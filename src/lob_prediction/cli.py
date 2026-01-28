#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from sklearn.metrics import confusion_matrix

from lob_prediction import config as cfg
from lob_prediction.utils import set_seed, banner, get_model_name, suppress_warnings
from lob_prediction.data import lobster_load, LOBDataset, LOBDataModule, LOBSTERPreprocessor, compute_returns
from lob_prediction.models import TLOB, TLOBDecay, DeepLOB, LiTTransformer, LiTDecayTransformer
from lob_prediction.training import TLOBTrainer, train_model
from lob_prediction.evaluation import (
    compute_metrics, print_metrics, analyze_model_decay, analyze_confidence,
    plot_confusion_matrix, plot_training_history
)


def create_model(model_name: str, decay: bool, n_features: int, seq_size: int) -> nn.Module:
    model_upper = model_name.upper().replace("-", "").replace("_", "")
    if model_upper == "TLOB":
        if decay:
            return TLOBDecay(hidden_dim=cfg.HIDDEN_DIM, num_layers=cfg.NUM_LAYERS, seq_size=seq_size,
                            num_features=n_features, num_heads=cfg.NUM_HEADS, init_decay=0.1)
        return TLOB(hidden_dim=cfg.HIDDEN_DIM, num_layers=cfg.NUM_LAYERS, seq_size=seq_size,
                   num_features=n_features, num_heads=cfg.NUM_HEADS)
    elif model_upper == "DEEPLOB":
        if decay:
            raise ValueError("DeepLOB does not support decay variant")
        return DeepLOB(n_features=n_features)
    elif model_upper == "LIT":
        if decay:
            return LiTDecayTransformer(n_features=n_features, window=seq_size, d_model=256, n_heads=8, n_layers=6, init_decay=0.1)
        return LiTTransformer(n_features=n_features, window=seq_size, d_model=256, n_heads=8, n_layers=6)
    raise ValueError(f"Unknown model: {model_name}")


def cmd_preprocess(args):
    suppress_warnings()
    set_seed(args.seed)
    
    raw_dir = f"./data/raw/{args.ticker}"
    output_dir = f"./data/preprocessed/{args.ticker}"
    if not os.path.exists(raw_dir):
        print(f"Error: Raw data not found: {raw_dir}")
        sys.exit(1)
    
    banner(f"PREPROCESSING {args.ticker}")
    print(f"Raw: {raw_dir} | Output: {output_dir} | Split: {cfg.SPLIT_RATES}\n")
    LOBSTERPreprocessor(raw_data_dir=raw_dir, output_dir=output_dir).preprocess()


def cmd_train(args):
    suppress_warnings()
    set_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('high')
    
    data_dir = f"./data/preprocessed/{args.ticker}"
    if not os.path.exists(data_dir):
        print(f"Error: Data not found: {data_dir}")
        print(f"Run: python -m lob_prediction preprocess --ticker {args.ticker}")
        sys.exit(1)
    
    banner(f"TRAINING TLOB - {args.ticker}")
    print(f"Data: {data_dir} | Epochs: {args.epochs} | Horizon: {args.horizon} | Device: {cfg.DEVICE}\n")
    
    train_x, train_y = lobster_load(os.path.join(data_dir, "train.npy"), horizon=args.horizon)
    val_x, val_y = lobster_load(os.path.join(data_dir, "val.npy"), horizon=args.horizon)
    test_x, test_y = lobster_load(os.path.join(data_dir, "test.npy"), horizon=args.horizon)
    
    if args.data_fraction < 1.0:
        n_train, n_val = int(len(train_y) * args.data_fraction), int(len(val_y) * args.data_fraction)
        train_x, train_y = train_x[:n_train], train_y[:n_train]
        val_x, val_y = val_x[:n_val], val_y[:n_val]
        print(f"Using {args.data_fraction*100:.0f}% data: {n_train} train, {n_val} val")
    
    print(f"Train: {train_x.shape} | Val: {val_x.shape} | Test: {test_x.shape}")
    for name, labels in [("Train", train_y), ("Val", val_y), ("Test", test_y)]:
        unique, counts = torch.unique(labels, return_counts=True)
        print(f"{name}: " + " ".join([f"{u.item()}:{(c/len(labels)).item():.1%}" for u, c in zip(unique, counts)]))
    
    train_set = LOBDataset(train_x, train_y, cfg.SEQ_SIZE)
    val_set = LOBDataset(val_x, val_y, cfg.SEQ_SIZE)
    test_set = LOBDataset(test_x, test_y, cfg.SEQ_SIZE)
    data_module = LOBDataModule(train_set, val_set, test_set, batch_size=args.batch_size, num_workers=args.num_workers)
    
    model = TLOBTrainer(num_features=train_x.shape[1], max_epochs=args.epochs, horizon=args.horizon, ticker=args.ticker)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = L.Trainer(
        accelerator="cpu" if cfg.DEVICE == "cpu" else "gpu", precision=cfg.PRECISION, max_epochs=args.epochs,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=1, verbose=False, min_delta=0.002),
                   TQDMProgressBar(refresh_rate=100)],
        num_sanity_val_steps=0, check_val_every_n_epoch=1, enable_progress_bar=True, enable_model_summary=False, logger=False)
    
    print("\nTraining...")
    trainer.fit(model, datamodule=data_module)
    
    print("\nTesting...")
    best_path = model.last_ckpt_path
    if best_path and os.path.exists(best_path):
        best_model = TLOBTrainer.load_from_checkpoint(best_path, map_location=cfg.DEVICE)
    else:
        best_model = model
    best_model.experiment_type = "EVALUATION"
    trainer.test(best_model, data_module.test_dataloader())
    banner("COMPLETE")


def cmd_run(args):
    suppress_warnings()
    set_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('high')
    
    model_name = get_model_name(args.model, args.decay)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("./results")
    run_dir = output_dir / f"{args.ticker}_{model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    banner("LOB PREDICTION PIPELINE")
    print(f"  Model: {model_name} | Ticker: {args.ticker} | Epochs: {args.epochs} | Device: {cfg.DEVICE}")
    start_time = time.time()
    
    banner("LOADING DATA")
    data_dir = f"./data/preprocessed/{args.ticker}"
    if not os.path.exists(data_dir):
        print(f"Error: Data not found: {data_dir}")
        sys.exit(1)
    
    train_x, train_y = lobster_load(f"{data_dir}/train.npy", horizon=args.horizon)
    val_x, val_y = lobster_load(f"{data_dir}/val.npy", horizon=args.horizon)
    test_x, test_y, test_midprice = lobster_load(f"{data_dir}/test.npy", horizon=args.horizon, return_midprice=True)
    test_returns = compute_returns(test_midprice, horizon=args.horizon)
    
    if args.data_fraction < 1.0:
        n_train, n_val = int(len(train_y) * args.data_fraction), int(len(val_y) * args.data_fraction)
        train_x, train_y = train_x[:n_train], train_y[:n_train]
        val_x, val_y = val_x[:n_val], val_y[:n_val]
    
    print(f"  Train: {train_x.shape[0]:,} | Val: {val_x.shape[0]:,} | Test: {test_x.shape[0]:,}")
    
    train_set = LOBDataset(train_x, train_y, cfg.SEQ_SIZE)
    val_set = LOBDataset(val_x, val_y, cfg.SEQ_SIZE)
    test_set = LOBDataset(test_x, test_y, cfg.SEQ_SIZE)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    banner("CREATING MODEL")
    model = create_model(args.model, args.decay, train_x.shape[1], cfg.SEQ_SIZE)
    print(f"  {model_name}: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    banner("TRAINING")
    model, history = train_model(model, train_loader, val_loader, args.epochs, args.lr, patience=args.patience)
    torch.save(model.state_dict(), run_dir / f"{model_name}_{args.ticker}_best.pt")
    plot_training_history(history, model_name, run_dir / "training_history.png")
    
    banner("EVALUATION")
    results = {"model": model_name, "ticker": args.ticker}
    
    model.eval()
    model = model.to(cfg.DEVICE)
    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                all_logits.append(model(x.to(cfg.DEVICE)).cpu())
                all_labels.append(y)
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0).numpy()
        probs = torch.softmax(all_logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        
        metrics = compute_metrics(all_labels, preds, y_prob=probs)
        for k, v in metrics.items():
            results[f"{split}_{k}"] = v
        
        cm = confusion_matrix(all_labels, preds, labels=[0, 1, 2])
        plot_confusion_matrix(cm, model_name, split, run_dir / f"confusion_matrix_{split}.png")
        
        if split == "test":
            results["test_labels"], results["test_preds"], results["test_probs"] = all_labels, preds, probs
    
    print_metrics({k.replace("test_", ""): v for k, v in results.items() if k.startswith("test_")}, "test")
    
    if args.decay:
        banner("DECAY ANALYSIS")
        decay_results = analyze_model_decay(model, analysis_dir, model_name)
        results["decay_analysis"] = decay_results
        if decay_results.get("decay_rates"):
            for layer, heads in decay_results["decay_rates"].items():
                for head, lam in heads.items():
                    print(f"  {layer} {head}: λ = {lam:.4f}")
    
    banner("CONFIDENCE ANALYSIS")
    try:
        n_test = len(test_set)
        y_ret = test_returns[:n_test] if len(test_returns) >= n_test else None
        confidence_results = analyze_confidence(model, val_loader, test_loader, y_ret, analysis_dir, model_name)
        results["confidence_analysis"] = confidence_results
    except Exception as e:
        print(f"  [WARN] Confidence analysis failed: {e}")
    
    banner("SAVING RESULTS")
    pd.DataFrame([{"Model": model_name, "Ticker": args.ticker, "Test Accuracy": results.get("test_accuracy", 0),
                   "Test Macro F1": results.get("test_macro_f1", 0)}]).to_csv(run_dir / "metrics.csv", index=False)
    
    with open(run_dir / "results.json", "w") as f:
        json.dump({"model": model_name, "ticker": args.ticker, "test_accuracy": results.get("test_accuracy"),
                   "test_macro_f1": results.get("test_macro_f1"), "history": history,
                   "elapsed_minutes": (time.time() - start_time) / 60}, f, indent=2,
                  default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    
    banner("COMPLETE")
    print(f"  Test Accuracy: {results.get('test_accuracy', 0):.4f} | Test F1: {results.get('test_macro_f1', 0):.4f}")
    print(f"  Time: {(time.time() - start_time)/60:.1f} min | Output: {run_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="LOB Prediction Pipeline", prog="lob_prediction")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess LOBSTER data")
    preprocess_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker")
    preprocess_parser.add_argument("--seed", type=int, default=42)
    
    train_parser = subparsers.add_parser("train", help="Train TLOB with Lightning")
    train_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker")
    train_parser.add_argument("--epochs", type=int, default=cfg.MAX_EPOCHS)
    train_parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    train_parser.add_argument("--horizon", type=int, default=10, choices=[10, 20, 50, 100])
    train_parser.add_argument("--data-fraction", type=float, default=1.0)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=1)
    
    run_parser = subparsers.add_parser("run", help="Full pipeline with analysis")
    run_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker")
    run_parser.add_argument("--model", type=str, default="TLOB", choices=["TLOB", "DeepLOB", "LiT"])
    run_parser.add_argument("--decay", action="store_true", help="Use decay variant")
    run_parser.add_argument("--epochs", type=int, default=10)
    run_parser.add_argument("--batch-size", type=int, default=128)
    run_parser.add_argument("--lr", type=float, default=0.0001)
    run_parser.add_argument("--patience", type=int, default=5)
    run_parser.add_argument("--horizon", type=int, default=10, choices=[10, 20, 50, 100])
    run_parser.add_argument("--data-fraction", type=float, default=1.0)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--output-dir", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "run":
        if args.model == "DeepLOB" and args.decay:
            parser.error("DeepLOB does not support --decay")
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
