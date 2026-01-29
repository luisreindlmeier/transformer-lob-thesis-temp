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
from torch.utils.data import DataLoader, WeightedRandomSampler
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src_prediction import config as cfg
from src_prediction.utils import set_seed, banner, get_model_name, suppress_warnings
from src_prediction.data import lobster_load, LOBDataset, LOBDataModule, LOBSTERPreprocessor, compute_returns
from src_prediction.models import TLOB, TLOBDecay, DeepLOB, LiTTransformer, LiTDecayTransformer
from src_prediction.training import TLOBTrainer, train_model
from src_prediction.evaluation import (
    compute_metrics, print_metrics, analyze_model_decay, analyze_confidence,
    plot_confusion_matrix, plot_training_history
)
from src_prediction.evaluation.calibration import TemperatureScaler
from src_prediction.evaluation.backtest import run_backtest_outputs


def create_model(model_name: str, decay: bool, n_features: int, seq_size: int,
                 lit_use_bin: bool = False, lit_use_event_embed: bool = False,
                 lit_use_mean_pool: bool = False) -> nn.Module:
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
        kw = dict(use_bin=lit_use_bin, use_event_embed=lit_use_event_embed, use_mean_pool=lit_use_mean_pool)
        if decay:
            return LiTDecayTransformer(n_features=n_features, window=seq_size, d_model=256, n_heads=8, n_layers=6, init_decay=0.1, **kw)
        return LiTTransformer(n_features=n_features, window=seq_size, d_model=256, n_heads=8, n_layers=6, **kw)
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
        print(f"Run: python -m src_prediction preprocess --ticker {args.ticker}")
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
    if args.test_fraction < 1.0:
        n_test = int(len(test_y) * args.test_fraction)
        test_x, test_y = test_x[:n_test], test_y[:n_test]
        print(f"Using {args.test_fraction*100:.0f}% test data: {n_test} samples")
    
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
    if args.test_fraction < 1.0:
        n_test = int(len(test_y) * args.test_fraction)
        test_x, test_y = test_x[:n_test], test_y[:n_test]
        test_midprice = test_midprice[:n_test]
        test_returns = test_returns[:n_test]
        print(f"  Using {args.test_fraction*100:.0f}% test data: {n_test:,} samples")
    
    print(f"  Train: {train_x.shape[0]:,} | Val: {val_x.shape[0]:,} | Test: {test_x.shape[0]:,}")
    
    train_set = LOBDataset(train_x, train_y, cfg.SEQ_SIZE)
    val_set = LOBDataset(val_x, val_y, cfg.SEQ_SIZE)
    test_set = LOBDataset(test_x, test_y, cfg.SEQ_SIZE)

    is_lit = args.model.upper() == "LIT"
    if is_lit:
        n_train = len(train_set)
        train_y_np = np.asarray(train_y).ravel()
        counts = np.bincount(train_y_np[:n_train].astype(np.int32), minlength=3)
        counts = np.maximum(counts, 1)
        class_weights_arr = (n_train / (3 * counts)).astype(np.float32)
        # Stat (index 1) mindestens 1.0, damit Modell Stationary nicht komplett weglässt
        class_weights_arr[1] = max(class_weights_arr[1], 1.0)
        class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32)
        sample_weights = torch.tensor([1.0 / counts[train_y_np[i]] for i in range(n_train)], dtype=torch.float32)
        train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=n_train)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, shuffle=False, num_workers=0)
        print(f"  LiT class weights (Down/Stat/Up): [{class_weights_arr[0]:.3f}, {class_weights_arr[1]:.3f}, {class_weights_arr[2]:.3f}]")
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    banner("CREATING MODEL")
    lit_bin = getattr(args, "lit_use_bin", False)
    lit_embed = getattr(args, "lit_use_event_embed", False)
    lit_pool = getattr(args, "lit_use_mean_pool", False)
    model = create_model(args.model, args.decay, train_x.shape[1], cfg.SEQ_SIZE,
                         lit_use_bin=lit_bin, lit_use_event_embed=lit_embed, lit_use_mean_pool=lit_pool)
    print(f"  {model_name}: {sum(p.numel() for p in model.parameters()):,} parameters")
    if args.model.upper() == "LIT" and (lit_bin or lit_embed or lit_pool):
        print(f"  LiT options: bin={lit_bin}, event_embed={lit_embed}, mean_pool={lit_pool}")

    banner("TRAINING")
    use_lr_schedule = getattr(args, "lit_use_lr_schedule", False)
    if is_lit:
        model, history = train_model(model, train_loader, val_loader, args.epochs, args.lr, patience=args.patience,
                                     use_lr_schedule=use_lr_schedule,
                                     class_weights=class_weights_tensor, use_f1_for_best=True, label_smoothing=0.1)
    else:
        model, history = train_model(model, train_loader, val_loader, args.epochs, args.lr, patience=args.patience,
                                     use_lr_schedule=use_lr_schedule)
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
    
    run_results = {"model": model_name, "ticker": args.ticker, "test_accuracy": results.get("test_accuracy"),
                   "test_macro_f1": results.get("test_macro_f1"), "history": history,
                   "elapsed_minutes": (time.time() - start_time) / 60}
    if args.model.upper() == "LIT":
        run_results["lit_use_bin"] = getattr(args, "lit_use_bin", False)
        run_results["lit_use_event_embed"] = getattr(args, "lit_use_event_embed", False)
        run_results["lit_use_mean_pool"] = getattr(args, "lit_use_mean_pool", False)
        run_results["lit_use_lr_schedule"] = getattr(args, "lit_use_lr_schedule", False)
        run_results["lit_class_weights"] = True
        run_results["lit_use_f1_for_best"] = True
        run_results["lit_label_smoothing"] = 0.1
        run_results["lit_stat_min_weight"] = True
    with open(run_dir / "results.json", "w") as f:
        json.dump(run_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    
    banner("COMPLETE")
    print(f"  Test Accuracy: {results.get('test_accuracy', 0):.4f} | Test F1: {results.get('test_macro_f1', 0):.4f}")
    print(f"  Time: {(time.time() - start_time)/60:.1f} min | Output: {run_dir}")
    return results


def cmd_eval(args):
    """Load a saved model (best.pt) and run evaluation + confidence + backtesting only (no training)."""
    suppress_warnings()
    set_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    results_path = run_dir / "results.json"
    if not results_path.exists():
        print(f"Error: No results.json in {run_dir}")
        sys.exit(1)

    with open(results_path) as f:
        run_results = json.load(f)
    model_name = run_results.get("model") or run_results.get("model_name", "TLOB")
    ticker = run_results.get("ticker", "CSCO")
    run_args = run_results.get("args", {})
    model_type = run_args.get("model", "LiT" if "LiT" in model_name else "TLOB")
    decay = run_args.get("decay", "Decay" in model_name or model_name.endswith("-Decay"))
    horizon = run_args.get("horizon", 10)

    model_path = getattr(args, "model_path", None)
    if model_path:
        model_path = Path(model_path).resolve()
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}")
            sys.exit(1)
    else:
        pt_candidates = list(run_dir.glob("*_best.pt"))
        if not pt_candidates:
            print(f"Error: No *_best.pt in {run_dir} and no --model-path given. Run 'run' once to save the model.")
            sys.exit(1)
        model_path = pt_candidates[0]

    banner("EVAL (no training)")
    print(f"  Run dir: {run_dir} | Model: {model_name} | Ticker: {ticker} | Checkpoint: {model_path.name}")
    start_time = time.time()

    banner("LOADING DATA")
    data_dir = f"./data/preprocessed/{ticker}"
    if not os.path.exists(data_dir):
        print(f"Error: Data not found: {data_dir}")
        sys.exit(1)

    train_x, train_y = lobster_load(f"{data_dir}/train.npy", horizon=horizon)
    val_x, val_y = lobster_load(f"{data_dir}/val.npy", horizon=horizon)
    test_x, test_y, test_midprice = lobster_load(f"{data_dir}/test.npy", horizon=horizon, return_midprice=True)
    test_returns = compute_returns(test_midprice, horizon=horizon)

    data_fraction = getattr(args, "data_fraction", 1.0)
    if data_fraction < 1.0:
        n_train = max(1, int(len(train_y) * data_fraction))
        n_val = max(1, int(len(val_y) * data_fraction))
        train_x, train_y = train_x[:n_train], train_y[:n_train]
        val_x, val_y = val_x[:n_val], val_y[:n_val]
        pct = data_fraction * 100
        print(f"  Using {pct:.2f}% train/val: {train_x.shape[0]:,} train, {val_x.shape[0]:,} val")

    test_fraction = getattr(args, "test_fraction", 1.0)
    if test_fraction < 1.0:
        n_test = int(len(test_y) * test_fraction)
        test_x, test_y = test_x[:n_test], test_y[:n_test]
        test_midprice = test_midprice[:n_test]
        test_returns = test_returns[:n_test]
        print(f"  Using {test_fraction*100:.0f}% test data: {test_x.shape[0]:,} samples")

    print(f"  Train: {train_x.shape[0]:,} | Val: {val_x.shape[0]:,} | Test: {test_x.shape[0]:,}")

    batch_size = getattr(args, "batch_size", cfg.BATCH_SIZE)
    train_set = LOBDataset(train_x, train_y, cfg.SEQ_SIZE)
    val_set = LOBDataset(val_x, val_y, cfg.SEQ_SIZE)
    test_set = LOBDataset(test_x, test_y, cfg.SEQ_SIZE)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    banner("LOADING MODEL")
    lit_bin = run_results.get("lit_use_bin", False)
    lit_embed = run_results.get("lit_use_event_embed", False)
    lit_pool = run_results.get("lit_use_mean_pool", False)
    model = create_model(model_type, decay, train_x.shape[1], cfg.SEQ_SIZE,
                         lit_use_bin=lit_bin, lit_use_event_embed=lit_embed, lit_use_mean_pool=lit_pool)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(cfg.DEVICE)
    model.eval()
    print(f"  Loaded: {model_path}")

    banner("EVALUATION")
    results = {"model": model_name, "ticker": ticker, "eval_from": str(run_dir)}

    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        all_logits, all_labels = [], []
        pbar = tqdm(loader, desc=f"  {split}", unit="batch", leave=False)
        with torch.no_grad():
            for x, y in pbar:
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
        analysis_dir = run_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        plot_confusion_matrix(cm, model_name, split, run_dir / f"confusion_matrix_{split}.png")

        if split == "test":
            results["test_labels"], results["test_preds"], results["test_probs"] = all_labels, preds, probs

    print_metrics({k.replace("test_", ""): v for k, v in results.items() if k.startswith("test_")}, "test")

    if decay:
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

    results["elapsed_minutes"] = (time.time() - start_time) / 60
    eval_json = run_dir / "eval_results.json"
    with open(eval_json, "w") as f:
        json.dump({k: v for k, v in results.items() if k not in ("test_labels", "test_preds", "test_probs")},
                  f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))

    banner("COMPLETE")
    print(f"  Test Accuracy: {results.get('test_accuracy', 0):.4f} | Test F1: {results.get('test_macro_f1', 0):.4f}")
    print(f"  Time: {results['elapsed_minutes']:.1f} min | Output: {run_dir}")
    return results


def cmd_backtest(args):
    """Backtest a trained model only: PnL curve, Sharpe, MDD, hit rate, profit factor, confidence plots, trade table."""
    suppress_warnings()
    set_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    results_path = run_dir / "results.json"
    if not results_path.exists():
        print(f"Error: No results.json in {run_dir}")
        sys.exit(1)

    with open(results_path) as f:
        run_results = json.load(f)
    model_name = run_results.get("model") or run_results.get("model_name", "TLOB")
    ticker = run_results.get("ticker", "CSCO")
    run_args = run_results.get("args", {})
    model_type = run_args.get("model", "TLOB")
    decay = run_args.get("decay", "Decay" in model_name or model_name.endswith("-Decay"))
    horizon = run_args.get("horizon", 10)

    model_path = getattr(args, "model_path", None)
    if model_path:
        model_path = Path(model_path).resolve()
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}")
            sys.exit(1)
    else:
        pt_candidates = list(run_dir.glob("*_best.pt"))
        if not pt_candidates:
            print(f"Error: No *_best.pt in {run_dir}. Run 'run' once to save the model.")
            sys.exit(1)
        model_path = pt_candidates[0]

    out_dir = Path(args.output_dir).resolve() if getattr(args, "output_dir", None) else (run_dir / "backtesting")
    out_dir.mkdir(parents=True, exist_ok=True)

    banner("BACKTEST")
    print(f"  Run dir: {run_dir} | Model: {model_name} | Ticker: {ticker}")
    print(f"  Output: {out_dir}")
    start_time = time.time()

    data_dir = f"./data/preprocessed/{ticker}"
    if not os.path.exists(data_dir):
        print(f"Error: Data not found: {data_dir}")
        sys.exit(1)

    train_x, _ = lobster_load(f"{data_dir}/train.npy", horizon=horizon)
    val_x, val_y = lobster_load(f"{data_dir}/val.npy", horizon=horizon)
    test_x, test_y, test_midprice = lobster_load(f"{data_dir}/test.npy", horizon=horizon, return_midprice=True)
    test_returns = compute_returns(test_midprice, horizon=horizon)

    data_fraction = getattr(args, "data_fraction", 1.0)
    if data_fraction < 1.0:
        n_val = max(1, int(len(val_y) * data_fraction))
        val_x, val_y = val_x[:n_val], val_y[:n_val]
    test_fraction = getattr(args, "test_fraction", 1.0)
    if test_fraction < 1.0:
        n_test = int(len(test_y) * test_fraction)
        test_x, test_y = test_x[:n_test], test_y[:n_test]
        test_midprice = test_midprice[:n_test]
        test_returns = test_returns[:n_test]

    batch_size = getattr(args, "batch_size", cfg.BATCH_SIZE)
    val_set = LOBDataset(val_x, val_y, cfg.SEQ_SIZE)
    test_set = LOBDataset(test_x, test_y, cfg.SEQ_SIZE)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = create_model(model_type, decay, train_x.shape[1], cfg.SEQ_SIZE)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(cfg.DEVICE)
    model.eval()

    scaler = TemperatureScaler(model, cfg.DEVICE)
    optimal_temp = scaler.set_temperature(val_loader, use_tqdm=True)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="  Backtest inference", unit="batch", leave=True, mininterval=0.5):
            all_logits.append(model(x.to(cfg.DEVICE)).cpu())
            all_labels.append(y)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    y_prob = torch.softmax(all_logits / optimal_temp, dim=1).numpy()
    y_pred = np.argmax(y_prob, axis=1)
    y_true = all_labels
    n_test = len(test_set)
    y_ret = test_returns[:n_test] if len(test_returns) >= n_test else test_returns

    results = run_backtest_outputs(
        y_true, y_pred, y_prob, y_ret,
        output_dir=out_dir,
        model_name=model_name,
    )
    elapsed = (time.time() - start_time) / 60
    with open(out_dir / "backtest_metrics.json", "w") as f:
        json.dump({**results, "elapsed_minutes": elapsed}, f, indent=2)

    banner("COMPLETE")
    print(f"  Sharpe: {results['sharpe_ratio']:.4f} | MDD: {results['max_drawdown']:.4f} | Hit rate: {results['hit_rate']:.2%} | PF: {results['profit_factor']:.4f}")
    print(f"  Time: {elapsed:.1f} min | Output: {out_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="LOB Prediction Pipeline", prog="src_prediction")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess LOBSTER data")
    preprocess_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker")
    preprocess_parser.add_argument("--seed", type=int, default=42)
    
    train_parser = subparsers.add_parser("train", help="Train TLOB with Lightning")
    train_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker")
    train_parser.add_argument("--epochs", type=int, default=cfg.MAX_EPOCHS)
    train_parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    train_parser.add_argument("--horizon", type=int, default=10, choices=[10, 20, 50, 100])
    train_parser.add_argument("--data-fraction", type=float, default=1.0, help="Fraction of train/val data to use (0–1)")
    train_parser.add_argument("--test-fraction", type=float, default=1.0, help="Fraction of test data to use (0–1)")
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
    run_parser.add_argument("--data-fraction", type=float, default=1.0, help="Fraction of train/val data to use (0–1)")
    run_parser.add_argument("--test-fraction", type=float, default=1.0, help="Fraction of test data to use (0–1)")
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--output-dir", type=str, default=None)
    run_parser.add_argument("--lit-use-bin", action="store_true", help="LiT: use BiN input normalization (step 1)")
    run_parser.add_argument("--lit-use-event-embed", action="store_true", help="LiT: embed event_type column (step 2)")
    run_parser.add_argument("--lit-use-mean-pool", action="store_true", help="LiT: CLS + mean-pool aggregation (step 4)")
    run_parser.add_argument("--lit-use-lr-schedule", action="store_true", help="LiT: halve LR on no val improvement (step 3)")

    eval_parser = subparsers.add_parser("eval", help="Eval + confidence + backtesting from saved best.pt (no training)")
    eval_parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory (contains results.json; optional *_best.pt)")
    eval_parser.add_argument("--model-path", type=str, default=None, help="Path to .pt file if not in run-dir (e.g. from another run)")
    eval_parser.add_argument("--data-fraction", type=float, default=1.0, help="Fraction of train/val data (0–1); use e.g. 0.01 for quick check")
    eval_parser.add_argument("--test-fraction", type=float, default=1.0, help="Fraction of test data (0–1)")
    eval_parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    eval_parser.add_argument("--seed", type=int, default=42)

    backtest_parser = subparsers.add_parser("backtest", help="Backtest trained model only: PnL, Sharpe, MDD, hit rate, PF, confidence plots, trade table")
    backtest_parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory (results.json + *_best.pt)")
    backtest_parser.add_argument("--model-path", type=str, default=None, help="Path to .pt if not in run-dir")
    backtest_parser.add_argument("--data-fraction", type=float, default=1.0, help="Fraction of val data for temperature scaling (0–1)")
    backtest_parser.add_argument("--test-fraction", type=float, default=1.0, help="Fraction of test data (0–1)")
    backtest_parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    backtest_parser.add_argument("--output-dir", type=str, default=None, help="Override: write backtesting here instead of <run-dir>/backtesting/")
    backtest_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "run":
        if args.model == "DeepLOB" and args.decay:
            parser.error("DeepLOB does not support --decay")
        cmd_run(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
