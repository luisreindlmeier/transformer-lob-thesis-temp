#!/usr/bin/env python3
"""LOB Prediction Pipeline - Main Entry Point."""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import config as cfg
from utils import set_seed, banner, get_model_name
from data import lobster_load, LOBDataset, compute_returns
from models.tlob import TLOB
from models.deeplob import DeepLOB
from models.lit import LiTTransformer
from models.tlob_decay import TLOBDecay
from models.lit_decay import LiTDecayTransformer
from evaluation.metrics import compute_metrics, print_metrics
from evaluation.decay import analyze_model_decay
from evaluation.confidence import analyze_confidence

COLORS = {"green": "#A8D5BA", "red": "#F5B7B1", "blue": "#AED6F1", "orange": "#FAD7A0",
          "purple": "#D7BDE2", "teal": "#A3E4D7", "gray": "#D5D8DC", "yellow": "#F9E79F"}
CLASS_NAMES = ["Down", "Stationary", "Up"]


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


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int,
                lr: float = 0.0001, device: str = cfg.DEVICE, patience: int = 5):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss, best_state, patience_counter = float('inf'), None, 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if batch_idx % 500 == 0 and batch_idx > 0:
                print(f"    Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {np.mean(train_losses[-100:]):.4f}")
        
        history["train_loss"].append(np.mean(train_losses))
        
        model.eval()
        val_losses, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_losses.append(criterion(logits, y).item())
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(y.cpu().numpy())
        
        avg_val_loss = np.mean(val_losses)
        from sklearn.metrics import f1_score
        val_f1 = f1_score(np.concatenate(all_labels), np.concatenate(all_preds), average='macro', labels=[0,1,2], zero_division=0)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(val_f1)
        
        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    return model, history


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str = cfg.DEVICE):
    model.eval()
    model = model.to(device)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            all_logits.append(model(x.to(device)).cpu())
            all_labels.append(y)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.softmax(all_logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    return compute_metrics(all_labels, preds, y_prob=probs), all_labels, preds, probs


def plot_confusion_matrix(cm: np.ndarray, model_name: str, split: str, save_path: Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("pastel_blue", ["#FFFFFF", COLORS["blue"], "#2E86AB"])
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set(xticks=np.arange(3), yticks=np.arange(3), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           ylabel="True Label", xlabel="Predicted Label")
    ax.set_title(f"{model_name} - {split.upper()}", fontweight='bold')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_norm[i, j]:.1%}", ha="center", va="center",
                   color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()


def plot_training_history(history: dict, model_name: str, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], color=COLORS["blue"], linewidth=2, marker='o', markersize=4, label="Train")
    axes[0].plot(epochs, history["val_loss"], color=COLORS["orange"], linewidth=2, marker='s', markersize=4, label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Loss", fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["val_f1"], color=COLORS["green"], linewidth=2, marker='o', markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title(f"{model_name} - Val F1", fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()


def run_pipeline(args) -> dict:
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
    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        metrics, labels, preds, probs = evaluate_model(model, loader)
        for k, v in metrics.items():
            results[f"{split}_{k}"] = v
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
        plot_confusion_matrix(cm, model_name, split, run_dir / f"confusion_matrix_{split}.png")
        if split == "test":
            results["test_labels"], results["test_preds"], results["test_probs"] = labels, preds, probs
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
    parser = argparse.ArgumentParser(description="LOB Prediction Pipeline")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker (e.g., CSCO)")
    parser.add_argument("--model", type=str, default="TLOB", choices=["TLOB", "DeepLOB", "LiT"])
    parser.add_argument("--decay", action="store_true", help="Use decay variant")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=10, choices=[10, 20, 50, 100])
    parser.add_argument("--data-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    if args.model == "DeepLOB" and args.decay:
        parser.error("DeepLOB does not support --decay")
    run_pipeline(args)


if __name__ == "__main__":
    main()
