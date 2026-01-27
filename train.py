#!/usr/bin/env python3
"""Minimal TLOB Training Script for LOBSTER data."""
import argparse
import os
import sys
import warnings
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import config as cfg
from utils import set_seed
from data import LOBSTERPreprocessor, lobster_load, LOBDataset, LOBDataModule
from trainer import TLOBTrainer


def preprocess(args):
    raw_dir = f"./data/raw/{args.ticker}"
    output_dir = f"./data/preprocessed/{args.ticker}"
    if not os.path.exists(raw_dir):
        print(f"Error: Raw data not found: {raw_dir}")
        sys.exit(1)
    print(f"\n{'='*60}\nPREPROCESSING {args.ticker}\n{'='*60}")
    print(f"Raw: {raw_dir} | Output: {output_dir} | Split: {cfg.SPLIT_RATES}\n")
    LOBSTERPreprocessor(raw_data_dir=raw_dir, output_dir=output_dir).preprocess()


def train(args):
    data_dir = f"./data/preprocessed/{args.ticker}"
    if not os.path.exists(data_dir):
        print(f"Error: Data not found: {data_dir}")
        print(f"Run: python train.py --ticker {args.ticker} --preprocess")
        sys.exit(1)
    
    print(f"\n{'='*60}\nTRAINING TLOB - {args.ticker}\n{'='*60}")
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
    print(f"\n{'='*60}\nCOMPLETE\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="TLOB Training")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
    parser.add_argument("--epochs", type=int, default=cfg.MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--horizon", type=int, default=10, choices=[10, 20, 50, 100])
    parser.add_argument("--data-fraction", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    
    set_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('high')
    
    if args.preprocess:
        preprocess(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
