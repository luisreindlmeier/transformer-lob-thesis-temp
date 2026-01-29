from contextlib import nullcontext
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
from src_prediction import config as cfg
from src_prediction.evaluation.metrics import compute_metrics


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int,
                lr: float = 0.0001, device: str = cfg.DEVICE, patience: int = 5,
                use_ema: bool = True, ema_decay: float = 0.999,
                use_lr_schedule: bool = False,
                class_weights: Optional[torch.Tensor] = None,
                use_f1_for_best: bool = False,
                label_smoothing: float = 0.0) -> Tuple[nn.Module, Dict]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
    ema.to(device)

    best_val_loss, best_val_f1, best_state, patience_counter = float("inf"), -1.0, None, 0
    min_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    
    pbar_epochs = tqdm(range(epochs), desc="Epochs", unit="epoch", leave=True)
    for epoch in pbar_epochs:
        model.train()
        train_losses = []
        pbar_batches = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False,
            unit="batch",
        )
        for x, y in pbar_batches:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            ema.update()
            train_losses.append(loss.item())
            pbar_batches.set_postfix(loss=f"{np.mean(train_losses[-100:]):.4f}")
        
        history["train_loss"].append(np.mean(train_losses))
        
        model.eval()
        val_losses, all_preds, all_labels = [], [], []
        with torch.no_grad():
            eval_ctx = ema.average_parameters() if use_ema else nullcontext()
            with eval_ctx:
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    val_losses.append(criterion(logits, y).item())
                    all_preds.append(logits.argmax(dim=1).cpu().numpy())
                    all_labels.append(y.cpu().numpy())
        
        avg_val_loss = np.mean(val_losses)
        val_f1 = f1_score(np.concatenate(all_labels), np.concatenate(all_preds), average='macro', labels=[0,1,2], zero_division=0)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(val_f1)
        
        pbar_epochs.set_postfix(
            train_loss=f"{np.mean(train_losses):.4f}",
            val_loss=f"{avg_val_loss:.4f}",
            val_f1=f"{val_f1:.4f}",
        )
        tqdm.write(f"  Epoch {epoch+1}/{epochs} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if use_f1_for_best:
            improved = val_f1 > best_val_f1
            if improved:
                best_val_f1 = val_f1
        else:
            improved = avg_val_loss < best_val_loss
            if improved:
                best_val_loss = avg_val_loss

        if improved:
            if use_ema:
                with ema.average_parameters():
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                tqdm.write(f"  Early stopping at epoch {epoch+1}")
                break

        if use_lr_schedule:
            if avg_val_loss < min_val_loss:
                improvement = min_val_loss - avg_val_loss
                min_val_loss = avg_val_loss
                if improvement < 0.002:
                    for g in optimizer.param_groups:
                        g["lr"] /= 2
                    tqdm.write(f"  Small improvement ({improvement:.4f}), halving LR -> {optimizer.param_groups[0]['lr']:.2e}")
            else:
                for g in optimizer.param_groups:
                    g["lr"] /= 2
                tqdm.write(f"  No improvement, halving LR -> {optimizer.param_groups[0]['lr']:.2e}")
    
    if best_state:
        model.load_state_dict(best_state)
    return model, history


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str = cfg.DEVICE,
                   temperature: float = 1.0, return_probs: bool = True) -> Tuple[Dict, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    model = model.to(device)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            all_logits.append(model(x.to(device)).cpu())
            all_labels.append(y)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.softmax(all_logits / temperature, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    metrics = compute_metrics(all_labels, preds, y_prob=probs if return_probs else None)
    return (metrics, all_labels, probs) if return_probs else (metrics, all_labels, None)
