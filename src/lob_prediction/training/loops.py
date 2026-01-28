from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from lob_prediction import config as cfg
from lob_prediction.evaluation.metrics import compute_metrics


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int,
                lr: float = 0.0001, device: str = cfg.DEVICE, patience: int = 5) -> Tuple[nn.Module, Dict]:
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
