"""Model evaluation metrics."""
from typing import Dict, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, classification_report,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, roc_auc_score)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=[0, 1, 2], zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    
    precision = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    
    for i, cls in enumerate(["down", "stat", "up"]):
        metrics[f"precision_{cls}"] = float(precision[i])
        metrics[f"recall_{cls}"] = float(recall[i])
        metrics[f"f1_{cls}"] = float(f1_per_class[i])
    
    if y_prob is not None:
        try:
            metrics["roc_auc_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            metrics["roc_auc_weighted"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except ValueError:
            pass
        from evaluation.confidence import expected_calibration_error, maximum_calibration_error, brier_score
        ece, _, _, _ = expected_calibration_error(y_true, y_prob)
        metrics["ece"] = float(ece)
        metrics["mce"] = float(maximum_calibration_error(y_true, y_prob))
        metrics["brier_score"] = float(brier_score(y_true, y_prob))
    
    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, normalize: Optional[str] = None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize=normalize)


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = ["Down", "Stat", "Up"], normalize: bool = True) -> None:
    cm = compute_confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    print("\n  Confusion Matrix" + (" (normalized)" if normalize else "") + ":")
    print(f"  {'':>8}", end="")
    for name in class_names:
        print(f"  {name:>8}", end="")
    print("  <- Predicted")
    print(f"  {'-' * (10 + 10 * len(class_names))}")
    for i, name in enumerate(class_names):
        print(f"  {name:>8}", end="")
        for j in range(len(class_names)):
            print(f"  {cm[i, j]:>8.2%}" if normalize else f"  {int(cm[i, j]):>8d}", end="")
        print()


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str = cfg.DEVICE,
                   temperature: float = 1.0, return_probs: bool = True):
    model.eval()
    model = model.to(device)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            all_logits.append(model(X_batch.to(device)).cpu())
            all_labels.append(y_batch)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.softmax(all_logits / temperature, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    metrics = compute_metrics(all_labels, preds, y_prob=probs if return_probs else None)
    return (metrics, all_labels, probs) if return_probs else (metrics, all_labels, None)


def print_metrics(metrics: Dict[str, float], split: str = "test") -> None:
    print(f"\n  {split.upper()} Metrics:")
    print(f"  {'Metric':<22} {'Value':>10}")
    print(f"  {'-'*34}")
    for key in ["accuracy", "macro_f1", "weighted_f1", "mcc", "cohen_kappa", "balanced_accuracy"]:
        if key in metrics:
            print(f"  {key.replace('_', ' ').title():<22} {metrics[key]:>10.4f}")
    if "roc_auc_macro" in metrics:
        print(f"  {'ROC-AUC (macro)':<22} {metrics['roc_auc_macro']:>10.4f}")
    if "ece" in metrics:
        print(f"  {'-'*34}")
        for key in ["ece", "mce", "brier_score"]:
            print(f"  {key.upper():<22} {metrics[key]:>10.4f}")
    print(f"  {'-'*34}")
    for cls in ["down", "stat", "up"]:
        if f"f1_{cls}" in metrics:
            print(f"  {f'F1-{cls.title()}':<22} {metrics[f'f1_{cls}']:>10.4f}")


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(y_true, y_pred, target_names=["Down", "Stationary", "Up"], digits=4)
