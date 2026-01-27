"""Decay attention analysis for TLOBDecay/LiTDecay models."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_decay_rates(state_dict: dict) -> dict:
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    decay_rates = {}
    for key, value in state_dict.items():
        if "lambda_raw" in key:
            parts = key.split(".")
            layer_idx = next((int(parts[i + 1]) for i, part in enumerate(parts) 
                            if part in ("layers", "encoder") and i + 1 < len(parts) and parts[i + 1].isdigit()), 0)
            layer_name = f"Layer {layer_idx + 1}"
            lambda_vals = np.log(1 + np.exp(value.cpu().numpy()))
            if layer_name not in decay_rates:
                decay_rates[layer_name] = {}
            for head_idx, lam in enumerate(lambda_vals):
                decay_rates[layer_name][f"Head {head_idx + 1}"] = float(lam)
    return decay_rates


def interpret_lambda(lam: float) -> str:
    if lam < 0.05: return "Very slow (global)"
    if lam < 0.1: return "Slow (long-range)"
    if lam < 0.2: return "Medium"
    if lam < 0.5: return "Fast (local)"
    return "Very fast (recent)"


def analyze_decay(checkpoint_path: str | Path, output_dir: Path | None = None, model_name: str = "Decay Model") -> dict:
    checkpoint_path = Path(checkpoint_path)
    output_dir = output_dir or Path("./results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}\n  H2 Decay Analysis: {model_name}\n{'='*60}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    decay_rates = extract_decay_rates(state_dict)
    
    if not decay_rates:
        print("[WARN] No decay rates found")
        return {"decay_rates": {}, "summary": {}}
    
    all_lambdas = []
    print(f"\n  {'Layer':<12} {'Head':<10} {'lambda':>10} {'Interpretation':>25}")
    print(f"  {'-'*60}")
    for layer_name, heads in sorted(decay_rates.items()):
        for head_name, lam in sorted(heads.items()):
            all_lambdas.append(lam)
            print(f"  {layer_name:<12} {head_name:<10} {lam:>10.4f} {interpret_lambda(lam):>25}")
    
    summary = {"min_lambda": float(np.min(all_lambdas)), "max_lambda": float(np.max(all_lambdas)),
               "mean_lambda": float(np.mean(all_lambdas)), "std_lambda": float(np.std(all_lambdas)), "n_heads": len(all_lambdas)}
    
    plot_path = output_dir / f"decay_curves_{model_name.replace(' ', '_').replace('-', '_')}.png"
    _plot_decay_curves(decay_rates, model_name, plot_path)
    return {"decay_rates": decay_rates, "summary": summary, "plot_path": str(plot_path)}


def _plot_decay_curves(decay_rates: dict, model_name: str, save_path: Path, max_distance: int = 100) -> None:
    COLORS = ["#A8D5BA", "#AED6F1", "#FAD7A0", "#F5B7B1", "#D7BDE2", "#A3E4D7", "#F9E79F", "#D5D8DC"]
    n_layers = len(decay_rates)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5))
    if n_layers == 1: axes = [axes]
    
    distances = np.arange(max_distance)
    for ax, (layer_name, heads) in zip(axes, sorted(decay_rates.items())):
        for i, (head_name, lam) in enumerate(sorted(heads.items())):
            ax.plot(distances, np.exp(-lam * distances), color=COLORS[i % len(COLORS)], 
                   linewidth=2.5, label=f"{head_name} (λ={lam:.3f})")
        ax.set_xlabel("Distance (time steps)")
        ax.set_ylabel("Attention Weight")
        ax.set_title(f"{model_name} - {layer_name}", fontweight='bold')
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
    plt.close()


def analyze_model_decay(model: torch.nn.Module, output_dir: Path | None = None, model_name: str = "Model") -> dict:
    output_dir = output_dir or Path("./results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if hasattr(model, "get_decay_rates"):
        rates = model.get_decay_rates()
        decay_rates = {key.replace("_", " ").title(): {f"Head {i+1}": float(v) for i, v in enumerate(vals.tolist())}
                       for key, vals in rates.items()}
    else:
        decay_rates = extract_decay_rates(model.state_dict())
    
    if not decay_rates:
        return {"decay_rates": {}, "summary": {}}
    
    all_lambdas = [lam for heads in decay_rates.values() for lam in heads.values()]
    summary = {"min_lambda": float(np.min(all_lambdas)), "max_lambda": float(np.max(all_lambdas)),
               "mean_lambda": float(np.mean(all_lambdas)), "std_lambda": float(np.std(all_lambdas)), "n_heads": len(all_lambdas)}
    
    plot_path = output_dir / f"decay_curves_{model_name.replace(' ', '_')}.png"
    _plot_decay_curves(decay_rates, model_name, plot_path)
    return {"decay_rates": decay_rates, "summary": summary, "plot_path": str(plot_path)}

analyze_decay_rates = analyze_decay
