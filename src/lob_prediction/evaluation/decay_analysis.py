from pathlib import Path
from typing import Optional
import numpy as np
import torch


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


def analyze_decay(checkpoint_path, output_dir: Optional[Path] = None, model_name: str = "Decay Model") -> dict:
    from lob_prediction.evaluation.plotting import plot_decay_curves
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
    plot_decay_curves(decay_rates, model_name, plot_path)
    return {"decay_rates": decay_rates, "summary": summary, "plot_path": str(plot_path)}


def analyze_model_decay(model: torch.nn.Module, output_dir: Optional[Path] = None, model_name: str = "Model") -> dict:
    from lob_prediction.evaluation.plotting import plot_decay_curves
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
    plot_decay_curves(decay_rates, model_name, plot_path)
    return {"decay_rates": decay_rates, "summary": summary, "plot_path": str(plot_path)}


analyze_decay_rates = analyze_decay
