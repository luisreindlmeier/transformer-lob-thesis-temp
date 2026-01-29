from typing import Dict, List
import numpy as np


def metrics_to_latex(metrics: Dict[str, float], model_name: str = "Model", caption: str = "Model Performance") -> str:
    rows = []
    for key in ["accuracy", "macro_f1", "weighted_f1", "mcc", "cohen_kappa", "balanced_accuracy"]:
        if key in metrics:
            label = key.replace("_", " ").title()
            rows.append(f"    {label} & {metrics[key]:.4f} \\\\")
    
    return f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{tab:{model_name.lower().replace(' ', '_')}_metrics}}
\\begin{{tabular}}{{lr}}
\\toprule
Metric & Value \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def confusion_matrix_to_latex(cm: np.ndarray, class_names: List[str] = ["Down", "Stat", "Up"],
                               caption: str = "Confusion Matrix", normalize: bool = True) -> str:
    cm = cm.astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums
    
    header = " & ".join([""] + class_names) + " \\\\"
    rows = []
    for i, name in enumerate(class_names):
        if normalize:
            vals = " & ".join([f"{cm[i, j]:.1%}" for j in range(len(class_names))])
        else:
            vals = " & ".join([f"{int(cm[i, j])}" for j in range(len(class_names))])
        rows.append(f"    {name} & {vals} \\\\")
    
    return f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{tab:confusion_matrix}}
\\begin{{tabular}}{{l{'r' * len(class_names)}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def save_latex(content: str, path: str) -> None:
    with open(path, "w") as f:
        f.write(content)
