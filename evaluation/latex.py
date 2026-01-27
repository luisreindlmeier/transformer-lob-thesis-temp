"""LaTeX export utilities for result tables."""
from typing import Dict, List, Optional
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


def comparison_to_latex(results: List[Dict], caption: str = "Model Comparison") -> str:
    if not results:
        return ""
    
    rows = []
    for r in results:
        name = r.get("model", "Model")
        acc = r.get("test_accuracy", 0)
        f1 = r.get("test_macro_f1", 0)
        rows.append(f"    {name} & {acc:.4f} & {f1:.4f} \\\\")
    
    return f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{tab:model_comparison}}
\\begin{{tabular}}{{lrr}}
\\toprule
Model & Accuracy & Macro F1 \\\\
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


def significance_to_latex(comparisons: List[Dict], caption: str = "Statistical Significance") -> str:
    if not comparisons:
        return ""
    
    rows = []
    for c in comparisons:
        a, b = c.get("model_a", "A"), c.get("model_b", "B")
        diff = c.get("f1_diff", 0)
        p = c.get("mcnemar_p", 1.0)
        sig = "$^{***}$" if c.get("significant", False) else ""
        rows.append(f"    {a} vs {b} & {diff:+.4f} & {p:.4f}{sig} \\\\")
    
    return f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{tab:significance}}
\\begin{{tabular}}{{lrr}}
\\toprule
Comparison & F1 Diff & p-value \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\multicolumn{{3}}{{l}}{{\\footnotesize $^{{***}}$ p < 0.05}}
\\end{{tabular}}
\\end{{table}}"""


def save_latex(content: str, path: str) -> None:
    with open(path, "w") as f:
        f.write(content)
