from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Tuple[float, float]:
    correct_a = (y_pred_a == y_true).astype(int)
    correct_b = (y_pred_b == y_true).astype(int)
    b = np.sum((correct_a == 1) & (correct_b == 0))
    c = np.sum((correct_a == 0) & (correct_b == 1))
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return float(chi2), float(1 - stats.chi2.cdf(chi2, df=1))


def paired_permutation_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray,
                            n_permutations: int = 10000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    correct_a = (y_pred_a == y_true).astype(int)
    correct_b = (y_pred_b == y_true).astype(int)
    observed_diff = correct_a.mean() - correct_b.mean()
    n_samples = len(y_true)
    perm_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        swap_mask = rng.random(n_samples) < 0.5
        perm_diffs[i] = np.where(swap_mask, correct_b, correct_a).mean() - np.where(swap_mask, correct_a, correct_b).mean()
    return float(observed_diff), float(np.mean(np.abs(perm_diffs) >= np.abs(observed_diff)))


def bootstrap_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray, metric_fn=None,
                                   n_bootstrap: int = 1000, confidence_level: float = 0.95,
                                   seed: int = 42) -> Tuple[float, float, float]:
    if metric_fn is None:
        metric_fn = lambda yt, yp: np.mean(yt == yp)
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    point_estimate = metric_fn(y_true, y_pred)
    bootstrap_metrics = np.array([metric_fn(y_true[idx := rng.integers(0, n_samples, size=n_samples)], y_pred[idx]) for _ in range(n_bootstrap)])
    alpha = 1 - confidence_level
    return float(point_estimate), float(np.percentile(bootstrap_metrics, 100 * alpha / 2)), float(np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2)))


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)


def paired_t_test(scores_a: np.ndarray, scores_b: np.ndarray) -> Tuple[float, float]:
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return float(t_stat), float(p_value)


def wilcoxon_signed_rank_test(scores_a: np.ndarray, scores_b: np.ndarray) -> Tuple[float, float]:
    differences = scores_a - scores_b
    if np.sum(differences != 0) < 5:
        return 0.0, 1.0
    try:
        stat, p_value = stats.wilcoxon(scores_a, scores_b, alternative='two-sided')
        return float(stat), float(p_value)
    except ValueError:
        return 0.0, 1.0


def friedman_test(scores_matrix: np.ndarray) -> Tuple[float, float]:
    stat, p_value = stats.friedmanchisquare(*[scores_matrix[:, i] for i in range(scores_matrix.shape[1])])
    return float(stat), float(p_value)


def nemenyi_critical_difference(n_models: int, n_datasets: int, alpha: float = 0.05) -> float:
    q_alpha = {0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164},
               0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920}}
    q = q_alpha[0.05 if alpha <= 0.05 else 0.10].get(n_models, 3.164 if alpha <= 0.05 else 2.920)
    return float(q * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets)))


def nemenyi_test(scores_matrix: np.ndarray, model_names: List[str], alpha: float = 0.05) -> Dict:
    n_datasets, n_models = scores_matrix.shape
    ranks = np.array([stats.rankdata(-scores_matrix[i]) for i in range(n_datasets)])
    avg_ranks = ranks.mean(axis=0)
    cd = nemenyi_critical_difference(n_models, n_datasets, alpha)
    significant_pairs = [(model_names[i], model_names[j]) for i in range(n_models) for j in range(i + 1, n_models)
                         if abs(avg_ranks[i] - avg_ranks[j]) > cd]
    return {"ranks": {name: float(rank) for name, rank in zip(model_names, avg_ranks)},
            "critical_difference": cd, "significant_pairs": significant_pairs, "rank_matrix": ranks, "alpha": alpha}


def multi_model_comparison(scores_matrix: np.ndarray, model_names: List[str], alpha: float = 0.05) -> Dict:
    friedman_stat, friedman_p = friedman_test(scores_matrix)
    result = {"friedman_statistic": friedman_stat, "friedman_p": friedman_p, "friedman_significant": friedman_p < alpha,
              "n_models": len(model_names), "n_datasets": scores_matrix.shape[0]}
    result["nemenyi"] = nemenyi_test(scores_matrix, model_names, alpha) if friedman_p < alpha else None
    return result


def print_multi_model_comparison(result: Dict, model_names: List[str]) -> None:
    print("\n" + "=" * 70)
    print("MULTI-MODEL COMPARISON (Friedman + Nemenyi)")
    print("=" * 70)
    print(f"\nFriedman Test: chi2={result['friedman_statistic']:.4f}, p={result['friedman_p']:.6f}")
    if result['friedman_significant']:
        print("  *** Significant: Models perform differently ***")
        if result['nemenyi']:
            print(f"\nNemenyi Post-hoc (CD={result['nemenyi']['critical_difference']:.4f}):")
            for name, rank in sorted(result['nemenyi']['ranks'].items(), key=lambda x: x[1]):
                print(f"  {name}: {rank:.3f}")
            if result['nemenyi']['significant_pairs']:
                print("\n  Significant pairs:", result['nemenyi']['significant_pairs'])
    else:
        print("  Not significant")


def compare_models(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray,
                   model_name_a: str = "Model A", model_name_b: str = "Model B", alpha: float = 0.05) -> Dict:
    acc_a, acc_b = np.mean(y_pred_a == y_true), np.mean(y_pred_b == y_true)
    f1_a, f1_b = f1_macro(y_true, y_pred_a), f1_macro(y_true, y_pred_b)
    mcnemar_chi2, mcnemar_p = mcnemar_test(y_true, y_pred_a, y_pred_b)
    acc_diff, perm_p = paired_permutation_test(y_true, y_pred_a, y_pred_b)
    significant = (mcnemar_p < alpha) or (perm_p < alpha)
    better = model_name_a if f1_a > f1_b else (model_name_b if f1_b > f1_a else "Tie")
    return {"model_a": model_name_a, "model_b": model_name_b, "accuracy_a": float(acc_a), "accuracy_b": float(acc_b),
            "f1_a": float(f1_a), "f1_b": float(f1_b), "accuracy_diff": float(acc_diff), "f1_diff": float(f1_a - f1_b),
            "mcnemar_chi2": float(mcnemar_chi2), "mcnemar_p": float(mcnemar_p), "permutation_p": float(perm_p),
            "significant": bool(significant), "better_model": better if significant else "No significant difference", "alpha": alpha}


def compare_all_models(y_true: np.ndarray, predictions: Dict[str, np.ndarray], alpha: float = 0.05) -> List[Dict]:
    model_names = list(predictions.keys())
    return [compare_models(y_true, predictions[model_names[i]], predictions[name_b], model_names[i], name_b, alpha)
            for i, _ in enumerate(model_names) for name_b in model_names[i + 1:]]


def print_comparison_table(comparisons: List[Dict]) -> None:
    print("\n" + "=" * 90)
    print("STATISTICAL SIGNIFICANCE TESTING RESULTS")
    print("=" * 90)
    print(f"{'Model A':<15} {'Model B':<15} {'F1 Diff':>10} {'McNemar p':>12} {'Perm p':>10} {'Significant':<12}")
    print("-" * 90)
    for c in comparisons:
        print(f"{c['model_a']:<15} {c['model_b']:<15} {c['f1_diff']:>+10.4f} {c['mcnemar_p']:>12.4f} {c['permutation_p']:>10.4f} {'***' if c['significant'] else '':<12}")


def create_significance_report(y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                               output_path: Optional[str] = None) -> str:
    comparisons = compare_all_models(y_true, predictions)
    lines = ["=" * 80, "STATISTICAL SIGNIFICANCE REPORT", "=" * 80, "", "1. MODEL PERFORMANCE SUMMARY", "-" * 40]
    for name, preds in predictions.items():
        acc_est, acc_lo, acc_hi = bootstrap_confidence_interval(y_true, preds)
        lines.extend([f"  {name}: Acc={acc_est:.4f} (95% CI: [{acc_lo:.4f}, {acc_hi:.4f}]), F1={f1_macro(y_true, preds):.4f}", ""])
    lines.extend(["", "2. PAIRWISE COMPARISONS", "-" * 40])
    for c in comparisons:
        lines.extend([f"  {c['model_a']} vs {c['model_b']}: F1 diff={c['f1_diff']:+.4f}, McNemar p={c['mcnemar_p']:.4f}, Perm p={c['permutation_p']:.4f}",
                      f"    {'*** SIGNIFICANT: ' + c['better_model'] + ' better ***' if c['significant'] else 'Not significant'}", ""])
    report = "\n".join(lines)
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
    return report
