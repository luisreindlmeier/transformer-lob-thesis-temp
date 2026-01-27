"""Evaluation modules for LOB prediction models."""
from evaluation.metrics import evaluate_model, print_metrics, compute_metrics, compute_confusion_matrix, print_confusion_matrix
from evaluation.decay import analyze_decay, extract_decay_rates, interpret_lambda
from evaluation.confidence import (expected_calibration_error, maximum_calibration_error, brier_score,
    TemperatureScaler, analyze_confidence, confidence_binned_metrics, optimal_confidence_threshold,
    compute_sharpe_ratio, strategy_returns, filtered_strategy_analysis)
from evaluation.comparison import (mcnemar_test, paired_permutation_test, paired_t_test, wilcoxon_signed_rank_test,
    friedman_test, nemenyi_test, multi_model_comparison, print_multi_model_comparison, bootstrap_confidence_interval,
    compare_models, compare_all_models, print_comparison_table, create_significance_report)
from evaluation.latex import metrics_to_latex, comparison_to_latex, confusion_matrix_to_latex, significance_to_latex, save_latex

__all__ = [
    "evaluate_model", "print_metrics", "compute_metrics", "compute_confusion_matrix", "print_confusion_matrix",
    "analyze_decay", "extract_decay_rates", "interpret_lambda",
    "expected_calibration_error", "maximum_calibration_error", "brier_score", "TemperatureScaler",
    "analyze_confidence", "confidence_binned_metrics", "optimal_confidence_threshold",
    "compute_sharpe_ratio", "strategy_returns", "filtered_strategy_analysis",
    "mcnemar_test", "paired_permutation_test", "paired_t_test", "wilcoxon_signed_rank_test",
    "bootstrap_confidence_interval", "compare_models", "compare_all_models", "print_comparison_table",
    "create_significance_report", "friedman_test", "nemenyi_test", "multi_model_comparison", "print_multi_model_comparison",
    "metrics_to_latex", "comparison_to_latex", "confusion_matrix_to_latex", "significance_to_latex", "save_latex",
]
