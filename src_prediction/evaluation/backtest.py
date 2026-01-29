"""Backtesting outputs: PnL curve, Sharpe, MDD, hit rate, profit factor, confidence plots, trade table."""
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src_prediction.evaluation.calibration import (
    TemperatureScaler,
    compute_sharpe_ratio,
    strategy_returns,
    filtered_strategy_analysis,
)
from src_prediction.evaluation.plotting import COLORS


def max_drawdown(cumulative_pnl: np.ndarray) -> float:
    """Peak-to-trough decline. Returns positive number (e.g. 0.05 = 5% MDD)."""
    if len(cumulative_pnl) == 0:
        return 0.0
    peak = np.maximum.accumulate(cumulative_pnl)
    drawdown = peak - cumulative_pnl
    return float(np.max(drawdown)) if np.any(peak > 0) else 0.0


def hit_rate(strat_ret: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of directional trades (pred != 1) that are profitable. Ideally > 0.5."""
    mask = y_pred != 1
    if np.sum(mask) == 0:
        return 0.0
    rets = strat_ret[mask]
    return float(np.mean(rets > 0))


def profit_factor(strat_ret: np.ndarray, y_pred: np.ndarray) -> float:
    """Gross profit / gross loss (only directional trades). > 1 profitable, > 1.5 good."""
    mask = y_pred != 1
    if np.sum(mask) == 0:
        return 0.0
    rets = strat_ret[mask]
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = np.abs(np.sum(rets[rets < 0]))
    if gross_loss == 0:
        return float(gross_profit) if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def run_backtest_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    y_ret: np.ndarray,
    output_dir: Path,
    model_name: str = "Model",
    n_bins: int = 10,
    confidence_thresholds: Optional[np.ndarray] = None,
) -> dict:
    """
    Produce all 8 backtest outputs and save to output_dir.
    No model or dataloader needed – uses precomputed y_true, y_pred, y_prob, y_ret.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strat_ret = strategy_returns(y_pred, y_ret)
    confidences = np.max(y_prob, axis=1)
    cum_pnl = np.cumsum(strat_ret)
    sharpe = compute_sharpe_ratio(strat_ret)
    mdd = max_drawdown(cum_pnl)
    hit = hit_rate(strat_ret, y_pred)
    pf = profit_factor(strat_ret, y_pred)

    # 1. Cumulative PnL curve – baseline (all trades)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(cum_pnl)), cum_pnl, color=COLORS["blue"], linewidth=1.5)
    ax.set_xlabel("Time (events)")
    ax.set_ylabel("Cumulative PnL")
    ax.set_title(f"{model_name} – Cumulative PnL (baseline)", fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_pnl.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    # 1b. Separate cumulative PnL for each confidence threshold 0.6, 0.7, 0.8, 0.9
    for thresh in [0.6, 0.7, 0.8, 0.9]:
        mask = confidences >= thresh
        strat_ret_f = np.zeros_like(strat_ret, dtype=np.float64)
        strat_ret_f[mask] = strat_ret[mask]
        cum_pnl_f = np.cumsum(strat_ret_f)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.arange(len(cum_pnl_f)), cum_pnl_f, color=COLORS["teal"], linewidth=1.5)
        ax.set_xlabel("Time (events)")
        ax.set_ylabel("Cumulative PnL")
        ax.set_title(f"{model_name} – Cumulative PnL (conf >= {thresh})", fontweight="bold")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"cumulative_pnl_conf_{thresh}.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

    # 2–5. Scalars: save to metrics JSON + short summary
    scalars = {
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "hit_rate": hit,
        "profit_factor": pf,
    }
    pd.DataFrame([scalars]).to_csv(output_dir / "backtest_metrics.csv", index=False)
    with open(output_dir / "backtest_summary.txt", "w") as f:
        f.write(f"Sharpe ratio:      {sharpe:.4f}  (>1 good, >2 excellent, <0 losing)\n")
        f.write(f"Max drawdown:      {mdd:.4f}  (lower = more stable)\n")
        f.write(f"Hit rate:         {hit:.2%}  (ideally >50%)\n")
        f.write(f"Profit factor:    {pf:.4f}  (>1 profitable, >1.5 good)\n")

    # Threshold analysis for trade count table and optional use in plots
    bt_analysis = filtered_strategy_analysis(
        y_true, y_pred, y_prob, y_ret,
        confidence_thresholds=confidence_thresholds,
    )
    thresholds = np.array(bt_analysis["thresholds"])
    n_trades = np.array(bt_analysis["n_trades"])
    sharpe_by_thresh = np.array(bt_analysis["sharpe_ratios"])
    coverages = np.array(bt_analysis["coverages"])

    # 8. Trade count by threshold (table)
    trade_table = pd.DataFrame({
        "confidence_threshold": thresholds,
        "n_trades": n_trades,
        "sharpe_ratio": sharpe_by_thresh,
        "coverage": coverages,
    })
    trade_table.to_csv(output_dir / "trade_count_by_threshold.csv", index=False)

    # Comparison: baseline vs. confidence-filtered strategies (>= 0.6, 0.7, 0.8, 0.9)
    comparison_thresholds = [None, 0.6, 0.7, 0.8, 0.9]  # None = baseline (all trades)
    comparison_rows = []
    for thresh in comparison_thresholds:
        if thresh is None:
            mask = np.ones(len(y_ret), dtype=bool)
            label = "baseline (all)"
        else:
            mask = confidences >= thresh
            if np.sum(mask) < 10:
                comparison_rows.append({
                    "strategy": f"conf >= {thresh}",
                    "sharpe_ratio": np.nan,
                    "max_drawdown": np.nan,
                    "hit_rate": np.nan,
                    "profit_factor": np.nan,
                    "n_trades": int(np.sum((y_pred != 1) & mask)),
                })
                continue
            label = f"conf >= {thresh}"
        ret_f = strat_ret[mask]
        pred_f = y_pred[mask]
        cum_f = np.cumsum(ret_f)
        comparison_rows.append({
            "strategy": label,
            "sharpe_ratio": compute_sharpe_ratio(ret_f),
            "max_drawdown": max_drawdown(cum_f),
            "hit_rate": hit_rate(ret_f, pred_f),
            "profit_factor": profit_factor(ret_f, pred_f),
            "n_trades": int(np.sum(pred_f != 1)),
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "confidence_strategy_comparison.csv", index=False)

    # Plot: Sharpe and n_trades by strategy for quick comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(comparison_df))
    width = 0.35
    axes[0].bar(x - width / 2, comparison_df["sharpe_ratio"], width=width, color=COLORS["blue"], label="Sharpe")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comparison_df["strategy"], rotation=15, ha="right")
    axes[0].set_ylabel("Sharpe ratio")
    axes[0].set_title("Sharpe by strategy")
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.7)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, comparison_df["n_trades"], width=0.6, color=COLORS["teal"], edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(comparison_df["strategy"], rotation=15, ha="right")
    axes[1].set_ylabel("Number of trades")
    axes[1].set_title("Trade count by strategy")
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_strategy_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    # 6. Returns by confidence bin (bar chart)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_ret_by_bin = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if np.sum(mask) > 0:
            mean_ret_by_bin[i] = np.mean(strat_ret[mask])
        else:
            mean_ret_by_bin[i] = np.nan

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 1 / n_bins * 0.8
    valid = ~np.isnan(mean_ret_by_bin)
    ax.bar(bin_centers[valid], mean_ret_by_bin[valid], width=width, color=COLORS["teal"], edgecolor="white")
    ax.set_xlabel("Confidence (bin center)")
    ax.set_ylabel("Mean return")
    ax.set_title(f"{model_name} – Returns by confidence bin (calibration check)", fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "returns_by_confidence_bin.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    # 7. Accuracy vs. confidence (line plot)
    acc_by_bin = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if np.sum(mask) > 0:
            acc_by_bin[i] = np.mean(y_true[mask] == y_pred[mask])
        else:
            acc_by_bin[i] = np.nan

    fig, ax = plt.subplots(figsize=(8, 5))
    valid_acc = ~np.isnan(acc_by_bin)
    ax.plot(bin_centers[valid_acc], acc_by_bin[valid_acc], color=COLORS["blue"], linewidth=2, marker="o", markersize=6)
    ax.set_xlabel("Confidence (bin center)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model_name} – Accuracy vs. confidence (monotonic = well calibrated)", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_confidence.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    recs = comparison_df.to_dict(orient="records")
    for r in recs:
        for k in r:
            if isinstance(r[k], (float, np.floating)) and np.isnan(r[k]):
                r[k] = None

    return {
        **scalars,
        "baseline_sharpe": float(bt_analysis["baseline_sharpe"]),
        "optimal_sharpe": float(bt_analysis["optimal_sharpe"]),
        "optimal_threshold": float(bt_analysis["optimal_threshold"]),
        "n_samples": len(y_true),
        "n_directional_trades": int(np.sum(y_pred != 1)),
        "confidence_strategy_comparison": recs,
    }
