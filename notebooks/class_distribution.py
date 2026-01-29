#!/usr/bin/env python3
"""
Up/Down/Stationary-Verteilung pro Ticker und Split.
Nutzt preprocessed .npy (Label-Spalte für Horizon 10 = Index -4).
Keine Änderung am bestehenden Code – nur lesender Zugriff.
"""
import sys
from pathlib import Path

import numpy as np

# Repo-Root = Parent von notebooks/
REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED_DIR = REPO_ROOT / "data" / "preprocessed"
# Label 0=Up, 1=Stationary, 2=Down (vgl. loading.py, plotting.py)
CLASS_NAMES = ["Up", "Stationary", "Down"]
# Horizon 10 → Spalte -4 (loading.py: horizon_to_col[10] == 4)
LABEL_COL = -4


def get_tickers():
    """Alle Ticker mit preprocessed Daten."""
    if not PREPROCESSED_DIR.exists():
        return []
    return sorted([d.name for d in PREPROCESSED_DIR.iterdir() if d.is_dir() and (d / "train.npy").exists()])


def distribution_for_path(path: Path):
    """Liefert (counts 0/1/2, n_valid, n_total) für eine .npy-Datei."""
    data = np.load(path, mmap_mode="r")
    labels = np.asarray(data[:, LABEL_COL]).flatten()
    n_total = len(labels)
    valid = np.isfinite(labels)
    n_valid = int(valid.sum())
    y = labels[valid].astype(int)
    counts = np.bincount(y, minlength=3)[:3]
    return counts, n_valid, n_total


def main():
    tickers = get_tickers()
    if not tickers:
        print("Keine preprocessed Daten in", PREPROCESSED_DIR)
        sys.exit(1)

    print("Up/Down/Stationary-Verteilung (Horizon 10)")
    print("=" * 70)
    rows = []
    for ticker in tickers:
        ticker_dir = PREPROCESSED_DIR / ticker
        for split in ("train", "val", "test"):
            path = ticker_dir / f"{split}.npy"
            if not path.exists():
                continue
            counts, n_valid, n_total = distribution_for_path(path)
            if n_valid == 0:
                pct = (0.0, 0.0, 0.0)
            else:
                pct = tuple(100 * counts[i] / n_valid for i in range(3))
            rows.append({
                "ticker": ticker,
                "split": split,
                "n_valid": n_valid,
                "n_total": n_total,
                "up": pct[0],
                "stat": pct[1],
                "down": pct[2],
                "count_up": int(counts[0]),
                "count_stat": int(counts[1]),
                "count_down": int(counts[2]),
            })
            print(f"  {ticker:6} {split:5}  n={n_valid:>10,}  Up:{pct[0]:5.1f}%  Stat:{pct[1]:5.1f}%  Down:{pct[2]:5.1f}%")

    # --- GENAUE WERTE pro Ticker × Split ---
    print()
    print("GENAUE WERTE (Anzahlen + Prozent, 4 Dezimalstellen)")
    print("=" * 100)
    for ticker in tickers:
        ticker_rows = [r for r in rows if r["ticker"] == ticker]
        if not ticker_rows:
            continue
        print(f"\n  {ticker}")
        print("  " + "-" * 96)
        for r in ticker_rows:
            n = r["n_valid"]
            cu, cs, cd = r["count_up"], r["count_stat"], r["count_down"]
            pu, ps, pd = r["up"], r["stat"], r["down"]
            print(f"    {r['split']:5}  n_valid = {n:>12,}")
            print(f"           Up:         {cu:>12,}  ({pu:7.4f} %)")
            print(f"           Stationary: {cs:>12,}  ({ps:7.4f} %)")
            print(f"           Down:       {cd:>12,}  ({pd:7.4f} %)")
        print()

    print("=" * 100)
    # Plausibilität
    for r in rows:
        if r["n_valid"] > 0:
            s = r["up"] + r["stat"] + r["down"]
            if abs(s - 100.0) > 0.01:
                print("  ⚠ Summe != 100%:", r["ticker"], r["split"], s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
