#!/bin/bash
# LiT Verbesserungsschritte: 5 Läufe mit 0.01 data/test fraction.
# LiT nutzt dabei immer: Class Weights, F1-Best-Checkpoint, WeightedRandomSampler (nur LiT, TLOB unverändert).
# Nach jedem Lauf: results/<ticker>_LiT_<timestamp> mit results.json und confusion_matrix_*.png
set -e
TICKER=${TICKER:-CSCO}
OUT=${OUT:-./results}
echo "Ticker=$TICKER | Output=$OUT | data-fraction=0.01 | test-fraction=0.01"

echo "=== 0. Initial (Baseline) ==="
python -m src_prediction run --ticker "$TICKER" --model LiT --epochs 1 --data-fraction 0.01 --test-fraction 0.01 --output-dir "$OUT"
echo ""

echo "=== 1. Step: BiN (--lit-use-bin) ==="
python -m src_prediction run --ticker "$TICKER" --model LiT --epochs 1 --data-fraction 0.01 --test-fraction 0.01 --output-dir "$OUT" --lit-use-bin
echo ""

echo "=== 2. Step: + event_type Embedding (--lit-use-event-embed) ==="
python -m src_prediction run --ticker "$TICKER" --model LiT --epochs 1 --data-fraction 0.01 --test-fraction 0.01 --output-dir "$OUT" --lit-use-bin --lit-use-event-embed
echo ""

echo "=== 3. Step: + LR-Schedule (--lit-use-lr-schedule) ==="
python -m src_prediction run --ticker "$TICKER" --model LiT --epochs 1 --data-fraction 0.01 --test-fraction 0.01 --output-dir "$OUT" --lit-use-bin --lit-use-event-embed --lit-use-lr-schedule
echo ""

echo "=== 4. Step: + Mean-Pool (--lit-use-mean-pool) ==="
python -m src_prediction run --ticker "$TICKER" --model LiT --epochs 1 --data-fraction 0.01 --test-fraction 0.01 --output-dir "$OUT" --lit-use-bin --lit-use-event-embed --lit-use-lr-schedule --lit-use-mean-pool
echo ""

echo "Done. Check $OUT for CSCO_LiT_* run dirs and fill notebooks/LIT_STEPS_RESULTS_TABLE.md"
