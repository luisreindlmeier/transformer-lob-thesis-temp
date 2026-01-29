# Transformer-Based LOB Prediction

Bachelor thesis: *Transformer-Based Prediction of Limit Order Book Dynamics for Market Making*

## Features

- **Multiple Models**: TLOB, DeepLOB, LiT (with optional Decay variants)
- **Preprocessing**: LOBSTER Level 2 data preprocessing with multi-horizon labels
- **Training**: PyTorch Lightning with EMA and manual LR scheduling
- **Evaluation**: Comprehensive metrics, calibration analysis, and statistical testing

## Project Structure

```
├── src_prediction/
│   ├── config.py              # Configuration constants
│   ├── cli.py                 # Unified CLI entry point
│   ├── data/
│   │   ├── dataset.py         # LOBDataset, LOBDataModule
│   │   ├── loading.py         # Data loading utilities
│   │   └── preprocessing.py   # LOBSTER preprocessing
│   ├── models/
│   │   ├── components.py      # BiN, MLP, positional embeddings
│   │   ├── attention.py       # Standard & Decay attention
│   │   ├── tlob.py            # TLOB & TLOBDecay
│   │   ├── deeplob.py         # DeepLOB (CNN-GRU)
│   │   └── lit.py             # LiT & LiTDecay Transformer
│   ├── training/
│   │   ├── trainer.py         # Lightning training module
│   │   └── loops.py           # Vanilla PyTorch training
│   ├── evaluation/
│   │   ├── metrics.py         # Accuracy, F1, MCC, etc.
│   │   ├── calibration.py     # Calibration & backtesting
│   │   ├── decay_analysis.py  # Decay rate analysis
│   │   ├── statistical.py     # Statistical tests
│   │   ├── plotting.py        # Visualization
│   │   └── export.py          # LaTeX export
│   └── utils/
│       ├── seed.py            # Reproducibility
│       ├── helpers.py         # Utility functions
│       └── logging.py         # Logging setup
├── data/
│   ├── raw/{TICKER}/          # Raw LOBSTER CSV files
│   └── preprocessed/{TICKER}/
├── checkpoints/               # Saved models
└── results/                   # Outputs
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data

**Source**: LOBSTER Level 2 data (academic license from [lobsterdata.com](https://lobsterdata.com))

Place raw data in `data/raw/{TICKER}/`:

```
data/raw/CSCO/
├── CSCO_2023-07-03_34200000_57600000_message_10.csv
├── CSCO_2023-07-03_34200000_57600000_orderbook_10.csv
└── ...
```

See `DATASET.md` for format details.

## Usage

```bash
# Preprocessing
python -m src_prediction preprocess --ticker CSCO

# Training (Lightning with EMA)
python -m src_prediction train --ticker CSCO --epochs 10

# Full pipeline with analysis
python -m src_prediction run --ticker CSCO --model TLOB --epochs 50

# With decay attention
python -m src_prediction run --ticker CSCO --model TLOB --decay --epochs 50

# Quick test
python -m src_prediction run --ticker CSCO --model TLOB --epochs 1 --data-fraction 0.01
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `preprocess` | Preprocess raw LOBSTER data |
| `train` | Train TLOB with Lightning (EMA, LR scheduling) |
| `run` | Full pipeline with training, evaluation, and analysis |

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ticker` | Required | Stock ticker (CSCO, INTC, ...) |
| `--model` | TLOB | Model: TLOB, DeepLOB, or LiT |
| `--decay` | False | Use decay attention variant |
| `--epochs` | 10 | Training epochs |
| `--batch-size` | 128 | Batch size |
| `--horizon` | 10 | Prediction horizon (10, 20, 50, 100) |
| `--lr` | 0.0001 | Learning rate |
| `--data-fraction` | 1.0 | Fraction of data to use |

## Models

| Model | Architecture | Key Features |
|-------|--------------|--------------|
| **TLOB** | Transformer | Bilinear Normalization, alternating attention |
| **TLOBDecay** | Transformer | TLOB + learnable decay attention |
| **DeepLOB** | CNN-GRU | Convolutional + recurrent baseline |
| **LiT** | Transformer | Lightweight with CLS token |
| **LiTDecay** | Transformer | LiT + learnable decay attention |

```python
from src_prediction.models import TLOB, TLOBDecay, DeepLOB, LiTTransformer

model = TLOB(hidden_dim=46, num_layers=4, seq_size=128)
model = TLOBDecay(hidden_dim=46, num_layers=4, seq_size=128)
model = DeepLOB(n_features=46)
model = LiTTransformer(n_features=46, window=128)
```

## Data Format

**Input** (46 features):
- 6 order features: `[time, event_type, size, price, direction, depth]`
- 40 LOB features: `[ask_p1, ask_s1, bid_p1, bid_s1, ..., ask_p10, ask_s10, bid_p10, bid_s10]`

**Labels** (3-class): 0=Up, 1=Stationary, 2=Down

## Citation

```bibtex
@software{reindlmeier2026thesis,
  author = {Reindlmeier, Luis},
  title = {Transformer-Based Prediction of Limit Order Book Dynamics},
  year = {2026},
  url = {https://github.com/luisreindlmeier/BT_Transformer_LOB}
}
```

## License

MIT - See `LICENSE`
