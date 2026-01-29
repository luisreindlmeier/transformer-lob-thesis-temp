# Technology Stack

## 1. Overview

The project uses a modern Python deep learning stack optimized for reproducibility and maintainability.

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  src_prediction (CLI, data, models, evaluation, utils)  │
├─────────────────────────────────────────────────────────┤
│                   Framework Layer                        │
│         PyTorch Lightning (training orchestration)       │
├─────────────────────────────────────────────────────────┤
│                    Core ML Layer                         │
│       PyTorch (neural networks, autograd, CUDA)         │
├─────────────────────────────────────────────────────────┤
│                   Scientific Layer                       │
│    NumPy | Pandas | SciPy | scikit-learn | matplotlib   │
├─────────────────────────────────────────────────────────┤
│                    Infrastructure                        │
│              Python 3.10+ | pip | venv                   │
└─────────────────────────────────────────────────────────┘
```

## 2. Core Dependencies

### 2.1 Deep Learning

| Package | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | ≥2.0.0 | Neural network framework |
| **Lightning** | ≥2.0.0 | Training orchestration |
| **torch_ema** | ≥0.3 | Exponential moving average |
| **einops** | ≥0.7.0 | Tensor operations (rearrange) |

**Why PyTorch?**
- Industry standard for research
- Dynamic computation graphs
- Excellent debugging (vs. TensorFlow)
- Strong community and documentation

**Why PyTorch Lightning?**
- Separates engineering from research code
- Handles distributed training, checkpointing, logging
- Reduces boilerplate significantly
- Reproducible training loops

**Why torch_ema?**
- Production-quality EMA implementation
- Handles edge cases (device movement, state dict)
- Used in TLOB original implementation

**Why einops?**
- Readable tensor reshaping
- `rearrange(x, 'b s f -> b f s')` vs. `x.permute(0, 2, 1)`
- Reduces dimension-related bugs

### 2.2 Scientific Computing

| Package | Version | Purpose |
|---------|---------|---------|
| **NumPy** | ≥1.26.0 | Array operations |
| **Pandas** | ≥2.0.0 | Data loading, preprocessing |
| **SciPy** | ≥1.12.0 | Statistical tests |
| **scikit-learn** | ≥1.4.0 | Metrics, classification |
| **matplotlib** | ≥3.8.0 | Visualization |

**Why these versions?**
- Recent stable releases
- Python 3.10+ compatibility
- Performance improvements in NumPy 1.26+
- Pandas 2.0 has improved type system

### 2.3 Version Pinning Strategy

```toml
# pyproject.toml
dependencies = [
    "numpy>=1.26.0",      # Minimum version
    "torch>=2.0.0",        # Minimum version
    "lightning>=2.0.0",    # Minimum version
]
```

**Why minimum versions (not exact)?**
- Allows bug fixes and security patches
- Reduces dependency conflicts
- pip resolves compatible versions

## 3. Project Structure

### 3.1 Package Layout

```
thesis/
├── src_prediction/                 # Main package
│   ├── __init__.py                 # Package exports
│   ├── __main__.py                 # Entry point (python -m)
│   ├── cli.py                      # CLI commands
│   ├── config.py                   # Configuration constants
│   ├── data/                       # Data handling
│   ├── models/                     # Neural networks
│   ├── training/                   # Training loops
│   ├── evaluation/                 # Metrics, analysis
│   └── utils/                      # Utilities
├── build/                          # egg-info, build artifacts (gitignored)
├── data/
│   ├── raw/                        # LOBSTER CSVs
│   └── preprocessed/               # NumPy arrays
├── checkpoints/                    # Saved models
├── results/                        # Experiment outputs
├── pyproject.toml                  # Package configuration
├── setup.cfg                       # egg_info: egg_base = build
└── README.md
```

### 3.2 Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `cli.py` | Command-line interface, argument parsing |
| `config.py` | All configuration constants |
| `data/` | Dataset, DataLoader, preprocessing |
| `models/` | Neural network architectures |
| `training/` | Training loops, Lightning module |
| `evaluation/` | Metrics, calibration, plots |
| `utils/` | Seeds, logging, helpers |

## 4. Development Tools

### 4.1 Code Quality

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **Ruff** | Linting + formatting | `pyproject.toml` |
| **MyPy** | Type checking | `pyproject.toml` |

**Ruff configuration:**
```toml
[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP"]
ignore = ["E501"]  # Line length handled by formatter
```

**Why Ruff?**
- 10-100x faster than Flake8 + isort + Black
- Single tool for linting and formatting
- Excellent defaults

### 4.2 Testing (Optional)

| Tool | Purpose |
|------|---------|
| **pytest** | Test framework |
| **pytest-cov** | Coverage reporting |

```bash
# Run tests (if implemented)
pytest tests/ -v --cov=src_prediction
```

## 5. Compute Requirements

### 5.1 Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 16 GB | 32+ GB |
| **GPU** | Not required | NVIDIA with 8+ GB VRAM |
| **Storage** | 10 GB | 50+ GB (for raw data) |

**Why GPU optional?**
- Models are relatively small (~2M params max)
- CPU training feasible for single-ticker experiments
- GPU provides 5-10x speedup

### 5.2 Training Time Estimates

| Model | Device | Time per Epoch | Full Training |
|-------|--------|----------------|---------------|
| DeepLOB | CPU | ~15 min | ~1.5 hours |
| DeepLOB | GPU | ~2 min | ~15 min |
| TLOB | CPU | ~20 min | ~2 hours |
| TLOB | GPU | ~3 min | ~20 min |
| LiT | CPU | ~30 min | ~3 hours |
| LiT | GPU | ~5 min | ~30 min |

*Estimates for CSCO dataset (~10M events)*

### 5.3 Memory Usage

| Model | Batch 128 GPU Memory | Peak RAM |
|-------|---------------------|----------|
| DeepLOB | ~2 GB | ~8 GB |
| TLOB | ~3 GB | ~10 GB |
| LiT | ~6 GB | ~12 GB |

## 6. Installation

### 6.1 Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install in development mode
pip install -e .

# Verify installation
python -m src_prediction --help
```

### 6.2 GPU Support

For CUDA support:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install PyTorch with CUDA (if needed)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 7. Execution

### 7.1 CLI Usage

```bash
# Preprocess data
python -m src_prediction preprocess --ticker CSCO

# Train model
python -m src_prediction train --ticker CSCO --epochs 10

# Full pipeline
python -m src_prediction run --ticker CSCO --model TLOB --decay
```

### 7.2 Programmatic Usage

```python
from src_prediction.models import TLOB, TLOBDecay
from src_prediction.data import lobster_load, LOBDataset
from src_prediction.training import train_model
from src_prediction.evaluation import compute_metrics

# Load data
train_x, train_y = lobster_load("data/preprocessed/CSCO/train.npy")
dataset = LOBDataset(train_x, train_y, seq_size=128)

# Create model
model = TLOBDecay(hidden_dim=46, num_layers=4, seq_size=128)

# Train
model, history = train_model(model, train_loader, val_loader, epochs=10)

# Evaluate
metrics = compute_metrics(y_true, y_pred, y_prob)
```

## 8. Output Management

### 8.1 Directory Structure

```
results/
└── CSCO_TLOB-Decay_20260128_143000/
    ├── TLOB-Decay_CSCO_best.pt        # Best model weights
    ├── training_history.png            # Loss/F1 curves
    ├── confusion_matrix_train.png
    ├── confusion_matrix_val.png
    ├── confusion_matrix_test.png
    ├── metrics.csv                     # Summary metrics
    ├── results.json                    # Detailed results
    └── analysis/
        ├── reliability_TLOB_Decay.png  # Calibration plot
        ├── decay_curves_TLOB_Decay.png # Decay visualization
        └── confidence_TLOB_Decay.json  # Confidence analysis
```

### 8.2 Checkpoint Format

PyTorch state dict format:
```python
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

**Why state_dict (not full model)?**
- Smaller file size
- Version-independent
- Explicit model instantiation

## 9. Reproducibility

### 9.1 Random Seeds

```python
from src_prediction.utils import set_seed
set_seed(42)
```

Affects:
- PyTorch (model init, dropout)
- NumPy (shuffling)
- Python random (any stdlib usage)
- CUDA (if available)

### 9.2 Determinism Settings

```python
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
```

**Note:** Full determinism (`torch.use_deterministic_algorithms(True)`) is not enabled due to:
- Performance overhead (10-20%)
- Some operations have no deterministic implementation

### 9.3 Version Tracking

Record versions in experiment logs:
```python
import torch
import lightning
print(f"PyTorch: {torch.__version__}")
print(f"Lightning: {lightning.__version__}")
print(f"CUDA: {torch.version.cuda}")
```

## 10. Limitations and Future Work

### 10.1 Current Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Single-GPU only | Slower for large experiments | Lightning supports multi-GPU |
| No experiment tracking | Manual result comparison | Could add W&B/MLflow |
| No hyperparameter tuning | Suboptimal configs possible | Could add Optuna |

### 10.2 Potential Enhancements

1. **W&B/MLflow integration** for experiment tracking
2. **Optuna** for hyperparameter optimization
3. **ONNX export** for production deployment
4. **Docker** containerization for reproducibility
5. **DVC** for data versioning
