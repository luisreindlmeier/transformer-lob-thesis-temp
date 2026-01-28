# Hyperparameters and Configuration

## 1. Parameter Categories

Parameters are divided into three categories:

| Category | Description | Examples |
|----------|-------------|----------|
| **Fixed** | Determined by data/problem structure | Sequence length, n_features |
| **Preset** | Set based on prior work, not tuned | Learning rate, batch size |
| **Learned** | Optimized during training | Model weights, decay rates |

## 2. Data Parameters (Fixed)

These parameters are determined by the LOBSTER dataset structure:

| Parameter | Value | Source | Rationale |
|-----------|-------|--------|-----------|
| `LEN_ORDER` | 6 | Dataset | Number of order-level features |
| `LEN_LEVEL` | 4 | Dataset | Features per LOB level (ask_p, ask_s, bid_p, bid_s) |
| `N_LOB_LEVELS` | 10 | Dataset | LOB depth (Level 2 data) |
| `N_FEATURES` | 46 | 6 + 40 | Total input features |
| `LOBSTER_HORIZONS` | [10, 20, 50, 100] | Convention | Standard prediction horizons |
| `LEN_SMOOTH` | 10 | TLOB paper | Label smoothing window |

**Why 10 LOB levels?**
- Standard in literature (DeepLOB, TLOB)
- Captures meaningful depth beyond best bid/ask
- Deeper levels add diminishing information

**Why 10-event smoothing?**
- Reduces label noise from bid-ask bounce
- TLOB paper demonstrates effectiveness
- Balances noise reduction vs. information loss

## 3. Model Architecture Parameters (Preset)

### 3.1 TLOB Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `HIDDEN_DIM` | 46 | Matches input dimension (no projection) |
| `NUM_LAYERS` | 4 | 4 alternating pairs = 8 total layers |
| `NUM_HEADS` | 1 | Single head for interpretability |
| `SEQ_SIZE` | 128 | ~1-2 seconds of market history |
| `IS_SIN_EMB` | True | Sinusoidal generalizes to unseen lengths |

**Why hidden_dim = 46?**
- TLOB paper design: hidden dimension matches feature dimension
- Alternating attention operates on both dimensions
- Avoids information loss from projection

**Why num_heads = 1?**
- TLOB paper uses single head
- Enables direct attention weight interpretation
- Multi-head showed no improvement in original work

**Why seq_size = 128?**
- Power of 2 for efficient computation
- ~1-2 seconds of market history
- Balances context length vs. computational cost
- Consistent with prior work

### 3.2 LiT Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `d_model` | 256 | Standard transformer width |
| `n_heads` | 8 | Standard for d_model=256 |
| `n_layers` | 6 | ViT-Base uses 12; we use smaller |
| `dropout` | 0.2 | Regularization for financial data |
| `dim_feedforward` | 1024 | 4 × d_model (standard) |

**Why d_model = 256?**
- Sufficient capacity for 46 input features
- Standard in ViT-style models
- Allows 8 heads with d_head = 32

**Why n_heads = 8?**
- Standard for d_model = 256
- Allows multi-head specialization
- Each head: d_head = 32 (sufficient for LOB patterns)

**Why dropout = 0.2?**
- Financial data is noisy
- Prevents overfitting to spurious patterns
- Higher than NLP (0.1) due to data characteristics

### 3.3 DeepLOB Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Conv1 filters | 32 | Zhang et al. (2019) |
| Conv2 filters | 64 | Zhang et al. (2019) |
| GRU hidden | 128 | Zhang et al. (2019) |
| GRU layers | 1 | Sufficient for 128-length sequences |

**Why follow original DeepLOB?**
- Fair baseline comparison
- Proven effective architecture
- Modifications would confound comparisons

### 3.4 Decay Attention Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `init_decay` | 0.1 | Moderate initial decay |
| Parameterization | Softplus | Ensures λ > 0, smooth gradients |

**Why init_decay = 0.1?**
- Corresponds to ~37% weight at distance 10
- Neither too slow (global) nor too fast (local)
- Allows learning in both directions

**Why softplus parameterization?**
- Ensures positivity: λ = log(1 + exp(λ_raw)) > 0
- Smooth gradients (unlike ReLU)
- Stable optimization

## 4. Training Parameters (Preset)

### 4.1 Optimization

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `LR` | 0.0001 | Conservative for transformers |
| Optimizer | Adam | Standard for deep learning |
| `eps` | 1e-8 | Adam numerical stability |
| Weight decay | 0 | Not used; dropout sufficient |

**Why LR = 0.0001?**
- Transformers are sensitive to learning rate
- 1e-4 is conservative but stable
- Higher rates (1e-3) cause divergence

**Why no weight decay?**
- Dropout provides regularization
- Weight decay can hurt attention learning
- TLOB paper does not use it

### 4.2 Batch and Epochs

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `BATCH_SIZE` | 128 | Balances memory and gradient quality |
| `MAX_EPOCHS` | 10 | Early stopping typically triggers earlier |
| `patience` | 5 | Early stopping patience |

**Why batch_size = 128?**
- Large enough for stable gradient estimates
- Small enough to fit in GPU memory
- Power of 2 for efficiency

**Why max_epochs = 10?**
- Early stopping typically triggers at epoch 3-5
- 10 epochs provides margin
- Avoids wasted computation

### 4.3 Learning Rate Scheduling (Lightning Trainer)

Manual scheduling based on validation loss:

```python
if val_loss < min_loss:
    if improvement < 0.002:
        lr = lr / 2  # Small improvement → halve LR
    min_loss = val_loss
else:
    lr = lr / 2  # No improvement → halve LR
```

**Why this schedule?**
- Aggressive reduction on plateaus
- Threshold (0.002) prevents premature reduction
- Effective for short training runs

### 4.4 Exponential Moving Average (EMA)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| EMA decay | 0.999 | Slow averaging for stability |

**Why EMA?**
- Smooths noisy gradient updates
- Improves generalization
- Critical for reproducibility (TLOB paper)

**Implementation:**
```python
self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
# Training: self.ema.update() after each step
# Validation: with self.ema.average_parameters(): ...
```

### 4.5 Early Stopping

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Monitor | `val_loss` | Primary validation metric |
| Mode | `min` | Lower is better |
| Patience | 1 or 5 | Epochs without improvement |
| Min delta | 0.002 | Minimum improvement threshold |

**Why patience = 1 (Lightning)?**
- Aggressive early stopping
- Combined with LR halving
- Prevents overfitting

**Why min_delta = 0.002?**
- Ignores negligible improvements
- Prevents false "improvement" from noise
- Approximately 0.2% loss reduction

## 5. Data Split Parameters (Fixed)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `SPLIT_RATES` | [0.85, 0.05, 0.10] | Train/Val/Test |
| Split method | Temporal | No data leakage |

**Why 85/5/10 split?**
- Maximizes training data (85%)
- Small validation (5%) because we use early stopping
- Sufficient test data (10%) for statistical significance
- ~1.2M test samples for CSCO

**Why temporal split?**
- Financial data is non-stationary
- Random splits leak future information
- Real deployment sees only past data

## 6. Inference Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | Learned | Calibration via temperature scaling |
| Confidence threshold | Grid search | Optimal trade-off |

**Temperature scaling:**
- Learned on validation set
- Typically T ∈ [0.8, 1.5]
- T > 1: Reduces overconfidence

## 7. Computational Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `DEVICE` | Auto (cuda/cpu) | Use GPU if available |
| `PRECISION` | 32 | Full precision for stability |
| `num_workers` | 4 | DataLoader parallelism |

**Why precision = 32?**
- Mixed precision (16) can cause instability
- Memory not a bottleneck for these models
- Reproducibility priority

## 8. Parameter Summary Table

| Component | Parameter | Value | Tunable? |
|-----------|-----------|-------|----------|
| **Data** | seq_size | 128 | No (fixed) |
| | n_features | 46 | No (fixed) |
| | horizon | 10 | Yes (CLI arg) |
| **TLOB** | hidden_dim | 46 | No (preset) |
| | num_layers | 4 | No (preset) |
| | num_heads | 1 | No (preset) |
| **LiT** | d_model | 256 | No (preset) |
| | n_heads | 8 | No (preset) |
| | n_layers | 6 | No (preset) |
| **Decay** | init_decay | 0.1 | Yes (learned) |
| **Training** | lr | 0.0001 | No (preset) |
| | batch_size | 128 | Yes (CLI arg) |
| | max_epochs | 10 | Yes (CLI arg) |
| | patience | 5 | No (preset) |
| **EMA** | decay | 0.999 | No (preset) |

## 9. CLI Configuration

All configurable parameters are exposed via CLI:

```bash
python -m lob_prediction run \
    --ticker CSCO \
    --model TLOB \
    --decay \
    --epochs 10 \
    --batch-size 128 \
    --lr 0.0001 \
    --patience 5 \
    --horizon 10 \
    --data-fraction 1.0 \
    --seed 42
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--ticker` | Required | Stock ticker |
| `--model` | TLOB | Model architecture |
| `--decay` | False | Use decay attention |
| `--epochs` | 10 | Max training epochs |
| `--batch-size` | 128 | Training batch size |
| `--lr` | 0.0001 | Learning rate |
| `--patience` | 5 | Early stopping patience |
| `--horizon` | 10 | Prediction horizon |
| `--data-fraction` | 1.0 | Fraction of data to use |
| `--seed` | 42 | Random seed |

## 10. Reproducibility

**Fixed random seeds:**
```python
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

**Deterministic operations:**
```python
torch.set_float32_matmul_precision('high')
```

**Note:** Full determinism requires `torch.use_deterministic_algorithms(True)`, which is not enabled due to performance overhead.
