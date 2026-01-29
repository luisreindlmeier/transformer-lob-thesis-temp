# Decay Attention Analysis

## 1. Research Hypothesis

**H2: Temporal Decay in Financial Attention**

> In limit order book prediction, attention patterns should exhibit temporal decay—recent events should receive higher attention weights than older events because market microstructure is inherently non-stationary and more recent information is more relevant for short-term price prediction.

This hypothesis motivates the learnable decay attention mechanism introduced in TLOB-Decay and LiT-Decay models.

## 2. Mathematical Foundation

### 2.1 Standard Attention

Standard scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

**Problem:** All positions compete equally for attention weight. Position 1 and position 128 have the same a priori importance.

### 2.2 Decay-Modulated Attention

Our decay attention introduces a distance-based mask:

```
Attention_decay(Q, K, V) = softmax((QK^T / √d_k) ⊙ M_λ) · V
```

**Where:**
```
M_λ[i,j] = exp(-λ · |i - j|)
```

- λ: Learnable decay rate (per attention head)
- |i - j|: Temporal distance between positions i and j
- ⊙: Element-wise multiplication

### 2.3 Effect on Attention Distribution

For a query at position i, the attention weight to position j is:

```
a[i,j] ∝ exp(q_i · k_j / √d_k) · exp(-λ · |i - j|)
       = exp(q_i · k_j / √d_k - λ · |i - j|)
```

**Interpretation:**
- Content similarity: `q_i · k_j` (standard attention)
- Distance penalty: `-λ · |i - j|` (decay)
- Combined: Similar content + close positions → high attention

### 2.4 Decay Function Properties

The exponential decay function exp(-λd) has desirable properties:

| Distance (d) | λ = 0.1 | λ = 0.2 | λ = 0.5 |
|--------------|---------|---------|---------|
| 0 | 1.000 | 1.000 | 1.000 |
| 10 | 0.368 | 0.135 | 0.007 |
| 50 | 0.007 | 0.000 | 0.000 |
| 100 | 0.000 | 0.000 | 0.000 |

**Key observations:**
- λ = 0: No decay → standard attention
- λ → ∞: Only self-attention (position i attends only to i)
- Intermediate λ: Smooth transition from local to global

## 3. Learning Dynamics

### 3.1 Initialization

Decay rates are initialized to λ_init = 0.1 (medium decay):

```python
init_raw = torch.log(torch.exp(torch.tensor(0.1)) - 1)  # ≈ -2.25
lambda_raw = nn.Parameter(torch.full((n_heads,), init_raw))
```

**Why 0.1?**
- Allows model to learn both faster and slower decay
- Corresponds to moderate distance penalty
- Empirically stable during training

### 3.2 Gradient Flow

The gradient of the loss w.r.t. λ_raw:

```
∂L/∂λ_raw = ∂L/∂a · ∂a/∂M · ∂M/∂λ · ∂λ/∂λ_raw
```

**Where:**
- ∂λ/∂λ_raw = sigmoid(λ_raw) (softplus derivative)
- ∂M/∂λ = -d · exp(-λd) (distance-weighted)

**Implication:** Gradient magnitude depends on typical attention distances. Heads attending to distant positions receive larger gradient signals for λ.

### 3.3 Convergence Behavior

Typical learning trajectory:

```
Epoch 1:  λ ≈ 0.10 (initialization)
Epoch 5:  λ ∈ [0.05, 0.20] (differentiation begins)
Epoch 10: λ ∈ [0.03, 0.50] (specialization)
Epoch 20: λ stabilizes (convergence)
```

Different heads converge to different decay rates, suggesting specialized roles.

## 4. Analysis Implementation

### 4.1 Extracting Decay Rates

```python
from src_prediction.evaluation import analyze_model_decay

# From trained model
model = TLOBDecay(...)
model.load_state_dict(torch.load("checkpoint.pt"))
results = analyze_model_decay(model, output_dir="./analysis", model_name="TLOB-Decay")

# Output: decay_rates, summary statistics, visualization
```

### 4.2 Interpretation Function

```python
def interpret_lambda(lam: float) -> str:
    if lam < 0.05: return "Very slow (global)"      # Attends globally
    if lam < 0.1:  return "Slow (long-range)"       # Long-range dependencies
    if lam < 0.2:  return "Medium"                  # Balanced
    if lam < 0.5:  return "Fast (local)"            # Local patterns
    return "Very fast (recent)"                      # Immediate context only
```

### 4.3 Visualization

Decay curves show the attention weight multiplier as a function of distance:

```python
def plot_decay_curves(decay_rates, model_name, save_path):
    distances = np.arange(100)
    for layer, heads in decay_rates.items():
        for head, lam in heads.items():
            plt.plot(distances, np.exp(-lam * distances), 
                    label=f"{head} (λ={lam:.3f})")
```

## 5. Expected Findings

### 5.1 Layer-Wise Patterns

**Early layers (close to input):**
- Expected: Faster decay (larger λ)
- Rationale: Early layers capture local patterns, recent context

**Later layers (close to output):**
- Expected: Slower decay (smaller λ)
- Rationale: Later layers aggregate global information

### 5.2 Feature vs. Temporal Attention (TLOB-Decay)

TLOB alternates between feature and temporal attention:

**Feature attention layers:**
- Attend across features at same timestep
- Decay across "feature distance" (less meaningful)
- Expected: Lower λ (more global)

**Temporal attention layers:**
- Attend across time at same feature
- Decay across temporal distance (meaningful)
- Expected: Higher λ (more local)

### 5.3 Multi-Head Specialization (LiT-Decay)

With 8 attention heads, expect diverse specialization:

| Head Type | Expected λ | Role |
|-----------|------------|------|
| Global head | λ < 0.05 | Captures overall trend |
| Long-range head | λ ≈ 0.1 | Captures momentum patterns |
| Medium-range head | λ ≈ 0.2 | Captures volatility clustering |
| Local head | λ > 0.3 | Captures immediate bid-ask dynamics |

## 6. Financial Interpretation

### 6.1 Market Microstructure Connection

Learned decay rates can reveal market dynamics:

**Fast decay (λ > 0.3):**
- Suggests short-lived information
- Consistent with high-frequency noise
- Relevant for: Order flow imbalance, immediate supply/demand

**Medium decay (0.1 < λ < 0.3):**
- Suggests medium-term persistence
- Consistent with momentum/mean-reversion
- Relevant for: Price trend, volatility clustering

**Slow decay (λ < 0.1):**
- Suggests long-range dependencies
- Consistent with regime persistence
- Relevant for: Overall market state, liquidity level

### 6.2 Comparison to Known Market Dynamics

| Phenomenon | Expected Decay | λ Range |
|------------|----------------|---------|
| Order book imbalance | Fast | 0.3-0.5 |
| Momentum | Medium | 0.1-0.2 |
| Volatility clustering | Medium | 0.1-0.3 |
| Market regime | Slow | 0.01-0.1 |

## 7. Ablation Studies

### 7.1 Fixed vs. Learned Decay

Compare models with:
1. **No decay** (standard attention)
2. **Fixed decay** (λ = 0.1, not learned)
3. **Learned decay** (λ initialized to 0.1, learned)

**Expected result:** Learned decay should outperform fixed decay, which should outperform no decay, validating the value of adaptive temporal weighting.

### 7.2 Initialization Sensitivity

Train with different λ_init values:
- λ_init = 0.01 (slow start)
- λ_init = 0.1 (moderate start)
- λ_init = 0.5 (fast start)

**Expected result:** Models converge to similar final λ distributions, showing robustness to initialization.

### 7.3 Per-Head vs. Global Decay

Compare:
1. **Per-head λ** (current implementation)
2. **Per-layer λ** (all heads share one λ)
3. **Global λ** (entire model shares one λ)

**Expected result:** Per-head provides most flexibility and best performance.

## 8. Output Format

The decay analysis produces:

### 8.1 Numeric Summary

```json
{
  "decay_rates": {
    "Layer 1": {"Head 1": 0.12, "Head 2": 0.08},
    "Layer 2": {"Head 1": 0.15, "Head 2": 0.22}
  },
  "summary": {
    "min_lambda": 0.08,
    "max_lambda": 0.22,
    "mean_lambda": 0.14,
    "std_lambda": 0.05,
    "n_heads": 4
  }
}
```

### 8.2 Visualization

Generated plots:
1. **Decay curves**: exp(-λd) for each head
2. **λ distribution**: Histogram of learned decay rates
3. **Layer-wise comparison**: Box plots by layer

### 8.3 Console Output

```
==========================================================
  H2 Decay Analysis: TLOB-Decay
==========================================================

  Layer        Head            lambda              Interpretation
  ------------------------------------------------------------
  Layer 1      Head 1          0.1234                      Medium
  Layer 2      Head 1          0.0567         Slow (long-range)
  Layer 3      Head 1          0.3456              Fast (local)
  ...
```

## 9. Code Example

```python
from src_prediction.models import TLOBDecay
from src_prediction.evaluation import analyze_model_decay
from pathlib import Path

# Train model
model = TLOBDecay(hidden_dim=46, num_layers=4, seq_size=128, init_decay=0.1)
# ... training code ...

# Analyze decay after training
results = analyze_model_decay(
    model=model,
    output_dir=Path("./results/decay_analysis"),
    model_name="TLOB-Decay-CSCO"
)

# Access learned rates
for layer, heads in results["decay_rates"].items():
    print(f"{layer}:")
    for head, lam in heads.items():
        print(f"  {head}: λ = {lam:.4f}")

# Summary statistics
print(f"Mean λ: {results['summary']['mean_lambda']:.4f}")
print(f"Std λ: {results['summary']['std_lambda']:.4f}")
```
