# Calibration and Confidence Analysis

## 1. Research Hypothesis

**H3: Confidence-Based Trade Selection**

> Well-calibrated model confidence scores can improve trading performance by filtering out low-confidence predictions, resulting in higher precision trades even at reduced coverage.

This hypothesis motivates comprehensive calibration analysis and confidence-based strategy evaluation.

## 2. Calibration Fundamentals

### 2.1 What is Calibration?

A classifier is **calibrated** if its predicted probabilities match empirical frequencies:

```
P(y = c | p(y = c) = p) = p
```

**Example:**
- If model predicts 70% confidence for "Up" across 1000 samples
- A calibrated model should have ~700 correct predictions among those samples

### 2.2 Why Calibration Matters for Trading

**Overconfident models:**
- Predict high probability when actually uncertain
- Lead to excessive trading, higher transaction costs
- Poor risk management

**Underconfident models:**
- Predict low probability even when correct
- Lead to missed opportunities
- Conservative but suboptimal

**Well-calibrated models:**
- Enable rational decision-making
- Allow confidence-based position sizing
- Support risk-adjusted strategy optimization

## 3. Calibration Metrics

### 3.1 Expected Calibration Error (ECE)

ECE measures the average gap between confidence and accuracy across bins:

```
ECE = Σ (|B_m| / N) · |acc(B_m) - conf(B_m)|
```

**Where:**
- B_m: Set of samples in confidence bin m
- acc(B_m): Accuracy of predictions in bin m
- conf(B_m): Average confidence in bin m
- N: Total samples

**Implementation:**
```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += np.sum(in_bin) / len(y_true) * np.abs(bin_acc - bin_conf)
    return ece
```

**Interpretation:**
- ECE = 0: Perfect calibration
- ECE = 0.1: Average 10% gap between confidence and accuracy
- Lower is better

### 3.2 Maximum Calibration Error (MCE)

MCE measures the worst-case calibration gap:

```
MCE = max_m |acc(B_m) - conf(B_m)|
```

**Why MCE?**
- ECE can hide large errors in sparse bins
- MCE identifies worst-case scenarios
- Important for risk management

### 3.3 Brier Score

Proper scoring rule for probabilistic predictions:

```
Brier = (1/N) Σ Σ (p_c - y_c)²
```

**Where:**
- p_c: Predicted probability for class c
- y_c: 1 if true class is c, else 0

**Properties:**
- Brier = 0: Perfect predictions
- Brier = 1: Worst possible
- Decomposes into calibration + refinement + uncertainty

### 3.4 Reliability Diagram

Visual representation of calibration:

```
┌────────────────────────────┐
│                     ▲      │
│                   ▲   ▲    │ Perfect
│                ▲    /      │ calibration
│             ▲    /         │ (diagonal)
│          ▲    /            │
│       ▲    /   Actual      │
│    ▲    /      reliability │
│ ▲    /                     │
│──/─────────────────────────│
│ Confidence                  │
└────────────────────────────┘
```

**Above diagonal:** Overconfident
**Below diagonal:** Underconfident
**On diagonal:** Well-calibrated

## 4. Temperature Scaling

### 4.1 Method

Temperature scaling is a post-hoc calibration technique:

```
p_calibrated = softmax(z / T)
```

**Where:**
- z: Model logits (pre-softmax)
- T: Temperature parameter (learned on validation set)
- T > 1: Softens probabilities (reduces overconfidence)
- T < 1: Sharpens probabilities (increases confidence)

### 4.2 Learning Temperature

Temperature is optimized to minimize NLL on validation set:

```python
class TemperatureScaler(nn.Module):
    def __init__(self, model):
        self.temperature = nn.Parameter(torch.ones(1))
    
    def set_temperature(self, val_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature])
        
        def closure():
            loss = criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
```

### 4.3 Why Temperature Scaling?

**Advantages:**
- Simple: Single parameter to learn
- Preserves accuracy: Only scales probabilities, doesn't change predictions
- Effective: Works well for neural networks
- Fast: LBFGS converges in ~50 iterations

**Limitation:**
- Class-agnostic: Same temperature for all classes
- Can be extended to per-class temperature (rarely needed)

## 5. Confidence-Based Analysis

### 5.1 Confidence-Accuracy Correlation

**Spearman correlation** between confidence and correctness:

```python
corr, _ = spearmanr(confidences, (predictions == y_true).astype(float))
```

**Interpretation:**
- ρ > 0: Higher confidence → higher accuracy (desirable)
- ρ ≈ 0: Confidence uninformative
- ρ < 0: Higher confidence → lower accuracy (miscalibrated)

**Expected range:** ρ ∈ [0.1, 0.4] for well-trained classifiers

### 5.2 Binned F1 Analysis

F1 score computed within confidence bins:

```python
def confidence_binned_metrics(y_true, y_pred, y_prob, n_bins=10):
    confidences = np.max(y_prob, axis=1)
    bin_f1s = []
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if np.sum(in_bin) >= 3:
            bin_f1 = f1_score(y_true[in_bin], y_pred[in_bin], average='macro')
            bin_f1s.append(bin_f1)
```

**Expected pattern:**
- F1 should increase monotonically with confidence
- Non-monotonic patterns indicate calibration issues

### 5.3 Monotonicity Check

A model is **monotonically calibrated** if:

```
conf(B_1) < conf(B_2) → acc(B_1) ≤ acc(B_2)
```

**Implementation:**
```python
monotonic = np.sum(np.diff(bin_f1s) >= -0.01) >= len(bin_f1s) * 0.7 - 1
```

We allow small violations (0.01 tolerance) in 30% of bins for robustness.

## 6. Confidence Thresholding

### 6.1 Trade-off: Coverage vs. Precision

Filtering predictions by confidence threshold τ:

```
Predictions_filtered = {p | confidence(p) ≥ τ}
```

| Threshold τ | Coverage | Expected F1 |
|-------------|----------|-------------|
| 0.33 | 100% | Baseline |
| 0.50 | ~70% | Higher |
| 0.70 | ~40% | Highest |
| 0.90 | ~10% | Very high (if calibrated) |

### 6.2 Optimal Threshold Selection

Find threshold maximizing F1 while maintaining minimum coverage:

```python
def optimal_confidence_threshold(y_true, y_pred, y_prob, min_coverage=0.1):
    best_threshold, best_f1 = 0.33, 0.0
    for thresh in np.linspace(0.33, 0.95, 50):
        mask = confidences >= thresh
        coverage = np.mean(mask)
        if coverage < min_coverage:
            break
        f1 = f1_score(y_true[mask], y_pred[mask], average='macro')
        if f1 > best_f1:
            best_f1, best_threshold = f1, thresh
    return best_threshold, best_f1
```

### 6.3 Trading Strategy Implications

| Strategy | Threshold | Interpretation |
|----------|-----------|----------------|
| Aggressive | τ = 0.33 | Trade all signals |
| Moderate | τ = 0.50 | Trade confident signals |
| Conservative | τ = 0.70 | Trade high-confidence only |
| Ultra-conservative | τ = 0.90 | Rare, high-conviction trades |

## 7. Backtesting with Confidence

### 7.1 Strategy Definition

Simple directional strategy based on predictions:

```python
def strategy_returns(y_pred, y_ret):
    strat_ret = np.zeros_like(y_ret)
    strat_ret[y_pred == 2] = y_ret[y_pred == 2]   # Long on Down (price will rise)
    strat_ret[y_pred == 0] = -y_ret[y_pred == 0]  # Short on Up (price will fall)
    # y_pred == 1 (Stationary) → no position
    return strat_ret
```

**Note:** Label encoding: 0=Up, 2=Down. Strategy goes:
- Long when predicting Down (expecting price recovery)
- Short when predicting Up (expecting price drop)

### 7.2 Sharpe Ratio

Risk-adjusted performance metric:

```
Sharpe = mean(returns) / std(returns)
```

**Annualization:** For high-frequency returns, multiply by √(252 × n_events_per_day)

### 7.3 Filtered Strategy Analysis

Evaluate strategy performance at different confidence thresholds:

```python
def filtered_strategy_analysis(y_true, y_pred, y_prob, y_ret, thresholds):
    for thresh in thresholds:
        mask = confidences >= thresh
        filtered_returns = strategy_returns(y_pred[mask], y_ret[mask])
        sharpe = compute_sharpe_ratio(filtered_returns)
        coverage = np.mean(mask)
        n_trades = np.sum(y_pred[mask] != 1)
```

**Expected finding:** Sharpe ratio improves with higher thresholds (up to a point), demonstrating the value of confidence-based filtering.

## 8. Analysis Pipeline

### 8.1 Full Analysis Function

```python
def analyze_confidence(model, val_loader, test_loader, y_ret_test, output_dir):
    # 1. Learn temperature on validation set
    scaler = TemperatureScaler(model)
    optimal_temp = scaler.set_temperature(val_loader)
    
    # 2. Get calibrated probabilities on test set
    probs_uncal = softmax(logits)
    probs_cal = softmax(logits / optimal_temp)
    
    # 3. Compute calibration metrics
    ece_uncal = expected_calibration_error(y_true, probs_uncal)
    ece_cal = expected_calibration_error(y_true, probs_cal)
    
    # 4. Analyze confidence-accuracy relationship
    binned_metrics = confidence_binned_metrics(y_true, y_pred, probs_cal)
    
    # 5. Generate reliability diagram
    plot_reliability_diagram(y_true, probs_cal, model_name, save_path)
    
    # 6. Backtest with confidence filtering
    backtest_results = filtered_strategy_analysis(y_true, y_pred, probs_cal, y_ret)
    
    return results
```

### 8.2 Output Structure

```json
{
  "optimal_temperature": 1.23,
  "calibration": {
    "uncalibrated": {"ece": 0.15, "mce": 0.25, "brier": 0.42},
    "calibrated": {"ece": 0.08, "mce": 0.12, "brier": 0.38}
  },
  "binned_metrics": {
    "correlation": 0.32,
    "monotonic": true
  },
  "backtesting": {
    "baseline_sharpe": 0.45,
    "optimal_threshold": 0.65,
    "optimal_sharpe": 0.78,
    "thresholds": [0.33, 0.40, ...],
    "sharpe_ratios": [0.45, 0.52, ...],
    "coverages": [1.0, 0.85, ...]
  }
}
```

## 9. Expected Findings

### 9.1 Calibration Improvement

| Metric | Uncalibrated | Calibrated | Improvement |
|--------|--------------|------------|-------------|
| ECE | 0.10-0.20 | 0.05-0.10 | 50%+ |
| MCE | 0.20-0.35 | 0.10-0.20 | 40%+ |
| Brier | 0.40-0.50 | 0.35-0.45 | 10-20% |

### 9.2 Confidence-Performance Correlation

| Model | Expected ρ | Interpretation |
|-------|-----------|----------------|
| DeepLOB | 0.15-0.25 | Moderate correlation |
| TLOB | 0.20-0.30 | Good correlation |
| TLOB-Decay | 0.25-0.35 | Strong correlation |
| LiT | 0.20-0.30 | Good correlation |
| LiT-Decay | 0.25-0.35 | Strong correlation |

### 9.3 Sharpe Ratio Improvement

At optimal threshold (τ ≈ 0.6-0.7):
- Expected Sharpe improvement: 30-70% over baseline
- Expected coverage: 30-50%
- Expected F1 improvement: 5-15%

## 10. Limitations

1. **Transaction costs not modeled:** Real trading incurs costs per trade
2. **Execution assumptions:** Perfect execution at mid-price assumed
3. **Single-asset strategy:** No portfolio effects considered
4. **Horizon mismatch:** Strategy horizon may differ from prediction horizon

## 11. Code Example

```python
from src_prediction.evaluation import analyze_confidence
from src_prediction.data import lobster_load, compute_returns

# Load data
test_x, test_y, test_midprice = lobster_load("test.npy", return_midprice=True)
test_returns = compute_returns(test_midprice, horizon=10)

# Analyze confidence
results = analyze_confidence(
    model=trained_model,
    val_loader=val_loader,
    test_loader=test_loader,
    y_ret_test=test_returns,
    output_dir="./results/confidence",
    model_name="TLOB-Decay"
)

print(f"Temperature: {results['optimal_temperature']:.3f}")
print(f"ECE improvement: {results['calibration']['uncalibrated']['ece']:.4f} → {results['calibration']['calibrated']['ece']:.4f}")
print(f"Sharpe improvement: {results['backtesting']['baseline_sharpe']:.3f} → {results['backtesting']['optimal_sharpe']:.3f}")
```
