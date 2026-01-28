# Data Preprocessing Pipeline

## 1. Pipeline Overview

The preprocessing pipeline transforms raw LOBSTER CSV files into model-ready NumPy arrays. The pipeline ensures:

1. **Temporal integrity**: Data remains chronologically ordered
2. **Normalization**: Features are z-score normalized for stable training
3. **Label generation**: Multi-horizon labels computed with smoothing
4. **Split isolation**: Training statistics never leak to validation/test sets

```
Raw CSVs → Event Filtering → Feature Engineering → Normalization → Labeling → NumPy Arrays
```

## 2. Event Filtering

### 2.1 Retained Events

Only events that directly impact the visible order book are retained:

| Event Type | Name | Included | Rationale |
|------------|------|----------|-----------|
| 1 | New Order | ✓ | New liquidity added to order book |
| 2 | Partial Cancel | ✗ | Modifies existing order, no structural change |
| 3 | Total Cancel | ✓ | Liquidity removed from order book |
| 4 | Execution | ✓ | Trade occurred, liquidity consumed |
| 5 | Hidden Execution | ✗ | Hidden liquidity not observable |
| 6 | Cross Trade | ✗ | Auction mechanism, different dynamics |

**Implementation:**
```python
indexes_to_drop = dataframes[0][dataframes[0]["event_type"].isin([2, 5, 6, 7])].index
```

**Why filter these events?**
- Events 2, 5, 6  either don't change the visible order book or represent non-standard market mechanisms
- Keeping only visible, structural changes ensures the model learns from actionable market information
- Reduces noise and improves signal-to-noise ratio


## 3. Feature Engineering

### 3.1 Input Features (46 dimensions)

The model receives 46 features per event, combining order-level and LOB-level information:

#### Order Features (6 dimensions)

| Index | Feature | Description | Transformation |
|-------|---------|-------------|----------------|
| 0 | `time` | Inter-event time delta | Z-score normalized |
| 1 | `event_type` | Order event type
| 2 | `size` | Order volume | Z-score normalized |
| 3 | `price` | Order price | Z-score normalized |
| 4 | `direction` | Buy (+1) or Sell (-1) | Sign-adjusted for executions |
| 5 | `depth` | Price levels from BBO | Computed, Z-score normalized |


#### LOB Features (40 dimensions)

| Indices | Description |
|---------|-------------|
| 6-25 | Ask side: [price₁, size₁, price₂, size₂, ..., price₁₀, size₁₀] |
| 26-45 | Bid side: [price₁, size₁, price₂, size₂, ..., price₁₀, size₁₀] |

**Why 10 levels?**
- Captures queue dynamics beyond best bid/offer
- Shows liquidity depth and potential support/resistance
- Standard in literature (DeepLOB, TLOB)
- Diminishing information value beyond 10 levels

### 3.2 Depth Calculation

The `depth` feature measures how far an order's price is from the best bid/offer:

```python
if direction == 1:  # Buy order
    depth = (best_bid - order_price) / tick_size
else:  # Sell order
    depth = (order_price - best_ask) / tick_size
```

**Interpretation:**
- `depth = 0`: Order at best bid/offer (aggressive)
- `depth > 0`: Order deeper in the book (passive)
- Higher depth → less likely to be executed immediately

**Why compute depth?**
- Order aggressiveness is a key predictor of short-term price movements
- Aggressive orders (depth=0) signal urgency and often precede price changes
- Depth normalizes across different price levels

### 3.3 Time Transformation

Raw timestamps are converted to inter-event time deltas:


**Why time deltas?**
- Absolute timestamps have no predictive value
- Time deltas capture market activity rhythm
- Short deltas indicate high activity periods (potentially more volatility)
- Long deltas indicate quiet periods

### 3.4 Direction Sign Adjustment

For execution events (type 4), the direction sign is inverted:

```python
direction = direction * (-1 if event_type == 4 else 1)
```

**Why?**
- Execution events represent the *aggressor* side
- A buy execution means someone *sold* to a buyer's limit order
- Inverting preserves the information flow direction

## 4. Normalization

### 4.1 Z-Score Normalization

All continuous features are z-score normalized:

```
x_normalized = (x - μ) / σ
```

**Where:**
- μ = mean computed from training data only
- σ = standard deviation computed from training data only

**Implementation:**
```python
def z_score_orderbook(data, mean_size=None, std_size=None, mean_prices=None, std_prices=None):
    if mean_size is None:
        mean_size = data.iloc[:, 1::2].stack().mean()  # All size columns
        std_size = data.iloc[:, 1::2].stack().std()
    # Apply normalization...
```

### 4.2 Separate Statistics for Prices and Sizes

Prices and sizes are normalized with separate statistics:

| Feature Group | Statistics |
|---------------|------------|
| Prices (order + LOB) | Computed jointly across all price columns |
| Sizes (order + LOB) | Computed jointly across all size columns |
| Time deltas | Separate μ and σ |
| Depth | Separate μ and σ |

**Why separate statistics?**
- Prices and sizes have fundamentally different distributions
- Prices are typically normally distributed around a mean
- Sizes follow a heavy-tailed distribution (many small, few large orders)
- Joint normalization ensures consistent scaling within each feature type

### 4.3 Train-Only Statistics

**Critical:** Normalization statistics are computed **only from training data**:

**Why?**
- Prevents data leakage from future observations
- Simulates real-world deployment where future statistics are unknown
- Ensures honest evaluation on test set

## 5. Label Generation

### 5.1 Smoothed Mid-Price

Labels are based on smoothed mid-prices, not raw mid-prices:

```python
def labeling(X, len_smooth, h):
    # Sliding window average for smoothing
    previous_mid = sliding_window_mean(midprice[t-len_smooth:t])
    future_mid = sliding_window_mean(midprice[t+h-len_smooth:t+h])
```

**Parameters:**
- `len_smooth = 10`: Window size for smoothing
- `h`: Prediction horizon (10, 20, 50, or 100 events)

**Why smoothing?**
- Raw mid-prices are noisy (bid-ask bounce)
- Smoothing captures the underlying trend, not microstructure noise
- Window of 10 events is empirically effective (TLOB paper)


## 6. Temporal Split

### 6.1 Split Ratios

| Split | Days | Ratio | Purpose |
|-------|------|-------|---------|
| Train | Days 1-17 | 85% | Model learning |
| Validation | Day 18 | 5% | Early stopping, hyperparameter selection |
| Test | Days 19-20 | 10% | Final evaluation |

**Implementation:**
```python
SPLIT_RATES = [0.85, 0.05, 0.10]
train_days = int(num_days * 0.85)
val_days = int(num_days * 0.05)
test_days = num_days - train_days - val_days
```

### 6.2 Why Temporal (Not Random) Split?

**Random splits would cause data leakage:**
- Adjacent events are highly correlated
- Random sampling would mix training and test events from the same market conditions
- Model would see "future" patterns during training

**Temporal splits ensure:**
- Training data is strictly before validation/test
- Model must generalize to unseen market conditions
- Evaluation reflects real deployment scenario

## 7. Output Format

### 7.1 NumPy Array Structure

Output shape: `(N, 50)` where N = number of valid samples

| Columns | Content |
|---------|---------|
| 0-5 | Order features: [time, event_type, size, price, direction, depth] |
| 6-45 | LOB features: [ask_p1, ask_s1, bid_p1, bid_s1, ..., ask_p10, ask_s10, bid_p10, bid_s10] |
| 46-49 | Labels: [label_h100, label_h50, label_h20, label_h10] |

### 7.2 Sequence Construction

At training time, sequences are constructed from the flat array:

```python
class LOBDataset:
    def __getitem__(self, i):
        x = self.data[i:i + seq_size, :46]  # (128, 46) input
        y = self.labels[i]                   # Single label
        return x, y
```

**Sequence length: 128 events**

**Why 128?**
- Captures ~1-2 seconds of market history
- Power of 2 for efficient tensor operations
- Balances context length vs. computational cost
- Consistent with TLOB/DeepLOB literature

## 8. Pipeline Diagram

```
┌─────────────────┐     ┌─────────────────┐
│  Message CSV    │     │  Orderbook CSV  │
│ (per day)       │     │ (per day)       │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────▼──────┐
              │   Merge &   │
              │ Synchronize │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │   Filter    │
              │ Event Types │
              │ (keep 1,3,4)│
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  Engineer   │
              │  Features   │
              │ (depth,time)│
              └──────┬──────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
    ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
    │  Train  │ │   Val   │ │  Test   │
    │ (85%)   │ │  (5%)   │ │ (10%)   │
    └────┬────┘ └────┬────┘ └────┬────┘
         │           │           │
    ┌────▼────┐      │           │
    │ Compute │      │           │
    │  Stats  │──────┼───────────┤
    │ (μ, σ)  │      │           │
    └────┬────┘      │           │
         │           │           │
    ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
    │Normalize│ │Normalize│ │Normalize│
    │(own μ,σ)│ │(train   │ │(train   │
    │         │ │ μ, σ)   │ │ μ, σ)   │
    └────┬────┘ └────┬────┘ └────┬────┘
         │           │           │
    ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
    │ Generate│ │ Generate│ │ Generate│
    │ Labels  │ │ Labels  │ │ Labels  │
    └────┬────┘ └────┬────┘ └────┬────┘
         │           │           │
    ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
    │train.npy│ │ val.npy │ │test.npy │
    └─────────┘ └─────────┘ └─────────┘
```





### 5.2 Threshold Calculation

The threshold α for class assignment is data-adaptive:

```python
pct_change = (future_mid - previous_mid) / previous_mid
alpha = abs(pct_change).mean() / 2
```

**Why adaptive threshold?**
- Fixed thresholds don't account for varying volatility
- Adaptive threshold ensures ~balanced classes
- Factor of 2 empirically yields ~30/40/30 split

### 5.3 Class Assignment

```python
label = 0 (Up)        if pct_change > +α
label = 1 (Stationary) if -α ≤ pct_change ≤ +α
label = 2 (Down)      if pct_change < -α
```

**Why 3 classes?**
- **Trading relevance**: Long (Up), Hold (Stationary), Short (Down)
- **Balanced problem**: Avoids extreme class imbalance
- **Standard in literature**: Comparable to prior work