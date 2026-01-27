# Dataset Documentation

## Source

**LOBSTER** (Limit Order Book System - The Efficient Reconstructor)  
Academic license from [lobsterdata.com](https://lobsterdata.com)

## Scope

| Attribute | Value |
|-----------|-------|
| **Exchange** | NASDAQ |
| **Tickers** | AAPL, CSCO, GOOG, INTC |
| **Period** | July 2023 (20 trading days) |
| **Market Hours** | 9:30 AM - 4:00 PM ET |
| **LOB Depth** | Level 2 (10 price levels) |

## File Structure

Each trading day produces two files per ticker:

```
data/raw/{TICKER}/
├── {TICKER}_{DATE}_34200000_57600000_message_10.csv
└── {TICKER}_{DATE}_34200000_57600000_orderbook_10.csv
```

- `34200000` = 9:30 AM in milliseconds from midnight
- `57600000` = 4:00 PM in milliseconds from midnight
- `10` = LOB depth (10 levels)

## Message File Format

Each row represents a market event:

| Column | Description | Values |
|--------|-------------|--------|
| Time | Seconds since midnight | float |
| Event Type | Order event | 1=New, 2=Cancel, 3=Delete, 4=Execute, 5=Hidden, 6=Cross, 7=Halt |
| Order ID | Unique identifier | int |
| Size | Order volume | int |
| Price | Price in $ (×10000) | int |
| Direction | Buy/Sell | -1=Sell, 1=Buy |

## Orderbook File Format

LOB snapshot after each event (40 columns for 10 levels):

| Columns | Description |
|---------|-------------|
| 1, 3, 5, ..., 19 | Ask prices (level 1-10) |
| 2, 4, 6, ..., 20 | Ask volumes (level 1-10) |
| 21, 23, 25, ..., 39 | Bid prices (level 1-10) |
| 22, 24, 26, ..., 40 | Bid volumes (level 1-10) |

## Preprocessed Format

After running `python train.py --ticker CSCO --preprocess`:

```
data/preprocessed/{TICKER}/
└── {TICKER}_preprocessed.npy
```

The `.npy` file contains a NumPy array with shape `(N, 51)`:
- Columns 0-5: Order features `[time, event_type, size, price, direction, depth]`
- Columns 6-45: LOB features (40 columns, normalized)
- Columns 46-50: Labels for horizons `[100, 50, 20, 10, 10]` ticks

## Label Generation

Labels are computed based on mid-price movement:

```
label = sign(future_midprice - current_midprice)
```

- **0 (Up)**: Mid-price increases by > threshold
- **1 (Stationary)**: Mid-price change within threshold
- **2 (Down)**: Mid-price decreases by > threshold

Threshold: 2× tick size (LOBSTER default)

## Data Statistics (CSCO, July 2023)

| Metric | Value |
|--------|-------|
| Total events | ~12M |
| Events per day | ~600K |
| Class balance (Up/Stat/Down) | ~30% / ~40% / ~30% |
| Sequence length | 128 events |

## Train/Validation/Test Split

| Split | Days | Percentage |
|-------|------|------------|
| Train | 1-14 | 70% |
| Validation | 15-17 | 15% |
| Test | 18-20 | 15% |

Split is temporal (no data leakage).

## License

LOBSTER data is proprietary. Academic license required for research use.  
See: [lobsterdata.com/info/Terms.php](https://lobsterdata.com/info/Terms.php)

## References

Huang, R., & Polak, T. (2011). *LOBSTER: Limit Order Book Reconstruction System.*
