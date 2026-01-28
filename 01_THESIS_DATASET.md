# Dataset: LOBSTER Limit Order Book Data

## 1. Data Source

The dataset originates from **LOBSTER** (Limit Order Book System - The Efficient Reconstructor), a commercial data provider that reconstructs high-frequency limit order book data from NASDAQ's ITCH data feed. LOBSTER is widely used in academic research on market microstructure and algorithmic trading.

## 2. Dataset Scope

| Attribute | Value | Rationale |
|-----------|-------|-----------|
| **Exchange** | NASDAQ | Largest electronic equity exchange, high liquidity |
| **Tickers** | AAPL, CSCO, GOOG, INTC | Large-cap tech stocks with high trading volume |
| **Period** | July 2023 | Recent data, 20+ trading days for sufficient samples |
| **Market Hours** | 9:30 AM - 4:00 PM ET | Regular trading session only (no pre/post-market) |
| **LOB Depth** | Level 2 (10 levels) | Captures order flow dynamics beyond best bid/ask |

**Why these tickers?**
- High liquidity ensures tight spreads and representative order flow
- Sufficient data volume
- Prior research comparability (INTC and AAPL are commonly used)

## 3. Raw Data Format

Each trading day produces two synchronized files per ticker:

### 3.1 Message File (Order Events)

Contains every market event that changes the order book state:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `time` | float | Seconds since midnight (nanosecond precision) | 34200.123456789 |
| `event_type` | int | Type of order event (see below) | 1 |
| `order_id` | int | Unique identifier for the order | 12345678 |
| `size` | int | Order volume in shares | 100 |
| `price` | int | Price in $ × 10,000 (4 decimal places) | 1234567 → $123.4567 |
| `direction` | int | Side: -1 = Sell (ask), 1 = Buy (bid) | 1 |

**Event Types:**

| Code | Name | Description |
|------|------|-------------|
| 1 | New Order | Limit order submission |
| 2 | Partial Cancellation | Reduction in order size |
| 3 | Total Cancellation | Complete order removal |
| 4 | Execution (Visible) | Order matched against visible liquidity |
| 5 | Execution (Hidden) | Order matched against hidden liquidity |
| 6 | Cross Trade | Auction trade (opening/closing) |
| 7 | Trading Halt | Market suspension indicator |

### 3.2 Orderbook File (LOB Snapshots)

Contains the order book state after each event (40 columns for 10 levels):

| Columns | Description |
|---------|-------------|
| 1, 5, 9, ..., 37 | Ask prices (Level 1-10, best to worst) |
| 2, 6, 10, ..., 38 | Ask sizes (Level 1-10) |
| 3, 7, 11, ..., 39 | Bid prices (Level 1-10, best to worst) |
| 4, 8, 12, ..., 40 | Bid sizes (Level 1-10) |


## 4. Data Characteristics

### 4.1 Event Frequency

```
how many events?!
```

**Why this matters:**
- High-frequency data requires efficient processing
- Sequence models must capture patterns across varying time scales
- Event-based (not time-based) sampling preserves market dynamics

## 6. Prediction Horizons

**Why event-based horizons?**
- Time-based horizons create variable information content
- Event-based horizons ensure consistent market activity
- Each event represents a real market action (order submission, cancellation, or execution)

**Primary horizon (k=10):**
- Most challenging (short-term noise)
- Most relevant for high-frequency market making
- Consistent with DeepLOB and TLOB benchmarks

## 7. Data Quality Considerations

### 7.1 Filtered Events

The following event types are removed during preprocessing:

| Event Type | Reason for Removal |
|------------|-------------------|
| 2 (Partial Cancel) | Modifies existing orders, doesn't change book structure |
| 5 (Hidden Execution) | Hidden liquidity not visible in order book |
| 6 (Cross Trade) | Auction mechanics differ from continuous trading |
| 7 (Trading Halt) | No trading activity during halts |

**Rationale:** Retained events (1, 3, 4) represent the visible order flow that shapes the limit order book and are directly actionable for market making.
