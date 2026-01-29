# 4.3 Preprocessing Pipeline

## Overview (before 4.3.1)

- **Input:** Raw LOBSTER CSVs (message + orderbook per day).
- **Output:** NumPy arrays `train.npy`, `val.npy`, `test.npy` of shape \((N, 50)\).
- **Pipeline order:** Merge → Event filtering → Feature engineering → **Temporal split** → Z-score normalization (train stats) → Label generation → Save per split.
- **Constraints:** Chronological order preserved; normalization statistics from training data only (no leakage).

**Figure 4.X — Preprocessing pipeline.**

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

---

## 4.3.1 Event Filtering and Feature Engineering

### Event filtering

- **Retained event types:** 1 (New Order), 3 (Total Cancel), 4 (Execution).
- **Dropped:** 2 (Partial Cancel), 5 (Hidden Execution), 6 (Cross Trade) — no visible LOB change or non-standard mechanism.
- **Result:** One row per event that alters the visible book or represents a trade.

### Order-level features (6 dims)

| Index | Name     | Definition / formula |
|-------|----------|-----------------------|
| 0     | `time`   | Inter-event time delta: \(\Delta t_i = t_i - t_{i-1}\); \(t_0\): time since session start. |
| 1     | `event_type` | Encoded: 1→0, 3→1, 4→2. |
| 2     | `size`   | Order volume (raw). |
| 3     | `price`  | Order price (raw, same scale as LOB). |
| 4     | `direction` | \(+1\) buy, \(-1\) sell; for executions: sign inverted (aggressor side). |
| 5     | `depth`  | Distance from best quote in ticks (see below). |

- **Direction for executions:**
  \[
  \text{direction} \leftarrow \text{direction} \times (-1)^{\mathbb{1}[\text{event\_type}=4]}.
  \]
- **Depth (ticks):** With best bid \(B\), best ask \(A\), order price \(p\), tick size \(\tau\):
  \[
  \text{depth} = \max\left(0,\; \frac{B - p}{\tau}\right) \;\text{(buy)}, \qquad
  \text{depth} = \max\left(0,\; \frac{p - A}{\tau}\right) \;\text{(sell)}.
  \]
  - depth = 0: at BBO (aggressive); depth > 0: passive.

### LOB features (40 dims)

- **Indices 6–25:** Ask side — \([p_1^{\text{ask}}, v_1^{\text{ask}}, \ldots, p_{10}^{\text{ask}}, v_{10}^{\text{ask}}]\).
- **Indices 26–45:** Bid side — \([p_1^{\text{bid}}, v_1^{\text{bid}}, \ldots, p_{10}^{\text{bid}}, v_{10}^{\text{bid}}]\).
- **Levels:** \(L = 10\); same as DeepLOB/TLOB; price \(p_\ell\), volume \(v_\ell\) per level.

### Feature count

- **Total input dimension:** \(6 + 40 = 46\) per event.

---

## 4.3.2 Temporal Split

- **Splits:** Train / Validation / Test by **calendar time** (no shuffling).
- **Ratios:** \(r_{\text{train}} = 0.85\), \(r_{\text{val}} = 0.05\), \(r_{\text{test}} = 0.10\) of trading days.
- **Day counts:** With \(D\) days,
  \[
  D_{\text{train}} = \lfloor D \cdot r_{\text{train}} \rfloor,\quad
  D_{\text{val}} = \lfloor D \cdot r_{\text{val}} \rfloor,\quad
  D_{\text{test}} = D - D_{\text{train}} - D_{\text{val}}.
  \]
- **Order:** Train = days \(1..D_{\text{train}}\), Val = next \(D_{\text{val}}\) days, Test = remaining days.
- **Reason:** Avoids leakage from future; validation/test use strictly later time; evaluation reflects deployment.

---

## 4.3.3 Z-Score Normalization

- **Formula (per scalar feature):**
  \[
  x' = \frac{x - \mu}{\sigma}, \qquad \mu = \frac{1}{n}\sum_{i=1}^n x_i,\quad \sigma = \sqrt{\frac{1}{n}\sum_{i=1}^n (x_i-\mu)^2}.
  \]
- **Statistics:** \(\mu,\sigma\) computed **only on the training split**; same \((\mu,\sigma)\) applied to validation and test.
- **Feature groups and statistics:**

| Group        | Columns / source      | Single \((\mu,\sigma)\)? |
|-------------|------------------------|---------------------------|
| Prices (LOB)| All LOB price columns  | Yes (one \(\mu_p,\sigma_p\)) |
| Sizes (LOB) | All LOB size columns   | Yes (one \(\mu_s,\sigma_s\)) |
| Order: time | `time`                 | Own \(\mu_t,\sigma_t\)     |
| Order: size | `size`                 | Shared with LOB sizes      |
| Order: price| `price`                | Shared with LOB prices     |
| Order: depth| `depth`                | Own \(\mu_d,\sigma_d\)     |

- **Not normalized:** `event_type`, `direction` (discrete).
- **Leakage:** No use of val/test data when computing \(\mu,\sigma\).

---

## 4.3.4 Label Generation and Sequence Construction

### Smoothed mid-price

- **Mid-price:** \(m_t = (p_1^{\text{ask}}(t) + p_1^{\text{bid}}(t))/2\).
- **Smoothing window:** length \(W = 10\) (events).
- **Past/future smoothed mid:**
  \[
  \bar{m}_t^{\text{past}} = \frac{1}{W}\sum_{k=t-W}^{t-1} m_k,\qquad
  \bar{m}_{t+h}^{\text{fut}} = \frac{1}{W}\sum_{k=t+h-W}^{t+h-1} m_k.
  \]

### Return and adaptive threshold

- **Relative change (horizon \(h\)):**
  \[
  r_t^{(h)} = \frac{\bar{m}_{t+h}^{\text{fut}} - \bar{m}_t^{\text{past}}}{\bar{m}_t^{\text{past}}}.
  \]
- **Threshold:** \(\alpha^{(h)} = \frac{1}{2}\,\overline{|r^{(h)}|}\) (mean absolute return on training, divided by 2).

### Class assignment (3 classes)

- **Labels:** \(y_t^{(h)} \in \{0,1,2\}\):
  \[
  y_t^{(h)} = \begin{cases}
  0\;\text{(Up)}       & \text{if } r_t^{(h)} > +\alpha^{(h)}, \\
  1\;\text{(Stationary)} & \text{if } -\alpha^{(h)} \le r_t^{(h)} \le +\alpha^{(h)}, \\
  2\;\text{(Down)}     & \text{if } r_t^{(h)} < -\alpha^{(h)}.
  \end{cases}
  \]
- **Horizons:** \(h \in \{10, 20, 50, 100\}\) events; one label per horizon (columns 46–49).

### Output array and sequence construction

- **Stored array shape:** \((N, 50)\).
  - Columns \(0\)–\(45\): input features (order 0–5, LOB 6–45).
  - Columns \(46\)–\(49\): labels for \(h = 10, 20, 50, 100\).
- **Sequence length:** \(T = 128\) events.
- **Training sample at index \(i\):** Input = events \(i, \ldots, i+T-1\); label = label of event \(i+T-1\) (end of window):
  \[
  \mathbf{x}_i = \mathbf{X}[i:i+T,\, :46] \in \mathbb{R}^{128 \times 46},\qquad
  y_i = \mathbf{X}[i+T-1,\, 46:50].
  \]
- **Number of sequences:** \(N_{\text{seq}} = N - T + 1\) (sliding window, stride 1).
- **Rationale for \(T=128\):** ~1–2 s of history; power of 2; aligned with TLOB/DeepLOB.
