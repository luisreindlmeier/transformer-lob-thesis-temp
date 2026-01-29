# Model Architectures

## 1. Overview

This thesis implements and compares five neural network architectures for limit order book prediction:

| Model | Architecture | Novel Contribution |
|-------|--------------|-------------------|
| **DeepLOB** | CNN + GRU | Baseline (Zhang et al., 2019) |
| **TLOB** | Transformer | Bilinear Normalization, Alternating Attention |
| **TLOB-Decay** | Transformer | TLOB + Learnable Temporal Decay |
| **LiT** | Transformer | Lightweight with CLS token (ViT-style) |
| **LiT-Decay** | Transformer | LiT + Learnable Temporal Decay |

All models solve a **3-class classification problem**: predicting whether the mid-price will go Up (0), stay Stationary (1), or go Down (2) within the next k events.

## 2. DeepLOB (Baseline)

### 2.1 Architecture

DeepLOB is a CNN-GRU hybrid that serves as the baseline from Zhang et al. (2019):

```
Input: (B, 128, 46)
    ↓
Conv Block 1: 2× Conv2d(1→32, kernel=(1,5))
    ↓
Conv Block 2: 2× Conv2d(32→64, kernel=(3,1))
    ↓
Reshape: (B, 128, 64×46)
    ↓
GRU: hidden_size=128
    ↓
FC: 128 → 3
    ↓
Output: (B, 3) logits
```

### 2.2 Design Rationale

**Convolutional layers:**
- `kernel=(1,5)`: Captures cross-feature patterns (price-size interactions)
- `kernel=(3,1)`: Captures temporal patterns (event sequences)
- Separate kernels exploit the 2D structure of LOB data

**GRU layer:**
- Captures long-range temporal dependencies
- Single layer is sufficient for 128-event sequences
- Hidden state (128) compresses temporal information

**Why as baseline?**
- Widely cited benchmark (~2000 citations)
- Strong performance on LOBSTER data
- Represents pre-transformer state-of-the-art

### 2.3 Implementation

```python
class DeepLOB(nn.Module):
    def __init__(self, n_features: int = 46, num_classes: int = 3):
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 5), padding=(0, 2)), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 5), padding=(0, 2)), nn.ReLU(),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)), nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=64 * n_features, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
```

## 3. TLOB (Transformer for LOB)

### 3.1 Key Innovation: Bilinear Normalization (BiN)

Standard LayerNorm normalizes across features, losing cross-sequence information. BiN normalizes in two directions:

```
BiN(x) = γ₁ · Z₁ + γ₂ · Z₂
```

**Where:**
- Z₁: Normalized across the temporal dimension (captures feature patterns)
- Z₂: Normalized across the feature dimension (captures temporal patterns)
- γ₁, γ₂: Learnable mixing weights (initialized to 0.5)

**Mathematical formulation:**

```
Z₁[b,t,f] = (x[b,t,f] - μ_t) / σ_t    # Temporal normalization
Z₂[b,t,f] = (x[b,t,f] - μ_f) / σ_f    # Feature normalization
BiN(x) = γ₁ · (λ₁ · Z₁ + β₁) + γ₂ · (λ₂ · Z₂ + β₂)
```

**Why BiN?**
- LOB data has structure in both dimensions (time and features)
- Standard normalization destroys cross-sequence relationships
- Learnable mixing adapts to the relative importance of each direction

### 3.2 Alternating Attention

TLOB applies self-attention in alternating directions:

```
Layer 1: Attention over features (d_model = 46)
Layer 2: Attention over time (d_model = 128)
Layer 3: Attention over features
Layer 4: Attention over time
...
```

**Implementation:**
```python
for i in range(num_layers):
    self.layers.append(TransformerLayer(hidden_dim, ...))   # Feature attention
    self.layers.append(TransformerLayer(seq_size, ...))     # Temporal attention
```

**Why alternating?**
- Features and time have different semantic meanings
- Feature attention: "Which LOB levels matter for this timestep?"
- Temporal attention: "Which historical events matter for this feature?"
- Alternating allows joint feature-temporal learning

### 3.3 Architecture

```
Input: (B, 128, 46)
    ↓
Order Type Embedding: event_type → learned embedding
    ↓
BiN: Bilinear Normalization
    ↓
Linear Projection: 46 → 46 (hidden_dim)
    ↓
+ Sinusoidal Positional Encoding
    ↓
8× Alternating Transformer Layers:
   - Feature attention (46 → 46)
   - Transpose
   - Temporal attention (128 → 128)
   - Transpose
   (Last layer reduces to hidden_dim//4, seq_size//4)
    ↓
Flatten: (B, 11 × 32) = (B, 352)
    ↓
MLP Head: 352 → 88 → 3
    ↓
Output: (B, 3) logits
```

### 3.4 Positional Encoding

Sinusoidal positional encoding (Vaswani et al., 2017):

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**Why sinusoidal (not learned)?**
- Generalizes to unseen sequence lengths
- Encodes relative positions through linear combinations
- Fewer parameters than learned embeddings

## 4. TLOB-Decay (Novel Contribution)

### 4.1 Decay Attention Mechanism

Standard attention treats all positions equally. In financial markets, recent events are typically more relevant than older events. Decay attention introduces a learnable temporal discount:

```
Attention(Q, K, V) = softmax(QK^T / √d_k · M_decay) · V
```

**Where the decay mask:**
```
M_decay[i,j] = exp(-λ · |i - j|)
```

**Properties:**
- λ > 0: Decay rate (larger = faster decay)
- |i - j|: Distance between positions
- λ is learnable (per attention head)

### 4.2 Softplus Parameterization

To ensure λ > 0, we parameterize via softplus:

```python
lambda_raw = nn.Parameter(...)  # Unconstrained
lambda_decay = F.softplus(lambda_raw)  # Always positive
```

**Why softplus?**
- Smooth, differentiable everywhere
- No gradient issues at boundary (unlike ReLU)
- Natural interpretation: log(1 + exp(x))

### 4.3 Implementation

```python
class DecayAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, init_decay=0.1):
        # Initialize λ via inverse softplus
        init_raw = torch.log(torch.exp(torch.tensor(init_decay)) - 1)
        self.lambda_raw = nn.Parameter(torch.full((n_heads,), init_raw.item()))
        
        # Precompute distance matrix
        positions = torch.arange(max_seq_len)
        self.distances = torch.abs(positions[:, None] - positions[None, :]).float()
    
    def forward(self, x):
        # Standard attention computation
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply decay mask
        lambdas = self.lambda_decay.view(1, n_heads, 1, 1)
        decay_mask = torch.exp(-lambdas * self.distances[:T, :T])
        attn = attn * decay_mask
        
        return softmax(attn) @ v
```

### 4.4 Interpretation

After training, learned λ values reveal what the model learned:

| λ Value | Interpretation | Attention Pattern |
|---------|----------------|-------------------|
| λ < 0.05 | Very slow decay | Global attention (all events matter) |
| λ ≈ 0.1 | Slow decay | Long-range dependencies |
| λ ≈ 0.2 | Medium decay | Balanced local/global |
| λ ≈ 0.5 | Fast decay | Mostly local attention |
| λ > 0.5 | Very fast decay | Recent events dominate |

## 5. LiT (Lightweight Transformer)

### 5.1 ViT-Style Architecture

LiT uses the Vision Transformer (ViT) paradigm with a CLS token:

```
Input: (B, 128, 46)
    ↓
Linear Projection: 46 → 256 (d_model)
    ↓
Prepend CLS token: (B, 129, 256)
    ↓
+ Learned Positional Embedding
    ↓
6× Transformer Encoder Layers (Pre-LN)
    ↓
Extract CLS token: (B, 256)
    ↓
MLP Head: 256 → 256 → 3
    ↓
Output: (B, 3) logits
```

### 5.2 Design Choices

**CLS token:**
- Aggregates sequence information into a single vector
- Eliminates need for global pooling
- Successful in NLP (BERT) and vision (ViT)

**Pre-LN (LayerNorm before attention):**
```python
x = x + Attention(LayerNorm(x))  # Pre-LN
# vs.
x = LayerNorm(x + Attention(x))  # Post-LN
```
- More stable gradients during training
- Allows deeper networks without warm-up

**Larger model:**
- d_model = 256 (vs. 46 in TLOB)
- n_heads = 8 (vs. 1 in TLOB)
- n_layers = 6 (similar to TLOB's effective 8)

### 5.3 Implementation

```python
class LiTTransformer(nn.Module):
    def __init__(self, n_features=46, window=128, d_model=256, n_heads=8, n_layers=6):
        self.input_proj = nn.Linear(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, window + 1, d_model))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, 4*d_model, 
                                        activation="gelu", norm_first=True),
            num_layers=n_layers
        )
        self.head = nn.Sequential(Linear(d_model, 256), GELU(), Linear(256, 3))
```

## 6. LiT-Decay

Combines LiT architecture with learnable decay attention:

```python
class LiTDecayTransformer(BaseLiTTransformer):
    def __init__(self, ..., init_decay=0.1):
        self.encoder = nn.ModuleList([
            DecayEncoderLayer(d_model, n_heads, window + 1, init_decay=init_decay)
            for _ in range(n_layers)
        ])
```

**Key differences from TLOB-Decay:**
- Uses CLS token (no alternating attention)
- Larger model capacity (256 vs. 46 hidden dim)
- More attention heads (8 vs. 1)
- Each head learns its own decay rate

## 7. Model Comparison

### 7.1 Parameter Counts

| Model | Parameters | Relative Size |
|-------|------------|---------------|
| DeepLOB | ~390K | 1.0× |
| TLOB | ~180K | 0.5× |
| TLOB-Decay | ~185K | 0.5× |
| LiT | ~2.1M | 5.4× |
| LiT-Decay | ~2.1M | 5.4× |

### 7.2 Architectural Comparison

| Aspect | DeepLOB | TLOB | LiT |
|--------|---------|------|-----|
| **Feature extraction** | CNN | BiN + Attention | Linear + Attention |
| **Temporal modeling** | GRU | Attention | Attention |
| **Attention type** | None | Alternating | Standard |
| **Normalization** | BatchNorm | BiN | LayerNorm |
| **Positional encoding** | Implicit (RNN) | Sinusoidal | Learned |
| **Output aggregation** | Last hidden | Flatten | CLS token |

### 7.3 Inductive Biases

| Model | Key Inductive Bias |
|-------|-------------------|
| **DeepLOB** | Local patterns via CNN, sequential via GRU |
| **TLOB** | Feature-time structure via BiN and alternating attention |
| **TLOB-Decay** | + Recency bias via learnable decay |
| **LiT** | Global attention, minimal structure assumption |
| **LiT-Decay** | + Recency bias via learnable decay |

## 8. Forward Pass Example

```python
from src_prediction.models import TLOB, TLOBDecay, LiTTransformer

# Input shape: (batch_size, seq_len, n_features)
x = torch.randn(32, 128, 46)

# TLOB forward
model = TLOB(hidden_dim=46, num_layers=4, seq_size=128)
logits = model(x)  # (32, 3)

# TLOB-Decay forward (same interface)
model = TLOBDecay(hidden_dim=46, num_layers=4, seq_size=128, init_decay=0.1)
logits = model(x)  # (32, 3)

# Inspect learned decay rates
decay_rates = model.get_decay_rates()
# {'layer_0_feature': tensor([0.12]), 'layer_1_temporal': tensor([0.08]), ...}
```
