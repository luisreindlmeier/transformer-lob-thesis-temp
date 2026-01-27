"""TLOB Model Architecture."""
import torch
from torch import nn
from einops import rearrange
from typing import Tuple, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


class BiN(nn.Module):
    """Bilinear Normalization for LOB data."""
    def __init__(self, d1: int, t1: int) -> None:
        super().__init__()
        self.t1 = t1
        self.d1 = d1
        self.B1 = nn.Parameter(torch.zeros(t1, 1))
        self.l1 = nn.Parameter(torch.empty(t1, 1))
        nn.init.xavier_normal_(self.l1)
        self.B2 = nn.Parameter(torch.zeros(d1, 1))
        self.l2 = nn.Parameter(torch.empty(d1, 1))
        nn.init.xavier_normal_(self.l2)
        self.y1 = nn.Parameter(torch.tensor([0.5]))
        self.y2 = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.y1[0] < 0:
            self.y1 = nn.Parameter(torch.tensor([0.01], device=x.device))
        if self.y2[0] < 0:
            self.y2 = nn.Parameter(torch.tensor([0.01], device=x.device))

        T2 = torch.ones([self.t1, 1], device=x.device)
        x2 = torch.mean(x, dim=2).unsqueeze(2)
        std = torch.std(x, dim=2).unsqueeze(2)
        std[std < 1e-4] = 1
        Z2 = (x - x2 @ T2.T) / (std @ T2.T)
        X2 = self.l2 @ T2.T * Z2 + self.B2 @ T2.T

        T1 = torch.ones([self.d1, 1], device=x.device)
        x1 = torch.mean(x, dim=1).unsqueeze(2)
        std = torch.std(x, dim=1).unsqueeze(2)
        op1 = (x1 @ T1.T).permute(0, 2, 1)
        op2 = (std @ T1.T).permute(0, 2, 1)
        z1 = (x - op1) / op2
        X1 = T1 @ self.l1.T * z1 + T1 @ self.B1.T

        return self.y1 * X1 + self.y2 * X2


class MLP(nn.Module):
    def __init__(self, start_dim: int, hidden_dim: int, final_dim: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(final_dim)
        self.fc = nn.Linear(start_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, final_dim)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.gelu(self.fc(x))
        x = self.fc2(x)
        if x.shape[2] == residual.shape[2]:
            x = x + residual
        return self.gelu(self.layer_norm(x))


class ComputeQKV(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim * num_heads)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q(x), self.k(x), self.v(x)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, final_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = ComputeQKV(hidden_dim, num_heads)
        self.attention = nn.MultiheadAttention(hidden_dim * num_heads, num_heads, batch_first=True, device=cfg.DEVICE)
        self.mlp = MLP(hidden_dim, hidden_dim * 4, final_dim)
        self.w0 = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        res = x
        q, k, v = self.qkv(x)
        x, att = self.attention(q, k, v, average_attn_weights=False, need_weights=True)
        x = self.w0(x) + res
        x = self.mlp(self.norm(x))
        if x.shape[-1] == res.shape[-1]:
            x = x + res
        return x, att


def sinusoidal_positional_embedding(seq_size: int, dim: int, n: float = 10000.0, device: Optional[str] = None) -> torch.Tensor:
    positions = torch.arange(0, seq_size).unsqueeze(1)
    denominators = torch.pow(n, 2 * torch.arange(0, dim // 2) / dim)
    embeddings = torch.zeros(seq_size, dim)
    embeddings[:, 0::2] = torch.sin(positions / denominators)
    embeddings[:, 1::2] = torch.cos(positions / denominators)
    return embeddings.to(device or cfg.DEVICE, non_blocking=True)


class TLOB(nn.Module):
    """Transformer for Limit Order Book prediction."""
    def __init__(self, hidden_dim: int = cfg.HIDDEN_DIM, num_layers: int = cfg.NUM_LAYERS,
                 seq_size: int = cfg.SEQ_SIZE, num_features: int = 46,
                 num_heads: int = cfg.NUM_HEADS, is_sin_emb: bool = cfg.IS_SIN_EMB) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.order_type_embedder = nn.Embedding(3, 1)
        self.norm_layer = BiN(num_features, seq_size)
        self.emb_layer = nn.Linear(num_features, hidden_dim)
        
        if is_sin_emb:
            self.pos_encoder = sinusoidal_positional_embedding(seq_size, hidden_dim)
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, seq_size, hidden_dim))

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim))
                self.layers.append(TransformerLayer(seq_size, num_heads, seq_size))
            else:
                self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim // 4))
                self.layers.append(TransformerLayer(seq_size, num_heads, seq_size // 4))
        
        total_dim = (hidden_dim // 4) * (seq_size // 4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim // 4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim // 4
        self.final_layers.append(nn.Linear(total_dim, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Order type at index 41 needs embedding
        continuous_features = torch.cat([x[:, :, :41], x[:, :, 42:]], dim=2)
        order_type = x[:, :, 41].long()
        order_type_emb = self.order_type_embedder(order_type).detach()
        x = torch.cat([continuous_features, order_type_emb], dim=2)
        
        x = rearrange(x, 'b s f -> b f s')
        x = self.norm_layer(x)
        x = rearrange(x, 'b f s -> b s f')
        x = self.emb_layer(x) + self.pos_encoder
        
        for layer in self.layers:
            x, _ = layer(x)
            x = x.permute(0, 2, 1)
        
        x = x.reshape(x.shape[0], -1)
        for layer in self.final_layers:
            x = layer(x)
        return x
