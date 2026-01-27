"""Decay Attention Module - learnable per-head temporal decay."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecayAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1, init_decay: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable decay rate per head (softplus parameterization)
        init_raw = torch.log(torch.exp(torch.tensor(init_decay)) - 1)
        self.lambda_raw = nn.Parameter(torch.full((n_heads,), init_raw.item()))
        
        positions = torch.arange(max_seq_len)
        self.register_buffer('distances', torch.abs(positions[:, None] - positions[None, :]).float())
    
    @property
    def lambda_decay(self) -> torch.Tensor:
        return F.softplus(self.lambda_raw)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply decay: exp(-lambda * |i-j|)
        lambdas = self.lambda_decay.view(1, self.n_heads, 1, 1)
        decay_mask = torch.exp(-lambdas * self.distances[:T, :T].unsqueeze(0).unsqueeze(0))
        attn = attn * decay_mask
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)
