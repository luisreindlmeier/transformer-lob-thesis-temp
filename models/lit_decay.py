"""LiT Transformer with Decay Attention."""
from typing import Iterable, Sequence, Optional
import torch
import torch.nn as nn
from models.decay_attention import DecayAttention


class DecayEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dim_feedforward: Optional[int] = None,
                 dropout: float = 0.1, init_decay: float = 0.1):
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = DecayAttention(d_model, n_heads, max_seq_len, dropout, init_decay)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class LiTDecayTransformer(nn.Module):
    def __init__(self, n_features: int, window: int, feature_mode: str = "all",
                 feature_indices: Optional[Sequence[int]] = None, event_count: Optional[int] = None,
                 lob_count: Optional[int] = None, d_model: int = 256, n_heads: int = 8, n_layers: int = 6,
                 num_classes: int = 3, dropout: float = 0.2, init_decay: float = 0.1):
        super().__init__()
        self.window = window
        self.init_decay = init_decay
        
        indices = self._resolve_feature_indices(n_features, feature_mode.lower(), feature_indices, event_count, lob_count)
        if indices is None:
            self.feature_indices = None
            input_features = n_features
        else:
            self.register_buffer("feature_indices", torch.tensor(list(indices), dtype=torch.long))
            input_features = len(indices)
        
        self.input_proj = nn.Linear(input_features, d_model)
        self.encoder = nn.ModuleList([
            DecayEncoderLayer(d_model, n_heads, window + 1, 4 * d_model, dropout, init_decay)
            for _ in range(n_layers)
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, window + 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, num_classes))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_indices is not None:
            x = torch.index_select(x, dim=2, index=self.feature_indices)
        B = x.size(0)
        x = self.input_proj(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_emb
        for layer in self.encoder:
            x = layer(x)
        return self.head(self.norm(x[:, 0]))
    
    def get_decay_rates(self) -> dict:
        return {f"layer_{i}": layer.attn.lambda_decay.detach() for i, layer in enumerate(self.encoder)}
    
    def get_all_lambda_raw(self) -> list:
        return [layer.attn.lambda_raw for layer in self.encoder]
    
    @staticmethod
    def _resolve_feature_indices(n_features: int, mode: str, indices: Optional[Sequence[int]],
                                  event_count: Optional[int], lob_count: Optional[int]) -> Optional[Iterable[int]]:
        if indices is not None:
            return indices
        if mode == "all":
            return None
        if event_count is None or lob_count is None:
            raise ValueError("event_count and lob_count required for feature_mode != 'all'")
        if mode == "events":
            return range(0, event_count)
        if mode == "lob":
            return range(event_count, event_count + lob_count)
        if mode == "no-raw":
            return list(range(3, event_count)) + list(range(event_count, event_count + lob_count))
        raise ValueError(f"Unknown feature_mode: {mode}")
