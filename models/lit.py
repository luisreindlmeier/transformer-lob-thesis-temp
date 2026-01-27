"""LiT (Lightweight Transformer) Model."""
from typing import Iterable, Sequence, Optional
import torch
import torch.nn as nn


class LiTTransformer(nn.Module):
    def __init__(self, n_features: int, window: int, feature_mode: str = "all",
                 feature_indices: Optional[Sequence[int]] = None, event_count: Optional[int] = None,
                 lob_count: Optional[int] = None, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 6, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.window = window
        
        indices = self._resolve_feature_indices(n_features, feature_mode.lower(), feature_indices, event_count, lob_count)
        if indices is None:
            self.feature_indices = None
            input_features = n_features
        else:
            self.register_buffer("feature_indices", torch.tensor(list(indices), dtype=torch.long))
            input_features = len(indices)

        self.input_proj = nn.Linear(input_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
                                                    dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
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
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))

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
