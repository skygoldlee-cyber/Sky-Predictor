from __future__ import annotations

"""Transformer model implementation.

This module provides a lightweight Transformer encoder model for predicting
directional probability P(up) from a fixed-length feature sequence.

If `torch` is not available, the `PriceTransformer` class remains importable
but cannot be instantiated/loaded.
"""

import math
from pathlib import Path
from typing import Any

from config import PAST_UNKNOWN_DIM


try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class _PositionalEncoding(nn.Module):
        """_PositionalEncoding.
"""
        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
            """__init__.

Args:
    d_model:
    max_len:
    dropout:
"""
            super().__init__()
            self.dropout = nn.Dropout(p=float(dropout))

            pe = torch.zeros(int(max_len), int(d_model))
            pos = torch.arange(int(max_len)).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, int(d_model), 2).float() * (-math.log(10000.0) / float(d_model)))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """forward.

Args:
    x:
"""
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)


    class PriceTransformer(nn.Module):
        """PriceTransformer.
"""
        def __init__(
            self,
            feature_dim: int = PAST_UNKNOWN_DIM,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            d_ff: int = 128,
            seq_len: int = 60,
            dropout: float = 0.1,
            pooling: str = "cls",
        ):
            """__init__.

Args:
    feature_dim:
    d_model:
    n_heads:
    n_layers:
    d_ff:
    seq_len:
    dropout:
"""
            super().__init__()
            self.feature_dim = int(feature_dim)
            self.d_model = int(d_model)
            self.seq_len = int(seq_len)

            p = str(pooling or "cls").strip().lower()
            if p not in {"cls", "recency_weighted"}:
                p = "cls"
            self.pooling = str(p)

            self.input_proj = nn.Linear(int(feature_dim), int(d_model))

            self.cls_token = nn.Parameter(torch.zeros(1, 1, int(d_model)))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

            self.pos_enc = _PositionalEncoding(int(d_model), max_len=int(seq_len) + 1, dropout=float(dropout))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=int(d_model),
                nhead=int(n_heads),
                dim_feedforward=int(d_ff),
                dropout=float(dropout),
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))

            self.head = nn.Sequential(
                nn.LayerNorm(int(d_model)),
                nn.Linear(int(d_model), 32),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """forward.

Args:
    x:
"""
            batch = x.size(0)
            x = self.input_proj(x)
            cls = self.cls_token.expand(batch, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = self.pos_enc(x)
            x = self.encoder(x)

            if str(getattr(self, "pooling", "cls")) == "recency_weighted":
                try:
                    # Exclude CLS token (index 0). Favor recent timesteps with exponential decay.
                    h = x[:, 1:, :]
                    t = int(h.size(1))
                    weights = torch.exp(torch.linspace(-2.0, 0.0, t, device=h.device))
                    weights = weights / (weights.sum() + 1e-9)
                    pooled = (h * weights.view(1, -1, 1)).sum(dim=1)
                    return self.head(pooled).squeeze(-1)
                except Exception:
                    pass

            return self.head(x[:, 0]).squeeze(-1)

        @classmethod
        def load(
            cls,
            weights_path: str,
            *,
            feature_dim: int = PAST_UNKNOWN_DIM,
            seq_len: int = 60,
            device: str = "cpu",
            **kwargs: Any,
        ) -> "PriceTransformer":
            """load.

Args:
    weights_path:
    feature_dim:
    seq_len:
    device:
    kwargs:
"""
            model = cls(feature_dim=int(feature_dim), seq_len=int(seq_len), **kwargs)
            obj = torch.load(str(weights_path), map_location=str(device), weights_only=False)
            state = None
            if isinstance(obj, dict) and "state_dict" in obj:
                state = obj.get("state_dict")
            else:
                state = obj
            model.load_state_dict(state)
            model.eval()
            return model

        def save(self, path: str) -> None:
            """save.

Args:
    path:
"""
            Path(str(path)).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), str(path))


else:

    class PriceTransformer:  # pragma: no cover
        """PriceTransformer.
"""
        def __init__(self, *args: Any, **kwargs: Any):
            """__init__.

Args:
    args:
    kwargs:
"""
            raise ImportError("torch is required to use PriceTransformer")

        @classmethod
        def load(cls, *args: Any, **kwargs: Any) -> "PriceTransformer":
            """load.

Args:
    args:
    kwargs:
"""
            raise ImportError("torch is required to load PriceTransformer weights")

