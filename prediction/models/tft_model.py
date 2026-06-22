"""Temporal Fusion Transformer (TFT) implementation (binary classification).

This module is intentionally self-contained and mirrors the design guide
`TFT_DUAL_MODEL_DESIGN_GUIDE.md`.

Torch is an optional dependency in this repository; when torch is unavailable,
constructing/loading the model raises ImportError.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from config import FUTURE_KNOWN_DIM, HORIZON_SEC, PAST_UNKNOWN_DIM

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class _GatedResidualNetwork(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float = 0.1,
            context_dim: int = 0,
        ):
            super().__init__()
            self.fc1 = nn.Linear(int(input_dim) + int(context_dim), int(hidden_dim))
            self.fc2 = nn.Linear(int(hidden_dim), int(output_dim))
            self.gate = nn.Linear(int(hidden_dim), int(output_dim))
            self.skip = (
                nn.Linear(int(input_dim), int(output_dim))
                if int(input_dim) != int(output_dim)
                else nn.Identity()
            )
            self.norm = nn.LayerNorm(int(output_dim))
            self.drop = nn.Dropout(float(dropout))
            self.act = nn.ELU()

        def forward(
            self,
            x: "torch.Tensor",
            context: "Optional[torch.Tensor]" = None,
        ) -> "torch.Tensor":
            if context is not None:
                inp = torch.cat([x, context], dim=-1)
            else:
                inp = x

            h = self.act(self.fc1(inp))
            h = self.drop(h)
            sig = torch.sigmoid(self.gate(h))
            out = sig * self.fc2(h)
            return self.norm(out + self.skip(x))


    class _VariableSelectionNetwork(nn.Module):
        def __init__(
            self,
            num_vars: int,
            d_model: int,
            dropout: float = 0.1,
            static_context_dim: int = 0,
        ):
            super().__init__()
            self.num_vars = int(num_vars)
            self.d_model = int(d_model)

            self.var_grns = nn.ModuleList(
                [_GatedResidualNetwork(1, self.d_model, self.d_model, float(dropout)) for _ in range(self.num_vars)]
            )
            self.weight_grn = _GatedResidualNetwork(
                self.num_vars * self.d_model,
                self.d_model,
                self.num_vars,
                float(dropout),
                context_dim=int(static_context_dim),
            )
            self.softmax = nn.Softmax(dim=-1)

        def forward(
            self,
            x: "torch.Tensor",  # (B, T, V)
            static_ctx: "Optional[torch.Tensor]" = None,
        ) -> "tuple[torch.Tensor, torch.Tensor]":
            b, t, v = x.shape
            if v != self.num_vars:
                raise ValueError(f"VSN expected num_vars={self.num_vars}, got {v}")

            processed = []
            for i in range(self.num_vars):
                vi = x[..., i : i + 1]
                processed.append(self.var_grns[i](vi))
            flat = torch.cat(processed, dim=-1)

            if static_ctx is not None:
                ctx = static_ctx.unsqueeze(1).expand(-1, t, -1)
                weights = self.softmax(self.weight_grn(flat, ctx))
            else:
                weights = self.softmax(self.weight_grn(flat))

            stacked = torch.stack(processed, dim=-2)
            out = (weights.unsqueeze(-1) * stacked).sum(dim=-2)
            return out, weights


    class TemporalFusionTransformer(nn.Module):
        def __init__(
            self,
            past_unknown_dim: int = PAST_UNKNOWN_DIM,
            future_known_dim: int = FUTURE_KNOWN_DIM,
            static_dim: int = 0,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            d_ff: int = 128,
            seq_len: int = 60,
            horizon: int = HORIZON_SEC,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.seq_len = int(seq_len)
            self.horizon = int(horizon)
            self.d_model = int(d_model)
            self.has_static = int(static_dim) > 0

            if self.has_static:
                self.static_grn = _GatedResidualNetwork(int(static_dim), self.d_model, self.d_model, float(dropout))
                sctx_dim = self.d_model
            else:
                self.static_grn = None
                sctx_dim = 0

            self.vsn_past_unknown = _VariableSelectionNetwork(
                int(past_unknown_dim), self.d_model, float(dropout), static_context_dim=int(sctx_dim)
            )
            self.vsn_past_known = _VariableSelectionNetwork(
                int(future_known_dim), self.d_model, float(dropout), static_context_dim=int(sctx_dim)
            )
            self.vsn_future_known = _VariableSelectionNetwork(
                int(future_known_dim), self.d_model, float(dropout), static_context_dim=int(sctx_dim)
            )

            self.encoder_lstm = nn.LSTM(
                input_size=self.d_model * 2,
                hidden_size=self.d_model,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
            )
            self.decoder_lstm = nn.LSTM(
                input_size=self.d_model,
                hidden_size=self.d_model,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
            )

            self.gate_enc = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.Sigmoid())
            self.gate_dec = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.Sigmoid())

            self.static_enrich = _GatedResidualNetwork(
                self.d_model,
                self.d_model,
                self.d_model,
                float(dropout),
                context_dim=int(sctx_dim),
            )

            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=int(n_heads),
                dim_feedforward=int(d_ff),
                dropout=float(dropout),
                batch_first=True,
                norm_first=True,
            )
            self.attn = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))
            self.pwff = _GatedResidualNetwork(self.d_model, int(d_ff), self.d_model, float(dropout))

            self.head = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, 32),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            past_unknown: "torch.Tensor",  # (B, seq_len, past_unknown_dim)
            past_known: "torch.Tensor",  # (B, seq_len, future_known_dim)
            future_known: "torch.Tensor",  # (B, horizon, future_known_dim)
            static: "Optional[torch.Tensor]" = None,
        ) -> "torch.Tensor":
            if past_unknown.ndim != 3 or past_known.ndim != 3 or future_known.ndim != 3:
                raise ValueError("TFT inputs must be 3D tensors")

            if past_unknown.size(1) != self.seq_len or past_known.size(1) != self.seq_len:
                raise ValueError("seq_len mismatch")
            if future_known.size(1) != self.horizon:
                raise ValueError("horizon mismatch")

            sctx = None
            if self.has_static and static is not None:
                sctx = self.static_grn(static)

            pu, _ = self.vsn_past_unknown(past_unknown, sctx)
            pk, _ = self.vsn_past_known(past_known, sctx)
            fk, _ = self.vsn_future_known(future_known, sctx)

            enc_in = torch.cat([pu, pk], dim=-1)
            enc_out, (h, c) = self.encoder_lstm(enc_in)
            enc_out = enc_out * self.gate_enc(enc_out)

            dec_out, _ = self.decoder_lstm(fk, (h, c))
            dec_out = dec_out * self.gate_dec(dec_out)

            combined = torch.cat([enc_out, dec_out], dim=1)
            if sctx is not None:
                se_ctx = sctx.unsqueeze(1).expand(-1, combined.size(1), -1)
                combined = self.static_enrich(combined, se_ctx)
            else:
                combined = self.static_enrich(combined)

            attn_out = self.attn(combined)
            attn_out = self.pwff(attn_out)

            first_future = attn_out[:, self.seq_len, :]
            return self.head(first_future).squeeze(-1)

        @classmethod
        def load(
            cls,
            weights_path: str,
            *,
            past_unknown_dim: int = PAST_UNKNOWN_DIM,
            future_known_dim: int = FUTURE_KNOWN_DIM,
            seq_len: int = 60,
            horizon: int = HORIZON_SEC,
            device: str = "cpu",
            **kwargs: Any,
        ) -> "TemporalFusionTransformer":
            model = cls(
                past_unknown_dim=int(past_unknown_dim),
                future_known_dim=int(future_known_dim),
                seq_len=int(seq_len),
                horizon=int(horizon),
                **kwargs,
            )
            state = torch.load(str(weights_path), map_location=str(device), weights_only=True)
            model.load_state_dict(state)
            model.eval()
            return model

        def save(self, path: str) -> None:
            Path(str(path)).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), str(path))


else:

    class TemporalFusionTransformer:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError("torch is required to use TemporalFusionTransformer")

        @classmethod
        def load(cls, *args: Any, **kwargs: Any):
            raise ImportError("torch is required to load TFT weights")
