"""PatchTST model implementation for KP200 futures direction prediction.

PriceTransformer(model.py) 와 완전히 동일한 인터페이스를 제공한다.
- PriceTransformer.load() / save() / forward() 시그니처 동일
- predictor.py / train.py 에서 import 경로만 바꾸면 드롭인 교체 가능

핵심 아이디어 (Nie et al., 2023 "A Time Series is Worth 64 Words"):
    - 시계열을 겹치는 패치(patch)로 분할 → 각 패치 = 하나의 토큰
    - 표준 Transformer는 타임스텝 하나 = 토큰 하나이므로
      seq_len=60일 때 60개 어텐션 계산
    - PatchTST는 patch_len=8, stride=4이면 14개 패치만 처리
      → 연산량 감소 + 국소 패턴(캔들 군집) 포착력 향상

KP200 1분봉에 최적화된 기본값:
    patch_len = 8   (8분봉 구간 하나를 하나의 의미 단위로 처리)
    stride    = 4   (4분 간격으로 패치를 슬라이딩, 50% 겹침)
    seq_len   = 60  (현재 파이프라인과 동일)
    → num_patches = (60 - 8) // 4 + 1 = 14

torch 가 없으면 클래스는 import 가능하지만 인스턴스화 시 ImportError 를 발생시킨다.
이는 model.py 의 PriceTransformer 와 동일한 폴백 패턴이다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from constants import PAST_UNKNOWN_DIM


try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class _PatchEmbedding(nn.Module):
        """시계열을 겹치는 패치로 분할하고 d_model 차원으로 선형 투영한다.

        Args:
            feature_dim: 입력 피처 수 (PAST_UNKNOWN_DIM).
            patch_len:   패치 하나의 길이 (타임스텝 수).
            stride:      패치 슬라이딩 간격.
            d_model:     출력 임베딩 차원.
            dropout:     드롭아웃 비율.
        """

        def __init__(
            self,
            feature_dim: int,
            patch_len: int,
            stride: int,
            d_model: int,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.patch_len = int(patch_len)
            self.stride = int(stride)
            # 패치 하나의 원시 차원 = patch_len × feature_dim (채널 독립 X, 채널 혼합)
            self.proj = nn.Linear(int(patch_len) * int(feature_dim), int(d_model))
            self.norm = nn.LayerNorm(int(d_model))
            self.drop = nn.Dropout(float(dropout))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """(B, T, F) → (B, num_patches, d_model)"""
            B, T, F = x.shape
            pl = self.patch_len
            st = self.stride

            # 패치 추출: unfold over 시간 축
            # x_unfold: (B, num_patches, patch_len, F)
            x_unfold = x.unfold(dimension=1, size=pl, step=st)
            # flatten patch 내부: (B, num_patches, patch_len * F)
            x_flat = x_unfold.contiguous().view(B, x_unfold.size(1), pl * F)
            out = self.drop(self.norm(self.proj(x_flat)))
            return out  # (B, num_patches, d_model)


    class _PositionalEncoding(nn.Module):
        """고정 사인/코사인 위치 인코딩."""

        def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1) -> None:
            super().__init__()
            self.dropout = nn.Dropout(p=float(dropout))
            pe = torch.zeros(int(max_len), int(d_model))
            pos = torch.arange(int(max_len)).unsqueeze(1).float()
            div = torch.exp(
                torch.arange(0, int(d_model), 2).float()
                * (-math.log(10000.0) / float(d_model))
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)


    class PatchTSTModel(nn.Module):
        """PatchTST 기반 방향 예측 모델.

        PriceTransformer 와 완전히 동일한 외부 인터페이스:
            forward(x) → scalar sigmoid 출력 (B,) 또는 (B, 1)
            load(weights_path, ...) → PatchTSTModel
            save(path)

        Args:
            feature_dim:  입력 피처 차원 (PAST_UNKNOWN_DIM).
            d_model:      Transformer 내부 임베딩 차원.
            n_heads:      Multi-head Attention 헤드 수.
            n_layers:     Transformer Encoder 레이어 수.
            d_ff:         Feed-Forward 내부 차원.
            seq_len:      입력 시퀀스 길이 (타임스텝).
            patch_len:    패치 하나의 길이.
            stride:       패치 슬라이딩 간격.
            dropout:      드롭아웃 비율.
            pooling:      최종 풀링 방식. 'cls' | 'mean' | 'recency_weighted'.
        """

        def __init__(
            self,
            feature_dim: int = PAST_UNKNOWN_DIM,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 3,
            d_ff: int = 256,
            seq_len: int = 60,
            patch_len: int = 8,
            stride: int = 4,
            dropout: float = 0.1,
            pooling: str = "cls",
        ) -> None:
            super().__init__()
            self.feature_dim = int(feature_dim)
            self.d_model = int(d_model)
            self.seq_len = int(seq_len)
            self.patch_len = int(patch_len)
            self.stride = int(stride)

            p = str(pooling or "cls").strip().lower()
            if p not in {"cls", "mean", "recency_weighted"}:
                p = "cls"
            self.pooling = p

            # 패치 수 계산 (seq_len 과 patch_len, stride 로 결정)
            self.num_patches = (int(seq_len) - int(patch_len)) // int(stride) + 1

            # 패치 임베딩 레이어
            self.patch_embed = _PatchEmbedding(
                feature_dim=int(feature_dim),
                patch_len=int(patch_len),
                stride=int(stride),
                d_model=int(d_model),
                dropout=float(dropout),
            )

            # [CLS] 토큰 (cls pooling 용)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, int(d_model)))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

            # 위치 인코딩: num_patches + 1 (CLS)
            self.pos_enc = _PositionalEncoding(
                d_model=int(d_model),
                max_len=self.num_patches + 1,
                dropout=float(dropout),
            )

            # Transformer Encoder (Pre-LN)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=int(d_model),
                nhead=int(n_heads),
                dim_feedforward=int(d_ff),
                dropout=float(dropout),
                batch_first=True,
                norm_first=True,  # Pre-LayerNorm: 안정적 학습
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))

            # 분류 헤드
            self.head = nn.Sequential(
                nn.LayerNorm(int(d_model)),
                nn.Linear(int(d_model), 32),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """순전파.

            Args:
                x: (B, seq_len, feature_dim) 입력 텐서.

            Returns:
                (B,) 형태의 sigmoid 출력 [0, 1].
            """
            B = x.size(0)

            # 1. 패치 임베딩: (B, num_patches, d_model)
            patches = self.patch_embed(x)

            # 2. CLS 토큰 prepend
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, patches], dim=1)  # (B, num_patches+1, d_model)

            # 3. 위치 인코딩
            tokens = self.pos_enc(tokens)

            # 4. Transformer Encoder
            encoded = self.encoder(tokens)  # (B, num_patches+1, d_model)

            # 5. 풀링
            if self.pooling == "mean":
                # 패치 토큰(CLS 제외) 평균
                pooled = encoded[:, 1:, :].mean(dim=1)
            elif self.pooling == "recency_weighted":
                # 최근 패치에 지수적으로 높은 가중치 부여
                try:
                    h = encoded[:, 1:, :]
                    t = int(h.size(1))
                    weights = torch.exp(torch.linspace(-2.0, 0.0, t, device=h.device))
                    weights = weights / (weights.sum() + 1e-9)
                    pooled = (h * weights.view(1, -1, 1)).sum(dim=1)
                except Exception:
                    pooled = encoded[:, 0]  # fallback → CLS
            else:
                # 기본값: CLS 토큰
                pooled = encoded[:, 0]

            return self.head(pooled).squeeze(-1)

        @classmethod
        def load(
            cls,
            weights_path: str,
            *,
            feature_dim: int = PAST_UNKNOWN_DIM,
            seq_len: int = 60,
            device: str = "cpu",
            **kwargs: Any,
        ) -> "PatchTSTModel":
            """저장된 가중치에서 모델을 로드한다.

            PriceTransformer.load() 와 동일한 시그니처.

            체크포인트에 'model_kwargs' 키가 있으면 아키텍처를 복원하고,
            없으면 kwargs 로 전달된 값을 사용한다.
            """
            obj = torch.load(
                str(weights_path), map_location=str(device), weights_only=False
            )

            # 아키텍처 복원: 체크포인트 우선, kwargs 보조
            arch_keys = {
                "d_model", "n_heads", "n_layers", "d_ff",
                "patch_len", "stride", "dropout", "pooling",
            }
            init_kwargs: dict[str, Any] = {k: v for k, v in kwargs.items() if k in arch_keys}

            state = None
            if isinstance(obj, dict):
                if "model_kwargs" in obj:
                    for k, v in obj["model_kwargs"].items():
                        if k in arch_keys:
                            init_kwargs[k] = v
                state = obj.get("state_dict", obj)
            else:
                state = obj

            model = cls(
                feature_dim=int(feature_dim),
                seq_len=int(seq_len),
                **init_kwargs,
            )
            model.load_state_dict(state)
            model.eval()
            return model

        def save(self, path: str) -> None:
            """가중치를 저장한다. 체크포인트에 아키텍처 정보를 함께 포함한다."""
            Path(str(path)).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": self.state_dict(),
                    "model_kwargs": {
                        "d_model": self.d_model,
                        "n_heads": self.encoder.layers[0].self_attn.num_heads,
                        "n_layers": len(self.encoder.layers),
                        "d_ff": self.encoder.layers[0].linear1.out_features,
                        "patch_len": self.patch_len,
                        "stride": self.stride,
                        "pooling": self.pooling,
                    },
                },
                str(path),
            )


else:

    class PatchTSTModel:  # pragma: no cover
        """torch 미설치 시 import 가능하지만 인스턴스화 불가."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("torch is required to use PatchTSTModel")

        @classmethod
        def load(cls, *args: Any, **kwargs: Any) -> "PatchTSTModel":
            raise ImportError("torch is required to load PatchTSTModel weights")
