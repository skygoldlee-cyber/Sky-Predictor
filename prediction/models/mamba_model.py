"""Mamba SSM 모델 구현 (순수 PyTorch).

PriceTransformer / PatchTSTModel 과 완전히 동일한 외부 인터페이스:
    forward(x)  → (B,) sigmoid 출력
    load(path)  → MambaModel
    save(path)

핵심 특성:
    - 시퀀스 길이에 선형 복잡도 O(L) — Transformer 의 O(L²) 대비 우수
    - seq_len=240 (4시간 1분봉) 까지 지연 없이 처리 가능
    - mamba-ssm 패키지 의존성 없음: 순수 PyTorch 로 구현
      (실제 CUDA 커널 최적화가 없으므로 속도는 참조 구현 수준)

아키텍처 (Simplified Mamba S4D):
    Input (B, L, F)
      → LinearProjection → (B, L, d_model)
      → N × MambaBlock
           ├─ SSM 경로: x → Linear → Δ,A,B,C 파라미터 → SSM recurrence
           └─ Gate 경로: x → Linear → SiLU → 곱
      → LayerNorm → Linear(1) → Sigmoid
      → (B,)

KP200 1분봉 권장 설정:
    seq_len   = 240   # 4시간 (기존 60 → 4배 확장 가능)
    d_model   = 64
    d_state   = 16    # SSM 상태 차원
    n_layers  = 4
    dropout   = 0.1

torch 가 없으면 클래스는 import 가능하지만 인스턴스화 시 ImportError.
이는 model.py / patch_tst_model.py 와 동일한 폴백 패턴.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

from config import PAST_UNKNOWN_DIM


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class _SSMKernel(nn.Module):
        """단순화된 S4D (Diagonal State Space) 커널.

        Mamba 의 핵심 SSM 연산을 순수 PyTorch 로 구현.
        실제 Mamba 논문의 selective scan 과 동일한 수식을 따르되
        병렬화는 sequential scan 으로 처리 (CUDA 커널 미사용).

        Args:
            d_model:  입력/출력 차원.
            d_state:  SSM 내부 상태 차원 N.
            dt_rank:  Δ (time step) 투영 rank.
            bias:     Linear 레이어 bias 사용 여부.
        """

        def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            dt_rank: Optional[int] = None,
            bias: bool = False,
        ) -> None:
            super().__init__()
            self.d_model = int(d_model)
            self.d_state = int(d_state)
            self.dt_rank = int(dt_rank or max(1, d_model // 16))

            # ── 입력 투영 (x → x_ssm, z) ────────────────────────────────
            self.in_proj = nn.Linear(self.d_model, self.d_model * 2, bias=bool(bias))

            # ── SSM 파라미터 ─────────────────────────────────────────────
            # x_proj: x_ssm → (Δ, B, C)
            self.x_proj = nn.Linear(
                self.d_model,
                self.dt_rank + self.d_state * 2,
                bias=False,
            )
            # dt_proj: Δ_rank → d_model (time step 확장)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_model, bias=True)
            nn.init.uniform_(
                self.dt_proj.bias,
                -math.log(d_model), math.log(d_model),
            )

            # A: (d_model, d_state) — 대각 상태 전이 행렬 (log 공간 초기화)
            A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0)
            A = A.expand(self.d_model, -1)
            self.A_log = nn.Parameter(torch.log(A))

            # D: skip connection 스케일
            self.D = nn.Parameter(torch.ones(self.d_model))

            # 출력 투영
            self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bool(bias))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """(B, L, d_model) → (B, L, d_model)"""
            B, L, D = x.shape

            # 입력 투영 → SSM 경로(x_ssm)와 게이트 경로(z)
            xz = self.in_proj(x)                          # (B, L, 2D)
            x_ssm, z = xz.chunk(2, dim=-1)               # 각 (B, L, D)

            # SSM 파라미터 계산
            x_dbl = self.x_proj(x_ssm)                   # (B, L, dt_rank + 2*N)
            dt, B_param, C_param = x_dbl.split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = self.dt_proj(dt)                         # (B, L, D)
            dt = F.softplus(dt)                           # Δ > 0 보장

            A = -torch.exp(self.A_log.float())            # (D, N) — 음수로 안정적 발산 방지

            # 이산화: ZOH (Zero-Order Hold)
            # Ā = exp(Δ * A),  B̄ = (Ā - I) / A * B ≈ Δ * B (단순화)
            dA = torch.exp(
                dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
            )  # (B, L, D, N)
            dB = dt.unsqueeze(-1) * B_param.unsqueeze(2)  # (B, L, D, N)

            # Sequential scan: h_t = Ā * h_{t-1} + B̄ * x_t
            h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
            ys = []
            for t in range(L):
                h = dA[:, t] * h + dB[:, t] * x_ssm[:, t].unsqueeze(-1)
                # h: (B, D, N)  C_param[:,t]: (B, N) → unsqueeze(1) → (B, 1, N)
                y_t = (h * C_param[:, t].unsqueeze(1)).sum(dim=-1)  # (B, D)
                ys.append(y_t)
            y = torch.stack(ys, dim=1)                    # (B, L, D)

            # Skip connection + 게이트
            y = y + x_ssm * self.D.unsqueeze(0).unsqueeze(0)
            y = y * F.silu(z)

            return self.out_proj(y)


    class _MambaBlock(nn.Module):
        """Mamba 블록: SSMKernel + 잔차 연결 + LayerNorm (Pre-Norm).

        Args:
            d_model:  모델 차원.
            d_state:  SSM 내부 상태 차원.
            dropout:  드롭아웃 비율.
        """

        def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(int(d_model))
            self.ssm  = _SSMKernel(int(d_model), d_state=int(d_state))
            self.drop = nn.Dropout(float(dropout))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """(B, L, d_model) → (B, L, d_model) — Pre-LN 잔차"""
            return x + self.drop(self.ssm(self.norm(x)))


    class MambaModel(nn.Module):
        """Mamba SSM 기반 방향 예측 모델.

        PriceTransformer / PatchTSTModel 과 완전히 동일한 외부 인터페이스.

        Args:
            feature_dim:  입력 피처 차원 (PAST_UNKNOWN_DIM).
            d_model:      SSM 내부 차원.
            d_state:      SSM 상태 차원 N (클수록 장기 의존성 포착 강화).
            n_layers:     MambaBlock 스택 수.
            seq_len:      입력 시퀀스 길이 (60 ~ 240 권장).
            dropout:      드롭아웃 비율.
            pooling:      최종 풀링 방식. 'last' | 'mean' | 'recency_weighted'.
        """

        def __init__(
            self,
            feature_dim: int = PAST_UNKNOWN_DIM,
            d_model: int = 64,
            d_state: int = 16,
            n_layers: int = 4,
            seq_len: int = 60,
            dropout: float = 0.1,
            pooling: str = "last",
        ) -> None:
            super().__init__()
            self.feature_dim = int(feature_dim)
            self.d_model     = int(d_model)
            self.d_state     = int(d_state)
            self.seq_len     = int(seq_len)

            p = str(pooling or "last").strip().lower()
            if p not in {"last", "mean", "recency_weighted"}:
                p = "last"
            self.pooling = p

            # 입력 투영
            self.input_proj = nn.Linear(int(feature_dim), int(d_model))
            self.input_norm = nn.LayerNorm(int(d_model))

            # Mamba 블록 스택
            self.blocks = nn.ModuleList([
                _MambaBlock(int(d_model), d_state=int(d_state), dropout=float(dropout))
                for _ in range(int(n_layers))
            ])

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
                x: (B, seq_len, feature_dim)

            Returns:
                (B,) sigmoid 확률 [0, 1]
            """
            # 입력 투영
            h = self.input_norm(self.input_proj(x))   # (B, L, d_model)

            # Mamba 블록 순차 통과
            for block in self.blocks:
                h = block(h)                           # (B, L, d_model)

            # 풀링
            if self.pooling == "mean":
                pooled = h.mean(dim=1)
            elif self.pooling == "recency_weighted":
                L = h.size(1)
                w = torch.exp(torch.linspace(-2.0, 0.0, L, device=h.device))
                w = w / (w.sum() + 1e-9)
                pooled = (h * w.view(1, -1, 1)).sum(dim=1)
            else:  # "last" — SSM 의 자연스러운 선택 (마지막 상태가 전체 이력 요약)
                pooled = h[:, -1, :]

            return self.head(pooled).squeeze(-1)       # (B,)

        @classmethod
        def load(
            cls,
            weights_path: str,
            *,
            feature_dim: int = PAST_UNKNOWN_DIM,
            seq_len: int = 60,
            device: str = "cpu",
            **kwargs: Any,
        ) -> "MambaModel":
            """저장된 가중치에서 모델을 로드한다.

            PriceTransformer.load() / PatchTSTModel.load() 와 동일한 시그니처.
            체크포인트의 'model_kwargs' 키로 아키텍처를 복원한다.
            """
            obj = torch.load(
                str(weights_path), map_location=str(device), weights_only=False
            )

            arch_keys = {"d_model", "d_state", "n_layers", "dropout", "pooling"}
            init_kwargs: dict[str, Any] = {
                k: v for k, v in kwargs.items() if k in arch_keys
            }

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
            """가중치를 아키텍처 정보와 함께 저장한다."""
            Path(str(path)).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": self.state_dict(),
                    "model_kwargs": {
                        "d_model":  self.d_model,
                        "d_state":  self.d_state,
                        "n_layers": len(self.blocks),
                        "pooling":  self.pooling,
                    },
                },
                str(path),
            )


else:

    class MambaModel:  # pragma: no cover
        """torch 미설치 시 import 가능하지만 인스턴스화 불가."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("torch is required to use MambaModel")

        @classmethod
        def load(cls, *args: Any, **kwargs: Any) -> "MambaModel":
            raise ImportError("torch is required to load MambaModel weights")
