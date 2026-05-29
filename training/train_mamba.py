"""Mamba SSM 훈련 스크립트.

train.py / train_patch_tst.py 와 동일한 데이터셋(.npz) 및 CLI 인터페이스.
Mamba 전용 하이퍼파라미터(--d-state, --seq-len-ext)가 추가됨.

핵심 차별점:
    - seq_len 을 240 까지 늘려도 선형 복잡도로 처리 (Transformer 대비 4배 맥락)
    - data_builder 는 --seq-len 240 으로 재빌드 필요
    - 기존 60-step 데이터셋도 그대로 사용 가능 (--seq-len 60)

Usage:
    # 240-step 데이터셋 빌드 (4시간 맥락)
    python -m prediction.data_builder \\
        --files ticks_replay_*.jsonl.gz \\
        --out dataset_4h.npz --seq-len 240 --horizon 5

    # Mamba 훈련
    python train_mamba.py \\
        --data dataset_4h.npz \\
        --out prediction/weights/mamba_4h.pt \\
        --epochs 60 --batch-size 128 --lr 5e-4 \\
        --d-state 16 --n-layers 4

    # config.json 에서 활성화
    # "model_class": "mamba"
    # "mamba_weights_path": "prediction/weights/mamba_4h.pt"
    # "mamba_seq_len": 240
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from config import load_config, FUTURE_KNOWN_DIM
from prediction.features import ADAPT_KEYS, CD_KEYS, MS5_KEYS, OB_KEYS, get_opt_keys
from core.utils import set_seed


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers (train.py 와 동일)
# ─────────────────────────────────────────────────────────────────────────────

def _focal_loss(prob, y, *, gamma: float, pos_weight):
    import torch
    pt = prob * y + (1.0 - prob) * (1.0 - y)
    weight = pos_weight * y + (1.0 - y)
    return -(weight * (1.0 - pt) ** float(gamma) * torch.log(pt + 1e-9)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str):
    data = np.load(str(path))
    X, y = data["X"], data["y"]
    meta = None
    try:
        if "metadata" in data.files:
            ms = data["metadata"]
            if hasattr(ms, "tolist"):
                ms = ms.tolist()
            if str(ms or "").strip():
                meta = json.loads(str(ms))
    except Exception:
        meta = None
    return X, y, meta


# ─────────────────────────────────────────────────────────────────────────────
# 훈련 메인 루프
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from prediction.mamba_model import MambaModel
    except ImportError as e:
        raise RuntimeError(
            "torch 와 prediction/mamba_model.py 가 필요합니다."
        ) from e

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    X_np, y_np, meta = load_data(args.data)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    x_mean_np = x_std_np = None
    try:
        x_mean_np = np.mean(X_np, axis=(0, 1)).astype(np.float32).reshape(-1)
        x_std_np  = np.std(X_np,  axis=(0, 1)).astype(np.float32).reshape(-1)
        x_std_np  = np.where(x_std_np > 1e-6, x_std_np, 1e-6).astype(np.float32)
    except Exception:
        pass

    N, seq_len, feature_dim = X.shape

    # ── config & feature dim 검증 ─────────────────────────────────────────
    cfg = None
    try:
        cfg = load_config(str(getattr(args, "config", "config.json") or "config.json"))
    except Exception:
        pass

    adaptive_enabled   = False
    option_feature_set = "v1"
    multiscale_5m      = bool(getattr(args, "multiscale_5m", False))

    if cfg is not None:
        try:
            adaptive_enabled = bool(
                getattr(cfg, "adaptive_indicator", None) and cfg.adaptive_indicator.enabled
            )
        except Exception:
            pass
        try:
            option_feature_set = str(
                getattr(getattr(cfg, "prediction", None), "option_feature_set", "v1") or "v1"
            )
        except Exception:
            pass
        if not multiscale_5m:
            try:
                multiscale_5m = bool(
                    getattr(getattr(cfg, "prediction", None), "multiscale_5m", False)
                )
            except Exception:
                pass

    opt_keys = list(get_opt_keys(str(option_feature_set or "v1")))
    expected_dim = int(
        len(OB_KEYS) + len(CD_KEYS) + len(opt_keys)
        + (len(MS5_KEYS) if multiscale_5m else 0)
        + (len(ADAPT_KEYS) if adaptive_enabled else 0)
        + int(FUTURE_KNOWN_DIM)
    )

    if isinstance(meta, dict) and meta:
        m_dim = int(meta.get("feature_dim") or feature_dim)
        if int(feature_dim) != int(m_dim):
            raise ValueError(
                f"Dataset feature_dim mismatch: got {feature_dim}, metadata={m_dim}"
            )

    if int(feature_dim) != int(expected_dim):
        raise ValueError(
            f"Dataset feature_dim mismatch: got {feature_dim}, expected={expected_dim} "
            f"(adaptive={adaptive_enabled}, opt={option_feature_set}, ms5={multiscale_5m}). "
            "Rebuild dataset to match current feature set."
        )

    logger.info(
        "data: N=%d seq=%d feat=%d pos=%.1f%%",
        N, seq_len, feature_dim, float(y.mean().item()) * 100.0,
    )

    # ── Train / Val 분할 ─────────────────────────────────────────────────
    n_train = int(N * 0.8)
    train_loader = DataLoader(
        TensorDataset(X[:n_train], y[:n_train]),
        batch_size=int(args.batch_size), shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(X[n_train:], y[n_train:]),
        batch_size=int(args.batch_size), shuffle=False,
    )

    # ── 모델 생성 ─────────────────────────────────────────────────────────
    model_kwargs = {
        "d_model":  int(getattr(args, "d_model",  64) or 64),
        "d_state":  int(getattr(args, "d_state",  16) or 16),
        "n_layers": int(getattr(args, "n_layers",  4) or 4),
        "dropout":  float(getattr(args, "dropout", 0.1) or 0.1),
        "pooling":  str(getattr(args, "pooling",  "last") or "last"),
    }

    model = MambaModel(
        feature_dim=int(feature_dim),
        seq_len=int(seq_len),
        **model_kwargs,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("MambaModel params: %d  (seq_len=%d, d_state=%d)", total_params, seq_len, model_kwargs["d_state"])

    # ── 클래스 불균형 보정 ────────────────────────────────────────────────
    pos_rate = float(max(1e-6, min(1.0 - 1e-6, float(y.mean().item()))))
    pos_weight_val = float((1.0 - pos_rate) / max(pos_rate, 1e-9))
    pos_weight = torch.tensor([pos_weight_val], device=device)
    logger.info("pos_weight: %.4f", pos_weight_val)

    loss_name   = str(getattr(args, "loss", "bce") or "bce").strip().lower()
    focal_gamma = float(getattr(args, "focal_gamma", 2.0) or 2.0)
    if loss_name not in {"bce", "focal"}:
        loss_name = "bce"
    logger.info("loss: %s%s", loss_name, f" (gamma={focal_gamma})" if loss_name == "focal" else "")

    # ── Optimizer / Scheduler ─────────────────────────────────────────────
    # Mamba 는 일반적으로 AdamW + 낮은 lr (5e-4) 을 권장
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epochs), eta_min=float(args.lr) * 0.05
    )

    # ── 훈련 루프 ─────────────────────────────────────────────────────────
    monitor       = str(getattr(args, "monitor", "acc") or "acc").strip().lower()
    best_metric   = 0.0 if monitor == "acc" else float("inf")
    best_epoch    = 0
    patience_cnt  = 0
    patience      = int(getattr(args, "patience", 0) or 0)
    min_delta     = float(getattr(args, "min_delta", 0.0) or 0.0)
    buy_threshold = float(getattr(args, "buy_threshold", 0.62) or 0.62)
    sell_threshold= float(getattr(args, "sell_threshold", 0.38) or 0.38)
    history: list[dict] = []

    Path(str(args.out)).parent.mkdir(parents=True, exist_ok=True)
    tag_date = str(getattr(args, "tag_date", "") or "")
    if not tag_date:
        try:
            tag_date = datetime.now().strftime("%Y%m%d")
        except Exception:
            tag_date = ""

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            prob = model(xb)

            if loss_name == "focal":
                loss = _focal_loss(prob, yb, gamma=focal_gamma, pos_weight=pos_weight)
            else:
                loss = -(
                    pos_weight * yb * torch.log(prob + 1e-9)
                    + (1.0 - yb) * torch.log(1.0 - prob + 1e-9)
                ).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        scheduler.step()

        # ── 검증 ─────────────────────────────────────────────────────────
        model.eval()
        correct = total_val = 0
        brier_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                prob = model(xb)
                pred = torch.where(
                    prob >= buy_threshold, torch.ones_like(prob),
                    torch.where(prob <= sell_threshold, torch.zeros_like(prob),
                                torch.full_like(prob, -1.0)),
                )
                label = (yb >= 0.5).float()
                mask  = pred >= 0.0
                correct   += int(((pred == label) & mask).sum().item())
                total_val += int(mask.sum().item())
                brier_sum += float(((prob - yb) ** 2).sum().item())

        avg_loss  = total_loss / max(1, len(train_loader))
        val_acc   = float(correct / max(1, total_val))
        val_brier = brier_sum / max(1, len(X) - n_train)

        logger.info(
            "epoch %3d | loss %.4f | val_acc %.4f | val_brier %.4f | lr %.2e",
            epoch, avg_loss, val_acc, val_brier,
            float(scheduler.get_last_lr()[0]),
        )
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "val_acc":    round(val_acc, 4),
            "val_brier":  round(val_brier, 4),
        })

        # ── Best 체크포인트 저장 ─────────────────────────────────────────
        val_metric = val_acc if monitor == "acc" else (1.0 - val_brier)
        improved   = val_metric >= best_metric + min_delta

        if improved:
            best_metric = val_metric
            best_epoch  = epoch
            patience_cnt = 0
            ckpt = {
                "state_dict": model.state_dict(),
                "model_kwargs": {**model_kwargs},
                "epoch": epoch,
                "val_acc":   round(val_acc, 4),
                "val_brier": round(val_brier, 4),
                "feature_dim": int(feature_dim),
                "seq_len":     int(seq_len),
                "tag_date":    tag_date,
            }
            if x_mean_np is not None and x_std_np is not None:
                ckpt["x_mean"] = x_mean_np.tolist()
                ckpt["x_std"]  = x_std_np.tolist()
            torch.save(ckpt, str(args.out))
            logger.info(
                "  → best saved (epoch %d, val_acc=%.4f, val_brier=%.4f)",
                epoch, val_acc, val_brier,
            )
        else:
            patience_cnt += 1
            if patience > 0 and patience_cnt >= patience:
                logger.info("Early stopping at epoch %d (best=%d)", epoch, best_epoch)
                break

    # ── CSV 이력 저장 ─────────────────────────────────────────────────────
    csv_path = str(args.out).replace(".pt", "_history.csv")
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_acc", "val_brier"])
            writer.writeheader()
            writer.writerows(history)
        logger.info("history saved: %s", csv_path)
    except Exception as e:
        logger.warning("history save failed: %s", e)

    if best_epoch > 0:
        best = history[best_epoch - 1]
        logger.info(
            "Done. best epoch=%d val_acc=%.4f val_brier=%.4f → %s",
            best_epoch, best["val_acc"], best["val_brier"], args.out,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mamba SSM 훈련 스크립트")

    # ── 데이터 / 출력 ─────────────────────────────────────────────────────
    p.add_argument("--data",   required=True, help="훈련 데이터 경로 (.npz)")
    p.add_argument("--out",    default="prediction/weights/mamba_4h.pt")
    p.add_argument("--config", default="config.json")

    # ── 훈련 하이퍼파라미터 ───────────────────────────────────────────────
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--batch-size",  type=int,   default=128,
                   help="Mamba 는 sequential scan 으로 메모리 사용량이 많으므로 128 권장")
    p.add_argument("--lr",          type=float, default=5e-4,
                   help="Mamba 권장 lr (Transformer 보다 낮게)")
    p.add_argument("--loss",        choices=["bce", "focal"], default="bce")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--patience",    type=int,   default=0)
    p.add_argument("--min-delta",   type=float, default=0.0)

    # ── Mamba 전용 하이퍼파라미터 ─────────────────────────────────────────
    p.add_argument("--d-state",  type=int, default=16,
                   help="SSM 내부 상태 차원 N. 클수록 장기 의존성 강화. (기본 16)")
    p.add_argument("--n-layers", type=int, default=4,
                   help="MambaBlock 스택 수 (기본 4)")
    p.add_argument("--d-model",  type=int, default=64)
    p.add_argument("--dropout",  type=float, default=0.1)
    p.add_argument("--pooling",
                   choices=["last", "mean", "recency_weighted"], default="last",
                   help="'last': SSM 마지막 상태 (권장), 'mean': 전체 평균")

    # ── 신호 분류 임계값 ──────────────────────────────────────────────────
    p.add_argument("--buy-threshold",  type=float, default=0.62)
    p.add_argument("--sell-threshold", type=float, default=0.38)

    # ── 멀티스케일 ────────────────────────────────────────────────────────
    p.add_argument("--multiscale-5m", action="store_true",
                   help="MS5_KEYS(8개) 포함 데이터셋 사용")

    # ── 모니터링 ──────────────────────────────────────────────────────────
    p.add_argument("--monitor",  choices=["acc", "brier"], default="acc")
    p.add_argument("--tag-date", default="")

    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
