"""PatchTST 훈련 스크립트.

기존 train.py 와 동일한 데이터셋 형식(.npz)과 CLI 인터페이스를 사용한다.
PriceTransformer → PatchTSTModel 로만 교체되었으며
PatchTST 전용 하이퍼파라미터(--patch-len, --stride)가 추가되었다.

Usage:
  # Step 1: 데이터셋 빌드 (기존 data_builder 그대로 사용)
  python -m prediction.data_builder \\
    --files ticks_replay_20250210.jsonl ticks_replay_20250211.jsonl \\
    --out dataset_5m.npz --seq-len 60 --horizon 5

  # Step 2: PatchTST 훈련
  python train_patch_tst.py \\
    --data dataset_5m.npz \\
    --out prediction/weights/patch_tst_5m.pt \\
    --epochs 60 --batch-size 256 --lr 1e-3 \\
    --patch-len 8 --stride 4 \\
    --d-model 64 --n-heads 4 --n-layers 3 --d-ff 256

  # Step 3: predictor.py 에서 PatchTST 사용
  #   TransformerPredictor 의 model import 경로만 바꾸면 된다:
  #   from prediction.patch_tst_model import PatchTSTModel as PriceTransformer
  #   (또는 config.json 에 "model_class": "patch_tst" 를 추가해 자동 선택 가능)

검증 지표:
  - Val Accuracy (방향 정확도, 기존과 동일)
  - Val Brier Score (확률 캘리브레이션)
  - 두 지표 모두 best 체크포인트 저장에 사용 가능 (--monitor 옵션)
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
    """Binary focal loss using probability output."""
    pt = prob * y + (1.0 - prob) * (1.0 - y)
    weight = pos_weight * y + (1.0 - y)
    import torch
    return -(weight * (1.0 - pt) ** float(gamma) * torch.log(pt + 1e-9)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (train.py 와 동일)
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str):
    """npz 파일에서 (X, y, meta) 를 로드한다."""
    data = np.load(str(path))
    X = data["X"]
    y = data["y"]
    meta = None
    try:
        if "metadata" in data.files:
            ms = data["metadata"]
            if hasattr(ms, "tolist"):
                ms = ms.tolist()
            ms = str(ms or "")
            if ms.strip():
                meta = json.loads(ms)
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
        from prediction.patch_tst_model import PatchTSTModel
    except ImportError as e:
        raise RuntimeError(
            "torch 와 prediction/patch_tst_model.py 가 필요합니다. "
            "torch 를 먼저 설치하세요."
        ) from e

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    X_np, y_np, meta = load_data(args.data)
    y = torch.tensor(y_np, dtype=torch.float32)

    # ── 정규화 통계 계산 및 학습 데이터에 실제 적용 ──────────────────────────
    # [FIX-NORM-1] 기존 코드는 x_mean/x_std를 계산만 하고 X_np에 적용하지 않았다.
    # 결과: 모델이 비정규화 원시값(OI 수만 단위 등)으로 학습 → Attention Saturation
    #       → sigmoid 출력이 0.9993 / 0.0007 에 고착되는 포화 현상 발생.
    # 수정: 정규화를 학습 데이터에 실제 적용하고 체크포인트에 저장한다.
    x_mean_np = None
    x_std_np = None
    X_norm_np = X_np  # 폴백: 정규화 실패 시 원시값 사용
    try:
        x_mean_np = np.mean(X_np, axis=(0, 1)).astype(np.float32).reshape(-1)
        x_std_np  = np.std(X_np,  axis=(0, 1)).astype(np.float32).reshape(-1)
        x_std_np  = np.where(x_std_np > 1e-6, x_std_np, 1e-6).astype(np.float32)
        # ★ 실제 정규화 적용 (이 줄이 핵심 수정)
        X_norm_np = ((X_np - x_mean_np.reshape(1, 1, -1))
                     / x_std_np.reshape(1, 1, -1)).astype(np.float32)
        logger.info(
            "[NORM] 정규화 적용 완료: mean 범위 [%.3f, %.3f] std 범위 [%.4f, %.4f]",
            float(x_mean_np.min()), float(x_mean_np.max()),
            float(x_std_np.min()),  float(x_std_np.max()),
        )
    except Exception as _ne:
        logger.warning("[NORM] 정규화 통계 계산/적용 실패 — 비정규화로 학습 진행: %s", _ne)
        x_mean_np = None
        x_std_np  = None
        X_norm_np = X_np

    X = torch.tensor(X_norm_np, dtype=torch.float32)

    N, seq_len, feature_dim = X.shape

    # config 에서 feature_dim 검증
    cfg = None
    try:
        cfg = load_config(str(getattr(args, "config", "config.json") or "config.json"))
    except Exception:
        cfg = None

    adaptive_enabled = False
    option_feature_set = "v1"
    if cfg is not None:
        try:
            adaptive_enabled = bool(
                getattr(cfg, "adaptive_indicator", None) and cfg.adaptive_indicator.enabled
            )
        except Exception:
            adaptive_enabled = False
        try:
            option_feature_set = str(
                getattr(getattr(cfg, "prediction", None), "option_feature_set", "v1") or "v1"
            )
        except Exception as e:
            logger.debug("[TRAIN_PATCH_TST] option_feature_set parsing fallback: %s", e)
            option_feature_set = "v1"

    # --multiscale-5m CLI 인자 우선, 없으면 config 값 사용
    multiscale_5m = bool(getattr(args, "multiscale_5m", False))
    if not multiscale_5m and cfg is not None:
        try:
            multiscale_5m = bool(getattr(getattr(cfg, "prediction", None), "multiscale_5m", False))
        except Exception as e:
            logger.debug("[TRAIN_PATCH_TST] multiscale_5m parsing skipped: %s", e)

    opt_keys = list(get_opt_keys(str(option_feature_set or "v1")))
    expected_dim = int(
        len(OB_KEYS)
        + len(CD_KEYS)
        + len(opt_keys)
        + (len(MS5_KEYS) if multiscale_5m else 0)
        + (len(ADAPT_KEYS) if adaptive_enabled else 0)
        + int(FUTURE_KNOWN_DIM)
    )

    # 메타데이터 / config 일관성 검증 (train.py 와 동일)
    try:
        if isinstance(meta, dict) and meta:
            m_dim = int(meta.get("feature_dim") or feature_dim)
            if int(feature_dim) != int(m_dim):
                raise ValueError(
                    f"Dataset feature_dim mismatch vs metadata: "
                    f"got {feature_dim} but metadata.feature_dim={m_dim}."
                )
            m_seq = int(meta.get("seq_len") or seq_len)
            if int(seq_len) != int(m_seq):
                raise ValueError(
                    f"Dataset seq_len mismatch vs metadata: "
                    f"got {seq_len} but metadata.seq_len={m_seq}."
                )
    except Exception:
        raise

    if int(feature_dim) != int(expected_dim):
        raise ValueError(
            f"Dataset feature_dim mismatch: got {feature_dim} but expected={expected_dim} "
            f"(adaptive={adaptive_enabled}, opt={option_feature_set}, time_dim={FUTURE_KNOWN_DIM}). "
            "Rebuild dataset to match current feature set."
        )

    logger.info(
        "data: N=%d seq=%d feat=%d pos=%.1f%%",
        N, seq_len, feature_dim, float(y.mean().item()) * 100.0,
    )

    # ── 패치 수 사전 계산 및 안전 검증 ──────────────────────────────────────
    patch_len = int(getattr(args, "patch_len", 8) or 8)
    stride = int(getattr(args, "stride", 4) or 4)
    num_patches = (seq_len - patch_len) // stride + 1

    if num_patches < 2:
        raise ValueError(
            f"patch_len={patch_len}, stride={stride}, seq_len={seq_len} 조합으로 "
            f"num_patches={num_patches} < 2. patch_len 을 줄이거나 stride 를 줄이세요."
        )
    logger.info(
        "PatchTST: patch_len=%d stride=%d → num_patches=%d",
        patch_len, stride, num_patches,
    )

    # ── Train / Val 분할 ─────────────────────────────────────────────────────
    n_train = int(N * 0.8)
    train_ds = TensorDataset(X[:n_train], y[:n_train])
    val_ds = TensorDataset(X[n_train:], y[n_train:])
    train_loader = DataLoader(
        train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False)

    # ── 모델 생성 ─────────────────────────────────────────────────────────────
    model_kwargs = {
        "d_model": int(getattr(args, "d_model", 64) or 64),
        "n_heads": int(getattr(args, "n_heads", 4) or 4),
        "n_layers": int(getattr(args, "n_layers", 3) or 3),
        "d_ff": int(getattr(args, "d_ff", 256) or 256),
        "dropout": float(getattr(args, "dropout", 0.1) or 0.1),
        "pooling": str(getattr(args, "pooling", "cls") or "cls"),
        "patch_len": patch_len,
        "stride": stride,
    }

    model = PatchTSTModel(
        feature_dim=int(feature_dim),
        seq_len=int(seq_len),
        **model_kwargs,
    ).to(device)

    total_params = sum(int(p.numel()) for p in model.parameters())
    logger.info("PatchTSTModel params: %d", total_params)

    # ── 클래스 불균형 보정 ────────────────────────────────────────────────────
    pos_rate = float(max(1e-6, min(1.0 - 1e-6, float(y.mean().item()))))
    pos_weight_val = float((1.0 - pos_rate) / max(pos_rate, 1e-9))
    pos_weight = torch.tensor([pos_weight_val], device=device)
    logger.info("pos_weight: %.4f", pos_weight_val)

    loss_name = str(getattr(args, "loss", "bce") or "bce").strip().lower()
    focal_gamma = float(getattr(args, "focal_gamma", 2.0) or 2.0)
    if loss_name not in {"bce", "focal"}:
        loss_name = "bce"
    logger.info("loss: %s%s", loss_name, f" (gamma={focal_gamma})" if loss_name == "focal" else "")

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epochs), eta_min=float(args.lr) * 0.1
    )

    # ── 모니터링 설정 ─────────────────────────────────────────────────────────
    monitor = str(getattr(args, "monitor", "acc") or "acc").strip().lower()
    if monitor not in {"acc", "brier"}:
        monitor = "acc"
    logger.info("best checkpoint monitor: val_%s", monitor)

    best_val_metric = 0.0 if monitor == "acc" else float("inf")
    best_epoch = 0
    patience_counter = 0
    patience = int(getattr(args, "patience", 0) or 0)
    min_delta = float(getattr(args, "min_delta", 0.0) or 0.0)

    buy_threshold = float(getattr(args, "buy_threshold", 0.62) or 0.62)
    sell_threshold = float(getattr(args, "sell_threshold", 0.38) or 0.38)

    history: list[dict] = []
    Path(str(args.out)).parent.mkdir(parents=True, exist_ok=True)

    tag_date = str(getattr(args, "tag_date", "") or "").strip()
    if not tag_date:
        try:
            tag_date = datetime.now().strftime("%Y%m%d")
        except Exception:
            tag_date = ""

    # ── 훈련 루프 ─────────────────────────────────────────────────────────────
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
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

        # ── 검증 ─────────────────────────────────────────────────────────────
        model.eval()
        correct = 0
        total_val = 0
        brier_sum = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                prob = model(xb)

                pred = torch.where(
                    prob >= buy_threshold,
                    torch.ones_like(prob),
                    torch.where(prob <= sell_threshold, torch.zeros_like(prob), torch.full_like(prob, -1.0)),
                )
                label = torch.where(yb >= 0.5, torch.ones_like(yb), torch.zeros_like(yb))
                mask = pred >= 0.0
                correct += int(((pred == label) & mask).sum().item())
                total_val += int(mask.sum().item())

                # Brier Score
                brier_sum += float(((prob - yb) ** 2).sum().item())

        avg_train_loss = total_loss / max(1, len(train_loader))
        val_acc = float(correct / max(1, total_val))
        val_brier = brier_sum / max(1, len(val_ds))

        logger.info(
            "epoch %3d | loss %.4f | val_acc %.4f | val_brier %.4f | lr %.2e",
            epoch, avg_train_loss, val_acc, val_brier,
            float(scheduler.get_last_lr()[0]),
        )

        row = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_acc": round(val_acc, 4),
            "val_brier": round(val_brier, 4),
        }
        history.append(row)

        # ── Best 체크포인트 저장 ─────────────────────────────────────────────
        val_metric = val_acc if monitor == "acc" else (1.0 - val_brier)
        improved = (
            val_metric >= best_val_metric + min_delta
            if monitor == "acc"
            else val_metric >= best_val_metric + min_delta
        )

        if improved:
            best_val_metric = val_metric
            best_epoch = epoch
            patience_counter = 0

            ckpt = {
                "state_dict": model.state_dict(),
                "model_kwargs": {**model_kwargs},
                "epoch": epoch,
                "val_acc": round(val_acc, 4),
                "val_brier": round(val_brier, 4),
                "feature_dim": int(feature_dim),
                "seq_len": int(seq_len),
                "tag_date": tag_date,
            }
            if x_mean_np is not None and x_std_np is not None:
                ckpt["x_mean"] = x_mean_np.tolist()
                ckpt["x_std"] = x_std_np.tolist()

            torch.save(ckpt, str(args.out))
            logger.info("  → best saved (epoch %d, val_acc=%.4f, val_brier=%.4f)", epoch, val_acc, val_brier)
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                logger.info("Early stopping at epoch %d (best=%d)", epoch, best_epoch)
                break

    # ── CSV 이력 저장 ─────────────────────────────────────────────────────────
    csv_path = str(args.out).replace(".pt", "_history.csv")
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_acc", "val_brier"])
            writer.writeheader()
            writer.writerows(history)
        logger.info("history saved: %s", csv_path)
    except Exception as e:
        logger.warning("history save failed: %s", e)

    logger.info(
        "Done. Best epoch=%d val_acc=%.4f val_brier=%.4f → %s",
        best_epoch,
        history[best_epoch - 1]["val_acc"] if best_epoch > 0 else 0.0,
        history[best_epoch - 1]["val_brier"] if best_epoch > 0 else 0.0,
        args.out,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatchTST 훈련 스크립트")

    # ── 데이터 / 출력 ─────────────────────────────────────────────────────────
    p.add_argument("--data", required=True, help="훈련 데이터 경로 (.npz)")
    p.add_argument(
        "--out",
        default="prediction/weights/patch_tst_5m.pt",
        help="모델 가중치 저장 경로 (.pt)",
    )
    p.add_argument("--config", default="config.json", help="config.json 경로")

    # ── 훈련 하이퍼파라미터 ───────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", choices=["bce", "focal"], default="bce")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument(
        "--patience", type=int, default=0,
        help="Early stopping patience (0 = 비활성화)",
    )
    p.add_argument("--min-delta", type=float, default=0.0)

    # ── PatchTST 전용 하이퍼파라미터 ─────────────────────────────────────────
    p.add_argument(
        "--patch-len", type=int, default=8,
        help="패치 하나의 길이 (타임스텝 수). 기본 8 = 8분 단위 패치.",
    )
    p.add_argument(
        "--stride", type=int, default=4,
        help="패치 슬라이딩 간격. 기본 4 (50%% 겹침).",
    )

    # ── 모델 아키텍처 (train.py 와 동일) ─────────────────────────────────────
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=3, help="기본 3층 (PriceTransformer 는 2층)")
    p.add_argument("--d-ff", type=int, default=256, help="기본 256 (PriceTransformer 는 128)")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pooling", choices=["cls", "mean", "recency_weighted"], default="cls")

    # ── 신호 분류 임계값 ──────────────────────────────────────────────────────
    p.add_argument("--buy-threshold", type=float, default=0.62)
    p.add_argument("--sell-threshold", type=float, default=0.38)

    # ── 멀티스케일 ────────────────────────────────────────────────────────────
    p.add_argument(
        "--multiscale-5m", action="store_true",
        help="5분봉 MS5_KEYS(8개) 포함 데이터셋 사용. data_builder --multiscale-5m 으로 빌드한 npz 전용.",
    )

    # ── 모니터링 ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--monitor", choices=["acc", "brier"], default="acc",
        help="best 체크포인트 기준 지표. 'brier' 선택 시 Brier Score 최솟값 기준.",
    )
    p.add_argument("--tag-date", default="", help="체크포인트 태그 날짜 (YYYYMMDD)")

    args = p.parse_args()

    # argparse 는 --patch-len 을 patch_len 으로, --d-model 을 d_model 로 변환
    # 하지만 argparse default 는 dest 기준이므로 수동 매핑
    if not hasattr(args, "patch_len"):
        args.patch_len = getattr(args, "patch_len", 8)
    if not hasattr(args, "d_model"):
        args.d_model = getattr(args, "d_model", 64)
    if not hasattr(args, "n_heads"):
        args.n_heads = getattr(args, "n_heads", 4)
    if not hasattr(args, "n_layers"):
        args.n_layers = getattr(args, "n_layers", 3)
    if not hasattr(args, "d_ff"):
        args.d_ff = getattr(args, "d_ff", 256)
    if not hasattr(args, "focal_gamma"):
        args.focal_gamma = getattr(args, "focal_gamma", 2.0)
    if not hasattr(args, "buy_threshold"):
        args.buy_threshold = getattr(args, "buy_threshold", 0.62)
    if not hasattr(args, "sell_threshold"):
        args.sell_threshold = getattr(args, "sell_threshold", 0.38)

    return args


if __name__ == "__main__":
    run(_parse_args())
