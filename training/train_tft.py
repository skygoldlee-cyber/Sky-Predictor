"""Offline training script for `TemporalFusionTransformer`.

Usage:
  # Step 1: build dataset (TFT mode)
  python -m prediction.data_builder \
    --files ticks_replay_20250210.jsonl ticks_replay_20250211.jsonl \
    --out dataset_tft_5m.npz --seq-len 60 --horizon 5 --tft --tft-horizon-sec 300

  # Step 2: train
  python train_tft.py \
    --data dataset_tft_5m.npz \
    --out prediction/weights/tft_5m.pt \
    --epochs 80 --batch-size 128 --lr 5e-4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import shutil

import numpy as np

from config import load_config, FUTURE_KNOWN_DIM, HORIZON_SEC, PAST_UNKNOWN_DIM
from prediction.features import ADAPT_KEYS, CD_KEYS, OB_KEYS, get_opt_keys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# [IMP-2-3] set_seed → utils.py 로 이동하여 train.py와 공유
from core.utils import set_seed  # noqa: E402


def load_data(path: str):
    """Load TFT dataset (X, past_known, future_known, y) from npz."""
    data = np.load(str(path))
    X = data["X"]
    y = data["y"]
    PK = data["past_known"]
    FK = data["future_known"]
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
    return X, PK, FK, y, meta


def run(args: argparse.Namespace) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from prediction.tft_model import TemporalFusionTransformer
    except Exception as e:
        raise RuntimeError("torch is required for training; install torch first") from e

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    X_np, PK_np, FK_np, y_np, meta = load_data(args.data)

    X = torch.tensor(X_np, dtype=torch.float32)
    PK = torch.tensor(PK_np, dtype=torch.float32)
    FK = torch.tensor(FK_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    N, seq_len, past_unknown_dim = X.shape
    _, _, future_known_dim = PK.shape
    _, horizon, _ = FK.shape

    cfg = None
    try:
        cfg = load_config(str(getattr(args, "config", "config.json") or "config.json"))
    except Exception:
        cfg = None

    adaptive_enabled = False
    option_feature_set = "v1"
    if cfg is not None:
        try:
            adaptive_enabled = bool(getattr(cfg, "adaptive_indicator", None) and cfg.adaptive_indicator.enabled)
        except Exception:
            adaptive_enabled = False

        try:
            option_feature_set = str(getattr(getattr(cfg, "prediction", None), "option_feature_set", "v1") or "v1")
        except Exception:
            option_feature_set = "v1"

    opt_keys = list(get_opt_keys(str(option_feature_set or "v1")))
    expected_pu_cfg = int(len(OB_KEYS) + len(CD_KEYS) + len(opt_keys) + (len(ADAPT_KEYS) if adaptive_enabled else 0))
    try:
        if isinstance(meta, dict) and meta:
            m_dim = int(meta.get("feature_dim") or past_unknown_dim)
            if int(past_unknown_dim) != int(m_dim):
                raise ValueError(
                    f"Dataset past_unknown_dim mismatch vs metadata: got {int(past_unknown_dim)} but metadata.feature_dim={int(m_dim)}."
                )
            m_seq = int(meta.get("seq_len") or seq_len)
            if int(seq_len) != int(m_seq):
                raise ValueError(
                    f"Dataset seq_len mismatch vs metadata: got {int(seq_len)} but metadata.seq_len={int(m_seq)}."
                )
            m_h = int(meta.get("tft_horizon_sec") or horizon)
            if int(horizon) != int(m_h):
                raise ValueError(
                    f"Dataset horizon mismatch vs metadata: got {int(horizon)} but metadata.tft_horizon_sec={int(m_h)}."
                )
            if cfg is not None:
                m_opt = str(meta.get("option_feature_set") or option_feature_set)
                if str(option_feature_set or "v1").strip().lower() != str(m_opt).strip().lower():
                    raise ValueError(
                        f"Dataset option_feature_set mismatch: config={str(option_feature_set)} metadata={str(m_opt)}. Rebuild dataset."
                    )
                m_adapt = bool(meta.get("adaptive_enabled"))
                if bool(adaptive_enabled) != bool(m_adapt):
                    raise ValueError(
                        f"Dataset adaptive_enabled mismatch: config={bool(adaptive_enabled)} metadata={bool(m_adapt)}. Rebuild dataset."
                    )
    except Exception:
        raise
    if int(past_unknown_dim) != int(expected_pu_cfg):
        raise ValueError(
            f"Dataset past_unknown_dim mismatch: got {int(past_unknown_dim)} but expected={int(expected_pu_cfg)} "
            f"(adaptive_indicator.enabled={bool(adaptive_enabled)}, option_feature_set={str(option_feature_set)}). Rebuild dataset."
        )

    if cfg is not None:
        try:
            expected_seq_len = int(getattr(getattr(cfg, "prediction", None), "seq_len", seq_len) or seq_len)
            if int(seq_len) != int(expected_seq_len):
                raise ValueError(
                    f"Dataset seq_len mismatch: got {int(seq_len)} but config.prediction.seq_len={int(expected_seq_len)}. "
                    "Rebuild dataset with matching --seq-len."
                )
        except Exception as e:
            logger.debug("[TRAIN_TFT] seq_len validation skipped: %s", e)

        try:
            expected_h = int(getattr(getattr(cfg, "prediction", None), "tft_horizon", HORIZON_SEC) or HORIZON_SEC)
            if int(horizon) != int(expected_h):
                raise ValueError(
                    f"Dataset horizon mismatch: got {int(horizon)} but config.prediction.tft_horizon={int(expected_h)}. "
                    "Rebuild dataset with matching --tft-horizon-sec."
                )
        except Exception as e:
            logger.debug("[TRAIN_TFT] horizon validation skipped: %s", e)

    # Backward-compat: keep the constant check only when config is unavailable.
    # When config is present, the config-derived expected_pu_cfg check above is authoritative.
    if cfg is None:
        try:
            expected_pu = int(PAST_UNKNOWN_DIM)
        except Exception:
            expected_pu = int(past_unknown_dim)
        if int(past_unknown_dim) != int(expected_pu):
            raise ValueError(
                f"Dataset past_unknown_dim mismatch: got {int(past_unknown_dim)} but PAST_UNKNOWN_DIM={int(expected_pu)}. "
                "Rebuild dataset to match current feature set."
            )

    try:
        expected_fk = int(FUTURE_KNOWN_DIM)
    except Exception:
        expected_fk = int(future_known_dim)
    if int(future_known_dim) != int(expected_fk):
        raise ValueError(
            f"Dataset future_known_dim mismatch: got {int(future_known_dim)} but FUTURE_KNOWN_DIM={int(expected_fk)}. "
            "Rebuild dataset with current time feature config."
        )

    try:
        expected_h = int(HORIZON_SEC)
    except Exception:
        expected_h = int(horizon)
    if int(horizon) != int(expected_h):
        raise ValueError(
            f"Dataset horizon mismatch: got {int(horizon)} but HORIZON_SEC={int(expected_h)}. "
            "Rebuild dataset with matching --tft-horizon-sec."
        )

    try:
        pos_rate = float(y.mean().item())
    except Exception:
        pos_rate = float(np.mean(y_np))

    logger.info(
        "data: N=%d seq=%d past_unknown=%d future_known=%d horizon=%d pos=%.1f%%",
        int(N),
        int(seq_len),
        int(past_unknown_dim),
        int(future_known_dim),
        int(horizon),
        float(pos_rate) * 100.0,
    )

    # 시간순 분할 (반드시 시간순, shuffle 금지)
    n_train = int(N * 0.8)
    train_ds = TensorDataset(X[:n_train], PK[:n_train], FK[:n_train], y[:n_train])
    val_ds = TensorDataset(X[n_train:], PK[n_train:], FK[n_train:], y[n_train:])

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=min(256, int(args.batch_size) * 2), shuffle=False)

    model = TemporalFusionTransformer(
        past_unknown_dim=int(past_unknown_dim),
        future_known_dim=int(future_known_dim),
        seq_len=int(seq_len),
        horizon=int(horizon),
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
    ).to(device)

    logger.info("params: %d", sum(int(p.numel()) for p in model.parameters()))

    # pos_weight로 클래스 불균형 보정
    pos_weight = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-9)], device=device)
    criterion = torch.nn.BCELoss(weight=pos_weight, reduction="mean")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.epochs),
        eta_min=float(args.lr) * 0.1,
    )

    best_val_acc = 0.0
    Path(str(args.out)).parent.mkdir(parents=True, exist_ok=True)

    tag_date = ""
    try:
        tag_date = str(getattr(args, "tag_date", "") or "").strip()
    except Exception:
        tag_date = ""
    if not tag_date:
        try:
            tag_date = datetime.now().strftime("%Y%m%d")
        except Exception:
            tag_date = ""

    # NW-ARC-02: Early stopping
    patience = int(getattr(args, "patience", 0) or 0)
    min_delta = float(getattr(args, "min_delta", 0.0) or 0.0)
    patience_counter = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0

        for pu, pk, fk, labels in train_loader:
            pu = pu.to(device)
            pk = pk.to(device)
            fk = fk.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            prob = model(pu, pk, fk)
            loss = criterion(prob, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        scheduler.step()

        # DataLoader 기반 검증 루프
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for vpu, vpk, vfk, vlabels in val_loader:
                vpu = vpu.to(device)
                vpk = vpk.to(device)
                vfk = vfk.to(device)
                vlabels = vlabels.to(device)
                preds = (model(vpu, vpk, vfk) >= 0.5).float()
                correct += int((preds == vlabels).sum().item())
                total += int(len(vlabels))

        val_acc = float(correct) / float(total) * 100.0 if total > 0 else 0.0
        avg_loss = float(total_loss) / float(max(1, len(train_loader)))
        logger.info(
            "epoch %3d/%d loss=%.4f val_acc=%.2f%%",
            int(epoch),
            int(args.epochs),
            float(avg_loss),
            float(val_acc),
        )

        if val_acc > float(best_val_acc) + float(min_delta):
            best_val_acc = float(val_acc)
            patience_counter = 0
            model.save(str(args.out))
            logger.info("  -> checkpoint saved (best=%.2f%%): %s", float(best_val_acc), str(args.out))
        else:
            patience_counter += 1
            if int(patience) > 0 and int(patience_counter) >= int(patience):
                logger.info(
                    "Early stopping at epoch %d/%d (best_val_acc=%.2f%%, patience=%d)",
                    int(epoch), int(args.epochs), float(best_val_acc), int(patience),
                )
                break

    logger.info("training done. best_val_acc=%.2f%% -> %s", float(best_val_acc), str(args.out))

    try:
        if tag_date:
            out_path = Path(str(args.out))
            dated_path = out_path.with_name(f"{out_path.stem}_{tag_date}{out_path.suffix}")
            if out_path.exists() and out_path.is_file():
                shutil.copyfile(str(out_path), str(dated_path))
                logger.info("  -> dated checkpoint saved: %s", str(dated_path))
    except Exception as e:
        logger.warning("failed to save dated checkpoint: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TemporalFusionTransformer (TFT)")
    parser.add_argument("--config", default="config.json", help="config.json path (used to derive expected feature set)")
    parser.add_argument("--data", required=True, help="dataset_tft_*.npz path (must contain X, past_known, future_known, y)")
    parser.add_argument("--out", default="prediction/weights/tft_5m.pt")
    parser.add_argument("--tag-date", default="", help="Optional YYYYMMDD tag used to also save a dated checkpoint")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    # NW-ARC-02: Early stopping (train.py 와 동일한 인터페이스)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (검증 정확도 개선 없을 때 최대 epoch 수). 0이면 비활성. 기본값: 10")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Early stopping 최소 개선량 (퍼센트포인트). 기본값: 0.0")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
