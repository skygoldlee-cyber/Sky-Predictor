"""Offline training script for `PriceTransformer`.

Usage:
  # Step 1: build dataset
  python -m prediction.data_builder \
    --files ticks_replay_20250210.jsonl ticks_replay_20250211.jsonl \
    --out dataset_5m.npz --seq-len 60 --horizon 5

  # Step 2: train
  python train.py \
    --data dataset_5m.npz \
    --out prediction/weights/transformer_5m.pt \
    --epochs 50 --batch-size 256 --lr 1e-3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
from pathlib import Path
from datetime import datetime
import shutil

import numpy as np

from config import load_config, FUTURE_KNOWN_DIM
from prediction.features import ADAPT_KEYS, CD_KEYS, MS5_KEYS, OB_KEYS, get_opt_keys


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _focal_loss(prob: "torch.Tensor", y: "torch.Tensor", *, gamma: float, pos_weight: "torch.Tensor") -> "torch.Tensor":
    """Binary focal loss using probability output.

    prob: model output in [0,1]
    y: labels in {0,1}
    pos_weight: scalar tensor (applied to positive class)
    """

    pt = prob * y + (1.0 - prob) * (1.0 - y)
    weight = pos_weight * y + (1.0 - y)
    return -(weight * (1.0 - pt) ** float(gamma) * torch.log(pt + 1e-9)).mean()


# [IMP-2-3] set_seed → utils.py 로 이동하여 train_tft.py와 공유
from core.utils import set_seed  # noqa: E402


def load_data(path: str):
    """Load training dataset (X, y) from npz."""
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


def run(args: argparse.Namespace) -> None:
    """run.

Args:
    args:
"""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        from prediction.model import PriceTransformer
    except Exception as e:
        raise RuntimeError("torch is required for training; install torch first") from e

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    X_np, y_np, meta = load_data(args.data)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    x_mean_np = None
    x_std_np = None
    try:
        x_mean_np = np.mean(X_np, axis=(0, 1)).astype(np.float32).reshape(-1)
        x_std_np = np.std(X_np, axis=(0, 1)).astype(np.float32).reshape(-1)
        x_std_np = np.where(x_std_np > 1e-6, x_std_np, 1e-6).astype(np.float32)
    except Exception:
        x_mean_np = None
        x_std_np = None

    N, seq_len, feature_dim = X.shape

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

    # --multiscale-5m CLI 인자 우선, 없으면 config 값 사용
    multiscale_5m = bool(getattr(args, "multiscale_5m", False))
    if not multiscale_5m and cfg is not None:
        try:
            multiscale_5m = bool(getattr(getattr(cfg, "prediction", None), "multiscale_5m", False))
        except Exception:
            pass

    opt_keys = list(get_opt_keys(str(option_feature_set or "v1")))
    expected_dim = int(
        len(OB_KEYS)
        + len(CD_KEYS)
        + len(opt_keys)
        + (len(MS5_KEYS) if multiscale_5m else 0)
        + (len(ADAPT_KEYS) if adaptive_enabled else 0)
        + int(FUTURE_KNOWN_DIM)
    )
    try:
        if isinstance(meta, dict) and meta:
            m_dim = int(meta.get("feature_dim") or feature_dim)
            if int(feature_dim) != int(m_dim):
                raise ValueError(
                    f"Dataset feature_dim mismatch vs metadata: got {int(feature_dim)} but metadata.feature_dim={int(m_dim)}."
                )
            m_seq = int(meta.get("seq_len") or seq_len)
            if int(seq_len) != int(m_seq):
                raise ValueError(
                    f"Dataset seq_len mismatch vs metadata: got {int(seq_len)} but metadata.seq_len={int(m_seq)}."
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
    if int(feature_dim) != int(expected_dim):
        raise ValueError(
            f"Dataset feature_dim mismatch: got {int(feature_dim)} but expected={int(expected_dim)} "
            f"(adaptive_indicator.enabled={bool(adaptive_enabled)}, option_feature_set={str(option_feature_set)}, time_dim={int(FUTURE_KNOWN_DIM)}). "
            "Rebuild dataset to match current feature set."
        )

    if cfg is not None:
        try:
            expected_seq_len = int(getattr(getattr(cfg, "prediction", None), "seq_len", seq_len) or seq_len)
            if int(seq_len) != int(expected_seq_len):
                raise ValueError(
                    f"Dataset seq_len mismatch: got {int(seq_len)} but config.prediction.seq_len={int(expected_seq_len)}. "
                    "Rebuild dataset with matching --seq-len."
                )
        except Exception:
            pass
    try:
        pos_rate = float(y.mean().item())
    except Exception:
        pos_rate = float(np.mean(y_np))

    try:
        pos_n = int((y_np >= 0.5).sum())
        neg_n = int((y_np < 0.5).sum())
    except Exception:
        try:
            pos_n = int((y >= 0.5).sum().item())
            neg_n = int((y < 0.5).sum().item())
        except Exception:
            pos_n = -1
            neg_n = -1

    try:
        pos_rate = float(max(1e-6, min(1.0 - 1e-6, float(pos_rate))))
    except Exception:
        pos_rate = 0.5

    logger.info(
        "data: N=%d seq=%d feat=%d pos=%.1f%%",
        int(N),
        int(seq_len),
        int(feature_dim),
        float(pos_rate) * 100.0,
    )
    try:
        if int(pos_n) >= 0 and int(neg_n) >= 0:
            logger.info("label_counts: pos=%d neg=%d", int(pos_n), int(neg_n))
    except Exception:
        pass

    n_train = int(N * 0.8)
    train_ds = TensorDataset(X[:n_train], y[:n_train])
    val_ds = TensorDataset(X[n_train:], y[n_train:])

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False)

    model_kwargs = {
        "d_model": int(getattr(args, "d_model", 64) or 64),
        "n_heads": int(getattr(args, "n_heads", 4) or 4),
        "n_layers": int(getattr(args, "n_layers", 2) or 2),
        "d_ff": int(getattr(args, "d_ff", 128) or 128),
        "dropout": float(getattr(args, "dropout", 0.1) or 0.1),
        "pooling": str(getattr(args, "pooling", "cls") or "cls"),
    }

    model = PriceTransformer(
        feature_dim=int(feature_dim),
        seq_len=int(seq_len),
        **model_kwargs,
    ).to(device)

    logger.info("params: %d", sum(int(p.numel()) for p in model.parameters()))

    pos_rate = float(pos_rate)
    pos_weight_val = float((1.0 - pos_rate) / max(pos_rate, 1e-9))
    pos_weight = torch.tensor([pos_weight_val], device=device)
    try:
        logger.info("pos_weight: %.4f", float(pos_weight_val))
    except Exception:
        pass

    loss_name = str(getattr(args, "loss", "bce") or "bce").strip().lower()
    focal_gamma = 2.0
    try:
        focal_gamma = float(getattr(args, "focal_gamma", 2.0) or 2.0)
    except Exception:
        focal_gamma = 2.0
    if loss_name not in {"bce", "focal"}:
        loss_name = "bce"
    try:
        logger.info("loss: %s", loss_name)
        if loss_name == "focal":
            logger.info("focal_gamma: %.4f", float(focal_gamma))
    except Exception:
        pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.epochs),
        eta_min=float(args.lr) * 0.1,
    )

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict] = []
    try:
        patience = int(getattr(args, "patience", 0) or 0)
    except Exception:
        patience = 0
    try:
        min_delta = float(getattr(args, "min_delta", 0.0) or 0.0)
    except Exception:
        min_delta = 0.0
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

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            prob = model(xb)

            if loss_name == "focal":
                loss = _focal_loss(prob, yb, gamma=float(focal_gamma), pos_weight=pos_weight)
            else:
                # weighted log-loss (BCE-like) to handle class imbalance
                loss = -(
                    pos_weight * yb * torch.log(prob + 1e-9)
                    + (1.0 - yb) * torch.log(1.0 - prob + 1e-9)
                ).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = (model(xb) >= 0.5).float()
                correct += int((pred == yb).sum().item())
                total += int(len(yb))

                try:
                    y_true = yb
                    y_pred = pred
                    tp += int(((y_pred >= 0.5) & (y_true >= 0.5)).sum().item())
                    fp += int(((y_pred >= 0.5) & (y_true < 0.5)).sum().item())
                    tn += int(((y_pred < 0.5) & (y_true < 0.5)).sum().item())
                    fn += int(((y_pred < 0.5) & (y_true >= 0.5)).sum().item())
                except Exception:
                    pass

        val_acc = float(correct) / float(total) * 100.0 if total > 0 else 0.0
        avg_loss = float(total_loss) / float(max(1, len(train_loader)))

        buy_precision = 0.0
        sell_precision = 0.0
        buy_recall = 0.0
        sell_recall = 0.0
        try:
            buy_precision = float(tp) / float(max(1, tp + fp))
            sell_precision = float(tn) / float(max(1, tn + fn))
            buy_recall = float(tp) / float(max(1, tp + fn))
            sell_recall = float(tn) / float(max(1, tn + fp))
        except Exception:
            buy_precision = 0.0
            sell_precision = 0.0
            buy_recall = 0.0
            sell_recall = 0.0

        logger.info(
            "epoch %3d/%d loss=%.4f val_acc=%.2f%% P@BUY=%.1f%% P@SELL=%.1f%% R@BUY=%.1f%% R@SELL=%.1f%% (TP=%d FP=%d TN=%d FN=%d)",
            int(epoch),
            int(args.epochs),
            float(avg_loss),
            float(val_acc),
            float(buy_precision) * 100.0,
            float(sell_precision) * 100.0,
            float(buy_recall) * 100.0,
            float(sell_recall) * 100.0,
            int(tp),
            int(fp),
            int(tn),
            int(fn),
        )

        try:
            history.append(
                {
                    "epoch": int(epoch),
                    "loss": float(avg_loss),
                    "val_acc": float(val_acc),
                    "p_buy": float(buy_precision),
                    "p_sell": float(sell_precision),
                    "r_buy": float(buy_recall),
                    "r_sell": float(sell_recall),
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                }
            )
        except Exception:
            pass

        if val_acc > (float(best_val_acc) + float(min_delta)):
            best_val_acc = float(val_acc)
            best_epoch = int(epoch)
            patience_counter = 0
            try:
                if x_mean_np is not None and x_std_np is not None:
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "x_mean": x_mean_np.tolist(),
                            "x_std": x_std_np.tolist(),
                            "model_kwargs": dict(model_kwargs),
                        },
                        str(args.out),
                    )
                else:
                    model.save(str(args.out))
            except Exception:
                model.save(str(args.out))
            logger.info("  -> checkpoint saved (best=%.2f%%): %s", float(best_val_acc), str(args.out))
        else:
            patience_counter += 1
            if int(patience) > 0 and int(patience_counter) >= int(patience):
                logger.info(
                    "early stopping at epoch %d (best=%.2f%% at epoch %d)",
                    int(epoch),
                    float(best_val_acc),
                    int(best_epoch),
                )
                break

    logger.info("training done. best_val_acc=%.2f%% -> %s", float(best_val_acc), str(args.out))

    try:
        if not bool(getattr(args, "no_report", False)):
            out_path = Path(str(args.out))
            report_base = out_path.with_suffix("")
            report_json = report_base.with_name(f"{report_base.name}__report.json")
            report_csv = report_base.with_name(f"{report_base.name}__report.csv")

            payload = {
                "out": str(args.out),
                "created_at": datetime.now().isoformat(),
                "dataset": str(getattr(args, "data", "")),
                "config": str(getattr(args, "config", "config.json")),
                "schema_version": (meta.get("schema_version") if isinstance(meta, dict) else None),
                "feature_dim": int(feature_dim),
                "seq_len": int(seq_len),
                "best": {
                    "epoch": int(best_epoch),
                    "val_acc": float(best_val_acc),
                },
                "model_kwargs": dict(model_kwargs),
                "train": {
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "loss": str(loss_name),
                    "focal_gamma": float(focal_gamma),
                    "pos_weight": float(pos_weight_val),
                },
                "history": list(history),
            }

            try:
                report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            try:
                with report_csv.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(
                        f,
                        fieldnames=[
                            "epoch",
                            "loss",
                            "val_acc",
                            "p_buy",
                            "p_sell",
                            "r_buy",
                            "r_sell",
                            "tp",
                            "fp",
                            "tn",
                            "fn",
                        ],
                    )
                    w.writeheader()
                    for row in history:
                        try:
                            w.writerow(row)
                        except Exception:
                            pass
            except Exception:
                pass
    except Exception:
        pass

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
    """main.
"""
    parser = argparse.ArgumentParser(description="Train PriceTransformer")
    parser.add_argument("--config", default="config.json", help="config.json path (used to derive expected feature set)")
    parser.add_argument("--data", required=True, help="dataset.npz path")
    parser.add_argument("--out", default="prediction/weights/transformer_5m.pt")
    parser.add_argument("--tag-date", default="", help="Optional YYYYMMDD tag used to also save a dated checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (검증 정확도 개선 없을 때 최대 epoch 수). "
                             "0이면 비활성. 기본값: 10")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["cls", "recency_weighted"], default="cls")
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--multiscale-5m", action="store_true",
        help="5분봉 MS5_KEYS(8개) 포함 데이터셋 사용. data_builder --multiscale-5m 으로 빌드한 npz 전용.",
    )
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
