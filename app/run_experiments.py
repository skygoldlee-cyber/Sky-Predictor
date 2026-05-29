from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str]) -> int:
    p = subprocess.run(cmd)
    return int(p.returncode)


def _load_npz_metadata(path: str) -> Dict[str, Any]:
    try:
        import numpy as np

        data = np.load(str(path), allow_pickle=True)
        if "metadata" not in data.files:
            return {}
        ms = data["metadata"]
        if hasattr(ms, "tolist"):
            ms = ms.tolist()
        ms = str(ms or "").strip()
        return json.loads(ms) if ms else {}
    except Exception:
        return {}


def _default_schema_tag(*, config_path: str) -> str:
    # Best-effort schema tag from current code + config.
    try:
        from constants import FUTURE_KNOWN_DIM
        from config import load_config
        from prediction.features import ADAPT_KEYS, CD_KEYS, OB_KEYS, get_opt_keys

        cfg = load_config(str(config_path or "config.json"))
        opt_set = "v1"
        adaptive_enabled = False
        try:
            opt_set = str(getattr(getattr(cfg, "prediction", None), "option_feature_set", "v1") or "v1")
        except Exception:
            opt_set = "v1"
        try:
            adaptive_enabled = bool(getattr(cfg, "adaptive_indicator", None) and cfg.adaptive_indicator.enabled)
        except Exception:
            adaptive_enabled = False

        opt_keys = list(get_opt_keys(str(opt_set)))
        adapt_dim = int(len(ADAPT_KEYS)) if adaptive_enabled else 0
        schema_version = (
            f"ob{len(OB_KEYS)}_cd{len(CD_KEYS)}_opt{len(opt_keys)}_adapt{adapt_dim}_time{int(FUTURE_KNOWN_DIM)}"
        )
        return str(schema_version)
    except Exception:
        return "schema_unknown"


def _make_run_name(
    *,
    schema_version: str,
    horizon: int,
    seq_len: int,
    min_profit_ticks: float,
    loss: str,
    focal_gamma: float,
    pooling: str,
    d_model: int,
    n_layers: int,
    d_ff: int,
) -> str:
    parts = [
        f"h{int(horizon)}",
        f"seq{int(seq_len)}",
        str(schema_version),
        f"mpt{min_profit_ticks:g}",
        f"loss{str(loss)}" + (f"g{focal_gamma:g}" if str(loss) == "focal" else ""),
        f"pool{str(pooling)}",
        f"dm{int(d_model)}",
        f"nl{int(n_layers)}",
        f"ff{int(d_ff)}",
        "norm1",
    ]
    return "__".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build datasets + run training experiments with consistent naming")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--files", nargs="+", required=True, help="ticks_replay_*.jsonl or .jsonl.gz")
    ap.add_argument("--out-dir", default="experiments")
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--min-profit-ticks", type=float, default=1.5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--skip-dataset", action="store_true")
    ap.add_argument("--dataset", default="", help="Use an existing dataset instead of building")
    args = ap.parse_args()

    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve file globs.
    files: List[str] = []
    for p in list(args.files or []):
        matches = glob.glob(str(p))
        if matches:
            files.extend(matches)
        else:
            files.append(str(p))
    files = sorted(list(dict.fromkeys(files)))

    dataset_path = str(args.dataset).strip()
    schema_version = ""
    if not dataset_path:
        schema_version = _default_schema_tag(config_path=str(args.config))
        dataset_path = str(out_dir / f"dataset__{schema_version}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")

    if (not bool(args.skip_dataset)) and (not Path(dataset_path).exists()):
        cmd = [
            sys.executable,
            "-m",
            "prediction.data_builder",
            "--files",
            *files,
            "--out",
            dataset_path,
            "--config",
            str(args.config),
            "--seq-len",
            str(int(args.seq_len)),
            "--horizon",
            str(int(args.horizon)),
            "--min-profit-ticks",
            str(float(args.min_profit_ticks)),
        ]
        rc = _run(cmd)
        if rc != 0:
            return rc

    meta = _load_npz_metadata(dataset_path)
    schema_version = str(meta.get("schema_version") or schema_version or "schema_unknown")

    # Experiments (extend as needed).
    exps: List[Dict[str, Any]] = [
        {
            "loss": "bce",
            "focal_gamma": 2.0,
            "pooling": "cls",
            "d_model": 64,
            "n_layers": 2,
            "d_ff": 128,
        },
        {
            "loss": "focal",
            "focal_gamma": 2.0,
            "pooling": "cls",
            "d_model": 64,
            "n_layers": 2,
            "d_ff": 128,
        },
        {
            "loss": "bce",
            "focal_gamma": 2.0,
            "pooling": "recency_weighted",
            "d_model": 64,
            "n_layers": 2,
            "d_ff": 128,
        },
        {
            "loss": "bce",
            "focal_gamma": 2.0,
            "pooling": "recency_weighted",
            "d_model": 128,
            "n_layers": 3,
            "d_ff": 256,
        },
    ]

    for e in exps:
        run_name = _make_run_name(
            schema_version=schema_version,
            horizon=int(args.horizon),
            seq_len=int(args.seq_len),
            min_profit_ticks=float(args.min_profit_ticks),
            loss=str(e["loss"]),
            focal_gamma=float(e.get("focal_gamma", 2.0)),
            pooling=str(e["pooling"]),
            d_model=int(e["d_model"]),
            n_layers=int(e["n_layers"]),
            d_ff=int(e["d_ff"]),
        )
        out_path = str(out_dir / f"transformer__{run_name}.pt")

        cmd = [
            sys.executable,
            "train.py",
            "--config",
            str(args.config),
            "--data",
            dataset_path,
            "--out",
            out_path,
            "--epochs",
            str(int(args.epochs)),
            "--batch-size",
            str(int(args.batch_size)),
            "--lr",
            str(float(args.lr)),
            "--loss",
            str(e["loss"]),
            "--focal-gamma",
            str(float(e.get("focal_gamma", 2.0))),
            "--pooling",
            str(e["pooling"]),
            "--d-model",
            str(int(e["d_model"])),
            "--n-layers",
            str(int(e["n_layers"])),
            "--d-ff",
            str(int(e["d_ff"])),
        ]
        rc = _run(cmd)
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
