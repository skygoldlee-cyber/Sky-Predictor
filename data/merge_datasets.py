from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np


_DATE_RE = re.compile(r"(\d{8})")
_DATE_FULL_RE = re.compile(r"^\d{8}$")


def _extract_yyyymmdd(path: Path) -> Optional[str]:
    m = _DATE_RE.search(path.name)
    if not m:
        return None
    return m.group(1)


def _select_last_n(files: List[Path], n: int) -> List[Path]:
    dated: List[Tuple[str, Path]] = []
    undated: List[Path] = []

    for p in files:
        d = _extract_yyyymmdd(p)
        if d is None:
            undated.append(p)
        else:
            dated.append((d, p))

    dated.sort(key=lambda x: x[0])
    if n > 0:
        dated = dated[-int(n) :]

    # If nothing had a date, fall back to lexicographic filename order.
    if not dated and undated:
        undated_sorted = sorted(undated, key=lambda x: x.name)
        return undated_sorted[-int(n) :] if n > 0 else undated_sorted

    return [p for _, p in dated]


def _load_rollover_marker(path: Path) -> Optional[str]:
    try:
        if not path.exists() or not path.is_file():
            return None
        s = path.read_text(encoding="utf-8").strip()
        return s if _DATE_FULL_RE.fullmatch(s) else None
    except Exception:
        return None


def _save_rollover_marker(path: Path, yyyymmdd: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(yyyymmdd), encoding="utf-8")
    except Exception:
        return


def _default_monthly_marker(now: datetime) -> Path:
    """Return a monthly marker path for the relevant expiry cycle.

    Marker is keyed by the most recent expiry month (YYYYMM) so each option cycle is isolated.
    """
    exp = _most_recent_expiry(now)
    if exp is None:
        return Path(".rollover_start.txt")
    yyyymm = exp.strftime("%Y%m")
    return Path(f".rollover_start_{yyyymm}.txt")


def _is_rollover_day(now: datetime) -> bool:
    """Legacy helper (kept for backward compatibility)."""
    try:
        from core.utils import get_option_expiry_date

        expiry_dt = get_option_expiry_date(now.year, now.month)
        rollover_dt = expiry_dt + timedelta(days=1)
        return now.date() == rollover_dt.date() and int(now.weekday()) == 4
    except Exception:
        return False


def _most_recent_expiry(now: datetime) -> Optional[datetime]:
    """Return the most recent monthly expiry datetime (second Thursday).

    If `now` is before this month's expiry, return previous month's expiry.
    """
    try:
        from core.utils import get_option_expiry_date

        cur = get_option_expiry_date(now.year, now.month)
        if now.date() >= cur.date():
            return cur

        # previous month
        pm = int(now.month) - 1
        py = int(now.year)
        if pm <= 0:
            pm = 12
            py -= 1
        return get_option_expiry_date(py, pm)
    except Exception:
        return None


def _expected_rollover_start(expiry_dt: datetime) -> datetime:
    """First weekday after expiry (Fri/Mon depending on weekends/holidays).

    Holidays are not known; merge routine additionally requires today's dataset to exist.
    """
    d = expiry_dt + timedelta(days=1)
    while int(d.weekday()) >= 5:
        d = d + timedelta(days=1)
    return d


def _has_dataset_for_date(files: List[Path], yyyymmdd: str) -> bool:
    for p in files:
        d = _extract_yyyymmdd(p)
        if d == yyyymmdd:
            return True
    return False


def _detect_rollover_start(now: datetime, files: List[Path], marker_val: Optional[str]) -> Optional[str]:
    """Return rollover start YYYYMMDD to set as marker, or None.

    We set rollover marker on the first run after expiry where today's dataset exists.
    This is robust to holidays because the training routine typically runs only when a
    daily dataset was produced.
    """
    exp = _most_recent_expiry(now)
    if exp is None:
        return None
    start = _expected_rollover_start(exp)
    start_s = start.strftime("%Y%m%d")

    today_s = now.strftime("%Y%m%d")
    if now.date() < start.date():
        return None

    # If marker is already set to a date >= expected start, do not reset again.
    if marker_val and _DATE_FULL_RE.fullmatch(marker_val):
        try:
            if int(marker_val) >= int(start_s):
                return None
        except Exception:
            pass

    # Trigger only when today's dataset exists (proxy for first trading day observed).
    if _has_dataset_for_date(files, today_s):
        return today_s
    return None


def _filter_by_rollover(files: List[Path], rollover_start_yyyymmdd: str) -> List[Path]:
    keep: List[Tuple[str, Path]] = []
    for p in files:
        d = _extract_yyyymmdd(p)
        if d is None:
            continue
        if d >= str(rollover_start_yyyymmdd):
            keep.append((d, p))
    keep.sort(key=lambda x: x[0])
    return [p for _, p in keep]


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(path))
    return {k: data[k] for k in data.files}


def _validate_and_collect(
    npz_list: List[Dict[str, np.ndarray]],
    *,
    tft: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    PKs: List[np.ndarray] = []
    FKs: List[np.ndarray] = []

    for obj in npz_list:
        if "X" not in obj or "y" not in obj:
            raise ValueError("npz must contain X and y")

        X = obj["X"]
        y = obj["y"]
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (N, seq_len, feat), got shape={X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D (N,), got shape={y.shape}")
        if int(X.shape[0]) != int(y.shape[0]):
            raise ValueError(f"N mismatch: X={X.shape} y={y.shape}")

        if bool(tft):
            if "past_known" not in obj or "future_known" not in obj:
                raise ValueError("TFT merge requires past_known and future_known in npz")
            PK = obj["past_known"]
            FK = obj["future_known"]
            if PK.ndim != 3 or FK.ndim != 3:
                raise ValueError(f"past_known/future_known must be 3D, got PK={PK.shape} FK={FK.shape}")
            if int(PK.shape[0]) != int(X.shape[0]) or int(FK.shape[0]) != int(X.shape[0]):
                raise ValueError(f"N mismatch: X={X.shape} PK={PK.shape} FK={FK.shape}")
            if int(PK.shape[1]) != int(X.shape[1]):
                raise ValueError(f"seq_len mismatch: X={X.shape} PK={PK.shape}")

            PKs.append(PK)
            FKs.append(FK)

        Xs.append(X)
        ys.append(y)

    return Xs, ys, (PKs if tft else None), (FKs if tft else None)


def merge(files: List[Path], *, tft: bool, out: Path, max_samples: int = 0) -> None:
    npz_list = [_load_npz(p) for p in files]
    Xs, ys, PKs, FKs = _validate_and_collect(npz_list, tft=bool(tft))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    if int(max_samples) > 0 and int(X.shape[0]) > int(max_samples):
        # Keep most recent samples assuming input files are time-sorted.
        X = X[-int(max_samples) :]
        y = y[-int(max_samples) :]

    out.parent.mkdir(parents=True, exist_ok=True)

    if bool(tft):
        assert PKs is not None and FKs is not None
        PK = np.concatenate(PKs, axis=0)
        FK = np.concatenate(FKs, axis=0)
        if int(max_samples) > 0 and int(PK.shape[0]) > int(max_samples):
            PK = PK[-int(max_samples) :]
            FK = FK[-int(max_samples) :]
        np.savez(str(out), X=X, y=y, past_known=PK, future_known=FK)
        print(f"saved: {out} X={X.shape} y={y.shape} past_known={PK.shape} future_known={FK.shape}")
        return

    np.savez(str(out), X=X, y=y)
    print(f"saved: {out} X={X.shape} y={y.shape}")


def main(now_fn: Optional[Callable[[], datetime]] = None) -> None:
    """Merge daily dataset files.

    Args:
        now_fn: 시간 함수 (테스트/백테스트용 주입 가능)
    """
    parser = argparse.ArgumentParser(description="Merge daily dataset_*.npz files into one training dataset")
    parser.add_argument("--pattern", required=True, help="Glob pattern for input NPZ files (e.g. dataset_tft_*.npz)")
    parser.add_argument("--last", type=int, default=20, help="Select last N dates from matched files (default: 20)")
    parser.add_argument("--tft", action="store_true", help="Merge TFT datasets (requires past_known/future_known)")
    parser.add_argument("--out", required=True, help="Output NPZ path")
    parser.add_argument("--max-samples", type=int, default=0, help="If >0, keep only last K samples after merge")
    parser.add_argument("--asof", default="", help="Override current date (YYYYMMDD) for rollover logic")
    parser.add_argument(
        "--no-reset-on-rollover",
        dest="reset_on_rollover",
        action="store_false",
        default=True,
        help="Disable rollover reset (by default, reset starts from the first trading day after expiry, when a daily dataset exists)",
    )
    parser.add_argument(
        "--rollover-marker",
        default="",
        help="Marker file storing rollover start YYYYMMDD (used to exclude pre-rollover data). If empty, a monthly marker (.rollover_start_YYYYMM.txt) is used.",
    )
    args = parser.parse_args()

    files = [Path(p) for p in sorted(Path(".").glob(str(args.pattern)))]
    if not files:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    # Rollover reset: from the first trading day after expiry (observed), invalidate all prior training days.
    _now = now_fn if now_fn is not None else datetime.now
    now = _now()
    asof = str(getattr(args, "asof", "") or "").strip()
    if asof and _DATE_FULL_RE.fullmatch(asof):
        try:
            now = datetime.strptime(asof, "%Y%m%d")
        except Exception:
            now = _now()

    marker_arg = str(getattr(args, "rollover_marker", "") or "").strip()
    marker_path = Path(marker_arg) if marker_arg else _default_monthly_marker(now)
    rollover_start = _load_rollover_marker(marker_path)
    try:
        print(f"[rollover] marker_path={marker_path} marker_val={rollover_start or '(none)'}")
    except Exception:
        pass
    if bool(getattr(args, "reset_on_rollover", True)):
        detected = _detect_rollover_start(now, files, rollover_start)
        if detected:
            rollover_start = str(detected)
            _save_rollover_marker(marker_path, str(rollover_start))
            print(f"[rollover] reset window start: {rollover_start} (marker={marker_path})")

    candidate_files = files
    if rollover_start:
        filtered = _filter_by_rollover(files, str(rollover_start))
        if filtered:
            candidate_files = filtered
            print(f"[rollover] using only files >= {rollover_start} (count={len(candidate_files)})")
        else:
            print(f"[rollover] marker={rollover_start} but no matched files after marker; falling back to default selection")

    selected = _select_last_n(candidate_files, int(args.last))
    if not selected:
        raise SystemExit("No files selected")

    print("selected files:")
    for p in selected:
        print(f"  - {p}")

    merge(selected, tft=bool(args.tft), out=Path(str(args.out)), max_samples=int(args.max_samples))


if __name__ == "__main__":
    main()
