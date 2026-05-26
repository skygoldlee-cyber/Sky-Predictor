from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


_DATE_PAT = re.compile(r"^(?P<stem>.+)_(?P<date>\d{8})\.pt$")


@dataclass
class WeightSelection:
    transformer_path: Optional[str]
    tft_path: Optional[str]
    reason: str


def _find_latest_dated_weight(weights_dir: Path, base_name: str, *, on_or_before: datetime) -> Optional[Path]:
    if not weights_dir.exists():
        return None

    cutoff = int(on_or_before.strftime("%Y%m%d"))
    best_date = -1
    best_path: Optional[Path] = None

    prefix = f"{base_name}_"
    for p in weights_dir.glob(f"{base_name}_*.pt"):
        try:
            m = _DATE_PAT.match(p.name)
            if not m:
                continue
            if not m.group("stem") == base_name:
                continue
            d = int(m.group("date"))
            if d <= cutoff and d > best_date:
                best_date = d
                best_path = p
        except Exception:
            continue

    return best_path


def select_weights_for_datetime(
    *,
    now: datetime,
    weights_dir: str = "prediction/weights",
    default_transformer: str = "transformer_5m.pt",
    default_tft: str = "tft_5m.pt",
    freeze_on_expiry_week: bool = True,
) -> WeightSelection:
    """Select weight files.

    Strategy:
    - Normal days: use default weights if present.
    - Expiry week (Mon~Thu, including second Thursday expiry day): freeze to the latest dated weights on or before
      week_start-1 day (i.e., previous Sunday's snapshot), if available.

    This keeps the live model stable during expiry week.
    """

    wdir = Path(weights_dir)
    t_path = wdir / default_transformer
    f_path = wdir / default_tft

    try:
        from core.utils import get_expiry_week_info

        info = get_expiry_week_info(now)
        is_expiry_week = bool(info.get("is_expiry_week"))
        expiry_dt = info.get("expiry_second_thursday")
    except Exception:
        is_expiry_week = False
        expiry_dt = None

    if freeze_on_expiry_week and is_expiry_week and isinstance(expiry_dt, datetime):
        week_start = expiry_dt - timedelta(days=int(expiry_dt.weekday()))
        # Only freeze during Mon~Thu of the expiry week.
        in_freeze_window = week_start.date() <= now.date() <= expiry_dt.date()
        # [IMP-1-4] 만기일(두 번째 목요일) 15:30 이후에는 freeze 해제.
        # 장마감 후에는 당일 만기 영향이 소멸되므로 기본 가중치로 복귀한다.
        if in_freeze_window and now.date() == expiry_dt.date():
            if now.hour > 15 or (now.hour == 15 and now.minute >= 30):
                in_freeze_window = False
        if not in_freeze_window:
            return WeightSelection(
                transformer_path=str(t_path) if t_path.exists() else None,
                tft_path=str(f_path) if f_path.exists() else None,
                reason="default",
            )

        cutoff_dt = week_start - timedelta(days=1)

        t_dated = _find_latest_dated_weight(wdir, "transformer_5m", on_or_before=cutoff_dt)
        f_dated = _find_latest_dated_weight(wdir, "tft_5m", on_or_before=cutoff_dt)

        reason = f"expiry_week_freeze_mon_thu cutoff={cutoff_dt.strftime('%Y-%m-%d')}"
        return WeightSelection(
            transformer_path=str(t_dated) if t_dated and t_dated.exists() else (str(t_path) if t_path.exists() else None),
            tft_path=str(f_dated) if f_dated and f_dated.exists() else (str(f_path) if f_path.exists() else None),
            reason=reason,
        )

    return WeightSelection(
        transformer_path=str(t_path) if t_path.exists() else None,
        tft_path=str(f_path) if f_path.exists() else None,
        reason="default",
    )
