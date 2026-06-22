"""Shared time feature builder for TFT inputs."""

from __future__ import annotations

from datetime import datetime
from typing import List

import numpy as np


def build_time_features(dt: datetime) -> List[float]:
    try:
        from core.utils import get_expiry_week_info

        expiry_info = get_expiry_week_info(dt)
        days_to_expiry = float(expiry_info.get("days_to_expiry"))
        is_expiry_week = 1.0 if bool(expiry_info.get("is_expiry_week")) else 0.0
    except Exception:
        days_to_expiry = 0.0
        is_expiry_week = 0.0

    dte_scaled = 0.0
    try:
        dte_scaled = min(float(days_to_expiry) / 30.0, 1.0)
        if not np.isfinite(dte_scaled):
            dte_scaled = 0.0
    except Exception:
        dte_scaled = 0.0

    dow = 0
    try:
        dow = int(dt.weekday())
        if dow < 0 or dow > 6:
            dow = 0
    except Exception:
        dow = 0

    dow_onehot = [0.0] * 7
    try:
        dow_onehot[dow] = 1.0
    except Exception:
        dow_onehot = [0.0] * 7

    try:
        session_start = dt.replace(hour=8, minute=45, second=0, microsecond=0)  # KP200 선물 세션 시작 08:45
        session_end   = dt.replace(hour=15, minute=45, second=0, microsecond=0)  # KP200 선물 세션 종료 15:45
        # [IMP-3-2] 세션 외 시간(장 전/후)은 is_session=0 으로 마스킹하고
        # tod_sin/tod_cos를 0/1(중립값)으로 고정하여 왜곡을 방지한다.
        # NOTE: KP200 선물 실제 세션 08:45~15:45 기준 적용 (390분 → 411분).
        #       모델 재학습 시 data_builder.py 동일 기준으로 일치시킬 것.
        is_session = 1.0 if session_start <= dt <= session_end else 0.0
        if is_session:
            denom = float((session_end - session_start).total_seconds() or 1.0)
            offset = float((dt - session_start).total_seconds())
            frac = max(0.0, min(1.0, offset / denom))
            tod_sin = float(np.sin(2.0 * np.pi * frac))
            tod_cos = float(np.cos(2.0 * np.pi * frac))
        else:
            tod_sin = 0.0
            tod_cos = 1.0
    except Exception:
        is_session = 0.0
        tod_sin = 0.0
        tod_cos = 1.0

    return [float(dte_scaled)] + [float(x) for x in dow_onehot] + [float(tod_sin), float(tod_cos)] + [float(is_expiry_week)]
