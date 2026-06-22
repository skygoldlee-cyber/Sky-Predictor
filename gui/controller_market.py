"""장 운영 시간 판별 (`gui_controller` 3단계 분리).

KOSPI 현물 정규장을 가정: 평일 09:00~15:45, ``datetime``은 **로컬(통상 KST)** 기준 naive 값으로 사용한다.
(``gui_controller.run()``에서 ``datetime.now()``와 동일한 기준.)

리플레이 버튼·비동기 태스크 연동은 UI 상태에 묶여 있어 ``gui_controller``에 남긴다.
"""

from __future__ import annotations

from datetime import datetime, timedelta

__all__ = [
    "MARKET_OPEN_HOUR",
    "MARKET_OPEN_MINUTE",
    "MARKET_CLOSE_HOUR",
    "MARKET_CLOSE_MINUTE",
    "is_market_open",
    "next_market_open",
]

# 정규장 (가장 단순 모델; 공휴일·조기종료 미반영)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 0
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 45


def is_market_open(dt: datetime) -> bool:
    """평일이고 정규장 시각 범위 안이면 True."""
    try:
        if int(dt.weekday()) >= 5:
            return False
        open_dt = dt.replace(
            hour=MARKET_OPEN_HOUR,
            minute=MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0,
        )
        close_dt = dt.replace(
            hour=MARKET_CLOSE_HOUR,
            minute=MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0,
        )
        return open_dt <= dt <= close_dt
    except Exception:
        return False


def next_market_open(dt: datetime) -> datetime:
    """다음 장 시작 시각(naive). 주말·장 종료 후는 다음 평일 09:00."""
    base = dt.replace(second=0, microsecond=0)
    open_today = base.replace(
        hour=MARKET_OPEN_HOUR,
        minute=MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0,
    )
    close_today = base.replace(
        hour=MARKET_CLOSE_HOUR,
        minute=MARKET_CLOSE_MINUTE,
        second=0,
        microsecond=0,
    )

    if int(base.weekday()) >= 5:
        days_ahead = 7 - int(base.weekday())
        cand = (base + timedelta(days=days_ahead)).replace(
            hour=MARKET_OPEN_HOUR,
            minute=MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0,
        )
        return cand

    if base < open_today:
        return open_today
    if base > close_today:
        cand = (base + timedelta(days=1)).replace(
            hour=MARKET_OPEN_HOUR,
            minute=MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0,
        )
        while int(cand.weekday()) >= 5:
            cand = (cand + timedelta(days=1)).replace(
                hour=MARKET_OPEN_HOUR,
                minute=MARKET_OPEN_MINUTE,
                second=0,
                microsecond=0,
            )
        return cand

    return open_today
