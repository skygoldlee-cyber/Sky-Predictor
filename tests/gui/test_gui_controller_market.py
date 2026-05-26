"""gui_controller_market: 장중 판별 단위 테스트 (네트워크/UI 없음)."""

from __future__ import annotations

from datetime import datetime

import pytest

from gui.controller_market import is_market_open, next_market_open


def test_weekend_not_open() -> None:
    # 2026-04-04 토요일
    dt = datetime(2026, 4, 4, 10, 0, 0)
    assert is_market_open(dt) is False


def test_weekday_session_open() -> None:
    dt = datetime(2026, 4, 6, 10, 30, 0)  # 월요일 장중
    assert is_market_open(dt) is True


def test_next_open_from_weekend() -> None:
    dt = datetime(2026, 4, 4, 12, 0, 0)  # 토
    n = next_market_open(dt)
    assert n.weekday() == 0
    assert n.hour == 9 and n.minute == 0


def test_next_open_after_close_same_day() -> None:
    dt = datetime(2026, 4, 6, 16, 0, 0)  # 월, 장 후
    n = next_market_open(dt)
    assert n.date() > dt.date()
    assert n.weekday() < 5
    assert n.hour == 9
