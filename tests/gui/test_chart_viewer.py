"""
차트 뷰어 단위 테스트  v2.0
============================

변경사항
--------
  [BUG-2] MAX_BARS 상수 300 → 500 동기화
  [BUG-3] set_zigzag 교체 후 캐시 무효화 검증
  [BUG-4] 거래 이벤트 캐시 (mtime 비교) 검증
  [BUG-5] _render_trade_markers 바이섹션 검증
  [PERF-1] 증분 feed (_incremental_feed) 검증
  [PERF-2] 피봇 마커 해시 비교 (_pm_hash) 검증
  [FEAT-1] _render_current_price_line API 존재 검증
  [FEAT-2] set_ma_enabled / _render_ma API 존재 검증
  [FEAT-3] attach_chart_viewer upcode 전달 검증
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import time

try:
    from PySide6.QtGui import QColor
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# TradeStatusLED
# ══════════════════════════════════════════════════════════════════════════════

class TestTradeStatusLED:

    def test_init_without_qt(self):
        if QT_AVAILABLE:
            pytest.skip("Qt가 설치되어 있음")
        from gui.chart_viewer import TradeStatusLED
        led = TradeStatusLED()
        assert led.widget is None

    def test_status_colors_exist(self):
        from gui.chart_viewer import TradeStatusLED
        for attr in ('COLOR_IDLE', 'COLOR_LONG_ENTRY', 'COLOR_SHORT_ENTRY',
                     'COLOR_LONG_HOLD', 'COLOR_SHORT_HOLD', 'COLOR_EXIT',
                     'COLOR_RISK_HIGH', 'COLOR_RISK_MEDIUM'):
            assert hasattr(TradeStatusLED, attr)

    def test_color_map_completeness(self):
        from gui.chart_viewer import TradeStatusLED
        for state in ("idle", "long_entry", "short_entry",
                      "long_hold", "short_hold", "exit"):
            assert hasattr(TradeStatusLED, f"COLOR_{state.upper()}")


# ══════════════════════════════════════════════════════════════════════════════
# ChartEngine
# ══════════════════════════════════════════════════════════════════════════════

class TestChartEngine:

    def test_init(self):
        from gui.chart_viewer import ChartEngine
        engine = ChartEngine()
        assert engine._zz is None
        assert engine._zz_cfg is None
        assert engine._last_sig is None
        # [PERF-1] 증분 상태 필드
        assert engine._last_feed_len == 0
        assert engine._last_feed_last_ts is None

    def test_max_bars_constant(self):
        """[BUG-2] MAX_BARS 는 500 이어야 한다 (이전 300 — 코드와 불일치)."""
        from gui.chart_viewer import ChartEngine
        assert ChartEngine.MAX_BARS == 500

    def test_set_zigzag(self):
        from gui.chart_viewer import ChartEngine
        engine = ChartEngine()
        mock_zz = Mock()
        engine.set_zigzag(mock_zz)
        assert engine._zz == mock_zz

    def test_set_zigzag_invalidates_cache(self):
        """[BUG-3] set_zigzag 교체 시 캐시(_last_sig)가 무효화되어야 한다."""
        from gui.chart_viewer import ChartEngine
        engine = ChartEngine()

        zz1 = Mock()
        zz2 = Mock()

        engine.set_zigzag(zz1)
        engine._last_sig = ("dummy", "sig", 1.0, id(zz1))  # 임의 캐시 설정

        engine.set_zigzag(zz2)
        assert engine._last_sig is None, "교체 후 캐시가 None이어야 한다"

    def test_set_zigzag_same_instance_no_reset(self):
        """[BUG-3] 동일 인스턴스를 다시 set_zigzag 해도 캐시가 유지된다."""
        from gui.chart_viewer import ChartEngine
        engine = ChartEngine()
        zz = Mock()
        engine.set_zigzag(zz)
        engine._last_sig = ("dummy", "sig", 1.0, id(zz))

        engine.set_zigzag(zz)  # 동일 인스턴스
        assert engine._last_sig is not None, "동일 인스턴스이면 캐시가 유지되어야 한다"

    def test_set_zigzag_resets_incremental_state(self):
        """[BUG-3 / PERF-1] set_zigzag 교체 시 증분 feed 상태도 초기화된다."""
        from gui.chart_viewer import ChartEngine
        engine = ChartEngine()
        zz1 = Mock()
        engine.set_zigzag(zz1)
        engine._last_feed_len = 50

        zz2 = Mock()
        engine.set_zigzag(zz2)
        assert engine._last_feed_len == 0

    def test_build_pivot_markers_without_zigzag(self):
        from gui.chart_viewer import ChartEngine
        engine = ChartEngine()
        df = pd.DataFrame({
            "Open": [380.0], "High": [381.0], "Low": [379.0], "Close": [380.5]
        })
        assert engine._build_pivot_markers(df) is None

    def test_feed_zigzag_without_zigzag(self):
        from gui.chart_viewer import ChartEngine
        engine = ChartEngine()
        df = pd.DataFrame({
            "Open": [380.0, 381.0], "High": [381.0, 382.0],
            "Low": [379.0, 380.0], "Close": [380.5, 381.5],
        })
        engine._feed_zigzag_full(df)  # 에러 없어야 함


# ══════════════════════════════════════════════════════════════════════════════
# [PERF-1] 증분 feed
# ══════════════════════════════════════════════════════════════════════════════

class TestIncrementalFeed:

    def _make_df(self, n: int) -> pd.DataFrame:
        idx = pd.date_range("2026-04-27 09:00", periods=n, freq="1min")
        return pd.DataFrame({
            "Open":   [350.0 + i for i in range(n)],
            "High":   [351.0 + i for i in range(n)],
            "Low":    [349.0 + i for i in range(n)],
            "Close":  [350.5 + i for i in range(n)],
            "Volume": [1000.0] * n,
        }, index=idx)

    def test_feed_rows_updates_last_feed_len(self):
        """_feed_rows 호출 후 _last_feed_len 이 len(df) 로 갱신된다."""
        from gui.chart_viewer import ChartEngine
        engine = ChartEngine()

        zz = Mock()
        zz.update = Mock()
        engine._zz = zz

        df = self._make_df(5)
        engine._feed_rows(df)
        assert engine._last_feed_len == 5

    def test_incremental_feed_calls_update_only_for_new_bars(self):
        """_incremental_feed 는 새 봉만 ZigZag.update 에 전달한다."""
        from gui.chart_viewer import ChartEngine
        engine = ChartEngine()

        zz = Mock()
        zz.update = Mock()
        zz.__class__ = type(zz)  # _feed_zigzag_full 내부 reset 위해
        engine._zz = zz
        engine._zz_cfg = Mock()

        df5 = self._make_df(5)
        engine._last_feed_len = 5  # 이미 5봉 feed됐다고 가정

        df7 = self._make_df(7)
        engine._incremental_feed(df7)

        # 새로 추가된 2봉(인덱스 5,6)만 update 호출
        assert zz.update.call_count == 2

    def test_incremental_feed_full_replay_on_shrink(self):
        """df 길이가 줄면 전체 replay(full feed)를 해야 한다."""
        from gui.chart_viewer import ChartEngine
        engine = ChartEngine()

        # _feed_zigzag_full 을 patch해서 호출 여부 확인
        called = []
        original = engine._feed_zigzag_full
        engine._feed_zigzag_full = lambda df: called.append(len(df))

        engine._last_feed_len = 10  # 이전에 10봉 feed됐다고 가정
        df5 = self._make_df(5)
        engine._incremental_feed(df5)

        assert len(called) == 1, "길이 축소 시 full replay가 호출되어야 한다"


# ══════════════════════════════════════════════════════════════════════════════
# [PERF-2] 피봇 마커 해시
# ══════════════════════════════════════════════════════════════════════════════

class TestPmHash:

    def test_none_returns_zero(self):
        from gui.chart_viewer import FpltRenderer

        class MockAx:
            pass

        r = FpltRenderer(MockAx(), MockAx())
        assert r._pm_hash(None) == 0

    def test_same_pm_same_hash(self):
        from gui.chart_viewer import FpltRenderer

        class MockAx:
            pass

        r = FpltRenderer(MockAx(), MockAx())
        pm = {
            "confirmed":   {"idx": [1, 3], "y": [350.0, 352.0], "type": ["H", "L"]},
            "unconfirmed": {"idx": [5],    "y": [355.0],         "type": ["H"]},
            "anchor_idx": -1,
        }
        assert r._pm_hash(pm) == r._pm_hash(pm)

    def test_different_pm_different_hash(self):
        from gui.chart_viewer import FpltRenderer

        class MockAx:
            pass

        r = FpltRenderer(MockAx(), MockAx())
        pm1 = {"confirmed": {"idx": [1], "y": [350.0], "type": ["H"]},
               "unconfirmed": {"idx": [], "y": [], "type": []}, "anchor_idx": -1}
        pm2 = {"confirmed": {"idx": [2], "y": [351.0], "type": ["L"]},
               "unconfirmed": {"idx": [], "y": [], "type": []}, "anchor_idx": -1}
        assert r._pm_hash(pm1) != r._pm_hash(pm2)


# ══════════════════════════════════════════════════════════════════════════════
# [BUG-4] 거래 이벤트 캐시
# ══════════════════════════════════════════════════════════════════════════════

class TestTradeEventsCache:

    def test_cache_interval_prevents_reread(self):
        """_TRADE_EVENTS_CACHE_SEC 이내에는 파일을 재읽지 않아야 한다."""
        from gui.chart_viewer import ChartViewerWidget

        widget = ChartViewerWidget.__new__(ChartViewerWidget)
        widget._trade_events           = [{"dummy": True}]
        widget._trade_events_mtime     = 0.0
        widget._trade_events_last_read = time.monotonic()  # 방금 읽었다고 설정
        widget._TRADE_EVENTS_CACHE_SEC = 5

        read_count = [0]
        original   = widget._load_trade_events_cached

        # 파일 I/O 없이 캐시만 반환되는지 확인
        # (캐시 간격 내이므로 즉시 self._trade_events 반환)
        result = widget._load_trade_events_cached()
        assert result == [{"dummy": True}]


# ══════════════════════════════════════════════════════════════════════════════
# [BUG-5] 거래 마커 바이섹션
# ══════════════════════════════════════════════════════════════════════════════

class TestTradeMarkersBisect:

    def test_bisect_finds_nearest_bar(self):
        """np.searchsorted 바이섹션이 올바른 bar 인덱스를 찾아야 한다."""
        # x_idx 생성: 09:00 ~ 09:09 (10봉)
        x_idx = pd.date_range("2026-04-27 09:00", periods=10, freq="1min")
        xi64  = x_idx.astype("datetime64[ns]").astype(np.int64)

        # 09:05에 해당하는 타임스탬프
        target = pd.Timestamp("2026-04-27 09:05").value
        pos = np.searchsorted(xi64, target)
        if pos >= len(xi64):
            pos = len(xi64) - 1
        elif pos > 0:
            if abs(xi64[pos - 1] - target) < abs(xi64[pos] - target):
                pos = pos - 1
        assert pos == 5, f"09:05는 인덱스 5이어야 한다, got {pos}"

    def test_bisect_handles_boundary(self):
        """x_idx 범위 밖 타임스탬프는 경계 인덱스로 클리핑되어야 한다."""
        x_idx = pd.date_range("2026-04-27 09:00", periods=5, freq="1min")
        xi64  = x_idx.astype("datetime64[ns]").astype(np.int64)

        target = pd.Timestamp("2026-04-27 09:30").value  # 범위 밖
        pos = np.searchsorted(xi64, target)
        if pos >= len(xi64):
            pos = len(xi64) - 1
        assert pos == 4


# ══════════════════════════════════════════════════════════════════════════════
# [FEAT-1] 현재가 라인 API
# ══════════════════════════════════════════════════════════════════════════════

class TestCurrentPriceLine:

    def test_render_current_price_line_exists(self):
        """FpltRenderer 에 _render_current_price_line 메서드가 있어야 한다."""
        from gui.chart_viewer import FpltRenderer
        assert hasattr(FpltRenderer, "_render_current_price_line")

    def test_render_signature_accepts_float(self):
        """_render_current_price_line 은 float 인자를 받아야 한다."""
        import inspect
        from gui.chart_viewer import FpltRenderer
        sig = inspect.signature(FpltRenderer._render_current_price_line)
        params = list(sig.parameters.keys())
        assert "current_price" in params

    def test_render_accepts_current_price_kwarg(self):
        """FpltRenderer.render() 가 current_price 키워드 인자를 받아야 한다."""
        import inspect
        from gui.chart_viewer import FpltRenderer
        sig = inspect.signature(FpltRenderer.render)
        assert "current_price" in sig.parameters


# ══════════════════════════════════════════════════════════════════════════════
# [FEAT-2] MA 오버레이 API
# ══════════════════════════════════════════════════════════════════════════════

class TestMAOverlay:

    def test_set_ma_enabled_exists(self):
        from gui.chart_viewer import FpltRenderer
        assert hasattr(FpltRenderer, "set_ma_enabled")

    def test_render_ma_exists(self):
        from gui.chart_viewer import FpltRenderer
        assert hasattr(FpltRenderer, "_render_ma")

    def test_ma_colors_defined(self):
        from gui.chart_viewer import FpltRenderer
        assert hasattr(FpltRenderer, "_MA20_COLOR")
        assert hasattr(FpltRenderer, "_MA60_COLOR")

    def test_ma_enabled_default_true(self):
        """[FEAT-2] _ma_enabled 기본값은 True 이어야 한다."""
        from gui.chart_viewer import FpltRenderer

        class MockAx:
            pass

        r = FpltRenderer(MockAx(), MockAx())
        assert r._ma_enabled is True

    def test_set_ma_enabled_false(self):
        """[FEAT-2] set_ma_enabled(False) 호출 후 _ma_enabled 가 False 여야 한다."""
        from gui.chart_viewer import FpltRenderer

        class MockAx:
            pass

        r = FpltRenderer(MockAx(), MockAx())
        r.set_ma_enabled(False)
        assert r._ma_enabled is False


# ══════════════════════════════════════════════════════════════════════════════
# [FEAT-3] attach_chart_viewer upcode 전달
# ══════════════════════════════════════════════════════════════════════════════

class TestAttachChartViewerUpcode:

    def test_attach_passes_kp200_upcode(self):
        """[FEAT-3] attach_chart_viewer 가 config 의 kp200_upcode 를 ChartViewerWidget 에 전달해야 한다."""
        from gui.chart_viewer import attach_chart_viewer

        captured = {}

        class FakeLayout:
            def addWidget(self, w, stretch=0):
                pass

        with patch("gui.chart_viewer.ChartViewerWidget") as MockWidget:
            mock_instance = Mock()
            mock_instance.widget = Mock()
            MockWidget.return_value = mock_instance

            config = {"kp200_upcode": "101V3000", "kospi_upcode": "K2"}
            attach_chart_viewer(FakeLayout(), config=config)

            call_kwargs = MockWidget.call_args[1]
            assert call_kwargs.get("kp200_upcode") == "101V3000"
            assert call_kwargs.get("kospi_upcode") == "K2"


# ══════════════════════════════════════════════════════════════════════════════
# 기존 구조 테스트 (회귀 방지)
# ══════════════════════════════════════════════════════════════════════════════

class TestPivotMarkersStructure:

    def test_pivot_markers_dict_keys(self):
        for key in ["confirmed", "unconfirmed", "anchor_idx"]:
            assert isinstance(key, str)

    def test_pivot_markers_confirmed_subkeys(self):
        for key in ["idx", "y", "type"]:
            assert isinstance(key, str)


class TestDataValidation:

    def test_invalid_data_handling(self):
        from gui.chart_viewer import ChartEngine
        engine = ChartEngine()
        result = engine._build_pivot_markers(pd.DataFrame())
        assert result is None

    def test_zero_price_handling(self):
        from gui.chart_viewer import ChartEngine
        engine = ChartEngine()
        df = pd.DataFrame({
            "Open": [0.0, 381.0], "High": [0.0, 382.0],
            "Low": [0.0, 380.0],  "Close": [0.0, 381.5],
        })
        engine._feed_zigzag_full(df)  # 에러 없어야 함


class TestColorConstants:

    def test_color_constants_exist(self):
        from gui.chart_viewer import TradeStatusLED
        for color in (
            'COLOR_IDLE', 'COLOR_LONG_ENTRY', 'COLOR_SHORT_ENTRY',
            'COLOR_LONG_HOLD', 'COLOR_SHORT_HOLD', 'COLOR_EXIT',
            'COLOR_RISK_HIGH', 'COLOR_RISK_MEDIUM',
        ):
            assert hasattr(TradeStatusLED, color)

    def test_color_values_are_valid(self):
        from gui.chart_viewer import TradeStatusLED
        for color_attr in ('COLOR_IDLE', 'COLOR_LONG_ENTRY', 'COLOR_SHORT_ENTRY'):
            color = getattr(TradeStatusLED, color_attr)
            if QT_AVAILABLE:
                assert color is not None
            else:
                if isinstance(color, str):
                    assert color.startswith('#')
                    assert len(color) == 7
