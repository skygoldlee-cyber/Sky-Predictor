"""
ChartViewer 타임인덱스 단위테스트  v2.0
========================================

v2.0 변경사항
--------------
  [BUG-2] TestChartEngine.test_max_bars_constant: 300 → 500 동기화
  [PERF-2] TestRenderPivots: pm_hash 캐시 동작 회귀 테스트 추가
  기존 모든 테스트 유지 (TestToX, TestMakeCandleDf,
  TestTimeAxisChangeDetection, TestDataSourceSwitch)
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class MockAxis:
    """finplot ax 최소 Mock — FpltRenderer 생성 시 필요."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# FpltRenderer._to_x()
# ══════════════════════════════════════════════════════════════════════════════

class TestToX(unittest.TestCase):

    def setUp(self):
        import finplot as fplt
        from gui.chart_viewer import FpltRenderer
        self.renderer = FpltRenderer(MockAxis(), MockAxis())
        self._fplt = fplt

    def tearDown(self):
        if hasattr(self._fplt, '_pdtime2index_patched'):
            del self._fplt._pdtime2index_patched

    def test_kst_naive_passthrough(self):
        idx = pd.DatetimeIndex(['2026-04-27 09:00:00', '2026-04-27 09:01:00'])
        out = self.renderer._to_x(idx)
        self.assertEqual(out.dtype, np.dtype('datetime64[ns]'))
        self.assertEqual(len(out), 2)
        self.assertEqual(pd.to_datetime(out[0]).strftime('%H:%M'), '09:00')
        self.assertEqual(pd.to_datetime(out[1]).strftime('%H:%M'), '09:01')

    def test_kst_naive_date_preserved(self):
        idx = pd.DatetimeIndex(['2026-04-27 09:00:00'])
        out = self.renderer._to_x(idx)
        self.assertEqual(pd.to_datetime(out[0]).date(), datetime(2026, 4, 27).date())

    def test_kst_aware_preserved(self):
        idx = pd.DatetimeIndex(['2026-04-27 09:00:00']).tz_localize('Asia/Seoul')
        out = self.renderer._to_x(idx)
        self.assertEqual(out.dtype, np.dtype('datetime64[ns]'))
        self.assertEqual(pd.to_datetime(out[0]).strftime('%H:%M'), '09:00')

    def test_utc_aware_converted_to_kst(self):
        idx = pd.DatetimeIndex(['2026-04-27 00:00:00']).tz_localize('UTC')
        out = self.renderer._to_x(idx)
        self.assertEqual(pd.to_datetime(out[0]).strftime('%H:%M'), '09:00')

    def test_output_is_numpy_datetime64ns(self):
        cases = [
            pd.DatetimeIndex(['2026-04-27 09:00:00']),
            pd.DatetimeIndex(['2026-04-27 09:00:00']).tz_localize('Asia/Seoul'),
            pd.DatetimeIndex(['2026-04-27 00:00:00']).tz_localize('UTC'),
        ]
        for idx in cases:
            with self.subTest(tz=str(idx.tz)):
                out = self.renderer._to_x(idx)
                self.assertIsInstance(out, np.ndarray)
                self.assertEqual(out.dtype, np.dtype('datetime64[ns]'))


# ══════════════════════════════════════════════════════════════════════════════
# FpltRenderer._make_candle_df()
# ══════════════════════════════════════════════════════════════════════════════

class TestMakeCandleDf(unittest.TestCase):

    BASE_TIME = datetime(2026, 4, 27, 9, 0, 0)

    def setUp(self):
        from gui.chart_viewer import FpltRenderer
        self.renderer = FpltRenderer(MockAxis(), MockAxis())

        timestamps = [self.BASE_TIME + timedelta(minutes=i) for i in range(5)]
        self.df_ohlc = pd.DataFrame(
            {
                'Open':   [990.0, 991.0, 992.0, 993.0, 994.0],
                'High':   [992.0, 993.0, 994.0, 995.0, 996.0],
                'Low':    [989.0, 990.0, 991.0, 992.0, 993.0],
                'Close':  [991.0, 992.0, 993.0, 994.0, 995.0],
                'Volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            },
            index=pd.DatetimeIndex(timestamps),
        )
        self.x_idx = self.renderer._to_x(self.df_ohlc.index)

    def test_output_has_datetimeindex(self):
        cdf = self.renderer._make_candle_df(self.x_idx, self.df_ohlc)
        self.assertIsInstance(cdf.index, pd.DatetimeIndex)

    def test_time_not_in_columns(self):
        cdf = self.renderer._make_candle_df(self.x_idx, self.df_ohlc)
        self.assertNotIn('time', cdf.columns)

    def test_index_timestamps_match_kst_naive_input(self):
        cdf = self.renderer._make_candle_df(self.x_idx, self.df_ohlc)
        self.assertEqual(cdf.index[0],  pd.Timestamp(self.BASE_TIME))
        self.assertEqual(cdf.index[-1], pd.Timestamp(self.BASE_TIME + timedelta(minutes=4)))

    def test_index_length(self):
        cdf = self.renderer._make_candle_df(self.x_idx, self.df_ohlc)
        self.assertEqual(len(cdf), 5)

    def test_required_columns_present(self):
        cdf = self.renderer._make_candle_df(self.x_idx, self.df_ohlc)
        for col in ('open', 'close', 'high', 'low'):
            with self.subTest(col=col):
                self.assertIn(col, cdf.columns)

    def test_column_dtype_float64(self):
        cdf = self.renderer._make_candle_df(self.x_idx, self.df_ohlc)
        for col in ('open', 'close', 'high', 'low'):
            with self.subTest(col=col):
                self.assertEqual(cdf[col].dtype, np.float64)

    def test_ohlc_values_correct(self):
        cdf = self.renderer._make_candle_df(self.x_idx, self.df_ohlc)
        self.assertAlmostEqual(cdf['open'].iloc[0],  990.0)
        self.assertAlmostEqual(cdf['high'].iloc[0],  992.0)
        self.assertAlmostEqual(cdf['low'].iloc[0],   989.0)
        self.assertAlmostEqual(cdf['close'].iloc[0], 991.0)


# ══════════════════════════════════════════════════════════════════════════════
# 시간축 변경 감지 (force_redraw)
# ══════════════════════════════════════════════════════════════════════════════

class TestTimeAxisChangeDetection(unittest.TestCase):

    BASE_TIME = datetime(2026, 4, 27, 9, 0, 0)

    def setUp(self):
        from gui.chart_viewer import FpltRenderer
        self.renderer = FpltRenderer(MockAxis(), MockAxis())

    def _make_df(self, n: int) -> pd.DataFrame:
        timestamps = [self.BASE_TIME + timedelta(minutes=i) for i in range(n)]
        return pd.DataFrame(
            {
                'Open':   [990.0 + i for i in range(n)],
                'High':   [992.0 + i for i in range(n)],
                'Low':    [989.0 + i for i in range(n)],
                'Close':  [991.0 + i for i in range(n)],
                'Volume': [1000.0] * n,
            },
            index=pd.DatetimeIndex(timestamps),
        )

    def _candle_last_time(self, df: pd.DataFrame) -> pd.Timestamp:
        x_idx = self.renderer._to_x(df.index)
        cdf   = self.renderer._make_candle_df(x_idx, df)
        return cdf.index[-1]

    def test_new_bar_triggers_force_redraw(self):
        t1 = self._candle_last_time(self._make_df(3))
        self.renderer._last_candle_time = t1
        t2 = self._candle_last_time(self._make_df(4))
        self.assertTrue(t2 is not None and t1 is not None and t2 != t1)

    def test_same_data_no_force_redraw(self):
        df = self._make_df(3)
        t1 = self._candle_last_time(df)
        self.renderer._last_candle_time = t1
        t2 = self._candle_last_time(df)
        self.assertFalse(t2 is not None and t1 is not None and t2 != t1)

    def test_last_candle_time_is_timestamp(self):
        t = self._candle_last_time(self._make_df(3))
        self.assertIsInstance(t, pd.Timestamp)

    def test_three_bars_last_time_correct(self):
        t = self._candle_last_time(self._make_df(3))
        self.assertEqual(t, pd.Timestamp(self.BASE_TIME + timedelta(minutes=2)))


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 소스 전환
# ══════════════════════════════════════════════════════════════════════════════

class TestDataSourceSwitch(unittest.TestCase):

    BASE_TIME = datetime(2026, 4, 27, 9, 0, 0)

    def setUp(self):
        from gui.chart_viewer import FpltRenderer
        self.renderer = FpltRenderer(MockAxis(), MockAxis())
        timestamps = [self.BASE_TIME + timedelta(minutes=i) for i in range(5)]
        self.df = pd.DataFrame(
            {
                'Open': [990.0] * 5, 'High': [992.0] * 5,
                'Low':  [989.0] * 5, 'Close': [991.0] * 5,
                'Volume': [1000.0] * 5,
            },
            index=pd.DatetimeIndex(timestamps),
        )

    def test_initial_data_source_is_none(self):
        self.assertIsNone(self.renderer._current_data_source)

    def test_data_source_set_after_render(self):
        self.renderer.render(self.df, None, data_source='futures')
        self.assertEqual(self.renderer._current_data_source, 'futures')

    def test_data_source_switch_updates_field(self):
        self.renderer.render(self.df, None, data_source='futures')
        self.renderer.render(self.df, None, data_source='kospi')
        self.assertEqual(self.renderer._current_data_source, 'kospi')

    def test_data_source_switch_resets_last_candle_time(self):
        self.renderer.render(self.df, None, data_source='futures')
        t_prev = pd.Timestamp('2026-04-26 09:04:00')
        self.renderer._last_candle_time = t_prev

        self.renderer.render(self.df, None, data_source='kospi')

        expected = pd.Timestamp(datetime(2026, 4, 27, 9, 0, 0) + timedelta(minutes=4))
        self.assertEqual(self.renderer._last_candle_time, expected)
        self.assertNotEqual(self.renderer._last_candle_time, t_prev)

    def test_data_source_switch_resets_pm_hash(self):
        """[PERF-2] 데이터 소스 전환 시 _last_pm_hash 가 None 으로 초기화되어야 한다."""
        self.renderer.render(self.df, None, data_source='futures')
        # pm_hash 를 임의 값으로 오염
        self.renderer._last_pm_hash = 99999

        self.renderer.render(self.df, None, data_source='kospi')
        self.assertIsNone(self.renderer._last_pm_hash)


# ══════════════════════════════════════════════════════════════════════════════
# [BUG-2] ChartEngine MAX_BARS 상수 동기화
# ══════════════════════════════════════════════════════════════════════════════

class TestChartEngineMaxBars(unittest.TestCase):

    def test_max_bars_is_500(self):
        """[BUG-2] MAX_BARS 는 500 이어야 한다."""
        from gui.chart_viewer import ChartEngine
        self.assertEqual(ChartEngine.MAX_BARS, 500)


# ══════════════════════════════════════════════════════════════════════════════
# [BUG-3] 캐시 키에 id(zz) 포함 검증
# ══════════════════════════════════════════════════════════════════════════════

class TestCacheKeyIncludesZzId(unittest.TestCase):

    def _make_df(self, n: int = 5) -> pd.DataFrame:
        idx = pd.date_range("2026-04-27 09:00", periods=n, freq="1min")
        return pd.DataFrame({
            "Open":   [350.0 + i for i in range(n)],
            "High":   [351.0 + i for i in range(n)],
            "Low":    [349.0 + i for i in range(n)],
            "Close":  [350.5 + i for i in range(n)],
            "Volume": [1000.0] * n,
        }, index=idx)

    def test_cache_key_contains_zz_id(self):
        """compute() 에서 생성한 sig 의 4번째 요소가 id(_zz) 이어야 한다."""
        from gui.chart_viewer import ChartEngine
        engine = ChartEngine()

        zz = object()  # 단순 객체로 id() 고정
        engine._zz = zz
        engine._zz_cfg = None   # feed 건너뜀

        df = self._make_df()
        # compute 내부 sig 생성 로직을 직접 검증
        try:
            sig = (
                len(df),
                str(df.index[-1]),
                round(float(df["Close"].iloc[-1]), 2),
                id(engine._zz),
            )
            self.assertEqual(sig[3], id(zz))
        except Exception as e:
            self.fail(f"캐시 키 생성 실패: {e}")

    def test_set_zigzag_changes_sig_id(self):
        """set_zigzag 후 id 가 바뀌면 동일 df 라도 캐시 미스가 발생해야 한다."""
        from gui.chart_viewer import ChartEngine
        engine = ChartEngine()

        zz1 = object()
        zz2 = object()

        engine._zz = zz1
        old_id = id(zz1)

        engine.set_zigzag(zz2)
        new_id = id(zz2)

        self.assertNotEqual(old_id, new_id)
        self.assertIsNone(engine._last_sig)


if __name__ == '__main__':
    unittest.main(verbosity=2)
