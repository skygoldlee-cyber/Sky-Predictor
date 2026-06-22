"""
실시간 플롯 기능 단위 테스트
============================

테스트 대상:
  - 데이터 컴퓨팅 스레드
  - 실시간 업데이트 콜백
  - 차트 엔진 실시간 계산
  - 렌더러 실시간 렌더링
  - 슈퍼트렌드 실시간 업데이트
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import time

try:
    from PySide6.QtCore import QObject, Signal
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# 테스트 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

def generate_ohlcv_data(n_bars: int, start_time: datetime = None) -> pd.DataFrame:
    """테스트용 OHLCV 데이터 생성"""
    if start_time is None:
        start_time = datetime(2024, 1, 1, 9, 0)
    
    timestamps = pd.date_range(start=start_time, periods=n_bars, freq='1min')
    base_price = 1000.0
    
    # 랜덤 워크 데이터 생성
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, n_bars)
    prices = base_price * (1 + returns).cumprod()
    
    high = prices * (1 + np.random.uniform(0, 0.005, n_bars))
    low = prices * (1 - np.random.uniform(0, 0.005, n_bars))
    close = prices
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    
    volume = np.random.randint(1000, 10000, n_bars)
    
    df = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=timestamps)
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 차트 엔진 실시간 계산 테스트
# ══════════════════════════════════════════════════════════════════════════════

class TestChartEngineRealtime:
    """차트 엔진 실시간 계산 테스트"""

    def test_compute_caching(self):
        """캐싱 메커니즘 테스트"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")
        
        engine = ChartEngine()
        df = generate_ohlcv_data(100)
        
        # 첫 번째 계산
        df1, pm1 = engine.compute(df, force_recompute=True)
        assert df1 is not None
        
        # 두 번째 계산 (캐시 히트)
        df2, pm2 = engine.compute(df, force_recompute=False)
        assert df2 is not None
        # 동일한 데이터 반환 확인
        assert len(df1) == len(df2)

    def test_compute_with_new_bar(self):
        """새 봉 추가 시 계산 테스트"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")
        
        engine = ChartEngine()
        df_initial = generate_ohlcv_data(100)
        
        # 초기 계산
        df1, pm1 = engine.compute(df_initial, force_recompute=True)
        
        # 새 봉 추가
        new_bar = generate_ohlcv_data(1, start_time=df_initial.index[-1] + timedelta(minutes=1))
        df_updated = pd.concat([df_initial, new_bar])
        
        # 업데이트 계산
        df2, pm2 = engine.compute(df_updated, force_recompute=False)
        assert len(df2) == len(df1) + 1

    def test_supertrend_cache_mismatch_handling(self):
        """슈퍼트렌드 캐시 길이 불일치 처리 테스트"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")
        
        engine = ChartEngine()
        # 슈퍼트렌드 초기화
        engine._init_supertrend()
        
        df = generate_ohlcv_data(50)
        
        # 첫 계산
        df1, pm1 = engine.compute(df, force_recompute=True)
        
        # 데이터 소스 변경 시뮬레이션 (길이 변화)
        df_short = df.iloc[:30]
        df2, pm2 = engine.compute(df_short, force_recompute=False)
        
        # 캐시 초기화 후 재계산되어야 함
        assert df2 is not None

    def test_supertrend_padding_with_valid_value(self):
        """슈퍼트렌드 첫 유효값 패딩 테스트"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")
        
        engine = ChartEngine()
        engine._init_supertrend()
        
        df = generate_ohlcv_data(100)
        df1, pm1 = engine.compute(df, force_recompute=True)
        
        # 슈퍼트렌드 컬럼 확인
        if "SuperTrend" in df1.columns:
            # NaN이 아닌 첫 값 찾기
            st_values = df1["SuperTrend"].values
            valid_values = st_values[~np.isnan(st_values)]
            assert len(valid_values) > 0, "슈퍼트렌드 유효값이 있어야 함"


# ══════════════════════════════════════════════════════════════════════════════
# 렌더러 실시간 렌더링 테스트
# ══════════════════════════════════════════════════════════════════════════════

# Qt/finplot 필요한 테스트 클래스 skip 조건
try:
    from gui.renderers.fplt_renderer import FpltRenderer
    from gui.chart_viewer import DataComputeThread
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    FpltRenderer = None
    DataComputeThread = None


@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI 모듈 미설치")
class TestFpltRendererRealtime:
    """렌더러 실시간 렌더링 테스트"""

    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI 모듈 미설치")
    def test_render_candles_update_data(self):
        """캔들 update_data 테스트"""
        # Mock finplot
        mock_fplt = MagicMock()
        mock_ax = MagicMock()

        renderer = FpltRenderer(mock_fplt, mock_ax)
        df = generate_ohlcv_data(50)

        # x_idx 생성
        x_idx = np.arange(len(df))

        # 첫 렌더링
        renderer._render_candles(x_idx, df)

        # 데이터 업데이트
        df_updated = df.copy()
        df_updated['Close'] *= 1.01

        # 두 번째 렌더링 (update_data 사용)
        renderer._render_candles(x_idx, df_updated)

        # 플롯 존재 확인
        assert "_candle" in renderer._plots

    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI 모듈 미설치")
    def test_render_supertrend_no_overlap(self):
        """슈퍼트렌드 중첩 방지 테스트"""
        mock_fplt = MagicMock()
        mock_ax = MagicMock()

        renderer = FpltRenderer(mock_fplt, mock_ax)
        df = generate_ohlcv_data(100)

        # 슈퍼트렌드 데이터 추가
        df["SuperTrend"] = df["Close"] * 0.99
        df["SuperTrend_Dir"] = np.where(df["Close"].diff() > 0, 1, -1)

        x_idx = np.arange(len(df))

        # 렌더링
        renderer._render_supertrend(x_idx, df)

        # up/down 플롯 확인
        assert "_supertrend_up" in renderer._plots or "_supertrend_down" in renderer._plots

    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI 모듈 미설치")
    def test_upsert_nan_filtering(self):
        """_upsert NaN 필터링 테스트"""
        mock_fplt = MagicMock()
        mock_ax = MagicMock()

        renderer = FpltRenderer(mock_fplt, mock_ax)
        
        # 모두 NaN인 데이터
        x_all_nan = np.array([1, 2, 3])
        y_all_nan = np.array([np.nan, np.nan, np.nan])
        
        # 렌더링 시도 (스킵되어야 함)
        renderer._upsert("test_plot", x_all_nan, y_all_nan, mock_ax, 
                         color="red", style="-", width=1.0)
        
        # 플롯이 생성되지 않거나 숨겨져야 함
        if "test_plot" in renderer._plots:
            # 숨겨져 있어야 함
            assert renderer._plots["test_plot"].visible == False

    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI 모듈 미설치")
    def test_render_pivot_markers_nan_filtering(self):
        """피봇 마커 NaN 필터링 테스트"""
        mock_fplt = MagicMock()
        mock_ax = MagicMock()

        renderer = FpltRenderer(mock_fplt, mock_ax)
        
        # 피봇 데이터 생성 (NaN 포함)
        pm = {
            "confirmed": {
                "idx": [10, 20, 30],
                "y": [1000.0, np.nan, 1050.0],
                "t": ["H", "L", "H"]
            }
        }
        
        x_idx = np.arange(50)
        
        # 렌더링
        renderer._render_pivots(x_idx, pm)
        
        # 플롯 확인
        assert "_zz_conf_H" in renderer._plots or "_zz_conf_L" in renderer._plots


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 컴퓨팅 스레드 테스트
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI 모듈 미설치")
class TestDataComputeThread:
    """데이터 컴퓨팅 스레드 테스트"""

    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI 모듈 미설치")
    def test_thread_completion(self):
        """스레드 완료 테스트"""
        
        # Mock 콜백
        callback = Mock()
        
        # Mock 데이터
        df = generate_ohlcv_data(100)
        engine = MagicMock()
        engine.compute.return_value = (df, {})
        
        # 스레드 생성
        thread = DataComputeThread(df, engine, force_clear=False, callback=callback)
        
        # 스레드 실행 (동기적으로)
        thread.run()
        
        # 콜백 호출 확인
        callback.assert_called_once()

    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI 모듈 미설치")
    def test_thread_error_handling(self):
        """스레드 에러 처리 테스트"""
        callback = Mock()
        df = generate_ohlcv_data(100)

        # 에러 발생 엔진
        engine = MagicMock()
        engine.compute.side_effect = Exception("테스트 에러")

        thread = DataComputeThread(df, engine, force_clear=False, callback=callback)
        
        # 스레드 실행
        thread.run()
        
        # 에러 시에도 콜백 호출 (에러 처리 확인)
        callback.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# 실시간 업데이트 시뮬레이션 테스트
# ══════════════════════════════════════════════════════════════════════════════

class TestRealtimeUpdateSimulation:
    """실시간 업데이트 시뮬레이션 테스트"""

    def test_incremental_data_update(self):
        """증분 데이터 업데이트 시뮬레이션"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")
        
        engine = ChartEngine()
        
        # 초기 데이터
        df_initial = generate_ohlcv_data(50)
        df1, pm1 = engine.compute(df_initial, force_recompute=True)
        initial_len = len(df1)
        
        # 10개 봉 순차적 추가
        for i in range(10):
            new_bar = generate_ohlcv_data(
                1, 
                start_time=df_initial.index[-1] + timedelta(minutes=i+1)
            )
            df_initial = pd.concat([df_initial, new_bar])
            df_updated, pm_updated = engine.compute(df_initial, force_recompute=False)
            
            # 길이 증가 확인
            assert len(df_updated) == initial_len + i + 1

    def test_same_bar_tick_update(self):
        """동일 봉 틱 업데이트 시뮬레이션"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")
        
        engine = ChartEngine()
        
        df = generate_ohlcv_data(50)
        df1, pm1 = engine.compute(df, force_recompute=True)
        
        # 마지막 봉의 종가만 변경 (틱 업데이트)
        df_tick = df.copy()
        df_tick.iloc[-1, df_tick.columns.get_loc('Close')] *= 1.001
        
        df2, pm2 = engine.compute(df_tick, force_recompute=False)
        
        # 길이는 동일해야 함
        assert len(df2) == len(df1)

    def test_supertrend_realtime_update(self):
        """슈퍼트렌드 실시간 업데이트 테스트"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")
        
        engine = ChartEngine()
        engine._init_supertrend()
        
        df = generate_ohlcv_data(100)
        df1, pm1 = engine.compute(df, force_recompute=True)
        
        # 슈퍼트렌드 컬럼 확인
        if "SuperTrend" in df1.columns:
            st_initial = df1["SuperTrend"].iloc[-1]
            
            # 새 봉 추가
            new_bar = generate_ohlcv_data(1, start_time=df.index[-1] + timedelta(minutes=1))
            df_updated = pd.concat([df, new_bar])
            df2, pm2 = engine.compute(df_updated, force_recompute=False)
            
            if "SuperTrend" in df2.columns:
                st_updated = df2["SuperTrend"].iloc[-1]
                # 값이 업데이트되어야 함
                assert not np.isnan(st_updated)


# ══════════════════════════════════════════════════════════════════════════════
# 렌더링 깜박임 테스트
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI 모듈 미설치")
class TestRenderingFlicker:
    """렌더링 깜박임 테스트"""

    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI 모듈 미설치")
    def test_data_source_switch_flicker(self):
        """데이터 소스 빈번 전환 시 깜박임 테스트"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")
        
        engine = ChartEngine()
        engine._init_supertrend()
        
        # 두 개의 다른 데이터 소스
        df_kospi = generate_ohlcv_data(381, start_time=datetime(2024, 1, 1))
        df_futures = generate_ohlcv_data(411, start_time=datetime(2024, 1, 1))
        
        # 렌더링 시간 측정
        render_times = []
        
        # 10번 데이터 소스 전환
        for i in range(10):
            start_time = time.time()
            
            # KOSPI
            engine.set_zigzag(None, data_source="kospi")
            df1, pm1 = engine.compute(df_kospi, force_recompute=True)
            
            # KP200
            engine.set_zigzag(None, data_source="futures")
            df2, pm2 = engine.compute(df_futures, force_recompute=True)
            
            elapsed = time.time() - start_time
            render_times.append(elapsed)
            
            # 각 전환은 1초 이내여야 함
            assert elapsed < 1.0, f"렌더링 시간이 너무 깁니다: {elapsed:.3f}초"
        
        # 평균 렌더링 시간 확인
        avg_time = sum(render_times) / len(render_times)
        print(f"\n데이터 소스 전환 평균 렌더링 시간: {avg_time:.3f}초")
        
        # 평균 0.5초 이내여야 깜박임이 없음
        assert avg_time < 0.5, f"평균 렌더링 시간이 너무 깁니다: {avg_time:.3f}초"

    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI 모듈 미설치")
    def test_rapid_data_update_flicker(self):
        """빈번한 데이터 업데이트 시 깜박임 테스트"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")
        
        engine = ChartEngine()
        engine._init_supertrend()
        
        # 초기 데이터
        df = generate_ohlcv_data(100)
        df1, pm1 = engine.compute(df, force_recompute=True)
        
        # 빠른 업데이트 시뮬레이션 (20개 봉)
        update_times = []
        
        for i in range(20):
            start_time = time.time()
            
            # 새 봉 추가
            new_bar = generate_ohlcv_data(1, start_time=df.index[-1] + timedelta(minutes=1))
            df = pd.concat([df, new_bar])
            df_updated, pm_updated = engine.compute(df, force_recompute=False)
            
            elapsed = time.time() - start_time
            update_times.append(elapsed)
            
            # 각 업데이트는 0.1초 이내여야 함
            assert elapsed < 0.1, f"업데이트 시간이 너무 깁니다: {elapsed:.3f}초"
        
        # 평균 업데이트 시간 확인
        avg_time = sum(update_times) / len(update_times)
        print(f"\n데이터 업데이트 평균 시간: {avg_time:.3f}초")
        
        # 평균 0.05초 이내여야 깜박임이 없음
        assert avg_time < 0.05, f"평균 업데이트 시간이 너무 깁니다: {avg_time:.3f}초"

    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI 모듈 미설치")
    def test_full_vs_partial_render_performance(self):
        """전체 렌더링 vs 부분 렌더링 성능 비교"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")
        
        engine = ChartEngine()
        engine._init_supertrend()
        
        df = generate_ohlcv_data(500)
        
        # 전체 렌더링 시간 측정
        full_render_times = []
        for i in range(5):
            start_time = time.time()
            df_result, pm = engine.compute(df, force_recompute=True)
            elapsed = time.time() - start_time
            full_render_times.append(elapsed)
        
        avg_full = sum(full_render_times) / len(full_render_times)
        
        # 부분 렌더링 시간 측정 (캐시 히트)
        partial_render_times = []
        for i in range(5):
            start_time = time.time()
            df_result, pm = engine.compute(df, force_recompute=False)
            elapsed = time.time() - start_time
            partial_render_times.append(elapsed)
        
        avg_partial = sum(partial_render_times) / len(partial_render_times)
        
        print(f"\n전체 렌더링 평균 시간: {avg_full:.3f}초")
        print(f"부분 렌더링 평균 시간: {avg_partial:.3f}초")
        print(f"성능 향상: {(avg_full / avg_partial):.2f}x")
        
        # 부분 렌더링이 전체 렌더링보다 빨라야 함
        assert avg_partial < avg_full, "부분 렌더링이 전체 렌더링보다 빨라야 함"
        
        # 최소 2배 이상 빨라야 깜박임 방지에 효과적
        assert avg_full / avg_partial >= 2.0, "부분 렌더링 성능 향상이 부족합니다"

    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI 모듈 미설치")
    def test_supertrend_recalculation_performance(self):
        """SuperTrend 전체 재계산 성능 테스트"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")
        
        engine = ChartEngine()
        engine._init_supertrend()
        
        # 큰 데이터셋
        df_large = generate_ohlcv_data(1000)
        
        # SuperTrend 전체 재계산 시간 측정
        recalc_times = []
        for i in range(5):
            start_time = time.time()
            df_result, pm = engine.compute(df_large, force_recompute=True)
            elapsed = time.time() - start_time
            recalc_times.append(elapsed)
            
            # 전체 재계산은 1초 이내여야 함
            assert elapsed < 1.0, f"전체 재계산 시간이 너무 깁니다: {elapsed:.3f}초"
        
        avg_recalc = sum(recalc_times) / len(recalc_times)
        print(f"\nSuperTrend 전체 재계산 평균 시간 (1000봉): {avg_recalc:.3f}초")
        
        # 평균 0.5초 이내여야 깜박임이 없음
        assert avg_recalc < 0.5, f"전체 재계산 평균 시간이 너무 깁니다: {avg_recalc:.3f}초"

    @pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI 모듈 미설치")
    def test_rendering_stability_under_load(self):
        """부하 하에서 렌더링 안정성 테스트"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")
        
        engine = ChartEngine()
        engine._init_supertrend()
        
        df = generate_ohlcv_data(200)
        
        # 연속 업데이트 시뮬레이션
        render_times = []
        
        for i in range(30):
            start_time = time.time()
            
            # 데이터 소스 전환 + 데이터 업데이트 혼합
            if i % 5 == 0:
                # 데이터 소스 전환
                engine.set_zigzag(None, data_source=f"source_{i % 3}")
                df_result, pm = engine.compute(df, force_recompute=True)
            else:
                # 일반 업데이트
                new_bar = generate_ohlcv_data(1, start_time=df.index[-1] + timedelta(minutes=1))
                df = pd.concat([df, new_bar])
                df_result, pm = engine.compute(df, force_recompute=False)
            
            elapsed = time.time() - start_time
            render_times.append(elapsed)
            
            # 각 렌더링은 1초 이내여야 함
            assert elapsed < 1.0, f"렌더링 시간이 너무 깁니다: {elapsed:.3f}초"
        
        # 표준편차 확인 (안정성)
        avg_time = sum(render_times) / len(render_times)
        std_time = (sum((t - avg_time) ** 2 for t in render_times) / len(render_times)) ** 0.5
        
        print(f"\n부하 하에서 렌더링 평균 시간: {avg_time:.3f}초")
        print(f"렌더링 시간 표준편차: {std_time:.3f}초")
        
        # 표준편차가 평균의 50% 이내여야 안정적
        assert std_time < avg_time * 0.5, f"렌더링 시간이 불안정합니다: std={std_time:.3f}s, avg={avg_time:.3f}s"


# ══════════════════════════════════════════════════════════════════════════════
# 통합 테스트
# ══════════════════════════════════════════════════════════════════════════════

class TestRealtimePlotIntegration:
    """실시간 플롯 통합 테스트"""

    def test_full_pipeline_simulation(self):
        """전체 파이프라인 시뮬레이션"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")
        
        engine = ChartEngine()
        engine._init_supertrend()
        
        # 초기 로드
        df = generate_ohlcv_data(100)
        df1, pm1 = engine.compute(df, force_recompute=True)
        
        # 실시간 업데이트 시뮬레이션 (20개 봉)
        for i in range(20):
            new_bar = generate_ohlcv_data(1, start_time=df.index[-1] + timedelta(minutes=i+1))
            df = pd.concat([df, new_bar])
            df_updated, pm_updated = engine.compute(df, force_recompute=False)
            
            # 데이터 무결성 확인
            assert df_updated is not None
            assert len(df_updated) == 100 + i + 1
            
            # 슈퍼트렌드 컬럼 존재 확인
            if "SuperTrend" in df_updated.columns:
                st_values = df_updated["SuperTrend"].values
                valid_count = np.sum(~np.isnan(st_values))
                # 유효값이 있어야 함
                assert valid_count > 0

    def test_cache_invalidation_on_source_change(self):
        """데이터 소스 변경 시 캐시 무효화 테스트"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")
        
        engine = ChartEngine()
        engine._init_supertrend()
        
        # 첫 번째 데이터 소스
        df1 = generate_ohlcv_data(100, start_time=datetime(2024, 1, 1))
        engine.set_zigzag(None, data_source="source1")
        df_result1, pm1 = engine.compute(df1, force_recompute=True)
        
        # 두 번째 데이터 소스
        df2 = generate_ohlcv_data(100, start_time=datetime(2024, 1, 2))
        engine.set_zigzag(None, data_source="source2")
        df_result2, pm2 = engine.compute(df2, force_recompute=True)
        
        # 캐시 무효화로 인해 재계산되어야 함
        assert df_result2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
