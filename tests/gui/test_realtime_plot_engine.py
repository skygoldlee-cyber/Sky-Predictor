"""
차트 엔진 실시간 플롯 기능 단위 테스트
========================================

테스트 대상:
  - 차트 엔진 실시간 계산
  - 슈퍼트렌드 실시간 업데이트
  - 캐싱 메커니즘
  - 데이터 소스 변경 처리
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


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

    def test_intermittent_break_simulation(self):
        """간헐적 깨짐 시뮬레이션: 범위 변경 + 데이터 길이 급변"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")

        engine = ChartEngine()
        engine._init_supertrend()

        # 1. 초기 로드 (100봉)
        df = generate_ohlcv_data(100)
        df1, pm1 = engine.compute(df, force_recompute=True)
        assert df1 is not None
        assert len(df1) == 100

        # 2. 실시간 업데이트 (10봉 추가)
        for i in range(10):
            new_bar = generate_ohlcv_data(1, start_time=df.index[-1] + timedelta(minutes=i+1))
            df = pd.concat([df, new_bar])
            df_updated, pm_updated = engine.compute(df, force_recompute=False)
            assert df_updated is not None

        # 3. 급격한 길이 감소 시뮬레이션 (110봉 → 50봉)
        df_short = df.iloc[-50:].copy()
        df_short_result, pm_short = engine.compute(df_short, force_recompute=False)
        assert df_short_result is not None
        assert len(df_short_result) == 50

        # 4. 다시 급격한 길이 증가 시뮬레이션 (50봉 → 150봉)
        df_long = generate_ohlcv_data(150, start_time=df_short.index[0])
        df_long_result, pm_long = engine.compute(df_long, force_recompute=False)
        assert df_long_result is not None
        assert len(df_long_result) == 150

        # 5. 슈퍼트렌드 데이터 무결성 확인
        if "SuperTrend" in df_long_result.columns:
            st_values = df_long_result["SuperTrend"].values
            valid_count = np.sum(~np.isnan(st_values))
            assert valid_count > 0, "슈퍼트렌드 유효값이 있어야 함"

    def test_rapid_source_switching(self):
        """급격한 데이터 소스 전환 시뮬레이션"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")

        engine = ChartEngine()
        engine._init_supertrend()

        # 데이터 소스 빈번 전환
        sources = ["source1", "source2", "source3", "source1"]
        results = []

        for i, source in enumerate(sources):
            df = generate_ohlcv_data(100, start_time=datetime(2024, 1, 1) + timedelta(days=i))
            engine.set_zigzag(None, data_source=source)
            df_result, pm = engine.compute(df, force_recompute=True)
            assert df_result is not None
            results.append(df_result)

            # 슈퍼트렌드 컬럼 확인
            if "SuperTrend" in df_result.columns:
                st_values = df_result["SuperTrend"].values
                valid_count = np.sum(~np.isnan(st_values))
                assert valid_count > 0, f"소스 {source}에서 슈퍼트렌드 유효값 있어야 함"

    def test_concurrent_update_simulation(self):
        """연속 업데이트 중 캐시 무효화 시뮬레이션"""
        try:
            from gui.engines.chart_engine import ChartEngine
        except ImportError as e:
            pytest.skip(f"ChartEngine import 실패: {e}")

        engine = ChartEngine()
        engine._init_supertrend()

        # 초기 데이터
        df = generate_ohlcv_data(100)
        df1, pm1 = engine.compute(df, force_recompute=True)

        # 연속 업데이트 중간에 force_recompute 호출
        for i in range(15):
            new_bar = generate_ohlcv_data(1, start_time=df.index[-1] + timedelta(minutes=i+1))
            df = pd.concat([df, new_bar])

            # 5번째마다 강제 재계산
            if i % 5 == 0:
                df_result, pm = engine.compute(df, force_recompute=True)
            else:
                df_result, pm = engine.compute(df, force_recompute=False)

            assert df_result is not None
            assert len(df_result) == 100 + i + 1

            # 슈퍼트렌드 확인
            if "SuperTrend" in df_result.columns:
                st_values = df_result["SuperTrend"].values
                valid_count = np.sum(~np.isnan(st_values))
                assert valid_count > 0, f"업데이트 {i}에서 슈퍼트렌드 유효값 있어야 함"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
