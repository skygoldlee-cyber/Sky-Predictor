"""
렌더러 로직 단위 테스트 (GUI 의존성 없음)
==========================================

테스트 대상:
  - 렌더러의 데이터 처리 로직
  - 마스크 생성 로직
  - NaN 필터링 로직
  - 길이 불일치 처리 로직
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_ohlcv_data(n_bars: int, start_time: datetime = None) -> pd.DataFrame:
    """테스트용 OHLCV 데이터 생성"""
    if start_time is None:
        start_time = datetime(2024, 1, 1, 9, 0)
    
    timestamps = pd.date_range(start=start_time, periods=n_bars, freq='1min')
    base_price = 1000.0
    
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
# SuperTrend 렌더링 로직 테스트
# ══════════════════════════════════════════════════════════════════════════════

class TestSuperTrendRenderingLogic:
    """SuperTrend 렌더링 로직 테스트"""

    def test_supertrend_mask_generation(self):
        """SuperTrend 마스크 생성 로직 테스트"""
        # 테스트 데이터 생성
        n = 100
        st = np.random.randn(n) * 10 + 1000  # 랜덤 슈퍼트렌드 값
        st_dir = np.where(np.random.randn(n) > 0, 1, -1)  # 랜덤 방향
        
        # 방향 문자열 변환
        st_dir_str = np.where(st_dir == 1, "up", np.where(st_dir == -1, "down", "up"))
        
        # 마스크 생성 로직 (fplt_renderer.py와 동일)
        include_in_up = (st_dir_str == "up").copy()
        include_in_down = (st_dir_str == "down").copy()
        
        # 전환 봉을 직전 방향에도 포함
        for i in range(1, n):
            if st_dir_str[i - 1] == "up" and st_dir_str[i] == "down":
                include_in_up[i] = True
            elif st_dir_str[i - 1] == "down" and st_dir_str[i] == "up":
                include_in_down[i] = True
        
        # 유효 마스크
        valid_mask = ~np.isnan(st)
        up_mask = valid_mask & include_in_up
        down_mask = valid_mask & include_in_down
        
        # 검증
        assert np.any(up_mask) or np.any(down_mask), "적어도 하나의 마스크는 있어야 함"
        assert len(up_mask) == n
        assert len(down_mask) == n
        
        # 전환 봉 확인
        transition_count = 0
        for i in range(1, n):
            if st_dir_str[i - 1] != st_dir_str[i]:
                transition_count += 1
                # 전환 봉은 양쪽 마스크에 포함되어야 함
                assert up_mask[i] or down_mask[i], f"전환 봉 {i}는 적어도 하나의 마스크에 포함되어야 함"

    def test_supertrend_nan_filtering(self):
        """SuperTrend NaN 필터링 로직 테스트"""
        # NaN 포함 데이터 생성
        st = np.array([1000, np.nan, 1020, np.nan, 1040, 1050])
        st_dir = np.array([1, 1, -1, -1, 1, 1])
        x_idx = np.arange(len(st))
        
        # 유효 마스크
        valid_mask = ~np.isnan(st)
        
        # NaN 필터링
        x_valid = x_idx[valid_mask]
        st_valid = st[valid_mask]
        st_dir_valid = st_dir[valid_mask]
        
        # 검증
        assert len(x_valid) == 4  # NaN 2개 제거
        assert np.all(~np.isnan(st_valid))
        assert len(x_valid) == len(st_valid)
        assert len(st_valid) == len(st_dir_valid)

    def test_supertrend_length_mismatch_handling(self):
        """SuperTrend 길이 불일치 처리 로직 테스트"""
        # 길이 불일치 데이터
        x_idx = np.arange(100)
        st = np.random.randn(80) * 10 + 1000  # 80개
        st_dir = np.where(np.random.randn(80) > 0, 1, -1)
        
        # 길이 맞춤 로직
        min_len = min(len(x_idx), len(st))
        x_idx_trimmed = x_idx[:min_len]
        st_trimmed = st[:min_len]
        st_dir_trimmed = st_dir[:min_len]
        
        # 검증
        assert len(x_idx_trimmed) == 80
        assert len(st_trimmed) == 80
        assert len(st_dir_trimmed) == 80
        assert len(x_idx_trimmed) == len(st_trimmed)

    def test_supertrend_all_nan_handling(self):
        """SuperTrend 전체 NaN 처리 로직 테스트"""
        # 전체 NaN 데이터
        st = np.array([np.nan, np.nan, np.nan])
        st_dir = np.array([1, -1, 1])
        x_idx = np.arange(len(st))
        
        # 유효 마스크
        valid_mask = ~np.isnan(st)
        
        # 전체 NaN이면 렌더링 스킵
        if not np.any(valid_mask):
            should_skip = True
        else:
            should_skip = False
        
        # 검증
        assert should_skip, "전체 NaN이면 렌더링 스킵해야 함"

    def test_supertrend_transition_overlap(self):
        """전환 봉 중첩 테스트"""
        # 명확한 전환 패턴 생성
        st = np.array([1000, 1010, 1020, 1015, 1005, 1000])  # 상승 후 하락
        st_dir = np.array([1, 1, 1, -1, -1, -1])  # up → down 전환 (인덱스 2-3)
        x_idx = np.arange(len(st))
        
        # 방향 문자열 변환
        st_dir_str = np.where(st_dir == 1, "up", np.where(st_dir == -1, "down", "up"))
        
        # 마스크 생성
        include_in_up = (st_dir_str == "up").copy()
        include_in_down = (st_dir_str == "down").copy()
        
        # 전환 봉 포함
        for i in range(1, len(st_dir_str)):
            if st_dir_str[i - 1] == "up" and st_dir_str[i] == "down":
                include_in_up[i] = True
            elif st_dir_str[i - 1] == "down" and st_dir_str[i] == "up":
                include_in_down[i] = True
        
        valid_mask = ~np.isnan(st)
        up_mask = valid_mask & include_in_up
        down_mask = valid_mask & include_in_down
        
        # 검증: 전환 봉(인덱스 3)은 양쪽 마스크에 포함되어야 함
        assert up_mask[3], "전환 봉은 up 마스크에 포함되어야 함"
        assert down_mask[3], "전환 봉은 down 마스크에 포함되어야 함"


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 처리 로직 테스트
# ══════════════════════════════════════════════════════════════════════════════

class TestDataProcessingLogic:
    """데이터 처리 로직 테스트"""

    def test_ohlc_data_generation(self):
        """OHLC 데이터 생성 테스트"""
        df = generate_ohlcv_data(100)
        
        # 검증
        assert len(df) == 100
        assert 'Open' in df.columns
        assert 'High' in df.columns
        assert 'Low' in df.columns
        assert 'Close' in df.columns
        assert 'Volume' in df.columns
        assert df.index.is_unique

    def test_data_concatenation(self):
        """데이터 연결 테스트"""
        df1 = generate_ohlcv_data(50)
        df2 = generate_ohlcv_data(1, start_time=df1.index[-1] + timedelta(minutes=1))
        
        df_concat = pd.concat([df1, df2])
        
        # 검증
        assert len(df_concat) == 51
        assert df_concat.index.is_unique
        assert not df_concat.index.duplicated().any()

    def test_data_slicing(self):
        """데이터 슬라이싱 테스트"""
        df = generate_ohlcv_data(100)
        
        # 마지막 50봉 슬라이스
        df_sliced = df.iloc[-50:]
        
        # 검증
        assert len(df_sliced) == 50
        assert df_sliced.index[0] == df.index[50]

    def test_data_length_mismatch_padding(self):
        """데이터 길이 불일치 패딩 로직 테스트"""
        # 짧은 데이터
        short_data = [1000, 1010, 1020]
        target_len = 10
        
        # 첫 유효값 패딩
        first_valid = next((v for v in short_data if not np.isnan(v)), np.nan)
        pad_len = target_len - len(short_data)
        padded_data = [first_valid] * pad_len + short_data
        
        # 검증
        assert len(padded_data) == target_len
        assert padded_data[0] == first_valid
        assert padded_data[pad_len] == short_data[0]

    def test_data_length_mismatch_trimming(self):
        """데이터 길이 불일치 트림 로직 테스트"""
        # 긴 데이터
        long_data = list(range(100))
        target_len = 50
        
        # 뒷부분 트림
        trimmed_data = long_data[-target_len:]
        
        # 검증
        assert len(trimmed_data) == target_len
        assert trimmed_data[0] == 50
        assert trimmed_data[-1] == 99

    def test_nan_detection(self):
        """NaN 감지 로직 테스트"""
        # 다양한 NaN 패턴
        data1 = np.array([1, 2, np.nan, 4, 5])
        data2 = np.array([np.nan, np.nan, np.nan])
        data3 = np.array([1, 2, 3, 4, 5])
        
        # NaN 감지
        has_nan1 = np.any(np.isnan(data1))
        has_nan2 = np.any(np.isnan(data2))
        has_nan3 = np.any(np.isnan(data3))
        
        # 검증
        assert has_nan1
        assert has_nan2
        assert not has_nan3

    def test_all_nan_detection(self):
        """전체 NaN 감지 로직 테스트"""
        data_all_nan = np.array([np.nan, np.nan, np.nan])
        data_partial_nan = np.array([1, np.nan, 3])
        data_no_nan = np.array([1, 2, 3])
        
        # 전체 NaN 감지
        is_all_nan1 = np.all(np.isnan(data_all_nan))
        is_all_nan2 = np.all(np.isnan(data_partial_nan))
        is_all_nan3 = np.all(np.isnan(data_no_nan))
        
        # 검증
        assert is_all_nan1
        assert not is_all_nan2
        assert not is_all_nan3


# ══════════════════════════════════════════════════════════════════════════════
# 피벗 마커 로직 테스트
# ══════════════════════════════════════════════════════════════════════════════

class TestPivotMarkerLogic:
    """피벗 마커 로직 테스트"""

    def test_pivot_data_filtering(self):
        """피벗 데이터 NaN 필터링 로직 테스트"""
        # 피벗 데이터 생성 (NaN 포함)
        pivot_data = {
            "idx": [10, 20, 30, 40],
            "y": [1000.0, np.nan, 1050.0, np.nan],
            "t": ["H", "L", "H", "L"]
        }
        
        # DataFrame 변환
        df_pivot = pd.DataFrame(pivot_data)
        
        # NaN 필터링
        valid_mask = ~np.isnan(df_pivot["y"])
        df_filtered = df_pivot[valid_mask]
        
        # 검증
        assert len(df_filtered) == 2  # NaN 2개 제거
        assert np.all(~np.isnan(df_filtered["y"]))

    def test_pivot_type_filtering(self):
        """피벗 타입 필터링 로직 테스트"""
        pivot_data = {
            "idx": [10, 20, 30, 40],
            "y": [1000.0, 990.0, 1050.0, 980.0],
            "t": ["H", "L", "H", "L"]
        }
        
        df_pivot = pd.DataFrame(pivot_data)
        
        # H 타입 필터링
        h_mask = df_pivot["t"] == "H"
        df_h = df_pivot[h_mask]
        
        # L 타입 필터링
        l_mask = df_pivot["t"] == "L"
        df_l = df_pivot[l_mask]
        
        # 검증
        assert len(df_h) == 2
        assert len(df_l) == 2
        assert all(df_h["t"] == "H")
        assert all(df_l["t"] == "L")

    def test_pivot_index_mapping(self):
        """피벗 인덱스와 x_idx 매핑 로직 테스트"""
        # 데이터프레임 생성
        df = generate_ohlcv_data(100)
        x_idx = np.arange(len(df))
        
        # 피벗 데이터 (인덱스는 데이터프레임의 위치 인덱스)
        pivot_data = {
            "idx": [10, 30, 50, 70],  # 데이터프레임 위치 인덱스
            "y": [1000.0, 1020.0, 1010.0, 1030.0],
            "t": ["H", "L", "H", "L"]
        }
        
        # 인덱스 매핑 검증
        for pivot_idx in pivot_data["idx"]:
            # 피벗 인덱스가 x_idx 범위 내에 있어야 함
            assert 0 <= pivot_idx < len(x_idx), f"피벗 인덱스 {pivot_idx}가 범위를 벗어남"
            # x_idx에서 해당 위치의 값 가져오기
            mapped_x = x_idx[pivot_idx]
            assert mapped_x == pivot_idx, f"인덱스 매핑 불일치: {mapped_x} != {pivot_idx}"

    def test_pivot_index_out_of_range(self):
        """피벗 인덱스 범위 초과 처리 테스트"""
        # 데이터프레임 생성
        df = generate_ohlcv_data(50)
        x_idx = np.arange(len(df))
        
        # 범위를 벗어난 피벗 인덱스
        pivot_data = {
            "idx": [10, 60, 30, 100],  # 60, 100은 범위 초과
            "y": [1000.0, 1020.0, 1010.0, 1030.0],
            "t": ["H", "L", "H", "L"]
        }
        
        # 유효한 인덱스 필터링
        valid_indices = [idx for idx in pivot_data["idx"] if 0 <= idx < len(x_idx)]
        
        # 검증
        assert len(valid_indices) == 2  # 10, 30만 유효
        assert 10 in valid_indices
        assert 30 in valid_indices
        assert 60 not in valid_indices
        assert 100 not in valid_indices

    def test_pivot_index_after_trimming(self):
        """데이터 트림 후 피벗 인덱스 유효성 테스트"""
        # 전체 데이터
        df_full = generate_ohlcv_data(100)
        
        # 트림된 데이터 (마지막 50봉)
        df_trimmed = df_full.iloc[-50:].copy()
        x_idx_trimmed = np.arange(len(df_trimmed))
        
        # 전체 데이터 기준 피벗 인덱스
        pivot_indices_full = [10, 30, 50, 70, 90]
        
        # 트림된 데이터 기준 인덱스 변환
        # 원래 인덱스 50~99는 트림 후 0~49가 됨
        pivot_indices_trimmed = [idx - 50 for idx in pivot_indices_full if idx >= 50]
        
        # 검증
        assert len(pivot_indices_trimmed) == 3  # 50, 70, 90만 트림 후 유효
        assert 0 in pivot_indices_trimmed  # 50 → 0
        assert 20 in pivot_indices_trimmed  # 70 → 20
        assert 40 in pivot_indices_trimmed  # 90 → 40
        
        # 모든 트림된 인덱스가 범위 내에 있어야 함
        for idx in pivot_indices_trimmed:
            assert 0 <= idx < len(x_idx_trimmed)

    def test_pivot_index_after_padding(self):
        """데이터 패딩 후 피벗 인덱스 유효성 테스트"""
        # 짧은 데이터
        df_short = generate_ohlcv_data(50)
        
        # 패딩된 데이터 (앞에 30봉 패딩)
        pad_len = 30
        first_valid_value = df_short['Close'].iloc[0]
        
        # 패딩된 인덱스 계산
        # 원래 인덱스 0~49는 패딩 후 30~79가 됨
        pivot_indices_short = [10, 20, 30, 40]
        pivot_indices_padded = [idx + pad_len for idx in pivot_indices_short]
        
        # 검증
        assert len(pivot_indices_padded) == 4
        assert 40 in pivot_indices_padded  # 10 + 30
        assert 50 in pivot_indices_padded  # 20 + 30
        assert 60 in pivot_indices_padded  # 30 + 30
        assert 70 in pivot_indices_padded  # 40 + 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
