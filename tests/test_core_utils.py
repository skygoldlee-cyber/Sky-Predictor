"""
core/utils.py 단위 테스트
"""

import pytest
import json
import tempfile
from datetime import datetime, date, timedelta
import numpy as np
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.unit
class TestTypeConversion:
    """타입 변환 유틸리티 테스트."""

    def test_safe_float_valid_string(self):
        """유효한 문자열 float 변환."""
        from core.utils import safe_float
        assert safe_float("123.45") == 123.45
        assert safe_float("-50.5") == -50.5
    
    def test_safe_float_invalid_string(self):
        """잘못된 문자열 float 변환."""
        from core.utils import safe_float
        assert safe_float("invalid", default=0.0) == 0.0
        assert safe_float(None, default=-1.0) == -1.0
    
    def test_safe_float_already_float(self):
        """이미 float인 경우."""
        from core.utils import safe_float
        assert safe_float(123.45) == 123.45
    
    def test_safe_int_valid_string(self):
        """유효한 문자열 int 변환."""
        from core.utils import safe_int
        assert safe_int("123") == 123
        assert safe_int("-50") == -50
    
    def test_safe_int_float_string(self):
        """float 문자열 int 변환."""
        from core.utils import safe_int
        assert safe_int("12.7") == 12
    
    def test_safe_int_invalid_string(self):
        """잘못된 문자열 int 변환."""
        from core.utils import safe_int
        assert safe_int("invalid", default=0) == 0
    
    def test_normalize_ohlcv_columns(self):
        """OHLCV 컬럼 정규화 테스트."""
        from core.utils import normalize_ohlcv_columns
        import pandas as pd
        
        # 소문자 컬럼
        df = pd.DataFrame({
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [100, 200]
        })
        result = normalize_ohlcv_columns(df)
        assert "Open" in result.columns
        assert "High" in result.columns
    
    def test_normalize_ohlcv_columns_none(self):
        """None 입력 처리."""
        from core.utils import normalize_ohlcv_columns
        assert normalize_ohlcv_columns(None) is None
    
    def test_normalize_ohlcv_columns_empty(self):
        """빈 DataFrame 처리."""
        from core.utils import normalize_ohlcv_columns
        import pandas as pd
        df = pd.DataFrame()
        assert normalize_ohlcv_columns(df).empty


class TestCalcDirection:
    """방향 계산 테스트."""
    
    def test_calc_direction_up(self):
        """상승 방향."""
        from core.utils import calc_direction
        assert calc_direction(predicted=390, current=380) == "up"
    
    def test_calc_direction_down(self):
        """하락 방향."""
        from core.utils import calc_direction
        assert calc_direction(predicted=370, current=380) == "down"
    
    def test_calc_direction_neutral(self):
        """중립 방향."""
        from core.utils import calc_direction
        assert calc_direction(predicted=380, current=380) == "neutral"
    
    def test_calc_direction_threshold(self):
        """임계값 테스트."""
        from core.utils import calc_direction
        # 0.5% 이상 변동해야 up/down
        assert calc_direction(predicted=382, current=380, threshold_pct=0.5) == "up"  # 0.5% 초과
        assert calc_direction(predicted=380.2, current=380, threshold_pct=1.0) == "neutral"  # 0.05% 미만
    
    def test_calc_direction_zero_current(self):
        """현재가 0인 경우."""
        from core.utils import calc_direction
        assert calc_direction(predicted=100, current=0) == "up"
        assert calc_direction(predicted=-100, current=0) == "down"
        assert calc_direction(predicted=0, current=0) == "neutral"


class TestStatistics:
    """통계 함수 테스트."""
    
    def test_norm_cdf(self):
        """정규분포 CDF 테스트."""
        from core.utils import norm_cdf
        
        # 표준 정규분포의 특성
        assert abs(norm_cdf(0) - 0.5) < 0.01  # 중앙값
        assert norm_cdf(-3) < 0.01  # 하위 0.1%
        assert norm_cdf(3) > 0.99  # 상위 0.1%
    
    def test_norm_pdf(self):
        """정규분포 PDF 테스트."""
        from core.utils import norm_pdf
        
        # 표준 정규분포의 특성
        assert norm_pdf(0) > 0  # 중앙값에서 최대
        assert norm_pdf(0) > norm_pdf(1)  # 중앙값이 1보다 높음
        assert norm_pdf(0) > norm_pdf(-1)  # 중앙값이 -1보다 높음


class TestDateUtilities:
    """날짜/시간 유틸리티 테스트."""
    
    def test_get_second_thursday_date(self):
        """두 번째 목요일 계산."""
        from core.utils import get_second_thursday_date
        
        # 2025년 2월 두 번째 목요일은 13일
        result = get_second_thursday_date(2025, 2)
        assert result.day == 13
        assert result.month == 2
        assert result.year == 2025
    
    def test_set_expiry_holidays_valid(self):
        """유효한 휴장일 설정."""
        from core.utils import set_expiry_holidays
        
        count = set_expiry_holidays(["2025-01-01", "2025-02-01"])
        assert count == 2
    
    def test_set_expiry_holidays_invalid(self):
        """잘못된 형식 처리."""
        from core.utils import set_expiry_holidays
        
        count = set_expiry_holidays(["invalid", "2025-01-01"])
        assert count == 1  # 유효한 것만 반영
    
    def test_set_expiry_holidays_yyyymmdd(self):
        """YYYYMMDD 형식 처리."""
        from core.utils import set_expiry_holidays
        
        count = set_expiry_holidays(["20250101", "20250201"])
        assert count == 2
    
    def test_get_previous_business_day_weekday(self):
        """평일 이전 영업일."""
        from core.utils import get_previous_business_day
        
        # 수요일 → 화요일
        wednesday = date(2025, 1, 8)  # 수요일
        result = get_previous_business_day(wednesday, days_back=1)
        assert result == date(2025, 1, 7)  # 화요일
    
    def test_get_previous_business_day_weekend(self):
        """주말 이전 영업일."""
        from core.utils import get_previous_business_day
        
        # 일요일 → 금요일
        sunday = date(2025, 1, 12)  # 일요일
        result = get_previous_business_day(sunday, days_back=1)
        assert result == date(2025, 1, 10)  # 금요일
    
    def test_get_previous_business_day_with_holiday(self):
        """휴장일 포함 이전 영업일."""
        from core.utils import set_expiry_holidays, get_previous_business_day
        
        # 휴장일 설정
        set_expiry_holidays(["2025-01-09"])  # 목요일 휴장
        
        # 금요일 → 수요일 (목요일 휴장)
        friday = date(2025, 1, 10)  # 금요일
        result = get_previous_business_day(friday, days_back=1)
        assert result == date(2025, 1, 8)  # 수요일
    
    def test_get_option_expiry_date(self):
        """옵션 만기일 계산."""
        from core.utils import get_option_expiry_date
        
        result = get_option_expiry_date(2025, 2)
        assert result.month == 2
        assert result.year == 2025
    
    def test_get_expiry_week_info(self):
        """만기주 정보."""
        from core.utils import get_expiry_week_info
        
        # 만기주에 있는 날
        expiry_date = datetime(2025, 2, 13)  # 두 번째 목요일
        result = get_expiry_week_info(expiry_date)
        assert "is_expiry_week" in result
        assert "expiry_second_thursday" in result
        assert "days_to_expiry" in result
    
    def test_get_option_month_yyyymm(self):
        """옵션 만기월."""
        from core.utils import get_option_month_yyyymm
        
        # 만기 전
        result = get_option_month_yyyymm(datetime(2025, 2, 10))
        assert result == "202502"
        
        # 만기 후
        result = get_option_month_yyyymm(datetime(2025, 2, 14))
        assert result == "202503"
    
    def test_parse_chetime_valid(self):
        """유효한 체결시간 파싱."""
        from core.utils import parse_chetime
        
        ref = datetime(2025, 1, 1, 12, 0, 0)
        result = parse_chetime("130430", reference=ref)
        assert result.hour == 13
        assert result.minute == 4
        assert result.second == 30
    
    def test_parse_chetime_none(self):
        """None 체결시간 처리."""
        from core.utils import parse_chetime
        
        ref = datetime(2025, 1, 1, 12, 0, 0)
        result = parse_chetime(None, reference=ref)
        assert result == ref.replace(microsecond=0)
    
    def test_parse_chetime_invalid(self):
        """잘못된 체결시간 처리."""
        from core.utils import parse_chetime
        
        ref = datetime(2025, 1, 1, 12, 0, 0)
        result = parse_chetime("invalid", reference=ref)
        assert result == ref.replace(microsecond=0)


class TestJsonUtilities:
    """JSON 유틸리티 테스트."""
    
    def test_make_json_safe_dict(self):
        """딕셔너리 JSON 안전 변환."""
        from core.utils import make_json_safe
        
        result = make_json_safe({"key": "value"})
        assert result == {"key": "value"}
    
    def test_make_json_safe_datetime(self):
        """datetime JSON 안전 변환."""
        from core.utils import make_json_safe
        
        dt = datetime(2025, 1, 1, 12, 0, 0)
        result = make_json_safe(dt)
        assert isinstance(result, str)
        assert "2025" in result
    
    def test_make_json_safe_numpy(self):
        """numpy 타입 JSON 안전 변환."""
        from core.utils import make_json_safe
        
        result = make_json_safe(np.int64(123))
        assert result == 123
        
        result = make_json_safe(np.float64(123.45))
        assert result == 123.45
    
    def test_write_jsonl_line(self):
        """JSONL 라인 쓰기."""
        from core.utils import write_jsonl_line
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            write_jsonl_line(f, {"key": "value"})
            f.flush()
            
            # 읽어서 확인
            with open(f.name, 'r') as rf:
                line = rf.readline()
                data = json.loads(line)
                assert data == {"key": "value"}


class TestOptionUtilities:
    """옵션 관련 유틸리티 테스트."""
    
    def test_validate_strike_price_valid(self):
        """유효한 행사가."""
        from core.utils import validate_strike_price
        
        assert validate_strike_price(380.0) is True
        assert validate_strike_price(250.0) is True
        assert validate_strike_price(450.0) is True
    
    def test_validate_strike_price_invalid(self):
        """잘못된 행사가."""
        from core.utils import validate_strike_price
        
        assert validate_strike_price(150.0) is False  # 최소 미만
        assert validate_strike_price(600.0) is False  # 최대 초과
    
    def test_validate_strike_price_custom_range(self):
        """커스텀 범위."""
        from core.utils import validate_strike_price
        
        assert validate_strike_price(300.0, min_strike=250.0, max_strike=350.0) is True
        assert validate_strike_price(200.0, min_strike=250.0, max_strike=350.0) is False
    
    def test_parse_strike_from_code_valid(self):
        """유효한 행사가 코드 파싱."""
        from core.utils import parse_strike_from_code
        
        assert parse_strike_from_code("385") == 385.0
        assert parse_strike_from_code("430") == 430.0
    
    def test_parse_strike_from_code_invalid(self):
        """잘못된 행사가 코드 파싱."""
        from core.utils import parse_strike_from_code
        
        assert parse_strike_from_code("invalid") is None
        assert parse_strike_from_code("") is None
    
    def test_parse_strike_from_code_alpha(self):
        """알파벳 행사가 코드."""
        from core.utils import parse_strike_from_code
        
        # A01 = 1000.0
        result = parse_strike_from_code("A01")
        assert result == 1000.0


class TestSeed:
    """시드 설정 테스트."""
    
    def test_set_seed(self):
        """시드 설정."""
        from core.utils import set_seed
        
        # 에러 없이 실행되어야 함
        set_seed(42)
        set_seed(123)
    
    def test_set_seed_default(self):
        """기본 시드."""
        from core.utils import set_seed
        
        # 에러 없이 실행되어야 함
        set_seed()  # 기본값 42
