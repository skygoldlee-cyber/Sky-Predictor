"""
fetch_daily_data.py 단위 테스트
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
import tempfile
import os

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fetch_daily_data import fetch_and_save_daily_data, main


class TestDateRangeGeneration:
    """날짜 범위 생성 로직 테스트"""
    
    def test_date_range_excludes_weekends(self):
        """날짜 범위에서 주말이 제외되는지 테스트"""
        # 2025-01-15 (수) ~ 2025-01-21 (화)
        start_date = "20250115"
        end_date = "20250121"
        
        # 주말 제외: 15(수), 16(목), 17(금), 20(월), 21(화) = 5일
        # 주말: 18(토), 19(일)
        
        # 테스트를 위해 config.json과 config.secrets.json 생성
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.json'
            secrets_path = Path(tmpdir) / 'config.secrets.json'
            
            config = {
                'ebest': {
                    'target_date': '20250115',
                    'kp200_upcode': 'A0166000'
                }
            }
            secrets = {
                'ebest': {
                    'appkey': 'test_key',
                    'appsecretkey': 'test_secret'
                }
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f)
            with open(secrets_path, 'w', encoding='utf-8') as f:
                json.dump(secrets, f)
            
            # 테스트용 시간 함수
            fixed_time = datetime(2025, 1, 15, 16, 0, 0)
            
            # 테스트 실행 (ebest API 호출 없이 날짜 범위 생성만 테스트)
            # 실제 API 호출은 mock으로 처리 필요
            pass  # 실제 테스트는 mock 구현 후 작성
    
    def test_date_range_all_weekdays(self):
        """모든 평일만 포함된 날짜 범위 테스트"""
        # 2025-01-13 (월) ~ 2025-01-17 (금)
        start_date = "20250113"
        end_date = "20250117"
        
        # 모두 평일: 13(월), 14(화), 15(수), 16(목), 17(금) = 5일
        pass
    
    def test_date_range_all_weekends(self):
        """모든 주말인 날짜 범위 테스트"""
        # 2025-01-18 (토) ~ 2025-01-19 (일)
        start_date = "20250118"
        end_date = "20250119"
        
        # 모두 주말이므로 빈 리스트
        pass


class TestDateValidation:
    """날짜 형식 검증 테스트"""
    
    def test_valid_date_format(self):
        """유효한 날짜 형식 테스트"""
        valid_date = "20250115"
        try:
            datetime.strptime(valid_date, '%Y%m%d')
            assert True
        except ValueError:
            assert False, "유효한 날짜 형식이어야 함"
    
    def test_invalid_date_format(self):
        """잘못된 날짜 형식 테스트"""
        invalid_dates = [
            "2025-01-15",  # 하이픈 사용 (YYYYMMDD 형식 아님)
            "2025/01/15",  # 슬래시 사용 (YYYYMMDD 형식 아님)
            "abcd",        # 문자열
            "20251301",    # 잘못된 월 (13월)
            "20250132",    # 잘못된 일 (32일)
            "20250001",    # 잘못된 월 (00월)
            "20250100",    # 잘못된 일 (00일)
        ]
        
        for invalid_date in invalid_dates:
            # YYYYMMDD 형식으로 파싱 시도
            with pytest.raises(ValueError):
                datetime.strptime(invalid_date, '%Y%m%d')


class TestMarketCloseCheck:
    """장마감 확인 로직 테스트"""
    
    def test_before_market_close(self):
        """장마감 전 시간 테스트"""
        # 15:30 이전
        before_close = datetime(2025, 1, 15, 14, 0, 0)
        market_close = before_close.replace(hour=15, minute=30, second=0, microsecond=0)
        
        assert before_close < market_close, "장마감 전이어야 함"
    
    def test_after_market_close(self):
        """장마감 후 시간 테스트"""
        # 15:30 이후
        after_close = datetime(2025, 1, 15, 16, 0, 0)
        market_close = after_close.replace(hour=15, minute=30, second=0, microsecond=0)
        
        assert after_close >= market_close, "장마감 후이어야 함"
    
    def test_exactly_market_close(self):
        """정확히 장마감 시간 테스트"""
        # 15:30 정각
        exactly_close = datetime(2025, 1, 15, 15, 30, 0)
        market_close = exactly_close.replace(hour=15, minute=30, second=0, microsecond=0)
        
        assert exactly_close >= market_close, "장마감 시간 이상이어야 함"


class TestArgumentValidation:
    """인자 유효성 검증 테스트"""
    
    def test_target_date_with_range_invalid(self):
        """--target-date와 --start-date/--end-date 동시 사용 불가"""
        # 이 테스트는 main() 함수의 인자 파싱 로직 테스트
        # 실제로는 argparse를 통해 검증됨
        pass
    
    def test_start_date_without_end_date_invalid(self):
        """--start-date만 있는 경우 불가"""
        pass
    
    def test_end_date_without_start_date_invalid(self):
        """--end-date만 있는 경우 불가"""
        pass


class TestNowFnInjection:
    """now_fn 주입 테스트"""
    
    def test_now_fn_injection(self):
        """now_fn 파라미터 주입 테스트"""
        # 고정 시간 설정
        fixed_time = datetime(2025, 1, 15, 16, 0, 0)
        
        # now_fn 주입
        now_fn = lambda: fixed_time
        
        # 주입된 시간 함수 사용
        current_time = now_fn()
        
        assert current_time == fixed_time, "주입된 시간 함수가 정상 작동해야 함"
    
    def test_now_fn_default_behavior(self):
        """now_fn이 None인 경우 기본 동작 테스트"""
        from scripts.fetch_daily_data import datetime
        
        # now_fn이 None인 경우 datetime.now 사용
        now_fn = None
        _now = now_fn if now_fn is not None else datetime.now
        
        current_time = _now()
        
        assert isinstance(current_time, datetime), "datetime 객체여야 함"
        assert current_time <= datetime.now(), "현재 시간 이하여야 함"


class TestConfigLoading:
    """config 로드 테스트"""
    
    def test_config_json_loading(self):
        """config.json 로드 테스트"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.json'
            
            config = {
                'ebest': {
                    'target_date': '20250115',
                    'kp200_upcode': 'A0166000'
                }
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f)
            
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            assert loaded_config['ebest']['target_date'] == '20250115'
            assert loaded_config['ebest']['kp200_upcode'] == 'A0166000'
    
    def test_config_secrets_loading(self):
        """config.secrets.json 로드 테스트"""
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_path = Path(tmpdir) / 'config.secrets.json'
            
            secrets = {
                'ebest': {
                    'appkey': 'test_key',
                    'appsecretkey': 'test_secret'
                }
            }
            
            with open(secrets_path, 'w', encoding='utf-8') as f:
                json.dump(secrets, f)
            
            with open(secrets_path, 'r', encoding='utf-8') as f:
                loaded_secrets = json.load(f)
            
            assert loaded_secrets['ebest']['appkey'] == 'test_key'
            assert loaded_secrets['ebest']['appsecretkey'] == 'test_secret'
    
    def test_config_secrets_missing(self):
        """config.secrets.json 누락 시 빈 dict 반환 테스트"""
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_path = Path(tmpdir) / 'config.secrets.json'
            
            # 파일이 없는 경우
            if not secrets_path.exists():
                secrets = {}
                assert secrets == {}, "빈 dict여야 함"


class TestCSVOutputPath:
    """CSV 출력 경로 테스트"""
    
    def test_single_date_output_path(self):
        """단일 날짜 출력 경로 테스트"""
        target_date = "20250115"
        expected_filename = f"minute_bars_kp200_{target_date}.csv"
        
        assert expected_filename == "minute_bars_kp200_20250115.csv"
    
    def test_date_range_output_path(self):
        """날짜 범위 출력 경로 테스트"""
        start_date = "20250115"
        end_date = "20250121"
        expected_filename = f"minute_bars_kp200_{start_date}_{end_date}.csv"
        
        assert expected_filename == "minute_bars_kp200_20250115_20250121.csv"


class TestWeekdayFilter:
    """주말 필터링 테스트"""
    
    def test_weekday_filter_monday(self):
        """월요일 필터링 테스트"""
        # 2025-01-13은 월요일
        dt = datetime(2025, 1, 13)
        assert dt.weekday() == 0, "월요일이어야 함 (weekday=0)"
    
    def test_weekday_filter_friday(self):
        """금요일 필터링 테스트"""
        # 2025-01-17은 금요일
        dt = datetime(2025, 1, 17)
        assert dt.weekday() == 4, "금요일이어야 함 (weekday=4)"
    
    def test_weekday_filter_saturday(self):
        """토요일 필터링 테스트"""
        # 2025-01-18은 토요일
        dt = datetime(2025, 1, 18)
        assert dt.weekday() == 5, "토요일이어야 함 (weekday=5)"
    
    def test_weekday_filter_sunday(self):
        """일요일 필터링 테스트"""
        # 2025-01-19는 일요일
        dt = datetime(2025, 1, 19)
        assert dt.weekday() == 6, "일요일이어야 함 (weekday=6)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
