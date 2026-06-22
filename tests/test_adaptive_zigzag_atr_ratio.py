"""
시간대별 동적 ATR 비율 필터링 단위 테스트
"""
import datetime
from typing import List, Tuple

from indicators.adaptive_zigzag import _get_time_based_atr_ratio


def test_get_time_based_atr_ratio_empty_table():
    """빈 테이블이면 기본값 1.0 반환"""
    ratio_table: List[Tuple[str, str, float]] = []
    current_time = datetime.datetime(2026, 5, 1, 9, 15)
    
    result = _get_time_based_atr_ratio(current_time, ratio_table)
    assert result == 1.0


def test_get_time_based_atr_ratio_match():
    """시간대가 매칭되면 해당 비율 반환"""
    ratio_table = [
        ("09:00", "09:30", 0.8),
        ("09:30", "10:30", 1.2),
        ("10:30", "13:00", 1.8),
    ]
    
    # 장 시작 시간대
    current_time = datetime.datetime(2026, 5, 1, 9, 15)
    result = _get_time_based_atr_ratio(current_time, ratio_table)
    assert result == 0.8
    
    # 오전 시간대
    current_time = datetime.datetime(2026, 5, 1, 10, 0)
    result = _get_time_based_atr_ratio(current_time, ratio_table)
    assert result == 1.2
    
    # 점심 시간대
    current_time = datetime.datetime(2026, 5, 1, 12, 0)
    result = _get_time_based_atr_ratio(current_time, ratio_table)
    assert result == 1.8


def test_get_time_based_atr_ratio_no_match():
    """매칭되는 시간대가 없으면 기본값 1.0 반환"""
    ratio_table = [
        ("09:00", "09:30", 0.8),
        ("09:30", "10:30", 1.2),
    ]
    
    # 테이블에 없는 시간 (오후)
    current_time = datetime.datetime(2026, 5, 1, 14, 0)
    result = _get_time_based_atr_ratio(current_time, ratio_table)
    assert result == 1.0


def test_get_time_based_atr_ratio_boundary():
    """시간대 경계 테스트"""
    ratio_table = [
        ("09:00", "09:30", 0.8),
        ("09:30", "10:30", 1.2),
    ]
    
    # 시작 시간 포함
    current_time = datetime.datetime(2026, 5, 1, 9, 0)
    result = _get_time_based_atr_ratio(current_time, ratio_table)
    assert result == 0.8
    
    # 종료 시간 미포함 (다음 구간으로)
    current_time = datetime.datetime(2026, 5, 1, 9, 30)
    result = _get_time_based_atr_ratio(current_time, ratio_table)
    assert result == 1.2


def test_get_time_based_atr_ratio_invalid_format():
    """잘못된 형식은 무시하고 다음 구간 검색"""
    ratio_table = [
        ("invalid", "09:30", 0.8),
        ("09:30", "10:30", 1.2),
    ]
    
    current_time = datetime.datetime(2026, 5, 1, 9, 45)
    result = _get_time_based_atr_ratio(current_time, ratio_table)
    assert result == 1.2


def test_get_time_based_atr_ratio_balanced_strategy():
    """균형 전략 테이블 테스트"""
    ratio_table = [
        ("09:00", "09:30", 0.8),   # 장 시작: 빠른 반응
        ("09:30", "10:30", 1.2),   # 오전: 안정적
        ("10:30", "13:00", 1.8),   # 점심: 노이즈 필터링
        ("13:00", "14:30", 1.2),   # 오후: 안정적
        ("14:30", "15:20", 0.8),   # 마감 전: 빠른 반응
        ("15:20", "15:30", 0.5),   # 마감: 최고 민감도
    ]
    
    # 장 시작
    current_time = datetime.datetime(2026, 5, 1, 9, 15)
    result = _get_time_based_atr_ratio(current_time, ratio_table)
    assert result == 0.8
    
    # 오전
    current_time = datetime.datetime(2026, 5, 1, 10, 0)
    result = _get_time_based_atr_ratio(current_time, ratio_table)
    assert result == 1.2
    
    # 점심 (가장 높은 비율)
    current_time = datetime.datetime(2026, 5, 1, 12, 0)
    result = _get_time_based_atr_ratio(current_time, ratio_table)
    assert result == 1.8
    
    # 오후
    current_time = datetime.datetime(2026, 5, 1, 14, 0)
    result = _get_time_based_atr_ratio(current_time, ratio_table)
    assert result == 1.2
    
    # 마감 전
    current_time = datetime.datetime(2026, 5, 1, 15, 0)
    result = _get_time_based_atr_ratio(current_time, ratio_table)
    assert result == 0.8
    
    # 마감 (가장 낮은 비율)
    current_time = datetime.datetime(2026, 5, 1, 15, 25)
    result = _get_time_based_atr_ratio(current_time, ratio_table)
    assert result == 0.5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
