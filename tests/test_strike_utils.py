"""
core/strike_utils.py 단위 테스트
"""

from core.strike_utils import (
    strike_code_to_pt,
    extract_strike_pt,
    strike_sort_key,
    pt_to_strike_code,
    ALPHA_STRIKE_MAP,
    PT_TO_ALPHA
)


class TestStrikeCodeToPt:
    """strike_code_to_pt 테스트."""
    
    def test_numeric_code_valid(self):
        """유효한 숫자 코드."""
        assert strike_code_to_pt("385") == 385.0
        assert strike_code_to_pt("430") == 430.0
        assert strike_code_to_pt("100") == 100.0
    
    def test_numeric_code_invalid_range(self):
        """잘못된 범위의 숫자 코드."""
        assert strike_code_to_pt("099") is None  # 100 미만
        assert strike_code_to_pt("998") is None  # 997 초과
    
    def test_alpha_code_valid(self):
        """유효한 알파벳 코드."""
        assert strike_code_to_pt("A01") == 1000.0
        assert strike_code_to_pt("A05") == 1010.0
        assert strike_code_to_pt("A11") == 1025.0
    
    def test_alpha_code_invalid(self):
        """잘못된 알파벳 코드."""
        assert strike_code_to_pt("A12") is None  # 존재하지 않는 코드
        assert strike_code_to_pt("B01") is None  # 잘못된 접두사
    
    def test_invalid_format(self):
        """잘못된 형식."""
        assert strike_code_to_pt("") is None
        assert strike_code_to_pt("1234") is None  # 3자리 초과
        assert strike_code_to_pt("AB") is None  # 2자리
        assert strike_code_to_pt("invalid") is None


class TestExtractStrikePt:
    """extract_strike_pt 테스트."""
    
    def test_extract_3_digit_code(self):
        """3자리 코드 직접 전달."""
        assert extract_strike_pt("385") == 385.0
        assert extract_strike_pt("A01") == 1000.0
    
    def test_extract_from_full_symbol(self):
        """전체 심볼에서 추출."""
        assert extract_strike_pt("B016385") == 385.0
        assert extract_strike_pt("B016A01") == 1000.0
    
    def test_extract_float_string(self):
        """숫자 float 문자열."""
        assert extract_strike_pt("385.0") == 385.0
        assert extract_strike_pt("430.5") == 430.5
    
    def test_extract_invalid_range(self):
        """잘못된 범위."""
        assert extract_strike_pt("50.0") is None  # 100 미만
        assert extract_strike_pt("2500.0") is None  # 2000 초과
    
    def test_extract_empty(self):
        """빈 문자열."""
        assert extract_strike_pt("") is None
        assert extract_strike_pt(None) is None


class TestStrikeSortKey:
    """strike_sort_key 테스트."""
    
    def test_sort_key_numeric(self):
        """숫자 정렬 키."""
        assert strike_sort_key(385) == 385.0
        assert strike_sort_key(430.5) == 430.5
    
    def test_sort_key_string_code(self):
        """문자열 코드 정렬 키."""
        assert strike_sort_key("385") == 385.0
        assert strike_sort_key("A01") == 1000.0
    
    def test_sort_key_full_symbol(self):
        """전체 심볼 정렬 키."""
        assert strike_sort_key("B016385") == 385.0
        assert strike_sort_key("B016A01") == 1000.0
    
    def test_sort_key_none(self):
        """None 처리."""
        assert strike_sort_key(None) is None
        assert strike_sort_key("") is None
        assert strike_sort_key("NAN") is None
        assert strike_sort_key("NONE") is None
        assert strike_sort_key("NULL") is None
    
    def test_sort_key_invalid_range(self):
        """잘못된 범위."""
        assert strike_sort_key("50") is None  # 100 미만
        assert strike_sort_key("2500") is None  # 2000 초과


class TestPtToStrikeCode:
    """pt_to_strike_code 테스트."""
    
    def test_pt_to_code_numeric(self):
        """숫자 행사가 코드 변환."""
        assert pt_to_strike_code(385.0) == "385"
        assert pt_to_strike_code(430) == "430"
        assert pt_to_strike_code(100) == "100"
        assert pt_to_strike_code(997) == "997"
    
    def test_pt_to_code_alpha(self):
        """알파벳 행사가 코드 변환."""
        assert pt_to_strike_code(1000.0) == "A01"
        assert pt_to_strike_code(1010.0) == "A05"
        assert pt_to_strike_code(1025.0) == "A11"
    
    def test_pt_to_code_invalid_range(self):
        """잘못된 범위."""
        assert pt_to_strike_code(99.0) is None  # 100 미만
        assert pt_to_strike_code(998.0) is None  # 997 초과 (알파벳 코드 범위 밖)
        assert pt_to_strike_code(1026.0) is None  # 알파벳 코드 최대 초과
    
    def test_pt_to_code_non_integer(self):
        """정수가 아닌 경우."""
        assert pt_to_strike_code(385.5) is None  # 소수점
    
    def test_pt_to_code_string_input(self):
        """문자열 입력."""
        assert pt_to_strike_code("385") == "385"
        assert pt_to_strike_code("1000") == "A01"


class TestAlphaStrikeMap:
    """알파벳 행사가 맵 테스트."""
    
    def test_alpha_strike_map_completeness(self):
        """알파벽 행사가 맵 완전성."""
        assert "A01" in ALPHA_STRIKE_MAP
        assert "A11" in ALPHA_STRIKE_MAP
        assert len(ALPHA_STRIKE_MAP) == 11
    
    def test_alpha_strike_map_values(self):
        """알파벳 행사가 맵 값."""
        assert ALPHA_STRIKE_MAP["A01"] == 1000.0
        assert ALPHA_STRIKE_MAP["A05"] == 1010.0
        assert ALPHA_STRIKE_MAP["A11"] == 1025.0
    
    def test_pt_to_alpha_completeness(self):
        """역방향 맵 완전성."""
        assert 1000.0 in PT_TO_ALPHA
        assert 1025.0 in PT_TO_ALPHA
        assert len(PT_TO_ALPHA) == 11
    
    def test_pt_to_alpha_consistency(self):
        """맵 일관성."""
        for code, pt in ALPHA_STRIKE_MAP.items():
            assert PT_TO_ALPHA[pt] == code


class TestEdgeCases:
    """엣지 케이스 테스트."""
    
    def test_case_insensitive(self):
        """대소문자 구분 없음."""
        assert strike_code_to_pt("a01") == 1000.0
        assert strike_code_to_pt("A01") == 1000.0
        assert strike_code_to_pt("385") == 385.0
        assert strike_code_to_pt("385") == 385.0
    
    def test_whitespace_handling(self):
        """공백 처리."""
        assert strike_code_to_pt(" 385 ") == 385.0
        assert extract_strike_pt(" 385 ") == 385.0
    
    def test_boundary_values(self):
        """경계값."""
        assert strike_code_to_pt("100") == 100.0  # 최소
        assert strike_code_to_pt("997") == 997.0  # 최대
        assert strike_code_to_pt("099") is None  # 최소 미만
        assert strike_code_to_pt("998") is None  # 최대 초과
