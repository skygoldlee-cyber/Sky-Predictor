"""
models/strike_utils.py
======================
KP200 옵션 행사가 코드 변환 유틸리티 — 단일 소스

행사가 코드 체계 (2026년 기준):
  "375"~"997" : 10진수 3자리 → 그 값이 직접 행사가(pt)
                간격: 2pt/3pt 교대 (평균 2.5pt)
  "A01"~"A11" : 알파벳 연장 코드 (1000pt 이상 행사가)
                A01=1000, A02=1002, A03=1005, A04=1007, A05=1010,
                A06=1012, A07=1015, A08=1017, A09=1020, A10=1022, A11=1025
  전체 심볼   : 마지막 3자리가 위의 행사가 코드
                예: "B016385" → "385" → 385.0pt

사용법:
    from models.strike_utils import strike_code_to_pt, extract_strike_pt, strike_sort_key
"""

from __future__ import annotations
from typing import Optional

# ── 알파벳 연장 행사가 코드 룩업 테이블 ──────────────────────────────────────
# 997 다음 2pt/3pt 교대: 1000, 1002, 1005, 1007, 1010, 1012, 1015, 1017, 1020, 1022, 1025
ALPHA_STRIKE_MAP: dict[str, float] = {
    "A01": 1000.0, "A02": 1002.0, "A03": 1005.0, "A04": 1007.0,
    "A05": 1010.0, "A06": 1012.0, "A07": 1015.0, "A08": 1017.0,
    "A09": 1020.0, "A10": 1022.0, "A11": 1025.0,
}

# 역방향 룩업 (pt → 코드) — ATM 탐색 등에서 사용
PT_TO_ALPHA: dict[float, str] = {v: k for k, v in ALPHA_STRIKE_MAP.items()}


def strike_code_to_pt(code: str) -> Optional[float]:
    """3자리 행사가 코드 → 실제 행사가(pt).

    '385'  → 385.0    (10진수 직접, 유효 범위 100~997)
    'A01'  → 1000.0   (알파벳 연장 룩업)
    기타   → None
    """
    try:
        s = str(code).strip().upper()
        if len(s) != 3:
            return None
        if s in ALPHA_STRIKE_MAP:
            return ALPHA_STRIKE_MAP[s]
        if s.isdigit():
            v = int(s)
            if 100 <= v <= 997:
                return float(v)
            return None
        return None
    except Exception:
        return None


def extract_strike_pt(symbol: str) -> Optional[float]:
    """심볼 코드(전체 또는 3자리)에서 행사가(pt) 추출.

    우선순위:
      1. 3자리 코드 직접 전달 ("385" / "A01") → strike_code_to_pt()
      2. 전체 심볼(≥6자리) → 마지막 3자리 추출 후 변환
      3. 순수 숫자 float 문자열 ("385.0") → float() 직접
    """
    try:
        s = str(symbol).strip().upper()
        if not s:
            return None
        if len(s) == 3:
            return strike_code_to_pt(s)
        if len(s) >= 6:
            return strike_code_to_pt(s[-3:])
        try:
            pt = float(s)
            if 100.0 <= pt <= 2000.0:
                return pt
        except Exception:
            pass
        return None
    except Exception:
        return None


def strike_sort_key(v: object) -> Optional[float]:
    """행사가 코드/심볼/pt값을 정렬용 float로 변환.

    None 반환 시 sort에서 맨 뒤로 이동 (None-safe 정렬).

    사용 예:
        keys.sort(key=lambda x: (strike_sort_key(x) is None, strike_sort_key(x) or 0))
    """
    if v is None:
        return None
    try:
        s = str(v).strip().upper()
        if not s or s in ("NAN", "NONE", "NULL"):
            return None
        # 이미 숫자 float/int 인 경우
        try:
            pt = float(v)
            if 100.0 <= pt <= 2000.0:
                return pt
        except Exception:
            pass
        # 3자리 코드 또는 전체 심볼
        return extract_strike_pt(s)
    except Exception:
        return None


def pt_to_strike_code(pt: float) -> Optional[str]:
    """행사가(pt) → 3자리 코드 역변환.

    1000.0 → 'A01',  385.0 → '385',  기타 → None
    """
    try:
        f = float(pt)
        if f in PT_TO_ALPHA:
            return PT_TO_ALPHA[f]
        i = int(f)
        if float(i) == f and 100 <= i <= 997:
            return str(i).zfill(3)
        return None
    except Exception:
        return None
