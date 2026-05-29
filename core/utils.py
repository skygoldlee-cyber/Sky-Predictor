"""
유틸리티 함수 모듈

개선사항:
- 타입 힌팅 완비
- 함수 중복 제거
- 에러 처리 개선
- 문서화 강화
"""

import calendar
import json
import logging
import math
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np

from .strike_utils import strike_code_to_pt  # noqa: E402

logger = logging.getLogger(__name__)

# 옵션 만기일 보정용 휴장일(YYYY-MM-DD) 집합.
_EXPIRY_HOLIDAYS: set[date] = set()

# scipy가 없어도 작동하도록
try:
    from scipy.stats import norm as _scipy_norm
except ImportError:
    _scipy_norm = None
    logger.warning("scipy not available, using fallback normal distribution")


# ============================================================================
# 타입 변환 유틸리티
# ============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    안전한 float 변환
    
    Args:
        value: 변환할 값
        default: 기본값
        
    Returns:
        변환된 float 값
        
    Example:
        >>> safe_float("123.45")
        123.45
        >>> safe_float("invalid", 0.0)
        0.0
    """
    try:
        return float(value)
    except (TypeError, ValueError, AttributeError):
        return default


def normalize_ohlcv_columns(df: Any) -> Any:
    try:
        import pandas as pd

        if df is None:
            return df
        if not isinstance(df, pd.DataFrame):
            return df
        if df.empty:
            return df

        cols = list(df.columns)
        norm = {str(c).strip().lower(): str(c) for c in cols}
        rename = {}
        for want, key in (
            ("Open", "open"),
            ("High", "high"),
            ("Low", "low"),
            ("Close", "close"),
            ("Volume", "volume"),
        ):
            src = norm.get(key)
            if src is not None and str(src) != str(want):
                rename[str(src)] = str(want)

        return df.rename(columns=rename) if rename else df
    except Exception:
        return df


def calc_direction(*, predicted: float, current: float, threshold_pct: float = 0.0, epsilon: float = 1e-9) -> str:
    """Compute a discrete direction label from predicted/current prices.

    Args:
        predicted: Predicted price.
        current: Current price.
        threshold_pct: Minimum absolute percentage move required to be considered up/down.
        epsilon: Numerical floor used when `threshold_pct` is near zero.

    Returns:
        One of: `"up"`, `"down"`, `"neutral"`.
    """
    c = float(current)
    p = float(predicted)
    th = float(threshold_pct)
    eps = float(epsilon)

    if c == 0.0:
        if p > 0.0:
            return "up"
        if p < 0.0:
            return "down"
        return "neutral"

    delta_pct = (p - c) / c * 100.0

    eff_th = abs(th)
    if eff_th < eps:
        eff_th = eps

    if delta_pct > eff_th:
        return "up"
    if delta_pct < -eff_th:
        return "down"
    return "neutral"


def safe_int(value: Any, default: int = 0) -> int:
    """
    안전한 int 변환
    
    Args:
        value: 변환할 값
        default: 기본값
        
    Returns:
        변환된 int 값
        
    Example:
        >>> safe_int("123")
        123
        >>> safe_int("12.7")
        12
        >>> safe_int("invalid", 0)
        0
    """
    try:
        return int(float(value))
    except (TypeError, ValueError, AttributeError):
        return default


# ============================================================================
# 통계 함수
# ============================================================================

def norm_cdf(x: float) -> float:
    """
    표준 정규분포의 누적분포함수
    
    Args:
        x: 입력값
        
    Returns:
        P(X <= x)
    """
    try:
        if _scipy_norm is not None:
            return float(_scipy_norm.cdf(x))
    except Exception as e:
        logger.debug(f"scipy norm_cdf failed: {e}, using fallback")
    
    # Fallback: error function 사용
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """
    표준 정규분포의 확률밀도함수
    
    Args:
        x: 입력값
        
    Returns:
        f(x)
    """
    try:
        if _scipy_norm is not None:
            return float(_scipy_norm.pdf(x))
    except Exception as e:
        logger.debug(f"scipy norm_pdf failed: {e}, using fallback")
    
    # Fallback: 직접 계산
    xx = float(x)
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * xx * xx)


# ============================================================================
# 날짜/시간 유틸리티
# ============================================================================

def get_second_thursday_date(year: int, month: int) -> datetime:
    """
    해당 월의 두 번째 목요일 날짜 반환
    
    Args:
        year: 연도
        month: 월 (1-12)
        
    Returns:
        두 번째 목요일의 datetime
        
    Example:
        >>> date = get_second_thursday_date(2025, 2)
        >>> print(date)
        2025-02-13 00:00:00
    """
    month_calendar = calendar.monthcalendar(year, month)
    thursdays = [week[3] for week in month_calendar if week[3] != 0]
    
    # 두 번째 목요일 (없으면 첫 번째)
    second_thursday_day = thursdays[1] if len(thursdays) > 1 else thursdays[0]
    
    return datetime(year, month, second_thursday_day)


def set_expiry_holidays(holiday_dates: Any) -> int:
    """옵션 만기일 보정에 사용할 휴장일 목록을 설정한다.

    Args:
        holiday_dates:
            - ["YYYY-MM-DD", ...] 또는
            - ["YYYYMMDD", ...]

    Returns:
        유효하게 반영된 휴장일 개수.
    """
    global _EXPIRY_HOLIDAYS
    parsed: set[date] = set()
    try:
        items = holiday_dates if isinstance(holiday_dates, list) else []
        for raw in items:
            try:
                s = str(raw or "").strip()
                if not s:
                    continue
                if len(s) == 8 and s.isdigit():
                    dt = datetime.strptime(s, "%Y%m%d").date()
                else:
                    dt = datetime.strptime(s, "%Y-%m-%d").date()
                parsed.add(dt)
            except Exception:
                continue
    except Exception:
        parsed = set()

    _EXPIRY_HOLIDAYS = set(parsed)
    try:
        if parsed:
            logger.info(f"[EXPIRY] holiday calendar loaded: {len(parsed)} days")
    except Exception:
        pass
    return int(len(parsed))


def _is_business_day(d: date) -> bool:
    try:
        if int(d.weekday()) >= 5:
            return False
        if d in _EXPIRY_HOLIDAYS:
            return False
        return True
    except Exception:
        return True


def get_previous_business_day(target_date: Optional[date] = None, days_back: int = 1) -> Optional[date]:
    """특정 날짜 기준 이전 영업일 계산.
    
    토요일, 일요일, 공휴일을 제외하고 이전 영업일을 반환한다.
    
    Args:
        target_date: 기준 날짜 (기본: 오늘)
        days_back: 몇 영업일 전 (기본: 1)
    
    Returns:
        이전 영업일 (계산 불가능 시 None)
    
    Example:
        # 어제 영업일
        prev_day = get_previous_business_day()
        
        # 3 영업일 전
        prev_3_day = get_previous_business_day(days_back=3)
    """
    try:
        if target_date is None:
            target_date = datetime.now().date()
        
        current = target_date
        business_days_found = 0
        
        # 최대 30일까지만 역추적 (무한 루프 방지)
        for _ in range(30):
            current = current - timedelta(days=1)
            if _is_business_day(current):
                business_days_found += 1
                if business_days_found >= days_back:
                    return current
        
        logger.warning(f"[get_previous_business_day] 30일 내에 영업일을 찾지 못함: target={target_date}, days_back={days_back}")
        return None
        
    except Exception as e:
        logger.error(f"[get_previous_business_day] 계산 실패: {e}")
        return None


async def fetch_previous_market_data(
    api: Any,
    symbol: str,
    target_date: Optional[date] = None,
    days_back: int = 1,
    ncnt: int = 1
) -> Optional[Dict[str, Any]]:
    """이전장 OHLCV 데이터 가져오기.
    
    토요일, 일요일, 공휴일을 제외하고 이전 영업일의 데이터를 가져온다.
    
    Args:
        api: eBest wrapper client
        symbol: 선물 심볼 코드 (예: "101V3000")
        target_date: 기준 날짜 (기본: 오늘)
        days_back: 몇 영업일 전 (기본: 1)
        ncnt: 분 단위 (1=1분, 5=5분 등)
    
    Returns:
        {
            "date": "YYYYMMDD",
            "bars": [OHLCV bars],
            "prev_business_date": date object
        }
        실패 시 None
    """
    try:
        # 이전 영업일 계산
        prev_business_date = get_previous_business_day(target_date, days_back)
        if prev_business_date is None:
            logger.warning("[fetch_previous_market_data] 이전 영업일 계산 실패")
            return None
        
        yyyymmdd = prev_business_date.strftime("%Y%m%d")
        logger.info(f"[fetch_previous_market_data] 이전 영업일: {yyyymmdd} (심볼: {symbol})")
        
        # eBest API import
        try:
            from ebestapi.api import _ebest_fetch_kp200_ohlcv_t8415
        except ImportError:
            logger.error("[fetch_previous_market_data] eBest API import 실패")
            return None
        
        # OHLCV 데이터 가져오기
        bars = await _ebest_fetch_kp200_ohlcv_t8415(
            api,
            symbol=symbol,
            yyyymmdd=yyyymmdd,
            ncnt=ncnt
        )
        
        if bars is None:
            logger.warning(f"[fetch_previous_market_data] 데이터 없음: {yyyymmdd}")
            return None
        
        logger.info(f"[fetch_previous_market_data] 데이터 가져오기 성공: {yyyymmdd} ({len(bars)} bars)")
        
        return {
            "date": yyyymmdd,
            "bars": bars,
            "prev_business_date": prev_business_date
        }
        
    except Exception as e:
        logger.error(f"[fetch_previous_market_data] 데이터 가져오기 실패: {e}")
        return None


def get_option_expiry_date(year: int, month: int) -> datetime:
    """옵션 만기일 반환.

    기본은 둘째 목요일이며, 해당일이 휴장일/주말이면 직전 영업일로 보정한다.
    """
    expiry_dt = get_second_thursday_date(year, month)
    d = expiry_dt.date()
    while not _is_business_day(d):
        d = d - timedelta(days=1)
    return datetime.combine(d, datetime.min.time())


def get_expiry_week_info(now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    옵션 만기주 정보 반환
    
    Args:
        now: 기준 시각 (None이면 현재)
        
    Returns:
        {
            'is_expiry_week': bool,
            'expiry_second_thursday': datetime,
            'days_to_expiry': int
        }
    """
    if now is None:
        now = datetime.now()
    
    expiry_dt = get_option_expiry_date(now.year, now.month)
    
    # 만기일이 지났으면 다음 달
    if now.date() > expiry_dt.date():
        next_month = now.month + 1
        next_year = now.year
        if next_month > 12:
            next_month = 1
            next_year += 1
        expiry_dt = get_option_expiry_date(next_year, next_month)
    
    # 만기주 여부 (만기일이 포함된 주, 월~일)
    week_start = expiry_dt - timedelta(days=expiry_dt.weekday())
    week_end = week_start + timedelta(days=6)
    is_expiry_week = week_start.date() <= now.date() <= week_end.date()
    
    days_to_expiry = (expiry_dt.date() - now.date()).days
    
    return {
        'is_expiry_week': is_expiry_week,
        'expiry_second_thursday': expiry_dt,
        'days_to_expiry': days_to_expiry,
    }


def get_option_month_yyyymm(now: Optional[datetime] = None) -> str:
    """
    현재 또는 다음 옵션 만기월 반환
    
    Args:
        now: 기준 시각 (None이면 현재)
        
    Returns:
        만기월 (YYYYMM 형식)
        
    Example:
        >>> # 2월 10일 (만기 전)
        >>> month = get_option_month_yyyymm(datetime(2025, 2, 10))
        >>> print(month)
        '202502'
        
        >>> # 2월 14일 (만기 후)
        >>> month = get_option_month_yyyymm(datetime(2025, 2, 14))
        >>> print(month)
        '202503'
    """
    if now is None:
        now = datetime.now()
    
    expiry_dt = get_option_expiry_date(now.year, now.month)
    
    # 만기일이 지났으면 다음 달
    if now.date() > expiry_dt.date():
        month = now.month + 1
        year = now.year
        if month > 12:
            month = 1
            year += 1
    else:
        month = now.month
        year = now.year
    
    return f"{year:04d}{month:02d}"


def parse_chetime(chetime: str, reference: Optional[datetime] = None) -> datetime:
    """
    체결시간 파싱 (HHMMSS → datetime)
    
    Args:
        chetime: 체결시간 (HHMMSS 형식)
        reference: 기준 시각 (None이면 현재)
        
    Returns:
        파싱된 datetime
        
    Note:
        - 기준 시각과 12시간 이상 차이나면 날짜 자동 조정
        
    Example:
        >>> dt = parse_chetime("130430")
        >>> print(dt.time())
        13:04:30
    """
    if reference is None:
        reference = datetime.now()
    
    # chetime=None 은 장 시작/종료 경계 및 데이터 공백 구간에서 정상 수신됨.
    # 예측 흐름에 영향 없는 정상 케이스이므로 WARNING 대신 DEBUG 로 처리한다.
    if chetime is None:
        logger.debug("parse_chetime: chetime=None (장 경계/공백 구간 정상 케이스) — reference 반환")
        return reference.replace(microsecond=0)

    s = str(chetime).strip()
    if len(s) != 6 or not s.isdigit():
        logger.debug("Invalid chetime format: %s — reference 반환", chetime)
        return reference.replace(microsecond=0)
    
    try:
        hour = int(s[:2])
        minute = int(s[2:4])
        second = int(s[4:6])
        
        # 시간 검증
        if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
            logger.debug("Invalid time values in chetime: %s — reference 반환", chetime)
            return reference.replace(microsecond=0)
        
        ts = reference.replace(hour=hour, minute=minute, second=second, microsecond=0)
        
        # 날짜 조정 (12시간 이상 차이)
        if ts - reference > timedelta(hours=12):
            ts = ts - timedelta(days=1)
        elif reference - ts > timedelta(hours=12):
            ts = ts + timedelta(days=1)
        
        return ts
        
    except Exception as e:
        logger.error(f"Failed to parse chetime {chetime}: {e}")
        return reference.replace(microsecond=0)


def parse_ebest_tick_datetime(
    time_val: Any,
    *,
    reference: Optional[datetime] = None,
) -> datetime:
    """eBest IJ_ 등 지수 틱의 time 필드를 분봉 버킷용 datetime으로 (FC0 chetime과 동일 분 정렬)."""
    ref = reference if isinstance(reference, datetime) else datetime.now()
    ref = ref.replace(microsecond=0)
    if time_val is None:
        return ref
    if isinstance(time_val, datetime):
        return time_val.replace(microsecond=0)
    s = str(time_val).strip()
    if not s:
        return ref
    if s.isdigit() and len(s) == 14:
        try:
            return datetime(
                int(s[0:4]),
                int(s[4:6]),
                int(s[6:8]),
                int(s[8:10]),
                int(s[10:12]),
                int(s[12:14]),
            )
        except Exception:
            return ref
    if s.isdigit() and len(s) == 6:
        return parse_chetime(s, reference=ref)
    if s.isdigit() and len(s) == 8:
        try:
            return datetime(
                int(s[0:4]),
                int(s[4:6]),
                int(s[6:8]),
                ref.hour,
                ref.minute,
                ref.second,
            ).replace(microsecond=0)
        except Exception:
            return ref
    try:
        if "T" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt.replace(microsecond=0)
    except Exception:
        pass
    if ":" in s:
        parts = [int(x) for x in re.findall(r"\d+", s)[:3]]
        while len(parts) < 3:
            parts.append(0)
        h, m, sec = int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
        try:
            ts = ref.replace(
                hour=h % 24,
                minute=min(max(m, 0), 59),
                second=min(max(sec, 0), 59),
                microsecond=0,
            )
            if ts - ref > timedelta(hours=12):
                ts -= timedelta(days=1)
            elif ref - ts > timedelta(hours=12):
                ts += timedelta(days=1)
            return ts
        except Exception:
            return ref
    return ref


def normalize_adaptive_indicator_symbol(sym: Any) -> str:
    raw = str(sym or "").strip()
    return re.sub(r"\s+", "", raw.upper()).replace("_", "").replace("-", "")


def get_pipeline_adaptive_indicator_symbol(pipeline: Any) -> str:
    """`PredictionPipeline` 등에서 적응형 심볼 문자열을 가져온다.

    dual_mode는 항상 true이므로 kospi_symbol을 반환한다.
    """
    try:
        cfg = getattr(pipeline, "config", None)
        if cfg is not None:
            ad = getattr(cfg, "adaptive_indicator", None)
            if ad is not None:
                sym = getattr(ad, "kospi_symbol", None)
                if sym is not None and str(sym).strip():
                    return str(sym).strip()
    except Exception:
        pass
    try:
        ad_any = getattr(pipeline, "_adaptive_indicator", None)
        if isinstance(ad_any, dict):
            s = ad_any.get("kospi_symbol")
            if s is not None and str(s).strip():
                return str(s).strip()
    except Exception:
        pass
    return "KOSPI 지수"


def adaptive_uses_kospi_spot_index_minute_bars(pipeline: Any) -> bool:
    """적응형 번들 분봉이 KOSPI 현물 지수(IJ_) 분봉을 써야 하면 True.

    dual_mode는 항상 true이므로 항상 True를 반환한다.
    """
    return True


# ============================================================================
# JSON 유틸리티
# ============================================================================

def make_json_safe(obj: Any, _depth: int = 0) -> Any:
    """
    JSON 직렬화 가능한 형태로 변환
    
    Args:
        obj: 변환할 객체
        
    Returns:
        JSON 직렬화 가능한 객체
    """
    if int(_depth) > 20:
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if isinstance(obj, dict):
            return {str(k): make_json_safe(v, _depth + 1) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_safe(v, _depth + 1) for v in obj]
        elif isinstance(obj, (datetime, np.datetime64)):
            return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return str(obj)


def write_jsonl_line(file_handle, obj: Any) -> None:
    """
    JSONL 형식으로 한 줄 쓰기
    
    Args:
        file_handle: 파일 핸들
        obj: 쓸 객체
    """
    try:
        safe_obj = make_json_safe(obj)
        file_handle.write(json.dumps(safe_obj, ensure_ascii=False) + "\n")
        file_handle.flush()
    except Exception as e:
        logger.error(f"Failed to write JSONL line: {e}")


# ============================================================================
# 파일 경로 유틸리티
# ============================================================================

def get_default_ticks_output_path(now: Optional[datetime] = None) -> str:
    """
    기본 틱 저장 파일명 생성
    
    Args:
        now: 기준 시각 (None이면 현재)
        
    Returns:
        파일명 (ticks_replay_YYYYMMDD_HHMMSS.jsonl)
    """
    if now is None:
        now = datetime.now()
    return f"ticks_replay_{now.strftime('%Y%m%d_%H%M%S')}.jsonl"


# ============================================================================
# 옵션 관련 유틸리티
# ============================================================================

def validate_strike_price(strike: float, min_strike: float = 200.0, 
                         max_strike: float = 500.0) -> bool:
    """
    행사가 유효성 검증
    
    Args:
        strike: 행사가
        min_strike: 최소 행사가
        max_strike: 최대 행사가
        
    Returns:
        True if valid
    """
    return min_strike <= strike <= max_strike


def parse_strike_from_code(strike_str: str) -> Optional[float]:
    """옵션 코드에서 행사가 파싱.

    3자리 행사가 코드("385", "A01" 등)를 실수로 변환한다.
    실제 변환은 strike_utils.strike_code_to_pt()에 위임한다.

    Args:
        strike_str: 3자리 행사가 코드 문자열 (예: "430", "A01")

    Returns:
        행사가 (float) 또는 None

    Example:
        >>> parse_strike_from_code("430")
        430.0
        >>> parse_strike_from_code("A01")
        1000.0
        >>> parse_strike_from_code("invalid")
        None
    """
    if not strike_str:
        return None
    result = strike_code_to_pt(strike_str)
    if result is None:
        logger.warning(f"Failed to parse strike code: {strike_str}")
    return result


# [IMP-2-3] set_seed를 utils로 이동. train.py / train_tft.py 에서 공유 사용.
def set_seed(seed: int = 42) -> None:
    """재현성을 위한 전역 시드 설정 (numpy / random / torch).

    Args:
        seed: 시드 값 (기본값: 42)
    """
    import random
    import numpy as np

    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"set_seed torch 설정 실패: {e}")
