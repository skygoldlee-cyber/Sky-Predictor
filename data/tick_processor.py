"""
실시간 틱 데이터 처리 모듈

Features:
- 선물/옵션 틱 처리
- 분봉 자동 집계
- 옵션 코드 파싱
- 메모리 관리
- 백테스팅용 데이터 자동 저장
"""

import logging
import threading
import time
from collections import defaultdict, deque
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

_KST = ZoneInfo("Asia/Seoul")

from config import (
    TRCode,
    FUTURES_MINUTE_RETENTION_HOURS,
    OPTION_MINUTE_RETENTION_HOURS,
    MAX_FUTURES_TICKS,
    load_config,
)
from core.utils import normalize_ohlcv_columns, parse_ebest_tick_datetime, safe_float, safe_int, parse_chetime
from core.strike_utils import extract_strike_pt

logger = logging.getLogger(__name__)


class RealTimeTickProcessor:
    """
    실시간 틱 데이터 처리 및 분봉 집계 클래스
    
    Features:
        - FC0 (선물) 틱 처리 및 분봉 집계
        - IJ_ (KOSPI 현물 지수) 틱 분봉 집계 — adaptive_indicator.symbol=「KOSPI 지수」일 때 적응형 번들에 사용
        - OC0 (옵션) 틱 처리 및 콜/풋 분류
        - 옵션 코드 자동 파싱 (B016: 콜, C016: 풋)
        - 만기월, 행사가 자동 추출
        - 자동 메모리 관리 (오래된 데이터 삭제)
    
    Attributes:
        futures_ticks: 선물 틱 리스트
        futures_minute_data: 분봉 집계 데이터
        call_options: 콜옵션 데이터 (symbol → data)
        put_options: 풋옵션 데이터 (symbol → data)
        call_option_ticks: 콜옵션 틱 카운터
        put_option_ticks: 풋옵션 틱 카운터
    
    Example:
        >>> processor = RealTimeTickProcessor()
        >>> 
        >>> # 틱 처리
        >>> processor.process_tick(futures_tick)
        >>> processor.process_tick(call_tick)
        >>> 
        >>> # 분봉 DataFrame
        >>> df = processor.get_futures_minute_df(60)
        >>> 
        >>> # 현재 가격
        >>> price = processor.get_current_price()
        >>> print(f"현재가: {price:.2f}")
    """
    
    def __init__(
        self,
        *,
        default_futures_minutes: int = 120,
        default_options_minutes: int = 120,
        fetch_market_service=None,
    ):
        """초기화"""
        self.futures_ticks: deque[Dict[str, Any]] = deque(maxlen=int(MAX_FUTURES_TICKS))
        self.futures_minute_data: defaultdict[datetime, List[Dict[str, Any]]] = defaultdict(list)
        self.kospi_minute_data: defaultdict[datetime, List[Dict[str, Any]]] = defaultdict(list)
        self._kospi_lock: threading.Lock = threading.Lock()
        self._last_kospi_timestamp: Optional[datetime] = None
        self.call_options: Dict[str, Dict] = {}
        self.put_options: Dict[str, Dict] = {}
        self.call_option_ticks = 0
        self.put_option_ticks = 0
        self.market_closed: bool = False
        self.option_minute_enabled: bool = False
        self.option_minute_atm_window: int = 2
        self._option_minute_allowed_symbols: set[str] = set()
        self.options_minute_data: Dict[str, Dict[datetime, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        self._options_minute_sweep_counter: int = 0
        self._options_minute_last_sweep_epoch: float = 0.0
        self.logger = logger
        self.fetch_market_service = fetch_market_service  # eBest API 서비스 (선택사항)

        # [FIX-AMP-1] 당일 세션 OHLC 누적 관리.
        # futures_minute_data 는 FUTURES_MINUTE_RETENTION_HOURS(2시간)치만 보관하므로
        # 오후 장에서는 session_high/low/open 추출 시 당일 전체 범위를 포함하지 못한다.
        # → 틱 수신 시마다 직접 누적하고 날짜 변경 시 리셋한다.
        self.session_open: Optional[float] = None
        self.session_high: float = -float('inf')
        self.session_low: float = float('inf')
        self.session_date: Optional[datetime] = None

        # 백테스팅 데이터 저장
        self._backtest_data_saver = None
        self._last_save_date: Optional[datetime.date] = None
        self._save_check_counter: int = 0
        self._save_check_interval: int = 100  # 100틱마다 체크
        self._daily_session_high:  float = 0.0      # 당일 누적 고가
        self._daily_session_low:   float = 0.0      # 당일 누적 저가
        self._daily_session_open:  float = 0.0      # 당일 시가 (첫 틱의 open 필드)
        self._daily_session_date:  str = ""         # 당일 날짜 (YYYYMMDD)

        # 장 초기 t2301에서 수신한 옵션 시가 맵 (symbol → open_price).
        # set_option_open_map()으로 주입되며, process_option_tick() 내에서
        # 신규 tick 수신 시 "open_price" 필드를 자동 주입하는 데 사용된다.
        self._call_open_map: Dict[str, float] = {}
        self._put_open_map: Dict[str, float] = {}

        # call_options / put_options 동시 접근 보호.
        # process_option_tick(I/O 스레드) + set_option_open_map / calc_otm_premium_change
        # (예측 스레드)가 동일 딕셔너리를 읽고 쓰므로 Lock이 필요하다.
        self._options_lock: threading.Lock = threading.Lock()

        # futures_ticks 동시 접근 보호.
        # process_futures_tick(eBest 콜백 스레드) + get_current_price / get_price_at /
        # get_price_near / get_statistics (예측·피드백 스레드)가 동시에 접근한다.
        # deque.append/popleft는 GIL로 보호되지만 iteration + 정리 복합 연산은 race condition이
        # 발생할 수 있으므로 명시적 Lock으로 직렬화한다.
        self._futures_lock: threading.Lock = threading.Lock()

        try:
            self.default_futures_minutes = max(1, int(default_futures_minutes))
        except Exception:
            self.default_futures_minutes = 120
        try:
            self.default_options_minutes = max(1, int(default_options_minutes))
        except Exception:
            self.default_options_minutes = 120

        # ── [PERF-3] 분봉 병합 결과 캐시 ──────────────────────────────────
        # get_futures_minute_df / get_kospi_minute_df 가 500ms 마다 호출될 때
        # _futures_minute_df.copy() + pd.concat + sort_index 를 매번 실행하지 않도록
        # 신규 틱 봉 수가 동일한 구간에서는 이전 병합 결과를 재사용한다.
        #
        # 캐시 키: (base_df 행 수, live_minute_data 키 수)
        # 신봉이 추가될 때만 키가 변경되므로 분봉 1개 추가 간격(1분)에 1회만 concat 발생.
        self._merged_futures_df:      Optional[pd.DataFrame] = None
        self._merged_futures_key:     Optional[tuple]        = None   # (base_len, new_key_count)
        self._merged_spot_df:         Optional[pd.DataFrame] = None
        self._merged_spot_key:        Optional[tuple]        = None

    def set_market_closed(self, value: bool) -> None:
        """market_closed 플래그를 설정한다.

        CON-04: 외부에서 setattr(tp, 'market_closed', True) 로 직접 접근하던 것을
        명시적 세터 메서드로 교체. 상태 변경 시 로깅을 보장하고 향후 부수 작업
        (이벤트 발행, 리소스 정리 등)을 추가하기 위한 단일 진입점을 제공한다.
        """
        prev = self.market_closed
        self.market_closed = bool(value)
        if bool(value) and not prev:
            self.logger.info("[TickProcessor] 장 종료 감지 — market_closed=True")
        elif not bool(value) and prev:
            self.logger.info("[TickProcessor] 장 재개 — market_closed=False")

    def clear_minute_cache(self) -> None:
        """초기 데이터 캐시를 초기화한다.

        _futures_minute_df, _kospi_minute_df 캐시를 삭제하여
        다음 데이터 요청 시 새로운 데이터를 가져오도록 한다.
        """
        if hasattr(self, "_futures_minute_df"):
            self._futures_minute_df = None
        if hasattr(self, "_kospi_minute_df"):
            self._kospi_minute_df = None
        if hasattr(self, "_merged_futures_df"):
            self._merged_futures_df = None
        if hasattr(self, "_merged_spot_df"):
            self._merged_spot_df = None
        self.logger.info("[TickProcessor] 분봉 캐시 초기화 완료")

    def set_option_open_map(
        self,
        call_open_map: Dict[str, float],
        put_open_map: Dict[str, float],
    ) -> None:
        """장 초기 t2301에서 수신한 옵션 시가를 주입한다.

        이미 call_options/put_options에 저장된 항목에는 즉시 "open_price"를
        갱신하고, 이후 process_option_tick()으로 수신되는 신규 tick에도
        _call_open_map/_put_open_map 캐시를 통해 자동 주입된다.

        Args:
            call_open_map: {symbol: open_price} 콜옵션 시가 맵.
            put_open_map:  {symbol: open_price} 풋옵션 시가 맵.
        """
        with self._options_lock:
            self._call_open_map = {str(k): float(v) for k, v in (call_open_map or {}).items() if float(v or 0.0) > 0.0}
            self._put_open_map  = {str(k): float(v) for k, v in (put_open_map  or {}).items() if float(v or 0.0) > 0.0}

            # 이미 수신된 tick에 open_price 소급 주입
            for sym, op in self._call_open_map.items():
                if sym in self.call_options:
                    self.call_options[sym]["open_price"] = op
            for sym, op in self._put_open_map.items():
                if sym in self.put_options:
                    self.put_options[sym]["open_price"] = op

        self.logger.info(
            "[TickProcessor] set_option_open_map: call=%d put=%d",
            len(self._call_open_map),
            len(self._put_open_map),
        )

    def configure_option_minute_ohlcv(self, *, enabled: bool, atm_window: int) -> None:
        try:
            self.option_minute_enabled = bool(enabled)
        except Exception:
            self.option_minute_enabled = False
        try:
            self.option_minute_atm_window = max(0, int(atm_window))
        except Exception:
            self.option_minute_atm_window = 2
        if not self.option_minute_enabled:
            self._option_minute_allowed_symbols = set()

    def update_option_minute_allowed_symbols(self, *, underlying_price: float, strike_gap: float = 2.5) -> None:
        if not self.option_minute_enabled:
            return
        try:
            upx = float(underlying_price or 0.0)
        except Exception:
            upx = 0.0
        if upx <= 0.0:
            return

        try:
            g = float(strike_gap or 2.5)
        except Exception:
            g = 2.5
        if g <= 0.0:
            g = 2.5

        # Round to nearest strike grid.
        atm = float(int(upx / g + 0.5) * g)

        strikes = set()
        for v in (self.call_options or {}).values():
            try:
                k = float(v.get("strike") or 0.0)
                if k > 0.0:
                    strikes.add(k)
            except Exception:
                continue
        for v in (self.put_options or {}).values():
            try:
                k = float(v.get("strike") or 0.0)
                if k > 0.0:
                    strikes.add(k)
            except Exception:
                continue
        if not strikes:
            return

        # Choose nearest available strike as ATM.
        atm_k = min(sorted(strikes), key=lambda x: abs(float(x) - atm))

        allowed_strikes = set(
            float(atm_k + i * g)
            for i in range(-int(self.option_minute_atm_window), int(self.option_minute_atm_window) + 1)
        )

        allowed: set[str] = set()
        for sym, v in (self.call_options or {}).items():
            try:
                if float(v.get("strike") or 0.0) in allowed_strikes:
                    allowed.add(str(sym))
            except Exception:
                continue
        for sym, v in (self.put_options or {}).items():
            try:
                if float(v.get("strike") or 0.0) in allowed_strikes:
                    allowed.add(str(sym))
            except Exception:
                continue

        self._option_minute_allowed_symbols = allowed

    def parse_option_code(self, symbol: str, *, now: Optional[datetime] = None) -> Dict[str, Any]:
        """
        옵션 코드 파싱
        
        Args:
            symbol: 옵션 심볼 (예: B0162503430, C0162502425)
            
        Returns:
            파싱된 정보 딕셔너리
            {
                'year': int,
                'month': int,
                'is_weekly': bool,
                'option_type': str,  # 'call' or 'put'
                'strike': float,
                'symbol': str
            }
            
        Raises:
            ValueError: 유효하지 않은 심볼
            
        Examples:
            >>> processor.parse_option_code("B0162503430")
            {'year': 2025, 'month': 3, 'option_type': 'call', 'strike': 430.0, ...}
            
            >>> processor.parse_option_code("C0162502425")
            {'year': 2025, 'month': 2, 'option_type': 'put', 'strike': 242.5, ...}
        """
        symbol_str = str(symbol).strip()
        
        # 길이 검증
        if not symbol_str or len(symbol_str) < 4:
            raise ValueError(f"Invalid symbol length: {symbol_str}")
        
        # 타입 식별 (B=콜, C=풋)
        first_char = symbol_str[0]
        if first_char == "B":
            option_type = "call"
        elif first_char == "C":
            option_type = "put"
        else:
            raise ValueError(f"Invalid option type: {first_char}")
        
        # 기초자산 확인 (016=KP200)
        underlying_code = symbol_str[1:4]
        if underlying_code != "016":
            raise ValueError(f"Unsupported underlying: {underlying_code}")

        code_suffix = symbol_str[4:]

        # 만기월/행사가 파싱
        now = now or datetime.now()
        year = now.year
        month = now.month
        strike_str = ""

        # 표준 형식: YYMM + STRIKE (예: 2503430)
        if len(code_suffix) >= 6:
            try:
                year_code = code_suffix[0:2]
                month_code = code_suffix[2:4]
                year = 2000 + int(year_code)
                month = int(month_code)
                strike_str = code_suffix[4:]

                if not (1 <= month <= 12):
                    self.logger.warning(f"Invalid month: {month} in {symbol_str}")
                    year = now.year
                    month = now.month
                    
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Expiry parsing failed: {symbol_str}, error: {e}")
                year = now.year
                month = now.month
                strike_str = code_suffix[4:] if len(code_suffix) >= 5 else ""

        # 단축 형식: M + STRIKE (예: 2430)
        elif len(code_suffix) == 4 and code_suffix.isdigit():
            try:
                month = int(code_suffix[0])
                strike_str = code_suffix[1:]
                if not (1 <= month <= 12):
                    month = now.month
            except Exception:
                month = now.month
                strike_str = ""

        # 행사가 변환 — strike_utils.extract_strike_pt()에 위임
        # 표준 형식(code_suffix[4:] = 3자리 코드), 알파벳 연장(A01~A11) 모두 처리
        strike = 0.0
        if strike_str:
            parsed = extract_strike_pt(strike_str)
            if parsed is not None:
                strike = parsed
            else:
                self.logger.warning("Strike parsing failed: symbol=%s, strike_str=%s", symbol_str, strike_str)

        return {
            "year": year,
            "month": month,
            "is_weekly": False,
            "option_type": option_type,
            "strike": strike,
            "symbol": symbol_str,
        }

    def process_futures_tick(self, tick_data: Dict) -> None:
        """
        선물 틱 데이터 처리 및 분봉 집계
        
        Args:
            tick_data: 선물 틱 데이터
                {
                    'trcode': 'FC0',
                    'symbol': 'A016XXXX',
                    'tick': {
                        'price': str,
                        'volume': str,
                        'chetime': str,  # HHMMSS
                        'k200jisu': str,
                        'openyak': str,
                        'bidho1': str,
                        'offerho1': str
                    }
                }
        
        Processing:
            1. 체결시간 파싱
            2. 틱 리스트에 추가
            3. 분봉 데이터에 추가
            4. 오래된 데이터 정리
        """
        tick = tick_data.get("tick_norm") or tick_data.get("tick") or {}
        ref = None
        try:
            ref = getattr(self, "_last_futures_timestamp", None)
        except Exception:
            ref = None
        timestamp = parse_chetime(str(tick.get("chetime") or ""), reference=ref)
        try:
            # Defensive: if timestamp jumps too far back, keep reference.
            if ref is not None and isinstance(ref, datetime):
                if (ref - timestamp) > timedelta(hours=18):
                    timestamp = ref.replace(microsecond=0)
        except Exception as _e:
            logger.debug("오류 무시: %s", _e)
        minute_key = timestamp.replace(second=0, microsecond=0)

        # FC0_OC0_SCHEMA.md:
        # - cvolume: 체결량(단건)
        # - volume: 누적거래량
        # 일부 환경/로깅에서는 volume만 제공될 수 있어 호환을 위해 fallback 처리
        cvol = safe_int(tick.get("cvolume"))
        cumvol = safe_int(tick.get("volume"))
        if cvol <= 0 and cumvol > 0:
            cvol = int(cumvol)

        # 틱 레코드 생성
        # 주의: tick의 open/high/low는 당일 누적값이므로 분봉 OHLC 계산에 사용하지 않음
        # 분봉 OHLC는 각 틱의 price에서 계산됨
        tick_record = {
            "timestamp": timestamp,
            "price": safe_float(tick.get("price")),
            # keep legacy key name `volume` as cumulative volume for minute-bar aggregation
            "volume": int(cumvol),
            "cvolume": int(cvol),
            # open/high/low는 당일 누적값 (분봉 OHLC 계산에 미사용)
            # 분봉 OHLC는 price 필드에서 계산됨
            "open": safe_float(tick.get("open")),
            "high": safe_float(tick.get("high")),
            "low": safe_float(tick.get("low")),
            # tick_norm: "k200jisu" / raw tick: "kospijisu" 또는 "k200jisu"
            "k200_index": (
                safe_float(tick.get("k200jisu"))
                or safe_float(tick.get("kospijisu"))
            ),
            # sbasis: 시장BASIS = KP200선물 - KP200현물 (FC0 OutBlock 직접 제공)
            # tick_norm 경로: "sbasis" / raw tick 경로: "sbasis"
            "sbasis": safe_float(tick.get("sbasis")),
            "open_interest": safe_int(tick.get("openyak")),
            "bid": safe_float(tick.get("bidho")) or safe_float(tick.get("bidho1")),
            "ask": safe_float(tick.get("offerho")) or safe_float(tick.get("offerho1")),
        }

        # 틱 추가 + 오래된 데이터 정리 (atomic)
        # maxlen=MAX_FUTURES_TICKS deque는 append 시 자동 popleft되므로
        # 별도 while-popleft 정리 루프는 불필요하다. (4-3 수정)
        with self._futures_lock:
            self.futures_ticks.append(tick_record)
        self.futures_minute_data[minute_key].append(tick_record)
        logger.debug("[TickProcessor] FC0 틱 추가: minute_key=%s, futures_minute_data[%s] 길이=%d", minute_key, minute_key, len(self.futures_minute_data[minute_key]))

        try:
            self._last_futures_timestamp = timestamp
        except Exception as _e:
            logger.debug("오류 무시: %s", _e)

        # [FIX-AMP-1] 당일 세션 OHLC 누적 갱신.
        # FC0 tick의 "high"/"low"/"open"은 eBest가 제공하는 당일 누적값이다.
        # FUTURES_MINUTE_RETENTION_HOURS 제한으로 분봉 데이터가 소실되더라도
        # 여기서 누적된 값은 장 내내 유지된다.
        try:
            _tick_date = timestamp.strftime("%Y%m%d")
            if _tick_date != self._daily_session_date:
                # 날짜 변경 → 당일 누적값 리셋
                self._daily_session_date = _tick_date
                self._daily_session_high  = 0.0
                self._daily_session_low   = 0.0
                self._daily_session_open  = 0.0

            _tick_high = float(tick_record.get("high") or 0.0)
            _tick_low  = float(tick_record.get("low")  or 0.0)
            _tick_open = float(tick_record.get("open") or 0.0)

            # open: FC0 스키마의 "open"은 당일 시가 — 첫 유효값만 기록
            if _tick_open > 0.0 and self._daily_session_open == 0.0:
                self._daily_session_open = _tick_open

            # high/low: eBest가 매 틱마다 당일 누적 고가/저가를 전달하므로 최신값 우선
            if _tick_high > 0.0:
                self._daily_session_high = max(self._daily_session_high, _tick_high)
            if _tick_low > 0.0:
                if self._daily_session_low == 0.0:
                    self._daily_session_low = _tick_low
                else:
                    self._daily_session_low = min(self._daily_session_low, _tick_low)
        except Exception as _e:
            logger.debug("오류 무시: %s", _e)

        # futures_minute_data 정리 (메모리 누수 방지)
        try:
            cutoff = timestamp - timedelta(hours=FUTURES_MINUTE_RETENTION_HOURS)
            cutoff_minute = cutoff.replace(second=0, microsecond=0)
            stale_keys = [k for k in self.futures_minute_data if k <= cutoff_minute]
            for k in stale_keys:
                del self.futures_minute_data[k]
        except Exception as e:
            self.logger.warning(f"Failed to clean minute_data: {e}")

    def process_spot_index_tick(self, tick_data: Dict) -> None:
        """IJ_ 등 KOSPI 현물 지수 틱을 분봉으로 누적한다.

        FC0와 동일하게 ``timestamp.replace(second=0, microsecond=0)`` 로 분 버킷을 맞춘다.

        [FIX] KOSPI 현물 정규 세션(09:00~15:30)에 해당하는 틱만 누적한다.
              KOSPI 현물은 09:00 개장이며, 08:45~08:59 구간 틱이 유입되면
              ZigZag 스윙 감지가 오염되므로 해당 구간은 수집 단계에서 차단한다.
        """
        tick = tick_data.get("tick_norm") or tick_data.get("tick") or {}
        ref: Optional[datetime] = None
        try:
            ref = getattr(self, "_last_spot_index_timestamp", None) or getattr(
                self, "_last_futures_timestamp", None
            )
        except Exception:
            ref = None
        timestamp = parse_ebest_tick_datetime(tick.get("time"), reference=ref)
        minute_key = timestamp.replace(second=0, microsecond=0)

        # [FIX] KOSPI 정규 세션 시간 필터: 09:00 ~ 15:30
        _hm = (minute_key.hour, minute_key.minute)
        if not ((9, 0) <= _hm <= (15, 30)):
            return

        px = safe_float(tick.get("jisu"))
        if px <= 0.0:
            return

        tick_record: Dict[str, Any] = {
            "timestamp": timestamp,
            "price": float(px),
            "volume": 0,
            "cvolume": 1,
            "open": 0.0,
            "high": 0.0,
            "low": 0.0,
            "k200_index": 0.0,
        }
        with self._kospi_lock:
            self.kospi_minute_data[minute_key].append(tick_record)
            self._last_kospi_timestamp = timestamp

        try:
            cutoff = timestamp - timedelta(hours=FUTURES_MINUTE_RETENTION_HOURS)
            cutoff_minute = cutoff.replace(second=0, microsecond=0)
            with self._kospi_lock:
                stale_keys = [k for k in self.kospi_minute_data if k <= cutoff_minute]
                for k in stale_keys:
                    del self.kospi_minute_data[k]
        except Exception as e:
            self.logger.warning("Failed to clean kospi_minute_data: %s", e)

    def process_option_tick(self, tick_data: Dict) -> None:
        """
        옵션 틱 데이터 처리
        
        Args:
            tick_data: 옵션 틱 데이터
                {
                    'trcode': 'OC0',
                    'symbol': 'B016XXXX' (콜) or 'C016XXXX' (풋),
                    'tick': {
                        'price': str,
                        'volume': str,
                        'openyak': str,
                        'impv': str,  # Implied Volatility (%)
                        'bidho1': str,
                        'offerho1': str,
                        'chetime': str
                    }
                }
        
        Processing:
            1. 심볼에서 콜/풋 구분
            2. 옵션 코드 파싱
            3. IV를 소수로 변환
            4. 딕셔너리에 저장
        """
        symbol = tick_data.get("symbol")
        tick = tick_data.get("tick_norm") or tick_data.get("tick") or {}
        raw_tick = tick_data.get("tick") or {}

        cvol = safe_int(tick.get("cvolume"))
        cumvol = safe_int(tick.get("volume"))
        if cvol <= 0 and cumvol > 0:
            cvol = int(cumvol)
        
        try:
            # 콜/풋 구분
            is_call = str(symbol).startswith("B")
            
            # 옵션 코드 파싱
            opt_info = self.parse_option_code(str(symbol))
            
        except ValueError as e:
            self.logger.error(f"Option code parsing failed: {e}")
            return

        # 옵션 데이터 생성
        processed = {
            "symbol": str(symbol),
            "optcode": str(tick.get("optcode") or symbol),
            "strike": float(opt_info.get("strike") or 0.0),
            "option_type": str(opt_info.get("option_type")),
            "price": safe_float(tick.get("price")),
            # keep legacy key name `volume` as cumulative volume (schema: 누적거래량)
            "volume": int(cumvol),
            "cvolume": int(cvol),
            "open_interest": safe_int(tick.get("openyak")),
            "iv": safe_float(tick.get("impv")) / 100.0 if tick.get("impv") is not None else 0.0,
            "theory_price": safe_float(tick.get("theoryprice")),
            # Greeks field names per schema: delt/gama/ceta/vega/rhox
            # tick_norm does not carry greeks; prefer raw tick keys.
            "delta": safe_float(raw_tick.get("delt")),
            "gamma": safe_float(raw_tick.get("gama")),
            "theta": safe_float(raw_tick.get("ceta")),
            "vega": safe_float(raw_tick.get("vega")),
            "rho": safe_float(raw_tick.get("rhox")),
            # Prefer tick_norm keys when present (bid1/ask1); fallback to raw aliases.
            "bid": safe_float(tick.get("bid1")) or safe_float(tick.get("bidho")) or safe_float(tick.get("bidho1")),
            "ask": safe_float(tick.get("ask1")) or safe_float(tick.get("offerho")) or safe_float(tick.get("offerho1")),
            "bid_depth": [safe_float(tick.get(f"bidho{i}")) for i in range(1, 6)],
            "ask_depth": [safe_float(tick.get(f"offerho{i}")) for i in range(1, 6)],
            "bid_qty_depth": [safe_float(tick.get(f"bidrem{i}")) for i in range(1, 6)],
            "ask_qty_depth": [safe_float(tick.get(f"offerrem{i}")) for i in range(1, 6)],
            "tot_bid_qty": safe_float(tick.get("totbidrem")),
            "tot_ask_qty": safe_float(tick.get("totofferrem")),
            "k200_index": safe_float(tick.get("kospijisu")) or safe_float(tick.get("k200jisu")),
            "timestamp": str(tick.get("chetime") or ""),
            # ── 가격 레벨 탐색용 당일 고가·저가 (OC0 스키마: high/low 문자열) ──
            "high": safe_float(tick.get("high")),
            "low":  safe_float(tick.get("low")),
        }

        # open_price: set_option_open_map()으로 캐시된 t2301 시가를 주입한다.
        # OC0 tick 스키마에는 "open" 필드가 없으므로 open_map 캐시 경로만 사용된다.
        with self._options_lock:
            _om = self._call_open_map if is_call else self._put_open_map
            _op = float(_om.get(str(symbol)) or 0.0)
            if _op > 0.0:
                processed["open_price"] = _op

            # 콜/풋 딕셔너리에 저장
            if is_call:
                self.call_options[str(symbol)] = processed
                self.call_option_ticks += 1
            else:
                self.put_options[str(symbol)] = processed
                self.put_option_ticks += 1

        # Optional: build option minute OHLCV for selected symbols (ATM ± N).
        if self.option_minute_enabled:
            try:
                sym = str(symbol)
            except Exception:
                sym = ""
            if sym and (not self._option_minute_allowed_symbols or sym in self._option_minute_allowed_symbols):
                try:
                    timestamp_dt = parse_chetime(str(tick.get("chetime") or ""))
                    minute_key = timestamp_dt.replace(second=0, microsecond=0)
                    tick_record = {
                        "timestamp": timestamp_dt,
                        "price": safe_float(tick.get("price")),
                        "volume": int(cumvol),
                        "cvolume": int(cvol),
                        "iv": safe_float(tick.get("impv")) / 100.0 if tick.get("impv") is not None else 0.0,
                    }
                    self.options_minute_data[sym][minute_key].append(tick_record)

                    cutoff = timestamp_dt - timedelta(hours=OPTION_MINUTE_RETENTION_HOURS)
                    cutoff_minute = cutoff.replace(second=0, microsecond=0)
                    try:
                        stale_keys = [k for k in self.options_minute_data[sym] if k <= cutoff_minute]
                        for k in stale_keys:
                            del self.options_minute_data[sym][k]
                    except Exception as _e:
                        logger.debug("오류 무시: %s", _e)

                    # Global sweep: symbols that stopped receiving ticks never get pruned by per-symbol cleanup.
                    # Sweep is intentionally lightweight and runs infrequently.
                    try:
                        self._options_minute_sweep_counter = int(self._options_minute_sweep_counter) + 1
                        do_sweep = False
                        if int(self._options_minute_sweep_counter) >= 500:
                            do_sweep = True
                        if float(time.time()) - float(self._options_minute_last_sweep_epoch) >= 60.0:
                            do_sweep = True

                        if do_sweep:
                            self._options_minute_sweep_counter = 0
                            self._options_minute_last_sweep_epoch = float(time.time())
                            for s in list(self.options_minute_data.keys()):
                                try:
                                    store = self.options_minute_data.get(s)
                                    if not store:
                                        del self.options_minute_data[s]
                                        continue
                                    latest_key = max(store.keys()) if store else None
                                    if latest_key is None or latest_key <= cutoff_minute:
                                        del self.options_minute_data[s]
                                except Exception:
                                    continue
                    except Exception as _e:
                        logger.debug("오류 무시: %s", _e)
                except Exception as _e:
                    logger.debug("오류 무시: %s", _e)

    def get_option_minute_df(self, symbol: str, minutes: Optional[int] = None) -> pd.DataFrame:
        if not symbol:
            return pd.DataFrame()
        sym = str(symbol)
        if minutes is None:
            minutes = int(getattr(self, "default_options_minutes", 120) or 120)
        store = self.options_minute_data.get(sym)
        if not store:
            return pd.DataFrame()

        minute_bars = []
        keys = sorted(store.keys())[-int(minutes or 0) :]
        prev_cum_volume: Optional[float] = None

        for minute_key in keys:
            ticks = store.get(minute_key) or []
            if not ticks:
                continue

            open_price = safe_float(ticks[0].get("price"))
            high_price = max(safe_float(t.get("price")) for t in ticks)
            low_price = min(safe_float(t.get("price")) for t in ticks)
            close_price = safe_float(ticks[-1].get("price"))

            vol_vals = [t.get("volume") for t in ticks if t.get("volume") is not None]
            cum_volume = max((safe_float(v) for v in vol_vals), default=0.0)
            if cum_volume <= 0.0:
                cvol_vals = [t.get("cvolume") for t in ticks if t.get("cvolume") is not None]
                cum_volume = float(sum((safe_float(v) for v in cvol_vals), start=0.0))

            if prev_cum_volume is None:
                volume = float(cum_volume)
            else:
                delta = float(cum_volume) - float(prev_cum_volume)
                volume = float(delta) if delta >= 0.0 else float(cum_volume)
            prev_cum_volume = float(cum_volume)

            minute_bars.append(
                {
                    "timestamp": minute_key,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "iv": safe_float(ticks[-1].get("iv", 0.0)),
                }
            )

        df = pd.DataFrame(minute_bars)
        try:
            df = normalize_ohlcv_columns(df)
        except Exception as _e:
            logger.debug("오류 무시: %s", _e)
        return df

    def process_option_quote_tick(self, tick_data: Dict) -> None:
        symbol = tick_data.get("symbol")
        tick = tick_data.get("tick_norm") or tick_data.get("tick") or {}

        try:
            is_call = str(symbol).startswith("B")
            opt_info = self.parse_option_code(str(symbol))
        except ValueError as e:
            self.logger.error(f"Option code parsing failed: {e}")
            return

        hotime = tick.get("hotime") or tick.get("chetime")
        ts_str = str(hotime or "")
        try:
            _ = parse_chetime(str(hotime or ""))
        except Exception as _e:
            logger.debug("[process_option_quote_tick] 오류 무시: %s", _e)

        bid_depth = tick.get("bidhos") if isinstance(tick.get("bidhos"), list) else [safe_float(tick.get(f"bidho{i}")) for i in range(1, 6)]
        ask_depth = tick.get("offerhos") if isinstance(tick.get("offerhos"), list) else [safe_float(tick.get(f"offerho{i}")) for i in range(1, 6)]
        bid_qty_depth = tick.get("bidrems") if isinstance(tick.get("bidrems"), list) else [safe_float(tick.get(f"bidrem{i}")) for i in range(1, 6)]
        ask_qty_depth = tick.get("offerrems") if isinstance(tick.get("offerrems"), list) else [safe_float(tick.get(f"offerrem{i}")) for i in range(1, 6)]

        bid = safe_float(tick.get("bid1"))
        ask = safe_float(tick.get("ask1"))
        if (bid or 0.0) <= 0.0 and bid_depth:
            try:
                bid = float(bid_depth[0] or 0.0)
            except Exception:
                bid = 0.0
        if (ask or 0.0) <= 0.0 and ask_depth:
            try:
                ask = float(ask_depth[0] or 0.0)
            except Exception:
                ask = 0.0

        quote_update: Dict[str, object] = {
            "symbol": str(symbol),
            "optcode": str(tick.get("optcode") or symbol),
            "strike": float(opt_info.get("strike") or 0.0),
            "option_type": str(opt_info.get("option_type")),
            "timestamp": ts_str,
        }
        if (bid or 0.0) > 0.0:
            quote_update["bid"] = float(bid)
        if (ask or 0.0) > 0.0:
            quote_update["ask"] = float(ask)
        if any((safe_float(v) or 0.0) > 0.0 for v in (bid_depth or [])):
            quote_update["bid_depth"] = [safe_float(v) for v in (bid_depth or [])][:5]
        if any((safe_float(v) or 0.0) > 0.0 for v in (ask_depth or [])):
            quote_update["ask_depth"] = [safe_float(v) for v in (ask_depth or [])][:5]
        if any((safe_float(v) or 0.0) > 0.0 for v in (bid_qty_depth or [])):
            quote_update["bid_qty_depth"] = [safe_float(v) for v in (bid_qty_depth or [])][:5]
        if any((safe_float(v) or 0.0) > 0.0 for v in (ask_qty_depth or [])):
            quote_update["ask_qty_depth"] = [safe_float(v) for v in (ask_qty_depth or [])][:5]

        tot_bid_qty = safe_float(tick.get("totbidrem"))
        tot_ask_qty = safe_float(tick.get("totofferrem"))
        if (tot_bid_qty or 0.0) > 0.0:
            quote_update["tot_bid_qty"] = float(tot_bid_qty)
        if (tot_ask_qty or 0.0) > 0.0:
            quote_update["tot_ask_qty"] = float(tot_ask_qty)

        store = self.call_options if is_call else self.put_options
        key = str(symbol)
        prev = store.get(key) or {}
        merged = dict(prev)
        merged.update(quote_update)
        store[key] = merged

    def process_tick(self, tick_data: Dict) -> None:
        """
        틱 데이터 자동 분기 처리
        
        Args:
            tick_data: 틱 데이터 (trcode에 따라 자동 분류)
        
        Routing:
            - FC0 → process_futures_tick()
            - OC0 → process_option_tick()
            - 그 외 → 무시
        
        Example:
            >>> for tick in stream:
            ...     processor.process_tick(tick)
        """
        trcode = tick_data.get("trcode")
        
        if trcode == TRCode.FUTURES.value:
            self.process_futures_tick(tick_data)
        elif trcode == TRCode.OPTIONS.value:
            self.process_option_tick(tick_data)
        elif trcode == TRCode.FUTURES_BOOK.value:
            # FO0(호가/오더북)은 PredictionPipeline에서 별도 버퍼링/feature 처리합니다.
            # tick_processor는 분봉/옵션 체결 스냅샷만 담당합니다.
            return
        elif getattr(TRCode, "OPTIONS_QUOTE", None) and trcode == TRCode.OPTIONS_QUOTE.value:
            self.process_option_quote_tick(tick_data)
            return

        # 백테스팅 데이터 저장 체크 (주기적으로)
        self._save_check_counter += 1
        if self._save_check_counter >= self._save_check_interval:
            self._save_check_counter = 0
            self.save_backtest_data_if_needed()

    def get_daily_session_ohlc(self) -> Dict[str, float]:
        """[FIX-AMP-1] 당일 세션 OHLC 반환.

        FC0 틱 수신 시마다 누적·갱신하므로 FUTURES_MINUTE_RETENTION_HOURS 제한과
        무관하게 장 내내 유효한 당일 시가/고가/저가를 제공한다.

        Returns:
            {"session_open": float, "session_high": float, "session_low": float}
            값이 없으면 0.0.
        """
        try:
            return {
                "session_open": float(self._daily_session_open or 0.0),
                "session_high": float(self._daily_session_high or 0.0),
                "session_low":  float(self._daily_session_low  or 0.0),
            }
        except Exception:
            return {"session_open": 0.0, "session_high": 0.0, "session_low": 0.0}

    async def fetch_futures_minute_from_api(
        self,
        upcode: str,
        date: Optional[str] = None,
        minutes: Optional[int] = None
    ) -> pd.DataFrame:
        """
        eBest API(t8415)에서 선물 분봉 데이터 직접 가져오기

        Args:
            upcode: 종목 코드 (예: KP200 선물 코드)
            date: 조회 날짜 (YYYYMMDD 형식, None인 경우 오늘)
            minutes: 가져올 분봉 개수

        Returns:
            OHLCV 분봉 DataFrame
        """
        if self.fetch_market_service is None:
            logger.warning("[TickProcessor] fetch_market_service가 설정되지 않음")
            return pd.DataFrame()

        # 캐시 초기화 (target_day 변경 시 새로운 데이터 가져오기)
        self.clear_minute_cache()

        # config에서 target_day 읽기
        config = load_config()
        target_day = config.get("prediction", {}).get("target_day", None)
        # 빈 문자열인 경우 None로 처리
        if target_day == "" or target_day is None:
            target_day = None

        if date is None:
            # target_day가 설정되어 있으면 해당 날짜 사용, 없으면 오늘 날짜 사용
            if target_day:
                date = target_day
            else:
                date = datetime.now().strftime("%Y%m%d")

        # 더미 뷰 객체 생성 (API 서비스에 필요)
        class DummyView:
            use_replay = False
            prev_target_day = target_day

        try:
            import asyncio
            # 비동기 실행
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 이벤트 루프가 실행 중인 경우
                future = asyncio.ensure_future(
                    self.fetch_market_service.fetch_market_data(
                        DummyView(),
                        "t8415",
                        upcode,
                        date,
                        timeframe=1
                    )
                )
                # 동기적으로 대기 (주의: 이벤트 루프 블로킹 가능)
                _, _, _, _, _, _, _, df = await asyncio.wait_for(future, timeout=10.0)
            else:
                # 새로운 이벤트 루프 생성
                _, _, _, _, _, _, _, df = await self.fetch_market_service.fetch_market_data(
                    DummyView(),
                    "t8415",
                    upcode,
                    date,
                    timeframe=1
                )

            # 최근 N분봉만 반환
            if minutes is not None and not df.empty:
                df = df.tail(minutes)

            return df
        except Exception as e:
            logger.error(f"[TickProcessor] API 분봉 가져오기 실패: {e}")
            return pd.DataFrame()

    async def fetch_kospi_minute_from_api(
        self,
        upcode: str,
        date: Optional[str] = None,
        minutes: Optional[int] = None
    ) -> pd.DataFrame:
        """
        eBest API(t8418)에서 KOSPI 지수 분봉 데이터 직접 가져오기

        Args:
            upcode: 종목 코드 (예: KOSPI 지수 코드)
            date: 조회 날짜 (YYYYMMDD 형식, None인 경우 오늘)
            minutes: 가져올 분봉 개수

        Returns:
            OHLCV 분봉 DataFrame
        """
        if self.fetch_market_service is None:
            logger.warning("[TickProcessor] fetch_market_service가 설정되지 않음")
            return pd.DataFrame()

        # 캐시 초기화 (target_day 변경 시 새로운 데이터 가져오기)
        self.clear_minute_cache()

        # config에서 target_day 읽기
        config = load_config()
        target_day = config.get("prediction", {}).get("target_day", None)
        # 빈 문자열인 경우 None로 처리
        if target_day == "" or target_day is None:
            target_day = None

        if date is None:
            # target_day가 설정되어 있으면 해당 날짜 사용, 없으면 오늘 날짜 사용
            if target_day:
                date = target_day
            else:
                date = datetime.now().strftime("%Y%m%d")

        # 더미 뷰 객체 생성
        class DummyView:
            use_replay = False
            prev_target_day = target_day

        try:
            import asyncio
            # 비동기 실행
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.ensure_future(
                    self.fetch_market_service.fetch_market_data(
                        DummyView(),
                        "t8418",
                        upcode,
                        date,
                        timeframe=1
                    )
                )
                _, _, _, _, _, _, _, df = await asyncio.wait_for(future, timeout=10.0)
            else:
                _, _, _, _, _, _, _, df = await self.fetch_market_service.fetch_market_data(
                    DummyView(),
                    "t8418",
                    upcode,
                    date,
                    timeframe=1
                )

            # 최근 N분봉만 반환
            if minutes is not None and not df.empty:
                df = df.tail(minutes)

            return df
        except Exception as e:
            logger.error(f"[TickProcessor] KOSPI API 분봉 가져오기 실패: {e}")
            return pd.DataFrame()

    def get_futures_minute_df(self, minutes: Optional[int] = None, use_api: bool = False, upcode: Optional[str] = None) -> pd.DataFrame:
        """
        선물 분봉 DataFrame 생성

        Args:
            minutes: 가져올 분봉 개수 (기본값: 120)
            use_api: True인 경우 eBest API(t8415)에서 직접 가져옴
            upcode: API 사용 시 종목 코드 (예: KP200 선물 코드)

        Returns:
            OHLCV 분봉 DataFrame

        Columns:
            - timestamp: 분봉 시각
            - Open: 시가
            - High: 고가
            - Low: 저가
            - Close: 종가
            - Volume: 거래량
            - k200_index: KP200 지수

        Note:
            - use_api=True인 경우 eBest API에서 직접 분봉 가져옴
            - use_api=False인 경우 틱 데이터 집계 사용 (기본값)
            - 최근 N분봉을 반환
            - eBest FC0 틱의 volume이 누적값인 경우: 분봉 거래량 = 해당 분의 최대 누적값
            - 데이터 없으면 빈 DataFrame

        Example:
            >>> df = processor.get_futures_minute_df(60)
            >>> print(df.tail())
            >>>
            >>> # 기술적 분석
            >>> df['MA5'] = df['Close'].rolling(5).mean()
            >>> df['returns'] = df['Close'].pct_change()
        """
        if minutes is None:
            minutes = int(getattr(self, "default_futures_minutes", 120) or 120)
        try:
            minutes = max(1, int(minutes))
        except Exception:
            minutes = int(getattr(self, "default_futures_minutes", 120) or 120)
            minutes = max(1, int(minutes))

        # API 사용 요청 시
        if use_api and self.fetch_market_service is not None and upcode:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 이미 실행 중인 루프에서는 동기 방식으로 처리 불가
                    logger.warning("[TickProcessor] 이벤트 루프 실행 중: API 사용 불가, 틱 집계 사용")
                else:
                    return loop.run_until_complete(
                        self.fetch_futures_minute_from_api(upcode, minutes=minutes)
                    )
            except Exception as e:
                logger.warning(f"[TickProcessor] API 호출 실패, 틱 집계 사용: {e}")

        # _futures_minute_df(t8415 초기 데이터)가 있으면 실시간 틱 집계와 병합
        if hasattr(self, "_futures_minute_df") and self._futures_minute_df is not None and not self._futures_minute_df.empty:
            base_df      = self._futures_minute_df   # [PERF-3] copy() 제거 — 읽기 전용 참조
            last_base_ts = base_df.index[-1]
            base_len     = len(base_df)

            if self.futures_minute_data:
                try:
                    new_keys = sorted(
                        k for k in self.futures_minute_data.keys()
                        if pd.Timestamp(k) > last_base_ts
                    )
                    logger.debug("[TickProcessor] new_keys 찾음: last_base_ts=%s, new_keys=%s (len=%d)", last_base_ts, new_keys, len(new_keys))
                    # [PERF-3] 캐시 키: base 봉 수 + 신규 키 수 + 현재 분봉 틱 수 + 현재 분(1분마다 캐시 무효화)
                    # 현재 분봉(incomplete bar)의 틱이 추가되어도 업데이트되도록 틱 수 포함
                    current_minute_ticks = 0
                    if new_keys:
                        current_minute_ticks = len(self.futures_minute_data.get(new_keys[-1], []))
                    now = datetime.now(tz=_KST)
                    current_minute_key = now.replace(second=0, microsecond=0)
                    cache_key: tuple = (base_len, len(new_keys), current_minute_ticks, current_minute_key)

                    if (self._merged_futures_df is not None
                            and cache_key == self._merged_futures_key):
                        # ── 캐시 히트: concat·sort 없이 이전 결과 재사용 ──
                        df = self._merged_futures_df
                    else:
                        # ── 캐시 미스: 신규 봉만 조립 후 concat ──
                        if new_keys:
                            new_bars = []
                            prev_cum: Optional[float] = None
                            for mk in new_keys:
                                ticks = self.futures_minute_data.get(mk) or []
                                if not ticks:
                                    continue
                                o = safe_float(ticks[0]["price"])
                                h = max(safe_float(t["price"]) for t in ticks)
                                l = min(safe_float(t["price"]) for t in ticks)
                                c = safe_float(ticks[-1]["price"])
                                vol_vals = [t.get("volume") for t in ticks if t.get("volume") is not None]
                                cum = max((safe_float(v) for v in vol_vals), default=0.0)
                                if cum <= 0.0:
                                    cum = float(sum(safe_float(t.get("cvolume")) for t in ticks))
                                if prev_cum is None:
                                    vol = cum
                                else:
                                    d = cum - prev_cum
                                    vol = d if d >= 0.0 else cum
                                prev_cum = cum
                                new_bars.append({"timestamp": mk,
                                    "Open": o, "High": h, "Low": l, "Close": c, "Volume": vol})
                            if new_bars:
                                new_df = pd.DataFrame(new_bars)
                                new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
                                new_df = new_df.set_index("timestamp")
                                new_df.index.name = None
                                merged = pd.concat([base_df, new_df])
                                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                                logger.debug("[TickProcessor] merged: base=%d + new=%d = %d bars",
                                             base_len, len(new_df), len(merged))
                                # 캐시 저장
                                self._merged_futures_df  = merged
                                self._merged_futures_key = cache_key
                                df = merged
                            else:
                                df = base_df
                        else:
                            df = base_df
                except Exception as _e:
                    logger.debug("[TickProcessor] 병합 실패, base만 반환: %s", _e)
                    df = base_df
            else:
                df = base_df

            logger.debug("[TickProcessor] _futures_minute_df total=%d, minutes=%s", len(df), str(minutes))
            if minutes is not None:
                if int(minutes) >= 9999:
                    return df
                df = df.tail(minutes)
            return df

        # _futures_minute_df 없으면 틱 집계만 사용
        logger.debug("[TickProcessor] _futures_minute_df not set, using tick aggregation")
        if not self.futures_minute_data:
            return pd.DataFrame()

        # 캐시 키: 총 분봉 수 + 현재 분(1분마다 캐시 무효화)
        now = datetime.now(tz=_KST)
        current_minute = now.replace(second=0, microsecond=0)
        total_keys = len([k for k in self.futures_minute_data.keys() if k < current_minute])
        cache_key: tuple = (total_keys, current_minute)

        if (self._merged_futures_df is not None
                and cache_key == self._merged_futures_key):
            # 캐시 히트
            df = self._merged_futures_df
            if minutes is not None and int(minutes) < 9999:
                df = df.tail(minutes)
            logger.debug("[TickProcessor] Tick aggregation 캐시 히트: total_keys=%d (current_minute=%s)", total_keys, current_minute)
            return df

        minute_bars = []
        # 현재 시간보다 1분 이전인 완성된 분봉만 사용
        
        # minutes가 9999 이상이면 전체 길이 사용 (장전체)
        if int(minutes) >= 9999:
            keys = sorted([k for k in self.futures_minute_data.keys() if k < current_minute])
            logger.debug("[TickProcessor] Tick aggregation 장전체: total keys=%d (current_minute=%s)", len(keys), current_minute)
        else:
            keys = sorted([k for k in self.futures_minute_data.keys() if k < current_minute])[-int(minutes):]
            logger.debug("[TickProcessor] Tick aggregation minutes=%d: keys=%d (current_minute=%s)", int(minutes), len(keys), current_minute)
        prev_cum_volume: Optional[float] = None
        
        for minute_key in keys:
            ticks = self.futures_minute_data.get(minute_key) or []
            if not ticks:
                continue

            # OHLC 계산: 해당 분봉의 틱 price에서 계산 (당일 누적값 아님)
            open_price = safe_float(ticks[0]["price"])
            high_price = max(safe_float(t["price"]) for t in ticks)
            low_price = min(safe_float(t["price"]) for t in ticks)
            close_price = safe_float(ticks[-1]["price"])
            
            # 거래량 (누적값의 최대 → 직전 분 대비 증분으로 변환)
            vol_vals = [t.get("volume") for t in ticks if t.get("volume") is not None]
            cum_volume = max((safe_float(v) for v in vol_vals), default=0.0)
            if cum_volume <= 0.0:
                # fallback when only trade volume is available
                cvol_vals = [t.get("cvolume") for t in ticks if t.get("cvolume") is not None]
                cum_volume = float(sum((safe_float(v) for v in cvol_vals), start=0.0))

            if prev_cum_volume is None:
                volume = float(cum_volume)
            else:
                delta = float(cum_volume) - float(prev_cum_volume)
                volume = float(delta) if delta >= 0.0 else float(cum_volume)

            prev_cum_volume = float(cum_volume)

            minute_bars.append({
                "timestamp": minute_key,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "k200_index": safe_float(ticks[-1].get("k200_index", 0.0)),
            })

        df = pd.DataFrame(minute_bars)
        try:
            df = normalize_ohlcv_columns(df)
        except Exception as _e:
            logger.debug("오류 무시: %s", _e)
        
        # 타임스탬프를 인덱스로 설정
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            df.index.name = None  # 인덱스 이름 제거
        
        # 캐시 저장
        if not df.empty:
            self._merged_futures_df = df
            self._merged_futures_key = cache_key
            logger.debug("[TickProcessor] Tick aggregation 캐시 저장: total_keys=%d (current_minute=%s)", total_keys, current_minute)
        
        return df

    def get_futures_ohlcv(self, minutes: Optional[int] = None) -> pd.DataFrame:
        """선물 OHLCV DataFrame 생성 (get_futures_minute_df 별칭)

        prediction_mixin.py와의 호환성을 위해 제공.
        """
        return self.get_futures_minute_df(minutes=minutes)

    def get_kospi_minute_df(self, minutes: Optional[int] = None, use_api: bool = False, upcode: Optional[str] = None) -> pd.DataFrame:
        """
        KOSPI 현물 지수 분봉 DataFrame.

        컬럼 스키마는 ``get_futures_minute_df`` 와 동일(Open/High/Low/Close/Volume 등)하여
        적응형 번들이 동일 코드로 처리할 수 있다.

        Args:
            minutes: 가져올 분봉 개수 (기본값: 120)
            use_api: True인 경우 eBest API(t8418)에서 직접 가져옴
            upcode: API 사용 시 종목 코드 (예: KOSPI 지수 코드)
        """
        if minutes is None:
            minutes = int(getattr(self, "default_futures_minutes", 120) or 120)
        try:
            minutes = max(1, int(minutes))
        except Exception:
            minutes = int(getattr(self, "default_futures_minutes", 120) or 120)
            minutes = max(1, int(minutes))

        # API 사용 요청 시
        if use_api and self.fetch_market_service is not None and upcode:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 이미 실행 중인 루프에서는 동기 방식으로 처리 불가
                    logger.warning("[TickProcessor] 이벤트 루프 실행 중: API 사용 불가, 틱 집계 사용")
                else:
                    return loop.run_until_complete(
                        self.fetch_kospi_minute_from_api(upcode, minutes=minutes)
                    )
            except Exception as e:
                logger.warning(f"[TickProcessor] KOSPI API 호출 실패, 틱 집계 사용: {e}")

        # _kospi_minute_df(t8418 초기 데이터)가 있으면 실시간 틱 집계와 병합
        if hasattr(self, "_kospi_minute_df") and self._kospi_minute_df is not None and not self._kospi_minute_df.empty:
            base_df      = self._kospi_minute_df   # [PERF-3] copy() 제거
            last_base_ts = base_df.index[-1]
            base_len     = len(base_df)

            with self._kospi_lock:
                live_data = dict(self.kospi_minute_data)
            if live_data:
                try:
                    new_keys = sorted(
                        k for k in live_data.keys()
                        if pd.Timestamp(k) > last_base_ts
                        and (9, 0) <= (k.hour, k.minute) <= (15, 30)
                    )
                    # [PERF-3] 캐시 키: base 봉 수 + 신규 키 수 + 현재 분봉 틱 수
                    # 현재 분봉(incomplete bar)의 틱이 추가되어도 업데이트되도록 틱 수 포함
                    current_minute_ticks = 0
                    if new_keys:
                        current_minute_ticks = len(live_data.get(new_keys[-1], []))
                    cache_key_s: tuple = (base_len, len(new_keys), current_minute_ticks)

                    if (self._merged_spot_df is not None
                            and cache_key_s == self._merged_spot_key):
                        # ── 캐시 히트 ──
                        df = self._merged_spot_df
                    else:
                        # ── 캐시 미스 ──
                        if new_keys:
                            new_bars = []
                            prev_cum_s: Optional[float] = None
                            for mk in new_keys:
                                ticks = live_data.get(mk) or []
                                if not ticks:
                                    continue
                                o = safe_float(ticks[0]["price"])
                                h = max(safe_float(t["price"]) for t in ticks)
                                l = min(safe_float(t["price"]) for t in ticks)
                                c = safe_float(ticks[-1]["price"])
                                vol_vals = [t.get("volume") for t in ticks if t.get("volume") is not None]
                                cum = max((safe_float(v) for v in vol_vals), default=0.0)
                                if cum <= 0.0:
                                    cum = float(sum(safe_float(t.get("cvolume")) for t in ticks))
                                if prev_cum_s is None:
                                    vol = cum
                                else:
                                    d = cum - prev_cum_s
                                    vol = d if d >= 0.0 else cum
                                prev_cum_s = cum
                                new_bars.append({"timestamp": mk,
                                    "Open": o, "High": h, "Low": l, "Close": c, "Volume": vol})
                            if new_bars:
                                new_df = pd.DataFrame(new_bars)
                                new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
                                new_df = new_df.set_index("timestamp")
                                new_df.index.name = None
                                merged_s = pd.concat([base_df, new_df])
                                merged_s = merged_s[~merged_s.index.duplicated(keep="last")].sort_index()
                                logger.debug("[TickProcessor] KOSPI merged: base=%d + new=%d = %d bars",
                                             base_len, len(new_df), len(merged_s))
                                self._merged_spot_df  = merged_s
                                self._merged_spot_key = cache_key_s
                                df = merged_s
                            else:
                                df = base_df
                        else:
                            df = base_df
                except Exception as _e:
                    logger.debug("[TickProcessor] KOSPI 병합 실패, base만 반환: %s", _e)
                    df = base_df
            else:
                df = base_df
                logger.debug("[TickProcessor] _kospi_minute_df total=%d, minutes=%s", len(df), str(minutes))
                if minutes is not None:
                    if int(minutes) >= 9999:
                        logger.debug("[TickProcessor] _kospi_minute_df total=%d, minutes=%s", len(df), str(minutes))
            
            # minutes 파라미터 처리
            if minutes is not None and int(minutes) < 9999:
                df = df.tail(int(minutes))
            return df
        
        # _kospi_minute_df 없으면 틱 집계만 사용
        logger.debug("[TickProcessor] _kospi_minute_df not set, using tick aggregation")
        with self._kospi_lock:
            if not self.kospi_minute_data:
                return pd.DataFrame()
            # [FIX] KOSPI 정규 세션(09:00~15:30) 버킷만 반환 (process_spot_index_tick에서
            #       이미 차단되지만, 만약 누락 틱이 유입된 경우를 대비한 이중 방어 필터.
            #       KP200 선물(08:45 개장)과 달리 KOSPI 현물은 09:00 개장이다.)
            _all_valid_keys = sorted(
                k for k in self.kospi_minute_data.keys()
                if (9, 0) <= (k.hour, k.minute) <= (15, 30)
            )
            # minutes가 9999 이상이면 전체 길이 사용 (장전체)
            if int(minutes) >= 9999:
                _valid_keys = _all_valid_keys
                logger.debug("[TickProcessor] KOSPI tick aggregation 장전체: total keys=%d", len(_valid_keys))
            else:
                _valid_keys = _all_valid_keys[-int(minutes):]
                logger.debug("[TickProcessor] KOSPI tick aggregation minutes=%d: keys=%d", int(minutes), len(_valid_keys))
            snapshot: Dict[datetime, List[Dict[str, Any]]] = {
                k: list(self.kospi_minute_data.get(k) or []) for k in _valid_keys
            }

        minute_bars: List[Dict[str, Any]] = []
        prev_cum_volume: Optional[float] = None

        for minute_key in _valid_keys:
            ticks = snapshot.get(minute_key) or []
            if not ticks:
                continue

            open_price = safe_float(ticks[0]["price"])
            high_price = max(safe_float(t["price"]) for t in ticks)
            low_price = min(safe_float(t["price"]) for t in ticks)
            close_price = safe_float(ticks[-1]["price"])

            vol_vals = [t.get("volume") for t in ticks if t.get("volume") is not None]
            cum_volume = max((safe_float(v) for v in vol_vals), default=0.0)
            if cum_volume <= 0.0:
                cvol_vals = [t.get("cvolume") for t in ticks if t.get("cvolume") is not None]
                cum_volume = float(sum((safe_float(v) for v in cvol_vals), start=0.0))

            if prev_cum_volume is None:
                volume = float(cum_volume)
            else:
                delta = float(cum_volume) - float(prev_cum_volume)
                volume = float(delta) if delta >= 0.0 else float(cum_volume)

            prev_cum_volume = float(cum_volume)

            minute_bars.append(
                {
                    "timestamp": minute_key,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "k200_index": 0.0,
                }
            )

        df = pd.DataFrame(minute_bars)
        try:
            df = normalize_ohlcv_columns(df)
        except Exception as _e:
            logger.debug("오류 무시: %s", _e)
        
        # 타임스탬프를 인덱스로 설정
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            df.index.name = None  # 인덱스 이름 제거
        
        return df

    def get_current_price(self) -> float:
        """
        현재 선물 가격 반환
        
        Returns:
            현재 가격 (데이터 없으면 0.0)
        
        Example:
            >>> price = processor.get_current_price()
            >>> if price > 0:
            ...     print(f"현재가: {price:.2f}")
        """
        with self._futures_lock:
            if not self.futures_ticks:
                return 0.0
            return safe_float(self.futures_ticks[-1].get("price"), 0.0)

    def get_latest_k200_index(self) -> float:
        """FC0 틱에 포함된 최신 KP200 현물지수 반환.

        FC0 틱의 ``k200jisu`` 필드는 선물 체결 시각마다 함께 수신되므로
        IJ_ 별도 구독 없이 실시간 KP200 현물지수를 얻을 수 있다.

        Returns:
            최신 KP200 현물지수. 틱 데이터가 없거나 필드가 0이면 0.0.

        Example:
            >>> k200 = processor.get_latest_k200_index()
            >>> if k200 > 0:
            ...     basis = futures_price - k200
        """
        with self._futures_lock:
            if not self.futures_ticks:
                return 0.0
            # 최신 틱부터 역순으로 유효한(>0) k200_index 탐색
            # (일부 틱에서 k200_index=0이 섞여 들어올 수 있으므로)
            for tick in reversed(self.futures_ticks):
                v = safe_float(tick.get("k200_index"), 0.0)
                if v > 0.0:
                    return v
        return 0.0

    def get_latest_spot_index_price(self) -> float:
        """KOSPI 현물 지수의 최신 가격 반환.

        IJ_ 틱에서 누적된 kospi_minute_data에서 최신 가격을 가져온다.

        Returns:
            최신 KOSPI 현물 지수. 데이터가 없으면 0.0.
        """
        with self._kospi_lock:
            if not self.kospi_minute_data:
                logger.warning("[TickProcessor] kospi_minute_data 비어있음")
                return 0.0
            # 최신 분 버킷 찾기
            latest_minute = max(self.kospi_minute_data.keys())
            ticks = self.kospi_minute_data.get(latest_minute, [])
            if not ticks:
                logger.warning("[TickProcessor] 최신 분 버킷 틱 없음: %s", latest_minute)
                return 0.0
            # 해당 분의 마지막 틱 가격 반환
            price = safe_float(ticks[-1].get("price"), 0.0)
            logger.debug("[TickProcessor] KOSPI 현재가: %s (분 버킷: %s, 틱 수: %d)", price, latest_minute, len(ticks))
            return price

    def get_latest_sbasis(self) -> Optional[float]:
        """FC0 틱에 포함된 최신 베이시스(선물-현물) 반환.

        FC0 틱의 ``sbasis`` 필드는 eBest가 직접 제공하는
        KP200선물 - KP200현물 값이다. 별도 계산 없이 그대로 사용한다.

        Returns:
            최신 sbasis 값. 틱 데이터가 없거나 필드가 None이면 None.

        Example:
            >>> basis = processor.get_latest_sbasis()
            >>> if basis is not None:
            ...     print(f"basis={basis:+.2f}")
        """
        with self._futures_lock:
            if not self.futures_ticks:
                return None
            # 최신 틱부터 역순으로 sbasis가 None이 아닌 값 탐색
            for tick in reversed(self.futures_ticks):
                v = safe_float(tick.get("sbasis"))
                if v is not None:
                    return float(v)
            return None

    def get_price_at(self, dt: datetime) -> Optional[float]:
        """get_price_at.

Args:
    dt:
"""
        t: Optional[datetime] = None
        if isinstance(dt, datetime):
            t = dt
        else:
            try:
                t = datetime.fromisoformat(str(dt))
            except Exception:
                t = None

        if t is None:
            return None
        if not self.futures_minute_data:
            return None

        minute_key = t.replace(second=0, microsecond=0)
        ticks = self.futures_minute_data.get(minute_key)
        if ticks:
            try:
                v = ticks[-1].get("price")
                return float(v) if v is not None else None
            except Exception:
                return None

        try:
            keys = sorted(self.futures_minute_data.keys())
            idx = bisect_right(keys, minute_key) - 1
            if idx < 0:
                return None
            k = keys[idx]
            ticks2 = self.futures_minute_data.get(k)
            if not ticks2:
                return None
            try:
                v2 = ticks2[-1].get("price")
                return float(v2) if v2 is not None else None
            except Exception:
                return None
        except Exception:
            return None

    def get_price_near(self, dt: datetime, *, tolerance_sec: float = 30.0) -> Optional[float]:
        t: Optional[datetime] = None
        if isinstance(dt, datetime):
            t = dt
        else:
            try:
                t = datetime.fromisoformat(str(dt))
            except Exception:
                t = None

        if t is None:
            return None

        try:
            tol = float(tolerance_sec)
        except Exception:
            tol = 0.0
        if tol <= 0.0:
            return None

        if not self.futures_ticks:
            return None

        best_price: Optional[float] = None
        best_diff: Optional[float] = None

        lo = t - timedelta(seconds=float(tol))
        hi = t + timedelta(seconds=float(tol))

        try:
            # 스냅샷 복사 후 순회 — Lock 보유 시간 최소화 (2-3 수정)
            with self._futures_lock:
                ticks_snapshot = list(self.futures_ticks)
            # Scan from newest to oldest; break once we're older than the low bound.
            for rec in reversed(ticks_snapshot):
                ts = rec.get("timestamp")
                if not isinstance(ts, datetime):
                    continue
                if ts < lo:
                    break
                if ts > hi:
                    continue
                try:
                    px = rec.get("price")
                    if px is None:
                        continue
                    px_f = float(px)
                except Exception:
                    continue

                diff = abs((ts - t).total_seconds())
                if best_diff is None or diff < float(best_diff):
                    best_diff = float(diff)
                    best_price = float(px_f)
                    if best_diff <= 0.0:
                        break
        except Exception:
            return best_price

        return best_price

    def update_oi_from_t2301(self, t2301_snapshot: Dict[str, Any]) -> int:
        """t2301 REST 스냅샷으로 call_options / put_options의 OI를 갱신한다.

        OC0 실시간 구독은 ATM ±N개 범위만 커버하지만, t2301은 전 행사가를 한 번에
        반환한다. 이 메서드를 장중 주기적으로 호출하면 ATM 이동과 무관하게 전 행사가
        OI가 항상 최신 상태로 유지된다.

        갱신 전략:
            1. t2301 oi_calls / oi_puts 리스트를 optcode(심볼) 기준으로 처리한다.
            2. 해당 심볼이 이미 call_options / put_options에 있으면 open_interest,
               iv, delta, gamma, theta, vega, bid, ask만 덮어쓴다 (체결가·거래량 등
               실시간 값은 OC0 틱 값을 보존한다).
            3. 해당 심볼이 없으면 (OC0 구독 범위 밖) 새 항목으로 삽입한다.
               이 경우 price=0으로 삽입되며, 이후 OC0 틱이 오면 덮어써진다.

        Args:
            t2301_snapshot: _ebest_fetch_t2301_snapshot() 반환 dict.
                            'oi_calls', 'oi_puts' 키를 포함해야 한다.

        Returns:
            갱신(삽입 포함)된 심볼 수.
        """
        if not isinstance(t2301_snapshot, dict):
            return 0

        oi_calls = t2301_snapshot.get("oi_calls")
        oi_puts  = t2301_snapshot.get("oi_puts")
        if not isinstance(oi_calls, list):
            oi_calls = []
        if not isinstance(oi_puts, list):
            oi_puts = []

        updated = 0


        # ── 콜/풋 옵션 갱신 (call_options/put_options 동시 접근 보호) ────────
        # process_option_tick(I/O 콜백)과 동시 실행될 수 있으므로 lock이 필요하다.
        with self._options_lock:
            # ── 콜 옵션 갱신 ────────────────────────────────────────────────────
            for row in oi_calls:
                if not isinstance(row, dict):
                    continue
                try:
                    k     = float(row.get("strike") or 0.0)
                    oi    = float(row.get("open_interest") or 0.0)
                    optcode = str(row.get("optcode") or "").strip()
                except Exception:
                    continue

                if k <= 0.0 or not optcode:
                    continue

                # optcode를 키로 매핑 (OC0 틱은 symbol=optcode로 저장됨)
                sym = optcode

                if sym in self.call_options:
                    # 기존 항목: OI·Greeks·IV·호가만 t2301 값으로 갱신
                    entry = self.call_options[sym]
                    if oi > 0.0:
                        entry["open_interest"] = int(oi)
                    _iv = float(row.get("iv") or 0.0)
                    if _iv > 0.0:
                        entry["iv"] = float(_iv)
                    for fld in ("delta", "gamma", "theta", "vega", "rho", "theory_price"):
                        v = row.get(fld)
                        if v is not None:
                            try:
                                entry[fld] = float(v)
                            except Exception as _e:
                                logger.debug("오류 무시: %s", _e)
                    for fld in ("bid", "ask"):
                        v = row.get(fld)
                        if v is not None:
                            try:
                                fv = float(v)
                                if fv > 0.0:
                                    entry[fld] = fv
                            except Exception as _e:
                                logger.debug("오류 무시: %s", _e)
                    # ── 가격 레벨 탐색용: t2301 당일 고가·저가 갱신 ──────────
                    # OC0 틱에서도 누적되지만, t2301은 전 행사가를 한 번에 커버하므로
                    # OC0 구독 범위 밖 종목도 갱신된다.
                    for fld in ("high", "low", "open"):
                        v = row.get(fld)
                        if v is not None:
                            try:
                                fv = float(v)
                                if fv > 0.0:
                                    entry[fld] = fv
                            except Exception as _e:
                                logger.debug("오류 무시: %s", _e)
                    entry["_oi_from_t2301"] = True   # 갱신 출처 마킹
                else:
                    # 신규 삽입: OC0 구독 범위 밖 행사가 — OI 분포 파악용
                    try:
                        opt_info = self.parse_option_code(sym)
                    except Exception:
                        opt_info = {"strike": k, "option_type": "call"}
                    self.call_options[sym] = {
                        "symbol": sym,
                        "optcode": sym,
                        "strike": float(k),
                        "option_type": "call",
                        "price": float(row.get("price") or 0.0),
                        "volume": int(float(row.get("volume") or 0.0)),
                        "cvolume": 0,
                        "open_interest": int(oi),
                        "iv": float(row.get("iv") or 0.0),
                        "delta": float(row.get("delta") or 0.0),
                        "gamma": float(row.get("gamma") or 0.0),
                        "theta": float(row.get("theta") or 0.0),
                        "vega": float(row.get("vega") or 0.0),
                        "rho": float(row.get("rho") or 0.0),
                        "theory_price": float(row.get("theory_price") or 0.0),
                        "bid": float(row.get("bid") or 0.0),
                        "ask": float(row.get("ask") or 0.0),
                        "bid_depth": [],
                        "ask_depth": [],
                        "bid_qty_depth": [],
                        "ask_qty_depth": [],
                        "tot_bid_qty": 0.0,
                        "tot_ask_qty": 0.0,
                        "k200_index": 0.0,
                        "timestamp": "",
                        "_oi_from_t2301": True,
                        "_oc0_received": False,  # OC0 틱 미수신 표시
                    }
                    # open_price: _call_open_map 캐시에서 소급 주입 (장중 리밸런싱 대응)
                    _op_new = float(self._call_open_map.get(sym) or 0.0)
                    if _op_new > 0.0:
                        self.call_options[sym]["open_price"] = _op_new
                updated += 1

            # ── 풋 옵션 갱신 ────────────────────────────────────────────────────
            for row in oi_puts:
                if not isinstance(row, dict):
                    continue
                try:
                    k     = float(row.get("strike") or 0.0)
                    oi    = float(row.get("open_interest") or 0.0)
                    optcode = str(row.get("optcode") or "").strip()
                except Exception:
                    continue

                if k <= 0.0 or not optcode:
                    continue

                sym = optcode

                if sym in self.put_options:
                    entry = self.put_options[sym]
                    if oi > 0.0:
                        entry["open_interest"] = int(oi)
                    _iv = float(row.get("iv") or 0.0)
                    if _iv > 0.0:
                        entry["iv"] = float(_iv)
                    for fld in ("delta", "gamma", "theta", "vega", "rho", "theory_price"):
                        v = row.get(fld)
                        if v is not None:
                            try:
                                entry[fld] = float(v)
                            except Exception as _e:
                                logger.debug("오류 무시: %s", _e)
                    for fld in ("bid", "ask"):
                        v = row.get(fld)
                        if v is not None:
                            try:
                                fv = float(v)
                                if fv > 0.0:
                                    entry[fld] = fv
                            except Exception as _e:
                                logger.debug("오류 무시: %s", _e)
                    # ── 가격 레벨 탐색용: t2301 당일 고가·저가 갱신 ──────────
                    for fld in ("high", "low", "open"):
                        v = row.get(fld)
                        if v is not None:
                            try:
                                fv = float(v)
                                if fv > 0.0:
                                    entry[fld] = fv
                            except Exception as _e:
                                logger.debug("오류 무시: %s", _e)
                    entry["_oi_from_t2301"] = True
                else:
                    try:
                        opt_info = self.parse_option_code(sym)
                    except Exception:
                        opt_info = {"strike": k, "option_type": "put"}
                    self.put_options[sym] = {
                        "symbol": sym,
                        "optcode": sym,
                        "strike": float(k),
                        "option_type": "put",
                        "price": float(row.get("price") or 0.0),
                        "volume": int(float(row.get("volume") or 0.0)),
                        "cvolume": 0,
                        "open_interest": int(oi),
                        "iv": float(row.get("iv") or 0.0),
                        "delta": float(row.get("delta") or 0.0),
                        "gamma": float(row.get("gamma") or 0.0),
                        "theta": float(row.get("theta") or 0.0),
                        "vega": float(row.get("vega") or 0.0),
                        "rho": float(row.get("rho") or 0.0),
                        "theory_price": float(row.get("theory_price") or 0.0),
                        "bid": float(row.get("bid") or 0.0),
                        "ask": float(row.get("ask") or 0.0),
                        "bid_depth": [],
                        "ask_depth": [],
                        "bid_qty_depth": [],
                        "ask_qty_depth": [],
                        "tot_bid_qty": 0.0,
                        "tot_ask_qty": 0.0,
                        "k200_index": 0.0,
                        "timestamp": "",
                        "_oi_from_t2301": True,
                        "_oc0_received": False,
                    }
                    # open_price: _put_open_map 캐시에서 소급 주입 (장중 리밸런싱 대응)
                    _op_new = float(self._put_open_map.get(sym) or 0.0)
                    if _op_new > 0.0:
                        self.put_options[sym]["open_price"] = _op_new
                updated += 1

            if updated > 0:
                self.logger.debug(
                    "[TP] update_oi_from_t2301: %d행 갱신 (call=%d put=%d → total call=%d put=%d)",
                    updated,
                    len(oi_calls),
                    len(oi_puts),
                    len(self.call_options),
                    len(self.put_options),
                )

        return updated

    def get_statistics(self) -> Dict[str, Any]:
        """
        처리 통계 반환
        
        Returns:
            통계 딕셔너리
        """
        try:
            with self._kospi_lock:
                _spot_m = len(self.kospi_minute_data)
        except Exception:
            _spot_m = 0
        return {
            "futures_ticks": len(self.futures_ticks),
            "futures_minutes": len(self.futures_minute_data),
            "spot_index_minutes": _spot_m,
            "call_options": len(self.call_options),
            "put_options": len(self.put_options),
            "call_option_ticks": self.call_option_ticks,
            "put_option_ticks": self.put_option_ticks,
        }

    def _get_backtest_data_saver(self):
        """백테스팅 데이터 저장 인스턴스 가져오기."""
        if self._backtest_data_saver is None:
            try:
                from data.backtest_data_saver import BacktestDataSaver
                self._backtest_data_saver = BacktestDataSaver()
            except Exception as e:
                self.logger.warning("[BacktestDataSaver] 데이터 저장 인스턴스 로드 실패: %s", e)
        return self._backtest_data_saver

    def save_backtest_data_if_needed(self) -> bool:
        """필요한 경우 백테스팅 데이터 저장.

        장마감 시 (15:35 이후) 당일 데이터를 자동으로 저장합니다.

        Returns:
            저장 여부
        """
        saver = self._get_backtest_data_saver()
        if saver is None:
            return False

        # 장마감 확인
        if not saver.is_market_closed():
            return False

        # 이미 저장된 날짜 확인
        today = datetime.now().date()
        if self._last_save_date == today:
            return False

        try:
            # KP200 선물 1분봉 저장
            df_futures = self.get_futures_minute_df(minutes=None)
            if df_futures is not None and not df_futures.empty:
                saved = saver.save_if_needed(
                    df_futures,
                    data_source="futures",
                    timeframe="1m",
                    data_date=today,
                )
                if saved:
                    self.logger.info("[BacktestDataSaver] KP200 선물 1분봉 저장 완료")

            # KOSPI 지수 1분봉 저장
            df_kospi = self.get_kospi_minute_df(minutes=None)
            if df_kospi is not None and not df_kospi.empty:
                saved = saver.save_if_needed(
                    df_kospi,
                    data_source="kospi",
                    timeframe="1m",
                    data_date=today,
                )
                if saved:
                    self.logger.info("[BacktestDataSaver] KOSPI 지수 1분봉 저장 완료")

            self._last_save_date = today
            return True

        except Exception as e:
            self.logger.error("[BacktestDataSaver] 데이터 저장 실패: %s", e)
            return False
