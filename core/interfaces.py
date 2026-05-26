"""core/interfaces.py — 패키지 간 의존성 역전을 위한 Protocol 정의.

역할
----
- 레이어 간 직접 import 대신 Protocol을 통해 느슨한 결합을 유지한다.
- 구현체(RealTimeTickProcessor 등)가 Protocol을 명시적으로 상속할 필요 없음
  (structural subtyping — duck typing).

사용 예
-------
    # prediction/pipeline.py
    from core.interfaces import TickDataProvider

    class PredictionPipeline:
        def __init__(self, tick_provider: TickDataProvider): ...
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    from typing import Protocol, runtime_checkable
except ImportError:          # Python 3.7 fallback
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class TickDataProvider(Protocol):
    """실시간 틱 데이터 공급자 인터페이스.

    RealTimeTickProcessor 가 이 Protocol을 구조적으로 충족한다.
    prediction / gui 레이어는 이 Protocol만 의존하고
    data.tick_processor 를 직접 import하지 않는다.
    """

    # ── 분봉 OHLCV ───────────────────────────────────────────────────────────
    def get_futures_minute_df(self, minutes: Optional[int] = None) -> "pd.DataFrame":
        """KP200 선물 분봉 DataFrame 반환."""
        ...

    def get_kospi_minute_df(self, minutes: Optional[int] = None) -> "pd.DataFrame":
        """KOSPI 현물지수 분봉 DataFrame 반환."""
        ...

    def get_daily_session_ohlc(self) -> Dict[str, float]:
        """당일 세션 OHLC (session_open / session_high / session_low) 반환."""
        ...

    # ── 현재가 / 지수 ────────────────────────────────────────────────────────
    def get_current_price(self) -> float:
        """KP200 선물 현재가 반환."""
        ...

    def get_latest_k200_index(self) -> float:
        """KP200 지수 현재값 반환."""
        ...

    def get_latest_sbasis(self) -> Optional[float]:
        """최근 베이시스(선물-현물) 반환."""
        ...

    # ── 틱 처리 ─────────────────────────────────────────────────────────────
    def process_tick(self, tick_data: Dict[str, Any]) -> None:
        """범용 틱 처리 (TR 코드 자동 분기)."""
        ...

    # ── 옵션 데이터 ──────────────────────────────────────────────────────────
    def update_option_minute_allowed_symbols(
        self, *, underlying_price: float, strike_gap: float = 2.5
    ) -> None:
        """ATM ± N 범위 기준 옵션 분봉 허용 심볼 갱신."""
        ...

    def update_oi_from_t2301(self, t2301_snapshot: Dict[str, Any]) -> int:
        """t2301 스냅샷에서 OI 데이터 갱신. 갱신 심볼 수 반환."""
        ...

    def set_option_open_map(
        self,
        call_open_map: Dict[str, float],
        put_open_map: Dict[str, float],
    ) -> None:
        """옵션 시가 맵 설정."""
        ...

    # ── 이력 조회 ────────────────────────────────────────────────────────────
    def get_price_at(self, dt: Any) -> Optional[float]:
        """특정 시각의 선물 가격 반환."""
        ...

    def get_price_near(
        self, dt: Any, *, tolerance_sec: float = 30.0
    ) -> Optional[float]:
        """특정 시각 근방의 선물 가격 반환."""
        ...

    def get_option_minute_df(
        self, symbol: str, minutes: Optional[int] = None
    ) -> "pd.DataFrame":
        """특정 옵션 심볼의 분봉 DataFrame 반환."""
        ...

    # ── 설정 ────────────────────────────────────────────────────────────────
    def configure_option_minute_ohlcv(
        self, *, enabled: bool, atm_window: int
    ) -> None:
        """옵션 분봉 OHLCV 수집 설정."""
        ...

    # ── 통계/속성 ────────────────────────────────────────────────────────────
    def get_statistics(self) -> Dict[str, Any]:
        """처리 통계 반환."""
        ...
