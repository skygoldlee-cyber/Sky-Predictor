"""
eBest OpenAPI t8415/t8418 분봉 데이터 수집 서비스

t8415: 선물/옵션 분봉 OHLCV 데이터 조회
t8418: KOSPI 지수 분봉 데이터 조회
"""
import asyncio
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)


class FetchMarketDataService:
    """eBest OpenAPI 분봉 데이터 수집 서비스"""

    def __init__(self, api_client=None):
        """
        Args:
            api_client: eBest API 클라이언트 인스턴스 (필요한 경우)
        """
        self.api_client = api_client

    async def fetch_market_data(
        self,
        view: Any,
        query_type: str,
        upcode: str,
        date: str,
        timeframe: int = 1,
        *,
        kosdaq_upcode: Optional[str] = None
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], pd.DataFrame]:
        """
        t8415/t8418 쿼리로 분봉 데이터 수집

        Args:
            view: 뷰 객체 (use_replay, prev_target_day 등 속성 접근용)
            query_type: "t8415" 또는 "t8418"
            upcode: 종목 코드
            date: 조회 날짜 (YYYYMMDD 형식)
            timeframe: 분봉 단위 (기본값 1분)
            kosdaq_upcode: 코스닥 종목 코드 (필요한 경우)

        Returns:
            (prev_high, prev_low, prev_close, pp, r1, r2, s1, df)
            - prev_high: 전일 고가
            - prev_low: 전일 저가
            - prev_close: 전일 종가
            - pp: Pivot Point
            - r1: 저항선 1
            - r2: 저항선 2
            - s1: 지지선 1
            - df: OHLCV DataFrame
        """
        try:
            # 리플레이 모드 옵션 데이터 스킵
            if query_type == "t8415" and bool(getattr(view, "use_replay", False)):
                if self._is_option_symbol(upcode):
                    logger.debug(f"[FetchMarketData] 리플레이 모드: 옵션 분봉 스킵 {upcode}")
                    return (None, None, None, None, None, None, None, pd.DataFrame())

            # 장 시작 전 옵션 데이터 스킵
            if query_type == "t8415" and (not bool(getattr(view, "use_replay", False))):
                if self._is_option_symbol(upcode):
                    today = datetime.now().strftime("%Y%m%d")
                    now_time = datetime.now().time()
                    market_open = time(8, 45)  # 선물 개장 시간
                    if date == today and now_time < market_open:
                        logger.debug(f"[FetchMarketData] 장 시작 전: 옵션 분봉 스킵 {upcode}")
                        return (None, None, None, None, None, None, None, pd.DataFrame())

            # 전일 데이터 처리 (60분봉)
            if hasattr(view, 'prev_target_day') and date == str(getattr(view, "prev_target_day", "")):
                timeframe = 60

            # API 요청 파라미터 구성
            inputs = {
                f"{query_type}InBlock": {
                    "shcode": upcode,
                    "ncnt": timeframe,
                    "qrycnt": 1,
                    "nday": "",
                    "sdate": date,
                    "stime": "",
                    "edate": date,
                    "etime": "",
                    "cts_date": "",
                    "cts_time": "",
                    "comp_yn": "N",
                },
            }

            # API 요청 (실제 구현 필요)
            # 여기서는 더미 구현 - 실제 eBest API 연결 필요
            df = await self._mock_api_request(query_type, inputs)

            # 전일 데이터 추출 (OutBlock)
            prev_high, prev_low, prev_close, current_open = self._extract_prev_data(df)

            # Pivot 포인트 계산
            pp, r1, r2, s1 = self._calculate_pivot(
                prev_high, prev_low, prev_close, current_open, upcode
            )

            return (prev_high, prev_low, prev_close, pp, r1, r2, s1, df)

        except Exception as e:
            logger.error(f"[FetchMarketData] 데이터 수집 실패: {e}")
            return (None, None, None, None, None, None, None, pd.DataFrame())

    def _is_option_symbol(self, symbol: str) -> bool:
        """옵션 종목인지 확인"""
        call_prefixes = ["101", "102", "103", "104", "105"]
        put_prefixes = ["201", "202", "203", "204", "205"]
        return (any(symbol.startswith(p) for p in call_prefixes) or
                any(symbol.startswith(p) for p in put_prefixes))

    def _extract_prev_data(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """전일 데이터 추출"""
        if df.empty:
            return (None, None, None, None)

        # OutBlock 데이터가 있는 경우 추출 (더미 구현)
        # 실제 API 응답에서 OutBlock을 파싱해야 함
        prev_high = df.get("prev_high")
        prev_low = df.get("prev_low")
        prev_close = df.get("prev_close")
        current_open = df.get("current_open")

        return (prev_high, prev_low, prev_close, current_open)

    def _calculate_pivot(
        self,
        prev_high: Optional[float],
        prev_low: Optional[float],
        prev_close: Optional[float],
        current_open: Optional[float],
        upcode: str
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Pivot 포인트 계산"""
        if None in [prev_high, prev_low, prev_close, current_open]:
            return (None, None, None, None)

        try:
            # 표준 Pivot 계산
            pp = (prev_high + prev_low + prev_close) / 3
            r1 = 2 * pp - prev_low
            r2 = pp + (prev_high - prev_low)
            s1 = 2 * pp - prev_high
            return (pp, r1, r2, s1)
        except Exception as e:
            logger.debug(f"[FetchMarketData] Pivot 계산 실패: {e}")
            return (None, None, None, None)

    async def _mock_api_request(self, query_type: str, inputs: Dict[str, Any]) -> pd.DataFrame:
        """
        더미 API 요청 (실제 eBest API 연결 필요)

        실제 구현에서는 여기서 eBest API를 호출하고 응답을 파싱해야 합니다.
        """
        # 임시: 빈 DataFrame 반환
        # 실제 구현 시 eBest API 호출 및 응답 파싱 필요
        logger.warning(f"[FetchMarketData] 실제 eBest API 연결 필요: {query_type}")
        return pd.DataFrame()

    def parse_t8415_response(self, outblock: Dict[str, Any], outblock1: list) -> pd.DataFrame:
        """
        t8415 응답 파싱

        Args:
            outblock: OutBlock (전일 데이터)
            outblock1: OutBlock1 (분봉 데이터 리스트)

        Returns:
            OHLCV DataFrame
        """
        if not outblock1:
            return pd.DataFrame()

        rows = []
        for item in outblock1:
            try:
                dt_str = f"{item.get('date', '')} {item.get('time', '')}"
                dt = pd.to_datetime(dt_str, format="%Y%m%d %H%M%S", errors="coerce")

                row = {
                    "Datetime": dt,
                    "Open": float(item.get("open", 0) or 0),
                    "High": float(item.get("high", 0) or 0),
                    "Low": float(item.get("low", 0) or 0),
                    "Close": float(item.get("close", 0) or 0),
                    "Volume": float(item.get("jdiff_vol", 0) or 0),
                }

                # 옵션인 경우 추가 필드
                if "openyak" in item:
                    row["OpenInterest"] = float(item.get("openyak", 0) or 0)
                    row["OIChange"] = float(item.get("openyakcha", 0) or 0)

                rows.append(row)
            except Exception as e:
                logger.debug(f"[FetchMarketData] 파싱 실패: {e}")
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # 컬럼명 변경
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "jdiff_vol": "Volume",
        })

        # Datetime을 인덱스로 설정
        if "Datetime" in df.columns:
            df = df.set_index("Datetime")
            df.index.name = None

        # RangePct 계산
        if "Open" in df.columns and df["Open"].notna().any():
            open_val = df["Open"].dropna().iloc[0]
            if open_val > 0:
                df["RangePct"] = ((df["High"].cummax() - df["Low"].cummin()) / open_val) * 100

        return df

    def parse_t8418_response(self, outblock: Dict[str, Any], outblock1: list) -> pd.DataFrame:
        """
        t8418 응답 파싱 (KOSPI 지수)

        Args:
            outblock: OutBlock (전일 데이터)
            outblock1: OutBlock1 (분봉 데이터 리스트)

        Returns:
            OHLCV DataFrame
        """
        # t8418은 t8415와 유사한 구조
        return self.parse_t8415_response(outblock, outblock1)
