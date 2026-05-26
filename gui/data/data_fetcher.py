"""데이터 페처 모듈"""

import logging
from typing import Optional, Any, Dict, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class DataFetcher:
    """데이터 페처 - predictor/tick_processor에서 데이터를 가져옴."""
    
    def __init__(
        self,
        predictor: Optional[Any] = None,
        config: Optional[Any] = None,
        selected_plot: str = "futures",
        minutes: int = 120,
        use_api: bool = False,
        kp200_upcode: str = "",
        kospi_upcode: str = "",
        csv_file_path: Optional[str] = None
    ):
        """
        Args:
            predictor: KP200HybridPredictor 또는 PredictionPipeline 인스턴스
            config: AppConfig
            selected_plot: 선택된 플롯 타입 ("futures" 또는 "kospi")
            minutes: 가져올 분봉 수
            use_api: API 사용 여부
            kp200_upcode: KP200 종목 코드
            kospi_upcode: KOSPI 종목 코드
            csv_file_path: CSV 백테스트 파일 경로
        """
        self._predictor = predictor
        self._config = config
        self._selected_plot = selected_plot
        self._minutes = minutes
        self._use_api = use_api
        self._kp200_upcode = kp200_upcode
        self._kospi_upcode = kospi_upcode
        self._csv_file_path = csv_file_path
        self._csv_df = None
    
    def set_predictor(self, predictor: Any) -> None:
        """predictor 설정."""
        self._predictor = predictor
    
    def set_config(self, config: Any) -> None:
        """config 설정."""
        self._config = config
    
    def set_selected_plot(self, selected_plot: str) -> None:
        """선택된 플롯 타입 설정."""
        self._selected_plot = selected_plot
    
    def set_minutes(self, minutes: int) -> None:
        """가져올 분봉 수 설정."""
        self._minutes = minutes
    
    def set_csv_file_path(self, csv_file_path: Optional[str]) -> None:
        """CSV 파일 경로 설정."""
        self._csv_file_path = csv_file_path
        self._csv_df = None  # 경로 변경 시 캐시 초기화
    
    def fetch(self) -> pd.DataFrame:
        """데이터 가져오기.
        
        Returns:
            데이터프레임
        """
        import time as _time
        _fetch_t0 = _time.perf_counter()
        logger.info(
            "[DataFetcher][RT] fetch 시작 (csv=%s predictor=%s plot=%s minutes=%s)",
            self._csv_file_path is not None,
            self._predictor is not None,
            getattr(self, "_selected_plot", "?"),
            getattr(self, "_minutes", "?"),
        )
        
        # CSV 백테스트 모드
        if self._csv_file_path:
            logger.info("[DataFetcher] CSV 백테스트 모드 - 파일 로드 시도")
            df = self._fetch_from_csv()
            if df is not None and not df.empty:
                logger.info("[DataFetcher] CSV 데이터 로드 성공: %d 봉", len(df))
                return df
            else:
                logger.warning("[DataFetcher] CSV 데이터 로드 실패")
        
        # predictor에서 직접 가져오기
        logger.info("[DataFetcher] predictor에서 데이터 가져오기 시도")
        df = self._fetch_from_predictor()
        if df is not None:
            _fetch_elapsed = _time.perf_counter() - _fetch_t0
            logger.info(
                "[DataFetcher][RT] predictor 데이터 성공: bars=%d range=[%s~%s] elapsed=%.3fs",
                len(df),
                df.index[0] if len(df) else "N/A",
                df.index[-1] if len(df) else "N/A",
                _fetch_elapsed,
            )
            return df
        
        # tick_processor에서 가져오기
        logger.info("[DataFetcher] tick_processor에서 데이터 가져오기 시도")
        tp = self._get_tick_processor()
        if tp is not None:
            logger.info("[DataFetcher] tick_processor 찾음")
            df = self._fetch_from_tick_processor(tp)
            if df is not None:
                _fetch_elapsed = _time.perf_counter() - _fetch_t0
                logger.info(
                    "[DataFetcher][RT] tick_processor 데이터 성공: bars=%d range=[%s~%s] elapsed=%.3fs",
                    len(df),
                    df.index[0] if len(df) else "N/A",
                    df.index[-1] if len(df) else "N/A",
                    _fetch_elapsed,
                )
                return df
        else:
            logger.warning("[DataFetcher] tick_processor 없음")
        
        _fetch_elapsed = _time.perf_counter() - _fetch_t0
        logger.error(
            "[DataFetcher][RT] 모든 데이터 소스 실패 — 빈 DataFrame 반환 elapsed=%.3fs "
            "(csv=%s predictor=%s)",
            _fetch_elapsed,
            self._csv_file_path is not None,
            self._predictor is not None,
        )
        return pd.DataFrame()
    
    def _get_tick_processor(self) -> Optional[Any]:
        """predictor에서 tick_processor 찾기."""
        if self._predictor is None:
            return None
        
        if hasattr(self._predictor, "tick_processor"):
            logger.debug("[DataFetcher] Using predictor.tick_processor")
            return self._predictor.tick_processor
        elif hasattr(self._predictor, "pipeline") and hasattr(
            self._predictor.pipeline, "tick_processor"
        ):
            logger.debug("[DataFetcher] Using predictor.pipeline.tick_processor")
            return self._predictor.pipeline.tick_processor
        
        return None
    
    def _fetch_from_csv(self) -> Optional[pd.DataFrame]:
        """CSV 파일에서 데이터 가져오기."""
        if self._csv_file_path is None:
            return None
        
        # 캐시된 데이터가 있으면 재사용
        if self._csv_df is not None:
            return self._csv_df
        
        try:
            logger.info("[DataFetcher] CSV 파일 로드: %s", self._csv_file_path)
            df = pd.read_csv(self._csv_file_path)
            
            # 컬럼 이름 소문자로 변환 (대소문자 구분 없이 처리)
            df.columns = [col.lower() for col in df.columns]
            
            # 필수 컬럼 확인 (대소문자 구분 없이)
            required_cols = ['time', 'datetime', 'open', 'high', 'low', 'close']
            # time 또는 datetime 중 하나가 있어야 함
            time_col = 'time' if 'time' in df.columns else ('datetime' if 'datetime' in df.columns else None)
            if time_col is None:
                logger.warning("[DataFetcher] CSV 파일에 time/datetime 컬럼 없음")
                return None
            
            ohlc_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in ohlc_cols):
                logger.warning("[DataFetcher] CSV 파일에 필수 OHLC 컬럼 없음: %s", ohlc_cols)
                return None
            
            # time 컬럼 이름을 time으로 통일
            if time_col != 'time':
                df['time'] = df[time_col]
            
            # time 컬럼을 datetime으로 변환
            df['time'] = pd.to_datetime(df['time'])
            
            # time 컬럼을 인덱스로 설정 (ChartEngine에서 필요)
            df = df.set_index('time')
            
            # 데이터 소스 감지 (파일명 기반)
            file_lower = self._csv_file_path.lower()
            if 'kospi' in file_lower:
                self._selected_plot = 'kospi'
                logger.info("[DataFetcher] CSV 데이터 소스 감지: KOSPI")
            elif 'futures' in file_lower or 'kp200' in file_lower:
                self._selected_plot = 'futures'
                logger.info("[DataFetcher] CSV 데이터 소스 감지: Futures")
            else:
                # 가격 범위로 판단 (KOSPI는 보통 2000-3000, Futures는 1000-1500)
                price_range = df['close'].max() - df['close'].min()
                avg_price = df['close'].mean()
                if avg_price > 2000:
                    self._selected_plot = 'kospi'
                    logger.info("[DataFetcher] CSV 데이터 소스 감지 (가격 기반): KOSPI (avg=%.2f)", avg_price)
                else:
                    self._selected_plot = 'futures'
                    logger.info("[DataFetcher] CSV 데이터 소스 감지 (가격 기반): Futures (avg=%.2f)", avg_price)
            
            # 분봉 수 제한
            if self._minutes > 0 and len(df) > self._minutes:
                df = df.tail(self._minutes).copy()
            
            self._csv_df = df
            logger.info("[DataFetcher] CSV 로드 완료: %d 봉 (source=%s)", len(df), self._selected_plot)
            return df
        except Exception as e:
            logger.error("[DataFetcher] CSV 로드 실패: %s", e)
            return None
    
    def _fetch_from_predictor(self) -> Optional[pd.DataFrame]:
        """predictor에서 직접 데이터 가져오기."""
        if self._selected_plot == "kospi":
            if hasattr(self._predictor, "get_kospi_minute_df"):
                logger.debug("[DataFetcher] Calling predictor.get_kospi_minute_df: minutes=%s", str(self._minutes))
                upcode = self._get_upcode("kospi")
                df = self._predictor.get_kospi_minute_df(
                    minutes=self._minutes,
                    use_api=self._use_api,
                    upcode=upcode if self._use_api else None
                )
                logger.debug("[DataFetcher] get_kospi_minute_df returned: %d bars", len(df) if df is not None else 0)
                return df
            return None

        # futures
        if hasattr(self._predictor, "get_futures_minute_df"):
            upcode = self._get_upcode("futures")
            logger.debug("[DataFetcher] Calling predictor.get_futures_minute_df: minutes=%s", str(self._minutes))
            df = self._predictor.get_futures_minute_df(
                minutes=self._minutes,
                use_api=self._use_api,
                upcode=upcode if self._use_api else None
            )
            logger.debug("[DataFetcher] get_futures_minute_df returned: %d bars", len(df) if df is not None else 0)
            return df
        return None
    
    def _fetch_from_tick_processor(self, tp: Any) -> Optional[pd.DataFrame]:
        """tick_processor에서 데이터 가져오기."""
        logger.info("[DataFetcher] _fetch_from_tick_processor 호출 (selected_plot=%s)", self._selected_plot)
        
        if self._selected_plot == "kospi" and hasattr(tp, "get_kospi_minute_df"):
            upcode = self._get_upcode("kospi")
            logger.info("[DataFetcher] tp.get_kospi_minute_df 호출: minutes=%s, upcode=%s", self._minutes, upcode)
            df = tp.get_kospi_minute_df(
                minutes=self._minutes,
                use_api=self._use_api,
                upcode=upcode if self._use_api else None
            )
            logger.info("[DataFetcher] tp.get_kospi_minute_df 반환: %d bars", len(df) if df is not None else 0)
            if df is not None and not df.empty:
                logger.info("[DataFetcher] KOSPI 데이터 수신 성공: 마지막 타임스탬프=%s", df.index[-1])
                return df
            else:
                logger.warning("[DataFetcher] KOSPI 데이터 없음")
        elif hasattr(tp, "get_futures_minute_df"):
            upcode = self._get_upcode("futures")
            logger.info("[DataFetcher] tp.get_futures_minute_df 호출: minutes=%s, upcode=%s", self._minutes, upcode)
            df = tp.get_futures_minute_df(
                minutes=self._minutes,
                use_api=self._use_api,
                upcode=upcode if self._use_api else None
            )
            logger.info("[DataFetcher] tp.get_futures_minute_df 반환: %d bars", len(df) if df is not None else 0)
            if df is not None and not df.empty:
                logger.info("[DataFetcher] Futures 데이터 수신 성공: 마지막 타임스탬프=%s", df.index[-1])
                return df
            else:
                logger.warning("[DataFetcher] Futures 데이터 없음")
        else:
            logger.warning("[DataFetcher] tick_processor에 해당 메서드 없음")
        
        return None
    
    def _get_upcode_from_config(self, plot_type: str) -> Optional[str]:
        """config에서 종목 코드 가져오기."""
        if not self._config:
            return None
        
        try:
            if plot_type == "futures":
                upcode = self._config.get("kp200_upcode") or self._config.get("ebest", {}).get("kp200_upcode")
                return upcode
            if plot_type == "kospi":
                upcode = self._config.get("kospi_upcode") or self._config.get("ebest", {}).get("kospi_upcode")
                return upcode
        except Exception:
            pass
        
        return None
    
    def _get_upcode_from_predictor(self, plot_type: str) -> Optional[str]:
        """predictor에서 종목 코드 가져오기."""
        if not self._predictor:
            return None
        
        try:
            if hasattr(self._predictor, "kp200_symbol") and plot_type == "futures":
                return self._predictor.kp200_symbol
            if hasattr(self._predictor, "kospi_symbol") and plot_type == "kospi":
                return self._predictor.kospi_symbol
        except Exception:
            pass
        
        return None
    
    def _get_upcode(self, plot_type: str) -> Optional[str]:
        """종목 코드 가져오기 (순서: 파라미터 > predictor > config)."""
        # 파라미터로 전달된 upcode 우선
        if plot_type == "futures" and self._kp200_upcode:
            return self._kp200_upcode
        if plot_type == "kospi" and self._kospi_upcode:
            return self._kospi_upcode
        
        # predictor에서 가져오기
        upcode = self._get_upcode_from_predictor(plot_type)
        if upcode:
            return upcode
        
        # config에서 가져오기
        upcode = self._get_upcode_from_config(plot_type)
        if upcode:
            return upcode
        
        return None
    
    def get_current_price(self) -> float:
        """tick_processor에서 현재가를 가져온다.
        
        Returns:
            현재가 (없으면 0.0)
        """
        tp = self._get_tick_processor()
        if tp is not None:
            price = self._get_price_from_tick_processor(tp)
            logger.info("[DataFetcher] tick_processor 현재가: %s (plot=%s)", price, self._selected_plot)
            return price
        else:
            logger.warning("[DataFetcher] tick_processor 없음")
        return 0.0
    
    def _get_price_from_tick_processor(self, tp: Any) -> float:
        """tick_processor에서 현재가 가져오기."""
        logger.info("[DataFetcher] _get_price_from_tick_processor 호출 (plot=%s)", self._selected_plot)
        if self._selected_plot == "kospi" and hasattr(tp, "get_latest_spot_index_price"):
            price = float(tp.get_latest_spot_index_price())
            logger.info("[DataFetcher] get_latest_spot_index_price 반환: %s", price)
            return price
        if hasattr(tp, "get_current_price"):
            price = float(tp.get_current_price())
            logger.info("[DataFetcher] get_current_price 반환: %s", price)
            return price
        logger.warning("[DataFetcher] 현재가 가져오기 메서드 없음")
        return 0.0
