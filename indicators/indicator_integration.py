"""Adaptive Indicators Integration
===================================
AdaptiveSuperTrend + AdaptiveZigZag 통합 관리자.

두 지표를 동시에 업데이트하고, Cross-feature 및 통합 LLM 컨텍스트를 생성합니다.
Transformer 피처(22개+)와 SkyEbest 실시간 파이프라인 모두 이 클래스를 사용할 수 있습니다.
"""

from __future__ import annotations

import copy
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from .adaptive_supertrend import AdaptiveSuperTrend, AdaptiveSuperTrendConfig, SuperTrendState
    from .adaptive_zigzag import AdaptiveZigZag, AdaptiveZigZagConfig, ZigZagState
except ImportError:
    from adaptive_supertrend import AdaptiveSuperTrend, AdaptiveSuperTrendConfig, SuperTrendState
    from adaptive_zigzag import AdaptiveZigZag, AdaptiveZigZagConfig, ZigZagState


@dataclass
class IndicatorManagerConfig:
    supertrend: Optional[AdaptiveSuperTrendConfig] = None
    zigzag:     Optional[AdaptiveZigZagConfig]     = None
    kospi_supertrend: Optional[AdaptiveSuperTrendConfig] = None
    kospi_zigzag: Optional[AdaptiveZigZagConfig] = None
    futures_supertrend: Optional[AdaptiveSuperTrendConfig] = None
    futures_zigzag: Optional[AdaptiveZigZagConfig] = None
    symbol:         str = "KP200 선물"
    kospi_symbol:   str = "KOSPI 지수"
    futures_symbol: str = "KP200 선물"
    dual_mode:  bool = False
    pivot_proximity_alert_enabled: bool = True
    pivot_proximity_max_bars_diff: int = 1
    pivot_proximity_telegram_enabled: bool = True
    pivot_candidate_alert_enabled: bool = True
    pivot_candidate_alert_events: List[str] = field(default_factory=lambda: ["registered", "changed", "cancelled"])
    pivot_candidate_alert_telegram_enabled: bool = True
    pivot_candidate_alert_change_cooldown_sec: float = 60.0
    # ZigZag 최소 스윙 수 (is_ready 판단 기준)
    min_swings_for_ready: int = 4

    def __post_init__(self) -> None:
        if self.supertrend is None:
            self.supertrend = AdaptiveSuperTrendConfig()
        if self.zigzag is None:
            self.zigzag = AdaptiveZigZagConfig()
        if self.kospi_supertrend is None:
            self.kospi_supertrend = self.supertrend
        if self.kospi_zigzag is None:
            self.kospi_zigzag = self.zigzag
        if self.futures_supertrend is None:
            self.futures_supertrend = self.supertrend
        if self.futures_zigzag is None:
            self.futures_zigzag = self.zigzag
        
        # [FIX] config 설정과 무관하게 심볼 이름을 prefix로 설정
        from core.utils import normalize_adaptive_indicator_symbol
        _sym_n = normalize_adaptive_indicator_symbol(self.symbol)
        self.zigzag.pivot_lifecycle_log_prefix = f"[{_sym_n}]" if _sym_n else ""
        self.zigzag.pivot_lifecycle_log = True
        
        if self.dual_mode:
            _kospi_sym_n = normalize_adaptive_indicator_symbol(self.kospi_symbol)
            self.kospi_zigzag.pivot_lifecycle_log_prefix = "[KOSPI]"
            self.kospi_zigzag.pivot_lifecycle_log = True
            
            _futures_sym_n = normalize_adaptive_indicator_symbol(self.futures_symbol)
            self.futures_zigzag.pivot_lifecycle_log_prefix = "[KP200]"
            self.futures_zigzag.pivot_lifecycle_log = True


class AdaptiveIndicatorManager:
    """AdaptiveSuperTrend + AdaptiveZigZag 통합 관리자.

    Transformer 피처 (약 26개):
        ast_*  : AdaptiveSuperTrend 9개
        azz_*  : AdaptiveZigZag 14개
        cross_*: 크로스 피처 4개 (이 클래스 전용)

    SkyEbest 실시간 파이프라인에서 사용 시:
        manager = AdaptiveIndicatorManager()
        result = manager.update(high, low, close)
        features = result["transformer"]
        llm_ctx  = result["llm_context"]
    """

    def __init__(self, config: Optional[IndicatorManagerConfig] = None) -> None:
        self.config = config or IndicatorManagerConfig()
        self.supertrend = AdaptiveSuperTrend(self.config.supertrend)
        self.zigzag     = AdaptiveZigZag(self.config.zigzag)
        self.zigzag.set_symbol("KP200 선물")  # 기본 심볼 설정
        self._bar_count:  int   = 0
        self._last_close: float = 0.0
        
        # 피봇 예측 파이프라인 (선택적)
        self._pivot_pipeline = None
        
        # 듀얼 모드: KOSPI 지수와 KP200 선물 각각의 SuperTrend와 ZigZag
        self.kospi_supertrend = None
        self.kospi_zigzag = None
        self.futures_supertrend = None
        self.futures_zigzag = None
        if self.config.dual_mode:
            kospi_st_config = self.config.kospi_supertrend or self.config.supertrend
            kospi_zz_config = self.config.kospi_zigzag or self.config.zigzag
            futures_st_config = self.config.futures_supertrend or self.config.supertrend
            futures_zz_config = self.config.futures_zigzag or self.config.zigzag
            logger.info(f"[INDICATOR] kospi_zigzag config: use_atr_based_filtering={kospi_zz_config.use_atr_based_filtering}")
            logger.info(f"[INDICATOR] futures_zigzag config: use_atr_based_filtering={futures_zz_config.use_atr_based_filtering}")
            
            self.kospi_supertrend = AdaptiveSuperTrend(kospi_st_config)
            self.kospi_zigzag = AdaptiveZigZag(kospi_zz_config)
            self.kospi_zigzag.set_symbol(self.config.kospi_symbol or "KOSPI 지수")

            self.futures_supertrend = AdaptiveSuperTrend(futures_st_config)
            self.futures_zigzag = AdaptiveZigZag(futures_zz_config)
            self.futures_zigzag.set_symbol(self.config.futures_symbol or "KP200 선물")
        
        # 피봇 근접 알림 콜백 및 쿨다운
        self._pivot_proximity_callback = None
        self._last_proximity_alert_time = None
        self._proximity_alert_cooldown_sec = 60  # 기본 1분 쿨다운

        # 피봇 후보 알림 콜백 및 쿨다운
        self._pivot_candidate_callback = None
        self._last_candidate_alert_time: Dict[str, float] = {}  # candidate_id -> last_alert_time
        self._candidate_alert_cooldown_sec = 60  # 기본 1분 쿨다운

        # 피봇 후보 콜백 설정 (adaptive_zigzag의 pivot_collector에 전달)
        self._setup_pivot_candidate_callback()

        # 멀티스케일 데이터 저장소 (5분봉/15분봉 피처)
        self._multiscale_enabled = False
        self._multiscale_5m_atr = None  # 5분봉 ATR (최근값)
        self._multiscale_15m_atr = None  # 15분봉 ATR (최근값)
        self._multiscale_5m_pivot_direction = None  # 5분봉 피봇 방향 (1=상승, -1=하락, 0=중립)
        self._multiscale_15m_pivot_direction = None  # 15분봉 피봇 방향

    def init_pivot_pipeline(
        self,
        classifier_path: Optional[str] = None,
        regressor_path: Optional[str] = None,
        lifespan_path: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        """피봇 예측 파이프라인 초기화.
        
        Args:
            classifier_path: 분류 모델 경로
            regressor_path: 회귀 모델 경로
            lifespan_path: 시계열 모델 경로
            device: 장치
        """
        try:
            from prediction.pivot_pipeline import PivotPredictionPipeline
            self._pivot_pipeline = PivotPredictionPipeline(
                classifier_path=classifier_path,
                regressor_path=regressor_path,
                lifespan_path=lifespan_path,
                zigzag=self.zigzag,
                device=device,
            )
        except Exception as e:
            _logger = __import__('logging').getLogger(__name__)
            _logger.warning(f"피봇 예측 파이프라인 초기화 실패: {e}")
    
    @property
    def pivot_pipeline(self):
        """피봇 예측 파이프라인."""
        return self._pivot_pipeline

    def update_multiscale_data(
        self,
        *,
        enabled: bool = False,
        atr_5m: Optional[float] = None,
        atr_15m: Optional[float] = None,
        pivot_direction_5m: Optional[int] = None,
        pivot_direction_15m: Optional[int] = None,
    ) -> None:
        """멀티스케일 데이터 업데이트.

        Args:
            enabled: 멀티스케일 기능 활성화 여부
            atr_5m: 5분봉 ATR (최근값)
            atr_15m: 15분봉 ATR (최근값)
            pivot_direction_5m: 5분봉 피봇 방향 (1=상승, -1=하락, 0=중립)
            pivot_direction_15m: 15분봉 피봇 방향
        """
        self._multiscale_enabled = bool(enabled)
        self._multiscale_5m_atr = float(atr_5m) if atr_5m is not None else None
        self._multiscale_15m_atr = float(atr_15m) if atr_15m is not None else None
        self._multiscale_5m_pivot_direction = int(pivot_direction_5m) if pivot_direction_5m is not None else None
        self._multiscale_15m_pivot_direction = int(pivot_direction_15m) if pivot_direction_15m is not None else None

    def get_multiscale_atr(self) -> Optional[float]:
        """멀티스케일 ATR 혼합값 반환.

        1분봉 ATR과 5분/15분봉 ATR을 가중 평균하여 반환.
        멀티스케일 비활성화 시 None 반환.
        """
        if not self._multiscale_enabled:
            return None

        # SuperTrend의 현재 ATR 가져오기
        try:
            current_atr = self.supertrend.atr
        except Exception:
            current_atr = None

        if current_atr is None:
            return None

        # 가중 평균: 1분봉 50%, 5분봉 30%, 15분봉 20%
        weights = [0.5, 0.3, 0.2]
        atrs = [current_atr]

        if self._multiscale_5m_atr is not None:
            atrs.append(self._multiscale_5m_atr)
        else:
            weights[1] = 0  # 5분봉 없으면 가중치 재조정

        if self._multiscale_15m_atr is not None:
            atrs.append(self._multiscale_15m_atr)
        else:
            weights[2] = 0  # 15분봉 없으면 가중치 재조정

        # 가중치 정규화
        total_weight = sum(weights)
        if total_weight == 0:
            return current_atr

        normalized_weights = [w / total_weight for w in weights]

        # 가중 평균 계산
        mixed_atr = sum(atr * weight for atr, weight in zip(atrs, normalized_weights))
        return mixed_atr

    def should_filter_zigzag_signal(self, signal: str) -> tuple[bool, str]:
        """상위 타임프레임 피봇 방향 기반 ZigZag 신호 필터링.

        Args:
            signal: 현재 ZigZag 신호 ("buy", "sell", "hold")

        Returns:
            (should_filter, reason): 필터링 여부와 사유
        """
        if not self._multiscale_enabled:
            return False, ""

        if signal == "hold":
            return False, ""

        # 현재 1분봉 ZigZag 방향 확인
        try:
            current_direction = self.zigzag.state.direction
        except Exception:
            current_direction = 0

        # 5분봉 피봇 방향과 일치하는지 확인
        if self._multiscale_5m_pivot_direction is not None:
            if current_direction == 1 and self._multiscale_5m_pivot_direction == -1:
                # 1분봉 상승인데 5분봉 하락 추세 -> buy 신호 필터
                if signal == "buy":
                    return True, "5분봉 하락 추세와 불일치"
            elif current_direction == -1 and self._multiscale_5m_pivot_direction == 1:
                # 1분봉 하락인데 5분봉 상승 추세 -> sell 신호 필터
                if signal == "sell":
                    return True, "5분봉 상승 추세와 불일치"

        # 15분봉 피봇 방향과 일치하는지 확인 (더 강력한 필터)
        if self._multiscale_15m_pivot_direction is not None:
            if current_direction == 1 and self._multiscale_15m_pivot_direction == -1:
                if signal == "buy":
                    return True, "15분봉 하락 추세와 불일치"
            elif current_direction == -1 and self._multiscale_15m_pivot_direction == 1:
                if signal == "sell":
                    return True, "15분봉 상승 추세와 불일치"

        return False, ""
    
    def _check_pivot_proximity(
        self,
        kospi_state: Optional[Any],
        futures_state: Optional[Any],
    ) -> None:
        """KOSPI와 KP200 선물의 피봇 위치 근접 감지.
        
        두 시장의 피봇 발생 위치가 설정된 봉 차이 이내인 경우 주요 분봉으로 판단하고 로그/알림.
        """
        # 설정 확인
        if not self.config.pivot_proximity_alert_enabled:
            return
        
        if kospi_state is None or futures_state is None:
            return
        
        try:
            # 마지막 확정 피봇 인덱스 가져오기
            kospi_last_idx = None
            futures_last_idx = None
            
            # KOSPI 마지막 피봇 인덱스 (고점 또는 저점 중 최신)
            if kospi_state.last_swing_high_idx > kospi_state.last_swing_low_idx:
                kospi_last_idx = kospi_state.last_swing_high_idx
            else:
                kospi_last_idx = kospi_state.last_swing_low_idx
            
            # KP200 마지막 피봇 인덱스 (고점 또는 저점 중 최신)
            if futures_state.last_swing_high_idx > futures_state.last_swing_low_idx:
                futures_last_idx = futures_state.last_swing_high_idx
            else:
                futures_last_idx = futures_state.last_swing_low_idx
            
            # 두 인덱스 모두 유효한지 확인
            if kospi_last_idx is None or futures_last_idx is None:
                return
            
            # 인덱스 차이 계산
            idx_diff = abs(kospi_last_idx - futures_last_idx)
            
            # 설정된 봉 차이 이내인 경우 감지
            max_diff = self.config.pivot_proximity_max_bars_diff
            if idx_diff <= max_diff:
                import time
                current_time = time.time()
                
                # 쿨다운 확인
                if self._last_proximity_alert_time is not None:
                    elapsed = current_time - self._last_proximity_alert_time
                    if elapsed < self._proximity_alert_cooldown_sec:
                        return  # 쿨다운 기간 내
                
                # 피봇 타입 확인
                kospi_type = "H" if kospi_last_idx == kospi_state.last_swing_high_idx else "L"
                futures_type = "H" if futures_last_idx == futures_state.last_swing_high_idx else "L"
                
                # 피봇 가격
                kospi_price = kospi_state.last_swing_high if kospi_type == "H" else kospi_state.last_swing_low
                futures_price = futures_state.last_swing_high if futures_type == "H" else futures_state.last_swing_low
                
                # 로그 출력
                _logger.info(
                    "[PIVOT_PROXIMITY] KOSPI와 KP200 선물 피봇 근접 감지 "
                    f"(차이: {idx_diff}봉) | "
                    f"KOSPI: {kospi_type}@{kospi_price:.2f} (idx:{kospi_last_idx}) | "
                    f"KP200: {futures_type}@{futures_price:.2f} (idx:{futures_last_idx}) | "
                    f"⚠️ 주요 분봉 가능성 높음"
                )
                
                # 쿨다운 갱신
                self._last_proximity_alert_time = current_time
                
                # 알림 콜백 호출 (설정된 경우)
                if hasattr(self, '_pivot_proximity_callback') and callable(self._pivot_proximity_callback):
                    try:
                        self._pivot_proximity_callback(
                            kospi_type=kospi_type,
                            kospi_price=kospi_price,
                            kospi_idx=kospi_last_idx,
                            futures_type=futures_type,
                            futures_price=futures_price,
                            futures_idx=futures_last_idx,
                            idx_diff=idx_diff,
                        )
                    except Exception:
                        pass
        except Exception:
            pass
    
    def set_pivot_proximity_callback(self, callback: Callable) -> None:
        """피봇 근접 감지 알림 콜백 설정."""
        self._pivot_proximity_callback = callback

    def set_pivot_candidate_callback(self, callback: Callable) -> None:
        """피봇 후보 알림 콜백 설정."""
        self._pivot_candidate_callback = callback

    def _setup_pivot_candidate_callback(self) -> None:
        """피봇 후보 콜백 설정 (adaptive_zigzag의 pivot_collector에 전달)."""
        try:
            # 메인 zigzag의 pivot_collector에 콜백 설정
            if hasattr(self.zigzag, 'pivot_collector'):
                collector = self.zigzag.pivot_collector
                if collector is not None:
                    cooldown = self.config.pivot_candidate_alert_change_cooldown_sec
                    collector.set_callback(
                        callback=self._on_pivot_candidate_event,
                        change_cooldown_sec=cooldown
                    )
                    logger.info("[PIVOT_CANDIDATE] 메인 zigzag 콜백 설정 완료")

            # 듀얼 모드: KOSPI와 KP200의 pivot_collector에도 콜백 설정
            if self.config.dual_mode:
                if self.kospi_zigzag and hasattr(self.kospi_zigzag, 'pivot_collector'):
                    collector = self.kospi_zigzag.pivot_collector
                    if collector is not None:
                        cooldown = self.config.pivot_candidate_alert_change_cooldown_sec
                        collector.set_callback(
                            callback=self._on_pivot_candidate_event,
                            change_cooldown_sec=cooldown
                        )
                        logger.info("[PIVOT_CANDIDATE] KOSPI zigzag 콜백 설정 완료")
                if self.futures_zigzag and hasattr(self.futures_zigzag, 'pivot_collector'):
                    collector = self.futures_zigzag.pivot_collector
                    if collector is not None:
                        cooldown = self.config.pivot_candidate_alert_change_cooldown_sec
                        collector.set_callback(
                            callback=self._on_pivot_candidate_event,
                            change_cooldown_sec=cooldown
                        )
                        logger.info("[PIVOT_CANDIDATE] KP200 zigzag 콜백 설정 완료")
        except Exception as e:
            logger.warning("[PIVOT_CANDIDATE] 콜백 설정 실패: %s", e)

    def _on_pivot_candidate_event(self, **kwargs: Any) -> None:
        """피봇 후보 이벤트 콜백 (pivot_collector에서 호출됨)."""
        try:
            # 설정에서 활성화된 이벤트만 전송
            enabled_events = self.config.pivot_candidate_alert_events
            event_type = kwargs.get("event_type", "")
            if event_type not in enabled_events:
                return

            # 콜백 호출
            if self._pivot_candidate_callback:
                symbol = kwargs.get("symbol", "")
                candidate_type = kwargs.get("candidate_type", "")
                candidate_price = kwargs.get("candidate_price", 0.0)
                logger.info(
                    "[PIVOT_CANDIDATE] 콜백 전송: %s, %s, %s@%.2f",
                    event_type, symbol, candidate_type, candidate_price
                )
                self._pivot_candidate_callback(**kwargs)
        except Exception as e:
            logger.warning("[PIVOT_CANDIDATE] 콜백 전송 실패: %s", e)

    # ──────────────────── 공개 API ────────────────────────

    def is_ready(self) -> bool:
        """안정적인 피처 생성이 가능한 상태인지 반환."""
        try:
            st = self.config.supertrend
            zz = self.config.zigzag
            min_bars = int(max(
                getattr(st, "atr_max_period", 21),
                getattr(st, "adx_period",     14),
                getattr(st, "er_period",       10),
                getattr(st, "bb_period",       20),
                getattr(zz, "atr_period",      14),
            ) + 5)
            if self._bar_count < min_bars:
                return False
            # ZigZag: 최소 스윙 수 확정 필요 (config에서 설정)
            # anchor pivot(index==0) 제외 (초기화용, 실시간 신호 아님)
            min_swings = int(getattr(self.config, "min_swings_for_ready", 4) or 4)
            all_swings = getattr(self.zigzag, "_all_swings", []) or []
            # anchor 제외: index==0인 swing는 anchor
            actual_swings = [s for s in all_swings if s.index != 0]
            if len(actual_swings) < min_swings:
                return False
            return True
        except Exception:
            return self._bar_count >= 30

    def update(
        self,
        high: float,
        low: float,
        close: float,
        open: Optional[float] = None,
        bar_time: Any = None,
        *,
        skip_zigzag: bool = False,
        # [FIX] 데이터 소스 분리: KOSPI와 Futures 데이터를 별도로 받음
        kospi_high: Optional[float] = None,
        kospi_low: Optional[float] = None,
        kospi_close: Optional[float] = None,
        kospi_open: Optional[float] = None,
    ) -> Dict[str, Any]:
        """새 봉 데이터로 두 지표 업데이트 후 통합 결과 딕셔너리 반환.

        Parameters
        ----------
        high, low, close : float
            KP200 선물 데이터 (기본)
        kospi_high, kospi_low, kospi_close : float, optional
            KOSPI 지수 데이터 (dual_mode일 때 사용)
        open : float, optional
            당일(세션) 첫 번째 봉의 시가.  _bar_count == 0 일 때만 anchor pivot 주입에
            사용되며, 이후 봉에서는 무시된다.  기존 호출부는 수정 불필요(기본값 None).
        kospi_open : float, optional
            KOSPI 지수 시가 (dual_mode일 때 사용)

        Returns
        -------
        {
            "transformer"     : Dict[str, float],   # 26개 정규화 피처
            "llm_context"     : str,                # 통합 자연어 컨텍스트
            "supertrend_state": SuperTrendState,
            "zigzag_state"    : ZigZagState,
            "kospi_zigzag_state": ZigZagState,     # dual_mode일 때 추가
            "futures_zigzag_state": ZigZagState,   # dual_mode일 때 추가
            "bar_count"       : int,
            "is_ready"        : bool,
        }

        skip_zigzag : bool, default False
            True이면 이번 봉은 ZigZag에 반영하지 않고(피봇 확정 흔들림 방지),
            직전까지의 ZigZag 상태로 피처·컨텍스트만 갱신한다. 배치
            ``compute_from_df``의 마지막 행과 동일한 의미.
        """
        # ── anchor pivot injection ────────────────────────────────────────
        # [FIX] 시가 anchor를 활성화하되, 충분한 데이터 누적 후 anchor 심도록 수정
        # 장 초반 변동성이 크더라도 피봇을 빠르게 탐지하기 위해 시가 anchor 유지
        # 첫 번째 봉이 아닌 최소 3봉 이상 데이터가 누적된 후 anchor 심도록 수정
        # [FIX] 심볼별 장 시작 시간 고려: KP200 08:45부터, KOSPI 09:00부터
        if self._bar_count == 3 and open is not None:  # 4번째 봉에서 anchor 심음 (0-based index)
            try:
                from kospi_indicators.adaptive_zigzag import SwingType as _SwingType
            except ImportError:
                try:
                    from .adaptive_zigzag import SwingType as _SwingType
                except ImportError:
                    _SwingType = None
            if _SwingType is not None:
                try:
                    open_f = float(open)
                    import math as _math
                    if _math.isfinite(open_f) and open_f > 0:
                        # [FIX] 심볼별 장 시작 시간 확인
                        market_start_hour = 9
                        market_start_minute = 0
                        if "KP200" in self.config.symbol or "선물" in self.config.symbol:
                            market_start_hour = 8
                            market_start_minute = 45
                        
                        # 현재 봉 시간 확인
                        bar_hour = None
                        bar_minute = None
                        if bar_time:
                            try:
                                from datetime import datetime as _datetime
                                if isinstance(bar_time, str):
                                    bar_dt = _datetime.strptime(bar_time, "%H:%M")
                                elif isinstance(bar_time, _datetime):
                                    bar_dt = bar_time
                                else:
                                    bar_dt = None
                                if bar_dt:
                                    bar_hour = bar_dt.hour
                                    bar_minute = bar_dt.minute
                            except Exception:
                                pass
                        
                        # 장 시작 시간 이후에만 anchor 심기
                        if bar_hour is not None and (bar_hour > market_start_hour or (bar_hour == market_start_hour and bar_minute >= market_start_minute)):
                            # [FIX] anchor 타입 결정 개선
                            # 첫 4봉의 고가/저가 중 시가보다 더 큰 움직임 쪽을 anchor로 선택
                            # 시가 대비 상승 폭이 하락 폭보다 크면 HIGH anchor
                            # 시가 대비 하락 폭이 상승 폭보다 크면 LOW anchor
                            # 하지만 현재 봉의 high/low만으로는 첫 4봉의 움직임을 알 수 없으므로
                            # 시가와 현재 봉의 high/low 거리로 결정
                            anc_type = _SwingType.HIGH if abs(open_f - high) < abs(open_f - low) else _SwingType.LOW
                            self.zigzag.seed_anchor(open_f, anc_type)
                            logger.info("[AdaptiveIndicatorManager] anchor pivot 주입 (4번째 봉): %.2f, type=%s (장 시작: %02d:%02d)", open_f, anc_type, market_start_hour, market_start_minute)
                        else:
                            logger.debug("[AdaptiveIndicatorManager] anchor pivot 주입 차단: 장 시작 전 (bar_time=%s, 장 시작: %02d:%02d)", bar_time, market_start_hour, market_start_minute)
                except Exception as e:
                    logger.warning("[AdaptiveIndicatorManager] anchor pivot 주입 실패: %s", e)
        # ─────────────────────────────────────────────────────────────────

        st_state = self.supertrend.update(high, low, close)
        
        # [SUPERTREND-INTEGRATION] 슈퍼트렌드 방향을 ZigZag에 전달 - 후보 등록 단계 필터용
        # [BUG-FIX] 기존: st_state.signal("buy"/"sell"/"hold") → "bull"/"bear" 불일치로 전 피봇 차단
        # 수정:    st_state.direction(1=bull, -1=bear, 지속값) → "bull"/"bear" 변환
        #          direction은 추세가 유지되는 한 값이 지속되므로 시점 문제 없음
        if self.config.supertrend_pivot_filter and hasattr(self.zigzag, 'set_supertrend_signal'):
            st_dir = getattr(st_state, 'direction', None)
            if st_dir is not None:
                _st_sig = "bull" if int(st_dir) == 1 else ("bear" if int(st_dir) == -1 else "")
                if _st_sig:
                    self.zigzag.set_supertrend_signal(_st_sig)
        
        # 멀티스케일 ATR 적용 (활성화 시)
        if self._multiscale_enabled:
            mixed_atr = self.get_multiscale_atr()
            if mixed_atr is not None:
                # SuperTrend의 ATR 값을 혼합 ATR로 덮어쓰기
                try:
                    self.supertrend._atr = mixed_atr
                    self.supertrend._prev_atr = mixed_atr
                    # 마지막 ATR 값도 업데이트
                    if len(self.supertrend._atr_values) > 0:
                        self.supertrend._atr_values[-1] = mixed_atr
                except Exception:
                    pass
        
        if skip_zigzag:
            # 마지막(미완결) 봉을 ZigZag에 넣지 않음 — 피봇 확정이 흔들리지 않도록
            zz_state = self.zigzag.state
        else:
            zz_state = self.zigzag.update(
                high, low, close, bar_time=bar_time, open=float(open) if open is not None else 0.0
            )
            # 후보 상태 로그: 후보 갱신/대기(rem, urgency 등)를 분봉마다 관찰할 수 있게 출력
            try:
                self.zigzag.emit_pending_status_log(close=float(close))
            except Exception:
                pass
        
        # 듀얼 모드: KOSPI 지수와 KP200 선물 각각 업데이트
        # [FIX] 데이터 소스 분리: predictor 내부에서는 dual_mode ZigZag 업데이트 비활성화
        # predictor는 주로 futures 데이터만 사용하며, KOSPI 데이터는 chart_viewer에서 별도 처리
        # dual_mode는 chart_viewer에서만 사용하며, predictor 내부에서는 비활성화하여 데이터 오염 방지
        kospi_st_state = None
        kospi_zz_state = None
        futures_st_state = None
        futures_zz_state = None
        
        # predictor 내부 dual_mode 비활성화 (데이터 소스 분리 강화)
        # chart_viewer에서는 predictor의 ZigZag를 사용하지 않으므로 안전
        if False:  # predictor 내부 dual_mode 비활성화
            if self.config.dual_mode:
                # KOSPI 지수 업데이트 (별도 데이터 사용)
                if self.kospi_supertrend is not None:
                    try:
                        kospi_h = kospi_high if kospi_high is not None else high
                        kospi_l = kospi_low if kospi_low is not None else low
                        kospi_c = kospi_close if kospi_close is not None else close
                        kospi_st_state = self.kospi_supertrend.update(kospi_h, kospi_l, kospi_c)
                        # [SUPERTREND-INTEGRATION] KOSPI ZigZag에 슈퍼트렌드 신호 전달
                        # [BUG-FIX] signal → direction 변경 (메인 ZigZag와 동일)
                        if self.config.supertrend_pivot_filter and hasattr(self.kospi_zigzag, 'set_supertrend_signal'):
                            kospi_st_dir = getattr(kospi_st_state, 'direction', None)
                            if kospi_st_dir is not None:
                                _kospi_st_sig = "bull" if int(kospi_st_dir) == 1 else ("bear" if int(kospi_st_dir) == -1 else "")
                                if _kospi_st_sig:
                                    self.kospi_zigzag.set_supertrend_signal(_kospi_st_sig)
                    except Exception:
                        pass
                if self.kospi_zigzag is not None:
                    try:
                        kospi_h = kospi_high if kospi_high is not None else high
                        kospi_l = kospi_low if kospi_low is not None else low
                        kospi_c = kospi_close if kospi_close is not None else close
                        kospi_o = kospi_open if kospi_open is not None else open
                        kospi_zz_state = self.kospi_zigzag.update(
                            kospi_h, kospi_l, kospi_c, bar_time=bar_time, open=float(kospi_o) if kospi_o is not None else 0.0
                        )
                        try:
                            self.kospi_zigzag.emit_pending_status_log(close=float(kospi_c))
                        except Exception:
                            pass
                    except Exception:
                        pass
                
                # KP200 선물 업데이트 (기본 데이터 사용)
                if self.futures_supertrend is not None:
                    futures_st_state = self.futures_supertrend.update(high, low, close)
                    # [SUPERTREND-INTEGRATION] Futures ZigZag에 슈퍼트렌드 신호 전달
                    # [BUG-FIX] signal → direction 변경 (메인 ZigZag와 동일)
                    if self.config.supertrend_pivot_filter and hasattr(self.futures_zigzag, 'set_supertrend_signal'):
                        futures_st_dir = getattr(futures_st_state, 'direction', None)
                        if futures_st_dir is not None:
                            _futures_st_sig = "bull" if int(futures_st_dir) == 1 else ("bear" if int(futures_st_dir) == -1 else "")
                            if _futures_st_sig:
                                self.futures_zigzag.set_supertrend_signal(_futures_st_sig)
                if self.futures_zigzag is not None:
                    try:
                        futures_zz_state = self.futures_zigzag.update(
                            high, low, close, bar_time=bar_time, open=float(open) if open is not None else 0.0
                        )
                        try:
                            self.futures_zigzag.emit_pending_status_log(close=float(close))
                        except Exception:
                            pass
                    except Exception:
                        pass
                
                # 피봇 위치 근접 감지
                self._check_pivot_proximity(kospi_zz_state, futures_zz_state)
        
        self._bar_count  += 1
        self._last_close  = float(close)

        feats: Dict[str, float] = {}
        feats.update(self.supertrend.get_transformer_features(close))
        feats.update(self.zigzag.get_transformer_features(close))
        feats.update(self._calc_cross_features(close, st_state, zz_state))
        
        # 피봇 예측
        pivot_prediction = None
        if self._pivot_pipeline is not None:
            try:
                pivot_prediction = self._pivot_pipeline.predict(float(close))
            except Exception:
                pass

        return {
            "transformer":      feats,
            "llm_context":      self._build_llm_context(close, st_state, zz_state),
            "supertrend_state": st_state,
            "zigzag_state":     zz_state,
            "kospi_supertrend_state": kospi_st_state,
            "kospi_zigzag_state": kospi_zz_state,
            "futures_supertrend_state": futures_st_state,
            "futures_zigzag_state": futures_zz_state,
            "bar_count":        self._bar_count,
            "is_ready":         self.is_ready(),
            "pivot_prediction": pivot_prediction,
        }

    def compute_from_df(
        self,
        df: pd.DataFrame,
        high_col:  str = "high",
        low_col:   str = "low",
        close_col: str = "close",
    ) -> pd.DataFrame:
        """DataFrame 전체를 처리해 ast_*, azz_*, cross_* 컬럼이 추가된 DataFrame 반환.

        ZigZag는 마지막(미완결) 봉의 OHLC를 update에 넣지 않는다. 피봇 확정이
        마지막 봉에 의해 흔들리지 않도록 하며, azz_*·cross_*의 마지막 행은
        직전 봉까지의 ZigZag 상태 + 해당 행 종가로 계산된다.
        """
        st_tmp = AdaptiveSuperTrend(self.config.supertrend)
        zz_tmp = AdaptiveZigZag(self.config.zigzag)
        zz_tmp.set_backtest_mode(True)  # 백테스트 모드: look-ahead bias 방지

        from .adaptive_supertrend import _resolve_col
        hc = _resolve_col(df, high_col)
        lc = _resolve_col(df, low_col)
        cc = _resolve_col(df, close_col)

        rows: List[Dict[str, float]] = []
        n = len(df)
        for i, row in enumerate(df.itertuples(index=False)):
            h = float(getattr(row, hc))
            lo = float(getattr(row, lc))
            c = float(getattr(row, cc))
            st_s = st_tmp.update(h, lo, c)
            # 마지막 봉은 ZigZag에 반영하지 않음(미완결봉으로 인한 피봇 확정 흔들림 방지).
            # SuperTrend·크로스 피처는 종가 기준으로 마지막 봉까지 반영.
            if i < n - 1:
                zz_s = zz_tmp.update(h, lo, c)
            else:
                zz_s = zz_tmp.state
            feats: Dict[str, float] = {}
            feats.update(st_tmp.get_transformer_features(c))
            feats.update(zz_tmp.get_transformer_features(c))
            feats.update(self._calc_cross_features(c, st_s, zz_s))
            rows.append(feats)

        return df.assign(**pd.DataFrame(rows, index=df.index))

    def get_transformer_feature_names(self) -> List[str]:
        """피처 이름 목록 반환."""
        try:
            dummy = AdaptiveIndicatorManager(self.config)
            res = dummy.update(100.0, 99.0, 99.5)
            tf = res.get("transformer") if isinstance(res, dict) else None
            if isinstance(tf, dict) and tf:
                return list(tf.keys())
        except Exception:
            pass
        return []

    def reset(self) -> None:
        self.supertrend._reset_buffers()
        self.zigzag._reset_buffers()
        self._bar_count  = 0
        self._last_close = 0.0

    @property
    def feature_count(self) -> int:
        return len(self.get_transformer_feature_names())

    # ──────────────────── 내부 메서드 ────────────────────

    def _calc_cross_features(
        self, close: float, st: SuperTrendState, zz: ZigZagState
    ) -> Dict[str, float]:
        """두 지표를 결합한 크로스 피처 (4개)."""

        # 추세 방향 일치도
        agree = 0.0
        if st.direction != 0 and zz.current_direction != 0:
            agree = 1.0 if st.direction == zz.current_direction else -1.0

        # 지지선 근접
        at_sup = 0.0
        if zz.nearest_support > 0 and close > 0:
            d = (close - zz.nearest_support) / close * 100
            if d < 0.5:
                at_sup = 1.0 - d / 0.5

        # 저항선 근접
        at_res = 0.0
        if zz.nearest_resistance > 0 and close > 0:
            d = (zz.nearest_resistance - close) / close * 100
            if d < 0.5:
                at_res = 1.0 - d / 0.5

        # 돌파 잠재력
        bkout = 0.0
        win = 1.0
        if st.direction == 1 and zz.nearest_resistance > 0:
            nearness = max(0.0, 1.0 - float(zz.resistance_dist_pct or 0.0) / win)
            bkout = st.efficiency_ratio * nearness
        elif st.direction == -1 and zz.nearest_support > 0:
            nearness = max(0.0, 1.0 - float(zz.support_dist_pct or 0.0) / win)
            bkout = -(st.efficiency_ratio * nearness)

        return {
            "cross_trend_agreement":    float(agree),
            "cross_at_support":         float(np.clip(at_sup, 0, 1)),
            "cross_at_resistance":      float(np.clip(at_res, 0, 1)),
            "cross_breakout_potential": float(np.clip(bkout, -1, 1)),
        }

    def _build_llm_context(
        self, close: float, st: SuperTrendState, zz: ZigZagState
    ) -> str:
        """두 지표의 분석을 결합한 종합 LLM 컨텍스트.

        각 지표의 get_llm_context() 를 그대로 포함해
        완결봉 정보, 최근 스윙 고/저, 피보나치, 지지/저항 등
        상세 정보가 LLM 프롬프트에 누락되지 않도록 한다.
        """
        sym = self.config.kospi_symbol if self.config.dual_mode else "KOSPI 지수"

        # ── SuperTrend 전문 컨텍스트 ──────────────────────
        try:
            st_ctx = self.supertrend.get_llm_context(close, symbol=sym)
        except Exception:
            st_ctx = f"[Adaptive SuperTrend - {sym}] (컨텍스트 생성 실패)"

        # ── ZigZag 전문 컨텍스트 (완결봉·스윙 고저·피보·지지저항 포함) ──
        try:
            zz_ctx = self.zigzag.get_llm_context(close, symbol=sym)
        except Exception:
            zz_ctx = f"[Adaptive ZigZag - {sym}] (컨텍스트 생성 실패)"
        
        # ── 듀얼 모드: KOSPI와 KP200 슈퍼트렌드 컨텍스트 추가 ──
        dual_ctx = ""
        if self.config.dual_mode:
            kospi_st_ctx = ""
            futures_st_ctx = ""
            if self.kospi_supertrend is not None:
                try:
                    kospi_st_ctx = self.kospi_supertrend.get_llm_context(
                        close, symbol=self.config.kospi_symbol or "KOSPI 지수"
                    )
                except Exception:
                    kospi_st_ctx = "[KOSPI SuperTrend] (컨텍스트 생성 실패)"
            if self.futures_supertrend is not None:
                try:
                    futures_st_ctx = self.futures_supertrend.get_llm_context(
                        close, symbol=self.config.futures_symbol or "KP200 선물"
                    )
                except Exception:
                    futures_st_ctx = "[KP200 SuperTrend] (컨텍스트 생성 실패)"
            
            if kospi_st_ctx or futures_st_ctx:
                dual_ctx = f"\n\n{'='*60}\n[DUAL SUPER-TREND]\n{'='*60}\n"
                if kospi_st_ctx:
                    dual_ctx += kospi_st_ctx + "\n\n"
                if futures_st_ctx:
                    dual_ctx += futures_st_ctx

        # ── 크로스 요약 한 줄 ────────────────────────────
        agree = (
            st.direction == zz.current_direction
            and st.direction != 0
            and zz.current_direction != 0
        )
        signals = []
        if st.signal in ("buy", "sell"):
            signals.append(f"SuperTrend {st.signal.upper()} 플립")
        if zz.new_swing_signal != "none":
            signals.append(
                f"ZigZag {'고점' if zz.new_swing_signal == 'new_high' else '저점'} 확정"
            )
        sig_line = "현재 신호: " + " | ".join(signals) if signals else "현재 신호 없음"
        cross_line = (
            f"[CROSS] bars={self._bar_count} | {sig_line} | "
            f"agree={'Y' if agree else 'N'} | "
            f"trend={st.trend_strength}"
        )

        return "\n\n".join([st_ctx, zz_ctx, cross_line, dual_ctx])


# ──────────────────────────────────────────────────────────
# 배치 일관성 검증 유틸
# ──────────────────────────────────────────────────────────

def validate_consistency(
    manager: AdaptiveIndicatorManager,
    df: pd.DataFrame,
    *,
    high_col:  str   = "high",
    low_col:   str   = "low",
    close_col: str   = "close",
    atol:      float = 1e-6,
) -> bool:
    """배치(compute_from_df)와 스트리밍(update 루프) 결과가 일치하는지 검증."""
    try:
        tmp_batch = copy.deepcopy(manager)
    except Exception:
        tmp_batch = AdaptiveIndicatorManager(getattr(manager, "config", None))
    batch = tmp_batch.compute_from_df(df.copy(), high_col=high_col, low_col=low_col, close_col=close_col)

    try:
        tmp_stream = copy.deepcopy(manager)
        tmp_stream.reset()
    except Exception:
        tmp_stream = AdaptiveIndicatorManager(getattr(manager, "config", None))

    from .adaptive_supertrend import _resolve_col
    hc = _resolve_col(df, high_col)
    lc = _resolve_col(df, low_col)
    cc = _resolve_col(df, close_col)

    last_res: Optional[Dict[str, Any]] = None
    n = len(df)
    for i, row in enumerate(df.itertuples(index=False)):
        last_res = tmp_stream.update(
            float(getattr(row, hc)),
            float(getattr(row, lc)),
            float(getattr(row, cc)),
            skip_zigzag=(i == n - 1 and n > 0),
        )

    tf = (last_res or {}).get("transformer") if isinstance(last_res, dict) else None
    if not isinstance(tf, dict) or not tf:
        return False

    keys = [k for k in tf.keys() if k in batch.columns]
    if not keys:
        return False

    a = np.array([float(tf.get(k) or 0.0) for k in keys], dtype=np.float64)
    b = batch[keys].iloc[-1].values.astype(np.float64)
    return bool(np.allclose(a, b, atol=float(atol), rtol=0.0, equal_nan=True))
