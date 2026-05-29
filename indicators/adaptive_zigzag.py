"""Adaptive ZigZag Indicator
============================
스윙 임계값을 ATR 비율로 동적 결정하는 Adaptive ZigZag 구현.

수정된 버그
-----------
수정된 버그에 대한 상세 내용은 BUG_FIXES.md 참조.
(본 파일 상단 주석은 유지보수를 위해 별도 문서로 분리 권장)
"""

import logging
import warnings
import datetime
import copy
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# 상수 정의 (Constants)
# =============================================================================

class ZigZagConstants:
    """ZigZag 알고리즘 상수"""
    
    # ATR 관련 상수
    ATR_ROUNDING_DECIMALS = 6  # ATR 값 반올림 자릿수 (EDGE-CASE-3)
    ATR_PERIOD_MULTIPLIER = 2  # max_wait_bars 기본값 배수 (EDGE-CASE-2)
    DEFAULT_ATR_PERIOD = 14   # 기본 ATR 주기
    
    # 버퍼 관련 상수
    DEFAULT_MAX_BUF = 100     # 기본 버퍼 최대 크기
    WARMUP_MULTIPLIER = 5      # 웜업 구간 배수 (ATR 주기 * 5)
    
    # 로그 관련 상수
    MAX_LOG_VIOLATIONS = 5    # 교번 위배 로그 최대 출력 건수
    
    # 기타 상수
    DEFAULT_MAX_SWINGS = 20   # 기본 최대 스윙 수
    DEFAULT_CONFIRMATION_BARS = 2  # 기본 확인 바 수
    
    # 갭 보정 관련 상수
    DEFAULT_GAP_THRESHOLD = 2.0  # 기본 갭 보정 임계값 (%)

try:
    from .wilder_smooth import WilderRMA
except ImportError:
    from wilder_smooth import WilderRMA

_logger = logging.getLogger(__name__)



# ──────────────────────────────────────────────────────────
# 보조 타입
# ──────────────────────────────────────────────────────────

class FibLevels(dict):
    """피보나치 레벨 딕셔너리 — 레거시 float 키를 fib_NNN 형식으로 자동 변환."""

    def _alias(self, key: Any) -> Any:
        try:
            ks = str(key)
        except Exception:
            return key
        if ks.startswith("fib_"):
            return ks
        try:
            r = float(ks)
        except Exception:
            return key
        if r <= 0:
            return key
        try:
            new_key = f"fib_{int(round(r * 1000.0))}"
        except Exception:
            return key
        try:
            warnings.warn(
                f"legacy fib key '{ks}' -> use '{new_key}'",
                DeprecationWarning, stacklevel=2,
            )
        except Exception:
            pass
        return new_key

    def get(self, key: Any, default: Any = None):
        return super().get(self._alias(key), default)

    def __getitem__(self, key: Any):
        return super().__getitem__(self._alias(key))


class SwingType(Enum):
    HIGH = "high"
    LOW  = "low"


@dataclass
class SwingPoint:
    index:        int
    price:        float
    swing_type:   SwingType
    atr_at_swing: float
    is_major:     bool = False
    confirmed:    bool = False
    confirmed_at_idx: int = -1
    confirmed_close: float = 0.0  # 확정봉 종가
    registered_at_idx: int = -1  # 후보 등록 봉 인덱스 (차트 표시용)


# ──────────────────────────────────────────────────────────
# Config / State
# ──────────────────────────────────────────────────────────

@dataclass
class AdaptiveZigZagConfig:
    atr_multiplier:           float = 1.5
    atr_period:               int   = 14
    er_period:                int   = 10
    atr_multiplier_min:       float = 1.0
    atr_multiplier_max:       float = 4.0
    pivot_threshold_min_pct:  float = 0.3
    pivot_threshold_max_pct:  float = 3.0
    major_swing_ratio:        float = 2.0
    max_swings:               int   = 20
    confirmation_bars:  int   = 2
    freeze_on_confirm:  bool  = True
    min_wave_bars:      int   = 5
    min_wave_pct:       float = 0.25
    structure_lookback_swings: int = 8
    structure_points:          int = 3
    # 장초반 시간 기반 ATR multiplier 조절 (변동성 큰 장초반 피봇 잦은 변경 방지)
    early_session_start_time: str = "09:00"  # 장초반 시작 시간
    early_session_end_time: str = "09:30"   # 장초반 종료 시간
    # [MAINT-2 FIX] early_session_atr_multiplier_max 제거
    # session_min_wave_atr_ratio_table로 통합 완료 — 이 필드는 더 이상 사용되지 않음
    # early_session_atr_multiplier_max: float = 8.0  ← 삭제
    fib_ratios:         List[float] = field(
        default_factory=lambda: [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
    )
    cluster_tolerance_pct: float = 0.3
    max_wait_bars:          int   = 0   # [FIX-7] 0=무제한, >0: pending 자동취소 봉수 (SkyEbest에서 이식)
    # KOSPI 지수 / KP200 선물 등: 후보등록/취소/확정을 로그로 남길 때 True
    pivot_lifecycle_log: bool = False
    pivot_lifecycle_log_prefix: str = ""
    # ── 피봇 후보 수집기 설정 ─────────────────────────────────────
    enable_pivot_collector: bool = False  # 머신러닝 학습 데이터 수집 활성화
    pivot_collector_max_sequence: int = 120  # 시계열 최대 길이 (봉수)

    # ── [보완-1] confirmation_bars 동적 조절 ──────────────────────
    # ranging/unknown 구간에서 더 많은 봉을 대기해 허위 확정 억제
    confirmation_bars_ranging: int   = 2   # ranging 구간 대기 봉 수
    confirmation_bars_unknown: int   = 3   # unknown 구간 대기 봉 수

    # ── [보완-2] min_wave_pct 기본값 상향 ─────────────────────────
    # 0.0 → 0.25: 경계 잡음 피봇 차단 (dist_pct < 0.25% 이면 후보 등록 차단)
    # min_wave_pct: float = 0.25  ← 기존 필드, 기본값만 변경 (아래에서 재정의)

    # ── [보완-3] structure 다수결 임계값 ──────────────────────────
    structure_majority_threshold: float = 0.7   # 70% 이상 일관 → 추세 판정

    # ── [보완-5] bars_since 기반 임계값 decay ────────────────────
    decay_start_bars: int    = 30     # decay 시작 봉 수
    decay_rate_per_bar: float = 0.005  # 봉당 감소율 (%)
    decay_max_pct: float     = 0.3    # 최대 감소폭 (%)

    # ── [다중 시간프레임] 설정 ────────────────────────────────────
    multi_timeframe_enabled: bool = False
    multi_timeframe_scales: List[int] = field(default_factory=lambda: [1, 5, 15])
    multi_timeframe_consensus_threshold: int = 2
    multi_timeframe_price_tolerance_pct: float = 1.0
    multi_timeframe_index_tolerance_multiplier: float = 2.0

    # ── [보완-7] is_major 파동 비율 기준 ─────────────────────────
    major_wave_ratio: float  = 1.5   # 평균 파동의 N배 이상 → major
    major_wave_lookback: int = 3     # 평균 계산에 사용할 이전 파동 수

    # ── [보완-8] 방향 ER (Directional ER) ────────────────────────
    der_mismatch_threshold: float = 0.3   # DER 불일치 판정 임계값
    der_mismatch_mult_ratio: float = 0.7  # 불일치 시 mmax 감소 비율

    # ── [SESSION-MW] 시간대별 min_wave_bars 테이블 ─────────────────
    # 리스트 순서대로 평가하며 처음 일치하는 구간의 값을 사용한다.
    # 형식: List[Tuple[시작HH:MM, 종료HH:MM(미포함), min_wave_bars]]
    # - 빈 리스트(기본값) → 기존 min_wave_bars 단일값 적용 (하위 호환)
    # - 테이블에 해당 없는 구간 → min_wave_bars 단일값 폴백
    # 예시:
    #   session_min_wave_bars_table = [
    #       ("09:00", "09:30", 10),   # 장초반:  피봇 적게
    #       ("09:30", "10:30",  7),   # 오전장:  중간
    #       ("10:30", "14:30",  5),   # 중반:    기본
    #       ("14:30", "15:20",  7),   # 장마감전: 조임
    #       ("15:20", "15:31", 10),   # 동시호가: 매우 엄격
    #   ]
    session_min_wave_bars_table: List[Tuple[str, str, int]] = field(
        default_factory=list
    )

    # ── [ATR-FILTER] 시간대별 동적 ATR 비율 테이블 ─────────────────
    # 리스트 순서대로 평가하며 처음 일치하는 구간의 값을 사용한다.
    # 형식: List[Tuple[시작HH:MM, 종료HH:MM(미포함), min_wave_atr_ratio]]
    # - 빈 리스트(기본값) → 기존 min_wave_atr_ratio 단일값 적용 (하위 호환)
    # - 테이블에 해당 없는 구간 → min_wave_atr_ratio 단일값 폴백
    # 예시 (균형 전략):
    #   session_min_wave_atr_ratio_table = [
    #       ("09:00", "09:30", 0.8),   # 장 시작: 빠른 반응
    #       ("09:30", "10:30", 1.2),   # 오전: 안정적
    #       ("10:30", "13:00", 1.8),   # 점심: 노이즈 필터링
    #       ("13:00", "14:30", 1.2),   # 오후: 안정적
    #       ("14:30", "15:20", 0.8),   # 마감 전: 빠른 반응
    #       ("15:20", "15:30", 0.5),   # 마감: 최고 민감도
    #   ]
    session_min_wave_atr_ratio_table: List[Tuple[str, str, float]] = field(
        default_factory=list
    )

    # ── [ATR-FILTER] ATR 기반 필터링 파라미터 ─────────────────────
    use_atr_based_filtering: bool = False  # ATR 기반 필터링 활성화 여부
    min_wave_atr_ratio: float = 0.5  # 피봇으로 인식되기 위한 최소 파동 크기 (ATR 배수)
    cluster_atr_ratio: float = 0.5  # 피봇 클러스터링에 사용되는 ATR 배수

    # ── [HYBRID-MODE] 하이브리드 모드 파라미터 (ATR + 퍼센트 결합) ─────
    use_hybrid_mode: bool = False  # 하이브리드 모드 활성화 여부
    base_pct: float = 0.3  # 기본 퍼센트 임계값 (%)
    atr_weight: float = 1.0  # ATR 가중치 (0~1). 1.0=ATR만, 0.0=퍼센트만, 0.5=혼합
    multiplier_min: float = 0.8  # 퍼센트 기반 ER 배수 하한
    multiplier_max: float = 2.0  # 퍼센트 기반 ER 배수 상한
    session_multiplier_table: List[Tuple[str, str, float]] = field(
        default_factory=list
    )  # 시간대별 퍼센트 배율 테이블

    def __post_init__(self) -> None:
        try:
            self.structure_lookback_swings = int(self.structure_lookback_swings)
        except Exception:
            self.structure_lookback_swings = 8
        if int(self.structure_lookback_swings) < 4:
            self.structure_lookback_swings = 4

        try:
            self.structure_points = int(self.structure_points)
        except Exception:
            self.structure_points = 3
        if int(self.structure_points) < 2:
            self.structure_points = 2


@dataclass
class ZigZagState:
    current_direction:     int   = 0
    last_swing_high:       float = 0.0
    last_swing_low:        float = 0.0
    last_swing_high_idx:   int   = 0
    last_swing_low_idx:    int   = 0
    pending_high:          float = 0.0
    pending_low:           float = 0.0
    pending_high_idx:      int   = 0
    pending_low_idx:       int   = 0
    recent_swings:         List[SwingPoint] = field(default_factory=list)
    wave_size:             float = 0.0
    wave_size_pct:         float = 0.0
    wave_direction:        int   = 0
    fib_levels:            Dict[str, float] = field(default_factory=FibLevels)
    nearest_resistance:    float = 0.0
    nearest_support:       float = 0.0
    resistance_dist_pct:   float = 0.0
    support_dist_pct:      float = 0.0
    is_making_higher_highs: bool = False
    is_making_lower_lows:   bool = False
    # [REVIEW-FIX-5] 스윙 버전: 클러스터링 in-place 갱신 시 렌더링 캐시 무효화용
    swing_version:         int   = 0
    structure:             str   = "unknown"
    adaptive_threshold_pct: float = 0.0
    atr:                   float = 0.0
    # ── [ATR-MONITOR] ATR 변화 추적 ─────────────────────────────
    atr_change_pct:        float = 0.0          # 이전 ATR 대비 변화율 (%)
    atr_trend:             str   = "unknown"     # "rising", "falling", "stable"
    atr_spike_detected:    bool  = False         # 급격 변동 감지 여부
    atr_ma:                float = 0.0          # ATR 이동평균
    new_swing_signal:      str   = "none"
    bars_since_last_swing: int   = 0
    # ── 확정 피봇 완결봉 정보 ──────────────────────────────────
    # 피봇이 확정된 시점(new_swing_signal != "none")에 채워진다.
    # 텔레그램 송출 등 외부에서 "어느 봉이 피봇인지" 바로 참조 가능.
    last_swing_high_time:         Optional[str] = None  # 고점 피봇 봉 시각 "HH:MM"
    last_swing_low_time:          Optional[str] = None  # 저점 피봇 봉 시각 "HH:MM"
    last_swing_high_confirm_time: Optional[str] = None  # 고점 확정봉 시각 "HH:MM"
    last_swing_low_confirm_time:  Optional[str] = None  # 저점 확정봉 시각 "HH:MM"
    last_swing_high_lag_bars:     int   = 0             # 고점 피봇봉→확정봉 경과 봉수
    last_swing_low_lag_bars:      int   = 0             # 저점 피봇봉→확정봉 경과 봉수
    last_swing_high_open:         float = 0.0           # 고점 확정봉 시가
    last_swing_high_close:        float = 0.0           # 고점 확정봉 종가
    last_swing_low_open:          float = 0.0           # 저점 확정봉 시가
    last_swing_low_close:         float = 0.0           # 저점 확정봉 종가
    # 텔레그램/리포트용 누적 확정 피봇 요약
    confirmed_pivot_count:        int   = 0
    confirmed_pivot_tail_hhmm:    str   = ""
    # 취소된 피봇 후보 목록 (장종료시 텔레그램 송출용)
    cancelled_candidates:         List[Dict[str, Any]] = field(default_factory=list)
    # ── 피봇 후보 상태 (텔레그램 송출용) ─────────────────────────────
    pending_candidate_type:      Optional[str] = None   # "high" 또는 "low"
    pending_candidate_time:      Optional[str] = None   # 후보 등록 시각 "HH:MM"
    pending_candidate_price:     float = 0.0            # 후보 가격
    pending_candidate_remaining: int   = 0              # 남은 대기 봉 수
    pending_candidate_status:    Optional[str] = None   # "등록", "갱신", "취소"
    # ── [보완-3] 단기 구조 및 신뢰도 ─────────────────────────────
    micro_structure:       str   = "unknown"   # 최근 2피봇 기반 단기 구조
    structure_confidence:  float = 0.0         # 구조 판정 일관성 (0~1)
    # ── [보완-6] 잠정 S/R (pending 후보 기반) ───────────────────
    pending_support:    float = 0.0    # 잠정 지지 (pending low 후보)
    pending_resistance: float = 0.0   # 잠정 저항 (pending high 후보)


# ──────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────

class AdaptiveZigZag:
    """Adaptive ZigZag 지표 (모든 알려진 버그 수정 완료)."""

    def __init__(self, config: Optional[AdaptiveZigZagConfig] = None) -> None:
        self.config = config or AdaptiveZigZagConfig()
        self._symbol = "KP200 선물"  # 기본 심볼
        self._symbol_name = "ZIGZAG"  # 디버그 로그용 심볼 이름
        self._backtest_mode: bool = False  # 백테스트 모드 플래그 (look-ahead bias 방지)
        
        # ── [REGIME-INTEGRATION] 런타임 파라미터 (레짐 기반 오버라이드) ─────────
        # config를 직접 뮤테이션하지 않고 별도 딕셔너리로 관리
        self._runtime_params: Dict[str, Any] = {}
        
        # [FIX v3-6] __init__에서 full_reset() 직접 호출 (명확성 개선)
        self.full_reset()

        # ── [ATR-MONITOR] ATR 모니터 초기화 ─────────────────────────
        self._atr_monitor = ATRMonitor(spike_threshold_pct=30.0, ma_period=14)
        self._dynamic_atr_ratio: float = 0.0  # 급격 변동 시 동적 조정된 비율
        
        # 피봇 후보 수집기 초기화
        self._pivot_collector = None
        self._pivot_event_callback = None  # [PIVOT-EVENT-LOG] 피봇 이벤트 콜백
        if bool(getattr(self.config, "enable_pivot_collector", False)):
            try:
                from .pivot_collector import PivotCandidateCollector
                max_seq = int(getattr(self.config, "pivot_collector_max_sequence", 120) or 120)
                self._pivot_collector = PivotCandidateCollector(max_sequence_length=max_seq)
                _logger.info("[AdaptiveZigZag] 피봇 후보 수집기 활성화")
            except Exception as e:
                _logger.error("[AdaptiveZigZag] 피봇 후보 수집기 초기화 실패: %s", e)

        # ── [자기완결형 적응 엔진] ─────────────────────────────────────
        self._adaptive_engine = None
        try:
            from .adaptive_param_engine import AdaptiveParamEngine
            self._adaptive_engine = AdaptiveParamEngine(self.config)
            _logger.info("[AdaptiveZigZag] 적응형 파라미터 엔진 활성화")
        except Exception as e:
            _logger.warning("[AdaptiveZigZag] 적응형 파라미터 엔진 초기화 실패: %s", e)

        # 피봇 통계 카운터 (백테스트용)
        self._candidate_registered_count = 0
        self._candidate_cancelled_count = 0
        self._candidate_updated_count = 0
        
        # 텔레그램 이벤트 콜백 (피봇 후보 등록/갱신/취소 시 텔레그램 전송)
        self._telegram_event_callback: Optional[Callable[..., None]] = None

        # ── [다중 시간프레임] 초기화 ───────────────────────────────
        self._multi_tf_zz: Optional['MultiTimeframeZigZag'] = None
        self._upper_tf_zz_instances: Dict[int, 'AdaptiveZigZag'] = {}  # 상위 시간프레임 ZigZag 인스턴스
        self._upper_tf_data_buffers: Dict[int, List[Dict[str, float]]] = {}  # 상위 시간프레임 데이터 버퍼
        
        if bool(getattr(self.config, "multi_timeframe_enabled", False)):
            try:
                from .multi_timeframe_zigzag import MultiTimeframeZigZag
                scales = getattr(self.config, "multi_timeframe_scales", [1, 5, 15])
                # 1분봉은 제외하고 상위 시간프레임만 사용
                upper_scales = [s for s in scales if s > 1]
                threshold = getattr(self.config, "multi_timeframe_consensus_threshold", 2)
                
                price_tol = getattr(self.config, "multi_timeframe_price_tolerance_pct", 1.0)
                index_tol = getattr(self.config, "multi_timeframe_index_tolerance_multiplier", 2.0)
                
                self._multi_tf_zz = MultiTimeframeZigZag(
                    scales=upper_scales if upper_scales else [5, 15],
                    consensus_threshold=threshold,
                    price_tolerance_pct=price_tol,
                    index_tolerance_multiplier=index_tol
                )
                
                # 상위 시간프레임 ZigZag 인스턴스 생성
                for scale in upper_scales:
                    try:
                        # 상위 시간프레임 인스턴스는 다중 시간프레임 기능 비활성화 (무한 재귀 방지)
                        upper_config = AdaptiveZigZagConfig()
                        # 기본 설정 복사 (다중 시간프레임 제외)
                        upper_config.atr_multiplier = self.config.atr_multiplier
                        upper_config.confirmation_bars = self.config.confirmation_bars
                        upper_config.pivot_threshold_min_pct = self.config.pivot_threshold_min_pct
                        upper_config.pivot_threshold_max_pct = self.config.pivot_threshold_max_pct
                        upper_config.use_atr_based_filtering = self.config.use_atr_based_filtering
                        upper_config.major_swing_ratio = self.config.major_swing_ratio
                        upper_config.cluster_tolerance_pct = self.config.cluster_tolerance_pct
                        # 다중 시간프레임은 비활성화
                        upper_config.multi_timeframe_enabled = False
                        
                        upper_zz = AdaptiveZigZag(config=upper_config)
                        upper_zz.set_symbol(f"{self._symbol} ({scale}m)")
                        self._upper_tf_zz_instances[scale] = upper_zz
                        self._upper_tf_data_buffers[scale] = []
                        _logger.info("[AdaptiveZigZag] 상위 시간프레임 ZigZag 초기화: %d분봉", scale)
                    except Exception as e:
                        _logger.error("[AdaptiveZigZag] %d분봉 ZigZag 초기화 실패: %s", scale, e)
                
                _logger.info("[AdaptiveZigZag] 다중 시간프레임 결합 활성화: scales=%s, threshold=%d", upper_scales, threshold)
            except Exception as e:
                _logger.error("[AdaptiveZigZag] 다중 시간프레임 초기화 실패: %s", e)

    def set_atr_telegram_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """ATR 급격 변동 감지 시 텔레그램 콜백 설정."""
        self._atr_monitor.set_telegram_callback(callback)

    def set_symbol(self, symbol: str) -> None:
        """심볼 설정."""
        old_symbol = self._symbol
        self._symbol = symbol
        # [PIVOT-EVENT-LOG] _symbol_name도 업데이트 (로그용)
        self._symbol_name = symbol
        _logger.info("[AdaptiveZigZag] 심볼 변경: %s -> %s", old_symbol, symbol)

    def set_backtest_mode(self, enabled: bool) -> None:
        """백테스트 모드 설정.

        백테스트 모드에서는 look-ahead bias를 방지하기 위해
        _validate_direction_change()에서 미래 데이터 검증을 건너뜁니다.

        Args:
            enabled: 백테스트 모드 활성화 여부
        """
        self._backtest_mode = bool(enabled)
        _logger.info("[AdaptiveZigZag] 백테스트 모드: %s", "활성화" if enabled else "비활성화")
    
    def _notify_collector_candidate_registered(
        self,
        candidate_type: str,
        candidate_price: float,
        bar_idx: int,
        close: float,
    ) -> None:
        """후보 등록 시 수집기에 통지."""
        if self._pivot_collector is None:
            return
        
        try:
            candidate_id = self._pivot_collector.generate_candidate_id(
                candidate_type, bar_idx
            )
            timestamp = self._bar_hhmm(bar_idx) or "?"
            
            # 현재 피처 추출
            features = self.get_transformer_features(close)
            
            self._pivot_collector.on_candidate_registered(
                candidate_id=candidate_id,
                candidate_type=candidate_type,
                candidate_price=candidate_price,
                bar_idx=bar_idx,
                timestamp=timestamp,
                features=features,
                close=close,
            )
            
            # [PIVOT-EVENT-LOG] 후보 등록 시 직접 콜백 호출 (과거 데이터 replay 지원)
            if self._pivot_event_callback:
                try:
                    self._pivot_event_callback(
                        event_type="registered",
                        symbol=self._symbol,
                        candidate_type=candidate_type.upper(),
                        candidate_price=candidate_price,
                        bar_idx=bar_idx,
                        timestamp=timestamp,
                        reason=""
                    )
                    _logger.info("[AdaptiveZigZag] 피봇 후보 등록 콜백 호출 성공")
                except Exception as e:
                    _logger.warning("[AdaptiveZigZag] 피봇 후보 등록 콜백 호출 실패: %s", e)
            
            # 후보 ID 저장 (확정/취소 시 사용)
            self._current_candidate_id = candidate_id
        except Exception as e:
            _logger.error("[AdaptiveZigZag] 후보 등록 통지 실패: %s", e)
    
    def _notify_collector_candidate_confirmed(
        self,
        candidate_type: str,
        candidate_price: float,
        bar_idx: int,
        close: float,
    ) -> None:
        """후보 확정 시 수집기에 통지."""
        if self._pivot_collector is None:
            return
        
        try:
            candidate_id = getattr(self, "_current_candidate_id", None)
            if candidate_id is None:
                # 후보 ID가 없으면 생성
                candidate_id = self._pivot_collector.generate_candidate_id(
                    candidate_type, bar_idx
                )
            
            timestamp = self._bar_hhmm(self._bar_idx) or "?"
            
            self._pivot_collector.on_candidate_confirmed(
                candidate_id=candidate_id,
                confirmed_bar=self._bar_idx,
                confirmed_time=timestamp,
                confirmed_close=close,
                symbol=self._symbol,
            )
        except Exception as e:
            _logger.error("[AdaptiveZigZag] 후보 확정 통지 실패: %s", e)
    
    def _notify_collector_bar_update(self, close: float) -> None:
        """매 봉마다 수집기에 통지."""
        if self._pivot_collector is None or self._pending_confirm is None:
            return
        
        try:
            candidate_id = getattr(self, "_current_candidate_id", None)
            if candidate_id is None:
                return
            
            timestamp = self._bar_hhmm(self._bar_idx) or "?"
            features = self.get_transformer_features(close)
            
            self._pivot_collector.on_bar_update(
                candidate_id=candidate_id,
                bar_idx=self._bar_idx,
                timestamp=timestamp,
                features=features,
                close=close,
            )
        except Exception as e:
            _logger.error("[AdaptiveZigZag] 봉 업데이트 통지 실패: %s", e)
    
    def _notify_collector_confirmed(
        self,
        confirmed_bar: int,
        confirmed_close: float,
    ) -> None:
        """확정 시 수집기에 통지."""
        if self._pivot_collector is None:
            return
        
        try:
            candidate_id = getattr(self, "_current_candidate_id", None)
            if candidate_id is None:
                return
            
            confirmed_time = self._bar_hhmm(confirmed_bar) or "?"
            
            self._pivot_collector.on_candidate_confirmed(
                candidate_id=candidate_id,
                confirmed_bar=confirmed_bar,
                confirmed_time=confirmed_time,
                confirmed_close=confirmed_close,
                symbol=self._symbol,
            )
            
            self._current_candidate_id = None
        except Exception as e:
            _logger.error("[AdaptiveZigZag] 확정 통지 실패: %s", e)
    
    def _notify_collector_cancelled(
        self,
        cancelled_bar: int,
        cancelled_close: float,
        reason: str,
    ) -> None:
        """취소 시 수집기에 통지."""
        if self._pivot_collector is None:
            return
        
        try:
            candidate_id = getattr(self, "_current_candidate_id", None)
            if candidate_id is None:
                return
            
            cancelled_time = self._bar_hhmm(cancelled_bar) or "?"
            
            self._pivot_collector.on_candidate_cancelled(
                candidate_id=candidate_id,
                cancelled_bar=cancelled_bar,
                cancelled_time=cancelled_time,
                cancelled_close=cancelled_close,
                reason=reason,
                symbol=self._symbol,
            )
            
            # [PIVOT-EVENT-LOG] 취소 시 직접 콜백 호출 (과거 데이터 replay 지원)
            if self._pivot_event_callback:
                try:
                    # 후보 타입 추정 (candidate_id에서 추출)
                    candidate_type = "HIGH" if "high" in candidate_id.lower() else "LOW"
                    # 후보 가격 추정 (취소된 후보 가격을 저장해두지 않으면 알 수 없음)
                    # 여기서는 0.0을 전달하고 UI에서 처리
                    self._pivot_event_callback(
                        event_type="cancelled",
                        symbol=self._symbol,
                        candidate_type=candidate_type,
                        candidate_price=0.0,  # 취소된 후보 가격은 저장되지 않음
                        bar_idx=cancelled_bar,
                        timestamp=cancelled_time,
                        reason=reason
                    )
                    _logger.info("[AdaptiveZigZag] 피봇 후보 취소 콜백 호출 성공")
                except Exception as e:
                    _logger.warning("[AdaptiveZigZag] 피봇 후보 취소 콜백 호출 실패: %s", e)
            
            self._current_candidate_id = None
        except Exception as e:
            _logger.error("[AdaptiveZigZag] 취소 통지 실패: %s", e)

    def _format_hhmm(self, bar_time: Any) -> Optional[str]:
        try:
            if bar_time is None:
                return None
            ts = pd.Timestamp(bar_time)
            return ts.strftime("%H:%M")
        except Exception:
            try:
                s = str(bar_time).strip()
                return s[:5] if len(s) >= 5 else None
            except Exception:
                return None

    def _remember_bar_time(self, bar_time: Any) -> None:
        hhmm = self._format_hhmm(bar_time)
        _logger.debug("[ZZ][_remember_bar_time] bar_idx=%d, bar_time=%s, hhmm=%s, map_size=%d",
                      self._bar_idx, bar_time, hhmm, len(self._bar_hhmm_map))
        if hhmm:
            self._bar_hhmm_map[int(self._bar_idx)] = str(hhmm)
            # [OPT-1 FIX] KOSPI200 1분봉 1주일 기준 ~2000봉으로 상한 조정
            # 기존 4096은 과도하게 큼; 오래된 시각 정보는 렌더링에서 불필요
            while len(self._bar_hhmm_map) > 2000:
                self._bar_hhmm_map.popitem(last=False)

    def _bar_hhmm(self, idx: Any) -> Optional[str]:
        try:
            result = self._bar_hhmm_map.get(int(idx))
            _logger.debug("[ZZ][_bar_hhmm] idx=%s, result=%s, map_size=%d", idx, result, len(self._bar_hhmm_map))
            return result
        except Exception as e:
            _logger.debug("[ZZ][_bar_hhmm] error: %s", e)
            return None

    def _pending_status_kwargs(self, close: float) -> Dict[str, Any]:
        """현재 pending_confirm 상태를 로그 kwargs로 반환하는 내부 헬퍼.

        후보갱신/후보상태 로그에서 공통으로 사용한다.
        pending이 없으면 빈 딕셔너리 반환.
        """
        import math
        try:
            pc = self._pending_confirm
            if not isinstance(pc, dict) or not pc:
                return {}
            _pc_type  = str(pc.get("type") or "")
            _pc_price = float(pc.get("price") or 0.0)
            _pc_rem   = int(pc.get("remaining") or 0)
            _cb_f     = float(getattr(self.config, "confirmation_bars", 2) or 2)

            _dist_str = f"{(_pc_price - close) / close * 100:+.2f}%" if close > 0 and _pc_price > 0 else "?"
            _urgency  = round(max(0.0, min(1.0, 1.0 - _pc_rem / max(_cb_f, 1.0))), 3)

            try:
                _reg    = int(getattr(self, "_pending_confirm_registered_bar", -1))
                _waited = self._bar_idx - _reg if _reg >= 0 else -1
            except Exception:
                _waited = -1
            _age        = round(math.exp(-max(_waited, 0) / 5.0), 3) if _waited >= 0 else 0.0
            _waited_str = str(_waited) if _waited >= 0 else "?"
            _pt_dir     = "↓반전 가능성" if _pc_type == "high" else "↑반전 가능성"

            return dict(
                dist=_dist_str,
                urgency=_urgency,
                age=_age,
                waited=_waited_str,
                note=f"확정까지 {_pc_rem}봉 {_pt_dir}",
            )
        except Exception:
            return {}

    def _confirmed_pivot_summary_kwargs(self, tail_n: int = 8) -> Dict[str, Any]:
        """누적 확정 피봇 요약(kwargs) 반환."""
        try:
            swings = [s for s in (self._all_swings or []) if bool(getattr(s, "confirmed", False))]
            # anchor pivot 제외 (초기화용, 실시간 신호 아님) - index==0인 swing는 anchor
            swings = [s for s in swings if s.index != 0]
            n = int(len(swings))
            tail = swings[-max(1, int(tail_n)):] if n > 0 else []
            parts: List[str] = []
            for sw in tail:
                try:
                    _tp = "H" if sw.swing_type == SwingType.HIGH else "L"
                    _tm = self._bar_hhmm(sw.index) or "?"
                    parts.append(f"{_tp}@{_tm}:{float(sw.price):.2f}")
                except Exception:
                    continue
            return dict(
                confirmed_count=n,
                confirmed_tail=(" | ".join(parts) if parts else "none"),
            )
        except Exception:
            return dict(confirmed_count=0, confirmed_tail="none")

    def _pivot_trace_kwargs(self, close: float = 0.0) -> Dict[str, Any]:
        """피봇 디버그용 공통 트레이스(kwargs): pending + confirmed 요약."""
        out: Dict[str, Any] = {}
        try:
            out.update(self._pending_status_kwargs(float(close)))
        except Exception:
            pass
        try:
            out.update(self._confirmed_pivot_summary_kwargs())
        except Exception:
            pass
        return out

    def _pivot_event_emit(self, action: str, *, close: float = 0.0, **kwargs: Any) -> None:
        """피봇 이벤트 로그 출력 + 공통 트레이스 정보 병합.

        중복 로그를 피하기 위해 표준 출력은 cross-project 비교 포맷
        (`ZZ_ENGINE_*`, `ZZ_CANDIDATE`, `ZZ_CONFIRMED_PIVOTS`)만 사용한다.
        """
        payload: Dict[str, Any] = {}
        try:
            payload.update(kwargs or {})
        except Exception:
            pass
        try:
            payload.update(self._pivot_trace_kwargs(float(close)))
        except Exception:
            pass
        try:
            self._emit_cross_project_debug_logs(action=action, close=float(close), payload=payload)
        except Exception as e:
            _logger.error("[AdaptiveZigZag] _emit_cross_project_debug_logs 예외: action=%s, error=%s", action, e, exc_info=True)
        
        # 텔레그램 이벤트 콜백 호출 (피봇 후보 등록/갱신/취소 시)
        if self._telegram_event_callback is not None and action in ("후보등록", "후보갱신", "취소"):
            try:
                self._telegram_event_callback(action=action, close=float(close), payload=payload)
            except Exception as e:
                _logger.warning("[AdaptiveZigZag] 텔레그램 이벤트 콜백 호출 실패: %s", e)

    def _next_pivot_event_id(self) -> int:
        try:
            self._pivot_event_seq = int(getattr(self, "_pivot_event_seq", 0) or 0) + 1
        except Exception:
            self._pivot_event_seq = 1
        return int(self._pivot_event_seq)

    def _candidate_signature(self) -> str:
        try:
            pc = self._pending_confirm
            if not isinstance(pc, dict) or not pc:
                return "none"
            _tp = str(pc.get("type") or "")
            _idx = int(pc.get("idx") or -1)
            _pr = round(float(pc.get("price") or 0.0), 4)
            _rm = int(pc.get("remaining") or 0)
            return f"{_tp}|{_idx}|{_pr}|{_rm}"
        except Exception:
            return "none"

    def _confirmed_signature(self) -> str:
        try:
            swings = [s for s in (self._all_swings or []) if bool(getattr(s, "confirmed", False))]
            if not swings:
                return "none"
            parts: List[str] = []
            for sw in swings:
                _tp = "H" if sw.swing_type == SwingType.HIGH else "L"
                _idx = int(sw.index)
                _pr = round(float(sw.price), 4)
                parts.append(f"{_tp}|{_idx}|{_pr}")
            return ";".join(parts)
        except Exception:
            return "none"

    def _emit_cross_project_debug_logs(self, *, action: str, close: float, payload: Dict[str, Any]) -> None:
        """피봇 이벤트를 prediction.log 에 사람이 읽기 좋은 형태로 기록한다.

        출력 형식 (샘플)
        ---------
        [ZZ][후보등록] ZIGZAG | H@10:30=355.00 | 대기=2봉 | 역전임계=1.20pt | prob=0.75 | bar=142
        [ZZ][후보갱신] ZIGZAG | H@10:30=355.00 | 대기=1봉 | close=353.00 | bar=143
        [ZZ][취소]    ZIGZAG | H@10:30=355.00 취소 | 사유=반대후보교체 | bar=144
        [ZZ][확정]    ZIGZAG | H@10:30=355.00 확정✓ | 확정봉=10:32 | lag=2봉 | close=352.50 | bar=144
        [ZZ][피봇목록] ZIGZAG | 확정3개: 1)09:15 L 350.25 | 2)10:30 H 355.00 | 3)11:20 L 348.50
        """
        # 모든 피봇 이벤트 항상 기록 (pivot_lifecycle_log 설정 무시)
        # if action != "취소" and not bool(getattr(self.config, "pivot_lifecycle_log", False)):
        #     return
        # [PIVOT-EVENT-LOG] _symbol_name 사용 (set_symbol로 업데이트됨)
        prefix = self._symbol_name
        bar = int(self._bar_idx)
        p = payload or {}

        # ── 이벤트별 핵심 메시지 구성 ─────────────────────────────────────────
        try:
            if action == "후보등록":
                ctype      = str(p.get("candidate") or "").upper()[:1] or "?"
                swing_time = str(p.get("swing_time") or "?")
                swing_px   = float(p.get("swing_price") or 0.0)
                rem        = int(p.get("remaining") or 0)
                thr        = float(p.get("thr_abs") or 0.0)
                prob       = float(p.get("prob", 0.0) or 0.0)
                prob_str   = f" | prob={prob:.2f}" if prob > 0 else ""
                log_msg = f"[ZZ][후보등록] {prefix} | {ctype}@{swing_time}={swing_px:.2f} | 대기={rem}봉 | 역전임계={thr:.2f}pt{prob_str} | bar={bar}"
                _logger.debug(  # [FIX] 반복 로그 방지를 위해 DEBUG로 변경
                    "[ZZ][후보등록] %s | %s@%s=%.2f | 대기=%d봉 | 역전임계=%.2fpt%s | bar=%d",
                    prefix, ctype, swing_time, swing_px, rem, thr, prob_str, bar,
                )
                print(log_msg)

            elif action in ("후보갱신", "후보상태"):
                ctype      = str(p.get("candidate") or "").upper()[:1] or "?"
                swing_time = str(p.get("swing_time") or "?")
                swing_px   = float(p.get("swing_price") or 0.0)
                rem        = int(p.get("remaining") or 0)
                reason     = str(p.get("reason") or "")
                reason_str = f" ({reason})" if reason else ""
                prob       = float(p.get("prob", 0.0) or 0.0)
                prob_str   = f" | prob={prob:.2f}" if prob > 0 else ""
                log_msg = f"[ZZ][후보갱신] {prefix} | {ctype}@{swing_time}={swing_px:.2f} | 잔여={rem}봉 | close={float(close):.2f}{reason_str}{prob_str} | bar={bar}"
                _logger.debug(  # [FIX] 반복 로그 방지를 위해 DEBUG로 변경
                    "[ZZ][후보갱신] %s | %s@%s=%.2f | 잔여=%d봉 | close=%.2f%s%s | bar=%d",
                    prefix, ctype, swing_time, swing_px, rem, float(close), reason_str, prob_str, bar,
                )
                print(log_msg)

            elif action == "취소":
                # 취소된 피봇 정보
                ctype      = str(p.get("cancelled_type") or "").upper()[:1] or "?"
                swing_time = str(p.get("cancelled_time") or "?")
                swing_px   = float(p.get("cancelled_price") or 0.0)
                reason     = str(p.get("reason") or "사유불명")
                cancel_tm  = str(p.get("cancel_time") or self._bar_hhmm(bar) or "?")
                log_msg = f"[ZZ][취소]    {prefix} | {ctype}@{swing_time}={swing_px:.2f} 취소 | 사유={reason} | 취소봉={cancel_tm} | close={float(close):.2f} | bar={bar}"
                _logger.warning(  # [FIX] ZZLogFilter가 INFO 레벨을 필터링하므로 WARNING으로 변경
                    "[ZZ][취소] %s | %s@%s=%.2f 취소 | 사유=%s | 취소봉=%s | close=%.2f | bar=%d",
                    prefix, ctype, swing_time, swing_px, reason, cancel_tm, float(close), bar,
                )
                print(log_msg)

            elif action == "확정":
                stype      = str(p.get("swing_type") or "").upper()[:1] or "?"
                swing_time = str(p.get("swing_time") or "?")
                swing_px   = float(p.get("swing_price") or 0.0)
                signal     = str(p.get("signal") or "")
                confirm_tm = self._bar_hhmm(bar) or "?"
                lag        = max(0, bar - int(
                    (self._pending_confirm or {}).get("idx", bar) if isinstance(self._pending_confirm, dict) else bar
                ))
                mode       = str(p.get("mode") or "")
                # [FIX] 피봇 후보 등록 없이 바로 확정되는 경우 별도 로그 출력
                if mode == "초기범위":
                    log_msg = f"[ZZ][확정-무후보] {prefix} | {stype}@{swing_time}={swing_px:.2f} 확정✓ (후보등록없음) | 확정봉={confirm_tm} | lag={lag}봉 | close={float(close):.2f} | signal={signal} | bar={bar}"
                    _logger.debug(  # [FIX] 반복 로그 방지를 위해 DEBUG로 변경
                        "[ZZ][확정-무후보] %s | %s@%s=%.2f 확정✓ (후보등록없음) | 확정봉=%s | lag=%d봉 | close=%.2f | signal=%s | bar=%d",
                        prefix, stype, swing_time, swing_px, confirm_tm, lag, float(close), signal, bar,
                    )
                    print(log_msg)
                else:
                    log_msg = f"[ZZ][확정]    {prefix} | {stype}@{swing_time}={swing_px:.2f} 확정✓ | 확정봉={confirm_tm} | lag={lag}봉 | close={float(close):.2f} | signal={signal} | bar={bar}"
                    _logger.warning(  # [FIX] 중요 이벤트는 WARNING 유지
                        "[ZZ][확정] %s | %s@%s=%.2f 확정✓ | 확정봉=%s | lag=%d봉 | close=%.2f | signal=%s | bar=%d",
                        prefix, stype, swing_time, swing_px, confirm_tm, lag, float(close), signal, bar,
                    )
                    print(log_msg)

            else:
                # 기타 (후보상태 등)
                log_msg = f"[ZZ][{action}] {prefix} | bar={bar} | {' '.join(f'{k}={v}' for k, v in p.items() if v is not None)}"
                _logger.warning(  # [FIX] ZZLogFilter가 INFO 레벨을 필터링하므로 WARNING으로 변경
                    "[ZZ][%s] %s | bar=%d | %s", action, prefix, bar,
                    " ".join(f"{k}={v}" for k, v in p.items() if v is not None)
                )
                print(log_msg)

        except Exception as _e:
            _logger.debug("[ZZ][emit_error] action=%s err=%s", action, _e)

        # ── 피봇 목록 (확정 수 변경 시만 출력) ────────────────────────────────
        try:
            swings = [s for s in (self._all_swings or []) if bool(getattr(s, "confirmed", False))]
            swings = [s for s in swings if self._bar_hhmm(s.index) != "09:00"]
            conf_sig = self._confirmed_signature()
            if conf_sig != str(getattr(self, "_last_confirmed_sig", "") or ""):
                self._last_confirmed_sig = conf_sig
                if swings:
                    parts: List[str] = []
                    for i, sw in enumerate(swings, start=1):
                        _tp  = "H" if sw.swing_type == SwingType.HIGH else "L"
                        _tm  = self._bar_hhmm(sw.index) or "?"
                        _pr  = float(sw.price)
                        _maj = "★" if getattr(sw, "is_major", False) else " "
                        parts.append(f"{i}){_maj}{_tm} {_tp} {_pr:.2f}")
                    log_msg = f"[ZZ][피봇목록] {prefix} | 확정{len(parts)}개: {' | '.join(parts)}"
                    _logger.debug(  # [FIX] 반복 로그 방지를 위해 DEBUG로 변경
                        "[ZZ][피봇목록] %s | 확정%d개: %s",
                        prefix, len(parts), " | ".join(parts),
                    )
                    print(log_msg)
                else:
                    log_msg = f"[ZZ][피봇목록] {prefix} | 확정0개 (초기화 또는 전체취소)"
                    _logger.debug(  # [FIX] 반복 로그 방지를 위해 DEBUG로 변경
                        "[ZZ][피봇목록] %s | 확정0개 (초기화 또는 전체취소)", prefix
                    )
                    print(log_msg)
        except Exception as _e:
            _logger.debug("[ZZ][pivot_list_error] %s", _e)

    def _process_pending_confirmation(
        self,
        high: float,
        low: float,
        close: float,
        atr: float,
        thr_pct: float
    ) -> str:
        """
        Pending confirmation window 처리.

        Args:
            high: 현재 봉 고가
            low: 현재 봉 저가
            close: 현재 봉 종가
            atr: 현재 ATR 값
            thr_pct: 임계값 (%)

        Returns:
            new_swing_signal: "new_high", "new_low", 또는 "none"
        """
        cfg = self.config
        new_swing_signal = "none"

        if not isinstance(self._pending_confirm, dict) or not self._pending_confirm:
            return new_swing_signal

        stype   = self._pending_confirm.get("type")
        rem     = int(self._pending_confirm.get("remaining") or 0)
        c_price = float(self._pending_confirm.get("price") or 0.0)
        c_idx   = int(self._pending_confirm.get("idx") or -1)
        c_atr   = float(self._pending_confirm.get("atr") or atr)

        # [FIX-7] max_wait_bars 초과 시 pending 자동취소
        # 0=무제한(기본). >0이면 등록 후 해당 봉 수 경과 시 취소
        # [EDGE-CASE-2] 무한 추세 시 확정 지연 방지: max_wait_bars=0이어도 ATR 주기 연동 기본값 적용
        _max_wait = int(getattr(cfg, "max_wait_bars", 0) or 0)
        if _max_wait == 0:
            # ATR 주기에 연동하여 타임프레임에 상관없이 적응적 방어
            _max_wait = int(getattr(cfg, "atr_period", ZigZagConstants.DEFAULT_ATR_PERIOD) or ZigZagConstants.DEFAULT_ATR_PERIOD) * ZigZagConstants.ATR_PERIOD_MULTIPLIER
        if self._pending_confirm_registered_bar >= 0:
            _waited = self._bar_idx - self._pending_confirm_registered_bar
            if _waited >= _max_wait:
                # ZigZagState 피봇 후보 상태 업데이트 (취소)
                self._state.pending_candidate_status = "취소"
                self._pivot_event_emit(
                    "취소",
                    close=close,
                    reason="max_wait_bars",
                    prev_type=stype,
                    prev_time=self._bar_hhmm(c_idx),
                    prev_price=c_price,
                    waited=_waited,
                    max_wait=_max_wait,
                )
                self._candidate_cancelled_count += 1
                # Collector 통지
                self._notify_collector_cancelled(
                    cancelled_bar=self._bar_idx,
                    cancelled_close=close,
                    reason="max_wait_bars",
                )
                self._pending_confirm = None
                self._pending_confirm_registered_bar = -1
                return new_swing_signal  # 취소 시 남은 처리 건너뜀

        # [DESIGN-1] freeze_on_confirm 로직 분기 단순화
        freeze_on_confirm = bool(getattr(cfg, "freeze_on_confirm", True))
        updated = False

        if not freeze_on_confirm:
            # freeze=False 경로: 신고점/신저점 갱신 허용
            if stype == "high" and high > c_price:
                c_price = high; c_idx = self._bar_idx; c_atr = atr; updated = True
            elif stype == "low" and low < c_price:
                c_price = low; c_idx = self._bar_idx; c_atr = atr; updated = True
        # freeze=True 경로: c_price/c_idx/c_atr 갱신 없음 (등록 시점 값 유지)

        rem -= 1
        # [BUG-4] freeze=False 시 신고점/신저점 갱신 → remaining 부분 리셋
        # 직전 봉에 극값이 오면 rem=0이 되어 즉시 확정되는 것을 방지
        # rem > 0 조건: 이미 0 이하면 이번 봉 확정 우선
        if updated and rem > 0:
            _cb = int(cfg.confirmation_bars or 2)
            _reset_to = max(1, _cb // 2 + 1)
            if rem < _reset_to:
                rem = _reset_to
        self._pending_confirm["remaining"] = rem
        if updated:
            self._pending_confirm.update(price=c_price, idx=c_idx, atr=c_atr)
            self._candidate_updated_count += 1
            self._pivot_event_emit(
                "후보갱신",
                close=close,
                candidate=stype,
                swing_time=self._bar_hhmm(c_idx),
                swing_price=round(c_price, 4),
                remaining=rem,
                prob=round(self.get_pending_confirmation_probability(close), 3),
            )

        if rem <= 0:
            # [FIX v2-2] 확정 시점에 해당 봉의 실제 고가/저가를 사용하도록 수정
            # freeze_on_confirm 설정과 관계없이 항상 실제 봉의 극값을 사용
            # [FIX v2-2] 절대 인덱스 → deque 상대 인덱스 변환
            base_offset = self._bar_idx - len(self._highs)
            relative_c_idx = c_idx - base_offset
            if 0 <= relative_c_idx < len(self._highs):
                actual_high = self._highs[relative_c_idx]
                actual_low = self._lows[relative_c_idx]
            else:
                # 인덱스 범위를 벗어나면 c_price 사용 (fallback)
                actual_high = c_price
                actual_low = c_price

            if stype == "high":
                # 디버그 로그: H 피봇 확정 시점
                logger.debug("[ZZ][확정]    %s | H@%s=%.2f 확정✓ | 확정봉=%s | lag=%d봉 | close=%.2f | signal=%s | bar=%d      ",
                    self._symbol_name, self._bar_hhmm(c_idx), actual_high, self._bar_hhmm(self._bar_idx),
                    max(0, self._bar_idx - c_idx), close, new_swing_signal if 'new_swing_signal' in locals() else 'N/A', self._bar_idx)
                
                # ── [다중 시간프레임] 합의도 확인 (State 업데이트 전) ──────────
                consensus_passed = self._check_multiframe_consensus(c_idx, actual_high, "H", close)
                
                # 합의도 미통과 시 피봇 필터링
                if not consensus_passed:
                    logger.warning("[ZZ][필터링] %s 피봇 합의도 부족으로 필터링: H@%s=%.2f", self._symbol_name, self._bar_hhmm(c_idx), actual_high)
                    # [BUG-2 FIX] direction을 반전해 탐색 방향 복원 (BUG-1과 동일 패턴)
                    # 미복원 시: direction=1 유지 → 다음 봉 재등록 → 반복 합의 실패
                    self._pending_confirm = None
                    self._pending_confirm_registered_bar = -1
                    self._current_direction = -1
                    self._pending_low       = low
                    self._pending_low_idx   = self._bar_idx
                    self._pending_high      = 0.0
                    self._pending_high_idx  = -1
                    return "none"
                
                # 합의도 통과 후에만 State 업데이트
                self._state.last_swing_high     = actual_high
                self._state.last_swing_high_idx = c_idx
                
                added = self._add_swing(c_idx, actual_high, SwingType.HIGH, c_atr, confirmed_at_idx=self._bar_idx, confirmed_close=self._last_bar_close)
                if added:
                    new_swing_signal = "new_high"
                    # 완결봉 정보: 피봇이 확정되는 현재 봉의 OHLC + 시각
                    self._state.last_swing_high_time         = self._bar_hhmm(c_idx) or self._bar_hhmm(self._bar_idx)
                    self._state.last_swing_high_confirm_time = self._bar_hhmm(self._bar_idx)
                    self._state.last_swing_high_lag_bars     = max(0, self._bar_idx - c_idx)
                    self._state.last_swing_high_open         = self._last_bar_open
                    self._state.last_swing_high_close        = self._last_bar_close
                    # 정상 확정: HIGH → 다음은 LOW 탐색
                    self._current_direction = -1
                    self._pending_low       = low
                    self._pending_low_idx   = self._bar_idx
                    # [PIVOT-EVENT-LOG] 피봇 확정 시 수집기에 통지
                    self._notify_collector_candidate_confirmed(
                        candidate_type="high",
                        candidate_price=actual_high,
                        bar_idx=c_idx,
                        close=self._last_bar_close,
                    )
                    # [PIVOT-EVENT-LOG] 피봇 확정 시 직접 콜백 호출 (과거 데이터 replay 지원)
                    _logger.info("[AdaptiveZigZag] 피봇 확정 콜백 호출 시도: callback=%s", self._pivot_event_callback is not None)
                    if self._pivot_event_callback:
                        try:
                            self._pivot_event_callback(
                                event_type="confirmed",
                                symbol=self._symbol,
                                candidate_type="HIGH",
                                candidate_price=actual_high,
                                bar_idx=c_idx,
                                timestamp=self._bar_hhmm(self._bar_idx) or "?",
                                reason=""
                            )
                            _logger.info("[AdaptiveZigZag] 피봇 확정 콜백 호출 성공")
                        except Exception as e:
                            _logger.warning("[AdaptiveZigZag] 피봇 이벤트 콜백 호출 실패: %s", e)
                    else:
                        _logger.warning("[AdaptiveZigZag] 피봇 확정 콜백이 설정되지 않음")
                    self._pending_high      = 0.0
                    self._pending_high_idx  = -1
                    self._last_confirmed_bar_idx = self._bar_idx
                else:
                    # [BUG-1 FIX] added=False(교번 거부/병합) 시 direction을 반전해 탐색 방향 복원
                    # 미복원 시: direction=1 유지 → 다음 봉에서 또 HIGH 후보 등록 → 교번 루프
                    logger.info(
                        "[ZZ][BUG1-FIX] HIGH _add_swing 거부 → direction -1 강제 전환 | bar=%d",
                        self._bar_idx,
                    )
                    self._current_direction = -1
                    self._pending_low       = low
                    self._pending_low_idx   = self._bar_idx
                    self._pending_high      = 0.0
                    self._pending_high_idx  = -1
            elif stype == "low":
                # 디버그 로그: L 피봇 확정 시점
                logger.debug("[ZZ][확정]    %s | L@%s=%.2f 확정✓ | 확정봉=%s | lag=%d봉 | close=%.2f | signal=%s | bar=%d      ",
                    self._symbol_name, self._bar_hhmm(c_idx), actual_low, self._bar_hhmm(self._bar_idx),
                    max(0, self._bar_idx - c_idx), close, new_swing_signal if 'new_swing_signal' in locals() else 'N/A', self._bar_idx)
                
                # ── [다중 시간프레임] 합의도 확인 (State 업데이트 전) ──────────
                consensus_passed = self._check_multiframe_consensus(c_idx, actual_low, "L", close)
                
                # 합의도 미통과 시 피봇 필터링
                if not consensus_passed:
                    logger.warning("[ZZ][필터링] %s 피봇 합의도 부족으로 필터링: L@%s=%.2f", self._symbol_name, self._bar_hhmm(c_idx), actual_low)
                    # [BUG-2 FIX] direction을 반전해 탐색 방향 복원
                    self._pending_confirm = None
                    self._pending_confirm_registered_bar = -1
                    self._current_direction = 1
                    self._pending_high      = high
                    self._pending_high_idx  = self._bar_idx
                    self._pending_low       = float("inf")
                    self._pending_low_idx   = -1
                    return "none"
                
                # 합의도 통과 후에만 State 업데이트
                self._state.last_swing_low     = actual_low
                self._state.last_swing_low_idx = c_idx
                
                added = self._add_swing(c_idx, actual_low, SwingType.LOW, c_atr, confirmed_at_idx=self._bar_idx, confirmed_close=self._last_bar_close)
                if added:
                    new_swing_signal = "new_low"
                    # 완결봉 정보
                    self._state.last_swing_low_time         = self._bar_hhmm(c_idx) or self._bar_hhmm(self._bar_idx)
                    self._state.last_swing_low_confirm_time = self._bar_hhmm(self._bar_idx)
                    self._state.last_swing_low_lag_bars     = max(0, self._bar_idx - c_idx)
                    self._state.last_swing_low_open         = self._last_bar_open
                    self._state.last_swing_low_close        = self._last_bar_close
                    # 정상 확정: LOW → 다음은 HIGH 탐색
                    self._current_direction = 1
                    self._pending_high      = high
                    self._pending_high_idx  = self._bar_idx
                    self._pending_low       = float("inf")
                    self._pending_low_idx   = -1
                    # [PIVOT-EVENT-LOG] 피봇 확정 시 수집기에 통지
                    self._notify_collector_candidate_confirmed(
                        candidate_type="low",
                        candidate_price=actual_low,
                        bar_idx=c_idx,
                        close=self._last_bar_close,
                    )
                    # [PIVOT-EVENT-LOG] 피봇 확정 시 직접 콜백 호출 (과거 데이터 replay 지원)
                    _logger.info("[AdaptiveZigZag] 피봇 확정 콜백 호출 시도: callback=%s", self._pivot_event_callback is not None)
                    if self._pivot_event_callback:
                        try:
                            self._pivot_event_callback(
                                event_type="confirmed",
                                symbol=self._symbol,
                                candidate_type="LOW",
                                candidate_price=actual_low,
                                bar_idx=c_idx,
                                timestamp=self._bar_hhmm(self._bar_idx) or "?",
                                reason=""
                            )
                            _logger.info("[AdaptiveZigZag] 피봇 확정 콜백 호출 성공")
                        except Exception as e:
                            _logger.warning("[AdaptiveZigZag] 피봇 이벤트 콜백 호출 실패: %s", e)
                    else:
                        _logger.warning("[AdaptiveZigZag] 피봇 확정 콜백이 설정되지 않음")
                    self._last_confirmed_bar_idx = self._bar_idx
                else:
                    # [BUG-1 FIX] added=False(교번 거부/병합) 시 direction을 반전해 탐색 방향 복원
                    logger.info(
                        "[ZZ][BUG1-FIX] LOW _add_swing 거부 → direction 1 강제 전환 | bar=%d",
                        self._bar_idx,
                    )
                    self._current_direction = 1
                    self._pending_high      = high
                    self._pending_high_idx  = self._bar_idx
                    self._pending_low       = float("inf")
                    self._pending_low_idx   = -1
            # 이벤트 발행 시에도 실제 봉의 극값 사용
            event_price = actual_high if stype == "high" else actual_low
            # lag 미리 계산 (순서 의존성 방지)
            _lag = max(0, self._bar_idx - c_idx)
            self._pivot_event_emit(
                "확정",
                close=close,
                mode="pending_confirm",
                signal=new_swing_signal,
                swing_type=stype,
                swing_time=self._bar_hhmm(c_idx),
                swing_price=round(event_price, 4),
                thr_pct=round(thr_pct, 4),
                lag=_lag,
            )
            # Collector 통지
            self._notify_collector_confirmed(
                confirmed_bar=self._bar_idx,
                confirmed_close=close,
            )
            self._pending_confirm = None
            self._pending_confirm_registered_bar = -1

        # 디버그 로그: 피봇 확정 후 피봇 목록 출력
        confirmed_swings = [s for s in self._all_swings if s.confirmed]
        if confirmed_swings:
            pivot_list_str = " | ".join([
                f"{i+1})★{self._bar_hhmm(s.index)} {'H' if s.swing_type == SwingType.HIGH else 'L'} {s.price:.2f}"
                for i, s in enumerate(confirmed_swings)
            ])
            logger.debug("[ZZ][피봇목록] %s | 확정%d개: %s",
                self._symbol_name, len(confirmed_swings), pivot_list_str)

        return new_swing_signal


    def _update_multi_timeframe(self, high: float, low: float, close: float, bar_time: Any, open: float, volume: float) -> None:
        """다중 시간프레임 데이터 업데이트"""
        # [OPT-4 FIX] multi_tf_zz 비활성화 시(기본값) 함수 호출 자체를 생략
        if self._multi_tf_zz is not None:
            self._update_upper_timeframe_data(high, low, close, bar_time, open, volume)

    def _update_runtime_params(self, bar_time: Any) -> None:
        """런타임 파라미터 계산 (시간대별 캐싱 포함)"""
        _cur_hhmm = self._format_hhmm(bar_time) or ""
        _cached_hhmm = getattr(self, '_runtime_params_hhmm', None)
        if _cur_hhmm != _cached_hhmm or not self._runtime_params:
            self._runtime_params = self._get_runtime_params(bar_time)
            self._runtime_params_hhmm = _cur_hhmm

    def _finalize_update(self, close: float, atr: float, thr_pct: float, new_swing_signal: str) -> ZigZagState:
        """파동 크기, 피보나치, S/R, 구조 분석 및 상태 갱신"""
        # 4. 파동 크기
        wave_size, wave_pct = self._calculate_wave_size(close)

        # 5-7. 피보나치, S/R, 구조
        fib_levels = self._calc_fibonacci()
        support, resistance = self._find_nearest_sr(close)
        # [DESIGN-4 FIX] _enforce_hl_alternation을 매 봉 O(N log N) 대신
        # 피봇 변경이 실제로 발생한 경우에만 호출 (new_swing_signal != "none")
        # 이미 _add_swing()이 교번을 1차 방어하므로 사후 보정은 변경 시에만 충분
        if new_swing_signal != "none":
            self._enforce_hl_alternation()
        structure, hh, ll, micro_structure, structure_confidence = self._analyze_market_structure(new_swing_signal)

        # 8. 상태 갱신
        self._update_state(close, wave_size, wave_pct, thr_pct, atr, new_swing_signal,
                           fib_levels, support, resistance, structure, hh, ll,
                           micro_structure, structure_confidence)

        # Collector 통지 (매 봉마다)
        self._notify_collector_bar_update(float(close))

        # _bar_idx 증가: 다음 봉을 위한 카운터 증가
        # 세션 리셋 시에도 계속 증가하여 인덱스 충돌 방지
        self._bar_idx += 1
        return self._state

    def _process_direction_based_pivots(self, high: float, low: float, close: float, atr: float, thr_abs: float, thr_pct: float, new_swing_signal: str) -> str:
        """방향 기반 피봇 처리 (direction=0, 1, -1)"""
        # [P-FIX-B] 이번 봉에서 3-a 확정이 발생했으면 3-b 진입 차단
        # → 확정과 신규 후보등록이 같은 봉에 혼재되는 것을 방지
        # [BUG-INIT-DIR0] 수정: direction=0 초기범위 확정도 같은 봉 재진입 차단에 포함.
        #   new_swing_signal != "none" 조건이 초기범위 확정 경우도 커버하므로
        #   아래 direction==1/-1 블록의 P-FIX-B(_bar_idx <= _last_confirmed_bar_idx)와
        #   이중 방어를 구성한다. 초기범위 확정 즉시 _last_confirmed_bar_idx=self._bar_idx
        #   를 세팅하므로 다음 봉에서도 동일 봉 차단이 올바르게 동작한다.
        if new_swing_signal != "none":
            pass  # 다음 봉부터 탐색 재개 (3-a pending_confirm 확정 및 초기범위 확정 모두 포함)
        elif self._current_direction == 0:
            new_swing_signal = self._process_direction_zero(high, low, close, atr, thr_abs, thr_pct)
        elif self._current_direction == 1:
            new_swing_signal = self._process_direction_one(high, low, close, atr, thr_abs, thr_pct)
        elif self._current_direction == -1:
            new_swing_signal = self._process_direction_minus_one(high, low, close, atr, thr_abs, thr_pct)
        return new_swing_signal

    def _process_pending_confirmation_with_error_handling(self, high: float, low: float, close: float, atr: float, thr_pct: float) -> str:
        """pending confirmation 처리 및 예외 핸들링"""
        new_swing_signal = "none"
        try:
            new_swing_signal = self._process_pending_confirmation(high, low, close, atr, thr_pct)
        except Exception as exc:
            # [FIX v2-8] 예외 발생 시 new_swing_signal은 초기값 "none" 유지
            # _process_pending_confirmation 내부에서 이미 self._pending_confirm = None 설정 경로가 있으나
            # 예외 핸들러에서도 안전하게 None으로 설정 (중복이지만 안전)
            pc_err = self._pending_confirm if isinstance(self._pending_confirm, dict) else None
            if pc_err:
                # ZigZagState 피봇 후보 상태 업데이트 (취소)
                self._state.pending_candidate_status = "취소"
                self._pivot_event_emit(
                    "취소",
                    close=float(close),
                    reason="pending_confirm_exception",
                    prev_type=pc_err.get("type"),
                    prev_time=self._bar_hhmm(pc_err.get("idx")),
                    prev_price=pc_err.get("price"),
                    error=str(exc)[:160],
                )
                self._candidate_cancelled_count += 1
                # Collector 통지
                self._notify_collector_cancelled(
                    cancelled_bar=self._bar_idx,
                    cancelled_close=float(close),
                    reason="pending_confirm_exception",
                )
            _logger.warning("ZigZag pending_confirm error at bar %d: %s", self._bar_idx, exc)
            self._pending_confirm = None
            self._pending_confirm_registered_bar = -1
        return new_swing_signal

    def _calculate_atr_and_threshold(self, high: float, low: float, close: float, bar_time: Any) -> tuple:
        """ATR 계산 및 적응형 임계값 계산"""
        # 1. True Range & ATR
        atr = self._update_atr(high, low, close)

        # ── [Layer A × Layer B] 런타임 파라미터 계산 ───────────────────────
        self._update_runtime_params(bar_time)

        # ── [ATR-MONITOR] ATR 변화 추적 ─────────────────────────
        atr_monitor_result = self._update_atr_monitor(atr)

        # ── [ATR-MONITOR] 급격 변동 시 동적 비율 조정 ─────────────────
        # ATR 급변 시 배율 저장 (시간대 테이블 값에 적용됨)
        # [REVIEW-FIX-2] 논리적 일관성 복구: 변동성 증가 시 임계값 높여 노이즈 억제
        self._update_dynamic_atr_ratio(atr_monitor_result)

        # 2. 적응형 임계값  [FIX-1: ER 방향 역전 수정]
        # [FIX v3-5] 명시적 bar_idx 전달로 암묵적 의존성 제거
        thr_pct = self._calc_threshold_pct(atr, close, self._bar_idx)
        thr_abs = close * thr_pct / 100.0

        return atr, thr_abs, thr_pct

    def _buffer_and_check_initial_bars(self, high: float, low: float, close: float, open: float, volume: float, bar_time: Any) -> bool:
        """데이터 버퍼링 및 초기 봉 체크 - False 반환 시 update 중단"""
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._opens.append(float(open) if open else 0.0)  # [REGIME-INTEGRATION] 시가 버퍼에 추가
        self._volumes.append(volume)  # [REGIME-INTEGRATION] 거래량 버퍼에 추가
        self._remember_bar_time(bar_time)
        # 완결봉 OHLC 보관 (피봇 확정 시 State에 기록)
        self._last_bar_open  = float(open) if open else 0.0
        self._last_bar_high  = high
        self._last_bar_low   = low
        self._last_bar_close = close
        n = len(self._closes)

        # [FIX] 첫 번째, 두 번째 봉에서 피봇 생성 방지 - 시가 anchor를 3번째 봉에서 심도록 수정
        # 초기 데이터 누적 후 anchor 심도록 하여 장 초반 잘못된 피봇 방지
        if n <= 2:
            self._bar_idx += 1
            return False

        # [FIX] 장 시작 전 시간 피봇 생성 방지 - 심볼별 장 시작 시간 적용
        if not self._check_market_open_time(bar_time):
            self._bar_idx += 1
            return False

        # [DEBUG] 피봇 생성 추적
        if n == 3 or n % 50 == 0:
            _logger.debug("[ZZ][DEBUG] update: n=%d, h=%.2f, l=%.2f, c=%.2f, _current_direction=%s, _pending_high=%.2f, _pending_low=%.2f, _all_swings=%d",
                        n, high, low, close, self._current_direction, self._pending_high, self._pending_low, len(self._all_swings))

        return True

    def _update_state(self, close: float, wave_size: float, wave_pct: float, thr_pct: float, atr: float,
                      new_swing_signal: str, fib_levels, support, resistance,
                      structure, hh, ll, micro_structure, structure_confidence) -> None:
        """ZigZagState 상태 갱신"""
        s = self._state
        s.current_direction      = self._current_direction
        s.wave_size              = wave_size
        s.wave_size_pct          = wave_pct
        s.wave_direction         = self._current_direction
        s.fib_levels             = fib_levels
        s.nearest_support        = support
        s.nearest_resistance     = resistance
        s.support_dist_pct       = (close - support) / close * 100 if close > 0 and support > 0 else 0.0
        s.resistance_dist_pct    = (resistance - close) / close * 100 if close > 0 and resistance > 0 else 0.0
        s.is_making_higher_highs = hh
        s.is_making_lower_lows   = ll
        s.structure              = structure
        s.micro_structure        = micro_structure        # [보완-3]
        s.structure_confidence   = structure_confidence  # [보완-3]
        s.adaptive_threshold_pct = thr_pct
        s.atr                    = atr
        s.new_swing_signal       = new_swing_signal
        # [REVIEW-FIX-5] 스윙 버전 동기화
        s.swing_version          = self._swing_version
        # [BUG-CLUSTER-1] recent_swings deepcopy: _add_swing()에서 in-place 갱신 시
        # 외부 참조 오염 방지를 위해 deepcopy 사용
        # [PERF-OPT] 백테스트 모드에서는 얕은 복사 사용으로 성능 최적화
        cfg = self.config
        if self._backtest_mode:
            # 백테스트 모드: 읽기 전용 접근이므로 얕은 복사 사용
            s.recent_swings = list(self._all_swings[-cfg.max_swings:]) if self._all_swings else []
        else:
            # 실시간 모드: in-place 갱신 가능하므로 deepcopy 사용
            s.recent_swings = copy.deepcopy(self._all_swings[-cfg.max_swings:]) if self._all_swings else []
        s.bars_since_last_swing  = max(0, self._bar_idx - self._last_confirmed_bar_idx) \
                                   if self._last_confirmed_bar_idx >= 0 else 0
        try:
            _conf = [sw for sw in (self._all_swings or []) if bool(getattr(sw, "confirmed", False))]
            _tail = _conf[-6:]
            _parts: List[str] = []
            for sw in _tail:
                _tp = "H" if sw.swing_type == SwingType.HIGH else "L"
                _pivot_tm = self._bar_hhmm(sw.index) or "?"
                _confirm_tm = self._bar_hhmm(getattr(sw, "confirmed_at_idx", -1)) or "?"
                _parts.append(f"{_tp}@{_pivot_tm}->{_confirm_tm}:{float(sw.price):.2f}")
            s.confirmed_pivot_count = int(len(_conf))
            s.confirmed_pivot_tail_hhmm = " | ".join(_parts)
        except Exception:
            s.confirmed_pivot_count = 0
            s.confirmed_pivot_tail_hhmm = ""

    def _analyze_market_structure(self, new_swing_signal: str) -> tuple:
        """시장 구조 분석"""
        if new_swing_signal != "none":
            structure = self._analyze_structure()
            hh = self._is_higher_highs()
            ll = self._is_lower_lows()
            micro_structure      = self._analyze_micro_structure()   # [보완-3]
            structure_confidence = self._calc_structure_confidence()  # [보완-3]
        else:
            structure            = str(self._state.structure)
            hh = bool(self._state.is_making_higher_highs)
            ll = bool(self._state.is_making_lower_lows)
            micro_structure      = str(self._state.micro_structure)         # [보완-3]
            structure_confidence = float(self._state.structure_confidence)  # [보완-3]
        return structure, hh, ll, micro_structure, structure_confidence

    def _calculate_wave_size(self, close: float) -> tuple:
        """파동 크기 계산"""
        if self._state.last_swing_high > 0 and self._state.last_swing_low > 0:
            wave_size = self._state.last_swing_high - self._state.last_swing_low
            mid = (self._state.last_swing_high + self._state.last_swing_low) / 2.0
            wave_pct = wave_size / mid * 100.0 if mid > 0 else 0.0
        else:
            wave_size = wave_pct = 0.0
        return wave_size, wave_pct

    def _process_direction_minus_one(self, high: float, low: float, close: float, atr: float, thr_abs: float, thr_pct: float) -> str:
        """direction=-1 하락 방향 처리"""
        new_swing_signal = "none"
        if low < self._pending_low:
            self._pending_low = low; self._pending_low_idx = self._bar_idx
        if high - self._pending_low >= thr_abs:
            if self._is_wave_length_ok(thr_abs, close,
                                       candidate_idx=self._pending_low_idx):
                # [FIX-2] 반대 타입인 경우 교체 허용
                # [P-FIX-B] 피봇 확정 봉과 동일 봉에서 후보 등록 차단
                if self._bar_idx <= self._last_confirmed_bar_idx and self._last_confirmed_bar_idx >= 0:
                    # 피봇 확정 봉과 동일하거나 이전 봉이면 후보 등록 차단
                    pass
                else:
                    if self._pending_confirm is None or self._pending_confirm.get("type") != "low":
                        # [BUG-REMAIN-1] 교번 사전 검증
                        if self._would_violate_alternation(SwingType.LOW):
                            pass  # 등록 차단 — _add_swing() FIX-ALT-1이 2차 방어
                        else:
                            # ── 이하 전체가 else 블록 내부 ──────────────────
                            old_pc = self._pending_confirm
                            if isinstance(old_pc, dict) and old_pc.get("type") == "high":
                                try:
                                    cancelled_info = {
                                        "type": "high",
                                        "time": self._bar_hhmm(old_pc.get("idx")),
                                        "price": round(old_pc.get("price") or 0.0, 4),
                                        "cancel_time": self._bar_hhmm(self._bar_idx),
                                        "cancel_price": round(close, 4),
                                        "reason": "반대후보교체",
                                    }
                                    self._state.cancelled_candidates.append(cancelled_info)
                                except Exception:
                                    pass
                                self._state.pending_candidate_status = "취소"
                                self._pivot_event_emit(
                                    "취소",
                                    close=close,
                                    reason="반대후보교체",
                                    cancelled_type="high",
                                    cancelled_time=self._bar_hhmm(old_pc.get("idx")),
                                    cancelled_price=round(old_pc.get("price") or 0.0, 4),
                                )
                                self._candidate_cancelled_count += 1
                                self._notify_collector_cancelled(
                                    cancelled_bar=self._bar_idx,
                                    cancelled_close=close,
                                    reason="반대후보교체",
                                )
                            if self._pending_low_idx != -1:
                                swing_time_low = self._bar_hhmm(self._pending_low_idx)
                                self._pending_confirm = dict(
                                    type="low",
                                    idx=self._pending_low_idx,
                                    price=self._pending_low,
                                    atr=atr,
                                    remaining=self._calc_confirmation_bars(),
                                )
                                self._pending_confirm_registered_bar = self._bar_idx
                                self._candidate_registered_count += 1
                                self._state.pending_candidate_type = "low"
                                self._state.pending_candidate_time = swing_time_low
                                self._state.pending_candidate_price = round(float(self._pending_low), 4)
                                self._state.pending_candidate_remaining = self._calc_confirmation_bars()
                                self._state.pending_candidate_status = "등록"
                                self._pivot_event_emit(
                                    "후보등록",
                                    close=float(close),
                                    candidate="low",
                                    swing_time=swing_time_low,
                                    swing_price=round(float(self._pending_low), 4),
                                    remaining=self._calc_confirmation_bars(),
                                    thr_abs=round(float(thr_abs), 4),
                                    thr_pct=round(float(thr_pct), 4),
                                    prob=round(self.get_pending_confirmation_probability(float(close)), 3),
                                )
                                self._notify_collector_candidate_registered(
                                    candidate_type="low",
                                    candidate_price=float(self._pending_low),
                                    bar_idx=self._pending_low_idx,
                                    close=float(close),
                                )
        return new_swing_signal

    def _process_direction_one(self, high: float, low: float, close: float, atr: float, thr_abs: float, thr_pct: float) -> str:
        """direction=1 상승 방향 처리"""
        new_swing_signal = "none"
        if high > self._pending_high:
            self._pending_high = high; self._pending_high_idx = self._bar_idx
        if self._pending_high - low >= thr_abs:
            if self._is_wave_length_ok(thr_abs, close,
                                       candidate_idx=self._pending_high_idx):
                # [FIX-2] 반대 타입인 경우 교체 허용
                if self._bar_idx <= self._last_confirmed_bar_idx and self._last_confirmed_bar_idx >= 0:
                    # 피봇 확정 봉과 동일하거나 이전 봉이면 후보 등록 차단
                    pass
                else:
                    if self._pending_confirm is None or self._pending_confirm.get("type") != "high":
                        if self._would_violate_alternation(SwingType.HIGH):
                            pass  # 등록 차단 — _add_swing() FIX-ALT-1이 2차 방어
                        else:
                            old_pc = self._pending_confirm
                            if isinstance(old_pc, dict) and old_pc.get("type") == "low":
                                try:
                                    cancelled_info = {
                                        "type": "low",
                                        "time": self._bar_hhmm(old_pc.get("idx")),
                                        "price": round(float(old_pc.get("price") or 0.0), 4),
                                        "cancel_time": self._bar_hhmm(self._bar_idx),
                                        "cancel_price": round(float(close), 4),
                                        "reason": "반대후보교체",
                                    }
                                    self._state.cancelled_candidates.append(cancelled_info)
                                except Exception:
                                    pass
                                self._state.pending_candidate_status = "취소"
                                self._pivot_event_emit(
                                    "취소",
                                    close=float(close),
                                    reason="반대후보교체",
                                    cancelled_type="low",
                                    cancelled_time=self._bar_hhmm(old_pc.get("idx")),
                                    cancelled_price=round(float(old_pc.get("price") or 0.0), 4),
                                )
                                self._candidate_cancelled_count += 1
                                self._notify_collector_cancelled(
                                    cancelled_bar=self._bar_idx,
                                    cancelled_close=float(close),
                                    reason="반대후보교체",
                                )
                            if self._pending_high_idx != -1:
                                swing_time_high = self._bar_hhmm(self._pending_high_idx)
                                self._pending_confirm = dict(
                                    type="high",
                                    idx=self._pending_high_idx,
                                    price=self._pending_high,
                                    atr=atr,
                                    remaining=self._calc_confirmation_bars(),
                                )
                                self._pending_confirm_registered_bar = self._bar_idx
                                self._candidate_registered_count += 1
                                self._state.pending_candidate_type = "high"
                                self._state.pending_candidate_time = swing_time_high
                                self._state.pending_candidate_price = round(
                                    float(self._pending_high), 4)
                                self._state.pending_candidate_remaining = \
                                    self._calc_confirmation_bars()
                                self._state.pending_candidate_status = "등록"
                                self._pivot_event_emit(
                                    "후보등록",
                                    close=float(close),
                                    candidate="high",
                                    swing_time=swing_time_high,
                                    swing_price=round(float(self._pending_high), 4),
                                    remaining=self._calc_confirmation_bars(),
                                    thr_abs=round(float(thr_abs), 4),
                                    thr_pct=round(float(thr_pct), 4),
                                    prob=round(
                                        self.get_pending_confirmation_probability(
                                            float(close)), 3),
                                )
                                self._notify_collector_candidate_registered(
                                    candidate_type="high",
                                    candidate_price=float(self._pending_high),
                                    bar_idx=self._pending_high_idx,
                                    close=float(close),
                                )
                    else:
                        self._pivot_event_emit(
                            "후보등록",
                            close=close,
                            candidate="high",
                            swing_time=self._bar_hhmm(self._pending_high_idx),
                            swing_price=round(float(self._pending_confirm.get("price") or 0.0), 4),
                            remaining=int(self._pending_confirm.get("remaining") or 0),
                            reason="same_type_retrigger",
                            prob=round(self.get_pending_confirmation_probability(float(close)), 3),
                        )
        return new_swing_signal

    def _process_direction_zero(self, high: float, low: float, close: float, atr: float, thr_abs: float, thr_pct: float) -> str:
        """direction=0 초기범위 처리"""
        new_swing_signal = "none"
        cfg = self.config

        if len(self._highs) >= 2:
            if high > self._pending_high:
                self._pending_high     = high
                self._pending_high_idx = self._bar_idx
            if low < self._pending_low:
                self._pending_low     = low
                self._pending_low_idx = self._bar_idx
            # [DEBUG] direction=0 상태 로깅
            _logger.debug(
                "[ZZ][DIR0] bar=%d high=%.2f low=%.2f pending_high=%.2f(idx=%d) pending_low=%.2f(idx=%d) thr_abs=%.2f",
                self._bar_idx, high, low, self._pending_high, self._pending_high_idx,
                self._pending_low, self._pending_low_idx, thr_abs
            )
            if self._pending_high > 0 and self._pending_low < float("inf"):
                if (self._pending_high - self._pending_low) >= thr_abs:
                    # [FIX] 초기범위 확정 조건부 허용 - seed_anchor가 없는 경우에만 허용
                    # seed_anchor가 이미 심어진 경우 초기범위 확정 차단
                    if len(self._all_swings) > 0:
                        _logger.debug(
                            "[ZZ][DIR0] 초기범위 확정 차단: seed_anchor 이미 존재 (swings=%d)",
                            len(self._all_swings)
                        )
                        pass  # seed_anchor가 있으면 초기범위 확정 차단
                    else:
                        # seed_anchor가 없는 경우 초기범위 확정 허용
                        # [FIX] 최소 봉 수 조건 추가 - 장 초반 잘못된 피봇 방지
                        min_wave_bars = int(getattr(cfg, "min_wave_bars", 5) or 5)
                        if self._bar_idx < min_wave_bars:
                            _logger.debug(
                                "[ZZ][DIR0] 초기범위 확정 차단: 봉 수 부족 (bar_idx=%d < min_wave_bars=%d)",
                                self._bar_idx, min_wave_bars
                            )
                            pass  # 봉 수 부족 시 초기범위 확정 차단
                        else:
                            _logger.debug(
                                "[ZZ][DIR0] 초기범위 조건 충족: pending_high_idx=%d pending_low_idx=%d diff=%.2f",
                                self._pending_high_idx, self._pending_low_idx,
                                self._pending_high - self._pending_low
                            )
                            # [EDGE-CASE-1] 동일 봉 내 발생 시 시가 기준 방향 결정
                            # 시가가 저점에 가까우면 LOW 우선, 고점에 가까우면 HIGH 우선
                            prefer_low_first = False  # 기본: HIGH 우선
                            if self._pending_high_idx == self._pending_low_idx:
                                open_price = self._last_bar_open if self._last_bar_open > 0 else (high + low) / 2
                                dist_to_high = abs(high - open_price)
                                dist_to_low = abs(low - open_price)
                                _logger.debug(
                                    "[ZZ][DIR0] 동일 봉 내 발생: open=%.2f dist_to_high=%.2f dist_to_low=%.2f",
                                    open_price, dist_to_high, dist_to_low
                                )
                                # 시가에서 더 먼 쪽을 먼저 확정 (더 큰 움직임)
                                prefer_low_first = (dist_to_low > dist_to_high)

                            # 인덱스 비교 대신 플래그 사용 (동일 봉 처리 시 미래 인덱스 방지)
                            if self._pending_high_idx > self._pending_low_idx or (self._pending_high_idx == self._pending_low_idx and prefer_low_first):
                                # [FIX v3-3] 초기범위 확정 시 deque 상대 인덱스 변환
                                base_offset = self._bar_idx - len(self._lows)
                                relative_low_idx = self._pending_low_idx - base_offset
                                if 0 <= relative_low_idx < len(self._lows):
                                    actual_low = self._lows[relative_low_idx]
                                else:
                                    actual_low = self._pending_low

                                self._current_direction = 1
                                self._state.last_swing_low     = actual_low
                                self._state.last_swing_low_idx = self._pending_low_idx
                                # [버그 C 수정] 초기범위 확정 전 교번 검사
                                _swing_added = False
                                _log_low_idx = _log_low_price = _lag = None
                                if not self._would_violate_alternation(SwingType.LOW):
                                    _logger.debug("[ZZ][DIR0] LOW 확정: actual_low=%.2f", actual_low)
                                    self._add_swing(
                                        self._pending_low_idx,
                                        actual_low,
                                        SwingType.LOW,
                                        atr,
                                        confirmed_at_idx=self._bar_idx,
                                        confirmed_close=self._last_bar_close,
                                    )
                                    self._last_confirmed_bar_idx = self._bar_idx
                                    new_swing_signal = "new_low"
                                    _swing_added = True
                                    # 완결봉: 초기범위 확정은 현재 봉 기준
                                    self._state.last_swing_low_time         = self._bar_hhmm(self._pending_low_idx) or self._bar_hhmm(self._bar_idx)
                                    self._state.last_swing_low_confirm_time = self._bar_hhmm(self._bar_idx)
                                    self._state.last_swing_low_lag_bars     = max(0, self._bar_idx - self._pending_low_idx)
                                    self._state.last_swing_low_open         = self._last_bar_open
                                    self._state.last_swing_low_close        = self._last_bar_close
                                    # 로그용 값을 리셋 전에 캡처
                                    _log_low_idx   = self._pending_low_idx
                                    _log_low_price = actual_low
                                    # lag 미리 계산 (순서 의존성 방지)
                                    _lag = max(0, self._bar_idx - _log_low_idx)
                                else:
                                    _logger.debug("[ZZ][DIR0] LOW 확정 차단: 교번 위반")
                                # [P-FIX-C] 초기 방향 확정 후 반대 방향 pending 초기화
                                self._pending_high     = high
                                self._pending_high_idx = self._bar_idx
                                self._pending_low      = float("inf")
                                self._pending_low_idx  = -1
                                if _swing_added:
                                    self._pivot_event_emit(
                                        "확정",
                                        close=close,
                                        mode="초기범위",
                                        signal=new_swing_signal,
                                        swing_type="low",
                                        swing_time=self._bar_hhmm(_log_low_idx),
                                        swing_price=round(_log_low_price, 4),
                                        thr_pct=round(thr_pct, 4),
                                        lag=_lag,
                                    )
                            else:
                                # [FIX v3-3] 초기범위 확정 시 deque 상대 인덱스 변환
                                base_offset = self._bar_idx - len(self._highs)
                                relative_high_idx = self._pending_high_idx - base_offset
                                if 0 <= relative_high_idx < len(self._highs):
                                    actual_high = self._highs[relative_high_idx]
                                else:
                                    actual_high = self._pending_high

                                self._current_direction = -1
                                self._state.last_swing_high     = actual_high
                                self._state.last_swing_high_idx = self._pending_high_idx
                                # [버그 C 수정] 초기범위 확정 전 교번 검사
                                _swing_added = False
                                _log_high_idx = _log_high_price = _lag = None
                                if not self._would_violate_alternation(SwingType.HIGH):
                                    _logger.debug("[ZZ][DIR0] HIGH 확정: actual_high=%.2f", actual_high)
                                    self._add_swing(
                                        self._pending_high_idx,
                                        actual_high,
                                        SwingType.HIGH,
                                        atr,
                                        confirmed_at_idx=self._bar_idx,
                                        confirmed_close=self._last_bar_close,
                                    )
                                    self._last_confirmed_bar_idx = self._bar_idx
                                    new_swing_signal = "new_high"
                                    _swing_added = True
                                    # 완결봉: 초기범위 확정은 현재 봉 기준
                                    self._state.last_swing_high_time         = self._bar_hhmm(self._pending_high_idx) or self._bar_hhmm(self._bar_idx)
                                    self._state.last_swing_high_confirm_time = self._bar_hhmm(self._bar_idx)
                                    self._state.last_swing_high_lag_bars     = max(0, self._bar_idx - self._pending_high_idx)
                                    self._state.last_swing_high_open         = self._last_bar_open
                                    self._state.last_swing_high_close        = self._last_bar_close
                                    # 로그용 값을 리셋 전에 캡처
                                    _log_high_idx   = self._pending_high_idx
                                    _log_high_price = actual_high
                                    # lag 미리 계산 (순서 의존성 방지)
                                    _lag = max(0, self._bar_idx - _log_high_idx)
                                else:
                                    _logger.debug("[ZZ][DIR0] HIGH 확정 차단: 교번 위반")
                                # [P-FIX-C] 초기 방향 확정 후 반대 방향 pending 초기화
                                self._pending_low      = low
                                self._pending_low_idx  = self._bar_idx
                                self._pending_high     = 0.0
                                self._pending_high_idx = -1
                                if _swing_added:
                                    self._pivot_event_emit(
                                        "확정",
                                        close=close,
                                        mode="초기범위",
                                        signal=new_swing_signal,
                                        swing_type="high",
                                        swing_time=self._bar_hhmm(_log_high_idx),
                                        swing_price=round(_log_high_price, 4),
                                        thr_pct=round(thr_pct, 4),
                                        lag=_lag,
                                    )
        return new_swing_signal

    def _update_dynamic_atr_ratio(self, atr_monitor_result: Dict) -> None:
        """ATR 급변 시 동적 비율 조정"""
        self._dynamic_atr_ratio = 0.0  # 0이면 급변 없음
        spike_detected = atr_monitor_result['spike_detected']
        change_pct = atr_monitor_result['change_pct']
        if spike_detected:
            # ATR 급증 시: 변동성 증가 → 임계값 높여 노이즈 억제
            if change_pct > 0:
                self._dynamic_atr_ratio = 1.3
            # ATR 급락 시: 변동성 감소 → 임계값 낮춰 민감도 회복
            else:
                self._dynamic_atr_ratio = 0.7

    def _update_atr_monitor(self, atr: float) -> Dict:
        """ATR 모니터 업데이트"""
        atr_monitor_result = self._atr_monitor.update(atr)
        self._state.atr = atr
        self._state.atr_change_pct = atr_monitor_result['change_pct']
        self._state.atr_trend = atr_monitor_result['trend']
        self._state.atr_spike_detected = atr_monitor_result['spike_detected']
        self._state.atr_ma = atr_monitor_result['ma']
        return atr_monitor_result

    def _update_atr(self, high: float, low: float, close: float) -> float:
        """ATR 계산 및 업데이트"""
        n = len(self._closes)
        if n >= 2:
            pc = self._closes[-2]
            tr = max(high - low, abs(high - pc), abs(low - pc))
        else:
            tr = high - low
        self._tr.append(tr)
        try:
            atr = self._atr_rma.update(tr)
        except (ValueError, TypeError, AttributeError):
            atr = np.mean(list(self._tr)) if self._tr else tr
        self._atr_values.append(atr)
        self._prev_atr = atr
        
        # [DATA-CONSISTENCY] ATR 값 저장 (세션 간 연속성 보존)
        if atr > 0:
            self._saved_atr = atr
        
        return atr

    def _check_market_open_time(self, bar_time: Any) -> bool:
        """장 시작 시간 체크 - 장 시작 전이면 False 반환"""
        if not bar_time:
            return True
        
        try:
            if isinstance(bar_time, str):
                bar_dt = datetime.datetime.strptime(bar_time, "%H:%M")
            elif isinstance(bar_time, datetime.datetime):
                bar_dt = bar_time
            else:
                bar_dt = None
            
            if bar_dt:
                bar_hour = bar_dt.hour
                bar_minute = bar_dt.minute
                # 심볼별 장 시작 시간 결정
                market_start_hour = 9
                market_start_minute = 0
                if "KP200" in self._symbol_name or "선물" in self._symbol_name:
                    market_start_hour = 8
                    market_start_minute = 45
                
                # 장 시작 시간 이전이면 피봇 생성 차단
                if bar_hour < market_start_hour or (bar_hour == market_start_hour and bar_minute < market_start_minute):
                    _logger.debug("[ZZ][UPDATE] 장 시작 전 시간 차단: %s (장 시작: %02d:%02d)", bar_time, market_start_hour, market_start_minute)
                    return False
        except Exception:
            pass  # 시간 파싱 실패 시 무시
        
        return True

    def update(self, high: float, low: float, close: float, bar_time: Any = None, open: float = 0.0, volume: float = 1.0) -> ZigZagState:
        # ── [다중 시간프레임] 상위 시간프레임 데이터 업데이트 ─────────
        self._update_multi_timeframe(high, low, close, bar_time, open, volume)
        
        # [FIX v2-4] _bar_idx는 이 메서드 전체에서 현재 처리 중인 봉 번호를 나타냄
        # _remember_bar_time()이 _bar_hhmm_map에 현재 봉 시각을 저장한 후
        # 내부 메서드들(_calc_threshold_pct 등)이 _bar_hhmm(_bar_idx)로 조회 가능
        if not self._buffer_and_check_initial_bars(high, low, close, open, volume, bar_time):
            return self.state

        # [DATA-CONSISTENCY] 시가 앵커 설정은 indicator_integration에서 seed_anchor로 직접 호출
        # update 메서드 내부에서는 시가 앵커 설정 제거 - 중복 방지

        # 1. True Range & ATR, 2. 적응형 임계값
        atr, thr_abs, thr_pct = self._calculate_atr_and_threshold(high, low, close, bar_time)

        # 3. ZigZag 로직
        new_swing_signal = "none"

        # 3-a. Pending confirmation window
        new_swing_signal = self._process_pending_confirmation_with_error_handling(high, low, close, atr, thr_pct)

        # 3-b. 방향 결정 / 전환
        new_swing_signal = self._process_direction_based_pivots(high, low, close, atr, thr_abs, thr_pct, new_swing_signal)

        # 4-8. 파동 크기, 피보나치, S/R, 구조 분석 및 상태 갱신
        return self._finalize_update(close, atr, thr_pct, new_swing_signal)

    def compute_from_df(
        self,
        df: pd.DataFrame,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> pd.DataFrame:
        """DataFrame 전체 처리. 컬럼명 대소문자 자동 탐지.

        마지막 봉은 ZigZag update에 넣지 않는다(미완결봉으로 피봇 확정이 흔들리는 것을 방지).
        출력 행의 ZigZag 컬럼은 직전 봉까지 확정된 상태를 반영한다.
        """
        hc = _resolve_col(df, high_col)
        lc = _resolve_col(df, low_col)
        cc = _resolve_col(df, close_col)
        self.full_reset()  # [FIX v2-1] _reset_buffers 대신 full_reset 사용 (_bar_idx 누적 방지)
        self.set_backtest_mode(True)  # 백테스트 모드: look-ahead bias 방지
        rows: List[dict] = []
        n = len(df)

        # [DATA-CONSISTENCY] 고정된 웜업 구간 확보 (ATR 안정화)
        cfg = self.config
        min_warmup_bars = cfg.atr_period * ZigZagConstants.WARMUP_MULTIPLIER  # 최소 atr_period의 5배
        if n < min_warmup_bars:
            _logger.warning("[ZZ][compute_from_df] 데이터 길이 부족: %d < %d (ATR 안정화를 위해 최소 %d봉 필요). "
                          "초반 임계값이 불안정할 수 있습니다.", n, min_warmup_bars, min_warmup_bars)
        for i, row in enumerate(df.itertuples(index=True)):
            # [BUG-5] bar_time을 DataFrame index(DatetimeIndex)에서 추출해 전달
            # 미전달 시 _bar_hhmm_map이 전혀 채워지지 않아 pivot_lifecycle_log에서
            # swing_time=? 만 출력되는 문제
            try:
                _bar_time = row.Index
            except Exception:
                _bar_time = None
            h = float(getattr(row, hc))
            lo = float(getattr(row, lc))
            c = float(getattr(row, cc))
            if i < n - 1:
                s = self.update(h, lo, c, bar_time=_bar_time)
            else:
                s = self.state
            f618 = float(s.fib_levels.get("fib_618") or 0.0)
            f382 = float(s.fib_levels.get("fib_382") or 0.0)
            rows.append({
                "azz_direction":        float(s.current_direction),
                "azz_last_high":        float(s.last_swing_high),
                "azz_last_low":         float(s.last_swing_low),
                "azz_wave_size_pct":    float(s.wave_size_pct),
                "azz_support":          float(s.nearest_support),
                "azz_resistance":       float(s.nearest_resistance),
                "azz_support_dist_pct": float(s.support_dist_pct),
                "azz_res_dist_pct":     float(s.resistance_dist_pct),
                "azz_threshold_pct":    float(s.adaptive_threshold_pct),
                "azz_structure":        s.structure,
                "azz_new_swing":        float(1 if s.new_swing_signal != "none" else 0),
                "azz_bars_since_swing": float(min(s.bars_since_last_swing / 50.0, 1.0)),
                "azz_fib382":           f382,
                "azz_fib618":           f618,
                # Transformer 호환 별칭
                "azz_fib_0382":         f382,
                "azz_fib_0618":         f618,
                "azz_higher_highs":     float(int(s.is_making_higher_highs)),
                "azz_lower_lows":       float(int(s.is_making_lower_lows)),
            })
        # [H/L-교번-사후처리] 데이터 로드 완료 후 H/L 교번 강제
        self._enforce_hl_alternation()
        return df.assign(**pd.DataFrame(rows, index=df.index))

    def get_transformer_features(self, close: float) -> Dict[str, float]:
        s = self._state

        def _fin(v: float, fb: float = 0.0) -> float:
            try:
                x = float(v)
            except Exception:
                return float(fb)
            return x if np.isfinite(x) else float(fb)

        f618 = float(s.fib_levels.get("fib_618") or dict.get(s.fib_levels, "0.618") or 0.0)
        f382 = float(s.fib_levels.get("fib_382") or dict.get(s.fib_levels, "0.382") or 0.0)
        fd618 = (close - f618) / close if close > 0 and f618 > 0 else 0.0
        fd382 = (close - f382) / close if close > 0 and f382 > 0 else 0.0

        try:
            age = float(s.bars_since_last_swing)
        except Exception:
            age = 0.0

        # ── pending_confirm 피처 계산 ──────────────────────────────
        # ZigZag 확정(6~20봉 후)을 기다리지 않고 후보 등록 시점부터
        # ML이 반전 가능성을 인식할 수 있도록 pending 상태를 피처화
        #
        # azz_pending_type  : +1=HIGH후보(매도방향), -1=LOW후보(매수방향), 0=없음
        # azz_pending_dist  : 후보가격과 현재가의 거리 비율 (-1~+1, ±5% 클리핑)
        #                     양수=후보가 현재가 위(HIGH), 음수=아래(LOW)
        # azz_pending_urgency: 확정 임박도. rem이 작을수록 1에 가까움
        #                     = 1 - rem/confirmation_bars (0=방금등록, 1=다음봉확정)
        # azz_pending_age   : 후보 등록 후 경과 비율. 길수록 신뢰도 낮음
        #                     exp(-waited/5): 0봉=1.0, 5봉≈0.37, 10봉≈0.14
        try:
            pc = self._pending_confirm
            _cb = float(getattr(self.config, "confirmation_bars", 2) or 2)
            if isinstance(pc, dict) and pc:
                _pc_type  = str(pc.get("type") or "")
                _pc_price = float(pc.get("price") or 0.0)
                _pc_rem   = float(pc.get("remaining") or 0.0)

                # type: +1=high, -1=low
                _pend_type = 1.0 if _pc_type == "high" else -1.0

                # 후보가격 ↔ 현재가 거리 정규화 (±5% 클리핑)
                if close > 0 and _pc_price > 0:
                    _raw_dist = (_pc_price - close) / close
                else:
                    _raw_dist = 0.0
                _pend_dist = float(np.clip(_raw_dist, -0.05, 0.05) / 0.05)

                # 확정 임박도: rem 작을수록 1
                _pend_urgency = float(np.clip(1.0 - _pc_rem / max(_cb, 1.0), 0.0, 1.0))

                # 등록 후 경과: _pending_confirm_registered_bar 참조
                try:
                    _reg_bar = int(getattr(self, "_pending_confirm_registered_bar", -1))
                    _waited  = float(self._bar_idx - _reg_bar) if _reg_bar >= 0 else 0.0
                except Exception:
                    _waited = 0.0
                _pend_age = float(np.exp(-_waited / 5.0))
            else:
                _pend_type    = 0.0
                _pend_dist    = 0.0
                _pend_urgency = 0.0
                _pend_age     = 0.0
        except Exception:
            _pend_type = _pend_dist = _pend_urgency = _pend_age = 0.0

        return {
            "azz_direction":         float(s.current_direction),
            # [BUG-6] raw 가격 그대로 반환하면 다른 0~1 정규화 피처들과 스케일 불일치
            # → 현재가 대비 거리(%)로 정규화: 양수=현재가 위, 음수=아래, ±5% 클리핑
            "azz_last_high":         _fin(float(np.clip(
                (s.last_swing_high - close) / close if close > 0 and s.last_swing_high > 0 else 0.0,
                -0.05, 0.05)) / 0.05),
            "azz_last_low":          _fin(float(np.clip(
                (s.last_swing_low - close) / close if close > 0 and s.last_swing_low > 0 else 0.0,
                -0.05, 0.05)) / 0.05),
            "azz_wave_size_pct":     _fin(min(s.wave_size_pct / 10.0, 1.0)),
            "azz_support_dist_pct":  _fin(min(s.support_dist_pct / 5.0, 1.0)),
            "azz_res_dist_pct":      _fin(min(s.resistance_dist_pct / 5.0, 1.0)),
            "azz_bars_since_swing":  _fin(min(s.bars_since_last_swing / 50.0, 1.0)),
            "azz_fib618_dist":       _fin(float(np.clip(fd618, -0.1, 0.1) / 0.1)),
            "azz_fib382_dist":       _fin(float(np.clip(fd382, -0.1, 0.1) / 0.1)),
            "azz_higher_highs":      float(int(s.is_making_higher_highs)),
            "azz_lower_lows":        float(int(s.is_making_lower_lows)),
            "azz_new_swing":         float(1 if s.new_swing_signal == "new_high" else
                                           (-1 if s.new_swing_signal == "new_low" else 0)),
            "azz_swing_recency":     _fin(float(np.exp(-age / 5.0))),
            "azz_threshold_pct":     _fin(float(np.clip(s.adaptive_threshold_pct / 3.0, 0.0, 1.0))),
            "azz_structure_up":      float(s.structure == "uptrend"),
            "azz_structure_down":    float(s.structure == "downtrend"),
            "azz_structure_ranging": float(s.structure == "ranging"),
            # ── 보완-3: micro_structure / structure_confidence ──────
            "azz_micro_up":          float(getattr(s, "micro_structure", "unknown") == "uptrend"),
            "azz_micro_down":        float(getattr(s, "micro_structure", "unknown") == "downtrend"),
            "azz_micro_ranging":     float(getattr(s, "micro_structure", "unknown") == "ranging"),
            "azz_structure_conf":    _fin(float(getattr(s, "structure_confidence", 0.0) or 0.0)),
            # ── 보완-6: pending 잠정 S/R ────────────────────────────
            "azz_pend_sr_dist":      _fin(float(np.clip(
                ((getattr(s, "pending_resistance", 0.0) or getattr(s, "pending_support", 0.0)) - close)
                / close if close > 0 else 0.0,
                -0.05, 0.05)) / 0.05),
            # ── pending 피처 (ZigZag 후행성 보완) ──────────────────
            "azz_pending_type":      _fin(_pend_type),
            "azz_pending_dist":      _fin(_pend_dist),
            "azz_pending_urgency":   _fin(_pend_urgency),
            "azz_pending_age":       _fin(_pend_age),
            "azz_pending_prob":      _fin(float(self.get_pending_confirmation_probability(float(close)))),
        }

    def get_llm_context(self, close: float, symbol: str = "KP200 선물") -> str:
        s = self._state

        dir_kor = {1: "상승", -1: "하락", 0: "미결정"}[s.current_direction]
        str_kor = {
            "uptrend":  "상승 구조 (고점·저점 모두 상승)",
            "downtrend":"하락 구조 (고점·저점 모두 하락)",
            "ranging":  "횡보 구조 (고점·저점 혼재)",
            "unknown":  "구조 분석 불충분",
        }.get(s.structure, "알 수 없음")
        micro_str_kor = {
            "uptrend":  "단기 상승",
            "downtrend":"단기 하락",
            "ranging":  "단기 횡보",
            "unknown":  "단기 미확정",
        }.get(getattr(s, "micro_structure", "unknown"), "단기 미확정")
        struct_conf = float(getattr(s, "structure_confidence", 0.0) or 0.0)

        fib_txt = []
        for r in ["0.236", "0.382", "0.5", "0.618", "0.786"]:
            k = f"fib_{int(round(float(r)*1000))}"
            lv = float(s.fib_levels.get(k) or 0.0)
            if lv > 0:
                d = (lv - close) / close * 100
                fib_txt.append(f"Fib {r} ({lv:.2f}) : {abs(d):.2f}% {'위' if d > 0 else '아래'}")

        swings = []
        for sw in reversed(s.recent_swings[-6:]):
            t = "고점" if sw.swing_type == SwingType.HIGH else "저점"
            swings.append(f"  {'[주요]' if sw.is_major else '[부차]'} {t} {sw.price:.2f}")

        if s.new_swing_signal == "new_high":
            sig = f"새 스윙 고점 확정: {s.last_swing_high:.2f}"
        elif s.new_swing_signal == "new_low":
            sig = f"새 스윙 저점 확정: {s.last_swing_low:.2f}"
        else:
            sig = f"마지막 스윙 이후 {s.bars_since_last_swing}봉 경과"

        # ── pending_confirm 컨텍스트 ────────────────────────────────
        # ZigZag 확정 전 후보 상태를 LLM에 전달해 반전 예측에 활용
        try:
            pc = self._pending_confirm
            if isinstance(pc, dict) and pc:
                _pc_type  = str(pc.get("type") or "")
                _pc_price = float(pc.get("price") or 0.0)
                _pc_rem   = int(pc.get("remaining") or 0)
                _pt_kor   = "고점" if _pc_type == "high" else "저점"
                _pt_dir   = "↓반전 가능" if _pc_type == "high" else "↑반전 가능"
                try:
                    _reg_bar = int(getattr(self, "_pending_confirm_registered_bar", -1))
                    _waited  = self._bar_idx - _reg_bar if _reg_bar >= 0 else 0
                except Exception:
                    _waited = 0
                _pc_swing_time = self._bar_hhmm(int(pc.get("idx") or -1)) or "?"
                pending_txt = (
                    f"피봇후보({_pt_kor}) 대기 중: {_pc_price:.2f}  "
                    f"봉시각={_pc_swing_time}  "
                    f"확정까지={_pc_rem}봉  경과={_waited}봉  {_pt_dir}"
                )
                # 확정 확률 추가
                prob = self.get_pending_confirmation_probability(close)
                if prob > 0:
                    pending_txt += f"  확정확률={prob*100:.0f}%"
                pending_note = "주의: 후보는 미확정 신호이며, 확정(new_swing) 전까지는 참고 지표로만 사용"
            else:
                pending_txt = "피봇후보 없음 (추세 탐색 중)"
                pending_note = "주의: 현재 확정 전환 후보가 없어, 기존 확정 스윙/구조를 우선 해석"
        except Exception:
            pending_txt = "피봇후보 상태 조회 불가"
            pending_note = "주의: 후보 상태 조회 실패로 확정 신호 중심 해석 필요"

        advice = {
            "uptrend":  "상승 구조 유지 — 매도 신호에 신중하세요.",
            "downtrend":"하락 구조 유지 — 매수 신호에 신중하세요.",
        }.get(s.structure, "횡보 구조 — 지지/저항 범위 매매가 유리합니다.")

        return (
            f"[Adaptive ZigZag - {symbol}]\n"
            f"현재가: {close:.2f}  방향: {dir_kor}  구조: {str_kor}  단기: {micro_str_kor}  신뢰도: {struct_conf:.2f}\n"
            f"신호: {sig}\n"
            f"[후보] {pending_txt}\n"
            f"[후보해석] {pending_note}\n"
            f"최근 스윙 고점: {s.last_swing_high:.2f}  저점: {s.last_swing_low:.2f}  파동: {s.wave_size_pct:.2f}%\n"
            f"스윙 목록:\n" + "\n".join(swings or ["  (데이터 부족)"]) + "\n"
            "피보나치:\n" + "\n".join(fib_txt or ["  (계산 중)"]) + "\n"
            f"지지: {s.nearest_support:.2f} ({s.support_dist_pct:.1f}% 아래)  "
            f"저항: {s.nearest_resistance:.2f} ({s.resistance_dist_pct:.1f}% 위)"
            + (f"  잠정지지: {s.pending_support:.2f}" if getattr(s, "pending_support", 0.0) > 0 else "")
            + (f"  잠정저항: {s.pending_resistance:.2f}" if getattr(s, "pending_resistance", 0.0) > 0 else "")
            + f"\n임계값: {s.adaptive_threshold_pct:.2f}%  ATR: {s.atr:.2f}\n"
            f"{advice}"
        )

    def get_pending_confirmation_probability(self, close: float = 0.0) -> float:
        """후보 피봇 확정 확률 반환 (0~1).

        Heuristic 기반 규칙으로 확률 추정.
        머신러닝 모델이 없으면 현재 규칙 기반으로 동작.

        Args:
            close: 현재 가격 (거리 계산용)

        Returns:
            확정 확률 (0.0 ~ 1.0). 후보가 없으면 0.0 반환.

        Heuristic 규칙:
            - urgency > 0.8 AND structure != 'ranging' → 높음 (0.8)
            - urgency > 0.5 AND dist_pct < threshold_pct * 0.5 → 중간 (0.6)
            - waited > max_wait_bars * 0.5 → 낮음 (0.3)
            - 그 외 → 기본 (0.4)
        """
        try:
            pc = self._pending_confirm
            if not isinstance(pc, dict) or not pc:
                return 0.0

            _pc_type = str(pc.get("type") or "")
            _pc_price = float(pc.get("price") or 0.0)
            _pc_rem = int(pc.get("remaining") or 0)
            _cb_f = float(getattr(self.config, "confirmation_bars", 2) or 2)
            _max_wait = float(getattr(self.config, "max_wait_bars", 0) or 0)  # [FIX] 기본값 0 (무제한)으로 수정

            # 기본 피처 계산
            kwargs = self._pending_status_kwargs(float(close))
            urgency = float(kwargs.get("urgency", 0.0))
            age = float(kwargs.get("age", 0.0))
            waited_str = kwargs.get("waited", "0")
            try:
                waited = int(waited_str) if waited_str.isdigit() else 0
            except Exception:
                waited = 0

            # 거리 계산
            dist_pct = 0.0
            if close > 0 and _pc_price > 0:
                dist_pct = abs(_pc_price - close) / close

            # 시장 구조
            structure = str(getattr(self._state, "structure", "unknown") or "unknown")
            struct_conf = float(getattr(self._state, "structure_confidence", 0.0) or 0.0)

            # 임계값 여유
            threshold_pct = float(getattr(self._state, "adaptive_threshold_pct", 0.0) or 0.0)
            threshold_margin = 0.0
            if threshold_pct > 0:
                threshold_margin = dist_pct / threshold_pct

            # Heuristic 규칙 기반 확률 계산
            prob = 0.4  # 기본값

            # 규칙 1: 긴급도 높고 횡보가 아니면 확률 높음
            if urgency > 0.8 and structure != "ranging" and struct_conf > 0.5:
                prob = 0.8
            # 규칙 2: 긴급도 중간이고 임계값 내에 있으면 중간 확률
            elif urgency > 0.5 and threshold_margin < 0.5:
                prob = 0.6
            # 규칙 3: 너무 오래 대기하면 확률 낮음 (반전 가능성)
            # [FIX v2-5] 규칙 3: max_wait_bars가 설정된 경우에만 적용
            elif _max_wait > 0 and waited > _max_wait * 0.5:
                prob = 0.3
            # 규칙 4: 구조 신뢰도가 높으면 확률 상승
            elif struct_conf > 0.7:
                prob = 0.5
            # 규칙 5: 상승/하락 구조에서 방향 일치 시 확률 상승
            elif structure in ("uptrend", "downtrend") and struct_conf > 0.4:
                prob = 0.5

            # age로 보정 (너무 오래된 후보는 확률 감소)
            prob *= (0.5 + 0.5 * age)

            return max(0.0, min(1.0, prob))
        except Exception:
            return 0.0

    def get_swing_points(self, n: int = 10) -> List[SwingPoint]:
        return self._all_swings[-n:]

    def emit_pending_status_log(self, close: float = 0.0) -> None:
        """분봉 갱신 시점에 현재 대기 중인 피봇 후보 상태를 로그로 출력.

        pivot_lifecycle_log=True 일 때만 동작한다.
        호출 예: 분봉이 갱신될 때마다 update() 직후 호출.

        동일 후보(타입·스윙봉 idx·가격)가 유지되는 동안은 로그를 생략해
        rem 카운트다운 등으로 매 봉 찍히는 노이즈를 줄인다.
        타입/idx/가격이 바뀌거나 후보가 없어졌다 생겼다 할 때만 출력한다.

        후보 대기 중:
            [ZZ_PIVOT][PREFIX] 후보상태 bar=N type=H price=370.25
                swing_at=09:15 rem=1 waited=4
                dist=+1.84% urgency=0.667 age=0.549 prob=0.75
                → 확정까지 1봉, ↓반전 가능성

        후보 없음:
            [ZZ_PIVOT][PREFIX] 후보상태 bar=N none
        """
        if not bool(getattr(self.config, "pivot_lifecycle_log", False)):
            return
        try:
            pc = self._pending_confirm
            if isinstance(pc, dict) and pc:
                _pc_type  = str(pc.get("type") or "")
                _pc_price = float(pc.get("price") or 0.0)
                _pc_rem   = int(pc.get("remaining") or 0)
                _pc_idx   = int(pc.get("idx") or -1)
                sig = f"{_pc_type}|{_pc_idx}|{round(_pc_price, 4)}"
            else:
                sig = "none"

            if getattr(self, "_last_pending_status_emit_sig", None) == sig:
                return
            self._last_pending_status_emit_sig = sig

            if isinstance(pc, dict) and pc:
                _pt       = "H" if _pc_type == "high" else "L"
                _swing_at = self._bar_hhmm(_pc_idx) or "?"
                _kwargs = self._pending_status_kwargs(float(close))
                _kwargs["prob"] = round(self.get_pending_confirmation_probability(float(close)), 3)
                # ZigZagState 피봇 후보 상태 업데이트 (갱신)
                self._state.pending_candidate_type = _pc_type
                self._state.pending_candidate_time = _swing_at
                self._state.pending_candidate_price = round(_pc_price, 4)
                self._state.pending_candidate_remaining = _pc_rem
                self._state.pending_candidate_status = "갱신"
                self._pivot_event_emit(
                    "후보상태",
                    close=float(close),
                    type=_pt,
                    price=round(_pc_price, 2),
                    swing_at=_swing_at,
                    rem=_pc_rem,
                    **_kwargs,
                )
            else:
                # 후보 없음
                self._state.pending_candidate_type = None
                self._state.pending_candidate_time = None
                self._state.pending_candidate_price = 0.0
                self._state.pending_candidate_remaining = 0
                self._state.pending_candidate_status = None
                self._pivot_event_emit("후보상태", close=float(close), note="none")
        except Exception:
            pass

    def seed_anchor(self, price: float, swing_type: SwingType) -> None:
        """장 시작 시가를 anchor SwingPoint 로 _all_swings[0] 에 주입.

        [FIX] 첫 번째 봉이 아닌 3번째 봉에서도 anchor 심도록 수정
        _bar_idx <= 2 이고 _all_swings 가 비어있을 때 anchor 심도록 변경

        효과:
          - _all_swings 에 피봇이 1개뿐일 때 last_swing_high / last_swing_low 한쪽이
            0 인 상태를 방지 → _find_nearest_sr / _calc_fibonacci 안정화.
          - _current_direction 을 anchor 반대 방향으로 설정:
            anchor=LOW  → direction=1  (상승 탐색 중)
            anchor=HIGH → direction=-1 (하락 탐색 중)
            이렇게 해야 direction=0 초기화 블록이 anchor 와 동일 타입 swing 을
            연속 추가하는 것을 막는다.
          - AdaptiveIndicatorManager.is_ready() 의 '최소 4개 스윙' 조건은 그대로이므로
            학습 피처 품질에는 영향 없음.  실시간 시그널 판단에서만 초반 공백을 줄임.
        """
        # [FIX] 첫 번째 봉이 아닌 4번째 봉까지 허용 - 초기 데이터 누적 후 anchor 심도록 수정
        if self._bar_idx > 3 or len(self._all_swings) != 0:
            return
        try:
            atr_est = float(self._prev_atr) if float(self._prev_atr) > 0 else 0.0
            self._all_swings.append(SwingPoint(
                index=0, price=float(price), swing_type=swing_type,
                atr_at_swing=atr_est, is_major=True, confirmed=True, confirmed_at_idx=self._bar_idx,
                registered_at_idx=self._bar_idx,  # seed anchor는 등록=확정
            ))
            # [BUG-2] seed anchor는 최초 확정 피봇이므로 _last_confirmed_bar_idx 설정
            # 미설정 시 min_wave_bars 조건이 오작동
            self._last_confirmed_bar_idx = self._bar_idx
            # state 에도 반영
            if swing_type == SwingType.HIGH:
                self._state.last_swing_high     = float(price)
                self._state.last_swing_high_idx = self._bar_idx
                # anchor 가 HIGH → 다음은 LOW 탐색 (하락 방향)
                self._current_direction = -1
                self._pending_low       = float("inf")
                self._pending_low_idx   = -1
            else:
                self._state.last_swing_low     = float(price)
                self._state.last_swing_low_idx = self._bar_idx
                # anchor 가 LOW → 다음은 HIGH 탐색 (상승 방향)
                self._current_direction = 1
                self._pending_high      = 0.0
                self._pending_high_idx  = -1
        except Exception:
            pass

    @property
    def state(self) -> ZigZagState:
        return self._state
    
    @property
    def pivot_collector(self) -> Optional["PivotCandidateCollector"]:
        """피봇 후보 수집기 인스턴스 반환 (데이터셋 저장/로드용)."""
        return self._pivot_collector

    # ────────────────── 내부 메서드 ───────────────────────

    # ── [FIX-ALT-3] H/L 교번 헬퍼 메서드 ─────────────────────────────────────

    def _last_confirmed_swing_type(self) -> Optional[SwingType]:
        """가장 최근 확정 피봇의 타입 반환. 없으면 None."""
        s = next((s for s in reversed(self._all_swings) if s.confirmed), None)
        return s.swing_type if s is not None else None

    def _would_violate_alternation(self, candidate_type: SwingType) -> bool:
        """후보 등록 시 H/L 교번을 위반하는지 사전 검사.

        마지막 확정 피봇과 동일 타입이면 True(위반).
        확정 피봇이 없으면 False(위반 없음).
        """
        last_type = self._last_confirmed_swing_type()
        if last_type is None:
            return False
        return last_type == candidate_type

    # ────────────────────────────────────────────────────────────────────────

    def _enforce_hl_alternation(self) -> None:
        """[H/L-교번-사후처리] _all_swings 리스트에서 연속된 동일 타입 확정 피봇 병합.

        [FIX-ALT-5] 수정 사항:
        - 미확정 항목을 병합 대상에서 완전 분리 후 재결합
        - 병합 후 전체 리스트를 index 기준 재정렬
        - 빈 리스트 / 단일 항목 조기 반환

        [FIX-ALT-6] 추가 개선:
        - confirmed_at_idx 기준 정렬 (확정 시점 기준)
        - 전체 피봇 리스트에서 교번 검사 (unconfirmed 포함)
        - 더 정확한 시간순서 기반 교번 보장

        [BUG-1] cfg 변수 선언 추가
        """
        cfg = self.config  # [BUG-1] cfg 변수 선언 추가
        if len(self._all_swings) < 2:
            return

        # [BUG-3 FIX] 정렬 키를 index(피봇 발생 시점)로 통일
        # 기존: confirmed_at_idx(확정 시점) 기준 병합 → index 기준 재정렬
        # 문제: confirmed_at_idx 기준 병합 결과가 index 기준 순서와 다를 수 있어
        #       "나중에 발생했지만 먼저 확정된" 피봇이 살아남는 인과율 위반 발생
        # 수정: 병합과 최종 정렬 모두 index(피봇 실제 발생 봉) 기준으로 일원화
        sorted_swings = sorted(
            self._all_swings,
            key=lambda s: (s.index, s.confirmed_at_idx if hasattr(s, 'confirmed_at_idx') and s.confirmed_at_idx is not None else s.index)
        )

        # [DEBUG] 교번 위배 확인을 위한 로그
        alt_violations = []
        for i in range(1, len(sorted_swings)):
            if sorted_swings[i].swing_type == sorted_swings[i-1].swing_type:
                alt_violations.append((i, sorted_swings[i-1], sorted_swings[i]))

        if alt_violations:
            _logger.debug("[ZZ][enforce_hl_alt] 교번 위배 감지: %d건", len(alt_violations))
            for idx, prev, curr in alt_violations[:ZigZagConstants.MAX_LOG_VIOLATIONS]:  # 최대 N건만 로그
                _logger.debug("[ZZ][enforce_hl_alt]   [%d] %s@%d(%s) -> %s@%d(%s)", 
                             idx, prev.swing_type.value, prev.index, 
                             prev.confirmed_at_idx if hasattr(prev, 'confirmed_at_idx') else "N/A",
                             curr.swing_type.value, curr.index,
                             curr.confirmed_at_idx if hasattr(curr, 'confirmed_at_idx') else "N/A")

        filtered: List[SwingPoint] = []
        i = 0
        removed_count = 0

        while i < len(sorted_swings):
            current    = sorted_swings[i]
            group_type = current.swing_type
            j = i + 1
            while j < len(sorted_swings) and sorted_swings[j].swing_type == group_type:
                j += 1

            if j - i > 1:
                group = sorted_swings[i:j]
                # [FIX-ALT-6] confirmed_at_idx 기준으로 더 최근인 피봇 유지
                # confirmed_at_idx가 있는 경우 확정 시점 기준, 없으면 등록 시점 기준
                # [EDGE-CASE-4] _enforce_hl_alternation의 '최선의 피봇' 선택 기준
                prefer_first_pivot = getattr(cfg, "prefer_first_pivot_in_alt", False)
                if group_type == SwingType.HIGH:
                    # 가격 극값 우선: 가장 높은 가격 선택
                    best = max(group, key=lambda s: s.price)
                else:
                    # 가격 극값 우선: 가장 낮은 가격 선택
                    best = min(group, key=lambda s: s.price)
                filtered.append(best)
                removed_count += j - i - 1
                
                # [DEBUG] 병합된 그룹 로그
                _logger.debug("[ZZ][enforce_hl_alt] %s 그룹 병합: %d개 -> %d개 (idx=%d, price=%.2f)", 
                             group_type.value, j - i, 1, best.index, best.price)
            else:
                filtered.append(current)

            i = j

        if removed_count > 0:
            self._swing_version += 1
            _logger.info(
                "[ZZ][enforce_hl_alt] %d 연속 동일타입 피봇 병합 제거 (confirmed_at_idx 기준 정렬)", removed_count
            )
        else:
            _logger.debug("[ZZ][enforce_hl_alt] 교번 원칙 준수: 병합 제거 없음")

        # [DESIGN-2] index 기준 재정렬
        # 교번 병합은 confirmed_at_idx(확정 시점) 기준으로 수행되었으나,
        # 최종 리스트는 피봇 발생 시점(index) 기준으로 재정렬합니다.
        # 이는 _analyze_structure(), _find_nearest_sr() 등 후속 메서드가
        # _all_swings를 시간순(index 기준)으로 가정한다는 암묵적 계약을 준수하기 위함입니다.
        filtered.sort(key=lambda s: s.index)
        self._all_swings = filtered

    def check_hl_alternation(self) -> Dict[str, Any]:
        """H/L 교번 원칙 준수 상태 점검.

        Returns:
            Dict[str, Any]: 교번 점검 결과
                - is_alternating: bool - 교번 준수 여부
                - violations: List[Dict] - 교번 위배 목록
                - total_pivots: int - 전체 피봇 수
                - confirmed_count: int - 확정 피봇 수
                - unconfirmed_count: int - 미확정 피봇 수
        """
        if len(self._all_swings) < 2:
            return {
                "is_alternating": True,
                "violations": [],
                "total_pivots": len(self._all_swings),
                "confirmed_count": 0,
                "unconfirmed_count": len(self._all_swings)
            }

        # confirmed_at_idx 기준 정렬
        sorted_swings = sorted(
            self._all_swings,
            key=lambda s: (
                s.confirmed_at_idx if hasattr(s, 'confirmed_at_idx') and s.confirmed_at_idx is not None else s.index,
                s.index
            )
        )

        violations = []
        for i in range(1, len(sorted_swings)):
            prev = sorted_swings[i-1]
            curr = sorted_swings[i]
            if prev.swing_type == curr.swing_type:
                violations.append({
                    "index": i,
                    "prev_type": prev.swing_type.value,
                    "prev_index": prev.index,
                    "prev_confirmed_at": prev.confirmed_at_idx if hasattr(prev, 'confirmed_at_idx') else None,
                    "prev_price": prev.price,
                    "curr_type": curr.swing_type.value,
                    "curr_index": curr.index,
                    "curr_confirmed_at": curr.confirmed_at_idx if hasattr(curr, 'confirmed_at_idx') else None,
                    "curr_price": curr.price,
                    "prev_confirmed": prev.confirmed,
                    "curr_confirmed": curr.confirmed
                })

        confirmed_count = sum(1 for s in self._all_swings if s.confirmed)
        unconfirmed_count = len(self._all_swings) - confirmed_count

        return {
            "is_alternating": len(violations) == 0,
            "violations": violations,
            "total_pivots": len(self._all_swings),
            "confirmed_count": confirmed_count,
            "unconfirmed_count": unconfirmed_count
        }

    def _init_buffers(self, cfg) -> None:
        """버퍼 초기화 공통 로직.

        full_reset()과 reset_for_new_session()에서 공통으로 사용되는
        버퍼 초기화 코드를 추출하여 중복을 제거합니다.
        """
        max_buf = int(max(cfg.atr_period * ZigZagConstants.WARMUP_MULTIPLIER, ZigZagConstants.DEFAULT_MAX_BUF))
        self._highs:       deque = deque(maxlen=max_buf)
        self._lows:        deque = deque(maxlen=max_buf)
        self._closes:      deque = deque(maxlen=max_buf)
        self._opens:       deque = deque(maxlen=max_buf)  # [REGIME-INTEGRATION] 시가 버퍼 추가
        self._volumes:     deque = deque(maxlen=max_buf)  # [REGIME-INTEGRATION] 거래량 버퍼 추가
        self._tr:          deque = deque(maxlen=max_buf)
        self._atr_values:  deque = deque(maxlen=max_buf)
        self._prev_atr:    float = 0.0
        self._atr_rma = WilderRMA(period=int(cfg.atr_period))
        self._dynamic_atr_ratio: float = 0.0

        # 피봇 탐색 상태 초기화
        self._last_confirmed_bar_idx: int = -1
        self._pending_high:     float = 0.0
        self._pending_high_idx: int   = -1
        self._pending_low:      float = float("inf")
        self._pending_low_idx:  int   = -1
        self._current_direction: int  = 0  # [FIX v3-1] 명시적 초기화 추가
        self._pending_confirm:   Optional[Dict[str, Any]] = None
        self._pending_confirm_registered_bar: int = -1
        self._last_pending_status_emit_sig: Optional[str] = None
        self._pivot_event_seq: int = 0
        self._last_candidate_sig: str = ""
        self._last_confirmed_sig: str = ""

        # 피봇 후보 수집기용
        self._current_candidate_id: Optional[str] = None
        # 직전 봉 OHLC
        self._last_bar_open:  float = 0.0
        self._last_bar_high:  float = 0.0
        self._last_bar_low:   float = 0.0
        self._last_bar_close: float = 0.0

        # [DATA-CONSISTENCY] 시가 앵커 및 ATR 초기값 보존 (데이터 길이 흔들림 방지)
        self._seed_anchor_open: float = 0.0  # 장 시작 시가 앵커
        self._saved_atr: float = 0.0  # 이전 세션 마지막 ATR 값

        # [SUPERTREND-INTEGRATION] 슈퍼트렌드 신호 참조를 위한 상태 저장
        self._supertrend_signal: str = ""  # "bull" 또는 "bear"

    def set_supertrend_signal(self, signal: str) -> None:
        """슈퍼트렌드 신호를 설정하여 피봇 확정 조건에 활용.
        
        Args:
            signal: "bull" (상승) 또는 "bear" (하락)
        """
        self._supertrend_signal = signal.lower() if signal else ""

    def set_pivot_event_callback(self, callback: Callable) -> None:
        """피봇 이벤트 콜백을 설정.
        
        Args:
            callback: 피봇 이벤트 발생 시 호출할 콜백 함수
        """
        self._pivot_event_callback = callback
        _logger.info("[AdaptiveZigZag] 피봇 이벤트 콜백 설정 완료: callback=%s", callback is not None)

    def reset(self) -> None:
        """완전 초기화: 버퍼, 피봇, 인덱스 모두 초기화.
        
        데이터 소스 변경 시 호출하여 완전히 초기화합니다.
        """
        cfg = self.config
        
        # [REFACTOR v2-7] 공통 버퍼 초기화 메서드 사용
        self._init_buffers(cfg)
        
        # state 생성
        self._state = ZigZagState()
        
        # _bar_idx 초기화 (완전 리셋)
        self._bar_idx = 0
        
        # 피봇 스윙 초기화
        self._all_swings = []
        
        # 시가 앵커 및 ATR 초기값 초기화
        self._seed_anchor_open = 0.0
        self._saved_atr = 0.0
        
        # 슈퍼트렌드 신호 초기화
        self._supertrend_signal = ""
        
        _logger.info("[AdaptiveZigZag] reset 완료: 완전 초기화")

    def reset_for_new_session(self) -> None:
        """장 시작 시 호출: 버퍼만 초기화, 피봇/인덱스는 유지.

        실시간 운용 중 장 시작 시 버퍼만 비우고 피봇 목록은 유지하여
        장 연속성을 보존합니다. _bar_idx는 계속 증가하여 인덱스 충돌을 방지합니다.
        """
        cfg = self.config

        # 기존 state 저장 (새 state 생성 전에 값 보존)
        old_state = getattr(self, "_state", None)

        # 시각 정보 보존 (reset 후에도 피봇 시각 정보 유지)
        saved_pivot_times = {}
        if old_state is not None:
            saved_pivot_times = {
                "last_swing_high_time": getattr(old_state, "last_swing_high_time", None),
                "last_swing_low_time": getattr(old_state, "last_swing_low_time", None),
                "last_swing_high_confirm_time": getattr(old_state, "last_swing_high_confirm_time", None),
                "last_swing_low_confirm_time": getattr(old_state, "last_swing_low_confirm_time", None),
                "last_swing_high_lag_bars": getattr(old_state, "last_swing_high_lag_bars", 0),
                "last_swing_low_lag_bars": getattr(old_state, "last_swing_low_lag_bars", 0),
                "confirmed_pivot_count": getattr(old_state, "confirmed_pivot_count", 0),
            }

        # 피봇 스윙 보존 (reset 후에도 피봇 목록 유지)
        saved_all_swings = list(getattr(self, "_all_swings", []))

        # [DATA-CONSISTENCY] 시가 앵커 및 ATR 초기값 보존
        saved_seed_anchor_open = getattr(self, "_seed_anchor_open", 0.0)
        saved_atr = getattr(self, "_saved_atr", 0.0)

        # [REFACTOR v2-7] 공통 버퍼 초기화 메서드 사용
        self._init_buffers(cfg)

        # state 생성 (저장된 상태 복원을 위해)
        self._state = ZigZagState()

        # _bar_idx 유지 (계속 증가하여 인덱스 충돌 방지)
        # _bar_idx: 전체 봉 카운터 (세션 리셋 시에도 계속 증가)
        # - 세션 간 인덱스 충돌 방지를 위해 reset_for_new_session()에서는 초기화하지 않음
        # - 피봇 등록/확정 시점 추적, lag 계산, _bar_hhmm_map 키로 사용
        # - full_reset()에서만 0으로 초기화
        if not hasattr(self, "_bar_idx"):
            self._bar_idx: int = 0

        # _all_swings 유지 (피봇 목록 보존)
        self._all_swings: List[SwingPoint] = saved_all_swings
        # [FIX v2-3] 마지막 확정 피봇 방향으로 direction 복원
        # [FIX v3-4] _last_confirmed_bar_idx도 복원하여 min_wave_bars 조건 유지
        last_confirmed = next(
            (s for s in reversed(self._all_swings) if s.confirmed), None
        )
        if last_confirmed is not None:
            self._current_direction = -1 if last_confirmed.swing_type == SwingType.HIGH else 1
            self._last_confirmed_bar_idx = last_confirmed.confirmed_at_idx if hasattr(last_confirmed, 'confirmed_at_idx') else last_confirmed.index
        else:
            self._current_direction = 0
            self._last_confirmed_bar_idx = -1

        # _bar_hhmm_map 유지 (시각 정보 보존)
        # [REFACTOR] OrderedDict 사용으로 메모리 정리 단순화
        if not hasattr(self, "_bar_hhmm_map"):
            self._bar_hhmm_map: OrderedDict[int, str] = OrderedDict()

        # 시각 정보 복원
        self._state.last_swing_high_time = saved_pivot_times.get("last_swing_high_time")
        self._state.last_swing_low_time = saved_pivot_times.get("last_swing_low_time")
        self._state.last_swing_high_confirm_time = saved_pivot_times.get("last_swing_high_confirm_time")
        self._state.last_swing_low_confirm_time = saved_pivot_times.get("last_swing_low_confirm_time")
        self._state.last_swing_high_lag_bars = saved_pivot_times.get("last_swing_high_lag_bars", 0)
        self._state.last_swing_low_lag_bars = saved_pivot_times.get("last_swing_low_lag_bars", 0)
        self._state.confirmed_pivot_count = saved_pivot_times.get("confirmed_pivot_count", 0)

        # recent_swings 복원 (deepcopy로 외부 참조 오염 방지)
        self._state.recent_swings = copy.deepcopy(self._all_swings[-cfg.max_swings:]) if self._all_swings else []

        # [DATA-CONSISTENCY] 시가 앵커 및 ATR 초기값 복원
        self._seed_anchor_open = saved_seed_anchor_open
        self._saved_atr = saved_atr
        # ATR 초기값 복원 (이전 세션의 마지막 ATR 값으로 초기화하여 수렴 속도 향상)
        if saved_atr > 0:
            self._prev_atr = saved_atr
            self._atr_rma._prev_value = saved_atr  # WilderRMA 내부 상태도 복원
            _logger.debug("[ZZ][reset_for_new_session] ATR 초기값 복원: %.6f", saved_atr)

        _logger.debug("[ZZ][reset_for_new_session] bar_idx=%d, map_size=%d, swings=%d, recent=%d, pivot_times restored, seed_anchor=%.2f, saved_atr=%.6f",
                     self._bar_idx, len(self._bar_hhmm_map), len(self._all_swings), len(self._state.recent_swings),
                     saved_seed_anchor_open, saved_atr)
        
        # 피봇 후보 수집기용
        self._current_candidate_id: Optional[str] = None
        # 직전 봉 OHLC
        self._last_bar_open:  float = 0.0
        self._last_bar_high:  float = 0.0
        self._last_bar_low:   float = 0.0
        self._last_bar_close: float = 0.0

    def full_reset(self) -> None:
        """완전 초기화: 모든 상태를 초기화합니다.

        백테스트 시작 시나 완전 재초기화가 필요한 경우 사용합니다.
        _bar_idx와 _all_swings도 모두 초기화됩니다.
        """
        cfg = self.config

        # 완전 초기화
        self._state = ZigZagState()

        # [REFACTOR v2-7] 공통 버퍼 초기화 메서드 사용
        self._init_buffers(cfg)

        # _bar_idx 완전 초기화 (full_reset 전용)
        # 전체 봉 카운터를 0으로 초기화하여 완전히 새로운 시퀀스 시작
        self._bar_idx: int = 0

        # _all_swings 완전 초기화
        self._all_swings: List[SwingPoint] = []
        # [REVIEW-FIX-5] 스윙 버전 카운터: 렌더링 캐시 무효화 트리거용
        self._swing_version: int = 0
        # _current_direction은 _init_buffers()에서 이미 초기화됨

        # _bar_hhmm_map 완전 초기화
        # [OPT-1 FIX] KOSPI200 1분봉 기준 1거래일 ~415봉 × 5일 = ~2000봉 상한
        # 무제한 성장 방지 — 오래된 시각 정보는 렌더링에서도 불필요
        self._bar_hhmm_map: OrderedDict[int, str] = OrderedDict()
        self._bar_hhmm_map_maxlen: int = 2000  # 상한 (full_reset 시만 변경)

        # [DATA-CONSISTENCY] 시가 앵커 및 ATR 초기값 완전 초기화
        self._seed_anchor_open = 0.0
        self._saved_atr = 0.0

        # [REGIME-INTEGRATION] 런타임 파라미터 초기화
        self._runtime_params: Dict[str, Any] = {}

        _logger.debug("[ZZ][full_reset] 완전 초기화 완료")

    def _reset_buffers(self) -> None:
        """레거시 호환용: 장 시작 시 버퍼만 초기화.
        
        Note:
            하위 호환성을 위해 유지합니다. 새 코드에서는 reset_for_new_session()을 사용하세요.
        """
        self.reset_for_new_session()

    def _add_swing(
        self,
        idx: int,
        price: float,
        swing_type: SwingType,
        atr: float,
        *,
        confirmed_at_idx: Optional[int] = None,
        confirmed_close: float = 0.0,
    ) -> bool:  # True=추가됨, False=무시/병합
        cfg   = self.config
        c_idx = int(self._bar_idx if confirmed_at_idx is None else confirmed_at_idx)

        # 디버그 로그: 피봇 추가 시점 (H/L 교번 확인)
        swing_type_str = "H" if swing_type == SwingType.HIGH else "L"
        
        # [SUPERTREND-INTEGRATION] 슈퍼트렌드 신호 확인 - 파동 중간 의미없는 피봇 방지
        # 슈퍼트렌드가 "bull"일 때만 HIGH 피봇 확정, "bear"일 때만 LOW 피봇 확정
        if self._supertrend_signal:
            if swing_type == SwingType.HIGH and self._supertrend_signal != "bull":
                logger.debug("[ZZ][피봇추가-스킵] %s | %s@%s=%.2f | 슈퍼트렌드=%s (상승 신호 아님)",
                    self._symbol_name, swing_type_str, self._bar_hhmm(idx), price, self._supertrend_signal)
                return False  # 슈퍼트렌드가 상승이 아니면 HIGH 피봇 확정 불가
            if swing_type == SwingType.LOW and self._supertrend_signal != "bear":
                logger.debug("[ZZ][피봇추가-스킵] %s | %s@%s=%.2f | 슈퍼트렌드=%s (하락 신호 아님)",
                    self._symbol_name, swing_type_str, self._bar_hhmm(idx), price, self._supertrend_signal)
                return False  # 슈퍼트렌드가 하락이 아니면 LOW 피봇 확정 불가
        
        logger.debug("[ZZ][피봇추가] %s | %s@%s=%.2f | bar=%d | st=%s",
            self._symbol_name, swing_type_str, self._bar_hhmm(idx), price, self._bar_idx, self._supertrend_signal)

        # ── [FIX-ALT-1 + BUG-REMAIN-4] 교번 강제 + 클러스터링 통합 ───────────
        # confirmed 피봇 중 가장 마지막 항목과 타입 비교
        # (미확정 항목을 건너뛰지 않고 confirmed만 역순 탐색)
        # 교번 강제 + 클러스터 병합을 통합하여 중복 제거
        last_confirmed_swing = next(
            (s for s in reversed(self._all_swings) if s.confirmed), None
        )
        if last_confirmed_swing is not None and last_confirmed_swing.swing_type == swing_type:
            is_more_extreme = (
                (swing_type == SwingType.HIGH and price > last_confirmed_swing.price) or
                (swing_type == SwingType.LOW  and price < last_confirmed_swing.price)
            )
            # cluster_tol 계산
            cluster_tol = float(getattr(cfg, "cluster_tolerance_pct", 0.3) or 0.0)
            use_atr_filter = getattr(cfg, "use_atr_based_filtering", False)
            if use_atr_filter and self._atr_values and price > 0:
                atr = float(self._atr_values[-1]) if self._atr_values else 0.0
                if atr > 0:
                    cluster_atr_ratio = getattr(cfg, "cluster_atr_ratio", 0.5)
                    cluster_tol_atr = atr * cluster_atr_ratio / price * 100.0
                    cluster_tol = max(cluster_tol, cluster_tol_atr)
            dist_pct = abs(price - last_confirmed_swing.price) / last_confirmed_swing.price * 100.0 if last_confirmed_swing.price > 0 else 0.0

            if is_more_extreme:
                # 더 극단적: 항상 in-place 갱신 (교번 강제 + 클러스터 병합 통합)
                prev_price_snapshot = float(last_confirmed_swing.price)
                last_confirmed_swing.index            = idx
                last_confirmed_swing.price            = price
                last_confirmed_swing.atr_at_swing     = atr
                last_confirmed_swing.confirmed_at_idx = c_idx
                last_confirmed_swing.confirmed_close  = confirmed_close
                self._swing_version += 1
                self._pivot_event_emit(
                    "H/L교번강제_병합",
                    close=float(confirmed_close),
                    prev_price=round(prev_price_snapshot, 4),
                    new_price=round(price, 4),
                    dist_pct=round(dist_pct, 4) if cluster_tol > 0 else None,
                )
                return True  # in-place 갱신 성공
            # 덜 극단적: 무시 (교번 유지)
            return False  # 무시

        # [보완-7] is_major: 평균 파동 비율 기반 (ATR 배수 fallback)
        avg_wave = self._calc_avg_wave_size(
            n=int(getattr(cfg, "major_wave_lookback", 3) or 3)
        )
        major_wave_ratio = float(getattr(cfg, "major_wave_ratio", 1.5) or 1.5)
        prev_any = next(
            (s for s in reversed(self._all_swings) if s.swing_type == swing_type),
            None,
        )
        if avg_wave > 0 and prev_any is not None:
            is_major = abs(price - float(prev_any.price)) >= avg_wave * major_wave_ratio
        else:
            # fallback: 기존 ATR 배수 방식
            is_major = (
                True if prev_any is None
                else abs(price - float(prev_any.price)) >= atr * float(cfg.major_swing_ratio)
            )

        # ── [PIVOT-VALIDATION] ZigZag 특성에 맞는 검증 로직 ─────────────────────
        # [BUG-2] 검증 로직 제거: 결과가 사용되지 않는 Dead Code
        # _validate_pivot, _validate_extreme_at_confirmation, _validate_direction_change, _validate_wave_size 메서드는
        # 현재 로깅 전용으로만 사용되며 실제 피봇 등록 차단에 사용되지 않음
        # 불필요한 연산을 제거하고 코드 단순화
        # ─────────────────────────────────────────────────────────────────────────

        self._all_swings.append(SwingPoint(
            index=idx, price=price, swing_type=swing_type,
            atr_at_swing=atr, is_major=is_major, confirmed=True,
            confirmed_at_idx=c_idx, confirmed_close=confirmed_close,
            registered_at_idx=getattr(self, "_pending_confirm_registered_bar", idx),
        ))
        # [FIX-3] 슬라이싱 재할당으로 통일 (del 방식 제거)
        # [FIX v3-8] 주석: 오래된 피봇 제거는 의도된 동작
        # 제거된 피봇은 이후 _find_nearest_sr, _analyze_structure 등 분석에서 제외됩니다
        if len(self._all_swings) > cfg.max_swings * 2:
            self._all_swings = self._all_swings[-cfg.max_swings:]

        return True  # 새 항목 추가 성공

    def _validate_pivot(
        self,
        idx: int,
        price: float,
        swing_type: SwingType,
        atr: float,
        confirmed_at_idx: int,
        prev_any: Optional[Any]
    ) -> Dict[str, bool]:
        """
        ZigZag 특성에 맞는 피봇 검증 로직
        
        Returns:
            {
                'extreme_valid': bool,  # 확정 시점 기준 극값 검증
                'direction_valid': bool,  # 방향 전환 유효성 검증
                'wave_size_valid': bool  # 파동 크기 검증
            }
        """
        results = {
            'extreme_valid': True,
            'direction_valid': True,
            'wave_size_valid': True
        }
        
        # 1. 확정 시점 기준 극값 검증
        results['extreme_valid'] = self._validate_extreme_at_confirmation(
            idx=idx,
            price=price,
            swing_type=swing_type,
            confirmed_at_idx=confirmed_at_idx
        )
        
        # 2. 방향 전환 유효성 검증
        results['direction_valid'] = self._validate_direction_change(
            idx=idx,
            price=price,
            swing_type=swing_type,
            lookforward=10
        )
        
        # 3. 파동 크기 검증
        if prev_any is not None:
            results['wave_size_valid'] = self._validate_wave_size(
                price=price,
                prev_price=float(prev_any.price),
                atr=atr
            )
        
        return results
    
    def _validate_extreme_at_confirmation(
        self,
        idx: int,
        price: float,
        swing_type: SwingType,
        confirmed_at_idx: int,
        lookback: int = 5,
        lookforward: int = 5
    ) -> bool:
        """
        확정 시점 기준 극값 검증

        확정 시점으로부터 ±N봉 내에서 극값인지 확인

        Note:
            start/end 계산은 절대 인덱스 기준이지만, 내부 루프에서
            상대 인덱스 변환을 통해 실제 deque 접근은 안전하게 수행됩니다.
        """
        if not self._highs or not self._lows:
            return True

        # [FIX v3-7] 주석: 절대 인덱스 기반 범위 계산 (내부 루프에서 상대 변환 수행)
        start = max(0, confirmed_at_idx - lookback)
        # end 계산을 절대 인덱스 기준으로 통일 (버퍼 크기가 아닌 현재까지의 최대 절대 인덱스 기준)
        abs_max = self._bar_idx  # 현재까지의 최대 절대 인덱스
        end = min(abs_max, confirmed_at_idx + lookforward + 1)

        if start >= end:
            return True

        # 리스트 슬라이싱 대신 반복문으로 처리
        nearby_highs = []
        nearby_lows = []
        # deque 상대 인덱스 변환: 절대 인덱스(i)를 deque 상대 인덱스로 변환
        base_offset = self._bar_idx - len(self._highs)
        for i in range(start, end):
            relative_i = i - base_offset
            if 0 <= relative_i < len(self._highs):
                nearby_highs.append(self._highs[relative_i])
                nearby_lows.append(self._lows[relative_i])

        if swing_type == SwingType.HIGH:
            # 확정 시점 기준으로 주변에서 최고가인지
            return price >= max(nearby_highs) if nearby_highs else True
        else:
            # 확정 시점 기준으로 주변에서 최저가인지
            return price <= min(nearby_lows) if nearby_lows else True

    # [DESIGN-3] _validate_direction_change 메서드 제거
    # 실시간 모드에서 항상 True를 반환하여 필터링 효과가 없음
    # 백테스트 모드에서도 look-ahead bias 방지를 위해 항상 True 반환
    # 불필요한 연산을 제거하고 코드 단순화

    def _validate_wave_size(
        self,
        price: float,
        prev_price: float,
        atr: float
    ) -> bool:
        """
        파동 크기 검증

        피봇 파동 크기가 ATR 기준으로 적절한지 확인
        """
        cfg = self.config  # [DESIGN-4] cfg 지역변수 패턴 통일
        if atr <= 0:
            return True

        wave_size = abs(price - prev_price)
        min_wave_atr_ratio = float(getattr(cfg, "min_wave_atr_ratio", 0.5) or 0.5)
        min_wave = atr * min_wave_atr_ratio
        
        return wave_size >= min_wave

    def _calc_fibonacci(self) -> FibLevels:
        cfg = self.config  # [DESIGN-4] cfg 지역변수 패턴 통일
        s = self._state
        if s.last_swing_high <= 0 or s.last_swing_low <= 0:
            return FibLevels()
        high = s.last_swing_high; low = s.last_swing_low; diff = high - low
        fib: FibLevels = FibLevels()
        for r in cfg.fib_ratios:
            try:
                k = f"fib_{int(round(float(r) * 1000.0))}"
            except Exception:
                k = str(r)
            fib[k] = (high - diff * r) if s.current_direction == 1 else (low + diff * r)
        return fib

    def _find_nearest_sr(self, close: float) -> Tuple[float, float]:
        if not self._all_swings:
            return 0.0, 0.0
        confirmed = [s for s in self._all_swings if s.confirmed]
        highs = [s.price for s in confirmed if s.swing_type == SwingType.HIGH and s.price > close]
        lows  = [s.price for s in confirmed if s.swing_type == SwingType.LOW  and s.price < close]

        # [보완-6] pending 후보를 잠정 S/R로 포함
        if isinstance(self._pending_confirm, dict) and self._pending_confirm:
            pc_type  = str(self._pending_confirm.get("type") or "")
            pc_price = float(self._pending_confirm.get("price") or 0.0)
            if pc_type == "high" and pc_price > close:
                highs.append(pc_price)
            elif pc_type == "low" and 0 < pc_price < close:
                lows.append(pc_price)

        # [FIX-4] 빈 리스트 fallback: 0.0 반환
        support    = max(lows)  if lows  else 0.0
        resistance = min(highs) if highs else 0.0

        # ZigZagState에 잠정 S/R 저장
        try:
            pc = self._pending_confirm if isinstance(self._pending_confirm, dict) else {}
            pc_t = str(pc.get("type") or "")
            pc_p = float(pc.get("price") or 0.0)
            self._state.pending_support    = pc_p if pc_t == "low"  and pc_p > 0 else 0.0
            self._state.pending_resistance = pc_p if pc_t == "high" and pc_p > 0 else 0.0
        except Exception:
            pass

        return support, resistance

    def _analyze_structure(self) -> str:
        cfg = self.config  # [DESIGN-4] cfg 지역변수 패턴 통일
        if len(self._all_swings) < 4:
            return "unknown"
        try:
            lookback = int(getattr(cfg, "structure_lookback_swings", 8) or 8)
        except Exception:
            lookback = 8
        if lookback < 4:
            lookback = 4

        try:
            points = int(getattr(cfg, "structure_points", 3) or 3)
        except Exception:
            points = 3
        if points < 2:
            points = 2

        # [FIX] 교번 보장된 스윙 추출 (연속 HIGH/LOW 방지)
        recent_swings = self._all_swings[-lookback:]
        # 교번 순서가 보장된 최근 피봇 추출
        alternating = []
        last_type = None
        for s in recent_swings:
            if s.swing_type != last_type:
                alternating.append(s)
                last_type = s.swing_type
        
        rh = [s.price for s in alternating if s.swing_type == SwingType.HIGH][-points:]
        rl = [s.price for s in alternating if s.swing_type == SwingType.LOW][-points:]
        if len(rh) < 2 or len(rl) < 2:
            return "unknown"

        # [보외-3] 다수결 방식: majority_threshold(70%) 이상 일관 → 추세 판정
        majority = float(getattr(self.config, "structure_majority_threshold", 0.7) or 0.7)
        n = len(rh) - 1  # 비교 쌍 수

        hh_score = sum(1 for i in range(1, len(rh)) if rh[i] > rh[i - 1])
        hl_score = sum(1 for i in range(1, len(rl)) if rl[i] > rl[i - 1])
        lh_score = sum(1 for i in range(1, len(rh)) if rh[i] < rh[i - 1])
        ll_score = sum(1 for i in range(1, len(rl)) if rl[i] < rl[i - 1])

        if n > 0:
            if hh_score / n >= majority and hl_score / n >= majority:
                return "uptrend"
            if lh_score / n >= majority and ll_score / n >= majority:
                return "downtrend"
        return "ranging"

    def _calc_structure_confidence(self) -> float:
        """구조 판정 일관성 점수 (0.0~1.0). 구조가 얼마나 명확한지 나타낸다."""
        try:
            lookback = int(getattr(self.config, "structure_lookback_swings", 8) or 8)
            points   = int(getattr(self.config, "structure_points", 3) or 3)
            rh = [s.price for s in self._all_swings[-lookback:]
                  if s.swing_type == SwingType.HIGH][-points:]
            rl = [s.price for s in self._all_swings[-lookback:]
                  if s.swing_type == SwingType.LOW][-points:]
            if len(rh) < 2 or len(rl) < 2:
                return 0.0
            n = len(rh) - 1
            hh = sum(1 for i in range(1, len(rh)) if rh[i] > rh[i - 1])
            hl = sum(1 for i in range(1, len(rl)) if rl[i] > rl[i - 1])
            lh = sum(1 for i in range(1, len(rh)) if rh[i] < rh[i - 1])
            ll = sum(1 for i in range(1, len(rl)) if rl[i] < rl[i - 1])
            up_score = (hh + hl) / (n * 2)
            dn_score = (lh + ll) / (n * 2)
            return float(max(up_score, dn_score))
        except Exception:
            return 0.0

    def _is_higher_highs(self) -> bool:
        try:
            lookback = int(getattr(self.config, "structure_lookback_swings", 6) or 6)
        except Exception:
            lookback = 6
        if lookback < 4:
            lookback = 4
        hs = [s.price for s in self._all_swings[-lookback:] if s.swing_type == SwingType.HIGH]
        return len(hs) >= 2 and hs[-1] > hs[-2]

    def _is_lower_lows(self) -> bool:
        try:
            lookback = int(getattr(self.config, "structure_lookback_swings", 6) or 6)
        except Exception:
            lookback = 6
        if lookback < 4:
            lookback = 4
        ls = [s.price for s in self._all_swings[-lookback:] if s.swing_type == SwingType.LOW]
        return len(ls) >= 2 and ls[-1] < ls[-2]

    # ── [보완-1] confirmation_bars 동적 조절 ────────────────────
    def _calc_confirmation_bars(self) -> int:
        """구조·파동 크기에 따른 동적 confirmation_bars 계산.

        ranging / unknown 구간에서 대기 봉을 늘려 허위 확정을 억제한다.
        파동 크기가 ATR 미만인 소파동에는 +1봉을 추가한다.

        [REGIME-INTEGRATION] 레짐 기반 파라미터가 설정된 경우 ranging/unknown 고정값보다 우선함.
        레짐 시스템이 이미 구조를 분류하므로 ranging 구조 판정은 보조적 역할로만 사용.
        """
        cfg = self.config
        # [REGIME-INTEGRATION] _runtime_params 우선 참조
        base = int(self._runtime_params.get('confirmation_bars', getattr(cfg, "confirmation_bars", 1) or 1))
        structure = str(self._state.structure or "unknown")

        # [REGIME-INTEGRATION] 레짐 파라미터가 설정된 경우 ranging/unknown 고정값 무시
        # _param_adjuster가 주입되어 있고 레짐 분류가 활성화된 경우
        if not (hasattr(self, '_param_adjuster') and self._param_adjuster is not None):
            # 레짐 시스템 없으면 기존 ranging 로직 사용
            if structure == "ranging":
                base = max(base, int(getattr(cfg, "confirmation_bars_ranging", 2) or 2))
            elif structure == "unknown":
                base = max(base, int(getattr(cfg, "confirmation_bars_unknown", 3) or 3))

        # 소파동 추가 검증: 직전 파동 < ATR 이면 +1
        if self._prev_atr > 0 and self._state.wave_size > 0:
            if abs(float(self._state.wave_size)) < float(self._prev_atr):
                base += 1

        # [DESIGN-1 FIX] ranging/unknown 자기강화 루프 완화
        # 문제: structure="unknown" → bars_unknown=3 → 피봇 더 안 생김 → 계속 unknown
        # 수정: bars_since가 decay_start_bars를 초과하면 confirmation_bars를 1씩 낮춰
        #       장기 무피봇 구간에서 감도를 점진적으로 회복
        if self._last_confirmed_bar_idx >= 0:
            bars_since = max(0, self._bar_idx - self._last_confirmed_bar_idx)
            decay_start = int(getattr(self.config, "decay_start_bars", 30) or 30)
            if bars_since > decay_start:
                # decay_start_bars 초과 봉마다 1 감소, 최소 1 유지
                reduce = min(base - 1, (bars_since - decay_start) // decay_start)
                base = max(1, base - reduce)

        return max(1, base)

    # ── [보완-3] 단기 구조 분석 (micro_structure) ────────────────
    def _analyze_micro_structure(self) -> str:
        """최근 2피봇 기반 단기 구조 (빠른 방향 판단용).

        [BUG-MICRO-ALT] 수정: _all_swings[-4:]에서 high/low를 독립 필터링하면
        교번이 깨진 상태([H,H,L,L] 등)에서도 각 리스트에 2개가 채워져 잘못된
        구조 판정을 반환한다.
        수정: 교번 순서가 보장된 최근 피봇 4개를 먼저 추출하고,
        그 중에서 HIGH/LOW를 분리한다.
        """
        # 교번 순서가 유지되는 확정 피봇만 취득
        confirmed = [s for s in self._all_swings if s.confirmed]
        if len(confirmed) < 4:
            return "unknown"
        recent4 = confirmed[-4:]

        # 교번 검증: 인접 피봇이 동일 타입이면 신뢰할 수 없음
        for i in range(1, len(recent4)):
            if recent4[i].swing_type == recent4[i - 1].swing_type:
                return "unknown"  # 교번 깨진 상태 → 판정 불가

        rh = [s.price for s in recent4 if s.swing_type == SwingType.HIGH]
        rl = [s.price for s in recent4 if s.swing_type == SwingType.LOW]
        if len(rh) < 2 or len(rl) < 2:
            return "unknown"
        hh = rh[-1] > rh[-2]
        hl = rl[-1] > rl[-2]
        lh = rh[-1] < rh[-2]
        ll = rl[-1] < rl[-2]
        if hh and hl:
            return "uptrend"
        if lh and ll:
            return "downtrend"
        return "ranging"

    # ── [보완-7] 평균 파동 크기 계산 ─────────────────────────────
    def _calc_avg_wave_size(self, n: int = 3) -> float:
        """최근 N개 확정 피봇 간 파동 평균 크기."""
        confirmed = [s for s in self._all_swings if s.confirmed]
        if len(confirmed) < 2:
            return 0.0
        sizes: List[float] = []
        for i in range(1, min(n + 1, len(confirmed))):
            sizes.append(abs(float(confirmed[-i].price) - float(confirmed[-i - 1].price)))
        return float(sum(sizes) / len(sizes)) if sizes else 0.0

    # ── [보완-8] 방향 ER (Directional ER) ────────────────────────
    def _calc_er_and_der(self) -> tuple:
        """[OPT-3] ER과 DER을 한 번의 순회로 계산. (er, der) 반환.

        기존: _calc_der()가 _calc_er()를 호출해 _closes 이중 순회
        수정: 단일 순회로 er, direction_sign 동시 계산
        """
        cfg = self.config
        n   = len(self._closes)
        try:
            period = int(getattr(cfg, "er_period", 14))
        except Exception:
            period = 14

        if n < period + 1:
            return 0.5, 0.0

        # [REVIEW-FIX-4] look-ahead 편향 방지: 현재 봉 제외 완결봉만 사용
        cs = list(self._closes)[-(period + 1):-1]
        if len(cs) < period:
            return 0.5, 0.0

        try:
            direction_price = abs(float(cs[-1]) - float(cs[0]))
            volatility      = sum(abs(float(cs[i]) - float(cs[i - 1])) for i in range(1, len(cs)))
            er = float(np.clip(direction_price / volatility, 0.0, 1.0)) if volatility > 1e-10 else 0.0
        except Exception:
            er = 0.5

        try:
            direction_sign = 1.0 if float(cs[-1]) >= float(cs[0]) else -1.0
        except Exception:
            direction_sign = 0.0

        return er, float(er * direction_sign)

    def _calc_der(self) -> float:
        """방향 Efficiency Ratio. _calc_er_and_der() 위임."""
        _, der = self._calc_er_and_der()
        return der

    def _calc_er(self) -> float:
        """Efficiency Ratio. _calc_er_and_der() 위임."""
        er, _ = self._calc_er_and_der()
        return er

    def _calc_threshold_pct(self, atr: float, close: float, bar_idx: Optional[int] = None) -> float:
        """적응형 임계값 계산.

        Args:
            atr: 현재 ATR 값
            close: 현재 종가
            bar_idx: 현재 봉 인덱스 (None이면 self._bar_idx 사용)

        Returns:
            임계값 (%)
        """
        cfg = self.config
        if close <= 0:
            return float(np.clip(1.0, cfg.pivot_threshold_min_pct, cfg.pivot_threshold_max_pct))

        # [HYBRID-MODE] 하이브리드 모드 체크
        use_hybrid = getattr(cfg, "use_hybrid_mode", False)
        
        # 하이브리드 모드: ATR 기반 + 퍼센트 기반 결합
        if use_hybrid:
            atr_threshold = self._calc_atr_threshold(atr, close, bar_idx)
            pct_threshold = self._calc_percent_threshold(close, bar_idx)
            atr_weight = float(getattr(cfg, "atr_weight", 1.0))
            atr_weight = np.clip(atr_weight, 0.0, 1.0)
            
            # 가중 평균
            hybrid_threshold = (1 - atr_weight) * pct_threshold + atr_weight * atr_threshold
            return float(np.clip(hybrid_threshold, cfg.pivot_threshold_min_pct, cfg.pivot_threshold_max_pct))

        # ATR 기반 필터링 비활성화 시 최소 임계값 반환
        use_atr_filter = getattr(cfg, "use_atr_based_filtering", False)
        if not use_atr_filter:
            return float(cfg.pivot_threshold_min_pct)

        # 기존 ATR 기반 임계값 계산
        return self._calc_atr_threshold(atr, close, bar_idx)
    
    def _calc_atr_threshold(self, atr: float, close: float, bar_idx: Optional[int] = None) -> float:
        """ATR 기반 임계값 계산.
        
        Args:
            atr: 현재 ATR 값
            close: 현재 종가
            bar_idx: 현재 봉 인덱스
            
        Returns:
            ATR 기반 임계값 (%)
        """
        cfg = self.config
        
        # [EDGE-CASE-3] ATR 값 반올림으로 미세 오차 무시 (데이터 길이에 따른 일관성 보장)
        # ATR 값을 소수점 6자리로 반올림하여 데이터 길이 차이로 인한 미세한 ATR 차이를 무시
        atr = round(atr, ZigZagConstants.ATR_ROUNDING_DECIMALS)

        # [FIX v3-5] 명시적 파라미터로 암묵적 의존성 제거
        _idx = bar_idx if bar_idx is not None else self._bar_idx

        # [OPT-3 FIX] ER과 DER을 한 번의 순회로 계산 (이중 순회 제거)
        er, der = self._calc_er_and_der()

        # [DESIGN-3 FIX] _runtime_params에 단일 'atr_multiplier'가 있으면
        # mmin==mmax가 되어 ER 적응이 무력화되는 문제 수정
        # 런타임 파라미터: min/max 키를 각각 조회하고 없을 때만 공통 키 폴백
        _rt_mult = self._runtime_params.get('atr_multiplier')
        if _rt_mult is not None:
            # 단일값 주입 시: min/max 모두 같은 값 → ER 적응 무력화
            # → 의도 명확화: 단일값은 고정 배수로 해석, range는 별도 키 사용
            mmin = float(_rt_mult)
            mmax = float(self._runtime_params.get('atr_multiplier_max', _rt_mult))
        else:
            mmin = float(self._runtime_params.get('atr_multiplier_min', getattr(cfg, "atr_multiplier_min", 1.0)))
            mmax = float(self._runtime_params.get('atr_multiplier_max', getattr(cfg, "atr_multiplier_max", 4.0)))

        if mmax < mmin:
            mmin, mmax = mmax, mmin

        # [보완-8] 방향 불일치 시 mmax 완화 → 임계값 낮춰 전환 조기 감지
        der_thr   = float(getattr(cfg, "der_mismatch_threshold", 0.3) or 0.3)
        der_ratio = float(getattr(cfg, "der_mismatch_mult_ratio", 0.7) or 0.7)
        direction_mismatch = (
            (self._current_direction ==  1 and der < -der_thr) or
            (self._current_direction == -1 and der >  der_thr)
        )
        if direction_mismatch:
            mmax = mmax * der_ratio

        n      = len(self._closes)
        warmup = int(getattr(cfg, "atr_period", 14)) + 5
        if n < max(10, warmup):
            mult = (mmin + mmax) / 2.0
        else:
            # [FIX-1] 방향 수정: ER 높을수록 mult 큼 → threshold 큼 → 노이즈 필터
            mult = mmin + er * (mmax - mmin)

        base = float(atr) / float(close) * 100.0 * float(mult)
        base = float(np.clip(base, cfg.pivot_threshold_min_pct, cfg.pivot_threshold_max_pct))

        # [보완-5] bars_since decay: 장시간 무피봇 구간 감도 향상
        decay_start = int(getattr(cfg, "decay_start_bars",    30)    or 30)
        decay_rate  = float(getattr(cfg, "decay_rate_per_bar", 0.005) or 0.005)
        decay_max   = float(getattr(cfg, "decay_max_pct",      0.3)   or 0.3)
        bars_since  = (
            max(0, self._bar_idx - self._last_confirmed_bar_idx)
            if self._last_confirmed_bar_idx >= 0 else 0
        )
        if bars_since > decay_start:
            excess = bars_since - decay_start
            decay  = min(decay_max, excess * decay_rate)
            base   = max(float(cfg.pivot_threshold_min_pct), base - decay)

        return float(base)
    
    def _calc_percent_threshold(self, close: float, bar_idx: Optional[int] = None) -> float:
        """퍼센트 기반 임계값 계산.
        
        Args:
            close: 현재 종가
            bar_idx: 현재 봉 인덱스
            
        Returns:
            퍼센트 기반 임계값 (%)
        """
        cfg = self.config
        
        # 기본 퍼센트 임계값
        base_pct = float(getattr(cfg, "base_pct", 0.3))
        
        # ER 기반 동적 배수
        _idx = bar_idx if bar_idx is not None else self._bar_idx
        er, _ = self._calc_er_and_der()
        
        multiplier_min = float(getattr(cfg, "multiplier_min", 0.8))
        multiplier_max = float(getattr(cfg, "multiplier_max", 2.0))
        
        n = len(self._closes)
        warmup = int(getattr(cfg, "er_period", 10)) + 5
        if n < max(10, warmup):
            mult = (multiplier_min + multiplier_max) / 2.0
        else:
            mult = multiplier_min + er * (multiplier_max - multiplier_min)
        
        # 시간대별 배율 (session_multiplier_table)
        session_scale = self._get_session_multiplier_scale(_idx)
        
        threshold = base_pct * mult * session_scale
        return float(np.clip(threshold, cfg.pivot_threshold_min_pct, cfg.pivot_threshold_max_pct))
    
    def _get_session_multiplier_scale(self, bar_idx: int) -> float:
        """시간대별 배율 반환.
        
        Args:
            bar_idx: 봉 인덱스
            
        Returns:
            배율 (기본 1.0)
        """
        cfg = self.config
        table = getattr(cfg, "session_multiplier_table", None)
        if not table:
            return 1.0
        
        current_time = self._bar_hhmm(bar_idx)
        if not current_time:
            return 1.0
        
        for start, end, scale in table:
            try:
                if start <= current_time < end:
                    return float(scale)
            except Exception:
                continue
        
        return 1.0

    def _get_session_min_wave_atr_ratio(self, candidate_idx: int = -1) -> float:
        """[SESSION-ATR] 현재 봉(또는 피봇 봉) 시각에 맞는 min_wave_atr_ratio 반환.
        
        [Layer A] 세션 시간대 테이블 기반 ATR 비율.
        """
        cfg = self.config
        base = float(getattr(cfg, "min_wave_atr_ratio", 1.0) or 1.0)
        table = getattr(cfg, "session_min_wave_atr_ratio_table", None) or []
        if not table:
            return base
        
        lookup_idx = int(candidate_idx) if candidate_idx >= 0 else (self._bar_idx - 1)
        current_time = self._bar_hhmm(lookup_idx)
        if not current_time:
            return base
        
        for start, end, ratio in table:
            try:
                if str(start) <= current_time < str(end):
                    # 테이블 값을 직접 사용 (max(base, ratio) 제거)
                    return float(ratio)
            except Exception:
                continue
        return base

    def _get_runtime_params(self, bar_time=None) -> Dict[str, Any]:
        """[Layer A × Layer C] 런타임 파라미터 계산.

        Layer A: 세션 시간대 테이블 (무지연, 순환 없음)
        Layer C: 적응형 엔진 (자기완결형, 외부 의존성 없음)

        Returns:
            런타임 파라미터 딕셔너리
        """
        cfg = self.config

        # Layer A: 세션 시간대 테이블 (이미 구현됨, 활성화만)
        a_atr_ratio = self._get_session_min_wave_atr_ratio(self._bar_idx - 1)
        a_wave_bars = self._get_session_min_wave_bars(self._bar_idx - 1)

        # Layer C: 적응형 엔진 (자기완결형)
        if self._adaptive_engine is not None:
            try:
                # 시장 구조 정보 가져오기
                structure = getattr(self._state, "structure", "unknown")
                
                adjusted = self._adaptive_engine.compute(
                    atr_values=list(self._atr_values),
                    all_swings=self._all_swings,
                    bar_idx=self._bar_idx,
                    er=float(self._calc_er()),
                    der=float(self._calc_der()),
                    direction=self._current_direction,
                    last_confirmed_bar_idx=self._last_confirmed_bar_idx,
                    structure=structure,
                )
                # 결합: config 수정 없이 런타임 dict로만
                return {
                    "atr_multiplier": float(np.clip(
                        cfg.atr_multiplier * adjusted.mult,
                        cfg.atr_multiplier_min, cfg.atr_multiplier_max,
                    )),
                    "confirmation_bars": adjusted.confirmation_bars,
                    "min_wave_atr_ratio": float(np.clip(
                        a_atr_ratio * adjusted.wave_ratio_mult,
                        0.5, 5.0,
                    )),
                    "min_wave_bars": a_wave_bars,
                    "pivot_threshold_min_pct": float(np.clip(
                        cfg.pivot_threshold_min_pct * adjusted.thr_mult,
                        cfg.pivot_threshold_min_pct * 0.5,
                        cfg.pivot_threshold_max_pct,
                    )),
                }
            except Exception as e:
                _logger.warning("[AdaptiveZigZag] 적응형 엔진 계산 실패, 기본값 사용: %s", e)

        # 폴백: 기존 Layer B 방식
        b_mult = 1.0
        if hasattr(self, '_param_adjuster') and self._param_adjuster is not None and hasattr(self._param_adjuster, 'get_vol_ratio'):
            b_mult = self._param_adjuster.get_vol_ratio()  # 0.85 / 1.0 / 1.25

        return {
            "atr_multiplier": float(np.clip(
                cfg.atr_multiplier * b_mult,
                cfg.atr_multiplier_min, cfg.atr_multiplier_max
            )),
            "confirmation_bars": cfg.confirmation_bars,
            "min_wave_atr_ratio": float(np.clip(a_atr_ratio * b_mult, 0.5, 5.0)),
            "min_wave_bars": a_wave_bars,
            "pivot_threshold_min_pct": cfg.pivot_threshold_min_pct,
        }

    def _get_session_min_wave_bars(self, candidate_idx: int = -1) -> int:
        """[SESSION-MW] 현재 봉(또는 피봇 봉) 시각에 맞는 min_wave_bars 반환.

        candidate_idx : 실제 피봇이 될 봉 인덱스 (_pending_high/low_idx).
                        -1 이면 self._bar_idx - 1 폴백 (기존 동작).
        session_min_wave_bars_table이 설정된 경우 테이블을 순서대로 평가해
        처음 일치하는 구간의 값을 반환한다.
        테이블이 비어 있거나 해당 구간이 없으면 config.min_wave_bars 폴백.
        """
        cfg = self.config
        base = int(getattr(cfg, "min_wave_bars", 0) or 0)
        table = getattr(cfg, "session_min_wave_bars_table", None) or []
        if not table:
            return base
        # [FIX-EARLY-B] 피봇 봉 시각 기준 조회 (candidate_idx 지정 시)
        # candidate_idx가 없으면 기존처럼 현재 봉(bar_idx-1) 기준
        lookup_idx = int(candidate_idx) if candidate_idx >= 0 else (self._bar_idx - 1)
        current_time = self._bar_hhmm(lookup_idx)
        if not current_time:
            return base
        for start, end, bars in table:
            try:
                if str(start) <= current_time < str(end):
                    return max(base, int(bars))
            except Exception:
                continue
        return base

    def _is_wave_length_ok(self, thr_abs: float, close: float,
                           candidate_idx: int = -1) -> bool:
        """피봇 후보 등록 가능 여부 판별.

        candidate_idx : 실제 피봇이 될 봉 인덱스 (pending_high/low_idx).
                        -1 이면 self._bar_idx 폴백 (하위 호환).
        """
        cfg = self.config
        # [FIX-EARLY-B] gap 기준을 candidate_idx(실제 피봇 봉)로 통일
        # self._bar_idx(현재 봉) 기준이면 후보가 여러 봉 전에 형성됐을 때
        # 간격이 실제보다 크게 잡혀 차단이 누락될 수 있음
        cand_idx = int(candidate_idx) if candidate_idx >= 0 else self._bar_idx

        # [SESSION-MW] 시간대별 min_wave_bars 테이블 적용 (피봇 봉 시각 기준)
        min_bars = self._get_session_min_wave_bars(candidate_idx=cand_idx)
        if min_bars > 0 and self._last_confirmed_bar_idx >= 0:
            bar_gap = cand_idx - self._last_confirmed_bar_idx
            if bar_gap < min_bars:
                return False

        if self._last_confirmed_bar_idx >= 0:
            if cand_idx <= self._last_confirmed_bar_idx:
                return False

        # min_wave_pct: 실제 이동 거리(dist_pct) 체크
        # [DESIGN-2 FIX] 기존 이중 체크(임계값 비교 + 실제 거리 비교) 중 임계값 비교 제거
        # 임계값(thr_abs)은 ATR 기반으로 이미 결정된 값이므로 min_wave_pct와 이중 비교는 중복
        # 실제 이동 거리(actual_pct)만 min_wave_pct와 비교하는 것으로 단순화
        min_pct = float(getattr(cfg, "min_wave_pct", 0.0) or 0.0)
        if min_pct > 0 and close > 0:
            if self._current_direction == 1 and self._pending_high > 0:
                actual_pct = abs(self._pending_high - close) / close * 100.0
                if actual_pct < min_pct:
                    return False
            elif self._current_direction == -1 and self._pending_low < float("inf"):
                actual_pct = abs(close - self._pending_low) / close * 100.0
                if actual_pct < min_pct:
                    return False

        # ── [ATR-FILTER] ATR 기반 필터링 ─────────────────────────────
        use_atr_filter = getattr(cfg, "use_atr_based_filtering", False)
        if use_atr_filter and self._atr_values and close > 0:
            atr = float(self._atr_values[-1]) if self._atr_values else 0.0
            if atr > 0:
                # 파라미터 합성 우선순위 (문서와 일치):
                # 1. 시간대 테이블 값 (base)
                # 2. ATR 급변 배율 적용 (×0.7 or ×1.3)
                # 3. 절대 하한/상한 클램프 (min=0.5, max=5.0)
                ratio_table = getattr(cfg, "session_min_wave_atr_ratio_table", None) or []

                # 현재 시간 가져오기 (봉 시각 기준으로 조회)
                # [BUG-4] datetime 파싱 대신 문자열 HH:MM 비교 사용
                # 자정 경계(23:50~00:10)에서 날짜 롤오버 문제 해결
                bar_hhmm = self._bar_hhmm(self._bar_idx)
                current_time = bar_hhmm  # 문자열 HH:MM 형식 사용

                # 1. 시간대 테이블 값 조회
                time_ratio = _get_time_based_atr_ratio(current_time, ratio_table, getattr(cfg, "min_wave_atr_ratio", 0.5))

                # 2. ATR 급변 시 배율 적용
                if self._dynamic_atr_ratio > 0:
                    # ATR 급증/급락 시 시간대 비율에 배율 적용
                    applied_ratio = time_ratio * self._dynamic_atr_ratio
                else:
                    # ATR 급변이 없으면 시간대 비율 사용
                    applied_ratio = time_ratio

                # 3. 절대 하한/상한 클램프
                applied_ratio = max(0.5, min(applied_ratio, 5.0))

                min_atr_wave = atr * applied_ratio

                if self._current_direction == 1 and self._pending_high > 0:
                    actual_wave = abs(self._pending_high - close)
                    if actual_wave < min_atr_wave:
                        return False
                elif self._current_direction == -1 and self._pending_low < float("inf"):
                    actual_wave = abs(close - self._pending_low)
                    if actual_wave < min_atr_wave:
                        return False

        return True
    
    def _update_upper_timeframe_data(
        self,
        high: float,
        low: float,
        close: float,
        bar_time: Any = None,
        open: float = 0.0,
        volume: float = 1.0
    ) -> None:
        """상위 시간프레임 데이터 업데이트.
        
        1분봉 데이터를 버퍼에 누적하고,
        상위 시간프레임(5분봉, 15분봉)이 완성되면 해당 ZigZag에 업데이트합니다.
        """
        if not self._upper_tf_zz_instances:
            return
        
        try:
            for scale, zz in self._upper_tf_zz_instances.items():
                # 버퍼 참조 가져오기 (누적을 위해)
                if scale not in self._upper_tf_data_buffers:
                    self._upper_tf_data_buffers[scale] = []
                buffer = self._upper_tf_data_buffers[scale]
                # [FIX] 스케일별 별도 dict 생성 (참조 오염 방지)
                bar_data = {
                    'high': high,
                    'low': low,
                    'close': close,
                    'open': float(open) if open else 0.0,
                    'volume': volume,
                    'time': bar_time
                }
                buffer.append(bar_data)
                
                # 버퍼가 해당 시간프레임의 봉 수만큼 쌓이면 리샘플링
                if len(buffer) >= scale:
                    # 리샘플링
                    resampled = self._resample_buffer(buffer, scale)
                    
                    # 상위 시간프레임 ZigZag 업데이트
                    if resampled:
                        zz.update(
                            high=resampled['high'],
                            low=resampled['low'],
                            close=resampled['close'],
                            open=resampled['open'],
                            volume=resampled['volume'],
                            bar_time=bar_time
                        )
                        
                        # 상위 시간프레임 피봇 캐시 업데이트
                        self._update_upper_tf_pivot_cache(scale, zz)
                    
                    # 버퍼 초기화
                    self._upper_tf_data_buffers[scale] = []
                    
        except Exception as e:
            _logger.error("[MultiTF] 상위 시간프레임 데이터 업데이트 실패: %s", e)
    
    def _resample_buffer(self, buffer: List[Dict[str, Any]], scale: int) -> Optional[Dict[str, float]]:
        """버퍼 데이터를 리샘플링.
        
        Args:
            buffer: 1분봉 데이터 버퍼
            scale: 시간프레임 (분 단위)
            
        Returns:
            리샘플링된 OHLC 데이터
        """
        if not buffer:
            return None
        
        try:
            opens = [d['open'] for d in buffer if d['open'] > 0]
            highs = [d['high'] for d in buffer]
            lows = [d['low'] for d in buffer]
            closes = [d['close'] for d in buffer]
            volumes = [d['volume'] for d in buffer]
            
            if not opens:
                return None
            
            resampled = {
                'open': opens[0],
                'high': max(highs),
                'low': min(lows),
                'close': closes[-1],
                'volume': sum(volumes)
            }
            
            return resampled
            
        except Exception as e:
            _logger.error("[MultiTF] 리샘플링 실패: %s", e)
            return None
    
    def _update_upper_tf_pivot_cache(self, scale: int, zz: 'AdaptiveZigZag') -> None:
        """상위 시간프레임 피봇 캐시 업데이트.
        
        Args:
            scale: 시간프레임
            zz: 상위 시간프레임 ZigZag 인스턴스
        """
        if self._multi_tf_zz is None:
            return
        
        try:
            # 상위 시간프레임 확정 피봇 추출
            confirmed_pivots = []
            for sw in zz._all_swings:
                if not getattr(sw, "confirmed", False):
                    continue
                # [FIX] 상위 TF 인덱스를 1분봉 기준 인덱스로 변환
                # 상위 TF의 bar_idx는 독립적이므로, 1분봉 기준으로 변환 필요
                # 예: 5분봉 20번째 = 1분봉 100번째 (20 * 5)
                base_index = int(sw.index) * scale
                confirmed_pivots.append({
                    'index': base_index,
                    'price': float(sw.price),
                    'pivot_type': 'H' if sw.swing_type == SwingType.HIGH else 'L'
                })
            
            # MultiTimeframeZigZag 캐시 업데이트
            self._multi_tf_zz.update_pivot_cache(scale, confirmed_pivots)
            
        except Exception as e:
            _logger.error("[MultiTF] 상위 시간프레임 피봇 캐시 업데이트 실패: %s", e)
    
    def _check_multiframe_consensus(
        self,
        pivot_index: int,
        pivot_price: float,
        pivot_type: str,
        current_close: float
    ) -> bool:
        """다중 시간프레임 합의도 확인.
        
        Args:
            pivot_index: 피봇 인덱스
            pivot_price: 피봇 가격
            pivot_type: 피봇 타입 ('H' or 'L')
            current_close: 현재 종가
            
        Returns:
            합의도 통과 여부 (True: 통과, False: 필터링)
        """
        if self._multi_tf_zz is None:
            return True  # 다중 시간프레임 비활성화 시 항상 통과
        
        try:
            # 합의도 확인
            result = self._multi_tf_zz.check_consensus(
                pivot_index=pivot_index,
                pivot_price=pivot_price,
                pivot_type=pivot_type
            )
            
            # 합의도가 임계값 미만이면 필터링
            if not result['passed']:
                _logger.warning(
                    "[MultiTF] %s 피봇 합의도 부족으로 필터링: index=%d, price=%.2f, consensus=%d/%d (%.1f%%)",
                    pivot_type, pivot_index, pivot_price, 
                    result['consensus'], result['total_scales'], 
                    result['consensus_ratio'] * 100
                )
                return False  # 필터링
            else:
                _logger.info(
                    "[MultiTF] %s 피봇 합의도 통과: index=%d, price=%.2f, consensus=%d/%d (%.1f%%)",
                    pivot_type, pivot_index, pivot_price,
                    result['consensus'], result['total_scales'],
                    result['consensus_ratio'] * 100
                )
                return True  # 통과
            
        except Exception as e:
            _logger.error("[MultiTF] 합의도 확인 실패: %s", e)
            return True  # 오류 시 통과 (안전장치)


# ── 유틸 ──────────────────────────────────────────────────


class ATRMonitor:
    """ATR 변화를 추적하고 급격 변동을 감지하는 모니터 클래스."""
    
    def __init__(self, spike_threshold_pct: float = 30.0, ma_period: int = 14):
        """
        Args:
            spike_threshold_pct: 급격 변동 감지 임계값 (%)
            ma_period: ATR 이동평균 계산 기간
        """
        self._prev_atr: float = 0.0
        self._atr_history: List[float] = []
        self._spike_threshold_pct: float = spike_threshold_pct
        self._ma_period: int = ma_period
        self._logger = logging.getLogger(__name__)
        # 텔레그램 콜백 (급격 변동 감지 시 호출)
        self._telegram_callback: Optional[Callable[[str], None]] = None

    def set_telegram_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """텔레그램 콜백 설정."""
        self._telegram_callback = callback
    
    def update(self, current_atr: float) -> Dict[str, Any]:
        """
        ATR 값 업데이트 및 변화 추적
        
        Args:
            current_atr: 현재 ATR 값
            
        Returns:
            {
                'change_pct': 변화율 (%),
                'trend': 추세 ('rising', 'falling', 'stable'),
                'spike_detected': 급격 변동 감지 여부,
                'ma': 이동평균
            }
        """
        if current_atr <= 0:
            return {
                'change_pct': 0.0,
                'trend': 'stable',
                'spike_detected': False,
                'ma': 0.0
            }
        
        # 변화율 계산
        change_pct = 0.0
        if self._prev_atr > 0:
            change_pct = ((current_atr - self._prev_atr) / self._prev_atr) * 100.0
        
        # 이동평균 계산 (텔레그램 송출보다 먼저 수행)
        self._atr_history.append(current_atr)
        if len(self._atr_history) > self._ma_period:
            self._atr_history.pop(0)
        ma = sum(self._atr_history) / len(self._atr_history) if self._atr_history else 0.0
        
        # 추세 판정
        trend = 'stable'
        if change_pct > 5.0:
            trend = 'rising'
        elif change_pct < -5.0:
            trend = 'falling'
        
        # 급격 변동 감지
        spike_detected = abs(change_pct) >= self._spike_threshold_pct
        if spike_detected:
            direction = '급증' if change_pct > 0 else '급락'
            
            # 텔레그램 송출
            if self._telegram_callback:
                try:
                    message = f"🚨 ATR {direction} 알림\n\n"
                    message += f"이전 ATR: {self._prev_atr:.2f}\n"
                    message += f"현재 ATR: {current_atr:.2f}\n"
                    message += f"변화율: {change_pct:+.1f}%\n"
                    message += f"이동평균: {ma:.2f}"
                    self._telegram_callback(message)
                except (TypeError, AttributeError, RuntimeError):
                    pass
        
        # 일반 로깅 (INFO 레벨)
        #     f"[ATR-MONITOR] ATR={current_atr:.2f}, change={change_pct:+.1f}%, "
        #     f"trend={trend}, spike={spike_detected}, MA={ma:.2f}"
        # )
        
        # 이전 값 업데이트
        self._prev_atr = current_atr
        
        return {
            'change_pct': change_pct,
            'trend': trend,
            'spike_detected': spike_detected,
            'ma': ma
        }


def _get_time_based_atr_ratio(current_time: datetime.datetime, ratio_table: List[Tuple[str, str, float]], fallback_ratio: float = 1.0) -> float:
    """
    현재 시간에 해당하는 min_wave_atr_ratio 반환

    Args:
        current_time: 현재 시간 (datetime.datetime)
        ratio_table: [(시작HH:MM, 종료HH:MM(미포함), 비율), ...] 형태의 테이블
        fallback_ratio: 매칭되는 시간대가 없을 때 사용할 기본값 (상위 min_wave_atr_ratio)

    Returns:
        해당 시간대의 min_wave_atr_ratio. 매칭되는 시간대가 없으면 fallback_ratio 반환
    """
    if not ratio_table:
        return fallback_ratio

    # 문자열인 경우 datetime으로 변환
    if isinstance(current_time, str):
        # HH:MM 형식인 경우 오늘 날짜를 추가
        if ":" in current_time and len(current_time) <= 5:
            today = datetime.datetime.now().date()
            current_time = datetime.datetime.combine(today, datetime.datetime.strptime(current_time, "%H:%M").time())
        else:
            current_time = pd.Timestamp(current_time).to_pydatetime()
    elif isinstance(current_time, pd.Timestamp):
        current_time = current_time.to_pydatetime()

    current_time_only = current_time.time()

    for start_str, end_str, ratio in ratio_table:
        try:
            start = datetime.datetime.strptime(start_str, "%H:%M").time()
            end = datetime.datetime.strptime(end_str, "%H:%M").time()
            if start <= current_time_only < end:
                return ratio
        except Exception:
            continue

    return fallback_ratio


def _resolve_col(df: pd.DataFrame, name: str) -> str:
    for col in df.columns:
        if str(col).lower() == name.lower():
            return col
    return name
