"""indicators
===================
KP200 선물 Adaptive Indicator 공유 라이브러리.

Transformer 프로젝트와 SkyEbest 프로젝트가 동일한 소스를 공유합니다.

Quick Start
-----------
    from indicators import AdaptiveIndicatorManager

    manager = AdaptiveIndicatorManager()
    result  = manager.update(high, low, close)
    features = result["transformer"]   # Dict[str, float]  — Transformer 입력
    context  = result["llm_context"]   # str               — LLM 프롬프트 삽입

각 지표 직접 사용 (SkyEbest 호환)
-----------------------------------
    from indicators import AdaptiveSuperTrend, AdaptiveZigZag

    ast = AdaptiveSuperTrend()
    zz  = AdaptiveZigZag()

    for h, lo, c in bars:
        st_state = ast.update(h, lo, c)
        zz_state = zz.update(h, lo, c)

수정된 버그 목록 (vs 이전 구현)
--------------------------------
- [ST]  bars_in_trend 플립 봉 누적 (just_flipped 패턴 적용)
- [ST]  ATR 재초기화 임계: 절댓값 → 비율 기준으로 통일
- [ST]  LLM advice 딕셔너리 키 불일치 수정
- [ST]  _prev_adx 초기값 0.0 → 25.0
- [ZZ]  ER-adaptive threshold 방향 역전 수정 (mmax-er*R → mmin+er*R)
- [ZZ]  pending_confirm 교체 조건: 반대 타입이면 교체 허용
- [ZZ]  _all_swings 정리: del → 슬라이싱 재할당
- [RMA] WilderRMA.ready: count > period → count >= period (1봉 지연 수정)
"""

from .wilder_smooth import WilderRMA
from .adaptive_supertrend import (
    AdaptiveSuperTrend,
    AdaptiveSuperTrendConfig,
    SuperTrendState,
)
from .adaptive_zigzag import (
    AdaptiveZigZag,
    AdaptiveZigZagConfig,
    ZigZagState,
    SwingPoint,
    SwingType,
    FibLevels,
)
from .pivot_collector import (
    PivotCandidateCollector,
    CandidateRecord,
    CandidateSnapshot,
)
from .indicator_integration import (
    AdaptiveIndicatorManager,
    IndicatorManagerConfig,
    validate_consistency,
)
# from .atr_adaptive_pivot import (
#     ATRAdaptivePivot,
#     ATRAdaptivePivotConfig,
#     ATRAdaptivePivotState,
#     PivotPoint,
#     PivotType,
# )
# from .percent_adaptive_pivot import (
#     PercentAdaptivePivot,
#     PercentAdaptivePivotConfig,
# )
# from .kalman_turning_point import (
#     KalmanTurningPoint,
#     KalmanConfig,
#     KalmanState,
# )
# from .pivot_score_integrator import (
#     PivotScoreIntegrator,
#     IntegratorConfig,
#     IntegratorResult,
# )
# from .market_structure_break import (
#     MarketStructureBreak,
#     MSBConfig,
#     MSBState,
#     OIStructureGate,
#     OIStructureConfig,
#     BOSType,
#     StructureType,
# )
try:
    from .fractal_confirmation import (
        FractalConfirmation,
        FractalConfig,
        FractalState,
        FractalPoint,
    )
    _FRACTAL_AVAILABLE = True
except ImportError:
    _FRACTAL_AVAILABLE = False
    FractalConfirmation = FractalConfig = FractalState = FractalPoint = None

__version__ = "1.0.0"

_base_all = [
    # Core
    "WilderRMA",
    # SuperTrend
    "AdaptiveSuperTrend",
    "AdaptiveSuperTrendConfig",
    "SuperTrendState",
    # ZigZag
    "AdaptiveZigZag",
    "AdaptiveZigZagConfig",
    "ZigZagState",
    "SwingPoint",
    "SwingType",
    "FibLevels",
    # Pivot Collector
    "PivotCandidateCollector",
    "CandidateRecord",
    "CandidateSnapshot",
    # Integration
    "AdaptiveIndicatorManager",
    "IndicatorManagerConfig",
    "validate_consistency",
    # KalmanTurningPoint (Step 3) - temporarily disabled
    # "KalmanTurningPoint",
    # "KalmanConfig",
    # "KalmanState",
    # PivotScoreIntegrator (Step 3 통합) - temporarily disabled
    # "PivotScoreIntegrator",
    # "IntegratorConfig",
    # "IntegratorResult",
    # MarketStructureBreak (Step 2) - temporarily disabled
    # "MarketStructureBreak",
    # "MSBConfig",
    # "MSBState",
    # "OIStructureGate",
    # "OIStructureConfig",
    # "BOSType",
    # "StructureType",
    # ATRAdaptivePivot (ZigZag 대체 Step 1) - temporarily disabled
    # "ATRAdaptivePivot",
    # "ATRAdaptivePivotConfig",
    # "ATRAdaptivePivotState",
    # "PivotPoint",
    # "PivotType",
    # PercentAdaptivePivot (ATR 없는 퍼센트 기반) - temporarily disabled
    # "PercentAdaptivePivot",
    # "PercentAdaptivePivotConfig",
]

if _FRACTAL_AVAILABLE:
    _base_all.extend([
        # FractalConfirmation (Step 1 확증 레이어)
        "FractalConfirmation",
        "FractalConfig",
        "FractalState",
        "FractalPoint",
    ])

__all__ = _base_all
