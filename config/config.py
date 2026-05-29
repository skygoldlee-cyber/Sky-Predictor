"""
설정 관리 모듈

개선사항:
- 타입 힌팅 완비
- 검증 로직 추가
- 에러 로깅
- 환경변수 우선순위 처리
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "load_config",
    "zigzag_settings_from_dict",
    "AppConfig",
    "AdaptiveZigZagSettings",
    "AdaptiveSuperTrendSettings",
    "HybridAdaptivePivotSettings",
    "AdaptiveIndicatorSettings",
]


def _deep_merge_dict(base: Dict, override: Dict) -> Dict:
    """Merge override into base recursively (dict-only)."""
    if not isinstance(base, dict):
        base = {}
    if not isinstance(override, dict):
        return dict(base)
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out.get(k) or {}, v)
        else:
            out[k] = v
    return out


def _load_json_file(path: Path) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"Config file read failed: {path} ({e})")
        return {}


def _resolve_secrets_path(config_path: str, secrets_path: Optional[str]) -> Path:
    if secrets_path:
        return Path(secrets_path)
    base = Path(config_path)
    parent = base.parent if base.parent else Path(".")
    return parent / "config.secrets.json"


def _strip_api_key(value: Any) -> Optional[str]:
    """API 키 문자열 정규화(공백 제거, 빈 문자열은 None)."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _secrets_paths_to_merge(config_path: str) -> list[Path]:
    """config.secrets.json 후보 경로(중복 제거). 순서대로 deep-merge 후, 뒤에 오는 파일이 우선한다."""
    seen: set[str] = set()
    out: list[Path] = []

    def add(p: Path) -> None:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)

    # (1) cwd — 작업 폴더가 프로젝트일 때
    add(Path.cwd() / "config.secrets.json")
    # (2) 이 모듈(config.py)이 있는 디렉터리 — cwd와 무관하게 프로젝트 루트 시크릿 사용
    add(Path(__file__).resolve().parent / "config.secrets.json")
    # (3) 실제 config 파일 옆 (절대 경로 등으로 지정된 경우)
    try:
        cp = Path(config_path).expanduser().resolve()
        if cp.exists():
            add(cp.parent / "config.secrets.json")
    except Exception:
        pass
    # (4) APP_SECRETS_CONFIG 또는 config.json과 동일 규칙의 경로 — 가장 나중에 merge되어 최우선
    add(_resolve_secrets_path(config_path, os.environ.get("APP_SECRETS_CONFIG")))
    return out


@dataclass
class AIProviderConfig:
    """AI 제공자 설정"""
    anthropic_key: Optional[str] = None
    openai_key: Optional[str] = None
    gemini_key: Optional[str] = None


def log_ai_provider_keys_loaded(
    ap: AIProviderConfig,
    *,
    log_to: Optional[logging.Logger] = None,
) -> None:
    """로드된 AI 키 요약(값은 로그에 남기지 않음). 하나 이상 있을 때만 INFO.

    ``load_config()``는 보통 ``setup_logging()``보다 먼저 호출되므로, 기본 루트
    로거는 INFO를 버린다. ``prediction.log``에 남기려면 로깅 초기화 직후
    ``log_to=`` 에 앱 로거를 넘겨 호출한다.
    """
    log = log_to or logging.getLogger(__name__)
    try:
        a = bool(ap.anthropic_key)
        o = bool(ap.openai_key)
        g = bool(ap.gemini_key)
    except Exception:
        return
    if not (a or o or g):
        return
    log.info(
        f"[CONFIG] AI provider keys loaded: anthropic={'yes' if a else 'no'}, openai={'yes' if o else 'no'}, gemini={'yes' if g else 'no'}"
    )


@dataclass
class EBestConfig:
    """eBest 인증 설정"""
    appkey: Optional[str] = None
    appsecretkey: Optional[str] = None


@dataclass
class OptionSubscriptionConfig:
    """옵션 구독 설정"""
    itm: int = 6
    otm_open_min: float = 0.30
    max_otm_calls: int = 0
    max_otm_puts: int = 0
    wait_sec: int = 2
    # OI 지지저항 분석용 구독 창 설정 (금일 추가)
    preopen_oh0_window: int = 10          # pre-open OH0 구독 창 (ATM ± N)
    oi_itm_count: int = 10               # OI 창 내가 방향 개수
    oi_otm_count: int = 10               # OI 창 외가 방향 개수
    oi_rebalance_interval_sec: float = 60.0  # 장중 OC0 재구독 주기(초)


@dataclass
class OptionMinuteOhlcvConfig:
    """옵션 OC0 틱을 심볼별 분봉 OHLCV로 집계하는 설정"""
    enabled: bool = False
    atm_window: int = 2


@dataclass
class MinuteLookbackConfig:
    """tick_processor 분봉 조회 기본 lookback 설정"""
    futures: int = 120
    options: int = 120


@dataclass
class TelegramConfig:
    """텔레그램 알림/명령 수신 설정"""
    enabled: bool = False
    option_flow_status_enabled: bool = True
    option_flow_status_cooldown_sec: float = 0.0
    option_flow_status_intraday_only: bool = False
    option_flow_status_disable_after_close: bool = True
    option_flow_interp_sr_warn: float = 1.5
    option_flow_interp_sr_hot: float = 2.0
    option_flow_interp_pt_low: float = 0.008
    option_flow_interp_pt_high: float = 0.03
    option_flow_interp_pcr_v_low: float = 0.90
    option_flow_interp_pcr_v_high: float = 1.10
    option_flow_interp_pcr_oi_low: float = 0.95
    option_flow_interp_pcr_oi_high: float = 1.05


@dataclass
class PredictionConfig:
    """예측 설정"""
    minutes: int = 5  # 5, 10, 30 중 하나
    use_llm: bool = True
    heuristic_fallback: bool = True
    # None이면 ebest_live에서 max(60, prediction_minutes*30) 초 사용
    heuristic_flip_min_interval_sec: Optional[float] = None
    heuristic_flip_include_hold_transition: bool = False
    rule_based_weights: Optional[Dict[str, float]] = None
    rule_based_mom_multiplier: float = 1.0
    numeric_predictor: str = "transformer"  # transformer|rule_based
    model_class: str = "transformer"        # transformer|patch_tst|mamba — 수치 예측 모델 구조 선택
    patch_len: int = 8                      # PatchTST 패치 길이 (patch_tst 선택 시 유효)
    stride: int = 4                         # PatchTST 슬라이딩 간격 (patch_tst 선택 시 유효)
    mamba_d_state: int = 16                # Mamba SSM 상태 차원 (mamba 선택 시 유효)
    mamba_seq_len: int = 60                # Mamba 입력 시퀀스 길이 (60~240, mamba 선택 시 유효)
    mamba_enabled: bool = False            # True: 앙상블에 Mamba 포함 (mamba_weights_path 필요)
    mamba_weights_path: str = ""           # Mamba 가중치 경로 (.pt). 비어 있으면 비활성
    mamba_weight: float = 0.33             # 앙상블 내 Mamba 가중치 비율 (0 < w < 1)
    multiscale_5m: bool = False             # True: 5분봉 MS5_KEYS 8개 피처 추가 (PAST_UNKNOWN_DIM +8)
    conformal_alpha: float = 0.1            # Conformal Prediction 오차율 (0.1 → 90% 커버리지 구간)
    conformal_path: str = ""               # Conformal 분위수 저장 경로 (.npz). 비어있으면 보정 없음
    option_feature_set: str = "v1"  # v1(기존 OPT 7) | v2(확장 OPT)
    # calc_pcr: ATM 행사가 기준 위·아래 각 N개 행사가만 합산. 0이면 ATM 1줄만, 크면 체인을 넓게.
    pcr_atm_strikes_each_side: int = 5
    min_minute_bars_required: int = 20
    seq_len: int = 60
    fo0_stale_sec: int = 10
    fo0_log_schema: bool = True
    preferred_provider: str = ""
    dual_llm: bool = False
    dual_llm_primary_provider: str = "gpt"
    buy_threshold: float = 0.62
    sell_threshold: float = 0.38
    confidence_high_margin: float = 0.15
    confidence_mid_margin: float = 0.08
    target_day: str = ""  # 조회할 타겟 날짜 (YYYYMMDD 형식, 빈 문자열이면 오늘)
    confidence_spread_max_for_high: float = 1.0
    # Conformal 구간 폭이 넓을 때 HIGH/MEDIUM을 낮춤 (검증 세트 Brier/ECE로 튜닝)
    confidence_conformal_width_max_for_high: float = 0.35
    confidence_conformal_width_max_for_medium: float = 0.55
    transformer_weight: float = 0.5
    transformer_weights_path: str = ""
    tft_weights_path: str = ""
    tft_horizon: int = 300
    disagreement_hold: bool = True
    disagreement_hold_prob_diff_max: float = 0.1
    # 레짐별 불일치 HOLD 임계(선택). 예: {"RANGE": 0.08, "STRONG_UP": 0.12}
    disagreement_hold_prob_diff_max_by_regime: Optional[Dict[str, float]] = None
    ensemble_agreement_confidence_boost: bool = True
    ensemble_agreement_prob_diff_max: float = 0.06
    guard_basis_hold_thr: float = 2.5
    guard_basis_downgrade_thr: float = 1.5
    guard_atm_spread_pct_thr: float = 1.5
    guard_atm_liq_log_thr: float = 2.0
    llm_min_interval_sec: float = 30.0
    gemini_timeout_sec: Optional[float] = None  # None이면 llm_timeout_sec 사용
    llm_provider_cooldown_on_timeout_sec: float = 60.0
    tick_size: float = 0.05
    feedback_threshold_ticks: int = 10
    feedback_skip_hold_ticks: int = 2
    feedback_weight_high: float = 1.0
    feedback_weight_mid: float = 0.5
    feedback_weight_low: float = 0.25
    feedback_use_price_snapshot: bool = True
    feedback_snapshot_tolerance_sec: float = 30.0
    feedback_snapshot_required: bool = False
    fc0_stale_threshold_sec: float = 10.0
    fc0_stale_cooldown_sec: float = 60.0
    oi_alert_cooldown_sec: float = 300.0  # OI 지지/저항 변경 알람 쿨다운 (초)


@dataclass
class AdaptiveSuperTrendSettings:
    atr_min_period: int = 7
    atr_max_period: int = 21
    multiplier_min: float = 1.5
    multiplier_max: float = 4.0
    er_period: int = 10
    adx_period: int = 14
    use_bb_correction: bool = True
    bb_period: int = 20
    bb_std: float = 2.0
    smooth_period: int = 3


@dataclass
class AdaptiveZigZagSettings:
    atr_multiplier: float = 1.5
    atr_period: int = 14
    er_period: int = 10
    atr_multiplier_min: float = 1.0
    atr_multiplier_max: float = 4.0
    pivot_threshold_min_pct: float = 0.3
    pivot_threshold_max_pct: float = 3.0
    major_swing_ratio: float = 2.0
    max_swings: int = 20
    confirmation_bars: int = 2
    confirmation_bars_ranging: int = 2
    confirmation_bars_unknown: int = 3
    freeze_on_confirm: bool = True
    min_wave_bars: int = 1
    min_wave_pct: float = 0.4
    max_wait_bars: int = 0          # [FIX-7] 0=무제한, >0: pending 자동취소 봉수
    cluster_tolerance_pct: float = 0.3
    structure_lookback_swings: int = 30
    structure_points: int = 4
    # [SESSION-MW] 시간대별 min_wave_bars 테이블
    # 빈 리스트 → 단일 min_wave_bars 폴백 (하위 호환)
    # JSON 형식: [[시작HH:MM, 종료HH:MM(미포함), min_wave_bars], ...]
    session_min_wave_bars_table: list = field(default_factory=list)
    # ── [ATR-FILTER] 시간대별 동적 ATR 비율 테이블 ─────────────────
    # 빈 리스트 → 단일 min_wave_atr_ratio 폴백 (하위 호환)
    # JSON 형식: [[시작HH:MM, 종료HH:MM(미포함), min_wave_atr_ratio], ...]
    session_min_wave_atr_ratio_table: list = field(default_factory=list)
    # ── [ATR-FILTER] ATR 기반 필터링 파라미터 ─────────────────────
    use_atr_based_filtering: bool = False  # ATR 기반 필터링 활성화 여부
    min_wave_atr_ratio: float = 0.5  # 피봇으로 인식되기 위한 최소 파동 크기 (ATR 배수)
    cluster_atr_ratio: float = 0.5  # 피봇 클러스터링에 사용되는 ATR 배수
    # ── 장초반 ATR multiplier 조절 ─────────────────────────────────
    # [MAINT-2 FIX] early_session_atr_multiplier_max 제거
    # session_min_wave_atr_ratio_table로 통합 완료
    early_session_start_time: str = "09:00"
    early_session_end_time: str = "09:30"
    # ── 피보나치 레벨 ───────────────────────────────────────────────
    fib_ratios: list = field(default_factory=lambda: [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618])
    # ── 구조 다수결 임계값 ─────────────────────────────────────────
    structure_majority_threshold: float = 0.7
    # ── bars_since 기반 임계값 decay ───────────────────────────────
    decay_start_bars: int = 30
    decay_rate_per_bar: float = 0.005
    decay_max_pct: float = 0.3
    # ── 다중 시간프레임 ─────────────────────────────────────────────
    multi_timeframe_enabled: bool = False
    multi_timeframe_scales: list = field(default_factory=lambda: [1, 5, 15])
    multi_timeframe_consensus_threshold: int = 2
    multi_timeframe_price_tolerance_pct: float = 1.0
    multi_timeframe_index_tolerance_multiplier: float = 2.0
    # ── major 파동 기준 ─────────────────────────────────────────────
    major_wave_ratio: float = 1.5
    major_wave_lookback: int = 3
    # ── Directional ER 불일치 처리 ─────────────────────────────────
    der_mismatch_threshold: float = 0.3
    der_mismatch_mult_ratio: float = 0.7
    # ── 피봇 후보 수집기 ────────────────────────────────────────────
    enable_pivot_collector: bool = False
    pivot_collector_max_sequence: int = 120

    def to_zigzag_config(
        self,
        pivot_lifecycle_log: bool = True,
        pivot_lifecycle_log_prefix: str = "",
    ) -> Any:
        """AdaptiveZigZagConfig 인스턴스를 생성하여 반환한다.

        단일 소스 원칙(SSOT): AdaptiveZigZag 인스턴스를 생성하는 모든 경로
        (pipeline.py, chart_engine.py, data_builder.py)는 이 메서드를 통해
        AdaptiveZigZagConfig 를 얻어야 한다. 파라미터 하드코딩/중복 파싱을 방지한다.

        Args:
            pivot_lifecycle_log: 피봇 이벤트 로그 활성화 여부
            pivot_lifecycle_log_prefix: 로그 prefix (예: "[KOSPI]", "[KP200]")

        Returns:
            AdaptiveZigZagConfig 인스턴스. indicators 패키지 import에 실패하면
            ImportError 를 그대로 전파한다.
        """
        from indicators.adaptive_zigzag import AdaptiveZigZagConfig  # type: ignore
        return AdaptiveZigZagConfig(
            atr_multiplier=self.atr_multiplier,
            atr_period=self.atr_period,
            er_period=self.er_period,
            atr_multiplier_min=self.atr_multiplier_min,
            atr_multiplier_max=self.atr_multiplier_max,
            pivot_threshold_min_pct=self.pivot_threshold_min_pct,
            pivot_threshold_max_pct=self.pivot_threshold_max_pct,
            major_swing_ratio=self.major_swing_ratio,
            max_swings=self.max_swings,
            confirmation_bars=self.confirmation_bars,
            confirmation_bars_ranging=self.confirmation_bars_ranging,
            confirmation_bars_unknown=self.confirmation_bars_unknown,
            freeze_on_confirm=self.freeze_on_confirm,
            min_wave_bars=self.min_wave_bars,
            min_wave_pct=self.min_wave_pct,
            max_wait_bars=self.max_wait_bars,
            cluster_tolerance_pct=self.cluster_tolerance_pct,
            structure_lookback_swings=self.structure_lookback_swings,
            structure_points=self.structure_points,
            session_min_wave_bars_table=list(self.session_min_wave_bars_table),
            session_min_wave_atr_ratio_table=list(self.session_min_wave_atr_ratio_table),
            use_atr_based_filtering=self.use_atr_based_filtering,
            min_wave_atr_ratio=self.min_wave_atr_ratio,
            cluster_atr_ratio=self.cluster_atr_ratio,
            pivot_lifecycle_log=pivot_lifecycle_log,
            pivot_lifecycle_log_prefix=pivot_lifecycle_log_prefix,
            # ── 장초반 ATR 조절 ─────────────────────────────────────
            # [MAINT-2 FIX] early_session_atr_multiplier_max 제거
            early_session_start_time=self.early_session_start_time,
            early_session_end_time=self.early_session_end_time,
            # ── 피보나치 ────────────────────────────────────────────
            fib_ratios=list(self.fib_ratios),
            # ── 구조 판정 ────────────────────────────────────────────
            structure_majority_threshold=self.structure_majority_threshold,
            # ── decay ───────────────────────────────────────────────
            decay_start_bars=self.decay_start_bars,
            decay_rate_per_bar=self.decay_rate_per_bar,
            decay_max_pct=self.decay_max_pct,
            # ── 다중 시간프레임 ─────────────────────────────────────
            multi_timeframe_enabled=self.multi_timeframe_enabled,
            multi_timeframe_scales=list(self.multi_timeframe_scales),
            multi_timeframe_consensus_threshold=self.multi_timeframe_consensus_threshold,
            multi_timeframe_price_tolerance_pct=self.multi_timeframe_price_tolerance_pct,
            multi_timeframe_index_tolerance_multiplier=self.multi_timeframe_index_tolerance_multiplier,
            # ── major 파동 ──────────────────────────────────────────
            major_wave_ratio=self.major_wave_ratio,
            major_wave_lookback=self.major_wave_lookback,
            # ── DER 불일치 ──────────────────────────────────────────
            der_mismatch_threshold=self.der_mismatch_threshold,
            der_mismatch_mult_ratio=self.der_mismatch_mult_ratio,
            # ── 피봇 수집기 ─────────────────────────────────────────
            enable_pivot_collector=self.enable_pivot_collector,
            pivot_collector_max_sequence=self.pivot_collector_max_sequence,
        )


@dataclass
class RangingFilterSettings:
    """횡보장 SuperTrend flip 억제 필터 설정.

    SuperTrend는 횡보장에서 잦은 flip(whipsaw)을 발생시킨다.
    4단계 필터를 통해 추세성이 약한 구간의 무의미한 신호를 억제한다.
    각 필터는 독립적으로 enabled/disabled 가능.

    Attributes:
        enabled: 전체 필터 활성화 여부 (False → 모든 단계 건너뜀)
        adx_min: ADX 최솟값. 미만이면 억제. (기본 15.0)
        er_min: Efficiency Ratio 최솟값. 미만이면 억제. (기본 0.15)
        use_zigzag_structure: ZigZag structure='ranging' 판정 시 억제 여부
        whipsaw_min_bars: flip 후 이 봉 수 미만에서 재flip 시 억제. (기본 2)
        use_adx_filter: ADX 필터 개별 on/off
        use_er_filter: ER 필터 개별 on/off
        use_whipsaw_filter: whipsaw 필터 개별 on/off
    """
    enabled: bool = True
    adx_min: float = 15.0
    er_min: float = 0.15
    use_zigzag_structure: bool = True
    whipsaw_min_bars: int = 2
    use_adx_filter: bool = True
    use_er_filter: bool = True
    use_whipsaw_filter: bool = True


@dataclass
class PivotProximityAlertSettings:
    enabled: bool = True
    max_bars_diff: int = 1
    telegram_enabled: bool = True


@dataclass
class PivotCandidateAlertSettings:
    enabled: bool = True
    telegram_enabled: bool = True
    events: List[str] = field(default_factory=lambda: ["registered", "changed", "cancelled"])
    change_cooldown_sec: float = 60.0


@dataclass
class HybridAdaptivePivotSettings:
    """HybridAdaptivePivot 설정 (ATR + 퍼센트 혼합)"""
    # 핵심 임계값
    base_pct: float = 0.3  # 퍼센트 임계값 (%)
    base_multiplier: float = 2.0  # ATR 배수
    atr_weight: float = 0.5  # ATR 혼합 비율 [0, 1]
    
    # ATR/ER 계산
    atr_period: int = 14  # WilderRMA 주기
    multiplier_min: float = 0.8  # ER 기반 배수 하한
    multiplier_max: float = 2.0  # ER 기반 배수 상한
    er_period: int = 10  # Kaufman ER 구간
    
    # 후보 확정 / 필터
    confirmation_bars: int = 1  # 확인 봉 수 (0=즉시 확정)
    cancel_ratio: float = 0.3  # 되돌림 취소 비율
    min_wave_pct: float = 0.15  # 최소 파동 퍼센트 필터
    min_wave_atr_ratio: float = 0.5  # 최소 파동 ATR 비율 필터
    
    # 웜업 / 저장
    warmup_bars: int = 20  # ATR/ER 안정화 최소 봉
    max_pivots: int = 30  # 보관 최대 피봇 수
    
    # Layer B: AdaptiveParamEngine (레짐 기반 atr_weight 동적 조정)
    use_adaptive_engine: bool = False
    regime_atr_weight_table: dict = field(default_factory=lambda: {
        "trend_strong_up": 0.75,
        "trend_strong_dn": 0.75,
        "trend_weak_up": 0.55,
        "trend_weak_dn": 0.55,
        "chop_low_vol": 0.35,
        "chop_high_vol": 0.85,
        "volatile": 0.90,
        "unknown": 0.50,
    })
    
    # Layer C: Fractal 교차 확증
    use_fractal_confirmation: bool = False
    fractal_lookback: int = 2
    fractal_volume_spike: float = 1.3
    fractal_price_tolerance_pct: float = 0.3
    fractal_bonus: float = 0.15


@dataclass
class AdaptiveIndicatorSettings:
    """적응형 지표 설정"""
    enabled: bool = True
    dual_mode: bool = False  # [FIX] 데이터 소스 분리: predictor 내부 dual_mode 비활성화
    kospi_symbol: str = "KOSPI 지수"
    futures_symbol: str = "KP200 선물"
    warmup_bars: int = 15
    min_swings_for_ready: int = 4
    supertrend_pivot_filter: bool = True  # 슈퍼트렌드 신호 참조 피봇 필터 활성화
    # 피봇 신호 간 최소 간격 (분봉) - 시간대별 테이블이 있을 경우 기본값으로만 사용
    min_pivot_interval_bars: int = 10
    # 시간대별 피봇 신호 최소 간격 테이블 [[시작시간, 종료시간, 간격], ...]
    session_min_pivot_interval_table: List[List[Any]] = field(default_factory=list)
    supertrend: AdaptiveSuperTrendSettings = field(default_factory=AdaptiveSuperTrendSettings)
    kospi_supertrend: Optional[AdaptiveSuperTrendSettings] = None
    futures_supertrend: Optional[AdaptiveSuperTrendSettings] = None
    zigzag: AdaptiveZigZagSettings = field(default_factory=AdaptiveZigZagSettings)
    kospi_zigzag: Optional[AdaptiveZigZagSettings] = None
    futures_zigzag: Optional[AdaptiveZigZagSettings] = None
    # HybridAdaptivePivot 설정 (ATR + 퍼센트 혼합)
    hap: HybridAdaptivePivotSettings = field(default_factory=HybridAdaptivePivotSettings)
    kospi_hap: Optional[HybridAdaptivePivotSettings] = None
    futures_hap: Optional[HybridAdaptivePivotSettings] = None
    ranging_filter: RangingFilterSettings = field(default_factory=RangingFilterSettings)
    pivot_proximity_alert: PivotProximityAlertSettings = field(default_factory=PivotProximityAlertSettings)
    pivot_candidate_alert: PivotCandidateAlertSettings = field(default_factory=PivotCandidateAlertSettings)


@dataclass
class AppConfig:
    """전체 애플리케이션 설정"""
    ai_providers: AIProviderConfig
    ebest: EBestConfig
    options_subscription: OptionSubscriptionConfig
    option_minute_ohlcv: OptionMinuteOhlcvConfig
    minute_lookback: MinuteLookbackConfig  # 피처 계산/LLM 컨텍스트용 (warmup_bars와 분리)
    prediction: PredictionConfig
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    adaptive_indicator: AdaptiveIndicatorSettings = field(default_factory=AdaptiveIndicatorSettings)
    trade_gate: Optional[Any] = None   # TradeGateConfig (trade_gate.py) — None 시 게이트 비활성
    log_file: str = "prediction.log"
    log_level: str = "INFO"
    adaptive_mode: bool = False

    @classmethod
    def from_file(cls, config_path: str) -> "AppConfig":
        """
        설정 파일에서 로드
        
        Args:
            config_path: 설정 파일 경로
            
        Returns:
            AppConfig 인스턴스
            
        Raises:
            FileNotFoundError: 파일이 없을 때
            ValueError: 설정 파일 형식 오류
        """
        path = Path(config_path)
        
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            cfg = cls._default_config()
            cls._backfill_missing_ai_keys(cfg, config_path)
            return cfg
        
        try:
            data = _load_json_file(path)

            for sp in _secrets_paths_to_merge(config_path):
                secrets_data = _load_json_file(sp)
                if secrets_data:
                    data = _deep_merge_dict(data, secrets_data)
            
            if not isinstance(data, dict):
                raise ValueError("Config must be a JSON object")
            
            cfg = cls._from_dict(data)
            try:
                from core.utils import set_expiry_holidays
                set_expiry_holidays(data.get("market_holidays"))
            except Exception:
                pass
            cls._backfill_missing_ai_keys(cfg, config_path)
            cfg.validate()
            return cfg
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse config file: {e}")
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    @classmethod
    def _parse_ai_providers(cls, data: Dict) -> AIProviderConfig:
        """AI 제공자 설정 파싱"""
        ai_data = data.get("ai_providers", {})
        ai_data = ai_data if isinstance(ai_data, dict) else {}
        ad = ai_data.get("anthropic", {}) if isinstance(ai_data.get("anthropic"), dict) else {}
        od = ai_data.get("openai", {}) if isinstance(ai_data.get("openai"), dict) else {}
        gd = ai_data.get("gemini", {}) if isinstance(ai_data.get("gemini"), dict) else {}
        return AIProviderConfig(
            anthropic_key=_strip_api_key(os.environ.get("ANTHROPIC_API_KEY"))
            or _strip_api_key(ad.get("api_key")),
            openai_key=_strip_api_key(os.environ.get("OPENAI_API_KEY"))
            or _strip_api_key(od.get("api_key")),
            gemini_key=_strip_api_key(os.environ.get("GEMINI_API_KEY"))
            or _strip_api_key(gd.get("api_key")),
        )

    @classmethod
    def _parse_ebest_config(cls, data: Dict) -> EBestConfig:
        """eBest 인증 설정 파싱"""
        ebest_data = data.get("ebest", {})
        ebest_data = ebest_data if isinstance(ebest_data, dict) else {}
        return EBestConfig(
            appkey=os.environ.get("EBEST_APPKEY")
            or os.environ.get("EBEST_APP_KEY")
            or ebest_data.get("appkey"),
            appsecretkey=os.environ.get("EBEST_APPSECRET")
            or os.environ.get("EBEST_APP_SECRET")
            or ebest_data.get("appsecretkey"),
        )

    @classmethod
    def _parse_option_subscription(cls, data: Dict) -> OptionSubscriptionConfig:
        """옵션 구독 설정 파싱"""
        opt_data = data.get("options_subscription", {})
        return OptionSubscriptionConfig(
            itm=cls._safe_int(opt_data.get("itm"), 6),
            otm_open_min=float(opt_data.get("otm_open_min", 0.30) or 0.30),
            max_otm_calls=cls._safe_int(opt_data.get("max_otm_calls"), 0),
            max_otm_puts=cls._safe_int(opt_data.get("max_otm_puts"), 0),
            wait_sec=cls._safe_int(opt_data.get("wait_sec"), 2),
            preopen_oh0_window=min(20, max(0, cls._safe_int(opt_data.get("preopen_oh0_window"), 10))),
            oi_itm_count=min(30, max(1, cls._safe_int(opt_data.get("oi_itm_count"), 10))),
            oi_otm_count=min(30, max(1, cls._safe_int(opt_data.get("oi_otm_count"), 10))),
            oi_rebalance_interval_sec=max(10.0, min(600.0, float(opt_data.get("oi_rebalance_interval_sec", 60.0) or 60.0))),
        )

    @classmethod
    def _parse_option_minute_ohlcv(cls, data: Dict) -> OptionMinuteOhlcvConfig:
        """옵션 분봉 OHLCV 집계 설정 파싱"""
        om_data = data.get("option_minute_ohlcv")
        om_data = om_data if isinstance(om_data, dict) else {}
        return OptionMinuteOhlcvConfig(
            enabled=bool(om_data.get("enabled", False)),
            atm_window=cls._safe_int(om_data.get("atm_window"), 2),
        )

    @classmethod
    def _parse_minute_lookback(cls, data: Dict) -> MinuteLookbackConfig:
        """분봉 lookback 설정 파싱"""
        ml_data = data.get("minute_lookback")
        ml_data = ml_data if isinstance(ml_data, dict) else {}
        return MinuteLookbackConfig(
            futures=cls._safe_int(ml_data.get("futures"), 60),
            options=cls._safe_int(ml_data.get("options"), 60),
        )

    @classmethod
    def _parse_telegram_config(cls, data: Dict) -> TelegramConfig:
        """텔레그램 설정 파싱"""
        telegram_data = data.get("telegram")
        telegram_data = telegram_data if isinstance(telegram_data, dict) else {}
        return TelegramConfig(
            enabled=bool(telegram_data.get("enabled", False)),
            option_flow_status_enabled=bool(telegram_data.get("option_flow_status_enabled", True)),
            option_flow_status_cooldown_sec=max(
                0.0,
                float(telegram_data.get("option_flow_status_cooldown_sec", 0.0) or 0.0),
            ),
            option_flow_status_intraday_only=bool(telegram_data.get("option_flow_status_intraday_only", False)),
            option_flow_status_disable_after_close=bool(
                telegram_data.get("option_flow_status_disable_after_close", True)
            ),
            option_flow_interp_sr_warn=float(telegram_data.get("option_flow_interp_sr_warn", 1.5) or 1.5),
            option_flow_interp_sr_hot=float(telegram_data.get("option_flow_interp_sr_hot", 2.0) or 2.0),
            option_flow_interp_pt_low=float(telegram_data.get("option_flow_interp_pt_low", 0.008) or 0.008),
            option_flow_interp_pt_high=float(telegram_data.get("option_flow_interp_pt_high", 0.03) or 0.03),
            option_flow_interp_pcr_v_low=float(telegram_data.get("option_flow_interp_pcr_v_low", 0.90) or 0.90),
            option_flow_interp_pcr_v_high=float(telegram_data.get("option_flow_interp_pcr_v_high", 1.10) or 1.10),
            option_flow_interp_pcr_oi_low=float(telegram_data.get("option_flow_interp_pcr_oi_low", 0.95) or 0.95),
            option_flow_interp_pcr_oi_high=float(telegram_data.get("option_flow_interp_pcr_oi_high", 1.05) or 1.05),
        )

    @classmethod
    def _from_dict(cls, data: Dict) -> "AppConfig":
        """딕셔너리에서 설정 생성"""
        ai_providers = cls._parse_ai_providers(data)
        ebest = cls._parse_ebest_config(data)
        options_subscription = cls._parse_option_subscription(data)
        option_minute_ohlcv = cls._parse_option_minute_ohlcv(data)
        minute_lookback = cls._parse_minute_lookback(data)
        telegram = cls._parse_telegram_config(data)
        
        # 예측 설정
        pred_cfg = data.get("prediction")
        pred_cfg = pred_cfg if isinstance(pred_cfg, dict) else {}

        # ARC-04: pred_cfg → data 우선순위 조회 헬퍼 (이중 get 패턴 제거)
        def _get(key: str, default, cast=None):
            """pred_cfg에 값이 있으면 그것을, 없으면 data에서 조회하고 default를 반환."""
            val = pred_cfg.get(key)
            if val is None:
                val = data.get(key)
            if val is None:
                val = default
            if cast is not None and val is not None:
                try:
                    return cast(val)
                except (TypeError, ValueError):
                    return cast(default) if default is not None else None
            return val

        pred_minutes = data.get("prediction_minutes") or data.get("prediction-minutes")
        pred_minutes = cls._safe_int(pred_minutes, 5)

        # 예측 시간 검증
        if pred_minutes not in (5, 10, 30):
            logger.warning(f"Invalid prediction_minutes: {pred_minutes}, using 5")
            pred_minutes = 5

        prediction = PredictionConfig(
            minutes=pred_minutes,
            use_llm=data.get("use_llm", True),
            heuristic_fallback=data.get("heuristic_fallback", True),
            heuristic_flip_min_interval_sec=_get("heuristic_flip_min_interval_sec", None, float),
            heuristic_flip_include_hold_transition=bool(pred_cfg.get("heuristic_flip_include_hold_transition", False)),
            rule_based_weights=cls._parse_regime_float_map(pred_cfg.get("rule_based_weights")),
            rule_based_mom_multiplier=float(pred_cfg.get("rule_based_mom_multiplier") or 1.0),
            numeric_predictor=str(pred_cfg.get("numeric_predictor") or "transformer"),
            model_class=str(pred_cfg.get("model_class") or "transformer").strip().lower(),
            patch_len=cls._safe_int(pred_cfg.get("patch_len"), 8),
            stride=cls._safe_int(pred_cfg.get("stride"), 4),
            mamba_d_state=cls._safe_int(pred_cfg.get("mamba_d_state"), 16),
            mamba_seq_len=cls._safe_int(pred_cfg.get("mamba_seq_len"), 60),
            mamba_enabled=bool(pred_cfg.get("mamba_enabled", False)),
            mamba_weights_path=str(pred_cfg.get("mamba_weights_path") or ""),
            mamba_weight=max(0.0, min(1.0, float(pred_cfg.get("mamba_weight") or 0.33))),
            multiscale_5m=bool(pred_cfg.get("multiscale_5m", False)),
            conformal_alpha=max(0.01, min(0.5, float(pred_cfg.get("conformal_alpha") or 0.1))),
            conformal_path=str(pred_cfg.get("conformal_path") or ""),
            option_feature_set=str(pred_cfg.get("option_feature_set") or "v1"),
            pcr_atm_strikes_each_side=max(
                0,
                min(50, cls._safe_int(pred_cfg.get("pcr_atm_strikes_each_side"), 5)),
            ),
            min_minute_bars_required=cls._safe_int(pred_cfg.get("min_minute_bars_required"), 20),
            seq_len=cls._safe_int(data.get("seq_len"), 60),
            fo0_stale_sec=cls._safe_int(data.get("fo0_stale_sec"), 10),
            fo0_log_schema=bool(data.get("fo0_log_schema", True)),
            preferred_provider=str(data.get("preferred_provider") or data.get("llm_preferred_provider") or ""),
            dual_llm=bool(pred_cfg.get("dual_llm", data.get("dual_llm", False))),
            dual_llm_primary_provider=str(
                pred_cfg.get("dual_llm_primary_provider")
                or data.get("dual_llm_primary_provider")
                or "gpt"
            ),
            buy_threshold=_get("buy_threshold", 0.62, float),
            sell_threshold=_get("sell_threshold", 0.38, float),
            transformer_weight=_get("transformer_weight", 0.5, float),
            transformer_weights_path=str(
                pred_cfg.get("transformer_weights_path")
                or data.get("transformer_weights_path")
                or ""
            ),
            tft_weights_path=str(pred_cfg.get("tft_weights_path") or data.get("tft_weights_path") or ""),
            tft_horizon=cls._safe_int(_get("tft_horizon", 300), 300),
            disagreement_hold=_get("disagreement_hold", True, bool),
            disagreement_hold_prob_diff_max=_get("disagreement_hold_prob_diff_max", 0.1, float),
            disagreement_hold_prob_diff_max_by_regime=cls._parse_regime_float_map(
                pred_cfg.get("disagreement_hold_prob_diff_max_by_regime")
            ),
            ensemble_agreement_confidence_boost=_get("ensemble_agreement_confidence_boost", True, bool),
            ensemble_agreement_prob_diff_max=_get("ensemble_agreement_prob_diff_max", 0.06, float),
            confidence_high_margin=_get("confidence_high_margin", 0.15, float),
            confidence_mid_margin=_get("confidence_mid_margin", 0.08, float),
            confidence_spread_max_for_high=_get("confidence_spread_max_for_high", 1.0, float),
            confidence_conformal_width_max_for_high=_get("confidence_conformal_width_max_for_high", 0.35, float),
            confidence_conformal_width_max_for_medium=_get("confidence_conformal_width_max_for_medium", 0.55, float),
            guard_basis_hold_thr=_get("guard_basis_hold_thr", 2.5, float),
            guard_basis_downgrade_thr=_get("guard_basis_downgrade_thr", 1.5, float),
            guard_atm_spread_pct_thr=_get("guard_atm_spread_pct_thr", 1.5, float),
            guard_atm_liq_log_thr=_get("guard_atm_liq_log_thr", 2.0, float),
            llm_min_interval_sec=_get("llm_min_interval_sec", 30.0, float),
            gemini_timeout_sec=_get("gemini_timeout_sec", None, float),
            llm_provider_cooldown_on_timeout_sec=max(
                0.0,
                _get("llm_provider_cooldown_on_timeout_sec", 60.0, float),
            ),
            tick_size=_get("tick_size", 0.05, float),
            feedback_threshold_ticks=cls._safe_int(_get("feedback_threshold_ticks", 10), 10),
            feedback_skip_hold_ticks=cls._safe_int(_get("feedback_skip_hold_ticks", 2), 2),
            feedback_weight_high=_get("feedback_weight_high", 1.0, float),
            feedback_weight_mid=_get("feedback_weight_mid", 0.5, float),
            feedback_weight_low=_get("feedback_weight_low", 0.25, float),
            feedback_use_price_snapshot=_get("feedback_use_price_snapshot", True, bool),
            feedback_snapshot_tolerance_sec=_get("feedback_snapshot_tolerance_sec", 30.0, float),
            feedback_snapshot_required=_get("feedback_snapshot_required", False, bool),
            fc0_stale_threshold_sec=_get("fc0_stale_threshold_sec", 10.0, float),
            fc0_stale_cooldown_sec=_get("fc0_stale_cooldown_sec", 60.0, float),
            oi_alert_cooldown_sec=max(10.0, _get("oi_alert_cooldown_sec", 300.0, float)),
        )


        # TradeGateConfig — trade_gate.py 에서 지연 임포트 (순환 방지)
        trade_gate_cfg = None
        try:
            from trading.gate import TradeGateConfig
            tg_data = data.get("trade_gate")
            tg_data = tg_data if isinstance(tg_data, dict) else {}
            trade_gate_cfg = TradeGateConfig.from_dict(tg_data)
        except ImportError:
            logger.debug("trade_gate module not found - trade_gate_cfg=None")
        except Exception as e:
            logger.warning(f"trade_gate config parse failed: {e}")

        ad_data = data.get("adaptive_indicator")
        ad_data = ad_data if isinstance(ad_data, dict) else {}
        # [SSOT] 모듈 레벨 함수 zigzag_settings_from_dict 를 classmethod 내에서
        # 호출하기 위해 forward-ref 방식으로 참조한다.
        # (함수가 클래스 하단에 정의되므로 클래스 정의 시점에 아직 없음)
        def _zz_from_dict(d, base=None):
            return zigzag_settings_from_dict(d, base)  # noqa: F821  모듈 로드 후 resolve
        st_data = ad_data.get("supertrend")
        st_data = st_data if isinstance(st_data, dict) else {}
        zz_data = ad_data.get("zigzag")
        zz_data = zz_data if isinstance(zz_data, dict) else {}

        # ── [ATR-FILTER] kospi_zigzag, futures_zigzag 파싱 ─────────────
        kospi_zz_data = ad_data.get("kospi_zigzag")
        kospi_zz_data = kospi_zz_data if isinstance(kospi_zz_data, dict) else {}
        futures_zz_data = ad_data.get("futures_zigzag")
        futures_zz_data = futures_zz_data if isinstance(futures_zz_data, dict) else {}

        # ── [HAP] hap, kospi_hap, futures_hap 파싱 ─────────────────────
        hap_data = ad_data.get("hap")
        hap_data = hap_data if isinstance(hap_data, dict) else {}
        kospi_hap_data = ad_data.get("kospi_hap")
        kospi_hap_data = kospi_hap_data if isinstance(kospi_hap_data, dict) else {}
        futures_hap_data = ad_data.get("futures_hap")
        futures_hap_data = futures_hap_data if isinstance(futures_hap_data, dict) else {}

        rf_data = ad_data.get("ranging_filter")
        rf_data = rf_data if isinstance(rf_data, dict) else {}

        adaptive_indicator = AdaptiveIndicatorSettings(
            enabled=bool(ad_data.get("enabled", True)),
            dual_mode=bool(ad_data.get("dual_mode", True)),
            kospi_symbol=str(ad_data.get("kospi_symbol") or "KOSPI 지수"),
            futures_symbol=str(ad_data.get("futures_symbol") or "KP200 선물"),
            warmup_bars=cls._safe_int(ad_data.get("warmup_bars"), 15),
            min_swings_for_ready=cls._safe_int(ad_data.get("min_swings_for_ready"), 4),
            min_pivot_interval_bars=cls._safe_int(ad_data.get("min_pivot_interval_bars") or 10, 10),
            session_min_pivot_interval_table=list(ad_data.get("session_min_pivot_interval_table") or []),
            # ── [SSOT] AdaptiveZigZagSettings 는 zigzag_settings_from_dict() 경유로 생성 ──
            # base(zigzag) 를 먼저 파싱한 뒤, kospi/futures 는 base 를 상속하여 덮어씀.
            # 이렇게 하면 from_file / zigzag_settings_from_dict / to_zigzag_config 가
            # 동일한 파싱 로직을 공유하며, 새 필드 추가 시 한 곳만 수정하면 된다.
            zigzag=_zz_from_dict(zz_data),
            kospi_zigzag=_zz_from_dict(kospi_zz_data, base=_zz_from_dict(zz_data)),
            futures_zigzag=_zz_from_dict(futures_zz_data, base=_zz_from_dict(zz_data)),
            # ── [HAP] HybridAdaptivePivotSettings 파싱 ─────────────────────
            hap=HybridAdaptivePivotSettings(
                base_pct=float(hap_data.get("base_pct", 0.3) or 0.3),
                base_multiplier=float(hap_data.get("base_multiplier", 2.0) or 2.0),
                atr_weight=float(hap_data.get("atr_weight", 0.5) or 0.5),
                atr_period=cls._safe_int(hap_data.get("atr_period"), 14),
                multiplier_min=float(hap_data.get("multiplier_min", 0.8) or 0.8),
                multiplier_max=float(hap_data.get("multiplier_max", 2.0) or 2.0),
                er_period=cls._safe_int(hap_data.get("er_period"), 10),
                confirmation_bars=cls._safe_int(hap_data.get("confirmation_bars"), 1),
                cancel_ratio=float(hap_data.get("cancel_ratio", 0.3) or 0.3),
                min_wave_pct=float(hap_data.get("min_wave_pct", 0.15) or 0.15),
                min_wave_atr_ratio=float(hap_data.get("min_wave_atr_ratio", 0.5) or 0.5),
                warmup_bars=cls._safe_int(hap_data.get("warmup_bars"), 20),
                max_pivots=cls._safe_int(hap_data.get("max_pivots"), 30),
                use_adaptive_engine=bool(hap_data.get("use_adaptive_engine", False)),
                use_fractal_confirmation=bool(hap_data.get("use_fractal_confirmation", False)),
                fractal_lookback=cls._safe_int(hap_data.get("fractal_lookback"), 2),
                fractal_volume_spike=float(hap_data.get("fractal_volume_spike", 1.3) or 1.3),
                fractal_price_tolerance_pct=float(hap_data.get("fractal_price_tolerance_pct", 0.3) or 0.3),
                fractal_bonus=float(hap_data.get("fractal_bonus", 0.15) or 0.15),
            ),
            kospi_hap=HybridAdaptivePivotSettings(
                base_pct=float(kospi_hap_data.get("base_pct", 0.3) or 0.3),
                base_multiplier=float(kospi_hap_data.get("base_multiplier", 2.0) or 2.0),
                atr_weight=float(kospi_hap_data.get("atr_weight", 0.5) or 0.5),
                atr_period=cls._safe_int(kospi_hap_data.get("atr_period"), 14),
                multiplier_min=float(kospi_hap_data.get("multiplier_min", 0.8) or 0.8),
                multiplier_max=float(kospi_hap_data.get("multiplier_max", 2.0) or 2.0),
                er_period=cls._safe_int(kospi_hap_data.get("er_period"), 10),
                confirmation_bars=cls._safe_int(kospi_hap_data.get("confirmation_bars"), 1),
                cancel_ratio=float(kospi_hap_data.get("cancel_ratio", 0.3) or 0.3),
                min_wave_pct=float(kospi_hap_data.get("min_wave_pct", 0.15) or 0.15),
                min_wave_atr_ratio=float(kospi_hap_data.get("min_wave_atr_ratio", 0.5) or 0.5),
                warmup_bars=cls._safe_int(kospi_hap_data.get("warmup_bars"), 20),
                max_pivots=cls._safe_int(kospi_hap_data.get("max_pivots"), 30),
                use_adaptive_engine=bool(kospi_hap_data.get("use_adaptive_engine", False)),
                use_fractal_confirmation=bool(kospi_hap_data.get("use_fractal_confirmation", False)),
                fractal_lookback=cls._safe_int(kospi_hap_data.get("fractal_lookback"), 2),
                fractal_volume_spike=float(kospi_hap_data.get("fractal_volume_spike", 1.3) or 1.3),
                fractal_price_tolerance_pct=float(kospi_hap_data.get("fractal_price_tolerance_pct", 0.3) or 0.3),
                fractal_bonus=float(kospi_hap_data.get("fractal_bonus", 0.15) or 0.15),
            ) if kospi_hap_data else None,
            futures_hap=HybridAdaptivePivotSettings(
                base_pct=float(futures_hap_data.get("base_pct", 0.3) or 0.3),
                base_multiplier=float(futures_hap_data.get("base_multiplier", 2.0) or 2.0),
                atr_weight=float(futures_hap_data.get("atr_weight", 0.5) or 0.5),
                atr_period=cls._safe_int(futures_hap_data.get("atr_period"), 14),
                multiplier_min=float(futures_hap_data.get("multiplier_min", 0.8) or 0.8),
                multiplier_max=float(futures_hap_data.get("multiplier_max", 2.0) or 2.0),
                er_period=cls._safe_int(futures_hap_data.get("er_period"), 10),
                confirmation_bars=cls._safe_int(futures_hap_data.get("confirmation_bars"), 1),
                cancel_ratio=float(futures_hap_data.get("cancel_ratio", 0.3) or 0.3),
                min_wave_pct=float(futures_hap_data.get("min_wave_pct", 0.15) or 0.15),
                min_wave_atr_ratio=float(futures_hap_data.get("min_wave_atr_ratio", 0.5) or 0.5),
                warmup_bars=cls._safe_int(futures_hap_data.get("warmup_bars"), 20),
                max_pivots=cls._safe_int(futures_hap_data.get("max_pivots"), 30),
                use_adaptive_engine=bool(futures_hap_data.get("use_adaptive_engine", False)),
                use_fractal_confirmation=bool(futures_hap_data.get("use_fractal_confirmation", False)),
                fractal_lookback=cls._safe_int(futures_hap_data.get("fractal_lookback"), 2),
                fractal_volume_spike=float(futures_hap_data.get("fractal_volume_spike", 1.3) or 1.3),
                fractal_price_tolerance_pct=float(futures_hap_data.get("fractal_price_tolerance_pct", 0.3) or 0.3),
                fractal_bonus=float(futures_hap_data.get("fractal_bonus", 0.15) or 0.15),
            ) if futures_hap_data else None,
            supertrend=AdaptiveSuperTrendSettings(
                atr_min_period=cls._safe_int(st_data.get("atr_min_period"), 7),
                atr_max_period=cls._safe_int(st_data.get("atr_max_period"), 21),
                multiplier_min=float(st_data.get("multiplier_min", 1.5) or 1.5),
                multiplier_max=float(st_data.get("multiplier_max", 4.0) or 4.0),
                er_period=cls._safe_int(st_data.get("er_period"), 10),
                adx_period=cls._safe_int(st_data.get("adx_period"), 14),
                use_bb_correction=bool(st_data.get("use_bb_correction", True)),
                bb_period=cls._safe_int(st_data.get("bb_period"), 20),
                bb_std=float(st_data.get("bb_std", 2.0) or 2.0),
                smooth_period=cls._safe_int(st_data.get("smooth_period"), 3),
            ),
            ranging_filter=RangingFilterSettings(
                enabled=bool(rf_data.get("enabled", True)),
                adx_min=float(rf_data.get("adx_min", 15.0) or 15.0),
                er_min=float(rf_data.get("er_min", 0.15) or 0.15),
                use_zigzag_structure=bool(rf_data.get("use_zigzag_structure", True)),
                whipsaw_min_bars=cls._safe_int(rf_data.get("whipsaw_min_bars"), 2),
                use_adx_filter=bool(rf_data.get("use_adx_filter", True)),
                use_er_filter=bool(rf_data.get("use_er_filter", True)),
                use_whipsaw_filter=bool(rf_data.get("use_whipsaw_filter", True)),
            ),
            pivot_proximity_alert=PivotProximityAlertSettings(
                enabled=bool(ad_data.get("pivot_proximity_alert", {}).get("enabled", True)),
                max_bars_diff=cls._safe_int(ad_data.get("pivot_proximity_alert", {}).get("max_bars_diff"), 1),
                telegram_enabled=bool(ad_data.get("pivot_proximity_alert", {}).get("telegram_enabled", True)),
            ),
            pivot_candidate_alert=PivotCandidateAlertSettings(
                enabled=bool(ad_data.get("pivot_candidate_alert", {}).get("enabled", True)),
                telegram_enabled=bool(ad_data.get("pivot_candidate_alert", {}).get("telegram_enabled", True)),
                events=list(ad_data.get("pivot_candidate_alert", {}).get("events") or ["registered", "changed", "cancelled"]),
                change_cooldown_sec=float(ad_data.get("pivot_candidate_alert", {}).get("change_cooldown_sec") or 60.0),
            ),
        )
        
        return cls(
            ai_providers=ai_providers,
            ebest=ebest,
            options_subscription=options_subscription,
            option_minute_ohlcv=option_minute_ohlcv,
            minute_lookback=minute_lookback,
            prediction=prediction,
            telegram=telegram,
            adaptive_indicator=adaptive_indicator,
            trade_gate=trade_gate_cfg,
            log_file=str(data.get("log_file", "prediction.log")),
            log_level=str(data.get("log_level", "INFO")),
            adaptive_mode=bool(data.get("adaptive_mode", False)),
        )

    @classmethod
    def _default_config(cls) -> "AppConfig":
        """기본 설정 반환"""
        return cls(
            ai_providers=AIProviderConfig(),
            ebest=EBestConfig(),
            options_subscription=OptionSubscriptionConfig(),
            option_minute_ohlcv=OptionMinuteOhlcvConfig(),
            minute_lookback=MinuteLookbackConfig(futures=60, options=60),
            prediction=PredictionConfig(),
            telegram=TelegramConfig(),
            adaptive_indicator=AdaptiveIndicatorSettings(),
        )

    @classmethod
    def _backfill_missing_ai_keys(cls, cfg: "AppConfig", config_path: str) -> None:
        """병합 후에도 비어 있는 AI 키를 시크릿 파일 후보에서 직접 채운다."""
        ap = cfg.ai_providers
        need_a = not bool(ap.anthropic_key)
        need_o = not bool(ap.openai_key)
        need_g = not bool(ap.gemini_key)
        if not (need_a or need_o or need_g):
            return
        for sp in _secrets_paths_to_merge(config_path):
            sd = _load_json_file(sp)
            ai = sd.get("ai_providers") if isinstance(sd, dict) else None
            if not isinstance(ai, dict):
                continue
            ad = ai.get("anthropic", {}) if isinstance(ai.get("anthropic"), dict) else {}
            od = ai.get("openai", {}) if isinstance(ai.get("openai"), dict) else {}
            gd = ai.get("gemini", {}) if isinstance(ai.get("gemini"), dict) else {}
            if need_a:
                v = _strip_api_key(ad.get("api_key"))
                if v:
                    ap.anthropic_key = v
                    need_a = False
            if need_o:
                v = _strip_api_key(od.get("api_key"))
                if v:
                    ap.openai_key = v
                    need_o = False
            if need_g:
                v = _strip_api_key(gd.get("api_key"))
                if v:
                    ap.gemini_key = v
                    need_g = False
            if not (need_a or need_o or need_g):
                break
        # 환경 변수는 _from_dict에서 이미 반영됨; 백필 이후에도 비어 있으면 env만 한 번 더
        if not ap.anthropic_key:
            ap.anthropic_key = _strip_api_key(os.environ.get("ANTHROPIC_API_KEY"))
        if not ap.openai_key:
            ap.openai_key = _strip_api_key(os.environ.get("OPENAI_API_KEY"))
        if not ap.gemini_key:
            ap.gemini_key = _strip_api_key(os.environ.get("GEMINI_API_KEY"))

    @staticmethod
    def _parse_regime_float_map(raw: Any) -> Optional[Dict[str, float]]:
        """JSON ``disagreement_hold_prob_diff_max_by_regime`` 등 레짐→float 맵."""
        if not isinstance(raw, dict):
            return None
        out: Dict[str, float] = {}
        for k, v in raw.items():
            try:
                out[str(k).strip().upper()] = float(v)
            except Exception:
                continue
        return out if out else None

    @staticmethod
    def _safe_int(value, default: int) -> int:
        """안전한 정수 변환"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_session_min_wave_bars_table(raw: Any) -> list:
        """[SESSION-MW] JSON → List[Tuple[str, str, int]] 변환.

        JSON 형식 (배열의 배열):
            [["09:00", "09:30", 12], ["09:30", "10:30", 7], ...]

        - 항목이 리스트/튜플이고 길이 3 이상이어야 유효
        - start/end은 "HH:MM" 문자열, bars는 정수로 강제 변환
        - 변환 실패 항목은 경고 후 건너뜀
        - raw가 None이거나 빈 리스트면 [] 반환 (폴백: min_wave_bars 단일값 사용)
        """
        if not raw:
            return []
        if not isinstance(raw, list):
            logger.warning(f"[SESSION-MW] session_min_wave_bars_table must be an array: {type(raw)}")
            return []
        result = []
        for i, item in enumerate(raw):
            try:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    raise ValueError(f"Item format error (length {len(item) if hasattr(item,'__len__') else '?'})")
                start = str(item[0]).strip()
                end   = str(item[1]).strip()
                bars  = int(item[2])
                if bars < 0:
                    raise ValueError(f"min_wave_bars must be >= 0: {bars}")
                result.append((start, end, bars))
            except Exception as e:
                logger.warning(f"[SESSION-MW] session_min_wave_bars_table[{i}] parse failed: {e}")
        return result

    @staticmethod
    def _parse_session_min_wave_atr_ratio_table(raw: Any) -> list:
        """[ATR-FILTER] JSON → List[Tuple[str, str, float]] 변환.

        JSON 형식 (배열의 배열):
            [["09:00", "09:30", 0.8], ["09:30", "10:30", 1.2], ...]

        - 항목이 리스트/튜플이고 길이 3 이상이어야 유효
        - start/end은 "HH:MM" 문자열, ratio는 float로 강제 변환
        - 변환 실패 항목은 경고 후 건너뜀
        - raw가 None이거나 빈 리스트면 [] 반환 (폴백: min_wave_atr_ratio 단일값 사용)
        """
        if not raw:
            return []
        if not isinstance(raw, list):
            logger.warning(f"[ATR-FILTER] session_min_wave_atr_ratio_table must be an array: {type(raw)}")
            return []
        result = []
        for i, item in enumerate(raw):
            try:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    raise ValueError(f"Item format error (length {len(item) if hasattr(item,'__len__') else '?'})")
                start = str(item[0]).strip()
                end   = str(item[1]).strip()
                ratio = float(item[2])
                if ratio < 0:
                    raise ValueError(f"min_wave_atr_ratio must be >= 0: {ratio}")
                result.append((start, end, ratio))
            except Exception as e:
                logger.warning(f"[ATR-FILTER] session_min_wave_atr_ratio_table[{i}] parse failed: {e}")
        return result

    def _validate_ai_keys(self) -> None:
        """AI API 키 검증"""
        if not any([
            self.ai_providers.anthropic_key,
            self.ai_providers.openai_key,
            self.ai_providers.gemini_key,
        ]):
            if self.prediction.use_llm:
                logger.warning("No AI API keys configured, LLM features will be disabled")

    def _validate_prediction_settings(self) -> None:
        """예측 기본 설정 검증"""
        if self.prediction.minutes not in (5, 10, 30):
            raise ValueError(f"Invalid prediction_minutes: {self.prediction.minutes}")

        allowed_providers = {"", "claude", "gpt", "gemini", "openai", "chatgpt"}
        if str(self.prediction.preferred_provider or "") not in allowed_providers:
            raise ValueError(f"Invalid preferred_provider: {self.prediction.preferred_provider!r}")

        allowed_dual_primary = {"gpt", "gemini", "openai", "chatgpt"}
        if str(self.prediction.dual_llm_primary_provider or "") not in allowed_dual_primary:
            raise ValueError(
                f"Invalid dual_llm_primary_provider: {self.prediction.dual_llm_primary_provider!r}"
            )

    def _validate_numeric_predictor(self) -> None:
        """숫자형 예측기 설정 검증"""
        allowed_numeric_predictors = {"transformer", "tft", "combined", "ensemble", "rule_based"}
        n = str(getattr(self.prediction, "numeric_predictor", "transformer") or "transformer").strip().lower()
        if n == "combined":
            n = "ensemble"
        if n not in allowed_numeric_predictors:
            raise ValueError(
                f"Invalid numeric_predictor: {getattr(self.prediction, 'numeric_predictor', None)!r} "
                f"(expected one of {sorted(allowed_numeric_predictors)})"
            )

    def _validate_model_class(self) -> None:
        """모델 클래스 검증"""
        allowed_model_classes = {"transformer", "patch_tst", "mamba"}
        mc_raw = getattr(self.prediction, "model_class", None)
        mc = str(mc_raw).strip().lower() if mc_raw is not None else ""
        if not mc or mc not in allowed_model_classes:
            raise ValueError(
                f"Invalid prediction.model_class: {mc_raw!r} "
                f"(expected one of {sorted(allowed_model_classes)})"
            )

    def _validate_patch_parameters(self) -> None:
        """Patch 파라미터 검증"""
        try:
            pl = int(getattr(self.prediction, "patch_len", 8))
            st_p = int(getattr(self.prediction, "stride", 4))
        except (TypeError, ValueError):
            raise ValueError("prediction.patch_len / stride must be integers")
        if pl < 2:
            raise ValueError(f"prediction.patch_len must be >= 2, got {pl}")
        if st_p < 1:
            raise ValueError(f"prediction.stride must be >= 1, got {st_p}")
        if pl > int(getattr(self.prediction, "seq_len", 60) or 60):
            raise ValueError(
                f"prediction.patch_len ({pl}) must be <= seq_len "
                f"({getattr(self.prediction, 'seq_len', 60)})"
            )

    def _validate_mamba_parameters(self) -> None:
        """Mamba 파라미터 검증"""
        try:
            mds = int(getattr(self.prediction, "mamba_d_state", 16))
            msl = int(getattr(self.prediction, "mamba_seq_len", 60))
        except (TypeError, ValueError):
            raise ValueError("prediction.mamba_d_state / mamba_seq_len must be integers")
        if mds < 1:
            raise ValueError(f"prediction.mamba_d_state must be >= 1, got {mds}")
        if msl < 10:
            raise ValueError(f"prediction.mamba_seq_len must be >= 10, got {msl}")

        try:
            mw = float(getattr(self.prediction, "mamba_weight", 0.33))
        except (TypeError, ValueError):
            raise ValueError("prediction.mamba_weight must be numeric")
        if not (0.0 < mw < 1.0):
            raise ValueError(f"prediction.mamba_weight must be in (0, 1), got {mw}")

    def _validate_conformal_alpha(self) -> None:
        """Conformal alpha 검증"""
        try:
            ca = float(getattr(self.prediction, "conformal_alpha", 0.1))
        except Exception:
            raise ValueError("prediction.conformal_alpha must be numeric")
        if not (0.01 <= ca <= 0.5):
            raise ValueError(f"prediction.conformal_alpha must be in [0.01, 0.5], got {ca}")

    def _validate_option_features(self) -> None:
        """옵션 피처 설정 검증"""
        opt_set = str(getattr(self.prediction, "option_feature_set", "v1") or "v1").strip().lower()
        if opt_set not in ("v1", "v2", "v3", "v4", "v5"):
            raise ValueError("prediction.option_feature_set must be 'v1', 'v2', 'v3', 'v4' or 'v5'")

        try:
            _pcr_n = int(getattr(self.prediction, "pcr_atm_strikes_each_side", 5))
        except Exception:
            raise ValueError("prediction.pcr_atm_strikes_each_side must be an integer")
        if not (0 <= _pcr_n <= 50):
            raise ValueError(f"prediction.pcr_atm_strikes_each_side must be in [0, 50], got {_pcr_n}")

    def _validate_thresholds(self) -> None:
        """매수/매도 임계값 검증"""
        try:
            bt = float(getattr(self.prediction, "buy_threshold", 0.62))
            st = float(getattr(self.prediction, "sell_threshold", 0.38))
        except Exception:
            raise ValueError("Invalid buy/sell threshold (must be numeric)")

        if not (0.0 <= bt <= 1.0):
            raise ValueError(f"Invalid buy_threshold: {bt} (expected 0~1)")
        if not (0.0 <= st <= 1.0):
            raise ValueError(f"Invalid sell_threshold: {st} (expected 0~1)")
        if bt <= st:
            raise ValueError(f"Invalid thresholds: buy_threshold({bt}) must be > sell_threshold({st})")

    def _validate_tft_parameters(self) -> None:
        """TFT/Ensemble 파라미터 검증"""
        try:
            tw = float(getattr(self.prediction, "transformer_weight", 0.5))
        except Exception:
            raise ValueError("Invalid transformer_weight (must be numeric)")
        if not (0.0 < tw < 1.0):
            raise ValueError(f"Invalid transformer_weight: {tw} (expected 0<tw<1)")

        try:
            th = int(getattr(self.prediction, "tft_horizon", 300))
        except Exception:
            raise ValueError("Invalid tft_horizon (must be int)")
        if th <= 0:
            raise ValueError(f"Invalid tft_horizon: {th} (expected >0)")

    def _validate_llm_parameters(self) -> None:
        """LLM 파라미터 검증"""
        try:
            lmi = float(getattr(self.prediction, "llm_min_interval_sec", 30.0) or 0.0)
        except Exception:
            raise ValueError("Invalid llm_min_interval_sec (must be numeric)")
        if lmi < 0.0:
            raise ValueError(f"Invalid llm_min_interval_sec: {lmi} (expected >=0)")
        
        try:
            _gto_raw = getattr(self.prediction, "gemini_timeout_sec", None)
            if _gto_raw is not None:
                _gto = float(_gto_raw)
                if _gto <= 0.0:
                    raise ValueError(
                        f"Invalid gemini_timeout_sec: {_gto} (expected >0 or null)"
                    )
        except ValueError:
            raise
        except Exception:
            raise ValueError("Invalid gemini_timeout_sec (must be numeric or null)")
        
        try:
            _pcd = float(
                getattr(self.prediction, "llm_provider_cooldown_on_timeout_sec", 60.0) or 0.0
            )
        except Exception:
            raise ValueError("Invalid llm_provider_cooldown_on_timeout_sec (must be numeric)")
        if _pcd < 0.0:
            raise ValueError(
                f"Invalid llm_provider_cooldown_on_timeout_sec: {_pcd} (expected >=0)"
            )

    def _validate_tick_parameters(self) -> None:
        """틱 관련 파라미터 검증"""
        try:
            tsz = float(getattr(self.prediction, "tick_size", 0.05) or 0.0)
        except Exception:
            raise ValueError("Invalid tick_size (must be numeric)")
        if tsz <= 0.0:
            raise ValueError(f"Invalid tick_size: {tsz} (expected >0)")

        try:
            ftt = int(getattr(self.prediction, "feedback_threshold_ticks", 10) or 10)
        except Exception:
            raise ValueError("Invalid feedback_threshold_ticks (must be int)")
        if ftt < 1:
            raise ValueError(f"Invalid feedback_threshold_ticks: {ftt} (expected >=1)")

        try:
            fsh = int(getattr(self.prediction, "feedback_skip_hold_ticks", 2) or 0)
        except Exception:
            raise ValueError("Invalid feedback_skip_hold_ticks (must be int)")
        if fsh < 0:
            raise ValueError(f"Invalid feedback_skip_hold_ticks: {fsh} (expected >=0)")
        if fsh >= int(ftt):
            raise ValueError(
                f"Invalid feedback_skip_hold_ticks: {fsh} (expected < feedback_threshold_ticks={ftt})"
            )

    def _validate_feedback_weights(self) -> None:
        """피드백 가중치 검증"""
        for k in ("feedback_weight_high", "feedback_weight_mid", "feedback_weight_low"):
            try:
                wv = float(getattr(self.prediction, k, 0.0) or 0.0)
            except Exception:
                raise ValueError(f"Invalid {k} (must be numeric)")
            if wv < 0.0:
                raise ValueError(f"Invalid {k}: {wv} (expected >=0)")

        try:
            _ = bool(getattr(self.prediction, "feedback_use_price_snapshot", True))
        except Exception:
            raise ValueError("Invalid feedback_use_price_snapshot (must be bool)")

        try:
            _ = bool(getattr(self.prediction, "feedback_snapshot_required", False))
        except Exception:
            raise ValueError("Invalid feedback_snapshot_required (must be bool)")

        try:
            tol = float(getattr(self.prediction, "feedback_snapshot_tolerance_sec", 30.0) or 0.0)
        except Exception:
            raise ValueError("Invalid feedback_snapshot_tolerance_sec (must be numeric)")
        if tol < 0.0:
            raise ValueError(
                f"Invalid feedback_snapshot_tolerance_sec: {tol} (expected >=0)"
            )

    def _validate_cooldown_parameters(self) -> None:
        """쿨다운 파라미터 검증"""
        for k, default_v in (
            ("fc0_stale_threshold_sec", 10.0),
            ("fc0_stale_cooldown_sec", 60.0),
            ("oi_alert_cooldown_sec", 300.0),
        ):
            try:
                v = float(getattr(self.prediction, k, default_v) or 0.0)
            except Exception:
                raise ValueError(f"Invalid {k} (must be numeric)")
            if v < 0.0:
                raise ValueError(f"Invalid {k}: {v} (expected >=0)")

    def validate(self) -> bool:
        """
        설정 검증
        
        Returns:
            True if valid
            
        Raises:
            ValueError: 검증 실패시
        """
        self._validate_ai_keys()
        self._validate_prediction_settings()
        self._validate_numeric_predictor()
        self._validate_model_class()
        self._validate_patch_parameters()
        self._validate_mamba_parameters()
        self._validate_conformal_alpha()
        self._validate_option_features()
        self._validate_thresholds()
        self._validate_tft_parameters()
        self._validate_llm_parameters()
        self._validate_tick_parameters()
        self._validate_feedback_weights()
        self._validate_cooldown_parameters()
        self._validate_adaptive_indicator()

    def _validate_adaptive_indicator(self) -> None:
        """적응형 지표 설정 검증"""
        try:
            ad = getattr(self, "adaptive_indicator", None)
        except Exception:
            ad = None
        if ad is not None:
            st = getattr(ad, "supertrend", None)
            zz = getattr(ad, "zigzag", None)

            if int(getattr(ad, "warmup_bars", 45) or 45) < 15:
                raise ValueError(
                    "adaptive_indicator.warmup_bars must be >= 15 "
                    "(최소 지표 윈도우 확보 필요)"
                )

            if st is not None:
                if int(getattr(st, "atr_min_period", 7) or 7) < 1:
                    raise ValueError("adaptive_indicator.supertrend.atr_min_period must be >= 1")
                if int(getattr(st, "atr_max_period", 21) or 21) < int(getattr(st, "atr_min_period", 7) or 7):
                    raise ValueError("adaptive_indicator.supertrend.atr_max_period must be >= atr_min_period")
                if float(getattr(st, "multiplier_min", 1.5) or 1.5) <= 0.0:
                    raise ValueError("adaptive_indicator.supertrend.multiplier_min must be > 0")
                if float(getattr(st, "multiplier_max", 4.0) or 4.0) < float(getattr(st, "multiplier_min", 1.5) or 1.5):
                    raise ValueError("adaptive_indicator.supertrend.multiplier_max must be >= multiplier_min")
                if int(getattr(st, "er_period", 10) or 10) < 1:
                    raise ValueError("adaptive_indicator.supertrend.er_period must be >= 1")
                if int(getattr(st, "adx_period", 14) or 14) < 1:
                    raise ValueError("adaptive_indicator.supertrend.adx_period must be >= 1")
                if int(getattr(st, "bb_period", 20) or 20) < 1:
                    raise ValueError("adaptive_indicator.supertrend.bb_period must be >= 1")
                if float(getattr(st, "bb_std", 2.0) or 2.0) <= 0.0:
                    raise ValueError("adaptive_indicator.supertrend.bb_std must be > 0")
                if int(getattr(st, "smooth_period", 3) or 3) < 1:
                    raise ValueError("adaptive_indicator.supertrend.smooth_period must be >= 1")

            if zz is not None:
                if float(getattr(zz, "atr_multiplier", 1.5) or 1.5) <= 0.0:
                    raise ValueError("adaptive_indicator.zigzag.atr_multiplier must be > 0")
                if int(getattr(zz, "atr_period", 14) or 14) < 1:
                    raise ValueError("adaptive_indicator.zigzag.atr_period must be >= 1")
                if float(getattr(zz, "pivot_threshold_min_pct", 0.3) or 0.3) < 0.0:
                    raise ValueError("adaptive_indicator.zigzag.pivot_threshold_min_pct must be >= 0")
                if float(getattr(zz, "pivot_threshold_max_pct", 3.0) or 3.0) < float(getattr(zz, "pivot_threshold_min_pct", 0.3) or 0.3):
                    raise ValueError("adaptive_indicator.zigzag.pivot_threshold_max_pct must be >= pivot_threshold_min_pct")
                if float(getattr(zz, "major_swing_ratio", 2.0) or 2.0) <= 0.0:
                    raise ValueError("adaptive_indicator.zigzag.major_swing_ratio must be > 0")
                if int(getattr(zz, "max_swings", 20) or 20) < 1:
                    raise ValueError("adaptive_indicator.zigzag.max_swings must be >= 1")
                if int(getattr(zz, "confirmation_bars", 1) or 1) < 0:
                    raise ValueError("adaptive_indicator.zigzag.confirmation_bars must be >= 0")
                if float(getattr(zz, "cluster_tolerance_pct", 0.3) or 0.3) < 0.0:
                    raise ValueError("adaptive_indicator.zigzag.cluster_tolerance_pct must be >= 0")

                if int(getattr(zz, "er_period", 10) or 10) < 2:
                    raise ValueError("adaptive_indicator.zigzag.er_period must be >= 2")
                try:
                    mmin = float(getattr(zz, "atr_multiplier_min", 1.0) or 1.0)
                    mmax = float(getattr(zz, "atr_multiplier_max", 4.0) or 4.0)
                except Exception:
                    raise ValueError("adaptive_indicator.zigzag.atr_multiplier_min/max must be numeric")
                if mmin <= 0.0 or mmax <= 0.0:
                    raise ValueError("adaptive_indicator.zigzag.atr_multiplier_min/max must be > 0")
                if int(getattr(zz, "min_wave_bars", 5) or 5) < 0:
                    raise ValueError("adaptive_indicator.zigzag.min_wave_bars must be >= 0")
                if float(getattr(zz, "min_wave_pct", 0.0) or 0.0) < 0.0:
                    raise ValueError("adaptive_indicator.zigzag.min_wave_pct must be >= 0")
        
        # 옵션 구독 설정 검증
        if self.options_subscription.itm < 0:
            raise ValueError("options_subscription.itm must be >= 0")
        try:
            oom = float(getattr(self.options_subscription, "otm_open_min", 0.30) or 0.30)
        except Exception:
            raise ValueError("options_subscription.otm_open_min must be numeric")
        if oom < 0.0:
            raise ValueError("options_subscription.otm_open_min must be >= 0")
        if int(getattr(self.options_subscription, "max_otm_calls", 0) or 0) < 0:
            raise ValueError("options_subscription.max_otm_calls must be >= 0")
        if int(getattr(self.options_subscription, "max_otm_puts", 0) or 0) < 0:
            raise ValueError("options_subscription.max_otm_puts must be >= 0")
        # OI 지지저항 분석용 구독 창 검증
        try:
            _oh0 = int(getattr(self.options_subscription, "preopen_oh0_window", 10) or 0)
        except Exception:
            raise ValueError("options_subscription.preopen_oh0_window must be int")
        if not (0 <= _oh0 <= 20):
            raise ValueError(f"options_subscription.preopen_oh0_window must be 0~20, got {_oh0}")
        try:
            _oi_itm = int(getattr(self.options_subscription, "oi_itm_count", 10))
            _oi_otm = int(getattr(self.options_subscription, "oi_otm_count", 10))
        except Exception:
            raise ValueError("options_subscription.oi_itm_count/oi_otm_count must be int")
        if not (1 <= _oi_itm <= 30):
            raise ValueError(f"options_subscription.oi_itm_count must be 1~30, got {_oi_itm}")
        if not (1 <= _oi_otm <= 30):
            raise ValueError(f"options_subscription.oi_otm_count must be 1~30, got {_oi_otm}")
        try:
            _oi_reb = float(getattr(self.options_subscription, "oi_rebalance_interval_sec", 60.0) or 60.0)
        except Exception:
            raise ValueError("options_subscription.oi_rebalance_interval_sec must be numeric")
        if not (10.0 <= _oi_reb <= 600.0):
            raise ValueError(f"options_subscription.oi_rebalance_interval_sec must be 10~600, got {_oi_reb}")

        # 옵션 분봉 OHLCV 설정 검증
        try:
            aw = int(getattr(self.option_minute_ohlcv, "atm_window", 2) or 2)
        except Exception:
            raise ValueError("option_minute_ohlcv.atm_window must be an int")
        if aw < 0:
            raise ValueError("option_minute_ohlcv.atm_window must be >= 0")

        # 피처 계산/LLM 컨텍스트용 분봉 lookback 검증
        try:
            f = int(getattr(self.minute_lookback, "futures", 60) or 60)
            o = int(getattr(self.minute_lookback, "options", 60) or 60)
        except Exception:
            raise ValueError("minute_lookback.futures/options must be int")
        if f < 1:
            raise ValueError("minute_lookback.futures must be >= 1")
        if o < 1:
            raise ValueError("minute_lookback.options must be >= 1")

        # 텔레그램 옵션 플로우 상태 메시지 설정 검증
        try:
            _ = bool(getattr(self.telegram, "option_flow_status_enabled", True))
            _ = bool(getattr(self.telegram, "option_flow_status_intraday_only", False))
            _ = bool(getattr(self.telegram, "option_flow_status_disable_after_close", True))
        except Exception:
            raise ValueError("telegram option_flow_status_* flags must be bool")
        try:
            _cd = float(getattr(self.telegram, "option_flow_status_cooldown_sec", 0.0) or 0.0)
        except Exception:
            raise ValueError("telegram.option_flow_status_cooldown_sec must be numeric")
        if _cd < 0.0:
            raise ValueError(
                f"telegram.option_flow_status_cooldown_sec must be >= 0, got {_cd}"
            )
        for _k, _d in (
            ("option_flow_interp_sr_warn", 1.5),
            ("option_flow_interp_sr_hot", 2.0),
            ("option_flow_interp_pt_low", 0.008),
            ("option_flow_interp_pt_high", 0.03),
            ("option_flow_interp_pcr_v_low", 0.90),
            ("option_flow_interp_pcr_v_high", 1.10),
            ("option_flow_interp_pcr_oi_low", 0.95),
            ("option_flow_interp_pcr_oi_high", 1.05),
        ):
            try:
                _ = float(getattr(self.telegram, _k, _d))
            except Exception:
                raise ValueError(f"telegram.{_k} must be numeric")
        if float(getattr(self.telegram, "option_flow_interp_sr_warn", 1.5)) > float(
            getattr(self.telegram, "option_flow_interp_sr_hot", 2.0)
        ):
            raise ValueError("telegram.option_flow_interp_sr_warn must be <= option_flow_interp_sr_hot")
        if float(getattr(self.telegram, "option_flow_interp_pt_low", 0.008)) > float(
            getattr(self.telegram, "option_flow_interp_pt_high", 0.03)
        ):
            raise ValueError("telegram.option_flow_interp_pt_low must be <= option_flow_interp_pt_high")
        if float(getattr(self.telegram, "option_flow_interp_pcr_v_low", 0.90)) > float(
            getattr(self.telegram, "option_flow_interp_pcr_v_high", 1.10)
        ):
            raise ValueError("telegram.option_flow_interp_pcr_v_low must be <= option_flow_interp_pcr_v_high")
        if float(getattr(self.telegram, "option_flow_interp_pcr_oi_low", 0.95)) > float(
            getattr(self.telegram, "option_flow_interp_pcr_oi_high", 1.05)
        ):
            raise ValueError("telegram.option_flow_interp_pcr_oi_low must be <= option_flow_interp_pcr_oi_high")

        return True


def load_config(config_path: str = "config.json") -> AppConfig:
    """
    설정 로드 헬퍼 함수
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        AppConfig 인스턴스
    """
    try:
        config = AppConfig.from_file(config_path)
        config.validate()
        return config
    except Exception as e:
        logger.error(f"Failed to load config, using defaults: {e}")
        cfg = AppConfig._default_config()
        try:
            # config.json 로드 실패 시에도 프로젝트 루트 등에서 시크릿만이라도 반영
            p = str(Path(__file__).resolve().parent / "config.json")
            AppConfig._backfill_missing_ai_keys(cfg, p)
        except Exception:
            pass
        return cfg


# Config 파일 변경 감지를 위한 캐시
_config_cache = {
    "config": None,
    "config_path": None,
    "config_mtime": None
}

def get_config_with_reload(config_path: str = "config.json") -> AppConfig:
    """
    Config 파일 변경 감지 및 자동 재로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        AppConfig 인스턴스
    """
    global _config_cache
    
    try:
        from pathlib import Path
        import os
        
        config_file = Path(config_path)
        if not config_file.exists():
            return load_config(config_path)
        
        # 파일 수정 시간 확인
        current_mtime = os.path.getmtime(config_file)
        
        # 캐시된 설정이 없거나, 파일이 변경되었으면 재로드
        if (_config_cache["config"] is None or 
            _config_cache["config_path"] != config_path or
            _config_cache["config_mtime"] is None or
            current_mtime > _config_cache["config_mtime"]):
            
            logger.info(f"Config file changed: {config_path} (mtime: {current_mtime}) - reloading")
            logger.info("Restart predictor to apply changes.")
            config = load_config(config_path)
            _config_cache["config"] = config
            _config_cache["config_path"] = config_path
            _config_cache["config_mtime"] = current_mtime
            return config
        else:
            return _config_cache["config"]
            
    except Exception as e:
        logger.warning(f"Config auto-reload failed, using default load: {e}")
        return load_config(config_path)


def parse_session_min_wave_bars_table(raw: Any) -> list:
    """[SESSION-MW] 모듈 레벨 공개 함수 — pipeline.py / data_builder.py 에서 직접 import해 사용.

    AppConfig._parse_session_min_wave_bars_table()와 동일한 로직.
    JSON 배열 [[시작HH:MM, 종료HH:MM, min_wave_bars], ...] → List[Tuple[str, str, int]] 변환.
    """
    return AppConfig._parse_session_min_wave_bars_table(raw)


def zigzag_settings_from_dict(
    zz,
    base=None,
):
    """dict(config.json 섹션) → AdaptiveZigZagSettings 변환 헬퍼.

    단일 소스 원칙(SSOT):
    pipeline.py / data_builder.py 처럼 AppConfig 객체 대신 raw dict 만 갖는
    호출자가 AdaptiveZigZagSettings 를 얻는 공식 경로.
    얻은 Settings 에서 .to_zigzag_config() 를 호출하면 AdaptiveZigZagConfig 가 생성된다.

    Args:
        zz:   config.json 의 adaptive_indicator.zigzag 등 해당 섹션 dict.
        base: 덮어쓸 기준 Settings 인스턴스. None 이면 기본값 인스턴스를 사용.

    Returns:
        AdaptiveZigZagSettings 인스턴스.
    """
    if not isinstance(zz, dict):
        zz = {}
    d = base if base is not None else AdaptiveZigZagSettings()
    _si = AppConfig._safe_int

    return AdaptiveZigZagSettings(
        atr_multiplier           = float(zz.get("atr_multiplier",           d.atr_multiplier)           or d.atr_multiplier),
        atr_period               = _si(  zz.get("atr_period",               d.atr_period),               d.atr_period),
        er_period                = _si(  zz.get("er_period",                d.er_period),                d.er_period),
        atr_multiplier_min       = float(zz.get("atr_multiplier_min",       d.atr_multiplier_min)       or d.atr_multiplier_min),
        atr_multiplier_max       = float(zz.get("atr_multiplier_max",       d.atr_multiplier_max)       or d.atr_multiplier_max),
        pivot_threshold_min_pct  = float(zz.get("pivot_threshold_min_pct",  d.pivot_threshold_min_pct)  or d.pivot_threshold_min_pct),
        pivot_threshold_max_pct  = float(zz.get("pivot_threshold_max_pct",  d.pivot_threshold_max_pct)  or d.pivot_threshold_max_pct),
        major_swing_ratio        = float(zz.get("major_swing_ratio",        d.major_swing_ratio)        or d.major_swing_ratio),
        max_swings               = _si(  zz.get("max_swings",               d.max_swings),               d.max_swings),
        confirmation_bars        = _si(  zz.get("confirmation_bars",        d.confirmation_bars),        d.confirmation_bars),
        confirmation_bars_ranging= _si(  zz.get("confirmation_bars_ranging",d.confirmation_bars_ranging),d.confirmation_bars_ranging),
        confirmation_bars_unknown= _si(  zz.get("confirmation_bars_unknown",d.confirmation_bars_unknown),d.confirmation_bars_unknown),
        freeze_on_confirm        = bool( zz.get("freeze_on_confirm",        d.freeze_on_confirm)),
        min_wave_bars            = _si(  zz.get("min_wave_bars",            d.min_wave_bars),            d.min_wave_bars),
        min_wave_pct             = float(zz.get("min_wave_pct",             d.min_wave_pct)             or 0.0),
        max_wait_bars            = _si(  zz.get("max_wait_bars",            d.max_wait_bars),            d.max_wait_bars),
        cluster_tolerance_pct    = float(zz.get("cluster_tolerance_pct",    d.cluster_tolerance_pct)    or d.cluster_tolerance_pct),
        structure_lookback_swings= _si(  zz.get("structure_lookback_swings",d.structure_lookback_swings),d.structure_lookback_swings),
        structure_points         = _si(  zz.get("structure_points",         d.structure_points),         d.structure_points),
        session_min_wave_bars_table=AppConfig._parse_session_min_wave_bars_table(
            zz.get("session_min_wave_bars_table") if "session_min_wave_bars_table" in zz
            else d.session_min_wave_bars_table
        ),
        session_min_wave_atr_ratio_table=AppConfig._parse_session_min_wave_atr_ratio_table(
            zz.get("session_min_wave_atr_ratio_table") if "session_min_wave_atr_ratio_table" in zz
            else d.session_min_wave_atr_ratio_table
        ),
        use_atr_based_filtering  = bool( zz.get("use_atr_based_filtering",  d.use_atr_based_filtering)),
        min_wave_atr_ratio       = float(zz.get("min_wave_atr_ratio",       d.min_wave_atr_ratio)       or d.min_wave_atr_ratio),
        cluster_atr_ratio        = float(zz.get("cluster_atr_ratio",        d.cluster_atr_ratio)        or d.cluster_atr_ratio),
        # ── 장초반 ATR 조절 ─────────────────────────────────────────────────────
        # [MAINT-2 FIX] early_session_atr_multiplier_max 제거
        early_session_start_time          = str(  zz.get("early_session_start_time",          d.early_session_start_time)),
        early_session_end_time            = str(  zz.get("early_session_end_time",            d.early_session_end_time)),
        # ── 피보나치 레벨 ────────────────────────────────────────────────────────
        fib_ratios = list(zz.get("fib_ratios", d.fib_ratios) or d.fib_ratios),
        # ── 구조 다수결 임계값 ────────────────────────────────────────────────────
        structure_majority_threshold      = float(zz.get("structure_majority_threshold",      d.structure_majority_threshold) or d.structure_majority_threshold),
        # ── bars_since 기반 decay ────────────────────────────────────────────────
        decay_start_bars                  = _si(  zz.get("decay_start_bars",                  d.decay_start_bars),              d.decay_start_bars),
        decay_rate_per_bar                = float(zz.get("decay_rate_per_bar",                d.decay_rate_per_bar)             or d.decay_rate_per_bar),
        decay_max_pct                     = float(zz.get("decay_max_pct",                     d.decay_max_pct)                  or d.decay_max_pct),
        # ── 다중 시간프레임 ──────────────────────────────────────────────────────
        multi_timeframe_enabled                   = bool( zz.get("multi_timeframe_enabled",                   d.multi_timeframe_enabled)),
        multi_timeframe_scales                    = list( zz.get("multi_timeframe_scales",                    d.multi_timeframe_scales)  or d.multi_timeframe_scales),
        multi_timeframe_consensus_threshold       = _si(  zz.get("multi_timeframe_consensus_threshold",       d.multi_timeframe_consensus_threshold), d.multi_timeframe_consensus_threshold),
        multi_timeframe_price_tolerance_pct       = float(zz.get("multi_timeframe_price_tolerance_pct",       d.multi_timeframe_price_tolerance_pct) or d.multi_timeframe_price_tolerance_pct),
        multi_timeframe_index_tolerance_multiplier= float(zz.get("multi_timeframe_index_tolerance_multiplier",d.multi_timeframe_index_tolerance_multiplier) or d.multi_timeframe_index_tolerance_multiplier),
        # ── major 파동 기준 ──────────────────────────────────────────────────────
        major_wave_ratio                  = float(zz.get("major_wave_ratio",                  d.major_wave_ratio)   or d.major_wave_ratio),
        major_wave_lookback               = _si(  zz.get("major_wave_lookback",               d.major_wave_lookback),            d.major_wave_lookback),
        # ── DER 불일치 처리 ──────────────────────────────────────────────────────
        der_mismatch_threshold            = float(zz.get("der_mismatch_threshold",            d.der_mismatch_threshold) or d.der_mismatch_threshold),
        der_mismatch_mult_ratio           = float(zz.get("der_mismatch_mult_ratio",           d.der_mismatch_mult_ratio) or d.der_mismatch_mult_ratio),
        # ── 피봇 수집기 ─────────────────────────────────────────────────────────
        enable_pivot_collector            = bool( zz.get("enable_pivot_collector",            d.enable_pivot_collector)),
        pivot_collector_max_sequence      = _si(  zz.get("pivot_collector_max_sequence",      d.pivot_collector_max_sequence),   d.pivot_collector_max_sequence),
    )
