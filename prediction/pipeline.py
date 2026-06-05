"""Prediction orchestrator for the Transformer(+stub) + LLM pipeline.

This module wires together:
- `RealTimeTickProcessor` (minute bars + option snapshots)
- FO0 buffering/downsampling (1Hz)
- Feature building (`prediction.features`)
- Numeric predictor (`prediction.predictor`)
- LLM judgment (`prediction.llm_judge`)

The main entrypoints used by the runtime are:
- `PredictionPipeline.add_realtime_tick()`
- `PredictionPipeline.get_prediction()`

리팩터링 구조 (Mixin 상속):
    LLMMixin          → llm_mixin.py         (LLM 호출/판단/executor)
    AmplitudeMixin    → amplitude_mixin.py   (진폭 예측/EMA 보정)
    GuardrailMixin    → guardrail_mixin.py   (옵션/베이시스/OI 가드레일)
    AdaptiveMixin     → adaptive_mixin.py    (SuperTrend/ZigZag/레짐)
    FeedbackMixin     → feedback_mixin.py    (예측 피드백/가중치 갱신)
    OptionMixin       → option_mixin.py      (옵션 스냅샷/OI 알람)
    PredictionMixin   → prediction_mixin.py  (get_prediction 흐름)
    TickMixin         → tick_mixin.py        (실시간 틱 처리)
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
import logging
import threading
from datetime import datetime
from typing import Any, Dict, Optional


# [SSOT] config.py 헬퍼 — AdaptiveZigZagSettings 경유 AdaptiveZigZagConfig 생성
try:
    from config import zigzag_settings_from_dict as _zz_settings_from_dict
except ImportError:
    _zz_settings_from_dict = None  # type: ignore[assignment]

from config import (
    FUTURE_KNOWN_DIM,
    HORIZON_SEC,
    LLM_PROVIDER_COOLDOWN_ON_TIMEOUT,
    LLM_TIMEOUT_SEC,
)
from data.tick_processor import RealTimeTickProcessor
from core.interfaces import TickDataProvider
from core.utils import normalize_adaptive_indicator_symbol

from .features.features import (
    ADAPT_KEYS,
    CD_KEYS,
    MS5_KEYS,
    MS15_KEYS,
    OB_KEYS,
    get_opt_keys,
)
from .llm_judge import LLMJudge
from .predictor import NumericPredictor, create_numeric_predictor

from .mixins.llm_mixin import LLMMixin
from .mixins.amplitude_mixin import AmplitudeMixin
from .mixins.guardrail_mixin import GuardrailMixin
from .mixins.adaptive_mixin import AdaptiveMixin
from .mixins.feedback_mixin import FeedbackMixin
from .mixins.option_mixin import OptionMixin
from .mixins.prediction_mixin import PredictionMixin  # [FIX] _NumericResult 단일화: pipeline.py 중복 선언 제거
from .mixins.tick_mixin import TickMixin

logger = logging.getLogger(__name__)

# _NumericResult는 prediction_mixin.py 에서 단일 정의 후 import한다. (위 import 참조)
# [FIX] 이중 정의 제거: @dataclass(slots=True) class _NumericResult was here.
# slots=True 중복 정의는 __slots__ 충돌 위험 + MRO 모호성을 유발한다.

class PredictionPipeline(
    LLMMixin,
    AmplitudeMixin,
    GuardrailMixin,
    AdaptiveMixin,
    FeedbackMixin,
    OptionMixin,
    PredictionMixin,
    TickMixin,
):
    """Transformer + LLM pipeline.

    This implements the role split described in Transformer_LLM_Pipeline.md:
    - Transformer: numeric prediction (currently a stub if weights are unavailable)
    - LLM: judgment/explanation based on model output + context

    It also preserves a predictor-like interface used by ebest_live.py.

    Public API (시그니처 불변):
        add_realtime_tick(tick_data)  → TickMixin
        get_prediction(...)           → PredictionMixin
    """

    def _metrics_inc(self, key: str, delta: int = 1) -> None:
        """_metrics 카운터를 Lock 하에 안전하게 증가시킨다."""
        with self._metrics_lock:
            self._metrics[key] = int(self._metrics.get(key) or 0) + delta

    def _metrics_set(self, key: str, value: Any) -> None:
        """_metrics 값을 Lock 하에 안전하게 설정한다."""
        with self._metrics_lock:
            self._metrics[key] = value

    def _metrics_get(self, key: str, default: Any = None) -> Any:
        """_metrics 값을 Lock 하에 안전하게 읽는다."""
        with self._metrics_lock:
            return self._metrics.get(key, default)

    def get_last_result(self) -> Dict[str, Any]:
        """마지막 예측 결과를 GUI/브리지에서 조회할 때 사용."""
        try:
            v = getattr(self, "_last_result", None)
            return dict(v) if isinstance(v, dict) else {}
        except Exception:
            return {}

    def __init__(
        self,
        *,
        anthropic_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
        preferred_provider: Optional[str] = None,
        numeric_predictor: str = "transformer",
        model_class: str = "transformer",
        patch_len: int = 8,
        stride: int = 4,
        conformal_alpha: float = 0.1,
        conformal_path: Optional[str] = None,
        multiscale_5m: bool = False,
        multiscale_enabled: bool = False,
        multiscale_time_scales: "list[int] | None" = None,  # [FIX] 뮤터블 기본값 제거: [1,5,15] → None
        mamba_enabled: bool = False,
        mamba_weights_path: Optional[str] = None,
        mamba_weight: float = 0.33,
        transformer_weight: float = 0.5,
        transformer_weights_path: Optional[str] = None,
        tft_weights_path: Optional[str] = None,
        tft_horizon: int = HORIZON_SEC,
        disagreement_hold: bool = True,
        disagreement_hold_prob_diff_max: float = 0.3,
        disagreement_hold_prob_diff_max_by_regime: Optional[Dict[str, float]] = None,
        ensemble_agreement_confidence_boost: bool = True,
        ensemble_agreement_prob_diff_max: float = 0.06,
        prediction_minutes: int = 5,
        min_minute_bars_required: int = 20,
        seq_len: int = 60,
        fo0_stale_sec: int = 10,
        fo0_log_schema: bool = True,
        llm_timeout_sec: float = LLM_TIMEOUT_SEC,
        llm_min_interval_sec: float = 30.0,
        llm_provider_cooldown_on_timeout_sec: float = LLM_PROVIDER_COOLDOWN_ON_TIMEOUT,
        gemini_timeout_sec: Optional[float] = None,
        tick_size: float = 0.05,
        feedback_threshold_ticks: int = 10,
        feedback_skip_hold_ticks: int = 2,
        feedback_weight_high: float = 1.0,
        feedback_weight_mid: float = 0.5,
        feedback_weight_low: float = 0.25,
        feedback_use_price_snapshot: bool = True,
        feedback_snapshot_tolerance_sec: float = 30.0,
        feedback_snapshot_required: bool = False,
        feedback_max_pending: int = 200,
        buy_threshold: float = 0.62,
        sell_threshold: float = 0.38,
        confidence_high_margin: float = 0.15,
        confidence_mid_margin: float = 0.08,
        confidence_spread_max_for_high: float = 1.0,
        confidence_conformal_width_max_for_high: float = 0.35,
        confidence_conformal_width_max_for_medium: float = 0.55,
        guard_basis_hold_thr: float = 2.5,
        guard_basis_downgrade_thr: float = 1.5,
        guard_atm_spread_pct_thr: float = 1.5,
        guard_atm_liq_log_thr: float = 2.0,
        fc0_stale_threshold_sec: float = 10.0,
        fc0_stale_cooldown_sec: float = 60.0,
        oi_alert_cooldown_sec: float = 300.0,
        config_path: Optional[str] = None,
        use_llm: bool = True,
        heuristic_fallback: bool = True,
        heuristic_flip_min_interval_sec: Optional[float] = None,
        heuristic_flip_include_hold_transition: bool = False,
        rule_based_weights: Optional[Dict[str, float]] = None,
        rule_based_mom_multiplier: float = 1.0,
        adaptive_indicator: Optional[dict] = None,
        option_minute_ohlcv: Optional[dict] = None,
        minute_lookback: Optional[dict] = None,  # 피처 계산/LLM 컨텍스트용
        option_feature_set: str = "v1",
        otm_open_min: float = 0.30,
        pcr_atm_strikes_each_side: int = 5,
        dump_llm_prompt: bool = False,
        dual_llm: bool = False,
        dual_llm_primary_provider: str = "gpt",
        notifier: Any = None,
        tick_provider: "Optional[TickDataProvider]" = None,
    ):
        """Create a new pipeline instance.

        Args:
            anthropic_key/openai_key/gemini_key: Optional LLM API keys.
            preferred_provider: Optional provider preference (`claude|gpt|gemini`).
            prediction_minutes: Prediction horizon (minutes).
            min_minute_bars_required: Minimum minute bars required before predicting.
            seq_len: FO0 buffer length (1Hz, e.g. 60 == last 60 seconds).
            fo0_stale_sec: FO0 stale warning threshold (seconds).
            fo0_log_schema: If True, log FO0 keys when parsed features look empty.
            config_path: Used for compatibility; currently stored for diagnostics.
            use_llm: If False, LLM is skipped and LLM fields are derived from transformer output.
        """
        # [REFACTOR] 분해: 파라미터 초기화 → 컴포넌트 빌드 → 상태 초기화
        self._init_parameters(
            prediction_minutes=prediction_minutes,
            config_path=config_path,
            use_llm=use_llm,
            heuristic_fallback=heuristic_fallback,
            heuristic_flip_min_interval_sec=heuristic_flip_min_interval_sec,
            heuristic_flip_include_hold_transition=heuristic_flip_include_hold_transition,
            rule_based_weights=rule_based_weights,
            rule_based_mom_multiplier=rule_based_mom_multiplier,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            confidence_high_margin=confidence_high_margin,
            confidence_mid_margin=confidence_mid_margin,
            confidence_spread_max_for_high=confidence_spread_max_for_high,
            confidence_conformal_width_max_for_high=confidence_conformal_width_max_for_high,
            confidence_conformal_width_max_for_medium=confidence_conformal_width_max_for_medium,
            guard_basis_hold_thr=guard_basis_hold_thr,
            guard_basis_downgrade_thr=guard_basis_downgrade_thr,
            guard_atm_spread_pct_thr=guard_atm_spread_pct_thr,
            guard_atm_liq_log_thr=guard_atm_liq_log_thr,
            fc0_stale_threshold_sec=fc0_stale_threshold_sec,
            fc0_stale_cooldown_sec=fc0_stale_cooldown_sec,
            oi_alert_cooldown_sec=oi_alert_cooldown_sec,
            seq_len=seq_len,
            multiscale_5m=multiscale_5m,
            multiscale_enabled=multiscale_enabled,
            multiscale_time_scales=multiscale_time_scales,
            tft_horizon=tft_horizon,
            fo0_stale_sec=fo0_stale_sec,
            fo0_log_schema=fo0_log_schema,
            llm_timeout_sec=llm_timeout_sec,
            gemini_timeout_sec=gemini_timeout_sec,
            llm_min_interval_sec=llm_min_interval_sec,
            llm_provider_cooldown_on_timeout_sec=llm_provider_cooldown_on_timeout_sec,
            dump_llm_prompt=dump_llm_prompt,
            tick_size=tick_size,
            feedback_threshold_ticks=feedback_threshold_ticks,
            feedback_skip_hold_ticks=feedback_skip_hold_ticks,
            feedback_weight_high=feedback_weight_high,
            feedback_weight_mid=feedback_weight_mid,
            feedback_weight_low=feedback_weight_low,
            feedback_use_price_snapshot=feedback_use_price_snapshot,
            feedback_snapshot_tolerance_sec=feedback_snapshot_tolerance_sec,
            feedback_snapshot_required=feedback_snapshot_required,
            feedback_max_pending=feedback_max_pending,
            disagreement_hold=disagreement_hold,
            disagreement_hold_prob_diff_max=disagreement_hold_prob_diff_max,
            disagreement_hold_prob_diff_max_by_regime=disagreement_hold_prob_diff_max_by_regime,
            ensemble_agreement_confidence_boost=ensemble_agreement_confidence_boost,
            ensemble_agreement_prob_diff_max=ensemble_agreement_prob_diff_max,
            dual_llm=dual_llm,
            dual_llm_primary_provider=dual_llm_primary_provider,
            adaptive_indicator=adaptive_indicator,
            minute_lookback=minute_lookback,
            option_feature_set=option_feature_set,
            otm_open_min=otm_open_min,
            pcr_atm_strikes_each_side=pcr_atm_strikes_each_side,
            option_minute_ohlcv=option_minute_ohlcv,
            notifier=notifier,
        )
        
        self._build_components(
            anthropic_key=anthropic_key,
            openai_key=openai_key,
            gemini_key=gemini_key,
            preferred_provider=preferred_provider,
            tick_provider=tick_provider,
            numeric_predictor=numeric_predictor,
            model_class=model_class,
            patch_len=patch_len,
            stride=stride,
            conformal_alpha=conformal_alpha,
            conformal_path=conformal_path,
            mamba_enabled=mamba_enabled,
            mamba_weights_path=mamba_weights_path,
            mamba_weight=mamba_weight,
            transformer_weight=transformer_weight,
            transformer_weights_path=transformer_weights_path,
            tft_weights_path=tft_weights_path,
            tft_horizon=tft_horizon,
            disagreement_hold=disagreement_hold,
            disagreement_hold_prob_diff_max=disagreement_hold_prob_diff_max,
            ensemble_agreement_confidence_boost=ensemble_agreement_confidence_boost,
            ensemble_agreement_prob_diff_max=ensemble_agreement_prob_diff_max,
            rule_based_weights=rule_based_weights,
            rule_based_mom_multiplier=rule_based_mom_multiplier,
            min_minute_bars_required=min_minute_bars_required,
            adaptive_indicator=adaptive_indicator,
            config_path=config_path,
        )
        
        self._init_state(config_path=config_path)

    def _init_parameters(
        self,
        *,
        prediction_minutes: int,
        config_path: Optional[str],
        use_llm: bool,
        heuristic_fallback: bool,
        heuristic_flip_min_interval_sec: Optional[float],
        heuristic_flip_include_hold_transition: bool,
        rule_based_weights: Optional[Dict[str, float]],
        rule_based_mom_multiplier: float,
        buy_threshold: float,
        sell_threshold: float,
        confidence_high_margin: float,
        confidence_mid_margin: float,
        confidence_spread_max_for_high: float,
        confidence_conformal_width_max_for_high: float,
        confidence_conformal_width_max_for_medium: float,
        guard_basis_hold_thr: float,
        guard_basis_downgrade_thr: float,
        guard_atm_spread_pct_thr: float,
        guard_atm_liq_log_thr: float,
        fc0_stale_threshold_sec: float,
        fc0_stale_cooldown_sec: float,
        oi_alert_cooldown_sec: float,
        seq_len: int,
        multiscale_5m: bool,
        multiscale_enabled: bool,
        multiscale_time_scales: Optional[list[int]],
        tft_horizon: int,
        fo0_stale_sec: int,
        fo0_log_schema: bool,
        llm_timeout_sec: float,
        gemini_timeout_sec: Optional[float],
        llm_min_interval_sec: float,
        llm_provider_cooldown_on_timeout_sec: float,
        dump_llm_prompt: bool,
        tick_size: float,
        feedback_threshold_ticks: int,
        feedback_skip_hold_ticks: int,
        feedback_weight_high: float,
        feedback_weight_mid: float,
        feedback_weight_low: float,
        feedback_use_price_snapshot: bool,
        feedback_snapshot_tolerance_sec: float,
        feedback_snapshot_required: bool,
        feedback_max_pending: int,
        disagreement_hold: bool,
        disagreement_hold_prob_diff_max: float,
        disagreement_hold_prob_diff_max_by_regime: Optional[Dict[str, float]],
        ensemble_agreement_confidence_boost: bool,
        ensemble_agreement_prob_diff_max: float,
        dual_llm: bool,
        dual_llm_primary_provider: str,
        adaptive_indicator: Optional[dict],
        minute_lookback: Optional[dict],
        option_feature_set: str,
        otm_open_min: float,
        pcr_atm_strikes_each_side: int,
        option_minute_ohlcv: Optional[dict],
        notifier: Any,
    ) -> None:
        """파라미터 초기화 및 검증."""
        self._prediction_minutes = int(prediction_minutes or 5)
        self._config_path = str(config_path or "config.json")
        self._use_llm = bool(use_llm)
        self._heuristic_fallback = bool(heuristic_fallback)
        try:
            self._heuristic_flip_min_interval_sec = (
                float(heuristic_flip_min_interval_sec)
                if heuristic_flip_min_interval_sec is not None
                else None
            )
        except Exception:
            self._heuristic_flip_min_interval_sec = None
        self._heuristic_flip_include_hold_transition = bool(heuristic_flip_include_hold_transition)
        self._rule_based_weights = (
            dict(rule_based_weights) if isinstance(rule_based_weights, dict) else None
        )
        try:
            self._rule_based_mom_multiplier = float(rule_based_mom_multiplier or 1.0)
        except Exception:
            self._rule_based_mom_multiplier = 1.0

        self._buy_threshold = float(buy_threshold)
        self._sell_threshold = float(sell_threshold)
        self._confidence_high_margin = float(confidence_high_margin)
        self._confidence_mid_margin = float(confidence_mid_margin)
        self._confidence_spread_max_for_high = float(confidence_spread_max_for_high)
        self._confidence_conformal_width_max_for_high = float(confidence_conformal_width_max_for_high)
        self._confidence_conformal_width_max_for_medium = float(confidence_conformal_width_max_for_medium)
        self._guard_basis_hold_thr = float(guard_basis_hold_thr)
        self._guard_basis_downgrade_thr = float(guard_basis_downgrade_thr)
        self._guard_atm_spread_pct_thr = float(guard_atm_spread_pct_thr)
        self._guard_atm_liq_log_thr = float(guard_atm_liq_log_thr)

        try:
            self._fc0_stale_threshold_sec = max(0.0, float(fc0_stale_threshold_sec or 0.0))
        except Exception:
            self._fc0_stale_threshold_sec = 10.0
        try:
            self._fc0_stale_cooldown_sec = max(0.0, float(fc0_stale_cooldown_sec or 0.0))
        except Exception:
            self._fc0_stale_cooldown_sec = 60.0

        self._seq_len = max(5, int(seq_len or 60))
        self._multiscale_5m = bool(multiscale_5m)
        self._multiscale_enabled = bool(multiscale_enabled)
        self._multiscale_time_scales = list(multiscale_time_scales or [1, 5, 15])
        self._tft_horizon = int(tft_horizon or HORIZON_SEC)
        self._fo0_stale_sec = max(1, int(fo0_stale_sec or 10))
        self._fo0_log_schema = bool(fo0_log_schema)
        self._llm_timeout_sec = max(0.1, float(llm_timeout_sec or LLM_TIMEOUT_SEC))
        try:
            _gto = (
                float(gemini_timeout_sec)
                if gemini_timeout_sec is not None
                else float(self._llm_timeout_sec) * 1.5
            )
            self._gemini_timeout_sec = max(float(self._llm_timeout_sec), float(_gto))
        except Exception:
            self._gemini_timeout_sec = float(self._llm_timeout_sec)
        self._llm_min_interval_sec = max(0.0, float(llm_min_interval_sec or 0.0))
        try:
            self._llm_provider_cooldown_on_timeout_sec = max(
                0.0,
                float(llm_provider_cooldown_on_timeout_sec or LLM_PROVIDER_COOLDOWN_ON_TIMEOUT),
            )
        except Exception:
            self._llm_provider_cooldown_on_timeout_sec = float(LLM_PROVIDER_COOLDOWN_ON_TIMEOUT)
        self._dump_llm_prompt = bool(dump_llm_prompt)
        self._llm_prompt_dumped = False

        self._tick_size = max(0.0, float(tick_size or 0.0))
        self._feedback_threshold_ticks = max(1, int(feedback_threshold_ticks or 10))
        self._feedback_skip_hold_ticks = max(0, int(feedback_skip_hold_ticks or 0))
        self._feedback_weight_high = float(feedback_weight_high)
        self._feedback_weight_mid = float(feedback_weight_mid)
        self._feedback_weight_low = float(feedback_weight_low)
        self._feedback_use_price_snapshot = bool(feedback_use_price_snapshot)
        try:
            self._feedback_snapshot_tolerance_sec = max(0.0, float(feedback_snapshot_tolerance_sec or 0.0))
        except Exception:
            self._feedback_snapshot_tolerance_sec = 0.0
        self._feedback_snapshot_required = bool(feedback_snapshot_required)
        self._feedback_max_pending = max(10, int(feedback_max_pending or 200))
        self._feedback_queue: deque[Dict[str, Any]] = deque(maxlen=int(self._feedback_max_pending))

        self._last_llm_call_epoch: float = 0.0
        self._last_llm_cache_key: str = ""
        self._last_llm_result = None

        self._llm_rate_limited_until_epoch: float = 0.0
        self._last_result: Dict[str, Any] = {}
        # provider별 개별 rate limit (gpt/gemini/claude 각각 독립 쿨다운)
        self._provider_rate_limited_until: dict = {}

        self._disagreement_hold = bool(disagreement_hold)
        try:
            self._disagreement_hold_prob_diff_max = float(disagreement_hold_prob_diff_max)
        except Exception:
            self._disagreement_hold_prob_diff_max = 0.3
        self._disagreement_hold_prob_diff_max_by_regime = (
            dict(disagreement_hold_prob_diff_max_by_regime)
            if isinstance(disagreement_hold_prob_diff_max_by_regime, dict)
            else None
        )
        self._ensemble_agreement_confidence_boost = bool(ensemble_agreement_confidence_boost)
        try:
            self._ensemble_agreement_prob_diff_max = float(ensemble_agreement_prob_diff_max)
        except Exception:
            self._ensemble_agreement_prob_diff_max = 0.06

        self._dual_llm = bool(dual_llm)
        self._dual_llm_primary_provider = str(dual_llm_primary_provider or "gemini").strip().lower() or "gemini"
        if self._dual_llm_primary_provider in ("openai", "chatgpt"):
            self._dual_llm_primary_provider = "gpt"
        if self._dual_llm_primary_provider not in ("gpt", "gemini"):
            self._dual_llm_primary_provider = "gemini"  # 기본: Gemini 우선

        ad = adaptive_indicator if isinstance(adaptive_indicator, dict) else {}
        # [SSOT] ad dict 를 보관 — 외부에서 adaptive_indicator 설정을 조회할 때 사용
        # dict(ad) 복사는 ad 자체가 이미 사본이므로 직접 대입해도 동일
        self._adaptive_indicator: Dict[str, Any] = ad  # ad 와 동일 객체 공유

        # 피처 계산/LLM 컨텍스트용 분봉 lookback (warmup_bars와 분리)
        ml = minute_lookback if isinstance(minute_lookback, dict) else {}
        try:
            default_futures_minutes = max(1, int(ml.get("futures", 60) or 60))
        except Exception:
            default_futures_minutes = 60
        try:
            default_options_minutes = max(1, int(ml.get("options", 60) or 60))
        except Exception:
            default_options_minutes = 60

        # tick_provider 외부 주입 지원 (DI / 테스트 목업용)
        # 미전달 시 기존처럼 RealTimeTickProcessor 직접 생성
        if tick_provider is not None:
            self.tick_processor: TickDataProvider = tick_provider
        else:
            self.tick_processor = RealTimeTickProcessor(
                default_futures_minutes=int(default_futures_minutes),
                default_options_minutes=int(default_options_minutes),
            )

        self._option_feature_set = str(option_feature_set or "v1").strip().lower() or "v1"
        if self._option_feature_set not in ("v1", "v2", "v3", "v4", "v5"):
            self._option_feature_set = "v1"

        # 각 guardrail을 버전 문자열 비교 대신 독립 bool 플래그로 제어한다.
        # config.json의 "guardrails" 섹션이 있으면 해당 값을 사용하고,
        # 없으면 option_feature_set 기반으로 기본값을 결정한다.
        _fs = self._option_feature_set
        _gr_cfg = {}
        try:
            import json as _json
            with open(str(self._config_path), "r", encoding="utf-8") as _f:
                _gr_cfg = _json.load(_f).get("guardrails") or {}
        except Exception as _e:
            logger.debug("오류 무시: %s", _e)
        self._guardrail_option_enabled = bool(_gr_cfg.get("option",  True))
        self._guardrail_basis_enabled  = bool(_gr_cfg.get("basis",   True))
        self._guardrail_parity_enabled = bool(_gr_cfg.get("parity",  _fs in ("v3", "v4", "v5")))
        self._guardrail_bleed_enabled  = bool(_gr_cfg.get("bleed",   _fs in ("v4", "v5")))
        self._guardrail_oi_enabled     = bool(_gr_cfg.get("oi",      _fs == "v5"))
        try:
            self._otm_open_min: float = max(0.0, float(otm_open_min or 0.30))
        except Exception:
            self._otm_open_min = 0.30
        try:
            self._pcr_atm_strikes_each_side = max(0, min(50, int(pcr_atm_strikes_each_side)))
        except Exception:
            self._pcr_atm_strikes_each_side = 5
        self._opt_keys = list(get_opt_keys(str(self._option_feature_set)))
        
        # tick_provider, minute_lookback, option_minute_ohlcv, notifier는 _build_components로 전달
        self._tick_provider_param = tick_provider
        self._minute_lookback_param = minute_lookback
        self._option_minute_ohlcv_param = option_minute_ohlcv
        self._notifier_param = notifier
        
        # OI alert cooldown은 _init_state로 전달
        self._oi_alert_cooldown_sec_param = oi_alert_cooldown_sec

    def _build_components(
        self,
        *,
        anthropic_key: Optional[str],
        openai_key: Optional[str],
        gemini_key: Optional[str],
        preferred_provider: Optional[str],
        tick_provider: "Optional[TickDataProvider]",
        numeric_predictor: str,
        model_class: str,
        patch_len: int,
        stride: int,
        conformal_alpha: float,
        conformal_path: Optional[str],
        mamba_enabled: bool,
        mamba_weights_path: Optional[str],
        mamba_weight: float,
        transformer_weight: float,
        transformer_weights_path: Optional[str],
        tft_weights_path: Optional[str],
        tft_horizon: int,
        disagreement_hold: bool,
        disagreement_hold_prob_diff_max: float,
        ensemble_agreement_confidence_boost: bool,
        ensemble_agreement_prob_diff_max: float,
        rule_based_weights: Optional[Dict[str, float]],
        rule_based_mom_multiplier: float,
        min_minute_bars_required: int,
        adaptive_indicator: Optional[dict],
        config_path: Optional[str],
    ) -> None:
        """컴포넌트 생성 및 초기화."""
        # tick_processor 설정
        ml = self._minute_lookback_param if isinstance(self._minute_lookback_param, dict) else {}
        try:
            default_futures_minutes = max(1, int(ml.get("futures", 60) or 60))
        except Exception:
            default_futures_minutes = 60
        try:
            default_options_minutes = max(1, int(ml.get("options", 60) or 60))
        except Exception:
            default_options_minutes = 60

        if self._tick_provider_param is not None:
            self.tick_processor: TickDataProvider = self._tick_provider_param
        else:
            self.tick_processor = RealTimeTickProcessor(
                default_futures_minutes=int(default_futures_minutes),
                default_options_minutes=int(default_options_minutes),
            )

        # option_minute_ohlcv 설정
        try:
            om = self._option_minute_ohlcv_param if isinstance(self._option_minute_ohlcv_param, dict) else {}
            self.tick_processor.configure_option_minute_ohlcv(
                enabled=bool(om.get("enabled", False)),
                atm_window=int(om.get("atm_window", 2) or 2),
            )
        except Exception as _e:
            logger.debug("오류 무시: %s", _e)
        
        self._notifier = self._notifier_param
        self.judge = (
            LLMJudge(
                anthropic_key=anthropic_key,
                openai_key=openai_key,
                gemini_key=gemini_key,
                preferred_provider=preferred_provider,
                notifier=self._notifier,
            )
            if self._use_llm
            else None
        )

        if self._use_llm and self.judge is not None:
            try:
                has_any_key = any(
                    [
                        bool(anthropic_key),
                        bool(openai_key),
                        bool(gemini_key),
                    ]
                )
                has_any_client = any(
                    [
                        bool(getattr(self.judge, "_anthropic", None)),
                        bool(getattr(self.judge, "_openai", None)),
                        bool(getattr(self.judge, "_gemini", None)),
                    ]
                )
                if not has_any_client:
                    logger.warning(
                        "LLM enabled but no provider client initialized (keys_present=%s, anthropic=%s, openai=%s, gemini=%s). "
                        "If running from Task Scheduler or before market open, ensure API keys are available in that environment.",
                        bool(has_any_key),
                        bool(anthropic_key),
                        bool(openai_key),
                        bool(gemini_key),
                    )
            except Exception as _e:
                logger.debug("오류 무시: %s", _e)

        self._llm_executor: Optional[ThreadPoolExecutor] = None
        self._llm_executor_lock = threading.Lock()
        if self._use_llm:
            try:
                workers = 2 if self._dual_llm else 1
                self._llm_executor = ThreadPoolExecutor(max_workers=int(workers))
            except Exception:
                self._llm_executor = ThreadPoolExecutor(max_workers=1)

        # Adaptive indicators 초기화
        self._adaptive_mgr = None
        self._adaptive_warmed = False
        self._adaptive_last_minute_ts: Optional[datetime] = None
        ad = adaptive_indicator if isinstance(adaptive_indicator, dict) else {}
        try:
            self._adaptive_enabled = bool(ad.get("enabled", True))
        except Exception:
            self._adaptive_enabled = True

        self._adaptive_last_features: Dict[str, float] = {}
        self._adaptive_last_context: str = ""

        # 지표 인스턴스 생성
        self._init_indicators(adaptive_indicator=adaptive_indicator)

    def _init_indicators(self, *, adaptive_indicator: Optional[dict]) -> None:
        """지표 인스턴스 초기화."""
        # 초기화 실패 시 None 유지 → adaptive_mixin 에서 None 체크 후 스킵
        self._aap: Optional[Any] = None          # ATRAdaptivePivot
        self._pap: Optional[Any] = None          # PercentAdaptivePivot
        self._msb: Optional[Any] = None          # MarketStructureBreak
        self._kf:  Optional[Any] = None          # KalmanTurningPoint
        self._oi_gate: Optional[Any] = None      # OIStructureGate
        self._integrator: Optional[Any] = None   # PivotScoreIntegrator
        try:
            from indicators import (
                ATRAdaptivePivot,    ATRAdaptivePivotConfig,
                PercentAdaptivePivot, PercentAdaptivePivotConfig,
                MarketStructureBreak, MSBConfig,
                OIStructureGate,     OIStructureConfig,
                KalmanTurningPoint,  KalmanConfig,
                PivotScoreIntegrator, IntegratorConfig,
            )
            ad = adaptive_indicator if isinstance(adaptive_indicator, dict) else {}
            # config.json adaptive_indicator.atr_pivot 섹션에서 파라미터 로드
            # 없으면 실전 검증된 기본값 사용
            _atr_cfg = dict((ad or {}).get("atr_pivot") or {})
            self._aap = ATRAdaptivePivot(ATRAdaptivePivotConfig(
                atr_period         = int(_atr_cfg.get("atr_period",       14)  or 14),
                base_multiplier    = float(_atr_cfg.get("base_multiplier", 2.0) or 2.0),
                multiplier_min     = float(_atr_cfg.get("multiplier_min",  1.2) or 1.2),
                multiplier_max     = float(_atr_cfg.get("multiplier_max",  3.5) or 3.5),
                er_period          = int(_atr_cfg.get("er_period",         10)  or 10),
                confirmation_bars  = int(_atr_cfg.get("confirmation_bars",  1)  or 1),
                min_wave_atr_ratio = float(_atr_cfg.get("min_wave_atr_ratio", 0.5) or 0.5),
                warmup_bars        = int(_atr_cfg.get("warmup_bars",       20)  or 20),
            ))
            self._aap.set_symbol(str(ad.get("symbol") or "KP200 선물"))

            # pivot_type에 따라 배타적으로 초기화
            _pivot_type = str((ad or {}).get("pivot_type", "atr") or "atr").lower()

            if _pivot_type == "atr":
                self._pap = None  # ATRAdaptivePivot 사용 시 PercentAdaptivePivot 비활성
                logger.info("[PIPELINE_INIT] pivot_type=atr → ATRAdaptivePivot 활성, PercentAdaptivePivot 비활성")
            elif _pivot_type == "percent":
                self._aap = None  # PercentAdaptivePivot 사용 시 ATRAdaptivePivot 비활성
                # config.json adaptive_indicator.percent_pivot 섹션에서 파라미터 로드
                _pap_cfg = dict((ad or {}).get("percent_pivot") or {})
                self._pap = PercentAdaptivePivot(PercentAdaptivePivotConfig(
                    base_pct             = float(_pap_cfg.get("base_pct",             0.3)  or 0.3),
                    multiplier_min       = float(_pap_cfg.get("multiplier_min",       0.8)  or 0.8),
                    multiplier_max       = float(_pap_cfg.get("multiplier_max",       2.0)  or 2.0),
                    er_period            = int(_pap_cfg.get("er_period",             10)   or 10),
                    confirmation_bars    = int(_pap_cfg.get("confirmation_bars",     1)    or 1),
                    min_wave_pct         = float(_pap_cfg.get("min_wave_pct",        0.15) or 0.15),
                    min_bar_gap          = int(_pap_cfg.get("min_bar_gap",           3)    or 3),
                    max_pivots           = int(_pap_cfg.get("max_pivots",           30)   or 30),
                    warmup_bars          = int(_pap_cfg.get("warmup_bars",           20)   or 20),
                    cancel_ratio         = float(_pap_cfg.get("cancel_ratio",        0.3)  or 0.3),
                ))
                logger.info("[PIPELINE_INIT] pivot_type=percent → PercentAdaptivePivot 활성, ATRAdaptivePivot 비활성")
            else:
                logger.warning("[PIPELINE_INIT] 알 수 없는 pivot_type=%s, 기본값 atr 사용", _pivot_type)
                self._pap = None

            _msb_cfg = dict((ad or {}).get("msb") or {})
            self._msb = MarketStructureBreak(MSBConfig(
                swing_lookback            = int(_msb_cfg.get("swing_lookback",           3)    or 3),
                bos_buffer_pct            = float(_msb_cfg.get("bos_buffer_pct",         0.20) or 0.20),
                structure_lookback_pivots = int(_msb_cfg.get("structure_lookback_pivots", 6)   or 6),
                choch_enabled             = bool(_msb_cfg.get("choch_enabled",            True)),
            ))

            _kf_cfg = dict((ad or {}).get("kalman") or {})
            self._kf = KalmanTurningPoint(KalmanConfig(
                q              = float(_kf_cfg.get("q",           0.01) or 0.01),
                r              = float(_kf_cfg.get("r",           2.0)  or 2.0),
                warmup_bars    = int(_kf_cfg.get("warmup_bars",   15)   or 15),
                slope_flip_min = float(_kf_cfg.get("slope_flip_min", 0.005) or 0.005),
                adaptive_q     = bool(_kf_cfg.get("adaptive_q",   True)),
            ))

            self._oi_gate = OIStructureGate(OIStructureConfig(
                oi_proximity_pct = float((ad or {}).get("oi_proximity_pct", 0.3) or 0.3),
            ))

            _int_cfg = dict((ad or {}).get("integrator") or {})
            # pivot_type에 따라 가중치 동적 조정
            if _pivot_type == "atr":
                _w_aap = float(_int_cfg.get("w_aap", 0.50) or 0.50)
                _w_pap = 0.0
            elif _pivot_type == "percent":
                _w_aap = 0.0
                _w_pap = float(_int_cfg.get("w_pap", 0.50) or 0.50)
            else:
                _w_aap = float(_int_cfg.get("w_aap", 0.50) or 0.50)
                _w_pap = 0.0

            self._integrator = PivotScoreIntegrator(IntegratorConfig(
                w_aap            = _w_aap,
                w_pap            = _w_pap,
                w_msb            = float(_int_cfg.get("w_msb",            0.25) or 0.25),
                w_oi             = float(_int_cfg.get("w_oi",             0.10) or 0.10),
                w_kf             = float(_int_cfg.get("w_kf",             0.15) or 0.15),
                entry_threshold  = float(_int_cfg.get("entry_threshold",  0.55) or 0.55),
                strong_threshold = float(_int_cfg.get("strong_threshold", 0.72) or 0.72),
                regime_boost     = float(_int_cfg.get("regime_boost",     1.15) or 1.15),
                regime_suppress  = float(_int_cfg.get("regime_suppress",  0.85) or 0.85),
            ))
            logger.info(
                "[PIPELINE_INIT] %s·MSB·Kalman·OIGate·Integrator 초기화 완료 (w_aap=%.2f, w_pap=%.2f)",
                "ATRAdaptivePivot" if _pivot_type == "atr" else "PercentAdaptivePivot",
                _w_aap,
                _w_pap
            )
        except Exception as _step_ex:
            logger.warning(
                "[PIPELINE_INIT] Step 1~3 지표 초기화 실패 (비필수 — 계속 진행): %s",
                _step_ex,
            )
            self._aap = self._pap = self._msb = self._kf = self._oi_gate = self._integrator = None

    def _init_state(self, *, config_path: Optional[str]) -> None:
        """상태 변수 초기화."""
        # CON-01: _metrics는 여러 스레드(틱 콜백, 예측 루프, 피드백 루프)에서 동시 접근.
        # GIL이 단순 대입은 보호하지만 read-modify-write 복합 연산은 race condition 발생 가능.
        self._metrics_lock = threading.Lock()
        # 옵션 틱 유입 강도(1분/20분 평균 배수) 집계 상태
        self._opt_tick_flow_window: deque[float] = deque(maxlen=20)
        self._opt_tick_flow_last_minute: Optional[datetime] = None
        self._opt_tick_flow_last_total_ticks: int = 0
        self._opt_tick_flow_last_call_ticks: int = 0
        self._opt_tick_flow_last_put_ticks: int = 0
        self._opt_tick_flow_last_price: float = 0.0

        # CON-02: LLM 캐시/rate-limit 변수는 dual_llm 모드에서 두 스레드가 동시에 접근.
        self._llm_cache_lock = threading.Lock()

        self._ob_lock = threading.Lock()
        self._ob_records: deque[Dict[str, Any]] = deque(maxlen=int(self._seq_len))
        self._last_ob_snapshot: Dict[str, Any] = {}
        self._fo0_schema_logged = False
        self._last_fo0_second: Optional[int] = None
        self._last_fo0_sig: Optional[tuple] = None
        self._last_fo0_seen_epoch: Optional[float] = None
        self._fo0_stale_warned = False
        self._last_fo0_stale_warn_epoch: Optional[float] = None

        self._last_fc0_seen_epoch: Optional[float] = None
        self._last_fc0_stale_warn_epoch: Optional[float] = None

        self._last_opt_sec_key: Optional[int] = None
        self._last_opt_features: Optional[Dict[str, Any]] = None

        # Best-effort background snapshots fetched during initialization.
        self._t2101_snapshot: Dict[str, Any] = {}
        self._t2301_snapshot: Dict[str, Any] = {}
        self._ij_realtime_snapshot: Dict[str, Any] = {}

        # v3 option feature: parity divergence 계산용 직전 틱 상태 캐시.
        # OB 버퍼 경로(1Hz)에서 _build_option_snapshot_safe(update_prev=True)가
        # 매 초 갱신한다. get_prediction() 경로는 update_prev=False로 호출하여
        # 이중 갱신을 방지한다.
        self._prev_underlying_price: Optional[float] = None
        self._prev_oi_levels: Dict[str, float] = {}  # OI velocity 계산용 이전 스냅샷
        self._oi_alert_last_epoch: float = 0.0        # OI 레벨 변경 알람 마지막 전송 시각
        self._OI_ALERT_COOLDOWN_SEC: float = max(10.0, float(self._oi_alert_cooldown_sec_param or 300.0))
        self._sigma_multiplier: float = 1.0       # calc_expected_amplitude 배율 (피드백 조정)
        self._exhaust_exceed_count: int = 0        # amplitude_exhaustion > 1.0 연속 횟수
        # 방안C: 실현 진폭 EMA — 장 종료 시 당일 realized_hl_range_pt를 반영해 다음날 IV 보정에 활용
        self._realized_amplitude_ema: float = 0.0  # 지수이동평균 (0이면 미초기화)
        self._realized_amplitude_ema_alpha: float = 0.2  # EMA 평활 계수 (≈ 최근 5일 가중)
        self._realized_amplitude_ema_updated_date: str = ""  # 마지막 갱신 날짜 (YYYYMMDD)
        self._prev_atm_call_price: Optional[float] = None
        self._prev_atm_put_price: Optional[float] = None

        # ── Step 1~3 신규 지표 인스턴스 (best-effort; adaptive_mgr 와 독립) ──
        # 초기화 실패 시 None 유지 → adaptive_mixin 에서 None 체크 후 스킵
        self._aap: Optional[Any] = None          # ATRAdaptivePivot
        self._pap: Optional[Any] = None          # PercentAdaptivePivot
        self._msb: Optional[Any] = None          # MarketStructureBreak
        self._kf:  Optional[Any] = None          # KalmanTurningPoint
        self._oi_gate: Optional[Any] = None      # OIStructureGate
        self._integrator: Optional[Any] = None   # PivotScoreIntegrator

        # Adaptive config 초기화
        self._init_adaptive_config(adaptive_indicator=self._adaptive_indicator_param)

    def _init_adaptive_config(self, *, adaptive_indicator: Optional[dict]) -> None:
        """Adaptive indicator 설정 초기화."""
        ad = adaptive_indicator if isinstance(adaptive_indicator, dict) else {}
        
        # P10: ADX 기반 confidence 조정
        self._adx_confidence_filter_enabled = bool(ad.get("adx_confidence_filter", {}).get("enabled", False))
        self._adx_hold_threshold = float(ad.get("adx_confidence_filter", {}).get("hold_threshold", 15.0))
        self._adx_weak_threshold = float(ad.get("adx_confidence_filter", {}).get("weak_threshold", 20.0))
        self._adx_strong_threshold = float(ad.get("adx_confidence_filter", {}).get("strong_threshold", 35.0))
        
        # 실시간 거래 이벤트 로거
        self._trade_logger = None
        try:
            trade_logging_enabled = bool(ad.get("trade_logging", {}).get("enabled", False))
            if trade_logging_enabled:
                from prediction.trade_logger import get_trade_logger
                from pathlib import Path
                log_dir = Path(ad.get("trade_logging", {}).get("log_dir", "logs/trades"))
                self._trade_logger = get_trade_logger()
                # 로거 로그 디렉토리 업데이트
                self._trade_logger.log_dir = log_dir
                self._trade_logger.log_dir.mkdir(parents=True, exist_ok=True)
                self._trade_logger.current_log_file = self._trade_logger._get_log_file()
        except Exception:
            pass
        
        # _compute_adaptive_bundle: 완결봉 미진행 시 features 캐시와 짝을 맞추기 위한 ZigZag 상태
        self._adaptive_last_zigzag_state: Any = None
        # FIX-HEURISTIC: 이전 라운드 ast_dir를 저장해 FLIP 감지에 사용한다.
        # azz_new_swing=0이 75/77회라 AND 조건이 사실상 항상 HOLD였던 문제를 해결.
        self._adaptive_prev_ast_dir: Optional[int] = None

        # FLIP-ACCUM: 예측 주기(5분) 사이에 발생한 SuperTrend flip을 누적한다.
        # _compute_adaptive_bundle 내에서 신규 봉들을 순서대로 처리하며 ast_signal != 0이면
        # "BUY"/"SELL"을 기록하고, 판정 시 소비(reset to None)한다.
        # 복수의 flip이 발생하면 마지막 flip 방향을 우선한다.
        self._adaptive_pending_flip: Optional[str] = None

        # ADAPT rewind 쿨다운 — 장 마감 직전 API 재전송으로 인한 반복 rewind 억제
        # 연속 rewind가 _ADAPT_REWIND_COOLDOWN_SEC 이내에 발생하면 reset을 건너뛰고
        # 직전 features를 재사용한다.
        self._adaptive_last_rewind_epoch: float = 0.0
        self._ADAPT_REWIND_COOLDOWN_SEC: float = 60.0   # 60초 내 재발 시 무시
        # 동일 rewind 사유 반복 억제(같은 last_complete_ts/prev 조합의 reset 스팸 방지)
        self._adaptive_last_rewind_reason: str = ""
        # rewind 사유 정규 키(full/incremental 공통) — 동일 ts쌍이면 한 번만 reset
        self._adaptive_last_rewind_key: str = ""

        # RANGING FILTER: adaptive_indicator dict(또는 빈 기본)에서 임계값 로드
        try:
            _rf_any = ad.get("ranging_filter")
            _rf = _rf_any if isinstance(_rf_any, dict) else {}
            self._rf_enabled: bool = bool(_rf.get("enabled", True))
            self._rf_adx_min: float = float(_rf.get("adx_min", 15.0) or 15.0)
            self._rf_er_min: float = float(_rf.get("er_min", 0.15) or 0.15)
            self._rf_use_zigzag: bool = bool(_rf.get("use_zigzag_structure", True))
            self._rf_whipsaw_min_bars: int = int(_rf.get("whipsaw_min_bars", 2) or 2)
            self._rf_use_adx: bool = bool(_rf.get("use_adx_filter", True))
            self._rf_use_er: bool = bool(_rf.get("use_er_filter", True))
            self._rf_use_whipsaw: bool = bool(_rf.get("use_whipsaw_filter", True))
        except Exception:
            self._rf_enabled = True
            self._rf_adx_min = 15.0
            self._rf_er_min = 0.15
            self._rf_use_zigzag = True
            self._rf_whipsaw_min_bars = 2
            self._rf_use_adx = True
            self._rf_use_er = True
            self._rf_use_whipsaw = True

        logger.info(
            "[RANGING_FILTER] 설정 로드 enabled=%s adx_min=%.1f er_min=%.2f "
            "zigzag=%s whipsaw_min_bars=%d use_adx=%s use_er=%s use_whipsaw=%s",
            self._rf_enabled, self._rf_adx_min, self._rf_er_min,
            self._rf_use_zigzag, self._rf_whipsaw_min_bars,
            self._rf_use_adx, self._rf_use_er, self._rf_use_whipsaw,
        )

        # warmup_bars: config 값 우선, fallback=45
        # (120 하드코딩 제거 — config에서 읽지 않던 버그 수정)
        # 최솟값 45 근거: ADX 이중 RMA 수렴(≈28봉) + ZigZag 구조 신호 안정화 버퍼
        try:
            self._adaptive_warmup_bars = max(
                45, int(ad.get("warmup_bars", 45) or 45)
            )
        except Exception:
            self._adaptive_warmup_bars = 45

        self._min_minute_bars_required = max(1, int(min_minute_bars_required or 20))
        self._metrics: Dict[str, Any] = {
            "predictions": 0,
            "prediction_failures": 0,
            "ticks_processed": 0,
            "last_latency_ms": None,
            "feedback_evaluations": 0,
            "feedback_weight_updates": 0,
            "guardrail_oi_skipped_no_data": 0,
            "heur_flip_triggered": 0,
            "heur_flip_skipped_interval": 0,
            "zz_confirm_triggered": 0,
            "zz_confirm_skipped_interval": 0,
        }

        # ── TransformerQualityTracker 초기화 ─────────────────────────────────
        # Telegram notifier는 bridge 연결 후 set_quality_notifier()로 주입
        try:
            from prediction.transformer_quality_tracker import TransformerQualityTracker
            self._quality_tracker = TransformerQualityTracker(notifier=None)
        except Exception as _qt_err:
            logger.warning("[Pipeline] TransformerQualityTracker 초기화 실패(무시): %s", _qt_err)
            self._quality_tracker = None

        # 옵션 센티먼트 분석기 초기화
        try:
            self._init_option_sentiment_analyzer(config_path=str(self._config_path))
        except Exception:
            self._option_sentiment_analyzer = None

    def warmup_llm(self, *, timeout_sec: Optional[float] = None) -> Dict[str, Any]:
        """Best-effort LLM warmup.

        This is intended to be called once at startup so connectivity/auth issues surface
        before the first prediction round.

        Returns a diagnostics dict and never raises.
        """
        out: Dict[str, Any] = {
            "attempted": False,
            "skipped": False,
            "provider": None,
            "timed_out": False,
            "error": None,
        }

        if not self._use_llm or self.judge is None:
            out["skipped"] = True
            return out

        out["attempted"] = True
        old_timeout = self._llm_timeout_sec
        try:
            if timeout_sec is not None:
                self._llm_timeout_sec = max(0.1, float(timeout_sec))

            system = "You are a connectivity warmup check. Reply strictly in JSON."
            user = "Return {\"action\":\"HOLD\",\"risk_level\":\"LOW\",\"rationale\":\"warmup\",\"caution\":\"\"}."
            judgment, timed_out, err = self._judge_with_timeout(system=system, user=user)
            out["timed_out"] = bool(timed_out)
            if err:
                out["error"] = str(err)
            try:
                if judgment is not None:
                    out["provider"] = str(getattr(judgment, "provider", "") or "")
            except Exception:
                out["provider"] = None
            return out
        except Exception as e:
            out["error"] = str(e)
            return out
        finally:
            try:
                self._llm_timeout_sec = old_timeout
            except Exception as _e:
                logger.debug("오류 무시: %s", _e)

    def set_market_snapshots(
        self,
        *,
        t2101: Optional[Dict[str, Any]] = None,
        t2301: Optional[Dict[str, Any]] = None,
        ij_: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Attach best-effort background snapshots for LLM context.

        This is optional and can be called by `ebest_live.py` after initialization.
        """
        if isinstance(t2101, dict) and t2101:
            # FIX: t2101 open 유효성 검증.
            # eBest t2101은 장 초기에 open=0 또는 전일 종가를 반환하는 경우가 있다.
            # - 새 open이 유효(>0)하면 항상 갱신
            # - 새 open이 0이지만 기존에 유효한 open이 없을 때만 갱신 (초기화)
            # - 새 open이 0이고 기존에 유효한 open이 있으면 기존 값 보존
            _new_open = float(t2101.get("open") or 0.0)
            _prev_open = float((self._t2101_snapshot or {}).get("open") or 0.0)
            if _new_open > 0.0 or _prev_open == 0.0:
                # open 이외 필드(high/low/price 등)는 최신 값으로 항상 갱신하되
                # open만 기존 유효값 보존 로직 적용
                _merged = dict(t2101)
                if _new_open == 0.0 and _prev_open > 0.0:
                    _merged["open"] = _prev_open
                self._t2101_snapshot = _merged
        if isinstance(t2301, dict) and t2301:
            self._t2301_snapshot = dict(t2301)
        if isinstance(ij_, dict) and ij_:
            self._ij_realtime_snapshot = dict(ij_)

    def _setup_pivot_proximity_callback(self) -> None:
        """피봇 근접 감지 시 텔레그램 알림 콜백 설정."""
        if self._adaptive_mgr is None or self._notifier is None:
            return
        
        def _pivot_proximity_alert_callback(**kwargs: Any) -> None:
            """피봇 근접 감지 알림 콜백."""
            try:
                kospi_type = kwargs.get("kospi_type", "")
                kospi_price = kwargs.get("kospi_price", 0.0)
                kospi_idx = kwargs.get("kospi_idx", 0)
                futures_type = kwargs.get("futures_type", "")
                futures_price = kwargs.get("futures_price", 0.0)
                futures_idx = kwargs.get("futures_idx", 0)
                idx_diff = kwargs.get("idx_diff", 0)
                
                message = (
                    f"🔔 피봇 근접 감지\n"
                    f"KOSPI: {kospi_type}@{kospi_price:.2f} (idx:{kospi_idx})\n"
                    f"KP200: {futures_type}@{futures_price:.2f} (idx:{futures_idx})\n"
                    f"차이: {idx_diff}봉\n"
                    f"⚠️ 주요 분봉 가능성 높음"
                )
                
                # 텔레그램 전송
                if hasattr(self._notifier, 'send_text'):
                    self._notifier.send_text(message)
            except Exception as e:
                logger.warning("[PIVOT_PROXIMITY] 텔레그램 알림 전송 실패: %s", e)
        
        try:
            self._adaptive_mgr.set_pivot_proximity_callback(_pivot_proximity_alert_callback)
        except Exception as e:
            logger.warning("[PIVOT_PROXIMITY] 콜백 설정 실패: %s", e)

    def _setup_pivot_candidate_callback(self) -> None:
        """피봇 후보 알림 텔레그램 콜백 설정."""
        if self._adaptive_mgr is None or self._notifier is None:
            return

        def _pivot_candidate_alert_callback(**kwargs: Any) -> None:
            """피봇 후보 알림 콜백."""
            try:
                event_type = kwargs.get("event_type", "")
                symbol = kwargs.get("symbol", "")
                candidate_type = kwargs.get("candidate_type", "")
                candidate_price = kwargs.get("candidate_price", 0.0)
                bar_idx = kwargs.get("bar_idx", 0)
                timestamp = kwargs.get("timestamp", "")
                reason = kwargs.get("reason", "")

                logger.info(
                    "[PIVOT_CANDIDATE] 이벤트: %s, 심볼: %s, 유형: %s, 가격: %.2f, 인덱스: %d, 시각: %s, 사유: %s",
                    event_type, symbol, candidate_type, candidate_price, bar_idx, timestamp, reason
                )

                # 이벤트 유형에 따라 이모지 결정
                if event_type == "registered":
                    emoji = "🔔"
                    title = "피봇 후보 등록"
                elif event_type == "changed":
                    emoji = "🔄"
                    title = "피봇 후보 변경"
                elif event_type == "cancelled":
                    emoji = "🚫"
                    title = "피봇 후보 취소"
                else:
                    return

                message = (
                    f"{emoji} {title}\n"
                    f"{symbol}: {candidate_type}@{candidate_price:.2f}\n"
                    f"시각: {timestamp}"
                )
                if reason:
                    message += f"\n사유: {reason}"

                # 텔레그램 전송
                if hasattr(self._notifier, 'send_text'):
                    self._notifier.send_text(message)
                    logger.info("[PIVOT_CANDIDATE] 텔레그램 전송 성공")
            except Exception as e:
                logger.warning("[PIVOT_CANDIDATE] 텔레그램 알림 전송 실패: %s", e)

        try:
            self._adaptive_mgr.set_pivot_candidate_callback(_pivot_candidate_alert_callback)
        except Exception as e:
            logger.warning("[PIVOT_CANDIDATE] 콜백 설정 실패: %s", e)

    @property
    def prediction_minutes(self) -> int:
        """prediction_minutes.
"""
        return int(self._prediction_minutes)

    @property
    def min_minute_bars_required(self) -> int:
        """min_minute_bars_required.
"""
        return int(self._min_minute_bars_required)

    def close(self) -> None:
        """파이프라인 리소스를 명시적으로 해제한다.

        ThreadPoolExecutor 워커 스레드를 종료하고 LLM 클라이언트 연결 풀을 닫는다.
        프로그램 종료 전 또는 파이프라인 교체 시 반드시 호출해야 한다.
        """
        with self._llm_executor_lock:
            if self._llm_executor is not None:
                try:
                    self._llm_executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    try:
                        self._llm_executor.shutdown(wait=False)
                    except Exception as _e:
                        logger.debug("[close] 오류 무시: %s", _e)
                self._llm_executor = None
                logger.info("[Pipeline] ThreadPoolExecutor 종료 완료")

        # LLM 클라이언트 HTTP 세션 해제
        if self.judge is not None:
            fn = getattr(self.judge, "close", None)
            if callable(fn):
                try:
                    fn()
                except Exception as _e:
                    logger.debug("[close] 오류 무시: %s", _e)

    def __del__(self) -> None:
        """GC 수거 시 close()를 best-effort로 호출한다."""
        try:
            self.close()
        except Exception as _e:
            logger.debug("[__del__] 오류 무시: %s", _e)

    def get_metrics(self) -> Dict[str, Any]:
        """Return internal runtime counters/latency metrics."""
        with self._metrics_lock:
            out = dict(self._metrics)
        try:
            fn = getattr(self.numeric_predictor, "get_transformer_weight", None)
            if callable(fn):
                w_t = float(fn())
                out["feedback_transformer_weight"] = float(w_t)
                out["feedback_tft_weight"] = float(1.0 - float(w_t))
        except Exception as _e:
            logger.debug("[get_metrics] 오류 무시: %s", _e)
        # ── TransformerQualityTracker 메트릭 병합 ────────────────────────────
        try:
            qt = getattr(self, "_quality_tracker", None)
            if qt is not None:
                out.update(qt.get_metrics_dict())
        except Exception as _qe:
            logger.debug("[get_metrics] quality_tracker 병합 오류(무시): %s", _qe)
        return out

    def set_quality_notifier(self, notifier: Any) -> None:
        """Telegram notifier를 quality tracker에 주입한다.

        PipelineTelegramBridge.start() 호출 후 한 번만 호출하면 된다.
        """
        try:
            qt = getattr(self, "_quality_tracker", None)
            if qt is not None:
                qt._notifier = notifier
                logger.info("[Pipeline] quality_tracker notifier 연결 완료")
        except Exception as e:
            logger.debug("[Pipeline] set_quality_notifier 오류(무시): %s", e)

    def log_quality_summary(self) -> None:
        """장마감 후 품질 요약을 로그로 출력한다 (run_daily_backtest에서 호출)."""
        try:
            qt = getattr(self, "_quality_tracker", None)
            if qt is not None:
                qt.log_daily_summary()
        except Exception as e:
            logger.debug("[Pipeline] log_quality_summary 오류(무시): %s", e)

    def reset_adaptive_weights(self) -> bool:
        fn = getattr(self.numeric_predictor, "reset_adaptive_weights", None)
        if not callable(fn):
            return False
        try:
            fn()
            return True
        except Exception:
            return False


