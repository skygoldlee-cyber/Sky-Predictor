"""PredictionPipeline 팩토리 모듈.

main.py에서 분리된 _build_pipeline() 단독 모듈.
config / args 를 받아 PredictionPipeline 인스턴스를 생성한다.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from typing import Optional

from config import AppConfig, HORIZON_SEC
from prediction import PredictionPipeline


def _build_pipeline(
    config: AppConfig,
    args: argparse.Namespace,
    *,
    transformer_weights_path: Optional[str] = None,
    tft_weights_path: Optional[str] = None,
    numeric_predictor_override: Optional[str] = None,
    transformer_weight_override: Optional[float] = None,
    notifier: Optional[object] = None,
) -> "PredictionPipeline":
    """AppConfig와 args로부터 PredictionPipeline을 생성한다.

    ARC-06: main() / run_live_mode()에 흩어져 있던 파이프라인 생성 로직을 단일화.
    파이프라인 파라미터 변경 시 이 함수만 수정하면 된다.

    Args:
        config: 로드된 AppConfig.
        args: parse_arguments() 또는 _make_args_from_gui()로 생성된 Namespace.
        transformer_weights_path: weights_selector 결과를 주입할 때 사용. None이면 config 값 사용.
        tft_weights_path: 동상.
        numeric_predictor_override: GUI에서 Transformer/TFT 체크박스로 결정한 모드 문자열.
        transformer_weight_override: ensemble이 아닐 때 1.0으로 고정하는 경우에 사용.
        notifier: TelegramNotifier 인스턴스. OI 레벨 변경 알람 등 pipeline 내부 알람에 사용.
                  None이면 pipeline 내부 텔레그램 알람 비활성.
    """
    from prediction.pipeline import PredictionPipeline

    pred = config.prediction
    ai = config.ai_providers

    numeric_predictor = str(
        numeric_predictor_override
        or getattr(args, "numeric_predictor", None)
        or getattr(pred, "numeric_predictor", "transformer")
        or "transformer"
    )

    _transformer_path = (
        str(transformer_weights_path)
        if transformer_weights_path
        else str(getattr(pred, "transformer_weights_path", "") or "") or None
    )
    _tft_path = (
        str(tft_weights_path)
        if tft_weights_path
        else str(getattr(pred, "tft_weights_path", "") or "") or None
    )

    _transformer_weight = float(
        transformer_weight_override
        if transformer_weight_override is not None
        else float(getattr(pred, "transformer_weight", 0.5) or 0.5)
    )

    return PredictionPipeline(
        # ── API 키 ──────────────────────────────────────────────────────
        anthropic_key=str(ai.anthropic_key or ""),
        openai_key=str(ai.openai_key or ""),
        gemini_key=str(ai.gemini_key or ""),
        # ── LLM 설정 ────────────────────────────────────────────────────
        use_llm=bool(pred.use_llm) and not bool(getattr(args, "heuristic_only", False)),
        heuristic_fallback=bool(getattr(pred, "heuristic_fallback", True)),
        heuristic_flip_min_interval_sec=getattr(pred, "heuristic_flip_min_interval_sec", None),
        heuristic_flip_include_hold_transition=bool(
            getattr(pred, "heuristic_flip_include_hold_transition", False)
        ),
        rule_based_weights=getattr(pred, "rule_based_weights", None),
        rule_based_mom_multiplier=float(getattr(pred, "rule_based_mom_multiplier", 1.0) or 1.0),
        preferred_provider=str(
            getattr(args, "preferred_provider", None)
            or getattr(pred, "preferred_provider", "") or ""
        ),
        dual_llm=bool(
            getattr(args, "dual_llm", None)
            if getattr(args, "dual_llm", None) is not None
            else getattr(pred, "dual_llm", False)
        ),
        dual_llm_primary_provider=str(
            getattr(args, "dual_llm_primary_provider", None)
            or getattr(pred, "dual_llm_primary_provider", "gpt") or "gpt"
        ),
        llm_timeout_sec=float(getattr(pred, "llm_timeout_sec", 8.0) or 8.0),
        llm_min_interval_sec=float(getattr(pred, "llm_min_interval_sec", 30.0) or 0.0),
        llm_provider_cooldown_on_timeout_sec=float(
            getattr(pred, "llm_provider_cooldown_on_timeout_sec", 60.0) or 0.0
        ),
        gemini_timeout_sec=(
            float(getattr(pred, "gemini_timeout_sec"))
            if getattr(pred, "gemini_timeout_sec", None) is not None
            else None
        ),
        dump_llm_prompt=bool(getattr(args, "dump_llm_prompt", False)),
        # ── 수치 예측기 ─────────────────────────────────────────────────
        numeric_predictor=numeric_predictor,
        model_class=str(getattr(pred, "model_class", "transformer") or "transformer"),
        patch_len=int(getattr(pred, "patch_len", 8) or 8),
        stride=int(getattr(pred, "stride", 4) or 4),
        conformal_alpha=float(getattr(pred, "conformal_alpha", 0.1) or 0.1),
        conformal_path=str(getattr(args, "conformal_path", None) or getattr(pred, "conformal_path", "") or "") or None,
        multiscale_5m=bool(getattr(pred, "multiscale_5m", False)),
        multiscale_enabled=bool(getattr(pred, "multiscale_enabled", False)),
        multiscale_time_scales=list(getattr(pred, "multiscale_time_scales", [1, 5, 15]) or [1, 5, 15]),
        mamba_enabled=bool(getattr(pred, "mamba_enabled", False)),
        mamba_weights_path=str(getattr(pred, "mamba_weights_path", "") or "") or None,
        mamba_weight=float(getattr(pred, "mamba_weight", 0.33) or 0.33),
        transformer_weights_path=_transformer_path,
        tft_weights_path=_tft_path,
        tft_horizon=int(getattr(pred, "tft_horizon", HORIZON_SEC) or HORIZON_SEC),
        transformer_weight=_transformer_weight,
        disagreement_hold=bool(getattr(pred, "disagreement_hold", True)),
        disagreement_hold_prob_diff_max=float(
            getattr(pred, "disagreement_hold_prob_diff_max", 0.1) or 0.1
        ),
        disagreement_hold_prob_diff_max_by_regime=getattr(
            pred, "disagreement_hold_prob_diff_max_by_regime", None
        ),
        ensemble_agreement_confidence_boost=bool(
            getattr(pred, "ensemble_agreement_confidence_boost", True)
        ),
        ensemble_agreement_prob_diff_max=float(
            getattr(pred, "ensemble_agreement_prob_diff_max", 0.06) or 0.06
        ),
        # ── 예측 설정 ───────────────────────────────────────────────────
        prediction_minutes=int(pred.minutes),
        buy_threshold=float(pred.buy_threshold),
        sell_threshold=float(pred.sell_threshold),
        confidence_high_margin=float(getattr(pred, "confidence_high_margin", 0.15) or 0.15),
        confidence_mid_margin=float(getattr(pred, "confidence_mid_margin", 0.08) or 0.08),
        confidence_spread_max_for_high=float(
            getattr(pred, "confidence_spread_max_for_high", 1.0) or 1.0
        ),
        confidence_conformal_width_max_for_high=float(
            getattr(pred, "confidence_conformal_width_max_for_high", 0.35) or 0.35
        ),
        confidence_conformal_width_max_for_medium=float(
            getattr(pred, "confidence_conformal_width_max_for_medium", 0.55) or 0.55
        ),
        option_feature_set=str(getattr(pred, "option_feature_set", "v1") or "v1"),
        otm_open_min=float(
            getattr(getattr(config, "options_subscription", None), "otm_open_min", 0.30) or 0.30
        ),
        pcr_atm_strikes_each_side=int(
            max(0, min(50, int(getattr(pred, "pcr_atm_strikes_each_side", 5) or 5)))
        ),
        min_minute_bars_required=int(getattr(pred, "min_minute_bars_required", 20) or 20),
        # ── 시퀀스 / FO0 ────────────────────────────────────────────────
        seq_len=int(
            getattr(args, "seq_len", None)
            if getattr(args, "seq_len", None) is not None
            else getattr(pred, "seq_len", 60) or 60
        ),
        fo0_stale_sec=int(
            getattr(args, "fo0_stale_sec", None)
            if getattr(args, "fo0_stale_sec", None) is not None
            else getattr(pred, "fo0_stale_sec", 10) or 10
        ),
        fo0_log_schema=bool(
            getattr(args, "fo0_log_schema", None)
            if getattr(args, "fo0_log_schema", None) is not None
            else getattr(pred, "fo0_log_schema", True)
        ),
        # ── 틱/피드백 ───────────────────────────────────────────────────
        tick_size=float(getattr(pred, "tick_size", 0.05) or 0.05),
        feedback_threshold_ticks=int(getattr(pred, "feedback_threshold_ticks", 10) or 10),
        feedback_skip_hold_ticks=int(getattr(pred, "feedback_skip_hold_ticks", 2) or 0),
        feedback_weight_high=float(getattr(pred, "feedback_weight_high", 1.0) or 0.0),
        feedback_weight_mid=float(getattr(pred, "feedback_weight_mid", 0.5) or 0.0),
        feedback_weight_low=float(getattr(pred, "feedback_weight_low", 0.25) or 0.0),
        feedback_use_price_snapshot=bool(getattr(pred, "feedback_use_price_snapshot", True)),
        feedback_snapshot_tolerance_sec=float(
            getattr(pred, "feedback_snapshot_tolerance_sec", 30.0) or 0.0
        ),
        feedback_snapshot_required=bool(getattr(pred, "feedback_snapshot_required", False)),
        # ── 가드레일 ────────────────────────────────────────────────────
        guard_basis_hold_thr=float(getattr(pred, "guard_basis_hold_thr", 2.5) or 2.5),
        guard_basis_downgrade_thr=float(getattr(pred, "guard_basis_downgrade_thr", 1.5) or 1.5),
        guard_atm_spread_pct_thr=float(getattr(pred, "guard_atm_spread_pct_thr", 1.5) or 1.5),
        guard_atm_liq_log_thr=float(getattr(pred, "guard_atm_liq_log_thr", 2.0) or 2.0),
        fc0_stale_threshold_sec=float(getattr(pred, "fc0_stale_threshold_sec", 10.0) or 0.0),
        fc0_stale_cooldown_sec=float(getattr(pred, "fc0_stale_cooldown_sec", 60.0) or 0.0),
        oi_alert_cooldown_sec=float(getattr(pred, "oi_alert_cooldown_sec", 300.0) or 300.0),
        # ── Adaptive indicator / 분봉 설정 ──────────────────────────────
        adaptive_indicator=asdict(config.adaptive_indicator),
        option_minute_ohlcv=asdict(config.option_minute_ohlcv),
        minute_lookback=asdict(config.minute_lookback),  # 피처 계산/LLM 컨텍스트용
        # ── 기타 ────────────────────────────────────────────────────────
        config_path=str(getattr(args, "config", "config.json") or "config.json"),
        notifier=notifier,
    )


