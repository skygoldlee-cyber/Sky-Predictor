"""애플리케이션 초기화 유틸리티.

main.py에서 분리된 초기화/설정 관련 함수:
- _make_args_from_gui()  : GUI 설정 → argparse.Namespace 변환
- _setup_logging()       : 로거 초기화
- display_startup_info() : 시작 정보 출력
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import AppConfig, log_ai_provider_keys_loaded, DEFAULT_LOG_FILE
from core.logging_utils import setup_logging, get_logger
from core.utils import get_expiry_week_info, get_option_expiry_date
from importlib.metadata import version, PackageNotFoundError

try:
    VERSION = version("skypredictor")
except PackageNotFoundError:
    VERSION = "1.0.0"

APP_NAME = "SkyPredictor"


def load_recommended_params(symbol: str = "KP200 선물", regime: str = "unknown") -> dict:
    """DB에서 추천 파라미터를 로드해 config에 반영."""
    try:
        from prediction.pivot_parameter_db import PivotParameterDB, ParameterRecommender
        
        db = PivotParameterDB("data/pivot_parameters.db")
        recommender = ParameterRecommender(db)
        params = recommender.recommend(symbol=symbol, regime=regime)
        
        logger = get_logger(__name__)
        logger.info(
            "[STARTUP] 추천 파라미터 로드: regime=%s source=%s score=%.3f",
            regime, params.get("_source"), params.get("_composite_score", 0.0)
        )
        
        db.close()
        return params
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning("[STARTUP] 파라미터 추천 실패, 기본값 사용: %s", e)
        return {}


def _make_args_from_gui(
    *,
    config_path: str,
    log_level: str,
    log_file: str,
    prediction_minutes: Optional[int],
    heuristic_only: bool,
    no_ebest_live: bool,
    duration_sec: int,
    include_options: bool,
    option_month: Optional[str],
    replay_speed: float = 0.0,
    replay_max_lines: Optional[int] = None,
) -> argparse.Namespace:
    """GUI 실행 시 argparse.Namespace를 생성한다.

    _build_pipeline()이 참조하는 모든 args 필드를 포함해야 한다.
    새 파라미터를 _build_pipeline에 추가할 때 이 함수도 함께 업데이트할 것.
    """
    return argparse.Namespace(
        cli=False,
        config=str(config_path or "config.json"),
        log_level=str(log_level or "INFO"),
        log_file=str(log_file or DEFAULT_LOG_FILE),
        prediction_minutes=prediction_minutes,
        buy_threshold=None,
        sell_threshold=None,
        numeric_predictor=None,
        transformer_weight=None,
        tft_weights_path=None,
        tft_horizon=None,
        disagreement_hold=None,
        heuristic_only=bool(heuristic_only),
        days_to_expiry=None,
        seq_len=None,
        fo0_stale_sec=None,
        fo0_log_schema=None,
        preferred_provider=None,
        dual_llm=None,
        dual_llm_primary_provider=None,
        dump_llm_prompt=True,
        test=False,
        replay=None,
        replay_speed=float(replay_speed),
        replay_max_lines=replay_max_lines,
        no_ebest_live=bool(no_ebest_live),
        duration_sec=int(duration_sec),
        include_options=bool(include_options),
        option_month=(str(option_month).strip() or None) if option_month is not None else None,
        out_ticks=None,
        no_save_ticks=False,
        compress_ticks=True,
        show_metrics=False,
        json_output=False,
        tee=True,
    )


# ── ARC-06: 책임 분리 헬퍼 ──────────────────────────────────────────────────

def _setup_logging(args: argparse.Namespace) -> logging.Logger:
    """로깅 시스템을 초기화하고 루트 로거를 반환한다.

    ARC-06: main()에서 로깅 초기화 책임을 분리.
    """
    from core.logging_utils import setup_logging
    log_level_str = str(getattr(args, "log_level", "INFO") or "INFO").upper()
    level = getattr(logging, log_level_str, logging.INFO)
    log_file = str(getattr(args, "log_file", DEFAULT_LOG_FILE) or DEFAULT_LOG_FILE)
    enable_tee = bool(getattr(args, "tee", True))
    try:
        return setup_logging(log_file=log_file, level=level, enable_tee=enable_tee)
    except Exception as e:
        logging.basicConfig(level=level)
        root = logging.getLogger()
        root.warning("로깅 초기화 실패 (기본 설정 사용): %s", e)
        return root




def display_startup_info(config: AppConfig, args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    시작 정보 표시
    
    Args:
        config: 설정
        args: 인자
        logger: 로거
    """
    logger.info("=" * 70)
    logger.info("%s v%s", APP_NAME, VERSION)
    logger.info("=" * 70)
    logger.info("Config file: %s", args.config)
    logger.info("Log level: %s", args.log_level)
    logger.info("Prediction minutes: %s", config.prediction.minutes)
    logger.info("Use LLM: %s", config.prediction.use_llm and not args.heuristic_only)
    try:
        log_ai_provider_keys_loaded(config.ai_providers, log_to=logger)
    except Exception as _e:
        logger.debug("오류 무시: %s", _e)

    # 만기 정보
    try:
        now = datetime.now()
        expiry_info = get_expiry_week_info(now)
        expiry_dt = get_option_expiry_date(now.year, now.month)
        is_expiry_day = now.date() == expiry_dt.date()
        
        logger.info("-" * 70)
        logger.info("Expiry Information:")
        logger.info("  Is expiry week: %s", expiry_info['is_expiry_week'])
        logger.info("  Is expiry day: %s", is_expiry_day)
        logger.info("  Expiry date: %s", expiry_info['expiry_second_thursday'])
        logger.info("  Days to expiry: %s", expiry_info['days_to_expiry'])

        # Rollover reset marker (for training window reset)
        try:
            from pathlib import Path
            from datetime import timedelta

            def _marker_for_expiry_cycle(expiry_dt: datetime) -> Path:
                try:
                    yyyymm = expiry_dt.strftime("%Y%m")
                    return Path(f".rollover_start_{yyyymm}.txt")
                except Exception:
                    return Path(".rollover_start.txt")

            marker_path = _marker_for_expiry_cycle(expiry_dt)
            marker_val = ""
            if marker_path.exists() and marker_path.is_file():
                marker_val = (marker_path.read_text(encoding="utf-8") or "").strip()
            after_expiry = now.date() > expiry_dt.date()
            logger.info("  After expiry date: %s", after_expiry)
            logger.info("  Rollover marker (first trading day after expiry): %s", marker_val or '(none)')
        except Exception as _e:
            logger.debug("[_marker_for_expiry_cycle] 오류 무시: %s", _e)
    except Exception as e:
        logger.warning("Failed to get expiry info: %s", e)
    
    logger.info("=" * 70)


