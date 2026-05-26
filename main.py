"""
메인 엔트리 포인트.

분리된 모듈:
    cli_args.py         : parse_arguments()
    app_setup.py        : _make_args_from_gui(), _setup_logging(), display_startup_info()
    pipeline_builder.py : _build_pipeline()
    run_modes.py        : run_test/replay/live/simple_mode()
    gui/controller.py   : GuiController (_run_gui 분리)
"""

import argparse
import asyncio
import io
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from config import load_config, AppConfig, get_config_with_reload
from core.logging_utils import setup_logging
from telegram.notifier import create_notifier_from_config, PipelineTelegramBridge
from importlib.metadata import version, PackageNotFoundError

try:
    VERSION = version("skypredictor")
except PackageNotFoundError:
    VERSION = "1.0.0"

APP_NAME = "SkyPredictor"

# ── 분리된 모듈 import ─────────────────────────────────────────────────────
from core.cli_args import parse_arguments
from app.app_setup import display_startup_info
from app.pipeline_builder import _build_pipeline
from app.run_modes import (
    run_test_mode,
    run_replay_mode,
    run_replay_mode_with_predictor,
    run_live_mode,
    run_simple_prediction,
)


# ── ZZ 로그 필터 ──────────────────────────────────────────────────────────
class ZZLogFilter(logging.Filter):
    """[ZZ]로 시작하는 로그를 필터링하여 장마감 시 불필요한 반복 방지"""
    def filter(self, record):
        # [ZZ]로 시작하는 로그는 INFO 레벨 이하에서만 필터링
        if record.getMessage().startswith('[ZZ]') and record.levelno < logging.WARNING:
            return False
        return True

# 모든 로거에 ZZ 필터 적용
_apply_zz_filter = False
def apply_zz_filter_to_all_loggers():
    global _apply_zz_filter
    if _apply_zz_filter:
        return
    _apply_zz_filter = True
    zz_filter = ZZLogFilter()
    # 기존 로거에 필터 적용
    for name in logging.root.manager.loggerDict.keys():
        logger = logging.getLogger(name)
        logger.addFilter(zz_filter)
    # 루트 로거에도 필터 적용
    logging.getLogger().addFilter(zz_filter)

# 로거 생성 시 필터 자동 적용
original_getLogger = logging.getLogger
def getLogger_with_zz_filter(name=''):
    logger = original_getLogger(name)
    if not any(isinstance(f, ZZLogFilter) for f in logger.filters):
        logger.addFilter(ZZLogFilter())
    return logger

logging.getLogger = getLogger_with_zz_filter



def main() -> int:
    """
    메인 함수
    
    Returns:
        Exit code (0: 성공, 1: 실패)
    """
    # 로깅 설정 초기화
    setup_logging(log_file="logs/prediction.log", level=logging.INFO, enable_tee=True)
    logger = logging.getLogger(__name__)
    logger.info("[main] main() start")

    def _startup_internet_time_sync(logger: logging.Logger) -> None:
        try:
            from utils.internet_time_sync import sync_best_effort

            env_v = str(os.environ.get("INTERNET_TIME_SYNC") or "").strip().lower()
            # Default ON. Set INTERNET_TIME_SYNC=0/false/off to disable.
            do_sync = env_v not in ("0", "false", "no", "n", "off")

            try:
                logger.info(f"[TIME_SYNC] startup (enabled={str(bool(do_sync)).lower()} env={env_v!r})")
            except Exception:
                pass
            r = sync_best_effort(
                sync=bool(do_sync),
                samples=2,
                timeout=1.0,
                min_abs_offset_ms_to_sync=500.0,
            )
            try:
                best = r.get("best") if isinstance(r, dict) else None
                best = best if isinstance(best, dict) else {}

                ok = bool(r.get("ok")) if isinstance(r, dict) else False
                reason = str(r.get("reason")) if isinstance(r, dict) else "unknown"
                sync_attempted = bool(r.get("sync_attempted")) if isinstance(r, dict) else False

                server = str(best.get("server") or "")
                samples = best.get("samples")
                offset_ms_avg = best.get("offset_ms_avg")
                offset_ms_p95 = best.get("offset_ms_p95")

                parts = []
                if ok and sync_attempted:
                    parts.append("Internet time sync success")
                elif ok and not sync_attempted:
                    if reason == "measured_only":
                        parts.append("Internet time offset measured (sync not performed)")
                    elif reason == "skip_small_offset":
                        parts.append("Internet time offset too small, sync skipped")
                    else:
                        parts.append("Internet time processing complete (sync not performed)")
                else:
                    if sync_attempted:
                        parts.append("Internet time sync failed")
                    else:
                        parts.append("Internet time offset measurement failed")

                if server:
                    parts.append(f"server={server}")
                if samples is not None:
                    parts.append(f"samples={samples}")
                if offset_ms_avg is not None:
                    try:
                        parts.append(f"avg_offset={float(offset_ms_avg):.2f}ms(+ means system is slow)")
                    except Exception:
                        parts.append(f"avg_offset={offset_ms_avg}ms")
                if offset_ms_p95 is not None:
                    try:
                        parts.append(f"P95_offset={float(offset_ms_p95):.2f}ms")
                    except Exception:
                        parts.append(f"P95_offset={offset_ms_p95}ms")
                if (not ok) or (reason and reason not in ("measured_only", "skip_small_offset", "SetSystemTime_ok", "w32tm_resync_ok")):
                    parts.append(f"reason={reason}")

                logger.info(f"[TIME_SYNC] {' / '.join(parts)}")
            except Exception:
                logger.info(f"[TIME_SYNC] {r}")
        except Exception as e:
            try:
                logger.warning(f"[TIME_SYNC] failed: {e}")
            except Exception:
                pass

    from gui.controller import GuiController

    def _run_gui() -> None:
        logger.info("[main] GUI mode start")
        GuiController().run()
        logger.info("[main] GUI mode complete")

    try:
        args = parse_arguments()
        # Always run in GUI mode
        logger.info("[main] calling _run_gui()")
        return _run_gui()
    except Exception as e:
        logger.error(f"[main] exception occurred: {e}")
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
