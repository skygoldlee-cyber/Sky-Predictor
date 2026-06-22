"""
메인 엔트리 포인트.

분리된 모듈:
    cli_args.py         : parse_arguments()
    app_setup.py        : _make_args_from_gui(), _setup_logging(), display_startup_info()
    pipeline_builder.py : _build_pipeline()
    run_modes.py        : run_test/replay/live/simple_mode()
    gui/controller.py   : GuiController (_run_gui 분리)
"""

import json
import logging
import os
import sys

from core.logging_utils import setup_logging
from importlib.metadata import version, PackageNotFoundError

try:
    VERSION = version("skypredictor")
except PackageNotFoundError:
    VERSION = "1.0.0"

APP_NAME = "SkyPredictor"

# ── 분리된 모듈 import ─────────────────────────────────────────────────────
from core.cli_args import parse_arguments


# ── ZZ 로그 필터 ──────────────────────────────────────────────────────────
class ZZLogFilter(logging.Filter):
    """[ZZ]로 시작하는 로그를 필터링하여 장마감 시 불필요한 반복 방지"""
    def filter(self, record):
        # [ZZ]로 시작하는 로그는 INFO 레벨 이하에서만 필터링
        if record.getMessage().startswith('[ZZ]') and record.levelno < logging.WARNING:
            return False
        return True

# 루트 로거에 ZZ 필터 적용 (전역 교체 대신)
_apply_zz_filter = False
def apply_zz_filter_to_root():
    global _apply_zz_filter
    if _apply_zz_filter:
        return
    _apply_zz_filter = True
    zz_filter = ZZLogFilter()
    # 루트 로거에 필터 적용
    logging.root.addFilter(zz_filter)



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

    # ZZ 필터 적용
    apply_zz_filter_to_root()

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
        import traceback
        logger.error(f"[main] exception occurred: {e}")
        logger.error(f"[main] traceback:\n{traceback.format_exc()}")
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
