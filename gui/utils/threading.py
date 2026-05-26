"""스레딩 유틸리티 모듈"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Callable

if TYPE_CHECKING:
    pass

try:
    from PySide6.QtCore import QObject, Signal, QThread, Slot
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False
    QObject = object
    Signal = None
    QThread = object
    Slot = lambda x: x

logger = logging.getLogger(__name__)


class DataComputeThread(QThread):
    """데이터 컴퓨팅을 백그라운드 스레드에서 수행하는 Thread."""
    
    # 시그널: df는 pd.DataFrame, pm은 Optional[Dict[str, Any]]
    finished = Signal(object, object, bool)  # (df: pd.DataFrame, pm: Optional[Dict[str, Any]], force_clear: bool)
    error = Signal(str, bool)  # (error_message: str, force_clear: bool)
    
    def __init__(self, compute_func: Callable, force_clear: bool):
        super().__init__()
        self._compute_func = compute_func
        self._force_clear = force_clear
        self._result_df = None
        self._result_pm = None
        self._stop_requested = False  # 취소 요청 플래그
    
    def request_stop(self) -> None:
        """스레드 취소 요청."""
        self._stop_requested = True
    
    def run(self) -> None:
        """백그라운드 스레드에서 데이터 컴퓨팅 실행."""
        compute_start_time = time.time()

        try:
            logger.info("[DataComputeThread] 데이터 컴퓨팅 시작")
            df, pm = self._compute_func()
            compute_elapsed = time.time() - compute_start_time

            # 데이터 컴퓨팅 시간에 따른 로그 레벨 결정
            if compute_elapsed >= 1.0:  # 1초 이상: 경고
                logger.warning(
                    "[DataComputeThread] 데이터 컴퓨팅 시간이 느립니다: "
                    "%.3f초 | df.shape=%s | force_clear=%s",
                    compute_elapsed, df.shape if hasattr(df, 'shape') else 'N/A', self._force_clear
                )
            elif compute_elapsed >= 0.5:  # 500ms-1초: 정보
                logger.info(
                    "[DataComputeThread] 데이터 컴퓨팅 시간: "
                    "%.3f초 | df.shape=%s | force_clear=%s",
                    compute_elapsed, df.shape if hasattr(df, 'shape') else 'N/A', self._force_clear
                )
            else:  # 500ms 미만: 디버그
                logger.debug(
                    "[DataComputeThread] 데이터 컴퓨팅 완료 (df.shape=%s, elapsed=%.3f초)",
                    df.shape if hasattr(df, 'shape') else 'N/A', compute_elapsed
                )

            self._result_df = df
            self._result_pm = pm
            self.finished.emit(df, pm, self._force_clear)
        except Exception as e:
            compute_elapsed = time.time() - compute_start_time
            logger.error(
                "[DataComputeThread] 컴퓨팅 오류: %s (elapsed=%.3f초)",
                e, compute_elapsed, exc_info=True
            )
            self.error.emit(str(e), self._force_clear)
