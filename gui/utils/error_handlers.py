"""에러 핸들러 모듈

finplot/PyQtGraph 관련 오류를 필터링합니다.
"""

from __future__ import annotations

import logging
import sys
import traceback
from typing import TYPE_CHECKING, Optional, Callable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FinplotTypeErrorHandler:
    """finplot의 Timestamp 비교 TypeError를 무시"""
    def __init__(self):
        self.original_excepthook = sys.excepthook

    def handle(self, exc_type, exc_value, exc_traceback):
        message = str(exc_value)
        # finplot/PyQtGraph 관련 오류 무시
        if exc_type == TypeError and (">=" in message and "Timestamp" in message):
            return
        # paintEvent 오류 무시 (PyQtGraph 내부 문제)
        if "paintEvent" in message or "QGraphicsView" in message:
            return
        # IndexError 무시 (finplot _pdtime2index - 데이터 부족 시)
        if exc_type == IndexError and "is out of bounds for axis" in message:
            return
        self.original_excepthook(exc_type, exc_value, exc_traceback)


handler = FinplotTypeErrorHandler()


class _StderrFilter:
    """finplot/PyQtGraph 관련 traceback 필터링"""
    def __init__(self):
        self.original_stderr = sys.stderr

    def write(self, text):
        # finplot/PyQtGraph 관련 traceback 무시
        if "is out of bounds for axis" in text:
            return
        self.original_stderr.write(text)

    def flush(self):
        self.original_stderr.flush()

    def __getattr__(self, name: str):
        # 누락된 속성은 original_stderr 에 위임 (fileno, isatty 등)
        # object.__getattribute__ 사용으로 무한 재귀 방지
        return getattr(object.__getattribute__(self, "original_stderr"), name)


# Qt 스레드 예외 핸들러 (PySide6)
try:
    from PySide6.QtCore import QCoreApplication, qInstallMessageHandler
    from PySide6.QtCore import QtMsgType, QMessageLogContext

    def qt_message_handler(mode, context, message):
        # paintEvent 관련 오류 무시
        if "paintEvent" in message or "QGraphicsView" in message:
            return
        # finplot IndexError 무시
        if "is out of bounds for axis" in message:
            return
        if "UnboundLocalError" in message and "painter" in message:
            return
        if mode == QtMsgType.QtCriticalMsg or mode == QtMsgType.QtFatalMsg:
            logger.error("[Qt] %s (file=%s, line=%d, function=%s)",
                        message, context.file, context.line, context.function)
        elif mode == QtMsgType.QtWarningMsg:
            logger.warning("[Qt] %s", message)
        else:
            logger.debug("[Qt] %s", message)

    # QCoreApplication이 초기화된 후에 설치해야 함
    # 여기서는 미리 함수만 정의하고, attach_chart_viewer에서 설치
    _qt_message_handler_func: Optional[Callable] = qt_message_handler
except Exception:
    _qt_message_handler_func = None
    logger.debug("[error_handlers] Qt 메시지 핸들러 설정 실패 (PySide6 없음)")
