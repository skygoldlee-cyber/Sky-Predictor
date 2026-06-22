"""Qt ↔ logging / stdout·stderr 라우팅.

`gui_controller` 분할 1단계: 로그 핸들러·표준출력 리다이렉트만 분리.
"""

from __future__ import annotations

import logging
from typing import Any


class QtLogEmitter:
    """Qt Signal을 통해 로그 레코드를 GUI로 전달하는 emitter."""

    def __init__(self) -> None:
        self._qt = None
        try:
            from PySide6.QtCore import QObject, Signal

            class _E(QObject):
                text = Signal(str)

            self._qt = _E()
        except Exception:
            self._qt = None

    @property
    def qt(self) -> Any:
        return self._qt


class QtLogHandler(logging.Handler):
    """QtLogEmitter를 통해 로그를 Qt Signal로 라우팅하는 Handler."""

    def __init__(self, emitter: Any) -> None:
        super().__init__()
        self._emitter = emitter
        # [FIX] ZZ 로그 필터 우회: QtLogHandler는 모든 로그를 GUI로 전달
        # ZZLogFilter가 로거 레벨에서 필터링하더라도, GUI에는 표시되도록 함
        self.addFilter(self._zz_bypass_filter)

    def _zz_bypass_filter(self, record: logging.LogRecord) -> bool:
        """ZZ 로그 필터 우회 - 항상 True를 반환하여 모든 로그 통과"""
        return True

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            return
        try:
            if self._emitter is not None and getattr(self._emitter, "qt", None) is not None:
                self._emitter.qt.text.emit(str(msg))
        except Exception:
            return


class QtStdIORedirect:
    """stdout/stderr를 Qt Signal로 넘겨 GUI 로그 뷰에 표시."""

    def __init__(self, emitter_obj: Any) -> None:
        self._emitter_obj = emitter_obj
        self._buf = ""

    def write(self, s: str) -> int:
        try:
            if s is None:
                return 0
            text = str(s)
            if not text:
                return 0
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.rstrip("\r")
                if line and getattr(self._emitter_obj, "qt", None) is not None:
                    self._emitter_obj.qt.text.emit(line)
            return len(text)
        except Exception:
            return 0

    def flush(self) -> None:
        try:
            if self._buf:
                line = self._buf.rstrip("\r\n")
                self._buf = ""
                if line and getattr(self._emitter_obj, "qt", None) is not None:
                    self._emitter_obj.qt.text.emit(line)
        except Exception:
            return

    def isatty(self) -> bool:
        return False
