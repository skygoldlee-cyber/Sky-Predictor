"""
로깅 시스템 모듈

개선사항:
- 구조화된 로깅 설정
- 파일 크기 제한 및 로테이션
- stdout/stderr tee 기능
- 컨텍스트 기반 로거
"""

import io
import atexit
import contextlib
import logging
import sys
import os
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

import re


class TeeStream(io.TextIOBase):
    """
    stdout/stderr를 콘솔과 파일에 동시 출력하는 스트림
    """
    
    def __init__(self, original: io.TextIOBase, file_handler: RotatingFileHandler):
        """
        Args:
            original: 원본 스트림 (stdout/stderr)
            file_handler: 파일 핸들러
        """
        self._original = original
        self._file_handler = file_handler
        self._buffer = ""
        self._ts_re = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
        # [IMP-6-4] 멀티스레드에서 동시 write 시 _buffer 손상 방지
        import threading as _threading
        self._lock = _threading.Lock()

    @property
    def encoding(self) -> str:
        """Return stream encoding (falls back to utf-8 when unknown)."""
        return getattr(self._original, "encoding", "utf-8")

    def write(self, text: str) -> int:
        """텍스트 쓰기"""
        if text is None:
            return 0
        
        text = str(text)
        written = len(text)

        def _prefix_if_needed(line: str) -> str:
            line_stripped = line.lstrip("\r\n")
            if self._ts_re.match(line_stripped):
                return line
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            return f"{ts} - {line}"

        # [IMP-6-4] _lock으로 _buffer 접근을 직렬화하여 멀티스레드 안전성 확보
        with self._lock:
            self._buffer += text
            try:
                while "\n" in self._buffer:
                    line_end = self._buffer.find("\n")
                    line = self._buffer[: line_end + 1]
                    self._buffer = self._buffer[line_end + 1 :]

                    out_line = _prefix_if_needed(line)
                    try:
                        self._original.write(out_line)
                    except Exception as _e:
                        print(f"[logging_utils._prefix_if_needed] 오류 무시: {_e}")

                    try:
                        stream = getattr(self._file_handler, "stream", None)
                        if stream is not None:
                            stream.write(out_line)
                    except Exception as _e:
                        print(f"[logging_utils._prefix_if_needed] 오류 무시: {_e}")
            except Exception:
                # best-effort fallback
                try:
                    self._original.write(text)
                except Exception as _e:
                    print(f"[logging_utils.] 오류 무시: {_e}")
                try:
                    stream = getattr(self._file_handler, "stream", None)
                    if stream is not None:
                        stream.write(text)
                except Exception as _e:
                    print(f"[logging_utils.] 오류 무시: {_e}")

        return written

    def flush(self) -> None:
        """버퍼 플러시"""
        if getattr(self, "_buffer", ""):
            buf = self._buffer
            self._buffer = ""
            try:
                if self._ts_re.match(buf.lstrip("\r\n")):
                    out = buf
                else:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    out = f"{ts} - {buf}"

                try:
                    self._original.write(out)
                except Exception as _e:
                    print(f"[logging_utils.flush] 오류 무시: {_e}")
                try:
                    stream = getattr(self._file_handler, "stream", None)
                    if stream is not None:
                        stream.write(out)
                except Exception as _e:
                    print(f"[logging_utils.flush] 오류 무시: {_e}")
            except Exception as _e:
                print(f"[logging_utils.flush] 오류 무시: {_e}")

        try:
            self._original.flush()
        except Exception as _e:
            print(f"[logging_utils.flush] 오류 무시: {_e}")

        try:
            stream = getattr(self._file_handler, "stream", None)
            if stream is not None:
                stream.flush()
        except Exception as _e:
            print(f"[logging_utils.] 오류 무시: {_e}")


class StdIOTee:
    """Context manager that tees stdout/stderr and restores originals."""

    def __init__(self, file_handler: RotatingFileHandler):
        self._file_handler = file_handler
        self._orig_stdout: Optional[io.TextIOBase] = None
        self._orig_stderr: Optional[io.TextIOBase] = None
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = TeeStream(sys.stdout, self._file_handler)  # type: ignore
        sys.stderr = TeeStream(sys.stderr, self._file_handler)  # type: ignore
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        try:
            if self._orig_stdout is not None:
                sys.stdout = self._orig_stdout  # type: ignore
            if self._orig_stderr is not None:
                sys.stderr = self._orig_stderr  # type: ignore
        finally:
            self._installed = False

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.restore()
        return False


def setup_logging(
    log_file: str = "app.log",
    level: int = logging.INFO,
    max_bytes: int = 0,
    backup_count: int = 0,
    enable_tee: bool = True,
    *,
    # ── 하위 호환 별칭 (잘못된 kwarg 유입 방지) ───────────────────────────────
    log_level: "Optional[Any]" = None,   # level 별칭
    tee: "Optional[bool]" = None,        # enable_tee 별칭
) -> logging.Logger:
    """
    로깅 시스템 설정

    Args:
        log_file:     로그 파일 경로 (기본 "app.log")
        level:        로그 레벨 (int 또는 "INFO"/"DEBUG" 문자열)
        max_bytes:    파일 최대 크기 (바이트, 0=무제한)
        backup_count: 백업 파일 개수
        enable_tee:   stdout/stderr tee 활성화 여부
        log_level:    level 의 별칭 (구 호출부 호환)
        tee:          enable_tee 의 별칭 (구 호출부 호환)

    Returns:
        설정된 로거 ("kp200_predictor")

    Example:
        >>> logger = setup_logging("prediction.log", logging.DEBUG)
        >>> logger.info("Application started")
    """
    # 별칭 처리 (구 호출부 호환)
    if log_level is not None:
        if isinstance(log_level, str):
            level = getattr(logging, log_level.upper(), logging.INFO)
        else:
            try:
                level = int(log_level)
            except Exception as _e:
                print(f"[logging_utils.] 오류 무시: {_e}")
    if tee is not None:
        enable_tee = bool(tee)
    # level이 문자열로 들어온 경우도 처리
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    # 로거 생성
    logger = logging.getLogger("kp200_predictor")
    logger.setLevel(level)

    # Prevent duplicate emission via root logger handlers.
    logger.propagate = False
    
    # 기존 핸들러 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()
    
    # 파일 핸들러
    file_handler = RotatingFileHandler(
        log_file,
        mode="w",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    
    # 콘솔 핸들러 (stdout으로 변경하여 터미널 출력)
    console_stream = getattr(sys, "__stdout__", None) or sys.stdout
    console_handler = logging.StreamHandler(stream=console_stream)
    console_handler.setLevel(level)
    
    # 포맷터
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Route logs from other modules (logging.getLogger(__name__)) through root handlers.
    # Keep `kp200_predictor` itself non-propagating to avoid duplicate emission.
    try:
        root = logging.getLogger()
        root.setLevel(level)
        # 중복 핸들러 방지를 위해 이미 핸들러가 있는지 확인
        if not root.handlers:
            root.addHandler(file_handler)
            root.addHandler(console_handler)
    except Exception as _e:
        print(f"[logging_utils.] 오류 무시: {_e}")

    # ── 외부 SDK 로거 억제 ─────────────────────────────────────────────────────
    # openai / google-genai SDK 의 httpx·AFC·retry INFO 로그가 root handler 를
    # 타고 파일/콘솔에 쏟아지는 것을 WARNING 이상으로 제한한다.
    _NOISY_LOGGERS = [
        "openai",
        "openai._base_client",
        "openai.http_client",
        "httpx",
        "httpcore",
        "httpcore.http11",
        "httpcore.connection",
        "google",
        "google.genai",
        "google.auth",
        "google.api_core",
        "google.api_core.retry",
        "anthropic",
        "anthropic._base_client",
        "urllib3",
        "urllib3.connectionpool",
    ]
    for _name in _NOISY_LOGGERS:
        try:
            logging.getLogger(_name).setLevel(logging.WARNING)
        except Exception as _e:
            print(f"[logging_utils.] 오류 무시: {_e}")
    # ──────────────────────────────────────────────────────────────────────────
    
    # 파일 핸들러를 속성으로 저장 (tee에서 사용)
    setattr(logger, "_file_handler", file_handler)
    
    # stdout/stderr tee 설정
    if enable_tee:
        install_stdout_stderr_tee(logger)

    try:
        install_watchdog_hooks(log_file=str(log_file), level=logging.WARNING)
    except Exception as _e:
        print(f"[logging_utils.] 오류 무시: {_e}")
    
    return logger


def install_stdout_stderr_tee(logger: logging.Logger) -> None:
    """
    stdout/stderr를 콘솔과 파일에 동시 출력하도록 설정
    
    Args:
        logger: 로거 (파일 핸들러 필요)
        
    Note:
        - print() 출력도 로그 파일에 기록됨
        - 에러 메시지도 자동으로 파일에 저장됨
    """
    file_handler = getattr(logger, "_file_handler", None)
    
    if file_handler is None:
        logging.warning("No file handler found, tee not installed")
        return
    
    if not isinstance(file_handler, RotatingFileHandler):
        logging.warning("File handler is not RotatingFileHandler, tee not installed")
        return

    # If a tee was previously installed on this logger, restore it first.
    try:
        prev = getattr(logger, "_stdio_tee", None)
        if isinstance(prev, StdIOTee):
            prev.restore()
    except Exception as _e:
        print(f"[logging_utils.] 오류 무시: {_e}")
    
    # 파일 스트림이 열려있는지 확인
    try:
        if getattr(file_handler, "stream", None) is None:
            file_handler.stream = file_handler._open()
    except Exception as e:
        logging.error("Failed to open file handler stream: %s", e)
        return

    tee = StdIOTee(file_handler)
    tee.install()
    setattr(logger, "_stdio_tee", tee)
    try:
        atexit.register(tee.restore)
    except Exception as _e:
        print(f"[logging_utils.] 오류 무시: {_e}")


def uninstall_stdout_stderr_tee(logger: logging.Logger) -> None:
    """Best-effort restore of stdout/stderr if previously installed via `install_stdout_stderr_tee`."""
    tee = getattr(logger, "_stdio_tee", None)
    if isinstance(tee, StdIOTee):
        tee.restore()


@contextlib.contextmanager
def stdout_stderr_tee(logger: logging.Logger):
    """Context manager variant of stdout/stderr tee.

    This avoids a process-global replacement beyond the `with` scope.
    """
    file_handler = getattr(logger, "_file_handler", None)
    if not isinstance(file_handler, RotatingFileHandler):
        yield
        return

    tee = StdIOTee(file_handler)
    try:
        tee.install()
        yield
    finally:
        tee.restore()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    로거 가져오기
    
    Args:
        name: 로거 이름 (None이면 기본 로거)
        
    Returns:
        로거
    """
    if name is None:
        return logging.getLogger("kp200_predictor")
    return logging.getLogger(name)


_WATCHDOG_INSTALLED = False


def _get_watchdog_logger(log_dir: str, level: int = logging.WARNING) -> logging.Logger:
    logger = logging.getLogger("watchdog")
    logger.setLevel(int(level))
    logger.propagate = False

    try:
        if getattr(logger, "_watchdog_ready", False):
            return logger
    except Exception as _e:
        print(f"[logging_utils.] 오류 무시: {_e}")

    try:
        logger.handlers.clear()
    except Exception as _e:
        print(f"[logging_utils.] 오류 무시: {_e}")

    try:
        os.makedirs(str(log_dir), exist_ok=True)
    except Exception as _e:
        print(f"[logging_utils.] 오류 무시: {_e}")

    path = None
    try:
        path = os.path.join(str(log_dir), "watchdog.log")
    except Exception:
        path = "watchdog.log"

    try:
        setattr(logger, "_watchdog_ready", True)
        setattr(logger, "_watchdog_path", str(path))
        setattr(logger, "_watchdog_log_dir", str(log_dir))
        setattr(logger, "_watchdog_level", int(level))
    except Exception as _e:
        print(f"[logging_utils.] 오류 무시: {_e}")
    return logger


def _ensure_watchdog_handler(logger: logging.Logger) -> None:
    """첫 로그 기록 시 핸들러 생성"""
    try:
        if getattr(logger, "_watchdog_handler_installed", False):
            return
    except Exception:
        return

    try:
        logger.handlers.clear()
    except Exception:
        pass

    path = getattr(logger, "_watchdog_path", None)
    log_dir = getattr(logger, "_watchdog_log_dir", ".")
    level = getattr(logger, "_watchdog_level", logging.WARNING)

    if not path:
        return

    try:
        fh = RotatingFileHandler(
            str(path),
            mode="a",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(int(level))
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
            )
        )
        logger.addHandler(fh)
        setattr(logger, "_watchdog_handler_installed", True)
    except Exception as _e:
        print(f"[logging_utils.] 오류 무시: {_e}")


def _cleanup_empty_watchdog_log(logger: logging.Logger) -> None:
    """watchdog.log가 비어있으면 삭제"""
    try:
        path = getattr(logger, "_watchdog_path", None)
        if path and os.path.exists(path):
            stat = os.stat(path)
            if stat.st_size == 0:
                os.remove(path)
    except Exception:
        pass


def install_watchdog_hooks(*, log_file: Optional[str] = None, level: int = logging.WARNING) -> None:
    global _WATCHDOG_INSTALLED
    if _WATCHDOG_INSTALLED:
        return

    log_dir = "."
    try:
        if log_file:
            log_dir = os.path.dirname(os.path.abspath(str(log_file))) or "."
    except Exception:
        log_dir = "."

    wd = _get_watchdog_logger(log_dir, level=level)

    prev_sys_hook = getattr(sys, "excepthook", None)

    def _sys_excepthook(exc_type, exc, tb):
        try:
            if exc_type is KeyboardInterrupt:
                if callable(prev_sys_hook):
                    return prev_sys_hook(exc_type, exc, tb)
                return
        except Exception as _e:
            print(f"[logging_utils._sys_excepthook] 오류 무시: {_e}")
        try:
            _ensure_watchdog_handler(wd)
            wd.error(
                "Unhandled exception (sys.excepthook): %s",
                "".join(traceback.format_exception(exc_type, exc, tb)).strip(),
            )
            _cleanup_empty_watchdog_log(wd)
        except Exception as _e:
            print(f"[logging_utils._sys_excepthook] 오류 무시: {_e}")
        try:
            if callable(prev_sys_hook):
                return prev_sys_hook(exc_type, exc, tb)
        except Exception as _e:
            print(f"[logging_utils._sys_excepthook] 오류 무시: {_e}")

    try:
        sys.excepthook = _sys_excepthook
    except Exception as _e:
        print(f"[logging_utils._sys_excepthook] 오류 무시: {_e}")

    try:
        import threading

        prev_thread_hook = getattr(threading, "excepthook", None)

        def _thread_excepthook(args):
            try:
                _ensure_watchdog_handler(wd)
                wd.error(
                    "Unhandled thread exception: thread=%s exc=%s",
                    getattr(args, "thread", None),
                    "".join(
                        traceback.format_exception(
                            getattr(args, "exc_type", None),
                            getattr(args, "exc_value", None),
                            getattr(args, "exc_traceback", None),
                        )
                    ).strip(),
                )
                _cleanup_empty_watchdog_log(wd)
            except Exception as _e:
                print(f"[logging_utils._thread_excepthook] 오류 무시: {_e}")
            try:
                if callable(prev_thread_hook):
                    return prev_thread_hook(args)
            except Exception as _e:
                print(f"[logging_utils._thread_excepthook] 오류 무시: {_e}")

        threading.excepthook = _thread_excepthook  # type: ignore[attr-defined]
    except Exception as _e:
        print(f"[logging_utils._thread_excepthook] 오류 무시: {_e}")

    try:
        import asyncio

        prev_factory = asyncio.get_event_loop_policy().get_event_loop

        def _get_loop_with_handler():
            loop = prev_factory()
            try:
                prev_handler = loop.get_exception_handler()

                def _handler(loop_obj, context):
                    try:
                        _ensure_watchdog_handler(wd)
                        msg = str((context or {}).get("message") or "asyncio exception")
                        exc = (context or {}).get("exception")
                        if exc is not None:
                            tb_txt = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
                        else:
                            tb_txt = str(context)
                        wd.error("Unhandled asyncio exception: %s | %s", msg, tb_txt)
                        _cleanup_empty_watchdog_log(wd)
                    except Exception as _e:
                        print(f"[logging_utils._handler] 오류 무시: {_e}")
                    try:
                        if callable(prev_handler):
                            return prev_handler(loop_obj, context)
                    except Exception as _e:
                        print(f"[logging_utils._handler] 오류 무시: {_e}")

                loop.set_exception_handler(_handler)
            except Exception as _e:
                print(f"[logging_utils._handler] 오류 무시: {_e}")
            return loop

        try:
            asyncio.get_event_loop_policy().get_event_loop = _get_loop_with_handler  # type: ignore[assignment]
        except Exception as _e:
            print(f"[logging_utils._handler] 오류 무시: {_e}")
    except Exception as _e:
        print(f"[logging_utils._handler] 오류 무시: {_e}")

    _WATCHDOG_INSTALLED = True


class LogContext:
    """
    로그 컨텍스트 매니저
    
    Example:
        >>> with LogContext("Processing tick data"):
        ...     process_data()
        ... # 자동으로 시작/종료 로그 출력
    """
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
        """Create a logging context.

        Args:
            operation: Human-readable operation name.
            logger: Logger to use (defaults to project logger).
        """
        self.operation = operation
        self.logger = logger or get_logger()
        self.start_time = None

    def __enter__(self):
        """Log a start message and return `self`."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log completion/failure and propagate exceptions (return False)."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation} ({elapsed:.2f}s)")
        else:
            self.logger.error(
                f"Failed: {self.operation} ({elapsed:.2f}s) - {exc_type.__name__}: {exc_val}"
            )
        
        return False  # 예외를 다시 발생시킴
