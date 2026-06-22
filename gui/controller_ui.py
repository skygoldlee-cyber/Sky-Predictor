"""GUI 메인 윈도우 레이아웃·폼 헬퍼 (`gui_controller` 2·6·7단계 분리).

스플리터·좌/우 패널·입력 검증·effective 행·UI 초기화 오류 기록.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

__all__ = [
    "MainWindowShell",
    "build_main_window_shell",
    "set_int_validator",
    "set_float_validator",
    "make_effective_row",
    "make_effective_row_widget",
    "record_gui_init_error",
]


@dataclass
class MainWindowShell:
    """좌측 스크롤(폼) + 우측 패널을 담은 스플리터 루트."""

    window: Any
    outer_root: Any
    splitter: Any
    left_scroll: Any
    left_content: Any
    left_root: Any
    right_panel: Any
    right_root: Any


def build_main_window_shell(
    *,
    base_dir: Path,
    window_title: str,
    settings: Optional[Any] = None,
    min_width: int = 1100,
    min_height: int = 780,
    left_panel_ratio: float = 0.45,
) -> MainWindowShell:
    """메인 QWidget, 스플리터, 좌·우 레이아웃을 생성하고 저장된 분할 상태를 복원한다."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QScrollArea, QSplitter, QVBoxLayout, QWidget

    w = QWidget()
    w.setWindowTitle(window_title)

    try:
        icon_path = base_dir / "assets" / "beacon.ico"
        if icon_path.exists():
            ic = QIcon(str(icon_path))
            if not ic.isNull():
                w.setWindowIcon(ic)
    except Exception:
        pass

    try:
        w.setMinimumSize(int(min_width), int(min_height))
    except Exception:
        pass

    outer_root = QVBoxLayout(w)
    outer_root.setContentsMargins(10, 10, 10, 10)

    splitter = QSplitter(Qt.Horizontal)
    outer_root.addWidget(splitter, stretch=1)

    left_scroll = QScrollArea(w)
    left_scroll.setWidgetResizable(True)
    try:
        left_scroll.setFrameShape(QScrollArea.NoFrame)
    except Exception:
        pass
    splitter.addWidget(left_scroll)

    left_content = QWidget()
    left_scroll.setWidget(left_content)
    left_root = QVBoxLayout(left_content)

    right_panel = QWidget()
    right_root = QVBoxLayout(right_panel)
    splitter.addWidget(right_panel)

    try:
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
    except Exception:
        pass

    try:
        if settings is not None:
            st = settings.value("gui/splitter_state")
            if st is not None:
                restored_splitter = False
                try:
                    restored_splitter = bool(splitter.restoreState(st))
                except Exception:
                    try:
                        restored_splitter = bool(splitter.restoreState(bytes(st)))
                    except Exception:
                        restored_splitter = False

                if not restored_splitter:
                    try:
                        base_w = max(int(w.width() or 0), int(w.minimumWidth() or 0), int(min_width))
                        left_w = int(base_w * float(left_panel_ratio))
                        right_w = max(1, int(base_w) - int(left_w))
                        splitter.setSizes([left_w, right_w])
                    except Exception:
                        pass
            else:
                try:
                    base_w = max(int(w.width() or 0), int(w.minimumWidth() or 0), int(min_width))
                    left_w = int(base_w * float(left_panel_ratio))
                    right_w = max(1, int(base_w) - int(left_w))
                    splitter.setSizes([left_w, right_w])
                except Exception:
                    pass
        else:
            try:
                base_w = max(int(w.width() or 0), int(w.minimumWidth() or 0), int(min_width))
                left_w = int(base_w * float(left_panel_ratio))
                right_w = max(1, int(base_w) - int(left_w))
                splitter.setSizes([left_w, right_w])
            except Exception:
                pass
    except Exception:
        pass

    return MainWindowShell(
        window=w,
        outer_root=outer_root,
        splitter=splitter,
        left_scroll=left_scroll,
        left_content=left_content,
        left_root=left_root,
        right_panel=right_panel,
        right_root=right_root,
    )


def set_int_validator(edit: Any, *, min_v: int = 0, max_v: int = 2147483647) -> None:
    try:
        if edit is None:
            return
        from PySide6.QtGui import QIntValidator

        v = QIntValidator(int(min_v), int(max_v))
        edit.setValidator(v)
    except Exception:
        pass


def set_float_validator(
    edit: Any,
    *,
    min_v: float = -1.0e308,
    max_v: float = 1.0e308,
    decimals: int = 8,
) -> None:
    try:
        if edit is None:
            return
        from PySide6.QtGui import QDoubleValidator

        v = QDoubleValidator(float(min_v), float(max_v), int(decimals))
        try:
            v.setNotation(QDoubleValidator.StandardNotation)
        except Exception:
            pass
        edit.setValidator(v)
    except Exception:
        pass


def make_effective_row(edit: Any) -> tuple[Any, Any]:
    """QLineEdit + 회색 effective 라벨을 한 줄로 묶은 QWidget 반환."""
    from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

    eff = QLabel("")
    try:
        eff.setStyleSheet("color: #9E9E9E;")
    except Exception:
        pass
    try:
        edit.setMinimumWidth(160)
        edit.setMaximumWidth(160)
    except Exception:
        pass
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(edit)
    row.addSpacing(8)
    row.addWidget(eff)
    wrap = QWidget()
    wrap.setLayout(row)
    return wrap, eff


def make_effective_row_widget(widget: Any) -> tuple[Any, Any]:
    """콤보 등 위젯 + effective 라벨 행."""
    from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

    eff = QLabel("")
    try:
        eff.setStyleSheet("color: #9E9E9E;")
    except Exception:
        pass
    try:
        widget.setMinimumWidth(160)
    except Exception:
        pass
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(widget)
    row.addSpacing(8)
    row.addWidget(eff)
    wrap = QWidget()
    wrap.setLayout(row)
    return wrap, eff


def record_gui_init_error(
    log: Any,
    errors: List[str],
    section: str,
    exc: BaseException,
) -> None:
    """폼 구성 중 예외를 로그·누적 문자열 리스트에 남긴다 (7단계: run() 클로저 축소)."""
    try:
        log.exception("[GUI_INIT_ERROR] section=%s", str(section))
    except Exception:
        pass
    try:
        errors.append(f"{str(section)}: {type(exc).__name__}: {str(exc)}")
    except Exception:
        pass
