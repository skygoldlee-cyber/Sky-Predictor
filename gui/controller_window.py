"""메인 창 기하·QSettings 연동 (`gui_controller` 5단계 분리)."""

from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "center_widget_on_screen",
    "apply_initial_window_geometry",
    "bind_save_gui_state_on_quit",
]


def center_widget_on_screen(w: Any) -> None:
    """윈도우를 커서가 있는 스크린의 가용 영역 중앙에 배치한다."""
    try:
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QApplication

        screen = None
        try:
            screen = QApplication.screenAt(QCursor.pos())
        except Exception:
            screen = None

        if screen is None:
            screen = w.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame = w.frameGeometry()
        frame.moveCenter(available.center())
        w.move(frame.topLeft())
    except Exception:
        pass


def apply_initial_window_geometry(
    w: Any,
    settings: Optional[Any],
    *,
    min_width: int = 1100,
    min_height: int = 780,
    default_width: int = 1100,
    default_height: int = 1200,
) -> bool:
    """최소 크기 설정 후 ``gui/window_geometry`` 복원. 실패 시 기본 ``resize``."""
    try:
        w.setMinimumSize(int(min_width), int(min_height))
    except Exception:
        pass

    restored_geom = False
    try:
        if settings is not None:
            geom = settings.value("gui/window_geometry")
            if geom is not None:
                try:
                    restored_geom = bool(w.restoreGeometry(geom))
                except Exception:
                    try:
                        restored_geom = bool(w.restoreGeometry(bytes(geom)))
                    except Exception:
                        restored_geom = False
    except Exception:
        restored_geom = False

    if not restored_geom:
        try:
            w.resize(int(default_width), int(default_height))
        except Exception:
            pass

    return bool(restored_geom)


def bind_save_gui_state_on_quit(
    app: Any,
    settings: Optional[Any],
    window: Any,
    splitter: Any,
) -> None:
    """종료 시 창 geometry·스플리터 상태를 QSettings에 저장."""
    if settings is None:
        return
    try:

        def _save_gui_state() -> None:
            try:
                settings.setValue("gui/window_geometry", window.saveGeometry())
            except Exception:
                pass
            try:
                settings.setValue("gui/splitter_state", splitter.saveState())
            except Exception:
                pass
            try:
                settings.sync()
            except Exception:
                pass

        app.aboutToQuit.connect(_save_gui_state)
    except Exception:
        pass
