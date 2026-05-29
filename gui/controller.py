"""GUI 컨트롤러 모듈.

main.py의 _run_gui() 함수를 GuiController 클래스로 분리.

사용법:
    from gui.controller import GuiController
    GuiController().run()
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from config import load_config, log_ai_provider_keys_loaded, VERSION, APP_NAME, DEFAULT_LOG_FILE
from core.logging_utils import setup_logging, get_logger
from core.utils import get_expiry_week_info
from app.pipeline_builder import _build_pipeline
from app.app_setup import _make_args_from_gui, display_startup_info, load_recommended_params
from app.run_modes import run_replay_mode_with_predictor, run_live_mode, run_simple_prediction
from telegram.notifier import create_notifier_from_config, PipelineTelegramBridge

from .qt_logging import QtLogEmitter, QtLogHandler, QtStdIORedirect
from .controller_ui import (
    build_main_window_shell,
    make_effective_row,
    make_effective_row_widget,
    record_gui_init_error,
    set_float_validator,
    set_int_validator,
)
from .controller_market import is_market_open, next_market_open
from .controller_window import (
    apply_initial_window_geometry,
    bind_save_gui_state_on_quit,
    center_widget_on_screen,
)
from .controller_logview import append_log_rich
from .controller_startup import run_startup_internet_time_sync
from .controller_rt_helpers import (
    fc0_is_stale,
    format_rt_status_line,
    open_replay_ticks_file_dialog,
    predictor_metrics_summary_strings,
)
from .controller_config_reload import merge_prediction_effective_from_loaded_config


logger = logging.getLogger(__name__)


class GuiController:
    """PySide6 GUI 애플리케이션 컨트롤러.

    기존 _run_gui() 함수에서 분리. task·리플레이 이벤트·effective 기본값은 self 멤버로 보관.

    Note:
        리팩토링 관련 TODO는 docs/TODO.md를 참조하세요.
    """

    def __init__(self) -> None:
        self.task: Optional[asyncio.Task] = None
        self.replay_pause_event: Optional[threading.Event] = None
        self.replay_paused: bool = False
        self.replay_stop_event: Optional[threading.Event] = None
        self.effective_pred_defaults: Dict[str, Any] = {}

    def _enter_gui_main_loop(
        self,
        *,
        app: Any,
        loop: Any,
        window: Any,
        splitter: Any,
        settings: Any,
    ) -> None:
        """창 geometry·종료 시 저장·표시 후 qasync 이벤트 루프 실행."""
        apply_initial_window_geometry(window, settings)
        bind_save_gui_state_on_quit(app, settings, window, splitter)
        window.show()
        try:
            from PySide6.QtCore import QTimer

            QTimer.singleShot(0, lambda: center_widget_on_screen(window))
        except Exception:
            center_widget_on_screen(window)
        with loop:
            loop.run_forever()
    
    # ── 레이아웃 제어 메서드 ────────────────────────────────────────────────────
    
    def save_layout(self, name: str = "default") -> bool:
        """현재 레이아웃 저장.
        
        Args:
            name: 레이아웃 이름 (기본: default)
        
        Returns:
            성공 여부
        """
        if self._layout_manager is None:
            logger.warning("[GuiController] 레이아웃 관리자 없음")
            return False
        return self._layout_manager.save_layout(name)
    
    def load_layout(self, name: str = "default") -> bool:
        """레이아웃 로드.
        
        Args:
            name: 레이아웃 이름 (기본: default)
        
        Returns:
            성공 여부
        """
        if self._layout_manager is None:
            logger.warning("[GuiController] 레이아웃 관리자 없음")
            return False
        return self._layout_manager.load_layout(name)
    
    def list_layouts(self) -> List[str]:
        """저장된 레이아웃 리스트.
        
        Returns:
            레이아웃 이름 리스트
        """
        if self._layout_manager is None:
            logger.warning("[GuiController] 레이아웃 관리자 없음")
            return []
        return self._layout_manager.list_layouts()
    
    def delete_layout(self, name: str) -> bool:
        """레이아웃 삭제.
        
        Args:
            name: 레이아웃 이름
        
        Returns:
            성공 여부
        """
        if self._layout_manager is None:
            logger.warning("[GuiController] 레이아웃 관리자 없음")
            return False
        return self._layout_manager.delete_layout(name)
    
    def move_tab(self, from_index: int, to_index: int):
        """탭 이동.
        
        Args:
            from_index: 이동할 탭 인덱스
            to_index: 이동할 위치 인덱스
        """
        if self._layout_manager is None:
            logger.warning("[GuiController] 레이아웃 관리자 없음")
            return
        self._layout_manager.move_tab(from_index, to_index)
    
    def toggle_tab_visibility(self, index: int):
        """탭 표시/숨기기 토글.
        
        Args:
            index: 탭 인덱스
        """
        if self._layout_manager is None:
            logger.warning("[GuiController] 레이아웃 관리자 없음")
            return
        self._layout_manager.toggle_tab_visibility(index)
    
    def _on_save_layout(self):
        """레이아웃 저장 버튼 핸들러."""
        try:
            from PySide6.QtWidgets import QInputDialog
            name, ok = QInputDialog.getText(None, "레이아웃 저장", "레이아웃 이름:", text="default")
            if ok and name:
                if self.save_layout(name):
                    logger.info("[GuiController] 레이아웃 저장 완료: %s", name)
                else:
                    logger.warning("[GuiController] 레이아웃 저장 실패: %s", name)
        except Exception as e:
            logger.error("[GuiController] 레이아웃 저장 실패: %s", e)
    
    def _on_load_layout(self):
        """레이아웃 로드 버튼 핸들러."""
        try:
            from PySide6.QtWidgets import QComboBox, QDialog, QVBoxLayout, QPushButton
            
            layouts = self.list_layouts()
            if not layouts:
                logger.info("[GuiController] 저장된 레이아웃 없음")
                return
            
            # 다이얼로그 생성
            dialog = QDialog()
            dialog.setWindowTitle("레이아웃 로드")
            dialog.setMinimumWidth(300)
            
            layout = QVBoxLayout(dialog)
            
            combo = QComboBox()
            combo.addItems(layouts)
            layout.addWidget(combo)
            
            btn_load = QPushButton("로드")
            btn_load.clicked.connect(lambda: self._load_selected_layout(combo.currentText(), dialog))
            layout.addWidget(btn_load)
            
            dialog.exec()
        except Exception as e:
            logger.error("[GuiController] 레이아웃 로드 실패: %s", e)
    
    def _load_selected_layout(self, name: str, dialog: QDialog):
        """선택된 레이아웃 로드.
        
        Args:
            name: 레이아웃 이름
            dialog: 다이얼로그
        """
        try:
            if self.load_layout(name):
                logger.info("[GuiController] 레이아웃 로드 완료: %s", name)
                dialog.accept()
            else:
                logger.warning("[GuiController] 레이아웃 로드 실패: %s", name)
        except Exception as e:
            logger.error("[GuiController] 레이아웃 로드 실패: %s", e)
    
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """GUI 진입점. 기존 _run_gui() 와 동일 동작."""
        try:
            from PySide6.QtCore import Qt, QDateTime, QTimer, QSettings, QSize
            from PySide6.QtGui import QColor, QPixmap, QFont, QIcon
            from PySide6.QtWidgets import (
                QApplication,
                QWidget,
                QVBoxLayout,
                QHBoxLayout,
                QGridLayout,
                QFormLayout,
                QGroupBox,
                QPushButton,
                QLineEdit,
                QCheckBox,
                QTextEdit,
                QComboBox,
                QLabel,
                QStyle,
            )
            from qasync import QEventLoop, asyncSlot
            from .alarm_led import AlarmLED
        except Exception as e:
            print(json.dumps({"error": f"PySide6/qasync not installed: {e}"}, ensure_ascii=False, indent=2))
            return 1

        app = QApplication.instance() or QApplication([])

        try:
            base_dir = Path(__file__).resolve().parent
        except Exception:
            base_dir = Path.cwd()

        try:
            icon_path = base_dir / "assets" / "beacon.ico"
            if icon_path.exists():
                ic = QIcon(str(icon_path))
                if not ic.isNull():
                    try:
                        app.setWindowIcon(ic)
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            app.setOrganizationName("Transformer")
            app.setApplicationName(str(APP_NAME))
        except Exception:
            pass

        try:
            settings = QSettings()
        except Exception:
            settings = None

        try:
            from qt_material import apply_stylesheet

            apply_stylesheet(app, theme="dark_teal.xml")

            try:
                app.setStyleSheet(
                    (app.styleSheet() or "")
                    + "\n"
                    + "QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {"
                    + " color: #EDEDED; background-color: rgba(30, 30, 30, 0.85); selection-background-color: #00796B; selection-color: #FFFFFF; }\n"
                    + "QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {"
                    + " color: #9E9E9E; background-color: rgba(60, 60, 60, 0.6); }\n"
                )
            except Exception:
                pass
        except Exception:
            pass

        loop = QEventLoop(app)
        asyncio.set_event_loop(loop)

        emitter = QtLogEmitter()
        qt_handler = QtLogHandler(emitter)
        qt_handler.setLevel(logging.INFO)
        qt_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(qt_handler)

        try:
            sys.stdout = QtStdIORedirect(emitter)
            sys.stderr = QtStdIORedirect(emitter)
        except Exception:
            pass

        shell = build_main_window_shell(
            base_dir=base_dir,
            window_title=str(APP_NAME),
            settings=settings,
        )
        w = shell.window
        outer_root = shell.outer_root
        splitter = shell.splitter
        left_scroll = shell.left_scroll
        left_content = shell.left_content
        left_root = shell.left_root
        right_panel = shell.right_panel
        right_root = shell.right_root

        ui_init_errors: list[str] = []

        form = QFormLayout()

        log_level_cb = QComboBox()
        for lv in ("DEBUG", "INFO", "WARNING", "ERROR"):
            log_level_cb.addItem(lv)
        log_level_cb.setCurrentText("INFO")

        log_file_edit = QLineEdit(DEFAULT_LOG_FILE)

        log_row = QHBoxLayout()
        log_row.setContentsMargins(0, 0, 0, 0)
        log_row.addWidget(QLabel("Level"))
        log_row.addWidget(log_level_cb)
        log_row.addSpacing(10)
        log_row.addWidget(QLabel("File"))
        log_row.addWidget(log_file_edit, stretch=1)
        log_wrap = QWidget()
        log_wrap.setLayout(log_row)
        form.addRow("Log", log_wrap)

        # prediction_minutes는 현재 0이므로 GUI에서 제외

        use_transformer_chk = QCheckBox("Transformer")
        use_transformer_chk.setChecked(False)

        use_tft_chk = QCheckBox("TFT")
        use_tft_chk.setChecked(False)

        # PatchTST: Transformer와 배타적 선택 (둘 중 하나만 체크 가능)
        use_patch_tst_chk = QCheckBox("PatchTST")
        use_patch_tst_chk.setChecked(True)
        use_patch_tst_chk.setToolTip(
            "Transformer 대신 PatchTST 모델 구조로 수치 예측.\n"
            "Transformer 와 배타적으로 선택됩니다 (둘 중 하나만 활성화)."
        )

        # Mamba: 앙상블에 Mamba SSM 추가 (mamba_weights_path 가 설정되어 있어야 유효)
        use_mamba_chk = QCheckBox("Mamba")
        use_mamba_chk.setChecked(False)
        use_mamba_chk.setToolTip(
            "앙상블에 Mamba SSM 모델 추가.\n"
            "config.prediction.mamba_weights_path 가 설정되어 있어야 활성화됩니다.\n"
            "경로 미설정 시 자동으로 비활성화됩니다."
        )

        # ── Transformer ↔ PatchTST 상호 배타적 연동 ──────────────────────
        def _on_transformer_toggled(checked: bool) -> None:
            if checked and use_patch_tst_chk.isChecked():
                use_patch_tst_chk.blockSignals(True)
                use_patch_tst_chk.setChecked(False)
                use_patch_tst_chk.blockSignals(False)

        def _on_patch_tst_toggled(checked: bool) -> None:
            if checked and use_transformer_chk.isChecked():
                use_transformer_chk.blockSignals(True)
                use_transformer_chk.setChecked(False)
                use_transformer_chk.blockSignals(False)

        use_transformer_chk.toggled.connect(_on_transformer_toggled)
        use_patch_tst_chk.toggled.connect(_on_patch_tst_toggled)
        # ──────────────────────────────────────────────────────────────────

        telegram_enable_chk = QCheckBox("Enable Telegram")
        telegram_enable_chk.setChecked(True)

        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.addWidget(use_transformer_chk)
        mode_row.addWidget(use_patch_tst_chk)
        mode_row.addWidget(use_tft_chk)
        mode_row.addWidget(use_mamba_chk)
        mode_row.addWidget(telegram_enable_chk)
        mode_row.addStretch(1)
        mode_wrap = QWidget()
        mode_wrap.setLayout(mode_row)

        # prediction_minutes 제외: Modes만 표시
        pm_modes_row = QHBoxLayout()
        pm_modes_row.setContentsMargins(0, 0, 0, 0)
        pm_modes_row.addWidget(QLabel("Modes"))
        pm_modes_row.addWidget(mode_wrap, stretch=1)
        pm_modes_wrap = QWidget()
        pm_modes_wrap.setLayout(pm_modes_row)
        form.addRow("Prediction", pm_modes_wrap)

        self.effective_pred_defaults = {
            "llm_min_interval_sec": 30.0,
            "tick_size": 0.05,
            "feedback_threshold_ticks": 10,
            "feedback_skip_hold_ticks": 2,
            "feedback_weight_high": 1.0,
            "feedback_weight_mid": 0.5,
            "feedback_weight_low": 0.25,
            "feedback_use_price_snapshot": True,
            "feedback_snapshot_tolerance_sec": 30.0,
            "feedback_snapshot_required": False,
            "fc0_stale_threshold_sec": 10.0,
            "fc0_stale_cooldown_sec": 60.0,
        }
        try:
            cfg0 = load_config("config.json")
            p0 = getattr(cfg0, "prediction", None)
            if p0 is not None:
                self.effective_pred_defaults.update(
                    {
                        "llm_min_interval_sec": float(getattr(p0, "llm_min_interval_sec", 30.0) or 0.0),
                        "tick_size": float(getattr(p0, "tick_size", 0.05) or 0.0),
                        "feedback_threshold_ticks": int(getattr(p0, "feedback_threshold_ticks", 10) or 10),
                        "feedback_skip_hold_ticks": int(getattr(p0, "feedback_skip_hold_ticks", 2) or 0),
                        "feedback_weight_high": float(getattr(p0, "feedback_weight_high", 1.0) or 0.0),
                        "feedback_weight_mid": float(getattr(p0, "feedback_weight_mid", 0.5) or 0.0),
                        "feedback_weight_low": float(getattr(p0, "feedback_weight_low", 0.25) or 0.0),
                        "feedback_use_price_snapshot": bool(getattr(p0, "feedback_use_price_snapshot", True)),
                        "feedback_snapshot_tolerance_sec": float(
                            getattr(p0, "feedback_snapshot_tolerance_sec", 30.0) or 0.0
                        ),
                        "feedback_snapshot_required": bool(getattr(p0, "feedback_snapshot_required", False)),
                        "fc0_stale_threshold_sec": float(getattr(p0, "fc0_stale_threshold_sec", 10.0) or 0.0),
                        "fc0_stale_cooldown_sec": float(getattr(p0, "fc0_stale_cooldown_sec", 60.0) or 0.0),
                    }
                )
        except Exception:
            pass

        def _refresh_prediction_effective_labels() -> None:
            def _eff_float(edit: QLineEdit, base: float) -> float:
                try:
                    s = edit.text().strip()
                    if s:
                        return float(s)
                except Exception:
                    pass
                return float(base)

            def _eff_int(edit: QLineEdit, base: int) -> int:
                try:
                    s = edit.text().strip()
                    if s:
                        return int(float(s))
                except Exception:
                    pass
                return int(base)

            def _eff_bool_combo(combo: Any, base: bool) -> bool:
                try:
                    v = combo.currentData()
                    if v is None:
                        return bool(base)
                    return bool(v)
                except Exception:
                    return bool(base)

            try:
                llm_min_interval_eff_lbl.setText(f"= {_eff_float(llm_min_interval_edit, float(self.effective_pred_defaults.get('llm_min_interval_sec') or 0.0)):.3g}")
            except Exception:
                pass
            try:
                tick_size_eff_lbl.setText(f"= {_eff_float(tick_size_edit, float(self.effective_pred_defaults.get('tick_size') or 0.0)):.6g}")
            except Exception:
                pass
            try:
                feedback_ticks_eff_lbl.setText(f"= {_eff_int(feedback_ticks_edit, int(self.effective_pred_defaults.get('feedback_threshold_ticks') or 0))}")
            except Exception:
                pass
            try:
                feedback_skip_hold_ticks_eff_lbl.setText(f"= {_eff_int(feedback_skip_hold_ticks_edit, int(self.effective_pred_defaults.get('feedback_skip_hold_ticks') or 0))}")
            except Exception:
                pass
            try:
                feedback_weight_high_eff_lbl.setText(f"= {_eff_float(feedback_weight_high_edit, float(self.effective_pred_defaults.get('feedback_weight_high') or 0.0)):.3g}")
            except Exception:
                pass
            try:
                feedback_weight_mid_eff_lbl.setText(f"= {_eff_float(feedback_weight_mid_edit, float(self.effective_pred_defaults.get('feedback_weight_mid') or 0.0)):.3g}")
            except Exception:
                pass
            try:
                feedback_weight_low_eff_lbl.setText(f"= {_eff_float(feedback_weight_low_edit, float(self.effective_pred_defaults.get('feedback_weight_low') or 0.0)):.3g}")
            except Exception:
                pass
            try:
                feedback_use_snapshot_eff_lbl.setText(
                    f"= {str(_eff_bool_combo(feedback_use_snapshot_cb, bool(self.effective_pred_defaults.get('feedback_use_price_snapshot')))).lower()}"
                )
            except Exception:
                pass
            try:
                feedback_snapshot_tol_eff_lbl.setText(
                    f"= {_eff_float(feedback_snapshot_tol_edit, float(self.effective_pred_defaults.get('feedback_snapshot_tolerance_sec') or 0.0)):.3g}s"
                )
            except Exception:
                pass
            try:
                feedback_snapshot_required_eff_lbl.setText(
                    f"= {str(_eff_bool_combo(feedback_snapshot_required_cb, bool(self.effective_pred_defaults.get('feedback_snapshot_required')))).lower()}"
                )
            except Exception:
                pass
            try:
                fc0_stale_thr_eff_lbl.setText(
                    f"= {_eff_float(fc0_stale_thr_edit, float(self.effective_pred_defaults.get('fc0_stale_threshold_sec') or 0.0)):.3g}s"
                )
            except Exception:
                pass
            try:
                fc0_stale_cool_eff_lbl.setText(
                    f"= {_eff_float(fc0_stale_cool_edit, float(self.effective_pred_defaults.get('fc0_stale_cooldown_sec') or 0.0)):.3g}s"
                )
            except Exception:
                pass

        llm_min_interval_edit = QLineEdit("")
        llm_min_interval_edit.setPlaceholderText("(config)")
        try:
            llm_min_interval_edit.setFixedWidth(160)
        except Exception:
            pass
        set_float_validator(llm_min_interval_edit, min_v=0.0)
        llm_min_interval_wrap, llm_min_interval_eff_lbl = make_effective_row(llm_min_interval_edit)

        tick_size_edit = QLineEdit("")
        tick_size_edit.setPlaceholderText("(config)")
        set_float_validator(tick_size_edit, min_v=0.0)
        tick_size_wrap, tick_size_eff_lbl = make_effective_row(tick_size_edit)

        feedback_ticks_edit = QLineEdit("")
        feedback_ticks_edit.setPlaceholderText("(config)")
        set_int_validator(feedback_ticks_edit, min_v=0)
        feedback_ticks_wrap, feedback_ticks_eff_lbl = make_effective_row(feedback_ticks_edit)

        feedback_skip_hold_ticks_edit = QLineEdit("")
        feedback_skip_hold_ticks_edit.setPlaceholderText("(config)")
        set_int_validator(feedback_skip_hold_ticks_edit, min_v=0)
        feedback_skip_hold_ticks_wrap, feedback_skip_hold_ticks_eff_lbl = make_effective_row(feedback_skip_hold_ticks_edit)

        feedback_weight_high_edit = QLineEdit("")
        feedback_weight_high_edit.setPlaceholderText("(config)")
        set_float_validator(feedback_weight_high_edit, min_v=0.0)
        feedback_weight_high_wrap, feedback_weight_high_eff_lbl = make_effective_row(feedback_weight_high_edit)

        feedback_weight_mid_edit = QLineEdit("")
        feedback_weight_mid_edit.setPlaceholderText("(config)")
        set_float_validator(feedback_weight_mid_edit, min_v=0.0)
        feedback_weight_mid_wrap, feedback_weight_mid_eff_lbl = make_effective_row(feedback_weight_mid_edit)

        feedback_weight_low_edit = QLineEdit("")
        feedback_weight_low_edit.setPlaceholderText("(config)")
        set_float_validator(feedback_weight_low_edit, min_v=0.0)
        feedback_weight_low_wrap, feedback_weight_low_eff_lbl = make_effective_row(feedback_weight_low_edit)

        feedback_use_snapshot_cb = QComboBox()
        feedback_use_snapshot_cb.addItem("(config)", None)
        feedback_use_snapshot_cb.addItem("True", True)
        feedback_use_snapshot_cb.addItem("False", False)
        feedback_use_snapshot_wrap, feedback_use_snapshot_eff_lbl = make_effective_row_widget(feedback_use_snapshot_cb)

        feedback_snapshot_tol_edit = QLineEdit("")
        feedback_snapshot_tol_edit.setPlaceholderText("(config)")
        set_float_validator(feedback_snapshot_tol_edit, min_v=0.0)
        feedback_snapshot_tol_wrap, feedback_snapshot_tol_eff_lbl = make_effective_row(feedback_snapshot_tol_edit)

        feedback_snapshot_required_cb = QComboBox()
        feedback_snapshot_required_cb.addItem("(config)", None)
        feedback_snapshot_required_cb.addItem("True", True)
        feedback_snapshot_required_cb.addItem("False", False)
        try:
            feedback_snapshot_required_cb.setFixedWidth(160)
        except Exception:
            pass
        feedback_snapshot_required_wrap, feedback_snapshot_required_eff_lbl = make_effective_row_widget(
            feedback_snapshot_required_cb
        )

        fc0_stale_thr_edit = QLineEdit("")
        fc0_stale_thr_edit.setPlaceholderText("(config)")
        set_float_validator(fc0_stale_thr_edit, min_v=0.0)
        fc0_stale_thr_wrap, fc0_stale_thr_eff_lbl = make_effective_row(fc0_stale_thr_edit)

        fc0_stale_cool_edit = QLineEdit("")
        fc0_stale_cool_edit.setPlaceholderText("(config)")
        set_float_validator(fc0_stale_cool_edit, min_v=0.0)
        fc0_stale_cool_wrap, fc0_stale_cool_eff_lbl = make_effective_row(fc0_stale_cool_edit)

        # Prediction/Feedback 그룹박스 초기화
        pred_group = None

        try:
            pred_grid = QGridLayout()
            pred_grid.setContentsMargins(0, 0, 0, 0)

            def _grid_row(r: int, left_label: str, left_widget: QWidget, right_label: str, right_widget: QWidget) -> None:
                pred_grid.addWidget(QLabel(str(left_label)), r, 0)
                pred_grid.addWidget(left_widget, r, 1)
                pred_grid.addWidget(QLabel(str(right_label)), r, 2)
                pred_grid.addWidget(right_widget, r, 3)

            _grid_row(0, "LLM min interval (sec)", llm_min_interval_wrap, "FC0 stale cooldown (sec)", fc0_stale_cool_wrap)
            _grid_row(1, "Tick size", tick_size_wrap, "FC0 stale threshold (sec)", fc0_stale_thr_wrap)
            _grid_row(2, "Feedback threshold (ticks)", feedback_ticks_wrap, "Feedback skip HOLD (ticks)", feedback_skip_hold_ticks_wrap)
            _grid_row(3, "Feedback weight HIGH", feedback_weight_high_wrap, "Feedback weight MID", feedback_weight_mid_wrap)
            _grid_row(4, "Feedback weight LOW", feedback_weight_low_wrap, "Feedback use price snapshot", feedback_use_snapshot_wrap)
            _grid_row(
                5,
                "Feedback snapshot tolerance (sec)",
                feedback_snapshot_tol_wrap,
                "Feedback snapshot required",
                feedback_snapshot_required_wrap,
            )

            pred_grid_wrap = QWidget()
            pred_grid_wrap.setLayout(pred_grid)
            pred_group = QGroupBox("Prediction / Feedback")
            pred_group_layout = QVBoxLayout()
            pred_group_layout.setContentsMargins(10, 10, 10, 10)
            pred_group_layout.addWidget(pred_grid_wrap)
            pred_group.setLayout(pred_group_layout)
            # 탭 위젯에 추가하기 위해 outer 위젯 생성 제거
        except Exception as e:
            record_gui_init_error(logger, ui_init_errors, "Prediction settings grid", e)
            try:
                err_lbl = QLabel(f"Prediction UI error: {type(e).__name__}: {str(e)}")
                err_lbl.setStyleSheet("color: #FF8A80;")
                pred_group = QGroupBox("Prediction / Feedback")
                pred_layout = QVBoxLayout(pred_group)
                pred_layout.addWidget(err_lbl)
            except Exception:
                pred_group = None

        try:
            llm_min_interval_edit.textChanged.connect(lambda _t: _refresh_prediction_effective_labels())
            tick_size_edit.textChanged.connect(lambda _t: _refresh_prediction_effective_labels())
            feedback_ticks_edit.textChanged.connect(lambda _t: _refresh_prediction_effective_labels())
            feedback_skip_hold_ticks_edit.textChanged.connect(lambda _t: _refresh_prediction_effective_labels())
            feedback_weight_high_edit.textChanged.connect(lambda _t: _refresh_prediction_effective_labels())
            feedback_weight_mid_edit.textChanged.connect(lambda _t: _refresh_prediction_effective_labels())
            feedback_weight_low_edit.textChanged.connect(lambda _t: _refresh_prediction_effective_labels())
            feedback_use_snapshot_cb.currentIndexChanged.connect(lambda _idx: _refresh_prediction_effective_labels())
            feedback_snapshot_tol_edit.textChanged.connect(lambda _t: _refresh_prediction_effective_labels())
            feedback_snapshot_required_cb.currentIndexChanged.connect(lambda _idx: _refresh_prediction_effective_labels())
            fc0_stale_thr_edit.textChanged.connect(lambda _t: _refresh_prediction_effective_labels())
            fc0_stale_cool_edit.textChanged.connect(lambda _t: _refresh_prediction_effective_labels())
        except Exception:
            pass

        try:
            _refresh_prediction_effective_labels()
        except Exception:
            pass

        replay_file_edit = QLineEdit("")
        replay_file_edit.setPlaceholderText("ticks_replay_YYYYMMDD_HHMMSS.jsonl.gz")
        btn_replay_file = QPushButton("...")
        replay_row = QHBoxLayout()
        replay_row.setContentsMargins(0, 0, 0, 0)
        replay_row.addWidget(replay_file_edit)
        replay_row.addWidget(btn_replay_file)
        replay_wrap = QWidget()
        replay_wrap.setLayout(replay_row)
        form.addRow("Replay file", replay_wrap)

        replay_speed_edit = QLineEdit("0")
        replay_speed_edit.setPlaceholderText("0=Max, 1=Realtime")

        replay_max_lines_edit = QLineEdit("")
        replay_max_lines_edit.setPlaceholderText("(optional)")

        replay_opts_row = QHBoxLayout()
        replay_opts_row.setContentsMargins(0, 0, 0, 0)
        replay_opts_row.addWidget(QLabel("Speed"))
        replay_opts_row.addWidget(replay_speed_edit)
        replay_opts_row.addSpacing(10)
        replay_opts_row.addWidget(QLabel("Max lines"))
        replay_opts_row.addWidget(replay_max_lines_edit)
        replay_opts_wrap = QWidget()
        replay_opts_wrap.setLayout(replay_opts_row)
        form.addRow("Replay", replay_opts_wrap)

        # Adaptive indicators 그룹박스 초기화
        adapt_group = None

        # Adaptive indicators
        try:
            adapt_group = QGroupBox("Adaptive indicators")
            adapt_root = QVBoxLayout(adapt_group)

            adapt_common_form = QFormLayout()

            # Multi-timeframe features (horizontal layout)
            multiscale_layout = QHBoxLayout()
            multiscale_5m_cb = QCheckBox("5m")
            multiscale_5m_cb.setChecked(True)  # 기본값: 5분봉 활성화
            multiscale_5m_cb.setToolTip("5분봉 기반 멀티스케일 피처 활성화 (중기 추세)")
            multiscale_15m_cb = QCheckBox("15m")
            multiscale_15m_cb.setChecked(False)  # 기본값: 15분봉 비활성화
            multiscale_15m_cb.setToolTip("15분봉 기반 멀티스케일 피처 활성화 (장기 추세)")
            multiscale_layout.addWidget(multiscale_5m_cb)
            multiscale_layout.addWidget(multiscale_15m_cb)
            adapt_common_form.addRow("Multiscale", multiscale_layout)

            adapt_warmup_edit = QLineEdit("")
            adapt_warmup_edit.setPlaceholderText("(config)")
            set_int_validator(adapt_warmup_edit, min_v=0)
            adapt_common_form.addRow("Warmup bars (5분봉 기준)", adapt_warmup_edit)

            ast_er_period = QLineEdit("")
            ast_er_period.setPlaceholderText("(config)")
            set_int_validator(ast_er_period, min_v=0)
            adapt_common_form.addRow("AST ER period", ast_er_period)

            azz_er_period = QLineEdit("")
            azz_er_period.setPlaceholderText("(config)")
            set_int_validator(azz_er_period, min_v=0)
            adapt_common_form.addRow("AZZ ER period", azz_er_period)

            ast_group = QGroupBox("Adaptive SuperTrend (AST)")
            ast_form = QGridLayout(ast_group)
            ast_form.setContentsMargins(6, 6, 6, 6)
            try:
                ast_form.setHorizontalSpacing(10)
                ast_form.setVerticalSpacing(6)
            except Exception:
                pass

            def _ast_row(r: int, l_label: str, l_widget: QWidget, r_label: str, r_widget: QWidget) -> None:
                ast_form.addWidget(QLabel(str(l_label)), r, 0)
                ast_form.addWidget(l_widget, r, 1)
                ast_form.addWidget(QLabel(str(r_label)), r, 2)
                ast_form.addWidget(r_widget, r, 3)

            ast_mult_min = QLineEdit("")
            ast_mult_min.setPlaceholderText("(config)")
            set_float_validator(ast_mult_min, min_v=0.0)
            ast_mult_max = QLineEdit("")
            ast_mult_max.setPlaceholderText("(config)")
            set_float_validator(ast_mult_max, min_v=0.0)

            ast_atr_min = QLineEdit("")
            ast_atr_min.setPlaceholderText("(config)")
            set_float_validator(ast_atr_min, min_v=0.0)
            ast_atr_max = QLineEdit("")
            ast_atr_max.setPlaceholderText("(config)")
            set_float_validator(ast_atr_max, min_v=0.0)

            ast_adx_period = QLineEdit("")
            ast_adx_period.setPlaceholderText("(config)")
            set_int_validator(ast_adx_period, min_v=0)

            ast_bb_correction_cb = QComboBox()
            ast_bb_correction_cb.addItem("(config)", userData=None)
            ast_bb_correction_cb.addItem("On", userData=True)
            ast_bb_correction_cb.addItem("Off", userData=False)

            ast_bb_period = QLineEdit("")
            ast_bb_period.setPlaceholderText("(config)")
            set_int_validator(ast_bb_period, min_v=0)
            ast_bb_std = QLineEdit("")
            ast_bb_std.setPlaceholderText("(config)")
            set_float_validator(ast_bb_std, min_v=0.0)

            ast_smooth = QLineEdit("")
            ast_smooth.setPlaceholderText("(config)")
            set_int_validator(ast_smooth, min_v=0)

            ast_smooth_lbl = QLabel("smooth")
            ast_form.addWidget(QLabel("mult min"), 0, 0)
            ast_form.addWidget(ast_mult_min, 0, 1)
            ast_form.addWidget(QLabel("mult max"), 0, 2)
            ast_form.addWidget(ast_mult_max, 0, 3)

            ast_form.addWidget(QLabel("ATR min"), 1, 0)
            ast_form.addWidget(ast_atr_min, 1, 1)
            ast_form.addWidget(QLabel("ATR max"), 1, 2)
            ast_form.addWidget(ast_atr_max, 1, 3)

            ast_form.addWidget(QLabel("ADX period"), 2, 0)
            ast_form.addWidget(ast_adx_period, 2, 1)
            ast_form.addWidget(QLabel("BB correction"), 2, 2)
            ast_form.addWidget(ast_bb_correction_cb, 2, 3)

            ast_form.addWidget(QLabel("BB period"), 3, 0)
            ast_form.addWidget(ast_bb_period, 3, 1)
            ast_form.addWidget(QLabel("BB std"), 3, 2)
            ast_form.addWidget(ast_bb_std, 3, 3)

            ast_form.addWidget(ast_smooth_lbl, 4, 0)
            ast_form.addWidget(ast_smooth, 4, 1)
            ast_form.addWidget(QLabel(""), 4, 2)
            ast_form.addWidget(QLabel(""), 4, 3)

            azz_group = QGroupBox("Adaptive ZigZag (AZZ)")
            azz_form = QGridLayout(azz_group)
            azz_form.setContentsMargins(6, 6, 6, 6)
            try:
                azz_form.setHorizontalSpacing(10)
                azz_form.setVerticalSpacing(6)
            except Exception:
                pass

            azz_atr_mult = QLineEdit("")
            azz_atr_mult.setPlaceholderText("(config)")
            set_float_validator(azz_atr_mult, min_v=0.0)
            azz_atr_period = QLineEdit("")
            azz_atr_period.setPlaceholderText("(config)")
            set_int_validator(azz_atr_period, min_v=0)

            azz_atr_mult_min = QLineEdit("")
            azz_atr_mult_min.setPlaceholderText("(config)")
            set_float_validator(azz_atr_mult_min, min_v=0.0)
            azz_atr_mult_max = QLineEdit("")
            azz_atr_mult_max.setPlaceholderText("(config)")
            set_float_validator(azz_atr_mult_max, min_v=0.0)

            azz_min_thr = QLineEdit("")
            azz_min_thr.setPlaceholderText("(config)")
            set_float_validator(azz_min_thr, min_v=0.0)
            azz_max_thr = QLineEdit("")
            azz_max_thr.setPlaceholderText("(config)")
            set_float_validator(azz_max_thr, min_v=0.0)

            azz_major_ratio = QLineEdit("")
            azz_major_ratio.setPlaceholderText("(config)")
            set_float_validator(azz_major_ratio, min_v=0.0)
            azz_confirm = QLineEdit("")
            azz_confirm.setPlaceholderText("(config)")
            set_int_validator(azz_confirm, min_v=0)

            azz_max_swings = QLineEdit("")
            azz_max_swings.setPlaceholderText("(config)")
            set_int_validator(azz_max_swings, min_v=0)
            azz_min_wave_bars = QLineEdit("")
            azz_min_wave_bars.setPlaceholderText("(config)")
            set_int_validator(azz_min_wave_bars, min_v=0)

            azz_min_wave_pct = QLineEdit("")
            azz_min_wave_pct.setPlaceholderText("(config)")
            set_float_validator(azz_min_wave_pct, min_v=0.0)
            azz_cluster_tol = QLineEdit("")
            azz_cluster_tol.setPlaceholderText("(config)")
            set_float_validator(azz_cluster_tol, min_v=0.0)

            azz_struct_lookback = QLineEdit("")
            azz_struct_lookback.setPlaceholderText("(config)")
            set_int_validator(azz_struct_lookback, min_v=0)
            azz_struct_points = QLineEdit("")
            azz_struct_points.setPlaceholderText("(config)")
            set_int_validator(azz_struct_points, min_v=0)

            azz_freeze_on_confirm = QComboBox()
            azz_freeze_on_confirm.addItem("(config)", userData=None)
            azz_freeze_on_confirm.addItem("On", userData=True)
            azz_freeze_on_confirm.addItem("Off", userData=False)

            azz_form.addWidget(QLabel("ATR mult"), 0, 0)
            azz_form.addWidget(azz_atr_mult, 0, 1)
            azz_form.addWidget(QLabel("ATR period"), 0, 2)
            azz_form.addWidget(azz_atr_period, 0, 3)

            azz_form.addWidget(QLabel("ATR mult min"), 1, 0)
            azz_form.addWidget(azz_atr_mult_min, 1, 1)
            azz_form.addWidget(QLabel("ATR mult max"), 1, 2)
            azz_form.addWidget(azz_atr_mult_max, 1, 3)

            azz_form.addWidget(QLabel("min thr %"), 2, 0)
            azz_form.addWidget(azz_min_thr, 2, 1)
            azz_form.addWidget(QLabel("max thr %"), 2, 2)
            azz_form.addWidget(azz_max_thr, 2, 3)

            azz_form.addWidget(QLabel("major ratio"), 3, 0)
            azz_form.addWidget(azz_major_ratio, 3, 1)
            azz_form.addWidget(QLabel("confirm bars"), 3, 2)
            azz_form.addWidget(azz_confirm, 3, 3)

            azz_form.addWidget(QLabel("max swings"), 4, 0)
            azz_form.addWidget(azz_max_swings, 4, 1)
            azz_form.addWidget(QLabel("min wave bars"), 4, 2)
            azz_form.addWidget(azz_min_wave_bars, 4, 3)

            azz_form.addWidget(QLabel("min wave %"), 5, 0)
            azz_form.addWidget(azz_min_wave_pct, 5, 1)
            azz_form.addWidget(QLabel("cluster tol %"), 5, 2)
            azz_form.addWidget(azz_cluster_tol, 5, 3)

            azz_form.addWidget(QLabel("struct lookback"), 6, 0)
            azz_form.addWidget(azz_struct_lookback, 6, 1)
            azz_form.addWidget(QLabel("struct points"), 6, 2)
            azz_form.addWidget(azz_struct_points, 6, 3)

            azz_form.addWidget(QLabel("freeze on confirm"), 7, 0)
            azz_form.addWidget(azz_freeze_on_confirm, 7, 1)
            azz_form.addWidget(QLabel(""), 7, 2)
            azz_form.addWidget(QLabel(""), 7, 3)

            adapt_root.addLayout(adapt_common_form)
            adapt_cols = QHBoxLayout()
            adapt_cols.addWidget(ast_group)
            adapt_cols.addWidget(azz_group)
            adapt_root.addLayout(adapt_cols)
        except Exception as e:
            record_gui_init_error(logger, ui_init_errors, "Adaptive indicators", e)
            adapt_group = QGroupBox("Adaptive indicators")
            adapt_root = QVBoxLayout(adapt_group)
            err_lbl = QLabel(f"Adaptive UI error: {type(e).__name__}: {str(e)}")
            err_lbl.setStyleSheet("color: #FF8A80;")
            adapt_root.addWidget(err_lbl)

        left_root.addLayout(form)

        # 탭 위젯 생성 (Prediction/Feedback + Adaptive Indicators)
        from PySide6.QtWidgets import QTabWidget
        settings_tab = QTabWidget()
        settings_tab.setTabPosition(QTabWidget.North)

        # Adaptive Indicators 탭 (첫 번째 탭)
        if adapt_group is not None:
            settings_tab.addTab(adapt_group, "Adaptive Indicators")

        # Prediction/Feedback 탭
        if pred_group is not None:
            settings_tab.addTab(pred_group, "Prediction/Feedback")

        left_root.addWidget(settings_tab)

        btn_row = QHBoxLayout()
        toggle_btn = QPushButton("🚀 Start")
        toggle_btn.setCheckable(True)
        btn_row.addWidget(toggle_btn)

        replay_btn = QPushButton("🔄 Replay")
        btn_row.addWidget(replay_btn)

        reload_cfg_btn = QPushButton("🔃 Reload config")
        btn_row.addWidget(reload_cfg_btn)

        reset_weights_btn = QPushButton("↩️ Reset weights")
        btn_row.addWidget(reset_weights_btn)

        # 레이아웃 제어 버튼
        try:
            btn_save_layout = QPushButton("💾 레이아웃 저장")
            btn_save_layout.setFixedWidth(140)
            btn_save_layout.clicked.connect(lambda: self._on_save_layout())
            btn_row.addWidget(btn_save_layout)

            btn_load_layout = QPushButton("📂 레이아웃 로드")
            btn_load_layout.setFixedWidth(140)
            btn_load_layout.clicked.connect(lambda: self._on_load_layout())
            btn_row.addWidget(btn_load_layout)
        except Exception as _lc_err:
            logger.warning("[LayoutControl] 버튼 추가 실패: %s", _lc_err)

        left_root.addLayout(btn_row)

        run_state_lbl = QLabel("Idle")
        right_root.addWidget(run_state_lbl)

        status_lbl = QLabel("RT: -")
        right_root.addWidget(status_lbl)

        summary_box = QGroupBox("Summary")
        summary_layout = QHBoxLayout(summary_box)

        summary_text = QLabel("")
        summary_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        summary_text.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        summary_layout.addWidget(summary_text, stretch=1)

        # LED 설정 헬퍼 함수
        def _configure_led(led: AlarmLED, size: int = 62, color: str = "gray", visible: bool = True) -> None:
            """LED 설정 헬퍼 함수."""
            try:
                led.setLedSize(size)
            except Exception:
                pass
            try:
                led.setColor(QColor(color))
            except Exception:
                pass
            try:
                led.setVisible(visible)
            except Exception:
                pass

        # 휴리스틱 LED 왼쪽에 레짐 LED 추가
        regime_led = AlarmLED("REGIME", "#EDEDED", font_size=10)
        _configure_led(regime_led)
        summary_layout.addWidget(regime_led, alignment=Qt.AlignRight | Qt.AlignVCenter)

        # 레짐 정보 라벨 추가
        regime_info_label = QLabel()
        regime_info_label.setStyleSheet("""
            QLabel {
                color: #FFD700;
                background-color: rgba(0, 0, 0, 120);
                border: 1px solid #FFD700;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 12px;
                font-family: Consolas, monospace;
            }
        """)
        regime_info_label.setText("Regime: -")
        regime_info_label.hide()
        summary_layout.addWidget(regime_info_label, alignment=Qt.AlignRight | Qt.AlignVCenter)

        # 레짐 라벨 업데이트 콜백
        def _set_regime_label_text(text: str, visible: bool = True) -> None:
            """레짐 라벨 텍스트 설정."""
            try:
                regime_info_label.setText(text)
                regime_info_label.setVisible(visible)
            except Exception as e:
                logger.warning("[GuiController] 레짐 라벨 텍스트 설정 실패: %s", e)

        heur_led = AlarmLED("HEUR", "#EDEDED", font_size=10)
        _configure_led(heur_led)
        summary_layout.addWidget(heur_led, alignment=Qt.AlignRight | Qt.AlignVCenter)

        # LLM 활성 상태 확인
        use_llm = False
        try:
            if cfg0 is not None:
                use_llm = bool(getattr(cfg0, "use_llm", False))
        except Exception:
            pass

        # LLM 비활성 시 GPT/GEM LED 숨김
        gpt_led = AlarmLED("GPT", "#EDEDED", font_size=10)
        _configure_led(gpt_led, visible=use_llm)
        summary_layout.addWidget(gpt_led, alignment=Qt.AlignRight | Qt.AlignVCenter)

        gem_led = AlarmLED("GEM", "#EDEDED", font_size=10)
        _configure_led(gem_led, visible=use_llm)
        summary_layout.addWidget(gem_led, alignment=Qt.AlignRight | Qt.AlignVCenter)

        cons_wrap = QWidget()
        cons_row = QHBoxLayout(cons_wrap)
        cons_row.setContentsMargins(0, 0, 0, 0)
        cons_row.setSpacing(6)

        cons_arrow = QLabel("")
        cons_arrow.setAlignment(Qt.AlignCenter)
        try:
            cons_arrow.setFixedSize(72, 72)
        except Exception:
            pass
        cons_arrow_default_font = QFont(cons_arrow.font()) if cons_arrow.font() else None
        cons_row.addWidget(cons_arrow)

        arrow_pixmaps: Dict[str, Optional[QPixmap]] = {
            "UP": None,
            "DOWN": None,
            "RIGHT": None,
            "UNKNOWN": None,
        }

        def _get_arrow_pix(which: str) -> Optional[QPixmap]:
            """화살표 픽스맵을 로드하고 캐싱."""
            key = str(which or "").strip().upper()
            if key not in arrow_pixmaps:
                return None
            if arrow_pixmaps.get(key) is not None:
                return arrow_pixmaps.get(key)
            
            try:
                base_dir = Path(__file__).resolve().parent
            except Exception:
                base_dir = Path.cwd()
            
            fn_map = {
                "UP": "arrow_up.png",
                "DOWN": "arrow_down.png",
                "RIGHT": "arrow_right.png",
                "UNKNOWN": "arrow_unknown.png",
            }
            fn = fn_map.get(key)
            if not fn:
                return None
            
            try:
                pm = QPixmap(str(base_dir / "assets" / fn))
                if pm is None or pm.isNull():
                    pm = None
            except Exception:
                pm = None
            
            arrow_pixmaps[key] = pm
            return pm

        try:
            cons_wrap.setToolTip("Consensus")
        except Exception:
            pass

        def _set_cons_arrow(cons_action: str) -> None:
            """컨센서스 화살표 설정."""
            try:
                a = str(cons_action or "").strip().upper()
            except Exception:
                a = ""

            try:
                style = app.style() if app is not None else None
            except Exception:
                style = None

            if style is None:
                return

            icon_sz = QSize(72, 72)

            def _set_unknown_arrow() -> None:
                """알 수 없는 화살표 설정."""
                try:
                    if cons_arrow_default_font is not None:
                        cons_arrow.setFont(QFont(cons_arrow_default_font))
                except Exception:
                    pass
                try:
                    cons_arrow.setStyleSheet("")
                except Exception:
                    pass
                
                pm = _get_arrow_pix("UNKNOWN")
                if pm is not None and (not pm.isNull()):
                    try:
                        if pm.size() != icon_sz:
                            pm = pm.scaled(icon_sz, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    except Exception:
                        pass
                    cons_arrow.setText("")
                    cons_arrow.setPixmap(pm)
                else:
                    try:
                        cons_arrow.setPixmap(QPixmap())
                    except Exception:
                        pass
                    try:
                        cons_arrow.setText("?")
                    except Exception:
                        pass
                    try:
                        f = QFont(cons_arrow.font())
                        f.setPointSize(32)
                        f.setBold(True)
                        cons_arrow.setFont(f)
                    except Exception:
                        pass
                    try:
                        cons_arrow.setStyleSheet("color: #EDEDED;")
                    except Exception:
                        pass

            if a not in ("BUY", "SELL", "HOLD"):
                _set_unknown_arrow()
                return

            try:
                if cons_arrow_default_font is not None:
                    cons_arrow.setFont(QFont(cons_arrow_default_font))
            except Exception:
                pass

            # 화살표 방향 매핑
            action_to_icon = {
                "BUY": (QStyle.SP_ArrowUp, "UP"),
                "SELL": (QStyle.SP_ArrowDown, "DOWN"),
                "HOLD": (QStyle.SP_ArrowRight, "RIGHT"),
            }
            
            style_icon, pix_key = action_to_icon.get(a, (None, "RIGHT"))
            pm = _get_arrow_pix(pix_key)
            
            if pm is None and style_icon is not None:
                ic = style.standardIcon(style_icon)
                pm = ic.pixmap(icon_sz)

            if pm is not None and (not pm.isNull()):
                try:
                    if pm.size() != icon_sz:
                        pm = pm.scaled(icon_sz, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                except Exception:
                    pass
                cons_arrow.setText("")
                try:
                    cons_arrow.setStyleSheet("")
                except Exception:
                    pass
                cons_arrow.setPixmap(pm)
            else:
                try:
                    cons_arrow.setPixmap(QPixmap())
                except Exception:
                    pass
                cons_arrow.setText("?")

        _set_cons_arrow("-")

        logger.info("[GuiController] summary_box를 right_root에 추가 전")
        summary_layout.addWidget(cons_wrap, alignment=Qt.AlignRight | Qt.AlignVCenter)

        right_root.addWidget(summary_box)
        logger.info("[GuiController] summary_box를 right_root에 추가 완료")

        # 레이아웃 제어 버튼은 left_root btn_row로 이동됨
        # ──────────────────────────────────────────────────────────────────────

        logger.info("[GuiController] 탭 위젯 초기화 코드 이전")
        # ── OHLC 차트 뷰어 & 거래 로그 뷰어 & 분석 대시보드 & 백업 뷰어 (탭으로 구성) ───
        self._chart_viewer = None
        self._trade_log_viewer = None
        self._trade_dashboard = None
        self._backup_viewer = None
        self._layout_manager = None
        tab_widget = None

        # 텔레그램 bridge 홀더 초기화
        telegram_bridge_holder: Dict[str, Any] = {"bridge": None, "last_flip_send_ts": 0.0}
        
        try:
            logger.info("[GuiController] 탭 위젯 초기화 try 블록 시작")
            from PySide6.QtWidgets import QTabWidget, QWidget, QVBoxLayout
            from gui.chart_viewer import attach_chart_viewer
            
            # 탭 위젯 생성
            tab_widget = QTabWidget()
            tab_widget.setMinimumHeight(400)  # 최소 높이 설정
            
            # 차트 탭
            chart_tab = QWidget()
            chart_layout = QVBoxLayout(chart_tab)
            chart_layout.setContentsMargins(0, 0, 0, 0)
            logger.info("[GuiController] attach_chart_viewer 호출 전")
            # config에서 refresh_ms 읽어서 전달
            chart_refresh_ms = getattr(getattr(cfg0, "chart", None), "refresh_ms", 5000) if hasattr(cfg0, "chart") else 5000

            # 레짐 LED 색상 설정 콜백 함수
            def _set_regime_led_color(color: str) -> None:
                try:
                    from PySide6.QtGui import QColor
                    regime_led.setColor(QColor(color))
                except Exception as e:
                    logger.warning("[GuiController] 레짐 LED 색상 설정 실패: %s", e)

            self._chart_viewer = attach_chart_viewer(
                chart_layout,
                predictor=None,   # predictor 준비 전 — set_predictor로 나중에 연결
                config=cfg0,       # config 전달
                refresh_ms=chart_refresh_ms,  # config.json의 refresh_ms 사용
                stretch=1,  # 탭 내에서는 stretch=1
                regime_led_callback=_set_regime_led_color,
                regime_label_callback=_set_regime_label_text,
                telegram_bridge_holder=telegram_bridge_holder,  # 텔레그램 bridge 홀더 전달
            )
            tab_widget.addTab(chart_tab, "📈 차트")
            
            # 거래 로그 탭 (별도 예외 처리)
            try:
                from gui.trade_log_viewer import attach_trade_log_viewer
                log_tab = QWidget()
                log_layout = QVBoxLayout(log_tab)
                log_layout.setContentsMargins(0, 0, 0, 0)
                self._trade_log_viewer = attach_trade_log_viewer(
                    log_layout,
                    log_dir=Path("logs/trades")
                )
                tab_widget.addTab(log_tab, "📋 거래 로그")
            except Exception as _log_err:
                logger.warning("[TradeLogViewer] 초기화 실패: %s", _log_err)
                # 거래 로그 뷰어 실패해도 차트는 계속 표시
            
            # 분석 대시보드 탭 (별도 예외 처리)
            try:
                from gui.trade_dashboard import attach_trade_dashboard
                dashboard_tab = QWidget()
                dashboard_layout = QVBoxLayout(dashboard_tab)
                dashboard_layout.setContentsMargins(0, 0, 0, 0)
                self._trade_dashboard = attach_trade_dashboard(
                    dashboard_layout,
                    log_dir=Path("logs/trades")
                )
                tab_widget.addTab(dashboard_tab, "📊 분석 대시보드")
            except Exception as _dash_err:
                logger.warning("[TradeDashboard] 초기화 실패: %s", _dash_err)
                # 분석 대시보드 실패해도 차트는 계속 표시
            
            # 백업 뷰어 탭 (별도 예외 처리)
            try:
                from gui.backup_viewer import attach_backup_viewer
                backup_tab = QWidget()
                backup_layout = QVBoxLayout(backup_tab)
                backup_layout.setContentsMargins(0, 0, 0, 0)
                self._backup_viewer = attach_backup_viewer(
                    backup_layout,
                    log_dir=Path("logs/trades")
                )
                tab_widget.addTab(backup_tab, "💾 백업")
            except Exception as _backup_err:
                logger.warning("[BackupViewer] 초기화 실패: %s", _backup_err)
                # 백업 뷰어 실패해도 차트는 계속 표시
            
            # 탭 위젯을 right_root에 추가
            right_root.addWidget(tab_widget, stretch=5)  # stretch=3 → 5 증가 (로그 줄임에 따라)
            
        except Exception as _cv_err:
            logger.error("[ChartViewer] 초기화 실패: %s", _cv_err)
            # 탭 위젯 실패 시 기존 방식으로 차트 뷰어 추가 (폴백 비활성화 - 중복 생성 방지)
            # from gui.chart_viewer import attach_chart_viewer
            # self._chart_viewer = attach_chart_viewer(
            #     right_root,
            #     predictor=None,
            #     config=cfg0,
            #     stretch=5,
            # )
        
        # 레이아웃 관리자 초기화 (탭 위젯이 있는 경우만)
        if tab_widget is not None:
            try:
                from gui.layout_manager import LayoutManager
                self._layout_manager = LayoutManager(tab_widget)
                logger.info("[LayoutManager] 레이아웃 관리자 초기화 완료")
            except Exception as _lm_err:
                logger.warning("[LayoutManager] 초기화 실패: %s", _lm_err)
        # ──────────────────────────────────────────────────────────────────────

        ui_error_view = QTextEdit()
        try:
            ui_error_view.setObjectName("ui_error_view")
            ui_error_view.setStyleSheet(
                "#ui_error_view { color: #FFCCBC; background-color: rgba(30, 10, 10, 0.85); }"
            )
        except Exception:
            pass
        try:
            ui_error_view.setAcceptRichText(False)
        except Exception:
            pass
        ui_error_view.setReadOnly(True)
        ui_error_view.setLineWrapMode(QTextEdit.WidgetWidth)
        try:
            ui_error_view.setMinimumHeight(80)
        except Exception:
            pass
        try:
            ui_error_view.hide()
        except Exception:
            pass
        right_root.addWidget(ui_error_view, stretch=1)

        log_view = QTextEdit()
        try:
            log_view.setObjectName("log_view")
            log_view.setStyleSheet(
                "#log_view { color: #EDEDED; background-color: rgba(10, 10, 10, 0.85); }"
            )
        except Exception:
            pass
        try:
            log_view.setAcceptRichText(True)
        except Exception:
            pass
        log_view.setReadOnly(True)
        log_view.setLineWrapMode(QTextEdit.NoWrap)
        try:
            log_view.setMinimumHeight(120)  # 220 → 120 줄임
        except Exception:
            pass
        right_root.addWidget(log_view, stretch=2)  # stretch=4 → 2 줄임

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        clock_lbl = QLabel("")
        clock_lbl.setMinimumWidth(80)
        clock_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_row.addWidget(clock_lbl)

        expiry_lbl = QLabel("")
        expiry_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        try:
            expiry_lbl.setMinimumWidth(90)
        except Exception:
            pass
        status_row.addWidget(expiry_lbl)

        tg_lbl = QLabel("TG: -")
        tg_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        try:
            tg_lbl.setMinimumWidth(80)
        except Exception:
            pass
        status_row.addWidget(tg_lbl)

        # 신규 피봇 상태 표시 (4종)
        aap_lbl = QLabel("AAP: -")
        aap_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        try:
            aap_lbl.setMinimumWidth(80)
        except Exception:
            pass
        try:
            aap_lbl.setStyleSheet("color: #4CAF50; font-weight: 600;")
        except Exception:
            pass
        status_row.addWidget(aap_lbl)
        self._aap_lbl = aap_lbl

        msb_lbl = QLabel("MSB: -")
        msb_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        try:
            msb_lbl.setMinimumWidth(80)
        except Exception:
            pass
        try:
            msb_lbl.setStyleSheet("color: #2196F3; font-weight: 600;")
        except Exception:
            pass
        status_row.addWidget(msb_lbl)
        self._msb_lbl = msb_lbl

        ktp_lbl = QLabel("KTP: -")
        ktp_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        try:
            ktp_lbl.setMinimumWidth(80)
        except Exception:
            pass
        try:
            ktp_lbl.setStyleSheet("color: #FF9800; font-weight: 600;")
        except Exception:
            pass
        status_row.addWidget(ktp_lbl)
        self._ktp_lbl = ktp_lbl

        int_lbl = QLabel("INT: -")
        int_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        try:
            int_lbl.setMinimumWidth(80)
        except Exception:
            pass
        try:
            int_lbl.setStyleSheet("color: #9C27B0; font-weight: 600;")
        except Exception:
            pass
        status_row.addWidget(int_lbl)
        self._int_lbl = int_lbl

        status_row.addWidget(QLabel(""), stretch=1)
        ver_lbl = QLabel(f"v{VERSION}")
        ver_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        try:
            from PySide6.QtGui import QFont

            f = QFont(ver_lbl.font())
            f.setPointSize(max(8, int(f.pointSize() or 9) - 1))
            ver_lbl.setFont(f)
        except Exception:
            pass
        try:
            ver_lbl.setStyleSheet("color: #9E9E9E;")
        except Exception:
            pass
        status_row.addWidget(ver_lbl)
        status_wrap = QWidget()
        status_wrap.setLayout(status_row)
        status_wrap.setMinimumHeight(clock_lbl.sizeHint().height() + 6)
        right_root.addWidget(status_wrap)

        try:
            if ui_init_errors:
                ui_error_view.setPlainText("\n".join([str(x) for x in ui_init_errors]))
                ui_error_view.show()
        except Exception:
            pass

        def _refresh_expiry_badge() -> None:
            try:
                now_dt = datetime.now()
                info = get_expiry_week_info(now_dt)
                is_expiry_week = bool((info or {}).get("is_expiry_week"))
                is_expiry_day = bool((info or {}).get("is_expiry_day"))
                exp_date = (info or {}).get("expiry_second_thursday")

                if is_expiry_day:
                    txt = "만기일"
                    style = "background-color: #B71C1C; color: #FFFFFF; padding: 2px 6px; border-radius: 6px; font-weight: 700;"
                elif is_expiry_week:
                    txt = "만기주"
                    style = "background-color: #D84315; color: #FFFFFF; padding: 2px 6px; border-radius: 6px; font-weight: 700;"
                else:
                    txt = "평시"
                    style = "background-color: rgba(120,120,120,0.35); color: #E0E0E0; padding: 2px 6px; border-radius: 6px;"

                expiry_lbl.setText(txt)
                try:
                    expiry_lbl.setStyleSheet(style)
                except Exception:
                    pass

                try:
                    tip = ""
                    if exp_date:
                        tip += f"만기일: {str(exp_date)}\n"
                    if is_expiry_week:
                        tip += "[추천 프리셋(시작점)]\n"
                        tip += "AST: multiplier_min/max ↑, smooth_period 3~5\n"
                        tip += "AZZ: confirmation_bars ↑, min_wave_bars ↑, pivot_threshold_min_pct ↑\n"
                        tip += "자세한 내용: ADAPTIVE_INDICATOR_GUIDE.md 8.7"
                    expiry_lbl.setToolTip(tip)
                except Exception:
                    pass
            except Exception:
                try:
                    expiry_lbl.setText("")
                except Exception:
                    pass

        def _tick_clock() -> None:
            try:
                clock_lbl.setText(QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss"))
            except Exception:
                pass

        _tick_clock()
        _refresh_expiry_badge()
        clock_timer = QTimer(w)
        clock_timer.timeout.connect(_tick_clock)
        clock_timer.start(1000)

        try:
            expiry_timer = QTimer(w)
            expiry_timer.timeout.connect(_refresh_expiry_badge)
            expiry_timer.start(30_000)
        except Exception:
            pass

        predictor_holder: Dict[str, Any] = {"predictor": None}

        try:
            from ebestapi.callbacks import get_gui_tick_stats
        except Exception:
            get_gui_tick_stats = None

        def _refresh_realtime_status() -> None:
            try:
                st = get_gui_tick_stats() if callable(get_gui_tick_stats) else {"counts": {}}
                m_fb = None
                try:
                    p = predictor_holder.get("predictor")
                    if p is not None and callable(getattr(p, "get_metrics", None)):
                        m_fb = p.get_metrics() or {}
                except Exception:
                    m_fb = None

                s = format_rt_status_line(
                    st if isinstance(st, dict) else None,
                    predictor_metrics=m_fb if isinstance(m_fb, dict) else None,
                )

                try:
                    b = telegram_bridge_holder.get("bridge")
                    n = getattr(b, "_notifier", None) if b is not None else None
                    if n is not None and callable(getattr(n, "get_send_count_total", None)):
                        try:
                            tg_lbl.setText(f"TG: {int(n.get_send_count_total() or 0)}")
                        except Exception:
                            pass
                    else:
                        try:
                            tg_lbl.setText("TG: -")
                        except Exception:
                            pass
                except Exception:
                    pass

                status_lbl.setText(s)
            except Exception:
                pass

            try:
                running = bool(self.task is not None and (not self.task.done()))
                pm_txt = "0"  # prediction_minutes는 현재 0으로 고정
                _nm_has_primary = use_transformer_chk.isChecked() or use_patch_tst_chk.isChecked()
                _nm_base = "patch_tst" if use_patch_tst_chk.isChecked() else "transformer"
                numeric_mode_txt = (
                    f"ensemble({_nm_base}+tft" + ("+mamba" if use_mamba_chk.isChecked() else "") + ")"
                    if (_nm_has_primary and use_tft_chk.isChecked())
                    else ("tft" if use_tft_chk.isChecked() else _nm_base)
                )

                m_panel = None
                try:
                    p = predictor_holder.get("predictor")
                    if p is not None and callable(getattr(p, "get_metrics", None)):
                        m_panel = p.get_metrics() or {}
                except Exception:
                    m_panel = None
                lines = predictor_metrics_summary_strings(m_panel if isinstance(m_panel, dict) else None)
                fc0_age_line = lines["fc0_age"]

                auto_ticks_path = "(auto)"
                try:
                    from core.utils import get_default_ticks_output_path

                    p = str(get_default_ticks_output_path() or "").strip()
                    if p:
                        if p.lower().endswith(".jsonl") and (not p.lower().endswith(".jsonl.gz")):
                            p = p + ".gz"
                        auto_ticks_path = p
                except Exception:
                    auto_ticks_path = "(auto)"
                
                # 옵션 의미가 상태 가져오기
                opt_sr_h = ""
                opt_sr_l = ""
                opt_sr_h_ts = ""
                opt_sr_l_ts = ""
                try:
                    tick_stats = get_gui_tick_stats()
                    opt_sr_h = tick_stats.get("opt_sr_h") or ""
                    opt_sr_l = tick_stats.get("opt_sr_l") or ""
                    opt_sr_h_ts = tick_stats.get("opt_sr_h_ts") or ""
                    opt_sr_l_ts = tick_stats.get("opt_sr_l_ts") or ""
                    logger.debug("[GuiController] 옵션 의미가 상태: opt_sr_h=%s, opt_sr_l=%s", opt_sr_h, opt_sr_l)
                except Exception as e:
                    logger.warning("[GuiController] 옵션 의미가 상태 가져오기 실패: %s", e)
                
                summary_lines = [
                    f"State: {'Running' if running else 'Idle'}",
                ]
                
                # 옵션 의미가 상태 추가 (없어도 표시)
                summary_lines.append("---")
                if opt_sr_h or opt_sr_l:
                    opt_lines = []
                    if opt_sr_h:
                        ts_str = opt_sr_h_ts[11:19] if opt_sr_h_ts else ""
                        opt_lines.append(f"의미가 H: {opt_sr_h} ({ts_str})")
                    if opt_sr_l:
                        ts_str = opt_sr_l_ts[11:19] if opt_sr_l_ts else ""
                        opt_lines.append(f"의미가 L: {opt_sr_l} ({ts_str})")
                    summary_lines.extend(opt_lines)
                else:
                    summary_lines.append("의미가: 없음")
                
                summary_text.setText(
                    "\n".join(summary_lines)
                )
            except Exception:
                pass

            try:
                m_stale = None
                try:
                    p = predictor_holder.get("predictor")
                    if p is not None and callable(getattr(p, "get_metrics", None)):
                        m_stale = p.get_metrics() or {}
                except Exception:
                    m_stale = None
                is_stale = fc0_is_stale(
                    stale_thr_edit_text=fc0_stale_thr_edit.text(),
                    effective_pred_defaults=self.effective_pred_defaults,
                    metrics=m_stale if isinstance(m_stale, dict) else None,
                )
                if is_stale:
                    summary_text.setStyleSheet("color: #FF8A80;")
                else:
                    summary_text.setStyleSheet("")
            except Exception:
                pass

        _refresh_realtime_status()
        rt_timer = QTimer(w)
        rt_timer.timeout.connect(_refresh_realtime_status)
        rt_timer.start(1000)

        def _append_log(s: str) -> None:
            append_log_rich(
                log_view,
                s,
                telegram_bridge_holder=telegram_bridge_holder,
                regime_led=regime_led,
                heur_led=heur_led,
                gpt_led=gpt_led,
                gem_led=gem_led,
                set_cons_arrow=_set_cons_arrow,
            )

        if emitter.qt is not None:
            emitter.qt.text.connect(_append_log)

        def _startup_internet_time_sync_gui() -> None:
            run_startup_internet_time_sync(_append_log, log=logger)

        try:
            QTimer.singleShot(0, _startup_internet_time_sync_gui)
        except Exception:
            pass

        def _choose_replay_file() -> None:
            p = open_replay_ticks_file_dialog(w, str(Path.cwd()))
            if p:
                replay_file_edit.setText(p)

        btn_replay_file.clicked.connect(_choose_replay_file)

        def _reload_config_effective_defaults() -> None:
            try:
                cfg = load_config("config.json")
                if not merge_prediction_effective_from_loaded_config(cfg, self.effective_pred_defaults):
                    return
            except Exception as e:
                try:
                    _append_log(f"Reload config failed: {e}")
                except Exception:
                    pass
                return

            try:
                _refresh_prediction_effective_labels()
            except Exception:
                pass
            try:
                _append_log("Reloaded config.json (effective defaults updated)")
            except Exception:
                pass

        try:
            reload_cfg_btn.clicked.connect(_reload_config_effective_defaults)
        except Exception:
            pass

        def _reset_adaptive_weights_clicked() -> None:
            try:
                p = predictor_holder.get("predictor")
                if p is None:
                    _append_log("Reset weights: predictor not initialized")
                    return
                fn = getattr(p, "reset_adaptive_weights", None)
                if not callable(fn):
                    _append_log("Reset weights: not supported by predictor")
                    return
                ok = bool(fn())
                _append_log("Reset weights: OK" if ok else "Reset weights: failed")
                try:
                    _refresh_realtime_status()
                except Exception:
                    pass
            except Exception:
                pass

        try:
            reset_weights_btn.clicked.connect(_reset_adaptive_weights_clicked)
        except Exception:
            pass

        # prediction_minutes 관련 함수 제외 (현재 0으로 고정)

        def _load_adaptive_fields_from_config(cfg_path: str) -> None:
            try:
                with open(str(cfg_path), "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

            ad = (data or {}).get("adaptive_indicator")
            if not isinstance(ad, dict):
                ad = {}

            # Load multiscale settings from prediction section
            pred = (data or {}).get("prediction")
            if not isinstance(pred, dict):
                pred = {}

            def _set_text(edit: Any, v: Any) -> None:
                try:
                    if v is None:
                        edit.setText("")
                    else:
                        edit.setText(str(v))
                except Exception:
                    pass

            # Multiscale checkboxes
            try:
                multiscale_5m_cb.setChecked(bool(pred.get("multiscale_5m", False)))
            except Exception:
                multiscale_5m_cb.setChecked(False)

            try:
                multiscale_15m_cb.setChecked(bool(pred.get("multiscale_enabled", False) and 15 in (pred.get("multiscale_time_scales") or [])))
            except Exception:
                multiscale_15m_cb.setChecked(False)

            _set_text(adapt_warmup_edit, ad.get("warmup_bars"))

            st = ad.get("supertrend")
            if not isinstance(st, dict):
                st = {}
            _set_text(ast_mult_min, st.get("multiplier_min"))
            _set_text(ast_mult_max, st.get("multiplier_max"))
            _set_text(ast_atr_min, st.get("atr_min_period"))
            _set_text(ast_atr_max, st.get("atr_max_period"))
            _set_text(ast_er_period, st.get("er_period"))
            _set_text(ast_adx_period, st.get("adx_period"))
            _set_text(ast_bb_period, st.get("bb_period"))
            _set_text(ast_bb_std, st.get("bb_std"))
            _set_text(ast_smooth, st.get("smooth_period"))
            try:
                v = st.get("use_bb_correction")
                if v is True:
                    ast_bb_correction_cb.setCurrentIndex(1)
                elif v is False:
                    ast_bb_correction_cb.setCurrentIndex(2)
                else:
                    ast_bb_correction_cb.setCurrentIndex(0)
            except Exception:
                pass

            zz = ad.get("zigzag")
            if not isinstance(zz, dict):
                zz = {}
            _set_text(azz_atr_mult, zz.get("atr_multiplier"))
            _set_text(azz_atr_period, zz.get("atr_period"))
            _set_text(azz_er_period, zz.get("er_period"))
            _set_text(azz_atr_mult_min, zz.get("atr_multiplier_min"))
            _set_text(azz_atr_mult_max, zz.get("atr_multiplier_max"))
            _set_text(azz_min_thr, zz.get("pivot_threshold_min_pct"))
            _set_text(azz_max_thr, zz.get("pivot_threshold_max_pct"))
            _set_text(azz_major_ratio, zz.get("major_swing_ratio"))
            _set_text(azz_confirm, zz.get("confirmation_bars"))
            _set_text(azz_max_swings, zz.get("max_swings"))
            _set_text(azz_min_wave_bars, zz.get("min_wave_bars"))
            _set_text(azz_min_wave_pct, zz.get("min_wave_pct"))
            _set_text(azz_cluster_tol, zz.get("cluster_tolerance_pct"))
            _set_text(azz_struct_lookback, zz.get("structure_lookback_swings"))
            _set_text(azz_struct_points, zz.get("structure_points"))
            try:
                v = zz.get("freeze_on_confirm")
                if v is True:
                    azz_freeze_on_confirm.setCurrentIndex(1)
                elif v is False:
                    azz_freeze_on_confirm.setCurrentIndex(2)
                else:
                    azz_freeze_on_confirm.setCurrentIndex(0)
            except Exception:
                try:
                    azz_freeze_on_confirm.setCurrentIndex(0)
                except Exception:
                    pass

        def _persist_adaptive_fields_to_config(cfg_path: str, config_obj: Any) -> None:
            try:
                with open(str(cfg_path), "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}

            try:
                ad = data.get("adaptive_indicator")
                if not isinstance(ad, dict):
                    ad = {}
                    data["adaptive_indicator"] = ad
            except Exception:
                ad = {}
                data["adaptive_indicator"] = ad

            # Prediction section for multiscale settings
            try:
                pred = data.get("prediction")
                if not isinstance(pred, dict):
                    pred = {}
                    data["prediction"] = pred
            except Exception:
                pred = {}
                data["prediction"] = pred

            try:
                st = ad.get("supertrend")
                if not isinstance(st, dict):
                    st = {}
                    ad["supertrend"] = st
            except Exception:
                st = {}
                ad["supertrend"] = st

            try:
                zz = ad.get("zigzag")
                if not isinstance(zz, dict):
                    zz = {}
                    ad["zigzag"] = zz
            except Exception:
                zz = {}
                ad["zigzag"] = zz

            # Save multiscale settings from GUI checkboxes
            try:
                pred["multiscale_5m"] = bool(multiscale_5m_cb.isChecked())
            except Exception:
                pass

            try:
                # Update multiscale_enabled and time_scales based on checkboxes
                multiscale_enabled = bool(multiscale_5m_cb.isChecked()) or bool(multiscale_15m_cb.isChecked())
                pred["multiscale_enabled"] = multiscale_enabled

                # Build time_scales list
                time_scales = [1]  # Always include 1-minute
                if multiscale_5m_cb.isChecked():
                    time_scales.append(5)
                if multiscale_15m_cb.isChecked():
                    time_scales.append(15)
                pred["multiscale_time_scales"] = time_scales
            except Exception:
                pass

            try:
                ad["warmup_bars"] = int(getattr(getattr(config_obj, "adaptive_indicator", None), "warmup_bars", ad.get("warmup_bars") or 0) or 0)
            except Exception:
                pass

            try:
                supertrend = getattr(getattr(config_obj, "adaptive_indicator", None), "supertrend", None)
                if supertrend is not None:
                    st["multiplier_min"] = float(getattr(supertrend, "multiplier_min", st.get("multiplier_min") or 0.0) or 0.0)
                    st["multiplier_max"] = float(getattr(supertrend, "multiplier_max", st.get("multiplier_max") or 0.0) or 0.0)
                    st["atr_min_period"] = int(getattr(supertrend, "atr_min_period", st.get("atr_min_period") or 0) or 0)
                    st["atr_max_period"] = int(getattr(supertrend, "atr_max_period", st.get("atr_max_period") or 0) or 0)
                    st["er_period"] = int(getattr(supertrend, "er_period", st.get("er_period") or 0) or 0)
                    st["adx_period"] = int(getattr(supertrend, "adx_period", st.get("adx_period") or 0) or 0)
                    st["use_bb_correction"] = bool(getattr(supertrend, "use_bb_correction", st.get("use_bb_correction")))
                    st["bb_period"] = int(getattr(supertrend, "bb_period", st.get("bb_period") or 0) or 0)
                    st["bb_std"] = float(getattr(supertrend, "bb_std", st.get("bb_std") or 0.0) or 0.0)
                    st["smooth_period"] = int(getattr(supertrend, "smooth_period", st.get("smooth_period") or 0) or 0)
            except Exception:
                pass

            try:
                zigzag = getattr(getattr(config_obj, "adaptive_indicator", None), "zigzag", None)
                if zigzag is not None:
                    zz["atr_multiplier"] = float(getattr(zigzag, "atr_multiplier", zz.get("atr_multiplier") or 0.0) or 0.0)
                    zz["atr_period"] = int(getattr(zigzag, "atr_period", zz.get("atr_period") or 0) or 0)
                    zz["er_period"] = int(getattr(zigzag, "er_period", zz.get("er_period") or 0) or 0)
                    zz["atr_multiplier_min"] = float(getattr(zigzag, "atr_multiplier_min", zz.get("atr_multiplier_min") or 0.0) or 0.0)
                    zz["atr_multiplier_max"] = float(getattr(zigzag, "atr_multiplier_max", zz.get("atr_multiplier_max") or 0.0) or 0.0)
                    zz["pivot_threshold_min_pct"] = float(getattr(zigzag, "pivot_threshold_min_pct", zz.get("pivot_threshold_min_pct") or 0.0) or 0.0)
                    zz["pivot_threshold_max_pct"] = float(getattr(zigzag, "pivot_threshold_max_pct", zz.get("pivot_threshold_max_pct") or 0.0) or 0.0)
                    zz["major_swing_ratio"] = float(getattr(zigzag, "major_swing_ratio", zz.get("major_swing_ratio") or 0.0) or 0.0)
                    zz["confirmation_bars"] = int(getattr(zigzag, "confirmation_bars", zz.get("confirmation_bars") or 0) or 0)
                    zz["max_swings"] = int(getattr(zigzag, "max_swings", zz.get("max_swings") or 0) or 0)
                    zz["freeze_on_confirm"] = bool(getattr(zigzag, "freeze_on_confirm", zz.get("freeze_on_confirm")))
                    zz["min_wave_bars"] = int(getattr(zigzag, "min_wave_bars", zz.get("min_wave_bars") or 0) or 0)
                    zz["min_wave_pct"] = float(getattr(zigzag, "min_wave_pct", zz.get("min_wave_pct") or 0.0) or 0.0)
                    zz["cluster_tolerance_pct"] = float(getattr(zigzag, "cluster_tolerance_pct", zz.get("cluster_tolerance_pct") or 0.0) or 0.0)
                    zz["structure_lookback_swings"] = int(getattr(zigzag, "structure_lookback_swings", zz.get("structure_lookback_swings") or 0) or 0)
                    zz["structure_points"] = int(getattr(zigzag, "structure_points", zz.get("structure_points") or 0) or 0)
            except Exception:
                pass

            try:
                with open(str(cfg_path), "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        def _persist_prediction_fields_to_config(cfg_path: str, config_obj: Any) -> None:
            try:
                with open(str(cfg_path), "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            try:
                pred = getattr(config_obj, "prediction", None)
            except Exception:
                pred = None
            if pred is None:
                return
            try:
                pred_dict = data.get("prediction")
                pred_dict = pred_dict if isinstance(pred_dict, dict) else {}
                data["prediction"] = pred_dict
            except Exception:
                pred_dict = {}
                data["prediction"] = pred_dict

            try:
                pred_dict["llm_min_interval_sec"] = float(getattr(pred, "llm_min_interval_sec", 30.0) or 0.0)
            except Exception:
                pass
            try:
                pred_dict["tick_size"] = float(getattr(pred, "tick_size", 0.05) or 0.0)
            except Exception:
                pass
            try:
                pred_dict["feedback_threshold_ticks"] = int(getattr(pred, "feedback_threshold_ticks", 10) or 10)
            except Exception:
                pass
            try:
                pred_dict["feedback_skip_hold_ticks"] = int(getattr(pred, "feedback_skip_hold_ticks", 2) or 0)
            except Exception:
                pass
            try:
                pred_dict["feedback_weight_high"] = float(getattr(pred, "feedback_weight_high", 1.0) or 0.0)
            except Exception:
                pass
            try:
                pred_dict["feedback_weight_mid"] = float(getattr(pred, "feedback_weight_mid", 0.5) or 0.0)
            except Exception:
                pass
            try:
                pred_dict["feedback_weight_low"] = float(getattr(pred, "feedback_weight_low", 0.25) or 0.0)
            except Exception:
                pass
            try:
                pred_dict["feedback_use_price_snapshot"] = bool(
                    getattr(pred, "feedback_use_price_snapshot", True)
                )
            except Exception:
                pass
            try:
                pred_dict["feedback_snapshot_tolerance_sec"] = float(
                    getattr(pred, "feedback_snapshot_tolerance_sec", 30.0) or 0.0
                )
            except Exception:
                pass
            try:
                pred_dict["feedback_snapshot_required"] = bool(
                    getattr(pred, "feedback_snapshot_required", False)
                )
            except Exception:
                pass
            try:
                pred_dict["fc0_stale_threshold_sec"] = float(
                    getattr(pred, "fc0_stale_threshold_sec", 10.0) or 0.0
                )
            except Exception:
                pass
            try:
                pred_dict["fc0_stale_cooldown_sec"] = float(
                    getattr(pred, "fc0_stale_cooldown_sec", 60.0) or 0.0
                )
            except Exception:
                pass

            try:
                with open(str(cfg_path), "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        class AdaptiveSettingsBinder:
            def apply_to_config(self, config: Any) -> None:
                _apply_gui_to_config_adaptive(config)

            def persist_to_config(self, cfg_path: str, config_obj: Any) -> None:
                _persist_adaptive_fields_to_config(cfg_path, config_obj)

        class PredictionSettingsBinder:
            def apply_to_config(self, config: Any) -> None:
                def _apply_float(line_edit: Any, setter, *, min_v: Optional[float] = None, max_v: Optional[float] = None) -> None:
                    try:
                        v = line_edit.text().strip()
                        if v:
                            x = float(v)
                            if min_v is not None and x < float(min_v):
                                x = float(min_v)
                            if max_v is not None and x > float(max_v):
                                x = float(max_v)
                            setter(x)
                    except Exception:
                        pass

                def _apply_int(line_edit: Any, setter, *, min_v: Optional[int] = None, max_v: Optional[int] = None) -> None:
                    try:
                        v = line_edit.text().strip()
                        if v:
                            x = int(float(v))
                            if min_v is not None and x < int(min_v):
                                x = int(min_v)
                            if max_v is not None and x > int(max_v):
                                x = int(max_v)
                            setter(x)
                    except Exception:
                        pass

                def _apply_bool_combo(combo: Any, setter) -> None:
                    try:
                        v = combo.currentData()
                        if v is not None:
                            setter(bool(v))
                    except Exception:
                        pass

                try:
                    pred = getattr(config, "prediction", None)
                except Exception:
                    pred = None
                if pred is None:
                    return

                _apply_float(llm_min_interval_edit, lambda x: setattr(pred, "llm_min_interval_sec", x), min_v=0.0)
                _apply_float(tick_size_edit, lambda x: setattr(pred, "tick_size", x), min_v=1e-9)
                _apply_int(feedback_ticks_edit, lambda x: setattr(pred, "feedback_threshold_ticks", x), min_v=1)
                _apply_int(feedback_skip_hold_ticks_edit, lambda x: setattr(pred, "feedback_skip_hold_ticks", x), min_v=0)
                _apply_float(feedback_weight_high_edit, lambda x: setattr(pred, "feedback_weight_high", x), min_v=0.0)
                _apply_float(feedback_weight_mid_edit, lambda x: setattr(pred, "feedback_weight_mid", x), min_v=0.0)
                _apply_float(feedback_weight_low_edit, lambda x: setattr(pred, "feedback_weight_low", x), min_v=0.0)
                _apply_bool_combo(feedback_use_snapshot_cb, lambda x: setattr(pred, "feedback_use_price_snapshot", x))
                _apply_float(feedback_snapshot_tol_edit, lambda x: setattr(pred, "feedback_snapshot_tolerance_sec", x), min_v=0.0)
                _apply_bool_combo(feedback_snapshot_required_cb, lambda x: setattr(pred, "feedback_snapshot_required", x))
                _apply_float(fc0_stale_thr_edit, lambda x: setattr(pred, "fc0_stale_threshold_sec", x), min_v=0.0)
                _apply_float(fc0_stale_cool_edit, lambda x: setattr(pred, "fc0_stale_cooldown_sec", x), min_v=0.0)

            def persist_to_config(self, cfg_path: str, config_obj: Any) -> None:
                _persist_prediction_fields_to_config(cfg_path, config_obj)

            def reload_effective_defaults(self) -> None:
                _reload_config_effective_defaults()

        adaptive_binder = AdaptiveSettingsBinder()
        prediction_binder = PredictionSettingsBinder()

        try:
            _load_adaptive_fields_from_config("config.json")
        except Exception:
            pass

        # prediction_minutes 관련 연결 제외 (현재 0으로 고정)

        def _apply_gui_to_config_adaptive(config: Any) -> None:
            def _apply_int(line_edit: Any, setter, *, min_v: Optional[int] = None, max_v: Optional[int] = None) -> None:
                try:
                    v = line_edit.text().strip()
                    if v:
                        x = int(v)
                        if min_v is not None and x < int(min_v):
                            x = int(min_v)
                        if max_v is not None and x > int(max_v):
                            x = int(max_v)
                        setter(x)
                except Exception:
                    pass

            def _apply_float(line_edit: Any, setter, *, min_v: Optional[float] = None, max_v: Optional[float] = None) -> None:
                try:
                    v = line_edit.text().strip()
                    if v:
                        x = float(v)
                        if min_v is not None and x < float(min_v):
                            x = float(min_v)
                        if max_v is not None and x > float(max_v):
                            x = float(max_v)
                        setter(x)
                except Exception:
                    pass

            def _apply_bool_combo(combo: Any, setter) -> None:
                try:
                    v = combo.currentData()
                    if v is not None:
                        setter(bool(v))
                except Exception:
                    pass

            def _apply_float_min_max(
                min_edit: Any,
                max_edit: Any,
                set_min,
                set_max,
                *,
                min_v: Optional[float] = None,
                max_v: Optional[float] = None,
            ) -> None:
                try:
                    a_s = min_edit.text().strip()
                    b_s = max_edit.text().strip()
                    if a_s and b_s:
                        a = float(a_s)
                        b = float(b_s)
                        if min_v is not None:
                            a = max(float(min_v), a)
                            b = max(float(min_v), b)
                        if max_v is not None:
                            a = min(float(max_v), a)
                            b = min(float(max_v), b)
                        if a > b:
                            a, b = b, a
                        set_min(a)
                        set_max(b)
                        return
                except Exception:
                    pass
                _apply_float(min_edit, set_min, min_v=min_v, max_v=max_v)
                _apply_float(max_edit, set_max, min_v=min_v, max_v=max_v)

            def _apply_int_min_max(
                min_edit: Any,
                max_edit: Any,
                set_min,
                set_max,
                *,
                min_v: Optional[int] = None,
                max_v: Optional[int] = None,
            ) -> None:
                try:
                    a_s = min_edit.text().strip()
                    b_s = max_edit.text().strip()
                    if a_s and b_s:
                        a = int(a_s)
                        b = int(b_s)
                        if min_v is not None:
                            a = max(int(min_v), a)
                            b = max(int(min_v), b)
                        if max_v is not None:
                            a = min(int(max_v), a)
                            b = min(int(max_v), b)
                        if a > b:
                            a, b = b, a
                        set_min(a)
                        set_max(b)
                        return
                except Exception:
                    pass
                _apply_int(min_edit, set_min, min_v=min_v, max_v=max_v)
                _apply_int(max_edit, set_max, min_v=min_v, max_v=max_v)

            try:
                adaptive = getattr(config, "adaptive_indicator", None)
                if adaptive is None:
                    return

                # Apply multiscale settings to prediction config
                try:
                    prediction = getattr(config, "prediction", None)
                    if prediction is not None:
                        # Apply multiscale_5m
                        setattr(prediction, "multiscale_5m", bool(multiscale_5m_cb.isChecked()))
                        # Apply multiscale_enabled and time_scales
                        multiscale_enabled = bool(multiscale_5m_cb.isChecked()) or bool(multiscale_15m_cb.isChecked())
                        setattr(prediction, "multiscale_enabled", multiscale_enabled)
                        # Build time_scales list
                        time_scales = [1]
                        if multiscale_5m_cb.isChecked():
                            time_scales.append(5)
                        if multiscale_15m_cb.isChecked():
                            time_scales.append(15)
                        setattr(prediction, "multiscale_time_scales", time_scales)
                except Exception:
                    pass

                supertrend = getattr(adaptive, "supertrend", None)
                zigzag = getattr(adaptive, "zigzag", None)

                _apply_int(adapt_warmup_edit, lambda x: setattr(adaptive, "warmup_bars", x), min_v=0)

                if supertrend is not None:
                    _apply_float_min_max(
                        ast_mult_min,
                        ast_mult_max,
                        lambda x: setattr(supertrend, "multiplier_min", x),
                        lambda x: setattr(supertrend, "multiplier_max", x),
                        min_v=0.0,
                    )
                    _apply_int_min_max(
                        ast_atr_min,
                        ast_atr_max,
                        lambda x: setattr(supertrend, "atr_min_period", x),
                        lambda x: setattr(supertrend, "atr_max_period", x),
                        min_v=1,
                    )
                    _apply_int(ast_er_period, lambda x: setattr(supertrend, "er_period", x), min_v=1)
                    _apply_int(ast_adx_period, lambda x: setattr(supertrend, "adx_period", x), min_v=1)
                    _apply_bool_combo(ast_bb_correction_cb, lambda x: setattr(supertrend, "use_bb_correction", x))
                    _apply_int(ast_bb_period, lambda x: setattr(supertrend, "bb_period", x), min_v=1)
                    _apply_float(ast_bb_std, lambda x: setattr(supertrend, "bb_std", x), min_v=0.0)
                    _apply_int(ast_smooth, lambda x: setattr(supertrend, "smooth_period", x), min_v=1)

                if zigzag is not None:
                    _apply_float(azz_atr_mult, lambda x: setattr(zigzag, "atr_multiplier", x), min_v=0.0)
                    _apply_int(azz_atr_period, lambda x: setattr(zigzag, "atr_period", x), min_v=1)
                    _apply_int(azz_er_period, lambda x: setattr(zigzag, "er_period", x), min_v=1)

                    _apply_float_min_max(
                        azz_atr_mult_min,
                        azz_atr_mult_max,
                        lambda x: setattr(zigzag, "atr_multiplier_min", x),
                        lambda x: setattr(zigzag, "atr_multiplier_max", x),
                        min_v=0.0,
                    )
                    _apply_float_min_max(
                        azz_min_thr,
                        azz_max_thr,
                        lambda x: setattr(zigzag, "pivot_threshold_min_pct", x),
                        lambda x: setattr(zigzag, "pivot_threshold_max_pct", x),
                        min_v=0.0,
                        max_v=100.0,
                    )

                    _apply_float(azz_major_ratio, lambda x: setattr(zigzag, "major_swing_ratio", x), min_v=1.0)
                    _apply_int(azz_confirm, lambda x: setattr(zigzag, "confirmation_bars", x), min_v=0)
                    _apply_int(azz_max_swings, lambda x: setattr(zigzag, "max_swings", x), min_v=4)
                    _apply_int(azz_min_wave_bars, lambda x: setattr(zigzag, "min_wave_bars", x), min_v=0)
                    _apply_float(azz_min_wave_pct, lambda x: setattr(zigzag, "min_wave_pct", x), min_v=0.0, max_v=100.0)
                    _apply_float(
                        azz_cluster_tol,
                        lambda x: setattr(zigzag, "cluster_tolerance_pct", x),
                        min_v=0.0,
                        max_v=100.0,
                    )
                    _apply_int(azz_struct_lookback, lambda x: setattr(zigzag, "structure_lookback_swings", x), min_v=4)
                    _apply_int(azz_struct_points, lambda x: setattr(zigzag, "structure_points", x), min_v=2)
                    _apply_bool_combo(azz_freeze_on_confirm, lambda x: setattr(zigzag, "freeze_on_confirm", x))
            except Exception:
                pass

        async def _run_pipeline() -> None:
            cfg_path = "config.json"
            try:
                _append_log("[RUN] pipeline starting")
            except Exception:
                pass
            try:
                config = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: load_config(cfg_path)
                )
            except Exception as e:
                _append_log(f"Failed to load config: {e}")
                run_state_lbl.setText("Config error")
                return

            # 추천 파라미터 로드 (DB에서 레짐별 최적 파라미터)
            try:
                recommended_params = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: load_recommended_params(symbol="KP200 선물", regime="unknown")
                )
                if recommended_params and recommended_params.get("_source") == "db":
                    # 추천 파라미터를 config에 적용
                    # HAP config 경로에 적용 (HybridAdaptivePivot 전용 필드)
                    try:
                        if hasattr(config, 'adaptive_indicator') and hasattr(config.adaptive_indicator, 'hap'):
                            hap = config.adaptive_indicator.hap
                            if 'atr_multiplier' in recommended_params:
                                hap.base_multiplier = recommended_params['atr_multiplier']
                            if 'base_pct' in recommended_params:
                                hap.base_pct = recommended_params['base_pct']
                            if 'atr_weight' in recommended_params:
                                hap.atr_weight = recommended_params['atr_weight']
                            if 'confirmation_bars' in recommended_params:
                                hap.confirmation_bars = recommended_params['confirmation_bars']
                            if 'er_period' in recommended_params:
                                hap.er_period = recommended_params['er_period']
                            if 'min_wave_pct' in recommended_params:
                                hap.min_wave_pct = recommended_params['min_wave_pct']
                            _append_log(f"[RUN] 추천 파라미터 적용 (HAP): score={recommended_params.get('_composite_score', 0):.3f}")
                    except Exception as e:
                        _append_log(f"[RUN] 추천 파라미터 적용 실패: {e}")
            except Exception as e:
                _append_log(f"[RUN] 추천 파라미터 로드 실패: {e}")

            bridge = None

            adaptive_binder.apply_to_config(config)
            prediction_binder.apply_to_config(config)

            try:
                _cfg_path_p = str(cfg_path)
                _config_p   = config
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: adaptive_binder.persist_to_config(_cfg_path_p, _config_p)
                )
            except Exception:
                pass

            try:
                _cfg_path_p2 = str(cfg_path)
                _config_p2   = config
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: prediction_binder.persist_to_config(_cfg_path_p2, _config_p2)
                )
            except Exception:
                pass

            pm = 0  # prediction_minutes는 현재 0으로 고정
            args = _make_args_from_gui(
                config_path=cfg_path,
                log_level=log_level_cb.currentText().strip() or "INFO",
                log_file=log_file_edit.text().strip() or DEFAULT_LOG_FILE,
                prediction_minutes=pm,
                heuristic_only=False,
                no_ebest_live=False,
                duration_sec=0,
                include_options=True,
                option_month=None,
            )

            try:
                args.no_save_ticks = False
                args.compress_ticks = True
                args.out_ticks = None
            except Exception:
                pass

            if args.prediction_minutes is not None:
                config.prediction.minutes = args.prediction_minutes
            if args.heuristic_only:
                config.prediction.use_llm = False
            if args.log_level:
                config.log_level = args.log_level
            if args.log_file:
                config.log_file = args.log_file

            log_level = getattr(logging, str(config.log_level).upper(), logging.INFO)
            setup_logging(
                log_file=config.log_file,
                level=log_level,
                enable_tee=True,
            )
            try:
                qt_handler.setLevel(int(log_level))
                qt_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
                logging.getLogger().addHandler(qt_handler)
            except Exception:
                pass
            logger = get_logger()  # setup_logging 완료 후 핸들러가 재구성된 logger 재취득
            try:
                if qt_handler not in getattr(logger, "handlers", []):
                    logger.addHandler(qt_handler)
            except Exception:
                pass

            try:
                logging.getLogger("telegram_notifier").setLevel(logging.INFO)
            except Exception:
                pass

            try:
                tg_debug = str(os.environ.get("TELEGRAM_DEBUG") or "").strip().lower() not in ("", "0", "false", "no")
            except Exception:
                tg_debug = False

            use_transformer = bool(use_transformer_chk.isChecked())
            use_tft        = bool(use_tft_chk.isChecked())
            use_patch_tst  = bool(use_patch_tst_chk.isChecked())
            use_mamba      = bool(use_mamba_chk.isChecked())
            # PatchTST가 체크되면 Transformer 계열 모델 구조를 patch_tst로 대체
            _model_class   = "patch_tst" if use_patch_tst else "transformer"
            # 수치 예측의 "primary" 역할: Transformer 또는 PatchTST 중 하나
            _has_primary   = use_transformer or use_patch_tst
            if (not _has_primary) and (not use_tft):
                _append_log("Invalid selection: Transformer/PatchTST/TFT 중 하나 이상 활성화하세요")
                run_state_lbl.setText("Config error")
                return
            # GUI 선택을 config에 즉시 반영 → pipeline_builder가 읽는 값에 우선 적용
            try:
                config.prediction.model_class   = _model_class
                config.prediction.mamba_enabled = use_mamba
            except Exception:
                pass

            display_startup_info(config, args, logger)

            selected_transformer_path = None
            selected_tft_path = None
            try:
                if _has_primary or use_tft:
                    from prediction.weights_selector import select_weights_for_datetime

                    _now_cap = datetime.now()
                    sel = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: select_weights_for_datetime(now=_now_cap)
                    )
                    if _has_primary:
                        selected_transformer_path = sel.transformer_path
                    if use_tft:
                        selected_tft_path = sel.tft_path
            except Exception:
                selected_transformer_path = None
                selected_tft_path = None

            # numeric_mode: TFT 포함 여부로 ensemble 결정
            numeric_mode = "ensemble" if (_has_primary and use_tft) else ("tft" if use_tft else "transformer")
            _tw_override = (
                float(getattr(config.prediction, "transformer_weight", 0.5) or 0.5)
                if numeric_mode == "ensemble"
                else 1.0
            )
            # primary(Transformer or PatchTST) / TFT 중 하나만 선택 시 None 처리
            _t_path = selected_transformer_path if _has_primary else None
            _f_path = selected_tft_path if use_tft else None

            # _build_pipeline은 동기 블로킹 함수(모델 로드 포함).
            # Qt 이벤트 루프를 멈추지 않도록 executor에서 실행한다.
            try:
                _append_log("[RUN] predictor 초기화 중...")
            except Exception:
                pass
            _t_path_cap = _t_path
            _f_path_cap = _f_path
            _nm_cap     = numeric_mode
            _tw_cap     = _tw_override
            _config_cap = config
            _args_cap   = args
            # notifier는 아래 telegram 초기화 블록에서 생성되므로
            # predictor 생성 전 미리 notifier를 만들어 주입한다.
            _notifier_for_pipeline = None
            try:
                _tg_en = bool(telegram_enable_chk.isChecked()) or bool(
                    getattr(getattr(config, "telegram", None), "enabled", False)
                )
                if _tg_en:
                    try:
                        _sp = os.environ.get("APP_SECRETS_CONFIG") or str(
                            Path(str(cfg_path or "config.json")).parent / "config.secrets.json"
                        )
                    except Exception:
                        _sp = os.environ.get("APP_SECRETS_CONFIG") or "config.secrets.json"
                    _pre_notifier = create_notifier_from_config(str(_sp))
                    if bool(getattr(_pre_notifier, "is_configured", False)):
                        _notifier_for_pipeline = _pre_notifier
            except Exception:
                pass
            _notifier_cap = _notifier_for_pipeline
            predictor = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _build_pipeline(
                    _config_cap,
                    _args_cap,
                    transformer_weights_path=_t_path_cap,
                    tft_weights_path=_f_path_cap,
                    numeric_predictor_override=_nm_cap,
                    transformer_weight_override=_tw_cap,
                    notifier=_notifier_cap,
                ),
            )
            try:
                _append_log("[RUN] predictor 초기화 완료")
            except Exception:
                pass
            # ── 차트 뷰어에 predictor 연결 ────────────────────────────────────
            try:
                if self._chart_viewer is not None:
                    self._chart_viewer.set_predictor(predictor)
            except Exception as _cv_e:
                try:
                    logger.debug("[ChartViewer] set_predictor 실패: %s", _cv_e)
                except Exception:
                    pass
            # ── Summary 패널용 predictor_holder 등록 ──────────────────────────
            try:
                predictor_holder["predictor"] = predictor
            except Exception:
                pass
            # ─────────────────────────────────────────────────────────────────
            try:
                tg_enabled = False
                try:
                    tg_enabled = bool(telegram_enable_chk.isChecked()) or bool(
                        getattr(getattr(config, "telegram", None), "enabled", False)
                    )
                except Exception:
                    tg_enabled = bool(telegram_enable_chk.isChecked())

                if tg_debug:
                    try:
                        _append_log(f"[TELEGRAM] enabled={tg_enabled} (gui_chk={bool(telegram_enable_chk.isChecked())})")
                    except Exception:
                        pass
                    try:
                        logger.info("[TELEGRAM] enabled=%s (gui_chk=%s)", tg_enabled, bool(telegram_enable_chk.isChecked()))
                    except Exception:
                        pass

                if tg_enabled:
                    try:
                        cfg_path_obj = Path(str(cfg_path or "config.json"))
                        secrets_path = os.environ.get("APP_SECRETS_CONFIG") or str(
                            cfg_path_obj.parent / "config.secrets.json"
                        )
                    except Exception:
                        secrets_path = os.environ.get("APP_SECRETS_CONFIG") or "config.secrets.json"

                    if tg_debug:
                        try:
                            _append_log(f"[TELEGRAM] secrets_path={secrets_path}")
                        except Exception:
                            pass
                        try:
                            logger.info("[TELEGRAM] secrets_path=%s", str(secrets_path))
                        except Exception:
                            pass

                    notifier = create_notifier_from_config(str(secrets_path))
                    is_cfg = bool(getattr(notifier, "is_configured", False))

                    if tg_debug:
                        try:
                            _append_log(f"[TELEGRAM] notifier_configured={is_cfg}")
                        except Exception:
                            pass
                        try:
                            logger.info("[TELEGRAM] notifier_configured=%s", is_cfg)
                        except Exception:
                            pass
                    if not is_cfg:
                        _append_log("[TELEGRAM] 토큰/채팅ID 미설정 — 브리지 생략")

                    if is_cfg:
                        # main.py 와 동일: ebest_live 예측 주기(prediction_minutes)와 브리지 간격 동기화.
                        _bridge_interval = 300.0
                        _pred_min_gui = 5
                        try:
                            _pred_min_gui = int(
                                getattr(getattr(config, "prediction", None), "minutes", 5) or 5
                            )
                            # prediction_minutes가 0이면 예측 전송 비활성화 (아주 큰 간격 설정)
                            if _pred_min_gui <= 0:
                                _bridge_interval = 999999.0
                            else:
                                _bridge_interval = float(max(60, _pred_min_gui * 60))
                        except Exception:
                            pass
                        try:
                            _append_log(
                                f"[TELEGRAM] predict_interval_sec={int(_bridge_interval)} "
                                f"(prediction.minutes={_pred_min_gui})"
                            )
                        except Exception:
                            pass
                        bridge = PipelineTelegramBridge(
                            predictor,
                            notifier,
                            predict_interval_sec=_bridge_interval,
                            only_consensus=True,
                            only_actionable=False,
                        )
                        bridge.start()

                        # TradeExecutionGate 설정 주입 (config.trade_gate 존재 시)
                        try:
                            tg_cfg = getattr(config, "trade_gate", None)
                            if tg_cfg is not None:
                                bridge.set_trade_gate_config(tg_cfg)
                        except Exception as _tg_e:
                            _append_log(f"[TRADE_GATE] 설정 주입 실패: {_tg_e}")

                        try:
                            bridge.start_polling()
                        except Exception as _poll_e:
                            _append_log(f"[TELEGRAM] start_polling failed: {_poll_e}")
                        try:
                            telegram_bridge_holder["bridge"] = bridge
                        except Exception:
                            pass
            except Exception as _e:
                try:
                    _append_log(f"[TELEGRAM] bridge init failed: {_e}")
                except Exception:
                    pass

            try:
                from ebestapi.live import install_ebest_zz_confirm_telegram_hook, install_ebest_zz_candidate_telegram_hook
                install_ebest_zz_confirm_telegram_hook(
                    predictor,
                    telegram_bridge_holder.get("bridge"),
                )
                install_ebest_zz_candidate_telegram_hook(
                    predictor,
                    telegram_bridge_holder.get("bridge"),
                )
            except Exception:
                pass

            run_state_lbl.setText("Running")

            try:
                if not args.no_ebest_live:
                    now_dt = datetime.now()
                    if not is_market_open(now_dt):
                        policy = "warn"
                        next_open = next_market_open(now_dt)
                        wait_sec = max(0, int((next_open - now_dt).total_seconds()))
                        msg = (
                            f"[MARKET] outside market hours now={now_dt.strftime('%Y-%m-%d %H:%M:%S')} "
                            f"next_open={next_open.strftime('%Y-%m-%d %H:%M:%S')} (wait {wait_sec}s) policy={policy}"
                        )
                        # _append_log만 쓰면 타임스탬프 없이 GUI/콘솔에 섞임 → logging으로 통일
                        try:
                            logger.info("%s", msg)
                        except Exception:
                            _append_log(msg)
                        try:
                            run_state_lbl.setText("Market closed")
                        except Exception:
                            pass

                        if policy == "exit":
                            return

                        if policy == "wait" and wait_sec > 0:
                            remaining = int(wait_sec)
                            while remaining > 0:
                                await asyncio.sleep(min(30, remaining))
                                remaining -= 30
                                try:
                                    run_state_lbl.setText(f"Waiting for market open... ({max(0, remaining)}s)")
                                except Exception:
                                    pass

                            try:
                                logger.info("[MARKET] market open wait completed; starting live mode")
                            except Exception:
                                _append_log("[MARKET] market open wait completed; starting live mode")
                            try:
                                run_state_lbl.setText("Running")
                            except Exception:
                                pass

                    res = await run_live_mode(config, args, predictor)
                else:
                    res = run_simple_prediction(predictor, args)
                if isinstance(res, dict) and (not res):
                    _append_log("Run completed (no result payload)")
                else:
                    res_json = json.dumps(res, ensure_ascii=False, indent=2)
                    _append_log(res_json)
                    try:
                        logger.info("\n\n" + res_json)
                    except Exception:
                        pass
                run_state_lbl.setText("Done")
            except asyncio.CancelledError:
                run_state_lbl.setText("Cancelled")
            except Exception as e:
                _append_log(f"Run failed: {e}")
                run_state_lbl.setText("Error")
            finally:
                try:
                    if self._chart_viewer is not None:
                        self._chart_viewer.stop()
                except Exception:
                    pass
                try:
                    if bridge is not None:
                        bridge.stop()
                        try:
                            getattr(bridge, "_notifier", None).stop_polling()
                        except Exception:
                            pass
                    try:
                        if telegram_bridge_holder.get("bridge") is bridge:
                            telegram_bridge_holder["bridge"] = None
                    except Exception:
                        pass
                except Exception:
                    pass
                self.task = None
                try:
                    predictor_holder["predictor"] = None
                except Exception:
                    pass
                try:
                    toggle_btn.blockSignals(True)
                    toggle_btn.setChecked(False)
                    toggle_btn.setText("Start")
                finally:
                    toggle_btn.blockSignals(False)

        async def _run_replay() -> None:
            replay_path = replay_file_edit.text().strip()
            if not replay_path:
                _append_log("Replay file is empty")
                return

            self.replay_pause_event = threading.Event()
            self.replay_stop_event = threading.Event()
            self.replay_paused = False
            try:
                replay_btn.setText("Pause")
            except Exception:
                pass

            cfg_path = "config.json"
            try:
                config = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: load_config(cfg_path)
                )
            except Exception as e:
                _append_log(f"Failed to load config: {e}")
                run_state_lbl.setText("Config error")
                return

            _apply_gui_to_config_adaptive(config)

            try:
                _persist_adaptive_fields_to_config(str(cfg_path), config)
            except Exception:
                pass

            log_level = getattr(logging, str(log_level_cb.currentText() or "INFO").upper(), logging.INFO)
            try:
                setup_logging(
                    log_file=str(log_file_edit.text().strip() or DEFAULT_LOG_FILE),
                    level=log_level,
                    enable_tee=True,
                )
            except Exception:
                pass
            logger = get_logger()  # setup_logging 완료 후 핸들러가 재구성된 logger 재취득
            try:
                log_ai_provider_keys_loaded(config.ai_providers, log_to=logger)
            except Exception:
                pass
            selected_tft_path = None
            use_transformer = bool(use_transformer_chk.isChecked())
            use_tft        = bool(use_tft_chk.isChecked())
            use_patch_tst  = bool(use_patch_tst_chk.isChecked())
            use_mamba      = bool(use_mamba_chk.isChecked())
            _model_class   = "patch_tst" if use_patch_tst else "transformer"
            _has_primary   = use_transformer or use_patch_tst
            if (not _has_primary) and (not use_tft):
                _append_log("Invalid selection: Transformer/PatchTST/TFT 중 하나 이상 활성화하세요")
                status_lbl.setText("Config error")
                return
            try:
                config.prediction.model_class   = _model_class
                config.prediction.mamba_enabled = use_mamba
            except Exception:
                pass

            try:
                if _has_primary or use_tft:
                    from prediction.weights_selector import select_weights_for_datetime

                    _now_cap = datetime.now()
                    sel = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: select_weights_for_datetime(now=_now_cap)
                    )
                    if _has_primary:
                        selected_transformer_path = sel.transformer_path
                    if use_tft:
                        selected_tft_path = sel.tft_path
            except Exception:
                selected_transformer_path = None
                selected_tft_path = None

            numeric_mode = "ensemble" if (_has_primary and use_tft) else ("tft" if use_tft else "transformer")
            _tw_override = (
                float(getattr(config.prediction, "transformer_weight", 0.5) or 0.5)
                if numeric_mode == "ensemble"
                else 1.0
            )
            _t_path = selected_transformer_path if _has_primary else None
            _f_path = selected_tft_path if use_tft else None

            # _run_replay는 _run_pipeline과 달리 args를 생성하지 않으므로
            # _build_pipeline이 참조하는 모든 args 필드를 포함한 Namespace를 직접 생성한다.
            replay_args = _make_args_from_gui(
                config_path=cfg_path,
                log_level=str(log_level_cb.currentText().strip() or "INFO"),
                log_file=str(log_file_edit.text().strip() or DEFAULT_LOG_FILE),
                prediction_minutes=0,  # prediction_minutes는 현재 0으로 고정
                heuristic_only=False,
                no_ebest_live=True,
                duration_sec=0,
                include_options=True,
                option_month=None,
                replay_speed=float(replay_speed_edit.text().strip() or 0.0),
                replay_max_lines=(int(replay_max_lines_edit.text().strip()) if replay_max_lines_edit.text().strip() else None),
            )

            # _build_pipeline 블로킹 → executor 비동기 실행
            try:
                _append_log("[REPLAY] predictor 초기화 중...")
            except Exception:
                pass
            _t_path_r  = _t_path
            _f_path_r  = _f_path
            _nm_r      = numeric_mode
            _tw_r      = _tw_override
            _config_r  = config
            _rargs_r   = replay_args
            # replay 모드는 telegram bridge 없이 실행되므로 notifier=None
            predictor = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _build_pipeline(
                    _config_r,
                    _rargs_r,
                    transformer_weights_path=_t_path_r,
                    tft_weights_path=_f_path_r,
                    numeric_predictor_override=_nm_r,
                    transformer_weight_override=_tw_r,
                    notifier=None,
                ),
            )
            try:
                _append_log("[REPLAY] predictor 초기화 완료")
            except Exception:
                pass
            # ── 차트 뷰어에 predictor 연결 (replay) ─────────────────────────
            try:
                if self._chart_viewer is not None:
                    self._chart_viewer.set_predictor(predictor)
            except Exception as _cv_e:
                try:
                    logger.debug("[ChartViewer] set_predictor 실패(replay): %s", _cv_e)
                except Exception:
                    pass
            # ── Summary 패널용 predictor_holder 등록 (replay) ─────────────────
            try:
                predictor_holder["predictor"] = predictor
            except Exception:
                pass
            # ─────────────────────────────────────────────────────────────────

            # Parse replay settings
            try:
                speed = float(replay_speed_edit.text().strip() or 0.0)
            except Exception:
                speed = 0.0
            max_lines = None
            try:
                v = replay_max_lines_edit.text().strip()
                if v:
                    max_lines = int(v)
            except Exception:
                max_lines = None

            run_state_lbl.setText("Replaying")
            _append_log(f"[REPLAY] file={replay_path} speed={speed} max_lines={max_lines}")

            try:
                rc = await asyncio.to_thread(
                    run_replay_mode_with_predictor,
                    replay_path,
                    predictor,
                    speed=float(speed),
                    max_lines=max_lines,
                    pause_event=self.replay_pause_event,
                    stop_event=self.replay_stop_event,
                )
                _append_log(f"[REPLAY] done rc={rc}")
                run_state_lbl.setText("Done")
            except asyncio.CancelledError:
                run_state_lbl.setText("Cancelled")
            except Exception as e:
                _append_log(f"Replay failed: {e}")
                run_state_lbl.setText("Error")
            finally:
                self.task = None
                self.replay_pause_event = None
                self.replay_stop_event = None
                self.replay_paused = False
                try:
                    replay_btn.setText("Replay")
                except Exception:
                    pass

        @asyncSlot(bool)
        async def _on_toggle(checked: bool) -> None:
            if checked:
                try:
                    toggle_btn.setText("Stop")
                except Exception:
                    pass
                try:
                    self.task = asyncio.create_task(_run_pipeline())
                except Exception:
                    try:
                        _append_log("[UI] Failed to create pipeline task")
                    except Exception:
                        pass
                    self.task = None
            else:
                try:
                    toggle_btn.setText("Start")
                except Exception:
                    pass
                if self.task is not None and not self.task.done():
                    self.task.cancel()

        def _on_replay_clicked() -> None:
            # If replay is running, toggle pause/resume.
            if self.task is not None and (not self.task.done()):
                if self.replay_pause_event is None:
                    return
                if not self.replay_paused:
                    self.replay_pause_event.set()
                    self.replay_paused = True
                    try:
                        replay_btn.setText("Resume")
                    except Exception:
                        pass
                    try:
                        run_state_lbl.setText("Paused")
                    except Exception:
                        pass
                else:
                    self.replay_pause_event.clear()
                    self.replay_paused = False
                    try:
                        replay_btn.setText("Pause")
                    except Exception:
                        pass
                    try:
                        run_state_lbl.setText("Replaying")
                    except Exception:
                        pass
                return

            # If another task is running (live mode), do not start replay.
            if self.task is not None and (not self.task.done()):
                _append_log("Another task is running; stop it before replay")
                return

            try:
                self.task = asyncio.create_task(_run_replay())
            except Exception:
                self.task = None

        toggle_btn.toggled.connect(_on_toggle)
        replay_btn.clicked.connect(_on_replay_clicked)

        # 피봇 로그 핸들러 추가
        try:
            pivot_handler = PivotLogHandler(self.update_pivot_status)
            pivot_handler.setLevel(logging.INFO)
            logger.addHandler(pivot_handler)
        except Exception as e:
            logger.debug("[PivotHandler] 추가 실패: %s", e)

        self._enter_gui_main_loop(app=app, loop=loop, window=w, splitter=splitter, settings=settings)
        return 0

    def update_pivot_status(self, pivot_type: str, count: int) -> None:
        """상태 표시줄에 피봇 카운트 업데이트.
        
        Args:
            pivot_type: 피벗 타입 (AAP, MSB, KTP, INT)
            count: 피벗 카운트
        """
        try:
            if pivot_type == "AAP" and hasattr(self, '_aap_lbl') and self._aap_lbl is not None:
                self._aap_lbl.setText(f"AAP: {count}")
            elif pivot_type == "MSB" and hasattr(self, '_msb_lbl') and self._msb_lbl is not None:
                self._msb_lbl.setText(f"MSB: {count}")
            elif pivot_type == "KTP" and hasattr(self, '_ktp_lbl') and self._ktp_lbl is not None:
                self._ktp_lbl.setText(f"KTP: {count}")
            elif pivot_type == "INT" and hasattr(self, '_int_lbl') and self._int_lbl is not None:
                self._int_lbl.setText(f"INT: {count}")
        except Exception as e:
            logger.debug("[PivotStatus] 업데이트 실패: %s", e)


class PivotLogHandler(logging.Handler):
    """피벗 카운트 로그를 감지하여 상태 표시줄에 표시하는 핸들러."""
    
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
    
    def emit(self, record):
        try:
            msg = record.getMessage()
            # 테스트 스크립트 요약 로그 패턴 감지
            # 예: "ATRAdaptivePivot 피봇: 11개"
            if "ATRAdaptivePivot 피봇:" in msg:
                parts = msg.split()
                if len(parts) >= 4:
                    count_str = parts[3].replace("개", "")
                    try:
                        count = int(count_str)
                        self.callback("AAP", count)
                    except Exception:
                        pass
            # 예: "MarketStructureBreak 신호: 31개"
            elif "MarketStructureBreak 신호:" in msg:
                parts = msg.split()
                if len(parts) >= 4:
                    count_str = parts[3].replace("개", "")
                    try:
                        count = int(count_str)
                        self.callback("MSB", count)
                    except Exception:
                        pass
            # 예: "KalmanTurningPoint 전환: 9개"
            elif "KalmanTurningPoint 전환:" in msg:
                parts = msg.split()
                if len(parts) >= 4:
                    count_str = parts[3].replace("개", "")
                    try:
                        count = int(count_str)
                        self.callback("KTP", count)
                    except Exception:
                        pass
            # 예: "통합 피봇 (Integrator): 10개"
            elif "통합 피봇 (Integrator):" in msg:
                parts = msg.split()
                if len(parts) >= 6:
                    count_str = parts[5].replace("개", "")
                    try:
                        count = int(count_str)
                        self.callback("INT", count)
                    except Exception:
                        pass
        except Exception:
            pass


