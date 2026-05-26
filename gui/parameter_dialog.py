"""AdaptiveZigZag 파라미터 실시간 조정 다이얼로그."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QComboBox,
)


class ParameterDialog(QDialog):
    """AdaptiveZigZag 파라미터 조정 다이얼로그."""

    # 파라미터 변경 시그널 (param_name, new_value)
    parameter_changed = Signal(str, object)

    def __init__(
        self,
        config_path: Path,
        data_source: str = "futures",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.config_path = config_path
        self.data_source = data_source
        self.config_key = f"{data_source}_zigzag"
        self.param_widgets: Dict[str, QWidget] = {}
        self.current_config: Dict[str, Any] = {}
        self.data_source_combo: Optional[QComboBox] = None

        self.setWindowTitle(f"{data_source.upper()} ZigZag 파라미터 조정")
        self.setMinimumWidth(500)
        self.setMinimumHeight(690)

        self._load_config()
        self._setup_ui()
        self._connect_signals()

    def _load_config(self) -> None:
        """config.json에서 설정 로드."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                # adaptive_indicator 섹션에서 해당 데이터 소스의 config 로드
                adaptive_config = config.get("adaptive_indicator", {})
                self.current_config = adaptive_config.get(self.config_key, {})
        except Exception as e:
            print(f"Config 로드 실패: {e}")
            self.current_config = {}

    def _save_config(self) -> None:
        """config.json에 설정 저장."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # 현재 파라미터 값으로 업데이트
            for param_name, widget in self.param_widgets.items():
                if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                    value = widget.value()
                elif isinstance(widget, QCheckBox):
                    value = widget.isChecked()
                elif isinstance(widget, QLineEdit):
                    text = widget.text()
                    # 숫자로 변환 시도
                    try:
                        value = float(text) if "." in text else int(text)
                    except ValueError:
                        value = text
                else:
                    continue

                self.current_config[param_name] = value

            # adaptive_indicator 섹션에 저장
            if "adaptive_indicator" not in config:
                config["adaptive_indicator"] = {}
            config["adaptive_indicator"][self.config_key] = self.current_config

            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            print(f"Config 저장 완료: adaptive_indicator.{self.config_key}")

        except Exception as e:
            QMessageBox.critical(self, "오류", f"Config 저장 실패: {e}")

    def _setup_ui(self) -> None:
        """UI 설정."""
        layout = QVBoxLayout(self)

        # 데이터 소스 선택
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("데이터 소스:"))
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItem("KOSPI", "kospi")
        self.data_source_combo.addItem("KP200 선물", "futures")
        
        # 현재 데이터 소스 선택
        idx = self.data_source_combo.findData(self.data_source)
        if idx >= 0:
            self.data_source_combo.setCurrentIndex(idx)
        
        self.data_source_combo.currentIndexChanged.connect(self._on_data_source_changed)
        source_layout.addWidget(self.data_source_combo)
        source_layout.addStretch()
        layout.addLayout(source_layout)

        # 스크롤 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # ATR 기반 필터링 그룹
        atr_group = QGroupBox("ATR 기반 필터링")
        atr_layout = QFormLayout()

        # use_atr_based_filtering
        self.param_widgets["use_atr_based_filtering"] = QCheckBox()
        self.param_widgets["use_atr_based_filtering"].setChecked(
            self.current_config.get("use_atr_based_filtering", False)
        )
        atr_layout.addRow("ATR 필터링 사용:", self.param_widgets["use_atr_based_filtering"])

        # min_wave_atr_ratio
        self.param_widgets["min_wave_atr_ratio"] = QDoubleSpinBox()
        self.param_widgets["min_wave_atr_ratio"].setRange(0.1, 5.0)
        self.param_widgets["min_wave_atr_ratio"].setSingleStep(0.1)
        self.param_widgets["min_wave_atr_ratio"].setValue(
            self.current_config.get("min_wave_atr_ratio", 1.5)
        )
        atr_layout.addRow("최소 파동 ATR 비율:", self.param_widgets["min_wave_atr_ratio"])

        # cluster_atr_ratio
        self.param_widgets["cluster_atr_ratio"] = QDoubleSpinBox()
        self.param_widgets["cluster_atr_ratio"].setRange(0.1, 5.0)
        self.param_widgets["cluster_atr_ratio"].setSingleStep(0.1)
        self.param_widgets["cluster_atr_ratio"].setValue(
            self.current_config.get("cluster_atr_ratio", 1.0)
        )
        atr_layout.addRow("클러스터 ATR 비율:", self.param_widgets["cluster_atr_ratio"])

        atr_group.setLayout(atr_layout)
        scroll_layout.addWidget(atr_group)

        # ZigZag 기본 파라미터 그룹
        zz_group = QGroupBox("ZigZag 기본 파라미터")
        zz_layout = QFormLayout()

        # atr_multiplier
        self.param_widgets["atr_multiplier"] = QDoubleSpinBox()
        self.param_widgets["atr_multiplier"].setRange(0.5, 10.0)
        self.param_widgets["atr_multiplier"].setSingleStep(0.1)
        self.param_widgets["atr_multiplier"].setValue(
            self.current_config.get("atr_multiplier", 1.5)
        )
        zz_layout.addRow("ATR 배수:", self.param_widgets["atr_multiplier"])

        # atr_period
        self.param_widgets["atr_period"] = QSpinBox()
        self.param_widgets["atr_period"].setRange(5, 50)
        self.param_widgets["atr_period"].setValue(
            self.current_config.get("atr_period", 14)
        )
        zz_layout.addRow("ATR 기간:", self.param_widgets["atr_period"])

        # confirmation_bars
        self.param_widgets["confirmation_bars"] = QSpinBox()
        self.param_widgets["confirmation_bars"].setRange(1, 10)
        self.param_widgets["confirmation_bars"].setValue(
            self.current_config.get("confirmation_bars", 2)
        )
        zz_layout.addRow("확정 봉 수:", self.param_widgets["confirmation_bars"])

        # freeze_on_confirm
        self.param_widgets["freeze_on_confirm"] = QCheckBox()
        self.param_widgets["freeze_on_confirm"].setChecked(
            self.current_config.get("freeze_on_confirm", True)
        )
        zz_layout.addRow("확정 시 가격 고정:", self.param_widgets["freeze_on_confirm"])

        zz_group.setLayout(zz_layout)
        scroll_layout.addWidget(zz_group)

        # 클러스터링 그룹
        cluster_group = QGroupBox("클러스터링")
        cluster_layout = QFormLayout()

        # cluster_tolerance_pct
        self.param_widgets["cluster_tolerance_pct"] = QDoubleSpinBox()
        self.param_widgets["cluster_tolerance_pct"].setRange(0.0, 5.0)
        self.param_widgets["cluster_tolerance_pct"].setSingleStep(0.1)
        self.param_widgets["cluster_tolerance_pct"].setValue(
            self.current_config.get("cluster_tolerance_pct", 0.3)
        )
        cluster_layout.addRow("클러스터 허용 오차(%):", self.param_widgets["cluster_tolerance_pct"])

        cluster_group.setLayout(cluster_layout)
        scroll_layout.addWidget(cluster_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # 버튼 영역
        button_layout = QHBoxLayout()

        self.save_button = QPushButton("저장 및 적용")
        self.save_button.clicked.connect(self._on_save_and_apply)
        button_layout.addWidget(self.save_button)

        self.reset_button = QPushButton("기본값으로 초기화")
        self.reset_button.clicked.connect(self._on_reset)
        button_layout.addWidget(self.reset_button)

        self.close_button = QPushButton("닫기")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def _connect_signals(self) -> None:
        """시그널 연결."""
        # 각 위젯의 값 변경 시 시그널 발생
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                widget.valueChanged.connect(
                    lambda value, name=param_name: self.parameter_changed.emit(name, value)
                )
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(
                    lambda state, name=param_name: self.parameter_changed.emit(
                        name, state == Qt.Checked
                    )
                )

    def _on_data_source_changed(self, index: int) -> None:
        """데이터 소스 변경 시 config 다시 로드."""
        self.data_source = self.data_source_combo.currentData()
        self.config_key = f"{self.data_source}_zigzag"
        self.setWindowTitle(f"{self.data_source.upper()} ZigZag 파라미터 조정")
        
        # config 다시 로드
        self._load_config()
        
        # 위젯 값 업데이트
        self._update_widget_values()

    def _update_widget_values(self) -> None:
        """위젯 값 업데이트."""
        for param_name, widget in self.param_widgets.items():
            value = self.current_config.get(param_name)
            if value is None:
                continue
            
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))

    def _on_save_and_apply(self) -> None:
        """저장 및 적용 버튼 클릭."""
        self._save_config()
        QMessageBox.information(self, "성공", "파라미터가 저장되었습니다.\n인디케이터를 재초기화하세요.")

    def _on_reset(self) -> None:
        """기본값으로 초기화."""
        reply = QMessageBox.question(
            self,
            "확인",
            "모든 파라미터를 기본값으로 초기화하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # 기본값 설정
            defaults = {
                "use_atr_based_filtering": False,
                "min_wave_atr_ratio": 1.5,
                "cluster_atr_ratio": 1.0,
                "atr_multiplier": 1.5,
                "atr_period": 14,
                "confirmation_bars": 2,
                "freeze_on_confirm": True,
                "cluster_tolerance_pct": 0.3,
            }

            for param_name, widget in self.param_widgets.items():
                if param_name in defaults:
                    if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                        widget.setValue(defaults[param_name])
                    elif isinstance(widget, QCheckBox):
                        widget.setChecked(defaults[param_name])

            self._save_config()

    def get_current_config(self) -> Dict[str, Any]:
        """현재 파라미터 값 반환."""
        config = {}
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                value = widget.value()
            elif isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                text = widget.text()
                try:
                    value = float(text) if "." in text else int(text)
                except ValueError:
                    value = text
            else:
                continue
            config[param_name] = value
        return config
