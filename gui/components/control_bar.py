"""컨트롤 바 컴포넌트"""

import logging
from typing import Optional, Any, Callable

logger = logging.getLogger(__name__)


class ControlBar:
    """컨트롤 바 위젯."""
    
    def __init__(
        self,
        parent: Any,
        minutes: int = 120,
        show_pivots_enabled: bool = True,
        adaptive_enabled: bool = False,
        on_range_changed: Optional[Callable] = None,
        on_ma_toggled: Optional[Callable] = None,
        on_pivot_toggled: Optional[Callable] = None,
        on_adaptive_toggled: Optional[Callable] = None,
        on_refresh_sim_toggle: Optional[Callable] = None,
        on_parameter_dialog: Optional[Callable] = None,
        on_plot_selection_changed: Optional[Callable] = None,
        on_csv_load: Optional[Callable] = None
    ):
        """
        Args:
            parent: 부모 위젯
            minutes: 기본 분봉 수
            show_pivots_enabled: 피봇 표시 활성화 여부
            adaptive_enabled: Adaptive 모드 활성화 여부
            on_range_changed: 범위 변경 콜백
            on_ma_toggled: MA 토글 콜백
            on_pivot_toggled: 피봇 토글 콜백
            on_adaptive_toggled: Adaptive 토글 콜백
            on_refresh_sim_toggle: 갱신/시뮬레이션 토글 콜백
            on_parameter_dialog: 파라미터 다이얼로그 콜백
            on_plot_selection_changed: 플롯 선택 변경 콜백
            on_csv_load: CSV 파일 로드 콜백
        """
        self._parent = parent
        self._minutes = minutes
        self._show_pivots_enabled = show_pivots_enabled
        self._adaptive_enabled = adaptive_enabled
        self._on_range_changed = on_range_changed
        self._on_ma_toggled = on_ma_toggled
        self._on_pivot_toggled = on_pivot_toggled
        self._on_adaptive_toggled = on_adaptive_toggled
        self._on_refresh_sim_toggle = on_refresh_sim_toggle
        self._on_parameter_dialog = on_parameter_dialog
        self._on_plot_selection_changed = on_plot_selection_changed
        self._on_csv_load = on_csv_load
        
        # 위젯 참조
        self._trade_led = None
        self._title_lbl = None
        self._range_cb = None
        self._ma_cb = None
        self._pivot_cb = None
        self._adaptive_cb = None
        self._sim_toggle_btn = None
        self._plot_button_group = None
        self._plot_kospi_rb = None
        self._plot_futures_rb = None
    
    def build(self, root: Any) -> Any:
        """컨트롤 바 빌드.
        
        Args:
            root: 루트 레이아웃 (QVBoxLayout)
            
        Returns:
            컨트롤 위젯 (QWidget)
        """
        from PySide6.QtWidgets import (
            QHBoxLayout, QPushButton, QLabel, QComboBox, QRadioButton, QButtonGroup, QCheckBox, QWidget,
        )
        
        ctrl_row = QHBoxLayout()
        ctrl_row.setContentsMargins(6, 2, 6, 2)

        # 매매 상태 LED
        from gui.components.trade_status_led import TradeStatusLED
        self._trade_led = TradeStatusLED(self._parent)
        ctrl_row.addWidget(self._trade_led.widget)
        
        self._title_lbl = QLabel("📈 KP200 선물 차트")  # 기본값
        ctrl_row.addWidget(self._title_lbl)
        ctrl_row.addStretch(1)
        ctrl_row.addWidget(QLabel("범위:"))

        # 범위 콤보박스
        self._range_cb = QComboBox()
        for m, lbl in ((30, "30분"), (60, "1시간"), (120, "2시간"), (9999, "장전체")):
            self._range_cb.addItem(lbl, userData=m)
        self._range_cb.setMinimumWidth(100)
        try:
            idx = self._range_cb.findData(self._minutes)
            if idx >= 0:
                self._range_cb.setCurrentIndex(idx)
        except Exception:
            pass
        if self._on_range_changed:
            self._range_cb.currentIndexChanged.connect(self._on_range_changed)
        ctrl_row.addWidget(self._range_cb)

        # MA 체크박스
        self._ma_cb = QCheckBox("MA")
        self._ma_cb.setChecked(False)
        if self._on_ma_toggled:
            self._ma_cb.toggled.connect(self._on_ma_toggled)
        ctrl_row.addWidget(self._ma_cb)

        # 피봇 체크박스
        self._pivot_cb = QCheckBox("피봇")
        self._pivot_cb.setChecked(self._show_pivots_enabled)
        if self._on_pivot_toggled:
            self._pivot_cb.toggled.connect(self._on_pivot_toggled)
        ctrl_row.addWidget(self._pivot_cb)

        # Adaptive 체크박스
        self._adaptive_cb = QCheckBox("Adaptive")
        self._adaptive_cb.setChecked(self._adaptive_enabled)
        if self._on_adaptive_toggled:
            self._adaptive_cb.toggled.connect(self._on_adaptive_toggled)
        ctrl_row.addWidget(self._adaptive_cb)

        # 갱신/시뮬레이션 토글 버튼 (장마감 후 시뮬레이션 기능)
        self._sim_toggle_btn = QPushButton("↺ 갱신")
        self._sim_toggle_btn.setFixedWidth(120)
        self._sim_toggle_btn.setCheckable(True)
        if self._on_refresh_sim_toggle:
            self._sim_toggle_btn.clicked.connect(self._on_refresh_sim_toggle)
        ctrl_row.addWidget(self._sim_toggle_btn)

        # 파라미터 조정 버튼
        btn_params = QPushButton("⚙ 파라미터")
        btn_params.setFixedWidth(120)
        if self._on_parameter_dialog:
            btn_params.clicked.connect(self._on_parameter_dialog)
        ctrl_row.addWidget(btn_params)

        # CSV 로드 버튼
        if self._on_csv_load:
            btn_csv = QPushButton("📁 CSV")
            btn_csv.setFixedWidth(80)
            btn_csv.clicked.connect(self._on_csv_load)
            ctrl_row.addWidget(btn_csv)

        # 플롯 선택 라디오 버튼
        self._plot_button_group = QButtonGroup()
        
        self._plot_kospi_rb = QRadioButton("KOSPI")
        self._plot_kospi_rb.setChecked(False)
        self._plot_button_group.addButton(self._plot_kospi_rb, 1)
        if self._on_plot_selection_changed:
            self._plot_kospi_rb.toggled.connect(self._on_plot_selection_changed)
        ctrl_row.addWidget(self._plot_kospi_rb)

        self._plot_futures_rb = QRadioButton("KP200 선물")
        self._plot_futures_rb.setChecked(True)  # 기본값: KP200 선물
        self._plot_button_group.addButton(self._plot_futures_rb, 2)
        if self._on_plot_selection_changed:
            self._plot_futures_rb.toggled.connect(self._on_plot_selection_changed)
        ctrl_row.addWidget(self._plot_futures_rb)

        ctrl_w = QWidget()
        ctrl_w.setLayout(ctrl_row)
        root.addWidget(ctrl_w)

        return ctrl_w
    
    # 위젯 접근자
    @property
    def trade_led(self) -> Any:
        """매매 상태 LED 위젯."""
        return self._trade_led
    
    @property
    def title_lbl(self) -> Any:
        """제목 라벨."""
        return self._title_lbl
    
    @property
    def range_cb(self) -> Any:
        """범위 콤보박스."""
        return self._range_cb
    
    @property
    def ma_cb(self) -> Any:
        """MA 체크박스."""
        return self._ma_cb
    
    @property
    def pivot_cb(self) -> Any:
        """피봇 체크박스."""
        return self._pivot_cb
    
    @property
    def adaptive_cb(self) -> Any:
        """Adaptive 체크박스."""
        return self._adaptive_cb

    @property
    def sim_toggle_btn(self) -> Any:
        """시뮬레이션 토글 버튼."""
        return self._sim_toggle_btn
    
    @property
    def plot_kospi_rb(self) -> Any:
        """KOSPI 라디오 버튼."""
        return self._plot_kospi_rb
    
    @property
    def plot_futures_rb(self) -> Any:
        """KP200 선물 라디오 버튼."""
        return self._plot_futures_rb
    
    def set_minutes(self, minutes: int) -> None:
        """분봉 수 설정."""
        self._minutes = minutes
        if self._range_cb:
            try:
                idx = self._range_cb.findData(minutes)
                if idx >= 0:
                    self._range_cb.setCurrentIndex(idx)
            except Exception:
                pass
    
    def get_selected_minutes(self) -> int:
        """선택된 분봉 수 가져오기."""
        if self._range_cb:
            return self._range_cb.currentData()
        return self._minutes
