"""chart_viewer.py — SkyPredictor OHLC 차트 뷰어
================================================

finplot으로 KP200 선물 분봉 OHLC 캔들스틱을 표시하고,
Adaptive ZigZag 피봇을 SkyPlot 스타일 마커로 오버레이합니다.

사용법 (gui_controller.py 에서)
--------------------------------
    from chart_viewer import attach_chart_viewer

    # predictor = KP200HybridPredictor 또는 PredictionPipeline
    self._chart_viewer = attach_chart_viewer(
        right_root,
        predictor=predictor,
        config=cfg,         # AppConfig (없으면 None)
        minutes=120,
    )

    # 파이프라인 시작 후 predictor 교체가 필요한 경우:
    if self._chart_viewer:
        self._chart_viewer.set_predictor(predictor)

    # 앱 종료 시:
    if self._chart_viewer:
        self._chart_viewer.stop()

독립 실행 (오프라인 테스트)
-------------------------------
    python chart_viewer.py

피봇 마커 스타일 (SkyPlot plot_manager.py 동일 규칙)
------------------------------------------------------
  confirmed H   → orange ▼  (finplot style='v')
  confirmed L   → orange ▲  (finplot style='^')
  unconfirmed   → yellow ◆  (finplot style='d')
  ZigZag 폴리라인 → orange 실선
"""

from __future__ import annotations

import logging
import os
import time
import json
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from PySide6.QtWidgets import QVBoxLayout, QWidget

import numpy as np
import pandas as pd

# ── 분리된 모듈 임포트 ─────────────────────────────────────────────────────────
from gui.utils.pivot_probability import HistoricalPivot, PivotProbabilityCalculator
from gui.utils.virtual_tick_generator import VirtualTickGenerator
from gui.utils.threading import DataComputeThread, Slot, QT_AVAILABLE
from gui.utils.error_handlers import FinplotTypeErrorHandler, _StderrFilter, _qt_message_handler_func
from gui.utils.cache_manager import CacheManager
from gui.data.data_fetcher import DataFetcher
from gui.components.trade_status_led import TradeStatusLED
from gui.components.control_bar import ControlBar
from gui.components.pivot_event_log import PivotEventLog
from gui.components.trade_event_log import TradeEventLog
from indicators.pivot_quality_monitor import PivotQualityMonitor
from gui.renderers.fplt_renderer import FpltRenderer
from gui.engines.chart_engine import ChartEngine

# ── 로거 설정 ────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── 설정 관리 ────────────────────────────────────────────────────────────────

@dataclass
class ChartViewerConfig:
    """차트 뷰어 설정."""
    refresh_ms: int = 500
    minutes: int = 120
    auto_refresh: bool = True
    show_ma: bool = True
    show_pivots: bool = True
    cache_ttl: float = 5.0
    loading_enabled: bool = True
    zoom_enabled: bool = True  # 줌 기능 활성화
    pan_enabled: bool = True   # 팬 기능 활성화


# ══════════════════════════════════════════════════════════════════════════════
# §3  Qt 위젯
# ══════════════════════════════════════════════════════════════════════════════

class ChartViewerWidget:
    """
    KP200 선물 OHLC 차트 뷰어 위젯.

    finplot FinWindow(GraphicsLayoutWidget) 을 Qt 레이아웃에 직접 embed 한다.
    QTimer 로 자동 갱신(기본 5초), 수동 refresh(), 범위 콤보박스 제공.
    """

    DEFAULT_REFRESH_MS = 5000  # 기본값 5초 (config.json과 일치)
    DEFAULT_MINUTES    = 120

    def __init__(
        self,
        predictor:     Any = None,
        config:        Any = None,
        refresh_ms:    int = DEFAULT_REFRESH_MS,
        minutes:       int = DEFAULT_MINUTES,
        parent_widget: Any = None,
        use_api:       bool = False,
        kp200_upcode:  Optional[str] = None,
        kospi_upcode:  Optional[str] = None,
        viewer_config: Optional[ChartViewerConfig] = None,
        data_fetcher:  Optional[callable] = None,  # 의존성 주입: 데이터 페처
        regime_led_callback: Optional[callable] = None,  # 레짐 LED 색상 설정 콜백
        regime_label_callback: Optional[callable] = None,  # 레짐 라벨 텍스트 설정 콜백
        telegram_bridge_holder: Optional[Dict[str, Any]] = None,  # 텔레그램 bridge 홀더
    ) -> None:
        logger.info("[ChartViewerWidget] __init__ 시작")
        # 설정 통합
        if viewer_config is not None:
            self._config_obj = viewer_config
            refresh_ms = viewer_config.refresh_ms
            minutes = viewer_config.minutes
            auto_refresh = viewer_config.auto_refresh
            show_ma = viewer_config.show_ma
            show_pivots = viewer_config.show_pivots
            cache_ttl = viewer_config.cache_ttl
            loading_enabled = viewer_config.loading_enabled
            zoom_enabled = viewer_config.zoom_enabled
            pan_enabled = viewer_config.pan_enabled
        else:
            self._config_obj = ChartViewerConfig(
                refresh_ms=refresh_ms,
                minutes=minutes,
                auto_refresh=True,
                show_ma=True,
                show_pivots=True,
                cache_ttl=5.0,
                loading_enabled=True,
                zoom_enabled=True,
                pan_enabled=True,
            )
            auto_refresh = True
            show_ma = True
            show_pivots = True
            cache_ttl = 5.0
            loading_enabled = True
            zoom_enabled = True
            pan_enabled = True

        self._parent     = parent_widget
        self._predictor  = predictor
        self._config     = config
        self._refresh_ms = int(refresh_ms)
        self._minutes    = int(minutes)
        self._use_api    = use_api
        self._kp200_upcode = kp200_upcode or ""
        self._kospi_upcode = kospi_upcode or ""
        self._csv_mode   = False  # CSV 백테스트 모드 플래그

        self._engine:     Optional[ChartEngine] = None  # _compute_data에서 초기화
        self._renderer:   Optional[FpltRenderer] = None

        # 데이터 페처 (의존성 주입 또는 기본 생성)
        if data_fetcher is not None:
            self._data_fetcher = data_fetcher
        else:
            self._data_fetcher = DataFetcher(
                predictor=predictor,
                config=config,
                selected_plot="futures",
                minutes=self._minutes,
                use_api=use_api,
                kp200_upcode=self._kp200_upcode,
                kospi_upcode=self._kospi_upcode
            )

        # predictor의 _adaptive_mgr에서도 콜백 설정 (듀얼 모드 지원)
        if self._predictor and hasattr(self._predictor, '_adaptive_mgr'):
            try:
                self._predictor._adaptive_mgr.set_pivot_candidate_callback(self._on_pivot_candidate_event)
                logger.debug("[ChartViewerWidget] predictor._adaptive_mgr 콜백 설정 완료")
            except Exception as e:
                logger.warning("[ChartViewerWidget] predictor._adaptive_mgr 콜백 설정 실패: %s", e)
        self._fplt_ref:   Optional[Any]          = None
        self._timer:      Optional[Any]          = None
        self._status_lbl: Optional[Any]         = None
        self._loading_lbl: Optional[Any]        = None
        self._selected_plot: str = "futures"    # "kospi" or "futures"
        self._position_tracker: Optional[Any] = None  # 포지션 트래커
        self._led_sync_enabled: bool = True  # LED 동기화 활성화
        self._trade_markers_enabled: bool = True  # 거래 마커 활성화
        self._trade_events: List[Dict] = []  # 거래 이벤트 캐시
        self._trade_events_mtime: float = 0.0  # 파일 수정 시간 캐시
        self._trade_events_last_read: float = 0.0  # 마지막 읽기 시간 캐시
        self._t0: float = time.perf_counter()  # 렌더링 시간 측정 기본값
        self._risk_monitor: Optional[Any] = None  # 리스크 모니터
        self._risk_monitoring_enabled: bool = True  # 리스크 모니터링 활성화
        self._auto_refresh_enabled: bool = auto_refresh  # 자동 갱신 활성화 (설정에서 가져옴)
        self._show_pivots_enabled: bool = show_pivots  # 피봇 마커 표시 활성화
        self._minutes_changed: bool = False  # 범위 변경 감지 플래그 (피봇 마커 재렌더용)
        # ── [보완-3] 멀티스레드 데이터 경합 방지용 요청 토큰 ──
        self._current_request_token: float = 0.0
        # ────────────────────────────────────────────────────────────────────────────

        # 피봇 확정 확률 계산기
        self._pivot_prob_calc: PivotProbabilityCalculator = PivotProbabilityCalculator(max_history=1000)

        # 원래 ZigZag 설정 백업 (Adaptive OFF 시 복원용)
        self._original_zz_configs: Dict[str, Dict] = {}  # {"kospi": {...}, "futures": {...}}
        self._adaptive_pending_reapply: bool = False  # 데이터 로드 후 피봇 조정 재적용 플래그
        self._adaptive_adjusting: bool = False  # 피봇 조정 중 플래그 (무한 루프 방지)
        self._last_data_source: Optional[str] = None  # 마지막 데이터 소스 (변경 감지용)

        # 이벤트 로그 컴포넌트
        self._pivot_event_log: PivotEventLog = PivotEventLog(max_lines=100)
        self._trade_event_log: TradeEventLog = TradeEventLog(max_lines=100)
        self._pivot_quality_monitor: PivotQualityMonitor = PivotQualityMonitor()

        # 거래 이벤트 콜백
        self._trade_event_callback: Optional[callable] = None
        
        # 렌더링 상태 추적
        self._is_rendering: bool = False  # 렌더링 중 여부
        self._render_lock = __import__('threading').Lock()  # [FIX] TOCTOU 방지용 Lock
        self._render_cancel_requested: bool = False  # 렌더링 취소 요청
        self._last_render_time: float = 0.0  # 마지막 렌더링 시간 (ms)
        
        # 백그라운드 컴퓨팅 스레드
        self._compute_thread: Optional[DataComputeThread] = None
        
        # crosshair 캐시 (캡슐화 개선)
        self._x_coords_cache: Optional[Any] = None

        # 시뮬레이션 상태
        self._sim_active: bool = False  # 시뮬레이션 활성화 여부
        self._sim_count: int = 0  # 시뮬레이션 카운트
        self._sim_max_count: int = 0  # 시뮬레이션 최대 카운트
        self._sim_timer: Optional[Any] = None  # 시뮬레이션 타이머
        self._sim_virtual_generator: Optional[Any] = None  # 가상 틱 생성기
        self._sim_virtual_data: Optional[pd.DataFrame] = None  # 가상 데이터프레임
        self._sim_current_price: Optional[float] = None  # 시뮬레이션 현재가
        self._refresh_scheduled: bool = False  # refresh 예약 상태 (중복 예약 방지)
        
        # 실시간 데이터 수신 추적
        self._last_data_timestamp: Optional[float] = None  # 마지막 데이터 타임스탬프
        self._last_data_count: int = 0  # 마지막 데이터 개수
        self._new_data_received: bool = False  # 새로운 데이터 수신 플래그
        
        # 데이터 캐싱
        self._cache_manager = CacheManager(cache_ttl=cache_ttl)
        self._loading_enabled: bool = loading_enabled  # 로딩 표시 활성화 (설정에서 가져옴)
        self._zoom_enabled: bool = zoom_enabled  # 줌 기능 활성화 (설정에서 가져옴)
        self._pan_enabled: bool = pan_enabled  # 팬 기능 활성화 (설정에서 가져옴)

        # 피봇 정보 패널 (crosshair용)
        self._pivot_info_panel: Optional[Any] = None
        self._pivot_text_item: Optional[Any] = None  # 차트 내부 배치용 TextItem

        # 피봇 확정 확률 계산기
        self._pivot_prob_calc = PivotProbabilityCalculator(max_history=1000)

        # 시장 레짐 분류기
        self._regime_classifier: Optional[Any] = None
        self._current_regime: Optional[Any] = None
        self._last_logged_regime: Optional[str] = None  # 레짐 변경 감지용
        self._adaptive_enabled: bool = False  # Adaptive 모드 활성화 상태
        self._regime_led_callback: Optional[callable] = regime_led_callback  # 레짐 LED 색상 설정 콜백
        self._regime_label_callback: Optional[callable] = regime_label_callback  # 레짐 라벨 텍스트 설정 콜백
        self._telegram_bridge_holder: Optional[Dict[str, Any]] = telegram_bridge_holder  # 텔레그램 bridge 홀더

        # 레짐 변경 통계 추적
        self._regime_change_stats: Dict[str, int] = {}  # {날짜: 변경 횟수}
        self._regime_change_history: List[Dict[str, Any]] = []  # [{시간, 이전, 현재, 심볼}]
        self._last_daily_report_date: Optional[str] = None  # 마지막 일일 리포트 날짜
        self._regime_stats_file = "logs/regime_change_stats.json"  # 통계 저장 파일

        # 레짐 트레이딩
        self._regime_trading_enabled: bool = False
        self._regime_trading_config: Dict[str, Any] = {}
        self._regime_confirmation_bars: int = 0  # 레짐 변경 후 확인 봉 수
        self._regime_change_hourly_count: int = 0  # 시간당 레짐 변경 횟수
        self._last_regime_change_hour: Optional[str] = None

        # 통계 파일 로드
        self._load_regime_stats()

        # 레짐 트레이딩 설정 로드 (Adaptive 체크박스와 무관)
        self._load_regime_trading_config()

        # crosshair 이벤트 연결 상태
        self._crosshair_event_connected: bool = False
        # [FIX] QTimer 1회 생성 재사용 (매 mousemove마다 새 객체 생성하면 메모리 누수)
        self._crosshair_debounce_timer: Optional[Any] = None  # _build_widget 후 초기화
        self._crosshair_last_pos: Optional[Any] = None  # 디바운스용 최신 마우스 위치

        # __init__에서 1회 import 시도 (매 refresh 동적 import 제거)
        self._try_init_position_tracker()
        self._try_init_risk_monitor()

        logger.info("[ChartViewerWidget] _build_widget 호출 전")
        self.widget = self._build_widget(parent_widget)
        # [FIX] widget 생성 후 QTimer 단일 초기화
        self._init_crosshair_debounce_timer()
        logger.info("[ChartViewerWidget] _build_widget 호출 후 (widget=%s)", self.widget is not None)

    # ── 1회 import 초기화 ─────────────────────────────────────────────────────

    def _init_crosshair_debounce_timer(self) -> None:
        """Crosshair 디바운스 QTimer 단일 초기화 (재사용 방식).

        _build_widget() 완료 후 호출해야 Qt 이벤트 루프가 준비된 상태다.
        """
        try:
            from PySide6.QtCore import QTimer
            self._crosshair_debounce_timer = QTimer(self.widget)
            self._crosshair_debounce_timer.setSingleShot(True)
            self._crosshair_debounce_timer.timeout.connect(self._on_crosshair_debounced)
            logger.debug("[ChartViewerWidget] crosshair debounce timer 초기화 완료")
        except Exception as e:
            logger.debug("[ChartViewerWidget] crosshair debounce timer 초기화 실패: %s", e)

    def _on_crosshair_debounced(self) -> None:
        """디바운스 타이머 만료 시 실제 crosshair 업데이트 실행."""
        if self._crosshair_last_pos is not None:
            self._do_crosshair_update(self._crosshair_last_pos)

    def _try_init_position_tracker(self) -> None:
        try:
            from prediction.trade_logger import get_position_tracker
            self._position_tracker = get_position_tracker()
        except Exception as e:
            logger.debug("[ChartViewerWidget] 포지션 트래커 초기화 실패: %s", e)

    def _try_init_risk_monitor(self) -> None:
        """리스크 모니터 초기화."""
        try:
            from prediction.risk_monitor import RiskMonitor
            # control_bar가 생성된 후에 호출됨
            trade_led = self._control_bar.trade_led if hasattr(self, '_control_bar') else None
            self._risk_monitor = RiskMonitor(led=trade_led, notifier=None)
        except Exception as e:
            logger.debug("[ChartViewerWidget] 리스크 모니터 초기화 실패: %s", e)

    # ── 위젯 구성 ─────────────────────────────────────────────────────────────

    def _build_control_bar(self, parent: Any, root: QVBoxLayout) -> QWidget:
        """컨트롤 바 빌드."""
        # config.adaptive_mode 확인 (config가 로드되지 않았으면 False 기본값)
        adaptive_enabled = False
        if self._config and hasattr(self._config, 'adaptive_mode'):
            adaptive_enabled = self._config.adaptive_mode

        self._control_bar = ControlBar(
            parent=parent,
            minutes=self._minutes,
            show_pivots_enabled=self._show_pivots_enabled,
            adaptive_enabled=adaptive_enabled,
            on_range_changed=self._on_range_changed,
            on_ma_toggled=self._on_ma_toggled,
            on_pivot_toggled=self._on_pivot_toggled,
            on_adaptive_toggled=self._on_adaptive_toggled,
            on_refresh_sim_toggle=self._on_refresh_sim_toggle,
            on_parameter_dialog=self._on_parameter_dialog,
            on_plot_selection_changed=self._on_plot_selection_changed,
            on_csv_load=self._on_csv_load
        )
        return self._control_bar.build(root)

    def _build_pivot_event_log(self, parent: Any, root: QVBoxLayout) -> None:
        """피봇 후보 이벤트 로그 영역 빌드."""
        self._pivot_event_log.build(parent, root)

    def _on_pivot_candidate_event(self, **kwargs: Any) -> None:
        """피봇 후보 이벤트 콜백."""
        event_type = kwargs.get("event_type", "")
        symbol = kwargs.get("symbol", "")
        candidate_type = kwargs.get("candidate_type", "")
        candidate_price = kwargs.get("candidate_price", 0.0)
        bar_idx = kwargs.get("bar_idx", 0)
        timestamp = kwargs.get("timestamp", "")
        reason = kwargs.get("reason", "")

        self._pivot_event_log.add_event(
            event_type=event_type,
            symbol=symbol,
            candidate_type=candidate_type,
            candidate_price=candidate_price,
            bar_idx=bar_idx,
            timestamp=timestamp,
            reason=reason
        )

    def _build_trade_event_log(self, parent: Any, root: QVBoxLayout) -> None:
        """거래 이벤트 로그 영역 빌드."""
        self._trade_event_log.build(parent, root)

    def _build_widget(self, parent: Any) -> Optional[Any]:
        logger.info("[ChartViewerWidget] _build_widget 시작")
        try:
            from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
            from PySide6.QtCore import QTimer, Qt
        except ImportError:
            logger.error("[ChartViewerWidget] PySide6 패키지를 찾을 수 없습니다. 설치 후 다시 시도하세요.")
            return None

        container = QWidget(parent)
        root = QVBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        # 컨트롤 바
        logger.info("[ChartViewerWidget] _build_control_bar 호출")
        self._build_control_bar(container, root)
        logger.info("[ChartViewerWidget] _build_control_bar 완료")

        # finplot 캔버스 (차트를 위에 배치)
        fplt_w = self._build_fplt_canvas()
        if fplt_w is not None:
            root.addWidget(fplt_w, stretch=3)  # 차트에 더 큰 stretch 할당
        else:
            fb = QLabel("⚠ 차트 라이브러리를 초기화할 수 없습니다.\nfinplot 패키지가 설치되어 있는지 확인하세요.")
            fb.setAlignment(Qt.AlignCenter)
            root.addWidget(fb, stretch=3)

        # 피봇 품질 모니터링 패널
        try:
            self._pivot_quality_monitor.build(container, root)
        except Exception as e:
            logger.warning("[ChartViewerWidget] 피봇 품질 모니터 빌드 실패: %s", e)

        # 피봇 후보 이벤트 로그 영역 (차트 아래에 배치, 고정 높이)
        try:
            self._build_pivot_event_log(container, root)
        except Exception as e:
            logger.warning("[ChartViewerWidget] 피봇 이벤트 로그 빌드 실패: %s", e)

        # 거래 이벤트 로그 영역 (차트 아래에 배치, 고정 높이) - 숨김 처리
        # try:
        #     self._build_trade_event_log(container, root)
        # except Exception as e:
        #     logger.warning("[ChartViewerWidget] 거래 이벤트 로그 빌드 실패: %s", e)

        # 상태 레이블
        self._status_lbl = QLabel("대기 중…")
        root.addWidget(self._status_lbl)
        
        # 로딩 라벨
        self._loading_lbl = QLabel("⏳ 로딩 중...")
        self._loading_lbl.setStyleSheet("color: #FFA500; font-weight: bold;")
        self._loading_lbl.setVisible(False)
        root.addWidget(self._loading_lbl)

        # 피봇 정보 패널 (crosshair용)
        # 차트 캔버스 생성 후에 부모를 설정하므로, 일단 container에 생성
        self._pivot_info_panel = QLabel(container)
        self._pivot_info_panel.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: #7CFC00;
                border: 1px solid #7CFC00;
                border-radius: 3px;
                padding: 5px;
                font-size: 11px;
                font-family: Consolas, monospace;
            }
        """)
        self._pivot_info_panel.hide()
        self._pivot_info_panel.setParent(container)
        self._pivot_info_panel.raise_()

        # 자동 갱신 타이머
        self._timer = QTimer(container)
        self._timer.setInterval(self._refresh_ms)
        self._timer.timeout.connect(self._auto_refresh_callback)
        self._timer.start()
        logger.info("[ChartViewer] 자동 갱신 타이머 시작 (interval=%dms, auto_refresh=%s)", self._refresh_ms, self._auto_refresh_enabled)

        return container

    # ── finplot 캔버스 embed ──────────────────────────────────────────────────

    def _build_fplt_canvas(self) -> Optional[Any]:
        """
        finplot FinWindow(GraphicsLayoutWidget) 을 생성하고 rows=2 ax 를 설정한다.
        FinWindow 는 QWidget 이므로 Qt 레이웃에 직접 addWidget 가능하다.
        """
        logger.info("[ChartViewerWidget] _build_fplt_canvas 시작")
        try:
            import finplot as fplt

            fplt.foreground = "#FFFFFF"
            fplt.background = "#0D0D0D"

            # FinWindow(GraphicsLayoutWidget) 생성 — show() 없이 embed용
            win = fplt.FinWindow("KP200 선물")
            win.show_maximized = False
            # 최대 높이 설정 (차트 높이 제한) - 화면 해상도 비율로 설정
            try:
                from PySide6.QtWidgets import QApplication
                screen = QApplication.primaryScreen()
                max_h = int(screen.availableGeometry().height() * 0.50) if screen else 600
            except Exception:
                max_h = 600
            win.setMaximumHeight(max_h)

            # rows=1: 캔들 단독 (볼륨 ax 없음 — ax_vol=None이면 x축이 ax_main에 표시됨)
            axs = fplt.create_plot_widget(
                master=win,
                rows=1,
                init_zoom_periods=9999,  # 초기 뷰: 전체 데이터 표시
            )

            ax_main = axs[0] if isinstance(axs, (list, tuple)) else axs
            ax_vol  = None   # 거래량 ax 미사용 (t8415 Volume=0)

            # 가로세로 격자선 표시
            try:
                ax_main.showGrid(x=True, y=True, alpha=0.3)
            except Exception:
                pass

            # 피봇 정보 패널용 crosshair 이벤트 연결
            # scene은 차트가 렌더링된 후에 생성되므로, 렌더링 후 연결 (_render_chart에서 처리)

            # 피봇 정보 TextItem 생성 (차트 내부 배치용)
            if self._pivot_text_item is None:
                try:
                    from pyqtgraph import TextItem
                    from PySide6.QtGui import QFont

                    self._pivot_text_item = TextItem(
                        color="#7CFC00",
                        anchor=(1, 0),  # 우측 상단 앵커
                        border="#7CFC00",
                        fill=(0, 0, 0, 180)
                    )
                    self._pivot_text_item.setFont(QFont("Consolas", 11))
                    self._pivot_text_item.hide()

                    # ViewBox에 추가
                    vb = getattr(ax_main, 'vb', None)
                    if vb is not None:
                        vb.addItem(self._pivot_text_item)
                        logger.debug("[ChartViewer] TextItem 생성 및 ViewBox에 추가 완료")
                except Exception as e:
                    logger.warning("[ChartViewer] TextItem 생성 실패: %s", e)
                    self._pivot_text_item = None

            # 줌/팬 기능 활성화
            if self._zoom_enabled:
                # 마우스 휠 줌 활성화
                try:
                    fplt.x_zoom_on(2)  # 마우스 휠 줌 배수
                except Exception:
                    pass
            if self._pan_enabled:
                # 드래그 팬 활성화 (finplot 기본 기능)
                try:
                    # finplot은 기본적으로 드래그 팬을 지원
                    pass
                except Exception:
                    pass

            # FinWindow 에 ax 등록
            try:
                win.addItem(ax_main, col=1)
                win.nextRow()
            except Exception:
                pass

            self._fplt_ref = fplt
            self._renderer = FpltRenderer(ax_main, ax_vol)

            logger.info("[ChartViewerWidget] finplot 캔버스 초기화 완료")
            return win   # GraphicsLayoutWidget → QWidget 역할

        except Exception as e:
            logger.warning("[ChartViewerWidget] finplot 캔버스 초기화 실패: %s", e)
            self._renderer = None
            self._fplt_ref  = None
            return None

    # ── 이벤트 핸들러 ─────────────────────────────────────────────────────────

    def _connect_crosshair_event(self) -> None:
        """Crosshair 이벤트 연결 (렌더링 후 scene 생성 확인)."""
        if self._crosshair_event_connected:
            return

        if self._renderer is None or self._renderer.ax_main is None:
            return

        try:
            ax_main = self._renderer.ax_main
            if hasattr(ax_main, 'vb') and hasattr(ax_main.vb, 'scene'):
                scene = ax_main.vb.scene()
                if scene is not None and hasattr(scene, 'sigMouseMoved'):
                    scene.sigMouseMoved.connect(self._on_crosshair_moved)
                    self._crosshair_event_connected = True
                    logger.info("[ChartViewerWidget] crosshair 이벤트 연결 완료")
                else:
                    logger.debug("[ChartViewerWidget] scene가 None이거나 sigMouseMoved 속성이 없음 (나중에 재시도)")
            else:
                logger.debug("[ChartViewerWidget] ax_main.vb 또는 scene 속성이 없음 (나중에 재시도)")
        except Exception as e:
            logger.warning("[ChartViewerWidget] crosshair 이벤트 연결 실패: %s", e)

    def _on_crosshair_moved(self, pos) -> None:
        """Crosshair 이동 이벤트 핸들러 (피봇 정보 패널용).

        [FIX] 매 호출마다 QTimer를 새로 생성하는 대신, __init__에서 단일 생성한
        _crosshair_debounce_timer를 restart() 방식으로 재사용한다.
        이전 방식은 매 mousemove 이벤트(수십 ms 간격)마다 QTimer 객체를 생성하여
        deleteLater 없이 누적 → 메모리 누수 발생.
        """
        if self._renderer is None:
            return

        logger.debug("[ChartViewerWidget] _on_crosshair_moved 호출: pos=%s", pos)

        # [FIX] 최신 위치 저장 후 기존 타이머 재시작 (객체 재생성 없음)
        self._crosshair_last_pos = pos
        if self._crosshair_debounce_timer is not None:
            self._crosshair_debounce_timer.start(50)  # 이미 실행 중이면 자동으로 재시작됨

    def _do_crosshair_update(self, pos) -> None:
        """Crosshair 업데이트 실제 처리 (디바운싱 후 호출)."""
        if self._renderer is None:
            return

        logger.debug("[ChartViewerWidget] _do_crosshair_update 호출: pos=%s", pos)

        try:
            df_index = self._renderer._df_index
            if df_index is None or len(df_index) == 0:
                if self._pivot_text_item:
                    self._pivot_text_item.hide()
                if self._pivot_info_panel:
                    self._pivot_info_panel.hide()
                return

            vb = getattr(self._renderer.ax_main, 'vb', None)
            if vb is None:
                return

            # Scene 좌표 → View 좌표
            mouse_point = vb.mapSceneToView(pos)
            x_coord = mouse_point.x()   # 뷰 좌표계 x값 (봉 인덱스)
            y_coord = mouse_point.y()

            # ── 핵심 수정: 뷰 좌표 x_coord가 이미 봉 인덱스임 ──
            # finplot 뷰 좌표계의 x축은 0-based 봉 인덱스를 사용
            nearest_idx = int(round(x_coord))
            # 데이터 범위를 엄격하게 제한
            nearest_idx = max(0, min(nearest_idx, len(df_index) - 1))
            # 피봇 정보 확인
            pivot_info = self._renderer._pivot_info
            if pivot_info is None or pivot_info.empty:
                if self._pivot_text_item:
                    self._pivot_text_item.hide()
                if self._pivot_info_panel:
                    self._pivot_info_panel.hide()
                return

            pivot_idx_arr = self._renderer._pivot_idx_arr  # 0-based 봉 인덱스
            pivot_y_arr   = self._renderer._pivot_y_arr

            if len(pivot_idx_arr) == 0:
                if self._pivot_text_item:
                    self._pivot_text_item.hide()
                if self._pivot_info_panel:
                    self._pivot_info_panel.hide()
                return

            # 20봉 이내 피봇 필터링
            mask = np.abs(pivot_idx_arr.astype(float) - nearest_idx) <= 20
            if not mask.any():
                if self._pivot_text_item:
                    self._pivot_text_item.hide()
                if self._pivot_info_panel:
                    self._pivot_info_panel.hide()
                return

            # 복합 거리 계산 (인덱스 거리 + 가격 거리)
            idx_diffs   = np.abs(pivot_idx_arr[mask].astype(float) - nearest_idx)
            price_diffs = np.abs(pivot_y_arr[mask] - y_coord)
            y_safe      = max(abs(y_coord), 1.0)
            distances   = idx_diffs * 0.7 + (price_diffs / y_safe * 100) * 0.3

            best_match_local = int(np.argmin(distances))

            # ── 핵심 수정: np.where(mask)[0][...] ──
            orig_indices = np.where(mask)[0]
            orig_idx     = int(orig_indices[best_match_local])
            pivot_row    = pivot_info.iloc[orig_idx]
            # pivot_row가 None인 경우 방어적 처리
            if pivot_row is None:
                if self._pivot_text_item:
                    self._pivot_text_item.hide()
                if self._pivot_info_panel:
                    self._pivot_info_panel.hide()
                return
            # ── 피봇 정보 텍스트 생성 ──
            pivot_type  = pivot_row['t']
            pivot_price = float(pivot_row['y'])
            # 안전한 인덱스 추출 (Series인 경우 iloc[0] 사용)
            pivot_idx_val = pivot_row['idx']
            pivot_idx_v = int(pivot_idx_val.iloc[0]) if hasattr(pivot_idx_val, 'iloc') else int(pivot_idx_val)

            time_str = (df_index[pivot_idx_v].strftime('%H:%M')
                        if pivot_idx_v < len(df_index) else f"idx:{pivot_idx_v}")

            is_confirmed = (
                'confirmed_at_idx' in pivot_row.index
                and pd.notna(pivot_row['confirmed_at_idx'])
                and int(pivot_row['confirmed_at_idx']) >= 0
            )
            
            # 지연확정 정보를 가장 앞에 배치
            if is_confirmed:
                confirmed_at = int(pivot_row['confirmed_at_idx'])
                delay = confirmed_at - pivot_idx_v
                delay_str = f"지연확정: +{delay}봉" if delay > 0 else "즉시확정"
                info_text = f"{delay_str} | 시간: {time_str} | 가격: {pivot_price:.2f} | 피봇: {pivot_type}"
            else:
                # 후보 피봇도 동일한 형식: 확정확률을 가장 앞에 배치
                prob_str = "확정확률: N/A"
                try:
                    candidate = HistoricalPivot(
                        idx=pivot_idx_v, price=pivot_price,
                        pivot_type=pivot_type, confirmed=False,
                    )
                    if self._pivot_prob_calc is not None:
                        prob = self._pivot_prob_calc.calculate_combined_probability(
                            candidate, y_coord, confirmation_bars_required=3
                        )
                        prob_str = f"확정확률: {prob*100:.1f}%"
                except Exception:
                    pass
                info_text = f"{prob_str} | 시간: {time_str} | 가격: {pivot_price:.2f} | 후보: {pivot_type}"

            # TextItem 사용 (차트 내부 배치)
            if self._pivot_text_item is not None:
                try:
                    vb = self._renderer.ax_main.vb
                    view_range = vb.viewRange()

                    # 현재 뷰 범위의 우측 상단 (데이터 좌표)
                    x_max = view_range[0][1]
                    y_max = view_range[1][1]

                    # 여백 추가 (데이터 범위의 2%)
                    x_range = view_range[0][1] - view_range[0][0]
                    y_range = view_range[1][1] - view_range[1][0]

                    x_pos = x_max - x_range * 0.02
                    y_pos = y_max - y_range * 0.02

                    self._pivot_text_item.setPos(x_pos, y_pos)
                    self._pivot_text_item.setText(info_text)
                    self._pivot_text_item.show()

                    # 외부 패널 숨김
                    if self._pivot_info_panel:
                        self._pivot_info_panel.hide()

                except Exception as e:
                    logger.warning("[ChartViewerWidget] TextItem 위치 설정 실패: %s", e)
                    # TextItem 실패 시 외부 패널 사용
                    if self._pivot_info_panel:
                        self._pivot_info_panel.setText(info_text)
                        self._pivot_info_panel.adjustSize()
                        self._pivot_info_panel.show()
            elif self._pivot_info_panel is not None:
                # TextItem이 없는 경우 기존 오버레이 방식 사용
                self._pivot_info_panel.setText(info_text)
                self._pivot_info_panel.adjustSize()

                # 패널 위치 (차트 캔버스 내부 우측 상단)
                if self._renderer is not None and self._renderer.ax_main is not None:
                    try:
                        ax_main = self._renderer.ax_main
                        # ax_main.vb의 parent widget (win)을 찾아서 win 내부 좌표 계산
                        vb = getattr(ax_main, 'vb', None)
                        if vb is not None:
                            win = vb.parentWidget()
                            if win is not None:
                                # win 내부 좌표 (패널은 win의 자식이므로)
                                win_width = win.width()
                                win_height = win.height()
                                win_pos = win.pos()

                                pw = self._pivot_info_panel.width()
                                ph = self._pivot_info_panel.height()

                                # win 내부 우측 상단 (여백 10px)
                                x = win_width - pw - 10
                                y = 10

                                logger.debug("[ChartViewerWidget] win: pos=%s, size=%s, panel: size=%s, move=(%d, %d)",
                                           win_pos, (win_width, win_height), (pw, ph), x, y)
                                self._pivot_info_panel.move(x, y)
                    except Exception as e:
                        logger.warning("[ChartViewerWidget] 패널 위치 설정 실패: %s", e)
                elif self.widget:
                    # fallback: widget 우측 상단
                    pw = self._pivot_info_panel.width()
                    x = self.widget.width() - pw - 20
                    y = 10
                    self._pivot_info_panel.move(x, y)

                self._pivot_info_panel.show()

        except Exception as e:
            logger.debug("[ChartViewerWidget] crosshair 처리 실패: %s", e, exc_info=True)
            if self._pivot_text_item:
                self._pivot_text_item.hide()
            if self._pivot_info_panel:
                self._pivot_info_panel.hide()

    def _on_range_changed(self) -> None:
        """범위 콤보박스 변경."""
        self._minutes = self._control_bar.get_selected_minutes()
        logger.info("[ChartViewer] 범위 변경: %d분", self._minutes)

        # MAX_BARS 동적 조정
        self._engine.set_max_bars(self._minutes)

        # DataFetcher 상태 업데이트
        self._data_fetcher.set_minutes(self._minutes)

        # 범위 변경 플래그 설정 (피봇 마커 재렌더용)
        self._minutes_changed = True

        # 캐시 삭제
        self._clear_cache()
        self.refresh()

        # 범위 변경 즉시 x축 뷰 갱신
        if self._renderer is not None:
            n = 0 if self._minutes >= 9999 else self._minutes
            self._renderer._reset_xaxis_view(n_bars=n)
    
    def _on_plot_selection_changed(self, checked: bool) -> None:
        """플롯 선택 변경."""
        if not checked:
            return

        if self._control_bar.plot_kospi_rb.isChecked():
            self._selected_plot = "kospi"
            if self._control_bar.title_lbl is not None:
                self._control_bar.title_lbl.setText("📈 KOSPI 지수 차트")
        elif self._control_bar.plot_futures_rb.isChecked():
            self._selected_plot = "futures"
            if self._control_bar.title_lbl is not None:
                self._control_bar.title_lbl.setText("📈 KP200 선물 차트")

        # DataFetcher 상태 업데이트
        self._data_fetcher.set_selected_plot(self._selected_plot)

        # 플롯 변경 시 캐시 삭제
        self._clear_cache()
        self.refresh()

    def _on_ma_toggled(self, checked: bool) -> None:
        """MA 체크박스 토글."""
        if self._renderer is not None:
            self._renderer.set_ma_enabled(checked)
        self.refresh()

    def _on_pivot_toggled(self, checked: bool) -> None:
        """피봇 체크박스 토글."""
        self._show_pivots_enabled = checked
        self.refresh()

    def _on_adaptive_toggled(self, checked: bool) -> None:
        """Adaptive 체크박스 토글."""
        # 시장 레짐 기반 적응형 기능 활성화/비활성화
        self._adaptive_enabled = checked
        logger.info("[ChartViewerWidget] Adaptive 모드: %s", checked)

        # ChartEngine에 Adaptive 모드 전파
        if self._engine is not None:
            self._engine.set_adaptive_enabled(checked)

        # 피봇 수 자동 조정 (Adaptive ON 시 10개 목표)
        if checked:
            self._apply_pivot_count_target(target=10)
        else:
            self._restore_original_zigzag_config()

        # 시장 레짐 분류기 초기화 (UI 표시용)
        if checked and self._regime_classifier is None:
            try:
                from services.market_regime_classifier import MarketRegimeClassifier
                self._regime_classifier = MarketRegimeClassifier()
                logger.info("[ChartViewerWidget] 시장 레짐 분류기 초기화 완료")
            except Exception as e:
                logger.warning("[ChartViewerWidget] 시장 레짐 분류기 초기화 실패 (피봇 조정은 계속 실행): %s", e)
                # 체크박스 해제하지 않음 - 피봇 조정 기능은 계속 작동

        self.refresh()

    def _on_csv_load(self) -> None:
        """CSV 파일 로드."""
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self._parent,
            "CSV 파일 선택",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            logger.info("[ChartViewer] CSV 파일 선택: %s", file_path)
            self._data_fetcher.set_csv_file_path(file_path)
            
            # CSV 로드 후 데이터 소스 갱신 (DataFetcher에서 자동 감지된 값)
            # fetch()를 호출하여 데이터 소스 감지 트리거
            self._data_fetcher.fetch()
            new_source = self._data_fetcher._selected_plot
            if new_source != self._selected_plot:
                logger.info("[ChartViewer] CSV 데이터 소스 변경: %s → %s", self._selected_plot, new_source)
                self._selected_plot = new_source
                # 라디오 버튼 상태 갱신
                if self._control_bar:
                    if new_source == "kospi":
                        self._control_bar._plot_kospi_rb.setChecked(True)
                    else:
                        self._control_bar._plot_futures_rb.setChecked(True)
            
            # CSV 모드 플래그 설정
            self._csv_mode = True
            logger.info("[ChartViewer] CSV 백테스트 모드 활성화")
            
            # Adaptive 모드 비활성화 (CSV 백테스트 모드에서는 레짐 매핑 비활성)
            if hasattr(self, '_engine') and self._engine is not None:
                self._engine.set_adaptive_enabled(False)
                logger.info("[ChartViewer] CSV 모드 - Adaptive 모드 비활성화")
            
            # Config 다시 로드 (CSV 모드는 최신 config 사용)
            try:
                from config import AppConfig
                self._config = AppConfig.from_file("config.json")
                logger.info("[ChartViewer] CSV 모드 - Config 다시 로드 완료")
            except Exception as e:
                logger.warning("[ChartViewer] Config 다시 로드 실패: %s", e)
            
            # Engine 캐시 초기화 (CSV 모드는 실시간 데이터와 다르므로)
            if hasattr(self, '_engine') and self._engine is not None:
                self._engine._last_sig = None
                self._engine._replay_signature = None
                self._engine._zz_state_cache = {}
                self._engine._confirmed_pivots_cache = []
                self._engine._last_completed_ts = None
                self._engine._anchor_ts = None
                self._engine._zz = None
                logger.info("[ChartViewer] CSV 로드 시 Engine 캐시 초기화")
            
            self._clear_cache()
            self.refresh()

    def _apply_pivot_count_target(self, target: int = 10) -> None:
        """현재 피봇 수 기반으로 min_wave_bars를 역산해 config에 주입."""
        # Engine이 초기화되지 않았으면 조정 건너뜀
        if self._engine is None:
            logger.warning("[ChartViewer] Engine이 초기화되지 않음, 피봇 수 조정 건너뜀")
            return
        
        # 현재 피봇 수 계산
        current_count = 0
        zz = self._engine._zz
        if zz is not None and hasattr(zz, "_all_swings"):
            current_count = len([s for s in zz._all_swings if getattr(s, "confirmed", False)])

        # 피봇 수가 0이거나 zz가 None이면, 조정하지 않음
        if current_count <= 0:
            logger.warning("[ChartViewer] 피봇 수를 확인할 수 없음, 조정 건너뜀")
            return
        else:
            # 비율 계산을 더 보수적으로 수정 (최대 1.2 제한)
            import math
            # 피봇 수가 목표보다 많을 때만 조정 (적으면 조정하지 않음)
            if current_count > target:
                ratio = min(1.2, 1 + 0.05 * (current_count - target))  # ex) 19->10: 1 + 0.05*9 = 1.45 -> 1.2
            else:
                ratio = 1.0  # 피봇 수가 목표보다 적으면 조정하지 않음

        # 현재 데이터 소스에 따른 config 선택
        cfg = self._config.adaptive_indicator
        if self._selected_plot == "kospi":
            zz_cfg = cfg.kospi_zigzag or cfg.zigzag
            ds_key = "kospi"
        else:
            zz_cfg = cfg.futures_zigzag or cfg.zigzag
            ds_key = "futures"

        # 원래 설정 백업 (첫 번째 호출 시만)
        if ds_key not in self._original_zz_configs:
            self._original_zz_configs[ds_key] = {
                "session_min_wave_bars_table": list(getattr(zz_cfg, "session_min_wave_bars_table", [])),
            }

        # session_min_wave_bars_table 역산
        table = list(getattr(zz_cfg, "session_min_wave_bars_table", []))
        new_table = [
            [s, e, max(5, int(round(bars * ratio)))]
            for s, e, bars in table
        ]

        # config 업데이트
        zz_cfg.session_min_wave_bars_table = new_table

        # [SSOT] _zz_external=True 상태에서는 _init_zigzag()가 guard에 막혀 동작하지 않음.
        # 대신 주입된 ZigZag 인스턴스의 config 를 직접 수정하고 reset() 으로 반영한다.
        if zz is not None and hasattr(zz, 'config'):
            zz.config.session_min_wave_bars_table = new_table
            logger.info("[ChartViewer] zz.config 업데이트 완료 (SSOT 직접 수정)")
            # reset()으로 내부 상태를 초기화해 새 테이블을 적용한다.
            if hasattr(zz, 'reset'):
                try:
                    zz.reset()
                    logger.info("[ChartViewer] zz.reset() 완료")
                except Exception as e:
                    logger.warning("[ChartViewer] zz.reset() 실패: %s", e)
        else:
            # zz 인스턴스 없음 → 데이터 로드 후 재시도 예약
            logger.info("[ChartViewer] zz가 None이어서 데이터 로드 후 재시작 예약")
            self._adaptive_pending_reapply = True

        # 캐시 무효화 → 다음 compute에서 재replay
        self._engine._last_sig = None
        self._engine._replay_signature = None
        self._engine._zz_state_cache = {}

        logger.info("[ChartViewer] Adaptive 피봇 목표 %d개: 현재=%d, ratio=%.2f, 새 테이블=%s",
                    target, current_count, ratio, new_table)

    def _restore_original_zigzag_config(self) -> None:
        """원래 ZigZag 설정 복원."""
        cfg = self._config.adaptive_indicator
        ds_key = self._selected_plot  # "kospi" or "futures"

        if ds_key not in self._original_zz_configs:
            logger.warning("[ChartViewer] 원래 설정 백업 없음: %s", ds_key)
            return

        # 원래 설정 복원
        original = self._original_zz_configs[ds_key]
        if ds_key == "kospi":
            zz_cfg = cfg.kospi_zigzag or cfg.zigzag
        else:
            zz_cfg = cfg.futures_zigzag or cfg.zigzag

        zz_cfg.session_min_wave_bars_table = original["session_min_wave_bars_table"]

        # [SSOT] 직접 주입된 ZigZag 인스턴스의 config 를 복원하고 reset()
        zz = self._engine._zz
        if zz is not None and hasattr(zz, 'config'):
            zz.config.session_min_wave_bars_table = original["session_min_wave_bars_table"]
            logger.info("[ChartViewer] zz.config 복원 완료 (SSOT 직접 수정)")
            if hasattr(zz, 'reset'):
                try:
                    zz.reset()
                    logger.info("[ChartViewer] zz.reset() 완료 (복원)")
                except Exception as e:
                    logger.warning("[ChartViewer] zz.reset() 실패 (복원): %s", e)

        # 캐시 무효화
        self._engine._last_sig = None
        self._engine._replay_signature = None
        self._engine._zz_state_cache = {}

        logger.info("[ChartViewer] 원래 ZigZag 설정 복원: %s", original["session_min_wave_bars_table"])

    def _generate_regime_change_report(self, period: str = "daily") -> str:
        """레짐 변경 통계 리포트 생성.

        Args:
            period: "daily" 또는 "weekly"

        Returns:
            리포트 문자열
        """
        current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        current_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

        if period == "daily":
            # 일일 리포트
            today_changes = self._regime_change_stats.get(current_date, 0)
            today_history = [h for h in self._regime_change_history if h["time"].startswith(current_date)]

            # 레짐별 빈도 계산
            regime_counts = {}
            for h in today_history:
                regime = h["current"]
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

            report = (
                f"📊 레짐 변경 일일 리포트\n"
                f"날짜: {current_date}\n"
                f"시간: {current_time}\n"
                f"총 변경 횟수: {today_changes}회\n\n"
            )

            if regime_counts:
                report += "레짐별 빈도:\n"
                for regime, count in sorted(regime_counts.items(), key=lambda x: x[1], reverse=True):
                    report += f"  - {regime}: {count}회\n"

            if today_history:
                report += "\n최근 변경 이력:\n"
                for h in today_history[-5:]:  # 최근 5개
                    report += f"  {h['time']}: {h['previous']} → {h['current']} ({h['symbol']})\n"

        elif period == "weekly":
            # 주간 리포트
            week_start = pd.Timestamp.now() - pd.Timedelta(days=6)
            week_dates = [(week_start + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]

            total_changes = 0
            regime_counts = {}
            daily_changes = {}

            for date in week_dates:
                changes = self._regime_change_stats.get(date, 0)
                daily_changes[date] = changes
                total_changes += changes

                # 해당 날짜의 이력에서 레짐별 빈도 계산
                for h in self._regime_change_history:
                    if h["time"].startswith(date):
                        regime = h["current"]
                        regime_counts[regime] = regime_counts.get(regime, 0) + 1

            report = (
                f"📊 레짐 변경 주간 리포트\n"
                f"기간: {week_dates[0]} ~ {week_dates[-1]}\n"
                f"시간: {current_time}\n"
                f"총 변경 횟수: {total_changes}회\n\n"
            )

            if daily_changes:
                report += "일별 변경 횟수:\n"
                for date in week_dates:
                    changes = daily_changes.get(date, 0)
                    report += f"  {date}: {changes}회\n"

            if regime_counts:
                report += "\n레짐별 빈도:\n"
                for regime, count in sorted(regime_counts.items(), key=lambda x: x[1], reverse=True):
                    report += f"  - {regime}: {count}회\n"

        return report

    def send_regime_change_report(self, period: str = "daily") -> None:
        """레짐 변경 통계 리포트를 텔레그램으로 전송.

        Args:
            period: "daily" 또는 "weekly"
        """
        if not self._telegram_bridge_holder:
            logger.warning("[ChartViewer] 텔레그램 bridge 홀더 없음 - 리포트 전송 불가")
            return

        bridge = self._telegram_bridge_holder.get("bridge")
        if not bridge or not hasattr(bridge, "notifier"):
            logger.warning("[ChartViewer] 텔레그램 notifier 없음 - 리포트 전송 불가")
            return

        try:
            report = self._generate_regime_change_report(period)
            bridge.notifier.send_text(report)
            logger.info("[ChartViewer] 레짐 변경 %s 리포트 전송 완료", period)
        except Exception as e:
            logger.warning("[ChartViewer] 레짐 변경 리포트 전송 실패: %s", e)

    def check_and_send_daily_report(self) -> None:
        """날짜가 변경되었으면 일일 리포트 전송."""
        current_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        if self._last_daily_report_date != current_date:
            # 이전 날짜에 대한 리포트 전송
            if self._last_daily_report_date and self._regime_change_stats.get(self._last_daily_report_date, 0) > 0:
                logger.info("[ChartViewer] 일일 리포트 전송: %s", self._last_daily_report_date)
                self.send_regime_change_report("daily")

            self._last_daily_report_date = current_date

    def _load_regime_stats(self) -> None:
        """레짐 변경 통계를 파일에서 로드."""
        try:
            import os
            from pathlib import Path

            stats_path = Path(self._regime_stats_file)
            if not stats_path.exists():
                logs_dir = stats_path.parent
                if not logs_dir.exists():
                    logs_dir.mkdir(parents=True, exist_ok=True)
                logger.info("[ChartViewer] 레짐 통계 파일 없음 - 새로 생성")
                return

            with open(stats_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._regime_change_stats = data.get('stats', {})
                self._regime_change_history = data.get('history', [])
                self._last_daily_report_date = data.get('last_report_date')

                # 최근 100개만 유지
                if len(self._regime_change_history) > 100:
                    self._regime_change_history = self._regime_change_history[-100:]

            logger.info("[ChartViewer] 레짐 통계 로드 완료: %d일 데이터, %d개 이력",
                      len(self._regime_change_stats), len(self._regime_change_history))
        except Exception as e:
            logger.warning("[ChartViewer] 레짐 통계 로드 실패: %s", e)

    def _save_regime_stats(self) -> None:
        """레짐 변경 통계를 파일에 저장."""
        try:
            import os
            from pathlib import Path

            stats_path = Path(self._regime_stats_file)
            logs_dir = stats_path.parent
            if not logs_dir.exists():
                logs_dir.mkdir(parents=True, exist_ok=True)

            data = {
                'stats': self._regime_change_stats,
                'history': self._regime_change_history,
                'last_report_date': self._last_daily_report_date,
                'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.debug("[ChartViewer] 레짐 통계 저장 완료")
        except Exception as e:
            logger.warning("[ChartViewer] 레짐 통계 저장 실패: %s", e)

    def _load_regime_trading_config(self) -> None:
        """레짐 트레이딩 설정 로드."""
        try:
            market_regime_config = {}
            if self._config and hasattr(self._config, 'market_regime'):
                market_regime_config = self._config.market_regime or {}
            elif isinstance(self._config, dict):
                market_regime_config = self._config.get('market_regime', {})

            trading_config = market_regime_config.get('trading', {})
            self._regime_trading_enabled = trading_config.get('enabled', False)
            self._regime_trading_config = trading_config

            if self._regime_trading_enabled:
                logger.info("[ChartViewer] 레짐 트레이딩 활성화: %s", trading_config)
            else:
                logger.info("[ChartViewer] 레짐 트레이딩 비활성화")
        except Exception as e:
            logger.warning("[ChartViewer] 레짐 트레이딩 설정 로드 실패: %s", e)

    def _generate_regime_signal(self, prev_regime: str, curr_regime: str, confidence: float) -> str:
        """레짐 변경 기반 매매 신호 생성.

        Args:
            prev_regime: 이전 레짐
            curr_regime: 현재 레짐
            confidence: 신뢰도

        Returns:
            신호: "STRONG_BUY", "BUY", "WEAK_BUY", "STRONG_SELL", "SELL", "WEAK_SELL", "HOLD"
        """
        if not self._regime_trading_enabled:
            return "HOLD"

        config = self._regime_trading_config
        confidence_threshold = config.get('confidence_threshold', 0.7)

        # 신뢰도 필터
        if confidence < confidence_threshold:
            logger.debug("[ChartViewer] 신뢰도 부족으로 신호 무시: %.2f < %.2f", confidence, confidence_threshold)
            return "HOLD"

        # 시간당 변경 횟수 필터
        current_hour = pd.Timestamp.now().strftime('%Y-%m-%d %H')
        if self._last_regime_change_hour != current_hour:
            self._regime_change_hourly_count = 0
            self._last_regime_change_hour = current_hour

        max_changes_per_hour = config.get('max_changes_per_hour', 3)
        if self._regime_change_hourly_count >= max_changes_per_hour:
            logger.warning("[ChartViewer] 시간당 변경 횟수 초과로 신호 무시: %d >= %d",
                         self._regime_change_hourly_count, max_changes_per_hour)
            return "HOLD"

        # 기본 신호 규칙
        signal = "HOLD"

        if prev_regime == "TREND_DOWN" and curr_regime == "TREND_UP":
            signal = "BUY"  # 기본 매수
        elif prev_regime == "TREND_UP" and curr_regime == "TREND_DOWN":
            signal = "SELL"  # 기본 매도
        elif prev_regime == "RANGING" and curr_regime == "TREND_UP":
            signal = "BUY"  # 횡보에서 상승 추세 전환
        elif prev_regime == "RANGING" and curr_regime == "TREND_DOWN":
            signal = "SELL"  # 횡보에서 하락 추세 전환
        elif prev_regime == "VOLATILE" and curr_regime == "TREND_UP":
            signal = "BUY"  # 변동성에서 상승 안정화
        elif prev_regime == "VOLATILE" and curr_regime == "TREND_DOWN":
            signal = "SELL"  # 변동성에서 하락 안정화
        elif prev_regime in ["TREND_UP", "TREND_DOWN"] and curr_regime == "RANGING":
            signal = "HOLD"  # 추세 소멸 - 관망
        elif prev_regime == "RANGING" and curr_regime == "VOLATILE":
            signal = "HOLD"  # 변동성 급증 - 관망

        # 지표 확인 (선택적)
        if signal in ["BUY", "SELL"]:
            signal = self._apply_indicator_confirmations(signal, prev_regime, curr_regime)

        if signal != "HOLD":
            self._regime_change_hourly_count += 1
            logger.info("[ChartViewer] 레짐 매매 신호 생성: %s (%s → %s, 신뢰도: %.2f)",
                      signal, prev_regime, curr_regime, confidence)

        return signal

    def _apply_indicator_confirmations(self, base_signal: str, prev_regime: str, curr_regime: str) -> str:
        """다른 지표와의 일치 여부 확인하여 신호 강도 조정.

        Args:
            base_signal: 기본 신호
            prev_regime: 이전 레짐
            curr_regime: 현재 레짐

        Returns:
            조정된 신호
        """
        config = self._regime_trading_config
        use_zigzag = config.get('use_zigzag_confirmation', True)
        use_supertrend = config.get('use_supertrend_confirmation', True)
        use_sentiment = config.get('use_sentiment_confirmation', True)

        confirmations = 0
        total_confirmations = 0

        # ZigZag 확인
        if use_zigzag:
            total_confirmations += 1
            try:
                zz = self._engine._zz
                if zz is not None and hasattr(zz, "_all_swings"):
                    # 최근 피봇 방향 확인
                    swings = zz._all_swings
                    if swings:
                        last_swing = swings[-1]
                        swing_direction = "up" if getattr(last_swing, "is_high", False) else "down"

                        if base_signal == "BUY" and swing_direction == "up":
                            confirmations += 1
                        elif base_signal == "SELL" and swing_direction == "down":
                            confirmations += 1
            except Exception as e:
                logger.debug("[ChartViewer] ZigZag 확인 실패: %s", e)

        # SuperTrend 확인
        if use_supertrend:
            total_confirmations += 1
            try:
                # SuperTrend 방향 확인 (구현 필요)
                # 현재는 pass로 처리
                pass
            except Exception as e:
                logger.debug("[ChartViewer] SuperTrend 확인 실패: %s", e)

        # 옵션 센티먼트 확인
        if use_sentiment:
            total_confirmations += 1
            try:
                # 옵션 센티먼트 확인 (구현 필요)
                # 현재는 pass로 처리
                pass
            except Exception as e:
                logger.debug("[ChartViewer] 옵션 센티먼트 확인 실패: %s", e)

        # 신호 강도 조정
        if total_confirmations > 0:
            confirmation_ratio = confirmations / total_confirmations

            if confirmation_ratio >= 0.67:  # 2/3 이상 일치
                if base_signal == "BUY":
                    return "STRONG_BUY"
                elif base_signal == "SELL":
                    return "STRONG_SELL"
            elif confirmation_ratio >= 0.33:  # 1/3 이상 일치
                return base_signal
            else:  # 일치하는 지표 없음
                if base_signal == "BUY":
                    return "WEAK_BUY"
                elif base_signal == "SELL":
                    return "WEAK_SELL"

        return base_signal

    def _on_parameter_dialog(self) -> None:
        """파라미터 조정 다이얼로그 열기."""
        from pathlib import Path
        try:
            from gui.parameter_dialog import ParameterDialog
        except ImportError:
            logger.error("[ChartViewer] parameter_dialog 모듈을 찾을 수 없습니다.")
            return

        config_path = Path("config.json")
        if not config_path.exists():
            logger.error("[ChartViewer] config.json 파일을 찾을 수 없습니다.")
            return

        dialog = ParameterDialog(
            config_path=config_path,
            data_source=self._selected_plot,
            parent=self.widget,
        )
        dialog.exec()

    # ── 데이터 취득 ───────────────────────────────────────────────────────────

    def _get_df(self) -> pd.DataFrame:
        """데이터 가져오기 (의존성 주입된 data_fetcher 사용)."""
        return self._data_fetcher.fetch()

    def _get_current_price(self) -> float:
        """tick_processor에서 현재가를 가져온다. 시뮬레이션 중에는 가상 현재가 사용."""
        logger.info("[ChartViewer] _get_current_price 호출 (selected_plot=%s)", self._selected_plot)
        # 시뮬레이션 중이면 가상 현재가 사용
        if self._sim_active and hasattr(self, "_sim_current_price"):
            logger.info("[ChartViewer] 시뮬레이션 현재가: %s", self._sim_current_price)
            return self._sim_current_price

        try:
            return self._data_fetcher.get_current_price()
        except Exception as e:
            logger.warning("[ChartViewer] 현재가 가져오기 실패: %s", e)
        return 0.0

    # ── 공개 메서드 ────────────────────────────────────────────────────────────

    def set_trade_status(self, status: str) -> None:
        """매매 상태 LED 설정.
        
        Args:
            status: 상태 ("idle", "long_entry", "short_entry", "long_hold", "short_hold", "exit")
        """
        if self._control_bar and self._control_bar.trade_led:
            self._control_bar.trade_led.set_status(status)
    
    def get_trade_status(self) -> str:
        """현재 매매 상태 반환.
        
        Returns:
            현재 상태
        """
        if self._control_bar and self._control_bar.trade_led:
            return self._control_bar.trade_led.get_status()
        return "idle"
    
    def set_led_sync_enabled(self, enabled: bool) -> None:
        """LED 동기화 활성화/비활성화.
        
        Args:
            enabled: 활성화 여부
        """
        self._led_sync_enabled = enabled
        logger.info("[ChartViewerWidget] LED 동기화: %s", "활성화" if enabled else "비활성화")
    
    def is_led_sync_enabled(self) -> bool:
        """LED 동기화 활성화 여부 반환.
        
        Returns:
            활성화 여부
        """
        return self._led_sync_enabled
    
    def set_trade_markers_enabled(self, enabled: bool) -> None:
        """거래 마커 활성화/비활성화.
        
        Args:
            enabled: 활성화 여부
        """
        self._trade_markers_enabled = enabled
        logger.info("[ChartViewerWidget] 거래 마커: %s", "활성화" if enabled else "비활성화")
    
    def is_trade_markers_enabled(self) -> bool:
        """거래 마커 활성화 여부 반환.
        
        Returns:
            활성화 여부
        """
        return self._trade_markers_enabled
    
    def set_risk_monitoring_enabled(self, enabled: bool) -> None:
        """리스크 모니터링 활성화/비활성화.
        
        Args:
            enabled: 활성화 여부
        """
        self._risk_monitoring_enabled = enabled
        logger.info("[ChartViewerWidget] 리스크 모니터링: %s", "활성화" if enabled else "비활성화")
    
    def is_risk_monitoring_enabled(self) -> bool:
        """리스크 모니터링 활성화 여부 반환.
        
        Returns:
            활성화 여부
        """
        return self._risk_monitoring_enabled

    def _is_market_closed(self) -> bool:
        """장 마감 시간 감지 (KP200: 08:45~15:45, KOSPI: 09:00~15:30, 주말 포함)"""
        try:
            from datetime import datetime
            now = datetime.now()
            # 주말
            if now.weekday() >= 5:
                return True

            # 선택된 플롯에 따른 장 시간 적용
            if self._selected_plot == "futures":  # KP200 선물
                # 장전 (08:45 이전)
                if now.hour < 8 or (now.hour == 8 and now.minute < 45):
                    return True
                # 장후 (15:45 이후)
                if now.hour > 15 or (now.hour == 15 and now.minute >= 45):
                    return True
            else:  # KOSPI 지수
                # 장전 (09:00 이전)
                if now.hour < 9:
                    return True
                # 장후 (15:30 이후)
                if now.hour > 15 or (now.hour == 15 and now.minute >= 30):
                    return True

            return False
        except Exception:
            return False

    def _should_auto_refresh(self) -> bool:
        """자동 갱신 여부 결정 (장 마감 시 중지)"""
        if not self._auto_refresh_enabled:
            logger.debug("[ChartViewer] 자동 갱신 비활성화됨")
            return False
        if self._is_market_closed():
            logger.debug("[ChartViewer] 장 마감 상태")
            return False
        logger.debug("[ChartViewer] 자동 갱신 조건 충족")
        return True

    def _get_cache_key(self) -> str:
        """캐시 키 생성"""
        return self._cache_manager.get_cache_key(self._selected_plot, self._minutes)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """캐시 유효성 검사"""
        return self._cache_manager.is_cache_valid(cache_key)

    def _clear_cache(self) -> None:
        """캐시 전체 삭제"""
        self._cache_manager.clear()

    def _prepare_refresh(self) -> None:
        """[SSOT] predictor._adaptive_mgr 의 ZigZag 인스턴스를 chart_engine 에 주입한다.

        단일 소스 원칙(SSOT):
        - 차트와 predictor 가 동일한 AdaptiveZigZag 인스턴스를 공유해야 한다.
        - data_source(futures/kospi)에 따라 적합한 인스턴스를 선택하여 주입한다.
        - 인스턴스가 이전과 동일하면 set_zigzag 를 호출하지 않아 불필요한 replay 를 방지한다.
        - predictor 가 없거나 _adaptive_mgr 가 없으면 chart_engine 내부 생성으로 폴백한다.
        """
        if not hasattr(self, '_engine') or self._engine is None:
            return  # _compute_data 에서 engine 초기화 후 재호출됨

        mgr = None
        if self._predictor is not None:
            mgr = getattr(self._predictor, '_adaptive_mgr', None)

        if mgr is None:
            # predictor 없음 → chart_engine 내부 생성 유지 (폴백)
            logger.warning("[ChartViewer][RT] _prepare_refresh: _adaptive_mgr 없음 — chart_engine 내부 ZigZag 폴백 사용")
            return

        # data_source 에 맞는 ZigZag 인스턴스 선택
        ds = getattr(self._engine, '_current_data_source', None) or self._selected_plot
        if ds == 'futures':
            zz_instance = getattr(mgr, 'futures_zigzag', None) or getattr(mgr, 'zigzag', None)
        elif ds == 'kospi':
            zz_instance = getattr(mgr, 'kospi_zigzag', None) or getattr(mgr, 'zigzag', None)
        else:
            zz_instance = getattr(mgr, 'zigzag', None)

        if zz_instance is None:
            logger.error(
                "[ChartViewer][RT] _prepare_refresh: mgr.zigzag=None (ds=%s) — SSOT 인스턴스 없음, 차트 ZigZag가 predictor와 분리됨",
                ds,
            )
            return

        # 이미 동일 인스턴스가 주입된 경우 skip (불필요한 replay 방지)
        # [FIX] _feed_zigzag에서 _zz가 재생성될 수 있으므로 _zz_external_origin으로 비교
        current_origin = getattr(self._engine, '_zz_external_origin', None)
        current_zz = getattr(self._engine, '_zz', None)
        if (current_origin is zz_instance) or (current_origin is None and current_zz is zz_instance):
            logger.debug("[ChartViewer] _prepare_refresh: ZigZag 인스턴스 동일 — skip")
            return

        logger.info(
            "[ChartViewer] _prepare_refresh: predictor ZigZag 주입 (data_source=%s, id=%s)",
            ds, id(zz_instance),
        )
        self._engine.set_zigzag(zz_instance, data_source=ds)

        # ── [SSOT] SuperTrend 주입 ────────────────────────────────────────────
        # set_zigzag()와 동일한 패턴:
        #   mgr.get_supertrend(ds) → engine.set_supertrend(st)
        # _st_external_origin 동일성 체크로 불필요한 캐시 무효화 방지
        try:
            st_instance = mgr.get_supertrend(ds) if callable(getattr(mgr, 'get_supertrend', None)) else None
            if st_instance is not None:
                current_st_origin = getattr(self._engine, '_st_external_origin', None)
                if current_st_origin is not st_instance:
                    self._engine.set_supertrend(st_instance, data_source=ds)
                    logger.info(
                        "[ChartViewer] SuperTrend 주입 (SSOT): ds=%s id=%s",
                        ds, id(st_instance),
                    )
                else:
                    logger.debug("[ChartViewer] SuperTrend 인스턴스 동일 — skip")
            else:
                logger.debug("[ChartViewer] _prepare_refresh: mgr.get_supertrend=None (ds=%s) — 폴백 유지", ds)
        except Exception as _st_e:
            logger.warning("[ChartViewer] SuperTrend 주입 실패: %s", _st_e)
        # ─────────────────────────────────────────────────────────────────────

    def _compute_data(self) -> Tuple[pd.DataFrame, Dict]:
        """데이터 계산: _get_df + engine.compute"""
        import time as _time
        _cd_t0 = _time.perf_counter()
        logger.info("[ChartViewer][RT] _compute_data 시작 (plot=%s csv=%s)", self._selected_plot, self._csv_mode)

        # _engine 초기화 체크
        if not hasattr(self, '_engine') or self._engine is None:
            from gui.engines.chart_engine import ChartEngine
            self._engine = ChartEngine()
            self._engine.set_max_bars(self._minutes)
            logger.info("[ChartViewer] _engine 초기화 완료")

        # 매번 콜백 설정 (초기화 상관없이)
        self._engine.set_pivot_candidate_callback(self._on_pivot_candidate_event)
        logger.info("[ChartViewer] _engine 콜백 설정 완료")

        # predictor._adaptive_mgr 콜백 설정 (듀얼 모드 지원)
        if self._predictor and hasattr(self._predictor, '_adaptive_mgr'):
            try:
                self._predictor._adaptive_mgr.set_pivot_candidate_callback(self._on_pivot_candidate_event)
                logger.info("[ChartViewer] predictor._adaptive_mgr 콜백 설정 완료")
            except Exception as e:
                logger.warning("[ChartViewer] predictor._adaptive_mgr 콜백 설정 실패: %s", e)

        # ZigZag 교체 가능성 확인 (CSV 모드 제외)
        if not self._csv_mode:
            self._prepare_refresh()

        logger.info("[ChartViewer][RT] _get_df 호출 (data_fetcher=%s)", self._data_fetcher is not None)
        _gdf_t0 = _time.perf_counter()
        df_raw = self._get_df()
        _gdf_elapsed = _time.perf_counter() - _gdf_t0
        if df_raw is None or (hasattr(df_raw, "empty") and df_raw.empty):
            logger.warning("[ChartViewer][RT] _get_df 결과: None/빈 DataFrame — elapsed=%.3fs", _gdf_elapsed)
        else:
            logger.info(
                "[ChartViewer][RT] _get_df 결과: bars=%d range=[%s ~ %s] elapsed=%.3fs",
                len(df_raw),
                df_raw.index[0] if len(df_raw) else "N/A",
                df_raw.index[-1] if len(df_raw) else "N/A",
                _gdf_elapsed,
            )
        
        # 데이터 수신 로그 및 플래그 확인
        if df_raw is not None and not df_raw.empty:
            current_count = len(df_raw)
            current_timestamp = df_raw.index[-1].timestamp() if hasattr(df_raw.index[-1], 'timestamp') else time.time()
            
            # 새로운 데이터 수신 확인
            if self._last_data_count == 0:
                self._new_data_received = True
                logger.info("[ChartViewer] 초기 데이터 수신: %d 봉, 마지막 타임스탬프: %s", current_count, df_raw.index[-1])
            elif current_count != self._last_data_count or current_timestamp != self._last_data_timestamp:
                self._new_data_received = True
                logger.info("[ChartViewer] 새로운 데이터 수신: 이전 %d 봉 -> 현재 %d 봉, 타임스탬프: %s", 
                           self._last_data_count, current_count, df_raw.index[-1])
            else:
                # 실시간 모드에서는 데이터가 있으면 항상 갱신 (틱 추가로 인한 갱신 필요)
                if not self._csv_mode:
                    self._new_data_received = True
                    logger.debug("[ChartViewer] 실시간 모드 - 데이터 갱신 (틱 추가로 인한 갱신): %d 봉", current_count)
                else:
                    self._new_data_received = False
                    logger.debug("[ChartViewer] 데이터 변경 없음: %d 봉", current_count)
            
            self._last_data_count = current_count
            self._last_data_timestamp = current_timestamp
        else:
            self._new_data_received = False
            logger.warning("[ChartViewer] 데이터 수신 실패 또는 빈 데이터")
        
        logger.info("[ChartViewer][refresh] _get_df 완료 rows=%d, new_data=%s", len(df_raw) if df_raw is not None else 0, self._new_data_received)

        # CSV 모드에서는 캐시 강제 초기화
        csv_force_recompute = False
        if self._csv_mode and hasattr(self, '_engine') and self._engine is not None:
            logger.info("[ChartViewer] CSV 모드 - Engine 캐시 강제 초기화")
            self._engine._last_sig = None
            self._engine._replay_signature = None
            self._engine._zz_state_cache = {}
            self._engine._confirmed_pivots_cache = []
            self._engine._last_completed_ts = None
            self._engine._anchor_ts = None
            # CSV 모드에서 ZigZag 인스턴스도 삭제하여 데이터 소스 분리 강화
            self._engine._zz = None
            logger.info("[ChartViewer] CSV 모드 - ZigZag 인스턴스 삭제 (데이터 소스 분리)")
            # CSV 모드에서 Adaptive 모드 비활성화
            self._adaptive_enabled = False  # 내부 상태도 비활성화
            self._engine.set_adaptive_enabled(False)
            # 체크박스 상태도 비활성화
            if self._control_bar and hasattr(self._control_bar, '_adaptive_cb'):
                self._control_bar._adaptive_cb.blockSignals(True)
                self._control_bar._adaptive_cb.setChecked(False)
                self._control_bar._adaptive_cb.blockSignals(False)
            # 레짐 라벨 숨김
            if self._regime_label_callback:
                try:
                    self._regime_label_callback("Regime: -", visible=False)
                except Exception as e:
                    logger.warning("[ChartViewer] 레짐 라벨 숨김 실패: %s", e)
            logger.info("[ChartViewer] CSV 모드 - Adaptive 모드 비활성화")
            # _zz 재초기화 (데이터 소스 변경으로 간주)
            self._engine._current_data_source = None  # 데이터 소스 변경 강제
            self._csv_mode = False  # 초기화 후 플래그 리셋
            csv_force_recompute = True  # CSV 모드 강제 재계산 플래그 저장
        else:
            # CSV 모드가 아닐 때는 config의 adaptive_mode 사용
            if self._config and hasattr(self._config, 'adaptive_mode'):
                adaptive_enabled = self._config.adaptive_mode
                self._adaptive_enabled = adaptive_enabled  # 내부 상태도 동기화
                self._engine.set_adaptive_enabled(adaptive_enabled)
                # 체크박스 상태도 동기화 (config 변경 시 UI 반영)
                if self._control_bar and hasattr(self._control_bar, '_adaptive_cb'):
                    self._control_bar._adaptive_cb.blockSignals(True)  # 신호 차단하여 무한 루프 방지
                    self._control_bar._adaptive_cb.setChecked(adaptive_enabled)
                    self._control_bar._adaptive_cb.blockSignals(False)
                logger.info("[ChartViewer] Adaptive 모드 설정: %s (config.adaptive_mode)", adaptive_enabled)

        # 범위 변경 감지 (플래그 확인)
        force_recompute = self._minutes_changed or csv_force_recompute  # 범위 변경 또는 CSV 모드 시 강제 재계산

        # [BUG-FIX] 데이터소스 변경 감지를 compute()에 위임
        # 기존: _engine._current_data_source를 여기서 사전 세팅 → compute() 내부의
        #        변경 감지 조건(current != data_source)이 항상 False가 되어 캐시 초기화 안 됨
        # 수정: 사전 세팅 제거, compute(data_source=selected_plot)에 그대로 전달
        #        compute() 내부에서 변경 감지 후 _zz, 캐시 초기화 수행
        source_changed = (self._engine._current_data_source != self._selected_plot)
        if source_changed:
            logger.info("[ChartViewer] 데이터 소스 변경: %s → %s",
                        self._engine._current_data_source, self._selected_plot)
            # 데이터 소스 변경 시 Predictor ZigZag 인스턴스 재주입
            # 기존: _zz=None으로 삭제하여 재초기화 유도
            # 수정: _prepare_refresh()를 호출하여 Predictor의 적절한 ZigZag 인스턴스 주입
            self._prepare_refresh()
            logger.info("[ChartViewer] 데이터 소스 변경 - Predictor ZigZag 인스턴스 재주입 완료")
            # 데이터 소스 변경 시 MA 체크박스 OFF 유지
            self._control_bar._ma_cb.setChecked(False)
            # 데이터 소스 변경 시 강제 재계산 (SuperTrend 캐시 초기화)
            force_recompute = True
            # 피봇 조정은 여기서 수행 (compute 전), 단 _current_data_source 세팅은 하지 않음
            if self._adaptive_enabled and not self._adaptive_adjusting:
                logger.info("[ChartViewer] 데이터 소스 변경 시 피벗 조정 수행 (데이터 계산 전)")
                self._adaptive_adjusting = True
                self._apply_pivot_count_target(target=10)
                self._adaptive_adjusting = False

        df, pm = self._engine.compute(df_raw, self._config, self._selected_plot, force_recompute=force_recompute)
        _eng_elapsed = _time.perf_counter() - _cd_t0
        if df is None or (hasattr(df, "empty") and df.empty):
            logger.error("[ChartViewer][RT] engine.compute 결과: 빈 df — elapsed=%.3fs", _eng_elapsed)
        else:
            _pm_conf  = len(pm.get("confirmed",  {}).get("idx", [])) if pm else 0
            _pm_unconf= len(pm.get("unconfirmed",{}).get("idx", [])) if pm else 0
            logger.info(
                "[ChartViewer][RT] engine.compute 완료: bars=%d pivot(확정=%d 미확정=%d) elapsed=%.3fs",
                len(df), _pm_conf, _pm_unconf, _eng_elapsed,
            )

        # 플래그 초기화
        self._minutes_changed = False

        logger.debug("[ChartViewer][refresh] engine.compute 완료 bars=%d", len(df))

        # ── 마지막 분봉 Close를 현재가로 실시간 동기화 ──────────────────
        if not df.empty:
            current_price = self._get_current_price()
            if current_price > 0:
                # 마지막 봉의 High/Low도 현재가에 맞게 확장
                df.loc[df.index[-1], "Close"] = current_price
                df.loc[df.index[-1], "High"] = max(
                    float(df["High"].iloc[-1]), current_price
                )
                df.loc[df.index[-1], "Low"] = min(
                    float(df["Low"].iloc[-1]), current_price
                )
        # ────────────────────────────────────────────────────────────────

        return df, pm

    def _render_chart(self, df: pd.DataFrame, pm: Dict, force_clear: bool) -> bool:
        """차트 렌더링

        Qt 위젯 조작은 메인 스레드에서 수행되어야 하므로,
        렌더링 중 주기적으로 Qt 이벤트 루프를 처리하여 UI 응답성을 유지합니다.

        Returns:
            실제로 차트가 재렌더링되었으면 True, 증분 업데이트만 수행되면 False
        """
        import time as _time
        _rc_t0 = _time.perf_counter()
        logger.info(
            "[ChartViewer][RT] _render_chart 시작: bars=%d pivot_markers=%s force_clear=%s",
            len(df), pm is not None, force_clear,
        )
        # 렌더링 시간 측정 시작
        render_start_time = time.time()

        logger.info("[ChartViewer] _render_chart 호출 (df.shape=%s, force_clear=%s)", df.shape if hasattr(df, 'shape') else 'N/A', force_clear)

        # Adaptive 피봇 조정 재적용 (데이터 로드 후)
        # force_clear일 때는 이미 데이터가 다시 계산되므로 조정 건너뜀
        # 데이터 소스 변경 감지는 _compute_data에서 처리
        if self._adaptive_enabled and not self._adaptive_adjusting and not force_clear:
            # 데이터 로드 후 피봇 조정 재적용
            if self._adaptive_pending_reapply:
                logger.info("[ChartViewer] 데이터 로드 후 Adaptive 피봇 조정 재적용")
                self._adaptive_adjusting = True
                self._apply_pivot_count_target(target=10)
                self._adaptive_adjusting = False
                self._adaptive_pending_reapply = False

        # 빈 데이터 체크
        if df is None or df.empty:
            logger.warning("[ChartViewer] 빈 데이터프레임 - 렌더링 스킵")
            with self._render_lock:
                self._is_rendering = False  # [BUG-1] lock 안에서 해제
            return False

        # 시장 레짐 분류 (Adaptive 체크박스와 무관하게 항상 실행)
        try:
            # 레짐 분류기 초기화 (필요한 경우)
            if self._regime_classifier is None:
                try:
                    from services.market_regime_classifier import MarketRegimeClassifier

                    # config에서 market_regime 설정 읽기
                    market_regime_config = {}
                    if self._config and hasattr(self._config, 'market_regime'):
                        market_regime_config = self._config.market_regime or {}
                    elif isinstance(self._config, dict):
                        market_regime_config = self._config.get('market_regime', {})

                    # 프리셋 로드
                    preset_name = market_regime_config.get('preset', 'balanced')
                    presets = market_regime_config.get('presets', {})
                    preset_config = presets.get(preset_name, {})

                    # 프리셋 설정 적용 (프리셋에 없으면 기본값 사용)
                    enable_option_sentiment = preset_config.get('enable_option_sentiment', False)
                    sentiment_boost = preset_config.get('sentiment_confidence_boost', 0.2)
                    sentiment_penalty = preset_config.get('sentiment_confidence_penalty', 0.2)
                    enable_enhanced_trend = preset_config.get('enable_enhanced_trend', True)
                    ma_short_period = preset_config.get('ma_short_period', 20)
                    ma_long_period = preset_config.get('ma_long_period', 60)
                    adx_trend_threshold = preset_config.get('adx_trend_threshold', 25)
                    adx_weak_threshold = preset_config.get('adx_weak_threshold', 15)
                    vol_high_threshold = preset_config.get('vol_high_threshold', 0.02)
                    vol_low_threshold = preset_config.get('vol_low_threshold', 0.005)

                    logger.info("[ChartViewer] 레짐 탐색 프리셋: %s", preset_name)

                    self._regime_classifier = MarketRegimeClassifier(
                        enable_option_sentiment=enable_option_sentiment,
                        sentiment_confidence_boost=sentiment_boost,
                        sentiment_confidence_penalty=sentiment_penalty,
                        enable_enhanced_trend=enable_enhanced_trend,
                        ma_short_period=ma_short_period,
                        ma_long_period=ma_long_period,
                        adx_trend_threshold=adx_trend_threshold,
                        adx_weak_threshold=adx_weak_threshold,
                        vol_high_threshold=vol_high_threshold,
                        vol_low_threshold=vol_low_threshold,
                    )
                    logger.info(
                        "[ChartViewerWidget] 시장 레짐 분류기 초기화 완료 "
                        "(옵션 센티먼트: %s, boost: %.2f, penalty: %.2f, 향상된 추세: %s, MA: %d/%d, "
                        "ADX: %d/%d, VOL: %.3f/%.3f)",
                        enable_option_sentiment, sentiment_boost, sentiment_penalty,
                        enable_enhanced_trend, ma_short_period, ma_long_period,
                        adx_trend_threshold, adx_weak_threshold, vol_high_threshold, vol_low_threshold
                    )
                except Exception as e:
                    logger.error("[ChartViewerWidget] 시장 레짐 분류기 초기화 실패: %s", e)

            # 시장 레짐 분류 실행
            if self._regime_classifier is not None:
                # 옵션 데이터 추출 (predictor에서 opt_snap 가져오기)
                skew = None
                volume_pcr = None
                oi_pcr = None

                if self._predictor and hasattr(self._predictor, '_last_opt_snap'):
                    opt_snap = self._predictor._last_opt_snap
                    if opt_snap:
                        try:
                            # iv_skew 변환: put_iv/call_iv → call_iv - put_iv
                            iv_skew_raw = opt_snap.get("iv_skew")
                            if iv_skew_raw is not None:
                                iv_skew = float(iv_skew_raw)
                                skew = 1.0 - iv_skew  # 변환
                            else:
                                skew = 0.0

                            # PCR 데이터 추출
                            volume_pcr = float(opt_snap.get("pcr_volume") or 1.0)
                            oi_pcr = float(opt_snap.get("pcr_oi") or 1.0)

                            logger.debug("[ChartViewer] 옵션 데이터 추출: skew=%.4f, volume_pcr=%.2f, oi_pcr=%.2f",
                                        skew, volume_pcr, oi_pcr)
                        except Exception as e:
                            logger.warning("[ChartViewer] 옵션 데이터 추출 실패: %s", e)

                self._current_regime = self._regime_classifier.classify(
                    df,
                    skew=skew,
                    volume_pcr=volume_pcr,
                    oi_pcr=oi_pcr,
                )
                if self._current_regime:
                    logger.info("[ChartViewer] 시장 레짐: %s (신뢰도: %.2f)",
                              self._current_regime.regime.value, self._current_regime.confidence)

                    # 레짐 변경 감지 및 텔레그램 전송
                    current_regime_value = self._current_regime.regime.value
                    if self._last_logged_regime != current_regime_value:
                        logger.info("[ChartViewer] 레짐 변경 감지: %s → %s", self._last_logged_regime, current_regime_value)
                        previous_regime = self._last_logged_regime

                        # 레짐 변경 통계 추적
                        current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
                        current_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        symbol = self._selected_plot.upper() if self._selected_plot else "UNKNOWN"

                        # 일일 변경 횟수 증가
                        if current_date not in self._regime_change_stats:
                            self._regime_change_stats[current_date] = 0
                        self._regime_change_stats[current_date] += 1

                        # 변경 이력 추가
                        self._regime_change_history.append({
                            "time": current_time,
                            "previous": previous_regime,
                            "current": current_regime_value,
                            "symbol": symbol,
                            "confidence": self._current_regime.confidence
                        })

                        # 최근 100개만 유지
                        if len(self._regime_change_history) > 100:
                            self._regime_change_history = self._regime_change_history[-100:]

                        logger.info("[ChartViewer] 레짐 변경 통계: %s - %d번째 변경 (오늘 총 %d회)",
                                   current_date, self._regime_change_stats[current_date],
                                   self._regime_change_stats[current_date])

                        # 통계 저장
                        self._save_regime_stats()

                        self._last_logged_regime = current_regime_value

                        # 매매 신호 생성
                        signal = self._generate_regime_signal(previous_regime, current_regime_value, self._current_regime.confidence)

                        # 텔레그램 전송
                        if self._telegram_bridge_holder:
                            bridge = self._telegram_bridge_holder.get("bridge")
                            if bridge and hasattr(bridge, "notifier"):
                                try:
                                    regime_text = current_regime_value.replace("_", " ").title()
                                    confidence_pct = self._current_regime.confidence * 100

                                    # 신호 이모지 매핑
                                    signal_emojis = {
                                        "STRONG_BUY": "🚀",
                                        "BUY": "📈",
                                        "WEAK_BUY": "⬆️",
                                        "STRONG_SELL": "💥",
                                        "SELL": "📉",
                                        "WEAK_SELL": "⬇️",
                                        "HOLD": "⏸️"
                                    }
                                    signal_emoji = signal_emojis.get(signal, "⏸️")

                                    message = (
                                        f"{signal_emoji} 레짐 변경 알림\n"
                                        f"심볼: {symbol}\n"
                                        f"레짐: {regime_text}\n"
                                        f"신뢰도: {confidence_pct:.1f}%\n"
                                        f"시간: {current_time}\n"
                                        f"오늘 변경 횟수: {self._regime_change_stats[current_date]}회\n"
                                        f"매매 신호: {signal}"
                                    )
                                    bridge.notifier.send_text(message)
                                    logger.info("[ChartViewer] 레짐 변경 텔레그램 전송 완료: %s (신호: %s)", current_regime_value, signal)
                                except Exception as e:
                                    logger.warning("[ChartViewer] 레짐 변경 텔레그램 전송 실패: %s", e)

                    # LED 색상 설정 콜백 호출
                    if self._regime_led_callback:
                        try:
                            regime_value = self._current_regime.regime.value
                            # 레짐에 따른 LED 색상 매핑
                            regime_colors = {
                                "TREND_UP": "lawngreen",
                                "TREND_DOWN": "red",
                                "RANGING": "yellow",
                                "VOLATILE": "orange",
                                "NORMAL": "gray",
                            }
                            color = regime_colors.get(regime_value, "gray")
                            self._regime_led_callback(color)
                        except Exception as e:
                            logger.warning("[ChartViewer] 레짐 LED 색상 설정 실패: %s", e)

                    # 레짐 라벨 표시 (adaptive_mode와 무관하게 항상 표시)
                    if self._regime_label_callback:
                        try:
                            regime_text = self._current_regime.regime.value.replace("_", " ").title()
                            self._regime_label_callback(f"Regime: {regime_text} ({self._current_regime.confidence:.0%})", visible=True)
                        except Exception as e:
                            logger.warning("[ChartViewer] 레짐 라벨 텍스트 설정 실패: %s", e)
                else:
                    # 레짐 분류 실패 시 LED 회색
                    if self._regime_led_callback:
                        try:
                            self._regime_led_callback("gray")
                        except Exception as e:
                            logger.warning("[ChartViewer] 레짐 LED 색상 설정 실패: %s", e)
                    if self._regime_label_callback:
                        try:
                            self._regime_label_callback("Regime: -", visible=False)
                        except Exception as e:
                            logger.warning("[ChartViewer] 레짐 라벨 숨김 실패: %s", e)
            else:
                # 분류기 없을 때 LED 회색
                if self._regime_led_callback:
                    try:
                        self._regime_led_callback("gray")
                    except Exception as e:
                        logger.warning("[ChartViewer] 레짐 LED 색상 설정 실패: %s", e)
                if self._regime_label_callback:
                    try:
                        self._regime_label_callback("Regime: -", visible=False)
                    except Exception as e:
                        logger.warning("[ChartViewer] 레짐 라벨 숨김 실패: %s", e)
        except Exception as e:
            logger.warning("[ChartViewer] 시장 레짐 분류 실패: %s", e)
            # 분류 실패 시 LED 회색
            if self._regime_led_callback:
                try:
                    self._regime_led_callback("gray")
                except Exception as e:
                    logger.warning("[ChartViewer] 레짐 LED 색상 설정 실패: %s", e)
            if self._regime_label_callback:
                try:
                    self._regime_label_callback("Regime: -", visible=False)
                except Exception as e:
                    logger.warning("[ChartViewer] 레짐 라벨 숨김 실패: %s", e)

        # 렌더러 체크
        if self._renderer is None:
            logger.warning("[ChartViewer] 렌더러가 None - 렌더링 스킵")
            with self._render_lock:
                self._is_rendering = False  # [BUG-1] lock 안에서 해제
            return False
        
        # [FIX] Lock으로 이중 렌더링 방지: 체크-수정 원자적 처리
        with self._render_lock:
            if self._is_rendering:
                if force_clear:
                    logger.warning("[ChartViewer] 렌더링 중복 요청 (force_clear) - 이전 렌더링 취소 요청")
                    self._render_cancel_requested = True
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(200, lambda: self.refresh(force_clear=force_clear))
                    return False
                else:
                    logger.debug("[ChartViewer] 렌더링 중 - 증분 업데이트 요청 무시")
                    return False
            self._is_rendering = True
            self._render_cancel_requested = False

        logger.debug("[ChartViewer][refresh] renderer.render 호출")

        render_start = time.perf_counter()
        current_price = self._get_current_price()
        trade_events = self._load_trade_events() if self._trade_markers_enabled else None

        # 거래 이벤트 콜백 설정
        if self._renderer is not None:
            self._renderer.set_trade_event_callback(self._trade_event_log.add_event)
            self._renderer.set_cancel_check_callback(lambda: self._render_cancel_requested)

        # 렌더링 시간 모니터링 및 이벤트 루프 처리
        from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

        # 렌더링 전 processEvents() 제거 (중첩 렌더링 방지)

        was_redrawn = False
        try:
            was_redrawn = self._renderer.render(df, pm,
                                  data_source=self._selected_plot,
                                  trade_events=trade_events,
                                  current_price=current_price,
                                  force_clear=force_clear,
                                  show_pivots=self._show_pivots_enabled,
                                  pivot_prob_calc=self._pivot_prob_calc,
                                  minutes=self._minutes)
        except Exception as e:
            logger.exception("[ChartViewer] 렌더링 오류: %s", e)
            self._set_status("렌더링 오류 발생")
            with self._render_lock:
                self._is_rendering = False  # [BUG-1] lock 안에서 해제
                self._render_cancel_requested = False
            return False
        finally:
            with self._render_lock:
                self._is_rendering = False  # [BUG-1] lock 안에서 해제
                self._render_cancel_requested = False
        
        render_elapsed = (time.perf_counter() - render_start) * 1000
        self._last_render_time = render_elapsed

        # 렌더링 원인 분석 (깜박임 원인 추적)
        render_cause = []
        if force_clear:
            render_cause.append("FORCE_CLEAR")
        if was_redrawn:
            render_cause.append("REDRAWN")
        if not was_redrawn:
            render_cause.append("INCREMENTAL")

        cause_str = ",".join(render_cause) if render_cause else "UNKNOWN"

        # 렌더링 시간에 따른 로그 레벨 결정 (깜박임 방지)
        if render_elapsed >= 500:  # 500ms 이상: 깜박임 발생 가능
            logger.error(
                "[ChartViewer] 렌더링 시간이 너무 깁니다 (깜박임 발생 가능): "
                "%.1fms | cause=%s | data_source=%s | bars=%d | force_clear=%s | redrawn=%s",
                render_elapsed, cause_str, self._selected_plot, len(df), force_clear, was_redrawn
            )
        elif render_elapsed >= 200:  # 200-500ms: 경고
            logger.warning(
                "[ChartViewer] 렌더링 시간이 느립니다: "
                "%.1fms | cause=%s | data_source=%s | bars=%d | force_clear=%s | redrawn=%s",
                render_elapsed, cause_str, self._selected_plot, len(df), force_clear, was_redrawn
            )
        elif render_elapsed >= 100:  # 100-200ms: 정보
            logger.info(
                "[ChartViewer] 렌더링 시간: "
                "%.1fms | cause=%s | data_source=%s | bars=%d | force_clear=%s | redrawn=%s",
                render_elapsed, cause_str, self._selected_plot, len(df), force_clear, was_redrawn
            )
        else:  # 100ms 미만: 디버그
            logger.debug(
                "[ChartViewer][refresh] renderer.render 완료 %.1fms (cause=%s, redrawn=%s)",
                render_elapsed, cause_str, was_redrawn
            )

        # 렌더 스킵 시에도 현재가 라인 업데이트 (항상 표시)
        if not was_redrawn and self._renderer is not None:
            self._renderer._render_current_price_line(current_price)

        # 피봇 정보 패널용 crosshair 이벤트 연결 (렌더링 후 scene 생성 확인)
        self._connect_crosshair_event()

        # 기존 렌더링 시간 경고 (하위 호환성 유지)
        if render_elapsed > 1000:
            logger.warning("[ChartViewer] 렌더링 시간 %.1fms 초과 - UI 응답성 저하 가능성", render_elapsed)
            if render_elapsed > 3000:
                logger.error("[ChartViewer] 렌더링 시간 %.1fms 심각 - 데이터 클리어 권장", render_elapsed)
                # 3초 초과 시 캐시 클리어 권장 (사용자가 수행 필요)
                self._set_status(f"렌더링 지연 ({render_elapsed/1000:.1f}s) - 캐시 클리어 권장")

        return was_redrawn
    
    def cancel_render(self) -> None:
        """렌더링 취소 요청."""
        if self._is_rendering:
            self._render_cancel_requested = True
            logger.info("[ChartViewer] 렌더링 취소 요청")
        else:
            logger.debug("[ChartViewer] 렌더링 중 아님 - 취소 요청 무시")
    
    def get_render_status(self) -> dict:
        """렌더링 상태 반환.
        
        Returns:
            렌더링 상태 딕셔너리
        """
        return {
            "is_rendering": self._is_rendering,
            "cancel_requested": self._render_cancel_requested,
            "last_render_time_ms": self._last_render_time,
        }

    def _update_status(self, df: pd.DataFrame, pm: Dict) -> None:
        """상태 라벨 업데이트"""
        n_bars   = len(df)
        n_conf   = len((pm or {}).get("confirmed",   {}).get("idx", []))
        n_unconf = len((pm or {}).get("unconfirmed", {}).get("idx", []))
        elapsed  = (time.perf_counter() - self._t0) * 1000

        plot_name = "KOSPI" if self._selected_plot == "kospi" else "KP200 선물"
        
        # 확정 피봇 상세 정보 생성 (엔진에서 이미 필터링됨)
        pivot_details = ""
        if pm and n_conf > 0 and not df.empty:
            conf = pm.get("confirmed", {})
            idxs = conf.get("idx", [])
            ys = conf.get("y", [])
            types = conf.get("type", [])
            # 최대 5개 표시
            details = []
            for i in range(min(len(idxs), 5)):
                bar_idx = idxs[i]
                if 0 <= bar_idx < len(df):
                    ts = df.index[bar_idx]
                    price = ys[i] if i < len(ys) else None
                    ptype = types[i] if i < len(types) else None
                    time_str = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)
                    details.append(f"{ptype}@{time_str} {price:.2f}" if price else f"{ptype}@{time_str}")
            if details:
                pivot_details = " | " + " ".join(details)
        
        self._set_status(
            f"{plot_name}  |  {n_bars}봉  |  피봇 확정 {n_conf}개 / 미확정 {n_unconf}개"
            f"{pivot_details}  |  {elapsed:.0f}ms"
        )

    def _auto_refresh_callback(self) -> None:
        """자동 갱신 타이머 콜백 (장 마감 시 중지)"""
        logger.info("[ChartViewer] 자동 갱신 콜백 호출 (new_data_received=%s)", self._new_data_received)

        # 일일 리포트 체크
        self.check_and_send_daily_report()

        # 상태 변경 시 1회만 설정 (중복 setText 방지)
        _prev_market_closed = getattr(self, "_prev_market_closed", False)
        is_closed = not self._should_auto_refresh()

        if is_closed:
            if not _prev_market_closed:  # 최초 1회만
                self._set_status("장 마감 - 자동 갱신 중지")
                # 장 마감 시 데이터 플래그 리셋하여 재계산 방지
                self._new_data_received = False
                logger.info("[ChartViewer] 장 마감 - 데이터 플래그 리셋")
            self._prev_market_closed = True
            # 장 마감 시 자동 갱신 건너뜀
            return
        else:
            self._prev_market_closed = False
            
            # 새로운 데이터 수신 확인
            if not self._new_data_received:
                logger.debug("[ChartViewer][RT] 자동 갱신 스킵: 새 데이터 없음 (plot=%s)", self._selected_plot)
                self._set_status(f"대기 중... (데이터 없음)")
                return
            
            logger.info("[ChartViewer] 새로운 데이터 수신됨 - refresh 호출")
            # 피봇 품질 모니터 업데이트
            try:
                if hasattr(self, '_engine') and self._engine is not None \
                        and self._engine._zz is not None:
                    self._pivot_quality_monitor.update(
                        zz=self._engine._zz,
                        cfg=self._engine._zz_cfg,
                    )
            except Exception as _e:
                logger.debug("[ChartViewer] PivotQualityMonitor update 실패: %s", _e)
            self.refresh()

    def refresh(self, force_clear: bool = False) -> None:
        """데이터를 가져와 차트를 갱신한다.
        
        Args:
            force_clear: 차트 전체 초기화 여부 (수동 갱신 시 True)
        """
        logger.info("[ChartViewer] refresh 호출 (force_clear=%s)", force_clear)
        # 렌더러 초기화는 데이터 컴퓨팅 후 _render_chart에서 수행

        # [FIX] Lock으로 TOCTOU 방지: 체크-수정을 원자적으로 처리
        with self._render_lock:
            if self._is_rendering and force_clear:
                logger.info("[ChartViewer] 렌더링 중 - 취소 후 재시도")
                self._render_cancel_requested = True
                from PySide6.QtCore import QTimer
                QTimer.singleShot(200, lambda: self.refresh(force_clear))
                return
            if self._is_rendering and not force_clear:
                logger.warning("[ChartViewer][RT] 렌더링 중 자동 갱신 스킵 — 이전 렌더가 완료되지 않음 (plot=%s)", self._selected_plot)
                return

        # 로딩 상태 표시
        self._loading_lbl.setVisible(True)

        # 수동 갱신 시 차트 전체 초기화는 render 메서드에서 force_clear로 처리
        # 추적 변수 초기화는 render 메서드 내에서 수행
        # 갱신 버튼 클릭 시 차트 지워지는 모습을 보여주기 위해 지연 추가
        if force_clear and self._renderer is not None:
            # 중복 예약 방지
            if self._refresh_scheduled:
                logger.debug("[ChartViewer] refresh 이미 예약됨 - 건너뜀")
                return
            self._refresh_scheduled = True

            from PySide6.QtCore import QTimer
            # 갱신 버튼 클릭 시 캐시 삭제
            self._clear_cache()
            # clear_all() 호출 제거 - 깜빡임 방지
            # 대신 캔들만 제거 (render 내부에서 처리됨)

            # 지연 호출 중 타이머 일시 정지 (경쟁 상태 방지)
            if self._timer is not None:
                self._timer.stop()

            def _delayed():
                try:
                    self._refresh_scheduled = False
                    self._do_refresh_after_clear(force_clear)
                except Exception as e:
                    logger.exception("[ChartViewer] refresh 오류: %s", e)
                    self._set_status("차트 갱신 중 오류가 발생했습니다.")
                    # 예외 발생 시에도 타이머 재시작
                    if force_clear and self._timer is not None:
                        self._timer.start()
                    if self._loading_enabled:
                        self._loading_lbl.setVisible(False)

            # 0ms 지연 후 render 호출 (현재 이벤트 처리 완료 후 즉시 실행, 깜빡임 방지)
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, _delayed)
            return

        self._do_refresh_after_clear(force_clear)

    def _do_refresh_after_clear(self, force_clear: bool) -> None:
        """clear_all 후 실제 refresh 로직 수행."""
        logger.info("[ChartViewer] _do_refresh_after_clear 호출 (force_clear=%s)", force_clear)
        # 백그라운드 스레드에서 데이터 컴퓨팅 실행
        if not QT_AVAILABLE:
            # Qt 없으면 기존 방식 (동기)
            self._do_refresh_sync(force_clear)
            return

        # 이미 실행 중인 스레드가 있으면 취소 요청 후 지연 재시도
        if self._compute_thread is not None and self._compute_thread.isRunning():
            logger.debug("[ChartViewer] 이전 컴퓨팅 스레드 취소 요청 (논블로킹)")
            self._compute_thread.request_stop()
            # [BUG-5] 취소 후 지연 재시도로 갱신 누락 방지
            from PySide6.QtCore import QTimer
            QTimer.singleShot(100, lambda: self._do_refresh_after_clear(force_clear))
            return

        # 백그라운드 컴퓨팅 스레드 생성
        # ── [보완-3] 요청 토큰 및 설정값 캡처 (멀티스레드 데이터 경합 방지) ──
        self._current_request_token = time.time()
        self._last_request_plot = self._selected_plot
        self._last_request_minutes = self._minutes
        # ────────────────────────────────────────────────────────────────────────────
        self._compute_thread = DataComputeThread(self._compute_data, force_clear)
        self._compute_thread.finished.connect(self._on_compute_finished)
        self._compute_thread.error.connect(self._on_compute_error)
        self._compute_thread.start()
        logger.info("[ChartViewer] 백그라운드 컴퓨팅 스레드 시작 (force_clear=%s)", force_clear)
    
    def _do_refresh_sync(self, force_clear: bool) -> None:
        """동기 방식 refresh (Qt 없을 때 폴백)."""
        self._t0 = time.perf_counter()
        try:
            logger.debug("[ChartViewer][refresh] 시작 (동기)")

            # 데이터 컴퓨팅
            df, pm = self._compute_data()

            if df.empty:
                self._set_status("데이터를 가져올 수 없습니다. 장 전이거나 연결이 끊어졌습니다.")
                return

            # 차트 렌더링
            was_redrawn = self._render_chart(df, pm, force_clear)

            # 시간축 변경이 있는 경우 x축 뷰 리셋 + Y축 autorange
            needs_reset = self._renderer and getattr(self._renderer, '_xaxis_needs_reset', False)
            if needs_reset:
                n = 0 if self._minutes >= 9999 else self._minutes
                self._renderer._reset_xaxis_view(n_bars=n)

            # finplot 갱신 (실제 변경이 있을 때만)
            try:
                if self._fplt_ref is not None and was_redrawn:
                    logger.debug("[ChartViewer][refresh] fplt.refresh 호출 (redrawn=%s)", was_redrawn)
                    self._fplt_ref.refresh()
                    # 데이터 소스 변경 후 Y축 autorange 트리거
                    if needs_reset and self._renderer:
                        self._renderer._force_yaxis_range(df)
                        # refresh() 후에도 v_autozoom 비활성화 유지
                        if hasattr(self._renderer.ax_main, 'vb') and hasattr(self._renderer.ax_main.vb, 'v_autozoom'):
                            self._renderer.ax_main.vb.v_autozoom = False
                    logger.debug("[ChartViewer][refresh] fplt.refresh 완료")
                    # processEvents() 제거 - 중간 상태 노출 방지
            except Exception as e:
                logger.warning("[ChartViewer][refresh] fplt.refresh 예외: %s", e)
                pass

            # 상태 업데이트
            self._update_status(df, pm)

            if self._led_sync_enabled:
                self._sync_led_with_positions()

            if self._risk_monitoring_enabled:
                self._check_position_risk()

        except Exception as e:
            logger.exception("[ChartViewerWidget] refresh 오류: %s", e)
            # 사용자 친화적 에러 메시지
            error_msg = str(e)
            if "connection" in error_msg.lower() or "연결" in error_msg:
                self._set_status("데이터 소스 연결 실패했습니다.")
            elif "timeout" in error_msg.lower() or "시간 초과" in error_msg:
                self._set_status("데이터 가져오기 시간 초과했습니다.")
            else:
                self._set_status("차트 갱신 중 오류가 발생했습니다.")
        finally:
            # 완료 후 타이머 재시작 (force_clear 시에만 정지했으므로 force_clear일 때만 재시작)
            if force_clear and self._timer is not None:
                self._timer.start()
            # 로딩 상태 숨김
            if self._loading_enabled:
                self._loading_lbl.setVisible(False)
    
    @Slot(object, object, bool)
    def _on_compute_finished(self, df: pd.DataFrame, pm: Optional[Dict], force_clear: bool) -> None:
        """백그라운드 컴퓨팅 완료 핸들러."""
        logger.info("[ChartViewer] 백그라운드 컴퓨팅 완료 (force_clear=%s, df.shape=%s)", force_clear, df.shape if hasattr(df, 'shape') else 'N/A')

        # ── [보완-3] 설정값 검증 (멀티스레드 데이터 경합 방지) ──
        # 스레드 시작 시점의 설정값과 현재 설정값이 다르면 무시
        if hasattr(self, '_last_request_plot') and hasattr(self, '_last_request_minutes'):
            if self._last_request_plot != self._selected_plot or self._last_request_minutes != self._minutes:
                logger.debug("[ChartViewer] 설정값 변경으로 컴퓨팅 결과 무시 (plot: %s->%s, minutes: %d->%d)",
                           self._last_request_plot, self._selected_plot,
                           self._last_request_minutes, self._minutes)
                return
        # ────────────────────────────────────────────────────────────────────────────

        self._t0 = time.perf_counter()
        try:
            # df 타입 검증
            if df is None:
                logger.warning("[ChartViewer] 컴퓨팅 결과 df가 None")
                self._set_status("데이터를 가져올 수 없습니다.")
                return
            if df.empty:
                self._set_status("데이터를 가져올 수 없습니다. 장 전이거나 연결이 끊어졌습니다.")
                return

            # 차트 렌더링 (메인 스레드에서 실행)
            was_redrawn = self._render_chart(df, pm, force_clear)
            logger.debug("[ChartViewer] 렌더링 완료 (was_redrawn=%s)", was_redrawn)

            # 시간축 변경이 있는 경우 x축 뷰 리셋 + Y축 autorange
            needs_reset = self._renderer and getattr(self._renderer, '_xaxis_needs_reset', False)
            if needs_reset:
                n = 0 if self._minutes >= 9999 else self._minutes
                self._renderer._reset_xaxis_view(n_bars=n)

            # finplot 갱신 (실제 변경이 있을 때만)
            try:
                if self._fplt_ref is not None and was_redrawn:
                    logger.debug("[ChartViewer][refresh] fplt.refresh 호출 (redrawn=%s)", was_redrawn)
                    self._fplt_ref.refresh()
                    # 데이터 소스 변경 후 Y축 autorange 트리거
                    if needs_reset and self._renderer:
                        self._renderer._force_yaxis_range(df)
                        # refresh() 후에도 v_autozoom 비활성화 유지
                        if hasattr(self._renderer.ax_main, 'vb') and hasattr(self._renderer.ax_main.vb, 'v_autozoom'):
                            self._renderer.ax_main.vb.v_autozoom = False
                    logger.debug("[ChartViewer][refresh] fplt.refresh 완료")
                    # processEvents() 제거 - 중간 상태 노출 방지
            except Exception as e:
                logger.warning("[ChartViewer][refresh] fplt.refresh 예외: %s", e)
                pass

            # 상태 업데이트
            self._update_status(df, pm)

            if self._led_sync_enabled:
                self._sync_led_with_positions()

            if self._risk_monitoring_enabled:
                self._check_position_risk()

        except Exception as e:
            logger.exception("[ChartViewerWidget] _on_compute_finished 오류: %s", e)
            self._set_status("차트 갱신 중 오류가 발생했습니다.")
        finally:
            # 완료 후 타이머 재시작 (force_clear 시에만 정지했으므로 force_clear일 때만 재시작)
            if force_clear and self._timer is not None:
                self._timer.start()
            # 로딩 상태 숨김
            if self._loading_enabled:
                self._loading_lbl.setVisible(False)
            # 스레드 정리
            if self._compute_thread is not None:
                self._compute_thread.deleteLater()
                self._compute_thread = None
    
    @Slot(str, bool)
    def _on_compute_error(self, error_msg: str, force_clear: bool) -> None:
        """백그라운드 컴퓨팅 에러 핸들러."""
        logger.error(
            "[ChartViewer][RT] 백그라운드 컴퓨팅 오류 (plot=%s force_clear=%s): %s",
            self._selected_plot, force_clear, error_msg,
        )
        
        # 사용자 친화적 에러 메시지
        if "connection" in error_msg.lower() or "연결" in error_msg:
            self._set_status("데이터 소스 연결 실패했습니다.")
        elif "timeout" in error_msg.lower() or "시간 초과" in error_msg:
            self._set_status("데이터 가져오기 시간 초과했습니다.")
        else:
            self._set_status("차트 갱신 중 오류가 발생했습니다.")
        
        # 완료 후 타이머 재시작
        if force_clear and self._timer is not None:
            self._timer.start()
        # 로딩 상태 숨김
        if self._loading_enabled:
            self._loading_lbl.setVisible(False)
        # 스레드 정리
        if self._compute_thread is not None:
            self._compute_thread.deleteLater()
            self._compute_thread = None

    def _set_status(self, msg: str) -> None:
        try:
            if self._status_lbl is not None:
                self._status_lbl.setText(msg)
        except Exception:
            pass
    
    def _sync_led_with_positions(self) -> None:
        """활성 포지션 상태로 LED 자동 업데이트."""
        if (self._control_bar is None or self._control_bar.trade_led is None 
            or self._position_tracker is None):
            return

        try:
            # 활성 포지션 조회
            positions = self._position_tracker.get_active_positions()
            
            if not positions:
                # 활성 포지션 없음 - 대기 상태
                self._control_bar.trade_led.set_status("idle")
            else:
                # 첫 번째 포지션 기준 (단일 포지션 가정)
                pos = positions[0]
                if pos.action == "BUY":
                    self._control_bar.trade_led.set_status("long_hold")
                else:  # SELL
                    self._control_bar.trade_led.set_status("short_hold")
            
        except Exception as e:
            logger.debug("[ChartViewerWidget] LED 동기화 실패: %s", e)
    
    def _check_position_risk(self) -> None:
        """포지션 리스크 체크."""
        if not self._risk_monitoring_enabled:
            return
        if self._risk_monitor is None or self._position_tracker is None:
            return

        try:
            # 활성 포지션 조회
            positions = self._position_tracker.get_active_positions()
            
            if not positions:
                return
            
            # 첫 번째 포지션 리스크 체크
            pos = positions[0]

            # 현재 가격 가져오기
            current_price = self._get_current_price()

            # 가격이 0이면 계산 불가 — 조기 반환 (ZeroDivisionError 방지)
            if current_price == 0 or getattr(pos, 'entry_price', 0) == 0:
                return

            # 미실현 손익률 계산
            if pos.action == "BUY":
                unrealized_pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
            else:
                unrealized_pnl_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100
            
            # 손절까지 거리 계산
            distance_to_stop_pct = None
            if pos.stop_loss is not None and pos.stop_loss != 0 and current_price > 0:
                if pos.action == "BUY":
                    distance_to_stop_pct = ((current_price - pos.stop_loss) / current_price) * 100
                else:
                    distance_to_stop_pct = ((pos.stop_loss - current_price) / current_price) * 100
            
            # 리스크 체크
            self._risk_monitor.check_risk(
                position_id=pos.position_id,
                current_price=current_price,
                unrealized_pnl_pct=unrealized_pnl_pct,
                distance_to_stop_pct=distance_to_stop_pct
            )
            
        except Exception as e:
            logger.debug("[ChartViewerWidget] 리스크 체크 실패: %s", e)
    
    def _load_trade_events(self) -> List[Dict]:
        """거래 로그에서 이벤트 로드 (mtime 캐시 기반).

        Returns:
            거래 이벤트 리스트
        """
        # 5초 간격 제한
        now = time.monotonic()
        if now - self._trade_events_last_read < 5.0:
            return self._trade_events

        self._trade_events_last_read = now

        try:
            from prediction.trade_logger import get_trade_logger
            logger_instance = get_trade_logger()

            # 오늘 날짜의 로그 파일 로드
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = logger_instance.log_dir / f"trades_{today}.jsonl"

            if not log_file.exists():
                self._trade_events = []
                return self._trade_events

            # mtime 비교
            mtime = os.path.getmtime(str(log_file))
            if mtime == self._trade_events_mtime:
                return self._trade_events

            # 파일이 변경된 경우 재읽기
            read_start = time.perf_counter()
            events = []
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        events.append(event)
                    except Exception:
                        pass
            read_time = time.perf_counter() - read_start
            if read_time > 0.1:  # 100ms 이상 걸리면 경고
                logger.warning("[ChartViewer] 거래 이벤트 로드가 %.2f초 걸렸습니다", read_time)

            # 최근 100개 이벤트만 유지
            self._trade_events = events[-100:]
            self._trade_events_mtime = mtime

            return self._trade_events
        except Exception as e:
            logger.debug("[ChartViewerWidget] 거래 이벤트 로드 실패: %s", e)
            return self._trade_events
    
    def _refresh_trade_markers(self) -> None:
        """거래 마커 갱신."""
        if not self._trade_markers_enabled:
            return
        
        self._trade_events = self._load_trade_events()

    def stop(self) -> None:
        """타이머 정지. 앱 종료 직전에 호출한다."""
        try:
            if self._timer is not None:
                self._timer.stop()
        except Exception:
            pass
        # 깜빡임 타이머 정지
        if self._renderer is not None and hasattr(self._renderer, '_blink_timer'):
            try:
                self._renderer._blink_timer.stop()
            except Exception:
                pass
        # 시뮬레이션 타이머 정지
        self.stop_simulation()

        # 레짐 변경 통계 저장
        self._save_regime_stats()

    def simulate_pivot_event_logs(self) -> None:
        """피봇 이벤트 로그창에 샘플 데이터를 표시하는 시뮬레이션."""
        if self._pivot_event_log is None:
            logger.warning("[ChartViewerWidget] 피봇 이벤트 로그창이 초기화되지 않았습니다.")
            return

        logger.info("[ChartViewerWidget] 피봇 이벤트 로그 시뮬레이션 시작")

        # 샘플 피봇 이벤트 데이터
        sample_events = [
            ("registered", "KP200", "HIGH", 355.00, 142, "10:30"),
            ("changed", "KP200", "HIGH", 354.50, 143, "10:31"),
            ("confirmed", "KP200", "HIGH", 355.00, 144, "10:32"),
            ("registered", "KP200", "LOW", 350.25, 145, "10:35"),
            ("cancelled", "KP200", "LOW", 350.25, 146, "10:36", "반대후보교체"),
            ("registered", "KP200", "LOW", 348.50, 147, "10:40"),
            ("confirmed", "KP200", "LOW", 348.50, 148, "10:42"),
        ]

        for event_data in sample_events:
            if len(event_data) == 6:
                event_type, symbol, candidate_type, candidate_price, bar_idx, timestamp = event_data
                reason = ""
            else:
                event_type, symbol, candidate_type, candidate_price, bar_idx, timestamp, reason = event_data

            self._add_pivot_event_log(
                event_type=event_type,
                symbol=symbol,
                candidate_type=candidate_type,
                candidate_price=candidate_price,
                bar_idx=bar_idx,
                timestamp=timestamp,
                reason=reason
            )

        logger.info("[ChartViewerWidget] 피봇 이벤트 로그 시뮬레이션 완료")

    def simulate_realtime_plot(self, duration_sec: int = 30, interval_ms: int = 500, use_virtual_data: bool = True) -> None:
        """실시간 플롯 시뮬레이션.

        Args:
            duration_sec: 시뮬레이션 지속 시간 (초)
            interval_ms: 갱신 간격 (밀리초)
            use_virtual_data: 가상 데이터 사용 여부
        """
        if self._renderer is None:
            logger.warning("[ChartViewerWidget] 렌더러가 초기화되지 않았습니다.")
            return

        from PySide6.QtCore import QTimer
        import random

        logger.info("[ChartViewerWidget] 실시간 플롯 시뮬레이션 시작 (%d초, %dms 간격, 가상데이터=%s)",
                    duration_sec, interval_ms, use_virtual_data)

        self._sim_active = True
        self._sim_count = 0
        self._sim_max_count = (duration_sec * 1000) // interval_ms

        # 가상 데이터 생성기 초기화
        if use_virtual_data:
            self._sim_virtual_generator = VirtualTickGenerator(
                base_price=1000.0,
                volatility=0.001,
                tick_size=0.05
            )
            # 기존 데이터 가져오기
            try:
                df_base, _ = self._compute_data()
                if not df_base.empty:
                    last_close = df_base['Close'].iloc[-1]
                    self._sim_virtual_generator = VirtualTickGenerator(
                        base_price=last_close,
                        volatility=0.0005,
                        tick_size=0.05
                    )
                    logger.info("[ChartViewerWidget] 가상 데이터 생성기 초기화 (기준가=%.2f)", last_close)
            except Exception as e:
                logger.warning("[ChartViewerWidget] 기존 데이터 로드 실패: %s", e)

        def _update_sim():
            if not getattr(self, "_sim_active", False):
                return

            self._sim_count += 1
            if self._sim_count >= self._sim_max_count:
                self._sim_active = False
                self._sim_current_price = None  # 현재가 초기화
                logger.info("[ChartViewerWidget] 실시간 플롯 시뮬레이션 완료")
                # 버튼 상태 복원
                if hasattr(self, "_sim_toggle_btn"):
                    self._sim_toggle_btn.setChecked(False)
                    self._sim_toggle_btn.setText("↺ 갱신")
                    self._sim_toggle_btn.setToolTip("갱신 (장마감 후 시뮬레이션 모드)")
                return

            # 실시간 갱신 시뮬레이션
            try:
                if use_virtual_data and hasattr(self, "_sim_virtual_generator"):
                    # 가상 틱 데이터 생성 및 업데이트
                    self._update_with_virtual_ticks()
                else:
                    # 실제 데이터 refresh
                    self.refresh(force_clear=False)
            except Exception as e:
                logger.warning("[ChartViewerWidget] 시뮬레이션 갱신 실패: %s", e)

        # 타이머 생성 및 시작
        sim_timer = QTimer()
        sim_timer.timeout.connect(_update_sim)
        sim_timer.start(interval_ms)

        # 타이머 참조 저장 (정지용)
        self._sim_timer = sim_timer

        logger.info("[ChartViewerWidget] 시뮬레이션 타이머 시작")

    def stop_simulation(self) -> None:
        """시뮬레이션 정지."""
        self._sim_active = False
        self._sim_current_price = None  # 현재가 초기화
        if hasattr(self, "_sim_timer") and self._sim_timer is not None:
            try:
                self._sim_timer.stop()
                logger.info("[ChartViewerWidget] 시뮬레이션 정지")
            except Exception:
                pass
        # 버튼 상태 복원
        if hasattr(self, "_sim_toggle_btn"):
            self._sim_toggle_btn.setChecked(False)
            self._sim_toggle_btn.setText("↺ 갱신")
            self._sim_toggle_btn.setToolTip("갱신 (장마감 후 시뮬레이션 모드)")

    def _update_with_virtual_ticks(self) -> None:
        """가상 틱 데이터로 차트 업데이트."""
        if self._sim_virtual_generator is None:
            logger.warning("[ChartViewerWidget] 가상 틱 생성기가 초기화되지 않았습니다.")
            return

        try:
            # 기존 데이터 로드
            df_base, pm = self._compute_data()

            if df_base.empty:
                logger.warning("[ChartViewerWidget] 기존 데이터가 없습니다.")
                return

            # 가상 OHLC 생성
            ohlc = self._sim_virtual_generator.generate_ohlc(num_ticks=5)

            # 마지막 봉 업데이트 또는 새 봉 추가
            last_row = df_base.iloc[-1].copy()
            last_idx = df_base.index[-1]

            # 시간 인덱스 증가 (1분)
            import pandas as pd
            new_idx = last_idx + pd.Timedelta(minutes=1)

            # 새로운 봉 데이터 생성
            new_row = pd.DataFrame({
                'Open': [ohlc['Open']],
                'High': [ohlc['High']],
                'Low': [ohlc['Low']],
                'Close': [ohlc['Close']],
                'Volume': [ohlc['Volume']]
            }, index=[new_idx])

            # 데이터프레임에 추가 (전체 유지 — tail로 자르면 pm bar_idx와 불일치)
            df_updated = pd.concat([df_base, new_row], ignore_index=False)
            # [BUG-SIM-1] tail(100) 제거: pm의 bar_idx는 원본 df 전체 기준이므로
            # df_updated를 100봉으로 잘라 render에 전달하면
            # x_idx 길이가 100이 되어 100 이상 인덱스의 피봇이 전부 필터링됨.
            # df_updated를 그대로 전달하고, ChartEngine.compute()의 max_bars가
            # 렌더 윈도우를 제한하므로 별도 슬라이싱 불필요.

            # pm 재계산: df_updated가 바뀌었으므로 engine으로 재계산
            df_updated, pm = self._engine.compute(
                df_updated, self._config, self._selected_plot, force_recompute=False
            )

            # 캐시 업데이트
            cache_key = self._get_cache_key()
            self._cache_manager.set(cache_key, df_updated, pm)

            # 현재가 업데이트 (시뮬레이션용)
            self._sim_current_price = ohlc['Close']

            # 렌더링 (시뮬레이션에서는 force_clear=True로 피봇 마커 표시)
            was_redrawn = self._render_chart(df_updated, pm, force_clear=True)

            if was_redrawn and self._fplt_ref is not None:
                self._fplt_ref.refresh()

            logger.debug("[ChartViewerWidget] 가상 틱 업데이트 완료 (가격=%.2f)", ohlc['Close'])

        except Exception as e:
            logger.exception("[ChartViewerWidget] 가상 틱 업데이트 실패: %s", e)

    def _on_refresh_sim_toggle(self, checked: bool) -> None:
        """갱신/시뮬레이션 토글 버튼 핸들러."""
        if checked:
            # 장 마감 상태 확인
            if self._is_market_closed():
                # 시뮬레이션 시작 (가상 데이터 사용)
                self._control_bar.sim_toggle_btn.setText("⏹ 시뮬 정지")
                self._control_bar.sim_toggle_btn.setToolTip("시뮬레이션 정지")
                self.simulate_realtime_plot(duration_sec=30, interval_ms=500, use_virtual_data=True)
            else:
                # 장 중이면 일반 갱신
                self._control_bar.sim_toggle_btn.setChecked(False)  # 체크 해제
                self.refresh(force_clear=True)
        else:
            # 시뮬레이션 정지 또는 일반 갱신
            self.stop_simulation()
            self._control_bar.sim_toggle_btn.setText("↺ 갱신")
            self._control_bar.sim_toggle_btn.setToolTip("갱신 (장마감 후 시뮬레이션 모드)")

    def set_predictor(self, predictor: Any) -> None:
        """런타임에 predictor 를 교체한다 (파이프라인 재시작 시 등)."""
        self._predictor = predictor
        self._data_fetcher.set_predictor(predictor)
        self.refresh()


# ══════════════════════════════════════════════════════════════════════════════
# §3.5  가상 틱 데이터 생성기
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# §4  gui_controller.py 통합 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def attach_chart_viewer(
    right_root: Any,
    *,
    predictor:  Any = None,
    config:     Any = None,
    refresh_ms: int = 5000,  # 기본값 5초 (config.json과 일치)
    minutes:    int = 9999,  # 기본값: 장전체
    stretch:    int = 3,
    data_fetcher: Optional[callable] = None,  # 의존성 주입: 데이터 페처
    viewer_config: Optional[ChartViewerConfig] = None,  # 설정 객체
    regime_led_callback: Optional[callable] = None,  # 레짐 LED 색상 설정 콜백
    regime_label_callback: Optional[callable] = None,  # 레짐 라벨 텍스트 설정 콜백
    telegram_bridge_holder: Optional[Dict[str, Any]] = None,  # 텔레그램 bridge 홀더
) -> Optional[ChartViewerWidget]:
    """
    gui_controller.py 의 right_root(QVBoxLayout)에 차트 뷰어를 삽입한다.

    Parameters
    ----------
    right_root  : QVBoxLayout
    predictor   : KP200HybridPredictor 또는 PredictionPipeline
    config      : AppConfig (optional — ZigZag 파라미터 전달용)
    refresh_ms  : 자동 갱신 주기 ms (기본 500, config.json의 chart.refresh_ms 우선)
    minutes     : 가져올 분봉 수 (기본 9999: 장전체)
    stretch     : QVBoxLayout stretch 값 (기본 3)
    data_fetcher: 데이터 페처 함수 (의존성 주입용, 테스트 시 사용)
    viewer_config: ChartViewerConfig 설정 객체

    Returns
    -------
    ChartViewerWidget | None

    # ─────── gui_controller.py 적용 예 ───────────────────────────────────────

    # (1) import 추가
    from chart_viewer import attach_chart_viewer

    # (2) right_root.addWidget(log_view ...) 바로 위에 삽입
    self._chart_viewer = attach_chart_viewer(
        right_root,
        predictor=None,   # 파이프라인 시작 전이므로 None
        config=cfg,
    )

    # (3) 파이프라인 시작 후 predictor 연결
    if self._chart_viewer:
        self._chart_viewer.set_predictor(predictor)

    # (4) 매매 상태 LED 제어
    if self._chart_viewer:
        # 진입 시
        self._chart_viewer.set_trade_status("long_entry")  # 매수 진입
        self._chart_viewer.set_trade_status("short_entry") # 매도 진입
        
        # 보유 중
        self._chart_viewer.set_trade_status("long_hold")   # 매수 보유
        self._chart_viewer.set_trade_status("short_hold")  # 매도 보유
        
        # 청산 시
        self._chart_viewer.set_trade_status("exit")
        
        # 대기 상태
        self._chart_viewer.set_trade_status("idle")
        
        # LED 동기화 활성화/비활성화 (기본: 활성화)
        self._chart_viewer.set_led_sync_enabled(True)  # 자동 동기화 활성화
        self._chart_viewer.set_led_sync_enabled(False) # 자동 동기화 비활성화
        
        # 거래 마커 활성화/비활성화 (기본: 활성화)
        self._chart_viewer.set_trade_markers_enabled(True)  # 거래 마커 활성화
        self._chart_viewer.set_trade_markers_enabled(False) # 거래 마커 비활성화
        
        # 리스크 모니터링 활성화/비활성화 (기본: 활성화)
        self._chart_viewer.set_risk_monitoring_enabled(True)  # 리스크 모니터링 활성화
        self._chart_viewer.set_risk_monitoring_enabled(False) # 리스크 모니터링 비활성화

    # (5) 앱 종료 시
    if self._chart_viewer:
        self._chart_viewer.stop()
    """
    logger.info("[attach_chart_viewer] 함수 시작")
    try:
        # Qt 메시지 핸들러 설치
        global _qt_message_handler_func
        if _qt_message_handler_func is not None:
            try:
                from PySide6.QtCore import qInstallMessageHandler
                qInstallMessageHandler(_qt_message_handler_func)
                logger.info("[attach_chart_viewer] Qt 메시지 핸들러 설치 완료")
            except Exception as e:
                logger.debug("[attach_chart_viewer] Qt 메시지 핸들러 설치 실패: %s", e)

        # config에서 refresh_ms 읽기 (우선)
        if config is not None:
            try:
                chart_config = getattr(config, "chart", None) or {}
                if isinstance(chart_config, dict):
                    config_refresh_ms = chart_config.get("refresh_ms")
                    if config_refresh_ms is not None:
                        refresh_ms = int(config_refresh_ms)
            except Exception:
                pass

        # config에서 upcode 추출해서 ChartViewerWidget에 전달
        kp200_upcode: Optional[str] = None
        kospi_upcode: Optional[str] = None
        if config is not None:
            try:
                kp200_upcode = (
                    getattr(config, "kp200_upcode", None)
                    or (config.get("kp200_upcode") if isinstance(config, dict) else None)
                    or (config.get("ebest", {}).get("kp200_upcode") if isinstance(config, dict) else None)
                )
                kospi_upcode = (
                    getattr(config, "kospi_upcode", None)
                    or (config.get("kospi_upcode") if isinstance(config, dict) else None)
                    or (config.get("ebest", {}).get("kospi_upcode") if isinstance(config, dict) else None)
                )
            except Exception:
                pass

        logger.info("[attach_chart_viewer] ChartViewerWidget 생성 전")
        viewer = ChartViewerWidget(
            predictor=predictor,
            config=config,
            refresh_ms=refresh_ms,
            minutes=minutes,
            kp200_upcode=kp200_upcode or "",
            kospi_upcode=kospi_upcode or "",
            data_fetcher=data_fetcher,
            viewer_config=viewer_config,
            regime_led_callback=regime_led_callback,
            regime_label_callback=regime_label_callback,
            telegram_bridge_holder=telegram_bridge_holder,
        )
        if viewer.widget is not None:
            right_root.addWidget(viewer.widget, stretch=stretch)
            logger.info("[attach_chart_viewer] 차트 뷰어 삽입 완료")
            return viewer
        logger.warning("[attach_chart_viewer] 차트 위젯을 생성할 수 없습니다. finplot 패키지를 확인하세요.")
        return None
    except Exception as e:
        logger.error("[attach_chart_viewer] 차트 뷰어 초기화 중 오류 발생: %s", e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# §5  독립 실행 (오프라인 테스트)
# ══════════════════════════════════════════════════════════════════════════════

def _make_dummy_df(n: int = 150) -> pd.DataFrame:
    """KST naive DatetimeIndex 를 가진 테스트용 OHLCV 분봉 DataFrame."""
    rng   = np.random.default_rng(42)
    base  = 350.0
    close = base + np.cumsum(rng.normal(0, 0.8, n))
    open_ = np.roll(close, 1); open_[0] = close[0]
    high  = np.maximum(open_, close) + rng.uniform(0.1, 1.2, n)
    low   = np.minimum(open_, close) - rng.uniform(0.1, 1.2, n)
    vol   = rng.integers(300, 4000, n).astype(float)
    idx   = pd.date_range(
        "2026-04-25 09:00", periods=n, freq="1min", tz="Asia/Seoul"
    ).tz_localize(None)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


if __name__ == "__main__":
    """
    독립 실행 — finplot 독립 윈도우 팝업.
    indicators 없으면 피봇 없이 캔들 + 볼륨만 표시된다.
    """
    import finplot as fplt

    # 독립 실행 시에만 stderr 리다이렉션 및 예외 핸들러 적용
    sys.stderr = _StderrFilter()
    sys.excepthook = handler.handle

    df = _make_dummy_df(150)

    engine   = ChartEngine()
    df_r, pm = engine.compute(df)

    n_conf   = len((pm or {}).get("confirmed",   {}).get("idx", []))
    n_unconf = len((pm or {}).get("unconfirmed", {}).get("idx", []))
    print(f"[TEST] bars={len(df_r)}  confirmed={n_conf}  unconfirmed={n_unconf}")

    fplt.foreground = "#FFFFFF"
    fplt.background = "#0D0D0D"

    # rows=2: [0]=캔들, [1]=볼륨
    axs     = fplt.create_plot("KP200 선물 — chart_viewer 테스트", rows=2)
    ax_main = axs if not isinstance(axs, (list, tuple)) else axs[0]
    ax_vol  = (axs[1] if isinstance(axs, (list, tuple)) and len(axs) > 1 else None)

    # 가로세로 격자선 표시
    try:
        ax_main.showGrid(x=True, y=True, alpha=0.3)
    except Exception:
        pass

    renderer = FpltRenderer(ax_main, ax_vol)
    renderer.render(df_r, pm, current_price=float(df_r["Close"].iloc[-1]))

    fplt.show()
