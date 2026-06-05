"""finplot 렌더러 모듈

OHLC 캔들스틱 + 피봇 마커를 finplot ax 위에 그립니다.
"""

from __future__ import annotations

import logging
import time
import zlib
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    pass

import numpy as np
import pandas as pd
from gui.utils.pivot_probability import HistoricalPivot

logger = logging.getLogger(__name__)

# finplot PerformanceWarning 무시
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
# pyqtgraph All-NaN slice 경고 무시 (데이터 유효성 검사 완료 후에도 발생)
warnings.filterwarnings('ignore', message='.*All-NaN slice.*', category=RuntimeWarning)

# ══════════════════════════════════════════════════════════════════════════════

class FpltRenderer:
    """finplot ax 위에 OHLC 캔들스틱 + 피봇 마커를 그리는 렌더러."""

    _CONF_H_COLOR  = "#FF00FF"  # H 피봇: 마젠타
    _CONF_L_COLOR  = "#FF00FF"  # L 피봇: 마젠타
    _UNCONF_COLOR  = "#00FFFF"
    _MARKER_WIDTH  = 2.0
    _BULL_COLOR    = "#7CFC00"
    _BEAR_COLOR    = "#FF5252"
    _ENTRY_LONG_COLOR  = "#00FF00"
    _ENTRY_SHORT_COLOR = "#FF0000"
    _EXIT_LONG_COLOR   = "#00BFFF"
    _EXIT_SHORT_COLOR  = "#FFA500"
    _CUR_PRICE_COLOR   = "#FFFFFF"   # 현재가 수평선 색상
    _MA20_COLOR        = "#00BFFF"   # 20 EMA 색상
    _MA60_COLOR        = "#FF8C00"   # 60 EMA 색상

    def __init__(self, ax_main: Any, ax_vol: Optional[Any]) -> None:
        import finplot as fplt
        
        # finplot 전역 시간 포맷 설정
        fplt.timestamp_format   = '%H:%M'  # HH:MM 만 표시
        fplt.truncate_timestamp = False     # 자동 절단 비활성화 (포맷 그대로 사용)
        fplt.display_timezone   = None      # None → UTC 기준, KST naive 데이터와 일치
        fplt.show_crosshair     = False     # 기본 십자선 비활성화 (깜빡임 방지)

        # finplot _pdtime2index monkey patch - 데이터 부족 시 IndexError 방지
        if not hasattr(fplt, '_pdtime2index_patched'):
            original_pdtime2index = fplt._pdtime2index
            def safe_pdtime2index(*args, **kwargs):
                try:
                    return original_pdtime2index(*args, **kwargs)
                except IndexError as e:
                    # 데이터 부족 시 빈 리스트 반환
                    if "is out of bounds" in str(e):
                        return []
                    raise
            fplt._pdtime2index = safe_pdtime2index
            fplt._pdtime2index_patched = True
            logger.debug("[FpltRenderer] _pdtime2index monkey patch 적용")
        
        self._fplt       = fplt
        self.ax_main     = ax_main
        self.ax_vol      = ax_vol

        # v_autozoom 비활성화하여 Y축 자동 줌 방지 (0값 포함 문제 방지)
        if hasattr(ax_main, 'vb') and hasattr(ax_main.vb, 'v_autozoom'):
            ax_main.vb.v_autozoom = False
        self._plots: Dict[str, Any] = {}
        self._xaxis_done = False
        self._xaxis_needs_reset = False
        self._yaxis_needs_reset = False  # Y축 범위 재설정 플래그
        self._last_candle_time = None
        self._current_data_source = None  # 데이터 소스 추적 (futures/kospi)
        self._df_index = None  # 데이터 인덱스 저장 (십자선 시간 변환용)
        self._ma_enabled = False  # MA 표시 활성화 플래그
        self._last_minutes: int = -1  # 범위 변경 감지 (BUG-C 수정)
        self._pivot_info: Optional[pd.DataFrame] = None  # 피봇 정보 (crosshair용)
        self._pivot_idx_arr: np.ndarray = np.array([], dtype=np.int32)  # 피봇 인덱스 캐시 (성능 최적화)
        self._pivot_y_arr: np.ndarray = np.array([], dtype=np.float64)  # 피봇 가격 캐시 (성능 최적화)
        self._pivot_prob_calc: Optional[Any] = None  # 피봇 확정 확률 계산기
        self._trade_event_callback: Optional[callable] = None  # 거래 이벤트 콜백
        self._cancel_check_callback: Optional[callable] = None  # 취소 확인 콜백
        # ── 증분 업데이트 관련 변수 (렌더링 성능 최적화) ─────────────────────────
        self._last_df_len = 0  # 데이터 길이 추적: 새 봉 추가 감지용
        self._last_pm_hash = None  # 피봇 마커 해시: 피봇 변경 감지용 (MD5)
        self._last_close = None  # 마지막 봉 Close 값: 틱 갱신 감지용
        self._last_full_sig = None  # 전체 시그니처: 렌더 스킵용
        # ───────────────────────────────────────────────────────────────────────
        # ── 후보 마커 깜빡임 애니메이션 ─────────────────────────────────────────────
        self._unconf_marker_names = set()  # 후보 마커 이름 목록 (set으로 경쟁 상태 방지)
        self._blink_visible = True  # 깜빡임 상태
        from PySide6.QtCore import QTimer
        self._blink_timer = QTimer()
        self._blink_timer.timeout.connect(self._toggle_blink)
        self._blink_timer.start(1000)  # 1000ms마다 깜빡임 (refresh_ms=500의 정확한 2배수)
        # ───────────────────────────────────────────────────────────────────────

        # 십자선 기능 활성화
        try:
            # finplot 십자선 색상 설정
            fplt.cross_hair_color = "gray"

            # 십자선 아이템 저장소
            self._mouse_crosshair_items: dict = {}


            # 마우스 이동 이벤트 연결 상태 추적
            self._mouse_move_connected_scenes: set = set()
            self._mouse_crosshair_bound: bool = False

            # finplot crosshair 제거 플래그
            self._crosshair_cleaned: set = set()

            # 커스텀 십자선 생성
            self._ensure_mouse_crosshair(ax_main)

            # Scene의 sigMouseMoved 시그널 연결
            vb = getattr(ax_main, 'vb', None)
            if vb is not None:
                sc = vb.scene()
                if sc is not None and hasattr(sc, 'sigMouseMoved'):
                    sid = id(sc)
                    if sid not in self._mouse_move_connected_scenes:
                        sc.sigMouseMoved.connect(self._on_scene_mouse_moved)
                        self._mouse_move_connected_scenes.add(sid)

                self._mouse_crosshair_bound = True
        except Exception as e:
            logger.debug("[FpltRenderer] 십자선 추가 실패: %s", e)

    def _remove_finplot_crosshair(self, ax) -> None:
        """finplot 기본 십자선 제거.
        
        참고: __init__에서 fplt.show_crosshair = False를 설정했으나,
        finplot 버전에 따라 동작하지 않을 수 있어 수동 제거 로직 유지.
        """
        import pyqtgraph as pg
        vb = ax.getViewBox() if hasattr(ax, 'getViewBox') else getattr(ax, 'vb', None)
        if vb is None:
            return

        # finplot 십자선 색상 가져오기
        target_color = None
        try:
            target_color = pg.mkPen(getattr(self._fplt, "cross_hair_color", None)).color().name()
        except Exception:
            pass

        # ViewBox의 모든 아이템 가져오기
        items = list(getattr(vb, "addedItems", []) or [])

        for it in items:
            col = None
            # InfiniteLine, TextItem, LabelItem인 경우 색상 확인
            if isinstance(it, pg.InfiniteLine):
                pen = getattr(it, "pen", None)
                col = pen.color().name() if pen is not None else None
            elif isinstance(it, (pg.TextItem, pg.LabelItem)):
                c = getattr(it, "color", None)
                col = c.name() if hasattr(c, "name") else None

            # finplot 십자선 색상과 일치하면 제거
            if target_color is not None and col == target_color:
                try:
                    vb.removeItem(it)
                except Exception:
                    it.setParentItem(None)

    def _ensure_mouse_crosshair(self, ax) -> None:
        """커스텀 십자선 생성."""
        import pyqtgraph as pg
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont

        items = getattr(self, "_mouse_crosshair_items", {})
        if not isinstance(items, dict):
            items = {}
            self._mouse_crosshair_items = items

        k = id(ax)
        # 이미 생성된 경우 스킵
        if k in items:
            return

        vb = getattr(ax, 'vb', None)
        if vb is None:
            return

        # 수직선 (90도, 점선, 회색)
        vline = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen("#666", style=Qt.DashLine))

        # 수평선 (0도, 점선, 회색)
        hline = pg.InfiniteLine(pos=0.0, angle=0, movable=False, pen=pg.mkPen("#666", style=Qt.DashLine))

        # 텍스트 라벨 (노란색)
        text = pg.TextItem(color="yellow", anchor=(0, 1))
        try:
            text.setFont(QFont("Consolas", 10))
        except Exception:
            pass

        # Z-Value 설정 (최상단 표시)
        try:
            vline.setZValue(1e9)
            hline.setZValue(1e9)
            text.setZValue(1e9)
        except Exception:
            pass

        # ViewBox에 아이템 추가 (ignoreBounds=True로 축 범위에 영향 없음)
        vb.addItem(vline, ignoreBounds=True)
        vb.addItem(hline, ignoreBounds=True)
        vb.addItem(text, ignoreBounds=True)

        # 아이템 저장
        items[k] = (ax, vline, hline, text)

    def _on_scene_mouse_moved(self, pos) -> None:
        """마우스 이동 시 십자선 업데이트."""
        items = getattr(self, "_mouse_crosshair_items", None)
        if not isinstance(items, dict) or not items:
            return

        cleaned = getattr(self, "_crosshair_cleaned", set())

        for _k, tpl in list(items.items()):
            try:
                ax, vline, hline, text = tpl
            except Exception:
                continue

            try:
                # ax별로 최초 1회만 finplot crosshair 제거
                ax_id = id(ax)
                if ax_id not in cleaned:
                    self._remove_finplot_crosshair(ax)
                    cleaned.add(ax_id)

                vb = getattr(ax, 'vb', None)
                if vb is None:
                    continue

                # 마우스가 해당 ViewBox 위에 있는지 확인
                hovered = True
                try:
                    if hasattr(vb, "sceneBoundingRect"):
                        hovered = bool(vb.sceneBoundingRect().contains(pos))
                except Exception:
                    pass

                # 십자선 가시성 설정
                try:
                    vline.setVisible(bool(hovered))
                    hline.setVisible(bool(hovered))
                    text.setVisible(bool(hovered))
                except Exception:
                    pass

                if not hovered:
                    continue

                # Scene 좌표를 View 좌표로 변환
                pt = vb.mapSceneToView(pos)
                x = float(pt.x())
                y = float(pt.y())

                # 십자선 위치 업데이트
                vline.setPos(x)
                hline.setPos(y)

                # 텍스트 라벨 업데이트
                dt_label = self._resolve_dt_label(ax, vb, x)
                try:
                    text.setText(f"{dt_label}  {y:.2f}")
                    text.setPos(float(x), float(y))
                except Exception:
                    pass
            except Exception:
                continue

        # cleaned 상태 저장
        try:
            self._crosshair_cleaned = cleaned
        except Exception:
            pass

    def _resolve_dt_label(self, ax, vb, x: float) -> str:
        """x 좌표를 HH:MM 문자열로 변환.

        _df_index가 있으면 항상 그것을 사용하고,
        없으면 ax.datasrc를 사용합니다. 단위 추측 폴백은 제거하여
        무의미한 값 생성을 방지합니다.
        """
        dt_label = None

        # 저장된 데이터 인덱스 사용 (우선)
        if self._df_index is not None and len(self._df_index) > 0:
            try:
                idx = int(round(x))
                idx = max(0, min(idx, len(self._df_index) - 1))
                ts = self._df_index[idx]
                if ts is not None:
                    dt_label = ts.strftime("%H:%M")
            except Exception:
                pass

        # ax.datasrc 기반
        if dt_label is None:
            try:
                ds = getattr(ax, "datasrc", None)
                if ds is not None:
                    ts_idx = None
                    # 가장 가까운 인덱스 찾기
                    if hasattr(ds, "index") and hasattr(ds.index, "get_indexer"):
                        gi = ds.index.get_indexer([x], method="nearest")
                        ts_idx = int(gi[0])

                    if ts_idx is None:
                        idx = int(round(x))
                        ds_len = int(len(ds))
                        if ds_len > 0:
                            ts_idx = max(0, min(idx, ds_len - 1))

                    if ts_idx is not None:
                        ts = None
                        if hasattr(ds, "columns") and "Datetime" in ds.columns:
                            ts = ds["Datetime"].iloc[ts_idx]
                        else:
                            ts = ds.index[ts_idx]

                        if ts is not None:
                            dt_label = ts.strftime("%H:%M")
            except Exception:
                pass

        # 명시적 fallback (단위 추측 제거)
        if dt_label is None:
            dt_label = "??:??"

        return dt_label

    # ── x축: KST naive → datetime64[us]  (finplot 내부에서 ns int64로 변환) ──

    def _to_x(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """DatetimeIndex → datetime64[ns] (KST naive).

        ChartEngine.compute()는 항상 KST naive를 출력한다
        (tz_convert("Asia/Seoul").tz_localize(None)).
        naive 입력은 이미 KST이므로 그대로 변환하고,
        tz-aware 입력만 KST로 변환 후 naive로 만든다.
        
        [개선] 타임존 정보가 없는(naive) 데이터에 대해 명시적으로 처리 강화
        """
        try:
            # 이미 naive한 경우라면 localize 과정에서 에러가 날 수 있으므로 체크 강화
            if idx.tz is not None:
                return idx.tz_convert("Asia/Seoul").tz_localize(None).to_numpy(dtype="datetime64[ns]")
            return idx.to_numpy(dtype="datetime64[ns]")
        except Exception:
            return np.asarray(idx.values, dtype="datetime64[ns]")

    # ── x축 포맷 (최초 1회만) ───────────────────────────────────────────────

    def _reset_xaxis_view(self, n_bars: int = 0) -> None:
        """시간축 변경 시 x축 뷰를 리셋.

        Args:
            n_bars: 표시할 봉 수. 0이면 전체 데이터 표시.
        """
        try:
            if self.ax_main is not None:
                # fplt.xaxis_format 은 이 버전의 finplot에 없음 — timestamp_format으로 대체
                try:
                    # finplot ViewBox를 통해 x축 뷰 범위 조정
                    vb = self.ax_main.vb if hasattr(self.ax_main, "vb") else None
                    if vb is not None:
                        if n_bars > 0:
                            # 최근 n_bars 봉만 표시
                            x_max = vb.viewRange()[0][1]
                            x_min = x_max - n_bars
                            vb.setXRange(x_min, x_max, padding=0.02)
                        # [FIX-NEW-BAR] autoRange() 제거
                        # 호출 시 ST 등 보조지표 bounds까지 포함해 Y축을 날림
                        # 필요 시 호출부에서 명시적 setXRange를 사용
                except Exception as _e:
                    logger.debug("[ChartViewer] x축 범위 설정 실패: %s", _e)
            self._xaxis_needs_reset = False
        except Exception as e:
            logger.debug("[ChartViewer] x축 리셋 실패: %s", e)

    def _setup_xaxis_format(self) -> None:
        """timestamp_format/display_timezone은 FpltRenderer.__init__에서 설정됨."""
        pass

    # ── scene에서 완전 제거 ─────────────────────────────────────────────────

    def _scene_remove(self, obj: Any, name: str = "") -> None:
        """객체 제거.

        [ANTI-BLINK] 각 제거 방법이 pyqtgraph 내부 prepareGeometryChange() +
        update() 이벤트를 발생시킨다. 방법을 여러 번 시도할수록 paint 이벤트가
        누적되어 깜빡임이 심해진다. 가장 효과적인 방법 하나만 시도한다.
        """
        if obj is None:
            return

        # InfiniteLine: ViewBox.removeItem이 가장 안전
        if self._is_infinite_line(obj):
            vb = getattr(self.ax_main, 'vb', None)
            if vb is not None:
                try:
                    vb.removeItem(obj)
                    return
                except Exception:
                    pass
            return

        # PlotDataItem(캔들/선): finplot remove_primitive 또는 scene 제거
        rm = getattr(self._fplt, "remove_primitive", None)
        if callable(rm):
            try:
                rm(obj)
                return
            except Exception:
                pass

        # fallback: scene에서 직접 제거
        self._remove_from_scene(obj)

    def _is_infinite_line(self, obj: Any) -> bool:
        """객체가 InfiniteLine인지 확인"""
        try:
            import pyqtgraph as pg
            return isinstance(obj, pg.InfiniteLine)
        except Exception:
            return False

    def _remove_from_scene(self, obj: Any) -> bool:
        """scene에서 객체 제거 (최적화)"""
        # [FIX-RENDER-1] 가장 안전한 방법 우선: ViewBox.removeItem
        vb = getattr(self.ax_main, 'vb', None)
        if vb is not None:
            try:
                vb.removeItem(obj)
                return True
            except Exception:
                pass
        
        # fallback: scene에서 직접 제거
        try:
            scene = obj.scene()
            if scene is not None:
                scene.removeItem(obj)
                return True
        except Exception:
            pass
        
        return False

    def _remove(self, name: str) -> None:
        obj = self._plots.pop(name, None)
        if obj is not None:
            self._scene_remove(obj, name)
        # 후보 마커 목록에서 제거
        if name in self._unconf_marker_names:
            self._unconf_marker_names.remove(name)

    # ── upsert ──────────────────────────────────────────────────────────────

    def _upsert_st(self, name: str, x: np.ndarray, y: np.ndarray, ax: Any,
                   *, color: str, style: str, width: float = 1.0) -> None:
        """SuperTrend 전용 upsert.

        [FIX-ROOT] 일반 _upsert는 NaN을 제거해 배열 길이가 달라지고
        finplot update_data()가 길이 불일치 예외를 던져 _remove→재생성 루프를 유발.
        이 메서드는 NaN 제거 없이 full-length 배열을 그대로 전달한다.
        finplot은 NaN 구간을 선 단절로 처리하므로 시각적으로 정상이다.
        """
        if x.size == 0 or y.size == 0:
            obj = self._plots.get(name)
            if obj is not None:
                try:
                    obj.setVisible(False)
                except Exception:
                    self._remove(name)
            return

        # x: datetime64[ns] 그대로 유지, y: float64 강제
        # [BUG] 이전 코드에서 x를 float64로 변환해 datetime 축 좌표가 깨졌음
        x = np.asarray(x).ravel()   # dtype 유지 (datetime64[ns])
        y = np.asarray(y, dtype=np.float64).ravel()

        existing = self._plots.get(name)
        if existing is not None:
            try:
                existing.update_data((x, y))
                try:
                    existing.setVisible(True)
                except Exception:
                    pass
                return
            except Exception as e:
                logger.debug("[FpltRenderer] _upsert_st update_data 실패(%s): %s — 재생성", name, e)
                self._remove(name)

        # 데이터 타입 명시적 변환 (np.where 결과가 object 타입일 수 있음)
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

        try:
            obj = self._fplt.plot(x, y, ax=ax,
                                  color=color, style=style, width=width)
            self._plots[name] = obj

            # [FIX-YAXIS] ST 아이템을 ViewBox autorange 계산에서 제외
            # NaN 포함 full 배열이 bounds에 영향을 주면 Y축이 점점 확장됨
            try:
                vb = getattr(ax, "vb", None)
                if vb is not None and hasattr(vb, "_itemBoundsCache"):
                    vb._itemBoundsCache.pop(obj, None)
                if hasattr(obj, "opts"):
                    obj.opts["ignoreBounds"] = True
            except Exception:
                pass
        except Exception as e:
            logger.warning("[FpltRenderer] _upsert_st 생성 실패: name=%s, error=%s", name, e)

    def _upsert(self, name: str, x: Any, y: Any, ax: Any,
                *, color: str, style: str, width: float = 1.0) -> None:
        try:
            xa = np.asarray(x).ravel()
            ya = np.asarray(y, dtype=np.float64).ravel()
        except Exception:
            xa = ya = np.array([])

        # 데이터 타입 변환 강화 (MA 업데이트 실패 해결)
        if xa.size > 0:
            # datetime 타입 처리 (finplot은 datetime64[ns]를 직접 지원)
            if xa.dtype.kind == 'O':
                try:
                    # datetime64로 변환 (초 단위 변환 제거 - finplot이 datetime64[ns] 처리)
                    xa = pd.to_datetime(xa)
                except Exception:
                    try:
                        xa = pd.to_numeric(xa, errors='coerce').astype(np.float64)
                    except Exception:
                        xa = np.array([])
            elif xa.dtype.kind == 'M':  # datetime64 - 변환 없이 그대로 사용
                pass  # finplot이 datetime64[ns]를 직접 처리하므로 변환 불필요
            elif xa.dtype.kind not in ('i', 'u', 'f'):
                try:
                    xa = pd.to_numeric(xa, errors='coerce').astype(np.float64)
                except Exception:
                    xa = np.array([])
        
        if ya.size > 0:
            if ya.dtype.kind == 'O':
                try:
                    ya = pd.to_numeric(ya, errors='coerce').astype(np.float64)
                except Exception:
                    ya = np.array([])
            elif ya.dtype.kind not in ('i', 'u', 'f'):
                try:
                    ya = ya.astype(np.float64)
                except Exception:
                    ya = np.array([])
        
        # NaN 제거 (datetime64 타입 처리)
        if xa.size > 0 and ya.size > 0:
            if xa.dtype.kind == 'M':  # datetime64
                nan_mask_x = np.zeros(len(xa), dtype=bool)
            else:
                nan_mask_x = np.isnan(xa) | np.isinf(xa)
            nan_mask_y = np.isnan(ya) | np.isinf(ya)
            mask = ~(nan_mask_x | nan_mask_y)
            xa = xa[mask]
            ya = ya[mask]

        # [ANTI-FLICKER] 빈 배열일 때 플롯 숨기기
        if xa.size == 0 or ya.size == 0:
            existing = self._plots.get(name)
            if existing is not None:
                try:
                    existing.setVisible(False)
                except Exception:
                    self._remove(name)
            return

        # [ANTI-NAN-WARNING] 필터링 후에도 유효 데이터가 있는지 확인
        if xa.dtype.kind != 'M':  # datetime64가 아닌 경우
            if np.all(np.isnan(xa)) or np.all(np.isnan(ya)):
                existing = self._plots.get(name)
                if existing is not None:
                    try:
                        existing.setVisible(False)
                    except Exception:
                        self._remove(name)
                return

        if len(xa) != len(ya):
            m = min(len(xa), len(ya))
            xa, ya = xa[:m], ya[:m]

        existing = self._plots.get(name)
        if existing is not None:
            try:
                existing.update_data((xa, ya))
                # 투명화 복원 (이전에 setVisible(False)로 숨겨진 경우)
                try:
                    existing.setVisible(True)
                except Exception:
                    pass
                return
            except Exception:
                self._remove(name)

        try:
            obj = self._fplt.plot(xa, ya, ax=ax,
                                  color=color, style=style, width=width)
            self._plots[name] = obj
        except Exception as e:
            logger.warning("[FpltRenderer] 플롯 생성 실패: name=%s, error=%s", name, str(e))

    def _toggle_blink(self) -> None:
        """후보 마커 깜빡임 토글 (setVisible 사용으로 렌더 트리거 최소화)"""
        self._blink_visible = not self._blink_visible
        for name in list(self._unconf_marker_names):
            plot_obj = self._plots.get(name)
            if plot_obj is not None:
                try:
                    plot_obj.setVisible(self._blink_visible)
                except RuntimeError:
                    # C++ 객체 이미 삭제됨
                    self._unconf_marker_names.discard(name)
                    pass

    # ── 캔들스틱 (candlestick_ochl 단일 item) ───────────────────────────────

    def _make_candle_df(self, x_idx: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
        """candlestick_ochl용 DataFrame. OHLC는 반드시 float64.

        finplot은 DataFrame의 첫 번째 컬럼 또는 DatetimeIndex를 x축으로 사용한다.
        DatetimeIndex 방식이 x축 눈금 표시에 가장 안정적이다.
        """
        # [FIX3] x_idx는 _to_x()에서 이미 datetime64[ns]로 변환됨.
        # pd.to_datetime() → .astype() 이중 변환 시 ns 단위 미세 오차로
        # finplot datasrc와 1ns 불일치 → 마지막 캔들 위치 어긋남 방지.
        time_idx = pd.DatetimeIndex(x_idx)  # datetime64[ns] 그대로 사용
        cdf = pd.DataFrame(
            {
                "open":  df["Open"].astype(np.float64).values,
                "close": df["Close"].astype(np.float64).values,
                "high":  df["High"].astype(np.float64).values,
                "low":   df["Low"].astype(np.float64).values,
            },
            index=time_idx,  # ← DatetimeIndex를 인덱스로 설정
        )
        return cdf

    def _render_candles(self, x_idx: np.ndarray, df: pd.DataFrame) -> None:
        """캔들스틱 렌더링.

        깜빡임 방지 원칙:
        - 기존 아이템이 있으면 update_data(cdf) 로 데이터만 교체 (clean 인자 사용 안 함).
          clean=True 는 내부적으로 아이템을 지우고 재생성하므로
          동일 분봉 틱 갱신에서도 공백(지우기→그리기) 이 발생해 깜빡임을 유발한다.
        - update_data 가 실패(구조 변경 등)할 때만 삭제 후 재생성한다.
        """
        try:
            cdf = self._make_candle_df(x_idx, df)
            logger.info(
                "[FpltRenderer] 캔들 렌더링: x_idx=%d, cdf=%d, existing=%s",
                len(x_idx), len(cdf), self._plots.get("_candle") is not None
            )
            # 전역 컬러 재설정 (삭제 방지)
            self._fplt.candle_bull_color      = self._BULL_COLOR
            self._fplt.candle_bull_body_color = self._BULL_COLOR
            self._fplt.candle_bear_color      = self._BEAR_COLOR

            existing = self._plots.get("_candle")

            if existing is not None:
                try:
                    # ★ clean 인자 없이 호출 → 기존 아이템 유지, 데이터만 교체 → 깜빡임 없음
                    existing.update_data(cdf)
                    if len(cdf) > 0:
                        self._last_candle_time = cdf.index[-1]

                    # [FIX-YAXIS] 캔들 업데이트 후 Y축 범위를 캔들 High/Low 기준으로 강제 고정
                    # finplot 내부 autorange가 Y축을 계속 확대하는 것을 방지
                    try:
                        vb = getattr(self.ax_main, 'vb', None)
                        if vb is not None and hasattr(vb, 'viewRange'):
                            # 캔들 데이터의 실제 High/Low 계산 (컬럼 이름 대소문자 호환)
                            if "High" in cdf.columns:
                                candle_high = cdf["High"].max()
                                candle_low = cdf["Low"].min()
                            elif "high" in cdf.columns:
                                candle_high = cdf["high"].max()
                                candle_low = cdf["low"].min()
                            else:
                                return

                            padding = (candle_high - candle_low) * 0.05
                            new_y_min = candle_low - padding
                            new_y_max = candle_high + padding

                            # v_autozoom 비활성화
                            if hasattr(vb, 'v_autozoom'):
                                vb.v_autozoom = False

                            vb.setYRange(new_y_min, new_y_max, padding=0)

                            # update() 후에도 v_autozoom 비활성화 유지
                            if hasattr(vb, 'update'):
                                vb.update()
                            if hasattr(vb, 'v_autozoom'):
                                vb.v_autozoom = False
                    except Exception:
                        pass

                    return
                except (TypeError, ValueError):
                    # 컬럼 구조가 바뀐 경우만 재생성
                    self._remove("_candle")
                except RuntimeError:
                    # C++ 객체 이미 삭제된 경우 재생성
                    self._remove("_candle")
                except Exception:
                    self._remove("_candle")

            if len(cdf) > 0:
                self._last_candle_time = cdf.index[-1]

            self._plots["_candle"] = self._fplt.candlestick_ochl(cdf, ax=self.ax_main)
            self._xaxis_needs_reset = True

        except Exception as e:
            logger.error("[FpltRenderer][RT] 캔들 렌더링 실패: %s", e, exc_info=True)

    # ── SuperTrend 선 ───────────────────────────────────────────────────────────

    def _render_supertrend(self, x_idx: np.ndarray, df: pd.DataFrame) -> None:
        """SuperTrend 선 렌더링.

        [FIX-ROOT] 기존 mask 서브셋 방식은 매 호출마다 up/down 요소 수가 달라져
        finplot update_data()가 길이 불일치 예외를 던지고 _remove → 재생성 루프를
        유발했다. full x_idx를 유지하고 해당 안 되는 봉은 NaN으로 채우는 방식으로
        변경하면 배열 길이가 항상 일정해 update_data()가 안정적으로 작동한다.
        """
        if "SuperTrend" not in df.columns or "SuperTrend_Dir" not in df.columns:
            logger.debug("[FpltRenderer] SuperTrend 컬럼 없음 - 렌더링 스킵")
            return

        try:
            st     = df["SuperTrend"].values.astype(np.float64)
            st_dir = df["SuperTrend_Dir"].values

            if len(x_idx) != len(st):
                min_len = min(len(x_idx), len(st))
                x_idx  = x_idx[:min_len]
                st     = st[:min_len]
                st_dir = st_dir[:min_len]

            valid_mask = ~np.isnan(st)
            if not np.any(valid_mask):
                logger.debug("[FpltRenderer] SuperTrend 유효 데이터 없음")
                return

            # numeric direction (1=up, -1=down)
            is_up   = (st_dir == 1)
            is_down = (st_dir == -1)

            # 방향 전환 봉을 양쪽에 포함 (선 연속성 유지)
            n = len(st_dir)
            include_in_up   = is_up.copy()
            include_in_down = is_down.copy()
            for i in range(1, n):
                if is_up[i - 1] and is_down[i]:
                    include_in_up[i] = True
                elif is_down[i - 1] and is_up[i]:
                    include_in_down[i] = True

            # [FIX-DATATYPE] x_idx를 float64로 변환 (pyqtgraph 호환성)
            x_idx = np.asarray(x_idx, dtype=np.float64).ravel()

            # [FIX-ROOT] full 배열 유지: 해당 구간은 실제값, 나머지는 NaN
            # → _upsert에 넘기는 배열 길이가 항상 len(x_idx)로 고정
            # → finplot update_data()가 길이 불일치 예외를 내지 않음
            up_vals   = np.where(valid_mask & include_in_up,   st, np.nan)
            down_vals = np.where(valid_mask & include_in_down, st, np.nan)

            up_has_data   = np.any(~np.isnan(up_vals))
            down_has_data = np.any(~np.isnan(down_vals))

            if up_has_data:
                self._upsert_st("_supertrend_up", x_idx, up_vals,
                             self.ax_main, color="#00FF00", style="--", width=2.0)
            else:
                obj = self._plots.get("_supertrend_up")
                if obj is not None:
                    try:
                        obj.setVisible(False)
                    except Exception:
                        pass

            if down_has_data:
                self._upsert_st("_supertrend_down", x_idx, down_vals,
                             self.ax_main, color="#FF0000", style="--", width=2.0)
            else:
                obj = self._plots.get("_supertrend_down")
                if obj is not None:
                    try:
                        obj.setVisible(False)
                    except Exception:
                        pass

            logger.debug(
                "[FpltRenderer] SuperTrend 렌더링: up_pts=%d, down_pts=%d",
                int(np.sum(~np.isnan(up_vals))), int(np.sum(~np.isnan(down_vals))),
            )

        except Exception as e:
            logger.warning("[FpltRenderer] SuperTrend 렌더링 실패: %s", e)

    # ── 종가선 (비활성화) ────────────────────────────────────────────────────

    def _render_price_lines(self, x_idx: np.ndarray, df: pd.DataFrame) -> None:
        # 종가선 비활성화 — 키가 존재할 때만 제거
        for nm in ("_close_line", "_open_line"):
            if nm in self._plots:
                self._remove(nm)

    # ── OBV (비활성화) ───────────────────────────────────────────────────────

    def _render_volume(self, x_idx: np.ndarray, df: pd.DataFrame) -> None:
        # OBV 비활성화 — 키가 존재할 때만 제거
        for nm in ("_obv_line", "_obv_zero"):
            if nm in self._plots:
                self._remove(nm)

    # ── 피봇 마커 ────────────────────────────────────────────────────────────

    @staticmethod
    def _pm_hash(pm: Optional[Dict]) -> Optional[int]:
        """pivot_markers 딕셔너리의 빠른 해시 (변경 감지용).

        경량 해시: 피봇 수 + 마지막 피봇 좌표만 비교 (hashlib.sha256 사용)
        [REVIEW-FIX-5] swing_version 추가: 클러스터링 in-place 갱신 시 캐시 무효화

        Returns:
            해시 값 또는 None (해시 불가 시 재렌더 강제)
        """
        if pm is None:
            return None
        try:
            import zlib
            conf = pm.get("confirmed", {})
            unconf = pm.get("unconfirmed", {})
            # 피봇 수 + 마지막 idx + 마지막 y 조합 + swing_version + minutes
            swing_version = pm.get("swing_version", 0)
            key = (
                len(conf.get("idx", [])),
                len(unconf.get("idx", [])),
                conf.get("idx", [-1])[-1] if conf.get("idx") else -1,
                round(conf.get("y", [0.0])[-1], 2) if conf.get("y") else 0.0,
                conf.get("confirmed_at_idx", [-1])[-1] if conf.get("confirmed_at_idx") else -1,
                swing_version,
                pm.get("minutes", 0),  # 범위 변경 시 강제 재렌더
            )
            # swing_version 변경 로그
            if swing_version > 0:
                logger.info("[FpltRenderer] pm_hash 계산: swing_version=%d", swing_version)
            # zlib.adler32 사용 (빠르고 충분히 안전)
            return zlib.adler32(str(key).encode())
        except Exception:
            return None

    def _render_pivots(self, x_idx: np.ndarray, pm: Dict, pm_hash: Optional[int] = None) -> None:
        # 피벗 해시 비교 제거 - render()의 pivot_changed 조건으로 일원화
        # pm_hash가 전달되지 않으면 내부 _pm_hash로 계산 (하위 호환)
        if pm_hash is None:
            pm_hash = self._pm_hash(pm)
        # 내부 해시 비교 제거 - render()에서 호출 여부 결정
        # _last_pm_hash 갱신은 render() finally에서 수행

        # 피벗 데이터 상태 로그 (DEBUG로 변경)
        logger.debug("[ChartViewer] _render_pivots 호출: x_idx 길이=%d, pm=%s", len(x_idx), bool(pm))
        if pm and isinstance(pm, dict):
            conf = pm.get("confirmed", {})
            unconf = pm.get("unconfirmed", {})
            conf_idx = conf.get("idx", [])
            unconf_idx = unconf.get("idx", [])
            logger.debug("[ChartViewer] 피벗 데이터: confirmed 개수=%d, unconfirmed 개수=%d", len(conf_idx), len(unconf_idx))

        # 스냅샷 캡처 (클로저 경쟁 상태 방지)
        prob_calc = self._pivot_prob_calc

        # 후보 마커 목록 초기화 — set으로 경쟁 상태 방지
        new_unconf_marker_names = set()

        # 피봇 정보 저장을 위한 리스트 (confirmed + unconfirmed)
        pivot_info_list = []

        def _bucket(bucket: str) -> None:
            b    = pm.get(bucket) or {}
            idxs = b.get("idx",  [])
            ys   = b.get("y",    [])
            tys  = b.get("type", [])
            confirmed_at_idxs = b.get("confirmed_at_idx", [])
            registered_at_idxs = b.get("registered_at_idx", [])
            n    = min(len(idxs), len(ys), len(tys) if tys else len(idxs))
            if n == 0:
                return

            # 피벗 인덱스 범위 검증 (위치 틀어짐 방지)
            valid_idx_mask = np.array([idx for idx in idxs[:n] if 0 <= idx < len(x_idx)])
            if len(valid_idx_mask) < n:
                logger.warning(
                    "[FpltRenderer] 피벗 인덱스 범위 초과: total=%d, valid=%d, x_idx_len=%d",
                    n, len(valid_idx_mask), len(x_idx)
                )
                # 유효한 인덱스만 사용
                valid_indices = [idx for idx in idxs[:n] if 0 <= idx < len(x_idx)]
                if not valid_indices:
                    return  # 유효한 인덱스가 없으면 렌더링 스킵
                # 데이터 재구성
                idxs = valid_indices
                ys = [ys[i] for i in range(n) if idxs[i] in valid_indices]
                tys = [tys[i] for i in range(n) if idxs[i] in valid_indices]
                if confirmed_at_idxs:
                    confirmed_at_idxs = [confirmed_at_idxs[i] for i in range(n) if idxs[i] in valid_indices]
                if registered_at_idxs:
                    registered_at_idxs = [registered_at_idxs[i] for i in range(n) if idxs[i] in valid_indices]
                n = len(idxs)

            # confirmed_at_idx, registered_at_idx가 있는 경우 길이 조정
            if confirmed_at_idxs:
                n = min(n, len(confirmed_at_idxs))
            if registered_at_idxs:
                n = min(n, len(registered_at_idxs))

            dfp = pd.DataFrame({
                "idx": pd.to_numeric(pd.Series(idxs[:n]), errors="coerce"),
                "y":   pd.to_numeric(pd.Series(ys[:n]),   errors="coerce"),
                "t":   pd.Series(tys[:n] if tys else [""]*n).astype(str).str.upper(),
            })

            # confirmed_at_idx 추가
            if confirmed_at_idxs:
                dfp["confirmed_at_idx"] = pd.to_numeric(pd.Series(confirmed_at_idxs[:n]), errors="coerce")

            # registered_at_idx 추가
            if registered_at_idxs:
                dfp["registered_at_idx"] = pd.to_numeric(pd.Series(registered_at_idxs[:n]), errors="coerce")
            
            dfp = dfp.dropna(subset=["idx", "y"])
            dfp = dfp.drop_duplicates(subset=["idx"], keep="last")
            n_before = len(dfp)

            # ── [FIX] 인덱스 범위 검증 강화 ──
            max_idx = len(x_idx) - 1
            dfp = dfp[dfp["idx"].astype(int).between(0, max_idx)].copy()
            if dfp.empty:
                if n_before > 0:
                    logger.warning(
                        "[FpltRenderer] 피봇 %d개가 x_idx 범위(%d) 벗어나 필터링됨 — minutes 변경 후 pm 재계산 필요",
                        n_before, len(x_idx)
                    )
                return

            # 인덱스 매핑 전 추가 검증
            bar = dfp["idx"].astype(int).to_numpy()
            # 인덱스가 음수이거나 범위를 벗어나는 경우 필터링
            valid_mask = (bar >= 0) & (bar <= max_idx)
            if not valid_mask.all():
                invalid_count = (~valid_mask).sum()
                logger.warning(
                    "[FpltRenderer] 피봇 인덱스 %d개가 범위 벗어남 - 필터링 (범위: 0~%d)",
                    invalid_count, max_idx
                )
                bar = bar[valid_mask]
                dfp = dfp.iloc[valid_mask].copy()
                if dfp.empty:
                    return

            xs  = x_idx[bar]
            ys2 = dfp["y"].to_numpy(dtype=np.float64)
            ts  = dfp["t"].to_numpy(dtype=str)

            # NaN 필터링
            valid_mask = ~np.isnan(ys2)
            xs = xs[valid_mask]
            ys2 = ys2[valid_mask]
            ts = ts[valid_mask]

            if bucket == "confirmed":
                mh = ts == "H"
                h_count = mh.sum() if mh.any() else 0
                self._upsert("_zz_conf_H",
                             xs[mh] if mh.any() else np.array([]),
                             ys2[mh] if mh.any() else np.array([]),
                             self.ax_main, color=self._CONF_H_COLOR,
                             style="v", width=self._MARKER_WIDTH)
                ml = ts == "L"
                l_count = ml.sum() if ml.any() else 0
                self._upsert("_zz_conf_L",
                             xs[ml] if ml.any() else np.array([]),
                             ys2[ml] if ml.any() else np.array([]),
                             self.ax_main, color=self._CONF_L_COLOR,
                             style="^", width=self._MARKER_WIDTH)
                
                # 피봇 정보 저장 (crosshair용)
                if "confirmed_at_idx" in dfp.columns and "registered_at_idx" in dfp.columns:
                    pivot_info_list.append(dfp[["idx", "y", "t", "confirmed_at_idx", "registered_at_idx"]].copy())
                elif "confirmed_at_idx" in dfp.columns:
                    pivot_info_list.append(dfp[["idx", "y", "t", "confirmed_at_idx"]].copy())
                else:
                    pivot_info_list.append(dfp[["idx", "y", "t"]].copy())
                
                # 확정 피봇을 과거 데이터에 저장 (확률 계산용)
                # 이미 저장된 idx는 건너뜀 (중복 방지)
                existing_idxs = {p.idx for p in prob_calc.historical_pivots} if prob_calc else set()
                for row in dfp.itertuples(index=False):
                    try:
                        pivot_idx = int(row.idx)
                        if pivot_idx in existing_idxs:
                            continue  # 이미 저장된 피봇은 건너뜀
                        if pivot_idx < len(x_idx):
                            timestamp = x_idx[pivot_idx] if hasattr(x_idx, '__getitem__') else None
                        else:
                            timestamp = None

                        hist_pivot = HistoricalPivot(
                            idx=pivot_idx,
                            price=float(row.y),
                            pivot_type=str(row.t),
                            confirmed=True,
                            confirmation_bars=0,  # 확정된 피봇은 0
                            price_deviation_pct=0.0,
                            timestamp=timestamp
                        )
                        # 피봇 확정 확률 계산기에 추가 (설정된 경우만)
                        if prob_calc is not None:
                            prob_calc.add_pivot(hist_pivot)
                    except Exception:
                        pass
            else:
                mh = ts == "H"
                self._upsert("_zz_unconf_H",
                             xs[mh] if mh.any() else np.array([]),
                             ys2[mh] if mh.any() else np.array([]),
                             self.ax_main, color=self._UNCONF_COLOR,
                             style="v", width=self._MARKER_WIDTH)
                if mh.any():
                    new_unconf_marker_names.add("_zz_unconf_H")
                ml = ts == "L"
                self._upsert("_zz_unconf_L",
                             xs[ml] if ml.any() else np.array([]),
                             ys2[ml] if ml.any() else np.array([]),
                             self.ax_main, color=self._UNCONF_COLOR,
                             style="^", width=self._MARKER_WIDTH)
                if ml.any():
                    new_unconf_marker_names.add("_zz_unconf_L")
                
                # unconfirmed 피봇도 crosshair용으로 저장
                if "registered_at_idx" in dfp.columns:
                    pivot_info_list.append(dfp[["idx", "y", "t", "registered_at_idx"]].copy())
                else:
                    pivot_info_list.append(dfp[["idx", "y", "t"]].copy())
        
        try:
            _bucket("confirmed")
            _bucket("unconfirmed")
            # 후보 마커 목록 원자적 교체 (경쟁 상태 방지)
            self._unconf_marker_names = new_unconf_marker_names
        except Exception as e:
            logger.error("[FpltRenderer][RT] 피봇 렌더링 실패: %s", e)

        # ZigZag 폴리라인 렌더링 (비활성화 - 불필요)
        # self._render_zigzag_line(x_idx, pm)

        # 피봇 정보 병합 (confirmed + unconfirmed)
        if pivot_info_list:
            self._pivot_info = pd.concat(pivot_info_list, ignore_index=True)
            # numpy 캐싱 (마우스 이동 성능 최적화)
            self._pivot_idx_arr = self._pivot_info["idx"].to_numpy(dtype=np.int32)
            self._pivot_y_arr = self._pivot_info["y"].to_numpy(dtype=np.float64)
        else:
            self._pivot_info = None
            self._pivot_idx_arr = np.array([], dtype=np.int32)
            self._pivot_y_arr = np.array([], dtype=np.float64)

    # ── 거래 마커 ────────────────────────────────────────────────────────────

    def _render_trade_markers(self, x_idx: np.ndarray,
                              trade_events: List[Dict]) -> None:
        # 거래 이벤트 해시 비교 (변경 감지)
        trade_hash = hash(tuple(
            (e.get("timestamp"), e.get("price")) for e in (trade_events or [])
        ))
        if trade_hash == getattr(self, "_last_trade_hash", None):
            return
        self._last_trade_hash = trade_hash

        # 기존 마커 숨기기 (삭제 금지)
        for nm in list(k for k in self._plots if k.startswith("_trade_")):
            obj = self._plots.get(nm)
            if obj is not None:
                try:
                    obj.setOpacity(0.0)
                except Exception:
                    self._remove(nm)
        if not trade_events:
            return

        # x_idx int64 배열을 미리 계산해 np.searchsorted 바이섹션 사용
        # x_idx는 시간순 정렬되어 있어야 함 (render() 진입 시 보장됨)
        xi64_sorted = x_idx.astype("datetime64[ns]").astype(np.int64)
        # 정렬 보장 검증 (디버깅용)
        if not np.all(xi64_sorted[:-1] <= xi64_sorted[1:]):
            logger.warning("[FpltRenderer] x_idx가 정렬되지 않음, searchsorted 정확도 보장 불가")
            xi64_sorted = np.sort(xi64_sorted)  # 정렬 수행

        for ev in trade_events:
            try:
                ts     = pd.to_datetime(ev.get("timestamp"))
                price  = float(ev.get("price", 0))
                action = ev.get("action", "")
                etype  = ev.get("event_type", "")
                tx64   = int(pd.Timestamp(ts).value)

                # O(N) argmin → O(log N) searchsorted
                pos = np.searchsorted(xi64_sorted, tx64)
                if pos >= len(xi64_sorted):
                    pos = len(xi64_sorted) - 1
                elif pos > 0:
                    if abs(xi64_sorted[pos - 1] - tx64) < abs(xi64_sorted[pos] - tx64):
                        pos = pos - 1
                idx = int(pos)

                if etype == "ENTRY":
                    color = self._ENTRY_LONG_COLOR if action=="BUY" else self._ENTRY_SHORT_COLOR
                    style = "^" if action=="BUY" else "v"
                    key   = f"_trade_entry_{ts.strftime('%Y%m%d%H%M%S')}"
                else:
                    color = self._EXIT_LONG_COLOR if action=="BUY" else self._EXIT_SHORT_COLOR
                    style, key = "o", f"_trade_exit_{ts.strftime('%Y%m%d%H%M%S')}"
                self._upsert(key,
                             np.array([x_idx[idx]]), np.array([price]),
                             self.ax_main, color=color, style=style, width=2.0)
                
                # 거래 이벤트 로그 기록 (피봇 상태 포함)
                if hasattr(self, '_pivot_info') and self._pivot_info is not None:
                    # 가장 가까운 피봇 찾기
                    pivot_row = None
                    if hasattr(self, '_pivot_idx_arr') and len(self._pivot_idx_arr) > 0:
                        pivot_idx_arr = self._pivot_idx_arr
                        pivot_y_arr = self._pivot_y_arr
                        # 거래 시점 인덱스에 가장 가까운 피봇 찾기
                        closest_idx = np.argmin(np.abs(pivot_idx_arr - idx))
                        pivot_row = self._pivot_info.iloc[closest_idx] if closest_idx < len(self._pivot_info) else None
                    
                    pivot_status = ""
                    pivot_info = ""
                    if pivot_row is not None:
                        # 피봇 타입 확인
                        pivot_type = str(pivot_row.get('t', ''))
                        # confirmed_at_idx가 있으면 확정, 없으면 미확정
                        if 'confirmed_at_idx' in pivot_row and pd.notna(pivot_row['confirmed_at_idx']):
                            pivot_status = "확정"
                        else:
                            pivot_status = "미확정"
                        pivot_info = f"{pivot_type}@{pivot_row['y']:.2f}"
                    else:
                        pivot_status = "없음"
                    
                    # 콜백 호출하여 로그 기록
                    if self._trade_event_callback is not None:
                        try:
                            self._trade_event_callback(
                                timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
                                action=action,
                                price=price,
                                event_type=etype,
                                pivot_status=pivot_status,
                                pivot_info=pivot_info
                            )
                        except Exception:
                            pass
            except Exception:
                pass

    def _refresh_trade_markers(self) -> None:
        pass

    # ── MA 오버레이 ─────────────────────────────────────────────────────────────

    def set_trade_event_callback(self, callback: callable) -> None:
        """거래 이벤트 콜백 설정."""
        self._trade_event_callback = callback

    def set_cancel_check_callback(self, callback: callable) -> None:
        """취소 확인 콜백 설정."""
        self._cancel_check_callback = callback

    def set_ma_enabled(self, enabled: bool) -> None:
        """MA 표시 on/off (컨트롤 바 체크박스와 연결).
        
        깜빡임 방지를 위해 _remove 대신 setOpacity(0.0)로 숨김
        """
        self._ma_enabled = enabled
        for name in ("_ma20", "_ma60"):
            obj = self._plots.get(name)
            if obj is not None:
                try:
                    obj.setOpacity(1.0 if enabled else 0.0)
                except Exception as e:
                    logger.debug("[FpltRenderer] setOpacity 실패 (%s): %s", name, e)
                    # [FIX-RENDER-2] _remove 대신 setVisible 사용
                    try:
                        obj.setVisible(enabled)
                    except Exception:
                        if not enabled:
                            self._remove(name)

    def _render_ma(self, x_idx: np.ndarray, df: pd.DataFrame) -> None:
        """20/60 EMA 오버레이."""
        if not self._ma_enabled:
            return

        close = df["Close"].astype(np.float64)

        # MA 렌더링 루프 (DRY 원칙 적용)
        for span, key, color in [
            (20, "_ma20", self._MA20_COLOR),
            (60, "_ma60", self._MA60_COLOR),
        ]:
            if len(close) >= 5:
                ma = close.ewm(span=span, adjust=False).mean().values
                valid = np.isfinite(ma)
                self._upsert(key, x_idx[valid], ma[valid],
                             self.ax_main, color=color, style="-", width=1.2)
            else:
                obj = self._plots.get(key)
                if obj is not None:
                    try:
                        obj.setOpacity(0.0)
                    except Exception:
                        self._remove(key)

    # ── 현재가 수평선 ───────────────────────────────────────────────────────────

    # ── 조건부 렌더링 (렌더링 성능 최적화) ─────────────────────────────────
    # 전략 A: 렌더 완전 스킵 (변경 없을 때 CPU 0%)
    #   - 시그니처: (데이터 길이, 마지막 Close, 피봇 해시, 피봇 표시)
    #   - 시그니처가 동일하면 렌더링 스킵
    # 전략 C: 캔들/피봇/MA 갱신 조건 분리
    #   - 캔들 갱신: 새 봉 추가 OR 마지막 봉 Close 변경
    #   - 피봇 갱신: 피봇 해시 변경 OR 피봇 표시 변경
    #   - MA 갱신: 새 봉 추가 시만
    # 결과: 불필요한 렌더링 최소화로 CPU 사용량 감소
    # ───────────────────────────────────────────────────────────────────────────────

    def _render_current_price_line(self, current_price: float) -> None:
        """현재가 수평 점선 표시.

        [ANTI-BLINK] removeItem/addItem 반복은 제거 순간~추가 순간 사이에
        빈 프레임이 노출되어 깜빡임을 유발한다.
        한 번만 생성하고 이후엔 setValue()로 위치만 갱신한다.
        """
        existing = self._plots.get("_cur_price")

        if current_price <= 0:
            if existing is not None:
                try:
                    existing.setVisible(False)
                except Exception:
                    self._remove("_cur_price")
            return

        try:
            import pyqtgraph as pg
            from PySide6.QtCore import Qt

            if existing is not None:
                # 위치만 갱신 — Scene 변경 없으므로 paint 이벤트 최소화
                try:
                    existing.setValue(float(current_price))
                    existing.setVisible(True)
                    try:
                        if hasattr(existing, 'label') and existing.label is not None:
                            existing.label.setFormat(f"{current_price:.2f}")
                    except Exception:
                        pass
                    return
                except Exception:
                    # 라인 객체가 깨진 경우에만 제거 후 재생성
                    vb_rm = getattr(self.ax_main, 'vb', None)
                    if vb_rm is not None:
                        try:
                            vb_rm.removeItem(existing)
                        except Exception:
                            pass
                    self._plots.pop("_cur_price", None)

            # 최초 1회 생성
            vb = getattr(self.ax_main, 'vb', None)
            if vb is None:
                return

            line = pg.InfiniteLine(
                pos=float(current_price),
                angle=0,
                movable=False,
                pen=pg.mkPen(self._CUR_PRICE_COLOR, width=1, style=Qt.DashLine),
                label=f"{current_price:.2f}",
                labelOpts={"color": self._CUR_PRICE_COLOR, "position": 0.98},
            )
            line.setZValue(100)
            vb.addItem(line, ignoreBounds=True)
            self._plots["_cur_price"] = line

        except Exception as e:
            logger.debug("[FpltRenderer] 현재가 라인 실패: %s", e)

    # ── 공개 render ──────────────────────────────────────────────────────────

    def render(self, df: pd.DataFrame, pm: Optional[Dict],
               data_source: str = "futures",
               trade_events: Optional[List[Dict]] = None,
               current_price: float = 0.0,
               force_clear: bool = False,
               show_pivots: bool = True,
               pivot_prob_calc: Optional[Any] = None,
               minutes: int = 120) -> bool:
        """차트 렌더링.

        Returns:
            실제로 차트가 재렌더링되었으면 True, 증분 업데이트만 수행되면 False
        """
        # 렌더링 시간 측정 시작
        render_start_time = time.time()

        if df is None or df.empty:
            logger.warning("[FpltRenderer][RT] render: df None/empty — 렌더링 스킵")
            return False

        # 데이터 시간순 정렬 보장 (searchsorted 정확성 보장)
        if isinstance(df.index, pd.DatetimeIndex):
            if not df.index.is_monotonic_increasing:
                logger.warning("[FpltRenderer] DataFrame index가 정렬되지 않음, 정렬 수행")
                df = df.sort_index()

        # 취소 확인 (초기 단계)
        # 취소 시 rendered_ok=False이므로 finally에서 상태 저장 안 됨
        # 다음 호출에서 full_sig가 동일해도 is_first_render=True로 간주되어 재시도 보장됨
        if self._cancel_check_callback is not None and self._cancel_check_callback():
            logger.debug("[FpltRenderer] 렌더링 취소됨 (초기 단계)")
            return False

        # 피봇 확정 확률 계산기 설정
        self._pivot_prob_calc = pivot_prob_calc

        # ── 증분 업데이트 감지 로직 ─────────────────────────────────────────────
        # 데이터 길이 비교로 새 봉 추가 감지
        current_len = len(df)
        last_len = self._last_df_len

        # 마지막 봉 Close 값 추적 (틱 갱신 감지용)
        last_close = getattr(self, "_last_close", None)
        current_close = float(df["Close"].iloc[-1]) if len(df) > 0 else None
        # 최초 렌더링 시 close_changed=False이지만 is_first_render=True로 처리됨
        close_changed = (last_close is not None and current_close is not None and last_close != current_close)

        # ── [FIX2] 캔들 데이터 실제 변경 감지 ──
        # full_sig candle_hash와 동일한 컬럼 기준(OHLC+SuperTrend)으로 통일.
        # 기존 OHLC만 해시 시: ST 변경 → candle_data_changed=False인데도
        # full_sig는 달라져 불필요한 렌더 진입 + ST 업데이트 누락이 반복되었음.
        last_candle_hash = getattr(self, "_last_candle_hash", None)
        try:
            _hash_cols = ["Open", "High", "Low", "Close"]
            if "SuperTrend" in df.columns:
                _hash_cols.append("SuperTrend")
            if "SuperTrend_Dir" in df.columns:
                _hash_cols.append("SuperTrend_Dir")
            last_candle = df.iloc[-1][_hash_cols].values.tobytes()
            current_candle_hash = zlib.adler32(last_candle)
            candle_data_changed = (last_candle_hash is not None and last_candle_hash != current_candle_hash)
        except Exception:
            current_candle_hash = 0
            candle_data_changed = False

        # 피봇 표시 변경 감지
        last_show_pivots = getattr(self, "_last_show_pivots", True)
        show_pivots_changed = (show_pivots != last_show_pivots)
        self._last_show_pivots = show_pivots

        # 피봇 마커 변경 감지: 경량 해시 비교
        pm_hash = self._pm_hash(pm)

        # 기본값 먼저 설정 (NameError 방지)
        is_new_bar = (current_len > last_len)
        pm_changed = (self._last_pm_hash is None) or (pm_hash != self._last_pm_hash)

        # 피봇 변경 감지 로그 (깜빡임 원인 추적)
        if pm_changed and self._last_pm_hash is not None:
            logger.warning("[FpltRenderer][RT] 피봇 변경 감지: last_pm_hash=%s, pm_hash=%s", self._last_pm_hash, pm_hash)

        # 범위 변경 감지 (캔들 전체 재렌더 강제 제거)
        minutes_changed = (minutes != self._last_minutes and self._last_minutes >= 0)
        self._last_minutes = minutes

        if minutes_changed:
            for nm in list(k for k in self._plots if k.startswith("_zz_")):
                self._remove(nm)
            self._reset_render_state("minutes_changed")  # _last_full_sig = None → is_first_render = True
            pm_changed = True

        # 강제 전체 초기화 (갱신 버튼 클릭 시)
        # force_clear 시에는 데이터가 완전히 새로 로드되므로 피봇 마커 삭제 후 재생성
        if force_clear:
            logger.info("[FpltRenderer][RT] 강제 전체 재렌더링 (갱신 버튼) - 피봇 마커 삭제")
            for nm in list(k for k in self._plots if k.startswith("_zz_")):
                self._remove(nm)
            self._reset_render_state("force_clear")
            is_new_bar = True  # 강제 전체 재렌더링
            pm_changed = True  # 피봇 마커도 무조건 재렌더링
        # _last_df_len은 finally에서 저장됨

        # 데이터 소스 변경 감지 (플래그 분리)
        source_changed = (self._current_data_source != data_source)
        if source_changed:
            logger.warning("[FpltRenderer][RT] 데이터 소스 변경 감지: %s → %s, 피벗 상태 초기화",
                          self._current_data_source, data_source)
            # 전체 플롯 삭제 - finplot 내부 datasrc 캐시 완전 초기화 (DataFrame 구조 변경 대응)
            self.clear_all()
            # ax.reset() 제거 - 캔들 표시 문제 방지
            # ax.reset()은 finplot 내부 datasrc를 완전 초기화하여 캔들 표시에 문제 유발
            # Y축 범위는 렌더링 후 재설정
            # 피벗 관련 상태 초기화 (위치 틀어짐 방지)
            self._pivot_info = None
            self._pivot_idx_arr = np.array([], dtype=np.int32)
            self._pivot_y_arr = np.array([], dtype=np.float64)
            self._last_pm_hash = None
            # zigzag 재초기화 (데이터 소스에 따른 config 변경)
            # FpltRenderer는 렌더링만 담당하며, zigzag 초기화는 ChartViewer에서 처리됨
            self._reset_render_state("source_changed")
            self._current_data_source = data_source
            self._xaxis_needs_reset = True
            self._yaxis_needs_reset = True  # Y축 범위도 재설정 필요
            is_new_bar = True  # 소스 변경 시 전체 재렌더링 (기본값 덮어쓰기)
            pm_changed = True  # 소스 변경 시 피벗도 무조건 재렌더링 (기본값 덮어쓰기)
        # _last_show_pivots는 상단에서 이미 갱신됨 (중복 제거)

        # 전략 A: 렌더 완전 스킵 (변경 없을 때)
        # 시그니처: (데이터 길이, 마지막 캔들 해시, current_close, 피봇 해시, 피봇 표시, 범위)
        # ── [FIX] close_changed 포함 (틱 업데이트 시 마지막 캔들 갱신) ──
        # close_changed 포함: 틱 업데이트 시 렌더링 호출하여 마지막 캔들 갱신
        # candle_hash 포함: 캔들 데이터 실제 변경 감지 (전체 렌더링 vs 마지막 캔들만 갱신 구분)
        # SuperTrend 포함: 슈퍼트렌드 변경 감지 (슈퍼트렌드 깨짐 방지)
        try:
            # 마지막 캔들 데이터 해시 계산 (캔들 변경 감지용)
            cols_for_hash = ["Open", "High", "Low", "Close"]
            # SuperTrend 컬럼이 있으면 해시에 포함
            if "SuperTrend" in df.columns:
                cols_for_hash.append("SuperTrend")
            if "SuperTrend_Dir" in df.columns:
                cols_for_hash.append("SuperTrend_Dir")
            last_candle = df.iloc[-1][cols_for_hash].values.tobytes()
            candle_hash = zlib.adler32(last_candle)
        except Exception:
            candle_hash = 0

        full_sig = (current_len, candle_hash, current_close, pm_hash, show_pivots, minutes)
        last_full_sig = getattr(self, "_last_full_sig", None)
        # 초기 로드 시에는 스킵하지 않음 (force_clear/source_changed 이후 판단)
        is_first_render = (self._last_full_sig is None)
        if not is_first_render and last_full_sig is not None and full_sig == last_full_sig:
            # 변경 없음: 렌더 스킵 (현재가 라인만 업데이트)
            logger.debug("[FpltRenderer] 렌더 스킵 (변경 없음) - full_sig=%s", full_sig)
            return False
        # 시그니처 저장은 렌더링 완료 후 finally에서 수행
        # ───────────────────────────────────────────────────────────────────────

        try:
            # OHLCV + SuperTrend 컬럼 복사 (인덱스 유지)
            cols_to_copy = ["Open", "High", "Low", "Close", "Volume"]
            if "SuperTrend" in df.columns:
                cols_to_copy.append("SuperTrend")
            if "SuperTrend_Dir" in df.columns:
                cols_to_copy.append("SuperTrend_Dir")
            df_ohlc = df[cols_to_copy].copy()
            # 인덱스 유지 확인
            df_ohlc.index = df.index
        except Exception:
            df_ohlc = df.copy()

        # 렌더링 로직 전체를 try-finally로 감싸서 시그니처 저장 보장
        rendered_ok = False
        try:
            # DatetimeIndex 보존 — reset_index 금지 (x축 눈금 표시에 필수)
            for col in df_ohlc.columns:
                try:
                    df_ohlc[col] = pd.to_numeric(df_ohlc[col], errors="coerce").astype(np.float64)
                except Exception:
                    pass

            idx_plot = (df_ohlc.index
                        if isinstance(df_ohlc.index, pd.DatetimeIndex)
                        else pd.to_datetime(df_ohlc.index))
            x_idx = self._to_x(idx_plot)

            # 데이터 인덱스 저장 (십자선 시간 변환용)
            self._df_index = idx_plot

            # 취소 확인 (캔들 렌더링 전)
            if self._cancel_check_callback is not None and self._cancel_check_callback():
                logger.debug("[FpltRenderer] 렌더링 취소됨 (캔들 렌더링 전)")
                return False

            if not self._xaxis_done:
                self._setup_xaxis_format()
                self._xaxis_done = True

            # ── 조건부 렌더링 로직 (렌더링 성능 최적화) ─────────────────────────────
            # 전략 A: 렌더 완전 스킵 (변경 없을 때)
            #   - 시그니처: (데이터 길이, 마지막 Close, 피봇 해시, 피봇 표시)
            #   - 시그니처가 동일하면 렌더링 스킵 (return False)
            # 전략 C: 캔들/피봇/MA 갱신 조건 분리
            #   - 캔들 갱신: 새 봉 추가 OR 마지막 봉 Close 변경
            #   - 피봇 갱신: 피봇 해시 변경 OR 피봇 표시 변경
            #   - MA 갱신: 새 봉 추가 시만
            # 결과: 불필요한 렌더링 최소화로 CPU 사용량 감소
            # ───────────────────────────────────────────────────────────────────────

            # 피봇 표시 상태 변경 시 피봇 마커 초기화
            if show_pivots_changed:
                for nm in list(k for k in self._plots if k.startswith("_zz_")):
                    self._remove(nm)
                # _last_pm_hash는 finally에서 pm_hash로 저장되므로 여기서 초기화 불필요
                # pm_changed는 이미 True이므로 피봇 재렌더링 보장됨

            # 전략 C: 캔들/피벗/MA/SuperTrend 갱신 조건 분리
            # ★ close_changed(동일 분봉 틱 갱신)도 candle_changed에 포함한다.
            #   _render_candles 내부에서 update_data(clean 없음)를 사용하므로
            #   기존 아이템을 지우지 않고 데이터만 교체 → 깜빡임 없음.
            candle_changed = is_new_bar or candle_data_changed or close_changed or minutes_changed
            pivot_changed = pm_changed or show_pivots_changed or minutes_changed  # 범위 변경 시에도 피벗 갱신
            ma_changed = is_new_bar or minutes_changed  # 범위 변경 시에도 MA 갱신
            # [FIX1] close_changed 추가: 틱 갱신 시 캔들은 update_data되지만 ST는 스킵되어
            # x축 불일치(캔들 N+1봉, ST N봉)가 발생하던 버그 수정.
            # close_changed 제거: 동일 분봉 틱 갱신 시 ST는 변하지 않음 → 매 틱 재렌더 방지
            supertrend_changed = is_new_bar or candle_data_changed or minutes_changed

            # 범위 변경 시 캔들/피벗 강제 재생성
            if minutes_changed:
                logger.warning("[FpltRenderer][RT] 범위 변경 감지: minutes=%d, last_minutes=%d, bars=%d",
                             minutes, self._last_minutes, len(x_idx))
                # clear_all 대신 캔들만 update_data로 갱신 (깜빡임 방지)
                # 피벗/MA/SuperTrend는 조건부 재렌더링
                # ax.reset() 제거 - 실시간 업데이트 시 깨짐 방지
                # ax.reset()은 finplot 내부 datasrc를 완전 초기화하여 깨짐 유발
                self._xaxis_needs_reset = True  # x축 뷰 재설정 트리거
                # 캔들은 candle_changed 조건으로 이미 갱신됨
                # 피벗/MA/SuperTrend는 각각의 changed 조건으로 재렌더링

            # [ANTI-BLINK] 모든 아이템 변경을 하나의 paint 이벤트로 통합
            # Scene 업데이트를 일시 중단 → 중간 상태(아이템 제거 후/추가 전)가
            # 화면에 노출되지 않아 깜빡임이 사라짐
            _blink_scene = None
            try:
                _blink_vb = getattr(self.ax_main, 'vb', None)
                if _blink_vb is not None:
                    _s = _blink_vb.scene() if callable(getattr(_blink_vb, 'scene', None)) else None
                    if _s is not None and hasattr(_s, 'setUpdatesEnabled'):
                        _s.setUpdatesEnabled(False)
                        _blink_scene = _s
            except Exception:
                pass

            # 캔들 갱신 (동일 분봉 틱 포함)
            if candle_changed:
                logger.debug("[FpltRenderer][RT] 캔들 렌더링: candle_changed=%s, bars=%d", candle_changed, len(x_idx))
                self._render_candles(x_idx, df_ohlc)
                self._render_price_lines(x_idx, df_ohlc)
                self._render_volume(x_idx, df_ohlc)

            # SuperTrend 갱신 (완결봉 추가 또는 범위 변경 시만)
            if supertrend_changed:
                logger.debug("[FpltRenderer] SuperTrend 렌더링: supertrend_changed=%s, bars=%d",
                             supertrend_changed, len(x_idx))
                self._render_supertrend(x_idx, df_ohlc)

            # 피벗 갱신
            # 초기 로드 시(force_clear 또는 데이터 소스 변경 또는 범위 변경)에는 무조건 렌더링
            is_initial_render = force_clear or source_changed or is_first_render or minutes_changed
            # ── [FIX] 불필요한 로그 출력 제거 (깜빡임 방지) ──
            # 실제 렌더링 호출 시에만 INFO 로그 출력
            if pivot_changed or is_initial_render:
                logger.info("[ChartViewer] 피벗 렌더링 호출: pivot_changed=%s, is_initial_render=%s, pm=%s", pivot_changed, is_initial_render, bool(pm))
                logger.info("[ChartViewer] pm 타입: %s, isinstance(pm, dict): %s", type(pm), isinstance(pm, dict) if pm is not None else "pm is None")
                if pm and isinstance(pm, dict) and show_pivots:
                    if pivot_changed:
                        logger.info("[FpltRenderer][RT] 피봇 변경 감지: pm=%s, show_pivots=%s, is_initial_render=%s",
                                    bool(pm), show_pivots, is_initial_render)
                    self._render_pivots(x_idx, pm, pm_hash=pm_hash)
                else:
                    for nm in list(k for k in self._plots if k.startswith("_zz_")):
                        obj = self._plots.get(nm)
                        if obj is not None:
                            try:
                                obj.setOpacity(0.0)
                            except Exception:
                                self._remove(nm)
                # _last_pm_hash 갱신은 finally에서 수행 (rendered_ok 보호)

            # MA 갱신
            if ma_changed:
                self._render_ma(x_idx, df_ohlc)

            logger.debug("[ChartViewer] 조건부 렌더링: candle=%s, pivot=%s, ma=%s, bars=%d",
                        candle_changed, pivot_changed, ma_changed, current_len)

            # 현재가 수평선 (항상 업데이트)
            self._render_current_price_line(current_price)

            # 거래 마커 렌더 (trade_events 가 주입된 경우)
            if trade_events is not None:
                self._render_trade_markers(x_idx, trade_events)

            # x축 범위를 전체 데이터로 설정 (최초 렌더 또는 소스 변경 시만)
            if is_first_render or source_changed:
                try:
                    if self.ax_main is not None and len(x_idx) > 0:
                        # finplot ViewBox API 사용
                        if hasattr(self.ax_main, 'vb'):
                            self.ax_main.vb.setXRange(x_idx[0], x_idx[-1], padding=0)
                except Exception as e:
                    logger.debug(f"[ChartViewer] x축 범위 설정 실패: {e}")

            # Y축 범위를 현재가를 포함하도록 조정 (최초 렌더 또는 소스 변경 시만)
            if is_first_render or source_changed:
                try:
                    if self.ax_main is not None and len(df_ohlc) > 0:
                        # 캔들 데이터의 최고가/최저가
                        df_high = df_ohlc["High"].max()
                        df_low = df_ohlc["Low"].min()

                        # 현재가가 범위 밖에 있으면 범위 확장
                        y_min = df_low
                        y_max = df_high
                        padding = (y_max - y_min) * 0.05  # 5% 패딩

                        if current_price > 0:
                            if current_price > y_max:
                                y_max = current_price + padding
                            elif current_price < y_min:
                                y_min = current_price - padding

                        # finplot ViewBox API 사용
                        if hasattr(self.ax_main, 'vb'):
                            self.ax_main.vb.setYRange(y_min, y_max, padding=0)
                            logger.info("[FpltRenderer] Y축 범위 설정: low=%.2f, high=%.2f", y_min, y_max)
                except Exception as e:
                    logger.debug(f"[ChartViewer] Y축 범위 설정 실패: {e}")

            # [ANTI-BLINK] Scene 업데이트 재개 — 여기서 한 번만 paint
            if _blink_scene is not None:
                try:
                    _blink_scene.setUpdatesEnabled(True)
                except Exception:
                    pass

            # 실제로 재렌더링되었는지 반환
            rendered_ok = True
            return candle_changed or pivot_changed or ma_changed
        finally:
            # 렌더링 시간 측정 및 로그
            render_elapsed = time.time() - render_start_time

            # 렌더링 원인 분석 (깜박임 원인 추적)
            render_cause = []
            if is_new_bar:
                render_cause.append("NEW_BAR")
            if candle_data_changed:
                render_cause.append("CANDLE_DATA_CHANGED")
            if close_changed:
                render_cause.append("TICK_UPDATE")
            if pivot_changed:
                render_cause.append("PIVOT_CHANGED")
            if source_changed:
                render_cause.append("SOURCE_CHANGED")
            if force_clear:
                render_cause.append("FORCE_CLEAR")
            if is_first_render:
                render_cause.append("FIRST_RENDER")

            cause_str = ",".join(render_cause) if render_cause else "NO_CHANGE"

            # 렌더링 시간에 따른 로그 레벨 결정
            if render_elapsed >= 0.5:  # 500ms 이상: 깜박임 발생 가능
                logger.error(
                    "[FpltRenderer][RT] 렌더링 시간이 너무 깁니다 (깜박임 발생 가능): "
                    "%.3f초 | cause=%s | data_source=%s | bars=%d | is_new_bar=%s | candle_changed=%s | pivot_changed=%s",
                    render_elapsed, cause_str, data_source, current_len, is_new_bar, candle_data_changed, pivot_changed
                )
            elif render_elapsed >= 0.2:  # 200-500ms: 경고
                logger.warning(
                    "[FpltRenderer][RT] 렌더링 시간이 느립니다: "
                    "%.3f초 | cause=%s | data_source=%s | bars=%d | is_new_bar=%s | candle_changed=%s | pivot_changed=%s",
                    render_elapsed, cause_str, data_source, current_len, is_new_bar, candle_data_changed, pivot_changed
                )
            elif render_elapsed >= 0.1:  # 100-200ms: 정보
                logger.info(
                    "[FpltRenderer][RT] 렌더링 시간: "
                    "%.3f초 | cause=%s | data_source=%s | bars=%d | is_new_bar=%s | candle_changed=%s | pivot_changed=%s",
                    render_elapsed, cause_str, data_source, current_len, is_new_bar, candle_data_changed, pivot_changed
                )
            else:  # 100ms 미만: 디버그
                logger.debug(
                    "[FpltRenderer][RT] 렌더링 시간: "
                    "%.3f초 | cause=%s | data_source=%s | bars=%d",
                    render_elapsed, cause_str, data_source, current_len
                )

            # 렌더링 성공 시에만 저장 (예외 발생 시 재시도 가능)
            if rendered_ok:
                self._last_full_sig = full_sig
                self._last_close = current_close
                self._last_candle_hash = current_candle_hash  # 캔들 해시 저장
                self._last_df_len = current_len  # force_clear 후에도 저장
                self._last_pm_hash = pm_hash  # 피봇 해시도 finally에서 통합 관리
                self._yaxis_needs_reset = False


    def _reset_render_state(self, reason: str) -> None:
        """렌더 상태 초기화 (소스 변경, 강제 초기화, 범위 변경 공통 처리)"""
        self._last_candle_time = None
        self._last_df_len = 0
        self._last_pm_hash = None
        self._last_close = None
        self._last_candle_hash = None
        self._last_full_sig = None
        logger.debug("[FpltRenderer] 렌더 상태 초기화: reason=%s", reason)

    def clear_all(self) -> None:
        for name in list(self._plots.keys()):
            self._remove(name)

# ══════════════════════════════════════════════════════════════════════════════
# §3  Qt 위젯
