"""피봇 품질 모니터링 시스템 v2.0

파일: gui/components/pivot_quality_monitor.py

chart_viewer.py 통합 (패치 4군데):
  [1] import 추가:
      from gui.components.pivot_quality_monitor import PivotQualityMonitor

  [2] __init__ 멤버:
      self._pivot_quality_monitor: PivotQualityMonitor = PivotQualityMonitor()

  [3] _build_widget() 내 _build_pivot_event_log 호출 바로 위:
      try:
          self._pivot_quality_monitor.build(container, root)
      except Exception as e:
          logger.warning("[ChartViewerWidget] 피봇 품질 모니터 빌드 실패: %s", e)

  [4] _auto_refresh_callback() 내 self.refresh() 바로 앞:
      try:
          if hasattr(self, '_engine') and self._engine is not None \
                  and self._engine._zz is not None:
              self._pivot_quality_monitor.update(
                  zz=self._engine._zz,
                  cfg=self._engine._zz_cfg,
              )
      except Exception as _e:
          logger.debug("[ChartViewer] PivotQualityMonitor update 실패: %s", _e)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── 메트릭 데이터 클래스 ─────────────────────────────────────────────

@dataclass
class PivotQualityMetrics:
    """피봇 품질 메트릭 스냅샷."""

    # 기본 카운터  (기준: 세션 전체 누적)
    total_confirmed: int   = 0   # 현재 _all_swings 내 확정 피봇 수
    total_registered: int  = 0   # 세션 누적 후보 등록 수
    total_cancelled: int   = 0   # 세션 누적 취소 수

    # 안정성
    cancel_rate: float     = 0.0  # cancelled / registered (0~1)

    # 밀도: 최근 60봉(=60분) 기준
    pivots_per_hour: float = 0.0

    # 지연(Lag): 피봇 발생 봉 → 확정 봉
    avg_lag_bars: float    = 0.0
    max_lag_bars: int      = 0

    # 지연 문제 진단
    lag_critical: bool     = False  # 평균 지연 10봉 이상
    lag_suggestion: str    = ""     # 지연 문제 대안 제안

    # 방향 정확도: 피봇 이후 반대 방향으로 실제 움직였는지 (피봇 봉 기준)
    accuracy_score: float  = 0.0
    accuracy_sample: int   = 0   # 정확도 계산에 사용된 피봇 수

    # 파동 크기
    avg_wave_pct: float    = 0.0

    # 현재 ZigZag 상태
    current_threshold_pct: float = 0.0
    current_atr: float           = 0.0
    current_structure: str       = "unknown"

    # 적응형 엔진 조정 상태
    adaptive_regime: str        = "unknown"
    adaptive_mult: float        = 1.0
    adaptive_er: float          = 0.5
    adaptive_atr_pct: float     = 50.0
    adaptive_density: str       = "normal"

    # 품질 등급 및 진단
    grade: str       = "N/A"
    grade_color: str = "#888888"
    suggestion: str  = ""
    suggestion_level: str = "info"   # "warn" | "info" | "ok"


# ── 품질 분석 엔진 ────────────────────────────────────────────────────

class PivotQualityAnalyzer:
    """AdaptiveZigZag 인스턴스에서 품질 메트릭 계산."""

    GRADE_THRESHOLDS = {
        "A": {"cancel_rate": 0.20, "pph": (4, 20),  "avg_lag": 3.0, "accuracy": 0.75},
        "B": {"cancel_rate": 0.35, "pph": (3, 30),  "avg_lag": 5.0, "accuracy": 0.60},
        "C": {"cancel_rate": 0.50, "pph": (2, 45),  "avg_lag": 8.0, "accuracy": 0.45},
    }
    GRADE_COLORS = {
        "A": "#4ec9b0", "B": "#dcdcaa", "C": "#f48771", "D": "#ff4444",
    }

    # ── 공개 API ─────────────────────────────────────────────────────

    def compute(self, zz: Any, cfg: Any) -> PivotQualityMetrics:
        m = PivotQualityMetrics()
        if zz is None:
            return m
        try:
            # ① 기본 카운터 ────────────────────────────────────────
            m.total_registered = int(getattr(zz, "_candidate_registered_count", 0))
            m.total_cancelled  = int(getattr(zz, "_candidate_cancelled_count",  0))

            # _all_swings 스냅샷 (반복 중 외부 변경 방지)
            all_swings_snap: List[Any] = list(getattr(zz, "_all_swings", []) or [])
            confirmed = [s for s in all_swings_snap if getattr(s, "confirmed", False)]
            m.total_confirmed = len(confirmed)

            if m.total_confirmed < 2:
                return m

            # ② 취소율 ─────────────────────────────────────────────
            if m.total_registered > 0:
                m.cancel_rate = m.total_cancelled / m.total_registered

            # ③ 지연 ───────────────────────────────────────────────
            lags = []
            for s in confirmed:
                c_at = getattr(s, "confirmed_at_idx", -1)
                idx  = getattr(s, "index", -1)
                if c_at >= 0 and idx >= 0:
                    lags.append(max(0, c_at - idx))
            if lags:
                m.avg_lag_bars = sum(lags) / len(lags)
                m.max_lag_bars = max(lags)

            # ③-① 지연 문제 진단 ────────────────────────────────────
            m.lag_critical = m.avg_lag_bars >= 10.0
            if m.lag_critical:
                m.lag_suggestion = "지연 10봉 이상 → pending 등록을 예비 신호로 사용, confirmation_bars 감소 권장"
            elif m.avg_lag_bars >= 6.0:
                m.lag_suggestion = "지연 6봉 이상 → confirmation_bars 동적 조정 검토"

            # ④ 밀도: 최근 60봉 ────────────────────────────────────
            bar_idx = int(getattr(zz, "_bar_idx", 0))
            m.pivots_per_hour = float(
                sum(1 for s in confirmed
                    if getattr(s, "confirmed_at_idx", -1) >= bar_idx - 60)
            )

            # ⑤ 파동 크기 ──────────────────────────────────────────
            m.avg_wave_pct = self._calc_avg_wave(confirmed)

            # ⑥ 방향 정확도 (피봇 봉 기준, deque 범위 내만) ────────
            m.accuracy_score, m.accuracy_sample = self._calc_accuracy(
                confirmed, zz, bar_idx, m.avg_wave_pct
            )

            # ⑦ ZigZag 상태 ────────────────────────────────────────
            state = getattr(zz, "_state", None)
            if state:
                m.current_threshold_pct = float(getattr(state, "adaptive_threshold_pct", 0.0))
                m.current_atr           = float(getattr(state, "atr", 0.0))
                m.current_structure     = str(getattr(state, "structure", "unknown"))

            # ⑧ 적응형 엔진 조정 상태 ───────────────────────────────
            adaptive_engine = getattr(zz, "_adaptive_engine", None)
            if adaptive_engine is not None:
                last_adj = getattr(adaptive_engine, "last_adjustment", None)
                if last_adj is not None:
                    m.adaptive_regime   = str(getattr(last_adj, "regime_label", "unknown"))
                    m.adaptive_mult     = float(getattr(last_adj, "mult", 1.0))
                    m.adaptive_er       = float(getattr(last_adj, "er", 0.5))
                    m.adaptive_atr_pct  = float(getattr(last_adj, "atr_pct", 50.0))
                    m.adaptive_density  = str(getattr(last_adj, "density_signal", "normal"))

            # ⑨ 등급 및 진단 ───────────────────────────────────────
            m.grade, m.grade_color = self._calc_grade(m)
            m.suggestion, m.suggestion_level = self._build_suggestion(m, cfg)

        except Exception as e:
            logger.warning("[PivotQualityAnalyzer] compute 실패: %s", e, exc_info=True)

        return m

    # ── 내부 헬퍼 ────────────────────────────────────────────────────

    def _calc_avg_wave(self, confirmed: List[Any]) -> float:
        """HIGH-LOW 교번 쌍에서 평균 파동 크기(%) 계산."""
        highs = [s for s in confirmed if self._is_high(s)]
        lows  = [s for s in confirmed if not self._is_high(s)]
        wave_pcts: List[float] = []
        for h in highs:
            prev_lows = [l for l in lows if getattr(l, "index", 0) < getattr(h, "index", 0)]
            if not prev_lows:
                continue
            pl  = prev_lows[-1]
            mid = (h.price + pl.price) / 2
            if mid > 0:
                wave_pcts.append(abs(h.price - pl.price) / mid * 100)
        return sum(wave_pcts) / len(wave_pcts) if wave_pcts else 0.0

    def _calc_accuracy(
        self,
        confirmed: List[Any],
        zz: Any,
        bar_idx: int,
        avg_wave_pct: float,
    ) -> tuple[float, int]:
        """
        피봇 발생 봉(s.index) 기준으로 이후 N봉 내 반대 방향 이동 여부 확인.

        수정(v2): c_at → s.index 기준, deque 범위 밖 피봇 자동 제외.
        """
        highs_arr: List[float] = list(getattr(zz, "_highs", []) or [])
        lows_arr:  List[float] = list(getattr(zz, "_lows",  []) or [])
        if not highs_arr:
            return 0.0, 0

        base_offset = bar_idx - len(highs_arr)

        # avg_wave_pct 기반 동적 lookforward (5~15봉)
        lookforward = max(5, min(15, int(avg_wave_pct / 0.1))) if avg_wave_pct > 0 else 5

        correct = 0
        total   = 0

        for s in confirmed[:-1]:   # 마지막 피봇은 미래 데이터 없음
            pivot_idx = getattr(s, "index", -1)
            if pivot_idx < 0:
                continue

            start_fi = pivot_idx + 1
            end_fi   = min(pivot_idx + lookforward, bar_idx)
            if start_fi > end_fi:
                continue

            is_high    = self._is_high(s)
            target_arr = lows_arr if is_high else highs_arr

            any_accessible = False
            matched        = False

            for fi in range(start_fi, end_fi + 1):
                rel = fi - base_offset
                if rel < 0:
                    continue            # deque 버퍼 밖 (너무 오래된 봉)
                if rel >= len(target_arr):
                    break               # 미래 봉
                any_accessible = True
                val = target_arr[rel]
                if is_high and val < s.price:
                    matched = True; break
                if not is_high and val > s.price:
                    matched = True; break

            if not any_accessible:
                continue                # 접근 가능 봉이 하나도 없으면 제외
            total += 1
            if matched:
                correct += 1

        score = correct / total if total > 0 else 0.0
        return score, total

    @staticmethod
    def _is_high(s: Any) -> bool:
        st = getattr(s, "swing_type", None)
        if st is None:
            return False
        return str(st).upper().endswith("HIGH")

    def _calc_grade(self, m: PivotQualityMetrics) -> tuple[str, str]:
        t = self.GRADE_THRESHOLDS
        for grade in ("A", "B", "C"):
            g = t[grade]
            lo, hi = g["pph"]
            if (m.cancel_rate    <= g["cancel_rate"] and
                    lo <= m.pivots_per_hour <= hi and
                    m.avg_lag_bars   <= g["avg_lag"] and
                    m.accuracy_score >= g["accuracy"]):
                return grade, self.GRADE_COLORS[grade]
        return "D", self.GRADE_COLORS["D"]

    def _build_suggestion(
        self, m: PivotQualityMetrics, cfg: Any
    ) -> tuple[str, str]:
        """(메시지, 수준) 반환. 수준: 'warn' | 'info' | 'ok'"""
        issues: List[str] = []

        if m.cancel_rate > 0.40:
            issues.append(f"취소율 {m.cancel_rate:.0%} → confirmation_bars +1 또는 min_wave_pct 상향")

        if m.pivots_per_hour > 30:
            issues.append(f"밀도 {m.pivots_per_hour:.0f}/h 과다 → pivot_threshold_min_pct 상향")
        elif m.pivots_per_hour < 3 and m.total_confirmed > 5:
            issues.append(f"밀도 {m.pivots_per_hour:.1f}/h 부족 → pivot_threshold_min_pct 하향")

        if m.avg_lag_bars > 6:
            if m.lag_critical:
                issues.append(f"지연 {m.avg_lag_bars:.1f}봉 심각 → 이중 신호 시스템: pending 등록을 예비 신호로 사용")
            else:
                issues.append(f"지연 {m.avg_lag_bars:.1f}봉 → confirmation_bars 감소 또는 레짐 기반 조기 확정")

        if m.accuracy_sample >= 5 and m.accuracy_score < 0.50:
            issues.append(f"정확도 {m.accuracy_score:.0%} 낮음 → use_atr_based_filtering 활성화 권장")

        # ATR 필터 비활성: 품질 C/D일 때만 경고, 그 외 info
        if cfg is not None and not getattr(cfg, "use_atr_based_filtering", False):
            if m.grade in ("C", "D"):
                issues.append("ATR필터 OFF → 시장 적응 비활성")
            # A/B 등급이면 별도 표시하지 않음 (파라미터 섹션에 ON/OFF 표시로 충분)

        if not issues:
            return "파라미터 양호", "ok"
        return " | ".join(issues), "warn"


# ── GUI 컴포넌트 ──────────────────────────────────────────────────────

class PivotQualityMonitor:
    """피봇 품질 모니터링 패널 (PySide6 위젯).

    chart_viewer.py에 임베드되어 auto_refresh 주기마다 update() 호출.
    """

    _PANEL_STYLE = """
        QWidget#qualityPanel {
            background-color: #1a1a2e;
            border: 1px solid #2d2d4e;
            border-radius: 4px;
        }
    """
    _LBL  = "font-size: 10px; color: #888; font-family: 'Consolas', monospace;"
    _VAL  = "font-size: 11px; font-weight: bold; font-family: 'Consolas', monospace; color: #d4d4d4;"
    _HDR  = "font-size: 11px; font-weight: bold; color: #9cdcfe;"

    def __init__(self) -> None:
        self._panel: Optional[Any] = None
        self._analyzer = PivotQualityAnalyzer()
        self._last_metrics: Optional[PivotQualityMetrics] = None

        # 위젯 참조
        self._grade_lbl:   Optional[Any]    = None
        self._metric_lbls: Dict[str, Any]   = {}
        self._param_lbls:  Dict[str, Any]   = {}
        self._suggest_lbl: Optional[Any]    = None
        self._bar_widget:  Optional[Any]    = None   # _BarWidget QWidget (직접 보유)

        # 변경 시에만 Qt 호출하기 위한 캐시
        self._cached_texts:  Dict[str, str] = {}
        self._cached_styles: Dict[str, str] = {}

    # ── 빌드 ─────────────────────────────────────────────────────────

    def build(self, parent: Any, root: Any) -> None:
        """피봇 품질 패널을 root 레이아웃에 추가."""
        try:
            from PySide6.QtWidgets import (
                QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
                QLabel, QFrame, QSizePolicy,
            )
            from PySide6.QtCore import Qt

            self._panel = QWidget(parent)
            self._panel.setObjectName("qualityPanel")
            self._panel.setStyleSheet(self._PANEL_STYLE)
            self._panel.setMaximumHeight(200)
            self._panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            hl = QHBoxLayout(self._panel)
            hl.setContentsMargins(6, 4, 6, 4)
            hl.setSpacing(8)

            # 섹션 1: 등급 + 카운터 ─────────────────────────────────
            col1 = QVBoxLayout()
            col1.setSpacing(2)
            col1.addWidget(self._make_hdr(parent, "📊 피봇 품질"))

            self._grade_lbl = QLabel("—", parent)
            self._grade_lbl.setStyleSheet(
                "font-size: 28px; font-weight: bold; color: #888;"
                "font-family: 'Consolas', monospace;"
            )
            self._grade_lbl.setAlignment(Qt.AlignCenter)
            col1.addWidget(self._grade_lbl)

            cg = QGridLayout(); cg.setSpacing(2)
            for row, (key, label) in enumerate([
                ("confirmed",  "확정"),
                ("registered", "등록"),
                ("cancelled",  "취소"),
            ]):
                cg.addWidget(self._make_lbl(parent, label), row, 0)
                v = QLabel("—", parent); v.setStyleSheet(self._VAL)
                v.setAlignment(Qt.AlignRight)
                cg.addWidget(v, row, 1)
                self._metric_lbls[key] = v
            col1.addLayout(cg)
            col1.addStretch()
            hl.addLayout(col1, stretch=2)
            hl.addWidget(self._make_sep(parent))

            # 섹션 2: 품질 메트릭 ────────────────────────────────────
            col2 = QVBoxLayout(); col2.setSpacing(2)
            col2.addWidget(self._make_hdr(parent, "🎯 메트릭"))
            mg = QGridLayout(); mg.setSpacing(2)
            for row, (key, label) in enumerate([
                ("cancel_rate", "취소율"),
                ("density",     "밀도/h"),
                ("avg_lag",     "평균지연"),
                ("max_lag",     "최대지연"),
                ("accuracy",    "방향정확"),
                ("avg_wave",    "평균파동"),
            ]):
                mg.addWidget(self._make_lbl(parent, label), row, 0)
                v = QLabel("—", parent); v.setStyleSheet(self._VAL)
                v.setAlignment(Qt.AlignRight)
                mg.addWidget(v, row, 1)
                self._metric_lbls[key] = v
            col2.addLayout(mg)
            col2.addStretch()
            hl.addLayout(col2, stretch=2)
            hl.addWidget(self._make_sep(parent))

            # 섹션 3: 현재 파라미터 ──────────────────────────────────
            col3 = QVBoxLayout(); col3.setSpacing(2)
            col3.addWidget(self._make_hdr(parent, "⚙ 현재 파라미터"))
            pg = QGridLayout(); pg.setSpacing(2)
            for row, (key, label) in enumerate([
                ("threshold", "임계값"),
                ("atr",       "ATR"),
                ("structure", "구조"),
                ("atr_filter","ATR필터"),
                ("conf_bars", "확인봉수"),
                ("regime",    "레짐"),
                ("mult",      "배율"),
                ("er",        "ER"),
                ("atr_pct",   "ATR%"),
                ("density",   "밀도"),
            ]):
                pg.addWidget(self._make_lbl(parent, label), row, 0)
                v = QLabel("—", parent); v.setStyleSheet(self._VAL)
                v.setAlignment(Qt.AlignRight)
                pg.addWidget(v, row, 1)
                self._param_lbls[key] = v
            col3.addLayout(pg)
            col3.addStretch()
            hl.addLayout(col3, stretch=2)
            hl.addWidget(self._make_sep(parent))

            # 섹션 4: 진단 + 히스토리 바 ─────────────────────────────
            col4 = QVBoxLayout(); col4.setSpacing(2)
            col4.addWidget(self._make_hdr(parent, "💡 진단"))
            self._suggest_lbl = QLabel("—", parent)
            self._suggest_lbl.setStyleSheet(
                "font-size: 9px; color: #ce9178; font-family: 'Consolas', monospace;"
            )
            self._suggest_lbl.setWordWrap(True)
            col4.addWidget(self._suggest_lbl)

            # [Fix-1] _BarWidget(QWidget)을 직접 생성 후 addWidget에 전달
            self._bar_widget = self._make_bar_widget(parent)
            if self._bar_widget is not None:
                self._bar_widget.setFixedHeight(40)
                col4.addWidget(self._bar_widget)

            col4.addStretch()
            hl.addLayout(col4, stretch=3)

            root.addWidget(self._panel)
            logger.info("[PivotQualityMonitor] 패널 빌드 완료")

        except Exception as e:
            logger.warning("[PivotQualityMonitor] build 실패: %s", e, exc_info=True)

    # ── 업데이트 ─────────────────────────────────────────────────────

    def update(self, zz: Any, cfg: Any = None) -> None:
        """ZigZag 인스턴스에서 메트릭 계산 후 위젯 갱신.

        Args:
            zz:  AdaptiveZigZag 인스턴스
            cfg: AdaptiveZigZagConfig 인스턴스 (None 가능)
        """
        if self._panel is None:
            return
        try:
            m = self._analyzer.compute(zz, cfg)
            self._last_metrics = m
            self._update_widgets(m, cfg)
            self._update_bar_widget(zz)
        except Exception as e:
            logger.warning("[PivotQualityMonitor] update 실패: %s", e)

    def get_metrics(self) -> Optional[PivotQualityMetrics]:
        return self._last_metrics

    # ── 위젯 업데이트 (내부) ─────────────────────────────────────────

    def _update_widgets(self, m: PivotQualityMetrics, cfg: Any) -> None:
        if m.total_confirmed < 2:
            return

        # 등급
        self._set_text("grade", self._grade_lbl, m.grade)
        self._set_style(
            "grade",
            self._grade_lbl,
            f"font-size: 28px; font-weight: bold; color: {m.grade_color};"
            "font-family: 'Consolas', monospace;",
        )

        # 카운터
        self._ml_text("confirmed",  str(m.total_confirmed))
        self._ml_text("registered", str(m.total_registered))
        self._ml_text("cancelled",  str(m.total_cancelled))

        # 취소율
        cr_col = ("#4ec9b0" if m.cancel_rate < 0.25
                  else "#dcdcaa" if m.cancel_rate < 0.40 else "#f48771")
        self._ml_text ("cancel_rate", f"{m.cancel_rate:.0%}")
        self._ml_style("cancel_rate",
            f"font-size:11px;font-weight:bold;color:{cr_col};"
            "font-family:'Consolas',monospace;")

        # 밀도
        dn_col = ("#4ec9b0" if 4 <= m.pivots_per_hour <= 20
                  else "#dcdcaa" if 2 <= m.pivots_per_hour <= 35 else "#f48771")
        self._ml_text ("density", f"{m.pivots_per_hour:.1f}")
        self._ml_style("density",
            f"font-size:11px;font-weight:bold;color:{dn_col};"
            "font-family:'Consolas',monospace;")

        self._ml_text("avg_lag",  f"{m.avg_lag_bars:.1f}봉")
        lag_col = "#f48771" if m.lag_critical else "#dcdcaa"
        self._ml_style("avg_lag", f"font-size:11px;font-weight:bold;color:{lag_col};font-family:'Consolas',monospace;")
        self._ml_text("max_lag",  f"{m.max_lag_bars}봉")
        acc_txt = (f"{m.accuracy_score:.0%}"
                   if m.accuracy_sample >= 3 else "—(샘플부족)")
        self._ml_text("accuracy", acc_txt)
        self._ml_text("avg_wave", f"{m.avg_wave_pct:.2f}%")

        # 파라미터
        self._pl_text("threshold", f"{m.current_threshold_pct:.3f}%")
        self._pl_text("atr",       f"{m.current_atr:.3f}")
        self._pl_text("structure", m.current_structure[:8])

        # 적응형 엔진 조정 상태
        if m.adaptive_regime != "unknown":
            self._pl_text("regime", m.adaptive_regime[:12])
            self._pl_text("mult", f"{m.adaptive_mult:.2f}x")
            self._pl_text("er", f"{m.adaptive_er:.2f}")
            self._pl_text("atr_pct", f"{m.adaptive_atr_pct:.0f}%")
            self._pl_text("density", m.adaptive_density[:6])

        if cfg is not None:
            use_atr = getattr(cfg, "use_atr_based_filtering", False)
            atr_col = "#4ec9b0" if use_atr else "#f48771"
            self._pl_text ("atr_filter", "ON" if use_atr else "OFF")
            self._pl_style("atr_filter",
                f"font-size:11px;font-weight:bold;color:{atr_col};"
                "font-family:'Consolas',monospace;")
            self._pl_text("conf_bars", str(getattr(cfg, "confirmation_bars", "?")))

        # 진단
        if self._suggest_lbl is not None:
            sug_col = {"ok": "#4ec9b0", "warn": "#ce9178", "info": "#888888"}.get(
                m.suggestion_level, "#888888"
            )
            self._set_text("suggest",  self._suggest_lbl, m.suggestion)
            self._set_style("suggest", self._suggest_lbl,
                f"font-size:9px;color:{sug_col};"
                "font-family:'Consolas',monospace;")

    def _update_bar_widget(self, zz: Any) -> None:
        """시간대별 피봇 분포 히스토리 업데이트."""
        if self._bar_widget is None or zz is None:
            return
        try:
            bar_idx = int(getattr(zz, "_bar_idx", 0))
            confirmed = [s for s in list(getattr(zz, "_all_swings", []) or [])
                         if getattr(s, "confirmed", False)]
            bins = [0] * 7
            for s in confirmed:
                c_at = getattr(s, "confirmed_at_idx", -1)
                if c_at < 0:
                    continue
                age_h = (bar_idx - c_at) // 60
                if 0 <= age_h < 7:
                    bins[age_h] += 1
            self._bar_widget.set_bins(list(reversed(bins)))
        except Exception as e:
            logger.debug("[PivotQualityMonitor] bar_widget 업데이트 실패: %s", e)

    # ── 캐시 기반 Qt 호출 헬퍼 ──────────────────────────────────────

    def _set_text(self, key: str, widget: Any, text: str) -> None:
        if widget is not None and self._cached_texts.get(key) != text:
            widget.setText(text)
            self._cached_texts[key] = text

    def _set_style(self, key: str, widget: Any, style: str) -> None:
        if widget is not None and self._cached_styles.get(key) != style:
            widget.setStyleSheet(style)
            self._cached_styles[key] = style

    def _ml_text(self, key: str, text: str) -> None:
        self._set_text(key, self._metric_lbls.get(key), text)

    def _ml_style(self, key: str, style: str) -> None:
        self._set_style(key, self._metric_lbls.get(key), style)

    def _pl_text(self, key: str, text: str) -> None:
        self._set_text(key, self._param_lbls.get(key), text)

    def _pl_style(self, key: str, style: str) -> None:
        self._set_style(key, self._param_lbls.get(key), style)

    # ── Qt 위젯 팩토리 ───────────────────────────────────────────────

    @staticmethod
    def _make_hdr(parent: Any, text: str) -> Any:
        from PySide6.QtWidgets import QLabel
        lbl = QLabel(text, parent)
        lbl.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #9cdcfe;"
        )
        return lbl

    @staticmethod
    def _make_lbl(parent: Any, text: str) -> Any:
        from PySide6.QtWidgets import QLabel
        lbl = QLabel(text, parent)
        lbl.setStyleSheet(
            "font-size: 10px; color: #888; font-family: 'Consolas', monospace;"
        )
        return lbl

    @staticmethod
    def _make_sep(parent: Any) -> Any:
        from PySide6.QtWidgets import QFrame
        sep = QFrame(parent)
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #2d2d4e;")
        return sep

    @staticmethod
    def _make_bar_widget(parent: Any) -> Optional[Any]:
        """[Fix-1] _BarWidget(QWidget 직접 상속)을 반환.

        QWidget이 아닌 프록시 객체를 addWidget에 전달하던 Bug-1 수정.
        """
        try:
            from PySide6.QtWidgets import QWidget
            from PySide6.QtGui import QPainter, QColor, QFont, QPen
            from PySide6.QtCore import Qt

            class _BarWidget(QWidget):
                def __init__(self, p: Any) -> None:
                    super().__init__(p)
                    self._bins: List[int] = [0] * 7

                def set_bins(self, bins: List[int]) -> None:
                    self._bins = bins[:7]
                    self.update()

                def paintEvent(self, event: Any) -> None:  # noqa: N802
                    painter = QPainter(self)
                    painter.setRenderHint(QPainter.Antialiasing)
                    w, h = self.width(), self.height()
                    n    = len(self._bins)
                    mx   = max(self._bins) if self._bins else 0
                    if n == 0 or mx == 0:
                        return

                    bar_w = max(1, (w - 4) // n - 1)
                    font  = QFont("Consolas", 7)
                    painter.setFont(font)

                    for i, cnt in enumerate(self._bins):
                        bar_h = int((cnt / mx) * (h - 14))
                        x = 2 + i * (bar_w + 1)
                        y = h - 12 - bar_h
                        alpha = 80 + int(175 * (i / max(n - 1, 1)))
                        color = QColor("#569cd6")
                        color.setAlpha(alpha)
                        painter.fillRect(x, y, bar_w, bar_h, color)
                        # 레이블: 왼쪽 '6h', 오른쪽 '0h'
                        label = f"{n-1-i}h" if i == 0 else ("0h" if i == n - 1 else "")
                        if label:
                            painter.setPen(QColor("#888888"))
                            # [Fix] y 좌표: h-2 (경계 이내)
                            painter.drawText(x, h - 2, label)

            return _BarWidget(parent)

        except Exception as e:
            logger.debug("[PivotQualityMonitor] _BarWidget 생성 실패: %s", e)
            return None
