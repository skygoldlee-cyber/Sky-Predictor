"""prediction/transformer_quality_tracker.py
=============================================
트랜스포머 예측 품질을 장중 실시간으로 추적하고
품질 저하 시 Telegram 알림을 발생시키는 모듈.

추가된 측정 항목
----------------
1. 일별 방향 정확도 이력  — 날짜별 hits/total 기록
2. ECE (Expected Calibration Error) 장중 측정
3. LLM 개별 정확도       — GPT·Gemini·Claude 별도 집계
4. 신뢰도 등급별 적중률  — HIGH/MEDIUM/LOW 실제 적중 여부
5. 품질 저하 자동 감지    — 임계값 이하 시 Telegram 알림

파이프라인 연결 지점
--------------------
prediction/mixins/feedback_mixin.py  _maybe_process_feedback()
prediction/pipeline.py               get_metrics(), __init__()

Telegram 알림
-------------
품질이 ALERT_ACCURACY_THRESHOLD 이하로 떨어지면
TelegramNotifier.send_text()를 통해 알림 전송.
ALERT_COOLDOWN_SEC 이내 중복 알림 억제.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── 판정 임계값 ──────────────────────────────────────────────────────────────

ALERT_ACCURACY_THRESHOLD = 0.45   # 방향 정확도 이 이하면 알림
ALERT_ECE_THRESHOLD       = 0.15   # ECE 이 이상이면 과신 경고
ALERT_COOLDOWN_SEC        = 1800.0 # 알림 최소 간격 (30분)
MIN_SAMPLES_FOR_ALERT     = 10     # 최소 평가 건수 미만이면 알림 억제


# ── 데이터 클래스 ─────────────────────────────────────────────────────────────

@dataclass
class DailyAccuracy:
    """하루치 방향 예측 정확도."""
    date_str:  str
    hits:      int = 0
    total:     int = 0
    brier_sum: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0

    @property
    def mean_brier(self) -> float:
        return self.brier_sum / self.total if self.total > 0 else 0.25


@dataclass
class ConfidenceBucket:
    """HIGH / MEDIUM / LOW 신뢰도 등급별 집계."""
    label: str
    hits:  int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0


@dataclass
class LLMAccuracy:
    """LLM 개별(GPT·Gemini·Claude) 방향 정확도."""
    name:  str
    hits:  int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0


@dataclass
class QualitySnapshot:
    """get_quality_snapshot() 반환 구조."""
    timestamp:          str
    today_accuracy:     float
    today_total:        int
    today_brier:        float
    ece_latest:         float
    transformer_weight: float
    confidence_buckets: Dict[str, Dict[str, Any]]
    llm_accuracies:     Dict[str, Dict[str, Any]]
    alert_fired:        bool
    alert_reason:       str


# ── 핵심 클래스 ───────────────────────────────────────────────────────────────

class TransformerQualityTracker:
    """트랜스포머 예측 품질 실시간 추적기.

    thread-safe: 모든 상태는 _lock 하에 접근.

    사용 예시
    ---------
    tracker = TransformerQualityTracker(notifier=tg_notifier)

    # 피드백 평가 완료 시 (feedback_mixin에서 호출)
    tracker.record_evaluation(
        correct=True,
        prob=0.73,
        confidence="HIGH",
        llm_actions={"gpt": "BUY", "gemini": "BUY", "claude": "HOLD"},
        actual_direction="BUY",
    )

    # get_metrics() 에 병합
    metrics = tracker.get_metrics_dict()

    # 장마감 후 일별 이력 출력
    tracker.log_daily_summary()
    """

    def __init__(
        self,
        notifier: Any = None,                   # TelegramNotifier (선택)
        ece_window: int = 50,                   # ECE 슬라이딩 윈도우 크기
        daily_history_days: int = 30,           # 일별 이력 보관 일수
        alert_accuracy_threshold: float = ALERT_ACCURACY_THRESHOLD,
        alert_ece_threshold: float = ALERT_ECE_THRESHOLD,
        alert_cooldown_sec: float = ALERT_COOLDOWN_SEC,
        min_samples_for_alert: int = MIN_SAMPLES_FOR_ALERT,
    ) -> None:
        self._notifier = notifier
        self._ece_window = max(10, int(ece_window))
        self._daily_history_days = max(7, int(daily_history_days))
        self._alert_accuracy_threshold = float(alert_accuracy_threshold)
        self._alert_ece_threshold = float(alert_ece_threshold)
        self._alert_cooldown_sec = float(alert_cooldown_sec)
        self._min_samples = max(1, int(min_samples_for_alert))

        self._lock = threading.Lock()

        # ── 일별 이력 ──────────────────────────────────────────────────────
        self._daily: Dict[str, DailyAccuracy] = {}

        # ── ECE 슬라이딩 윈도우 ────────────────────────────────────────────
        self._ece_probs:  deque[float] = deque(maxlen=self._ece_window)
        self._ece_labels: deque[float] = deque(maxlen=self._ece_window)

        # ── 신뢰도 등급별 집계 ─────────────────────────────────────────────
        self._confidence_buckets: Dict[str, ConfidenceBucket] = {
            "HIGH":   ConfidenceBucket("HIGH"),
            "MEDIUM": ConfidenceBucket("MEDIUM"),
            "LOW":    ConfidenceBucket("LOW"),
        }

        # ── LLM 개별 집계 ─────────────────────────────────────────────────
        self._llm_accuracy: Dict[str, LLMAccuracy] = {
            "gpt":    LLMAccuracy("gpt"),
            "gemini": LLMAccuracy("gemini"),
            "claude": LLMAccuracy("claude"),
        }

        # ── 알림 쿨다운 ────────────────────────────────────────────────────
        self._last_alert_epoch: float = 0.0
        self._last_alert_reason: str  = ""

        logger.info("[QualityTracker] 초기화 완료 (ece_window=%d)", self._ece_window)

    # ── 공개 API: 평가 기록 ──────────────────────────────────────────────────

    def record_evaluation(
        self,
        *,
        correct: bool,
        prob: float,
        confidence: str,
        actual_direction: str,                     # "BUY" | "SELL" | "HOLD"
        llm_actions: Optional[Dict[str, str]] = None,  # {"gpt": "BUY", ...}
        transformer_weight: float = 0.5,
    ) -> None:
        """피드백 평가 1건을 기록한다.

        Parameters
        ----------
        correct:
            Transformer 예측이 실제 방향과 일치했는가.
        prob:
            Transformer 출력 확률 (0~1, 0.5 초과 = BUY 예측).
        confidence:
            신뢰도 등급 문자열 ("HIGH" / "MEDIUM" / "LOW").
        actual_direction:
            실제 시장 방향 ("BUY" / "SELL" / "HOLD").
        llm_actions:
            LLM별 예측 action. 키: "gpt", "gemini", "claude".
        transformer_weight:
            현재 앙상블 Transformer 가중치 (0~1).
        """
        try:
            p = float(np.clip(prob, 0.0, 1.0))
            label = 1.0 if bool(correct) else 0.0
            brier = float((p - label) ** 2)
            today_str = date.today().isoformat()

            with self._lock:
                # 1. 일별 정확도 이력
                if today_str not in self._daily:
                    self._daily[today_str] = DailyAccuracy(today_str)
                    self._prune_old_daily()
                d = self._daily[today_str]
                d.total += 1
                if bool(correct):
                    d.hits += 1
                d.brier_sum += brier

                # 2. ECE 슬라이딩 윈도우
                self._ece_probs.append(p)
                self._ece_labels.append(label)

                # 3. 신뢰도 등급별 집계
                conf_key = str(confidence).upper()
                if conf_key in self._confidence_buckets:
                    bucket = self._confidence_buckets[conf_key]
                    bucket.total += 1
                    if bool(correct):
                        bucket.hits += 1

                # 4. LLM 개별 집계
                if llm_actions and actual_direction in ("BUY", "SELL"):
                    for llm_name, llm_act in llm_actions.items():
                        key = str(llm_name).lower()
                        if key in self._llm_accuracy:
                            acc = self._llm_accuracy[key]
                            acc.total += 1
                            if str(llm_act).upper() == actual_direction:
                                acc.hits += 1

                # 5. 품질 저하 자동 감지
                self._maybe_fire_alert(d, transformer_weight)

        except Exception as e:
            logger.debug("[QualityTracker] record_evaluation 오류(무시): %s", e)

    # ── 공개 API: 메트릭 조회 ────────────────────────────────────────────────

    def get_metrics_dict(self) -> Dict[str, Any]:
        """get_metrics()에 병합할 딕셔너리를 반환한다."""
        try:
            with self._lock:
                today_str = date.today().isoformat()
                d = self._daily.get(today_str, DailyAccuracy(today_str))
                ece = self._calc_ece()

                out: Dict[str, Any] = {
                    # 오늘 일별 정확도
                    "quality_today_accuracy":   round(d.accuracy, 4),
                    "quality_today_total":      d.total,
                    "quality_today_hits":       d.hits,
                    "quality_today_brier":      round(d.mean_brier, 4),
                    # ECE
                    "quality_ece":              round(ece, 4),
                    # 신뢰도 등급별
                    "quality_high_accuracy":    round(self._confidence_buckets["HIGH"].accuracy, 4),
                    "quality_high_total":       self._confidence_buckets["HIGH"].total,
                    "quality_medium_accuracy":  round(self._confidence_buckets["MEDIUM"].accuracy, 4),
                    "quality_medium_total":     self._confidence_buckets["MEDIUM"].total,
                    "quality_low_accuracy":     round(self._confidence_buckets["LOW"].accuracy, 4),
                    "quality_low_total":        self._confidence_buckets["LOW"].total,
                    # LLM 개별
                    "quality_llm_gpt_accuracy":    round(self._llm_accuracy["gpt"].accuracy, 4),
                    "quality_llm_gpt_total":       self._llm_accuracy["gpt"].total,
                    "quality_llm_gemini_accuracy": round(self._llm_accuracy["gemini"].accuracy, 4),
                    "quality_llm_gemini_total":    self._llm_accuracy["gemini"].total,
                    "quality_llm_claude_accuracy": round(self._llm_accuracy["claude"].accuracy, 4),
                    "quality_llm_claude_total":    self._llm_accuracy["claude"].total,
                }
                return out
        except Exception as e:
            logger.debug("[QualityTracker] get_metrics_dict 오류(무시): %s", e)
            return {}

    def get_quality_snapshot(self, transformer_weight: float = 0.5) -> QualitySnapshot:
        """현재 품질 상태 전체 스냅샷을 반환한다."""
        with self._lock:
            today_str = date.today().isoformat()
            d = self._daily.get(today_str, DailyAccuracy(today_str))
            ece = self._calc_ece()

            conf_dict = {
                k: {"accuracy": round(v.accuracy, 4), "total": v.total}
                for k, v in self._confidence_buckets.items()
            }
            llm_dict = {
                k: {"accuracy": round(v.accuracy, 4), "total": v.total}
                for k, v in self._llm_accuracy.items()
            }

            alert, reason = self._check_alert_condition(d)

            return QualitySnapshot(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                today_accuracy=round(d.accuracy, 4),
                today_total=d.total,
                today_brier=round(d.mean_brier, 4),
                ece_latest=round(ece, 4),
                transformer_weight=round(float(transformer_weight), 4),
                confidence_buckets=conf_dict,
                llm_accuracies=llm_dict,
                alert_fired=alert,
                alert_reason=reason,
            )

    def get_daily_history(self) -> List[Dict[str, Any]]:
        """일별 정확도 이력 리스트 (오래된 것부터)."""
        with self._lock:
            return [
                {
                    "date":     d.date_str,
                    "accuracy": round(d.accuracy, 4),
                    "total":    d.total,
                    "brier":    round(d.mean_brier, 4),
                }
                for d in sorted(self._daily.values(), key=lambda x: x.date_str)
            ]

    def log_daily_summary(self) -> None:
        """장마감 후 일별 요약을 로그로 출력한다."""
        snap = self.get_quality_snapshot()
        lines = [
            "",
            "=" * 55,
            "  트랜스포머 예측 품질 일별 요약",
            "=" * 55,
            f"  오늘 방향 정확도:  {snap.today_accuracy:.1%}  ({snap.today_total}건)",
            f"  오늘 Brier Score:  {snap.today_brier:.4f}",
            f"  ECE (최근 {self._ece_window}건): {snap.ece_latest:.4f}",
            f"  Transformer 가중치: {snap.transformer_weight:.3f}",
            "-" * 55,
            "  신뢰도 등급별 적중률",
        ]
        for label, info in snap.confidence_buckets.items():
            lines.append(
                f"    {label:<8}: {info['accuracy']:.1%}  ({info['total']}건)"
            )
        lines.append("-" * 55)
        lines.append("  LLM 개별 방향 적중률")
        for name, info in snap.llm_accuracies.items():
            lines.append(
                f"    {name:<8}: {info['accuracy']:.1%}  ({info['total']}건)"
            )
        if snap.alert_fired:
            lines += ["-" * 55, f"  경보: {snap.alert_reason}"]
        lines += ["=" * 55, ""]
        for line in lines:
            logger.info(line)

    def reset_daily(self) -> None:
        """일별 집계를 초기화한다 (장 시작 시 호출 가능)."""
        today_str = date.today().isoformat()
        with self._lock:
            self._daily.pop(today_str, None)
            self._ece_probs.clear()
            self._ece_labels.clear()
            for b in self._confidence_buckets.values():
                b.hits = 0
                b.total = 0
            for a in self._llm_accuracy.values():
                a.hits = 0
                a.total = 0
        logger.info("[QualityTracker] 일별 집계 초기화")

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

    def _calc_ece(self, n_bins: int = 10) -> float:
        """슬라이딩 윈도우에서 ECE 계산 (_lock 안에서 호출)."""
        probs  = list(self._ece_probs)
        labels = list(self._ece_labels)
        n = len(probs)
        if n < 5:
            return 0.0
        p = np.array(probs,  dtype=np.float64)
        y = np.array(labels, dtype=np.float64)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
            cnt = int(np.sum(mask))
            if cnt == 0:
                continue
            conf = float(np.mean(p[mask]))
            acc  = float(np.mean(y[mask]))
            ece += (cnt / n) * abs(acc - conf)
        return float(ece)

    def _check_alert_condition(
        self, d: DailyAccuracy
    ) -> Tuple[bool, str]:
        """알림 조건 확인 (_lock 안에서 호출). (should_alert, reason)."""
        if d.total < self._min_samples:
            return False, ""
        if d.accuracy < self._alert_accuracy_threshold:
            return True, (
                f"방향 정확도 저하: {d.accuracy:.1%} < {self._alert_accuracy_threshold:.1%}"
                f" ({d.total}건)"
            )
        ece = self._calc_ece()
        if ece > self._alert_ece_threshold:
            return True, (
                f"ECE 과신 경고: {ece:.4f} > {self._alert_ece_threshold:.4f}"
            )
        return False, ""

    def _maybe_fire_alert(
        self, d: DailyAccuracy, transformer_weight: float
    ) -> None:
        """품질 저하 알림 발생 (_lock 안에서 호출)."""
        should, reason = self._check_alert_condition(d)
        if not should:
            return

        import time as _time
        now = _time.time()
        if now - self._last_alert_epoch < self._alert_cooldown_sec:
            return

        self._last_alert_epoch = now
        self._last_alert_reason = reason

        # Telegram 알림 (notifier가 없으면 로그만)
        msg = self._format_alert_message(d, reason, transformer_weight)
        logger.warning("[QualityTracker] 품질 저하 감지: %s", reason)
        if self._notifier is not None:
            try:
                send_fn = getattr(self._notifier, "send_text", None)
                if callable(send_fn):
                    send_fn(msg, parse_mode="HTML")
            except Exception as e:
                logger.debug("[QualityTracker] Telegram 알림 실패(무시): %s", e)

    def _format_alert_message(
        self,
        d: DailyAccuracy,
        reason: str,
        transformer_weight: float,
    ) -> str:
        """Telegram HTML 알림 메시지 포매팅."""
        ece = self._calc_ece()
        conf_lines = "\n".join(
            f"  {k}: {v.accuracy:.1%} ({v.total}건)"
            for k, v in self._confidence_buckets.items()
            if v.total > 0
        )
        llm_lines = "\n".join(
            f"  {k}: {v.accuracy:.1%} ({v.total}건)"
            for k, v in self._llm_accuracy.items()
            if v.total > 0
        )
        return (
            f"<b>⚠ 트랜스포머 품질 저하 감지</b>\n\n"
            f"<b>경고:</b> {reason}\n\n"
            f"<b>오늘 현황</b>\n"
            f"  방향 정확도: <b>{d.accuracy:.1%}</b> ({d.total}건)\n"
            f"  Brier Score: {d.mean_brier:.4f}\n"
            f"  ECE: {ece:.4f}\n"
            f"  Transformer 가중치: {transformer_weight:.3f}\n\n"
            f"<b>신뢰도별 적중률</b>\n{conf_lines or '  (데이터 없음)'}\n\n"
            f"<b>LLM별 적중률</b>\n{llm_lines or '  (데이터 없음)'}"
        )

    def _prune_old_daily(self) -> None:
        """오래된 일별 이력 삭제 (_lock 안에서 호출)."""
        if len(self._daily) <= self._daily_history_days:
            return
        sorted_keys = sorted(self._daily.keys())
        for k in sorted_keys[: len(sorted_keys) - self._daily_history_days]:
            del self._daily[k]
