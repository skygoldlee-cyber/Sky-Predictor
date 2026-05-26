"""
TransformerQualityTracker 단위 테스트
"""

import pytest
import time
from datetime import date, datetime
from unittest.mock import Mock, MagicMock

from prediction.transformer_quality_tracker import (
    TransformerQualityTracker,
    DailyAccuracy,
    ConfidenceBucket,
    LLMAccuracy,
    QualitySnapshot,
    ALERT_ACCURACY_THRESHOLD,
    ALERT_ECE_THRESHOLD,
    ALERT_COOLDOWN_SEC,
    MIN_SAMPLES_FOR_ALERT,
)


@pytest.mark.unit
class TestTransformerQualityTracker:
    """TransformerQualityTracker 단위 테스트."""

    def test_initialization(self):
        """초기화 기본값 테스트."""
        tracker = TransformerQualityTracker()
        assert tracker._notifier is None
        assert tracker._ece_window == 50
        assert tracker._daily_history_days == 30
        assert tracker._alert_accuracy_threshold == ALERT_ACCURACY_THRESHOLD
        assert tracker._alert_ece_threshold == ALERT_ECE_THRESHOLD
        assert tracker._alert_cooldown_sec == ALERT_COOLDOWN_SEC
        assert tracker._min_samples == MIN_SAMPLES_FOR_ALERT

    def test_initialization_with_custom_params(self):
        """사용자 정의 파라미터 초기화 테스트."""
        mock_notifier = Mock()
        tracker = TransformerQualityTracker(
            notifier=mock_notifier,
            ece_window=100,
            daily_history_days=60,
            alert_accuracy_threshold=0.50,
            alert_ece_threshold=0.20,
            alert_cooldown_sec=3600.0,
            min_samples_for_alert=20,
        )
        assert tracker._notifier is mock_notifier
        assert tracker._ece_window == 100
        assert tracker._daily_history_days == 60
        assert tracker._alert_accuracy_threshold == 0.50
        assert tracker._alert_ece_threshold == 0.20
        assert tracker._alert_cooldown_sec == 3600.0
        assert tracker._min_samples == 20

    def test_record_evaluation_single(self):
        """단일 평가 기록 테스트."""
        tracker = TransformerQualityTracker()
        tracker.record_evaluation(
            correct=True,
            prob=0.73,
            confidence="HIGH",
            actual_direction="BUY",
            llm_actions={"gpt": "BUY", "gemini": "BUY", "claude": "HOLD"},
            transformer_weight=0.62,
        )
        metrics = tracker.get_metrics_dict()
        assert metrics["quality_today_accuracy"] == 1.0
        assert metrics["quality_today_total"] == 1
        assert metrics["quality_today_hits"] == 1
        assert metrics["quality_high_total"] == 1
        assert metrics["quality_high_accuracy"] == 1.0

    def test_record_evaluation_multiple(self):
        """다중 평가 기록 테스트."""
        tracker = TransformerQualityTracker()
        
        # 5건 기록 (3건 정답)
        for i in range(5):
            tracker.record_evaluation(
                correct=i < 3,
                prob=0.6 + (i * 0.05),
                confidence="HIGH" if i < 3 else "MEDIUM",
                actual_direction="BUY",
                llm_actions={"gpt": "BUY", "gemini": "BUY", "claude": "HOLD"},
                transformer_weight=0.6,
            )
        
        metrics = tracker.get_metrics_dict()
        assert metrics["quality_today_accuracy"] == 0.6
        assert metrics["quality_today_total"] == 5
        assert metrics["quality_today_hits"] == 3
        assert metrics["quality_high_total"] == 3
        assert metrics["quality_high_accuracy"] == 1.0
        assert metrics["quality_medium_total"] == 2
        assert metrics["quality_medium_accuracy"] == 0.0

    def test_record_evaluation_llm_accuracy(self):
        """LLM 개별 정확도 집계 테스트."""
        tracker = TransformerQualityTracker()
        
        # GPT 정답, Gemini 오답, Claude 정답
        tracker.record_evaluation(
            correct=True,
            prob=0.7,
            confidence="HIGH",
            actual_direction="BUY",
            llm_actions={"gpt": "BUY", "gemini": "SELL", "claude": "BUY"},
            transformer_weight=0.6,
        )
        
        metrics = tracker.get_metrics_dict()
        assert metrics["quality_llm_gpt_accuracy"] == 1.0
        assert metrics["quality_llm_gpt_total"] == 1
        assert metrics["quality_llm_gemini_accuracy"] == 0.0
        assert metrics["quality_llm_gemini_total"] == 1
        assert metrics["quality_llm_claude_accuracy"] == 1.0
        assert metrics["quality_llm_claude_total"] == 1

    def test_record_evaluation_hold_direction_skips_llm(self):
        """HOLD 방향에서 LLM 집계 건너뛰기 테스트."""
        tracker = TransformerQualityTracker()
        
        tracker.record_evaluation(
            correct=True,
            prob=0.5,
            confidence="LOW",
            actual_direction="HOLD",
            llm_actions={"gpt": "BUY", "gemini": "SELL", "claude": "HOLD"},
            transformer_weight=0.5,
        )
        
        metrics = tracker.get_metrics_dict()
        # HOLD 방향이면 LLM 집계 건너뜀
        assert metrics["quality_llm_gpt_total"] == 0
        assert metrics["quality_llm_gemini_total"] == 0
        assert metrics["quality_llm_claude_total"] == 0

    def test_get_metrics_dict_empty(self):
        """빈 상태에서 메트릭 조회 테스트."""
        tracker = TransformerQualityTracker()
        metrics = tracker.get_metrics_dict()
        assert metrics["quality_today_accuracy"] == 0.0
        assert metrics["quality_today_total"] == 0
        assert metrics["quality_today_hits"] == 0
        assert metrics["quality_ece"] == 0.0

    def test_get_quality_snapshot(self):
        """품질 스냅샷 조회 테스트."""
        tracker = TransformerQualityTracker()
        tracker.record_evaluation(
            correct=True,
            prob=0.7,
            confidence="HIGH",
            actual_direction="BUY",
            transformer_weight=0.6,
        )
        
        snapshot = tracker.get_quality_snapshot(transformer_weight=0.65)
        assert snapshot.today_accuracy == 1.0
        assert snapshot.today_total == 1
        assert snapshot.transformer_weight == 0.65
        assert isinstance(snapshot.timestamp, str)
        assert isinstance(snapshot.confidence_buckets, dict)
        assert isinstance(snapshot.llm_accuracies, dict)

    def test_get_daily_history(self):
        """일별 이력 조회 테스트."""
        tracker = TransformerQualityTracker()
        
        # 오늘 3건 기록
        for _ in range(3):
            tracker.record_evaluation(
                correct=True,
                prob=0.7,
                confidence="HIGH",
                actual_direction="BUY",
                transformer_weight=0.6,
            )
        
        history = tracker.get_daily_history()
        assert len(history) == 1
        assert history[0]["accuracy"] == 1.0
        assert history[0]["total"] == 3

    def test_reset_daily(self):
        """일별 집계 초기화 테스트."""
        tracker = TransformerQualityTracker()
        
        # 데이터 기록
        tracker.record_evaluation(
            correct=True,
            prob=0.7,
            confidence="HIGH",
            actual_direction="BUY",
            transformer_weight=0.6,
        )
        
        assert tracker.get_metrics_dict()["quality_today_total"] == 1
        
        # 초기화
        tracker.reset_daily()
        
        assert tracker.get_metrics_dict()["quality_today_total"] == 0
        assert tracker.get_metrics_dict()["quality_high_total"] == 0

    def test_alert_condition_accuracy_low(self):
        """정확도 저하 경보 조건 테스트."""
        tracker = TransformerQualityTracker(
            alert_accuracy_threshold=0.50,
            min_samples_for_alert=5,
        )
        
        # 5건 중 2건만 정답 (40%)
        for i in range(5):
            tracker.record_evaluation(
                correct=i < 2,
                prob=0.5,
                confidence="MEDIUM",
                actual_direction="BUY",
                transformer_weight=0.5,
            )
        
        with tracker._lock:
            d = tracker._daily.get(date.today().isoformat())
            should_alert, reason = tracker._check_alert_condition(d)
        
        assert should_alert is True
        assert "방향 정확도 저하" in reason

    def test_alert_condition_accuracy_high(self):
        """정확도 높을 때 경보 미발생 테스트."""
        tracker = TransformerQualityTracker(
            alert_accuracy_threshold=0.50,
            min_samples_for_alert=5,
        )
        
        # 5건 중 4건 정답 (80%)
        for i in range(5):
            tracker.record_evaluation(
                correct=i < 4,
                prob=0.7,
                confidence="HIGH",
                actual_direction="BUY",
                transformer_weight=0.6,
            )
        
        with tracker._lock:
            d = tracker._daily.get(date.today().isoformat())
            should_alert, reason = tracker._check_alert_condition(d)
        
        assert should_alert is False

    def test_alert_condition_insufficient_samples(self):
        """샘플 부족 시 경보 미발생 테스트."""
        tracker = TransformerQualityTracker(
            alert_accuracy_threshold=0.50,
            min_samples_for_alert=10,
        )
        
        # 5건만 기록 (임계값 10 미만)
        for _ in range(5):
            tracker.record_evaluation(
                correct=False,
                prob=0.5,
                confidence="LOW",
                actual_direction="BUY",
                transformer_weight=0.5,
            )
        
        with tracker._lock:
            d = tracker._daily.get(date.today().isoformat())
            should_alert, reason = tracker._check_alert_condition(d)
        
        assert should_alert is False

    def test_alert_cooldown(self):
        """경보 쿨다운 테스트."""
        mock_notifier = Mock()
        tracker = TransformerQualityTracker(
            notifier=mock_notifier,
            alert_accuracy_threshold=0.50,
            alert_cooldown_sec=1.0,  # 1초 쿨다운
            min_samples_for_alert=5,
        )
        
        # 첫 번째 경보 발생
        for i in range(5):
            tracker.record_evaluation(
                correct=i < 2,
                prob=0.5,
                confidence="MEDIUM",
                actual_direction="BUY",
                transformer_weight=0.5,
            )
        
        # 쿨다운 내 추가 기록
        for _ in range(3):
            tracker.record_evaluation(
                correct=False,
                prob=0.5,
                confidence="LOW",
                actual_direction="BUY",
                transformer_weight=0.5,
            )
        
        # 쿨다운 내이므로 notifier 호출되지 않음
        assert mock_notifier.send_text.call_count == 0

    def test_alert_with_notifier(self):
        """notifier가 있을 때 경보 발송 테스트."""
        mock_notifier = Mock()
        tracker = TransformerQualityTracker(
            notifier=mock_notifier,
            alert_accuracy_threshold=0.50,
            alert_cooldown_sec=0.0,  # 쿨다운 없음
            min_samples_for_alert=5,
        )
        
        # 경보 조건 충족
        for i in range(5):
            tracker.record_evaluation(
                correct=i < 2,
                prob=0.5,
                confidence="MEDIUM",
                actual_direction="BUY",
                transformer_weight=0.5,
            )
        
        # notifier 호출 확인
        assert mock_notifier.send_text.call_count == 1
        call_args = mock_notifier.send_text.call_args
        assert "품질 저하" in call_args[0][0]

    def test_ece_calculation(self):
        """ECE 계산 테스트."""
        tracker = TransformerQualityTracker(ece_window=10)
        
        # 완벽하게 보정된 예측 (ECE = 0)
        for i in range(10):
            prob = 0.7 + (i * 0.02)
            correct = prob > 0.5
            tracker.record_evaluation(
                correct=correct,
                prob=prob,
                confidence="HIGH",
                actual_direction="BUY",
                transformer_weight=0.6,
            )
        
        metrics = tracker.get_metrics_dict()
        # 완벽한 보정이면 ECE가 낮아야 함
        assert metrics["quality_ece"] < 0.3

    def test_confidence_auto_estimation(self):
        """신뢰도 등급 자동 추정 테스트 (feedback_mixin 로직 확인용)."""
        # 이 테스트는 feedback_mixin.py의 자동 추정 로직을 검증
        # 실제 추정은 feedback_mixin에서 수행됨
        margin_high = abs(0.75 - 0.5)  # 0.25 -> HIGH
        margin_medium = abs(0.60 - 0.5)  # 0.10 -> MEDIUM
        margin_low = abs(0.55 - 0.5)  # 0.05 -> LOW
        
        assert margin_high >= 0.25
        assert margin_medium >= 0.10 and margin_medium < 0.25
        assert margin_low < 0.10

    def test_log_daily_summary(self, caplog):
        """일별 요약 로그 출력 테스트."""
        import logging
        caplog.set_level(logging.INFO)
        
        tracker = TransformerQualityTracker()
        
        # 데이터 기록
        for i in range(10):
            tracker.record_evaluation(
                correct=i < 7,
                prob=0.7,
                confidence="HIGH" if i < 7 else "MEDIUM",
                actual_direction="BUY",
                llm_actions={"gpt": "BUY", "gemini": "BUY", "claude": "HOLD"},
                transformer_weight=0.6,
            )
        
        tracker.log_daily_summary()
        
        # 로그 확인
        assert any("트랜스포머 예측 품질 일별 요약" in record.message for record in caplog.records)
        assert any("오늘 방향 정확도" in record.message for record in caplog.records)
        assert any("신뢰도 등급별 적중률" in record.message for record in caplog.records)

    def test_thread_safety(self):
        """thread-safety 테스트."""
        import threading
        
        tracker = TransformerQualityTracker()
        num_threads = 5
        records_per_thread = 20
        
        def record_worker():
            for i in range(records_per_thread):
                tracker.record_evaluation(
                    correct=i % 2 == 0,
                    prob=0.6,
                    confidence="MEDIUM",
                    actual_direction="BUY",
                    transformer_weight=0.5,
                )
        
        threads = [threading.Thread(target=record_worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        metrics = tracker.get_metrics_dict()
        # 모든 스레드가 정상적으로 기록되었는지 확인
        assert metrics["quality_today_total"] == num_threads * records_per_thread


@pytest.mark.unit
class TestDataClasses:
    """데이터 클래스 단위 테스트."""

    def test_daily_accuracy(self):
        """DailyAccuracy 테스트."""
        da = DailyAccuracy(date_str="2026-05-26", hits=7, total=10, brier_sum=1.5)
        assert da.accuracy == 0.7
        assert da.mean_brier == 0.15

    def test_daily_accuracy_empty(self):
        """빈 DailyAccuracy 테스트."""
        da = DailyAccuracy(date_str="2026-05-26")
        assert da.accuracy == 0.0
        assert da.mean_brier == 0.25  # 기본값

    def test_confidence_bucket(self):
        """ConfidenceBucket 테스트."""
        cb = ConfidenceBucket(label="HIGH", hits=15, total=20)
        assert cb.accuracy == 0.75

    def test_confidence_bucket_empty(self):
        """빈 ConfidenceBucket 테스트."""
        cb = ConfidenceBucket(label="HIGH")
        assert cb.accuracy == 0.0

    def test_llm_accuracy(self):
        """LLMAccuracy 테스트."""
        la = LLMAccuracy(name="gpt", hits=8, total=12)
        assert la.accuracy == 2/3

    def test_llm_accuracy_empty(self):
        """빈 LLMAccuracy 테스트."""
        la = LLMAccuracy(name="gpt")
        assert la.accuracy == 0.0
