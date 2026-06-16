"""Pivot Candidate History Collector

후보 피봇의 등록부터 확정/취소까지의 히스토리를 수집하여
머신러닝 학습 데이터셋을 생성하는 수집기.

Usage:
    collector = PivotCandidateCollector()
    
    # 후보 등록 시
    collector.on_candidate_registered(
        candidate_id="high_142",
        features={...},
        candidate_type="high",
        candidate_price=370.25,
        bar_idx=142,
        timestamp="09:15",
    )
    
    # 매 봉마다 피처 업데이트
    collector.on_bar_update("high_142", features={...})
    
    # 확정 시
    collector.on_candidate_confirmed("high_142", confirmed_bar=144)
    
    # 취소 시
    collector.on_candidate_cancelled("high_142", cancelled_bar=143, reason="max_wait_bars")
    
    # 데이터셋 저장
    collector.save_dataset("pivot_candidates.pkl")
"""

import pickle
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import uuid

_logger = logging.getLogger(__name__)


@dataclass
class CandidateSnapshot:
    """후보 피봇의 특정 시점 스냅샷."""
    bar_idx: int
    timestamp: str
    features: Dict[str, float]
    close: float


@dataclass
class CandidateRecord:
    """후보 피봇 전체 기록."""
    candidate_id: str
    candidate_type: str  # "high" or "low"
    candidate_price: float
    
    # 등록 정보
    registered_bar: int
    registered_time: str
    registered_features: Dict[str, float]
    registered_close: float
    
    # 확정/취소 정보
    label: int  # 1=확정, 0=취소
    confirmed_bar: Optional[int] = None
    cancelled_bar: Optional[int] = None
    reason: Optional[str] = None
    
    # 수명
    lifespan_bars: int = 0
    
    # 시계열 히스토리 (매 봉마다의 피처 변화)
    sequence: List[CandidateSnapshot] = field(default_factory=list)
    
    # 추가 메타데이터
    symbol: str = "KP200 선물"
    date: str = ""  # YYYY-MM-DD


class PivotCandidateCollector:
    """후보 피봇 히스토리 수집기.
    
    AdaptiveZigZag의 피봇 이벤트를 훅하여 후보 등록부터 확정/취소까지의
    전체 히스토리를 수집합니다.
    """
    
    def __init__(self, max_sequence_length: int = 120, now_fn: Optional[Callable[[], datetime]] = None):
        """초기화.

        Args:
            max_sequence_length: 시계열 최대 길이 (봉수)
            now_fn: 시간 함수 (테스트/백테스트용 주입 가능)
        """
        self.max_sequence_length = max_sequence_length
        self.active_candidates: Dict[str, CandidateRecord] = {}
        self.completed_candidates: List[CandidateRecord] = []
        self._enabled = True
        self._callback = None  # 피봇 후보 알림 콜백
        self._last_alert_time: Dict[str, float] = {}  # candidate_id -> last_alert_time
        self._change_cooldown_sec = 60.0  # 변경 이벤트 쿨다운 (초)
        self._now_fn = now_fn if now_fn is not None else datetime.now
    
    def enable(self):
        """수집 활성화."""
        self._enabled = True
        _logger.info("[PivotCollector] 수집 활성화")
    
    def disable(self):
        """수집 비활성화."""
        self._enabled = False
        _logger.info("[PivotCollector] 수집 비활성화")
    
    def is_enabled(self) -> bool:
        """수집 활성화 여부."""
        return self._enabled

    def set_callback(self, callback, change_cooldown_sec: float = 60.0) -> None:
        """피봇 후보 알림 콜백 설정.

        Args:
            callback: 콜백 함수 (event_type, symbol, candidate_type, candidate_price, bar_idx, timestamp, reason)
            change_cooldown_sec: 변경 이벤트 쿨다운 (초)
        """
        self._callback = callback
        self._change_cooldown_sec = change_cooldown_sec
    
    def generate_candidate_id(self, candidate_type: str, bar_idx: int) -> str:
        """후보 ID 생성."""
        return f"{candidate_type}_{bar_idx}_{uuid.uuid4().hex[:8]}"
    
    def on_candidate_registered(
        self,
        candidate_id: str,
        candidate_type: str,
        candidate_price: float,
        bar_idx: int,
        timestamp: str,
        features: Dict[str, float],
        close: float,
        symbol: str = "KP200 선물",
    ) -> None:
        """후보 등록 시 호출.
        
        Args:
            candidate_id: 후보 고유 ID
            candidate_type: "high" or "low"
            candidate_price: 후보 가격
            bar_idx: 등록 봉 인덱스
            timestamp: 시각 "HH:MM"
            features: 등록 시점의 피처 딕셔너리
            close: 등록 시점의 종가
            symbol: 심볼명
        """
        if not self._enabled:
            return
        
        # 날짜 추출 (timestamp에서)
        date_str = self._now_fn().strftime("%Y-%m-%d")
        
        record = CandidateRecord(
            candidate_id=candidate_id,
            candidate_type=candidate_type,
            candidate_price=candidate_price,
            registered_bar=bar_idx,
            registered_time=timestamp,
            registered_features=features.copy(),
            registered_close=close,
            label=0,  # 초기값: 미확정
            symbol=symbol,
            date=date_str,
            sequence=[],
        )
        
        # 초기 스냅샷 추가
        record.sequence.append(CandidateSnapshot(
            bar_idx=bar_idx,
            timestamp=timestamp,
            features=features.copy(),
            close=close,
        ))
        
        self.active_candidates[candidate_id] = record
        _logger.debug(
            "[PivotCollector] 후보 등록: %s type=%s price=%.2f bar=%d",
            candidate_id, candidate_type, candidate_price, bar_idx
        )

        # 콜백 호출 (등록 이벤트)
        if self._callback:
            try:
                _logger.info(
                    "[PivotCollector] 콜백 호출: 등록, %s, %s@%.2f, idx:%d, time:%s",
                    candidate_id, candidate_type.upper(), candidate_price, bar_idx, timestamp
                )
                self._callback(
                    event_type="registered",
                    symbol=symbol,
                    candidate_type=candidate_type.upper(),
                    candidate_price=candidate_price,
                    bar_idx=bar_idx,
                    timestamp=timestamp,
                    reason=""
                )
            except Exception as e:
                _logger.warning("[PivotCollector] 콜백 호출 실패 (등록): %s", e)
    
    def on_bar_update(
        self,
        candidate_id: str,
        bar_idx: int,
        timestamp: str,
        features: Dict[str, float],
        close: float,
    ) -> None:
        """매 봉마다 피처 업데이트.
        
        Args:
            candidate_id: 후보 ID
            bar_idx: 현재 봉 인덱스
            timestamp: 시각 "HH:MM"
            features: 현재 피처
            close: 현재 종가
        """
        if not self._enabled:
            return
        
        if candidate_id not in self.active_candidates:
            return
        
        record = self.active_candidates[candidate_id]
        
        # 시퀀스 길이 제한
        if len(record.sequence) >= self.max_sequence_length:
            record.sequence.pop(0)  # 가장 오래된 것 제거
        
        # 스냅샷 추가
        record.sequence.append(CandidateSnapshot(
            bar_idx=bar_idx,
            timestamp=timestamp,
            features=features.copy(),
            close=close,
        ))

        # 콜백 호출 (변경 이벤트) - 쿨다운 적용
        if self._callback:
            try:
                import time
                now_t = time.time()
                last_alert_t = self._last_alert_time.get(candidate_id, 0)
                if now_t - last_alert_t >= self._change_cooldown_sec:
                    _logger.info(
                        "[PivotCollector] 콜백 호출: 변경, %s, %s@%.2f, idx:%d, time:%s, reason:%s",
                        candidate_id, record.candidate_type.upper(), record.candidate_price, bar_idx, timestamp, f"대기 {len(record.sequence)}봉"
                    )
                    self._callback(
                        event_type="changed",
                        symbol=record.symbol,
                        candidate_type=record.candidate_type.upper(),
                        candidate_price=record.candidate_price,
                        bar_idx=bar_idx,
                        timestamp=timestamp,
                        reason=f"대기 {len(record.sequence)}봉"
                    )
                    self._last_alert_time[candidate_id] = now_t
                else:
                    _logger.debug(
                        "[PivotCollector] 콜백 쿨다운: 변경, %s, %.0f초 남음",
                        candidate_id, self._change_cooldown_sec - (now_t - last_alert_t)
                    )
            except Exception as e:
                _logger.warning("[PivotCollector] 콜백 호출 실패 (변경): %s", e)
    
    def on_candidate_confirmed(
        self,
        candidate_id: str,
        confirmed_bar: int,
        confirmed_time: str,
        confirmed_close: float,
        symbol: str = "KP200 선물",
    ) -> None:
        """후보 확정 시 호출.
        
        Args:
            candidate_id: 후보 ID
            confirmed_bar: 확정 봉 인덱스
            confirmed_time: 확정 시각
            confirmed_close: 확정 시점 종가
        """
        if not self._enabled:
            return
        
        if candidate_id not in self.active_candidates:
            _logger.warning("[PivotCollector] 확정 시도: 활성 후보 없음 %s", candidate_id)
            return
        
        record = self.active_candidates.pop(candidate_id)
        record.label = 1  # 확정
        record.confirmed_bar = confirmed_bar
        record.lifespan_bars = confirmed_bar - record.registered_bar
        
        # 확정 시점 스냅샷 추가
        record.sequence.append(CandidateSnapshot(
            bar_idx=confirmed_bar,
            timestamp=confirmed_time,
            features={},  # 확정 시점 피처는 필요시 별도 수집
            close=confirmed_close,
        ))
        
        self.completed_candidates.append(record)
        _logger.info(
            "[PivotCollector] 후보 확정: %s lifespan=%d봉",
            candidate_id, record.lifespan_bars
        )

        # 콜백 호출 (확정 이벤트)
        if self._callback:
            try:
                self._callback(
                    event_type="confirmed",
                    candidate_id=candidate_id,
                    candidate_type=record.candidate_type,
                    candidate_price=record.candidate_price,
                    bar_idx=confirmed_bar,
                    timestamp=confirmed_time,
                    close=confirmed_close,
                    symbol=symbol,
                    reason=""
                )
            except Exception as e:
                _logger.warning("[PivotCollector] 콜백 호출 실패 (확정): %s", e)
    
    def on_candidate_cancelled(
        self,
        candidate_id: str,
        cancelled_bar: int,
        cancelled_time: str,
        cancelled_close: float,
        reason: str,
        symbol: str = "KP200 선물",
    ) -> None:
        """후보 취소 시 호출.
        
        Args:
            candidate_id: 후보 ID
            cancelled_bar: 취소 봉 인덱스
            cancelled_time: 취소 시각
            cancelled_close: 취소 시점 종가
            reason: 취소 사유
        """
        if not self._enabled:
            return
        
        if candidate_id not in self.active_candidates:
            _logger.warning("[PivotCollector] 취소 시도: 활성 후보 없음 %s", candidate_id)
            return
        
        record = self.active_candidates.pop(candidate_id)
        record.label = 0  # 취소
        record.cancelled_bar = cancelled_bar
        record.reason = reason
        record.lifespan_bars = cancelled_bar - record.registered_bar
        
        # 취소 시점 스냅샷 추가
        record.sequence.append(CandidateSnapshot(
            bar_idx=cancelled_bar,
            timestamp=cancelled_time,
            features={},
            close=cancelled_close,
        ))
        
        self.completed_candidates.append(record)
        _logger.info(
            "[PivotCollector] 후보 취소: %s reason=%s lifespan=%d봉",
            candidate_id, reason, record.lifespan_bars
        )

        # 콜백 호출 (취소 이벤트)
        if self._callback:
            try:
                _logger.info(
                    "[PivotCollector] 콜백 호출: 취소, %s, %s@%.2f, idx:%d, time:%s, reason:%s",
                    candidate_id, record.candidate_type.upper(), record.candidate_price, cancelled_bar, cancelled_time, reason
                )
                self._callback(
                    event_type="cancelled",
                    symbol=symbol,
                    candidate_type=record.candidate_type.upper(),
                    candidate_price=record.candidate_price,
                    bar_idx=cancelled_bar,
                    timestamp=cancelled_time,
                    reason=reason
                )
            except Exception as e:
                _logger.warning("[PivotCollector] 콜백 호출 실패 (취소): %s", e)
    
    def get_statistics(self) -> Dict[str, Any]:
        """수집 통계 반환."""
        total = len(self.completed_candidates)
        confirmed = sum(1 for r in self.completed_candidates if r.label == 1)
        cancelled = total - confirmed
        
        avg_lifespan_confirmed = 0
        avg_lifespan_cancelled = 0
        
        if confirmed > 0:
            confirmed_records = [r for r in self.completed_candidates if r.label == 1]
            avg_lifespan_confirmed = sum(r.lifespan_bars for r in confirmed_records) / confirmed
        
        if cancelled > 0:
            cancelled_records = [r for r in self.completed_candidates if r.label == 0]
            avg_lifespan_cancelled = sum(r.lifespan_bars for r in cancelled_records) / cancelled
        
        return {
            "total_candidates": total,
            "confirmed": confirmed,
            "cancelled": cancelled,
            "confirmation_rate": confirmed / total if total > 0 else 0.0,
            "avg_lifespan_confirmed": avg_lifespan_confirmed,
            "avg_lifespan_cancelled": avg_lifespan_cancelled,
            "active_candidates": len(self.active_candidates),
        }
    
    def save_dataset(self, path: str) -> None:
        """학습 데이터셋 저장.
        
        Args:
            path: 저장 경로 (.pkl)
        """
        try:
            data = {
                "completed_candidates": [asdict(r) for r in self.completed_candidates],
                "statistics": self.get_statistics(),
                "saved_at": self._now_fn().isoformat(),
            }
            
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            
            _logger.info(
                "[PivotCollector] 데이터셋 저장: %s (총 %d 건)",
                path, len(self.completed_candidates)
            )
        except Exception as e:
            _logger.error("[PivotCollector] 데이터셋 저장 실패: %s", e)
    
    def load_dataset(self, path: str) -> None:
        """학습 데이터셋 로드.
        
        Args:
            path: 로드 경로 (.pkl)
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # 데이터 복원
            self.completed_candidates = []
            for item in data["completed_candidates"]:
                # 스냅샷 복원
                snapshots = [
                    CandidateSnapshot(**s) for s in item["sequence"]
                ]
                item["sequence"] = snapshots
                self.completed_candidates.append(CandidateRecord(**item))
            
            _logger.info(
                "[PivotCollector] 데이터셋 로드: %s (총 %d 건)",
                path, len(self.completed_candidates)
            )
        except Exception as e:
            _logger.error("[PivotCollector] 데이터셋 로드 실패: %s", e)
    
    def clear_completed(self) -> None:
        """완료된 후보 기록 초기화."""
        self.completed_candidates.clear()
        _logger.info("[PivotCollector] 완료된 기록 초기화")
    
    def clear_active(self) -> None:
        """활성 후보 초기화."""
        self.active_candidates.clear()
        _logger.info("[PivotCollector] 활성 후보 초기화")
    
    def clear_all(self) -> None:
        """모든 기록 초기화."""
        self.clear_completed()
        self.clear_active()
