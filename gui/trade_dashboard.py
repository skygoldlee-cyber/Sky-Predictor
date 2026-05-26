"""거래 분석 대시보드

거래 로그를 분석하여 시각적으로 표시하는 대시보드 위젯.

Usage:
    from gui.trade_dashboard import TradeDashboard
    
    dashboard = TradeDashboard(parent=parent_widget)
    dashboard.load_and_analyze(date=datetime.now().date())
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class TradeDashboard:
    """거래 분석 대시보드 위젯."""
    
    def __init__(self, parent=None, log_dir: Optional[Path] = None):
        """초기화.
        
        Args:
            parent: 부모 위젯
            log_dir: 로그 디렉토리 (기본: logs/trades)
        """
        try:
            from PySide6.QtWidgets import (
                QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                QGroupBox, QGridLayout, QScrollArea
            )
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QFont
        except ImportError:
            logger.error("[TradeDashboard] PySide6 없음")
            self.widget = None
            return
        
        self.log_dir = log_dir or Path("logs/trades")
        self.current_date = date.today()
        
        # 위젯 생성
        self.widget = QWidget(None)
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 스크롤 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # 컨텐츠 위젯
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        # 제목
        title = QLabel("📊 거래 분석 대시보드")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        content_layout.addWidget(title)
        
        # 분석 결과 영역
        self.analysis_group = QGroupBox("분석 결과")
        analysis_layout = QGridLayout()
        self.analysis_group.setLayout(analysis_layout)
        content_layout.addWidget(self.analysis_group)
        
        # 상세 통계 영역
        self.details_group = QGroupBox("상세 통계")
        details_layout = QGridLayout()
        self.details_group.setLayout(details_layout)
        content_layout.addWidget(self.details_group)
        
        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        # 초기 분석
        self.load_and_analyze()
    
    def load_logs(self, target_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """날짜별 로그 로드.
        
        Args:
            target_date: 대상 날짜 (기본: 현재 날짜)
        
        Returns:
            거래 이벤트 리스트
        """
        if target_date is None:
            target_date = self.current_date
        
        log_file = self.log_dir / f"trades_{target_date.strftime('%Y-%m-%d')}.jsonl"
        
        if not log_file.exists():
            logger.debug("[TradeDashboard] 로그 파일 없음: %s", log_file)
            return []
        
        events = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        # 이벤트 형식 정규화
                        event = self._normalize_event(event)
                        events.append(event)
                    except Exception as e:
                        logger.debug("[TradeDashboard] 이벤트 파싱 실패: %s", e)
        except Exception as e:
            logger.error("[TradeDashboard] 로그 파일 읽기 실패: %s", e)
        
        return events
    
    def _normalize_event(self, event: Dict) -> Dict:
        """이벤트 형식 정규화 (TradeRecord 호환).
        
        Args:
            event: 원본 이벤트 딕셔너리
        
        Returns:
            정규화된 이벤트 딕셔너리
        """
        # 이미 event_type이 있으면 그대로 반환
        if "event_type" in event:
            return event
        
        # TradeRecord 형식인 경우 변환
        normalized = {
            "timestamp": event.get("entry_time") or event.get("close_time", ""),
            "event_type": "EXIT" if event.get("close_price") else "ENTRY",
            "action": event.get("side", event.get("entry_signal", "")),
            "price": event.get("entry_price") or event.get("close_price", 0),
            "size": event.get("position_size", 0),
            "confidence": event.get("entry_confidence", ""),
            "reason": event.get("close_reason", ""),
            "pnl": event.get("pnl_pt", 0),
        }
        return normalized
    
    def analyze_logs(self, events: List[Dict]) -> Dict[str, Any]:
        """거래 로그 분석.
        
        Args:
            events: 거래 이벤트 리스트
        
        Returns:
            분석 결과
        """
        if not events:
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
                "entries": 0,
                "exits": 0
            }
        
        # 진입/청산 분리
        entries = [e for e in events if e.get("event_type") == "ENTRY"]
        exits = [e for e in events if e.get("event_type") == "EXIT"]
        
        # 손익 계산
        pnls = [e.get("pnl", 0) for e in exits]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total_pnl = sum(pnls)
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        max_win = max(wins) if wins else 0
        max_loss = min(losses) if losses else 0
        
        return {
            "total_trades": len(exits),
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_win": max_win,
            "max_loss": max_loss,
            "entries": len(entries),
            "exits": len(exits)
        }
    
    def load_and_analyze(self, target_date: Optional[date] = None):
        """로그 로드 및 분석.
        
        Args:
            target_date: 대상 날짜 (기본: 현재 날짜)
        """
        events = self.load_logs(target_date)
        analysis = self.analyze_logs(events)
        self.update_dashboard(analysis)
    
    def update_dashboard(self, analysis: Dict[str, Any]):
        """대시보드 업데이트.
        
        Args:
            analysis: 분석 결과
        """
        try:
            from PySide6.QtWidgets import QLabel
            from PySide6.QtGui import QFont
            from PySide6.QtCore import Qt
        except ImportError:
            return
        
        # 분석 결과 그룹 업데이트
        analysis_layout = self.analysis_group.layout()
        
        # 기존 위젯 제거
        while analysis_layout.count():
            child = analysis_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # 핵심 지표 표시
        metrics = [
            ("총 거래수", f"{analysis['total_trades']}건"),
            ("총 손익", f"{analysis['total_pnl']:.2f}"),
            ("승률", f"{analysis['win_rate']:.1f}%"),
            ("평균 수익", f"{analysis['avg_win']:.2f}"),
            ("평균 손실", f"{analysis['avg_loss']:.2f}"),
            ("최대 수익", f"{analysis['max_win']:.2f}"),
            ("최대 손실", f"{analysis['max_loss']:.2f}"),
        ]
        
        for i, (label, value) in enumerate(metrics):
            lbl_label = QLabel(f"{label}:")
            lbl_value = QLabel(value)
            
            # 색상 적용
            if "손익" in label or "수익" in label:
                if float(value) > 0:
                    lbl_value.setStyleSheet("color: green; font-weight: bold;")
                elif float(value) < 0:
                    lbl_value.setStyleSheet("color: red; font-weight: bold;")
            
            analysis_layout.addWidget(lbl_label, i, 0)
            analysis_layout.addWidget(lbl_value, i, 1)
        
        # 상세 통계 그룹 업데이트
        details_layout = self.details_group.layout()
        
        # 기존 위젯 제거
        while details_layout.count():
            child = details_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        details = [
            ("진입 횟수", f"{analysis['entries']}회"),
            ("청산 횟수", f"{analysis['exits']}회"),
        ]
        
        for i, (label, value) in enumerate(details):
            lbl = QLabel(f"{label}: {value}")
            details_layout.addWidget(lbl, i, 0)
        
        logger.info("[TradeDashboard] 대시보드 업데이트 완료")


def attach_trade_dashboard(
    parent: Any,
    log_dir: Optional[Path] = None,
) -> Optional[TradeDashboard]:
    """
    부모 위젯에 거래 분석 대시보드를 삽입한다.
    
    Parameters
    ----------
    parent  : QWidget/QVBoxLayout
        부모 위젯 또는 레이아웃
    log_dir : Path
        로그 디렉토리 (기본: logs/trades)
    
    Returns
    -------
    TradeDashboard | None
    
    ─────── 사용 예 ───────────────────────────────────────
    
    from gui.trade_dashboard import attach_trade_dashboard
    
    # QVBoxLayout에 추가
    dashboard = attach_trade_dashboard(right_root)
    
    # QWidget에 추가
    dashboard = attach_trade_dashboard(container)
    parent_layout.addWidget(dashboard.widget)
    ──────────────────────────────────────────────────────────
    """
    try:
        from PySide6.QtWidgets import QVBoxLayout
    except ImportError:
        logger.error("[attach_trade_dashboard] PySide6 없음")
        return None
    
    dashboard = TradeDashboard(parent=None, log_dir=log_dir)
    
    if dashboard.widget is None:
        return None
    
    # 부모가 QVBoxLayout이면 위젯 추가
    if hasattr(parent, 'addWidget'):
        parent.addWidget(dashboard.widget)
    # 부모가 QWidget이면 레이아웃에 추가
    elif hasattr(parent, 'layout'):
        parent.layout().addWidget(dashboard.widget)
    
    return dashboard
