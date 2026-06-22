"""거래 로그 뷰어

GUI 내에서 거래 로그를 볼 수 있는 뷰어 위젯.

Usage:
    from gui.trade_log_viewer import TradeLogViewer
    
    viewer = TradeLogViewer(parent=parent_widget)
    viewer.load_logs(date=datetime.now().date())
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class TradeLogViewer:
    """거래 로그 뷰어 위젯."""
    
    def __init__(self, parent=None, log_dir: Optional[Path] = None):
        """초기화.
        
        Args:
            parent: 부모 위젯
            log_dir: 로그 디렉토리 (기본: logs/trades)
        """
        try:
            from PySide6.QtWidgets import (
                QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
                QTableWidgetItem, QHeaderView, QLabel, QComboBox,
                QPushButton, QDateEdit
            )
            from PySide6.QtCore import Qt, QDate
        except ImportError:
            logger.error("[TradeLogViewer] PySide6 없음")
            self.widget = None
            return
        
        self.log_dir = log_dir or Path("logs/trades")
        self.current_date = date.today()
        self.current_filter = "ALL"  # ALL, ENTRY, EXIT, TRAILING_STOP, ATR_SNAPSHOT
        
        # 위젯 생성 (parent는 무시 - 외부에서 레이아웃에 추가)
        self.widget = QWidget(None)
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 컨트롤 바
        ctrl_row = QHBoxLayout()
        ctrl_row.setContentsMargins(6, 2, 6, 2)
        
        # 날짜 선택
        ctrl_row.addWidget(QLabel("날짜:"))
        self.date_edit = QDateEdit()
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.dateChanged.connect(self._on_date_changed)
        ctrl_row.addWidget(self.date_edit)
        
        # 필터
        ctrl_row.addWidget(QLabel("필터:"))
        self.filter_cb = QComboBox()
        self.filter_cb.setMinimumWidth(150)  # 최소 너비 증가 (120 -> 150)
        self.filter_cb.addItem("전체", "ALL")
        self.filter_cb.addItem("진입", "ENTRY")
        self.filter_cb.addItem("청산", "EXIT")
        self.filter_cb.addItem("트레일링 스탑", "TRAILING_STOP")
        self.filter_cb.addItem("ATR 스냅샷", "ATR_SNAPSHOT")
        self.filter_cb.currentIndexChanged.connect(self._on_filter_changed)
        ctrl_row.addWidget(self.filter_cb)
        
        # 새로고침 버튼
        btn_refresh = QPushButton("↺ 새로고침")
        btn_refresh.setFixedWidth(100)  # 폭 증가 (80 -> 100)
        btn_refresh.clicked.connect(self.refresh)
        ctrl_row.addWidget(btn_refresh)
        
        ctrl_row.addStretch(1)
        
        # 통계 레이블
        self.stats_label = QLabel("")
        ctrl_row.addWidget(self.stats_label)
        
        ctrl_w = QWidget()
        ctrl_w.setLayout(ctrl_row)
        layout.addWidget(ctrl_w)
        
        # 테이블
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "시간", "타입", "방향", "가격", "사이즈", "신뢰도", "사유", "손익"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table)
        
        # 초기 로드
        self.refresh()
    
    def _on_date_changed(self, qdate):
        """날짜 변경 시 처리.
        
        Args:
            qdate: Qt 날짜
        """
        self.current_date = qdate.toPython()
        self.refresh()
    
    def _on_filter_changed(self, index):
        """필터 변경 시 처리.
        
        Args:
            index: 콤보박스 인덱스
        """
        self.current_filter = self.filter_cb.itemData(index)
        self.refresh()
    
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
            logger.debug("[TradeLogViewer] 로그 파일 없음: %s", log_file)
            return []
        
        events = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        events.append(event)
                    except Exception as e:
                        logger.debug("[TradeLogViewer] 이벤트 파싱 실패: %s", e)
        except Exception as e:
            logger.error("[TradeLogViewer] 로그 파일 읽기 실패: %s", e)
        
        return events
    
    def filter_by_type(self, events: List[Dict], event_type: str) -> List[Dict]:
        """이벤트 타입 필터링.
        
        Args:
            events: 거래 이벤트 리스트
            event_type: 이벤트 타입 (ALL, ENTRY, EXIT, TRAILING_STOP, ATR_SNAPSHOT)
        
        Returns:
            필터링된 이벤트 리스트
        """
        if event_type == "ALL":
            return events
        return [e for e in events if e.get("event_type") == event_type]
    
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
    
    def refresh(self):
        """로그 새로고침."""
        events = self.load_logs()
        # 이벤트 형식 정규화
        events = [self._normalize_event(e) for e in events]
        filtered_events = self.filter_by_type(events, self.current_filter)
        
        # 테이블 갱신
        self.table.setRowCount(len(filtered_events))
        
        for row, event in enumerate(filtered_events):
            try:
                # 시간
                timestamp = event.get("timestamp", "")
                if isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = dt.strftime("%H:%M:%S")
                    except Exception:
                        time_str = timestamp[:19]
                else:
                    time_str = str(timestamp)
                self.table.setItem(row, 0, QTableWidgetItem(time_str))
                
                # 타입
                event_type = event.get("event_type", "")
                self.table.setItem(row, 1, QTableWidgetItem(event_type))
                
                # 방향
                action = event.get("action", "")
                self.table.setItem(row, 2, QTableWidgetItem(action))
                
                # 가격
                price = event.get("price", 0)
                self.table.setItem(row, 3, QTableWidgetItem(f"{price:.2f}"))
                
                # 사이즈
                size = event.get("size", 0)
                self.table.setItem(row, 4, QTableWidgetItem(f"{size:.2f}"))
                
                # 신뢰도
                confidence = event.get("confidence", "")
                self.table.setItem(row, 5, QTableWidgetItem(confidence))
                
                # 사유
                reason = event.get("reason", event.get("signal_reason", ""))
                self.table.setItem(row, 6, QTableWidgetItem(str(reason)[:50]))
                
                # 손익 (청산 시)
                pnl = event.get("pnl", 0)
                if pnl != 0:
                    pnl_item = QTableWidgetItem(f"{pnl:.2f}")
                    if pnl > 0:
                        pnl_item.setForeground(Qt.GlobalColor.green)
                    else:
                        pnl_item.setForeground(Qt.GlobalColor.red)
                    self.table.setItem(row, 7, pnl_item)
                else:
                    self.table.setItem(row, 7, QTableWidgetItem(""))
                
            except Exception as e:
                logger.debug("[TradeLogViewer] 행 표시 실패: %s", e)
        
        # 통계 업데이트
        entry_count = len([e for e in events if e.get("event_type") == "ENTRY"])
        exit_count = len([e for e in events if e.get("event_type") == "EXIT"])
        total_pnl = sum(e.get("pnl", 0) for e in events if e.get("event_type") == "EXIT")
        
        self.stats_label.setText(
            f"진입: {entry_count} | 청산: {exit_count} | 총 손익: {total_pnl:.2f}"
        )
        
        logger.info(
            "[TradeLogViewer] 로드 완료: %s (필터: %s, 이벤트: %d)",
            self.current_date, self.current_filter, len(filtered_events)
        )
    
    def get_selected_event(self) -> Optional[Dict[str, Any]]:
        """선택된 이벤트 반환.
        
        Returns:
            선택된 이벤트
        """
        current_row = self.table.currentRow()
        if current_row < 0:
            return None
        
        events = self.load_logs()
        filtered_events = self.filter_by_type(events, self.current_filter)
        
        if current_row < len(filtered_events):
            return filtered_events[current_row]
        
        return None


def attach_trade_log_viewer(
    parent: Any,
    log_dir: Optional[Path] = None,
) -> Optional[TradeLogViewer]:
    """
    부모 위젯에 거래 로그 뷰어를 삽입한다.
    
    Parameters
    ----------
    parent  : QWidget/QVBoxLayout
        부모 위젯 또는 레이아웃
    log_dir : Path
        로그 디렉토리 (기본: logs/trades)
    
    Returns
    -------
    TradeLogViewer | None
    
    ─────── 사용 예 ───────────────────────────────────────
    
    from gui.trade_log_viewer import attach_trade_log_viewer
    
    # QVBoxLayout에 추가
    log_viewer = attach_trade_log_viewer(right_root)
    
    # QWidget에 추가
    log_viewer = attach_trade_log_viewer(container)
    parent_layout.addWidget(log_viewer.widget)
    ──────────────────────────────────────────────────────────
    """
    try:
        from PySide6.QtWidgets import QVBoxLayout
    except ImportError:
        logger.error("[attach_trade_log_viewer] PySide6 없음")
        return None
    
    viewer = TradeLogViewer(parent=None, log_dir=log_dir)
    
    if viewer.widget is None:
        return None
    
    # 부모가 QVBoxLayout이면 위젯 추가
    if hasattr(parent, 'addWidget'):
        parent.addWidget(viewer.widget)
    # 부모가 QWidget이면 레이아웃에 추가
    elif hasattr(parent, 'layout'):
        parent.layout().addWidget(viewer.widget)
    
    return viewer
