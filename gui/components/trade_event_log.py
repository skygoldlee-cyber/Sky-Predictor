"""거래 이벤트 로그 컴포넌트"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class TradeEventLog:
    """거래 이벤트 로그 컴포넌트."""
    
    def __init__(self, max_lines: int = 100):
        """
        Args:
            max_lines: 최대 라인 수
        """
        self._trade_event_log: Optional[Any] = None
        self._trade_event_log_max_lines: int = max_lines
    
    def build(self, parent: Any, root: Any) -> None:
        """거래 이벤트 로그 영역 빌드.
        
        Args:
            parent: 부모 위젯
            root: 루트 레이아웃 (QVBoxLayout)
        """
        from PySide6.QtWidgets import QTextEdit, QLabel
        
        # 라벨
        log_label = QLabel("💰 거래 이벤트")
        log_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        root.addWidget(log_label)

        # 로그 텍스트 에디터
        self._trade_event_log = QTextEdit()
        self._trade_event_log.setReadOnly(True)
        self._trade_event_log.setMaximumHeight(100)  # 최대 높이 제한
        self._trade_event_log.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
            }
        """)
        root.addWidget(self._trade_event_log)
    
    def add_event(self, timestamp: str, action: str, price: float, 
                  event_type: str, pivot_status: str = "", pivot_info: str = "") -> None:
        """거래 이벤트를 로그에 추가.
        
        Args:
            timestamp: 거래 시간
            action: BUY/SELL
            price: 거래 가격
            event_type: ENTRY/EXIT
            pivot_status: 해당 시점의 피봇 상태 (확정/미확정/없음)
            pivot_info: 가장 가까운 피봇 정보
        """
        if self._trade_event_log is None:
            return

        # 이벤트 유형에 따라 이모지와 색상 결정
        if event_type == "ENTRY":
            emoji = "📥" if action == "BUY" else "📤"
            color = "#4ec9b0" if action == "BUY" else "#f48771"
            label = "진입"
        elif event_type == "EXIT":
            emoji = "📤" if action == "SELL" else "📥"
            color = "#f48771" if action == "SELL" else "#4ec9b0"
            label = "청산"
        else:
            emoji = "❓"
            color = "#d4d4d4"
            label = event_type

        # 로그 메시지 생성
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{current_time}] {emoji} {label} {action}@{price:.2f}"
        
        # 피봇 상태 추가
        if pivot_status:
            pivot_emoji = "✅" if pivot_status == "확정" else "⏳" if pivot_status == "미확정" else "❓"
            log_msg += f" | 피봇: {pivot_emoji} {pivot_status}"
        
        if pivot_info:
            log_msg += f" ({pivot_info})"

        # HTML 형식으로 추가
        html = f'<span style="color: {color}">{log_msg}</span>'
        self._trade_event_log.append(html)

        # 최대 라인 수 제한
        if self._trade_event_log.document().blockCount() > self._trade_event_log_max_lines:
            cursor = self._trade_event_log.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.BlockUnderCursor)
            cursor.removeSelectedText()

        # 자동 스크롤
        self._trade_event_log.verticalScrollBar().setValue(
            self._trade_event_log.verticalScrollBar().maximum()
        )
