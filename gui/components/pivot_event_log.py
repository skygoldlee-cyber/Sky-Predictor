"""피봇 이벤트 로그 컴포넌트"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class PivotEventLog:
    """피봇 후보 이벤트 로그 컴포넌트."""
    
    def __init__(self, max_lines: int = 100):
        """
        Args:
            max_lines: 최대 라인 수
        """
        self._pivot_event_log: Optional[Any] = None
        self._pivot_event_log_max_lines: int = max_lines
    
    def build(self, parent: Any, root: Any) -> None:
        """피봇 후보 이벤트 로그 영역 빌드.
        
        Args:
            parent: 부모 위젯
            root: 루트 레이아웃 (QVBoxLayout)
        """
        from PySide6.QtWidgets import QTextEdit, QLabel
        
        # 라벨
        log_label = QLabel("📋 피봇 이벤트")
        log_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        root.addWidget(log_label)

        # 로그 텍스트 에디터
        self._pivot_event_log = QTextEdit()
        self._pivot_event_log.setReadOnly(True)
        self._pivot_event_log.setMaximumHeight(100)  # 최대 높이 제한
        self._pivot_event_log.setStyleSheet("""
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
        root.addWidget(self._pivot_event_log)
    
    def add_event(self, event_type: str, symbol: str, candidate_type: str,
                  candidate_price: float, bar_idx: int, timestamp: str, reason: str = "") -> None:
        """피봇 이벤트를 로그에 추가.

        Args:
            event_type: 이벤트 유형 ("registered", "changed", "cancelled", "confirmed")
            symbol: 심볼
            candidate_type: 후보 유형
            candidate_price: 후보 가격
            bar_idx: 봉 인덱스
            timestamp: 타임스탬프
            reason: 이유
        """
        if self._pivot_event_log is None:
            return

        # 이벤트 유형에 따라 이모지와 색상 결정
        if event_type == "registered":
            emoji = "🔔"
            color = "#4ec9b0"  # 청록색
            label = "등록"
        elif event_type == "changed":
            emoji = "🔄"
            color = "#dcdcaa"  # 노란색
            label = "변경"
        elif event_type == "cancelled":
            emoji = "🚫"
            color = "#f48771"  # 빨간색
            label = "취소"
        elif event_type == "confirmed":
            emoji = "✅"
            color = "#569cd6"  # 파란색
            label = "확정"
        else:
            return

        # 로그 메시지 생성 (현재 시간 타임스탬프 추가)
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{current_time}] {emoji} {label} [{timestamp}] {symbol} {candidate_type}@{candidate_price:.2f}"
        if reason:
            log_msg += f" ({reason})"

        # HTML 형식으로 추가
        html = f'<span style="color: {color}">{log_msg}</span>'
        self._pivot_event_log.append(html)

        # 최대 라인 수 제한
        if self._pivot_event_log.document().blockCount() > self._pivot_event_log_max_lines:
            cursor = self._pivot_event_log.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.BlockUnderCursor)
            cursor.removeSelectedText()

        # 자동 스크롤
        self._pivot_event_log.verticalScrollBar().setValue(
            self._pivot_event_log.verticalScrollBar().maximum()
        )
