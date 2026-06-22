"""
매매 상태 LED 표시 위젯

AlarmLED를 사용하여 매매 상태를 시각적으로 표시합니다.
"""

import logging

logger = logging.getLogger(__name__)

# QColor 임포트
try:
    from PySide6.QtGui import QColor
except ImportError:
    QColor = None


class TradeStatusLED:
    """매매 상태 LED 표시 위젯 (AlarmLED 상속)."""

    # Null Object 패턴: 더미 위젯 클래스
    try:
        from PySide6.QtWidgets import QWidget as _QW
        class _NullAlarmLED(_QW):
            """AlarmLED가 없을 때 사용하는 더미 위젯."""
            def setColor(self, color) -> None:
                pass
            def setLedSize(self, size) -> None:
                pass
    except ImportError:
        class _NullAlarmLED:
            """AlarmLED가 없을 때 사용하는 더미 위젯."""
            def setColor(self, color) -> None:
                pass
            def setLedSize(self, size) -> None:
                pass

    # 상태별 색상
    COLOR_IDLE = QColor(128, 128, 128) if QColor else "#808080"        # 회색 - 대기
    COLOR_LONG_ENTRY = QColor(0, 255, 0) if QColor else "#00FF00"      # 초록색 - 매수 진입
    COLOR_SHORT_ENTRY = QColor(255, 0, 0) if QColor else "#FF0000"     # 빨간색 - 매도 진입
    COLOR_LONG_HOLD = QColor(0, 191, 255) if QColor else "#00BFFF"     # 하늘색 - 매수 보유
    COLOR_SHORT_HOLD = QColor(255, 165, 0) if QColor else "#FFA500"   # 주황색 - 매도 보유
    COLOR_EXIT = QColor(255, 255, 255) if QColor else "#FFFFFF"        # 흰색 - 청산
    COLOR_RISK_HIGH = QColor(255, 0, 128) if QColor else "#FF0080"     # 자주색 - 리스크 높음
    COLOR_RISK_MEDIUM = QColor(255, 128, 0) if QColor else "#FF8000"   # 주황색 - 리스크 중간

    def __init__(self, parent=None):
        # super().__init__(parent) 제거 - object는 parent 인자를 받지 않음
        # AlarmLED 상속 (텍스트 없음, 크기 30)
        try:
            from gui.alarm_led import AlarmLED
            self.widget = AlarmLED(
                text="",
                text_color="#FFFFFF",
                font_size=0,
                parent=parent
            )
            self.widget.setLedSize(30)
        except ImportError:
            logger.error("[TradeStatusLED] AlarmLED 모듈을 찾을 수 없습니다. 매매 상태 LED가 비활성화됩니다.")
            # Null Object 패턴: 더미 위젯 사용
            self.widget = self._NullAlarmLED()
            self._current_status = "idle"
            return

        # 초기 색상 설정 (문자열을 QColor로 변환)
        initial_color = self.COLOR_IDLE
        if isinstance(initial_color, str) and QColor:
            initial_color = QColor(initial_color)
        self.widget.setColor(initial_color)

        self._current_status = "idle"

    def set_status(self, status: str) -> None:
        """LED 상태 설정.

        Args:
            status: 상태 ("idle", "long_entry", "short_entry", "long_hold", "short_hold", "exit")
        """
        # Null Object 패턴: widget이 항상 존재하므로 체크 불필요
        color_map = {
            "idle": self.COLOR_IDLE,
            "long_entry": self.COLOR_LONG_ENTRY,
            "short_entry": self.COLOR_SHORT_ENTRY,
            "long_hold": self.COLOR_LONG_HOLD,
            "short_hold": self.COLOR_SHORT_HOLD,
            "exit": self.COLOR_EXIT,
            "risk_alert_high": self.COLOR_RISK_HIGH,
            "risk_alert_medium": self.COLOR_RISK_MEDIUM
        }

        color = color_map.get(status, self.COLOR_IDLE)

        # 문자열 색상을 QColor로 변환
        if isinstance(color, str) and QColor:
            color = QColor(color)

        self.widget.setColor(color)

        self._current_status = status
        logger.debug("[TradeStatusLED] 상태 변경: %s", status)

    def get_status(self) -> str:
        """현재 상태 반환.

        Returns:
            현재 상태
        """
        return self._current_status
