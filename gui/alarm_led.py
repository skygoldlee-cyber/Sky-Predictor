from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QRectF, QSize
from PySide6.QtGui import QColor, QFont, QPainter, QRadialGradient
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QLabel, QWidget


class AlarmLED(QWidget):
    """LED 스타일 알람 위젯"""

    def __init__(
        self,
        text: str,
        text_color: str,
        font: str = "Arial",
        font_size: int = 10,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._color = QColor("red")
        self._radius = 100
        self.setFixedSize(self._radius, self._radius)

        self.label = QLabel(text, self)
        self.label.setStyleSheet(f"color: {text_color};")
        self.label.setFont(QFont(font, font_size))
        self.label.setAlignment(Qt.AlignCenter)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 255))
        shadow.setOffset(5, 5)
        self.setGraphicsEffect(shadow)

    def setColor(self, color: QColor) -> None:
        self._color = color
        self.update()

    def setLedSize(self, radius: int) -> None:
        self._radius = radius
        self.setFixedSize(radius, radius)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self._radius, self._radius)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        side = min(self.width(), self.height())
        square_rect = QRectF(
            (self.width() - side) / 2,
            (self.height() - side) / 2,
            side,
            side,
        )

        gradient = QRadialGradient(square_rect.center(), side / 2, square_rect.center())
        gradient.setColorAt(0, QColor(255, 255, 255, 150))
        gradient.setColorAt(0.7, self._color)
        gradient.setColorAt(1, QColor(0, 0, 0, 150))

        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(square_rect)
        self.label.setGeometry(0, 0, self._radius, self._radius)
