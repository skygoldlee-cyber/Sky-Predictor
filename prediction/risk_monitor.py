"""리스크 모니터링 시스템

리스크 메트릭 초과 시 LED/알림을 보냅니다.

Usage:
    from prediction.risk_monitor import RiskMonitor
    
    monitor = RiskMonitor(led=led, notifier=notifier)
    monitor.check_risk(position_id, current_price, unrealized_pnl_pct)
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


@dataclass
class RiskThreshold:
    """리스크 임계값 설정."""
    unrealized_loss_pct: float = 2.0  # 미실현 손실률 임계값 (%)
    max_drawdown_pct: float = 5.0  # 최대 낙폭 임계값 (%)
    position_size_pct: float = 95.0  # 포지션 사이즈 임계값 (%)
    distance_to_stop_pct: float = 0.5  # 손절까지 거리 임계값 (%)


class RiskMonitor:
    """리스크 모니터."""
    
    def __init__(
        self,
        led: Optional[Any] = None,
        notifier: Optional[Any] = None,
        threshold: Optional[RiskThreshold] = None
    ):
        """초기화.
        
        Args:
            led: LED 위젯 (set_status 메서드 필요)
            notifier: 알림 인스턴스 (notify_risk_alert 메서드 필요)
            threshold: 리스크 임계값
        """
        self.led = led
        self.notifier = notifier
        self.threshold = threshold or RiskThreshold()
        self._last_alert_time: Optional[float] = None
        self._alert_cooldown_seconds = 60  # 알림 쿨다운 (초)
        _logger.info("[RISK_MONITOR] 리스크 모니터 초기화")
    
    def check_risk(
        self,
        position_id: str,
        current_price: float,
        unrealized_pnl_pct: float,
        distance_to_stop_pct: Optional[float] = None,
        position_size_pct: Optional[float] = None
    ) -> Dict[str, Any]:
        """리스크 체크 및 경고.
        
        Args:
            position_id: 포지션 ID
            current_price: 현재 가격
            unrealized_pnl_pct: 미실현 손익률 (%)
            distance_to_stop_pct: 손절까지 거리 (%)
            position_size_pct: 포지션 사이즈 (%)
        
        Returns:
            리스크 체크 결과
        """
        alerts = []
        risk_level = "normal"
        
        # 미실현 손실 체크
        if unrealized_pnl_pct < -self.threshold.unrealized_loss_pct:
            alerts.append({
                "type": "unrealized_loss",
                "value": unrealized_pnl_pct,
                "threshold": self.threshold.unrealized_loss_pct,
                "message": f"미실현 손실률 {unrealized_pnl_pct:.2f}% 초과 (임계값: {self.threshold.unrealized_loss_pct}%)"
            })
            risk_level = "high"
        
        # 손절 근접 체크
        if distance_to_stop_pct is not None and distance_to_stop_pct < self.threshold.distance_to_stop_pct:
            alerts.append({
                "type": "stop_loss_near",
                "value": distance_to_stop_pct,
                "threshold": self.threshold.distance_to_stop_pct,
                "message": f"손절 근접 {distance_to_stop_pct:.2f}% (임계값: {self.threshold.distance_to_stop_pct}%)"
            })
            if risk_level == "normal":
                risk_level = "medium"
        
        # 포지션 사이즈 체크
        if position_size_pct is not None and position_size_pct > self.threshold.position_size_pct:
            alerts.append({
                "type": "position_size",
                "value": position_size_pct,
                "threshold": self.threshold.position_size_pct,
                "message": f"포지션 사이즈 {position_size_pct:.2f}% 초과 (임계값: {self.threshold.position_size_pct}%)"
            })
            if risk_level == "normal":
                risk_level = "medium"
        
        # 경고 처리
        if alerts:
            self._handle_alerts(position_id, alerts, risk_level)
        
        return {
            "position_id": position_id,
            "risk_level": risk_level,
            "alerts": alerts,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "current_price": current_price
        }
    
    def _handle_alerts(self, position_id: str, alerts: list, risk_level: str) -> None:
        """경고 처리.
        
        Args:
            position_id: 포지션 ID
            alerts: 경고 리스트
            risk_level: 리스크 레벨
        """
        # 쿨다운 체크
        import time
        current_time = time.time()
        if self._last_alert_time and (current_time - self._last_alert_time) < self._alert_cooldown_seconds:
            _logger.debug("[RISK_MONITOR] 알림 쿨다운 중")
            return
        
        # LED 업데이트
        if self.led:
            if risk_level == "high":
                self.led.set_status("risk_alert_high")
            elif risk_level == "medium":
                self.led.set_status("risk_alert_medium")
        
        # 알림 전송
        if self.notifier:
            try:
                # 첫 번째 경고 메시지 사용
                primary_alert = alerts[0]
                reason = primary_alert["message"]
                self.notifier.notify_risk_alert(
                    position_id=position_id,
                    current_price=primary_alert.get("value", 0),
                    unrealized_pnl_pct=primary_alert.get("value", 0),
                    reason=reason
                )
                _logger.warning("[RISK_MONITOR] 리스크 경고 알림 전송: %s", reason)
            except Exception as e:
                _logger.error("[RISK_MONITOR] 알림 전송 실패: %s", e)
        
        self._last_alert_time = current_time
        _logger.warning("[RISK_MONITOR] 리스크 경고: %s - %s", position_id, [a["message"] for a in alerts])
    
    def reset_alert_cooldown(self) -> None:
        """알림 쿨다운 리셋."""
        self._last_alert_time = None
        _logger.info("[RISK_MONITOR] 알림 쿨다운 리셋")
    
    def set_threshold(self, threshold: RiskThreshold) -> None:
        """리스크 임계값 설정.
        
        Args:
            threshold: 리스크 임계값
        """
        self.threshold = threshold
        _logger.info("[RISK_MONITOR] 리스크 임계값 업데이트")
