"""거래 알림 통합

실시간 거래 이벤트에 대한 알림을 전송합니다.
텔레그램 등 다양한 채널을 지원합니다.

Usage:
    from prediction.trade_notifier import TradeNotifier, get_trade_notifier
    
    notifier = get_trade_notifier()
    notifier.notify_entry(event)
    notifier.notify_exit(event, pnl)
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


@dataclass
class NotificationConfig:
    """알림 설정."""
    enabled: bool = True
    telegram_enabled: bool = True
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    notify_on_entry: bool = True
    notify_on_exit: bool = True
    notify_on_risk_alert: bool = True
    risk_alert_threshold_pct: float = 2.0  # 리스크 알림 임계값 (%)


class TradeNotifier:
    """거래 알림 통합."""
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        """초기화.
        
        Args:
            config: 알림 설정
        """
        self.config = config or NotificationConfig()
        self._telegram_client = None
        
        if self.config.enabled and self.config.telegram_enabled:
            self._init_telegram()
    
    def _init_telegram(self):
        """텔레그램 초기화."""
        try:
            if self.config.telegram_bot_token and self.config.telegram_chat_id:
                # 텔레그램 클라이언트 초기화 (실제 구현 필요)
                self._telegram_client = True  # 플레이스홀더
                _logger.info("[TRADE_NOTIFIER] 텔레그램 초기화 완료")
            else:
                _logger.warning("[TRADE_NOTIFIER] 텔레그램 설정 없음")
        except Exception as e:
            _logger.error("[TRADE_NOTIFIER] 텔레그램 초기화 실패: %s", e)
    
    def notify_entry(self, event: Dict[str, Any]):
        """진입 알림.
        
        Args:
            event: 거래 이벤트 딕셔너리
        """
        if not self.config.enabled or not self.config.notify_on_entry:
            return
        
        message = self._format_entry_message(event)
        self._send_notification(message)
    
    def notify_exit(self, event: Dict[str, Any], pnl: float):
        """청산 알림.
        
        Args:
            event: 거래 이벤트 딕셔너리
            pnl: 손익
        """
        if not self.config.enabled or not self.config.notify_on_exit:
            return
        
        message = self._format_exit_message(event, pnl)
        self._send_notification(message)
    
    def notify_risk_alert(
        self,
        position_id: str,
        current_price: float,
        unrealized_pnl_pct: float,
        reason: str
    ):
        """리스크 알림.
        
        Args:
            position_id: 포지션 ID
            current_price: 현재 가격
            unrealized_pnl_pct: 미실현 손익률
            reason: 알림 사유
        """
        if not self.config.enabled or not self.config.notify_on_risk_alert:
            return
        
        # 임계값 체크
        if abs(unrealized_pnl_pct) < self.config.risk_alert_threshold_pct:
            return
        
        message = self._format_risk_alert_message(
            position_id, current_price, unrealized_pnl_pct, reason
        )
        self._send_notification(message)
    
    def _format_entry_message(self, event: Dict[str, Any]) -> str:
        """진입 메시지 포맷팅."""
        action = event.get("action", "")
        price = event.get("price", 0)
        size = event.get("size", 0)
        confidence = event.get("confidence", "")
        signal_reason = event.get("signal_reason", "")
        stop_loss = event.get("stop_loss", 0)
        take_profit = event.get("take_profit", 0)
        
        emoji = "📈" if action == "BUY" else "📉"
        
        message = f"{emoji} 진입 알림\n"
        message += "━━━━━━━━━━━━━━━━━━\n"
        message += f"방향: {action}\n"
        message += f"가격: {price:.2f}\n"
        message += f"사이즈: {size:.2f}\n"
        message += f"신뢰도: {confidence}\n"
        message += f"신호: {signal_reason}\n"
        message += f"손절: {stop_loss:.2f}\n"
        message += f"이익실현: {take_profit:.2f}\n"
        message += "━━━━━━━━━━━━━━━━━━\n"
        message += f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return message
    
    def _format_exit_message(self, event: Dict[str, Any], pnl: float) -> str:
        """청산 메시지 포맷팅."""
        action = event.get("action", "")
        price = event.get("price", 0)
        reason = event.get("reason", "")
        
        emoji = "✅" if pnl > 0 else "❌"
        
        message = f"{emoji} 청산 알림\n"
        message += "━━━━━━━━━━━━━━━━━━\n"
        message += f"방향: {action}\n"
        message += f"가격: {price:.2f}\n"
        message += f"사유: {reason}\n"
        message += f"손익: {pnl:,.0f}원 ({pnl/abs(pnl)*100:.2f}%)\n"
        message += "━━━━━━━━━━━━━━━━━━\n"
        message += f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return message
    
    def _format_risk_alert_message(
        self,
        position_id: str,
        current_price: float,
        unrealized_pnl_pct: float,
        reason: str
    ) -> str:
        """리스크 알림 메시지 포맷팅."""
        emoji = "⚠️" if unrealized_pnl_pct < 0 else "💰"
        
        message = f"{emoji} 리스크 알림\n"
        message += "━━━━━━━━━━━━━━━━━━\n"
        message += f"포지션: {position_id}\n"
        message += f"현재가: {current_price:.2f}\n"
        message += f"미실현 손익: {unrealized_pnl_pct:.2f}%\n"
        message += f"사유: {reason}\n"
        message += "━━━━━━━━━━━━━━━━━━\n"
        message += f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return message
    
    def _send_notification(self, message: str):
        """알림 전송.
        
        Args:
            message: 알림 메시지
        """
        # 텔레그램 전송
        if self._telegram_client:
            try:
                # 실제 텔레그램 API 호출 (구현 필요)
                # requests.post(
                #     f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage",
                #     json={"chat_id": self.config.telegram_chat_id, "text": message}
                # )
                _logger.info("[TRADE_NOTIFIER] 알림 전송 (텔레그램): %s", message[:50] + "...")
            except Exception as e:
                _logger.error("[TRADE_NOTIFIER] 텔레그램 전송 실패: %s", e)
        else:
            # 로그 출력만
            _logger.info("[TRADE_NOTIFIER] 알림: %s", message)


# 전역 알림 인스턴스
_global_notifier: Optional[TradeNotifier] = None


def get_trade_notifier(config: Optional[NotificationConfig] = None) -> TradeNotifier:
    """전역 거래 알림 인스턴스 반환.
    
    Args:
        config: 알림 설정
    
    Returns:
        거래 알림 인스턴스
    """
    global _global_notifier
    if _global_notifier is None:
        _global_notifier = TradeNotifier(config)
    return _global_notifier
