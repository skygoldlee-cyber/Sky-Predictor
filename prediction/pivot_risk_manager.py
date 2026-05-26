"""Pivot-based Risk Management

피봇 기반 리스크 관리 시스템.

Usage:
    from prediction.pivot_risk_manager import PivotRiskManager
    
    risk_mgr = PivotRiskManager(config)
    position_size = risk_mgr.calculate_position_size(signal, confidence, current_price)
    stop_loss, take_profit = risk_mgr.calculate_exit_levels(entry_price, atr, signal)
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """리스크 관리 설정."""
    max_position_size_pct: float = 0.95  # 최대 포지션 사이즈 (자본의 95%)
    stop_loss_atr_multiplier: float = 2.0  # 손절 ATR 멀티플라이어
    take_profit_atr_multiplier: float = 3.0  # 이익실현 ATR 멀티플라이어
    high_confidence_size_pct: float = 0.95  # HIGH confidence 포지션 사이즈
    medium_confidence_size_pct: float = 0.70  # MEDIUM confidence 포지션 사이즈
    low_confidence_size_pct: float = 0.30  # LOW confidence 포지션 사이즈
    max_risk_per_trade_pct: float = 0.02  # 거래당 최대 리스크 (2%)
    trailing_stop_atr_multiplier: float = 1.5  # 트레일링 스탑 ATR 멀티플라이어


class PivotRiskManager:
    """피봇 기반 리스크 관리자."""
    
    def __init__(self, config: Optional[RiskConfig] = None):
        """초기화.
        
        Args:
            config: 리스크 관리 설정
        """
        self.config = config or RiskConfig()
    
    def calculate_position_size(
        self,
        signal: str,
        confidence: str,
        current_price: float,
        capital: float,
        atr: Optional[float] = None
    ) -> float:
        """신호와 confidence에 따른 포지션 사이즈 계산.
        
        Args:
            signal: 신호 (BUY/SELL/HOLD)
            confidence: 신뢰도 (HIGH/MEDIUM/LOW)
            current_price: 현재 가격
            capital: 총 자본
            atr: ATR (선택)
        
        Returns:
            포지션 사이즈 (계약 수)
        """
        if signal not in ("BUY", "SELL"):
            return 0.0
        
        # confidence에 따른 기본 사이즈
        if confidence == "HIGH":
            base_size_pct = self.config.high_confidence_size_pct
        elif confidence == "MEDIUM":
            base_size_pct = self.config.medium_confidence_size_pct
        else:  # LOW
            base_size_pct = self.config.low_confidence_size_pct
        
        # 최대 사이즈 제한
        base_size_pct = min(base_size_pct, self.config.max_position_size_pct)
        
        # 기본 포지션 가치
        position_value = capital * base_size_pct
        
        # ATR 기반 리스크 조정
        if atr is not None and atr > 0:
            # 손절 가격까지의 거리
            stop_distance = atr * self.config.stop_loss_atr_multiplier
            risk_per_unit = stop_distance / current_price
            
            # 최대 리스크 초과 체크
            max_position_value = capital * self.config.max_risk_per_trade_pct / risk_per_unit
            position_value = min(position_value, max_position_value)
        
        # 계약 수 계산
        position_size = position_value / current_price
        
        _logger.info(
            "[RISK_MGR] 포지션 사이즈 계산: signal=%s confidence=%s size=%.2f contracts (value=%.0f원)",
            signal, confidence, position_size, position_value
        )
        
        return position_size
    
    def calculate_exit_levels(
        self,
        entry_price: float,
        atr: float,
        signal: str,
        confidence: str
    ) -> tuple[float, float]:
        """진입 후 손절/이익실현 가격 계산.
        
        Args:
            entry_price: 진입 가격
            atr: ATR
            signal: 신호 (BUY/SELL)
            confidence: 신뢰도 (HIGH/MEDIUM/LOW)
        
        Returns:
            (stop_loss, take_profit)
        """
        if signal == "BUY":
            stop_loss = entry_price - (atr * self.config.stop_loss_atr_multiplier)
            take_profit = entry_price + (atr * self.config.take_profit_atr_multiplier)
        elif signal == "SELL":
            stop_loss = entry_price + (atr * self.config.stop_loss_atr_multiplier)
            take_profit = entry_price - (atr * self.config.take_profit_atr_multiplier)
        else:
            stop_loss = entry_price
            take_profit = entry_price
        
        # confidence에 따른 조정
        if confidence == "LOW":
            # LOW confidence는 더 타이트한 손절
            if signal == "BUY":
                stop_loss = entry_price - (atr * self.config.stop_loss_atr_multiplier * 0.7)
            else:
                stop_loss = entry_price + (atr * self.config.stop_loss_atr_multiplier * 0.7)
        
        _logger.info(
            "[RISK_MGR] 청산 가격 계산: entry=%.2f stop=%.2f take=%.2f (atr=%.2f)",
            entry_price, stop_loss, take_profit, atr
        )
        
        return stop_loss, take_profit
    
    def calculate_trailing_stop(
        self,
        current_price: float,
        atr: float,
        signal: str,
        current_stop: float
    ) -> float:
        """트레일링 스탑 가격 계산.
        
        Args:
            current_price: 현재 가격
            atr: ATR
            signal: 신호 (BUY/SELL)
            current_stop: 현재 스탑 가격
        
        Returns:
            새로운 스탑 가격
        """
        if signal == "BUY":
            new_stop = current_price - (atr * self.config.trailing_stop_atr_multiplier)
            # 스탑은 이전보다 높아야 함 (익절 보호)
            new_stop = max(new_stop, current_stop)
        elif signal == "SELL":
            new_stop = current_price + (atr * self.config.trailing_stop_atr_multiplier)
            # 스탑은 이전보다 낮아야 함 (익절 보호)
            new_stop = min(new_stop, current_stop)
        else:
            new_stop = current_stop
        
        return new_stop
    
    def should_exit_on_pivot_change(
        self,
        current_signal: str,
        new_signal: str,
        current_position: str
    ) -> bool:
        """피봇 구조 변화 시 청산 여부 판단.
        
        Args:
            current_signal: 현재 신호
            new_signal: 새로운 신호
            current_position: 현재 포지션 (LONG/SHORT)
        
        Returns:
            청산 여부
        """
        # 반대 신호 발생 시 청산
        if current_position == "LONG" and new_signal == "SELL":
            return True
        if current_position == "SHORT" and new_signal == "BUY":
            return True
        
        # HOLD로 전환 시 청산 (구조 약화)
        if new_signal == "HOLD":
            return True
        
        return False
    
    def get_risk_metrics(
        self,
        entry_price: float,
        current_price: float,
        stop_loss: float,
        position_size: float,
        capital: float
    ) -> Dict[str, float]:
        """현재 리스크 메트릭 계산.
        
        Args:
            entry_price: 진입 가격
            current_price: 현재 가격
            stop_loss: 손절 가격
            position_size: 포지션 사이즈
            capital: 총 자본
        
        Returns:
            리스크 메트릭 딕셔너리
        """
        # 현재 수익/손실
        unrealized_pnl = (current_price - entry_price) * position_size
        unrealized_pnl_pct = unrealized_pnl / (entry_price * position_size) * 100
        
        # 손절까지의 거리
        distance_to_stop = abs(current_price - stop_loss)
        distance_to_stop_pct = distance_to_stop / current_price * 100
        
        # 현재 리스크
        current_risk = distance_to_stop * position_size
        current_risk_pct = current_risk / capital * 100
        
        return {
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "distance_to_stop": distance_to_stop,
            "distance_to_stop_pct": distance_to_stop_pct,
            "current_risk": current_risk,
            "current_risk_pct": current_risk_pct,
        }
