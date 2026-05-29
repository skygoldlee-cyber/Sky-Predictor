"""
포지션 사이징 모듈

다양한 알고리즘을 사용하여 포지션 사이즈를 동적으로 계산합니다.

지원 알고리즘:
- Fixed Fractional: 고정 비율 (자본의 일정 비율)
- Kelly Criterion: 켈리 기준 (최적 성장률)
- Risk Parity: 리스크 패리티 (동일 리스크)
- Volatility-based: 변동성 기반 (ATR 활용)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SizingMethod(str, Enum):
    """포지션 사이징 방법."""
    FIXED_FRACTIONAL = "fixed_fractional"  # 고정 비율
    KELLY_CRITERION = "kelly_criterion"  # 켈리 기준
    RISK_PARITY = "risk_parity"  # 리스크 패리티
    VOLATILITY_BASED = "volatility_based"  # 변동성 기반


@dataclass
class SizingConfig:
    """사이징 설정."""
    method: SizingMethod = SizingMethod.FIXED_FRACTIONAL
    
    # Fixed Fractional
    fixed_fraction: float = 0.95  # 자본의 95% 투자
    
    # Kelly Criterion
    kelly_fraction: float = 0.5  # 켈리 비율 (보수적 적용)
    min_kelly: float = 0.1  # 최소 켈리 비율
    max_kelly: float = 0.25  # 최대 켈리 비율
    
    # Risk Parity
    risk_per_trade: float = 0.02  # 거래당 리스크 (자본의 2%)
    stop_loss_pt: float = 1.0  # 손절 포인트
    
    # Volatility-based
    atr_multiplier: float = 2.0  # ATR 멀티플라이어
    volatility_target: float = 0.15  # 목표 변동성
    
    # 공통
    max_position_size: float = 0.3  # 최대 포지션 사이즈 (자본의 30%)
    min_position_size: float = 0.05  # 최소 포지션 사이즈 (자본의 5%)


@dataclass
class PositionSize:
    """포지션 사이즈 결과."""
    size: float  # 포지션 사이즈 (계약/주 수)
    capital_used: float  # 사용 자본
    risk_amount: float  # 리스크 금액
    risk_pct: float  # 리스크 비율
    method_used: str  # 사용된 방법


class PositionSizer:
    """포지션 사이저."""
    
    def __init__(self, config: Optional[SizingConfig] = None):
        """초기화.
        
        Args:
            config: 사이징 설정
        """
        self.config = config or SizingConfig()
    
    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        win_rate: float = 0.5,
        avg_win: float = 2.0,
        avg_loss: float = 1.0,
        atr: Optional[float] = None
    ) -> PositionSize:
        """포지션 사이즈 계산.
        
        Args:
            capital: 총 자본
            entry_price: 진입 가격
            stop_loss_price: 손절 가격
            win_rate: 승률 (0~1)
            avg_win: 평균 승리 (pt)
            avg_loss: 평균 패배 (pt)
            atr: ATR (Average True Range)
        
        Returns:
            포지션 사이즈 결과
        """
        if self.config.method == SizingMethod.FIXED_FRACTIONAL:
            return self._fixed_fractional(capital, entry_price)
        elif self.config.method == SizingMethod.KELLY_CRITERION:
            return self._kelly_criterion(capital, entry_price, win_rate, avg_win, avg_loss)
        elif self.config.method == SizingMethod.RISK_PARITY:
            return self._risk_parity(capital, entry_price, stop_loss_price)
        elif self.config.method == SizingMethod.VOLATILITY_BASED:
            return self._volatility_based(capital, entry_price, atr)
        else:
            logger.warning("[SIZING] 알 수 없는 사이징 방법: %s, Fixed Fractional 사용", self.config.method)
            return self._fixed_fractional(capital, entry_price)
    
    def _fixed_fractional(self, capital: float, entry_price: float) -> PositionSize:
        """고정 비율 사이징.
        
        Args:
            capital: 총 자본
            entry_price: 진입 가격
        
        Returns:
            포지션 사이즈 결과
        """
        fraction = self.config.fixed_fraction
        capital_used = capital * fraction
        size = capital_used / entry_price
        
        # 최대/최소 제한
        max_capital = capital * self.config.max_position_size
        min_capital = capital * self.config.min_position_size
        
        if capital_used > max_capital:
            capital_used = max_capital
            size = capital_used / entry_price
        elif capital_used < min_capital:
            capital_used = min_capital
            size = capital_used / entry_price
        
        risk_amount = capital_used  # 최악의 경우 전체 손실
        risk_pct = fraction
        
        logger.debug(
            "[SIZING] Fixed Fractional: capital=%.0f, fraction=%.2f, size=%.2f",
            capital, fraction, size
        )
        
        return PositionSize(
            size=size,
            capital_used=capital_used,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            method_used="fixed_fractional"
        )
    
    def _kelly_criterion(
        self,
        capital: float,
        entry_price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> PositionSize:
        """켈리 기준 사이징.
        
        Kelly % = (bp - q) / b
        b = 평균 승리 / 평균 패배
        p = 승률
        q = 패배율 (1 - p)
        
        Args:
            capital: 총 자본
            entry_price: 진입 가격
            win_rate: 승률
            avg_win: 평균 승리 (pt)
            avg_loss: 평균 패배 (pt)
        
        Returns:
            포지션 사이즈 결과
        """
        if avg_loss <= 0:
            logger.warning("[SIZING] avg_loss가 0 이하, Fixed Fractional 사용")
            return self._fixed_fractional(capital, entry_price)
        
        # Kelly 비율 계산
        b = avg_win / avg_loss  # 승패비
        p = win_rate
        q = 1 - p
        
        kelly_pct = (b * p - q) / b if b > 0 else 0
        
        # 보수적 적용 (일반적으로 켈리의 절반만 사용)
        kelly_pct *= self.config.kelly_fraction
        
        # 최소/최대 제한
        kelly_pct = max(self.config.min_kelly, min(self.config.max_kelly, kelly_pct))
        
        # 사이즈 계산
        capital_used = capital * kelly_pct
        size = capital_used / entry_price
        
        # 최대/최소 제한
        max_capital = capital * self.config.max_position_size
        min_capital = capital * self.config.min_position_size
        
        if capital_used > max_capital:
            capital_used = max_capital
            size = capital_used / entry_price
        elif capital_used < min_capital:
            capital_used = min_capital
            size = capital_used / entry_price
        
        risk_amount = capital_used * avg_loss / avg_win if avg_win > 0 else capital_used
        risk_pct = kelly_pct
        
        logger.debug(
            "[SIZING] Kelly Criterion: win_rate=%.2f, b=%.2f, kelly=%.2f%%, size=%.2f",
            win_rate, b, kelly_pct * 100, size
        )
        
        return PositionSize(
            size=size,
            capital_used=capital_used,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            method_used="kelly_criterion"
        )
    
    def _risk_parity(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: Optional[float]
    ) -> PositionSize:
        """리스크 패리티 사이징.
        
        거래당 리스크를 일정하게 유지.
        
        Args:
            capital: 총 자본
            entry_price: 진입 가격
            stop_loss_price: 손절 가격
        
        Returns:
            포지션 사이즈 결과
        """
        if stop_loss_price is None:
            logger.warning("[SIZING] stop_loss_price 없음, Fixed Fractional 사용")
            return self._fixed_fractional(capital, entry_price)
        
        # 거래당 리스크 금액
        risk_amount = capital * self.config.risk_per_trade
        
        # 1계약당 리스크
        risk_per_contract = abs(entry_price - stop_loss_price)
        
        if risk_per_contract <= 0:
            logger.warning("[SIZING] risk_per_contract가 0 이하, Fixed Fractional 사용")
            return self._fixed_fractional(capital, entry_price)
        
        # 사이즈 계산
        size = risk_amount / risk_per_contract
        capital_used = size * entry_price
        
        # 최대/최소 제한
        max_capital = capital * self.config.max_position_size
        min_capital = capital * self.config.min_position_size
        
        if capital_used > max_capital:
            capital_used = max_capital
            size = capital_used / entry_price
        elif capital_used < min_capital:
            capital_used = min_capital
            size = capital_used / entry_price
        
        risk_pct = self.config.risk_per_trade
        
        logger.debug(
            "[SIZING] Risk Parity: risk_per_trade=%.2f%%, risk_per_contract=%.2f, size=%.2f",
            self.config.risk_per_trade * 100, risk_per_contract, size
        )
        
        return PositionSize(
            size=size,
            capital_used=capital_used,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            method_used="risk_parity"
        )
    
    def _volatility_based(
        self,
        capital: float,
        entry_price: float,
        atr: Optional[float]
    ) -> PositionSize:
        """변동성 기반 사이징.
        
        ATR을 활용하여 변동성에 따라 사이즈 조절.
        
        Args:
            capital: 총 자본
            entry_price: 진입 가격
            atr: ATR
        
        Returns:
            포지션 사이즈 결과
        """
        if atr is None or atr <= 0:
            logger.warning("[SIZING] ATR 없음, Fixed Fractional 사용")
            return self._fixed_fractional(capital, entry_price)
        
        # 변동성 기반 리스크
        volatility_risk = atr * self.config.atr_multiplier
        
        # 목표 변동성에 따른 사이징
        # 변동성이 높으면 사이즈 줄이기
        sizing_factor = self.config.volatility_target / (atr / entry_price)
        sizing_factor = max(0.5, min(2.0, sizing_factor))  # 0.5x ~ 2x
        
        capital_used = capital * self.config.fixed_fraction * sizing_factor
        size = capital_used / entry_price
        
        # 최대/최소 제한
        max_capital = capital * self.config.max_position_size
        min_capital = capital * self.config.min_position_size
        
        if capital_used > max_capital:
            capital_used = max_capital
            size = capital_used / entry_price
        elif capital_used < min_capital:
            capital_used = min_capital
            size = capital_used / entry_price
        
        risk_amount = capital_used * (atr / entry_price) * self.config.atr_multiplier
        risk_pct = risk_amount / capital
        
        logger.debug(
            "[SIZING] Volatility-based: atr=%.2f, sizing_factor=%.2f, size=%.2f",
            atr, sizing_factor, size
        )
        
        return PositionSize(
            size=size,
            capital_used=capital_used,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            method_used="volatility_based"
        )
