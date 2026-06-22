"""
포지션 사이징 단위 테스트
"""

import pytest
from trading.position_sizing import (
    SizingMethod,
    SizingConfig,
    PositionSizer,
    PositionSize
)


class TestSizingConfig:
    """SizingConfig 테스트."""
    
    def test_default_values(self):
        """기본값 확인."""
        config = SizingConfig()
        assert config.method == SizingMethod.FIXED_FRACTIONAL
        assert config.fixed_fraction == 0.95
        assert config.kelly_fraction == 0.5
        assert config.risk_per_trade == 0.02
        assert config.max_position_size == 0.3
        assert config.min_position_size == 0.05
    
    def test_custom_values(self):
        """커스텀 값 설정."""
        config = SizingConfig(
            method=SizingMethod.KELLY_CRITERION,
            fixed_fraction=0.8,
            kelly_fraction=0.6
        )
        assert config.method == SizingMethod.KELLY_CRITERION
        assert config.fixed_fraction == 0.8
        assert config.kelly_fraction == 0.6


class TestPositionSizer:
    """PositionSizer 테스트."""
    
    def test_init(self):
        """초기화 테스트."""
        config = SizingConfig()
        sizer = PositionSizer(config)
        assert sizer.config == config
    
    def test_fixed_fractional(self):
        """고정 비율 사이징 테스트."""
        config = SizingConfig(
            method=SizingMethod.FIXED_FRACTIONAL,
            fixed_fraction=0.95,
            max_position_size=1.0,  # 제한 없음
            min_position_size=0.0
        )
        sizer = PositionSizer(config)
        
        result = sizer._fixed_fractional(capital=1000000.0, entry_price=380.0)
        
        assert isinstance(result, PositionSize)
        assert result.size == pytest.approx(1000000.0 * 0.95 / 380.0)
        assert result.capital_used == pytest.approx(1000000.0 * 0.95)
        assert result.method_used == "fixed_fractional"
    
    def test_fixed_fractional_max_limit(self):
        """고정 비율 최대 제한 테스트."""
        config = SizingConfig(
            method=SizingMethod.FIXED_FRACTIONAL,
            fixed_fraction=0.95,
            max_position_size=0.3  # 30% 제한
        )
        sizer = PositionSizer(config)
        
        result = sizer._fixed_fractional(capital=1000000.0, entry_price=380.0)
        
        # 30% 제한으로 줄어야 함
        assert result.capital_used == pytest.approx(1000000.0 * 0.3)
    
    def test_fixed_fractional_min_limit(self):
        """고정 비율 최소 제한 테스트."""
        config = SizingConfig(
            method=SizingMethod.FIXED_FRACTIONAL,
            fixed_fraction=0.01,  # 너무 작음
            min_position_size=0.05  # 5% 최소
        )
        sizer = PositionSizer(config)
        
        result = sizer._fixed_fractional(capital=1000000.0, entry_price=380.0)
        
        # 5% 최소로 올라야 함
        assert result.capital_used == pytest.approx(1000000.0 * 0.05)
    
    def test_kelly_criterion(self):
        """켈리 기준 사이징 테스트."""
        config = SizingConfig(
            method=SizingMethod.KELLY_CRITERION,
            kelly_fraction=0.5,
            min_kelly=0.1,
            max_kelly=0.25
        )
        sizer = PositionSizer(config)
        
        # 승률 60%, 승패비 2:1
        result = sizer._kelly_criterion(
            capital=1000000.0,
            entry_price=380.0,
            win_rate=0.6,
            avg_win=2.0,
            avg_loss=1.0
        )
        
        assert isinstance(result, PositionSize)
        assert result.method_used == "kelly_criterion"
        # 켈리 비율 계산: (2*0.6 - 0.4) / 2 = 0.4
        # 보수적 적용: 0.4 * 0.5 = 0.2
        # 최대 제한: 0.25
        assert result.capital_used <= 1000000.0 * 0.25
    
    def test_kelly_criterion_zero_loss(self):
        """켈리 기준 - 평균 패배 0 테스트."""
        config = SizingConfig(method=SizingMethod.KELLY_CRITERION)
        sizer = PositionSizer(config)
        
        result = sizer._kelly_criterion(
            capital=1000000.0,
            entry_price=380.0,
            win_rate=0.6,
            avg_win=2.0,
            avg_loss=0.0  # 0으로 인해 fallback
        )
        
        # Fixed Fractional으로 fallback
        assert result.method_used == "fixed_fractional"
    
    def test_risk_parity(self):
        """리스크 패리티 사이징 테스트."""
        config = SizingConfig(
            method=SizingMethod.RISK_PARITY,
            risk_per_trade=0.02,
            stop_loss_pt=1.0
        )
        sizer = PositionSizer(config)
        
        result = sizer._risk_parity(
            capital=1000000.0,
            entry_price=380.0,
            stop_loss_price=379.0  # 1pt 차이
        )
        
        assert isinstance(result, PositionSize)
        assert result.method_used == "risk_parity"
        # 리스크 금액: 1000000 * 0.02 = 20000
        # 1계약당 리스크: 380 - 379 = 1
        # 사이즈: 20000 / 1 = 20000
        assert result.risk_amount == pytest.approx(20000.0)
    
    def test_risk_parity_no_stop_loss(self):
        """리스크 패리티 - 손절 가격 없음 테스트."""
        config = SizingConfig(method=SizingMethod.RISK_PARITY)
        sizer = PositionSizer(config)
        
        result = sizer._risk_parity(
            capital=1000000.0,
            entry_price=380.0,
            stop_loss_price=None  # 없음으로 인해 fallback
        )
        
        # Fixed Fractional으로 fallback
        assert result.method_used == "fixed_fractional"
    
    def test_volatility_based(self):
        """변동성 기반 사이징 테스트."""
        config = SizingConfig(
            method=SizingMethod.VOLATILITY_BASED,
            fixed_fraction=0.95,
            atr_multiplier=2.0,
            volatility_target=0.15
        )
        sizer = PositionSizer(config)
        
        result = sizer._volatility_based(
            capital=1000000.0,
            entry_price=380.0,
            atr=2.0  # 2pt ATR
        )
        
        assert isinstance(result, PositionSize)
        assert result.method_used == "volatility_based"
    
    def test_volatility_based_no_atr(self):
        """변동성 기반 - ATR 없음 테스트."""
        config = SizingConfig(method=SizingMethod.VOLATILITY_BASED)
        sizer = PositionSizer(config)
        
        result = sizer._volatility_based(
            capital=1000000.0,
            entry_price=380.0,
            atr=None  # 없음으로 인해 fallback
        )
        
        # Fixed Fractional으로 fallback
        assert result.method_used == "fixed_fractional"
    
    def test_calculate_fixed_fractional(self):
        """calculate 메서드 - Fixed Fractional 테스트."""
        config = SizingConfig(method=SizingMethod.FIXED_FRACTIONAL)
        sizer = PositionSizer(config)
        
        result = sizer.calculate(
            capital=1000000.0,
            entry_price=380.0
        )
        
        assert isinstance(result, PositionSize)
        assert result.method_used == "fixed_fractional"
    
    def test_calculate_kelly_criterion(self):
        """calculate 메서드 - Kelly Criterion 테스트."""
        config = SizingConfig(method=SizingMethod.KELLY_CRITERION)
        sizer = PositionSizer(config)
        
        result = sizer.calculate(
            capital=1000000.0,
            entry_price=380.0,
            win_rate=0.6,
            avg_win=2.0,
            avg_loss=1.0
        )
        
        assert isinstance(result, PositionSize)
        assert result.method_used == "kelly_criterion"
    
    def test_calculate_risk_parity(self):
        """calculate 메서드 - Risk Parity 테스트."""
        config = SizingConfig(method=SizingMethod.RISK_PARITY)
        sizer = PositionSizer(config)
        
        result = sizer.calculate(
            capital=1000000.0,
            entry_price=380.0,
            stop_loss_price=379.0
        )
        
        assert isinstance(result, PositionSize)
        assert result.method_used == "risk_parity"
    
    def test_calculate_volatility_based(self):
        """calculate 메서드 - Volatility-based 테스트."""
        config = SizingConfig(method=SizingMethod.VOLATILITY_BASED)
        sizer = PositionSizer(config)
        
        result = sizer.calculate(
            capital=1000000.0,
            entry_price=380.0,
            atr=2.0
        )
        
        assert isinstance(result, PositionSize)
        assert result.method_used == "volatility_based"
    
    def test_calculate_unknown_method(self):
        """알 수 없는 사이징 방법 테스트."""
        config = SizingConfig()
        # method를 잘못된 값으로 설정
        config.method = "unknown_method"  # type: ignore
        sizer = PositionSizer(config)
        
        result = sizer.calculate(
            capital=1000000.0,
            entry_price=380.0
        )
        
        # Fixed Fractional으로 fallback
        assert result.method_used == "fixed_fractional"


class TestPositionSize:
    """PositionSize 테스트."""
    
    def test_creation(self):
        """PositionSize 생성 테스트."""
        size = PositionSize(
            size=100.0,
            capital_used=95000.0,
            risk_amount=1000.0,
            risk_pct=0.1,
            method_used="fixed_fractional"
        )
        
        assert size.size == 100.0
        assert size.capital_used == 95000.0
        assert size.risk_amount == 1000.0
        assert size.risk_pct == 0.1
        assert size.method_used == "fixed_fractional"
