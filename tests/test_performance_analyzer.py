"""
성과 분석기 단위 테스트
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from prediction.performance_analyzer import PerformanceAnalyzer, PerformanceReport
from trading.state import TradeRecord, TradeSlot, PositionSide, CloseReason


@pytest.fixture
def sample_trades():
    """샘플 거래 데이터 생성."""
    trades = []
    
    # 승리 거래
    trades.append(TradeRecord(
        slot=TradeSlot.A,
        side=PositionSide.LONG,
        entry_price=380.0,
        entry_time=datetime(2026, 1, 1, 10, 0),
        entry_signal="BUY",
        entry_confidence="HIGH",
        entry_prob=0.75,
        position_size=100.0,
        capital_used=38000.0,
        risk_amount=380.0,
        risk_pct=0.01,
        sizing_method="fixed_fractional"
    ))
    trades[-1].close_price = 382.0
    trades[-1].close_time = datetime(2026, 1, 1, 10, 30)
    trades[-1].close_reason = CloseReason.TARGET_PROFIT
    
    # 패배 거래
    trades.append(TradeRecord(
        slot=TradeSlot.B,
        side=PositionSide.LONG,
        entry_price=381.0,
        entry_time=datetime(2026, 1, 1, 11, 0),
        entry_signal="BUY",
        entry_confidence="MEDIUM",
        entry_prob=0.65,
        position_size=100.0,
        capital_used=38100.0,
        risk_amount=381.0,
        risk_pct=0.01,
        sizing_method="fixed_fractional"
    ))
    trades[-1].close_price = 380.0
    trades[-1].close_time = datetime(2026, 1, 1, 11, 30)
    trades[-1].close_reason = CloseReason.STOP_LOSS
    
    # 또 다른 승리
    trades.append(TradeRecord(
        slot=TradeSlot.C,
        side=PositionSide.SHORT,
        entry_price=382.0,
        entry_time=datetime(2026, 1, 2, 10, 0),
        entry_signal="SELL",
        entry_confidence="HIGH",
        entry_prob=0.75,
        position_size=100.0,
        capital_used=38200.0,
        risk_amount=382.0,
        risk_pct=0.01,
        sizing_method="kelly_criterion"
    ))
    trades[-1].close_price = 380.0
    trades[-1].close_time = datetime(2026, 1, 2, 10, 30)
    trades[-1].close_reason = CloseReason.TARGET_PROFIT
    
    return trades


class TestPerformanceAnalyzer:
    """PerformanceAnalyzer 테스트."""
    
    def test_init(self):
        """초기화 테스트."""
        analyzer = PerformanceAnalyzer()
        assert analyzer is not None
    
    def test_generate_report(self, sample_trades):
        """리포트 생성 테스트."""
        analyzer = PerformanceAnalyzer()
        report = analyzer.generate_report(sample_trades)
        
        assert isinstance(report, PerformanceReport)
        assert report.basic_stats is not None
        assert report.slot_analysis is not None
        assert report.sizing_analysis is not None
        assert report.advanced_metrics is not None
    
    def test_generate_report_empty(self):
        """빈 거래 리스트 테스트."""
        analyzer = PerformanceAnalyzer()
        report = analyzer.generate_report([])
        
        assert isinstance(report, PerformanceReport)
        assert report.basic_stats == {}
    
    def test_basic_stats(self, sample_trades):
        """기본 통계 테스트."""
        analyzer = PerformanceAnalyzer()
        report = analyzer.generate_report(sample_trades)
        
        assert report.basic_stats["total_trades"] == 3
        assert report.basic_stats["win_trades"] == 2
        assert report.basic_stats["loss_trades"] == 1
        assert report.basic_stats["win_rate"] == pytest.approx(2/3)
    
    def test_slot_analysis(self, sample_trades):
        """슬롯별 분석 테스트."""
        analyzer = PerformanceAnalyzer()
        report = analyzer.generate_report(sample_trades)
        
        assert "A" in report.slot_analysis
        assert "B" in report.slot_analysis
        assert "C" in report.slot_analysis
        
        assert report.slot_analysis["A"]["count"] == 1
        assert report.slot_analysis["A"]["win_rate"] == 1.0
    
    def test_sizing_analysis(self, sample_trades):
        """사이징 분석 테스트."""
        analyzer = PerformanceAnalyzer()
        report = analyzer.generate_report(sample_trades)
        
        assert report.sizing_analysis is not None
        assert "sizing_method_analysis" in report.sizing_analysis
        assert "fixed_fractional" in report.sizing_analysis["sizing_method_analysis"]
        assert "kelly_criterion" in report.sizing_analysis["sizing_method_analysis"]
    
    def test_advanced_metrics(self, sample_trades):
        """고급 지표 테스트."""
        analyzer = PerformanceAnalyzer()
        report = analyzer.generate_report(sample_trades)
        
        assert "calmar_ratio" in report.advanced_metrics
        assert "win_loss_ratio" in report.advanced_metrics
        assert "expectancy" in report.advanced_metrics
        assert "risk_of_ruin" in report.advanced_metrics
        assert "recovery_factor" in report.advanced_metrics
    
    def test_risk_metrics(self, sample_trades):
        """리스크 메트릭 테스트."""
        analyzer = PerformanceAnalyzer()
        report = analyzer.generate_report(sample_trades)
        
        assert "max_drawdown" in report.risk_metrics
        assert "sharpe_ratio" in report.risk_metrics
        assert "sortino_ratio" in report.risk_metrics
        assert "profit_factor" in report.risk_metrics
    
    def test_time_analysis(self, sample_trades):
        """시간대별 분석 테스트."""
        analyzer = PerformanceAnalyzer()
        report = analyzer.generate_report(sample_trades)
        
        assert report.time_analysis is not None
    
    def test_day_analysis(self, sample_trades):
        """요일별 분석 테스트."""
        analyzer = PerformanceAnalyzer()
        report = analyzer.generate_report(sample_trades)
        
        assert report.day_of_week_analysis is not None
    
    def test_exit_reason_analysis(self, sample_trades):
        """청산 사유별 분석 테스트."""
        analyzer = PerformanceAnalyzer()
        report = analyzer.generate_report(sample_trades)
        
        assert report.exit_reason_analysis is not None
        # CloseReason의 value는 한글이므로 한글로 확인
        assert "목표수익" in report.exit_reason_analysis
        assert "손절" in report.exit_reason_analysis
    
    def test_print_report(self, sample_trades, capsys):
        """리포트 출력 테스트."""
        analyzer = PerformanceAnalyzer()
        report = analyzer.generate_report(sample_trades)
        
        analyzer.print_report(report)
        
        captured = capsys.readouterr()
        assert "성능 분석 리포트" in captured.out
        assert "기본 통계" in captured.out
        assert "슬롯별 분석" in captured.out
        assert "고급 성과 지표" in captured.out
        assert "포지션 사이징 분석" in captured.out
    
    def test_plot_equity_curve_no_matplotlib(self, sample_trades):
        """matplotlib 없을 때 테스트."""
        analyzer = PerformanceAnalyzer()
        
        # MATPLOTLIB_AVAILABLE을 False로 설정하려면 mock 필요
        # 여기서는 경고만 확인
        analyzer.plot_equity_curve(sample_trades)
    
    def test_save_report_to_excel_no_openpyxl(self, sample_trades, tmp_path):
        """openpyxl 없을 때 테스트."""
        analyzer = PerformanceAnalyzer()
        report = analyzer.generate_report(sample_trades)
        
        output_path = tmp_path / "report.xlsx"
        
        # OPENPYXL_AVAILABLE을 False로 설정하려면 mock 필요
        # 여기서는 경고만 확인
        analyzer.save_report_to_excel(report, output_path)


class TestPerformanceReport:
    """PerformanceReport 테스트."""
    
    def test_creation(self):
        """PerformanceReport 생성 테스트."""
        report = PerformanceReport(
            basic_stats={"total_trades": 10},
            time_analysis={},
            day_of_week_analysis={},
            confidence_analysis={},
            exit_reason_analysis={},
            holding_time_analysis={},
            risk_metrics={},
            excursion_analysis={}
        )
        
        assert report.basic_stats == {"total_trades": 10}
        assert report.slot_analysis == {}
        assert report.sizing_analysis == {}
        assert report.advanced_metrics == {}
