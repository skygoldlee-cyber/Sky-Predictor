"""
파라미터 튜너 단위 테스트
"""

import json
import pytest
from pathlib import Path
from datetime import datetime
from scripts.parameter_tuner import ParameterSpace, ParameterTuner, TuningResult


@pytest.fixture
def sample_log_dir(tmp_path):
    """샘플 로그 디렉토리 생성."""
    log_dir = tmp_path / "trade_history"
    log_dir.mkdir()
    
    # 샘플 로그 데이터 생성
    logs = [
        # 진입
        {
            "timestamp": "2026-01-01T10:00:00",
            "event_type": "ENTRY",
            "action": "BUY",
            "price": 380.0,
            "confidence": "HIGH",
            "prob": 0.75,
            "consecutive_count": 2
        },
        # 청산 (승리)
        {
            "timestamp": "2026-01-01T10:30:00",
            "event_type": "EXIT",
            "action": "SELL",
            "price": 382.0,
            "reason": "TARGET_PROFIT"
        },
        # 진입
        {
            "timestamp": "2026-01-01T11:00:00",
            "event_type": "ENTRY",
            "action": "BUY",
            "price": 381.0,
            "confidence": "MEDIUM",
            "prob": 0.65,
            "consecutive_count": 2
        },
        # 청산 (패배)
        {
            "timestamp": "2026-01-01T11:30:00",
            "event_type": "EXIT",
            "action": "SELL",
            "price": 380.0,
            "reason": "STOP_LOSS"
        },
        # 진입 (LOW confidence - 필터링될 수 있음)
        {
            "timestamp": "2026-01-01T12:00:00",
            "event_type": "ENTRY",
            "action": "BUY",
            "price": 380.5,
            "confidence": "LOW",
            "prob": 0.55,
            "consecutive_count": 2
        },
        # 청산
        {
            "timestamp": "2026-01-01T12:30:00",
            "event_type": "EXIT",
            "action": "SELL",
            "price": 381.5,
            "reason": "TARGET_PROFIT"
        }
    ]
    
    log_file = log_dir / "2026-01-01.jsonl"
    with open(log_file, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")
    
    return log_dir


class TestParameterSpace:
    """ParameterSpace 테스트."""
    
    def test_default_values(self):
        """기본값 확인."""
        space = ParameterSpace()
        assert len(space.target_profit_pt) == 4
        assert len(space.stop_loss_pt) == 4
        assert len(space.trailing_stop_enabled) == 2
        assert space.min_confidence == ["LOW", "MEDIUM", "HIGH"]
    
    def test_custom_values(self):
        """커스텀 값 설정."""
        space = ParameterSpace(
            target_profit_pt=[1.0, 2.0],
            stop_loss_pt=[0.5, 1.0]
        )
        assert space.target_profit_pt == [1.0, 2.0]
        assert space.stop_loss_pt == [0.5, 1.0]


class TestParameterTuner:
    """ParameterTuner 테스트."""
    
    def test_init(self, sample_log_dir):
        """초기화 테스트."""
        tuner = ParameterTuner(log_dir=sample_log_dir)
        assert tuner.log_dir == sample_log_dir
        assert tuner.parameter_space is not None
        assert tuner.results == []
    
    def test_load_trade_logs(self, sample_log_dir):
        """로그 로드 테스트."""
        tuner = ParameterTuner(log_dir=sample_log_dir)
        logs = tuner._load_trade_logs()
        assert len(logs) == 6
    
    def test_simulate_with_params(self, sample_log_dir):
        """파라미터 시뮬레이션 테스트."""
        tuner = ParameterTuner(log_dir=sample_log_dir)
        logs = tuner._load_trade_logs()
        
        # 기본 파라미터
        params = {
            "target_profit_pt": 2.0,
            "stop_loss_pt": 1.0,
            "trailing_stop_enabled": False,
            "trailing_stop_activation_pt": 1.0,
            "trailing_stop_distance_pt": 0.5,
            "max_consecutive_losses": 3,
            "max_daily_loss_pt": 5.0,
            "min_confidence": "MEDIUM",
            "min_prob_buy": 0.62,
            "max_prob_sell": 0.38,
            "min_consecutive_signals": 2
        }
        
        result = tuner._simulate_with_params(params, logs)
        
        assert isinstance(result, TuningResult)
        assert result.params == params
        assert result.total_trades >= 0
        assert 0 <= result.win_rate <= 1
        assert result.score >= 0
    
    def test_simulate_with_strict_filter(self, sample_log_dir):
        """엄격한 필터링 테스트."""
        tuner = ParameterTuner(log_dir=sample_log_dir)
        logs = tuner._load_trade_logs()
        
        # HIGH confidence만 허용
        params = {
            "target_profit_pt": 2.0,
            "stop_loss_pt": 1.0,
            "trailing_stop_enabled": False,
            "trailing_stop_activation_pt": 1.0,
            "trailing_stop_distance_pt": 0.5,
            "max_consecutive_losses": 3,
            "max_daily_loss_pt": 5.0,
            "min_confidence": "HIGH",  # 엄격
            "min_prob_buy": 0.62,
            "max_prob_sell": 0.38,
            "min_consecutive_signals": 2
        }
        
        result = tuner._simulate_with_params(params, logs)
        
        # LOW confidence 진입은 필터링되어야 함
        assert result.total_trades <= 2  # HIGH confidence만 2개
    
    def test_simulate_with_no_trades(self, sample_log_dir):
        """거래 없는 경우 테스트."""
        tuner = ParameterTuner(log_dir=sample_log_dir)
        logs = tuner._load_trade_logs()
        
        # 너무 엄격한 필터링
        params = {
            "target_profit_pt": 2.0,
            "stop_loss_pt": 1.0,
            "trailing_stop_enabled": False,
            "trailing_stop_activation_pt": 1.0,
            "trailing_stop_distance_pt": 0.5,
            "max_consecutive_losses": 3,
            "max_daily_loss_pt": 5.0,
            "min_confidence": "HIGH",
            "min_prob_buy": 0.90,  # 너무 높음
            "max_prob_sell": 0.38,
            "min_consecutive_signals": 5  # 너무 높음
        }
        
        result = tuner._simulate_with_params(params, logs)
        
        assert result.total_trades == 0
        assert result.win_rate == 0.0
        assert result.score == 0.0
    
    def test_random_search(self, sample_log_dir):
        """Random Search 테스트."""
        tuner = ParameterTuner(log_dir=sample_log_dir)
        results = tuner.random_search(n_iterations=5)
        
        assert len(results) == 5
        assert all(isinstance(r, TuningResult) for r in results)
        
        # 점수 순 정렬 확인
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_random_search_updates_results(self, sample_log_dir):
        """Random Search가 results를 업데이트하는지 테스트."""
        tuner = ParameterTuner(log_dir=sample_log_dir)
        assert tuner.results == []
        
        tuner.random_search(n_iterations=3)
        assert len(tuner.results) == 3
    
    def test_get_best_params(self, sample_log_dir):
        """최적 파라미터 반환 테스트."""
        tuner = ParameterTuner(log_dir=sample_log_dir)
        tuner.random_search(n_iterations=3)
        
        best_params = tuner.get_best_params()
        assert best_params is not None
        assert isinstance(best_params, dict)
        assert "target_profit_pt" in best_params
    
    def test_get_best_params_no_results(self, sample_log_dir):
        """결과가 없을 때 테스트."""
        tuner = ParameterTuner(log_dir=sample_log_dir)
        best_params = tuner.get_best_params()
        assert best_params is None
    
    def test_save_results(self, sample_log_dir, tmp_path):
        """결과 저장 테스트."""
        tuner = ParameterTuner(log_dir=sample_log_dir)
        tuner.random_search(n_iterations=3)
        
        output_file = tmp_path / "results.json"
        tuner.save_results(output_file)
        
        assert output_file.exists()
        
        # 저장된 내용 확인
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert len(data) == 3
        assert "params" in data[0]
        assert "score" in data[0]
    
    def test_print_results(self, sample_log_dir, capsys):
        """결과 출력 테스트."""
        tuner = ParameterTuner(log_dir=sample_log_dir)
        tuner.random_search(n_iterations=3)
        
        tuner.print_results(top_n=2)
        
        captured = capsys.readouterr()
        assert "파라미터 튜닝 결과" in captured.out
        assert "Score:" in captured.out


class TestTuningResult:
    """TuningResult 테스트."""
    
    def test_creation(self):
        """TuningResult 생성 테스트."""
        result = TuningResult(
            params={"target_profit_pt": 2.0},
            total_trades=10,
            win_rate=0.6,
            total_profit_pct=5.0,
            avg_profit_pct_per_trade=0.5,
            max_drawdown_pct=3.0,
            sharpe_ratio=1.5,
            score=0.4
        )
        
        assert result.params == {"target_profit_pt": 2.0}
        assert result.total_trades == 10
        assert result.win_rate == 0.6
        assert result.score == 0.4
