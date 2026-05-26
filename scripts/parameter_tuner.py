"""
파라미터 튜닝 스크립트

TradeExecutionGate의 파라미터들을 백테스트 기반으로 최적화합니다.

Usage:
    python scripts/parameter_tuner.py --data data/ohlcv.csv --log-dir trade_history/
    
    # Grid Search
    python scripts/parameter_tuner.py --method grid --target-profit 1.5 2.0 2.5 --stop-loss 0.8 1.0 1.2
    
    # Random Search
    python scripts/parameter_tuner.py --method random --n-iterations 50
"""

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from itertools import product
import random

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
_logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """파라미터 탐색 공간."""
    target_profit_pt: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])
    stop_loss_pt: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2, 1.5])
    trailing_stop_enabled: List[bool] = field(default_factory=lambda: [False, True])
    trailing_stop_activation_pt: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    trailing_stop_distance_pt: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    max_consecutive_losses: List[int] = field(default_factory=lambda: [2, 3, 4])
    max_daily_loss_pt: List[float] = field(default_factory=lambda: [3.0, 5.0, 7.0])
    min_confidence: List[str] = field(default_factory=lambda: ["LOW", "MEDIUM", "HIGH"])
    min_prob_buy: List[float] = field(default_factory=lambda: [0.60, 0.62, 0.65])
    max_prob_sell: List[float] = field(default_factory=lambda: [0.35, 0.38, 0.40])
    min_consecutive_signals: List[int] = field(default_factory=lambda: [1, 2, 3])


@dataclass
class TuningResult:
    """튜닝 결과."""
    params: Dict[str, Any]
    total_trades: int
    win_rate: float
    total_profit_pct: float
    avg_profit_pct_per_trade: float
    max_drawdown_pct: float
    sharpe_ratio: float
    score: float  # 종합 점수


class ParameterTuner:
    """파라미터 튜너."""
    
    def __init__(
        self,
        log_dir: Path,
        data_file: Optional[Path] = None,
        parameter_space: Optional[ParameterSpace] = None
    ):
        """초기화.
        
        Args:
            log_dir: 거래 로그 디렉토리
            data_file: OHLCV 데이터 파일 (선택 사항)
            parameter_space: 파라미터 탐색 공간
        """
        self.log_dir = Path(log_dir)
        self.data_file = Path(data_file) if data_file else None
        self.parameter_space = parameter_space or ParameterSpace()
        self.results: List[TuningResult] = []
        
        # 데이터 로드
        self.df = None
        if self.data_file and self.data_file.exists():
            self._load_data()
    
    def _load_data(self):
        """OHLCV 데이터 로드."""
        if self.data_file.suffix == '.csv':
            self.df = pd.read_csv(self.data_file, index_col=0, parse_dates=True)
        elif self.data_file.suffix == '.parquet':
            self.df = pd.read_parquet(self.data_file)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {self.data_file.suffix}")
        
        _logger.info("[TUNER] 데이터 로드 완료: %d rows", len(self.df))
    
    def _load_trade_logs(self) -> List[Dict]:
        """거래 로그 로드."""
        logs = []
        log_files = sorted(self.log_dir.glob("*.jsonl"))
        
        for log_file in log_files:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
        
        _logger.info("[TUNER] 로드된 로그: %d개 (파일: %d개)", len(logs), len(log_files))
        return logs
    
    def _simulate_with_params(self, params: Dict[str, Any], logs: List[Dict]) -> TuningResult:
        """파라미터로 시뮬레이션 실행.
        
        Args:
            params: 테스트할 파라미터
            logs: 거래 로그
        
        Returns:
            튜닝 결과
        """
        # 파라미터 적용하여 필터링
        filtered_trades = []
        
        for log in logs:
            # 진입 조건 체크
            if log.get("event_type") != "ENTRY":
                continue
            
            # 신뢰도 필터
            min_confidence = params.get("min_confidence", "MEDIUM")
            confidence_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            if confidence_rank.get(log.get("confidence", "LOW"), 0) < confidence_rank.get(min_confidence, 1):
                continue
            
            # 확률 필터
            min_prob_buy = params.get("min_prob_buy", 0.62)
            max_prob_sell = params.get("max_prob_sell", 0.38)
            prob = log.get("prob", 0.5)
            
            if log.get("action") == "BUY" and prob < min_prob_buy:
                continue
            if log.get("action") == "SELL" and prob > max_prob_sell:
                continue
            
            # 연속 신호 필터
            min_consecutive_signals = params.get("min_consecutive_signals", 2)
            consecutive_count = log.get("consecutive_count", 1)
            if consecutive_count < min_consecutive_signals:
                continue
            
            # 리스크 관리 필터 (일일 손실, 연속 손실)
            # 실제 시뮬레이션에서는 상태를 추적해야 함
            # 여기서는 단순화하여 진입만 필터링
            
            filtered_trades.append(log)
        
        # 필터링된 거래로 결과 계산
        if not filtered_trades:
            return TuningResult(
                params=params,
                total_trades=0,
                win_rate=0.0,
                total_profit_pct=0.0,
                avg_profit_pct_per_trade=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                score=0.0
            )
        
        # 실제 청산 로그 매칭
        all_logs = logs
        entries = [t for t in filtered_trades]
        exits = [e for e in all_logs if e.get("event_type") == "EXIT"]
        
        trades = []
        equity = 1000000.0  # 초기 자본 100만원
        equity_curve = [equity]
        
        for entry in entries:
            # 해당 진입의 청산 찾기
            entry_time = entry.get("timestamp")
            matching_exits = [e for e in exits if e.get("timestamp") > entry_time]
            
            if not matching_exits:
                continue
            
            exit = min(matching_exits, key=lambda x: x["timestamp"])
            
            # 수익 계산
            entry_price = entry.get("price", 0)
            exit_price = exit.get("price", 0)
            action = entry.get("action")
            
            if action == "BUY":
                profit_pct = (exit_price - entry_price) / entry_price * 100
            else:  # SELL
                profit_pct = (entry_price - exit_price) / entry_price * 100
            
            # 수수료 (0.015%)
            profit_pct -= 0.015 * 2
            
            equity *= (1 + profit_pct / 100)
            equity_curve.append(equity)
            
            trades.append({
                "profit_pct": profit_pct,
                "exit_reason": exit.get("reason", "UNKNOWN")
            })
        
        if not trades:
            return TuningResult(
                params=params,
                total_trades=0,
                win_rate=0.0,
                total_profit_pct=0.0,
                avg_profit_pct_per_trade=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                score=0.0
            )
        
        # 통계 계산
        total_trades = len(trades)
        win_trades = sum(1 for t in trades if t["profit_pct"] > 0)
        win_rate = win_trades / total_trades
        
        total_profit_pct = (equity / 1000000.0 - 1) * 100
        avg_profit_pct_per_trade = total_profit_pct / total_trades
        
        # MDD 계산
        max_equity = max(equity_curve)
        min_equity = min(equity_curve)
        max_drawdown_pct = (max_equity - min_equity) / max_equity * 100
        
        # Sharpe Ratio 계산
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # 종합 점수 (가중평균)
        # 승률 30%, 수익 40%, MDD -20%, Sharpe 10%
        score = (
            win_rate * 0.3 +
            (total_profit_pct / 100) * 0.4 -
            (max_drawdown_pct / 100) * 0.2 +
            min(sharpe_ratio / 2, 1) * 0.1
        )
        
        return TuningResult(
            params=params,
            total_trades=total_trades,
            win_rate=win_rate,
            total_profit_pct=total_profit_pct,
            avg_profit_pct_per_trade=avg_profit_pct_per_trade,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            score=score
        )
    
    def grid_search(self) -> List[TuningResult]:
        """Grid Search 실행.
        
        Returns:
            튜닝 결과 리스트 (점수 순 정렬)
        """
        _logger.info("[TUNER] Grid Search 시작...")
        
        # 파라미터 조합 생성
        param_names = [
            "target_profit_pt", "stop_loss_pt", "trailing_stop_enabled",
            "trailing_stop_activation_pt", "trailing_stop_distance_pt",
            "max_consecutive_losses", "max_daily_loss_pt",
            "min_confidence", "min_prob_buy", "max_prob_sell",
            "min_consecutive_signals"
        ]
        
        param_values = []
        for name in param_names:
            values = getattr(self.parameter_space, name)
            param_values.append(values)
        
        # 전체 조합 수
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        _logger.info("[TUNER] 전체 조합 수: %d", total_combinations)
        
        if total_combinations > 10000:
            _logger.warning("[TUNER] 조합 수가 너무 많습니다. Random Search를 권장합니다.")
        
        # 로그 로드
        logs = self._load_trade_logs()
        
        # 각 조합 시뮬레이션
        results = []
        for i, combination in enumerate(product(*param_values)):
            params = dict(zip(param_names, combination))
            
            if (i + 1) % 100 == 0:
                _logger.info("[TUNER] 진행률: %d/%d", i + 1, total_combinations)
            
            result = self._simulate_with_params(params, logs)
            results.append(result)
        
        # 점수 순 정렬
        results.sort(key=lambda x: x.score, reverse=True)
        self.results = results
        
        _logger.info("[TUNER] Grid Search 완료: %d개 조합 테스트", len(results))
        
        return results
    
    def random_search(self, n_iterations: int = 50) -> List[TuningResult]:
        """Random Search 실행.
        
        Args:
            n_iterations: 반복 횟수
        
        Returns:
            튜닝 결과 리스트 (점수 순 정렬)
        """
        _logger.info("[TUNER] Random Search 시작 (반복: %d)...", n_iterations)
        
        # 로그 로드
        logs = self._load_trade_logs()
        
        results = []
        for i in range(n_iterations):
            # 랜덤 파라미터 생성
            params = {
                "target_profit_pt": random.choice(self.parameter_space.target_profit_pt),
                "stop_loss_pt": random.choice(self.parameter_space.stop_loss_pt),
                "trailing_stop_enabled": random.choice(self.parameter_space.trailing_stop_enabled),
                "trailing_stop_activation_pt": random.choice(self.parameter_space.trailing_stop_activation_pt),
                "trailing_stop_distance_pt": random.choice(self.parameter_space.trailing_stop_distance_pt),
                "max_consecutive_losses": random.choice(self.parameter_space.max_consecutive_losses),
                "max_daily_loss_pt": random.choice(self.parameter_space.max_daily_loss_pt),
                "min_confidence": random.choice(self.parameter_space.min_confidence),
                "min_prob_buy": random.choice(self.parameter_space.min_prob_buy),
                "max_prob_sell": random.choice(self.parameter_space.max_prob_sell),
                "min_consecutive_signals": random.choice(self.parameter_space.min_consecutive_signals),
            }
            
            if (i + 1) % 10 == 0:
                _logger.info("[TUNER] 진행률: %d/%d", i + 1, n_iterations)
            
            result = self._simulate_with_params(params, logs)
            results.append(result)
        
        # 점수 순 정렬
        results.sort(key=lambda x: x.score, reverse=True)
        self.results = results
        
        _logger.info("[TUNER] Random Search 완료: %d개 조합 테스트", len(results))
        
        return results
    
    def print_results(self, top_n: int = 10):
        """결과 출력.
        
        Args:
            top_n: 상위 N개 출력
        """
        if not self.results:
            _logger.warning("[TUNER] 결과가 없습니다.")
            return
        
        print("\n" + "=" * 80)
        print(f"파라미터 튜닝 결과 (상위 {top_n}개)")
        print("=" * 80)
        
        for i, result in enumerate(self.results[:top_n], 1):
            print(f"\n#{i} (Score: {result.score:.4f})")
            print("-" * 80)
            print(f"파라미터:")
            for key, value in result.params.items():
                print(f"  {key}: {value}")
            print(f"\n성능:")
            print(f"  총 거래: {result.total_trades}")
            print(f"  승률: {result.win_rate:.2%}")
            print(f"  총 수익: {result.total_profit_pct:.2f}%")
            print(f"  평균 수익/거래: {result.avg_profit_pct_per_trade:.2f}%")
            print(f"  최대 낙폭: {result.max_drawdown_pct:.2f}%")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        
        print("\n" + "=" * 80)
    
    def save_results(self, output_file: Path):
        """결과 저장.
        
        Args:
            output_file: 출력 파일
        """
        if not self.results:
            _logger.warning("[TUNER] 결과가 없습니다.")
            return
        
        # 결과를 JSON으로 변환
        results_data = []
        for result in self.results:
            result_dict = {
                "params": result.params,
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "total_profit_pct": result.total_profit_pct,
                "avg_profit_pct_per_trade": result.avg_profit_pct_per_trade,
                "max_drawdown_pct": result.max_drawdown_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "score": result.score
            }
            results_data.append(result_dict)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        _logger.info("[TUNER] 결과 저장 완료: %s", output_file)
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """최적 파라미터 반환.
        
        Returns:
            최적 파라미터 (결과가 없으면 None)
        """
        if not self.results:
            return None
        
        return self.results[0].params


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(description="파라미터 튜닝 스크립트")
    
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="거래 로그 디렉토리"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        help="OHLCV 데이터 파일 (선택 사항)"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        choices=["grid", "random"],
        default="random",
        help="튜닝 방법 (grid/random)"
    )
    
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=50,
        help="Random Search 반복 횟수"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=f"tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="결과 출력 파일"
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="상위 N개 결과 출력"
    )
    
    # Grid Search용 파라미터 (선택 사항)
    parser.add_argument("--target-profit", type=float, nargs="+", help="목표수익 후보")
    parser.add_argument("--stop-loss", type=float, nargs="+", help="손절 후보")
    parser.add_argument("--trailing-activation", type=float, nargs="+", help="Trailing 활성화 후보")
    parser.add_argument("--trailing-distance", type=float, nargs="+", help="Trailing 거리 후보")
    
    args = parser.parse_args()
    
    # 파라미터 공간 설정
    param_space = ParameterSpace()
    
    if args.target_profit:
        param_space.target_profit_pt = args.target_profit
    if args.stop_loss:
        param_space.stop_loss_pt = args.stop_loss
    if args.trailing_activation:
        param_space.trailing_stop_activation_pt = args.trailing_activation
    if args.trailing_distance:
        param_space.trailing_stop_distance_pt = args.trailing_distance
    
    # 튜너 초기화
    tuner = ParameterTuner(
        log_dir=args.log_dir,
        data_file=args.data,
        parameter_space=param_space
    )
    
    # 튜닝 실행
    if args.method == "grid":
        results = tuner.grid_search()
    else:  # random
        results = tuner.random_search(n_iterations=args.n_iterations)
    
    # 결과 출력
    tuner.print_results(top_n=args.top_n)
    
    # 결과 저장
    tuner.save_results(Path(args.output))
    
    # 최적 파라미터 출력
    best_params = tuner.get_best_params()
    if best_params:
        print("\n" + "=" * 80)
        print("최적 파라미터 (config.json에 적용):")
        print("=" * 80)
        print(json.dumps(best_params, indent=2, ensure_ascii=False))
        print("=" * 80)


if __name__ == "__main__":
    main()
