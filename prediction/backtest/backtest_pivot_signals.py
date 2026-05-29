"""Pivot Signal Backtesting Framework

피봇 기반 매매 신호의 백테스팅 프레임워크.
실제 거래 로그 기반 백테스팅도 지원합니다.
실시간 거래 로직과 동일한 포지션 관리를 사용합니다.

Usage:
    from prediction.backtest_pivot_signals import PivotSignalBacktester

    backtester = PivotSignalBacktester()
    results = backtester.run_backtest(df, signals)

    # 로그 기반 백테스팅
    results = backtester.run_backtest_from_logs(log_file)

    # 로그 + OHLCV 기반 백테스팅
    results = backtester.run_backtest_from_logs_with_ohlcv(log_file, df)

    # 결과 저장
    backtester.save_all(results)  # 전체, 거래, 요약 모두 저장
    # 또는 개별 저장
    backtester.save_results_to_json(results)  # 전체 결과 JSON
    backtester.save_trades_to_csv(results)    # 거래 기록 CSV
    backtester.save_summary_to_json(results)  # 요약 결과 JSON
"""

import json
import logging
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """백테스팅 설정."""
    initial_capital: float = 10000000.0  # 초기 자본 (1000만원)
    tick_size: float = 0.05  # 틱 사이즈
    commission_rate: float = 0.00015  # 수수료율 (0.015%)
    slippage_ticks: int = 1  # 슬리피지 (틱 수)
    position_size_pct: float = 0.95  # 포지션 사이즈 (자본의 95%)
    stop_loss_atr_multiplier: float = 2.0  # 손절 ATR 멀티플라이어
    take_profit_atr_multiplier: float = 3.0  # 이익실현 ATR 멀티플라이어
    trailing_stop_atr_multiplier: float = 1.5  # 트레일링 스탑 ATR 멀티플라이어


@dataclass
class Trade:
    """거래 기록."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    action: str  # "BUY" or "SELL"
    exit_reason: Optional[str]
    profit: Optional[float]
    profit_pct: Optional[float]
    bars_held: Optional[int]


@dataclass
class BacktestResult:
    """백테스팅 결과."""
    total_trades: int
    win_trades: int
    loss_trades: int
    win_rate: float
    total_profit: float
    total_profit_pct: float
    avg_profit_per_trade: float
    avg_profit_pct_per_trade: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[Trade]
    equity_curve: List[float]


class PivotSignalBacktester:
    """피봇 신호 백테스터."""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """초기화.
        
        Args:
            config: 백테스팅 설정
        """
        self.config = config or BacktestConfig()
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        signals: pd.DataFrame,
        atr_col: str = "ATR"
    ) -> BacktestResult:
        """백테스팅 실행.
        
        Args:
            df: OHLCV 데이터프레임
            signals: 신호 데이터프레임 (timestamp, action, confidence)
            atr_col: ATR 컬럼명
        
        Returns:
            백테스팅 결과
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        
        # 신호와 데이터 병합
        merged = df.join(signals, how='left')
        
        # 거래 기록
        trades: List[Trade] = []
        equity_curve: List[float] = []
        current_equity = self.config.initial_capital
        max_equity = current_equity
        max_drawdown = 0.0
        
        # 현재 포지션
        position = None  # {"action": "BUY"/"SELL", "entry_price": float, "entry_time": datetime, "stop_loss": float, "take_profit": float}
        
        for idx, row in merged.iterrows():
            current_price = row["Close"]
            atr = row.get(atr_col, 0.0) or 0.0
            
            # 현재 포지션 청산 체크
            if position is not None:
                # 손절/이익실현 체크
                should_exit = False
                exit_reason = None
                exit_price = current_price
                
                if position["action"] == "BUY":
                    if current_price <= position["stop_loss"]:
                        should_exit = True
                        exit_reason = "stop_loss"
                        exit_price = position["stop_loss"]
                    elif current_price >= position["take_profit"]:
                        should_exit = True
                        exit_reason = "take_profit"
                        exit_price = position["take_profit"]
                else:  # SELL
                    if current_price >= position["stop_loss"]:
                        should_exit = True
                        exit_reason = "stop_loss"
                        exit_price = position["stop_loss"]
                    elif current_price <= position["take_profit"]:
                        should_exit = True
                        exit_reason = "take_profit"
                        exit_price = position["take_profit"]
                
                # 반대 신호로 청산
                signal_action = row.get("action", None)
                if signal_action is not None and signal_action != position["action"]:
                    should_exit = True
                    exit_reason = "signal_reversal"
                
                if should_exit:
                    # 청산
                    slippage = self.config.slippage_ticks * self.config.tick_size
                    if position["action"] == "BUY":
                        exit_price = exit_price - slippage
                        profit = (exit_price - position["entry_price"]) * position["size"]
                    else:  # SELL
                        exit_price = exit_price + slippage
                        profit = (position["entry_price"] - exit_price) * position["size"]
                    
                    # 수수료
                    commission = (position["entry_price"] + exit_price) * position["size"] * self.config.commission_rate
                    profit -= commission
                    
                    current_equity += profit
                    bars_held = (idx - position["entry_time"]).total_seconds() / 60  # 분 단위
                    
                    trade = Trade(
                        entry_time=position["entry_time"],
                        exit_time=idx,
                        entry_price=position["entry_price"],
                        exit_price=exit_price,
                        action=position["action"],
                        exit_reason=exit_reason,
                        profit=profit,
                        profit_pct=profit / (position["entry_price"] * position["size"]) * 100,
                        bars_held=int(bars_held)
                    )
                    trades.append(trade)
                    position = None
            
            # 신호로 진입
            if position is None:
                signal_action = row.get("action", None)
                signal_confidence = row.get("confidence", "LOW")
                
                if signal_action in ("BUY", "SELL") and signal_confidence in ("HIGH", "MEDIUM"):
                    # 진입
                    slippage = self.config.slippage_ticks * self.config.tick_size
                    if signal_action == "BUY":
                        entry_price = current_price + slippage
                    else:  # SELL
                        entry_price = current_price - slippage
                    
                    # 포지션 사이즈 계산
                    position_value = current_equity * self.config.position_size_pct
                    size = position_value / entry_price
                    
                    # 손절/이익실현 가격
                    stop_loss = entry_price - (atr * self.config.stop_loss_atr_multiplier) if signal_action == "BUY" else entry_price + (atr * self.config.stop_loss_atr_multiplier)
                    take_profit = entry_price + (atr * self.config.take_profit_atr_multiplier) if signal_action == "BUY" else entry_price - (atr * self.config.take_profit_atr_multiplier)
                    
                    position = {
                        "action": signal_action,
                        "entry_price": entry_price,
                        "entry_time": idx,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "size": size
                    }
            
            # equity curve 기록
            equity_curve.append(current_equity)
            
            # 최대 자본 갱신 및 MDD 계산
            if current_equity > max_equity:
                max_equity = current_equity
            drawdown = (max_equity - current_equity) / max_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 최종 포지션 청산
        if position is not None:
            current_price = df.iloc[-1]["Close"]
            slippage = self.config.slippage_ticks * self.config.tick_size
            if position["action"] == "BUY":
                exit_price = current_price - slippage
                profit = (exit_price - position["entry_price"]) * position["size"]
            else:
                exit_price = current_price + slippage
                profit = (position["entry_price"] - exit_price) * position["size"]
            
            commission = (position["entry_price"] + exit_price) * position["size"] * self.config.commission_rate
            profit -= commission
            
            current_equity += profit
            bars_held = (df.index[-1] - position["entry_time"]).total_seconds() / 60
            
            trade = Trade(
                entry_time=position["entry_time"],
                exit_time=df.index[-1],
                entry_price=position["entry_price"],
                exit_price=exit_price,
                action=position["action"],
                exit_reason="end_of_data",
                profit=profit,
                profit_pct=profit / (position["entry_price"] * position["size"]) * 100,
                bars_held=int(bars_held)
            )
            trades.append(trade)
        
        # 결과 계산
        if not trades:
            return BacktestResult(
                total_trades=0,
                win_trades=0,
                loss_trades=0,
                win_rate=0.0,
                total_profit=0.0,
                total_profit_pct=0.0,
                avg_profit_per_trade=0.0,
                avg_profit_pct_per_trade=0.0,
                max_drawdown=max_drawdown,
                max_drawdown_pct=max_drawdown * 100,
                sharpe_ratio=0.0,
                trades=[],
                equity_curve=equity_curve
            )
        
        win_trades = sum(1 for t in trades if t.profit and t.profit > 0)
        loss_trades = sum(1 for t in trades if t.profit and t.profit <= 0)
        win_rate = win_trades / len(trades) if trades else 0.0
        total_profit = sum(t.profit for t in trades if t.profit is not None)
        total_profit_pct = total_profit / self.config.initial_capital * 100
        avg_profit_per_trade = total_profit / len(trades)
        avg_profit_pct_per_trade = total_profit_pct / len(trades)
        
        # Sharpe Ratio 계산 (단순화)
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        return BacktestResult(
            total_trades=len(trades),
            win_trades=win_trades,
            loss_trades=loss_trades,
            win_rate=win_rate,
            total_profit=total_profit,
            total_profit_pct=total_profit_pct,
            avg_profit_per_trade=avg_profit_per_trade,
            avg_profit_pct_per_trade=avg_profit_pct_per_trade,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown * 100,
            sharpe_ratio=sharpe_ratio,
            trades=trades,
            equity_curve=equity_curve
        )
    
    def run_backtest_from_logs(
        self,
        log_file: Path,
        initial_capital: Optional[float] = None
    ) -> BacktestResult:
        """거래 로그 기반 백테스팅 실행.
        
        Args:
            log_file: 거래 로그 파일 (.jsonl)
            initial_capital: 초기 자본 (None이면 config 사용)
        
        Returns:
            백테스팅 결과
        """
        import json
        
        if not log_file.exists():
            _logger.error("[BACKTEST] 로그 파일 없음: %s", log_file)
            return BacktestResult(
                total_trades=0, win_trades=0, loss_trades=0, win_rate=0.0,
                total_profit=0.0, total_profit_pct=0.0, avg_profit_per_trade=0.0,
                avg_profit_pct_per_trade=0.0, max_drawdown=0.0, max_drawdown_pct=0.0,
                sharpe_ratio=0.0, trades=[], equity_curve=[]
            )
        
        # 이벤트 로드
        events = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        
        _logger.info("[BACKTEST] 로드된 이벤트: %d개", len(events))
        
        # 이벤트 정렬
        events.sort(key=lambda x: x["timestamp"])
        
        # 진입/청산 매칭
        entries = [e for e in events if e["event_type"] == "ENTRY"]
        exits = [e for e in events if e["event_type"] == "EXIT"]
        
        trades: List[Trade] = []
        equity_curve: List[float] = []
        capital = initial_capital or self.config.initial_capital
        equity_curve.append(capital)
        
        for entry in entries:
            # 해당 진입의 청산 찾기
            matching_exits = [e for e in exits if e["timestamp"] > entry["timestamp"]]
            if not matching_exits:
                continue
            
            # 가장 빠른 청산 사용
            exit = min(matching_exits, key=lambda x: x["timestamp"])
            
            # 수익 계산
            if entry["action"] == "BUY":
                profit = (exit["price"] - entry["price"]) * entry["size"]
            else:  # SELL
                profit = (entry["price"] - exit["price"]) * entry["size"]
            
            # 수수료
            commission = (entry["price"] + exit["price"]) * entry["size"] * self.config.commission_rate
            profit -= commission
            
            capital += profit
            equity_curve.append(capital)
            
            # 보유 기간 계산
            entry_time = datetime.fromisoformat(entry["timestamp"])
            exit_time = datetime.fromisoformat(exit["timestamp"])
            bars_held = int((exit_time - entry_time).total_seconds() / 60)  # 분 단위
            
            trade = Trade(
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry["price"],
                exit_price=exit["price"],
                action=entry["action"],
                exit_reason=exit["reason"],
                profit=profit,
                profit_pct=profit / (entry["price"] * entry["size"]) * 100,
                bars_held=bars_held
            )
            trades.append(trade)
        
        # 결과 계산
        if not trades:
            return BacktestResult(
                total_trades=0, win_trades=0, loss_trades=0, win_rate=0.0,
                total_profit=0.0, total_profit_pct=0.0, avg_profit_per_trade=0.0,
                avg_profit_pct_per_trade=0.0, max_drawdown=0.0, max_drawdown_pct=0.0,
                sharpe_ratio=0.0, trades=[], equity_curve=equity_curve
            )
        
        win_trades = sum(1 for t in trades if t.profit and t.profit > 0)
        loss_trades = sum(1 for t in trades if t.profit and t.profit <= 0)
        win_rate = win_trades / len(trades) if trades else 0.0
        total_profit = sum(t.profit for t in trades if t.profit is not None)
        total_profit_pct = total_profit / (initial_capital or self.config.initial_capital) * 100
        avg_profit_per_trade = total_profit / len(trades)
        avg_profit_pct_per_trade = total_profit_pct / len(trades)
        
        # MDD 계산
        max_equity = max(equity_curve) if equity_curve else capital
        min_equity = min(equity_curve) if equity_curve else capital
        max_drawdown = max_equity - min_equity
        max_drawdown_pct = max_drawdown / max_equity * 100 if max_equity > 0 else 0.0
        
        # Sharpe Ratio 계산
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        _logger.info(
            "[BACKTEST] 로그 기반 결과: %d거래, 승률=%.1f%%, 수익=%.2fpt",
            len(trades), win_rate * 100, total_profit
        )
        
    def run_backtest_from_logs_with_ohlcv(
        self,
        log_file: Path,
        df: pd.DataFrame,
        initial_capital: Optional[float] = None
    ) -> BacktestResult:
        """로그 + OHLCV 데이터 기반 백테스팅 실행.
        
        Args:
            log_file: 거래 로그 파일 (.jsonl)
            df: OHLCV 데이터프레임
            initial_capital: 초기 자본 (None이면 config 사용)
        
        Returns:
            백테스팅 결과
        """
        import json
        
        if not log_file.exists():
            _logger.error("[BACKTEST] 로그 파일 없음: %s", log_file)
            return BacktestResult(
                total_trades=0, win_trades=0, loss_trades=0, win_rate=0.0,
                total_profit=0.0, total_profit_pct=0.0, avg_profit_per_trade=0.0,
                avg_profit_pct_per_trade=0.0, max_drawdown=0.0, max_drawdown_pct=0.0,
                sharpe_ratio=0.0, trades=[], equity_curve=[]
            )
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        
        # 이벤트 로드
        events = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        
        _logger.info("[BACKTEST] 로드된 이벤트: %d개", len(events))
        
        # 이벤트 정렬
        events.sort(key=lambda x: x["timestamp"])
        
        # 진입/청산 매칭
        entries = [e for e in events if e["event_type"] == "ENTRY"]
        exits = [e for e in events if e["event_type"] == "EXIT"]
        
        trades: List[Trade] = []
        equity_curve: List[float] = []
        capital = initial_capital or self.config.initial_capital
        equity_curve.append(capital)
        
        for entry in entries:
            # 해당 진입의 청산 찾기
            matching_exits = [e for e in exits if e["timestamp"] > entry["timestamp"]]
            if not matching_exits:
                continue
            
            # 가장 빠른 청산 사용
            exit = min(matching_exits, key=lambda x: x["timestamp"])
            
            # OHLCV 데이터에서 해당 기간의 실제 가격 변동 확인
            entry_time = pd.to_datetime(entry["timestamp"])
            exit_time = pd.to_datetime(exit["timestamp"])
            
            # 해당 기간의 OHLCV 데이터 슬라이스
            period_df = df.loc[entry_time:exit_time]
            
            if len(period_df) == 0:
                _logger.warning("[BACKTEST] OHLCV 데이터 없음: %s ~ %s", entry_time, exit_time)
                continue
            
            # 실제 청산 가격 검증 (로그의 청산 가격 vs OHLCV의 실제 청산 시점)
            actual_exit_price = period_df.iloc[-1]["Close"]
            
            # 수익 계산
            if entry["action"] == "BUY":
                profit = (actual_exit_price - entry["price"]) * entry["size"]
            else:  # SELL
                profit = (entry["price"] - actual_exit_price) * entry["size"]
            
            # 수수료
            commission = (entry["price"] + actual_exit_price) * entry["size"] * self.config.commission_rate
            profit -= commission
            
            capital += profit
            equity_curve.append(capital)
            
            # 보유 기간 계산
            bars_held = len(period_df)
            
            trade = Trade(
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry["price"],
                exit_price=actual_exit_price,
                action=entry["action"],
                exit_reason=exit["reason"],
                profit=profit,
                profit_pct=profit / (entry["price"] * entry["size"]) * 100,
                bars_held=bars_held
            )
            trades.append(trade)
        
        # 결과 계산
        if not trades:
            return BacktestResult(
                total_trades=0, win_trades=0, loss_trades=0, win_rate=0.0,
                total_profit=0.0, total_profit_pct=0.0, avg_profit_per_trade=0.0,
                avg_profit_pct_per_trade=0.0, max_drawdown=0.0, max_drawdown_pct=0.0,
                sharpe_ratio=0.0, trades=[], equity_curve=equity_curve
            )
        
        win_trades = sum(1 for t in trades if t.profit and t.profit > 0)
        loss_trades = sum(1 for t in trades if t.profit and t.profit <= 0)
        win_rate = win_trades / len(trades) if trades else 0.0
        total_profit = sum(t.profit for t in trades if t.profit is not None)
        total_profit_pct = total_profit / (initial_capital or self.config.initial_capital) * 100
        avg_profit_per_trade = total_profit / len(trades)
        avg_profit_pct_per_trade = total_profit_pct / len(trades)
        
        # MDD 계산
        max_equity = max(equity_curve) if equity_curve else capital
        min_equity = min(equity_curve) if equity_curve else capital
        max_drawdown = max_equity - min_equity
        max_drawdown_pct = max_drawdown / max_equity * 100 if max_equity > 0 else 0.0
        
        # Sharpe Ratio 계산
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        _logger.info(
            "[BACKTEST] 로그+OHLCV 기반 결과: %d거래, 승률=%.1f%%, 수익=%.2fpt",
            len(trades), win_rate * 100, total_profit
        )
        
        return BacktestResult(
            total_trades=len(trades),
            win_trades=win_trades,
            loss_trades=loss_trades,
            win_rate=win_rate,
            total_profit=total_profit,
            total_profit_pct=total_profit_pct,
            avg_profit_per_trade=avg_profit_per_trade,
            avg_profit_pct_per_trade=avg_profit_pct_per_trade,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            trades=trades,
            equity_curve=equity_curve
        )

    
    def print_results(self, result: BacktestResult) -> None:
        """백테스팅 결과 출력."""
        print("\n" + "="*60)
        print("백테스팅 결과")
        print("="*60)
        print(f"총 거래: {result.total_trades}")
        print(f"승리: {result.win_trades} | 패배: {result.loss_trades}")
        print(f"승률: {result.win_rate:.2%}")
        print(f"총 수익: {result.total_profit:,.0f}원 ({result.total_profit_pct:.2f}%)")
        print(f"평균 수익/거래: {result.avg_profit_per_trade:,.0f}원 ({result.avg_profit_pct_per_trade:.2f}%)")
        print(f"최대 낙폭 (MDD): {result.max_drawdown_pct:.2f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print("="*60)

    def save_results_to_json(
        self,
        result: BacktestResult,
        output_dir: Optional[Path] = None,
        filename: Optional[str] = None
    ) -> Path:
        """백테스팅 전체 결과를 JSON으로 저장.

        Args:
            result: 백테스팅 결과
            output_dir: 출력 디렉토리 (기본: logs/backtest/results)
            filename: 파일명 (기본: backtest_full_YYYYMMDD_HHMMSS.json)

        Returns:
            저장된 파일 경로
        """
        if output_dir is None:
            output_dir = Path("logs/backtest/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_full_{timestamp}.json"

        output_path = output_dir / filename

        # 결과를 딕셔너리로 변환
        result_dict = asdict(result)

        # Trade 객체의 datetime을 ISO 문자열로 변환
        result_dict["trades"] = [
            {
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "action": t.action,
                "exit_reason": t.exit_reason,
                "profit": t.profit,
                "profit_pct": t.profit_pct,
                "bars_held": t.bars_held
            }
            for t in result.trades
        ]

        # 저장
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        _logger.info("[BACKTEST] 전체 결과 저장 완료: %s", output_path)
        return output_path

    def save_trades_to_csv(
        self,
        result: BacktestResult,
        output_dir: Optional[Path] = None,
        filename: Optional[str] = None
    ) -> Path:
        """거래 기록을 CSV로 저장.

        Args:
            result: 백테스팅 결과
            output_dir: 출력 디렉토리 (기본: logs/backtest/results)
            filename: 파일명 (기본: backtest_trades_YYYYMMDD_HHMMSS.csv)

        Returns:
            저장된 파일 경로
        """
        if output_dir is None:
            output_dir = Path("logs/backtest/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_trades_{timestamp}.csv"

        output_path = output_dir / filename

        # 거래 기록을 DataFrame으로 변환
        trades_df = pd.DataFrame([
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "action": t.action,
                "exit_reason": t.exit_reason,
                "profit": t.profit,
                "profit_pct": t.profit_pct,
                "bars_held": t.bars_held
            }
            for t in result.trades
        ])

        # CSV로 저장
        trades_df.to_csv(output_path, index=False, encoding="utf-8")

        _logger.info("[BACKTEST] 거래 기록 저장 완료: %s", output_path)
        return output_path

    def save_summary_to_json(
        self,
        result: BacktestResult,
        output_dir: Optional[Path] = None,
        filename: Optional[str] = None
    ) -> Path:
        """요약 결과를 JSON으로 저장.

        Args:
            result: 백테스팅 결과
            output_dir: 출력 디렉토리 (기본: logs/backtest/results)
            filename: 파일명 (기본: backtest_summary_YYYYMMDD_HHMMSS.json)

        Returns:
            저장된 파일 경로
        """
        if output_dir is None:
            output_dir = Path("logs/backtest/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_summary_{timestamp}.json"

        output_path = output_dir / filename

        # 요약 결과
        summary = {
            "backtest_date": datetime.now().isoformat(),
            "total_trades": result.total_trades,
            "win_trades": result.win_trades,
            "loss_trades": result.loss_trades,
            "win_rate": result.win_rate,
            "total_profit": result.total_profit,
            "total_profit_pct": result.total_profit_pct,
            "avg_profit_per_trade": result.avg_profit_per_trade,
            "avg_profit_pct_per_trade": result.avg_profit_pct_per_trade,
            "max_drawdown": result.max_drawdown,
            "max_drawdown_pct": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "config": {
                "initial_capital": self.config.initial_capital,
                "tick_size": self.config.tick_size,
                "commission_rate": self.config.commission_rate,
                "slippage_ticks": self.config.slippage_ticks,
                "position_size_pct": self.config.position_size_pct,
                "stop_loss_atr_multiplier": self.config.stop_loss_atr_multiplier,
                "take_profit_atr_multiplier": self.config.take_profit_atr_multiplier,
                "trailing_stop_atr_multiplier": self.config.trailing_stop_atr_multiplier
            }
        }

        # 저장
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        _logger.info("[BACKTEST] 요약 결과 저장 완료: %s", output_path)
        return output_path

    def save_all(
        self,
        result: BacktestResult,
        output_dir: Optional[Path] = None,
        timestamp: Optional[str] = None
    ) -> Dict[str, Path]:
        """모든 백테스팅 결과를 저장 (전체, 거래, 요약).

        Args:
            result: 백테스팅 결과
            output_dir: 출력 디렉토리 (기본: logs/backtest/results)
            timestamp: 타임스탬프 (기본: 현재 시간)

        Returns:
            저장된 파일 경로 딕셔너리 {"full": path, "trades": path, "summary": path}
        """
        if output_dir is None:
            output_dir = Path("logs/backtest/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 각 파일명 생성
        full_filename = f"backtest_full_{timestamp}.json"
        trades_filename = f"backtest_trades_{timestamp}.csv"
        summary_filename = f"backtest_summary_{timestamp}.json"

        # 저장
        full_path = self.save_results_to_json(result, output_dir, full_filename)
        trades_path = self.save_trades_to_csv(result, output_dir, trades_filename)
        summary_path = self.save_summary_to_json(result, output_dir, summary_filename)

        _logger.info(
            "[BACKTEST] 모든 결과 저장 완료: full=%s, trades=%s, summary=%s",
            full_path, trades_path, summary_path
        )

        return {
            "full": full_path,
            "trades": trades_path,
            "summary": summary_path
        }


@dataclass
class BacktestPositionState:
    """백테스팅 포지션 상태."""
    position_id: str
    action: str  # "BUY" or "SELL"
    entry_price: float
    entry_time: datetime
    size: float
    confidence: str
    signal_reason: str
    stop_loss: float
    take_profit: float
    atr: float
    current_stop: float
    is_active: bool = True
    max_favorable_excursion: float = 0.0  # 최대 호재
    max_adverse_excursion: float = 0.0  # 최대 악재


class BacktestPositionManager:
    """백테스팅용 포지션 매니저 (실제 PositionTracker와 동일한 로직)."""
    
    def __init__(self, config: BacktestConfig):
        """초기화.
        
        Args:
            config: 백테스팅 설정
        """
        self.config = config
        self.positions: Dict[str, BacktestPositionState] = {}
        self._position_counter = 0
    
    def create_position(
        self,
        action: str,
        entry_price: float,
        size: float,
        confidence: str,
        signal_reason: str,
        stop_loss: float,
        take_profit: float,
        atr: float,
        entry_time: datetime
    ) -> str:
        """새 포지션 생성.
        
        Args:
            action: 액션 (BUY/SELL)
            entry_price: 진입 가격
            size: 사이즈
            confidence: 신뢰도
            signal_reason: 신호 이유
            stop_loss: 손절 가격
            take_profit: 이익실현 가격
            atr: ATR 값
            entry_time: 진입 시간
        
        Returns:
            포지션 ID
        """
        self._position_counter += 1
        position_id = f"pos_{self._position_counter}"
        
        position = BacktestPositionState(
            position_id=position_id,
            action=action,
            entry_price=entry_price,
            entry_time=entry_time,
            size=size,
            confidence=confidence,
            signal_reason=signal_reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
            current_stop=stop_loss
        )
        
        self.positions[position_id] = position
        _logger.info(
            "[BACKTEST_POS] 포지션 생성: %s %s @ %.2f (size=%.2f, conf=%s)",
            action, position_id, entry_price, size, confidence
        )
        
        return position_id
    
    def update_position(
        self,
        position_id: str,
        current_price: float,
        atr: float
    ) -> Optional[float]:
        """포지션 상태 업데이트 (트레일링 스탑).
        
        실제 PositionTracker.update_position()과 동일한 로직.
        
        Args:
            position_id: 포지션 ID
            current_price: 현재 가격
            atr: ATR 값
        
        Returns:
            새로운 스탑 가격 (변경 시), None (변경 없음)
        """
        if position_id not in self.positions:
            return None
        
        pos = self.positions[position_id]
        
        if not pos.is_active:
            return None
        
        # 트레일링 스탑 계산 (실제 로직과 동일)
        trailing_stop_multiplier = self.config.trailing_stop_atr_multiplier
        if pos.action == "BUY":
            new_stop = current_price - (atr * trailing_stop_multiplier)
            # 스탑은 이전보다 높아야 함 (익절 보호)
            new_stop = max(new_stop, pos.current_stop)
        else:  # SELL
            new_stop = current_price + (atr * trailing_stop_multiplier)
            # 스탑은 이전보다 낮아야 함 (익절 보호)
            new_stop = min(new_stop, pos.current_stop)
        
        # 스탑 변경 시 기록
        if new_stop != pos.current_stop:
            pos.current_stop = new_stop
            _logger.debug(
                "[BACKTEST_POS] 트레일링 스탑 업데이트: %s %.2f -> %.2f (price=%.2f)",
                position_id, pos.stop_loss, new_stop, current_price
            )
            return new_stop
        
        return None
    
    def update_excursions(self, position_id: str, current_price: float):
        """최대 호재/악재 업데이트.
        
        Args:
            position_id: 포지션 ID
            current_price: 현재 가격
        """
        if position_id not in self.positions:
            return
        
        pos = self.positions[position_id]
        
        if pos.action == "BUY":
            favorable = current_price - pos.entry_price
            adverse = pos.entry_price - current_price
        else:  # SELL
            favorable = pos.entry_price - current_price
            adverse = current_price - pos.entry_price
        
        pos.max_favorable_excursion = max(pos.max_favorable_excursion, favorable)
        pos.max_adverse_excursion = max(pos.max_adverse_excursion, adverse)
    
    def should_exit(
        self,
        position_id: str,
        current_price: float
    ) -> tuple[bool, Optional[str]]:
        """청산 여부 판단.
        
        실제 PositionTracker.should_exit()과 동일한 로직.
        
        Args:
            position_id: 포지션 ID
            current_price: 현재 가격
        
        Returns:
            (청산 여부, 청산 사유)
        """
        if position_id not in self.positions:
            return False, None
        
        pos = self.positions[position_id]
        
        if not pos.is_active:
            return False, None
        
        # 손절/이익실현 체크 (실제 로직과 동일)
        if pos.action == "BUY":
            if current_price <= pos.current_stop:
                return True, "stop_loss"
            elif current_price >= pos.take_profit:
                return True, "take_profit"
        else:  # SELL
            if current_price >= pos.current_stop:
                return True, "stop_loss"
            elif current_price <= pos.take_profit:
                return True, "take_profit"
        
        return False, None
    
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str
    ) -> Optional[BacktestPositionState]:
        """포지션 청산.
        
        Args:
            position_id: 포지션 ID
            exit_price: 청산 가격
            reason: 청산 사유
        
        Returns:
            청산된 포지션 상태
        """
        if position_id not in self.positions:
            return None
        
        pos = self.positions[position_id]
        pos.is_active = False
        
        _logger.info(
            "[BACKTEST_POS] 포지션 청산: %s @ %.2f (reason=%s)",
            position_id, exit_price, reason
        )
        
        return pos
    
    def get_position(self, position_id: str) -> Optional[BacktestPositionState]:
        """포지션 상태 조회."""
        return self.positions.get(position_id)
    
    def get_active_positions(self) -> List[BacktestPositionState]:
        """활성 포지션 리스트 반환."""
        return [pos for pos in self.positions.values() if pos.is_active]
