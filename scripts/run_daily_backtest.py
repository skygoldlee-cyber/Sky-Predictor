"""장마감 후 당일 매매 백테스트 자동 실행 스크립트.

JIF 장마감 이벤트 수신 후 호출되어 당일 매매 백테스트를 실행합니다.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prediction.backtest.backtest_pivot_signals import PivotSignalBacktester, BacktestConfig
from prediction.pivot_parameter_db import PivotParameterDB, WalkForwardEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_daily_backtest(
    initial_capital: float = 10000000.0,
    output_dir: Optional[Path] = None
) -> bool:
    """당일 매매 백테스트 실행.

    Args:
        initial_capital: 초기 자본 (기본 1000만원)
        output_dir: 결과 저장 디렉토리 (기본: logs/backtest/results)

    Returns:
        실행 성공 여부
    """
    try:
        logger.info("[DAILY_BACKTEST] 당일 매매 백테스트 시작")

        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = Path("logs/backtest/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) 거래 로그 파일 경로 확인
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = Path("logs/trades") / f"trades_{today}.jsonl"

        if not log_file.exists():
            logger.warning("[DAILY_BACKTEST] 거래 로그 파일 없음: %s", log_file)
            return False

        # 2) 백테스터 생성
        config = BacktestConfig(initial_capital=initial_capital)
        backtester = PivotSignalBacktester(config)

        # 3) 로그 기반 백테스팅 실행
        logger.info("[DAILY_BACKTEST] 로그 기반 백테스팅 실행: %s", log_file)
        results = backtester.run_backtest_from_logs(log_file, initial_capital=initial_capital)

        # 4) 결과 출력
        backtester.print_results(results)

        # 5) 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = backtester.save_all(results, output_dir=output_dir, timestamp=timestamp)

        logger.info("[DAILY_BACKTEST] 백테스트 완료")
        logger.info("[DAILY_BACKTEST] 전체 결과: %s", paths["full"])
        logger.info("[DAILY_BACKTEST] 거래 기록: %s", paths["trades"])
        logger.info("[DAILY_BACKTEST] 요약 결과: %s", paths["summary"])

        # 6) 세션 파라미터 DB 저장 (백테스트 결과 기반)
        try:
            save_session_to_db_from_backtest(results, None, backtester.config, today)
        except Exception as e:
            logger.warning("[DAILY_BACKTEST] 세션 파라미터 DB 저장 실패: %s", e)

        # 7) 트랜스포머 품질 요약 로그 출력
        try:
            _pred = globals().get("_active_pipeline")
            if _pred is not None and callable(getattr(_pred, "log_quality_summary", None)):
                _pred.log_quality_summary()
        except Exception as _qe:
            logger.debug("[DAILY_BACKTEST] 품질 요약 출력 실패(무시): %s", _qe)

        return True

    except Exception as e:
        logger.exception("[DAILY_BACKTEST] 백테스트 실행 실패: %s", e)
        return False


def run_daily_backtest_with_ohlcv(
    initial_capital: float = 10000000.0,
    output_dir: Optional[Path] = None
) -> bool:
    """당일 매매 백테스트 실행 (OHLCV 데이터 포함).

    Args:
        initial_capital: 초기 자본 (기본 1000만원)
        output_dir: 결과 저장 디렉토리 (기본: logs/backtest/results)

    Returns:
        실행 성공 여부
    """
    try:
        logger.info("[DAILY_BACKTEST] 당일 매매 백테스트 시작 (OHLCV 포함)")

        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = Path("logs/backtest/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) 거래 로그 파일 경로 확인
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = Path("logs/trades") / f"trades_{today}.jsonl"

        if not log_file.exists():
            logger.warning("[DAILY_BACKTEST] 거래 로그 파일 없음: %s", log_file)
            return False

        # 2) OHLCV 데이터 파일 경로 확인 (futures 우선)
        today_str = datetime.now().strftime("%Y-%m-%d")
        ohlcv_file = Path("data/backtesting/futures") / datetime.now().strftime("%Y") / f"{today_str}_futures_1m.csv"

        if not ohlcv_file.exists():
            logger.warning("[DAILY_BACKTEST] OHLCV 데이터 파일 없음: %s", ohlcv_file)
            # OHLCV 없으면 일반 로그 기반 백테스트로 폴백
            return run_daily_backtest(initial_capital, output_dir)

        # 3) OHLCV 데이터 로드
        df = pd.read_csv(ohlcv_file, index_col=0, parse_dates=True)
        logger.info("[DAILY_BACKTEST] OHLCV 데이터 로드 완료: %d 봉", len(df))

        # 4) 백테스터 생성
        config = BacktestConfig(initial_capital=initial_capital)
        backtester = PivotSignalBacktester(config)

        # 5) 로그 + OHLCV 기반 백테스팅 실행
        logger.info("[DAILY_BACKTEST] 로그 + OHLCV 기반 백테스팅 실행")
        results = backtester.run_backtest_from_logs_with_ohlcv(log_file, df, initial_capital=initial_capital)

        # 6) 결과 출력
        backtester.print_results(results)

        # 7) 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = backtester.save_all(results, output_dir=output_dir, timestamp=timestamp)

        logger.info("[DAILY_BACKTEST] 백테스트 완료")
        logger.info("[DAILY_BACKTEST] 전체 결과: %s", paths["full"])
        logger.info("[DAILY_BACKTEST] 거래 기록: %s", paths["trades"])
        logger.info("[DAILY_BACKTEST] 요약 결과: %s", paths["summary"])

        # 8) 세션 파라미터 DB 저장 (백테스트 결과 기반)
        try:
            save_session_to_db_from_backtest(results, df, backtester.config, today_str)
        except Exception as e:
            logger.warning("[DAILY_BACKTEST] 세션 파라미터 DB 저장 실패: %s", e)

        return True

    except Exception as e:
        logger.exception("[DAILY_BACKTEST] 백테스트 실행 실패: %s", e)
        return False


def save_session_to_db_from_backtest(
    results: dict,
    df: Optional[pd.DataFrame],
    config: BacktestConfig,
    date: str,
) -> None:
    """백테스트 결과를 세션 파라미터로 DB에 저장."""
    db = PivotParameterDB("data/pivot_parameters.db")
    
    try:
        # 성능 메트릭 추출 (백테스트 결과에서)
        total_trades = results.get("total_trades", 0)
        winning_trades = results.get("winning_trades", 0)
        losing_trades = results.get("losing_trades", 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # bars_held에서 lag 메트릭 계산 (분 단위 → 봉 단위 변환)
        trades = results.get("trades", [])
        lag_values = []
        for trade in trades:
            if hasattr(trade, 'bars_held') and trade.bars_held is not None:
                # 분 단위를 봉 단위로 변환 (1분 봉 기준)
                lag_bars = trade.bars_held  # 이미 분 단위
                lag_values.append(lag_bars)
        
        avg_lag_bars = float(np.mean(lag_values)) if lag_values else 0.0
        lag_p95_bars = float(np.percentile(lag_values, 95)) if len(lag_values) > 0 else 0.0
        
        # pivot_quality_score 계산
        # 백테스트 거래 결과에서 피봇 품질 추정
        # 승률이 높고 평균 수익률이 양수이면 품질이 높다고 판단
        pivot_quality_score = 0.0
        if total_trades > 0:
            # 기본 품질 = 승률
            quality = win_rate
            # 수익성 보정: 평균 수익률이 양수면 품질 상향
            avg_profit_pct = results.get("avg_profit_pct_per_trade", 0.0)
            if avg_profit_pct > 0:
                quality = min(quality + (avg_profit_pct / 100.0), 1.0)
            pivot_quality_score = quality
        
        # alternation_rate 계산 (백테스트 거래의 BUY/SELL 교번 확인)
        alternation_rate = 0.5
        if len(trades) >= 2:
            violations = 0
            for i in range(1, len(trades)):
                if trades[i].action == trades[i-1].action:
                    violations += 1
            alternation_rate = 1.0 - (violations / len(trades))
        
        # avg_wave_size_pct 및 avg_wave_atr_ratio 계산 (OHLCV 데이터 필요)
        avg_wave_size_pct = 0.0
        avg_wave_atr_ratio = 0.0
        if df is not None and len(df) > 0 and 'Close' in df.columns:
            # 파동 크기 계산: 고점-저점 변동폭의 평균 퍼센트
            if 'High' in df.columns and 'Low' in df.columns:
                wave_sizes = []
                for i in range(1, len(df)):
                    high = df['High'].iloc[i]
                    low = df['Low'].iloc[i]
                    prev_close = df['Close'].iloc[i-1]
                    if prev_close > 0:
                        wave_pct = abs(high - low) / prev_close * 100.0
                        wave_sizes.append(wave_pct)
                avg_wave_size_pct = float(np.mean(wave_sizes)) if wave_sizes else 0.0
            
            # ATR 계산 및 파동 비율
            if 'High' in df.columns and 'Low' in df.columns:
                tr_values = []
                for i in range(1, len(df)):
                    high = df['High'].iloc[i]
                    low = df['Low'].iloc[i]
                    prev_close = df['Close'].iloc[i-1]
                    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                    tr_values.append(tr)
                
                if tr_values:
                    # Wilder's RMA로 ATR 계산 (14기간)
                    atr_period = 14
                    atr_values = []
                    alpha = 1.0 / atr_period
                    prev_atr = tr_values[0]
                    atr_values.append(prev_atr)
                    for tr in tr_values[1:]:
                        curr_atr = alpha * tr + (1 - alpha) * prev_atr
                        atr_values.append(curr_atr)
                        prev_atr = curr_atr
                    
                    avg_atr = float(np.mean(atr_values[-atr_period:])) if len(atr_values) >= atr_period else float(np.mean(atr_values))
                    
                    # 파동 크기 대비 ATR 비율
                    if avg_atr > 0 and wave_sizes:
                        avg_wave_abs = float(np.mean([df['High'].iloc[i] - df['Low'].iloc[i] for i in range(1, len(df))]))
                        avg_wave_atr_ratio = avg_wave_abs / avg_atr
        
        # 백테스트 결과를 세션 파라미터 형식으로 변환
        performance_metrics = {
            "bar_count": len(df) if df is not None else 0,
            "total_pivots": total_trades,
            "confirmed_pivots": winning_trades,
            "cancelled_pivots": losing_trades,
            "pivot_confirmation_rate": win_rate,
            "avg_lag_bars": avg_lag_bars,  # 백테스트 거래 보유 기간에서 계산
            "lag_p95_bars": lag_p95_bars,
            "pivot_quality_score": pivot_quality_score,  # 백테스트 승률 + 수익성 기반 품질 점수
            "alternation_rate": alternation_rate,  # 백테스트 거래 BUY/SELL 교번 비율
            "avg_wave_size_pct": avg_wave_size_pct,  # OHLCV 데이터에서 계산
            "avg_wave_atr_ratio": avg_wave_atr_ratio,  # OHLCV 데이터에서 계산
            "false_pivot_rate": losing_trades / total_trades if total_trades > 0 else 0.0,
        }
        
        # 시장 상태 (임시: unknown)
        market_state = {
            "dominant_regime": "unknown",
            "regime_stability": 0.5,
            "avg_atr": 0.0,
            "avg_er": 0.0,
            "atr_percentile": 0.5,
        }
        
        # 파라미터 설정 (config에서 추출)
        param_config = {
            "atr_multiplier": 1.5,  # 기본값
            "base_pct": 0.30,
            "atr_weight": 0.50,
            "base_multiplier": 2.0,
            "confirmation_bars": 2,
            "er_period": 10,
            "min_wave_pct": 0.15,
        }
        
        db.save_session_parameters(
            date=date,
            session_label="full",
            symbol="KP200 선물",
            indicator_type="hybrid_adaptive_pivot",
            config=param_config,
            performance_metrics=performance_metrics,
            market_state=market_state,
            time_start="09:00",
            time_end="15:30",
        )
        
        logger.info("[DAILY_BACKTEST] 세션 파라미터 DB 저장 완료: %s", date)
        
    finally:
        db.close()


def main():
    """메인 함수."""
    import argparse

    parser = argparse.ArgumentParser(description="장마감 후 당일 매매 백테스트 실행")
    parser.add_argument("--with-ohlcv",    action="store_true", help="OHLCV 데이터 포함 백테스팅")
    parser.add_argument("--capital",       type=float, default=10000000.0, help="초기 자본 (기본 1000만원)")
    parser.add_argument("--output-dir",    type=str,   default="logs/backtest/results", help="결과 저장 디렉토리")
    # ── Walk-forward 평가 옵션 (신규) ──────────────────────────────────────
    parser.add_argument("--evaluate",      action="store_true", help="백테스트 후 Walk-forward 평가 실행")
    parser.add_argument("--eval-lookback", type=int,   default=30, help="평가 훈련 윈도우 (일, 기본 30)")
    parser.add_argument("--eval-test-days",type=int,   default=20, help="평가 슬라이딩 횟수 (기본 20)")
    parser.add_argument("--eval-only",     action="store_true", help="백테스트 없이 평가만 실행")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # 평가만 실행
    if args.eval_only:
        run_walk_forward_evaluation(
            lookback_days=args.eval_lookback,
            test_days=args.eval_test_days,
        )
        sys.exit(0)

    # 백테스트 실행
    if args.with_ohlcv:
        success = run_daily_backtest_with_ohlcv(args.capital, output_dir)
    else:
        success = run_daily_backtest(args.capital, output_dir)

    # 백테스트 성공 후 Walk-forward 평가
    if success and args.evaluate:
        logger.info("[DAILY_BACKTEST] Walk-forward 평가 시작")
        run_walk_forward_evaluation(
            lookback_days=args.eval_lookback,
            test_days=args.eval_test_days,
            output_dir=Path("logs/evaluation"),
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward 평가 함수 (신규 추가)
# ─────────────────────────────────────────────────────────────────────────────

def run_walk_forward_evaluation(
    symbol: str = "KP200 선물",
    db_path: str = "data/pivot_parameters.db",
    lookback_days: int = 30,
    test_days: int = 20,
    output_dir: Optional[Path] = None,
) -> dict:
    """Walk-forward 평가를 실행하고 결과를 저장한다."""
    if output_dir is None:
        output_dir = Path("logs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    db = PivotParameterDB(db_path)
    try:
        evaluator = WalkForwardEvaluator(db, symbol)

        logger.info("[EVALUATION] Walk-forward 검증 시작 (lookback=%d, test=%d일)", lookback_days, test_days)
        wf_result = evaluator.run(lookback_days=lookback_days, test_days=test_days)

        logger.info("[EVALUATION] 레짐별 커버리지 측정")
        cov_result = evaluator.measure_coverage(lookback_days=lookback_days)

        full_result = {
            "timestamp":     datetime.now().isoformat(),
            "symbol":        symbol,
            "lookback_days": lookback_days,
            "test_days":     test_days,
            "walk_forward":  wf_result,
            "coverage":      cov_result,
        }

        today = datetime.now().strftime("%Y%m%d")
        json_path   = output_dir / f"walkforward_{today}.json"
        latest_path = output_dir / "walkforward_latest.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_result, f, ensure_ascii=False, indent=2)
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(full_result, f, ensure_ascii=False, indent=2)

        _print_evaluation_summary(full_result)
        logger.info("[EVALUATION] 결과 저장: %s", json_path)
        return full_result

    finally:
        db.close()


def _print_evaluation_summary(result: dict) -> None:
    """평가 결과 요약 로그 출력."""
    wf  = result.get("walk_forward", {})
    cov = result.get("coverage", {})

    verdict     = wf.get("verdict", "N/A")
    improvement = wf.get("avg_improvement_vs_fallback", 0.0)
    win_rate    = wf.get("win_rate_vs_fallback", 0.0)
    coverage    = cov.get("overall_coverage_rate", 0.0)
    stability   = wf.get("stability_cv", 0.0)

    lines = [
        "",
        "=" * 60,
        "  Walk-forward 평가 결과 요약",
        "=" * 60,
        f"  판정:              {verdict}",
        f"  평균 개선률:       {improvement:+.4f}  (목표 > +0.030)",
        f"  폴백 대비 승률:    {win_rate:.1%}",
        f"  레짐 커버리지:     {coverage:.1%}          (목표 >= 80%)",
        f"  파라미터 안정성:   CV={stability:.3f}       (목표 < 0.200)",
        "-" * 60,
    ]
    for regime, info in cov.get("regimes", {}).items():
        mark  = "OK" if info["covered"] else "NG"
        count = info["sample_count"]
        lines.append(f"  [{mark}] {regime:<22} {count:>3}샘플")
    lines += ["=" * 60, ""]
    for line in lines:
        logger.info(line)
