"""Optuna를 사용한 ZigZag 파라미터 최적화 스크립트.

ZigZag 파라미터를 Optuna를 사용하여 최적화합니다.

Usage:
    # 단순 버전 (min_wave_atr_ratio만)
    python scripts/optuna_zigzag_tuner.py --data data/backtesting/futures/2026/2026-05-03_futures_1m.csv --mode simple --n-trials 30

    # 고급 버전 (여러 파라미터)
    python scripts/optuna_zigzag_tuner.py --data data/backtesting/futures/2026/2026-05-03_futures_1m.csv --mode advanced --n-trials 60

    # 시간대별 테이블 최적화 (준비 중)
    python scripts/optuna_zigzag_tuner.py --data data/backtesting/futures/2026/2026-05-03_futures_1m.csv --mode session --n-trials 100
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional

import optuna

from prediction.zigzag_backtester import ZigZagBacktester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
_logger = logging.getLogger(__name__)


def main(now_fn: Optional[Callable[[], datetime]] = None):
    """메인 함수.

    Args:
        now_fn: 시간 함수 (테스트/백테스트용 주입 가능)
    """
    _now = now_fn if now_fn is not None else datetime.now
    parser = argparse.ArgumentParser(description="ZigZag 파라미터 Optuna 최적화")
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="OHLCV 데이터 파일 경로"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "advanced", "session"],
        default="simple",
        help="최적화 모드 (simple: min_wave_atr_ratio만, advanced: 여러 파라미터, session: 시간대별 테이블)"
    )
    
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Optuna 시도 횟수"
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="병렬 작업 수 (1=직렬, >1=병렬)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 파일 경로 (JSON)"
    )
    
    args = parser.parse_args()
    
    # 데이터 경로 확인
    data_path = Path(args.data)
    if not data_path.exists():
        _logger.error("데이터 파일 없음: %s", data_path)
        return
    
    # 백테스터 초기화
    _logger.info("백테스터 초기화 중...")
    backtester = ZigZagBacktester(data_path=data_path)
    
    # objective 함수 선택
    if args.mode == "simple":
        objective = backtester.objective_simple
        _logger.info("모드: simple (min_wave_atr_ratio만 최적화)")
    elif args.mode == "advanced":
        objective = backtester.objective_advanced
        _logger.info("모드: advanced (여러 파라미터 최적화)")
    elif args.mode == "session":
        _logger.error("session 모드는 아직 구현되지 않았습니다.")
        return
    else:
        _logger.error("알 수 없는 모드: %s", args.mode)
        return
    
    # Optuna study 생성
    _logger.info("Optuna study 생성 중...")
    study = optuna.create_study(direction="maximize")
    
    # 최적화 실행
    _logger.info("최적화 시작 (n_trials=%d, n_jobs=%d)...", args.n_trials, args.n_jobs)
    start_time = _now()
    
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )
    
    elapsed = (_now() - start_time).total_seconds()
    _logger.info("최적화 완료 (소요 시간: %.1f초)", elapsed)
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("최적화 결과")
    print("=" * 80)
    print(f"최적 파라미터: {study.best_params}")
    print(f"최적 점수: {study.best_value:.4f}")
    print(f"시도 횟수: {len(study.trials)}")
    print("=" * 80)
    
    # 상위 5개 결과 출력
    print("\n상위 5개 결과:")
    print("-" * 80)
    sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
    for i, trial in enumerate(sorted_trials, 1):
        print(f"#{i}: {trial.params} → 점수: {trial.value:.4f}")
    print("-" * 80)
    
    # 결과 저장
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        result = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "elapsed_seconds": elapsed,
            "timestamp": _now().isoformat(),
            "top_trials": [
                {
                    "params": t.params,
                    "value": t.value
                }
                for t in sorted_trials
            ]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        _logger.info("결과 저장 완료: %s", output_path)


if __name__ == "__main__":
    main()
