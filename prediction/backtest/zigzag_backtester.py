"""ZigZag 파라미터 백테스트 실행기.

ZigZag 파라미터를 백테스트하여 피봇 감지 성능을 평가합니다.
Optuna 등의 최적화 프레임워크와 함께 사용할 수 있습니다.

Usage:
    from prediction.zigzag_backtester import ZigZagBacktester
    from pathlib import Path

    backtester = ZigZagBacktester(data_path=Path("data/backtesting/futures/2026/2026-05-03_futures_1m.csv"))
    result = backtester.run_backtest(config)
"""

import pandas as pd
from indicators.adaptive_zigzag import AdaptiveZigZag, AdaptiveZigZagConfig
from typing import Dict, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


class ZigZagBacktester:
    """ZigZag 파라미터 백테스트 실행기."""
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        df: Optional[pd.DataFrame] = None
    ):
        """초기화.
        
        Args:
            data_path: OHLCV 데이터 파일 경로 (df와 둘 중 하나만 제공)
            df: OHLCV 데이터프레임 (data_path와 둘 중 하나만 제공)
        """
        if data_path is not None:
            self.df = self._load_data(data_path)
        elif df is not None:
            self.df = df.copy()
        else:
            raise ValueError("data_path 또는 df 중 하나는 제공해야 합니다.")
    
    def _load_data(self, data_path: Path) -> pd.DataFrame:
        """데이터 로드.
        
        Args:
            data_path: 데이터 파일 경로
            
        Returns:
            OHLCV 데이터프레임
        """
        if not data_path.exists():
            raise FileNotFoundError(f"데이터 파일 없음: {data_path}")
        
        # 파일 형식에 따라 로드
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        elif data_path.suffix == '.parquet':
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {data_path.suffix}")
        
        # 컬럼명 정규화 (소문자 → 대문자)
        col_map = {c: c.capitalize() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
        df = df.rename(columns=col_map)
        
        # 필수 컬럼 확인
        required_cols = ["High", "Low", "Close"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"필수 컬럼 누락: {missing_cols}")
        
        _logger.info("[BACKTESTER] 데이터 로드 완료: %d rows", len(df))
        return df
    
    def run_backtest(self, config: AdaptiveZigZagConfig) -> Dict:
        """백테스트 실행.
        
        Args:
            config: ZigZag 설정
            
        Returns:
            백테스트 결과 딕셔너리
            {
                "confirmed_count": 확정된 피봇 수,
                "unconfirmed_count": 미확정 피봇 수,
                "total_count": 전체 피봇 수,
                "confirm_rate": 확정율 (0~1)
            }
        """
        zz = AdaptiveZigZag(config)
        zz.set_backtest_mode(True)  # 백테스트 모드: look-ahead bias 방지
        
        confirmed_count = 0
        unconfirmed_count = 0
        pivot_details = []
        
        for idx, row in self.df.iterrows():
            state = zz.update(
                high=row["High"],
                low=row["Low"],
                close=row["Close"],
                bar_time=idx,
                open=row.get("Open", 0.0)
            )
            
            # 피봇 상태 추적
            if state.pivot_type is not None:
                if state.is_confirmed:
                    confirmed_count += 1
                    pivot_details.append({
                        "time": idx,
                        "type": state.pivot_type,
                        "price": state.pivot_price,
                        "confirmed": True
                    })
                else:
                    unconfirmed_count += 1
                    pivot_details.append({
                        "time": idx,
                        "type": state.pivot_type,
                        "price": state.pivot_price,
                        "confirmed": False
                    })
        
        total = confirmed_count + unconfirmed_count
        confirm_rate = confirmed_count / total if total > 0 else 0.0
        
        _logger.info(
            "[BACKTESTER] 백테스트 완료: 확정=%d, 미확정=%d, 확정율=%.2f%%",
            confirmed_count, unconfirmed_count, confirm_rate * 100
        )
        
        return {
            "confirmed_count": confirmed_count,
            "unconfirmed_count": unconfirmed_count,
            "total_count": total,
            "confirm_rate": confirm_rate,
            "pivot_details": pivot_details
        }
    
    def objective_simple(self, trial) -> float:
        """Optuna objective 함수 (단순 버전 - min_wave_atr_ratio만).
        
        Args:
            trial: Optuna trial 객체
            
        Returns:
            평가 점수 (높을수록 좋음)
        """
        # 파라미터 제안
        min_wave_atr_ratio = trial.suggest_float("min_wave_atr_ratio", 1.0, 4.0)
        
        # config 생성
        config = AdaptiveZigZagConfig()
        config.min_wave_atr_ratio = min_wave_atr_ratio
        config.use_atr_based_filtering = True
        
        # 백테스트 실행
        result = self.run_backtest(config)
        
        # 평가 점수 계산
        confirm_rate = result["confirm_rate"]
        # 피봇 수 패널티: 3~5개가 이상적
        count_penalty = abs(result["confirmed_count"] - 4) * 0.05
        score = confirm_rate - count_penalty
        
        return score
    
    def objective_advanced(self, trial) -> float:
        """Optuna objective 함수 (고급 버전 - 여러 파라미터).
        
        Args:
            trial: Optuna trial 객체
            
        Returns:
            평가 점수 (높을수록 좋음)
        """
        # 파라미터 제안
        min_wave_atr_ratio = trial.suggest_float("min_wave_atr_ratio", 1.0, 4.0)
        cluster_atr_ratio = trial.suggest_float("cluster_atr_ratio", 0.5, 3.0)
        min_wave_pct = trial.suggest_float("min_wave_pct", 0.1, 0.4)
        
        # config 생성
        config = AdaptiveZigZagConfig()
        config.min_wave_atr_ratio = min_wave_atr_ratio
        config.cluster_atr_ratio = cluster_atr_ratio
        config.min_wave_pct = min_wave_pct
        config.use_atr_based_filtering = True
        
        # 백테스트 실행
        result = self.run_backtest(config)
        
        # 평가 점수 계산
        confirm_rate = result["confirm_rate"]
        count_penalty = abs(result["confirmed_count"] - 4) * 0.05
        score = confirm_rate - count_penalty
        
        return score
