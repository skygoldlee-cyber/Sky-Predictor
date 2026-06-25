"""
샘플 가중치 조정 시스템
시간 기반, 변동성 기반, 성과 기반 가중치 조정
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import joblib

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "ml_data"
MODELS_DIR = Path(__file__).parent / "ml_models"


class SampleWeightCalculator:
    """샘플 가중치 계산기"""
    
    def __init__(self, weight_type='time_based'):
        self.weight_type = weight_type
        self.weight_params = {}
        
    def calculate_time_based_weights(self, df: pd.DataFrame, decay_factor=0.95) -> np.ndarray:
        """시간 기반 가중치 (최근 데이터에 더 높은 가중치)"""
        df['year'] = pd.to_datetime(df['entry_time']).dt.year
        years = sorted(df['year'].unique())
        
        weights = np.zeros(len(df))
        
        for i, year in enumerate(years):
            year_mask = df['year'] == year
            # 최근 연도에 더 높은 가중치
            weights[year_mask] = decay_factor ** (len(years) - 1 - i)
        
        # 정규화
        weights = weights / weights.sum() * len(weights)
        
        return weights
    
    def calculate_volatility_based_weights(self, df: pd.DataFrame, high_vol_weight=1.5) -> np.ndarray:
        """변동성 기반 가중치 (고변동성 기간에 더 높은 가중치)"""
        df['volatility'] = df['entry_atr'] / df['entry_close']
        
        # 변동성 분위수 기반 가중치
        volatility_quantiles = df['volatility'].quantile([0.33, 0.66])
        
        weights = np.ones(len(df))
        
        # 고변동성 (상위 33%)
        high_vol_mask = df['volatility'] >= volatility_quantiles[0.66]
        weights[high_vol_mask] = high_vol_weight
        
        # 중변동성 (중간 33%)
        mid_vol_mask = (df['volatility'] >= volatility_quantiles[0.33]) & (df['volatility'] < volatility_quantiles[0.66])
        weights[mid_vol_mask] = 1.0
        
        # 저변동성 (하위 33%)
        low_vol_mask = df['volatility'] < volatility_quantiles[0.33]
        weights[low_vol_mask] = 0.8
        
        # 정규화
        weights = weights / weights.sum() * len(weights)
        
        return weights
    
    def calculate_performance_based_weights(self, df: pd.DataFrame, good_period_weight=1.3) -> np.ndarray:
        """성과 기반 가중치 (좋은 성과 기간에 더 높은 가중치)"""
        # 연도별 승률 계산
        df['year'] = pd.to_datetime(df['entry_time']).dt.year
        yearly_winrate = df.groupby('year')['is_win'].mean()
        
        weights = np.ones(len(df))
        
        for year in df['year'].unique():
            year_mask = df['year'] == year
            winrate = yearly_winrate[year]
            
            # 승률이 높은 연도에 더 높은 가중치
            if winrate > 0.55:
                weights[year_mask] = good_period_weight
            elif winrate < 0.45:
                weights[year_mask] = 0.7
            else:
                weights[year_mask] = 1.0
        
        # 정규화
        weights = weights / weights.sum() * len(weights)
        
        return weights
    
    def calculate_difficulty_based_weights(self, df: pd.DataFrame, hard_sample_weight=1.5) -> np.ndarray:
        """어려운 샘플 가중치 (잘못 예측된 샘플에 더 높은 가중치)"""
        # 이 예시에서는 단순히 승/패 기반 가중치
        # 실제로는 모델 예측 오류 기반 가중치가 필요
        
        weights = np.ones(len(df))
        
        # 패배한 거래에 더 높은 가중치 (학습 개선)
        lose_mask = df['is_win'] == 0
        weights[lose_mask] = hard_sample_weight
        
        # 정규화
        weights = weights / weights.sum() * len(weights)
        
        return weights
    
    def calculate_combined_weights(self, df: pd.DataFrame, 
                                  time_weight=0.3, 
                                  vol_weight=0.3, 
                                  perf_weight=0.2, 
                                  diff_weight=0.2) -> np.ndarray:
        """결합 가중치 (다양한 가중치 조합)"""
        time_weights = self.calculate_time_based_weights(df)
        vol_weights = self.calculate_volatility_based_weights(df)
        perf_weights = self.calculate_performance_based_weights(df)
        diff_weights = self.calculate_difficulty_based_weights(df)
        
        # 가중 평균
        combined_weights = (
            time_weight * time_weights +
            vol_weight * vol_weights +
            perf_weight * perf_weights +
            diff_weight * diff_weights
        )
        
        # 정규화
        combined_weights = combined_weights / combined_weights.sum() * len(combined_weights)
        
        return combined_weights


class WeightedModelTrainer:
    """가중치 조정 모델 트레이너"""
    
    def __init__(self):
        self.weight_calculator = SampleWeightCalculator()
        
    def train_xgboost_with_weights(self, df: pd.DataFrame, weight_type='combined'):
        """가중치 조정 XGBoost 학습"""
        import xgboost as xgb
        
        # 가중치 계산
        if weight_type == 'time':
            sample_weights = self.weight_calculator.calculate_time_based_weights(df)
        elif weight_type == 'volatility':
            sample_weights = self.weight_calculator.calculate_volatility_based_weights(df)
        elif weight_type == 'performance':
            sample_weights = self.weight_calculator.calculate_performance_based_weights(df)
        elif weight_type == 'difficulty':
            sample_weights = self.weight_calculator.calculate_difficulty_based_weights(df)
        else:  # combined
            sample_weights = self.weight_calculator.calculate_combined_weights(df)
        
        # 피처 준비
        feature_cols = [
            'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
            'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
            'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
            'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
        ]
        
        X = df[feature_cols].fillna(0).values
        y = df['is_win'].values
        
        # 시간 기반 분할
        df['year'] = pd.to_datetime(df['entry_time']).dt.year
        train_mask = (df['year'] >= 2019) & (df['year'] <= 2023)
        val_mask = (df['year'] == 2024)
        test_mask = (df['year'] >= 2025)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        sample_weights_train = sample_weights[train_mask]
        
        # 가중치 분석
        print(f"\n샘플 가중치 분석 ({weight_type}):")
        print(f"  평균: {sample_weights_train.mean():.4f}")
        print(f"  표준편차: {sample_weights_train.std():.4f}")
        print(f"  최소: {sample_weights_train.min():.4f}")
        print(f"  최대: {sample_weights_train.max():.4f}")
        
        # XGBoost 모델 학습 (가중치 적용)
        model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            reg_alpha=0.5,
            reg_lambda=2.0
        )
        
        model.fit(X_train, y_train, sample_weight=sample_weights_train, 
                 eval_set=[(X_val, y_val)], verbose=False)
        
        # 성과 평가
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        print(f"테스트 정확도: {accuracy:.4f}")
        
        # 모델 저장
        joblib.dump(model, MODELS_DIR / f"xgboost_weighted_{weight_type}.pkl")
        
        return model, accuracy


def compare_weighting_methods(df: pd.DataFrame):
    """다양한 가중치 방법 비교"""
    print("=" * 80)
    print("샘플 가중치 방법 비교")
    print("=" * 80)
    
    trainer = WeightedModelTrainer()
    
    weight_types = ['time', 'volatility', 'performance', 'difficulty', 'combined']
    results = {}
    
    for weight_type in weight_types:
        print(f"\n{weight_type} 가중치:")
        model, accuracy = trainer.train_xgboost_with_weights(df, weight_type)
        results[weight_type] = accuracy
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("가중치 방법 비교 결과")
    print("=" * 80)
    for weight_type, accuracy in results.items():
        print(f"{weight_type:15s}: {accuracy:.4f}")
    
    # 최적 방법 선택
    best_method = max(results, key=results.get)
    print(f"\n최적 가중치 방법: {best_method} (정확도: {results[best_method]:.4f})")
    
    return results


def main():
    """메인 함수"""
    print("=" * 80)
    print("샘플 가중치 조정 시스템")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 연도별 승률 분석
    df['year'] = pd.to_datetime(df['entry_time']).dt.year
    yearly_stats = df.groupby('year').agg({
        'is_win': ['mean', 'count']
    })
    
    print(f"\n연도별 승률:")
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]
        winrate = year_data['is_win'].mean()
        count = len(year_data)
        print(f"  {year}년: {winrate:.2%} ({count}건)")
    
    # 가중치 방법 비교
    results = compare_weighting_methods(df)
    
    print(f"\n샘플 가중치 조정 시스템 구축 완료")


if __name__ == "__main__":
    main()
