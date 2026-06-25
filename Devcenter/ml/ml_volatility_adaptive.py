"""
변동성 적응형 모델 구축
변동성 기반 스케일링, 클러스터링, 포지션 사이징
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


class VolatilityAdaptiveScaler:
    """변동성 적응형 스케일러"""
    
    def __init__(self):
        self.volatility_bins = None
        self.scalers = {}
        
    def fit(self, df: pd.DataFrame):
        """변동성 기반 스케일러 학습"""
        from sklearn.preprocessing import StandardScaler
        
        # ATR 기반 변동성 계산
        df['volatility'] = df['entry_atr'] / df['entry_close']
        
        # 변동성 분위수 기반 클러스터링
        volatility_quantiles = df['volatility'].quantile([0.33, 0.66])
        
        self.volatility_bins = [
            (0, volatility_quantiles[0.33]),      # 저변동성
            (volatility_quantiles[0.33], volatility_quantiles[0.66]),  # 중변동성
            (volatility_quantiles[0.66], float('inf'))  # 고변동성
        ]
        
        # 각 변동성 클러스터별 스케일러 학습
        for i, (low, high) in enumerate(self.volatility_bins):
            cluster_data = df[(df['volatility'] >= low) & (df['volatility'] < high)]
            
            if len(cluster_data) > 10:
                feature_cols = [
                    'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
                    'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
                    'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
                    'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
                ]
                
                X = cluster_data[feature_cols].fillna(0).values
                scaler = StandardScaler()
                scaler.fit(X)
                self.scalers[i] = scaler
                
                print(f"변동성 클러스터 {i} ({low:.4f}-{high:.4f}): {len(cluster_data)}건")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """변동성 기반 스케일링"""
        df['volatility'] = df['entry_atr'] / df['entry_close']
        
        feature_cols = [
            'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
            'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
            'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
            'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
        ]
        
        X = df[feature_cols].fillna(0).values
        X_scaled = np.zeros_like(X)
        
        for i, (low, high) in enumerate(self.volatility_bins):
            mask = (df['volatility'] >= low) & (df['volatility'] < high)
            
            if i in self.scalers and mask.sum() > 0:
                X_scaled[mask] = self.scalers[i].transform(X[mask])
            else:
                # 해당 스케일러가 없으면 기본 스케일링
                X_scaled[mask] = X[mask]
        
        return X_scaled
    
    def get_volatility_cluster(self, df: pd.DataFrame) -> np.ndarray:
        """변동성 클러스터 할당"""
        df['volatility'] = df['entry_atr'] / df['entry_close']
        
        clusters = np.zeros(len(df), dtype=int)
        
        for i, (low, high) in enumerate(self.volatility_bins):
            mask = (df['volatility'] >= low) & (df['volatility'] < high)
            clusters[mask] = i
        
        return clusters


class VolatilityAdaptiveModel:
    """변동성 적응형 모델"""
    
    def __init__(self):
        self.scaler = VolatilityAdaptiveScaler()
        self.models = {}
        
    def train(self, df: pd.DataFrame):
        """변동성 적응형 모델 학습"""
        import xgboost as xgb
        
        # 변동성 스케일러 학습
        self.scaler.fit(df)
        
        # 변동성 클러스터 할당
        clusters = self.scaler.get_volatility_cluster(df)
        
        # 각 클러스터별 모델 학습
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_data = df[cluster_mask].copy()
            
            if len(cluster_data) < 30:
                print(f"클러스터 {cluster_id}: 데이터 부족 ({len(cluster_data)}건) - 건너뜀")
                continue
            
            print(f"\n클러스터 {cluster_id} 모델 학습 ({len(cluster_data)}건)")
            
            # 시간 기반 분할
            cluster_data['year'] = pd.to_datetime(cluster_data['entry_time']).dt.year
            train_mask = (cluster_data['year'] >= 2019) & (cluster_data['year'] <= 2023)
            val_mask = (cluster_data['year'] == 2024)
            test_mask = (cluster_data['year'] >= 2025)
            
            feature_cols = [
                'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
                'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
                'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
                'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
            ]
            
            X = cluster_data[feature_cols].fillna(0).values
            y = cluster_data['is_win'].values
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            
            if len(X_train) < 20 or len(X_test) < 5:
                print(f"  데이터 부족 - 건너뜀")
                continue
            
            # XGBoost 모델 학습
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
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # 성과 평가
            y_pred = model.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            
            print(f"  테스트 정확도: {accuracy:.4f}")
            
            self.models[cluster_id] = model
        
        # 모델 저장
        joblib.dump(self.scaler, MODELS_DIR / "volatility_scaler.pkl")
        for cluster_id, model in self.models.items():
            joblib.dump(model, MODELS_DIR / f"volatility_model_cluster_{cluster_id}.pkl")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """변동성 적응형 예측"""
        # 변동성 클러스터 할당
        clusters = self.scaler.get_volatility_cluster(df)
        
        feature_cols = [
            'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
            'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
            'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
            'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
        ]
        
        X = df[feature_cols].fillna(0).values
        predictions = np.zeros(len(df))
        
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            
            if cluster_id in self.models:
                model = self.models[cluster_id]
                predictions[cluster_mask] = model.predict_proba(X[cluster_mask])[:, 1]
            else:
                # 해당 클러스터 모델이 없으면 기본 예측
                predictions[cluster_mask] = 0.5
        
        return predictions


class VolatilityAdjustedPositionSizing:
    """변동성 조정 포지션 사이징"""
    
    def __init__(self, base_position_size=1.0):
        self.base_position_size = base_position_size
        self.volatility_adjustment_factor = 0.5  # 변동성 조정 계수
        
    def calculate_position_size(self, df: pd.DataFrame) -> np.ndarray:
        """변동성 기반 포지션 사이징"""
        df['volatility'] = df['entry_atr'] / df['entry_close']
        
        # 변동성 중앙값 기준 조정
        median_volatility = df['volatility'].median()
        
        # 변동성이 높으면 포지션 축소, 낮으면 확대
        position_sizes = self.base_position_size * (median_volatility / (df['volatility'] + 1e-8)) ** self.volatility_adjustment_factor
        
        # 포지션 사이즈 제한 (0.5 ~ 2.0 배)
        position_sizes = np.clip(position_sizes, 0.5, 2.0)
        
        return position_sizes.values


def evaluate_volatility_adaptive_model(model: VolatilityAdaptiveModel, df: pd.DataFrame):
    """변동성 적응형 모델 성과 평가"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # 테스트 데이터 분할 (2025-2026)
    test_mask = (pd.to_datetime(df['entry_time']).dt.year >= 2025)
    test_df = df[test_mask].copy()
    
    if len(test_df) == 0:
        print("테스트 데이터 부족")
        return
    
    # 예측
    y_pred_proba = model.predict(test_df)
    y_pred = (y_pred_proba > 0.5).astype(int)
    y_test = test_df['is_win'].values
    
    # 성과 평가
    result = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(y_test) > 1 else 0
    }
    
    print(f"\n변동성 적응형 모델 성과 (2025-2026):")
    print(f"  정확도: {result['accuracy']:.4f}")
    print(f"  정밀도: {result['precision']:.4f}")
    print(f"  재현율: {result['recall']:.4f}")
    print(f"  F1 점수: {result['f1']:.4f}")
    print(f"  ROC AUC: {result['roc_auc']:.4f}")
    
    # 변동성 클러스터 분포
    clusters = model.scaler.get_volatility_cluster(test_df)
    print(f"\n변동성 클러스터 분포:")
    unique, counts = np.unique(clusters, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"  클러스터 {cluster}: {count}건 ({count/len(clusters)*100:.1f}%)")
    
    return result


def main():
    """메인 함수"""
    print("=" * 80)
    print("변동성 적응형 모델 구축")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 변동성 분석
    df['volatility'] = df['entry_atr'] / df['entry_close']
    print(f"\n변동성 통계:")
    print(f"  평균: {df['volatility'].mean():.4f}")
    print(f"  표준편차: {df['volatility'].std():.4f}")
    print(f"  최소: {df['volatility'].min():.4f}")
    print(f"  최대: {df['volatility'].max():.4f}")
    print(f"  중앙값: {df['volatility'].median():.4f}")
    
    # 연도별 변동성
    print(f"\n연도별 변동성:")
    for year in sorted(df['entry_time'].dt.year.unique()):
        year_data = df[df['entry_time'].dt.year == year]
        print(f"  {year}년: {year_data['volatility'].mean():.4f}")
    
    # 변동성 적응형 모델 학습
    print("\n" + "=" * 80)
    print("변동성 적응형 모델 학습")
    print("=" * 80)
    model = VolatilityAdaptiveModel()
    model.train(df)
    
    # 성과 평가
    result = evaluate_volatility_adaptive_model(model, df)
    
    # 포지션 사이징 테스트
    print("\n" + "=" * 80)
    print("변동성 조정 포지션 사이징")
    print("=" * 80)
    position_sizer = VolatilityAdjustedPositionSizing()
    
    test_mask = (pd.to_datetime(df['entry_time']).dt.year >= 2025)
    test_df = df[test_mask].copy()
    
    position_sizes = position_sizer.calculate_position_size(test_df)
    
    print(f"\n포지션 사이징 통계 (2025-2026):")
    print(f"  평균: {position_sizes.mean():.4f}")
    print(f"  표준편차: {position_sizes.std():.4f}")
    print(f"  최소: {position_sizes.min():.4f}")
    print(f"  최대: {position_sizes.max():.4f}")
    
    print(f"\n변동성 적응형 모델 구축 완료")


if __name__ == "__main__":
    main()
