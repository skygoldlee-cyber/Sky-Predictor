"""
최적화된 ML 파이프라인
과적합 방지 강화, 변동성 기반 샘플 가중치, Time Series Split 교차 검증 통합
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


class OptimizedMLPipeline:
    """최적화된 ML 파이프라인"""
    
    def __init__(self):
        self.models = {}
        self.weight_calculator = None
        
    def calculate_volatility_weights(self, df: pd.DataFrame) -> np.ndarray:
        """변동성 기반 샘플 가중치 계산"""
        df['volatility'] = df['entry_atr'] / df['entry_close']
        
        # 변동성 분위수 기반 가중치
        volatility_quantiles = df['volatility'].quantile([0.33, 0.66])
        
        weights = np.ones(len(df))
        
        # 고변동성 (상위 33%)
        high_vol_mask = df['volatility'] >= volatility_quantiles[0.66]
        weights[high_vol_mask] = 1.5
        
        # 중변동성 (중간 33%)
        mid_vol_mask = (df['volatility'] >= volatility_quantiles[0.33]) & (df['volatility'] < volatility_quantiles[0.66])
        weights[mid_vol_mask] = 1.0
        
        # 저변동성 (하위 33%)
        low_vol_mask = df['volatility'] < volatility_quantiles[0.33]
        weights[low_vol_mask] = 0.8
        
        # 정규화
        weights = weights / weights.sum() * len(weights)
        
        return weights
    
    def train_xgboost_optimized(self, df: pd.DataFrame):
        """최적화된 XGBoost 학습 (과적합 방지 + 변동성 가중치)"""
        import xgboost as xgb
        
        # 변동성 기반 가중치
        sample_weights = self.calculate_volatility_weights(df)
        
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
        
        print(f"\n최적화된 XGBoost 학습:")
        print(f"  훈련: {len(X_train)}건 (승률: {y_train.mean():.2%})")
        print(f"  검증: {len(X_val)}건 (승률: {y_val.mean():.2%})")
        print(f"  테스트: {len(X_test)}건 (승률: {y_test.mean():.2%})")
        print(f"  가중치 평균: {sample_weights_train.mean():.4f}")
        
        # 과적합 방지 강화된 파라미터
        model = xgb.XGBClassifier(
            n_estimators=50,  # 복잡도 감소
            max_depth=4,  # 복잡도 감소
            learning_rate=0.05,  # 학습률 감소
            subsample=0.7,  # 샘플링 비율 감소
            colsample_bytree=0.7,  # 피처 샘플링 비율 감소
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            reg_alpha=0.5,  # L1 정규화 강화
            reg_lambda=2.0  # L2 정규화 강화
        )
        
        # 변동성 가중치 적용 학습
        model.fit(X_train, y_train, sample_weight=sample_weights_train, 
                 eval_set=[(X_val, y_val)], verbose=False)
        
        # 성과 평가
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        result = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(y_test) > 1 else 0
        }
        
        print(f"\n테스트 성과:")
        print(f"  정확도: {result['accuracy']:.4f}")
        print(f"  정밀도: {result['precision']:.4f}")
        print(f"  재현율: {result['recall']:.4f}")
        print(f"  F1 점수: {result['f1']:.4f}")
        print(f"  ROC AUC: {result['roc_auc']:.4f}")
        
        self.models['xgboost_optimized'] = model
        joblib.dump(model, MODELS_DIR / "xgboost_optimized_final.pkl")
        
        return result
    
    def train_random_forest_optimized(self, df: pd.DataFrame):
        """최적화된 Random Forest 학습 (과적합 방지 + 변동성 가중치)"""
        from sklearn.ensemble import RandomForestClassifier
        
        # 변동성 기반 가중치
        sample_weights = self.calculate_volatility_weights(df)
        
        # 피처 준비 (29개 피처)
        rf_base_features = [
            'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
            'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
            'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
            'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
        ]
        
        df_temp = df.copy()
        
        # 추가 피처 계산
        df_temp['rsi_oversold'] = (df_temp['entry_rsi'] < 30).astype(int)
        df_temp['rsi_overbought'] = (df_temp['entry_rsi'] > 70).astype(int)
        df_temp['macd_bullish'] = (df_temp['entry_macd'] > df_temp['entry_macd_signal']).astype(int)
        df_temp['macd_strength'] = abs(df_temp['entry_macd'] - df_temp['entry_macd_signal'])
        df_temp['price_above_ma20'] = (df_temp['entry_close'] > df_temp['entry_ma20']).astype(int)
        df_temp['price_above_ma60'] = (df_temp['entry_close'] > df_temp['entry_ma60']).astype(int)
        df_temp['bb_position'] = (df_temp['entry_close'] - df_temp['entry_bb_lower']) / (df_temp['entry_bb_upper'] - df_temp['entry_bb_lower'])
        df_temp['bb_lower_touch'] = (df_temp['entry_close'] <= df_temp['entry_bb_lower'] * 1.01).astype(int)
        df_temp['price_above_st'] = (df_temp['entry_close'] > df_temp['entry_supertrend']).astype(int)
        df_temp['is_morning'] = ((df_temp['entry_hour'] >= 9) & (df_temp['entry_hour'] < 12)).astype(int)
        df_temp['is_afternoon'] = ((df_temp['entry_hour'] >= 12) & (df_temp['entry_hour'] < 15)).astype(int)
        df_temp['is_bull'] = (df_temp['regime'] == 1).astype(int)
        df_temp['is_neutral'] = (df_temp['regime'] == 0).astype(int)
        
        rf_features = rf_base_features + [
            'rsi_oversold', 'rsi_overbought',
            'macd_bullish', 'macd_strength',
            'price_above_ma20', 'price_above_ma60',
            'bb_position', 'bb_lower_touch',
            'price_above_st',
            'is_morning', 'is_afternoon',
            'is_bull', 'is_neutral'
        ]
        
        X = df_temp[rf_features].fillna(0).values
        y = df_temp['is_win'].values
        
        # 시간 기반 분할
        df_temp['year'] = pd.to_datetime(df_temp['entry_time']).dt.year
        train_mask = (df_temp['year'] >= 2019) & (df_temp['year'] <= 2023)
        val_mask = (df_temp['year'] == 2024)
        test_mask = (df_temp['year'] >= 2025)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        sample_weights_train = sample_weights[train_mask]
        
        print(f"\n최적화된 Random Forest 학습:")
        print(f"  훈련: {len(X_train)}건 (승률: {y_train.mean():.2%})")
        print(f"  검증: {len(X_val)}건 (승률: {y_val.mean():.2%})")
        print(f"  테스트: {len(X_test)}건 (승률: {y_test.mean():.2%})")
        print(f"  가중치 평균: {sample_weights_train.mean():.4f}")
        
        # 과적합 방지 강화된 파라미터
        model = RandomForestClassifier(
            n_estimators=50,  # 복잡도 감소
            max_depth=6,  # 복잡도 감소
            min_samples_split=20,  # 증가
            min_samples_leaf=10,  # 증가
            max_features='sqrt',  # 피처 수 제한
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # 변동성 가중치 적용 학습
        model.fit(X_train, y_train, sample_weight=sample_weights_train)
        
        # 성과 평가
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        result = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(y_test) > 1 else 0
        }
        
        print(f"\n테스트 성과:")
        print(f"  정확도: {result['accuracy']:.4f}")
        print(f"  정밀도: {result['precision']:.4f}")
        print(f"  재현율: {result['recall']:.4f}")
        print(f"  F1 점수: {result['f1']:.4f}")
        print(f"  ROC AUC: {result['roc_auc']:.4f}")
        
        self.models['random_forest_optimized'] = model
        joblib.dump(model, MODELS_DIR / "random_forest_optimized_final.pkl")
        
        return result
    
    def evaluate_with_timeseries_cv(self, df: pd.DataFrame):
        """Time Series Split 교차 검증으로 평가"""
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # 피처 준비
        feature_cols = [
            'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
            'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
            'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
            'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
        ]
        
        X = df[feature_cols].fillna(0).values
        y = df['is_win'].values
        timestamps = pd.to_datetime(df['entry_time'])
        
        # 변동성 기반 가중치
        sample_weights = self.calculate_volatility_weights(df)
        
        # Time Series Split
        n_splits = 5
        n_samples = len(X)
        test_size = n_samples // (n_splits + 1)
        
        fold_results = []
        
        print(f"\nTime Series Split 교차 검증:")
        
        for i in range(n_splits):
            # 훈련 인덱스
            train_end = n_samples - (n_splits - i) * test_size
            train_indices = np.arange(train_end)
            
            # 검증 인덱스
            test_start = train_end
            test_end = n_samples - (n_splits - i - 1) * test_size
            test_indices = np.arange(test_start, test_end)
            
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            weights_train = sample_weights[train_indices]
            
            # 과적합 방지 강화된 파라미터
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
            
            # 변동성 가중치 적용 학습
            model.fit(X_train, y_train, sample_weight=weights_train)
            
            # 예측
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 성과 평가
            result = {
                'fold': i,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(y_test) > 1 else 0
            }
            
            fold_results.append(result)
            
            print(f"  Fold {i}: 정확도={result['accuracy']:.4f}, F1={result['f1']:.4f}")
        
        # 평균 성과
        avg_results = {
            'accuracy': np.mean([r['accuracy'] for r in fold_results]),
            'precision': np.mean([r['precision'] for r in fold_results]),
            'recall': np.mean([r['recall'] for r in fold_results]),
            'f1': np.mean([r['f1'] for r in fold_results]),
            'roc_auc': np.mean([r['roc_auc'] for r in fold_results])
        }
        
        print(f"\n평균 성과:")
        print(f"  정확도: {avg_results['accuracy']:.4f}")
        print(f"  정밀도: {avg_results['precision']:.4f}")
        print(f"  재현율: {avg_results['recall']:.4f}")
        print(f"  F1 점수: {avg_results['f1']:.4f}")
        print(f"  ROC AUC: {avg_results['roc_auc']:.4f}")
        
        return avg_results


def main():
    """메인 함수"""
    print("=" * 80)
    print("최적화된 ML 파이프라인")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 최적화된 파이프라인
    pipeline = OptimizedMLPipeline()
    
    # XGBoost 최적화 학습
    print("\n" + "=" * 80)
    print("1. 과적합 방지 + 변동성 가중치 XGBoost")
    print("=" * 80)
    xgb_result = pipeline.train_xgboost_optimized(df)
    
    # Random Forest 최적화 학습
    print("\n" + "=" * 80)
    print("2. 과적합 방지 + 변동성 가중치 Random Forest")
    print("=" * 80)
    rf_result = pipeline.train_random_forest_optimized(df)
    
    # Time Series Split 교차 검증
    print("\n" + "=" * 80)
    print("3. Time Series Split 교차 검증")
    print("=" * 80)
    cv_result = pipeline.evaluate_with_timeseries_cv(df)
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("최종 결과 요약")
    print("=" * 80)
    print(f"\nXGBoost (과적합 방지 + 변동성 가중치):")
    print(f"  정확도: {xgb_result['accuracy']:.4f}")
    print(f"  F1 점수: {xgb_result['f1']:.4f}")
    print(f"  ROC AUC: {xgb_result['roc_auc']:.4f}")
    
    print(f"\nRandom Forest (과적합 방지 + 변동성 가중치):")
    print(f"  정확도: {rf_result['accuracy']:.4f}")
    print(f"  F1 점수: {rf_result['f1']:.4f}")
    print(f"  ROC AUC: {rf_result['roc_auc']:.4f}")
    
    print(f"\nTime Series Split 교차 검증:")
    print(f"  정확도: {cv_result['accuracy']:.4f}")
    print(f"  F1 점수: {cv_result['f1']:.4f}")
    print(f"  ROC AUC: {cv_result['roc_auc']:.4f}")
    
    print(f"\n최적화된 ML 파이프라인 구축 완료")


if __name__ == "__main__":
    main()
