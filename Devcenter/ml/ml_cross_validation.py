"""
교차 검증 방법 개선
Purged K-Fold, Time Series Split, Nested Cross-Validation
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


class PurgedKFold:
    """Purged K-Fold 교차 검증 (시계열 데이터 누설 방지)"""
    
    def __init__(self, n_splits=5, purge_gap=5):
        self.n_splits = n_splits
        self.purge_gap = purge_gap  # 훈련/검증 사이 시간 간격
        
    def split(self, X: np.ndarray, timestamps: pd.Series):
        """Purged K-Fold 분할"""
        indices = np.arange(len(X))
        timestamps_sorted = timestamps.sort_values()
        
        fold_size = len(X) // self.n_splits
        
        for i in range(self.n_splits):
            # 훈련 인덱스
            train_end = (i + 1) * fold_size - self.purge_gap
            train_indices = indices[:train_end]
            
            # 검증 인덱스
            val_start = (i + 1) * fold_size
            val_end = min((i + 2) * fold_size, len(X))
            val_indices = indices[val_start:val_end]
            
            if len(train_indices) > 0 and len(val_indices) > 0:
                yield train_indices, val_indices


class TimeSeriesSplit:
    """시계열 교차 검증"""
    
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        
    def split(self, X: np.ndarray, timestamps: pd.Series):
        """시계열 분할"""
        indices = np.arange(len(X))
        timestamps_sorted = timestamps.sort_values()
        
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # 훈련 인덱스
            train_end = n_samples - (self.n_splits - i) * test_size
            train_indices = indices[:train_end]
            
            # 검증 인덱스
            test_start = train_end
            test_end = n_samples - (self.n_splits - i - 1) * test_size
            test_indices = indices[test_start:test_end]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


class CrossValidationEvaluator:
    """교차 검증 평가기"""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_model_with_cv(self, df: pd.DataFrame, cv_method='timeseries'):
        """교차 검증으로 모델 평가"""
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
        
        # 교차 검증 방법 선택
        if cv_method == 'purged':
            cv = PurgedKFold(n_splits=5, purge_gap=5)
        elif cv_method == 'timeseries':
            cv = TimeSeriesSplit(n_splits=5)
        else:
            cv = TimeSeriesSplit(n_splits=5)
        
        # 교차 검증 실행
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, timestamps)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
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
            
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 성과 평가
            result = {
                'fold': fold,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(y_test) > 1 else 0,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            fold_results.append(result)
            
            print(f"Fold {fold}: 정확도={result['accuracy']:.4f}, F1={result['f1']:.4f}")
        
        # 평균 성과 계산
        avg_results = {
            'accuracy': np.mean([r['accuracy'] for r in fold_results]),
            'precision': np.mean([r['precision'] for r in fold_results]),
            'recall': np.mean([r['recall'] for r in fold_results]),
            'f1': np.mean([r['f1'] for r in fold_results]),
            'roc_auc': np.mean([r['roc_auc'] for r in fold_results])
        }
        
        self.results[cv_method] = {
            'fold_results': fold_results,
            'avg_results': avg_results
        }
        
        return avg_results


def compare_cv_methods(df: pd.DataFrame):
    """다양한 교차 검증 방법 비교"""
    print("=" * 80)
    print("교차 검증 방법 비교")
    print("=" * 80)
    
    evaluator = CrossValidationEvaluator()
    
    cv_methods = ['timeseries', 'purged']
    results = {}
    
    for cv_method in cv_methods:
        print(f"\n{cv_method} 교차 검증:")
        avg_results = evaluator.evaluate_model_with_cv(df, cv_method)
        results[cv_method] = avg_results
        
        print(f"  평균 정확도: {avg_results['accuracy']:.4f}")
        print(f"  평균 F1 점수: {avg_results['f1']:.4f}")
        print(f"  평균 ROC AUC: {avg_results['roc_auc']:.4f}")
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("교차 검증 방법 비교 결과")
    print("=" * 80)
    for cv_method, avg_results in results.items():
        print(f"\n{cv_method}:")
        print(f"  정확도: {avg_results['accuracy']:.4f}")
        print(f"  정밀도: {avg_results['precision']:.4f}")
        print(f"  재현율: {avg_results['recall']:.4f}")
        print(f"  F1 점수: {avg_results['f1']:.4f}")
        print(f"  ROC AUC: {avg_results['roc_auc']:.4f}")
    
    # 최적 방법 선택
    best_method = max(results, key=lambda x: results[x]['f1'])
    print(f"\n최적 교차 검증 방법: {best_method} (F1: {results[best_method]['f1']:.4f})")
    
    return results


def main():
    """메인 함수"""
    print("=" * 80)
    print("교차 검증 방법 개선")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 교차 검증 방법 비교
    results = compare_cv_methods(df)
    
    print(f"\n교차 검증 방법 개선 완료")


if __name__ == "__main__":
    main()
