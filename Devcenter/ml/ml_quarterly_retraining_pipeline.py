"""
분기별 재학습 파이프라인
Walk-Forward Validation 기반 정기 모델 재학습 시스템
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
DATA_DIR = Path(__file__).parent / "ml_data"
MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    """ML 데이터셋 로드"""
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['year'] = df['entry_time'].dt.year
    df['quarter'] = df['entry_time'].dt.quarter
    return df


def train_random_forest_quarterly(train_data: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """분기별 Random Forest 모델 학습"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    X_train = train_data[feature_cols].copy().fillna(0).astype(float)
    y_train = train_data['is_win'].copy()
    
    # 보수적 파라미터
    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=3,
        min_samples_split=30,
        min_samples_leaf=15,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # 학습 데이터 성능 평가
    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    
    result = {
        'model': model,
        'accuracy': accuracy_score(y_train, y_pred),
        'precision': precision_score(y_train, y_pred, zero_division=0),
        'recall': recall_score(y_train, y_pred, zero_division=0),
        'f1': f1_score(y_train, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_train, y_pred_proba) if len(y_train) > 1 else 0,
        'feature_importance': dict(zip(feature_cols, model.feature_importances_))
    }
    
    return result


def get_feature_cols() -> List[str]:
    """피처 컬럼 목록 반환"""
    base_feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    derived_feature_cols = [
        'rsi_ma_ratio', 'price_ma_ratio', 'bb_position', 'supertrend_alignment'
    ]
    
    return base_feature_cols + derived_feature_cols


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """파생 피처 추가"""
    df = df.copy()
    df['rsi_ma_ratio'] = df['entry_rsi'] / df['entry_ma20']
    df['price_ma_ratio'] = df['entry_close'] / df['entry_ma20']
    df['bb_position'] = (df['entry_close'] - df['entry_bb_lower']) / (df['entry_bb_upper'] - df['entry_bb_lower'])
    df['supertrend_alignment'] = ((df['entry_close'] > df['entry_supertrend']) & (df['entry_supertrend_dir'] == 1)).astype(int)
    return df


def quarterly_retraining_pipeline(df: pd.DataFrame, start_year: int = 2019, end_year: int = 2026) -> Dict:
    """분기별 재학습 파이프라인 실행"""
    print("=" * 80)
    print("분기별 재학습 파이프라인")
    print("=" * 80)
    
    feature_cols = get_feature_cols()
    df = add_derived_features(df)
    
    results = {}
    
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            # 훈련 데이터: 현재 분기 이전 모든 데이터
            train_mask = (df['year'] < year) | ((df['year'] == year) & (df['quarter'] < quarter))
            train_data = df[train_mask].copy()
            
            # 테스트 데이터: 현재 분기
            test_mask = (df['year'] == year) & (df['quarter'] == quarter)
            test_data = df[test_mask].copy()
            
            if len(train_data) < 100 or len(test_data) == 0:
                print(f"\n{year}년 Q{quarter}: 데이터 부족 (훈련: {len(train_data)}, 테스트: {len(test_data)})")
                continue
            
            print(f"\n{year}년 Q{quarter}:")
            print(f"  훈련 데이터: {len(train_data)}건")
            print(f"  테스트 데이터: {len(test_data)}건")
            
            # 모델 학습
            train_result = train_random_forest_quarterly(train_data, feature_cols)
            model = train_result['model']
            
            # 테스트 데이터 성능 평가
            X_test = test_data[feature_cols].copy().fillna(0).astype(float)
            y_test = test_data['is_win'].copy()
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, zero_division=0)
            test_recall = recall_score(y_test, y_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_pred, zero_division=0)
            test_roc_auc = roc_auc_score(y_test, y_pred_proba) if len(y_test) > 1 else 0
            
            print(f"  훈련 정확도: {train_result['accuracy']:.4f}")
            print(f"  테스트 정확도: {test_accuracy:.4f}")
            print(f"  테스트 F1: {test_f1:.4f}")
            print(f"  테스트 ROC AUC: {test_roc_auc:.4f}")
            
            # 모델 저장
            model_path = MODELS_DIR / f"rf_model_{year}Q{quarter}.pkl"
            joblib.dump(model, model_path)
            print(f"  모델 저장: {model_path}")
            
            # 결과 저장
            key = f"{year}Q{quarter}"
            results[key] = {
                'year': year,
                'quarter': quarter,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'train_accuracy': train_result['accuracy'],
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_roc_auc': test_roc_auc,
                'feature_importance': train_result['feature_importance'],
                'model_path': str(model_path)
            }
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("분기별 재학습 결과 요약")
    print(f"{'='*80}")
    
    if results:
        avg_test_accuracy = np.mean([r['test_accuracy'] for r in results.values()])
        avg_test_f1 = np.mean([r['test_f1'] for r in results.values()])
        avg_test_roc_auc = np.mean([r['test_roc_auc'] for r in results.values()])
        
        print(f"평균 테스트 정확도: {avg_test_accuracy:.4f}")
        print(f"평균 테스트 F1: {avg_test_f1:.4f}")
        print(f"평균 테스트 ROC AUC: {avg_test_roc_auc:.4f}")
        
        # 결과 저장
        results_path = MODELS_DIR / "quarterly_retraining_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n결과 저장: {results_path}")
    else:
        print("재학습 결과 없음")
    
    return results


def main():
    """메인 함수"""
    print("=" * 80)
    print("분기별 재학습 파이프라인 실행")
    print("=" * 80)
    
    # 데이터 로드
    df = load_data()
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 분기별 재학습 파이프라인 실행
    results = quarterly_retraining_pipeline(df, start_year=2019, end_year=2026)
    
    return results


if __name__ == "__main__":
    results = main()
