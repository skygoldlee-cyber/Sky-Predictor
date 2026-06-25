"""
하이퍼파라미터 최적화 (Bayesian Optimization)
Optuna를 사용한 Random Forest 하이퍼파라미터 최적화
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
DATA_DIR = Path(__file__).parent / "ml_data"
MODELS_DIR = Path(__file__).parent / "ml_models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    """ML 데이터셋 로드"""
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['year'] = df['entry_time'].dt.year
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """피처 준비"""
    df = df.copy()
    
    # 기본 피처
    base_feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    # 파생 피처
    df['rsi_ma_ratio'] = df['entry_rsi'] / df['entry_ma20']
    df['price_ma_ratio'] = df['entry_close'] / df['entry_ma20']
    df['bb_position'] = (df['entry_close'] - df['entry_bb_lower']) / (df['entry_bb_upper'] - df['entry_bb_lower'])
    df['supertrend_alignment'] = ((df['entry_close'] > df['entry_supertrend']) & (df['entry_supertrend_dir'] == 1)).astype(int)
    
    all_feature_cols = base_feature_cols + ['rsi_ma_ratio', 'price_ma_ratio', 'bb_position', 'supertrend_alignment']
    
    return df, all_feature_cols


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna 목적 함수"""
    # 하이퍼파라미터 탐색 공간
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 5, 50)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 30)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    
    # 모델 학습
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # 성능 평가
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_pred_proba) if len(y_val) > 1 else 0
    
    # 목표: F1 점수 최대화
    return f1


def optimize_hyperparameters(df: pd.DataFrame, feature_cols: List[str], n_trials: int = 50) -> Dict:
    """하이퍼파라미터 최적화"""
    # 데이터 분할 (훈련/검증)
    train_data = df[df['year'] <= 2024].copy()
    val_data = df[df['year'] == 2025].copy()
    
    X_train = train_data[feature_cols].copy().fillna(0).astype(float)
    y_train = train_data['is_win'].copy()
    X_val = val_data[feature_cols].copy().fillna(0).astype(float)
    y_val = val_data['is_win'].copy()
    
    print(f"훈련 데이터: {len(X_train)}건")
    print(f"검증 데이터: {len(X_val)}건")
    
    # Optuna 스터디 생성
    study = optuna.create_study(direction='maximize')
    
    # 최적화 실행
    print(f"\n하이퍼파라미터 최적화 시작 (n_trials={n_trials})...")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
    
    # 최적 파라미터
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n최적 하이퍼파라미터:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"최적 F1 점수: {best_value:.4f}")
    
    # 최적 파라미터로 모델 학습 및 평가
    best_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    best_model.fit(X_train, y_train)
    
    # 테스트 데이터로 평가
    test_data = df[df['year'] == 2026].copy()
    X_test = test_data[feature_cols].copy().fillna(0).astype(float)
    y_test = test_data['is_win'].copy()
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba) if len(y_test) > 1 else 0
    
    print(f"\n테스트 성과:")
    print(f"  정확도: {test_accuracy:.4f}")
    print(f"  F1 점수: {test_f1:.4f}")
    print(f"  ROC AUC: {test_roc_auc:.4f}")
    
    result = {
        'best_params': best_params,
        'best_value': best_value,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'test_roc_auc': test_roc_auc,
        'model': best_model
    }
    
    return result


def main():
    """메인 함수"""
    print("=" * 80)
    print("하이퍼파라미터 최적화 (Bayesian Optimization)")
    print("=" * 80)
    
    # 데이터 로드
    df = load_data()
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 피처 준비
    df, feature_cols = prepare_features(df)
    print(f"피처 준비 완료: {len(feature_cols)}개")
    
    # 하이퍼파라미터 최적화
    result = optimize_hyperparameters(df, feature_cols, n_trials=50)
    
    # 최적 모델 저장
    import joblib
    model_path = MODELS_DIR / "rf_optimized.pkl"
    joblib.dump(result['model'], model_path)
    print(f"\n최적 모델 저장: {model_path}")
    
    # 결과 저장
    import json
    result_path = MODELS_DIR / "hyperparameter_optimization_result.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_params': result['best_params'],
            'best_value': result['best_value'],
            'test_accuracy': result['test_accuracy'],
            'test_f1': result['test_f1'],
            'test_roc_auc': result['test_roc_auc']
        }, f, indent=2, ensure_ascii=False)
    print(f"최적화 결과 저장: {result_path}")
    
    return result


if __name__ == "__main__":
    result = main()
