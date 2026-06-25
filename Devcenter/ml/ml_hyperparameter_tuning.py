"""
하이퍼파라미터 튜닝 (Optuna)
XGBoost 모델 자동 최적화
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
import joblib

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "ml_data"
MODELS_DIR = Path(__file__).parent / "ml_models"


def objective_xgboost(trial, X_train, y_train, X_val, y_val):
    """XGBoost 하이퍼파라미터 최적화 목적 함수"""
    import xgboost as xgb
    
    # 하이퍼파라미터 탐색 공간
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 30, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    
    # 모델 학습
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # 검증 성과
    y_pred = model.predict(X_val)
    accuracy = (y_pred == y_val).mean()
    
    return accuracy


def optimize_xgboost_hyperparameters(df: pd.DataFrame, n_trials: int = 50):
    """XGBoost 하이퍼파라미터 최적화"""
    import optuna
    
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
    
    print(f"데이터 분할:")
    print(f"  훈련: {len(X_train)}건 (승률: {y_train.mean():.2%})")
    print(f"  검증: {len(X_val)}건 (승률: {y_val.mean():.2%})")
    print(f"  테스트: {len(X_test)}건 (승률: {y_test.mean():.2%})")
    
    # Optuna 스터디 생성
    study = optuna.create_study(direction='maximize', study_name='xgboost_optimization')
    
    # 최적화 실행
    print(f"\n하이퍼파라미터 최적화 시작 (n_trials={n_trials})")
    study.optimize(
        lambda trial: objective_xgboost(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # 최적 하이퍼파라미터
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n최적 하이퍼파라미터:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"\n최적 검증 정확도: {best_value:.4f}")
    
    # 최적 모델로 테스트
    import xgboost as xgb
    best_model = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    y_test_pred = best_model.predict(X_test)
    test_accuracy = (y_test_pred == y_test).mean()
    
    print(f"테스트 정확도: {test_accuracy:.4f}")
    
    # 모델 저장
    joblib.dump(best_model, MODELS_DIR / "xgboost_optimized.pkl")
    joblib.dump(best_params, MODELS_DIR / "xgboost_best_params.pkl")
    
    return best_params, best_value, test_accuracy


def objective_random_forest(trial, X_train, y_train, X_val, y_val):
    """Random Forest 하이퍼파라미터 최적화 목적 함수"""
    from sklearn.ensemble import RandomForestClassifier
    
    # 하이퍼파라미터 탐색 공간
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 30, 100),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 10, 30),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 15),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 모델 학습
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # 검증 성과
    y_pred = model.predict(X_val)
    accuracy = (y_pred == y_val).mean()
    
    return accuracy


def optimize_random_forest_hyperparameters(df: pd.DataFrame, n_trials: int = 50):
    """Random Forest 하이퍼파라미터 최적화"""
    import optuna
    
    # 피처 준비 (Random Forest용 29개 피처)
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
    
    print(f"데이터 분할:")
    print(f"  훈련: {len(X_train)}건 (승률: {y_train.mean():.2%})")
    print(f"  검증: {len(X_val)}건 (승률: {y_val.mean():.2%})")
    print(f"  테스트: {len(X_test)}건 (승률: {y_test.mean():.2%})")
    
    # Optuna 스터디 생성
    study = optuna.create_study(direction='maximize', study_name='random_forest_optimization')
    
    # 최적화 실행
    print(f"\n하이퍼파라미터 최적화 시작 (n_trials={n_trials})")
    study.optimize(
        lambda trial: objective_random_forest(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # 최적 하이퍼파라미터
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n최적 하이퍼파라미터:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"\n최적 검증 정확도: {best_value:.4f}")
    
    # 최적 모델로 테스트
    from sklearn.ensemble import RandomForestClassifier
    best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    y_test_pred = best_model.predict(X_test)
    test_accuracy = (y_test_pred == y_test).mean()
    
    print(f"테스트 정확도: {test_accuracy:.4f}")
    
    # 모델 저장
    joblib.dump(best_model, MODELS_DIR / "random_forest_optimized.pkl")
    joblib.dump(best_params, MODELS_DIR / "random_forest_best_params.pkl")
    
    return best_params, best_value, test_accuracy


def main():
    """메인 함수"""
    print("=" * 80)
    print("하이퍼파라미터 튜닝 (Optuna)")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # XGBoost 하이퍼파라미터 최적화
    print("\n" + "=" * 80)
    print("XGBoost 하이퍼파라미터 최적화")
    print("=" * 80)
    xgb_best_params, xgb_best_val, xgb_test_acc = optimize_xgboost_hyperparameters(df, n_trials=30)
    
    # Random Forest 하이퍼파라미터 최적화
    print("\n" + "=" * 80)
    print("Random Forest 하이퍼파라미터 최적화")
    print("=" * 80)
    rf_best_params, rf_best_val, rf_test_acc = optimize_random_forest_hyperparameters(df, n_trials=30)
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("최적화 결과 요약")
    print("=" * 80)
    print(f"\nXGBoost:")
    print(f"  검증 정확도: {xgb_best_val:.4f}")
    print(f"  테스트 정확도: {xgb_test_acc:.4f}")
    print(f"\nRandom Forest:")
    print(f"  검증 정확도: {rf_best_val:.4f}")
    print(f"  테스트 정확도: {rf_test_acc:.4f}")
    
    print(f"\n하이퍼파라미터 튜닝 완료")


if __name__ == "__main__":
    main()
