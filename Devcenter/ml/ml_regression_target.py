"""
타겟 변수 재정의 (이진 → 회귀)
net_krw를 타겟으로 하는 회귀 모델 Walk-Forward Validation
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "ml_data"
MODELS_DIR = Path(__file__).parent / "ml_models"


def walk_forward_validation(df: pd.DataFrame, n_splits: int = 3) -> List[Dict]:
    """Walk-Forward Validation 실제 실행"""
    # 연도별로 데이터 분할
    years = sorted(df['year'].unique())
    
    results = []
    
    # 사용 가능한 최대 fold 수 계산
    max_folds = len(years) - n_splits
    if max_folds <= 0:
        print(f"데이터 부족: 최소 {n_splits+1}년 필요 (현재 {len(years)}년)")
        return results
    
    for i in range(max_folds):
        train_years = years[i:i+n_splits]
        test_year = years[i+n_splits]
        
        train_data = df[df['year'].isin(train_years)].copy()
        test_data = df[df['year'] == test_year].copy()
        
        print(f"\n{'='*80}")
        print(f"Fold {i+1}")
        print(f"{'='*80}")
        print(f"훈련 기간: {train_years[0]}-{train_years[-1]} ({len(train_data)}건)")
        print(f"테스트 기간: {test_year} ({len(test_data)}건)")
        
        # 모델 학습 및 평가
        fold_result = train_and_evaluate_models(train_data, test_data)
        fold_result['train_years'] = train_years
        fold_result['test_year'] = test_year
        
        results.append(fold_result)
    
    return results


def train_and_evaluate_models(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
    """모델 학습 및 평가 (회귀)"""
    results = {}
    
    # 1. XGBoost 회귀 모델
    print("\n[1/4] XGBoost 회귀 모델...")
    xgb_result = train_xgboost_regression(train_data, test_data)
    results['xgboost'] = xgb_result
    
    # 2. Random Forest 회귀 모델
    print("[2/4] Random Forest 회귀 모델...")
    rf_result = train_random_forest_regression(train_data, test_data)
    results['random_forest'] = rf_result
    
    # 3. LightGBM 회귀 모델
    print("[3/4] LightGBM 회귀 모델...")
    lgb_result = train_lightgbm_regression(train_data, test_data)
    results['lightgbm'] = lgb_result
    
    # 4. CatBoost 회귀 모델
    print("[4/4] CatBoost 회귀 모델...")
    cat_result = train_catboost_regression(train_data, test_data)
    results['catboost'] = cat_result
    
    return results


def train_xgboost_regression(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
    """XGBoost 회귀 모델 학습 및 평가"""
    import xgboost as xgb
    
    # 기본 피처
    base_feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    # 추가 피처 계산
    train_data = train_data.copy()
    test_data = test_data.copy()
    
    for df in [train_data, test_data]:
        df['rsi_ma_ratio'] = df['entry_rsi'] / df['entry_ma20']
        df['price_ma_ratio'] = df['entry_close'] / df['entry_ma20']
        df['bb_position'] = (df['entry_close'] - df['entry_bb_lower']) / (df['entry_bb_upper'] - df['entry_bb_lower'])
        df['supertrend_alignment'] = ((df['entry_close'] > df['entry_supertrend']) & (df['entry_supertrend_dir'] == 1)).astype(int)
    
    # 모든 피처
    all_feature_cols = base_feature_cols + ['rsi_ma_ratio', 'price_ma_ratio', 'bb_position', 'supertrend_alignment']
    
    # 학습 데이터 준비
    X_train = train_data[all_feature_cols].copy().fillna(0).astype(float)
    y_train = train_data['net_krw'].copy()
    X_test = test_data[all_feature_cols].copy().fillna(0).astype(float)
    y_test = test_data['net_krw'].copy()
    
    # XGBoost 회귀 모델 학습
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        reg_alpha=0.5,
        reg_lambda=2.0
    )
    
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 성능 평가
    result = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'predictions': y_pred,
        'actual': y_test
    }
    
    print(f"  MSE: {result['mse']:.2f}")
    print(f"  MAE: {result['mae']:.2f}")
    print(f"  R2: {result['r2']:.4f}")
    
    return result


def train_random_forest_regression(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
    """Random Forest 회귀 모델 학습 및 평가"""
    from sklearn.ensemble import RandomForestRegressor
    
    # 기본 피처
    base_feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    # 추가 피처 계산
    train_data = train_data.copy()
    test_data = test_data.copy()
    
    for df in [train_data, test_data]:
        df['rsi_ma_ratio'] = df['entry_rsi'] / df['entry_ma20']
        df['price_ma_ratio'] = df['entry_close'] / df['entry_ma20']
        df['bb_position'] = (df['entry_close'] - df['entry_bb_lower']) / (df['entry_bb_upper'] - df['entry_bb_lower'])
        df['supertrend_alignment'] = ((df['entry_close'] > df['entry_supertrend']) & (df['entry_supertrend_dir'] == 1)).astype(int)
    
    # 모든 피처
    all_feature_cols = base_feature_cols + ['rsi_ma_ratio', 'price_ma_ratio', 'bb_position', 'supertrend_alignment']
    
    # 학습 데이터 준비
    X_train = train_data[all_feature_cols].copy().fillna(0).astype(float)
    y_train = train_data['net_krw'].copy()
    X_test = test_data[all_feature_cols].copy().fillna(0).astype(float)
    y_test = test_data['net_krw'].copy()
    
    # Random Forest 회귀 모델 학습
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 성능 평가
    result = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'predictions': y_pred,
        'actual': y_test
    }
    
    print(f"  MSE: {result['mse']:.2f}")
    print(f"  MAE: {result['mae']:.2f}")
    print(f"  R2: {result['r2']:.4f}")
    
    return result


def train_lightgbm_regression(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
    """LightGBM 회귀 모델 학습 및 평가"""
    try:
        import lightgbm as lgb
    except ImportError:
        print("  LightGBM not installed, skipping...")
        return {'mse': 0, 'mae': 0, 'r2': 0}
    
    # 기본 피처
    base_feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    # 추가 피처 계산
    train_data = train_data.copy()
    test_data = test_data.copy()
    
    for df in [train_data, test_data]:
        df['rsi_ma_ratio'] = df['entry_rsi'] / df['entry_ma20']
        df['price_ma_ratio'] = df['entry_close'] / df['entry_ma20']
        df['bb_position'] = (df['entry_close'] - df['entry_bb_lower']) / (df['entry_bb_upper'] - df['entry_bb_lower'])
        df['supertrend_alignment'] = ((df['entry_close'] > df['entry_supertrend']) & (df['entry_supertrend_dir'] == 1)).astype(int)
    
    # 모든 피처
    all_feature_cols = base_feature_cols + ['rsi_ma_ratio', 'price_ma_ratio', 'bb_position', 'supertrend_alignment']
    
    # 학습 데이터 준비
    X_train = train_data[all_feature_cols].copy().fillna(0).astype(float)
    y_train = train_data['net_krw'].copy()
    X_test = test_data[all_feature_cols].copy().fillna(0).astype(float)
    y_test = test_data['net_krw'].copy()
    
    # LightGBM 회귀 모델 학습
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        reg_alpha=0.5,
        reg_lambda=2.0,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 성능 평가
    result = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'predictions': y_pred,
        'actual': y_test
    }
    
    print(f"  MSE: {result['mse']:.2f}")
    print(f"  MAE: {result['mae']:.2f}")
    print(f"  R2: {result['r2']:.4f}")
    
    return result


def train_catboost_regression(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
    """CatBoost 회귀 모델 학습 및 평가"""
    try:
        import catboost as cb
    except ImportError:
        print("  CatBoost not installed, skipping...")
        return {'mse': 0, 'mae': 0, 'r2': 0}
    
    # 기본 피처
    base_feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    # 추가 피처 계산
    train_data = train_data.copy()
    test_data = test_data.copy()
    
    for df in [train_data, test_data]:
        df['rsi_ma_ratio'] = df['entry_rsi'] / df['entry_ma20']
        df['price_ma_ratio'] = df['entry_close'] / df['entry_ma20']
        df['bb_position'] = (df['entry_close'] - df['entry_bb_lower']) / (df['entry_bb_upper'] - df['entry_bb_lower'])
        df['supertrend_alignment'] = ((df['entry_close'] > df['entry_supertrend']) & (df['entry_supertrend_dir'] == 1)).astype(int)
    
    # 모든 피처
    all_feature_cols = base_feature_cols + ['rsi_ma_ratio', 'price_ma_ratio', 'bb_position', 'supertrend_alignment']
    
    # 학습 데이터 준비
    X_train = train_data[all_feature_cols].copy().fillna(0).astype(float)
    y_train = train_data['net_krw'].copy()
    X_test = test_data[all_feature_cols].copy().fillna(0).astype(float)
    y_test = test_data['net_krw'].copy()
    
    # CatBoost 회귀 모델 학습
    model = cb.CatBoostRegressor(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        l2_leaf_reg=2.0,
        verbose=False
    )
    
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 성능 평가
    result = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'predictions': y_pred,
        'actual': y_test
    }
    
    print(f"  MSE: {result['mse']:.2f}")
    print(f"  MAE: {result['mae']:.2f}")
    print(f"  R2: {result['r2']:.4f}")
    
    return result


def print_walk_forward_results(results: List[Dict]):
    """Walk-Forward Validation 결과 출력"""
    print(f"\n{'='*80}")
    print("Walk-Forward Validation 결과 요약 (회귀)")
    print(f"{'='*80}")
    
    for i, result in enumerate(results):
        print(f"\nFold {i+1} ({result['train_years'][0]}-{result['train_years'][-1]} → {result['test_year']}):")
        print(f"  XGBoost: MSE={result['xgboost']['mse']:.2f}, MAE={result['xgboost']['mae']:.2f}, R2={result['xgboost']['r2']:.4f}")
        print(f"  Random Forest: MSE={result['random_forest']['mse']:.2f}, MAE={result['random_forest']['mae']:.2f}, R2={result['random_forest']['r2']:.4f}")
        print(f"  LightGBM: MSE={result['lightgbm']['mse']:.2f}, MAE={result['lightgbm']['mae']:.2f}, R2={result['lightgbm']['r2']:.4f}")
        print(f"  CatBoost: MSE={result['catboost']['mse']:.2f}, MAE={result['catboost']['mae']:.2f}, R2={result['catboost']['r2']:.4f}")
    
    # 평균 성과
    print(f"\n{'='*80}")
    print("평균 성과")
    print(f"{'='*80}")
    
    avg_xgb = {
        'mse': np.mean([r['xgboost']['mse'] for r in results]),
        'mae': np.mean([r['xgboost']['mae'] for r in results]),
        'r2': np.mean([r['xgboost']['r2'] for r in results])
    }
    
    avg_rf = {
        'mse': np.mean([r['random_forest']['mse'] for r in results]),
        'mae': np.mean([r['random_forest']['mae'] for r in results]),
        'r2': np.mean([r['random_forest']['r2'] for r in results])
    }
    
    avg_lgb = {
        'mse': np.mean([r['lightgbm']['mse'] for r in results]),
        'mae': np.mean([r['lightgbm']['mae'] for r in results]),
        'r2': np.mean([r['lightgbm']['r2'] for r in results])
    }
    
    avg_cat = {
        'mse': np.mean([r['catboost']['mse'] for r in results]),
        'mae': np.mean([r['catboost']['mae'] for r in results]),
        'r2': np.mean([r['catboost']['r2'] for r in results])
    }
    
    print(f"XGBoost: MSE={avg_xgb['mse']:.2f}, MAE={avg_xgb['mae']:.2f}, R2={avg_xgb['r2']:.4f}")
    print(f"Random Forest: MSE={avg_rf['mse']:.2f}, MAE={avg_rf['mae']:.2f}, R2={avg_rf['r2']:.4f}")
    print(f"LightGBM: MSE={avg_lgb['mse']:.2f}, MAE={avg_lgb['mae']:.2f}, R2={avg_lgb['r2']:.4f}")
    print(f"CatBoost: MSE={avg_cat['mse']:.2f}, MAE={avg_cat['mae']:.2f}, R2={avg_cat['r2']:.4f}")


def main():
    """메인 함수"""
    print("=" * 80)
    print("타겟 변수 재정의 (이진 → 회귀) Walk-Forward Validation")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['year'] = df['entry_time'].dt.year
    
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    print(f"연도별 데이터: {df['year'].value_counts().sort_index().to_dict()}")
    
    # 타겟 변수 통계
    print(f"\n타겟 변수 (net_krw) 통계:")
    print(f"  평균: {df['net_krw'].mean():.2f}")
    print(f"  표준편차: {df['net_krw'].std():.2f}")
    print(f"  최소값: {df['net_krw'].min():.2f}")
    print(f"  최대값: {df['net_krw'].max():.2f}")
    
    # Walk-Forward Validation 실행
    results = walk_forward_validation(df, n_splits=3)
    
    # 결과 출력
    print_walk_forward_results(results)
    
    return results


if __name__ == "__main__":
    results = main()
