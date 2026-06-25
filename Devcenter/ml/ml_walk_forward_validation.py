"""
Walk-Forward Validation 실제 실행 및 유효성 검증
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
    """모델 학습 및 평가"""
    results = {}
    
    # 1. XGBoost 필터링 모델
    print("\n[1/3] XGBoost 필터링 모델...")
    xgb_result = train_xgboost(train_data, test_data)
    results['xgboost'] = xgb_result
    
    # 2. Random Forest 진입 타이밍 모델
    print("[2/3] Random Forest 진입 타이밍 모델...")
    rf_result = train_random_forest(train_data, test_data)
    results['random_forest'] = rf_result
    
    # 3. LSTM 청산 타이밍 모델
    print("[3/3] LSTM 청산 타이밍 모델...")
    lstm_result = train_lstm(train_data, test_data)
    results['lstm'] = lstm_result
    
    return results


def train_xgboost(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
    """XGBoost 모델 학습 및 평가"""
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    X_train = train_data[feature_cols].copy().fillna(0).astype(float)
    y_train = train_data['is_win'].copy()
    X_test = test_data[feature_cols].copy().fillna(0).astype(float)
    y_test = test_data['is_win'].copy()
    
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        reg_alpha=0.5,  # 0.1→0.5 (L1 정규화 강화)
        reg_lambda=2.0  # 1.0→2.0 (L2 정규화 강화)
    )
    
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 성능 평가
    result = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(y_test) > 1 else 0
    }
    
    print(f"  정확도: {result['accuracy']:.4f}")
    print(f"  F1 점수: {result['f1']:.4f}")
    print(f"  ROC AUC: {result['roc_auc']:.4f}")
    
    return result


def train_random_forest(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
    """Random Forest 모델 학습 및 평가"""
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime',
        'rsi_ma_ratio', 'price_ma_ratio', 'bb_position', 'supertrend_alignment'
    ]
    
    # 추가 피처 계산
    train_data = train_data.copy()
    test_data = test_data.copy()
    
    for df in [train_data, test_data]:
        df['rsi_ma_ratio'] = df['entry_rsi'] / df['entry_ma20']
        df['price_ma_ratio'] = df['entry_close'] / df['entry_ma20']
        df['bb_position'] = (df['entry_close'] - df['entry_bb_lower']) / (df['entry_bb_upper'] - df['entry_bb_lower'])
        df['supertrend_alignment'] = ((df['entry_close'] > df['entry_supertrend']) & (df['entry_supertrend_dir'] == 1)).astype(int)
    
    X_train = train_data[feature_cols].copy().fillna(0).astype(float)
    y_train = train_data['is_win'].copy()
    X_test = test_data[feature_cols].copy().fillna(0).astype(float)
    y_test = test_data['is_win'].copy()
    
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=30,  # 100→30 (트리 수 감소)
        max_depth=4,  # 8→4 (깊이 감소)
        min_samples_split=25,  # 15→25 (분할 최소 샘플 증가)
        min_samples_leaf=12,  # 8→12 (리프 최소 샘플 증가)
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 성능 평가
    result = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(y_test) > 1 else 0
    }
    
    print(f"  정확도: {result['accuracy']:.4f}")
    print(f"  F1 점수: {result['f1']:.4f}")
    print(f"  ROC AUC: {result['roc_auc']:.4f}")
    
    return result


def train_lstm(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
    """LSTM 모델 학습 및 평가"""
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    from sklearn.preprocessing import MinMaxScaler
    
    # 시계열 데이터 준비
    train_data_sorted = train_data.sort_values('entry_time').reset_index(drop=True)
    test_data_sorted = test_data.sort_values('entry_time').reset_index(drop=True)
    
    X_train = train_data_sorted[feature_cols].values
    X_test = test_data_sorted[feature_cols].values
    
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    # 스케일링
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 시퀀스 생성
    sequence_length = 10
    
    def create_sequences(X, y, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, train_data_sorted['is_win'].values, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, test_data_sorted['is_win'].values, sequence_length)
    
    if len(X_test_seq) == 0:
        print("  테스트 시퀀스 부족 - 평가 불가")
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'roc_auc': 0}
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
             kernel_regularizer=l2(0.03), recurrent_regularizer=l2(0.03)),
        Dropout(0.6),  # 0.3→0.6 (드롭아웃 증가)
        LSTM(16, return_sequences=False,
             kernel_regularizer=l2(0.03), recurrent_regularizer=l2(0.03)),
        Dropout(0.6),  # 0.3→0.6 (드롭아웃 증가)
        Dense(8, activation='relu', kernel_regularizer=l2(0.03)),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0003), loss='binary_crossentropy', metrics=['accuracy'])  # 0.001→0.0003
    
    model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0)
    
    # 예측
    y_pred_proba = model.predict(X_test_seq, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # 성능 평가
    result = {
        'accuracy': accuracy_score(y_test_seq, y_pred),
        'precision': precision_score(y_test_seq, y_pred, zero_division=0),
        'recall': recall_score(y_test_seq, y_pred, zero_division=0),
        'f1': f1_score(y_test_seq, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test_seq, y_pred_proba) if len(y_test_seq) > 1 else 0
    }
    
    print(f"  정확도: {result['accuracy']:.4f}")
    print(f"  F1 점수: {result['f1']:.4f}")
    
    return result


def print_walk_forward_results(results: List[Dict]):
    """Walk-Forward Validation 결과 출력"""
    print(f"\n{'='*80}")
    print("Walk-Forward Validation 결과 요약")
    print(f"{'='*80}")
    
    for i, result in enumerate(results):
        print(f"\nFold {i+1} ({result['train_years'][0]}-{result['train_years'][-1]} → {result['test_year']}):")
        print(f"  XGBoost: 정확도={result['xgboost']['accuracy']:.4f}, F1={result['xgboost']['f1']:.4f}, ROC AUC={result['xgboost']['roc_auc']:.4f}")
        print(f"  Random Forest: 정확도={result['random_forest']['accuracy']:.4f}, F1={result['random_forest']['f1']:.4f}, ROC AUC={result['random_forest']['roc_auc']:.4f}")
        print(f"  LSTM: 정확도={result['lstm']['accuracy']:.4f}, F1={result['lstm']['f1']:.4f}")
    
    # 평균 성과
    print(f"\n{'='*80}")
    print("평균 성과")
    print(f"{'='*80}")
    
    avg_xgb = {
        'accuracy': np.mean([r['xgboost']['accuracy'] for r in results]),
        'f1': np.mean([r['xgboost']['f1'] for r in results]),
        'roc_auc': np.mean([r['xgboost']['roc_auc'] for r in results])
    }
    
    avg_rf = {
        'accuracy': np.mean([r['random_forest']['accuracy'] for r in results]),
        'f1': np.mean([r['random_forest']['f1'] for r in results]),
        'roc_auc': np.mean([r['random_forest']['roc_auc'] for r in results])
    }
    
    avg_lstm = {
        'accuracy': np.mean([r['lstm']['accuracy'] for r in results]),
        'f1': np.mean([r['lstm']['f1'] for r in results])
    }
    
    print(f"XGBoost: 정확도={avg_xgb['accuracy']:.4f}, F1={avg_xgb['f1']:.4f}, ROC AUC={avg_xgb['roc_auc']:.4f}")
    print(f"Random Forest: 정확도={avg_rf['accuracy']:.4f}, F1={avg_rf['f1']:.4f}, ROC AUC={avg_rf['roc_auc']:.4f}")
    print(f"LSTM: 정확도={avg_lstm['accuracy']:.4f}, F1={avg_lstm['f1']:.4f}")


def main():
    """메인 함수"""
    print("=" * 80)
    print("Walk-Forward Validation 실제 실행 및 유효성 검증")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['year'] = df['entry_time'].dt.year
    
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    print(f"연도별 데이터: {df['year'].value_counts().sort_index().to_dict()}")
    
    # Walk-Forward Validation 실행
    results = walk_forward_validation(df, n_splits=3)
    
    # 결과 출력
    print_walk_forward_results(results)
    
    return results


if __name__ == "__main__":
    results = main()
