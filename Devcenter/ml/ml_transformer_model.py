"""
딥러닝 기반 접근 - Transformer 모델
시계열 데이터에 적합한 Transformer 분류 모델 구현
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
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


def create_sequences(df: pd.DataFrame, feature_cols: List[str], sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """시퀀스 데이터 생성"""
    sequences = []
    targets = []
    
    for i in range(len(df) - sequence_length):
        seq = df[feature_cols].iloc[i:i+sequence_length].values
        target = df['is_win'].iloc[i+sequence_length]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


def build_transformer_model(input_shape: int, num_heads: int = 4, ff_dim: int = 64, dropout: float = 0.3) -> 'Model':
    """Transformer 모델 구축"""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model
    except ImportError:
        print("TensorFlow not installed, skipping...")
        return None
    
    # 입력 레이어
    inputs = layers.Input(shape=input_shape)
    
    # Multi-Head Attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=input_shape[1] // num_heads,
        dropout=dropout
    )(inputs, inputs)
    
    # Add & Norm
    attention_output = layers.Add()([inputs, attention_output])
    attention_output = layers.LayerNormalization()(attention_output)
    
    # Feed Forward
    ff_output = layers.Dense(ff_dim, activation='relu')(attention_output)
    ff_output = layers.Dropout(dropout)(ff_output)
    ff_output = layers.Dense(input_shape[1])(ff_output)
    
    # Add & Norm
    ff_output = layers.Add()([attention_output, ff_output])
    ff_output = layers.LayerNormalization()(ff_output)
    
    # Global Average Pooling
    pooled = layers.GlobalAveragePooling1D()(ff_output)
    
    # Dense layers
    x = layers.Dense(64, activation='relu')(pooled)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_transformer(train_data: pd.DataFrame, test_data: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """Transformer 모델 학습 및 평가"""
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        print("  TensorFlow not installed, skipping...")
        return {'accuracy': 0, 'f1': 0, 'roc_auc': 0}
    
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    # 시퀀스 생성
    sequence_length = 10
    X_train_seq, y_train_seq = create_sequences(train_data, feature_cols, sequence_length)
    X_test_seq, y_test_seq = create_sequences(test_data, feature_cols, sequence_length)
    
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        print("  데이터 부족으로 시퀀스 생성 불가")
        return {'accuracy': 0, 'f1': 0, 'roc_auc': 0}
    
    print(f"  훈련 시퀀스: {len(X_train_seq)}건")
    print(f"  테스트 시퀀스: {len(X_test_seq)}건")
    
    # 모델 구축
    model = build_transformer_model(X_train_seq.shape[1:])
    
    if model is None:
        return {'accuracy': 0, 'f1': 0, 'roc_auc': 0}
    
    # Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # 모델 학습
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # 예측
    y_pred_proba = model.predict(X_test_seq, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # 성능 평가
    result = {
        'accuracy': accuracy_score(y_test_seq, y_pred),
        'f1': f1_score(y_test_seq, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test_seq, y_pred_proba) if len(y_test_seq) > 1 else 0
    }
    
    print(f"  정확도: {result['accuracy']:.4f}")
    print(f"  F1 점수: {result['f1']:.4f}")
    print(f"  ROC AUC: {result['roc_auc']:.4f}")
    
    return result


def walk_forward_validation(df: pd.DataFrame, feature_cols: List[str], n_splits: int = 3) -> List[Dict]:
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
        
        # Transformer 모델 학습 및 평가
        print("\nTransformer 모델...")
        transformer_result = train_transformer(train_data, test_data, feature_cols)
        
        fold_result = {
            'train_years': train_years,
            'test_year': test_year,
            'transformer': transformer_result
        }
        
        results.append(fold_result)
    
    return results


def print_walk_forward_results(results: List[Dict]):
    """Walk-Forward Validation 결과 출력"""
    print(f"\n{'='*80}")
    print("Walk-Forward Validation 결과 요약 (Transformer)")
    print(f"{'='*80}")
    
    for i, result in enumerate(results):
        print(f"\nFold {i+1} ({result['train_years'][0]}-{result['train_years'][-1]} → {result['test_year']}):")
        print(f"  Transformer: 정확도={result['transformer']['accuracy']:.4f}, F1={result['transformer']['f1']:.4f}, ROC AUC={result['transformer']['roc_auc']:.4f}")
    
    # 평균 성과
    print(f"\n{'='*80}")
    print("평균 성과")
    print(f"{'='*80}")
    
    avg_transformer = {
        'accuracy': np.mean([r['transformer']['accuracy'] for r in results]),
        'f1': np.mean([r['transformer']['f1'] for r in results]),
        'roc_auc': np.mean([r['transformer']['roc_auc'] for r in results])
    }
    
    print(f"Transformer: 정확도={avg_transformer['accuracy']:.4f}, F1={avg_transformer['f1']:.4f}, ROC AUC={avg_transformer['roc_auc']:.4f}")


def main():
    """메인 함수"""
    print("=" * 80)
    print("딥러닝 기반 접근 - Transformer 모델")
    print("=" * 80)
    
    # 데이터 로드
    df = load_data()
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 피처 준비
    df, feature_cols = prepare_features(df)
    print(f"피처 준비 완료: {len(feature_cols)}개")
    
    # Walk-Forward Validation 실행
    results = walk_forward_validation(df, feature_cols, n_splits=3)
    
    # 결과 출력
    print_walk_forward_results(results)
    
    return results


if __name__ == "__main__":
    results = main()
