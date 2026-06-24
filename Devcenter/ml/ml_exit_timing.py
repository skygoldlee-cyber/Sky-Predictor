# -*- coding: utf-8 -*-
"""
청산 타이밍 최적화 (LSTM)

시계열 데이터를 사용하여 청산 시점을 최적화한다.
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent / "ml_models"
OUTPUT_DIR = Path(__file__).parent / "ml_models"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_optimized_data() -> pd.DataFrame:
    """최적화된 데이터 로드"""
    df = pd.read_csv(DATA_DIR / "optimized_trades.csv")
    
    print(f"최적화된 데이터 로드 완료: {len(df)}건")
    print(f"승률: {df['is_win'].mean() * 100:.2f}%")
    print(f"총 PnL: {df['net_krw'].sum():,.0f} 원")
    
    return df


def prepare_time_series_data(df: pd.DataFrame, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """시계열 데이터 준비"""
    # 피쳐 선택
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    # 거래 순서대로 정렬
    df_sorted = df.sort_values('entry_time').reset_index(drop=True)
    
    # 피쳐 데이터 추출
    X = df_sorted[feature_cols].values
    
    # 타겟 변수 (승/패)
    y = df_sorted['is_win'].values
    
    # 결측치 처리
    X = np.nan_to_num(X, nan=0.0)
    
    # 스케일링
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 시계열 시퀀스 생성
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:i+sequence_length])
        y_sequences.append(y[i+sequence_length])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    print(f"\n시계열 데이터 준비 완료")
    print(f"시퀀스 길이: {sequence_length}")
    print(f"시퀀스 수: {len(X_sequences)}")
    print(f"승률: {y_sequences.mean() * 100:.2f}%")
    
    return X_sequences, y_sequences, scaler


def build_lstm_model(input_shape: int) -> Sequential:
    """LSTM 모델 구축"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_lstm_model(X: np.ndarray, y: np.ndarray) -> Tuple[Sequential, np.ndarray, np.ndarray]:
    """LSTM 모델 학습"""
    # 학습/테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n학습 데이터: {len(X_train)}건 (승률: {y_train.mean() * 100:.2f}%)")
    print(f"테스트 데이터: {len(X_test)}건 (승률: {y_test.mean() * 100:.2f}%)")
    
    # 모델 구축
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # 모델 학습
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    # 테스트 데이터 예측
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # 성능 평가
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n테스트 데이터 성능:")
    print(f"  정확도: {accuracy:.4f}")
    print(f"  정밀도: {precision:.4f}")
    print(f"  재현율: {recall:.4f}")
    print(f"  F1 점수: {f1:.4f}")
    
    return model, X_test, y_test


def optimize_exit_timing(df: pd.DataFrame, model: Sequential, X: np.ndarray, 
                        scaler: MinMaxScaler, sequence_length: int = 10) -> pd.DataFrame:
    """청산 타이밍 최적화"""
    # 전체 데이터에 대한 예측
    y_pred_proba = model.predict(X, verbose=0).flatten()
    
    # 시퀀스 길이만큼의 데이터는 예측 불가능하므로 제외
    df_optimized = df.sort_values('entry_time').reset_index(drop=True)
    df_optimized = df_optimized.iloc[sequence_length:].copy()
    df_optimized['exit_quality_score'] = y_pred_proba
    
    # 고품질 청산만 선택 (threshold 0.7)
    threshold = 0.7
    df_optimized = df_optimized[df_optimized['exit_quality_score'] >= threshold]
    
    print(f"\n청산 타이밍 최적화 결과 (threshold={threshold}):")
    print(f"  최적화 전: {len(df)}건 (승률: {df['is_win'].mean() * 100:.2f}%)")
    print(f"  최적화 후: {len(df_optimized)}건 (승률: {df_optimized['is_win'].mean() * 100:.2f}%)")
    
    return df_optimized


def compare_exit_timing(df_optimized: pd.DataFrame, df_final: pd.DataFrame):
    """청산 타이밍 최적화 전후 성과 비교"""
    print(f"\n{'='*100}")
    print("청산 타이밍 최적화 전후 성과 비교")
    print(f"{'='*100}")
    
    # 최적화 전
    n_trades_opt = len(df_optimized)
    win_rate_opt = df_optimized['is_win'].mean() * 100
    total_pnl_opt = df_optimized['net_krw'].sum()
    avg_pnl_opt = df_optimized['net_krw'].mean()
    
    # 최적화 후
    n_trades_final = len(df_final)
    win_rate_final = df_final['is_win'].mean() * 100
    total_pnl_final = df_final['net_krw'].sum()
    avg_pnl_final = df_final['net_krw'].mean()
    
    print(f"\n{'지표':<20}{'최적화 전':>20}{'최적화 후':>20}{'변화':>20}")
    print(f"{'-'*80}")
    print(f"{'거래 수':<20}{n_trades_opt:>20}{n_trades_final:>20}{n_trades_final - n_trades_opt:>20}")
    print(f"{'승률 (%)':<20}{win_rate_opt:>20.2f}{win_rate_final:>20.2f}{win_rate_final - win_rate_opt:>20.2f}")
    print(f"{'총 PnL (원)':<20}{total_pnl_opt:>20,.0f}{total_pnl_final:>20,.0f}{total_pnl_final - total_pnl_opt:>20,.0f}")
    print(f"{'평균 PnL (원)':<20}{avg_pnl_opt:>20,.0f}{avg_pnl_final:>20,.0f}{avg_pnl_final - avg_pnl_opt:>20,.0f}")
    
    # 연도별 비교
    print(f"\n연도별 성과 비교:")
    print(f"{'연도':<10}{'최적화 전 거래':>15}{'최적화 후 거래':>15}{'최적화 전 승률':>15}{'최적화 후 승률':>15}")
    print(f"{'-'*70}")
    
    for year in sorted(df_optimized['year'].unique()):
        opt_year = df_optimized[df_optimized['year'] == year]
        final_year = df_final[df_final['year'] == year]
        
        print(f"{year:<10}{len(opt_year):>15}{len(final_year):>15}"
              f"{opt_year['is_win'].mean()*100:>15.2f}{final_year['is_win'].mean()*100:>15.2f}")


def main():
    """메인 함수"""
    print("=" * 100)
    print("청산 타이밍 최적화 (LSTM)")
    print("=" * 100)
    
    # 1) 최적화된 데이터 로드
    df = load_optimized_data()
    
    # 2) 시계열 데이터 준비
    X, y, scaler = prepare_time_series_data(df, sequence_length=10)
    
    # 3) LSTM 모델 학습
    model, X_test, y_test = train_lstm_model(X, y)
    
    # 4) 모델 저장
    model_path = OUTPUT_DIR / "exit_timing_lstm.keras"
    model.save(model_path)
    print(f"\n모델 저장 완료: {model_path}")
    
    # 5) 청산 타이밍 최적화
    df_final = optimize_exit_timing(df, model, X, scaler, sequence_length=10)
    
    # 6) 성과 비교
    compare_exit_timing(df, df_final)
    
    # 7) 최종 결과 요약
    print(f"\n{'='*100}")
    print("최종 결과 요약")
    print(f"{'='*100}")
    
    print(f"\n최종 거래 수: {len(df_final)}건")
    print(f"최종 승률: {df_final['is_win'].mean() * 100:.2f}%")
    print(f"최종 총 PnL: {df_final['net_krw'].sum():,.0f} 원")
    print(f"최종 평균 PnL: {df_final['net_krw'].mean():,.0f} 원")
    
    # 최종 데이터셋 저장
    final_path = OUTPUT_DIR / "final_trades.csv"
    df_final.to_csv(final_path, index=False)
    print(f"\n최종 데이터셋 저장 완료: {final_path}")


if __name__ == "__main__":
    main()
