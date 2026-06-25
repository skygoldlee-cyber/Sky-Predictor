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


def load_optimized_data() -> Tuple[pd.DataFrame, pd.Series]:
    """최적화된 데이터 로드"""
    df = pd.read_csv(DATA_DIR / "optimized_trades.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    timestamps = df['entry_time']
    
    print(f"최적화된 데이터 로드 완료: {len(df)}건")
    print(f"승률: {df['is_win'].mean() * 100:.2f}%")
    print(f"총 PnL: {df['net_krw'].sum():,.0f} 원")
    print(f"기간: {timestamps.min()} ~ {timestamps.max()}")
    
    return df, timestamps


def prepare_time_series_data(df: pd.DataFrame, timestamps: pd.Series, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, np.ndarray]:
    """시계열 데이터 준비 (시간 기반 분할로 데이터 누설 방지)"""
    # 피쳐 선택
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    # 거래 순서대로 정렬
    df_sorted = df.sort_values('entry_time').reset_index(drop=True)
    timestamps_sorted = timestamps.sort_values().reset_index(drop=True)
    
    # 피쳐 데이터 추출
    X = df_sorted[feature_cols].values
    
    # 타겟 변수 (승/패)
    y = df_sorted['is_win'].values
    
    # 결측치 처리
    X = np.nan_to_num(X, nan=0.0)
    
    # 스케일링 (훈련 데이터에만 fit, 검증/테스트에는 transform)
    train_mask = (timestamps_sorted.dt.year >= 2019) & (timestamps_sorted.dt.year <= 2023)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X[train_mask])  # 훈련 데이터에만 fit
    X_scaled = scaler.transform(X)  # 전체 데이터에 transform
    
    # 시계열 시퀀스 생성 (시간 순서 유지)
    X_sequences = []
    y_sequences = []
    year_sequences = []
    
    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:i+sequence_length])
        y_sequences.append(y[i+sequence_length])
        year_sequences.append(timestamps_sorted.iloc[i+sequence_length].year)
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    year_sequences = np.array(year_sequences)
    
    print(f"\n시계열 데이터 준비 완료")
    print(f"시퀀스 길이: {sequence_length}")
    print(f"시퀀스 수: {len(X_sequences)}")
    print(f"승률: {y_sequences.mean() * 100:.2f}%")
    
    return X_sequences, y_sequences, scaler, year_sequences


def build_lstm_model(input_shape: int) -> Sequential:
    """LSTM 모델 구축 (복잡도 감소, 정규화 강화)"""
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape,
             kernel_regularizer=l2(0.03), recurrent_regularizer=l2(0.03)),
        Dropout(0.6),  # 0.5→0.6 (드롭아웃 증가)
        LSTM(16, return_sequences=False,
             kernel_regularizer=l2(0.03), recurrent_regularizer=l2(0.03)),
        Dropout(0.6),  # 0.5→0.6 (드롭아웃 증가)
        Dense(8, activation='relu', kernel_regularizer=l2(0.03)),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0003),  # 0.0005→0.0003 (학습률 감소)
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_lstm_model(X: np.ndarray, y: np.ndarray, year_sequences: np.ndarray) -> Tuple[Sequential, np.ndarray, np.ndarray]:
    """LSTM 모델 학습 (시간 기반 분할로 데이터 누설 방지)"""
    # 시간 기반 train/validation/test 분할 (데이터 누설 방지)
    # 2019-2023: 훈련, 2024: 검증, 2025-2026: 테스트
    train_mask = (year_sequences >= 2019) & (year_sequences <= 2023)
    val_mask = (year_sequences == 2024)
    test_mask = (year_sequences >= 2025)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\n시간 기반 분할 (데이터 누설 방지):")
    print(f"  훈련 데이터 (2019-2023): {len(X_train)}건 (승률: {y_train.mean() * 100:.2f}%)")
    print(f"  검증 데이터 (2024): {len(X_val)}건 (승률: {y_val.mean() * 100:.2f}%)")
    print(f"  테스트 데이터 (2025-2026): {len(X_test)}건 (승률: {y_test.mean() * 100:.2f}%)")
    
    # 모델 구축
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # 모델 학습
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
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
    
    print(f"\n테스트 데이터 성능 (샘플 외):")
    print(f"  정확도: {accuracy:.4f}")
    print(f"  정밀도: {precision:.4f}")
    print(f"  재현율: {recall:.4f}")
    print(f"  F1 점수: {f1:.4f}")
    
    return model, X_test, y_test


def optimize_for_total_pnl_exit(df: pd.DataFrame, model: Sequential, X: np.ndarray, 
                                scaler: MinMaxScaler, sequence_length: int = 10) -> tuple:
    """총 PnL 기반 threshold 최적화 (청산 타이밍)"""
    best_pnl = 0
    best_threshold = 0.5
    best_df = None
    
    print(f"\n{'='*100}")
    print("총 PnL 기반 threshold 최적화 (청산 타이밍)")
    print(f"{'='*100}")
    
    for threshold in np.arange(0.5, 0.9, 0.05):
        df_optimized = optimize_exit_timing(df, model, X, scaler, sequence_length, threshold)
        total_pnl = df_optimized['net_krw'].sum()
        win_rate = df_optimized['is_win'].mean() * 100
        
        print(f"Threshold {threshold:.2f}: 총 PnL {total_pnl:,.0f}원, 승률 {win_rate:.2f}%, 거래 수 {len(df_optimized)}건")
        
        if total_pnl > best_pnl:
            best_pnl = total_pnl
            best_threshold = threshold
            best_df = df_optimized
    
    print(f"\n최적 threshold: {best_threshold:.2f}")
    print(f"최적 총 PnL: {best_pnl:,.0f}원")
    print(f"최적 승률: {best_df['is_win'].mean() * 100:.2f}%")
    print(f"최적 거래 수: {len(best_df)}건")
    
    return best_threshold, best_df


def optimize_for_sharpe_ratio_exit(df: pd.DataFrame, model: Sequential, X: np.ndarray, 
                                   scaler: MinMaxScaler, sequence_length: int = 10) -> tuple:
    """샤프 비율 기반 threshold 최적화 (청산 타이밍)"""
    best_sharpe = 0
    best_threshold = 0.5
    best_df = None
    
    print(f"\n{'='*100}")
    print("샤프 비율 기반 threshold 최적화 (청산 타이밍)")
    print(f"{'='*100}")
    
    for threshold in np.arange(0.5, 0.9, 0.05):
        df_optimized = optimize_exit_timing(df, model, X, scaler, sequence_length, threshold)
        returns = df_optimized['net_krw'].values
        
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            total_pnl = df_optimized['net_krw'].sum()
            win_rate = df_optimized['is_win'].mean() * 100
            
            print(f"Threshold {threshold:.2f}: 샤프 비율 {sharpe_ratio:.4f}, 총 PnL {total_pnl:,.0f}원, 승률 {win_rate:.2f}%, 거래 수 {len(df_optimized)}건")
            
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_threshold = threshold
                best_df = df_optimized
    
    print(f"\n최적 threshold: {best_threshold:.2f}")
    print(f"최적 샤프 비율: {best_sharpe:.4f}")
    print(f"최적 총 PnL: {best_df['net_krw'].sum():,.0f}원")
    print(f"최적 승률: {best_df['is_win'].mean() * 100:.2f}%")
    print(f"최적 거래 수: {len(best_df)}건")
    
    return best_threshold, best_df


def filter_by_time_and_month(df: pd.DataFrame) -> pd.DataFrame:
    """시간대별 및 월별 필터링 (완화: 시간대/월별 필터링 제거)"""
    df_filtered = df.copy()
    
    # 시간대별 필터링 제거 (거래 수 확보를 위해)
    # df_filtered = df_filtered[~df_filtered['entry_hour'].isin([9, 10, 11])]
    
    # 월별 필터링 제거 (거래 수 확보를 위해)
    # df_filtered['entry_month'] = pd.to_datetime(df_filtered['entry_time']).dt.month
    # df_filtered = df_filtered[~df_filtered['entry_month'].isin([3, 5, 6])]
    
    print(f"\n시간대별 및 월별 필터링 완화 (제거):")
    print(f"  필터링 전: {len(df)}건")
    print(f"  필터링 후: {len(df_filtered)}건")
    print(f"  제외된 거래: {len(df) - len(df_filtered)}건")
    
    return df_filtered


def optimize_exit_timing(df: pd.DataFrame, model: Sequential, X: np.ndarray, 
                        scaler: MinMaxScaler, sequence_length: int = 10, threshold: float = 0.7) -> pd.DataFrame:
    """청산 타이밍 최적화"""
    # 전체 데이터에 대한 예측
    y_pred_proba = model.predict(X, verbose=0).flatten()
    
    # 시퀀스 길이만큼의 데이터는 예측 불가능하므로 제외
    df_optimized = df.sort_values('entry_time').reset_index(drop=True)
    df_optimized = df_optimized.iloc[sequence_length:].copy()
    df_optimized['exit_quality_score'] = y_pred_proba
    
    # 고품질 청산만 선택
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
    df, timestamps = load_optimized_data()
    
    # 2) 시계열 데이터 준비
    X, y, scaler, year_sequences = prepare_time_series_data(df, timestamps, sequence_length=10)
    
    # 3) LSTM 모델 학습
    model, X_test, y_test = train_lstm_model(X, y, year_sequences)
    
    # 4) 모델 저장
    model_path = OUTPUT_DIR / "exit_timing_lstm.keras"
    model.save(model_path)
    print(f"\n모델 저장 완료: {model_path}")
    
    # 5) 총 PnL 기반 threshold 최적화 적용
    best_threshold, df_final = optimize_for_total_pnl_exit(df, model, X, scaler, sequence_length=10)
    
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
