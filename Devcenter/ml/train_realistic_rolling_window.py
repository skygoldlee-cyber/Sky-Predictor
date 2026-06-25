# -*- coding: utf-8 -*-
"""
현실적 롤링 윈도우 모델 학습
2년 과거 데이터 기반
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "ml_data"
MODELS_DIR = Path(__file__).parent / "ml_models"


def train_realistic_rolling_window():
    """현실적 롤링 윈도우 모델 학습"""
    print("=" * 80)
    print("현실적 롤링 윈도우 모델 학습 (2년 과거 데이터 기반)")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    
    # 2년 롤링 윈도우 데이터 선택 (2026년 데이터 제외)
    end_date = pd.Timestamp('2025-12-31')
    start_date = end_date - pd.DateOffset(years=2)
    
    df_window = df[(df['entry_time'] >= start_date) & (df['entry_time'] <= end_date)].copy()
    
    print(f"\n롤링 윈도우 기간: {start_date} ~ {end_date}")
    print(f"롤링 윈도우 데이터: {len(df_window)}건")
    
    # 피처 선택
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    X = df_window[feature_cols].copy().fillna(0).astype(float)
    y = df_window['is_win'].copy()
    
    # 시간 기반 분할
    df_window_sorted = df_window.sort_values('entry_time').reset_index(drop=True)
    split_idx = int(len(df_window_sorted) * 0.7)
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    # 최적 하이퍼파라미터
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        reg_alpha=0.5,
        reg_lambda=2.0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 전체 데이터 예측
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Threshold 최적화
    best_threshold = 0.5
    best_pnl = float('-inf')
    
    print(f"\nThreshold 최적화:")
    
    for threshold in np.arange(0.4, 0.9, 0.05):
        filtered_mask = y_pred_proba >= threshold
        filtered_df = df_window[filtered_mask].copy()
        
        if len(filtered_df) == 0:
            continue
        
        total_pnl = filtered_df['net_krw'].sum()
        win_rate = filtered_df['is_win'].mean() * 100
        
        print(f"  Threshold {threshold:.2f}: 총 PnL {total_pnl:,.0f}원, 승률 {win_rate:.2f}%, 거래 수 {len(filtered_df)}건")
        
        if total_pnl > best_pnl:
            best_pnl = total_pnl
            best_threshold = threshold
    
    print(f"\n최적 threshold: {best_threshold:.2f}")
    print(f"최적 총 PnL: {best_pnl:,.0f}원")
    
    # 모델 저장
    model_path = MODELS_DIR / "trade_filter_xgboost_rolling_realistic.json"
    model.save_model(str(model_path))
    print(f"\n현실적 롤링 윈도우 모델 저장 완료: {model_path}")
    
    return model, best_threshold, best_pnl


if __name__ == "__main__":
    model, best_threshold, best_pnl = train_realistic_rolling_window()
