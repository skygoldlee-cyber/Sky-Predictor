# -*- coding: utf-8 -*-
"""
2023년 특화 XGBoost 거래 필터링 모델 개선
threshold 조정으로 성과 개선
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


def improve_threshold_2023():
    """2023년 threshold 개선"""
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['year'] = df['entry_time'].dt.year
    
    # 2023년 데이터만 선택
    df_2023 = df[df['year'] == 2023].copy()
    
    print(f"2023년 데이터: {len(df_2023)}건")
    print(f"2023년 승률: {df_2023['is_win'].mean() * 100:.2f}%")
    
    # 기존 모델 로드
    model = xgb.XGBClassifier()
    model.load_model(str(MODELS_DIR / "trade_filter_xgboost_2023.json"))
    
    # 피처 선택
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    X = df_2023[feature_cols].copy().fillna(0).astype(float)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # threshold 최적화 (더 높은 범위)
    best_threshold = 0.50
    best_pnl = float('-inf')
    
    print(f"\n2023년 데이터로 threshold 최적화 (개선):")
    
    for threshold in np.arange(0.50, 0.85, 0.02):
        filtered_mask = y_pred_proba >= threshold
        filtered_df = df_2023[filtered_mask].copy()
        
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
    
    return best_threshold, best_pnl


def main():
    """메인 함수"""
    print("=" * 80)
    print("2023년 특화 모델 threshold 개선")
    print("=" * 80)
    
    best_threshold, best_pnl = improve_threshold_2023()
    
    print(f"\n개선된 threshold: {best_threshold:.2f}")
    print(f"개선된 총 PnL: {best_pnl:,.0f}원")
    
    return best_threshold, best_pnl


if __name__ == "__main__":
    best_threshold, best_pnl = main()
