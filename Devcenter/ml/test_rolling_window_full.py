# -*- coding: utf-8 -*-
"""
롤링 윈도우 모델 전체 테스트
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


def test_rolling_window_full():
    """롤링 윈도우 모델 전체 테스트"""
    print("=" * 80)
    print("롤링 윈도우 모델 전체 테스트")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['year'] = df['entry_time'].dt.year
    
    print(f"\n전체 데이터: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 롤링 윈도우 모델 로드
    model = xgb.XGBClassifier()
    model.load_model(str(MODELS_DIR / "trade_filter_xgboost_rolling.json"))
    
    # 피처 선택
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    X = df[feature_cols].copy().fillna(0).astype(float)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # 전체 데이터에 대한 threshold 최적화
    best_threshold = 0.5
    best_pnl = float('-inf')
    
    print(f"\n전체 데이터로 threshold 최적화:")
    
    for threshold in np.arange(0.4, 0.9, 0.05):
        filtered_mask = y_pred_proba >= threshold
        filtered_df = df[filtered_mask].copy()
        
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
    
    # 연도별 성과 분석
    print(f"\n{'='*80}")
    print("연도별 성과 분석")
    print(f"{'='*80}")
    
    for year in sorted(df['year'].unique()):
        df_year = df[df['year'] == year].copy()
        X_year = X[df['year'] == year]
        y_pred_proba_year = y_pred_proba[df['year'] == year]
        
        filtered_mask = y_pred_proba_year >= best_threshold
        filtered_df = df_year[filtered_mask].copy()
        
        if len(filtered_df) == 0:
            print(f"\n{year}년: 필터링된 거래 없음")
            continue
        
        total_pnl = filtered_df['net_krw'].sum()
        win_rate = filtered_df['is_win'].mean() * 100
        
        print(f"\n{year}년:")
        print(f"  필터링 전: {len(df_year)}건 (승률: {df_year['is_win'].mean() * 100:.2f}%)")
        print(f"  필터링 후: {len(filtered_df)}건 (승률: {win_rate:.2f}%)")
        print(f"  총 PnL: {total_pnl:,.0f}원")
    
    return best_threshold, best_pnl


if __name__ == "__main__":
    best_threshold, best_pnl = test_rolling_window_full()
