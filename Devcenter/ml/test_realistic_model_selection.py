# -*- coding: utf-8 -*-
"""
현실적 모델 선택 테스트
전년도 모델을 현재 연도에 적용
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


def test_realistic_model_selection():
    """현실적 모델 선택 테스트"""
    print("=" * 80)
    print("현실적 모델 선택 테스트 (전년도 모델을 현재 연도에 적용)")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['year'] = df['entry_time'].dt.year
    
    print(f"\n전체 데이터: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 피처 선택
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    # 연도별 threshold 설정
    threshold_map = {
        2019: 0.45,
        2020: 0.50,
        2021: 0.55,
        2022: 0.55,
        2023: 0.50,
        2024: 0.45,
        2025: 0.50,
        2026: 0.50
    }
    
    # 현실적 모델 선택 테스트
    print(f"\n{'='*80}")
    print("현실적 모델 선택 테스트 결과")
    print(f"{'='*80}")
    
    total_pnl = 0
    total_trades = 0
    total_wins = 0
    
    for year in range(2020, 2027):
        # 전년도 모델 로드
        prev_year = year - 1
        model_path = MODELS_DIR / f"trade_filter_xgboost_{prev_year}.json"
        
        if not model_path.exists():
            print(f"\n{year}년: {prev_year}년 모델 없음, 건너뜀")
            continue
        
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        
        # 현재 연도 데이터
        df_year = df[df['year'] == year].copy()
        
        if len(df_year) == 0:
            print(f"\n{year}년: 데이터 없음, 건너뜀")
            continue
        
        X = df_year[feature_cols].copy().fillna(0).astype(float)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # 전년도 threshold 사용
        threshold = threshold_map.get(prev_year, 0.50)
        
        filtered_mask = y_pred_proba >= threshold
        filtered_df = df_year[filtered_mask].copy()
        
        if len(filtered_df) == 0:
            print(f"\n{year}년: 필터링된 거래 없음")
            continue
        
        total_pnl_year = filtered_df['net_krw'].sum()
        win_rate = filtered_df['is_win'].mean() * 100
        
        total_pnl += total_pnl_year
        total_trades += len(filtered_df)
        total_wins += filtered_df['is_win'].sum()
        
        print(f"\n{year}년 ({prev_year}년 모델 사용):")
        print(f"  Threshold: {threshold:.2f}")
        print(f"  필터링 전: {len(df_year)}건 (승률: {df_year['is_win'].mean() * 100:.2f}%)")
        print(f"  필터링 후: {len(filtered_df)}건 (승률: {win_rate:.2f}%)")
        print(f"  총 PnL: {total_pnl_year:,.0f}원")
    
    # 전체 성과
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    print(f"\n{'='*80}")
    print("전체 성과")
    print(f"{'='*80}")
    print(f"총 거래 수: {total_trades}건")
    print(f"총 승률: {overall_win_rate:.2f}%")
    print(f"총 PnL: {total_pnl:,.0f}원")
    
    return total_pnl, total_trades, overall_win_rate


if __name__ == "__main__":
    total_pnl, total_trades, overall_win_rate = test_realistic_model_selection()
