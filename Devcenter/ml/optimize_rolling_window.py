# -*- coding: utf-8 -*-
"""
롤링 윈도우 모델 최적화
롤링 윈도우 크기, 하이퍼파라미터 튜닝
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


def optimize_rolling_window_size():
    """롤링 윈도우 크기 최적화"""
    print("=" * 80)
    print("롤링 윈도우 크기 최적화")
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
    
    # 롤링 윈도우 크기 테스트
    window_sizes = [1, 2, 3, 4, 5]
    results = []
    
    for window_size in window_sizes:
        print(f"\n{'='*80}")
        print(f"롤링 윈도우 크기: {window_size}년")
        print(f"{'='*80}")
        
        # 롤링 윈도우 데이터 선택
        end_date = df['entry_time'].max()
        start_date = end_date - pd.DateOffset(years=window_size)
        
        df_window = df[(df['entry_time'] >= start_date) & (df['entry_time'] <= end_date)].copy()
        
        if len(df_window) < 100:
            print(f"롤링 윈도우 데이터 부족: {len(df_window)}건")
            continue
        
        print(f"롤링 윈도우 기간: {start_date} ~ {end_date}")
        print(f"롤링 윈도우 데이터: {len(df_window)}건")
        
        # 시간 기반 분할
        df_window_sorted = df_window.sort_values('entry_time').reset_index(drop=True)
        split_idx = int(len(df_window_sorted) * 0.7)
        
        X = df_window[feature_cols].copy().fillna(0).astype(float)
        y = df_window['is_win'].copy()
        
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]
        
        # XGBoost 모델 학습
        model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.05,
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
        
        # 결과 저장
        results.append({
            'window_size': window_size,
            'data_count': len(df_window),
            'best_threshold': best_threshold,
            'best_pnl': best_pnl
        })
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("롤링 윈도우 크기 최적화 결과 요약")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\n롤링 윈도우 크기: {result['window_size']}년")
        print(f"  데이터 수: {result['data_count']}건")
        print(f"  최적 threshold: {result['best_threshold']:.2f}")
        print(f"  최적 총 PnL: {result['best_pnl']:,.0f}원")
    
    # 최적 롤링 윈도우 크기 선택
    best_result = max(results, key=lambda x: x['best_pnl'])
    
    print(f"\n{'='*80}")
    print("최적 롤링 윈도우 크기")
    print(f"{'='*80}")
    print(f"최적 크기: {best_result['window_size']}년")
    print(f"최적 총 PnL: {best_result['best_pnl']:,.0f}원")
    print(f"최적 threshold: {best_result['best_threshold']:.2f}")
    
    return results, best_result


def optimize_hyperparameters():
    """하이퍼파라미터 튜닝"""
    print(f"\n{'='*80}")
    print("하이퍼파라미터 튜닝")
    print(f"{'='*80}")
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    
    # 롤링 윈도우 데이터 선택 (2년)
    end_date = df['entry_time'].max()
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
    
    # 하이퍼파라미터 조합 테스트
    param_combinations = [
        {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.05},
        {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05},
        {'n_estimators': 50, 'max_depth': 6, 'learning_rate': 0.05},
        {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1},
        {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
    ]
    
    results = []
    
    for params in param_combinations:
        print(f"\n{'='*80}")
        print(f"하이퍼파라미터: {params}")
        print(f"{'='*80}")
        
        model = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
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
        
        for threshold in np.arange(0.4, 0.9, 0.05):
            filtered_mask = y_pred_proba >= threshold
            filtered_df = df_window[filtered_mask].copy()
            
            if len(filtered_df) == 0:
                continue
            
            total_pnl = filtered_df['net_krw'].sum()
            
            if total_pnl > best_pnl:
                best_pnl = total_pnl
                best_threshold = threshold
        
        print(f"최적 threshold: {best_threshold:.2f}")
        print(f"최적 총 PnL: {best_pnl:,.0f}원")
        
        results.append({
            'params': params,
            'best_threshold': best_threshold,
            'best_pnl': best_pnl
        })
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("하이퍼파라미터 튜닝 결과 요약")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\n하이퍼파라미터: {result['params']}")
        print(f"  최적 threshold: {result['best_threshold']:.2f}")
        print(f"  최적 총 PnL: {result['best_pnl']:,.0f}원")
    
    # 최적 하이퍼파라미터 선택
    best_result = max(results, key=lambda x: x['best_pnl'])
    
    print(f"\n{'='*80}")
    print("최적 하이퍼파라미터")
    print(f"{'='*80}")
    print(f"최적 하이퍼파라미터: {best_result['params']}")
    print(f"최적 총 PnL: {best_result['best_pnl']:,.0f}원")
    print(f"최적 threshold: {best_result['best_threshold']:.2f}")
    
    return results, best_result


if __name__ == "__main__":
    # 롤링 윈도우 크기 최적화
    window_results, best_window = optimize_rolling_window_size()
    
    # 하이퍼파라미터 튜닝
    param_results, best_param = optimize_hyperparameters()
