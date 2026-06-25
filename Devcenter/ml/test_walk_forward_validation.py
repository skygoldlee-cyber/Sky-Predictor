# -*- coding: utf-8 -*-
"""
Walk-Forward Validation 테스트
현실적인 테스트를 위한 Walk-Forward Validation
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "ml_data"
MODELS_DIR = Path(__file__).parent / "ml_models"


def walk_forward_validation():
    """Walk-Forward Validation 테스트"""
    print("=" * 80)
    print("Walk-Forward Validation 테스트")
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
    
    # 연도별 Walk-Forward Validation
    results = []
    
    for test_year in range(2020, 2027):
        # 훈련 데이터: test_year 이전 모든 데이터
        train_df = df[df['year'] < test_year].copy()
        
        # 테스트 데이터: test_year 데이터
        test_df = df[df['year'] == test_year].copy()
        
        if len(train_df) < 50 or len(test_df) < 10:
            print(f"\n{test_year}년: 데이터 부족, 건너뜀")
            continue
        
        print(f"\n{'='*80}")
        print(f"{test_year}년 Walk-Forward Validation")
        print(f"{'='*80}")
        print(f"훈련 데이터: {len(train_df)}건 ({train_df['year'].min()} ~ {train_df['year'].max()})")
        print(f"테스트 데이터: {len(test_df)}건 ({test_year})")
        
        # 훈련 데이터 준비
        X_train = train_df[feature_cols].copy().fillna(0).astype(float)
        y_train = train_df['is_win'].copy()
        
        # 테스트 데이터 준비
        X_test = test_df[feature_cols].copy().fillna(0).astype(float)
        y_test = test_df['is_win'].copy()
        
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
        
        model.fit(X_train, y_train)
        
        # 테스트 데이터 예측
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 성능 평가
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if len(y_test) > 1 else 0
        
        print(f"\n모델 성과:")
        print(f"  정확도: {accuracy:.4f}")
        print(f"  정밀도: {precision:.4f}")
        print(f"  재현율: {recall:.4f}")
        print(f"  F1 점수: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        
        # Threshold 최적화
        best_threshold = 0.5
        best_pnl = float('-inf')
        
        print(f"\nThreshold 최적화:")
        
        for threshold in np.arange(0.4, 0.9, 0.05):
            filtered_mask = y_pred_proba >= threshold
            filtered_df = test_df[filtered_mask].copy()
            
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
            'year': test_year,
            'train_years': f"{train_df['year'].min()}-{train_df['year'].max()}",
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'best_threshold': best_threshold,
            'best_pnl': best_pnl
        })
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("Walk-Forward Validation 결과 요약")
    print(f"{'='*80}")
    
    total_pnl = 0
    total_trades = 0
    total_wins = 0
    
    for result in results:
        print(f"\n{result['year']}년 (훈련: {result['train_years']}):")
        print(f"  최적 threshold: {result['best_threshold']:.2f}")
        print(f"  최적 총 PnL: {result['best_pnl']:,.0f}원")
        print(f"  정확도: {result['accuracy']:.4f}")
        print(f"  ROC AUC: {result['roc_auc']:.4f}")
        
        total_pnl += result['best_pnl']
    
    # 전체 성과
    print(f"\n{'='*80}")
    print("전체 성과")
    print(f"{'='*80}")
    print(f"총 PnL: {total_pnl:,.0f}원")
    
    return results, total_pnl


if __name__ == "__main__":
    results, total_pnl = walk_forward_validation()
