# -*- coding: utf-8 -*-
"""
모든 연도 특화 XGBoost 거래 필터링 모델 일괄 구현
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


def train_xgboost_model_year(df: pd.DataFrame, year: int) -> xgb.XGBClassifier:
    """특정 연도 데이터만 사용하여 XGBoost 모델 학습"""
    # 특정 연도 데이터만 선택
    df_year = df[df['year'] == year].copy()
    
    print(f"\n{year}년 데이터: {len(df_year)}건")
    print(f"{year}년 승률: {df_year['is_win'].mean() * 100:.2f}%")
    
    if len(df_year) < 50:
        print(f"{year}년 데이터가 부족하여 특화 모델 학습 불가")
        return None, None
    
    # 피처 선택
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    X = df_year[feature_cols].copy().fillna(0).astype(float)
    y = df_year['is_win'].copy()
    
    # 시간 기반 분할 (연도 내에서 훈련/검증 분할)
    df_year_sorted = df_year.sort_values('entry_time').reset_index(drop=True)
    split_idx = int(len(df_year_sorted) * 0.7)
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    print(f"{year}년 내 분할:")
    print(f"  훈련 데이터: {len(X_train)}건 (승률: {y_train.mean() * 100:.2f}%)")
    print(f"  검증 데이터: {len(X_val)}건 (승률: {y_val.mean() * 100:.2f}%)")
    
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
    
    # 검증 데이터 예측
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # 성능 평가
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_pred_proba) if len(y_val) > 1 else 0
    
    print(f"검증 데이터 성과:")
    print(f"  정확도: {accuracy:.4f}")
    print(f"  정밀도: {precision:.4f}")
    print(f"  재현율: {recall:.4f}")
    print(f"  F1 점수: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    return model, df_year


def optimize_threshold_year(df_year: pd.DataFrame, model: xgb.XGBClassifier, year: int) -> float:
    """특정 연도 데이터로 threshold 최적화"""
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    X = df_year[feature_cols].copy().fillna(0).astype(float)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    best_threshold = 0.5
    best_pnl = float('-inf')
    
    print(f"\n{year}년 데이터로 threshold 최적화:")
    
    for threshold in np.arange(0.4, 0.9, 0.05):
        filtered_mask = y_pred_proba >= threshold
        filtered_df = df_year[filtered_mask].copy()
        
        if len(filtered_df) == 0:
            continue
        
        total_pnl = filtered_df['net_krw'].sum()
        win_rate = filtered_df['is_win'].mean() * 100
        
        print(f"  Threshold {threshold:.2f}: 총 PnL {total_pnl:,.0f}원, 승률 {win_rate:.2f}%, 거래 수 {len(filtered_df)}건")
        
        if total_pnl > best_pnl:
            best_pnl = total_pnl
            best_threshold = threshold
    
    print(f"최적 threshold: {best_threshold:.2f}")
    print(f"최적 총 PnL: {best_pnl:,.0f}원")
    
    return best_threshold, best_pnl


def main():
    """메인 함수"""
    print("=" * 80)
    print("모든 연도 특화 XGBoost 거래 필터링 모델 일괄 구현")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['year'] = df['entry_time'].dt.year
    
    print(f"\n전체 데이터: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 결과 저장
    results = {}
    
    # 2020-2023년 특화 모델 구현
    for year in [2020, 2021, 2022, 2023]:
        print(f"\n{'='*80}")
        print(f"{year}년 특화 모델")
        print(f"{'='*80}")
        
        # 모델 학습
        model, df_year = train_xgboost_model_year(df, year)
        
        if model is None:
            print(f"{year}년 특화 모델 학습 불가 (데이터 부족)")
            continue
        
        # threshold 최적화
        best_threshold, best_pnl = optimize_threshold_year(df_year, model, year)
        
        # 모델 저장
        model_path = MODELS_DIR / f"trade_filter_xgboost_{year}.json"
        model.save_model(str(model_path))
        print(f"모델 저장 완료: {model_path}")
        
        # 결과 저장
        results[year] = {
            'threshold': best_threshold,
            'total_pnl': best_pnl
        }
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("결과 요약")
    print(f"{'='*80}")
    
    for year, result in results.items():
        print(f"{year}년: Threshold {result['threshold']:.2f}, 총 PnL {result['total_pnl']:,.0f}원")
    
    return results


if __name__ == "__main__":
    results = main()
