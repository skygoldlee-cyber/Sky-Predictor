# -*- coding: utf-8 -*-
"""
2019년 특화 XGBoost 거래 필터링 모델
2019년 데이터만 사용하여 학습
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


def train_xgboost_model_2019(df: pd.DataFrame) -> xgb.XGBClassifier:
    """2019년 데이터만 사용하여 XGBoost 모델 학습"""
    # 2019년 데이터만 선택
    df_2019 = df[df['year'] == 2019].copy()
    
    print(f"2019년 데이터: {len(df_2019)}건")
    print(f"2019년 승률: {df_2019['is_win'].mean() * 100:.2f}%")
    
    if len(df_2019) < 50:
        print("2019년 데이터가 부족하여 특화 모델 학습 불가")
        return None
    
    # 피처 선택
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    X = df_2019[feature_cols].copy().fillna(0).astype(float)
    y = df_2019['is_win'].copy()
    
    # 시간 기반 분할 (2019년 내에서 훈련/검증 분할)
    df_2019_sorted = df_2019.sort_values('entry_time').reset_index(drop=True)
    split_idx = int(len(df_2019_sorted) * 0.7)
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    print(f"\n2019년 내 분할:")
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
    
    print(f"\n검증 데이터 성과:")
    print(f"  정확도: {accuracy:.4f}")
    print(f"  정밀도: {precision:.4f}")
    print(f"  재현율: {recall:.4f}")
    print(f"  F1 점수: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    return model


def optimize_threshold_2019(df: pd.DataFrame, model: xgb.XGBClassifier) -> float:
    """2019년 데이터로 threshold 최적화"""
    df_2019 = df[df['year'] == 2019].copy()
    
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    X = df_2019[feature_cols].copy().fillna(0).astype(float)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    best_threshold = 0.5
    best_pnl = float('-inf')
    
    print(f"\n2019년 데이터로 threshold 최적화:")
    
    for threshold in np.arange(0.4, 0.9, 0.05):
        filtered_mask = y_pred_proba >= threshold
        filtered_df = df_2019[filtered_mask].copy()
        
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
    
    return best_threshold


def main():
    """메인 함수"""
    print("=" * 80)
    print("2019년 특화 XGBoost 거래 필터링 모델")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['year'] = df['entry_time'].dt.year
    
    print(f"\n전체 데이터: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 2019년 데이터만 사용하여 모델 학습
    model = train_xgboost_model_2019(df)
    
    if model is None:
        print("2019년 특화 모델 학습 불가 (데이터 부족)")
        return None, None
    
    # 2019년 데이터로 threshold 최적화
    best_threshold = optimize_threshold_2019(df, model)
    
    # 모델 저장
    model_path = MODELS_DIR / "trade_filter_xgboost_2019.json"
    model.save_model(str(model_path))
    print(f"\n모델 저장 완료: {model_path}")
    
    return model, best_threshold


if __name__ == "__main__":
    model, best_threshold = main()
