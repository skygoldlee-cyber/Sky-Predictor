# -*- coding: utf-8 -*-
"""
거래 필터링 모델 (XGBoost)

승률 예측 모델을 학습하여 고품질 거래만 선택하여 승률을 향상시킨다.
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path(__file__).parent / "ml_data"
OUTPUT_DIR = Path(__file__).parent / "ml_models"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """데이터 로드 및 전처리"""
    # 데이터셋 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    
    # 피쳐 선택
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    # 타겟 변수
    target_col = 'is_win'
    
    # 피쳐와 타겟 분리
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # 결측치 처리
    X = X.fillna(0)
    
    # 데이터 타입 변환
    X = X.astype(float)
    
    print(f"데이터 로드 완료: {len(df)}건")
    print(f"피쳐 수: {len(feature_cols)}")
    print(f"승률: {y.mean() * 100:.2f}%")
    
    return df, X, y


def train_xgboost_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """XGBoost 모델 학습"""
    # 학습/테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n학습 데이터: {len(X_train)}건 (승률: {y_train.mean() * 100:.2f}%)")
    print(f"테스트 데이터: {len(X_test)}건 (승률: {y_test.mean() * 100:.2f}%)")
    
    # XGBoost 모델 학습
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # 교차 검증
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n교차 검증 정확도: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 테스트 데이터 예측
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 성능 평가
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n테스트 데이터 성능:")
    print(f"  정확도: {accuracy:.4f}")
    print(f"  정밀도: {precision:.4f}")
    print(f"  재현율: {recall:.4f}")
    print(f"  F1 점수: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    # 피쳐 중요도
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n피쳐 중요도 (상위 10):")
    print(feature_importance.head(10))
    
    return model


def filter_trades_by_model(df: pd.DataFrame, model: xgb.XGBClassifier, 
                           X: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """모델을 사용하여 거래 필터링"""
    # 승률 예측
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # 필터링
    df_filtered = df.copy()
    df_filtered['win_probability'] = y_pred_proba
    df_filtered = df_filtered[df_filtered['win_probability'] >= threshold]
    
    print(f"\n필터링 결과 (threshold={threshold}):")
    print(f"  필터링 전: {len(df)}건 (승률: {df['is_win'].mean() * 100:.2f}%)")
    print(f"  필터링 후: {len(df_filtered)}건 (승률: {df_filtered['is_win'].mean() * 100:.2f}%)")
    
    return df_filtered


def compare_performance(df_original: pd.DataFrame, df_filtered: pd.DataFrame):
    """필터링 전후 성과 비교"""
    print(f"\n{'='*100}")
    print("필터링 전후 성과 비교")
    print(f"{'='*100}")
    
    # 필터링 전
    n_trades_orig = len(df_original)
    win_rate_orig = df_original['is_win'].mean() * 100
    total_pnl_orig = df_original['net_krw'].sum()
    avg_pnl_orig = df_original['net_krw'].mean()
    
    # 필터링 후
    n_trades_filt = len(df_filtered)
    win_rate_filt = df_filtered['is_win'].mean() * 100
    total_pnl_filt = df_filtered['net_krw'].sum()
    avg_pnl_filt = df_filtered['net_krw'].mean()
    
    print(f"\n{'지표':<20}{'필터링 전':>20}{'필터링 후':>20}{'변화':>20}")
    print(f"{'-'*80}")
    print(f"{'거래 수':<20}{n_trades_orig:>20}{n_trades_filt:>20}{n_trades_filt - n_trades_orig:>20}")
    print(f"{'승률 (%)':<20}{win_rate_orig:>20.2f}{win_rate_filt:>20.2f}{win_rate_filt - win_rate_orig:>20.2f}")
    print(f"{'총 PnL (원)':<20}{total_pnl_orig:>20,.0f}{total_pnl_filt:>20,.0f}{total_pnl_filt - total_pnl_orig:>20,.0f}")
    print(f"{'평균 PnL (원)':<20}{avg_pnl_orig:>20,.0f}{avg_pnl_filt:>20,.0f}{avg_pnl_filt - avg_pnl_orig:>20,.0f}")
    
    # 연도별 비교
    print(f"\n연도별 성과 비교:")
    print(f"{'연도':<10}{'필터링 전 거래':>15}{'필터링 후 거래':>15}{'필터링 전 승률':>15}{'필터링 후 승률':>15}")
    print(f"{'-'*70}")
    
    for year in sorted(df_original['year'].unique()):
        orig_year = df_original[df_original['year'] == year]
        filt_year = df_filtered[df_filtered['year'] == year]
        
        print(f"{year:<10}{len(orig_year):>15}{len(filt_year):>15}"
              f"{orig_year['is_win'].mean()*100:>15.2f}{filt_year['is_win'].mean()*100:>15.2f}")


def main():
    """메인 함수"""
    print("=" * 100)
    print("거래 필터링 모델 (XGBoost) 학습")
    print("=" * 100)
    
    # 1) 데이터 로드 및 전처리
    df, X, y = load_and_preprocess_data()
    
    # 2) XGBoost 모델 학습
    model = train_xgboost_model(X, y)
    
    # 3) 모델 저장
    model_path = OUTPUT_DIR / "trade_filter_xgboost.json"
    model.save_model(str(model_path))
    print(f"\n모델 저장 완료: {model_path}")
    
    # 4) 다양한 threshold로 필터링 테스트
    thresholds = [0.5, 0.6, 0.7, 0.8]
    
    for threshold in thresholds:
        print(f"\n{'='*100}")
        print(f"Threshold: {threshold}")
        print(f"{'='*100}")
        
        df_filtered = filter_trades_by_model(df, model, X, threshold)
        compare_performance(df, df_filtered)
    
    # 5) 최적 threshold 선택 (승률 70% 이상 목표)
    print(f"\n{'='*100}")
    print("최적 threshold 선택")
    print(f"{'='*100}")
    
    best_threshold = 0.6
    df_best = filter_trades_by_model(df, model, X, best_threshold)
    
    print(f"\n선택된 threshold: {best_threshold}")
    print(f"필터링 후 거래 수: {len(df_best)}건")
    print(f"필터링 후 승률: {df_best['is_win'].mean() * 100:.2f}%")
    print(f"필터링 후 총 PnL: {df_best['net_krw'].sum():,.0f} 원")
    
    # 필터링된 데이터셋 저장
    filtered_path = OUTPUT_DIR / "filtered_trades.csv"
    df_best.to_csv(filtered_path, index=False)
    print(f"\n필터링된 데이터셋 저장 완료: {filtered_path}")


if __name__ == "__main__":
    main()
