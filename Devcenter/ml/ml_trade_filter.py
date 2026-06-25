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

DATA_DIR = Path(__file__).parent / "ml_data"
OUTPUT_DIR = Path(__file__).parent / "ml_models"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """데이터 로드 및 전처리"""
    # 데이터셋 로드
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    
    # 타임스탬프 변환
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    timestamps = df['entry_time']
    
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
    print(f"기간: {timestamps.min()} ~ {timestamps.max()}")
    
    return df, X, y, timestamps


def train_xgboost_model(X: pd.DataFrame, y: pd.Series, timestamps: pd.Series) -> xgb.XGBClassifier:
    """XGBoost 모델 학습 (시간 기반 분할로 데이터 누설 방지)"""
    # 시간 기반 train/validation/test 분할 (데이터 누설 방지)
    # 2019-2023: 훈련, 2024: 검증, 2025-2026: 테스트
    train_mask = (timestamps.dt.year >= 2019) & (timestamps.dt.year <= 2023)
    val_mask = (timestamps.dt.year == 2024)
    test_mask = (timestamps.dt.year >= 2025)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\n시간 기반 분할 (데이터 누설 방지):")
    print(f"  훈련 데이터 (2019-2023): {len(X_train)}건 (승률: {y_train.mean() * 100:.2f}%)")
    print(f"  검증 데이터 (2024): {len(X_val)}건 (승률: {y_val.mean() * 100:.2f}%)")
    print(f"  테스트 데이터 (2025-2026): {len(X_test)}건 (승률: {y_test.mean() * 100:.2f}%)")
    
    # XGBoost 모델 학습 (검증 데이터로 조기 종료, 정규화 강화)
    model = xgb.XGBClassifier(
        n_estimators=50,  # 복잡도 유지
        max_depth=4,  # 복잡도 유지
        learning_rate=0.05,  # 학습률 유지
        subsample=0.7,  # 샘플링 비율 유지
        colsample_bytree=0.7,  # 피처 샘플링 비율 유지
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        reg_alpha=0.5,  # L1 정규화 유지
        reg_lambda=2.0  # L2 정규화 유지
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 테스트 데이터 예측
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 성능 평가
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n테스트 데이터 성능 (샘플 외):")
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


def filter_trades_by_model_dynamic(df: pd.DataFrame, model: xgb.XGBClassifier, 
                                    X: pd.DataFrame, year_thresholds: dict) -> pd.DataFrame:
    """연도별 동적 threshold를 사용하여 거래 필터링"""
    # 승률 예측
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # 필터링
    df_filtered = df.copy()
    df_filtered['win_probability'] = y_pred_proba
    df_filtered['year'] = pd.to_datetime(df_filtered['entry_time']).dt.year
    df_filtered['threshold'] = df_filtered['year'].map(year_thresholds)
    df_filtered = df_filtered[df_filtered['win_probability'] >= df_filtered['threshold']]
    
    print(f"\n연도별 동적 필터링 결과:")
    print(f"  필터링 전: {len(df)}건 (승률: {df['is_win'].mean() * 100:.2f}%)")
    print(f"  필터링 후: {len(df_filtered)}건 (승률: {df_filtered['is_win'].mean() * 100:.2f}%)")
    
    # 연도별 필터링 결과
    print("\n연도별 필터링 결과:")
    for year in sorted(year_thresholds.keys()):
        year_trades = df[df['year'] == year] if 'year' in df.columns else df[pd.to_datetime(df['entry_time']).dt.year == year]
        year_filtered = df_filtered[df_filtered['year'] == year]
        if len(year_trades) > 0:
            print(f"  {year}: {len(year_trades)}건 → {len(year_filtered)}건 (threshold={year_thresholds[year]})")
    
    return df_filtered


def filter_trades_by_model_volatility(df: pd.DataFrame, model: xgb.XGBClassifier, 
                                       X: pd.DataFrame, volatility_thresholds: dict) -> pd.DataFrame:
    """변동성 기반 동적 threshold를 사용하여 거래 필터링"""
    # 승률 예측
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # 변동성 계산 (원본 df와 필터링 df 모두)
    df_temp = df.copy()
    df_temp['volatility'] = df_temp['entry_atr'] / df_temp['entry_close']
    
    df_filtered = df.copy()
    df_filtered['win_probability'] = y_pred_proba
    df_filtered['volatility'] = df_filtered['entry_atr'] / df_filtered['entry_close']
    
    # 변동성 구간별 threshold 적용
    def get_volatility_threshold(vol):
        if vol < 0.0014:
            return volatility_thresholds['low']
        elif vol < 0.0022:
            return volatility_thresholds['medium']
        else:
            return volatility_thresholds['high']
    
    df_filtered['threshold'] = df_filtered['volatility'].apply(get_volatility_threshold)
    df_filtered = df_filtered[df_filtered['win_probability'] >= df_filtered['threshold']]
    
    print(f"\n변동성 기반 동적 필터링 결과:")
    print(f"  필터링 전: {len(df)}건 (승률: {df['is_win'].mean() * 100:.2f}%)")
    print(f"  필터링 후: {len(df_filtered)}건 (승률: {df_filtered['is_win'].mean() * 100:.2f}%)")
    
    # 변동성 구간별 필터링 결과
    print("\n변동성 구간별 필터링 결과:")
    for vol_type, threshold in volatility_thresholds.items():
        if vol_type == 'low':
            vol_mask = df_temp['volatility'] < 0.0014
        elif vol_type == 'medium':
            vol_mask = (df_temp['volatility'] >= 0.0014) & (df_temp['volatility'] < 0.0022)
        else:
            vol_mask = df_temp['volatility'] >= 0.0022
        
        vol_trades = df_temp[vol_mask]
        vol_filtered = df_filtered[df_filtered['volatility'].apply(get_volatility_threshold) == threshold]
        
        if len(vol_trades) > 0:
            print(f"  {vol_type} 변동성 (threshold={threshold}): {len(vol_trades)}건 → {len(vol_filtered)}건")
    
    return df_filtered


def optimize_for_total_pnl(df: pd.DataFrame, model: xgb.XGBClassifier, X: pd.DataFrame) -> tuple:
    """총 PnL 기반 threshold 최적화"""
    best_pnl = 0
    best_threshold = 0.5
    best_df = None
    
    print(f"\n{'='*100}")
    print("총 PnL 기반 threshold 최적화")
    print(f"{'='*100}")
    
    for threshold in np.arange(0.4, 0.9, 0.05):
        df_filtered = filter_trades_by_model(df, model, X, threshold)
        total_pnl = df_filtered['net_krw'].sum()
        win_rate = df_filtered['is_win'].mean() * 100
        
        print(f"Threshold {threshold:.2f}: 총 PnL {total_pnl:,.0f}원, 승률 {win_rate:.2f}%, 거래 수 {len(df_filtered)}건")
        
        if total_pnl > best_pnl:
            best_pnl = total_pnl
            best_threshold = threshold
            best_df = df_filtered
    
    print(f"\n최적 threshold: {best_threshold:.2f}")
    print(f"최적 총 PnL: {best_pnl:,.0f}원")
    print(f"최적 승률: {best_df['is_win'].mean() * 100:.2f}%")
    print(f"최적 거래 수: {len(best_df)}건")
    
    return best_threshold, best_df


def optimize_for_sharpe_ratio(df: pd.DataFrame, model: xgb.XGBClassifier, X: pd.DataFrame) -> tuple:
    """샤프 비율 기반 threshold 최적화 (수익/변동성)"""
    best_sharpe = 0
    best_threshold = 0.5
    best_df = None
    
    print(f"\n{'='*100}")
    print("샤프 비율 기반 threshold 최적화")
    print(f"{'='*100}")
    
    for threshold in np.arange(0.4, 0.9, 0.05):
        df_filtered = filter_trades_by_model(df, model, X, threshold)
        returns = df_filtered['net_krw'].values
        
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            total_pnl = df_filtered['net_krw'].sum()
            win_rate = df_filtered['is_win'].mean() * 100
            
            print(f"Threshold {threshold:.2f}: 샤프 비율 {sharpe_ratio:.4f}, 총 PnL {total_pnl:,.0f}원, 승률 {win_rate:.2f}%, 거래 수 {len(df_filtered)}건")
            
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_threshold = threshold
                best_df = df_filtered
    
    print(f"\n최적 threshold: {best_threshold:.2f}")
    print(f"최적 샤프 비율: {best_sharpe:.4f}")
    print(f"최적 총 PnL: {best_df['net_krw'].sum():,.0f}원")
    print(f"최적 승률: {best_df['is_win'].mean() * 100:.2f}%")
    print(f"최적 거래 수: {len(best_df)}건")
    
    return best_threshold, best_df


def filter_by_time_and_month(df: pd.DataFrame) -> pd.DataFrame:
    """시간대별 및 월별 필터링"""
    df_filtered = df.copy()
    
    # 시간대별 필터링 (9시, 10시, 11시 제외)
    df_filtered = df_filtered[~df_filtered['entry_hour'].isin([9, 10, 11])]
    
    # 월별 필터링 (3월, 5월, 6월 제외)
    df_filtered['entry_month'] = pd.to_datetime(df_filtered['entry_time']).dt.month
    df_filtered = df_filtered[~df_filtered['entry_month'].isin([3, 5, 6])]
    
    print(f"\n시간대별 및 월별 필터링 적용:")
    print(f"  필터링 전: {len(df)}건")
    print(f"  필터링 후: {len(df_filtered)}건")
    print(f"  제외된 거래: {len(df) - len(df_filtered)}건")
    
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
    df, X, y, timestamps = load_and_preprocess_data()
    
    # 2) XGBoost 모델 학습
    model = train_xgboost_model(X, y, timestamps)
    
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
    
    # 5) 총 PnL 기반 threshold 최적화 적용
    best_threshold, df_best = optimize_for_total_pnl(df, model, X)
    
    # 필터링된 데이터셋 저장
    filtered_path = OUTPUT_DIR / "filtered_trades.csv"
    df_best.to_csv(filtered_path, index=False)
    print(f"\n필터링된 데이터셋 저장 완료: {filtered_path}")


if __name__ == "__main__":
    main()
