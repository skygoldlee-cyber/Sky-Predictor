# -*- coding: utf-8 -*-
"""
진입 타이밍 최적화 (Random Forest)

필터링된 고품질 거래 데이터를 사용하여 진입 시점을 더 정교하게 최적화한다.
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
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = Path(__file__).parent / "ml_models"
OUTPUT_DIR = Path(__file__).parent / "ml_models"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_filtered_data() -> Tuple[pd.DataFrame, pd.Series]:
    """필터링된 데이터 로드"""
    df = pd.read_csv(DATA_DIR / "filtered_trades.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    timestamps = df['entry_time']
    
    print(f"필터링된 데이터 로드 완료: {len(df)}건")
    print(f"승률: {df['is_win'].mean() * 100:.2f}%")
    print(f"총 PnL: {df['net_krw'].sum():,.0f} 원")
    print(f"기간: {timestamps.min()} ~ {timestamps.max()}")
    
    return df, timestamps


def engineer_entry_timing_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """진입 타이밍 피쳐 엔지니어링"""
    # 기존 피쳐
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    X = df[feature_cols].copy()
    
    # 추가 피쳐 엔지니어링
    # 1. RSI 과매수/과매도 상태
    X['rsi_oversold'] = (X['entry_rsi'] < 30).astype(int)
    X['rsi_overbought'] = (X['entry_rsi'] > 70).astype(int)
    
    # 2. MACD 신호 강도
    X['macd_bullish'] = (X['entry_macd'] > X['entry_macd_signal']).astype(int)
    X['macd_strength'] = abs(X['entry_macd'] - X['entry_macd_signal'])
    
    # 3. 가격과 이동평균선 관계
    X['price_above_ma20'] = (df['entry_close'] > X['entry_ma20']).astype(int)
    X['price_above_ma60'] = (df['entry_close'] > X['entry_ma60']).astype(int)
    
    # 4. Bollinger Bands 위치
    X['bb_position'] = (df['entry_close'] - X['entry_bb_lower']) / (X['entry_bb_upper'] - X['entry_bb_lower'])
    X['bb_lower_touch'] = (df['entry_close'] <= X['entry_bb_lower'] * 1.01).astype(int)
    
    # 5. SuperTrend 방향과 가격 관계
    X['price_above_st'] = (df['entry_close'] > X['entry_supertrend']).astype(int)
    
    # 6. 시간대 특성
    X['is_morning'] = ((X['entry_hour'] >= 9) & (X['entry_hour'] < 12)).astype(int)
    X['is_afternoon'] = ((X['entry_hour'] >= 12) & (X['entry_hour'] < 15)).astype(int)
    
    # 7. 레짐 특성
    X['is_bull'] = (X['regime'] == 1).astype(int)
    X['is_neutral'] = (X['regime'] == 0).astype(int)
    
    # 결측치 처리
    X = X.fillna(0)
    
    # 타겟 변수
    y = df['is_win'].copy()
    
    print(f"\n피쳐 엔지니어링 완료")
    print(f"총 피쳐 수: {len(X.columns)}")
    
    return df, X, y


def train_random_forest_model(X: pd.DataFrame, y: pd.Series, timestamps: pd.Series) -> RandomForestClassifier:
    """Random Forest 모델 학습 (시간 기반 분할로 데이터 누설 방지)"""
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
    
    # Random Forest 모델 학습 (복잡도 감소 및 정규화 강화)
    model = RandomForestClassifier(
        n_estimators=30,  # 50→30 (트리 수 감소)
        max_depth=4,  # 6→4 (깊이 감소)
        min_samples_split=25,  # 20→25 (분할 최소 샘플 증가)
        min_samples_leaf=12,  # 10→12 (리프 최소 샘플 증가)
        max_features='sqrt',  # 피처 수 제한
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # 클래스 불균형 처리
    )
    
    model.fit(X_train, y_train)
    
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
    
    print(f"\n피쳐 중요도 (상위 15):")
    print(feature_importance.head(15))
    
    return model


def optimize_for_total_pnl_entry(df: pd.DataFrame, model: RandomForestClassifier, 
                                   X: pd.DataFrame) -> tuple:
    """총 PnL 기반 threshold 최적화 (진입 타이밍)"""
    best_pnl = 0
    best_threshold = 0.5
    best_df = None
    
    print(f"\n{'='*100}")
    print("총 PnL 기반 threshold 최적화 (진입 타이밍)")
    print(f"{'='*100}")
    
    for threshold in np.arange(0.5, 0.95, 0.05):
        df_optimized = optimize_entry_timing(df, model, X, threshold)
        total_pnl = df_optimized['net_krw'].sum()
        win_rate = df_optimized['is_win'].mean() * 100
        
        print(f"Threshold {threshold:.2f}: 총 PnL {total_pnl:,.0f}원, 승률 {win_rate:.2f}%, 거래 수 {len(df_optimized)}건")
        
        if total_pnl > best_pnl:
            best_pnl = total_pnl
            best_threshold = threshold
            best_df = df_optimized
    
    print(f"\n최적 threshold: {best_threshold:.2f}")
    print(f"최적 총 PnL: {best_pnl:,.0f}원")
    print(f"최적 승률: {best_df['is_win'].mean() * 100:.2f}%")
    print(f"최적 거래 수: {len(best_df)}건")
    
    return best_threshold, best_df


def optimize_for_sharpe_ratio_entry(df: pd.DataFrame, model: RandomForestClassifier, 
                                     X: pd.DataFrame) -> tuple:
    """샤프 비율 기반 threshold 최적화 (진입 타이밍)"""
    best_sharpe = 0
    best_threshold = 0.5
    best_df = None
    
    print(f"\n{'='*100}")
    print("샤프 비율 기반 threshold 최적화 (진입 타이밍)")
    print(f"{'='*100}")
    
    for threshold in np.arange(0.5, 0.95, 0.05):
        df_optimized = optimize_entry_timing(df, model, X, threshold)
        returns = df_optimized['net_krw'].values
        
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            total_pnl = df_optimized['net_krw'].sum()
            win_rate = df_optimized['is_win'].mean() * 100
            
            print(f"Threshold {threshold:.2f}: 샤프 비율 {sharpe_ratio:.4f}, 총 PnL {total_pnl:,.0f}원, 승률 {win_rate:.2f}%, 거래 수 {len(df_optimized)}건")
            
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_threshold = threshold
                best_df = df_optimized
    
    print(f"\n최적 threshold: {best_threshold:.2f}")
    print(f"최적 샤프 비율: {best_sharpe:.4f}")
    print(f"최적 총 PnL: {best_df['net_krw'].sum():,.0f}원")
    print(f"최적 승률: {best_df['is_win'].mean() * 100:.2f}%")
    print(f"최적 거래 수: {len(best_df)}건")
    
    return best_threshold, best_df


def filter_by_time_and_month(df: pd.DataFrame) -> pd.DataFrame:
    """시간대별 및 월별 필터링 (완화: 시간대/월별 필터링 제거)"""
    df_filtered = df.copy()
    
    # 시간대별 필터링 제거 (거래 수 확보를 위해)
    # df_filtered = df_filtered[~df_filtered['entry_hour'].isin([9, 10, 11])]
    
    # 월별 필터링 제거 (거래 수 확보를 위해)
    # df_filtered['entry_month'] = pd.to_datetime(df_filtered['entry_time']).dt.month
    # df_filtered = df_filtered[~df_filtered['entry_month'].isin([3, 5, 6])]
    
    print(f"\n시간대별 및 월별 필터링 완화 (제거):")
    print(f"  필터링 전: {len(df)}건")
    print(f"  필터링 후: {len(df_filtered)}건")
    print(f"  제외된 거래: {len(df) - len(df_filtered)}건")
    
    return df_filtered


def optimize_entry_timing(df: pd.DataFrame, model: RandomForestClassifier, 
                          X: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """진입 타이밍 최적화"""
    # 승률 예측
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # 최적 진입 시점 필터링
    df_optimized = df.copy()
    df_optimized['entry_quality_score'] = y_pred_proba
    df_optimized = df_optimized[df_optimized['entry_quality_score'] >= threshold]
    
    print(f"\n진입 타이밍 최적화 결과 (threshold={threshold}):")
    print(f"  최적화 전: {len(df)}건 (승률: {df['is_win'].mean() * 100:.2f}%)")
    print(f"  최적화 후: {len(df_optimized)}건 (승률: {df_optimized['is_win'].mean() * 100:.2f}%)")
    
    return df_optimized


def compare_entry_timing(df_filtered: pd.DataFrame, df_optimized: pd.DataFrame):
    """진입 타이밍 최적화 전후 성과 비교"""
    print(f"\n{'='*100}")
    print("진입 타이밍 최적화 전후 성과 비교")
    print(f"{'='*100}")
    
    # 최적화 전
    n_trades_filt = len(df_filtered)
    win_rate_filt = df_filtered['is_win'].mean() * 100
    total_pnl_filt = df_filtered['net_krw'].sum()
    avg_pnl_filt = df_filtered['net_krw'].mean()
    
    # 최적화 후
    n_trades_opt = len(df_optimized)
    win_rate_opt = df_optimized['is_win'].mean() * 100
    total_pnl_opt = df_optimized['net_krw'].sum()
    avg_pnl_opt = df_optimized['net_krw'].mean()
    
    print(f"\n{'지표':<20}{'최적화 전':>20}{'최적화 후':>20}{'변화':>20}")
    print(f"{'-'*80}")
    print(f"{'거래 수':<20}{n_trades_filt:>20}{n_trades_opt:>20}{n_trades_opt - n_trades_filt:>20}")
    print(f"{'승률 (%)':<20}{win_rate_filt:>20.2f}{win_rate_opt:>20.2f}{win_rate_opt - win_rate_filt:>20.2f}")
    print(f"{'총 PnL (원)':<20}{total_pnl_filt:>20,.0f}{total_pnl_opt:>20,.0f}{total_pnl_opt - total_pnl_filt:>20,.0f}")
    print(f"{'평균 PnL (원)':<20}{avg_pnl_filt:>20,.0f}{avg_pnl_opt:>20,.0f}{avg_pnl_opt - avg_pnl_filt:>20,.0f}")
    
    # 연도별 비교
    print(f"\n연도별 성과 비교:")
    print(f"{'연도':<10}{'최적화 전 거래':>15}{'최적화 후 거래':>15}{'최적화 전 승률':>15}{'최적화 후 승률':>15}")
    print(f"{'-'*70}")
    
    for year in sorted(df_filtered['year'].unique()):
        filt_year = df_filtered[df_filtered['year'] == year]
        opt_year = df_optimized[df_optimized['year'] == year]
        
        print(f"{year:<10}{len(filt_year):>15}{len(opt_year):>15}"
              f"{filt_year['is_win'].mean()*100:>15.2f}{opt_year['is_win'].mean()*100:>15.2f}")


def main():
    """메인 함수"""
    print("=" * 100)
    print("진입 타이밍 최적화 (Random Forest)")
    print("=" * 100)
    
    # 1) 필터링된 데이터 로드
    df, timestamps = load_filtered_data()
    
    # 2) 진입 타이밍 피쳐 엔지니어링
    df, X, y = engineer_entry_timing_features(df)
    
    # 3) Random Forest 모델 학습
    model = train_random_forest_model(X, y, timestamps)
    
    # 4) 모델 저장
    model_path = OUTPUT_DIR / "entry_timing_rf.pkl"
    import joblib
    joblib.dump(model, model_path)
    print(f"\n모델 저장 완료: {model_path}")
    
    # 5) 다양한 threshold로 최적화 테스트
    thresholds = [0.6, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        print(f"\n{'='*100}")
        print(f"Threshold: {threshold}")
        print(f"{'='*100}")
        
        df_optimized = optimize_entry_timing(df, model, X, threshold)
        compare_entry_timing(df, df_optimized)
    
    # 6) 총 PnL 기반 threshold 최적화 적용
    best_threshold, df_best = optimize_for_total_pnl_entry(df, model, X)
    
    print(f"\n선택된 threshold: {best_threshold}")
    print(f"최적화 후 거래 수: {len(df_best)}건")
    print(f"최적화 후 승률: {df_best['is_win'].mean() * 100:.2f}%")
    print(f"최적화 후 총 PnL: {df_best['net_krw'].sum():,.0f} 원")
    
    # 최적화된 데이터셋 저장
    optimized_path = OUTPUT_DIR / "optimized_trades.csv"
    df_best.to_csv(optimized_path, index=False)
    print(f"\n최적화된 데이터셋 저장 완료: {optimized_path}")


if __name__ == "__main__":
    main()
