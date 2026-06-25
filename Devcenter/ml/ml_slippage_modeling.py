"""
슬리피지 모델링
거래량, 시간대, ATR 기반 실제 시장 슬리피지 모델링
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
DATA_DIR = Path(__file__).parent / "ml_data"
MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    """ML 데이터셋 로드"""
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    return df


def calculate_slippage_features(df: pd.DataFrame) -> pd.DataFrame:
    """슬리피지 관련 피처 계산"""
    df = df.copy()
    
    # 시간대 피처
    df['entry_hour'] = df['entry_time'].dt.hour
    df['entry_minute'] = df['entry_time'].dt.minute
    
    # 시간대 구분 (야간/주간)
    df['is_night_session'] = ((df['entry_hour'] >= 18) | (df['entry_hour'] < 8)).astype(int)
    
    # 거래 시간대 구분 (거래량이 많은 시간대)
    df['is_high_volume_hour'] = ((df['entry_hour'] >= 9) & (df['entry_hour'] < 15)).astype(int)
    
    # ATR 기반 슬리피지 예측
    df['atr_slippage_factor'] = df['entry_atr'] / df['entry_close'] * 100
    
    # 거래량 기반 슬리피지 (데이터가 있는 경우)
    if 'volume' in df.columns:
        df['volume_slippage_factor'] = 1 / (df['volume'] + 1) * 100
    else:
        # 거래량 데이터가 없는 경우 ATR을 대신 사용
        df['volume_slippage_factor'] = df['atr_slippage_factor']
    
    # 시간대 기반 슬리피지
    df['time_slippage_factor'] = np.where(
        df['is_high_volume_hour'] == 1,
        0.01,  # 거래량이 많은 시간대: 슬리피지 낮음
        np.where(
            df['is_night_session'] == 1,
            0.03,  # 야간 세션: 슬리피지 높음
            0.02   # 기타 시간대: 중간
        )
    )
    
    # 종합 슬리피지 예측 (가중 평균)
    df['predicted_slippage_pct'] = (
        df['atr_slippage_factor'] * 0.4 +
        df['volume_slippage_factor'] * 0.3 +
        df['time_slippage_factor'] * 0.3
    )
    
    # 실제 슬리피지 계산 (진입/이탈 가격 차이)
    df['actual_slippage_ticks'] = np.abs(df['entry_close'] - df['exit_close']) / df['entry_close'] * 10000  # basis points
    
    return df


def analyze_slippage_by_features(df: pd.DataFrame) -> Dict:
    """피처별 슬리피지 분석"""
    results = {}
    
    # 시간대별 슬리피지
    hour_slippage = df.groupby('entry_hour')['predicted_slippage_pct'].agg(['mean', 'std', 'count'])
    results['hour_slippage'] = hour_slippage.to_dict()
    
    # 세션별 슬리피지
    session_slippage = df.groupby('is_night_session')['predicted_slippage_pct'].agg(['mean', 'std', 'count'])
    results['session_slippage'] = session_slippage.to_dict()
    
    # 거래량 시간대별 슬리피지
    volume_hour_slippage = df.groupby('is_high_volume_hour')['predicted_slippage_pct'].agg(['mean', 'std', 'count'])
    results['volume_hour_slippage'] = volume_hour_slippage.to_dict()
    
    # ATR 구간별 슬리피지
    df['atr_quartile'] = pd.qcut(df['entry_atr'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    atr_slippage = df.groupby('atr_quartile')['predicted_slippage_pct'].agg(['mean', 'std', 'count'])
    results['atr_slippage'] = atr_slippage.to_dict()
    
    return results


def build_slippage_model(df: pd.DataFrame) -> Dict:
    """슬리피지 예측 모델 구축"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    
    # 피처 선택
    feature_cols = [
        'entry_hour', 'entry_minute', 'is_night_session', 'is_high_volume_hour',
        'atr_slippage_factor', 'volume_slippage_factor', 'time_slippage_factor',
        'entry_atr', 'entry_close', 'entry_rsi', 'entry_macd'
    ]
    
    # 결측치 처리
    X = df[feature_cols].copy().fillna(0).astype(float)
    y = df['predicted_slippage_pct'].copy()
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    # 모델 학습
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 예측
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 성능 평가
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # 피처 중요도
    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    
    result = {
        'model': model,
        'train_mse': train_mse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'feature_importance': feature_importance
    }
    
    return result


def apply_slippage_to_backtest(df: pd.DataFrame, slippage_model) -> pd.DataFrame:
    """백테스트에 슬리피지 적용"""
    feature_cols = [
        'entry_hour', 'entry_minute', 'is_night_session', 'is_high_volume_hour',
        'atr_slippage_factor', 'volume_slippage_factor', 'time_slippage_factor',
        'entry_atr', 'entry_close', 'entry_rsi', 'entry_macd'
    ]
    
    X = df[feature_cols].copy().fillna(0).astype(float)
    df['model_predicted_slippage'] = slippage_model.predict(X)
    
    # 슬리피지를 반영한 PnL 계산
    # 슬리피지 비용: 예측 슬리피지 * 포지션 크기 (size_factor 사용)
    df['slippage_cost'] = df['model_predicted_slippage'] * df['size_factor'] * 0.01  # %를 실제 비용으로 변환
    
    # 슬리피지 반영 PnL
    df['net_krw_with_slippage'] = df['net_krw'] - df['slippage_cost']
    
    return df


def main():
    """메인 함수"""
    print("=" * 80)
    print("슬리피지 모델링")
    print("=" * 80)
    
    # 데이터 로드
    df = load_data()
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 슬리피지 피처 계산
    df = calculate_slippage_features(df)
    print(f"\n슬리피지 피처 계산 완료")
    
    # 슬리피지 분석
    slippage_analysis = analyze_slippage_by_features(df)
    print(f"\n슬리피지 분석 완료")
    
    print(f"\n시간대별 슬리피지:")
    for hour, stats in slippage_analysis['hour_slippage']['mean'].items():
        print(f"  {hour}시: {stats:.4f}%")
    
    print(f"\n세션별 슬리피지:")
    for session, stats in slippage_analysis['session_slippage']['mean'].items():
        session_name = "야간" if session == 1 else "주간"
        print(f"  {session_name}: {stats:.4f}%")
    
    print(f"\n거래량 시간대별 슬리피지:")
    for is_high, stats in slippage_analysis['volume_hour_slippage']['mean'].items():
        volume_name = "고거래량" if is_high == 1 else "저거래량"
        print(f"  {volume_name}: {stats:.4f}%")
    
    # 슬리피지 모델 구축
    slippage_model_result = build_slippage_model(df)
    print(f"\n슬리피지 모델 구축 완료")
    print(f"  훈련 MSE: {slippage_model_result['train_mse']:.6f}")
    print(f"  훈련 MAE: {slippage_model_result['train_mae']:.6f}")
    print(f"  훈련 R2: {slippage_model_result['train_r2']:.4f}")
    print(f"  테스트 MSE: {slippage_model_result['test_mse']:.6f}")
    print(f"  테스트 MAE: {slippage_model_result['test_mae']:.6f}")
    print(f"  테스트 R2: {slippage_model_result['test_r2']:.4f}")
    
    print(f"\n피처 중요도:")
    for feature, importance in sorted(slippage_model_result['feature_importance'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    # 슬리피지 적용
    df_with_slippage = apply_slippage_to_backtest(df, slippage_model_result['model'])
    print(f"\n슬리피지 적용 완료")
    
    # 슬리피지 적용 전후 PnL 비교
    original_total_pnl = df['net_krw'].sum()
    slippage_total_pnl = df_with_slippage['net_krw_with_slippage'].sum()
    slippage_cost = df_with_slippage['slippage_cost'].sum()
    
    print(f"\n슬리피지 적용 전후 PnL 비교:")
    print(f"  원본 총 PnL: {original_total_pnl:,.0f}원")
    print(f"  슬리피지 비용: {slippage_cost:,.0f}원")
    print(f"  슬리피지 적용 후 총 PnL: {slippage_total_pnl:,.0f}원")
    print(f"  PnL 감소율: {(slippage_cost / original_total_pnl * 100):.2f}%")
    
    # 모델 저장
    import joblib
    model_path = MODELS_DIR / "slippage_model.pkl"
    joblib.dump(slippage_model_result['model'], model_path)
    print(f"\n슬리피지 모델 저장: {model_path}")
    
    # 슬리피지 적용 데이터 저장
    df_with_slippage.to_csv(DATA_DIR / "ml_dataset_with_slippage.csv", index=False)
    print(f"슬리피지 적용 데이터 저장: {DATA_DIR / 'ml_dataset_with_slippage.csv'}")
    
    return slippage_model_result, df_with_slippage


if __name__ == "__main__":
    slippage_model_result, df_with_slippage = main()
