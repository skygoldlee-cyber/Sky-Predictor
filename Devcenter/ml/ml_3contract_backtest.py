"""
3계약 백테스트
시장 영향력, 유동성 제한 반영한 3계약 기준 실제 백테스트
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
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
    df['year'] = df['entry_time'].dt.year
    df['month'] = df['entry_time'].dt.month
    return df


def calculate_market_impact(df: pd.DataFrame, position_size: int = 3) -> pd.DataFrame:
    """시장 영향력 계산 (3계약 기준)"""
    df = df.copy()
    
    # 거래량 기반 시장 영향력
    if 'entry_volume' in df.columns:
        avg_volume = df['entry_volume'].mean()
        df['volume_ratio'] = df['entry_volume'] / avg_volume
        df['market_impact'] = (position_size / df['volume_ratio']) * 0.01  # % 단위
    else:
        # 거래량 데이터가 없는 경우 ATR 기반 추정
        df['market_impact'] = df['entry_atr'] / df['entry_close'] * 100 * 0.1
    
    # 유동성 제한 (거래량이 적은 시간대 추가 슬리피지)
    df['entry_hour'] = df['entry_time'].dt.hour
    df['is_low_liquidity'] = ((df['entry_hour'] >= 16) | (df['entry_hour'] < 9)).astype(int)
    df['liquidity_penalty'] = df['is_low_liquidity'] * 0.02  # 2% 추가 슬리피지
    
    # 종합 시장 영향력
    df['total_market_impact'] = df['market_impact'] + df['liquidity_penalty']
    
    return df


def simulate_3contract_backtest(df: pd.DataFrame, position_size: int = 3) -> Dict:
    """3계약 백테스트 시뮬레이션"""
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    # 시장 영향력 계산
    df = calculate_market_impact(df, position_size)
    
    # 시장 영향력 값 제한 (inf 방지)
    df['total_market_impact'] = df['total_market_impact'].clip(0, 1)  # 최대 1%로 제한
    
    # 피처 선택
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    # 파생 피처 추가
    df['rsi_ma_ratio'] = df['entry_rsi'] / df['entry_ma20'].replace(0, 1)
    df['price_ma_ratio'] = df['entry_close'] / df['entry_ma20'].replace(0, 1)
    df['bb_position'] = (df['entry_close'] - df['entry_bb_lower']) / (df['entry_bb_upper'] - df['entry_bb_lower']).replace(0, 1)
    df['supertrend_alignment'] = ((df['entry_close'] > df['entry_supertrend']) & (df['entry_supertrend_dir'] == 1)).astype(int)
    
    feature_cols.extend(['rsi_ma_ratio', 'price_ma_ratio', 'bb_position', 'supertrend_alignment'])
    
    # 모델 로드 또는 학습
    model_path = MODELS_DIR / "rf_model_2026Q2.pkl"
    if model_path.exists():
        model = joblib.load(model_path)
    else:
        # 모델이 없으면 학습
        X = df[feature_cols].copy().fillna(0).astype(float)
        y = df['is_win'].copy()
        
        model = RandomForestClassifier(
            n_estimators=20,
            max_depth=3,
            min_samples_split=30,
            min_samples_leaf=15,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X, y)
    
    # 예측
    X = df[feature_cols].copy().fillna(0).astype(float)
    df['predicted_prob'] = model.predict_proba(X)[:, 1]
    df['predicted_signal'] = (df['predicted_prob'] >= 0.5).astype(int)
    
    # 진입 필터: 예측 확률 0.6 이상만 진입
    df['entry_signal'] = (df['predicted_prob'] >= 0.6).astype(int)
    
    # 3계약 기준 백테스트
    trades = []
    
    for idx, row in df[df['entry_signal'] == 1].iterrows():
        # 시장 영향력 반영 슬리피지
        slippage_pct = row['total_market_impact']
        
        # 진입 가격 조정 (시장 영향력 반영)
        if row['direction'] == 1:  # Long
            adjusted_entry_price = row['entry_close'] * (1 + slippage_pct / 100)
            adjusted_exit_price = row['exit_close'] * (1 - slippage_pct / 100)
        else:  # Short
            adjusted_entry_price = row['entry_close'] * (1 - slippage_pct / 100)
            adjusted_exit_price = row['exit_close'] * (1 + slippage_pct / 100)
        
        # PnL 계산 (3계약 기준)
        points_diff = (adjusted_exit_price - adjusted_entry_price) if row['direction'] == 1 else (adjusted_entry_price - adjusted_exit_price)
        pnl_3contract = points_diff * position_size * 31500  # KP200 틱 가격 31,500원
        
        trades.append({
            'entry_time': row['entry_time'],
            'exit_time': row['exit_time'],
            'direction': row['direction'],
            'entry_price': row['entry_close'],
            'exit_price': row['exit_close'],
            'adjusted_entry_price': adjusted_entry_price,
            'adjusted_exit_price': adjusted_exit_price,
            'position_size': position_size,
            'market_impact': slippage_pct,
            'liquidity_penalty': row['liquidity_penalty'],
            'original_pnl': row['net_krw'],
            'adjusted_pnl': pnl_3contract,
            'is_win': pnl_3contract > 0
        })
    
    if len(trades) == 0:
        return {'error': 'No trades executed'}
    
    trades_df = pd.DataFrame(trades)
    
    # 결과 계산
    total_trades = len(trades_df)
    winning_trades = trades_df['is_win'].sum()
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    total_pnl_original = trades_df['original_pnl'].sum()
    total_pnl_adjusted = trades_df['adjusted_pnl'].sum()
    avg_pnl = trades_df['adjusted_pnl'].mean()
    max_drawdown = trades_df['adjusted_pnl'].cumsum().min()
    
    # 시장 영향력 통계
    avg_market_impact = trades_df['market_impact'].mean()
    avg_liquidity_penalty = trades_df['liquidity_penalty'].mean()
    
    # 연도별 성과
    trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
    yearly_performance = trades_df.groupby('year').agg({
        'adjusted_pnl': 'sum',
        'is_win': 'mean'
    })
    
    result = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl_original': total_pnl_original,
        'total_pnl_adjusted': total_pnl_adjusted,
        'pnl_impact': total_pnl_adjusted - total_pnl_original,
        'avg_pnl': avg_pnl,
        'max_drawdown': max_drawdown,
        'avg_market_impact': avg_market_impact,
        'avg_liquidity_penalty': avg_liquidity_penalty,
        'yearly_performance': yearly_performance.to_dict(),
        'trades': trades
    }
    
    return result


def analyze_3contract_performance(result: Dict) -> Dict:
    """3계약 백테스트 성과 분석"""
    if 'error' in result:
        return result
    
    analysis = {
        'total_trades': result['total_trades'],
        'win_rate': result['win_rate'],
        'total_pnl_original': result['total_pnl_original'],
        'total_pnl_adjusted': result['total_pnl_adjusted'],
        'pnl_impact': result['pnl_impact'],
        'pnl_impact_pct': (result['pnl_impact'] / result['total_pnl_original'] * 100) if result['total_pnl_original'] != 0 else 0,
        'avg_pnl': result['avg_pnl'],
        'max_drawdown': result['max_drawdown'],
        'avg_market_impact': result['avg_market_impact'],
        'avg_liquidity_penalty': result['avg_liquidity_penalty'],
        'yearly_performance': result['yearly_performance'],
        'is_successful': result['win_rate'] >= 50 and result['total_pnl_adjusted'] > 0
    }
    
    return analysis


def main():
    """메인 함수"""
    print("=" * 80)
    print("3계약 백테스트")
    print("=" * 80)
    
    # 데이터 로드
    df = load_data()
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 3계약 백테스트 시뮬레이션
    backtest_result = simulate_3contract_backtest(df, position_size=3)
    
    if 'error' in backtest_result:
        print(f"\n3계약 백테스트 오류: {backtest_result['error']}")
        return backtest_result
    
    # 성과 분석
    performance = analyze_3contract_performance(backtest_result)
    
    print(f"\n3계약 백테스트 성과:")
    print(f"  총 거래 수: {performance['total_trades']}건")
    print(f"  승리 거래: {backtest_result['winning_trades']}건")
    print(f"  패배 거래: {backtest_result['losing_trades']}건")
    print(f"  승률: {performance['win_rate']:.2f}%")
    print(f"  원본 총 PnL: {performance['total_pnl_original']:,.0f}원")
    print(f"  조정 총 PnL: {performance['total_pnl_adjusted']:,.0f}원")
    print(f"  PnL 영향: {performance['pnl_impact']:,.0f}원 ({performance['pnl_impact_pct']:.2f}%)")
    print(f"  평균 PnL: {performance['avg_pnl']:,.0f}원")
    print(f"  최대 손실: {performance['max_drawdown']:,.0f}원")
    print(f"  평균 시장 영향력: {performance['avg_market_impact']:.4f}%")
    print(f"  평균 유동성 페널티: {performance['avg_liquidity_penalty']:.4f}%")
    
    print(f"\n연도별 성과:")
    for year, pnl in performance['yearly_performance']['adjusted_pnl'].items():
        win_rate = performance['yearly_performance']['is_win'][year] * 100
        print(f"  {year}년: PnL {pnl:,.0f}원, 승률 {win_rate:.2f}%")
    
    # 성공 여부 판단
    if performance['is_successful']:
        print(f"\n3계약 백테스트 성공: 승률 {performance['win_rate']:.2f}%, 조정 총 PnL {performance['total_pnl_adjusted']:,.0f}원")
    else:
        print(f"\n3계약 백테스트 실패: 승률 {performance['win_rate']:.2f}%, 조정 총 PnL {performance['total_pnl_adjusted']:,.0f}원")
    
    # 결과 저장
    import json
    result_path = MODELS_DIR / "3contract_backtest_result.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(performance, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n3계약 백테스트 결과 저장: {result_path}")
    
    return performance


if __name__ == "__main__":
    performance = main()
