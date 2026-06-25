"""
실매매 수익성 테스트
Random Forest 보수적 파라미터 모델로 실매매 시뮬레이션
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
DATA_DIR = Path(__file__).parent / "ml_data"
MODELS_DIR = Path(__file__).parent / "ml_models"
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


def simulate_live_trading(df: pd.DataFrame, start_date: str, end_date: str, position_size: int = 1) -> Dict:
    """실매매 시뮬레이션"""
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    # 기간 필터링
    df = df[(df['entry_time'] >= start_date) & (df['entry_time'] <= end_date)].copy()
    
    if len(df) == 0:
        return {'error': 'No data in specified period'}
    
    # 피처 선택
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month', 'regime'
    ]
    
    # 파생 피처 추가
    df['rsi_ma_ratio'] = df['entry_rsi'] / df['entry_ma20']
    df['price_ma_ratio'] = df['entry_close'] / df['entry_ma20']
    df['bb_position'] = (df['entry_close'] - df['entry_bb_lower']) / (df['entry_bb_upper'] - df['entry_bb_lower'])
    df['supertrend_alignment'] = ((df['entry_close'] > df['entry_supertrend']) & (df['entry_supertrend_dir'] == 1)).astype(int)
    
    feature_cols.extend(['rsi_ma_ratio', 'price_ma_ratio', 'bb_position', 'supertrend_alignment'])
    
    # Random Forest 보수적 파라미터 모델 학습
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
    df['predicted_prob'] = model.predict_proba(X)[:, 1]
    df['predicted_signal'] = (df['predicted_prob'] >= 0.5).astype(int)
    
    # 진입 필터: 예측 확률 0.6 이상만 진입 (보수적)
    df['entry_signal'] = (df['predicted_prob'] >= 0.6).astype(int)
    
    # 실매매 시뮬레이션
    trades = []
    current_position = None
    initial_capital = 100000000  # 1억원 초기 자본
    current_capital = initial_capital
    
    for idx, row in df.iterrows():
        if row['entry_signal'] == 1 and current_position is None:
            # 진입
            current_position = {
                'entry_time': row['entry_time'],
                'entry_price': row['entry_close'],
                'direction': row['direction'],
                'size': position_size,
                'predicted_prob': row['predicted_prob'],
                'capital_before': current_capital
            }
        elif current_position is not None:
            # 이탈 조건 (실제 이탈 시간 사용)
            pnl = row['net_krw'] * position_size
            current_capital += pnl
            
            trades.append({
                'entry_time': current_position['entry_time'],
                'exit_time': row['entry_time'],
                'entry_price': current_position['entry_price'],
                'exit_price': row['exit_close'],
                'direction': current_position['direction'],
                'size': current_position['size'],
                'predicted_prob': current_position['predicted_prob'],
                'capital_before': current_position['capital_before'],
                'pnl': pnl,
                'capital_after': current_capital,
                'is_win': pnl > 0
            })
            current_position = None
    
    # 결과 계산
    if len(trades) == 0:
        return {'error': 'No trades executed'}
    
    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades_df)
    winning_trades = trades_df['is_win'].sum()
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    total_pnl = trades_df['pnl'].sum()
    avg_pnl = trades_df['pnl'].mean()
    max_drawdown = trades_df['capital_after'].min() - initial_capital
    max_profit = trades_df['capital_after'].max() - initial_capital
    
    final_capital = trades_df['capital_after'].iloc[-1]
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    
    # 월별 성과
    trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.month
    monthly_performance = trades_df.groupby('month').agg({
        'pnl': 'sum',
        'is_win': 'mean'
    })
    
    # 샤프 비율 (연율화)
    if len(trades_df) > 1:
        returns = trades_df['pnl'] / initial_capital
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    result = {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return': total_return,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'max_drawdown': max_drawdown,
        'max_profit': max_profit,
        'sharpe_ratio': sharpe_ratio,
        'monthly_performance': monthly_performance.to_dict(),
        'trades': trades
    }
    
    return result


def analyze_live_trading_performance(result: Dict, period: str) -> Dict:
    """실매매 성과 분석"""
    if 'error' in result:
        return result
    
    analysis = {
        'period': period,
        'initial_capital': result['initial_capital'],
        'final_capital': result['final_capital'],
        'total_return': result['total_return'],
        'total_trades': result['total_trades'],
        'win_rate': result['win_rate'],
        'total_pnl': result['total_pnl'],
        'avg_pnl': result['avg_pnl'],
        'max_drawdown': result['max_drawdown'],
        'max_profit': result['max_profit'],
        'sharpe_ratio': result['sharpe_ratio'],
        'monthly_performance': result['monthly_performance'],
        'is_profitable': result['total_pnl'] > 0,
        'is_successful': result['win_rate'] >= 50 and result['total_pnl'] > 0 and result['sharpe_ratio'] > 0
    }
    
    return analysis


def main():
    """메인 함수"""
    print("=" * 80)
    print("실매매 수익성 테스트 (Random Forest 보수적 파라미터)")
    print("=" * 80)
    
    # 데이터 로드
    df = load_data()
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 1년간 실매매 시뮬레이션 (최근 1년)
    end_date = df['entry_time'].max()
    start_date = end_date - timedelta(days=365)
    
    print(f"\n실매매 시뮬레이션 기간: {start_date} ~ {end_date}")
    
    # 실매매 시뮬레이션
    live_trading_result = simulate_live_trading(df, start_date, end_date, position_size=1)
    
    if 'error' in live_trading_result:
        print(f"\n실매매 시뮬레이션 오류: {live_trading_result['error']}")
        return live_trading_result
    
    # 성과 분석
    performance = analyze_live_trading_performance(live_trading_result, f"{start_date} ~ {end_date}")
    
    print(f"\n실매매 성과:")
    print(f"  초기 자본: {performance['initial_capital']:,.0f}원")
    print(f"  최종 자본: {performance['final_capital']:,.0f}원")
    print(f"  총 수익률: {performance['total_return']:.2f}%")
    print(f"  총 거래 수: {performance['total_trades']}건")
    print(f"  승리 거래: {live_trading_result['winning_trades']}건")
    print(f"  패배 거래: {live_trading_result['losing_trades']}건")
    print(f"  승률: {performance['win_rate']:.2f}%")
    print(f"  총 PnL: {performance['total_pnl']:,.0f}원")
    print(f"  평균 PnL: {performance['avg_pnl']:,.0f}원")
    print(f"  최대 손실: {performance['max_drawdown']:,.0f}원")
    print(f"  최대 이익: {performance['max_profit']:,.0f}원")
    print(f"  샤프 비율: {performance['sharpe_ratio']:.4f}")
    
    print(f"\n월별 성과:")
    for month, pnl in performance['monthly_performance']['pnl'].items():
        win_rate = performance['monthly_performance']['is_win'][month] * 100
        print(f"  {month}월: PnL {pnl:,.0f}원, 승률 {win_rate:.2f}%")
    
    # 성공 여부 판단
    if performance['is_successful']:
        print(f"\n실매매 성공: 승률 {performance['win_rate']:.2f}%, 총 수익률 {performance['total_return']:.2f}%, 샤프 비율 {performance['sharpe_ratio']:.4f}")
    else:
        print(f"\n실매매 실패: 승률 {performance['win_rate']:.2f}%, 총 수익률 {performance['total_return']:.2f}%, 샤프 비율 {performance['sharpe_ratio']:.4f}")
    
    # 결과 저장
    import json
    result_path = MODELS_DIR / "live_trading_simulation_result.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(performance, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n실매매 시뮬레이션 결과 저장: {result_path}")
    
    return performance


if __name__ == "__main__":
    performance = main()
