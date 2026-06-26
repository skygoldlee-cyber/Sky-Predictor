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
    """ML 데이터셋 로드 (실제 long+short)"""
    df = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['year'] = df['entry_time'].dt.year
    df['month'] = df['entry_time'].dt.month
    return df


def simulate_live_trading(df: pd.DataFrame, train_start: str, train_end: str, test_start: str, test_end: str, position_size: int = 1, entry_threshold: float = 0.6) -> Dict:
    """실매매 시뮬레이션 (Walk-forward validation: 학습 데이터로 학습, 테스트 데이터로 테스트)"""
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    # 학습 데이터와 테스트 데이터 분리
    train_df = df[(df['entry_time'] >= train_start) & (df['entry_time'] <= train_end)].copy()
    test_df = df[(df['entry_time'] >= test_start) & (df['entry_time'] <= test_end)].copy()
    
    if len(train_df) == 0:
        return {'error': 'No training data in specified period'}
    if len(test_df) == 0:
        return {'error': 'No test data in specified period'}
    
    print(f"  학습 데이터: {len(train_df)}건 ({train_start} ~ {train_end})")
    print(f"  테스트 데이터: {len(test_df)}건 ({test_start} ~ {test_end})")
    
    # 피처 선택
    feature_cols = [
        'entry_rsi', 'entry_macd', 'entry_macd_signal', 'entry_macd_hist',
        'entry_atr', 'entry_supertrend', 'entry_supertrend_dir',
        'entry_ma20', 'entry_ma60', 'entry_bb_upper', 'entry_bb_lower', 'entry_bb_middle',
        'entry_hour', 'entry_dayofweek', 'entry_month',
        'volatility_regime', 'trend_regime', 'momentum_regime'
    ]
    
    # 파생 피처 추가 (학습 데이터)
    train_df['rsi_ma_ratio'] = train_df['entry_rsi'] / train_df['entry_ma20'].replace(0, 1)
    train_df['price_ma_ratio'] = train_df['entry_close'] / train_df['entry_ma20'].replace(0, 1)
    train_df['bb_position'] = (train_df['entry_close'] - train_df['entry_bb_lower']) / (train_df['entry_bb_upper'] - train_df['entry_bb_lower']).replace(0, 1)
    train_df['supertrend_alignment'] = ((train_df['entry_close'] > train_df['entry_supertrend']) & (train_df['entry_supertrend_dir'] == 1)).astype(int)
    
    # 파생 피처 추가 (테스트 데이터)
    test_df['rsi_ma_ratio'] = test_df['entry_rsi'] / test_df['entry_ma20'].replace(0, 1)
    test_df['price_ma_ratio'] = test_df['entry_close'] / test_df['entry_ma20'].replace(0, 1)
    test_df['bb_position'] = (test_df['entry_close'] - test_df['entry_bb_lower']) / (test_df['entry_bb_upper'] - test_df['entry_bb_lower']).replace(0, 1)
    test_df['supertrend_alignment'] = ((test_df['entry_close'] > test_df['entry_supertrend']) & (test_df['entry_supertrend_dir'] == 1)).astype(int)
    
    # 무한대 값 처리
    train_df = train_df.replace([np.inf, -np.inf], 0)
    test_df = test_df.replace([np.inf, -np.inf], 0)
    
    feature_cols.extend(['rsi_ma_ratio', 'price_ma_ratio', 'bb_position', 'supertrend_alignment'])
    
    # Random Forest 보수적 파라미터 모델 학습 (학습 데이터만 사용)
    X_train = train_df[feature_cols].copy().fillna(0).astype(float)
    y_train = train_df['is_win'].copy()
    
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
    model.fit(X_train, y_train)
    
    # 테스트 데이터로 예측
    X_test = test_df[feature_cols].copy().fillna(0).astype(float)
    test_df['predicted_prob'] = model.predict_proba(X_test)[:, 1]
    test_df['predicted_signal'] = (test_df['predicted_prob'] >= 0.5).astype(int)
    
    # 진입 필터: 예측 확률 임계값 이상만 진입
    test_df['entry_signal'] = (test_df['predicted_prob'] >= entry_threshold).astype(int)
    
    # 시간 필터: 10-14시만 거래
    test_df['time_filter'] = ((test_df['entry_hour'] >= 10) & (test_df['entry_hour'] <= 14)).astype(int)
    test_df['entry_signal'] = (test_df['entry_signal'] & test_df['time_filter']).astype(int)
    
    # 실매매 시뮬레이션 (테스트 데이터만 사용)
    trades = []
    current_position = None
    initial_capital = 100000000  # 1억원 초기 자본
    current_capital = initial_capital
    
    for idx, row in test_df.iterrows():
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
    """메인 함수 (Walk-forward validation)"""
    print("=" * 80)
    print("Walk-forward Validation 테스트")
    print("=" * 80)
    
    # 데이터 로드
    df = load_data()
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # Walk-forward validation 설정
    # 학습 데이터: 2019-2024년
    # 테스트 데이터: 2025-2026년
    train_start = "2019-01-01"
    train_end = "2024-12-31"
    test_start = "2025-01-01"
    test_end = "2026-12-31"
    
    print(f"\nWalk-forward validation 설정:")
    print(f"  학습 기간: {train_start} ~ {train_end}")
    print(f"  테스트 기간: {test_start} ~ {test_end}")
    
    # 다른 진입 임계값으로 시뮬레이션
    thresholds = [0.52]  # 최적 임계값만 테스트
    position_sizes = [1, 2, 3]  # 포지션 사이즈 테스트
    results = {}
    
    for position_size in position_sizes:
        for threshold in thresholds:
            print(f"\n{'=' * 80}")
            print(f"포지션 사이즈: {position_size}, 진입 임계값: {threshold}")
            print(f"{'=' * 80}")
            
            # Walk-forward validation 시뮬레이션
            live_trading_result = simulate_live_trading(df, train_start, train_end, test_start, test_end, position_size=position_size, entry_threshold=threshold)
            
            if 'error' in live_trading_result:
                print(f"\n실매매 시뮬레이션 오류: {live_trading_result['error']}")
                results[f"pos{position_size}_thr{threshold}"] = live_trading_result
                continue
            
            # 성과 분석
            performance = analyze_live_trading_performance(live_trading_result, f"{test_start} ~ {test_end}")
            
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
            
            results[f"pos{position_size}_thr{threshold}"] = {
                'position_size': position_size,
                'threshold': threshold,
                'performance': performance,
                'trades': live_trading_result
            }
    
    # 결과 비교
    print(f"\n{'='*80}")
    print("포지션 사이즈별 성과 비교")
    print(f"{'='*80}")
    
    for key, result in results.items():
        if 'error' not in result:
            perf = result['performance']
            print(f"\n{key}:")
            print(f"  포지션 사이즈: {result['position_size']}")
            print(f"  임계값: {result['threshold']}")
            print(f"  총 거래 수: {perf['total_trades']}건")
            print(f"  승률: {perf['win_rate']:.2f}%")
            print(f"  총 수익률: {perf['total_return']:.2f}%")
            print(f"  샤프 비율: {perf['sharpe_ratio']:.4f}")
    
    # 결과 저장
    import json
    result_path = MODELS_DIR / "walk_forward_validation_result.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nWalk-forward validation 결과 저장: {result_path}")
    
    return results


if __name__ == "__main__":
    performance = main()
