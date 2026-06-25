"""
페이퍼 트레이딩 시스템
6개월 이상 페이퍼 트레이딩 시스템 구축 및 성과 검증
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


def simulate_paper_trading(df: pd.DataFrame, start_date: str, end_date: str, model_type: str = 'random_forest') -> Dict:
    """페이퍼 트레이딩 시뮬레이션"""
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
    
    # 모델 로드 또는 학습
    if model_type == 'random_forest':
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
    
    # 페이퍼 트레이딩 시뮬레이션
    # 진입 필터: 예측 확률 0.6 이상만 진입
    df['entry_signal'] = (df['predicted_prob'] >= 0.6).astype(int)
    
    # 포지션 관리
    trades = []
    current_position = None
    
    for idx, row in df.iterrows():
        if row['entry_signal'] == 1 and current_position is None:
            # 진입
            current_position = {
                'entry_time': row['entry_time'],
                'entry_price': row['entry_close'],
                'direction': row['direction'],
                'size': row['size_factor'],
                'predicted_prob': row['predicted_prob']
            }
        elif current_position is not None:
            # 이탈 조건 (실제 이탈 시간 사용)
            trades.append({
                'entry_time': current_position['entry_time'],
                'exit_time': row['entry_time'],
                'entry_price': current_position['entry_price'],
                'exit_price': row['exit_close'],
                'direction': current_position['direction'],
                'size': current_position['size'],
                'predicted_prob': current_position['predicted_prob'],
                'pnl': row['net_krw'],
                'is_win': row['is_win']
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
    max_drawdown = trades_df['pnl'].cumsum().min()
    
    # 월별 성과
    trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.month
    monthly_performance = trades_df.groupby('month').agg({
        'pnl': 'sum',
        'is_win': 'mean'
    })
    
    result = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'max_drawdown': max_drawdown,
        'monthly_performance': monthly_performance.to_dict(),
        'trades': trades
    }
    
    return result


def analyze_paper_trading_performance(result: Dict, period: str) -> Dict:
    """페이퍼 트레이딩 성과 분석"""
    if 'error' in result:
        return result
    
    analysis = {
        'period': period,
        'total_trades': result['total_trades'],
        'win_rate': result['win_rate'],
        'total_pnl': result['total_pnl'],
        'avg_pnl': result['avg_pnl'],
        'max_drawdown': result['max_drawdown'],
        'monthly_performance': result['monthly_performance'],
        'is_successful': result['win_rate'] >= 50 and result['total_pnl'] > 0
    }
    
    return analysis


def main():
    """메인 함수"""
    print("=" * 80)
    print("페이퍼 트레이딩 시스템")
    print("=" * 80)
    
    # 데이터 로드
    df = load_data()
    print(f"\n데이터 로드 완료: {len(df)}건")
    print(f"기간: {df['entry_time'].min()} ~ {df['entry_time'].max()}")
    
    # 6개월 페이퍼 트레이딩 시뮬레이션 (최근 6개월)
    end_date = df['entry_time'].max()
    start_date = end_date - timedelta(days=180)
    
    print(f"\n페이퍼 트레이딩 기간: {start_date} ~ {end_date}")
    
    # 페이퍼 트레이딩 시뮬레이션
    paper_trading_result = simulate_paper_trading(df, start_date, end_date)
    
    if 'error' in paper_trading_result:
        print(f"\n페이퍼 트레이딩 오류: {paper_trading_result['error']}")
        return paper_trading_result
    
    # 성과 분석
    performance = analyze_paper_trading_performance(paper_trading_result, f"{start_date} ~ {end_date}")
    
    print(f"\n페이퍼 트레이딩 성과:")
    print(f"  총 거래 수: {performance['total_trades']}건")
    print(f"  승리 거래: {paper_trading_result['winning_trades']}건")
    print(f"  패배 거래: {paper_trading_result['losing_trades']}건")
    print(f"  승률: {performance['win_rate']:.2f}%")
    print(f"  총 PnL: {performance['total_pnl']:,.0f}원")
    print(f"  평균 PnL: {performance['avg_pnl']:,.0f}원")
    print(f"  최대 손실: {performance['max_drawdown']:,.0f}원")
    
    print(f"\n월별 성과:")
    for month, pnl in performance['monthly_performance']['pnl'].items():
        win_rate = performance['monthly_performance']['is_win'][month] * 100
        print(f"  {month}월: PnL {pnl:,.0f}원, 승률 {win_rate:.2f}%")
    
    # 성공 여부 판단
    if performance['is_successful']:
        print(f"\n페이퍼 트레이딩 성공: 승률 {performance['win_rate']:.2f}%, 총 PnL {performance['total_pnl']:,.0f}원")
    else:
        print(f"\n페이퍼 트레이딩 실패: 승률 {performance['win_rate']:.2f}%, 총 PnL {performance['total_pnl']:,.0f}원")
    
    # 결과 저장
    import json
    result_path = MODELS_DIR / "paper_trading_result.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(performance, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n페이퍼 트레이딩 결과 저장: {result_path}")
    
    return performance


if __name__ == "__main__":
    performance = main()
