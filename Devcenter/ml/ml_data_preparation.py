# -*- coding: utf-8 -*-
"""
머신러닝 데이터 준비

BULL + NEUTRAL 레짐 완화 + Half Kelly + 인트라데이 모델의 거래 데이터를 추출하고
피쳐 엔지니어링을 수행하여 머신러닝 모델 학습용 데이터셋을 생성한다.
"""
import sys
import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg
import pivot_regime_optimizer as pro

DB_PATH = "c:/Project/SkyPredictor/Devcenter/duckdb/market_data.duckdb"
OUTPUT_DIR = Path(__file__).parent / "ml_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# 인트라데이 백테스트 설정 (Half Kelly + 인트라데이)
BT_HALF_KELLY_INTRADAY = pv.BacktestConfig(
    multiplier=31_500,  # Half Kelly 사이징
    commission_pct_per_side=0.00003,  # 0.003% per side
    slippage_ticks_per_side=1,
    session_boundary_hour=8,  # 야간세션 경계 시간
    intraday_only=True,  # 인트라데이 모드
    entry_on="next_open"
)

# 피봇 파라미터 (BULL 최적 파라미터)
PCFG_BULL = pv.HybridAdaptivePivotConfig(
    base_pct=0.5,
    base_multiplier=2.0,
    atr_weight=0.3,
    confirmation_bars=2
)

# 필터링 파라미터 (BULL 최적 파라미터)
FCFG_BULL = pv.FilterConfig(
    enabled=True,
    min_wave_pct=0.3,  # BULL 최적 파라미터
    min_pivot_interval_bars=10,  # BULL 최적 파라미터
    st_distance_threshold=0.1,  # BULL 최적 파라미터
    adx_hold_threshold=15.0  # BULL 최적 파라미터
)


def load_data_by_year(year: int):
    """특정 연도의 5분봉 데이터 로드"""
    import duckdb
    start_date = f"{year}-01-01 00:00:00"
    end_date = f"{year}-12-31 23:59:59"
    
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(
        f"SELECT * FROM futures_5min "
        f"WHERE timestamp >= '{start_date}' "
        f"AND timestamp <= '{end_date}' "
        f"ORDER BY timestamp"
    ).df()
    con.close()
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df.columns = df.columns.str.upper()
    
    df = pv.filter_day_session(df, start="08:45", end="15:45")
    df = pv.compute_indicators(df)
    
    return df


def run_pivot_bull_neutral_with_details(df: pd.DataFrame, pcfg: pv.HybridAdaptivePivotConfig, 
                                       fcfg: pv.FilterConfig, direction_mode: str = "long_only",
                                       bt_cfg: pv.BacktestConfig = None):
    """BULL + NEUTRAL 레짐 완화 피봇 전략 실행 (상세 거래 데이터 포함)"""
    bt = bt_cfg if bt_cfg is not None else BT_HALF_KELLY_INTRADAY
    bt.direction_mode = direction_mode
    
    # 레짐 신호 계산
    regime_signal = rg.daily_regime_signal(df, ma_short=20, ma_long=60)
    regime_per_bar = regime_signal.reindex(df.index, method='ffill')
    
    # BULL(1)과 NEUTRAL(0) 레짐만 필터링
    df_filtered = df[regime_per_bar.isin([0, 1])].copy()
    print(f"  레짐 필터링 전: {len(df)} 봉, 필터링 후: {len(df_filtered)} 봉")
    
    if len(df_filtered) == 0:
        return pv.BacktestResult()
    
    # 피봇 검출 및 백테스트 (일별 리셋)
    pivots = pv.detect_pivots_daily(df_filtered, pcfg, fcfg, bt.session_boundary_hour)
    print(f"  검출된 피봇 수: {len(pivots)}")
    
    if len(pivots) == 0:
        return pv.BacktestResult()
    
    # 백테스트 실행
    res = pv.backtest(df_filtered, pivots, bt)
    
    # 상세 거래 데이터 추가
    if res.trades is not None and len(res.trades) > 0:
        # 진입/청산 시점의 시장 데이터 추가
        for idx, trade in res.trades.iterrows():
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            
            # 진입 시점 데이터
            entry_data = df_filtered.loc[entry_time]
            res.trades.at[idx, 'entry_close'] = entry_data['CLOSE']
            res.trades.at[idx, 'entry_high'] = entry_data['HIGH']
            res.trades.at[idx, 'entry_low'] = entry_data['LOW']
            res.trades.at[idx, 'entry_volume'] = entry_data['VOLUME']
            
            # 청산 시점 데이터
            exit_data = df_filtered.loc[exit_time]
            res.trades.at[idx, 'exit_close'] = exit_data['CLOSE']
            res.trades.at[idx, 'exit_high'] = exit_data['HIGH']
            res.trades.at[idx, 'exit_low'] = exit_data['LOW']
            res.trades.at[idx, 'exit_volume'] = exit_data['VOLUME']
            
            # 레짐 정보
            res.trades.at[idx, 'regime'] = regime_per_bar.loc[entry_time]
    
    return res


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 계산"""
    df = df.copy()
    
    # RSI (14)
    delta = df['CLOSE'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    ema12 = df['CLOSE'].ewm(span=12, adjust=False).mean()
    ema26 = df['CLOSE'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']
    
    # ATR (14)
    high = df['HIGH']
    low = df['LOW']
    close = df['CLOSE']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # ADX (14)
    # ADX 계산 복잡성으로 인해 간소화된 버전 사용
    df['ADX'] = 0  # TODO: 완전한 ADX 구현 필요
    
    # SuperTrend (10, 1.5)
    hl2 = (high + low) / 2
    upper = hl2 + 1.5 * df['ATR']
    lower = hl2 - 1.5 * df['ATR']
    
    n = len(df)
    st = np.full(n, np.nan)
    direction = np.full(n, 1)
    
    for i in range(1, n):
        if pd.isna(df['ATR'].iloc[i]):
            continue
        
        if direction[i-1] == 1:
            if close.iloc[i] <= lower.iloc[i]:
                st[i] = lower.iloc[i]
                direction[i] = -1
            else:
                st[i] = max(upper.iloc[i], st[i-1] if not pd.isna(st[i-1]) else upper.iloc[i])
                direction[i] = 1
        else:
            if close.iloc[i] >= upper.iloc[i]:
                st[i] = upper.iloc[i]
                direction[i] = 1
            else:
                st[i] = min(lower.iloc[i], st[i-1] if not pd.isna(st[i-1]) else lower.iloc[i])
                direction[i] = -1
    
    st[0] = lower.iloc[0]
    df['SUPERTREND'] = st
    df['SUPERTREND_DIR'] = direction
    
    # 이동평균선 (다양한 시간 윈도우)
    df['MA5'] = df['CLOSE'].rolling(window=5).mean()
    df['MA10'] = df['CLOSE'].rolling(window=10).mean()
    df['MA20'] = df['CLOSE'].rolling(window=20).mean()
    df['MA60'] = df['CLOSE'].rolling(window=60).mean()
    
    # Bollinger Bands (20, 2)
    df['BB_MIDDLE'] = df['MA20']
    df['BB_STD'] = df['CLOSE'].rolling(window=20).std()
    df['BB_UPPER'] = df['BB_MIDDLE'] + 2 * df['BB_STD']
    df['BB_LOWER'] = df['BB_MIDDLE'] - 2 * df['BB_STD']
    
    return df


def extract_ml_dataset(years: List[int]):
    """머신러닝 데이터셋 추출"""
    all_trades = []
    
    for year in years:
        print(f"[{year}년 데이터 추출 중...]")
        
        # 데이터 로드
        df = load_data_by_year(year)
        if len(df) == 0:
            continue
        
        print(f"  로드된 데이터: {len(df)} 봉")
        
        # 기술적 지표 계산
        df = calculate_technical_indicators(df)
        
        # 백테스트 실행
        res = run_pivot_bull_neutral_with_details(
            df, PCFG_BULL, FCFG_BULL, "long_only", BT_HALF_KELLY_INTRADAY
        )
        
        print(f"  백테스트 결과: 거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}")
        
        if res.trades is not None and len(res.trades) > 0:
            # 연도 정보 추가
            res.trades['year'] = year
            
            # 진입 시점의 기술적 지표 추가 (다양한 시간 윈도우 및 변동성 적응형 피처)
            for idx, trade in res.trades.iterrows():
                entry_time = trade['entry_time']
                entry_data = df.loc[entry_time]
                
                res.trades.at[idx, 'entry_rsi'] = entry_data['RSI']
                res.trades.at[idx, 'entry_macd'] = entry_data['MACD']
                res.trades.at[idx, 'entry_macd_signal'] = entry_data['MACD_SIGNAL']
                res.trades.at[idx, 'entry_macd_hist'] = entry_data['MACD_HIST']
                res.trades.at[idx, 'entry_atr'] = entry_data['ATR']
                res.trades.at[idx, 'entry_supertrend'] = entry_data['SUPERTREND']
                res.trades.at[idx, 'entry_supertrend_dir'] = entry_data['SUPERTREND_DIR']
                res.trades.at[idx, 'entry_ma5'] = entry_data['MA5']
                res.trades.at[idx, 'entry_ma10'] = entry_data['MA10']
                res.trades.at[idx, 'entry_ma20'] = entry_data['MA20']
                res.trades.at[idx, 'entry_ma60'] = entry_data['MA60']
                res.trades.at[idx, 'entry_bb_upper'] = entry_data['BB_UPPER']
                res.trades.at[idx, 'entry_bb_lower'] = entry_data['BB_LOWER']
                res.trades.at[idx, 'entry_bb_middle'] = entry_data['BB_MIDDLE']
                
                # 변동성 적응형 피처
                res.trades.at[idx, 'entry_close'] = entry_data['CLOSE']
                res.trades.at[idx, 'atr_normalized'] = entry_data['ATR'] / entry_data['CLOSE'] if entry_data['CLOSE'] > 0 else 0
            
            all_trades.append(res.trades)
            print(f"  거래 수: {len(res.trades)}")
        else:
            print(f"  거래 없음")
    
    # 모든 거래 데이터 합치기
    if all_trades:
        ml_dataset = pd.concat(all_trades, ignore_index=True)
        
        # 레이블링
        ml_dataset['is_win'] = (ml_dataset['net_pts'] > 0).astype(int)
        
        # 시간 피쳐 추가
        ml_dataset['entry_hour'] = pd.to_datetime(ml_dataset['entry_time']).dt.hour
        ml_dataset['entry_dayofweek'] = pd.to_datetime(ml_dataset['entry_time']).dt.dayofweek
        ml_dataset['entry_month'] = pd.to_datetime(ml_dataset['entry_time']).dt.month
        
        # 저장
        output_path = OUTPUT_DIR / "ml_dataset.csv"
        ml_dataset.to_csv(output_path, index=False)
        print(f"\n데이터셋 저장 완료: {output_path}")
        print(f"총 거래 수: {len(ml_dataset)}")
        print(f"승률: {ml_dataset['is_win'].mean() * 100:.2f}%")
        
        return ml_dataset
    else:
        print("거래 데이터가 없습니다.")
        return None


def main():
    """메인 함수"""
    years = list(range(2019, 2027))  # 2019-2026
    
    print("=" * 100)
    print("머신러닝 데이터 준비 시작")
    print("=" * 100)
    
    ml_dataset = extract_ml_dataset(years)
    
    if ml_dataset is not None:
        print("\n" + "=" * 100)
        print("데이터셋 통계")
        print("=" * 100)
        print(f"총 거래 수: {len(ml_dataset)}")
        print(f"승률: {ml_dataset['is_win'].mean() * 100:.2f}%")
        print(f"평균 PnL: {ml_dataset['net_krw'].mean():,.0f} 원")
        print(f"총 PnL: {ml_dataset['net_krw'].sum():,.0f} 원")
        
        # 연도별 통계
        print("\n연도별 통계:")
        for year in sorted(ml_dataset['year'].unique()):
            year_data = ml_dataset[ml_dataset['year'] == year]
            print(f"  {year}년: 거래={len(year_data)}, 승률={year_data['is_win'].mean()*100:.2f}%, PnL={year_data['net_krw'].sum():,.0f}")


if __name__ == "__main__":
    main()
