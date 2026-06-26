"""
피봇 검출 테스트
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg

DB_PATH = "Devcenter/data/duckdb/market_data.duckdb"

# 백테스트 설정
BT_HALF_KELLY_INTRADAY = pv.BacktestConfig(
    multiplier=31_500,
    commission_pct_per_side=0.00003,
    slippage_ticks_per_side=1,
    session_boundary_hour=8,
    intraday_only=True,
    entry_on="next_open"
)

# 피봇 파라미터
PCFG_BULL = pv.HybridAdaptivePivotConfig(
    base_pct=0.3,
    base_multiplier=1.5,
    atr_weight=0.2,
    confirmation_bars=1
)

# 필터링 파라미터
FCFG_BULL = pv.FilterConfig(
    enabled=True,
    min_wave_pct=0.15,
    min_pivot_interval_bars=5,
    st_distance_threshold=0.05,
    adx_hold_threshold=10.0
)

def load_data():
    """데이터 로드"""
    import duckdb
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(
        "SELECT * FROM futures_1min "
        "WHERE timestamp >= '20250625' "
        "AND timestamp <= '20260619' "
        "ORDER BY timestamp"
    ).df()
    con.close()
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y%m%d %H%M')
    df = df.set_index("timestamp")
    df.columns = df.columns.str.upper()
    
    # 5분봉으로 집계
    df_5min = df.resample('5min').agg({
        'OPEN': 'first',
        'HIGH': 'max',
        'LOW': 'min',
        'CLOSE': 'last',
        'VOLUME': 'sum'
    }).dropna()
    
    df_5min = pv.filter_day_session(df_5min, start="08:45", end="15:45")
    df_5min = pv.compute_indicators(df_5min)
    
    return df_5min

def main():
    print("데이터 로드 중...")
    df = load_data()
    print(f"로드된 데이터: {len(df)} 봉")
    print(f"기간: {df.index.min()} ~ {df.index.max()}")
    
    # 레짐 신호 계산
    regime_signal = rg.daily_regime_signal(df, ma_short=20, ma_long=60)
    regime_per_bar = regime_signal.reindex(df.index, method='ffill')
    
    # 모든 레짐 포함
    df_filtered = df.copy()
    print(f"레짐 필터링 후: {len(df_filtered)} 봉")
    
    # 피봇 검출
    print("피봇 검출 중...")
    pivots = pv.detect_pivots_daily(df_filtered, PCFG_BULL, FCFG_BULL, BT_HALF_KELLY_INTRADAY.session_boundary_hour)
    print(f"검출된 피봇 수: {len(pivots)}")
    
    if len(pivots) > 0:
        print("피봇 샘플:")
        for i, pivot in enumerate(pivots[:5]):
            print(f"  {i+1}. {pivot}")
    
    # 백테스트
    print("백테스트 실행 중...")
    BT_HALF_KELLY_INTRADAY.direction_mode = "both"
    res = pv.backtest(df_filtered, pivots, BT_HALF_KELLY_INTRADAY)
    print(f"백테스트 결과: 거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}")

if __name__ == "__main__":
    main()
