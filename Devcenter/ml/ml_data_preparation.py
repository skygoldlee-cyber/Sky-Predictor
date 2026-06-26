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

DB_PATH = "Devcenter/data/duckdb/market_data.duckdb"
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

# 피봇 파라미터 (BULL 최적 파라미터 - 거래 수 확보를 위해 완화)
PCFG_BULL = pv.HybridAdaptivePivotConfig(
    base_pct=0.3,  # 0.5→0.3 (거래 수 확보를 위해 완화)
    base_multiplier=1.5,  # 2.0→1.5 (거래 수 확보를 위해 완화)
    atr_weight=0.2,  # 0.3→0.2 (거래 수 확보를 위해 완화)
    confirmation_bars=1  # 2→1 (거래 수 확보를 위해 완화)
)

# 필터링 파라미터 (BULL 최적 파라미터 - 거래 수 확보를 위해 완화)
FCFG_BULL = pv.FilterConfig(
    enabled=True,
    min_wave_pct=0.15,  # 0.3→0.15 (거래 수 확보를 위해 완화)
    min_pivot_interval_bars=5,  # 10→5 (거래 수 확보를 위해 완화)
    st_distance_threshold=0.05,  # 0.1→0.05 (거래 수 확보를 위해 완화)
    adx_hold_threshold=10.0  # 15.0→10.0 (거래 수 확보를 위해 완화)
)

# Short 전용 피봇 파라미터 (더 보수적 - 양질 Short 피봇만 선택)
PCFG_SHORT = pv.HybridAdaptivePivotConfig(
    base_pct=0.4,  # 0.3→0.4 (더 보수적)
    base_multiplier=2.0,  # 1.5→2.0 (더 보수적)
    atr_weight=0.3,  # 0.2→0.3 (더 보수적)
    confirmation_bars=2  # 1→2 (더 보수적)
)

# Short 전용 필터링 파라미터 (더 보수적)
FCFG_SHORT = pv.FilterConfig(
    enabled=True,
    min_wave_pct=0.25,  # 0.15→0.25 (더 보수적)
    min_pivot_interval_bars=10,  # 5→10 (더 보수적)
    st_distance_threshold=0.1,  # 0.05→0.1 (더 보수적)
    adx_hold_threshold=15.0  # 10.0→15.0 (더 보수적)
)


def load_data_by_year(year: int):
    """특정 연도의 1분봉 데이터 로드 후 5분봉으로 집계"""
    import duckdb
    start_date = f"{year}0101"
    end_date = f"{year}1231"
    
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(
        f"SELECT * FROM futures_1min "
        f"WHERE timestamp >= '{start_date}' "
        f"AND timestamp <= '{end_date}' "
        f"ORDER BY timestamp"
    ).df()
    con.close()
    
    if len(df) == 0:
        return df
    
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


def load_data_from_txt(txt_path: str):
    """since2019_future_data.txt 파일에서 데이터 로드 후 5분봉으로 집계"""
    # 공백으로 구분된 데이터 읽기 (헤더 없음)
    df = pd.read_csv(txt_path, sep='\s+', header=None, 
                     names=['index', 'datetime', 'open', 'high', 'low', 'close'])
    
    # 날짜/시간 파싱 (형식: 2019/06/03_0900)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y/%m/%d_%H%M')
    df = df.set_index('datetime')
    df = df.drop('index', axis=1)
    
    # 컬럼명 대문자로 변환
    df.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
    # VOLUME 컬럼 추가 (데이터가 없으므로 0으로 설정)
    df['VOLUME'] = 0
    
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


def run_pivot_bull_neutral_with_details(df: pd.DataFrame, pcfg: pv.HybridAdaptivePivotConfig, 
                                       fcfg: pv.FilterConfig, direction_mode: str = "long_only",
                                       bt_cfg: pv.BacktestConfig = None):
    """피봇 전략 실행 (상세 거래 데이터 포함, 방향별 파라미터 적용)"""
    bt = bt_cfg if bt_cfg is not None else BT_HALF_KELLY_INTRADAY
    bt.direction_mode = direction_mode
    
    df_filtered = df.copy()
    print(f"  사용 데이터: {len(df_filtered)} 봉")
    
    if len(df_filtered) == 0:
        return pv.BacktestResult()
    
    # 방향별 파라미터 적용
    if direction_mode == "both":
        # Long 피봇 검출 (BULL 파라미터)
        long_pivots = pv.detect_pivots_daily(df_filtered, PCFG_BULL, FCFG_BULL, bt.session_boundary_hour)
        print(f"  Long 피봇 수: {len(long_pivots)}")
        
        # Short 피봇 검출 (SHORT 파라미터)
        short_pivots = pv.detect_pivots_daily(df_filtered, PCFG_SHORT, FCFG_SHORT, bt.session_boundary_hour)
        print(f"  Short 피봇 수: {len(short_pivots)}")
        
        # 피봇 합치기 (DataFrame으로 유지)
        if isinstance(long_pivots, pd.DataFrame) and isinstance(short_pivots, pd.DataFrame):
            pivots = pd.concat([long_pivots, short_pivots], ignore_index=True)
        elif isinstance(long_pivots, list) and isinstance(short_pivots, list):
            pivots = long_pivots + short_pivots
        else:
            # 둘 중 하나가 DataFrame이면 다른 것도 DataFrame으로 변환
            if isinstance(long_pivots, pd.DataFrame):
                pivots = pd.concat([long_pivots, pd.DataFrame(short_pivots)], ignore_index=True)
            else:
                pivots = pd.concat([pd.DataFrame(long_pivots), short_pivots], ignore_index=True)
        print(f"  전체 피봇 수: {len(pivots)}")
    else:
        # 단일 방향 (기존 로직)
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


def calculate_regime_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """레짐 감지를 위한 피처 계산"""
    df = df.copy()
    
    # pv.compute_indicators 컬럼 이름 사용 (ATR_14, EMA_20 등)
    atr_col = 'ATR_14' if 'ATR_14' in df.columns else 'ATR'
    
    # MA20 컬럼 찾기 (여러 가능한 이름)
    ma20_col = None
    for col in ['EMA_20', 'MA20', 'MA_20', 'SMA_20']:
        if col in df.columns:
            ma20_col = col
            break
    
    # MA20가 없으면 직접 계산
    if ma20_col is None:
        df['MA20'] = df['CLOSE'].rolling(window=20).mean()
        ma20_col = 'MA20'
    
    # 변동성 기반 레짐
    df['volatility'] = df[atr_col] / df['CLOSE'] * 100
    df['volatility_ma'] = df['volatility'].rolling(window=window, min_periods=1).mean()
    df['volatility_regime'] = np.where(df['volatility'] > df['volatility_ma'] * 1.5, 2, 
                                       np.where(df['volatility'] < df['volatility_ma'] * 0.5, 0, 1))
    # 0: low, 1: normal, 2: high
    
    # 추세 기반 레짐
    df['trend_strength'] = (df['CLOSE'] - df[ma20_col]) / df['CLOSE'] * 100
    df['trend_regime'] = np.where(df['trend_strength'] > 1.0, 2,
                                  np.where(df['trend_strength'] < -1.0, 0, 1))
    # 0: downtrend, 1: sideways, 2: uptrend
    
    # 모멘텀 기반 레짐
    df['momentum'] = df['CLOSE'] / df['CLOSE'].shift(window) - 1
    df['momentum_regime'] = np.where(df['momentum'] > 0.01, 2,
                                     np.where(df['momentum'] < -0.01, 0, 1))
    # 0: negative, 1: neutral, 2: positive
    
    return df


def extract_ml_dataset(years: List[int], use_txt: bool = False, txt_path: str = None):
    """머신러닝 데이터셋 추출 (전체 기간 한 번에 처리)"""
    print("전체 기간 데이터 로드 중...")
    
    if use_txt and txt_path:
        # txt 파일 사용
        print(f"TXT 파일 사용: {txt_path}")
        df = load_data_from_txt(txt_path)
        if len(df) > 0:
            all_dfs = [df]
        else:
            print("TXT 파일 데이터 없음")
            return None
    else:
        # DuckDB 사용
        # 전체 기간 데이터 로드
        all_dfs = []
        for year in years:
            df = load_data_by_year(year)
            if len(df) > 0:
                all_dfs.append(df)
    
    if not all_dfs:
        print("데이터 없음")
        return None
    
    df = pd.concat(all_dfs)
    print(f"로드된 데이터: {len(df)} 봉")
    print(f"기간: {df.index.min()} ~ {df.index.max()}")
    
    # 기술적 지표 계산 (pv.compute_indicators 사용)
    df = pv.compute_indicators(df)
    
    # 레짐 피처 계산
    df = calculate_regime_features(df)
    
    # 백테스트 실행 (long+short 모두 포함)
    res = run_pivot_bull_neutral_with_details(
        df, PCFG_BULL, FCFG_BULL, "both", BT_HALF_KELLY_INTRADAY
    )
    
    print(f"백테스트 결과: 거래={res.n_trades}, PnL={res.total_pnl_krw:,.0f}")
    
    if res.trades is not None and len(res.trades) > 0:
        # 연도 정보 추가
        res.trades['year'] = pd.to_datetime(res.trades['entry_time']).dt.year
        
        # 진입 시점의 기술적 지표 추가
        for idx, trade in res.trades.iterrows():
            entry_time = trade['entry_time']
            if entry_time in df.index:
                entry_data = df.loc[entry_time]
                
                # pv.compute_indicators 컬럼 이름 사용
                res.trades.at[idx, 'entry_rsi'] = entry_data.get('RSI_14', 0)
                res.trades.at[idx, 'entry_macd'] = entry_data.get('MACD', 0)
                res.trades.at[idx, 'entry_macd_signal'] = entry_data.get('MACD_Signal', 0)
                res.trades.at[idx, 'entry_macd_hist'] = entry_data.get('MACD_Hist', 0)
                res.trades.at[idx, 'entry_atr'] = entry_data.get('ATR_14', 0)
                res.trades.at[idx, 'entry_supertrend'] = entry_data.get('Supertrend', 0)
                res.trades.at[idx, 'entry_supertrend_dir'] = entry_data.get('Supertrend_Dir', 0)
                res.trades.at[idx, 'entry_ma5'] = entry_data.get('EMA_5', 0)
                res.trades.at[idx, 'entry_ma10'] = entry_data.get('EMA_10', 0)
                res.trades.at[idx, 'entry_ma20'] = entry_data.get('EMA_20', 0)
                res.trades.at[idx, 'entry_ma60'] = entry_data.get('EMA_60', 0)
                res.trades.at[idx, 'entry_bb_upper'] = entry_data.get('BB_Upper', 0)
                res.trades.at[idx, 'entry_bb_lower'] = entry_data.get('BB_Lower', 0)
                res.trades.at[idx, 'entry_bb_middle'] = entry_data.get('BB_Middle', 0)
                
                # 변동성 적응형 피처
                res.trades.at[idx, 'entry_close'] = entry_data['CLOSE']
                res.trades.at[idx, 'atr_normalized'] = entry_data.get('ATR_14', 0) / entry_data['CLOSE'] if entry_data['CLOSE'] > 0 else 0
                
                # 레짐 피처
                res.trades.at[idx, 'volatility_regime'] = entry_data.get('volatility_regime', 1)
                res.trades.at[idx, 'trend_regime'] = entry_data.get('trend_regime', 1)
                res.trades.at[idx, 'momentum_regime'] = entry_data.get('momentum_regime', 1)
        
        # 레이블링
        res.trades['is_win'] = (res.trades['net_pts'] > 0).astype(int)
        
        # 시간 피쳐 추가
        res.trades['entry_hour'] = pd.to_datetime(res.trades['entry_time']).dt.hour
        res.trades['entry_dayofweek'] = pd.to_datetime(res.trades['entry_time']).dt.dayofweek
        res.trades['entry_month'] = pd.to_datetime(res.trades['entry_time']).dt.month
        
        # 저장
        output_path = OUTPUT_DIR / "ml_dataset.csv"
        res.trades.to_csv(output_path, index=False)
        print(f"\n데이터셋 저장 완료: {output_path}")
        print(f"총 거래 수: {len(res.trades)}")
        print(f"승률: {res.trades['is_win'].mean() * 100:.2f}%")
        
        return res.trades
    else:
        print("거래 데이터가 없습니다.")
        return None


def main():
    """메인 함수"""
    # since2019_future_data.txt 사용 (2019-2026 전체 기간)
    txt_path = "Devcenter/data/since2019_future_data.txt"
    use_txt = True
    years = []  # txt 파일 사용 시 years는 비워둠
    
    print("=" * 100)
    print("머신러닝 데이터 준비 시작")
    print("=" * 100)
    
    ml_dataset = extract_ml_dataset(years, use_txt=use_txt, txt_path=txt_path)
    
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
        
        # 방향별 통계
        print("\n방향별 통계:")
        for direction in sorted(ml_dataset['direction'].unique()):
            dir_data = ml_dataset[ml_dataset['direction'] == direction]
            dir_name = "Long" if direction == 1 else "Short"
            print(f"  {dir_name}: 거래={len(dir_data)}, 승률={dir_data['is_win'].mean()*100:.2f}%, PnL={dir_data['net_krw'].sum():,.0f}")


if __name__ == "__main__":
    main()
