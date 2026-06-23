# -*- coding: utf-8 -*-
"""
long_or_flat_strategy.py
=========================

롱-또는-플랫 당일매매 전략 실사용 스크립트

[전략 개요]
- 매일 시가에 롱 진입, 종가에 청산
- MA 20/60 기반 레짐 감지
- bull/중립 -> 롱, bear -> 플랫(현금)
- 하락장 방어 효과 기대

[사용법]
1. 전제 조건
   - DuckDB에 최근 60일 이상의 1분봉 데이터가 있어야 함
   - 위치: Devcenter/data/duckdb/market_data.duckdb
   - 테이블: futures_1min
   - 데이터 수집: 48. 오늘데이터_1분봉_자동수집.py 사용

2. 패키지 설치
   - pandas, numpy, duckdb, pyarrow

3. 실행 시간
   - 매일 08:45 이전: 스크립트 실행하여 오늘의 신호 확인
   - 08:45: 시가에 롱 진입 (신호가 1인 경우)
   - 15:45: 종가에 청산

4. 실행 방법
   cd c:\Project\SkyPredictor\Devcenter
   python long_or_flat_strategy.py

[신호 해석]
- 신호 1: 롱 진입
  * 08:45 시가에 매수
  * 15:45 종가에 매도
  * 당일 청산 (오버나잇 금지)

- 신호 0: 플랫 스킵
  * 진입하지 않음
  * 현금 유지
  * 내일 신호 다시 확인

[전략 로직]
- MA 20 > MA 60: bull/중립 -> 롱 진입
- MA 20 < MA 60: bear -> 플랫 스킵
- ADX > 25: 트렌드 강도 확인 (기본값)

[리스크 경고]
- 이 전략은 상승장 가정에 기반
- 하락장에서는 플랫으로 손실 회피 시도
- MA 20/60이 하락장을 제때 감지하지 못할 수 있음
- 포지션 사이징으로 리스크 관리 필요
- 최악 구간 MaxDD: -3,553만원 (베타 기준)

[win1 방어 효과]
- win1 (2025-06-25~2025-08-19): 하락장 감지 -> 플랫 -> 0원 손실
- 베타: -343만원 손실
- 롱/플랫: 0원 손실 (완전 방어)

[매일 사용 권장]
장 시작 전(08:45 이전)에 스크립트를 실행하여 오늘의 신호를 확인하고 거래하세요.
"""
from __future__ import annotations

import sys
import pandas as pd
import numpy as np
from datetime import datetime, time
from pathlib import Path
import os
import json

# 프로젝트 경로 추가
sys.path.append(r"c:\Project\SkyPredictor\Devcenter")

import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg


def load_config() -> dict:
    """설정 파일 로드
    
    Returns:
        dict: 설정 정보
    """
    config_path = Path(__file__).parent / 'config' / 'long_or_flat_config.json'
    
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def log_signal(date: str, signal: int, ma20: float, ma60: float, ma_diff: bool, adx: float, config: dict):
    """신호 로그 기록
    
    Args:
        date: 오늘 날짜 (YYYY-MM-DD)
        signal: 신호 (1=롱, 0=플랫)
        ma20: MA20 값
        ma60: MA60 값
        ma_diff: MA20 > MA60 여부
        adx: ADX 값
        config: 설정 정보
    """
    # 로그 설정 확인
    if not config.get('logging', {}).get('enabled', True):
        return
    
    # CSV 로그
    log_dir = Path(__file__).parent / config['logging']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / config['logging']['log_file']
    
    # 로그 데이터
    log_data = {
        'date': date,
        'signal': signal,
        'ma20': ma20,
        'ma60': ma60,
        'ma20_gt_ma60': ma_diff,
        'adx': adx,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # CSV 파일 저장
    if not log_file.exists():
        df = pd.DataFrame([log_data])
        df.to_csv(log_file, index=False, encoding='utf-8-sig')
        print(f"로그 파일 생성: {log_file}")
    else:
        df = pd.DataFrame([log_data])
        df.to_csv(log_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        print(f"로그 추가: {log_file}")
    
    # DB 저장
    save_signal_to_db(log_data, config)


def save_signal_to_db(log_data: dict, config: dict):
    """신호를 DB에 저장
    
    Args:
        log_data: 로그 데이터
        config: 설정 정보
    """
    try:
        import duckdb
        
        db_path = config['data']['db_path']
        table_name = 'long_or_flat_signals'
        
        # DB 연결
        con = duckdb.connect(str(db_path))
        
        # 기존 테이블 삭제 (스키마 변경을 위해)
        con.execute(f'DROP TABLE IF EXISTS {table_name}')
        
        # 테이블 생성
        con.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                date VARCHAR,
                signal INTEGER,
                ma20 DOUBLE,
                ma60 DOUBLE,
                ma20_gt_ma60 BOOLEAN,
                adx DOUBLE,
                timestamp VARCHAR
            )
        ''')
        
        # 데이터 삽입
        con.execute(f'''
            INSERT INTO {table_name} VALUES (
                '{log_data['date']}',
                {log_data['signal']},
                {log_data['ma20']},
                {log_data['ma60']},
                {log_data['ma20_gt_ma60']},
                {log_data['adx']},
                '{log_data['timestamp']}'
            )
        ''')
        
        con.close()
        print(f"DB 저장 완료: {table_name}")
        
    except Exception as e:
        print(f"DB 저장 실패: {e}")


def optimize_ma_parameters(df: pd.DataFrame, bt: pv.BacktestConfig, config: dict):
    """다양한 MA 조합 테스트 (파라미터 최적화)
    
    Args:
        df: 1분봉 데이터
        bt: 백테스트 설정
        config: 설정 정보
    """
    # 테스트할 MA 조합
    ma_configs = [
        (5, 15),
        (10, 30),
        (20, 60),
        (50, 200),
    ]
    
    adx_threshold = config['parameters']['adx_threshold']
    allow_short = config['parameters']['allow_short']
    
    print("=" * 80)
    print("MA 파라미터 최적화 테스트")
    print("=" * 80)
    
    # 일봉 변환
    daily = rg.to_daily(df, bt.session_boundary_hour)
    
    results = []
    for ma_short, ma_long in ma_configs:
        # 레짐 신호 계산
        signal = rg.daily_regime_signal(
            daily,
            regime_method="ma",
            ma_short=ma_short,
            ma_long=ma_long,
            adx_threshold=adx_threshold,
            allow_short=allow_short
        )
        
        # 백테스트
        result = rg._bt_from_daily_signal(daily, signal, bt)
        
        results.append({
            'ma_short': ma_short,
            'ma_long': ma_long,
            'sharpe': result.sharpe_daily,
            'pnl': result.total_pnl_krw,
            'max_dd': result.max_drawdown_krw,
            'n_trades': result.n_trades
        })
    
    # 결과 출력
    print(f"\n{'MA 조합':<12}{'Sharpe':>10}{'PnL(KRW)':>15}{'MaxDD(KRW)':>15}{'거래수':>8}")
    print("-" * 80)
    for r in results:
        print(f"MA{r['ma_short']}/{r['ma_long']:<8}{r['sharpe']:>10.3f}{r['pnl']:>15,.0f}{r['max_dd']:>15,.0f}{r['n_trades']:>8}")
    
    # 최적 조합 선택
    best = max(results, key=lambda x: x['sharpe'])
    print("-" * 80)
    print(f"최적 조합: MA{best['ma_short']}/{best['ma_long']} (Sharpe: {best['sharpe']:.3f})")
    print("=" * 80)
    
    return best



def load_today_data(config: dict) -> pd.DataFrame:
    """당일 1분봉 데이터 로드
    
    Args:
        config: 설정 정보
    
    Returns:
        pd.DataFrame: 1분봉 데이터
    """
    # 설정에서 데이터 경로 및 파라미터 가져오기
    db_path = config['data']['db_path']
    table_name = config['data']['table_name']
    session_start = config['data']['session_start']
    session_end = config['data']['session_end']
    warmup_days = config['data']['warmup_days']
    
    # 오늘 날짜 계산
    today = datetime.now()
    # DB에 240일 데이터가 있으므로 충분히 과거부터 로드
    start_date = (today - pd.Timedelta(days=240)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    
    # 데이터 로드
    df = pv.load_data_by_date(db_path, table_name, start=start_date, end=end_date)
    
    # 주간세션 필터
    df = pv.filter_day_session(df, start=session_start, end=session_end)
    
    # 최근 warmup_days 데이터만 사용
    df = df.tail(warmup_days * 420)  # warmup_days * 420분 (08:45-15:45)
    
    return df


def calculate_regime_signal(df: pd.DataFrame, bt: pv.BacktestConfig, config: dict) -> int:
    """오늘의 레짐 신호 계산 (전일까지 데이터 사용)
    
    Args:
        df: 1분봉 데이터
        bt: 백테스트 설정
        config: 설정 정보
    
    Returns:
        int: 1 (롱 진입), 0 (플랫 스킵)
    """
    # 설정에서 파라미터 가져오기
    ma_short = config['parameters']['ma_short']
    ma_long = config['parameters']['ma_long']
    adx_threshold = config['parameters']['adx_threshold']
    allow_short = config['parameters']['allow_short']
    
    # 일봉 변환
    daily = rg.to_daily(df, bt.session_boundary_hour)
    
    # 레짐 신호 계산 (전체 데이터, 워밍업 보존)
    signal = rg.daily_regime_signal(
        daily,
        regime_method="adx",
        ma_short=ma_short,
        ma_long=ma_long,
        adx_threshold=adx_threshold,
        allow_short=allow_short
    )
    
    # 오늘의 신호 (shift(1)로 이미 전일까지 데이터로 계산된 신호)
    today_signal = signal.iloc[-1]
    
    return int(today_signal)


def print_signal_info(df: pd.DataFrame, bt: pv.BacktestConfig, signal: int, config: dict):
    """신호 정보 출력 (전일까지 데이터 사용)
    
    Args:
        df: 1분봉 데이터
        bt: 백테스트 설정
        signal: 신호
        config: 설정 정보
    """
    daily = rg.to_daily(df, bt.session_boundary_hour)
    
    # 오늘 데이터 제외 (전일까지 데이터로 신호 계산)
    daily_yesterday = daily.iloc[:-1].copy()
    
    # 최근 데이터 (전일까지)
    recent_daily = daily_yesterday.tail(5)
    
    # 오늘 날짜
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 설정에서 파라미터 가져오기
    ma_short = config['parameters']['ma_short']
    ma_long = config['parameters']['ma_long']
    
    print("=" * 80)
    print("롱-또는-플랫 전략 신호")
    print("=" * 80)
    print(f"\n오늘 날짜: {today}")
    print(f"\n최근 일봉 데이터 (전일까지):")
    print(recent_daily[['OPEN', 'HIGH', 'LOW', 'CLOSE']].to_string())
    
    # MA 계산 (전일까지 데이터)
    close = daily_yesterday['CLOSE']
    ma20 = close.rolling(ma_short).mean().iloc[-1]
    ma60 = close.rolling(ma_long).mean().iloc[-1]
    ma_diff = ma20 > ma60
    
    # ADX 계산 (전일까지 데이터)
    adx = rg._daily_adx(daily_yesterday, period=14).iloc[-1]
    
    print(f"\nMA{ma_short} (전일): {ma20:.2f}")
    print(f"MA{ma_long} (전일): {ma60:.2f}")
    print(f"MA{ma_short} > MA{ma_long}: {ma_diff}")
    print(f"ADX (전일): {adx:.2f}")
    
    print(f"\n오늘의 신호: {signal}")
    if signal == 1:
        print("  -> 롱 진입: 시가에 매수, 종가에 매도")
    else:
        print("  -> 플랫 스킵: 진입하지 않음 (현금 유지)")
    print("=" * 80)
    
    # 로그 기록
    log_signal(today, signal, ma20, ma60, ma_diff, adx, config)


def main():
    """메인 함수"""
    # 설정 파일 로드
    print("설정 파일 로드 중...")
    config = load_config()
    print(f"설정 파일 로드 완료: {config['strategy']['name']}")
    
    # 백테스트 설정
    bt_config = config['backtest']
    bt = pv.BacktestConfig(
        multiplier=bt_config['multiplier'],
        commission_pct_per_side=bt_config['commission_pct_per_side'],
        slippage_ticks_per_side=bt_config['slippage_ticks_per_side'],
        tick_size=bt_config['tick_size'],
        entry_on=bt_config['entry_on'],
        annualization=bt_config['annualization'],
        intraday_only=bt_config['intraday_only'],
        session_boundary_hour=bt_config['session_boundary_hour'],
        direction_mode=bt_config['direction_mode'],
    )
    
    # 데이터 로드
    print("\n데이터 로드 중...")
    df = load_today_data(config)
    print(f"데이터 로드 완료: {len(df)} 봉")
    
    # 파라미터 최적화 (설정에 따라)
    if config.get('optimization', {}).get('enabled', False) and \
       config.get('optimization', {}).get('run_on_startup', False):
        print("\n파라미터 최적화 실행...")
        best = optimize_ma_parameters(df, bt, config)
        print(f"\n최적 파라미터 적용: MA{best['ma_short']}/{best['ma_long']}")
        config['parameters']['ma_short'] = best['ma_short']
        config['parameters']['ma_long'] = best['ma_long']
    
    # 신호 계산
    print("\n레짐 신호 계산 중...")
    signal = calculate_regime_signal(df, bt, config)
    
    # 신호 정보 출력
    print_signal_info(df, bt, signal, config)
    
    # 거래 가이드
    print("\n거래 가이드:")
    if signal == 1:
        print("  1. 08:45 시가에 매수 진입")
        print("  2. 15:45 종가에 매도 청산")
        print("  3. 당일 청산 (오버나잇 금지)")
    else:
        print("  1. 오늘은 진입하지 않음")
        print("  2. 현금 유지")
        print("  3. 내일 신호 다시 확인")
    
    # 리스크 경고
    print("\n리스크 경고:")
    print("  - 이 전략은 상승장 가정에 기반")
    print("  - 하락장에서는 플랫으로 손실 회피 시도")
    print("  - MA 20/60이 하락장을 제때 감지하지 못할 수 있음")
    print("  - 포지션 사이징으로 리스크 관리 필요")
    print("  - 최악 구간 MaxDD: -3,553만원 (베타 기준)")


if __name__ == "__main__":
    main()
