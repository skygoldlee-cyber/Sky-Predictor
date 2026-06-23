# -*- coding: utf-8 -*-
"""
BULL 피봇 롱 실시간 신호 생성기

최신 데이터를 기준으로:
    1. 오늘의 레짐(BULL/BEAR/NEUTRAL) 판단
    2. 오늘 5분봉에서 피봇 감지
    3. 현재 포지션 상태에 따른 신호 생성 (진입/청산/홀드/대기)
    4. 포지션 사이즈 계산 (Half Kelly)

실행 방식:
    python pivot_bull_signal_generator.py

주기적 실행은 OS 스케줄러(cron/Task Scheduler) 또는 루프 모드(--loop) 사용.
"""
import sys
import gc
import json
import math
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import duckdb
import regime_intraday_v2 as rg
import pivot_optuna_v2 as pv

from pivot_bull_data_collector import (
    LSOpenAPICollector, DataStore, LSRealtimeWebSocket, LSOpenAPIOrder,
    load_secrets, get_secret
)

# ────────────────────────────────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────────────────────────────────
DB_PATH = "c:/Project/SkyPredictor v1/Devcenter/duckdb/market_data.duckdb"
TABLE_NAME = "futures_5min"
SESSION_BOUNDARY_HOUR = 8

# BULL 피봇 롱 최적 파라미터 (2019-2025 WFO 결과)
PIVOT_CFG = pv.HybridAdaptivePivotConfig(
    base_pct=1.272989526401749,
    base_multiplier=1.3341908735602903,
    atr_weight=0.20831334967633547,
    confirmation_bars=1,
)
FILTER_CFG = pv.FilterConfig(
    enabled=True,
    min_wave_pct=0.07699392762885474,
    min_pivot_interval_bars=28,
    st_distance_threshold=0.1,
    adx_hold_threshold=15.0,
)

# Kelly 비율 (Half Kelly)
KELLY_FACTOR = 0.126
BASE_MULTIPLIER = 250_000.0
COMMISSION_PCT = 0.00003
SLIPPAGE_TICKS = 1.0
TICK_SIZE = 0.05

# MA 레짐 설정
MA_SHORT = 20
MA_LONG = 60

# ────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ────────────────────────────────────────────────────────────────────────────
def load_recent_data(days: int = 120) -> pd.DataFrame:
    """최근 N일치 5분봉 데이터를 DuckDB에서 로드."""
    start_dt = datetime.now() - timedelta(days=days)
    start_str = start_dt.strftime("%Y-%m-%d")
    
    con = duckdb.connect(database=DB_PATH, read_only=True)
    df = con.execute(
        f"SELECT * FROM {TABLE_NAME} WHERE timestamp >= '{start_str} 00:00:00' "
        f"ORDER BY timestamp"
    ).df()
    con.close()
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df.columns = df.columns.str.upper()
    return df


# ────────────────────────────────────────────────────────────────────────────
# 레짐 계산
# ────────────────────────────────────────────────────────────────────────────
def get_current_regime(df: pd.DataFrame) -> int:
    """현재 레짐 반환: 1=BULL, -1=BEAR, 0=NEUTRAL."""
    daily = rg.to_daily(df, SESSION_BOUNDARY_HOUR)
    close = daily["CLOSE"]
    ma_s = close.rolling(MA_SHORT).mean()
    ma_l = close.rolling(MA_LONG).mean()
    
    if ma_s.iloc[-1] > ma_l.iloc[-1]:
        return 1
    elif ma_s.iloc[-1] < ma_l.iloc[-1]:
        return -1
    return 0


# ────────────────────────────────────────────────────────────────────────────
# 피봇 감지
# ────────────────────────────────────────────────────────────────────────────
def detect_date_pivots(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """특정 날짜 5분봉에서 피봇 감지."""
    pivots = pv.detect_pivots_daily(df, PIVOT_CFG, FILTER_CFG, SESSION_BOUNDARY_HOUR)
    if pivots is None or len(pivots) == 0:
        return pd.DataFrame()
    
    date_pivots = pivots[pivots["confirm_time"].dt.normalize() == target_date.normalize()]
    return date_pivots


# ────────────────────────────────────────────────────────────────────────────
# 신호 생성
# ────────────────────────────────────────────────────────────────────────────
def generate_signal(df: pd.DataFrame, current_position: int = 0, capital: float = 0.0,
                     as_of: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
    """현재 데이터와 포지션 상태, 계좌 크기를 기준으로 신호 생성."""
    now = as_of if as_of is not None else pd.Timestamp.now()
    target_date = now.normalize()
    
    regime = get_current_regime(df)
    
    signal = {
        "timestamp": now.isoformat(),
        "regime": "BULL" if regime == 1 else "BEAR" if regime == -1 else "NEUTRAL",
        "action": "NO_SIGNAL",
        "reason": "",
        "position_size": 0,
        "entry_px": None,
        "stop_px": None,
        "target_px": None,
    }
    
    if regime != 1:
        signal["reason"] = "BULL 레짐 아님"
        if current_position != 0:
            signal["action"] = "FLATTEN"
            signal["reason"] = "BULL 레짐 종료 - 포지션 청산"
        return signal
    
    # BULL 레짐: 해당 날짜 피봇 감지
    date_pivots = detect_date_pivots(df, target_date)
    if len(date_pivots) == 0:
        signal["reason"] = "해당 날짜 피봇 신호 없음"
        return signal
    
    # as_of 시점까지 확정된 피봇만 고려
    if as_of is not None:
        valid_pivots = date_pivots[date_pivots["confirm_time"] <= as_of]
    else:
        valid_pivots = date_pivots
    
    if len(valid_pivots) == 0:
        signal["reason"] = "아직 피봇 신호 없음"
        return signal
    
    # 가장 최근 피봇
    latest_pivot = valid_pivots.iloc[-1]
    is_high = latest_pivot["is_high"]  # True=high 피봇, False=low 피봇
    confirm_px = latest_pivot["pivot_price"]
    confirm_time = latest_pivot["confirm_time"]
    
    # BULL에서 피봇 low 발생 → 롱 진입
    if not is_high:
        if current_position == 0:
            signal["action"] = "ENTER_LONG"
            signal["reason"] = f"BULL 피봇 low 확정 @ {confirm_time}"
            signal["entry_px"] = float(confirm_px)
            # ATR 기반 손절/목표/포지션 사이즈
            atr = pv._atr(df, 14).iloc[-1]
            signal["stop_px"] = float(confirm_px - 2 * atr)
            signal["target_px"] = float(confirm_px + 2 * atr)
            signal["position_size"] = calculate_position_size(capital, atr)
        elif current_position == 1:
            signal["action"] = "HOLD_LONG"
            signal["reason"] = "이미 롱 포지션 보유"
    elif is_high:
        if current_position == 1:
            signal["action"] = "EXIT_LONG"
            signal["reason"] = f"BULL 피봇 high 확정 @ {confirm_time}"
            signal["entry_px"] = float(confirm_px)
        elif current_position == 0:
            signal["action"] = "NO_SIGNAL"
            signal["reason"] = "피봇 high 발생 - 미보유"
    
    return signal


# ────────────────────────────────────────────────────────────────────────────
# 포지션 사이즈 계산
# ────────────────────────────────────────────────────────────────────────────
def calculate_position_size(capital: float, atr: float, multiplier: float = BASE_MULTIPLIER) -> int:
    """
    Half Kelly 기준 ATR 리스크 포지션 사이즈 계산 (계약 수).
    
    공식:
        risk_per_contract = 2 * ATR * multiplier
        target_risk = capital * KELLY_FACTOR
        contracts = target_risk / risk_per_contract
    """
    if capital <= 0 or atr <= 0 or multiplier <= 0:
        return 0
    
    risk_per_contract = 2.0 * atr * multiplier
    target_risk = capital * KELLY_FACTOR
    contracts = int(target_risk / risk_per_contract)
    
    return max(1, contracts)  # 최소 1계약


# ────────────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BULL 피봇 롱 실시간 신호 생성")
    parser.add_argument("--loop", action="store_true", help="5분마다 반복 실행")
    parser.add_argument("--live", action="store_true", help="LS OpenAPI 5분봉 실시간 데이터 수집 모드")
    parser.add_argument("--single-shot", action="store_true", help="실시간 모드에서 1회만 실행")
    parser.add_argument("--websocket", action="store_true", help="LS OpenAPI WebSocket 초 단위 실시간 체결 모드")
    parser.add_argument("--ws-duration", type=int, default=0, help="WebSocket 테스트 실행 시간(초), 0이면 무한")
    parser.add_argument("--position", type=int, default=0, help="현재 포지션 (0=없음, 1=롱)")
    parser.add_argument("--capital", type=float, default=0.0, help="계좌 크기 (원), 0이면 포지션 사이즈 계산 안함")
    parser.add_argument("--date", type=str, help="Backtesting mode: YYYY-MM-DD 형식의 특정 날짜")
    parser.add_argument("--json", action="store_true", help="JSON 형식 출력")
    parser.add_argument("--config", type=str, default="c:/Project/SkyPredictor v1/config.secrets.json",
                        help="LS OpenAPI 인증 정보 JSON 경로")
    parser.add_argument("--symbol", type=str, default="A0169000", help="KOSPI200 선물 단축코드")
    parser.add_argument("--order", action="store_true", help="신호 발생 시 자동 주문 실행")
    parser.add_argument("--live-order", action="store_true", help="실제 주문 제출 (없으면 dry-run)")
    parser.add_argument("--account", type=str, help="선물옵션 계좌번호")
    parser.add_argument("--password", type=str, help="선물옵션 계좌 비밀번호")
    args = parser.parse_args()
    
    def run_once():
        df = load_recent_data(days=120)
        if len(df) == 0:
            print("데이터가 없습니다.")
            return
        
        if args.date:
            # Backtesting mode: 해당 날짜의 5분봉 시점별로 신호 생성
            target_date = pd.Timestamp(args.date)
            day_df = df[df.index.strftime('%Y-%m-%d') == args.date]
            if len(day_df) == 0:
                print(f"{args.date} 데이터가 없습니다.")
                return
            
            print(f"Backtesting mode: {args.date}")
            print(f"Regime: {'BULL' if get_current_regime(df) == 1 else 'BEAR' if get_current_regime(df) == -1 else 'NEUTRAL'}")
            print("-" * 60)
            
            current_position = 0
            last_action = None
            for ts in day_df.index:  # 매 5분봉 종료 시점마다 체크
                as_of = ts + pd.Timedelta(minutes=5)  # 해당 봉 종료 시점
                signal = generate_signal(df, current_position=current_position, capital=args.capital, as_of=as_of)
                if signal["action"] != "NO_SIGNAL" and signal["action"] != last_action:
                    # 포지션 상태 업데이트
                    if signal["action"] == "ENTER_LONG":
                        current_position = 1
                    elif signal["action"] == "EXIT_LONG":
                        current_position = 0
                    last_action = signal["action"]
                    if args.json:
                        print(json.dumps(signal, ensure_ascii=False, indent=2))
                    else:
                        print(f"[{signal['timestamp']}] {signal['regime']} | {signal['action']}")
                        print(f"  사유: {signal['reason']}")
                        if signal['position_size']:
                            print(f"  포지션 사이즈: {signal['position_size']} 계약")
                        if signal['entry_px']:
                            print(f"  진입가: {signal['entry_px']:.2f}")
                        if signal['stop_px']:
                            print(f"  손절가: {signal['stop_px']:.2f}")
                        if signal['target_px']:
                            print(f"  목표가: {signal['target_px']:.2f}")
                    print("-" * 60)
        else:
            signal = generate_signal(df, current_position=args.position, capital=args.capital)
            if args.json:
                print(json.dumps(signal, ensure_ascii=False, indent=2))
            else:
                print(f"[{signal['timestamp']}] {signal['regime']} | {signal['action']}")
                print(f"  사유: {signal['reason']}")
                if signal['position_size']:
                    print(f"  포지션 사이즈: {signal['position_size']} 계약")
                if signal['entry_px']:
                    print(f"  진입가: {signal['entry_px']:.2f}")
                if signal['stop_px']:
                    print(f"  손절가: {signal['stop_px']:.2f}")
                if signal['target_px']:
                    print(f"  목표가: {signal['target_px']:.2f}")
        
        del df
        gc.collect()
    
    def run_live():
        """LS OpenAPI로 실시간 데이터 수집 후 신호 생성."""
        import time
        
        raw_secrets = load_secrets(args.config)
        nested = raw_secrets.get("ebest", {}) if isinstance(raw_secrets.get("ebest"), dict) else {}
        secrets = {**raw_secrets, **nested}
        
        appkey = get_secret(secrets, "appkey", "app_key", "APPKEY", "ls_appkey")
        appsecret = get_secret(secrets, "appsecret", "appsecretkey", "app_secret", "APPSECRET", "ls_appsecret")
        openapi_server = get_secret(secrets, "mode", "server", "openapi_server") or "demo"
        symbol = args.symbol or get_secret(secrets, "symbol", "shcode", "focode") or "A0169000"
        
        if not appkey or not appsecret:
            print(f"[오류] LS OpenAPI 인증 정보가 없습니다. ({args.config})")
            return
        
        collector = LSOpenAPICollector(appkey, appsecret, is_demo=(openapi_server == "demo"))
        store = DataStore()
        order_executor = None
        if args.order:
            account = args.account or get_secret(secrets, "account", "acct_no", "계좌번호")
            password = args.password or get_secret(secrets, "password", "acct_pw", "계좌비밀번호")
            order_executor = LSOpenAPIOrder(
                collector,
                account=account,
                password=password,
                dry_run=not args.live_order,
            )
            mode = "실주문" if args.live_order else "DRY-RUN"
            print(f"[Order] 자동 주문 실행기 초기화: {mode} 모드")
        
        if not collector.connect():
            return
        
        print(f"BULL 피봇 롱 실시간 신호 생성기 시작 (종목: {symbol})")
        print("-" * 60)
        
        try:
            while True:
                # 1. 최신 5분봉 수집
                data = collector.get_latest_ohlcv(symbol)
                if data is None:
                    print("[Live] 데이터 수집 실패, 5분 후 재시도")
                    if args.single_shot:
                        break
                    time.sleep(300)
                    continue
                
                print(f"[Live] 수집: {data['timestamp']} | O={data['open']} H={data['high']} L={data['low']} C={data['close']} V={data['volume']}")
                
                # 2. DuckDB 저장
                store.save(data)
                
                # 3. 최신 데이터 로드 및 신호 생성
                df = load_recent_data(days=120)
                if len(df) == 0:
                    print("[Live] 데이터 로드 실패")
                else:
                    signal = generate_signal(df, current_position=args.position, capital=args.capital)
                    if args.json:
                        print(json.dumps(signal, ensure_ascii=False, indent=2))
                    else:
                        print(f"[{signal['timestamp']}] {signal['regime']} | {signal['action']}")
                        print(f"  사유: {signal['reason']}")
                        if signal['position_size']:
                            print(f"  포지션 사이즈: {signal['position_size']} 계약")
                        if signal['entry_px']:
                            print(f"  진입가: {signal['entry_px']:.2f}")
                        if signal['stop_px']:
                            print(f"  손절가: {signal['stop_px']:.2f}")
                        if signal['target_px']:
                            print(f"  목표가: {signal['target_px']:.2f}")
                    
                    # 자동 주문 실행
                    if order_executor and signal['action'] in ("ENTER_LONG", "EXIT_LONG"):
                        side = "buy" if signal['action'] == "ENTER_LONG" else "sell"
                        qty = signal.get('position_size') or 1
                        price = signal.get('entry_px') if signal['action'] == "ENTER_LONG" else signal.get('exit_px')
                        if price:
                            order_executor.submit_order(symbol, side, qty, price=price, price_type="limit")
                        else:
                            order_executor.submit_order(symbol, side, qty, price_type="market")
                
                print("-" * 60)
                if args.single_shot:
                    break
                time.sleep(300)
        except KeyboardInterrupt:
            print("\n[Live] 중단")
        finally:
            collector.disconnect()
    
    def run_websocket():
        """WebSocket 초 단위 실시간 체결 → 5분봉 집계 → 신호 생성."""
        import time
        
        raw_secrets = load_secrets(args.config)
        nested = raw_secrets.get("ebest", {}) if isinstance(raw_secrets.get("ebest"), dict) else {}
        secrets = {**raw_secrets, **nested}
        
        appkey = get_secret(secrets, "appkey", "app_key", "APPKEY", "ls_appkey")
        appsecret = get_secret(secrets, "appsecret", "appsecretkey", "app_secret", "APPSECRET", "ls_appsecret")
        openapi_server = get_secret(secrets, "mode", "server", "openapi_server") or "demo"
        symbol = args.symbol or get_secret(secrets, "symbol", "shcode", "focode") or "A0169000"
        
        if not appkey or not appsecret:
            print(f"[오류] LS OpenAPI 인증 정보가 없습니다. ({args.config})")
            return
        
        # 1. REST 토큰 발급
        collector = LSOpenAPICollector(appkey, appsecret, is_demo=(openapi_server == "demo"))
        if not collector.connect():
            return
        
        # TODO: Telegram 알림 연동 (향후 구현 필요)
        # - 신호 발생 시 알림 (ENTER_LONG / EXIT_LONG / STOP)
        # - WebSocket 재연결 시 알림
        # - 토큰 갱신 시 알림
        # - 주문 체결/실패 시 알림
        
        store = DataStore()
        order_executor = None
        if args.order:
            account = args.account or get_secret(secrets, "account", "acct_no", "계좌번호")
            password = args.password or get_secret(secrets, "password", "acct_pw", "계좌비밀번호")
            order_executor = LSOpenAPIOrder(
                collector,
                account=account,
                password=password,
                dry_run=not args.live_order,
            )
            mode = "실주문" if args.live_order else "DRY-RUN"
            print(f"[Order] 자동 주문 실행기 초기화: {mode} 모드")
        
        print(f"BULL 피봇 롱 WebSocket 신호 생성기 시작 (종목: {symbol})")
        print("-" * 60)
        
        def on_tick(tick: Dict[str, Any]):
            print(f"[WebSocket] tick {tick['chetime']} | price={tick['price']} | vol={tick['volume']}")
        
        def on_bar(bar: Dict[str, Any]):
            print(f"[WebSocket] 5min bar {bar['timestamp']} | O={bar['open']} H={bar['high']} L={bar['low']} C={bar['close']} V={bar['volume']}")
            store.save(bar)
            df = load_recent_data(days=120)
            if len(df) > 0:
                signal = generate_signal(df, current_position=args.position, capital=args.capital)
                if args.json:
                    print(json.dumps(signal, ensure_ascii=False, indent=2))
                else:
                    print(f"[{signal['timestamp']}] {signal['regime']} | {signal['action']}")
                    print(f"  사유: {signal['reason']}")
                    if signal['position_size']:
                        print(f"  포지션 사이즈: {signal['position_size']} 계약")
                    if signal['entry_px']:
                        print(f"  진입가: {signal['entry_px']:.2f}")
                    if signal['stop_px']:
                        print(f"  손절가: {signal['stop_px']:.2f}")
                    if signal['target_px']:
                        print(f"  목표가: {signal['target_px']:.2f}")
                
                # 자동 주문 실행
                if order_executor and signal['action'] in ("ENTER_LONG", "EXIT_LONG"):
                    side = "buy" if signal['action'] == "ENTER_LONG" else "sell"
                    qty = signal.get('position_size') or 1
                    price = signal.get('entry_px') if signal['action'] == "ENTER_LONG" else signal.get('exit_px')
                    if price:
                        order_executor.submit_order(symbol, side, qty, price=price, price_type="limit")
                    else:
                        order_executor.submit_order(symbol, side, qty, price_type="market")
            print("-" * 60)
        
        # 2. WebSocket 연결 (토큰 자동 갱신 콜백 전달)
        ws = LSRealtimeWebSocket(
            access_token=collector.access_token,
            symbol=symbol,
            tr_cd="FC9",
            on_tick=on_tick,
            on_bar=on_bar,
            token_getter=collector.get_valid_token,
        )
        ws.connect()
        
        try:
            if args.ws_duration > 0:
                time.sleep(args.ws_duration)
            else:
                while ws.is_connected():
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\n[WebSocket] 중단")
        finally:
            ws.disconnect()
            collector.disconnect()
    
    if args.live:
        run_live()
    elif args.websocket:
        run_websocket()
    elif args.loop:
        import time
        print("BULL 피봇 롱 신호 생성기 시작 (5분마다 실행)")
        while True:
            run_once()
            print("-" * 60)
            time.sleep(300)
    else:
        run_once()


if __name__ == "__main__":
    main()
