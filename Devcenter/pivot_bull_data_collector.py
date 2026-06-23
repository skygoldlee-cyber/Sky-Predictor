# -*- coding: utf-8 -*-
"""
BULL 피봇 롱 전략용 실시간 데이터 수집 파이프라인

현재 지원:
    - LS증권(구 이베스트) OpenAPI (REST)
    - LS증권(구 이베스트) XingAPI (COM/OCX)
    - Mock Collector (파일/DB 기반 테스트)

사용 예시:
    python pivot_bull_data_collector.py --mode mock --source-db "c:/.../market_data.duckdb"
    python pivot_bull_data_collector.py --mode ls --id YOUR_ID --pw YOUR_PW --cert YOUR_CERT
    python pivot_bull_data_collector.py --mode openapi --appkey KEY --appsecret SECRET
"""
import sys
import time
import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import duckdb
import requests
import json
import websocket
import threading
import time as pytime

DB_PATH = "c:/Project/SkyPredictor v1/Devcenter/duckdb/market_data.duckdb"
TABLE_NAME = "futures_5min"
SESSION_BOUNDARY_HOUR = 8


# ────────────────────────────────────────────────────────────────────────────
# 추상 인터페이스
# ────────────────────────────────────────────────────────────────────────────
class DataCollector(ABC):
    """실시간 시세 수집기 추상 인터페이스."""

    @abstractmethod
    def connect(self) -> bool:
        """API 연결/로그인."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """API 연결 해제."""
        pass

    @abstractmethod
    def get_latest_ohlcv(self, symbol: str = "KP200") -> Optional[Dict[str, Any]]:
        """최신 5분봉 OHLCV 데이터 반환."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """연결 상태 확인."""
        pass


# ────────────────────────────────────────────────────────────────────────────
# LS증권(구 이베스트) XingAPI Collector
# ────────────────────────────────────────────────────────────────────────────
class LSEbestCollector(DataCollector):
    """
    LS증권(구 이베스트) XingAPI 기반 5분봉 수집기.
    
    필요 조건:
        - Windows OS
        - XingAPI OCX 등록 (xhopen, xingAPI 등)
        - LS증권 계좌 및 API 사용 신청
        - win32com 패키지 설치: pip install pywin32
    
    사용 예시:
        collector = LSEbestCollector(id="user", pw="pass", cert="certpw")
        if collector.connect():
            data = collector.get_latest_ohlcv("KP200")
    """

    def __init__(self, user_id: str, user_pw: str, cert_pw: str,
                 server: str = "demo"):
        self.user_id = user_id
        self.user_pw = user_pw
        self.cert_pw = cert_pw
        self.server = server
        self.xa_session = None
        self._connected = False

    def connect(self) -> bool:
        try:
            import win32com.client
            self.xa_session = win32com.client.Dispatch("XA_Session.XA_Session")
            # 서버 선택: "demo" (모의) 또는 "real" (실전)
            server_addr = "demo.ebestsec.co.kr" if self.server == "demo" else "hts.ebestsec.co.kr"
            server_port = 20001
            self._connected = self.xa_session.ConnectServer(server_addr, server_port)
            if not self._connected:
                print("[LSEbest] 서버 연결 실패")
                return False
            
            login_ok = self.xa_session.Login(self.user_id, self.user_pw, self.cert_pw, 0, False)
            if not login_ok:
                print("[LSEbest] 로그인 실패")
                return False
            
            print("[LSEbest] 로그인 성공")
            self._connected = True
            return True
        except Exception as e:
            print(f"[LSEbest] 연결 오류: {e}")
            return False

    def disconnect(self) -> None:
        if self.xa_session:
            self.xa_session.DisconnectServer()
        self._connected = False
        print("[LSEbest] 연결 종료")

    def is_connected(self) -> bool:
        return self._connected

    def get_latest_ohlcv(self, symbol: str = "KP200") -> Optional[Dict[str, Any]]:
        """
        KOSPI200 선물 최근 5분봉 1건 요청 (t8412 - 주식/선물 차트).
        
        참고: 실제 TR 번호는 LS증권 API 문서를 확인해야 합니다.
        """
        if not self._connected:
            print("[LSEbest] 연결되지 않음")
            return None
        
        try:
            import win32com.client
            # XAQuery 객체 생성 (예: t8412 - 선물 분봉 차트)
            query = win32com.client.Dispatch("XA_DataSet.XAQuery")
            query.ResFileName = "C:\\eBEST\\xingAPI\\Res\\t8412.res"  # 실제 Res 파일 경로
            
            # 입력값 설정 (예시 - 실제 문서 확인 필요)
            query.SetFieldData("t8412InBlock", "shcode", 0, symbol)  # 종목코드
            query.SetFieldData("t8412InBlock", "ncnt", 0, 5)        # 분봉 주기
            query.SetFieldData("t8412InBlock", "qrycnt", 0, 1)       # 요청 개수
            query.SetFieldData("t8412InBlock", "nday", 0, 1)          # 당일만
            
            query.Request(0)
            # 동기 응답 대기 (실제로는 이벤트 기반)
            time.sleep(1)
            
            count = query.GetBlockCount("t8412OutBlock1")
            if count == 0:
                return None
            
            # 마지막 데이터
            i = count - 1
            data = {
                "timestamp": pd.to_datetime(query.GetFieldData("t8412OutBlock1", "date", i) + " " + query.GetFieldData("t8412OutBlock1", "time", i)),
                "open": float(query.GetFieldData("t8412OutBlock1", "open", i)),
                "high": float(query.GetFieldData("t8412OutBlock1", "high", i)),
                "low": float(query.GetFieldData("t8412OutBlock1", "low", i)),
                "close": float(query.GetFieldData("t8412OutBlock1", "close", i)),
                "volume": int(query.GetFieldData("t8412OutBlock1", "volume", i)),
            }
            return data
        except Exception as e:
            print(f"[LSEbest] 데이터 요청 오류: {e}")
            return None


# ────────────────────────────────────────────────────────────────────────────
# LS증권 OpenAPI Collector
# ────────────────────────────────────────────────────────────────────────────
class LSOpenAPICollector(DataCollector):
    """
    LS증권(구 이베스트) OpenAPI (REST) 기반 5분봉 수집기.
    
    필요 조건:
        - LS증권 계좌 및 OpenAPI 사용 신청
        - appkey, appsecret 발급
        - requests 패키지: pip install requests
    
    사용 예시:
        collector = LSOpenAPICollector(appkey="...", appsecret="...", is_demo=True)
        if collector.connect():
            data = collector.get_latest_ohlcv("101TC000")  # KOSPI200 선물 코드
    
    참고:
        - 실제 엔드포인트는 LS증권 OpenAPI 문서를 확인하여 수정 필요
        - 토큰 유효기간 관리를 위해 자동 갱신 포함
    """

    def __init__(self, appkey: str, appsecret: str, is_demo: bool = True):
        self.appkey = appkey
        self.appsecret = appsecret
        self.is_demo = is_demo
        # eBest/LS OpenAPI URL (모의/실전 서버별로 수정 필요)
        self.base_url = "https://openapi.ls-sec.co.kr:8080" if is_demo else "https://openapi.ls-sec.co.kr"
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self._connected = False

    def _issue_token(self) -> bool:
        """OAuth access token 발급."""
        url = f"{self.base_url}/oauth2/token"
        headers = {"content-type": "application/x-www-form-urlencoded"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.appkey,
            "appsecretkey": self.appsecret,
            "scope": "oob",  # OOB (Out-Of-Band)
        }
        try:
            resp = requests.post(url, headers=headers, data=body, timeout=10)
            if resp.status_code != 200:
                print(f"[LSOpenAPI] 토큰 응답 {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            self.access_token = data.get("access_token")
            expires_in = data.get("expires_in", 86400)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)
            print(f"[LSOpenAPI] 토큰 발급 성공 (expires_in={expires_in})")
            return bool(self.access_token)
        except Exception as e:
            print(f"[LSOpenAPI] 토큰 발급 오류: {e}")
            return False

    def _ensure_token(self) -> bool:
        """토큰이 유효한지 확인하고, 만료되면 재발급."""
        if self.access_token and self.token_expires_at and datetime.now() < self.token_expires_at:
            return True
        return self._issue_token()

    def connect(self) -> bool:
        self._connected = self._issue_token()
        return self._connected

    def disconnect(self) -> None:
        self.access_token = None
        self.token_expires_at = None
        self._connected = False
        print("[LSOpenAPI] 연결 종료")

    def is_connected(self) -> bool:
        return self._connected and self._ensure_token()

    def get_valid_token(self) -> Optional[str]:
        """유효한 access_token 반환 (만료 시 자동 갱신)."""
        if self._ensure_token():
            return self.access_token
        return None

    # TR 코드별 access_url 매핑 (openapi.ls-sec.co.kr 기준)
    TR_ENDPOINTS = {
        "t8465": "/futureoption/chart",   # 선물/옵션 차트(N분)
        "t8467": "/futureoption/market-data",  # 지수선물 마스터조회
    }

    def _request_tr(self, tr_id: str, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """LS OpenAPI TR 공통 요청."""
        if not self._ensure_token():
            print("[LSOpenAPI] 토큰 없음")
            return None
        
        path = self.TR_ENDPOINTS.get(tr_id, "")
        url = f"{self.base_url}{path}"
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "tr_cd": tr_id,
            "tr_cont": "N",
            "tr_cont_key": "",
            "mac_address": "",
        }
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=10)
            if resp.status_code != 200:
                print(f"[LSOpenAPI] {tr_id} 응답 {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[LSOpenAPI] {tr_id} 요청 오류: {e}")
            return None

    def get_latest_ohlcv(self, symbol: str = "A0169000") -> Optional[Dict[str, Any]]:
        """
        KOSPI200 선물 최근 5분봉 1건 요청 (t8465 — 선물/옵션 차트 N분).
        
        EBEST_OPENAPI_SCHEMA.md 기준:
            - InBlock:  shcode, ncnt(5), qrycnt, nday, sdate, stime, edate, etime, cts_date, cts_time, comp_yn
            - OutBlock1: date, time, open, high, low, close, jdiff_vol
        """
        tr_id = "t8465"
        today = datetime.now().strftime("%Y%m%d")
        body = {
            f"{tr_id}InBlock": {
                "shcode": symbol,           # 단축코드
                "ncnt": 5,                   # 5분봉
                "qrycnt": 10,                # 최근 10건
                "nday": "0",                 # 당일 기준 사용안함
                "sdate": "",                 # space
                "stime": "",                 # space
                "edate": "99999999",         # 최근 데이터
                "etime": "",                 # space
                "cts_date": "",              # space
                "cts_time": "",              # space
                "comp_yn": "N",              # 비압축
            }
        }
        data = self._request_tr(tr_id, body)
        if data is None:
            return None
        
        rows = data.get(f"{tr_id}OutBlock1", [])
        if not rows:
            print(f"[LSOpenAPI] {tr_id} OutBlock1 데이터 없음")
            return None
        
        row = rows[-1]
        date_str = str(row.get("date", ""))
        time_str = str(row.get("time", ""))
        ts_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
        
        return {
            "timestamp": pd.to_datetime(ts_str),
            "open": float(row.get("open")),
            "high": float(row.get("high")),
            "low": float(row.get("low")),
            "close": float(row.get("close")),
            "volume": int(row.get("jdiff_vol", 0)),
        }

    def get_futures_symbols(self) -> List[Dict[str, Any]]:
        """
        지수선물 마스터조회 (t8467) — 유효한 선물 종목코드 목록 반환.
        """
        tr_id = "t8467"
        body = {
            f"{tr_id}InBlock": {
                "gubun": "1"  # 1: 전체
            }
        }
        data = self._request_tr(tr_id, body)
        if data is None:
            return []
        
        rows = data.get(f"{tr_id}OutBlock", []) or data.get(f"{tr_id}OutBlock1", [])
        return rows

    def backfill_ohlcv(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        KOSPI200 선물 과거 5분봉 백필 (t8465 연속 조회).
        
        Args:
            symbol: 단축코드 (예: A0169000)
            start_date: 시작일자 (YYYYMMDD)
            end_date: 종료일자 (YYYYMMDD)
        """
        tr_id = "t8465"
        all_rows: List[Dict[str, Any]] = []
        cts_date = ""
        cts_time = ""
        max_pages = 10  # 안전장치
        
        for page in range(max_pages):
            body = {
                f"{tr_id}InBlock": {
                    "shcode": symbol,
                    "ncnt": 5,
                    "qrycnt": 500,
                    "nday": "0",
                    "sdate": start_date,
                    "stime": "",
                    "edate": end_date,
                    "etime": "",
                    "cts_date": cts_date,
                    "cts_time": cts_time,
                    "comp_yn": "N",
                }
            }
            data = self._request_tr(tr_id, body)
            if data is None:
                break
            
            rows = data.get(f"{tr_id}OutBlock1", [])
            if not rows:
                break
            
            for row in rows:
                date_str = str(row.get("date", ""))
                time_str = str(row.get("time", ""))
                ts_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                all_rows.append({
                    "timestamp": pd.to_datetime(ts_str),
                    "open": float(row.get("open")),
                    "high": float(row.get("high")),
                    "low": float(row.get("low")),
                    "close": float(row.get("close")),
                    "volume": int(row.get("jdiff_vol", 0)),
                })
            
            # 연속 조회 키 확인
            header = data.get(f"{tr_id}OutBlock", {})
            next_cts_date = str(header.get("cts_date", "")).strip()
            next_cts_time = str(header.get("cts_time", "")).strip()
            
            if not next_cts_date or (next_cts_date == cts_date and next_cts_time == cts_time):
                break
            
            cts_date = next_cts_date
            cts_time = next_cts_time
            print(f"[LSOpenAPI] {tr_id} 연속 조회: cts_date={cts_date}, cts_time={cts_time}")
        
        # 중복 제거 및 시간순 정렬
        df = pd.DataFrame(all_rows)
        if len(df) == 0:
            return []
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        return df.to_dict("records")

    def get_today_ohlcv(self, symbol: str = "A0169000") -> List[Dict[str, Any]]:
        """
        KOSPI200 선물 당일 전체 5분봉 조회 (t8465).
        """
        tr_id = "t8465"
        today = datetime.now().strftime("%Y%m%d")
        body = {
            f"{tr_id}InBlock": {
                "shcode": symbol,
                "ncnt": 5,
                "qrycnt": 500,               # 당일 최대 500건
                "nday": "1",
                "sdate": "",
                "stime": "",
                "edate": today,
                "etime": "",
                "cts_date": "",
                "cts_time": "",
                "comp_yn": "N",
            }
        }
        data = self._request_tr(tr_id, body)
        if data is None:
            return []
        
        rows = data.get(f"{tr_id}OutBlock1", [])
        results = []
        for row in rows:
            date_str = str(row.get("date", ""))
            time_str = str(row.get("time", ""))
            ts_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
            results.append({
                "timestamp": pd.to_datetime(ts_str),
                "open": float(row.get("open")),
                "high": float(row.get("high")),
                "low": float(row.get("low")),
                "close": float(row.get("close")),
                "volume": int(row.get("jdiff_vol", 0)),
            })
        return results


# ────────────────────────────────────────────────────────────────────────────
# 선물/옵션 주문
# ────────────────────────────────────────────────────────────────────────────
class LSOpenAPIOrder:
    """LS증권 OpenAPI 선물/옵션 주문 실행기 (CFOAT00100)."""

    def __init__(
        self,
        collector: LSOpenAPICollector,
        account: Optional[str] = None,
        password: Optional[str] = None,
        dry_run: bool = True,
    ):
        self.collector = collector
        self.account = account
        self.password = password
        self.dry_run = dry_run

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: Optional[float] = None,
        price_type: str = "market",
    ) -> Optional[Dict[str, Any]]:
        """
        선물/옵션 주문 제출.

        Args:
            symbol: 단축코드 (예: A0169000)
            side: "buy" 또는 "sell"
            qty: 주문수량
            price: 지정가 주문 시 가격 (market이면 무시)
            price_type: "market" 또는 "limit"
        """
        if side not in ("buy", "sell"):
            print("[Order] side는 buy 또는 sell이어야 합니다")
            return None
        if qty <= 0:
            print("[Order] qty는 0보다 커야 합니다")
            return None

        bns_tp_code = "2" if side == "buy" else "1"
        if price_type == "market":
            price_pattern = "03"
            fno_ord_prc = "0"
        elif price_type == "limit":
            price_pattern = "00"
            fno_ord_prc = str(price) if price is not None else "0"
        else:
            print("[Order] 지원하지 않는 price_type")
            return None

        body = {
            "CFOAT00100InBlock1": {
                "FnoIsuNo": symbol,
                "BnsTpCode": bns_tp_code,
                "FnoOrdprcPtnCode": price_pattern,
                "FnoOrdPrc": fno_ord_prc,
                "OrdQty": str(qty),
            }
        }
        if self.account:
            body["CFOAT00100InBlock1"]["AcntNo"] = self.account
        if self.password:
            body["CFOAT00100InBlock1"]["Pwd"] = self.password

        action = "매수" if side == "buy" else "매도"
        if self.dry_run:
            print(f"[Order] DRY-RUN: {action} {qty}계약 {symbol} @ {price_type} {fno_ord_prc}")
            return {"dry_run": True, "body": body}

        if not self.collector._ensure_token():
            print("[Order] 토큰 없음")
            return None

        url = f"{self.collector.base_url}/futureoption/order"
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.collector.access_token}",
            "tr_cd": "CFOAT00100",
            "tr_cont": "N",
            "tr_cont_key": "",
            "mac_address": "",
        }
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=10)
            if resp.status_code != 200:
                print(f"[Order] 응답 {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            print(f"[Order] 주문 결과: {data.get('rsp_cd')} - {data.get('rsp_msg')}")
            return data
        except Exception as e:
            print(f"[Order] 주문 요청 오류: {e}")
            return None


# ────────────────────────────────────────────────────────────────────────────
# 실시간 WebSocket (초 단위 체결 → 5분봉 집계)
# ────────────────────────────────────────────────────────────────────────────
class TickTo5MinAggregator:
    """실시간 체결 tick을 5분봉으로 집계."""

    def __init__(self, on_bar: Optional[Any] = None):
        self.on_bar = on_bar
        self.current_bar: Optional[Dict[str, Any]] = None
        self.current_bucket: Optional[datetime] = None

    def _bucket_time(self, chetime: str) -> datetime:
        """chetime(HHMMSS)을 5분 단위 버킷으로 변환."""
        now = datetime.now()
        hour = int(chetime[:2])
        minute = int(chetime[2:4])
        second = int(chetime[4:6])
        bucket_minute = (minute // 5) * 5
        bucket = now.replace(hour=hour, minute=bucket_minute, second=0, microsecond=0)
        # 시간이 현재보다 많이 미래면 전일 거래로 간주 (예: 심야 실행 시)
        if bucket > now + timedelta(hours=2):
            bucket -= timedelta(days=1)
        return bucket

    def add_tick(self, tick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """tick 추가. 5분봉 완성 시 completed bar 반환."""
        chetime = tick.get("chetime", "")
        if len(chetime) < 6:
            return None
        
        bucket = self._bucket_time(chetime)
        price = float(tick.get("price", 0))
        volume = int(tick.get("volume", 0))
        
        completed_bar = None
        if self.current_bucket is None:
            self.current_bucket = bucket
            self.current_bar = {
                "timestamp": bucket,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
            }
        elif bucket == self.current_bucket:
            self.current_bar["high"] = max(self.current_bar["high"], price)
            self.current_bar["low"] = min(self.current_bar["low"], price)
            self.current_bar["close"] = price
            self.current_bar["volume"] += volume
        else:
            completed_bar = self.current_bar.copy()
            if self.on_bar:
                self.on_bar(completed_bar)
            self.current_bucket = bucket
            self.current_bar = {
                "timestamp": bucket,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
            }
        return completed_bar

    def flush(self) -> Optional[Dict[str, Any]]:
        """현재 진행 중인 봉 강제 반환."""
        bar = self.current_bar
        self.current_bar = None
        self.current_bucket = None
        return bar


class LSRealtimeWebSocket:
    """LS증권 OpenAPI WebSocket — FC9 선물 실시간 체결."""

    def __init__(
        self,
        access_token: str,
        symbol: str = "A0169000",
        tr_cd: str = "FC9",
        on_tick: Optional[Any] = None,
        on_bar: Optional[Any] = None,
        ws_url: str = "wss://openapi.ls-sec.co.kr:9443/websocket",
        max_reconnect: int = 10,
        token_getter: Optional[Any] = None,
    ):
        self.access_token = access_token
        self.symbol = symbol
        self.tr_cd = tr_cd
        self.ws_url = ws_url
        self.on_tick = on_tick
        self.on_bar = on_bar
        self.token_getter = token_getter
        self.ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._aggregator = TickTo5MinAggregator(on_bar=on_bar)
        self._running = False
        self._subscribed = False
        self._max_reconnect = max_reconnect
        self._reconnect_attempts = 0
        self._reconnect_timer: Optional[threading.Timer] = None

    def _normalize_tick(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """FC9 수신 body를 표준 tick dict로 변환."""
        return {
            "symbol": body.get("futcode") or body.get("optcode") or self.symbol,
            "chetime": str(body.get("chetime", "")),
            "price": float(body.get("price", 0) or 0),
            "volume": int(body.get("cvolume", 0) or 0),
            "open": float(body.get("open", 0) or 0),
            "high": float(body.get("high", 0) or 0),
            "low": float(body.get("low", 0) or 0),
            "cum_volume": int(body.get("volume", 0) or 0),
        }

    def _send_subscribe(self, ws: websocket.WebSocketApp, tr_type: str = "3"):
        msg = {
            "header": {
                "token": self.access_token,
                "tr_type": tr_type,
            },
            "body": {
                "tr_cd": self.tr_cd,
                "tr_key": self.symbol,
            }
        }
        ws.send(json.dumps(msg))
        action = "구독" if tr_type == "3" else "해제"
        print(f"[WebSocket] {self.tr_cd} {action}: {self.symbol}")

    def _on_open(self, ws: websocket.WebSocketApp):
        self._reconnect_attempts = 0  # 연결 성공 시 시도 횟수 초기화
        self._send_subscribe(ws, "3")
        self._subscribed = True

    def _on_message(self, ws: websocket.WebSocketApp, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            print(f"[WebSocket] JSON 파싱 오류: {message[:200]}")
            return
        
        body = data.get("body", {})
        header = data.get("header", {})
        
        # ping/pong 또는 시스템 메시지
        if not body or "chetime" not in body:
            print(f"[WebSocket] 시스템 메시지: {data}")
            return
        
        tick = self._normalize_tick(body)
        completed_bar = self._aggregator.add_tick(tick)
        if self.on_tick:
            self.on_tick(tick)
        if completed_bar and self.on_bar:
            self.on_bar(completed_bar)

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception):
        print(f"[WebSocket] 오류: {error}")

    def _on_close(self, ws: websocket.WebSocketApp, close_status_code: int, close_msg: str):
        self._subscribed = False
        print(f"[WebSocket] 연결 종료: {close_status_code} {close_msg}")
        if self._running:
            self._schedule_reconnect()

    def _cancel_reconnect_timer(self) -> None:
        """예약된 재연결 타이머 취소."""
        if self._reconnect_timer:
            self._reconnect_timer.cancel()
            self._reconnect_timer = None

    def _schedule_reconnect(self) -> None:
        """끊김 시 지수 백오프로 재연결 예약."""
        if not self._running:
            return
        if self._reconnect_attempts >= self._max_reconnect:
            print(f"[WebSocket] 최대 재연결 횟수 초과: {self._max_reconnect}")
            return
        
        self._reconnect_attempts += 1
        delay = min(2 ** (self._reconnect_attempts - 1), 30)
        print(f"[WebSocket] {delay}초 후 재연결 시도 ({self._reconnect_attempts}/{self._max_reconnect})")
        
        self._cancel_reconnect_timer()
        self._reconnect_timer = threading.Timer(delay, self._do_connect)
        self._reconnect_timer.daemon = True
        self._reconnect_timer.start()

    def _do_connect(self) -> None:
        """실제 WebSocket 연결 생성 (토큰 갱신 후)."""
        # 재연결 전 access_token 만료 확인 및 갱신
        if self.token_getter:
            new_token = self.token_getter()
            if new_token:
                self.access_token = new_token
            else:
                print("[WebSocket] 토큰 갱신 실패, 재연결 시도")
                self._schedule_reconnect()
                return
        
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self._thread = threading.Thread(target=self.ws.run_forever)
            self._thread.daemon = True
            self._thread.start()
            print(f"[WebSocket] 연결 시작: {self.ws_url}")
        except Exception as e:
            print(f"[WebSocket] 연결 시작 오류: {e}")
            self._schedule_reconnect()

    def connect(self) -> bool:
        """WebSocket 연결 시작 (별도 스레드, 끊김 시 자동 재연결)."""
        if not self.access_token:
            print("[WebSocket] 토큰 없음")
            return False
        
        self._running = True
        self._reconnect_attempts = 0
        self._cancel_reconnect_timer()
        self._do_connect()
        return True

    def disconnect(self) -> None:
        """WebSocket 연결 종료 및 구독 해제 (재연글 중단)."""
        self._running = False
        self._cancel_reconnect_timer()
        if self.ws and self._subscribed:
            try:
                self._send_subscribe(self.ws, "4")
                pytime.sleep(0.5)
            except Exception as e:
                print(f"[WebSocket] 해제 오류: {e}")
        if self.ws:
            self.ws.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        print("[WebSocket] 연결 종료")

    def is_connected(self) -> bool:
        return self._running and self._subscribed


# ────────────────────────────────────────────────────────────────────────────
# Mock Collector (테스트용)
# ────────────────────────────────────────────────────────────────────────────
class MockCollector(DataCollector):
    """
    기존 DB에서 마지막 데이터를 읽어오는 Mock 수집기.
    실제 API 없이 파이프라인 구조를 테스트할 때 사용.
    """

    def __init__(self, db_path: str = DB_PATH, table_name: str = TABLE_NAME):
        self.db_path = db_path
        self.table_name = table_name
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        print("[Mock] 연결 성공 (DB 기반)")
        return True

    def disconnect(self) -> None:
        self._connected = False
        print("[Mock] 연결 종료")

    def is_connected(self) -> bool:
        return self._connected

    def get_latest_ohlcv(self, symbol: str = "KP200") -> Optional[Dict[str, Any]]:
        if not self._connected:
            return None
        
        try:
            con = duckdb.connect(database=self.db_path, read_only=True)
            df = con.execute(
                f"SELECT * FROM {self.table_name} ORDER BY timestamp DESC LIMIT 1"
            ).df()
            con.close()
            
            if len(df) == 0:
                return None
            
            row = df.iloc[0]
            return {
                "timestamp": pd.to_datetime(row["timestamp"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }
        except Exception as e:
            print(f"[Mock] 데이터 조회 오류: {e}")
            return None


# ────────────────────────────────────────────────────────────────────────────
# 데이터 저장
# ────────────────────────────────────────────────────────────────────────────
class DataStore:
    """DuckDB에 5분봉 데이터 저장."""

    def __init__(self, db_path: str = DB_PATH, table_name: str = TABLE_NAME):
        self.db_path = db_path
        self.table_name = table_name

    def save(self, data: Dict[str, Any]) -> bool:
        return self.save_many([data])

    def save_many(self, data_list: List[Dict[str, Any]]) -> bool:
        """여러 5분봉 데이터를 한 번에 저장 (중복 제거)."""
        if not data_list:
            return True
        try:
            con = duckdb.connect(database=self.db_path)
            df = pd.DataFrame(data_list)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # 테이블이 없으면 생성
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    timestamp TIMESTAMP,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT
                )
            """)
            
            # 중복 방지 (timestamp 기준 UPSERT)
            timestamps = [f"'{d['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}'" for d in data_list]
            con.execute(f"""
                DELETE FROM {self.table_name} 
                WHERE timestamp IN ({', '.join(timestamps)})
            """)
            con.execute(f"INSERT INTO {self.table_name} SELECT * FROM df")
            con.close()
            print(f"[DataStore] {len(data_list)}건 저장 완료")
            return True
        except Exception as e:
            print(f"[DataStore] 저장 오류: {e}")
            return False


# ────────────────────────────────────────────────────────────────────────────
# 스케줄러
# ────────────────────────────────────────────────────────────────────────────
class CollectorScheduler:
    """주기적으로 5분봉 데이터를 수집하고 저장."""

    def __init__(self, collector: DataCollector, store: DataStore,
                 interval_seconds: int = 300, symbol: str = "A0169000"):
        self.collector = collector
        self.store = store
        self.interval_seconds = interval_seconds
        self.symbol = symbol
        self._running = False

    def run_once(self) -> bool:
        data = self.collector.get_latest_ohlcv(self.symbol)
        if data is None:
            print("[Scheduler] 수집된 데이터 없음")
            return False
        
        print(f"[Scheduler] 수집: {data['timestamp']} | O={data['open']} H={data['high']} L={data['low']} C={data['close']} V={data['volume']}")
        return self.store.save(data)

    def run_loop(self) -> None:
        self._running = True
        print(f"[Scheduler] {self.interval_seconds}초 간격으로 수집 시작")
        while self._running:
            self.run_once()
            print("-" * 60)
            time.sleep(self.interval_seconds)

    def stop(self) -> None:
        self._running = False


def load_secrets(config_path: str) -> Dict[str, Any]:
    """config.secrets.json에서 인증 정보 로드."""
    path = Path(config_path)
    if not path.exists():
        print(f"[설정] 파일 없음: {config_path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            secrets = json.load(f)
        print(f"[설정] 파일 로드 성공: {config_path} (keys: {list(secrets.keys())})")
        return secrets
    except Exception as e:
        print(f"[오류] 설정 파일 로드 실패: {e}")
        return {}


def get_secret(secrets: Dict[str, Any], *keys: str) -> str:
    """여러 후보 key 중 첫 번째로 존재하는 값 반환."""
    for key in keys:
        if key in secrets and secrets[key]:
            return str(secrets[key])
    return ""


# ────────────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BULL 피봇 롱 데이터 수집 파이프라인")
    parser.add_argument("--mode", choices=["mock", "ls", "openapi"], default="mock",
                        help="수집 모드: mock (DB), ls (XingAPI), openapi (REST)")
    parser.add_argument("--config", type=str, default="c:/Project/SkyPredictor v1/config.secrets.json",
                        help="인증 정보 JSON 파일 경로")
    parser.add_argument("--id", type=str, help="LS증권 ID (XingAPI)")
    parser.add_argument("--pw", type=str, help="LS증권 비밀번호 (XingAPI)")
    parser.add_argument("--cert", type=str, help="LS증권 공인인증서 비밀번호 (XingAPI)")
    parser.add_argument("--server", type=str, default="demo", help="LS증권 서버: demo 또는 real (XingAPI)")
    parser.add_argument("--appkey", type=str, help="LS증권 OpenAPI appkey")
    parser.add_argument("--appsecret", type=str, help="LS증권 OpenAPI appsecret")
    parser.add_argument("--openapi-server", type=str, default="demo", help="LS증권 OpenAPI 서버: demo 또는 real")
    parser.add_argument("--source-db", type=str, default=DB_PATH, help="Mock 모드 DB 경로")
    parser.add_argument("--symbol", type=str, default="A0169000", help="KOSPI200 선물 단축코드")
    parser.add_argument("--list-symbols", action="store_true", help="유효한 선물 종목코드 목록 출력")
    parser.add_argument("--backfill", action="store_true", help="과거 데이터 백필")
    parser.add_argument("--start-date", type=str, help="백필 시작일 (YYYYMMDD)")
    parser.add_argument("--end-date", type=str, help="백필 종료일 (YYYYMMDD)")
    parser.add_argument("--loop", action="store_true", help="주기적 수집 실행")
    parser.add_argument("--interval", type=int, default=300, help="수집 간격(초)")
    args = parser.parse_args()

    # 설정 파일 로드 (명령줄 인자가 우선)
    raw_secrets = load_secrets(args.config) if args.mode == "openapi" else {}
    # 중첩 객체 지원 (예: {"ebest": {"appkey": ...}})
    nested = raw_secrets.get("ebest", {}) if isinstance(raw_secrets.get("ebest"), dict) else {}
    secrets = {**raw_secrets, **nested}
    
    appkey = args.appkey or get_secret(secrets, "appkey", "app_key", "APPKEY", "ls_appkey")
    appsecret = args.appsecret or get_secret(secrets, "appsecret", "appsecretkey", "app_secret", "APPSECRET", "ls_appsecret")
    openapi_server = args.openapi_server or get_secret(secrets, "mode", "server", "openapi_server") or "demo"
    symbol = args.symbol or get_secret(secrets, "symbol", "shcode", "focode") or "105V6000"

    if args.mode == "ls":
        if not args.id or not args.pw or not args.cert:
            print("[오류] LS증권 XingAPI 모드는 --id, --pw, --cert 인자가 필요합니다.")
            return
        collector = LSEbestCollector(args.id, args.pw, args.cert, args.server)
    elif args.mode == "openapi":
        if not appkey or not appsecret:
            print(f"[오류] LS증권 OpenAPI 모드는 appkey/appsecret이 필요합니다. ({args.config} 또는 --appkey/--appsecret)")
            return
        collector = LSOpenAPICollector(appkey, appsecret, is_demo=(openapi_server == "demo"))
    else:
        collector = MockCollector(args.source_db)
    
    store = DataStore()
    scheduler = CollectorScheduler(collector, store, args.interval, symbol)
    
    if not collector.connect():
        return
    
    try:
        if args.mode == "openapi" and args.list_symbols:
            # 유효 선물 종목코드 목록 출력
            symbols = collector.get_futures_symbols()
            print(f"[LSOpenAPI] 유효 선물 종목 수: {len(symbols)}")
            for s in symbols[:20]:
                print(f"  {s}")
        elif args.mode == "openapi" and args.backfill:
            # 과거 데이터 백필
            if not args.start_date or not args.end_date:
                print("[오류] 백필은 --start-date, --end-date 인자가 필요합니다. (YYYYMMDD)")
            else:
                print(f"[Backfill] {args.start_date} ~ {args.end_date} 백필 시작")
                rows = collector.backfill_ohlcv(symbol, args.start_date, args.end_date)
                print(f"[Backfill] 수집 건수: {len(rows)}")
                if rows:
                    store.save_many(rows)
                    print(f"[Backfill] 저장 완료: {rows[0]['timestamp']} ~ {rows[-1]['timestamp']}")
        elif args.loop:
            scheduler.run_loop()
        else:
            scheduler.run_once()
    except KeyboardInterrupt:
        print("\n[Scheduler] 중단")
    finally:
        collector.disconnect()


if __name__ == "__main__":
    main()
