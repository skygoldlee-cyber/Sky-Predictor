# Tick Processing & Minute Bars

## 1) tick_processor.py

### 역할

- eBest realtime tick(FC0/OC0/OH0/FH0 일부)를 받아 내부 상태 업데이트
- 선물 틱을 **분봉 OHLCV**로 집계
- 옵션 틱을 **최신 스냅샷** 형태로 유지
- (옵션 설정 시) 특정 옵션 심볼만 분봉 OHLCV로 집계

### 핵심 클래스/메서드

| 이름 | 종류 | 설명 | 주요 I/O |
|---|---|---|---|
| `RealTimeTickProcessor` | class | 런타임 tick → 상태/분봉 생성의 중심 | stateful |
| `RealTimeTickProcessor.__init__(default_futures_minutes, default_options_minutes)` | method | 분봉 조회 기본값을 주입 받음(`config.minute_lookback`) | in: ints |
| `configure_option_minute_ohlcv(enabled, atm_window)` | method | 옵션 분봉 집계 on/off 및 ATM 윈도우 설정 | in: bool/int |
| `update_option_minute_allowed_symbols(underlying_price, strike_gap)` | method | ATM±N 범위로 옵션 분봉 집계 대상 심볼 갱신 | in: price |
| `process_tick(tick_data)` | method | `trcode`에 따라 내부 처리 분기 | in: dict |
| `process_futures_tick(tick_data)` | method | FC0 처리 및 분봉 버퍼 축적 | in: dict |
| `process_option_tick(tick_data)` | method | OC0 처리(스냅샷) + (옵션분봉 enabled 시) 분봉 집계 | in: dict |
| `process_option_quote_tick(tick_data)` | method | OH0(옵션호가) 스냅샷 반영(미세구조 피처에 사용) | in: dict |
| `get_futures_minute_df(minutes=None)` | method | 선물 분봉 DF 생성(최근 N개). `None`이면 기본값 사용 | out: `pd.DataFrame` |
| `get_option_minute_df(symbol, minutes=None)` | method | 옵션 심볼별 분봉 DF(최근 N개). `None`이면 기본값 사용 | out: `pd.DataFrame` |
| `get_current_price()` | method | 최신 선물 가격(best-effort) | out: float |

## 2) tick_normalizer.py

### 역할

- eBest wrapper가 주는 다양한 스키마를 **통일된 형태(`tick_norm`)로 정규화**
- callback에서 predictor에 전달하기 전에 “필수 키를 최대한 채워” 다운스트림 파싱 안정화

### 핵심 함수

| 이름 | 종류 | 설명 |
|---|---|---|
| `normalize_realtime_tick(trcode, symbol, tick)` | function | FC0/OC0/FH0/OH0 공통 키를 표준화한 dict 반환 |
