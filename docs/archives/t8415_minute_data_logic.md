# t8415 선물/옵션 분봉 데이터 수집 로직

## 개요

t8415는 eBest OpenAPI에서 선물/옵션의 분봉(또는 일봉) OHLCV 데이터를 가져오는 쿼리입니다. 본 문서는 KOSPI200 선물(KP200)과 KOSPI 지수 선물(KOSPI)의 분봉 데이터를 수집하는 로직을 설명합니다.

## 주요 컴포넌트

### 1. fetch_market_data_service.py

`t8415` 쿼리의 핵심 데이터 수집 로직이 구현된 서비스입니다.

#### 함수 시그니처

```python
async def fetch_market_data(view, query_type, upcode, date, timeframe=1, *, kosdaq_upcode: str)
```

#### 주요 파라미터

- `query_type`: "t8415" 또는 "t8418"
- `upcode`: 종목 코드 (예: KP200 선물 코드, KOSPI 지수 코드)
- `date`: 조회 날짜 (YYYYMMDD 형식)
- `timeframe`: 분봉 단위 (기본값 1분, 전일 데이터인 경우 60분)

#### t8415 요청 파라미터

```python
inputs = {
    f"{query_type}InBlock": {
        "shcode": upcode,           # 종목 코드
        "ncnt": timeframe,          # 분봉 단위
        "qrycnt": 1,                # 조회 건수
        "nday": "",                 # 일수
        "sdate": date,              # 시작 날짜
        "stime": "",                # 시작 시간
        "edate": date,              # 종료 날짜
        "etime": "",                # 종료 시간
        "cts_date": "",             # 연속 조회 날짜
        "cts_time": "",             # 연속 조회 시간
        "comp_yn": "N",             # 압축 여부
    },
}
```

#### 특별 처리 로직

**1. 리플레이 모드 옵션 데이터 스킵**

```python
if query_type == "t8415" and bool(getattr(view, "use_replay", False)):
    is_option = call_prefix 또는 put_prefix로 시작하는 종목인지 확인
    if is_option:
        빈 DataFrame 반환 (리플레이 모드에서는 옵션 분봉 수집 스킵)
```

**2. 장 시작 전 옵션 데이터 스킵**

```python
if query_type == "t8415" and (not bool(getattr(view, "use_replay", False))):
    is_option = 옵션 종목인지 확인
    if is_option and date == today and now_time < market_open_time:
        빈 DataFrame 반환 (장 시작 전에는 옵션 분봉 없음)
```

**3. 전일 데이터 처리**

```python
if date == view.prev_target_day:
    timeframe = 60  # 전일 데이터는 60분봉으로 조회
```

#### 응답 데이터 처리

**1. OutBlock (전일 데이터)**

- `jihigh`: 전일 고가
- `jilow`: 전일 저가
- `jiclose`: 전일 종가
- `disiga`: 당일 시가

**2. OutBlock1 (분봉 데이터)**

- `date`: 날짜 (YYYYMMDD)
- `time`: 시간 (HHMMSS)
- `open`: 시가
- `high`: 고가
- `low`: 저가
- `close`: 종가
- `jdiff_vol`: 거래량
- `openyak`: 미결제약정 (옵션만 해당)
- `openyakcha`: 미결제약정변동 (옵션만 해당)

#### 데이터 변환

```python
# Datetime 생성
df["Datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y%m%d %H%M%S")

# 컬럼명 변경
df = df.rename(columns={
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "jdiff_vol": "Volume",
    "openyak": "OpenInterest",
    "openyakcha": "OIChange",
})

# RangePct 계산 (시가 기준 고가-저가 범위 퍼센트)
if df["Open"].notna().any():
    open_val = df["Open"].dropna().iloc[0]
    if open_val > 0:
        df["RangePct"] = ((df["High"].cummax() - df["Low"].cummin()) / open_val) * 100
```

#### Pivot 포인트 계산

```python
if date == view.today:
    # OutBlock에서 전일 데이터 가져오기
    previous_high = outblock.get("jihigh")
    previous_low = outblock.get("jilow")
    previous_close = outblock.get("jiclose")
    current_open = outblock.get("disiga")
    
    # Pivot 계산
    pp, r1, r2, s1 = view.my_pivot(
        float(previous_high),
        float(previous_low),
        float(previous_close),
        float(current_open),
        upcode,
    )
```

#### 반환 값

```python
return (
    previous_high,   # 전일 고가
    previous_low,    # 전일 저가
    previous_close,  # 전일 종가
    pp,              # Pivot Point
    r1,              # 저항선 1
    r2,              # 저항선 2
    s1,              # 지지선 1
    df,              # OHLCV DataFrame
)
```

---

### 2. basic_market_data_service.py

KOSPI/KP200 선물의 기본 시장 데이터를 수집하는 서비스입니다.

#### KP200 선물 데이터 수집

```python
# 날짜 결정 (리플레이 모드면 전일 데이터 사용)
if bool(getattr(view, "use_replay", False)):
    kp200_date = str(getattr(view, "prev_target_day", date))
else:
    kp200_date = str(date)

# t8415로 KP200 분봉 수집
kp200_fetch_result = await view.fetch_market_data(
    "t8415", 
    view.kp200_symbol, 
    date=kp200_date
)

# 결과 저장
(
    view.kp200_prev_high,
    view.kp200_prev_low,
    view.kp200_prev_close,
    view.kp200_pivot,
    view.kp200_r1,
    view.kp200_r2,
    view.kp200_s1,
    view.df_fetch_kp200,
) = kp200_fetch_result
```

#### 장 시작 전 처리

```python
# 장 시작 전에는 t8415 분봉이 비어 있음
if view.df_fetch_kp200 is not None and not view.df_fetch_kp200.empty:
    kp200_price = view.df_fetch_kp200["Close"].iloc[-1]
else:
    # 전일종가를 기준가로 대체
    _fallback = getattr(view, "kp200_prev_close", None)
    if _fallback is not None:
        kp200_price = float(_fallback)
        view.log_message(
            f"[INFO] 장 시작 전: KP200 분봉 없음 → 전일종가({kp200_price:.2f})로 ATM 계산"
        )
    else:
        kp200_price = 0.0
```

#### KOSPI 지수 데이터 수집

```python
# KOSPI 지수는 t8418 사용 (일봉/분봉)
kospi_fetch_result = await view.fetch_market_data(
    "t8418", 
    view.kospi_symbol, 
    date=date
)

(
    view.kospi_prev_high,
    view.kospi_prev_low,
    view.kospi_prev_close,
    view.kospi_pivot,
    view.kospi_r1,
    view.kospi_r2,
    view.kospi_s1,
    view.df_fetch_kospi,
) = kospi_fetch_result
```

---

### 3. option_aggregator.py

옵션 종목의 분봉 데이터를 집계하는 서비스입니다.

#### t8415 옵션 데이터 수집

```python
result = await self._fetch_market_data("t8415", symbol, date=date)
```

#### 장시작 전 t8415 실패 처리

```python
if result is None:
    # t2301 스냅샷에서 전일 데이터 가져오기
    snap = t2301_snapshot or {}
    prev_close_snap = float(snap.get("price") or snap.get("prev_close") or 0.0)
    prev_high_snap = float(snap.get("high") or snap.get("prev_high") or 0.0)
    prev_low_snap = float(snap.get("low") or snap.get("prev_low") or 0.0)
    
    if prev_close_snap > 0:
        # t2301 데이터 있으면 prev_close만 채운 최소 agg 반환
        return OptionSymbolAggregate(
            symbol=str(symbol),
            label=str(label),
            prev_high=prev_high_snap,
            prev_low=prev_low_snap,
            prev_close=prev_close_snap,
            # ... (나머지 필드는 NaN 또는 0)
        )
```

---

## 데이터 흐름

### 1. KP200 선물 분봉 수집 흐름

```
basic_market_data_service.get_basic_market_data()
    ↓
fetch_market_data("t8415", kp200_symbol, date)
    ↓
eBest API 요청 (t8415InBlock)
    ↓
응답 처리 (t8415OutBlock, t8415OutBlock1)
    ↓
데이터 변환 및 정규화
    ↓
Pivot 포인트 계산
    ↓
반환: (prev_high, prev_low, prev_close, pp, r1, r2, s1, df)
    ↓
view.kp200_* 변수에 저장
```

### 2. KOSPI 지수 분봉 수집 흐름

```
basic_market_data_service.get_basic_market_data()
    ↓
fetch_market_data("t8418", kospi_symbol, date)
    ↓
eBest API 요청 (t8418InBlock)
    ↓
응답 처리
    ↓
데이터 변환
    ↓
반환: (prev_high, prev_low, prev_close, pp, r1, r2, s1, df)
    ↓
view.kospi_* 변수에 저장
```

### 3. 옵션 분봉 수집 흐름

```
option_aggregator.aggregate_symbol()
    ↓
fetch_market_data("t8415", option_symbol, date)
    ↓
[장시작 전/리플레이 모드] 빈 DataFrame 반환
    ↓
[장시작 전 실패] t2301 스냅샷 fallback
    ↓
데이터 변환
    ↓
OptionSymbolAggregate 생성
    ↓
반환
```

---

## 특수 케이스 처리

### 1. 리플레이 모드

- 옵션 종목의 t8415 요청 스킵 (빈 DataFrame 반환)
- KP200 선물은 전일 데이터(`prev_target_day`)로 조회

### 2. 장 시작 전 (당일 08:45 이전)

- 옵션 종목의 t8415 요청 스킵 (빈 DataFrame 반환)
- t2301 스냅샷에서 전일 데이터 fallback

### 3. 전일 데이터 조회

- `timeframe = 60` (60분봉)
- `date == view.prev_target_day`

### 4. 재시도 로직

```python
retry_count = config.getint("SETTINGS", "TR_RETRY_COUNT", fallback=3)
retry_delay_ms = config.getint("SETTINGS", "TR_RETRY_DELAY_MS", fallback=200)

for attempt in range(retry_count):
    response = await api.request(query_type, inputs)
    if outblock1:
        break
    if attempt < (retry_count - 1):
        await asyncio.sleep(retry_delay_ms / 1000.0)
```

---

## 참고 파일

- `services/fetch_market_data_service.py` - t8415 핵심 수집 로직
- `services/basic_market_data_service.py` - KOSPI/KP200 데이터 수집
- `services/option_aggregator.py` - 옵션 데이터 집계
- `controllers/option_fetch_controller.py` - 옵션 데이터 가져오기 컨트롤러
- `docs/eBest_OpenAPI_Schema.md` - eBest OpenAPI 스키마 문서
