# 실시간 신호 생성 및 자동 주문 연동 문서

## 1. 개요

본 문서는 `SkyPredictor` 프로젝트에서 LS증권(구 이베스트) OpenAPI를 활용한 실시간 데이터 수집, 5분봉 집계, 피봇 신호 생성, 재연결, 토큰 갱신, 자동 주문 연동 구조를 정리합니다.

**구현된 기능:**
- REST 5분봉 실시간 수집 (`--live`)
- WebSocket 초 단위 체결 수집 (`--websocket`)
- tick → 5분봉 자동 집계
- WebSocket 끊김 시 자동 재연결/재구독
- 재연결 전 access_token 자동 갱신
- 신호 발생 시 선물/옵션 자동 주문 (`CFOAT00100`)
- DRY-RUN / 실주문 모드 분리

**향후 구현 필요:**
- Telegram 알림 (신호/재연결/토큰갱신/주문)

---

## 2. 구성 요소

| 파일 | 역할 |
|------|------|
| `Devcenter/pivot_bull_data_collector.py` | LS OpenAPI REST/WebSocket/주문 클라이언트 |
| `Devcenter/pivot_bull_signal_generator.py` | 데이터 기반 피봇 신호 생성 및 주문 연동 |
| `Devcenter/duckdb/market_data.duckdb` | 5분봉 OHLCV 저장소 |
| `config.secrets.json` | appkey, appsecret, 계좌정보 등 인증 설정 |

---

## 3. 주요 클래스

### `LSOpenAPICollector`
- OAuth2 토큰 발급/갱신
- `t8465` 5분봉 REST 조회
- `get_valid_token()` 메서드로 유효 토큰 반환

### `LSRealtimeWebSocket`
- `FC9` 선물 실시간 체결 구독
- `TickTo5MinAggregator`로 5분봉 집계
- 끊김 시 지수 백오프(1s → 2s → 4s → ... → 30s) 자동 재연결
- `token_getter`로 재연결 전 토큰 갱신

### `LSOpenAPIOrder`
- `CFOAT00100` 선물/옵션 주문
- `dry_run=True` 기본 (출력만)
- `dry_run=False`로 실제 API 호출

### `DataStore`
- DuckDB 5분봉 UPSERT 저장

---

## 4. CLI 사용법

### 4.1 5분봉 실시간 모드

```powershell
# 5분마다 REST로 최신봉 수집 → 신호 생성
python "Devcenter/pivot_bull_signal_generator.py" --live

# 1회만 실행
python "Devcenter/pivot_bull_signal_generator.py" --live --single-shot

# 포지션 사이즈 계산 포함
python "Devcenter/pivot_bull_signal_generator.py" --live --single-shot --capital 100000000
```

### 4.2 WebSocket 실시간 모드

```powershell
# WebSocket 초 단위 체결 → 5분봉 집계 → 신호 생성
python "Devcenter/pivot_bull_signal_generator.py" --websocket

# 10초 테스트
python "Devcenter/pivot_bull_signal_generator.py" --websocket --ws-duration 10

# 포지션 사이즈 계산 포함
python "Devcenter/pivot_bull_signal_generator.py" --websocket --capital 100000000
```

### 4.3 자동 주문

```powershell
# DRY-RUN (주문 내용만 출력)
python "Devcenter/pivot_bull_signal_generator.py" --live --order --single-shot

# 실제 주문 (모의투자 계좌용 appkey/appsecret 필요)
python "Devcenter/pivot_bull_signal_generator.py" --live --order --live-order \
  --account "모의계좌번호" --password "모의비밀번호"

# WebSocket + 실주문
python "Devcenter/pivot_bull_signal_generator.py" --websocket --order --live-order \
  --account "모의계좌번호" --password "모의비밀번호"
```

**⚠️ 실제 계좌용 appkey/appsecret에서는 `--live-order`를 사용하지 마세요.**

---

## 5. CLI 인자 목록

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--live` | 5분봉 REST 실시간 모드 | `False` |
| `--single-shot` | 1회 실행 후 종료 | `False` |
| `--websocket` | WebSocket 초 단위 실시간 모드 | `False` |
| `--ws-duration` | WebSocket 테스트 실행 시간(초), 0이면 무한 | `0` |
| `--position` | 현재 포지션 (0=없음, 1=롱) | `0` |
| `--capital` | 계좌 크기 (원) | `0.0` |
| `--order` | 신호 발생 시 자동 주문 실행 | `False` |
| `--live-order` | 실제 주문 제출 (없으면 dry-run) | `False` |
| `--account` | 선물옵션 계좌번호 | `None` |
| `--password` | 선물옵션 계좌 비밀번호 | `None` |
| `--symbol` | KOSPI200 선물 단축코드 | `A0169000` |
| `--config` | 인증 정보 JSON 경로 | `config.secrets.json` |
| `--json` | JSON 형식 출력 | `False` |

---

## 6. 실시간 데이터 흐름

### 6.1 REST 5분봉 모드

```text
LS OpenAPI (t8465)
       ↓
최신 5분봉 수집
       ↓
DuckDB 저장 (futures_5min)
       ↓
최근 120일 데이터 로드
       ↓
generate_signal() → 진입/청산/홀드/대기
       ↓
[--order] 주문 실행
```

### 6.2 WebSocket 모드

```text
LS OpenAPI WebSocket (FC9)
       ↓
초 단위 체결 tick 수신
       ↓
TickTo5MinAggregator → 5분봉 OHLCV
       ↓
5분 경계마다 DuckDB 저장
       ↓
최근 120일 데이터 로드
       ↓
generate_signal()
       ↓
[--order] 주문 실행
```

---

## 7. 자동 주문 규칙

| 신호 | 주문 | 매매구분 | 가격 유형 |
|------|------|----------|-----------|
| `ENTER_LONG` | 매수 | `BnsTpCode=2` | `entry_px` 있으면 지정가, 없으면 시장가 |
| `EXIT_LONG` | 매도 | `BnsTpCode=1` | `exit_px` 있으면 지정가, 없으면 시장가 |
| `NO_SIGNAL` / `HOLD` | 미실행 | - | - |

---

## 8. 설정 파일 예시

```json
{
  "ebest": {
    "appkey": "YOUR_APPKEY",
    "appsecret": "YOUR_APPSECRET",
    "mode": "demo",
    "symbol": "A0169000",
    "account": "모의계좌번호",
    "password": "모의비밀번호"
  }
}
```

- `mode`: `"demo"` (모의투자) 또는 `"real"` (실전)
- `account`/`password`: 주문 시 필요 (선택)

---

## 9. WebSocket 재연결/토큰 갱신

- WebSocket 연결 끊김 시 자동 재연결
- 재연결 시 이전 구독 (`FC9`) 자동 복원
- 재연결 전 `get_valid_token()`으로 access_token 만료 확인 및 갱신
- 최대 재연결 횟수: 10회 (기본)
- 지수 백오프: 1s → 2s → 4s → 8s → 16s → 30s

---

## 10. 주의사항

1. **실전 계좌 주문 위험**
   - `--live-order`는 반드시 모의투자 계좌에서만 사용
   - 실전 계좌에서 실행 시 실제 손실 발생 가능

2. **장 운영 시간**
   - KOSPI200 선물 장중: 08:45 ~ 15:45
   - 장 마감 후에는 tick/체결 수신 불가, 주문도 거부됨

3. **API Rate Limit**
   - CFOAT00100: 10 TPS
   - t8465: 제한 없음
   - FC9: 실시간 스트리밍

4. **주문 TR 필드**
   - `CFOAT00100` 기준 `FnoIsuNo`, `BnsTpCode`, `FnoOrdprcPtnCode`, `FnoOrdPrc`, `OrdQty` 사용
   - `AcntNo`/`Pwd`는 필요 시 body에 포함

---

## 11. 향후 구현 필요 항목

### 11.1 Telegram 알림

`pivot_bull_signal_generator.py`에 다음과 같이 TODO로 명기되어 있습니다:

```python
# TODO: Telegram 알림 연동 (향후 구현 필요)
# - 신호 발생 시 알림 (ENTER_LONG / EXIT_LONG / STOP)
# - WebSocket 재연결 시 알림
# - 토큰 갱신 시 알림
# - 주문 체결/실패 시 알림
```

구현 시 `config.secrets.json`의 `telegram` 섹션을 활용할 예정입니다.

### 11.2 모의투자 실주문 테스트

- 모의투자용 `appkey`/`appsecret` 발급 후
- 장 중(08:45~15:45)에 `--live-order --order`로 1계약 매수/매도 테스트

### 11.3 장 운영 시간 필터

- WebSocket 실행 시 장 시작 전/후 자동 대기 또는 종료
- 주문 실행 시 장 운영 시간 확인 후 차단

---

## 12. 디버깅/확인 명령

```powershell
# WebSocket 연결 테스트
python "Devcenter/pivot_bull_signal_generator.py" --websocket --ws-duration 10

# 5분봉 수집 + dry-run 주문
python "Devcenter/pivot_bull_signal_generator.py" --live --single-shot --order

# 데이터 확인
python -c "import duckdb; con=duckdb.connect('Devcenter/duckdb/market_data.duckdb'); print(con.execute('SELECT * FROM futures_5min ORDER BY timestamp DESC LIMIT 5').df())"
```

---

## 13. 참고

- LS OpenAPI REST base: `https://openapi.ls-sec.co.kr:8080`
- LS OpenAPI WebSocket: `wss://openapi.ls-sec.co.kr:9443/websocket`
- 5분봉 차트 TR: `t8465` (`/futureoption/chart`)
- 실시간 체결 TR: `FC9` (선물)
- 주문 TR: `CFOAT00100` (`/futureoption/order`)

---

# 부록 — 통합 운영/엔트리포인트 가이드

> 원본 보고서는 리팩토링 과정에서 `docs/runtime/`에서 제거되었습니다.
> - `main.md` → 부록 A
> - `Market_Open_Subscription_Flow.md` → 부록 B
> - `live_run_troubleshooting.md` → 부록 C

---

## 부록 A. `main.py` 엔트리포인트 및 GUI 가이드

**원본**: `main.md`

### `main.py` 역할
- CLI/GUI 실행 모드 제공
- `config.json` 로드 및 오버라이드 적용
- `PredictionPipeline` 생성 후 라이브(eBest) 또는 리플레이 루프 실행

### LLM Warmup
- `PredictionPipeline` 생성 직후 LLM을 1회 warmup 호출
- 목적: 09:00 이전(예: 08:45) 실행에서 LLM 초기화/연결 문제 조기 발견
- warmup은 짧은 타임아웃으로 실행, 실패해도 런타임 계속 진행
- 결과는 `[LLM_WARMUP] ...` 로그로 출력

### Dual LLM (GPT + Gemini 동시 호출)
- 설정 키: `prediction.dual_llm` (bool), `prediction.dual_llm_primary_provider` (`gpt`|`gemini`)
- CLI: `--dual-llm`, `--dual-llm-primary-provider`
- `dual_llm_primary_provider`는 최종 `llm_action/risk_level/rationale`에 반영되는 provider
- `dual_llm=true`일 때 provider별 결과를 모두 보관/출력, 최종 채택은 primary provider 기준

### Ticks 저장 (JSONL/GZ)
- live 모드에서 `--out-ticks` 지정 시 tick 로그를 `.jsonl`로 저장
- `--compress-ticks`(기본값: True)이면 `.jsonl.gz`로 스트리밍 압축 저장
- 원본 유지: `--no-compress-ticks`
- GUI에서는 tick 저장이 항상 활성화, 출력 경로 자동 생성

### GUI 리플레이
- `Replay` 버튼은 tick 로그(`.jsonl`/`.jsonl.gz`)를 리플레이
- 실행 중 `Pause/Resume` 토글로 동작
- `Pick replay file` 버튼으로 파일 직접 선택 가능
- `Replay speed`와 `Replay max lines` 한 줄로 수평 배치

### GUI 실시간 상태 표시 (`RT:`)
- 1초 주기로 실시간 수신 상태 표시
- 항목: `FC0/FH0/OC0/OH0/JIF/IJ_` 수신 누적 카운트, 선물/옵션 가격, 현물지수(KP200), DIR 누적 성공율, 옵션 의미가 등
- 항목 사이에는 `|` 구분자 포함
- 텔레그램 송신/수신 로그도 GUI 로그 뷰에 출력 (`[TG][SEND] ...`, `[TG][RECV] ...`)

### Heuristic Flip 즉시 예측/알림
- live 모드에서 `prediction.minutes` 경계에서만 예측 수행
- Heuristic action이 `BUY↔SELL`로 flip되면 5분 경계와 무관하게 즉시 LLM 포함 예측 수행
- GUI 메시지 강조 및 Telegram 즉시 전송 트리거
- 중복 전송 방지를 위해 짧은 디바운스 적용

### GUI 설정
- 기본 창 크기: `1100x1200`
- 창 위치: 현재 마우스 커서가 위치한 모니터 중앙
- 설정 파일: 프로젝트 루트의 `config.json` 고정
- 버전: 하단 상태 라인 우측 끝에 `vX.Y.Z` 형태로 표시
- Adaptive indicator 항목은 `config.json` 값을 GUI 입력칸에 로드, 수정 시 `config.json`에 저장
- `Enable Telegram` 기본값 ON

### 핵심 함수/클래스

| 이름 | 설명 |
|------|------|
| `parse_arguments()` | CLI 인자 파싱 |
| `_make_args_from_gui(...)` | GUI 입력값을 args로 변환 |
| `display_startup_info(...)` | 시작 시 설정/모드 요약 로깅 |
| `run_test_mode()` | 테스트 모드 실행 |
| `run_replay_mode_with_predictor(...)` | ticks `.jsonl`/`.jsonl.gz` 리플레이 실행 |
| `run_simple_prediction(...)` | 1회 예측 수행 헬퍼 |
| `main()` | 전체 런타임 엔트리포인트 |

---

## 부록 B. 장 개장 구독 흐름 (Pre-Open → Open)

**원본**: `Market_Open_Subscription_Flow.md`

### 개요
- KP200 전일 종가(pre-open reference) 획득
- ATM 계산 및 옵션 콜/풋 리스트 구축
- 실시간 구독 등록 (JIF/FC0/IJ_/OC0)
- KP200 시가 기반 ATM 재계산
- 개장 후 누락된 OC0 심볼 추가 구독

### 핵심 파일/진입점
- `ebest_live.py::run_ebest_live_mode`: live-mode loop 진입점
- `ebest_live.py::_initialize_api`: pre-open 실시간 등록, `_post_open_init()` 백그라운드 태스크 시작
- `ebest_callbacks.py::_make_realtime_callback`: 실시간 tick 수신 (`JIF`, `FC0`, `FH0`, `OC0`, `OH0`, `IJ_`)
- `ebest_options.py::_filter_option_symbols_by_atm`: ATM 기준 심볼 선택
- `ebest_options.py::filter_option_symbols_dynamic_otm_by_open`: 개장 후 시가 기반 OTM 선택

### Pre-open: KP200 전일 종가 (`t8432`)
- `_initialize_api(...)`에서 `kp200_symbol`, `kp200_prev_close`, 옵션 심볼 리스트를 best-effort로 획득
- `state.kp200_prev_close` 및 `predictor.kp200_prev_close`에 저장
- `kp200_symbol` 누락 시 `_ebest_fetch_kp200_symbol(...)` fallback

### Pre-open: 실시간 등록
- 항상: `FC0`(KP200 선물 체결), `FH0`(선물 호가), `JIF`(장 운영), `IJ_`(현물지수)
- `include_options=True`일 때: `OC0`(옵션 체결), `OH0`(옵션 호가, ATM±N)
- pre-open OC0 등록 시 `state.subscribed_oc0`에 기록

### Market Open Trigger (JIF)
- `JIF` tick에서 `jangubun=="5" and jstatus=="21"`이면 `state.market_opened = True`
- JIF 미수신 시 `_post_open_init()`의 시간 기반 fallback: `[GATE_FALLBACK] ...`
- realtime tick(FC0/FH0/OC0/OH0/IJ_) 1회 수신 시에도 게이트 오픈: `[GATE_BY_TICK] ...`

### After Open: 시가 기반 ATM 재계산 및 누락 OC0 구독
- `_post_open_init()`에서 실행
- `t8415`, `t2101`, `t2301` snapshot 및 `t2301 open_map` 획득
- `filter_option_symbols_dynamic_otm_by_open(...)`로 desired 심볼 계산
- `missing = desired - state.subscribed_oc0`만 추가 구독
- `underlying_open`은 `t2101` snapshot `open` 우선, fallback으로 `predictor.tick_processor.get_current_price()`

### 주요 로그
```text
[OPTIONS_CFG] opt_itm=...->... wait_sec=...->... otm_open_min=... max_otm_calls=... max_otm_puts=... preopen_oh0_window=...
[eBest] subscribe OC0 (pre-open) calls=... puts=... ATM=... prev_close=...
[eBest] subscribe OH0 (pre-open, ATM±N) symbols=... ATM=... prev_close=...
[OPEN_FLOW] include_options=... option_month_info=... opt_itm=... otm_open_min=... max_otm_calls=... max_otm_puts=...
[GATE_FALLBACK] ... (JIF open 미수신 시)
[OPEN_FLOW] open_map sizes: call=... put=...
[eBest] subscribe OC0 (post-open) calls=... puts=... open=... ATM=...
[OPEN][OC0] desired=... missing=... open=... ATM=...
[OPEN][OC0] added_missing=N
```

---

## 부록 C. Live Run Troubleshooting

**원본**: `live_run_troubleshooting.md`

| # | 증상 | 원인 | 대응 |
|---|------|------|------|
| 1 | LLM 응답이 간헐적으로 누락 | 네트워크/서버 지연, API 일시 오류, JSON 파싱 실패 | `constants.py`의 `API_MAX_RETRIES`, `API_RETRY_DELAY_SECONDS`, `API_BACKOFF_MULTIPLIER` 확인; `llm_timeout_sec` 8초 → 12~15초 조정; 듀얼 LLM 모드에서 `model_outputs.gpt`/`gemini`에 `timed_out`/`error` 기록 확인 |
| 2 | Gemini 404 NOT_FOUND | 설정된 모델이 API 버전/SDK에서 미지원 | `constants.py`의 `GEMINI_MODEL`/`GEMINI_FALLBACK_MODELS` 교체; SDK 지원 모델 목록 확인 |
| 3 | 로그 가로 스크롤 발생 | `rationale`/`caution`이 한 줄로 길게 출력 | 모델 출력 블록은 긴 문자열 wrap; LLM provider 블록에서 큰 `raw` 필드 제외 |
| 4 | `market_opened=False` 지속 | 런타임 시작 시점이 장시작 이후라 JIF open 전이 이미 지남 | realtime tick 1회 수신 시 게이트 오픈 (`[GATE_BY_TICK] ...`); 시간 기반 fallback (`[GATE_FALLBACK] ...`) |
| 5 | `[IJ_REFRESH] ... jisu=0.00` 반복 | eBest wrapper/서버 환경에 따라 IJ 응답 필드명 차이 또는 장외/세션/권한 문제 | `ebest_api.py::_ebest_fetch_ij_snapshot()`가 여러 형태를 best-effort로 시도; `IJ_` realtime 수신 카운트 확인, 필요 시 TR key/블록 키 환경 조정 |
