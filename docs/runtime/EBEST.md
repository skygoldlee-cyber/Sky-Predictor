# eBest Live Runtime

## 1) ebest_live.py

### 역할

- eBest 로그인/초기화
- realtime 구독 등록(FC0/FH0/JIF/IJ_ + 옵션 OC0/OH0)
  - `JIF`는 스키마상 `tr_key="0"`로 구독합니다.
- `IJ_`는 KP200 현물지수 용도로 `tr_key="101"`로 구독합니다.
- predictor(`PredictionPipeline`)에 tick을 전달하여 예측 루프 구동

### 장 시작 게이트(JIF 기반)

- 장 시작 전(예: 08:45 이전)에는 REST 조회가 정상값을 못 가져오는 경우가 있어,
  런타임은 **realtime 구독은 즉시 등록**하되 **일부 REST 스냅샷 조회는 장 시작 이벤트 이후로 지연**합니다.
- 게이트 오픈 조건
  - `JIF` 수신에서 `jangubun == "5"` & `jstatus == "21"`(장시작) 감지
  - 로그: `[JIF_OPEN] ...`
  - 이후 로그: `[GATE] market open detected; starting t8415/t2101/t2301 snapshots`
  - `JIF`의 "장시작" 이벤트는 **08:45 이후 프로그램을 시작한 경우** 이미 지나간 상태일 수 있어 수신되지 않는 것이 정상일 수 있습니다.
  - 이 경우에도 런타임이 멈추지 않도록 다음 fallback이 함께 동작합니다.
    - realtime tick(FC0/FH0/OC0/OH0/IJ_)이 **1회라도 수신되면** 게이트를 오픈(`market_opened=True`)합니다.
      - 로그: `[GATE_BY_TICK] market_opened=True by realtime tick ...`
    - 추가 안전장치로, `JIF_OPEN`이 끝내 수신되지 않더라도 장중 시간(09:00~15:45, KST)이고 일정 시간 대기 후 게이트를 오픈할 수 있습니다.
      - 로그: `[GATE_FALLBACK] JIF open not received; opening gate by time policy ...`
- 게이트 대상(REST)
  - `t8415`(KP200 price)
  - `t2101`(선물/시장 스냅샷)
  - `t2301`(옵션 체인/IV 스냅샷)
- 게이트 비대상(realtime)
  - `FC0/FH0/JIF/IJ_` 및 옵션 `OC0/OH0` realtime 구독은 장 시작 전에도 등록해도 무방합니다.

realtime 구독은 항상 살아있기 때문에, 장 시작 전에도 tick 수신 카운트/로그를 통해 연결 상태를 확인할 수 있습니다.

### IJ_(KP200 현물지수) 처리

- `IJ_`는 Push 실시간 TR로 수신되며, 수신 즉시 다음을 갱신합니다.
  - GUI `RT:` 라인의 `spot`/`basis` 표시
  - `PredictionPipeline`의 background snapshot(`market_background.ij_`) 및 `market.spot_index/basis`
- 안정성을 위해(수신 누락/장외 등) 예측 직전 **60초 주기**로 `IJ` REST 스냅샷을 best-effort로 재조회하는 fallback이 함께 동작합니다.

### 옵션 호가(OH0) 구독 전략(ATM drift 대응)

- 목적
  - OH0는 옵션 전체를 구독하면 부하가 커서, 런타임에서는 **ATM 근방 미세구조(가드레일)** 목적에 맞게 구독 범위를 제한합니다.
- 정책(현재 구현)
  - `OC0`(옵션 체결)은 ATM 기준 ITM/ATM/OTM을 구독
    - pre-open에서도 `options_subscription.max_otm_calls/max_otm_puts` 캡을 적용할 수 있습니다.
  - `OH0`(옵션 호가)는 **ATM±N** 범위만 구독
    - `options_subscription.preopen_oh0_window`로 N을 조절(기본 2)
  - 변동성 큰 장초 ATM drift를 따라가기 위해 **60초마다** 현재가 기반으로 ATM±2를 재계산하고,
    필요한 OH0 심볼이 생기면 **추가 구독**합니다(초기 구독에 없던 신규 심볼만 추가).
- 로그
  - 신규 OH0 구독이 발생하면 `[OH0_REFRESH] ...` 로그가 출력됩니다.

pre-open/post-open 옵션 구독 구성은 아래 로그로 확인할 수 있습니다.

- `[OPTIONS_CFG] ...`
- `[eBest] pre-open subscription breakdown: ...`
- `[eBest] post-open subscription breakdown: ...`

### 옵션 의미가(당일 extreme) 표시

- `OC0`로 수신되는 옵션 가격들을 대상으로 당일 전체에서의 최고/최저를 추적하며,
  최고가/최저가가 사전 정의 의미가 레벨과 **정확히 일치**할 때 GUI `RT:` 라인에 `SRH`/`SRL` 형태로 표시됩니다.
  - `SRH`: Support/Resistance High (당일 최고가)
  - `SRL`: Support/Resistance Low (당일 최저가)

### 핵심 요소

| 이름 | 종류 | 설명 |
|---|---|---|
| `LiveState` | dataclass | 라이브 루프의 mutable 상태(카운터/평가/last_result 등) |
| `_initialize_api(...)` | async function | 로그인/심볼 조회/실시간 등록/옵션구독 초기화 |
| `_try_evaluate_pending(state, df)` | function | 예측 결과와 실제 분봉을 비교해 평가(지연 평가) |
| `_append_eval_metrics(result, state)` | function | 누적 평가 지표를 result dict에 부착 |
| `_log_model_outputs(result)` | function | 디버깅용 모델 출력 블록 출력 |

`dual_llm` 모드가 활성화되면 `_log_model_outputs()`는 `model_outputs.gpt`, `model_outputs.gemini`를 각각 `[GPT]`, `[GEMINI]` 블록으로 함께 출력합니다.

## 2) ebest_api.py

### 역할

- eBest wrapper REST 요청 헬퍼(로그인/심볼 조회/스냅샷 조회 등)

| 이름 | 종류 | 설명 |
|---|---|---|
| `_get_ebest_keys(config_path)` | function | env/config에서 appkey/appsecretkey 해결 |
| `_ebest_login(api, appkey, appsecretkey)` | async function | 로그인 래퍼 |
| `_ebest_fetch_kp200_symbol(api)` | async function | KP200 선물 심볼 조회 |
| `_ebest_fetch_t2301_open_map(api, yyyymm, gubun)` | async function | 옵션 체인 open map(심볼→open) 생성(best-effort) |

## 3) ebest_callbacks.py

### 역할

- eBest wrapper 이벤트 콜백을 만들고, tick을 JSONL 저장/정규화 후 predictor로 전달

| 이름 | 종류 | 설명 |
|---|---|---|
| `get_gui_tick_stats()` | function | GUI 상태 표시용 tick 통계 스냅샷 |
| `_make_realtime_callback(predictor, state, ticks_fh)` | function | realtime event handler 생성(핵심) |

### JIF(장상태) 수신 처리

- JIF는 장상태 push(`jangubun`, `jstatus`)가 수신될 수 있습니다.
- 로그
  - `jstatus` 변경(처음 관측 포함) 시: `[JIF_STATUS] ...` 로그를 즉시 1회 출력
  - 그 외에도 수신 여부 확인용으로 1Hz rate-limit된 `[JIF] ...` 로그를 출력
- 종료 트리거
  - `jangubun == "5"` 이고 `jstatus == "41"` 수신 시: `[JIF_CLOSE] ...` 로그를 남기고 `stop_requested` 플래그를 세워 라이브 루프가 정상 종료됩니다.

- 장 시작 트리거
  - `jangubun == "5"` 이고 `jstatus == "21"` 수신 시: `[JIF_OPEN] ...` 로그를 남기고
    (내부적으로) 장 시작 이후에만 수행해야 하는 REST 스냅샷 조회 게이트를 해제합니다.

## 4) ebest_options.py

### 역할

- 옵션 월물 심볼 리스트에서 ATM±N/OTM 조건으로 구독 대상을 선정

| 이름 | 종류 | 설명 |
|---|---|---|
| `_filter_option_symbols_by_atm(...)` | function | ATM 기준 ITM/OTM 카운트로 선택 |
| `filter_option_symbols_dynamic_otm_by_open(...)` | function | open interest(또는 open map) 기준으로 OTM 선택을 동적으로 제한 |
