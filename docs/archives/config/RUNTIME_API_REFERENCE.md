# Runtime API Reference

이 문서는 프로젝트의 **실시간 동작(runtime)** 경로에서 실제로 호출되는(또는 호출될 수 있는) 핵심 모듈들의 **클래스/함수**를 파일 단위로 요약합니다.

- 대상 범위: `main.py`, `config.py`, `tick_processor.py`, `tick_normalizer.py`, `ebest_*.py`, `prediction/`(파이프라인/피처/컨텍스트/LLM)
- 목적: “어디가 엔트리포인트이고, 어떤 데이터가 어디로 흐르며, 무엇을 수정해야 하는지” 빠르게 찾기

---

## 1) 엔트리포인트 및 실행 흐름

### 1.1 `main.py`

- **역할**
  - CLI/GUI 실행 모드 제공
  - `config.json` 로드 및 오버라이드 적용
  - `PredictionPipeline` 생성 후 라이브(eBest) 또는 리플레이 루프 실행

- **핵심 함수/클래스**

| 이름 | 종류 | 설명 | 주요 I/O |
|---|---|---|---|
| `parse_arguments()` | function | CLI 인자 파싱 | out: `argparse.Namespace` |
| `_make_args_from_gui(...)` | function | GUI 입력값을 `argparse` 유사 구조로 변환 | in: GUI 값, out: args |
| `display_startup_info(config, args, logger)` | function | 시작 시 설정/모드 요약 로깅 | in: `AppConfig`, args |
| `run_test_mode()` | function | 테스트 모드 실행 | out: exit code |
| `run_replay_mode(replay_file)` | function | ticks JSONL 리플레이 실행 | in: 파일 경로 |
| `run_simple_prediction(predictor, args)` | function | predictor에 입력을 넣고 1회 예측 수행(헬퍼) | in: predictor, out: result dict |
| `main()` | function | 전체 런타임 엔트리포인트 | out: exit code |
| `_QtLogEmitter` | class | GUI 로그 전달용 시그널 래퍼 | Qt signal |
| `_QtLogHandler` | class | Python logging → GUI로 전달 | logging.Handler |

---

## 2) 설정 로딩/검증

### 2.1 `config.py`

- **역할**
  - `config.json` + `config.secrets.json` 병합 로드
  - dataclass 기반 설정 구조 제공
  - 런타임에서 안전하게 접근할 수 있도록 검증

- **핵심 함수/클래스**

| 이름 | 종류 | 설명 |
|---|---|---|
| `_deep_merge_dict(base, override)` | function | dict 재귀 머지(시크릿 병합에 사용) |
| `_load_json_file(path)` | function | JSON 파일 안전 로드(best-effort) |
| `_resolve_secrets_path(config_path, secrets_path)` | function | secrets 파일 경로 결정 |
| `AIProviderConfig` | dataclass | Claude/OpenAI/Gemini 키 보관 |
| `EBestConfig` | dataclass | eBest appkey/appsecretkey |
| `OptionSubscriptionConfig` | dataclass | 옵션 구독 범위(ITM/OTM 등) |
| `OptionMinuteOhlcvConfig` | dataclass | 옵션 틱→분봉 OHLCV 집계 설정 |
| `MinuteLookbackConfig` | dataclass | `tick_processor` 분봉 DF 조회 기본 lookback(`minute_lookback`) |
| `PredictionConfig` | dataclass | 예측 관련 파라미터(시퀀스 길이/threshold 등) |
| `AdaptiveIndicatorSettings` | dataclass | adaptive_indicator 사용 여부/파라미터/warmup_bars |
| `AppConfig` | dataclass | 전체 설정 루트 |
| `AppConfig.from_file(config_path)` | classmethod | 파일에서 설정 로드(시크릿 병합 포함) |
| `AppConfig.validate()` | method | 설정 유효성 검사 |
| `load_config(config_path="config.json")` | function | 런타임 편의 로더(실패 시 defaults) |

- **런타임에서 중요한 설정 키(요약)**

| config key | 영향 범위 |
|---|---|
| `prediction_minutes` / `prediction.minutes` | LLM/모델 horizon |
| `seq_len` / `prediction.seq_len` | FO0 버퍼 길이(초) 및 모델 입력 길이 |
| `min_minute_bars_required` | 분봉이 충분히 쌓이기 전 예측 방지 |
| `minute_lookback.futures/options` | `tick_processor.get_*_minute_df()`의 기본 조회 길이 |
| `adaptive_indicator.enabled` | feature_dim 19/47 및 컨텍스트 블록 포함 여부 |
| `adaptive_indicator.warmup_bars` | adaptive 지표 warmup 길이 |
| `option_minute_ohlcv.enabled` | 옵션 틱 분봉 집계 on/off |

---

## 3) 실시간 틱 처리 / 분봉 집계

### 3.1 `tick_processor.py`

- **역할**
  - eBest realtime tick(FC0/OC0/OH0/FH0 일부)를 받아 내부 상태 업데이트
  - 선물 틱을 **분봉 OHLCV**로 집계
  - 옵션 틱을 **최신 스냅샷** 형태로 유지
  - (옵션 설정 시) 특정 옵션 심볼만 분봉 OHLCV로 집계

- **핵심 클래스/메서드**

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
| `get_futures_minute_df(minutes=None)` | method | 선물 분봉 DF 생성(최근 N개). `None`이면 config 주입 기본값 사용 | out: `pd.DataFrame` |
| `get_option_minute_df(symbol, minutes=None)` | method | 옵션 심볼별 분봉 DF(최근 N개). `None`이면 기본값 사용 | out: `pd.DataFrame` |
| `get_current_price()` | method | 최신 선물 가격(best-effort) | out: float |

---

### 3.2 `tick_normalizer.py`

- **역할**
  - eBest wrapper가 주는 다양한 스키마를 **통일된 형태(`tick_norm`)로 정규화**
  - callback에서 predictor에 전달하기 전에 “필수 키를 최대한 채워” 다운스트림 파싱 안정화

| 이름 | 종류 | 설명 |
|---|---|---|
| `normalize_realtime_tick(trcode, symbol, tick)` | function | FC0/OC0/FH0/OH0 공통 키를 표준화한 dict 반환 |

---

## 4) eBest 라이브 모드

### 4.1 `ebest_live.py`

- **역할**
  - eBest 로그인/초기화
  - realtime 구독 등록(FC0/FH0/JIF + 옵션 OC0/OH0)
  - predictor(`PredictionPipeline`)에 tick을 전달하여 예측 루프 구동

| 이름 | 종류 | 설명 |
|---|---|---|
| `LiveState` | dataclass | 라이브 루프의 mutable 상태(카운터/평가/last_result 등) |
| `_initialize_api(...)` | async function | 로그인/심볼 조회/실시간 등록/옵션구독 초기화 |
| `_try_evaluate_pending(state, df)` | function | 예측 결과와 실제 분봉을 비교해 평가(지연 평가) |
| `_append_eval_metrics(result, state)` | function | 누적 평가 지표를 result dict에 부착 |
| `_log_model_outputs(result)` | function | 디버깅용 모델 출력 블록 출력 |

### 4.2 `ebest_api.py`

- **역할**
  - eBest wrapper REST 요청 헬퍼(로그인/심볼 조회/스냅샷 조회 등)

| 이름 | 종류 | 설명 |
|---|---|---|
| `_get_ebest_keys(config_path)` | function | env/config에서 appkey/appsecretkey 해결 |
| `_ebest_login(api, appkey, appsecretkey)` | async function | 로그인 래퍼 |
| `_ebest_fetch_kp200_symbol(api)` | async function | KP200 선물 심볼 조회 |
| `_ebest_fetch_t2301_open_map(api, yyyymm, gubun)` | async function | 옵션 체인 open map(심볼→open) 생성(best-effort) |

### 4.3 `ebest_callbacks.py`

- **역할**
  - eBest wrapper 이벤트 콜백을 만들고, tick을 JSONL 저장/정규화 후 predictor로 전달

| 이름 | 종류 | 설명 |
|---|---|---|
| `get_gui_tick_stats()` | function | GUI 상태 표시용 tick 통계 스냅샷 |
| `_make_realtime_callback(predictor, state, ticks_fh)` | function | realtime event handler 생성(핵심) |

### 4.4 `ebest_options.py`

- **역할**
  - 옵션 월물 심볼 리스트에서 ATM±N/OTM 조건으로 구독 대상을 선정

| 이름 | 종류 | 설명 |
|---|---|---|
| `_filter_option_symbols_by_atm(...)` | function | ATM 기준 ITM/OTM 카운트로 선택 |
| `filter_option_symbols_dynamic_otm_by_open(...)` | function | open interest(또는 open map) 기준으로 OTM 선택을 동적으로 제한 |

---

## 5) 예측 파이프라인(`prediction/`)

### 5.1 `prediction/pipeline.py`

- **역할**
  - 런타임 핵심 오케스트레이터
  - tick을 받아 내부 버퍼(FO0 특징, 옵션 스냅샷, 분봉 DF)를 갱신
  - 수치 예측(Transformer/TFT/앙상블/룰베이스) + LLM 판단을 결합해 최종 결과 생성

| 이름 | 종류 | 설명 |
|---|---|---|
| `PredictionPipeline` | class | 실시간 예측 파이프라인 |
| `PredictionPipeline.add_realtime_tick(payload)` | method | eBest realtime payload를 받아 내부 상태 업데이트 |
| `PredictionPipeline.get_prediction()` | method | 현재까지 누적된 정보로 1회 예측 수행(결과 dict 반환) |
| `PredictionPipeline.set_market_snapshots(t2101, t2301)` | method | 배경 스냅샷 주입(가능할 때) |

### 5.2 `prediction/predictor.py`

- **역할**
  - 수치 예측기(Transformer/TFT) 로딩 및 추론
  - 가중치/차원 불일치 시 룰 기반 fallback

| 이름 | 종류 | 설명 |
|---|---|---|
| `TransformerPredictionResult` | dataclass | 수치 예측 결과(prob/signal/confidence/snapshot) |
| `ModelInput` | dataclass | 모델 입력 컨테이너(sequence/past_known/future_known/meta 등) |
| `TransformerPredictor` | class | Transformer 추론(또는 fallback) |
| `_classify(prob, spread, buy_threshold, sell_threshold)` | function | 확률→BUY/SELL/HOLD + confidence 매핑 |
| `create_numeric_predictor(...)` | function | numeric_predictor 옵션에 따라 predictor 조합 생성 |

### 5.3 `prediction/features.py`

- **역할**
  - FH0 오더북 스냅샷 → 수치 피처(OB)
  - 분봉 OHLCV → 캔들 피처(CD)
  - 옵션 스냅샷 → 옵션 피처(OPT)
  - (옵션) adaptive 지표 피처(ADAPT)
  - 최종적으로 `(seq_len, feature_dim)` 시퀀스를 구성

| 이름 | 종류 | 설명 |
|---|---|---|
| `OB_KEYS`, `CD_KEYS`, `OPT_KEYS`, `ADAPT_KEYS` | constants | 모델 입력 피처 키/순서(정본) |
| `calc_orderbook_features(quote)` | function | FH0-like dict → OB 피처 dict |
| `calc_candle_features(df)` | function | OHLCV DF → CD 피처 DF |
| `build_sequence(..., adaptive_features=None)` | function | OB+CD+OPT(+ADAPT) 시퀀스 생성 (`adaptive_features`가 None이면 19, dict이면 47) |

### 5.4 `prediction/option_features.py`

- **역할**
  - 옵션 tick 스냅샷에서 스칼라 피처 계산

| 이름 | 종류 | 설명 |
|---|---|---|
| `calc_pcr(calls, puts)` | function | PCR(volume/oi) 계산 |
| `calc_iv_skew(calls, puts, underlying_price)` | function | ATM put/call IV skew |
| `calc_max_pain(calls, puts, underlying_price)` | function | max pain strike 및 거리 |
| `calc_atm_microstructure(calls, puts, underlying_price)` | function | OH0 기반 ATM 미세구조 피처 |
| `build_option_snapshot(calls, puts, underlying_price)` | function | 위 피처들을 합쳐 snapshot dict 생성 |

### 5.5 `prediction/context_builder.py`

- **역할**
  - LLM에 전달할 컨텍스트 문자열을 구성
  - snapshot JSON + 최근 오더북 요약 + 옵션 스냅샷 + adaptive 텍스트 블록을 결합

| 이름 | 종류 | 설명 |
|---|---|---|
| `build_llm_context(snapshot, ob_records, adaptive_context)` | function | `[PIPELINE_INPUT]`, `[ORDERBOOK_SUMMARY_LAST_60S]`, `[OPTIONS_SNAPSHOT]`, `[ADAPTIVE_INDICATORS]` 블록 생성 |
| `build_llm_prompt(context, prediction_minutes)` | function | strict JSON 응답을 강제하는 (system,user) 프롬프트 생성 |

### 5.6 `prediction/llm_judge.py`

- **역할**
  - LLM provider(Claude/OpenAI/Gemini) 호출
  - strict JSON 파싱 및 필드 정규화

| 이름 | 종류 | 설명 |
|---|---|---|
| `LLMJudgment` | dataclass | 정규화된 LLM 출력(action/risk/rationale/caution/raw) |
| `LLMJudge` | class | provider 선택/호출/fallback/파싱 전체 담당 |
| `LLMJudge.judge(system, user, timeout=None)` | method | 실제 호출 및 `LLMJudgment` 반환 |

### 5.7 `prediction/time_features.py`

| 이름 | 종류 | 설명 |
|---|---|---|
| `build_time_features(dt)` | function | TFT용 시간 피처 벡터 생성(요일 one-hot 등) |

### 5.8 `prediction/weights_selector.py`

| 이름 | 종류 | 설명 |
|---|---|---|
| `WeightSelection` | dataclass | 선택된 weight 경로와 이유 |
| `select_weights_for_datetime(now, ...)` | function | 만기주 freeze 정책 포함해 사용할 weight 결정 |

---

## 6) 수정 시 빠른 가이드

- **실시간 예측 로직**을 바꾸려면:
  - `prediction/pipeline.py` (오케스트레이션)
  - `prediction/features.py` (피처 정의/순서/차원)
  - `tick_processor.py` (분봉 생성/옵션 스냅샷)

- **LLM 컨텍스트/프롬프트**를 바꾸려면:
  - `prediction/context_builder.py`
  - `LLM_INPUT_TABLE.md`(문서)

- **config 기반 동작**을 바꾸려면:
  - `config.py` + `config.json`
  - `main.py`(CLI/GUI 오버라이드 전달)

---

*마지막 업데이트: runtime 핵심 모듈 기준 요약*
