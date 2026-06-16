# Hardcoded Values Inventory

이 문서는 프로젝트 전반에 존재하는 **하드코딩된 값(매직 넘버/문자열/기본값/키 리스트/모델명/시간창/threshold)** 을 파일별로 정리합니다.

- 목적:
  - 값의 의미/사용처를 빠르게 파악
  - runtime/training 간 정합성에 민감한 값을 우선 식별
  - 필요 시 `config.json`으로 이동할 후보를 선별

> 참고: 일부 값은 이미 `constants.py`/`config.json`으로 중앙화되어 있습니다. 이 문서는 “어디에 어떤 값이 박혀있는지”를 인벤토리 형태로 나열합니다.

---

## 1) constants.py (중앙 상수)

### 1.1 데이터/시장 관련
- **MIN_MINUTE_BARS_REQUIRED = 20**
  - 예측 최소 분봉 요구치(런타임 기본)

### 1.2 타임아웃/재시도

- **API_TIMEOUT_SECONDS = 30**
- **TICK_SUBSCRIPTION_WAIT_SECONDS = 2**
- **GRACEFUL_SHUTDOWN_TIMEOUT = 5**
- **API_MAX_RETRIES = 3**
- **API_RETRY_DELAY_SECONDS = 1.0**
- **API_BACKOFF_MULTIPLIER = 2.0**

### 1.3 LLM 모델명/스키마

- **CLAUDE_MODEL = "claude-sonnet-4-20250514"**
- **CLAUDE_FALLBACK_MODELS = (... )**
- **GPT_MODEL = "gpt-4o"**
- **GEMINI_MODEL = "gemini-2.0-flash-exp"**
- **GEMINI_FALLBACK_MODELS = (... )**
- **LLM_OUTPUT_SCHEMA**
  - LLM 응답 JSON contract (action/risk_level/rationale/caution)

### 1.4 런타임 제한/보관 정책

- **MAX_FUTURES_TICKS = 100_000**
- **TICK_DATA_RETENTION_HOURS = 2**
- **MINUTE_DATA_RETENTION_HOURS = 4**

### 1.5 Black-Scholes/기술지표 기본값

- **DEFAULT_RISK_FREE_RATE = 0.03**
- **MIN_TIME_TO_EXPIRY = 1/365**
- **DEFAULT_VOLATILITY = 0.20**
- RSI/MACD/Bollinger 관련 기본 period/std

### 1.6 TFT 차원/호라이즌

- **FUTURE_KNOWN_DIM = 11**
- **HORIZON_SEC = 300**
- **PAST_UNKNOWN_DIM = 47**

> 리스크: `PAST_UNKNOWN_DIM=47`은 **v1 + ADAPT** 조합의 기본 dim에 해당합니다.
> 런타임/학습에서는 `prediction.option_feature_set`(v1/v2) 및 `adaptive_indicator.enabled` 조합에 따라
> 실제 `feature_dim`이 19/28/47/56으로 달라질 수 있습니다.

---

## 2) config.py (dataclass 기본값)

`AppConfig` 계열 dataclass의 필드는 **기본값 자체가 하드코딩**입니다(설정 파일이 없을 때 fallback).

### 2.1 OptionSubscriptionConfig

- `itm: 6`
- `otm_open_min: 0.30`
- `max_otm_calls: 0`
- `max_otm_puts: 0`
- `wait_sec: 2`

추가 런타임 키(설정 파일에서만 사용):

- `preopen_oh0_window: 2`

### 2.2 OptionMinuteOhlcvConfig

- `enabled: False`
- `atm_window: 2`

### 2.3 MinuteLookbackConfig

- `futures: 120`
- `options: 120`

### 2.4 PredictionConfig

- `minutes: 5`
- `use_llm: True`
- `numeric_predictor: "transformer"`
- `option_feature_set: "v1"`
- `min_minute_bars_required: 20`
- `seq_len: 60`
- `fo0_stale_sec: 10`
- `fo0_log_schema: True`
- `buy_threshold: 0.62`
- `sell_threshold: 0.38`
- `confidence_high_margin: 0.15`
- `confidence_mid_margin: 0.08`
- `confidence_spread_max_for_high: 1.0`
- `transformer_weight: 0.5`
- `tft_horizon: 300`
- `disagreement_hold: True`
- `disagreement_hold_prob_diff_max: 0.1`
- `guard_basis_hold_thr: 2.5`
- `guard_basis_downgrade_thr: 1.5`
- `guard_atm_spread_pct_thr: 1.5`
- `guard_atm_liq_log_thr: 2.0`

> 후보: 대부분은 이미 `config.json`로 오버라이드 가능하지만, 문서상 “기본값”으로 의미가 있으므로 유지 가치가 있습니다.

---

## 3) prediction/features.py (피처 키 리스트)

피처 순서/차원을 결정하는 핵심 하드코딩입니다.

- **OB_KEYS**: 7개 (orderbook)
- **CD_KEYS**: 5개 (candle)
- **OPT_KEYS_V1**: 7개 (options snapshot)
- **OPT_KEYS_V2**: v1 + 9개 (option minute micro-movement)
- **ADAPT_KEYS**: 28개 (adaptive_indicator)

> 리스크: 이 리스트 순서가 곧 모델 입력 스키마입니다. 변경 시 dataset 재생성/재학습이 필요합니다.

---

## 4) prediction/pipeline.py (런타임 기본 파라미터)

`PredictionPipeline.__init__` 디폴트값/내부 fallback 값들이 런타임 동작을 좌우합니다.

- `numeric_predictor="transformer"`
- `transformer_weight=0.5`
- `tft_horizon=HORIZON_SEC(=300)`
- `prediction_minutes=5`
- `min_minute_bars_required=20`
- `seq_len=60`
- `fo0_stale_sec=10`
- `llm_timeout_sec=8.0`
- `buy_threshold=0.62`, `sell_threshold=0.38`
- `minute_lookback` fallback(코드 내부 기본값):
  - futures/options default 120
  - 단, 실제 런타임에서는 `config.json`의 `minute_lookback.futures/options`가 전달되면 그 값을 사용
- `option_minute_ohlcv.atm_window` fallback 2
- `option_feature_set="v1"`

또한 adaptive 설정 dict parsing fallback:

- `adaptive.enabled` default `True`
- `adaptive.warmup_bars` fallback(코드 내부 기본값) 45
  - 단, 실제 런타임에서는 `config.json`의 `adaptive_indicator.warmup_bars`가 전달되면 그 값을 사용
- supertrend/zigzag 각 파라미터 기본값(atr_min/max, multiplier_min/max, er/adx period, bb period/std, smooth 등)

---

## 4.1) ebest_live.py (런타임 루프/로그 출력)

- 하트비트 로그(`[HB]`) 출력 주기: **60초**
- 방향 요약 로그(`[DIR_SUMMARY]`): 로거로 **1회**만 기록(중복 출력 방지)
- 모델 출력 블록(`[PIPELINE]`, `[GPT]`, `[GEMINI]`, `[HEURISTIC]`)
  - 긴 문자열 필드(`rationale`, `caution` 등)는 가독성을 위해 여러 줄로 wrap하여 출력
  - provider 블록(`[GPT]`, `[GEMINI]`)에서는 큰 `raw` 필드를 출력에서 제외(로그 간결화)
  - 예측 라운드마다 `eval_dir_hits/total/rate`가 `[PIPELINE]` payload에 포함될 수 있음

---

## 5) prediction/predictor.py (분류 규칙/기본 경로)

### 5.1 기본 가중치 경로

- `_DEFAULT_WEIGHTS = "prediction/weights/transformer_5m.pt"`
- `_DEFAULT_TFT_WEIGHTS = "prediction/weights/tft_5m.pt"`

### 5.2 Rule-based 분류 임계값

- `_classify()` 내부:
  - confidence 결정은 config로 조정 가능합니다:
    - `prediction.confidence_high_margin`
    - `prediction.confidence_mid_margin`
    - `prediction.confidence_spread_max_for_high`

### 5.3 Predictor ctor 기본값

- `feature_dim=PAST_UNKNOWN_DIM`
- `seq_len=60`
- `device="cpu"`
- `buy_threshold=0.62`, `sell_threshold=0.38`

> 후보: confidence 규칙은 성능/해석에 영향을 주므로 `config.json`화할 가치가 큽니다.

---

## 6) tick_processor.py (심볼 파싱/분봉 집계 관련)

### 6.1 ctor 기본값

- `default_futures_minutes` fallback(코드 내부 기본값): 120
- `default_options_minutes` fallback(코드 내부 기본값): 120
  - 단, runtime에서는 `PredictionPipeline`이 `config.json`의 `minute_lookback.futures/options` 값을 전달하면 그 값으로 초기화됨(예: 20)

### 6.2 option minute ohlcv 허용 심볼

- `option_minute_atm_window` 기본 2
- `update_option_minute_allowed_symbols(..., strike_gap=2.5)`
  - strike_gap 기본값 **2.5**

### 6.3 옵션 심볼 규칙(하드코딩)

- option type:
  - `"B"` → call
  - `"C"` → put
- underlying code:
  - `"016"`만 지원(KOSPI200)

> 리스크: 심볼 규칙은 거래소/브로커 포맷 변경 시 깨질 수 있어, 문서화/테스트가 중요합니다.

---

## 7) prediction/data_builder.py (오프라인 생성 기본값)

- `build_dataset(..., seq_len=60, horizon_min=5, tft_horizon_sec=HORIZON_SEC(=300), config_path="config.json")`

또한 config 기반으로:

- `prediction.option_feature_set`에 따라 OPT_KEYS 선택(v1/v2)
- `adaptive_indicator.enabled`에 따라 ADAPT 블록 포함

---

## 8) main.py (CLI 기본값/choices)

- `--config` default `config.json`
- `--log-file` default `logs/prediction.log`
- `--prediction-minutes` choices `[5, 10, 30]`
- `--numeric-predictor` choices `[transformer, tft, combined, ensemble, rule_based]`
- `--preferred-provider` choices `[claude, gpt, gemini, openai, chatgpt]`

> 후보: CLI choices는 제품 정책에 해당. 필요 시 문서/설정과 함께 관리.

---

## 9) train.py / train_tft.py (학습 기본값)

- train.py:
  - epochs default 50
  - batch-size 256
  - lr 1e-3
- train_tft.py:
  - epochs default 80
  - batch-size 128
  - lr 5e-4

또한 config 기반으로 expected_dim을 검증(옵션 feature_set v1/v2 포함).

---

## 10) “config로 이동 후보” (우선순위)

아래는 운영/실험에서 튜닝 필요성이 높고, 코드 변경 없이 조절 가능한 편이 좋은 값들입니다.

- prediction/predictor.py의 confidence rule 파라미터
  - `margin` 경계(0.15, 0.08)
  - `spread` 경계(1.0)
- prediction/pipeline.py의 timeout/버퍼/최소요구치
  - `llm_timeout_sec`, `min_minute_bars_required`, `minute_lookback.*`
- tick_processor.py의 strike grid
  - `strike_gap=2.5` (거래소 상품/단위 변경 가능)

---

## 11) 참고: 생성/갱신 방법

이 문서는 정적 인벤토리입니다. 새 하드코딩이 추가되면 다음을 기준으로 갱신합니다.

- `constants.py`에 새 상수 추가 여부
- `config.py` dataclass default 변경 여부
- `prediction/features.py` KEY 리스트 변경 여부(가장 중요)
- runtime/training 스크립트에 새로운 디폴트/threshold 추가 여부
