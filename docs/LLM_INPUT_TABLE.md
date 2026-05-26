# LLM Input Payload Reference (Table)

이 문서는 KOSPI200 가격 예측을 위해 LLM(GPT/Gemini/Claude)에 전달되는 **입력 JSON payload**를 코드 기준으로 정리한 표입니다.

- 코드 기준:
  - `prediction/pipeline.py` → `PredictionPipeline.get_prediction()` (LLM 판단 호출 포함)
  - `prediction/context_builder.py` → `build_llm_context()` / `build_llm_prompt()`
  - `prediction/llm_judge.py` → `LLMJudge.judge()` (Claude/GPT/Gemini)

> **변경 이력**
> - v2 (2026-02-14): 문서를 현행 `PredictionPipeline` + `context_builder` 구현 기준으로 단순화
> - v3 (2026-02-23): `market.spot_index/basis`, `ensemble`, `adaptive`, `dual_llm` 모드 반영; 출력 스키마 `regime/model_outputs` 추가

---

## 1) Payload 최상위 구조(현행)

```json
{
  "prediction_minutes": 5,
  "transformer": {
    "prob": 0.0,
    "signal": "BUY|SELL|HOLD"
  },
  "ensemble": {
    "prob": 0.0,
    "signal": "BUY|SELL|HOLD",
    "confidence": "HIGH|MEDIUM|LOW",
    "method": "transformer_only|weighted_avg|disagreement_hold|...",
    "agreement": true
  },
  "tft": {
    "prob": 0.0,
    "signal": "BUY|SELL|HOLD"
  },
  "market": {
    "current_price": 0.0,
    "spot_index": null,
    "basis": null
  },
  "market_background": {
    "t2101": {"basis": 0.0, "theoryprice": 0.0, "kospijisu": 0.0},
    "t2301": {"cimpv": 0.0, "pimpv": 0.0, "histimpv": 0.0, "jandatecnt": 0.0},
    "ij_":   {"jisu": 0.0}
  },
  "orderbook": {
    "obi": 0.0,
    "spread": 0.0,
    "level1_ratio": 0.0,
    "bid_slope": 0.0,
    "offer_slope": 0.0,
    "totbidrem": 0.0,
    "totofferrem": 0.0
  },
  "adaptive": {
    "ast_direction": 0.0,
    "ast_dist_pct": 0.0,
    "ast_signal": 0.0,
    "azz_direction": 0.0,
    "cross_trend_agreement": 0.0
  }
}
```

- `tft`는 TFT 확률이 계산된 경우에만 포함됩니다.
- `market.spot_index` / `market.basis`: IJ_ 실시간 지수 수신 시 채워지며, 없으면 `null`.  
  `basis = current_price - spot_index`. `basis` 절대값이 클 때 confidence 하향 또는 HOLD 강제.
- `market_background.ij_`: IJ_ 실시간 스냅샷 (jisu 등). `t2101/t2301`과 달리 실시간 갱신됨.
- `orderbook`은 FH0 기반 feature가 계산된 경우에만 포함됩니다.
- `options`는 스냅샷에서 `build_llm_context()`가 pop한 뒤 `[OPTIONS_SNAPSHOT]` 섹션으로 분리 포함됩니다.
- `adaptive`: `adaptive_indicator.enabled=true`이고 지표가 준비된 경우 포함. PIPELINE_INPUT JSON 내부에 그대로 포함됩니다.
- Adaptive indicator 텍스트 요약은 별도 섹션 `[ADAPTIVE_INDICATORS]`로 포함됩니다.

---

## 2) Field Table(현행)

표 컬럼:
- **Path**: JSON 경로
- **Type**: 자료형
- **Required**: 필수 여부(LLM 입력에 항상 존재하는지)
- **Mode**: 포함 조건
- **Source**: 계산/생성 위치
- **Description**: 의미

### 2.1 Top-level

| Path | Type | Required | Mode | Source | Description |
|---|---:|:---:|---|---|---|
| `prediction_minutes` | int | Y | all | `prediction/pipeline.py` | 예측 horizon(분). |
| `transformer` | object | Y | all | `prediction/pipeline.py` | 수치 예측 결과(Transformer 확률/시그널). |
| `ensemble` | object | Y | all | `prediction/pipeline.py` | 앙상블(또는 최종) 확률/시그널/신뢰도/방법. |
| `tft` | object | N | when available | `prediction/pipeline.py` | TFT 확률/시그널(가능할 때만). |
| `market` | object | Y | all | `prediction/pipeline.py` | 시장 스냅샷(최소 current_price). |
| `market_background` | object | N | when available | `prediction/pipeline.py` | 초기화 시 수집한 t2101/t2301 best-effort 스냅샷. |
| `orderbook` | object | N | when FH0 | `prediction/pipeline.py` | 최신 FH0 feature 스냅샷. |
| `adaptive` | object | N | when adaptive enabled and available | `prediction/pipeline.py` | Adaptive indicator 수치 피처 스냅샷(ADAPT_KEYS 기반, best-effort). |

### 2.2 `transformer`

| Path | Type | Required | Mode | Source | Description |
|---|---:|:---:|---|---|---|
| `transformer.prob` | float | Y | all | `TransformerPredictor.predict()` | 상승 확률(0~1). |
| `transformer.signal` | str | Y | all | `TransformerPredictor.predict()` | `BUY|SELL|HOLD`. |

### 2.3 `ensemble`

| Path | Type | Required | Mode | Source | Description |
|---|---:|:---:|---|---|---|
| `ensemble.prob` | float | Y | all | `prediction/pipeline.py` | 최종(앙상블) 상승 확률(0~1). |
| `ensemble.signal` | str | Y | all | `prediction/pipeline.py` | 최종 시그널 `BUY|SELL|HOLD`. |
| `ensemble.confidence` | str | Y | all | `prediction/predictor.py` | `HIGH|MEDIUM|LOW`. |
| `ensemble.method` | str | Y | all | `prediction/predictor.py` | 사용한 앙상블/결합 방법 식별자. |
| `ensemble.agreement` | bool | Y | all | `prediction/predictor.py` | 내부 모델(Transformer/TFT 등) 합의 여부(best-effort). |

### 2.4 `tft` (optional)

| Path | Type | Required | Mode | Source | Description |
|---|---:|:---:|---|---|---|
| `tft.prob` | float | N | when available | `prediction/predictor.py` | TFT 기반 상승 확률(0~1). |
| `tft.signal` | str | N | when available | `prediction/predictor.py` | `BUY|SELL|HOLD`. |

### 2.5 `market`

| Path | Type | Required | Mode | Source | Description |
|---|---:|:---:|---|---|---|
| `market.current_price` | float | Y | all | `RealTimeTickProcessor.get_current_price()` | 현재 선물 가격(최신). |
| `market.spot_index` | float\|null | N | when IJ_ available | `IJ_` 실시간 지수 구독 | KOSPI200 현물 지수. IJ_ 수신 전 또는 미구독 시 `null`. |
| `market.basis` | float\|null | N | when IJ_ available | `pipeline.py` | `current_price - spot_index`. 절대값 ≥ 2.5pt 시 BUY/SELL → HOLD 강제; ≥ 1.5pt 시 confidence 하향. |

### 2.6 `orderbook` (optional)

| Path | Type | Required | Mode | Source | Description |
|---|---:|:---:|---|---|---|
| `orderbook.obi` | float | N | when FH0 | `calc_orderbook_features` | 주문 불균형 지표(-1~+1). |
| `orderbook.spread` | float | N | when FH0 | `calc_orderbook_features` | 최우선 스프레드. |
| `orderbook.level1_ratio` | float | N | when FH0 | `calc_orderbook_features` | L1 잔량 압력 비율. |
| `orderbook.bid_slope` | float | N | when FH0 | `calc_orderbook_features` | 매수 잔량 기울기(깊이 기반). |
| `orderbook.offer_slope` | float | N | when FH0 | `calc_orderbook_features` | 매도 잔량 기울기(깊이 기반). |
| `orderbook.totbidrem` | float | N | when FH0 | `calc_orderbook_features` | 총 매수 잔량(FH0 또는 depth 합). |
| `orderbook.totofferrem` | float | N | when FH0 | `calc_orderbook_features` | 총 매도 잔량(FH0 또는 depth 합). |

---

## 3) Prompt-level additions

`build_llm_context()`는 snapshot 외에, 최근 `ob_records`(1Hz) 히스토리 요약을 다음 형태로 포함합니다.

- `[ORDERBOOK_SUMMARY_LAST_60S]`
  - `count`
  - `last`: 마지막 record의 핵심 값
  - `mean`: 기간 평균
  - `delta`: `last - first`

또한 옵션 지표가 계산된 경우 다음 섹션이 포함됩니다.

- `[OPTIONS_SNAPSHOT]`
  - `call_count`, `put_count`
  - `pcr_volume`, `pcr_oi`, `call_vol`, `put_vol`, `call_oi`, `put_oi`
  - `iv_skew`, `atm_strike`, `atm_call_iv`, `atm_put_iv`
  - `max_pain_price`, `max_pain_dist_pct`
  - `atm_spread_pct`, `atm_orderbook_imb`, `atm_liquidity_log`

Adaptive indicator 텍스트 요약이 제공되는 경우 다음 섹션이 포함됩니다.

- `[ADAPTIVE_INDICATORS]`
  - `AdaptiveIndicatorManager`가 생성한 자연어 요약 텍스트(통합 요약 + SuperTrend + ZigZag)
  - 포함 조건: `PredictionPipeline.get_prediction()`이 `adaptive_context`를 생성했고,
    `build_llm_context(..., adaptive_context=...)`로 전달된 경우

> **dual_llm 모드 주의**: `dual_llm=true`일 때는 GPT와 Gemini에 **동일한 system/user 프롬프트**를 전달합니다.
> Claude는 dual_llm 모드의 대상이 아닙니다 (단일 모드에서 `preferred_provider=claude`로 사용).
> `dual_llm_primary_provider` (기본 `"gpt"`)의 결과가 최종 `llm_action/risk_level/rationale`에 반영됩니다.

---

## 4) LLM Output Schema (expected)

`build_llm_prompt()`는 아래 JSON 단일 객체로만 응답하도록 강제합니다.

```json
{
  "action": "BUY|SELL|HOLD",
  "risk_level": "LOW|MEDIUM|HIGH",
  "rationale": "...",
  "caution": "..."
}
```

---

## 5) LLM 예측 호출 흐름(코드 트레이스)

이 섹션은 LLM 입력(payload) 생성부터 실제 LLM API 호출 및 결과 파싱까지의 **실행 흐름**을 코드 기준으로 요약합니다.

### 5.1 `get_prediction()` → LLM 호출 트리거 (`prediction/pipeline.py`)

- `PredictionPipeline.get_prediction()`은 현재가/분봉/오더북(FH0)을 기반으로 입력 스냅샷을 구성합니다.
- LLM 입력은 `prediction/context_builder.py`에서 컨텍스트를 생성합니다.
- `use_llm == True`이고 API key/의존성이 준비된 provider가 있으면 LLM 판단을 수행합니다.

### 5.2 `build_llm_context()`/`build_llm_prompt()` → LLM 입력 생성 (`prediction/context_builder.py`)

- 입력: 파이프라인 스냅샷 + 최근 오더북 버퍼(`ob_records`)
- 동작:
  - 스냅샷(JSON)을 포함하고,
  - 최근 60초 오더북 요약(평균/변화량/마지막)을 함께 포함하여 LLM이 판단하기 쉽게 구성합니다.

```json
{
  "prediction_minutes": 5,
  "transformer": { "prob": 0.0, "signal": "HOLD" },
  "ensemble": { "prob": 0.0, "signal": "HOLD", "confidence": "MEDIUM", "method": "transformer_only", "agreement": true },
  "market": { "current_price": 0.0 },
  "orderbook": { "obi": 0.0, "spread": 0.0 }
}
```

### 5.3 `LLMJudge.judge()` (프로바이더별 실제 API 호출, fallback)

`LLMJudge.judge()`는 사용 가능한 provider(Claude/GPT/Gemini)를 순서대로 시도하며, 실패 시 다음 provider로 fallback 합니다.

#### 5.3.1 Claude (`provider == "claude"`)

- `client.messages.create(...)`
  - `system=msgs["system"]`
  - `messages=[{"role":"user","content":msgs["user"]}]`

#### 5.3.2 OpenAI (`provider in ("gpt", "openai")`)

- `client.chat.completions.create(...)`
  - messages에 system/user를 함께 전달

#### 5.3.3 Gemini (그 외 provider 분기)

- 현재 구현은 버전 호환성을 위해 `system + "\n" + user`를 합쳐 `contents`로 전달합니다.
- `client.models.generate_content(model=..., contents=...)` 호출
- 텍스트 추출은 `resp.text` 우선, 없으면 최종적으로 `str(resp)`로 fallback 합니다.

### 5.4 공통 후처리: 응답 텍스트 → JSON 추출 → 정규화된 판단

- LLM 응답 텍스트에서 fenced block 제거/JSON 추출/정규화(`BUY|SELL|HOLD`, `LOW|MEDIUM|HIGH`)를 수행합니다.
- 파싱 실패/빈 응답인 경우에도 안전한 기본값으로 fallback 합니다.
