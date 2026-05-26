# Transformer + LLM 역할 분리 파이프라인

## 개요

5분 가격예측 시스템에서 두 모델의 역할을 명확히 분리한다.

테스트 실행(권장): `python -m pytest -q`

Transformer(피처 레이아웃/모델/학습/weights) 상세는 다음 문서를 참조하세요.
- [TRANSFORMER_GUIDE.md](TRANSFORMER_GUIDE.md)

> 참고(현행 구현): 현재 저장소는 `prediction/PredictionPipeline` 기반으로 동작하며,
> `prediction/predictor.py`는 **Transformer + TFT + Ensemble** 수치 예측기를 포함하며,
> 가중치가 존재하면 torch inference를 사용하고, 가중치가 없거나 torch 미설치 시 rule-based fallback으로 동작합니다.
> LLM은 `prediction/llm_judge.py`에서 **Claude/GPT/Gemini** 를 모두 지원하며, 실패 시 provider fallback 합니다.
> `dual_llm=true` 시 GPT + Gemini를 동시 호출하고, `dual_llm_primary_provider`(기본 `"gpt"`) 결과를 최종 판단에 반영합니다.
> **LLM 실패·타임아웃** 시 `prediction.heuristic_fallback`(기본 `true`)에 따라 adaptive 휴리스틱이 최종 `llm_action`을 보강할 수 있다(`prediction/llm_mixin.py`).
> 라이브 루프(`ebest_live.py`)에서는 휴리스틱 **방향 전환**에 `heuristic_flip_min_interval_sec` 등으로 최소 간격을 둘 수 있다.
> `IJ_` 실시간 지수를 구독하면 `spot_index`/`basis` 계산 및 basis 가드레일이 활성화됩니다.

| 모델 | 역할 | 입력 | 출력 |
|------|------|------|------|
| **NumericPredictor (Transformer/TFT/Ensemble)** | 수치 예측 | 오더북 + 분봉(+옵션+adaptive) 시계열, (TFT는 time features 포함) | 상승 확률 (0~1) + signal/confidence |
| **AdaptiveIndicatorManager** | 시장 국면/구조 판단 | 분봉 OHLCV | heuristic action + regime + ADAPT_KEYS(28) |
| **LLM** | 해석·판단 | 예측 결과 + 시장 컨텍스트 | 전략 판단 텍스트 |

---

## 1) 시스템 구조

```
┌──────────────────────────────────────────────────────────┐
│               데이터 수집 — eBest Live                     │
│                                                          │
│  실시간 TR: FC0 / OC0 / FH0 / OH0 / JIF / IJ_            │
│  (본 저장소는 ebest 래퍼 + ebest_live.py(오케스트레이터) 기반) │
└──────────────────┬────────────────────────────────────────┘
                   │  FC0/OC0/FH0/OH0/JIF/IJ_
                   ▼ 실시간 틱
┌──────────────────────────────────────────────────────────┐
│                  Feature Engineering                      │
│   calc_orderbook_features()  +  calc_candle_features()   │
│   → OBI, spread, slope, ret, vol_accel                   │
│   AdaptiveIndicatorManager → ADAPT_KEYS(28) + regime     │
│   option_features → OPT_KEYS v1(7) / v2(16)             │
└───────────────────────────┬──────────────────────────────┘
                            │ sequence (seq_len × feature_dim)
                            │  feature_dim = len(OB_KEYS)+len(CD_KEYS)+len(OPT_KEYS)+len(ADAPT_KEYS)+time_dim
                            │  (환경/config에 따라 OPT(v1/v2), adaptive on/off, time_dim이 달라짐)
                            ▼
┌──────────────────────────────────────────────────────────┐
│               수치 예측 (Transformer/TFT/Ensemble)         │
│   prob_up + signal/confidence + (옵션) agreement          │
│   + heuristic (adaptive_indicator)                       │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│               basis 가드레일                              │
│   IJ_ 실시간 지수 → spot_index / basis 계산               │
│   basis ≥ 2.5pt → HOLD 강제 / basis ≥ 1.5pt → 하향       │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│               Context Builder                             │
│   예측값 + 시장 상태(spot/basis/regime) → 자연어 컨텍스트  │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│               LLM 판단                                    │
│   단일 모드: Claude / GPT / Gemini (preferred 우선 fallback)│
│   dual_llm: GPT + Gemini 동시 호출                        │
│   → JSON {action / risk_level / rationale / caution}     │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
              {prob/signal/confidence,
               spot_index/basis/regime,
               llm_action/risk_level/rationale,
               model_outputs{heuristic/gpt/gemini},
               transformer_prob/tft_prob/ensemble_method}
```

참고:
- `JIF`는 장운영 상태 모니터링용으로 구독하며, 현행 구현에서는 피처/예측 입력으로 사용하지 않습니다.
- `IJ_`는 KP200 실시간 현물 지수입니다. `add_realtime_tick(ij_={...})`으로 전달하면 `spot_index`/`basis` 계산에 사용됩니다.

---

## 2) 디렉터리 구조

```
project/
│
├── prediction/              # 공통 — Feature·모델·파이프라인
│   ├── features.py          # Feature 추출 (오더북·분봉·OB_KEYS/CD_KEYS/OPT_KEYS/ADAPT_KEYS)
│   ├── model.py             # PriceTransformer (torch inference)
│   ├── tft_model.py         # TemporalFusionTransformer (torch inference)
│   ├── time_features.py     # TFT time features (past_known/future_known)
│   ├── predictor.py         # NumericPredictor (Transformer/TFT/Ensemble + fallback)
│   ├── option_features.py   # 옵션 지표 계산 (PCR / IV Skew / Max Pain)
│   ├── option_flow_features.py  # 옵션 미시 이동 피처 (OPT_KEYS_V2 추가 9개)
│   ├── context_builder.py   # LLM용 컨텍스트 생성
│   ├── llm_judge.py         # LLM 호출 및 판단 파싱
│   └── pipeline.py          # 전체 파이프라인 조합
│
├── adaptive_indicator/      # Adaptive 지표 모듈
│   ├── adaptive_supertrend.py
│   ├── adaptive_zigzag.py
│   ├── indicator_integration.py  # AdaptiveIndicatorManager
│   └── __init__.py
│
├── tick_processor.py        # FC0/OC0 틱 처리 + 분봉 집계
├── ebest_live.py            # eBest 실시간 예측 루프 오케스트레이션
├── ebest_api.py             # eBest 인증/REST 요청 헬퍼
├── ebest_options.py         # 옵션 심볼 필터링/ATM 유틸
├── ebest_callbacks.py       # realtime/message callback + ACK 판별
├── config.py                # config.json 로드/검증
└── main.py                  # 엔트리포인트

※ `auth_manager.py/rest_client.py/websocket_client.py/...` 등 LS OpenAPI용 모듈은 본 저장소에 포함되어 있지 않습니다.
   본 문서는 역할 분리 설계 문서로 유지하되, 현행 구현은 eBest 래퍼를 통해 실시간 틱을 수신합니다.
```

### 2.1 옵션 구독 범위(현행)

옵션 구독은 `config.json`의 `options_subscription`으로 제어됩니다.

- `itm`: ATM 기준 ITM 쪽 개수
- `otm_open_min`: OTM 구독 조건 — 옵션 시가(open) 하한
- `max_otm_calls`, `max_otm_puts`: OTM 콜/풋 구독 상한 (0이면 무제한)
- `wait_sec`: 옵션 구독 전 대기 시간

OTM 구독은 로그인 후 `t2301`로 받은 옵션 체인(open map)을 기준으로 동적으로 결정됩니다.

- `open >= otm_open_min` 인 OTM만 구독
- open map 수신이 실패하면 **OTM은 0으로 축소**되고 ITM/ATM만 구독

---

## 3) `features.py` — Feature 추출

```python
# prediction/features.py (핵심 상수 및 함수 발췌)

OB_KEYS = ["obi", "spread", "level1_ratio",
           "bid_slope", "offer_slope", "totbidrem", "totofferrem"]  # 7개

CD_KEYS = ["ret1", "ret3", "slope3", "vol_accel", "range_pct"]  # 5개

# OPT_KEYS_V1: 기본 옵션 지표 7개
OPT_KEYS_V1 = [
    "pcr_volume", "iv_skew", "max_pain_dist_pct", "atm_iv",
    "atm_spread_pct", "atm_orderbook_imb", "atm_liquidity_log",
]
# OPT_KEYS_V2: V1 + 옵션 분봉 미시이동 피처 9개 (option_flow_features.py)
OPT_KEYS_V2 = OPT_KEYS_V1 + [
    "optm_call_ret", "optm_put_ret", "optm_straddle_ret",
    "optm_call_range_pct", "optm_put_range_pct", "optm_straddle_range_pct",
    "optm_call_vol", "optm_put_vol", "optm_straddle_vol",
]
OPT_KEYS = OPT_KEYS_V1  # 하위 호환 기본값; pipeline은 option_feature_set 설정에 따라 선택

# ADAPT_KEYS: adaptive_indicator 28개 (순서 고정 — 모델 입력 차원에 영향)
ADAPT_KEYS = [
    # Adaptive SuperTrend (9)
    "ast_direction", "ast_dist_pct", "ast_atr_pct", "ast_efficiency_ratio",
    "ast_adx_norm", "ast_mult_norm", "ast_trend_duration", "ast_signal", "ast_band_width_pct",
    # Adaptive ZigZag (19)
    "azz_direction", "azz_wave_size_pct", "azz_support_dist_pct", "azz_res_dist_pct",
    "azz_bars_since_swing", "azz_fib618_dist", "azz_fib382_dist",
    "azz_higher_highs", "azz_lower_lows", "azz_new_swing", "azz_swing_recency",
    "azz_threshold_pct", "azz_structure_up", "azz_structure_down", "azz_structure_ranging",
    # Cross (4)
    "cross_trend_agreement", "cross_at_support", "cross_at_resistance", "cross_breakout_potential",
]

# feature_dim 요약 (constants.py)
# PAST_UNKNOWN_DIM = 47  →  OB(7) + CD(5) + OPT_V1(7) + ADAPT(28)  [adaptive.enabled=true, v1]
# PAST_UNKNOWN_DIM = 56  →  OB(7) + CD(5) + OPT_V2(16) + ADAPT(28) [adaptive.enabled=true, v2]
# adaptive.enabled=false 시 ADAPT 블록 없음 (19 또는 28)


def calc_orderbook_features(quote: dict) -> dict:
    """5단계 호가 dict → 오더북 지표 dict."""
    total_offer = float(quote.get("totofferrem", 0))
    total_bid   = float(quote.get("totbidrem", 0))
    total       = total_offer + total_bid or 1.0
    obi = (total_bid - total_offer) / total
    spread = float(quote.get("offerho", quote.get("offerho1", 0))) \
           - float(quote.get("bidho",   quote.get("bidho1",   0)))
    bid_rems   = [float(quote.get(f"bidrem{i}",   0)) for i in range(1, 6)]
    offer_rems = [float(quote.get(f"offerrem{i}", 0)) for i in range(1, 6)]
    level1_denom = (bid_rems[0] + offer_rems[0]) or 1.0
    level1_ratio = (bid_rems[0] - offer_rems[0]) / level1_denom
    bid_slope   = (bid_rems[-1]   - bid_rems[0])   / 4
    offer_slope = (offer_rems[-1] - offer_rems[0]) / 4
    return {
        "obi": obi, "spread": spread, "level1_ratio": level1_ratio,
        "bid_slope": bid_slope, "offer_slope": offer_slope,
        "totbidrem": total_bid, "totofferrem": total_offer,
    }


def calc_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """분봉 DataFrame → 모멘텀·거래량 지표 DataFrame."""
    out = pd.DataFrame(index=df.index)
    out["ret1"]      = df["Close"].pct_change(1)
    out["ret3"]      = df["Close"].pct_change(3)
    out["slope3"]    = df["Close"].diff(3) / 3
    out["vol_accel"] = df["Volume"] / (df["Volume"].rolling(5).mean() + 1e-9)
    out["range_pct"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)
    return out.fillna(0.0)


def build_sequence(
    ob_records: list[dict],
    candle_df: pd.DataFrame,
    seq_len: int = 60,
    opt_features: dict | None = None,       # OPT_KEYS v1 or v2 (스칼라, 전 행 복제)
    adaptive_features: dict | None = None,  # ADAPT_KEYS 28개 (스칼라, 전 행 복제)
) -> np.ndarray:
    """
    오더북 시계열 + 분봉 지표 + 옵션 스칼라 + adaptive 스칼라를 결합하여
    (seq_len, feature_dim) numpy 배열로 반환.

    adaptive_features=None → ADAPT 블록 없이 반환 (feature_dim = OB+CD+OPT)
    """
    ...
```

---

## 4) `model.py` / `tft_model.py` — Transformer / TFT

> 상태: **구현됨**. Torch가 설치되어 있고 가중치 파일이 존재하면 수치 예측은 torch inference로 수행됩니다.
> Torch 미설치/가중치 없음/로딩 실패 시에는 `prediction/predictor.py`가 안전하게 fallback 경로로 전환합니다.

```python
# prediction/model.py (핵심 발췌)
from config import PAST_UNKNOWN_DIM  # = 47 (v1+adaptive) 또는 56 (v2+adaptive)

class PriceTransformer(nn.Module):
    """
    5분 방향 예측 Transformer.
    입력:  (batch, seq_len, PAST_UNKNOWN_DIM)
    출력:  (batch, 1)  — sigmoid 전 logit
    """
    def __init__(
        self,
        feature_dim: int = PAST_UNKNOWN_DIM,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_drop   = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head    = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_drop(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.head(x)

    def predict_proba(self, x: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x.unsqueeze(0))).item()
```

> **feature_dim 매직 넘버 제거 완료** (v2.2): `model.py`, `tft_model.py`, `predictor.py`, `pipeline.py`, `data_builder.py`에서 모두 `constants.PAST_UNKNOWN_DIM` / `FUTURE_KNOWN_DIM` / `HORIZON_SEC` import로 교체.

---

## 5) `predictor.py` — Transformer 추론 래퍼

> 상태: **구현됨**
> - 구현 파일: `prediction/predictor.py`
> - 입력: `features.build_sequence()`로 만든 `(seq_len, feature_dim)` 시퀀스
> - 출력: `prob/signal/confidence`
>
> Torch Transformer가 준비되면, 동일한 `predict(input=ModelInput)` 인터페이스를 유지한 채 내부만 교체하는 것을 목표로 합니다.

현행 저장소의 `prediction/predictor.py`는 다음 정책으로 동작합니다.
- 가중치(`prediction/weights/transformer_5m.pt`)가 있으면 torch inference
- 없으면 rule-based fallback (`rule_based_weights`, `rule_based_mom_multiplier`로 휴리스틱 튜닝 가능)
- `numeric_predictor: "rule_based"`이면 **`RuleBasedPredictor`** 만 사용(학습 가중치 없이 동일 휴리스틱 코어)

입력 표준(contract):
- predictor는 `ModelInput` 단일 객체를 입력으로 받습니다.
- `ModelInput.sequence`는 `features.build_sequence()` 출력인 `(seq_len, feature_dim)` numpy 배열이며,
  `ModelInput.feature_snapshot`은 최신 오더북 feature(dict), `ModelInput.meta`는 진단/메타 정보(dict)입니다.
- TFT(Temporal Fusion Transformer) 도입을 위해 `ModelInput.future_known`(known-future covariates)와
  `ModelInput.schema_version`이 포함됩니다.

`ModelInput.future_known` (현행 파이프라인 생성 기준):
- shape: `(prediction_minutes*60, 11)`
- 컬럼(순서):
  - `dte_scaled`
  - `dow_onehot_0..6`
  - `tod_sin`, `tod_cos` (정규장 09:00~15:30 기준)
  - `is_expiry_week`

---

## 6) `context_builder.py` — LLM용 컨텍스트 생성

> 상태: **구현됨**
> - 구현 파일: `prediction/context_builder.py`
> - 특징:
>   - 파이프라인 스냅샷(JSON) + 최근 `ob_records`(기본 60초) 요약을 함께 포함
>   - `[OPTIONS_SNAPSHOT]` / `[ADAPTIVE_INDICATORS]` 별도 섹션 지원
>   - LLM 출력 JSON 스키마를 강제(`action/risk_level/rationale/caution`)

현행 구현은 `prediction/context_builder.py`에서 다음 형태로 동작합니다.

```python
def build_llm_context(
    *,
    snapshot: dict,
    ob_records: list[dict] | None = None,
    adaptive_context: str | None = None,   # AdaptiveIndicatorManager 자연어 요약
) -> str:
    """
    [PIPELINE_INPUT] : snapshot JSON (options 블록 제거 후)
    [ORDERBOOK_SUMMARY_LAST_60S] : ob_records count/last/mean/delta
    [OPTIONS_SNAPSHOT] : snapshot.pop("options") 결과
    [ADAPTIVE_INDICATORS] : adaptive_context 있을 때만 포함
    """
    ...


def build_llm_prompt(*, context: str, prediction_minutes: int) -> tuple[str, str]:
    """→ (system_str, user_str)"""
    ...
```

- `snapshot`은 `PredictionPipeline.get_prediction()` 내부에서 구성되며, 예측 horizon/transformer 결과/현재가/`spot_index`/`basis` 및(가능하면) 오더북 스냅샷을 포함합니다.
- `ob_records`는 `PredictionPipeline.add_realtime_tick()`가 FH0를 1Hz로 버퍼링한 히스토리입니다.
- `adaptive_context`는 `AdaptiveIndicatorManager.get_llm_context()`가 반환한 자연어 요약 문자열입니다.

---

## 7) `llm_judge.py` — LLM 호출 및 판단 파싱

> 상태: **구현됨(멀티 provider + dual_llm 모드)**
>
> - 구현 파일: `prediction/llm_judge.py`
> - 지원 provider:
>   - `claude` (anthropic)
>   - `gpt` (openai)
>   - `gemini` (google-genai)
> - 동작:
>   - 사용 가능한 provider를 순서대로 시도
>   - 실패 시 다음 provider로 fallback
>   - `preferred_provider`로 우선순위 지정 가능
>   - JSON 파싱/정규화(코드펜스 제거, dict 추출, enum 정규화) 포함
>   - model-level fallback: Claude → `CLAUDE_FALLBACK_MODELS`, Gemini → `GEMINI_FALLBACK_MODELS`

**dual_llm 모드** (`prediction.dual_llm=true`):
- `PredictionPipeline._judge_provider_with_timeout()`을 통해 GPT + Gemini를 **순차 호출**합니다.
- 결과는 `model_outputs["gpt"]` / `model_outputs["gemini"]`에 각각 저장됩니다.
- `dual_llm_primary_provider`(기본 `"gpt"`, 현재 config에서 `"gemini"` 사용)의 결과가 `llm_action/risk_level/rationale/caution`에 반영됩니다.
- ⚠️ **주의**: `_judge_provider_with_timeout()` 내부의 재시도 루프가 `API_MAX_RETRIES` / `API_RETRY_DELAY_SECONDS` / `API_BACKOFF_MULTIPLIER`를 `constants`에서 import해야 동작합니다. 현재 미임포트로 재시도 로직이 무력화됨 (P0 수정 필요).

현행 코드는 위의 단일-provider 예시보다 확장되어 있으므로, 최신 구현은 `prediction/llm_judge.py`를 참조합니다.

---

## 8) `pipeline.py` — 전체 파이프라인 조합

> 상태: **구현됨**
> - 구현 파일: `prediction/pipeline.py`
> - 입력 데이터:
>   - FC0/OC0: `tick_processor.py`가 분봉/옵션 스냅샷 구성
>   - FH0: `PredictionPipeline.add_realtime_tick()`에서 1Hz로 다운샘플링하여 `ob_records` 유지
>   - IJ_: `add_realtime_tick(ij_={...})`로 실시간 지수 갱신 → `spot_index`/`basis` 계산
> - 예측:
>   - `features.build_sequence(ob_records, candle_features, seq_len, opt_features, adaptive_features)` 생성
>   - NumericPredictor(Transformer/TFT/Ensemble + fallback)로 `prob/signal/confidence` 생성
>   - `AdaptiveIndicatorManager`로 `heuristic action` + `regime` + `ADAPT_KEYS(28)` 산출
>   - `basis` 가드레일: `|basis| ≥ 2.5pt` → HOLD 강제 / `≥ 1.5pt` → confidence 하향
> - 판단:
>   - `context_builder`로 프롬프트 구성 후 `LLMJudge`로 전략 판단(action/risk)
>   - `dual_llm=true` 시 GPT + Gemini 동시 호출 → `model_outputs`에 각각 저장
> - 출력(요약):
>   - `prob`, `signal`, `confidence`, `transformer_prob`, `tft_prob`, `ensemble_method`, `model_agreement`
>   - `spot_index`, `basis`, `regime`
>   - `llm_action`, `llm_provider`, `llm_timed_out`, `risk_level`, `rationale`, `caution`
>   - `model_outputs` (heuristic / gpt / gemini — 활성화된 구성 요소만 포함)
>   - `consensus`, `ob_records_len`, `fo0_age_sec`, `options`

현행 코드는 위의 독립 실행 예시보다 `main.py`/`ebest_live.py`와 결합된 형태로 동작하므로, 최신 구현은 `prediction/pipeline.py`를 참조합니다.

---

## 8.1) 수치 예측기 구성 — Transformer / TFT / Ensemble / Adaptive

`prediction/predictor.py`의 `create_numeric_predictor()`는 설정에 따라 다음 중 하나를 선택합니다.

- `transformer`: `TransformerPredictor` (PriceTransformer)
- `tft`: `TFTPredictor` (TemporalFusionTransformer)
- `ensemble`: `EnsemblePredictor` (Transformer + TFT 결합, 현행 config 기본값)

Ensemble 모드에서는 다음이 함께 산출될 수 있습니다.

- `transformer_prob`
- `tft_prob`
- `ensemble_prob` (= `ensemble_method`에 따라 가중합 또는 선택)
- `agreement` (두 모델의 방향성 합의 여부)
- `disagreement_hold=true` 시: 불일치하면 `signal="HOLD"`, `ensemble_method="disagreement_hold"`

**Adaptive Indicator** (`adaptive_indicator.enabled=true`):

`AdaptiveIndicatorManager`는 수치 예측기와 별도로 동작하며 다음을 제공합니다.
- `heuristic`: AdaptiveSuperTrend + AdaptiveZigZag 기반 action (BUY/SELL/HOLD)
- `regime`: 시장 국면 레이블 (`STRONG_UP` / `WEAK_UP` / `RANGE` / `WEAK_DOWN` / `STRONG_DOWN`)
- `ADAPT_KEYS` 28개: Transformer/TFT 입력 시퀀스에 편입 (전 행 복제)
- `adaptive_context`: LLM 컨텍스트의 `[ADAPTIVE_INDICATORS]` 섹션으로 제공

---

## 9) 실사용 예시

현행 실행은 `main.py`가 `PredictionPipeline`을 생성하고,
`ebest_live.py`가 FC0/OC0/FH0/OH0 실시간 틱을 `PredictionPipeline.add_realtime_tick()`으로 전달한 뒤,
주기적으로 `PredictionPipeline.get_prediction()`을 호출하는 구조입니다.

---

## 10) 두 모델 불일치(consensus=False) 처리 원칙

| Transformer | LLM | 권장 처리 |
|-------------|-----|-----------|
| BUY | HOLD | 포지션 축소 or HOLD 유지 |
| BUY | SELL | **진입 금지** — 불확실성 높음 |
| SELL | HOLD | 청산 보류, 모니터링 강화 |
| SELL | BUY | **진입 금지** — 불확실성 높음 |
| HOLD | BUY/SELL | LLM 단독 판단 — 참고만 |

> **원칙**: 두 모델이 반대 신호를 낼 경우 항상 보수적으로 HOLD.
> Transformer 신뢰도가 LOW이거나 LLM risk_level이 HIGH이면 동일하게 HOLD.

---

## 11) 환경 설정

```bash
# 기본 의존성
pip install -r requirements.txt

# 선택 의존성 (LLM)
pip install anthropic openai google-genai
```

환경변수로 키를 주입할 수 있습니다.

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `EBEST_APPKEY`
- `EBEST_APPSECRET`

PowerShell 예시(현재 터미널 세션에서만):

```powershell
$env:OPENAI_API_KEY="..."
$env:GEMINI_API_KEY="..."
$env:ANTHROPIC_API_KEY="..."
$env:EBEST_APPKEY="..."
$env:EBEST_APPSECRET="..."
python main.py
```

PowerShell 예시(사용자 환경변수로 영구 저장; 새 터미널부터 반영):

```powershell
setx OPENAI_API_KEY "..."
setx GEMINI_API_KEY "..."
setx ANTHROPIC_API_KEY "..."
setx EBEST_APPKEY "..."
setx EBEST_APPSECRET "..."
```

---

## 12) 데이터처리 흐름 — eBest Live (현행 구현)

> 본 저장소의 실시간 데이터 처리는 LS OpenAPI REST/WS 예시가 아니라,
> `ebest_live.py` + eBest 래퍼(OpenApi) 기반으로 동작합니다.

### 12.1 입력 데이터

- `FC0`: 선물 체결 틱
- `OC0`: 옵션 체결 틱
- `FH0`: 선물 호가/오더북 틱
- `OH0`: 옵션현재가/호가 틱
- `JIF`: 실시간 장운영 정보(장구분/장상태; 모니터링용, 예측 입력 미사용)
- `IJ_`: KP200 실시간 현물 지수 → `spot_index`/`basis` 계산에 사용

참고:
- 문서/로그에서 `fo0_*`라는 키 이름이 남아있을 수 있으나, 현행 구현에서 오더북 버퍼링 입력은 `FH0`입니다.
- `PredictionPipeline.get_prediction()`은 best-effort로 `t2101/t2301/ij_` 초기 스냅샷을 `market_background`로 함께 포함할 수 있습니다.

### 12.2 처리/예측 흐름(요약)

```
FC0/OC0/FH0/OH0/JIF/IJ_ 실시간 수신
  ├─ tick_normalizer.normalize_realtime_tick() → tick_norm 생성
  ├─ tick_processor.process_tick()  (FC0/OC0 분봉/스냅샷 업데이트)
  ├─ PredictionPipeline.add_realtime_tick() (FH0 오더북 1Hz 버퍼링)
  ├─ PredictionPipeline.add_realtime_tick(ij_={...}) (IJ_ 지수 갱신)
  └─ 주기적/조건부로 PredictionPipeline.get_prediction() 호출
        ├─ 분봉 데이터 충분성 체크 (min_minute_bars_required)
        ├─ features.calc_* + build_sequence(seq_len, opt_features, adaptive_features)
        ├─ predictor(torch if weights else rule-based) → prob/signal/confidence
        ├─ AdaptiveIndicatorManager → heuristic + regime + adaptive_features
        ├─ basis 가드레일 (spot_index → basis 계산 → 이상 시 HOLD/하향)
        ├─ context_builder → system/user prompt (IJ_/basis/regime/adaptive_context 포함)
        └─ llm_judge(claude/gpt/gemini) → action/risk (timeout + heuristic_fallback 분기)
             or dual_llm: gpt + gemini 동시 → model_outputs[gpt/gemini]
```

### 12.3 운영 파라미터(현행)

- `seq_len`: 오더북 버퍼 길이(기본 60; 1Hz 기준 60초)
- `min_minute_bars_required`: 예측 최소 분봉 수(기본 20; config에서 21)
- `fo0_stale_sec`: FH0 수신 지연 경고 임계값 (명칭은 과거 호환 용어)
- `fo0_log_schema`: FH0 스키마 키 로깅(필드명 확인용; 명칭은 과거 호환 용어)
- `llm_timeout_sec`: LLM 호출 타임아웃(초). 타임아웃/오류 시 수치 신호 + `heuristic_fallback` 정책 적용
- `heuristic_fallback`: LLM 실패 시 adaptive 휴리스틱으로 `llm_action` 보강 여부(기본 `true`)
- `heuristic_flip_min_interval_sec` / `heuristic_flip_include_hold_transition`: 라이브 휴리스틱 방향 전환 억제(미설정 시 기본 간격 공식)
- `rule_based_weights` / `rule_based_mom_multiplier`: 휴리스틱 확률 가중·모멘텀 배율
- `preferred_provider`: LLM provider 우선순위 지정(`claude|gpt|gemini`)
- `dual_llm`: GPT + Gemini 동시 호출 모드 (기본 `false`; 현재 config `true`)
- `dual_llm_primary_provider`: dual_llm에서 최종 판단으로 사용할 provider (기본 `"gpt"`; 현재 config `"gemini"`)
- `option_feature_set`: 옵션 피처 버전 (`v1`=7개 / `v2`=16개; 현재 config `"v2"`)
