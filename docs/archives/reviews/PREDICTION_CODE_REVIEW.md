# prediction/ 패키지 코드 리뷰

> **작성일**: 2026-03-02  
> **대상**: `prediction/` 패키지 전체  
> **파일**: `pipeline.py` · `predictor.py` · `features.py` · `model.py` · `tft_model.py` · `llm_judge.py` · `context_builder.py` · `option_features.py` · `weights_selector.py`

---

## 목차

1. [시스템 아키텍처 개요](#1-시스템-아키텍처-개요)
2. [features.py — 특징 추출](#2-featurespy--특징-추출)
3. [predictor.py — 수치 예측](#3-predictorpy--수치-예측)
4. [pipeline.py — 오케스트레이션](#4-pipelinepy--오케스트레이션)
5. [model.py — PriceTransformer](#5-modelpy--pricetransformer)
6. [tft_model.py — TFT](#6-tft_modelpy--tft)
7. [llm_judge.py — LLM 판단](#7-llm_judgepy--llm-판단)
8. [context_builder.py — LLM 컨텍스트](#8-context_builderpy--llm-컨텍스트)
9. [option_features.py — 옵션 피처](#9-option_featurespy--옵션-피처)
10. [weights_selector.py — 가중치 선택](#10-weights_selectorpy--가중치-선택)
11. [우선순위별 개선 로드맵](#11-우선순위별-개선-로드맵)

---

## 1. 시스템 아키텍처 개요

```
실시간 틱 (FC0 / OC0 / FH0)
        │
        ▼
 RealTimeTickProcessor
  ├─ 분봉 OHLCV 버퍼
  ├─ 옵션 스냅샷 (콜/풋)
  └─ 1Hz OB 레코드 deque
        │
        ▼
  PredictionPipeline.get_prediction()
  ├─ 1. _prepare_prediction_inputs()
  │    ├─ AdaptiveIndicatorManager  (SuperTrend + ZigZag)
  │    └─ _compute_regime()
  ├─ 2. _build_option_snapshot_safe()
  ├─ 3. _run_numeric_prediction_and_guardrails()
  │    ├─ build_sequence()          (OB + Candle + Option + Adaptive + Time)
  │    ├─ NumericPredictor.predict()
  │    │    ├─ TransformerPredictor  (PriceTransformer)
  │    │    └─ TFTPredictor          (TemporalFusionTransformer)
  │    │         └─ EnsemblePredictor (weighted avg)
  │    ├─ _apply_option_guardrail()
  │    └─ _apply_basis_guardrail()
  ├─ 4. _maybe_process_feedback()  (adaptive weight 업데이트)
  ├─ 5. _run_llm_judgment()        (Claude / GPT / Gemini)
  └─ 6. 최종 결과 dict 반환
```

**Feature Vector 구성 (per time-step)**

| 블록 | 키 수 | 설명 |
|------|-------|------|
| `OB_KEYS` | 10 | obi, spread, level1_ratio, bid/offer slope, totrem 등 |
| `CD_KEYS` | 5 | ret1, ret3, slope3, vol_accel, range_pct |
| `OPT_KEYS_V1` / `V2` | 7 / 16 | PCR, IV skew, max pain, ATM features |
| `ADAPT_KEYS` | 28 | AST 9 + AZZ 19 + Cross 4 |
| Time (`FUTURE_KNOWN_DIM`) | N | sin/cos 인코딩 |

---

## 2. features.py — 특징 추출

### 현황

`calc_orderbook_features()`는 FH0 스키마 다양성을 잘 처리한다. alias 폴백, L1 수량 보완, 가격 역전 방어, `np.polyfit` 기반 slope 계산까지 구현되어 있다.

`build_sequence()`는 타임스탬프 기반 분봉 매핑과 선형 매핑 폴백을 지원하며, 옵션/적응형 피처를 per-record 방식으로 붙인다.

### 발견된 문제 및 개선점

---

#### ✅ FIXED — slope 계산 이미 개선됨

리뷰 시점 코드에서 `bid_slope` / `offer_slope`는 이미 `np.polyfit`과 L1 정규화로 구현되어 있다.

```python
# 현재 코드 (features.py)
bid_slope = float(np.polyfit(x, bid_arr, 1)[0]) / float(b0)   # b0 = bid_arr[0]
offer_slope = float(np.polyfit(x, offer_arr, 1)[0]) / float(o0)
```

이전 리뷰에서 지적된 `(bid_rems[-1] - bid_rems[0]) / 4.0` 방식은 제거되었다. **조치 완료.**

---

#### ✅ FIXED — OBI 클램핑 이미 적용됨

```python
# 현재 코드 (features.py)
obi = float(np.clip((float(total_bid) - float(total_offer)) / denom, -1.0, 1.0))
```

서킷브레이커 극단값 방지를 위한 `np.clip`이 이미 적용되어 있다. **조치 완료.**

---

#### 🟡 PERF-01 — `build_sequence()` inner loop 성능

`seq_len(60)` 루프 내에서 `datetime.fromtimestamp()` + `pd.DatetimeIndex.searchsorted()` 조합이 per-record로 반복된다. 초당 호출 빈도가 높아지면 CPU 병목이 될 수 있다.

```python
# 현재 코드 — seq_len 루프마다 datetime 변환 + searchsorted
for i, rec in enumerate(tail):
    ts = float(rec.get("_ts_epoch"))
    minute = datetime.fromtimestamp(ts).replace(second=0, microsecond=0)
    if minute in cdf.index:
        cd_arr[start + i] = cdf.loc[minute, cd_keys].values.astype(np.float32)
    else:
        pos = int(cdf.index.searchsorted(minute, side="right")) - 1
        ...
```

```python
# 개선안: pd.merge_asof() 로 vectorize
epochs = np.array([float(r.get("_ts_epoch", 0.0)) for r in tail])
minutes = pd.to_datetime(epochs, unit="s").floor("min")
merged = pd.merge_asof(
    pd.DataFrame({"minute": minutes}),
    cdf[cd_keys].reset_index().rename(columns={"index": "minute"}),
    on="minute",
    direction="backward",
).fillna(0.0)
cd_arr[start:] = merged[cd_keys].values.astype(np.float32)
```

---

#### 🔵 FEAT-01 — OFI 피처 `pipeline.py`에만 존재, `features.py`와 미연동

`_compute_flow_features()`가 `pipeline.py`에 구현되어 있어 `ofi_1s`, `ofi_5s`, `vwap_dev`를 계산한다. 그러나 이 값들이 `build_sequence()`의 `ob_records`에 per-record로 붙지 않고 `last_ob_snapshot`에만 첨부된다. 결과적으로 시계열 60스텝에 OFI가 반영되지 않는다.

```python
# pipeline.py — last_ob_snapshot에만 추가됨
last_ob_snapshot.update(self._compute_flow_features(...))
# build_sequence()로는 전달되지 않음
```

시계열 전체에 OFI를 반영하려면 `ob_records`의 per-record에 첨부하거나, `OB_KEYS`에 추가해야 한다.

---

## 3. predictor.py — 수치 예측

### 현황

`TransformerPredictor`는 가중치 없을 때 `_rule_based()`로 fallback하며, `TFTPredictor`와 `EnsemblePredictor`가 잘 구조화되어 있다. `AdaptiveEnsembleWeightTracker`는 `get_weights()`로 정확도 기반 가중치를 계산한다.

피드백 루프는 **`pipeline.py`의 `_maybe_process_feedback()`이 실제로 `update_adaptive_weights()`를 호출**하도록 연결되어 있다. 예측 시 `_feedback_queue`에 기록하고, N분 후 matured된 예측에 대해 actual label을 계산해 가중치를 업데이트한다.

### 발견된 문제 및 개선점

---

#### ✅ FIXED — `disagreement_hold` 임계값 방향

```python
# 현재 코드 (predictor.py) — 수정된 상태
if prob_diff >= float(self._disagreement_hold_prob_diff_max):
    signal = "HOLD"
    confidence = "LOW"
    method = "disagreement_hold"
```

`>=` 로 올바르게 구현되어 있다. 차이가 클 때(≥ threshold) HOLD를 발동한다. **조치 완료.**

---

#### ✅ FIXED — `EnsemblePredictor._classify()` confidence 파라미터

```python
# 현재 코드 (predictor.py) — confidence 파라미터가 전달됨
signal, confidence = _classify(
    prob=float(ens_prob),
    spread=float(spread),
    buy_threshold=float(self._buy_threshold),
    sell_threshold=float(self._sell_threshold),
    confidence_high_margin=float(self._confidence_high_margin),
    confidence_mid_margin=float(self._confidence_mid_margin),
    confidence_spread_max_for_high=float(self._confidence_spread_max_for_high),
)
```

**조치 완료.**

---

#### 🟠 ARCH-01 — `_rule_based()` spread penalty 스케일링 검토 필요

spread penalty를 `_confidence_spread_max_for_high` 기준으로 정규화하는 방식으로 개선되어 있지만, 이 값의 기본값이 `1.0`이고 KP200 선물 호가 단위(0.05pt)와의 정합성 확인이 필요하다.

```python
# 현재 코드 (predictor.py)
spread_scale = float(getattr(self, "_confidence_spread_max_for_high", 1.0) or 1.0)
spread_penalty = max(0.0, min(0.25, (float(spread) / spread_scale) * 0.25))
```

`confidence_spread_max_for_high = 1.0`이면 spread 0.05pt → penalty 0.0125로 여전히 미미하다. 실제 운영 환경의 평균 스프레드를 기준으로 이 파라미터를 명시적으로 설정해야 의미 있는 페널티가 된다.

---

#### 🟡 ARCH-02 — `TFTPredictor._infer_vars()` 추론 방식 취약성

체크포인트 호환성 확인 시 `weight_grn.fc1.weight`의 shape로 `num_vars`를 추론한다.

```python
# predictor.py
def _infer_vars(prefix: str) -> Optional[int]:
    w = state.get(f"{prefix}.weight_grn.fc1.weight")
    ...
    if in_dim % out_dim != 0:
        return None
    return int(in_dim // out_dim)
```

`context_dim > 0`인 경우 `in_features = num_vars * d_model + context_dim`이 되므로 추론 결과가 틀릴 수 있다. 체크포인트 저장 시 `model_kwargs`를 함께 저장하는 방식이 더 안전하다.

```python
# 개선안: 저장 시
torch.save({
    "state_dict": model.state_dict(),
    "model_kwargs": {
        "past_unknown_dim": past_unknown_dim,
        "future_known_dim": future_known_dim,
        "seq_len": seq_len,
        "horizon": horizon,
    }
}, path)
```

---

#### 🔵 FEAT-02 — 피드백 루프 `tick_processor.get_price_at()` 의존성

`_maybe_process_feedback()`에서 `self.tick_processor.get_price_at(tgt_dt)`를 호출하는데, 이 메서드가 `tick_processor`에 구현되어 있지 않으면 `None`을 반환하고 `current_price`로 대체된다. 롱 구간(5분 이상 경과)에서 `current_price`가 이미 변동했다면 label 오염이 발생한다.

`feedback_snapshot_required=True`로 설정하거나 `get_price_at()` 구현 여부를 시작 시 검증하는 것을 권장한다.

---

## 4. pipeline.py — 오케스트레이션

### 현황

`PredictionPipeline`은 매우 풍부한 기능을 갖추고 있으며, 가드레일(basis, option), Dual LLM, disagreement hold, FC0 stale 경고, adaptive 워밍업, LLM 캐싱, 피드백 루프 등이 구현되어 있다. 전반적으로 이전 리뷰 대비 크게 개선되었다.

### 발견된 문제 및 개선점

---

#### ✅ FIXED — `ad` 변수명 오류

```python
# 현재 코드 (pipeline.py, line 194)
ad = adaptive_indicator if isinstance(adaptive_indicator, dict) else {}
symbol = str(ad.get("symbol") or "KP200 선물")
```

`ad` 변수가 올바르게 정의되어 있다. **조치 완료.**

---

#### ✅ FIXED — `_fo0_schema_logged` 속성명 통일

```python
# 현재 코드 (pipeline.py)
# __init__ (line 274):
self._fo0_schema_logged = False
# add_realtime_tick (line 1161):
if self._fo0_log_schema and (not self._fo0_schema_logged):
    ...
    self._fo0_schema_logged = True
```

`__init__`과 `add_realtime_tick` 모두 `_fo0_schema_logged`로 통일되어 있다. **조치 완료.**

---

#### ✅ FIXED — `_adaptive_last_features` 초기화

```python
# 현재 코드 (pipeline.py, line 300)
self._adaptive_last_features: Dict[str, float] = {}
```

`__init__`에서 올바르게 초기화된다. **조치 완료.**

---

#### ✅ FIXED — `_compute_regime()` 클래스 메서드 분리

```python
# 현재 코드 (pipeline.py, line 733)
def _compute_regime(
    self,
    *,
    adaptive_features: Optional[Dict[str, float]],
    adaptive_supertrend_state: Any,
) -> Optional[str]:
    ...
```

`get_prediction()` 내부 closure가 아닌 독립 메서드로 분리되어 있다. **조치 완료.**

---

#### ✅ FIXED — LLM 호출 캐싱 구현

```python
# 현재 코드 (pipeline.py, line 2021)
if (
    float(self._llm_min_interval_sec) > 0.0
    and float(self._last_llm_call_epoch) > 0.0
    and (now_epoch - float(self._last_llm_call_epoch)) < float(self._llm_min_interval_sec)
    and self._last_llm_result is not None
    and str(self._last_llm_cache_key or "") == str(cache_key or "")
):
    return self._last_llm_result
```

`llm_min_interval_sec`(기본 30초) + cache_key 비교 기반 캐싱이 구현되어 있다. **조치 완료.**

---

#### 🟠 ARCH-03 — `get_prediction()` 단일 책임 원칙 — 부분 개선됨

`_prepare_prediction_inputs()`, `_build_option_snapshot_safe()`, `_run_numeric_prediction_and_guardrails()`, `_run_llm_judgment()` 등으로 단계 분리가 상당히 진행되었다. 다만 `get_prediction()` 자체는 여전히 약 100줄이며, 각 단계의 오류 처리 분기가 중첩되어 있다. 더 단순한 흐름으로 리팩토링하면 가독성이 향상된다.

```python
# 현재 흐름 (단순화)
def get_prediction(self):
    # 1. 가격/시간 가져오기
    # 2. FC0 stale 감지
    # 3. 피드백 처리
    # 4. 예측 입력 준비 (try/except + 여러 RuntimeError 분기)
    # 5. 옵션 스냅샷
    # 6. 수치 예측 + 가드레일
    # 7. LLM 컨텍스트 구성
    # 8. LLM 캐시 확인
    # 9. LLM 호출
    # 10. 결과 dict 조립
```

---

#### 🟡 PERF-02 — `_build_option_snapshot_safe()` 중복 호출 가능성

1Hz 틱 수신 시 `add_realtime_tick()`에서 per-second 옵션 스냅샷을 `_last_opt_features`에 캐시한다. `get_prediction()`에서는 `_build_option_snapshot_safe()`를 직접 호출하므로 별도 계산이 한 번 더 실행된다.

```python
# 개선안: 최신 캐시 재사용
def _build_option_snapshot_safe(self, *, current_price: float) -> Dict[str, Any]:
    if (
        self._last_opt_features
        and self._last_opt_sec_key is not None
        and (time.time() - float(self._last_opt_sec_key)) < 5.0
    ):
        return dict(self._last_opt_features)
    return build_option_snapshot(...)
```

---

#### 🔵 OPS-01 — `prediction_failures` 임계값 알림 미구현

`self._metrics["prediction_failures"]`가 수집되고 있으나, 임계값 초과 시 Telegram 알림으로 내보내는 코드가 없다.

```python
# 권장 추가
FAILURE_ALERT_THRESHOLD = 3
FAILURE_ALERT_WINDOW_SEC = 600  # 10분

if self._consecutive_failures >= FAILURE_ALERT_THRESHOLD:
    await send_telegram(f"⚠️ 예측 연속 실패 {self._consecutive_failures}회")
    self._consecutive_failures = 0
```

---

## 5. model.py — PriceTransformer

### 현황

`PriceTransformer`는 CLS 토큰 풀링 + `recency_weighted` 폴링 두 가지 모드를 지원하며, `norm_first=True` Pre-LN 구조를 사용한다. 체크포인트에서 `x_mean`, `x_std`, `model_kwargs`를 로드하는 backward-compat 코드가 잘 갖춰져 있다.

### 발견된 문제 및 개선점

---

#### 🟡 MODEL-01 — CLS 토큰 초기화

```python
# 현재 코드 (model.py)
self.cls_token = nn.Parameter(torch.zeros(1, 1, int(d_model)))
nn.init.trunc_normal_(self.cls_token, std=0.02)
```

`trunc_normal_(std=0.02)`은 ViT 스타일로 적절하다. 다만 학습 초기에 CLS 토큰이 모든 시퀀스에서 동일한 gradient를 받으므로, **학습 초기 수렴이 느릴 수 있다.** Xavier 또는 kaiming 초기화를 실험해볼 수 있다.

---

#### 🔵 MODEL-02 — Positional Encoding의 주기적 패턴 최적화

Sinusoidal PE는 절대 위치 기반이다. KP200 선물의 1Hz OB 시퀀스는 장 개폐, 점심 시간 유동성 저하 등 **주기적 패턴**이 존재한다. `future_known`에 이미 sin/cos 시간 피처가 포함되어 있으므로 중복 인코딩이 발생할 수 있다. RoPE(Rotary Positional Embedding) 실험을 장기 과제로 고려해볼 만하다.

---

## 6. tft_model.py — TFT

### 현황

VSN + GRN + LSTM encoder + multi-head attention 구조로 TFT 핵심을 구현. binary classification head로 마무리. `weights_only=True`로 안전하게 로드한다.

### 발견된 문제 및 개선점

---

#### 🟡 MODEL-03 — VSN `_infer_vars()` context_dim 간섭

`predictor.py`에서 언급한 것과 동일하게, `weight_grn.fc1.weight`의 input_dim이 `num_vars * d_model + context_dim`일 수 있어 `in_dim % out_dim != 0` 조건이 False를 반환하지 않는 경우 잘못된 `num_vars`가 추론될 수 있다. (현재 `static_dim=0` 기본값이므로 즉각 위험은 낮으나, static context 사용 시 문제가 된다.)

---

#### 🔵 MODEL-04 — LSTM + Transformer 이중 처리 Ablation 필요

```python
# tft_model.py
enc_out, (h, c) = self.encoder_lstm(enc_in)   # LSTM으로 sequential 처리
...
attn_out = self.attn(combined)                  # Attention으로 재처리
```

KP200 60초 시퀀스에서 이 이중 구조가 과적합을 유발할 수 있다. LSTM 단독 vs Attention 단독 Ablation study를 통해 최적 구조를 검증할 것을 권장한다.

---

## 7. llm_judge.py — LLM 판단

### 현황

3개 프로바이더(Claude, GPT, Gemini) 지원, fallback 체인, Claude/Gemini 모델 자동 선택 로직이 잘 구현되어 있다. `parse_json()`은 5단계 fallback(fenced block → direct → raw_decode → balanced-brace → rfind)으로 강건하게 작성되어 있다.

### 발견된 문제 및 개선점

---

#### ✅ FIXED — JSON 파싱 강화

```python
# 현재 코드 (llm_judge.py) — 5단계 fallback
# 1) fenced code block
# 2) json.loads(s)
# 3) JSONDecoder.raw_decode (첫 번째 '{' 위치부터)
# 4) balanced-brace extraction (8192자 제한)
# 5) rfind('}') 기반 단순 슬라이싱
```

Gemini의 JSON 앞뒤 prose 문제가 `raw_decode` + balanced-brace로 대응된다. **조치 완료.**

---

#### 🟠 ARCH-04 — LLM executor timeout 후 zombie 스레드

```python
# pipeline.py
fut = self._llm_executor.submit(self.judge.judge, ...)
judgment = fut.result(timeout=float(self._llm_timeout_sec))
# ...
except FuturesTimeoutError:
    fut.cancel()
    self._reset_llm_executor("timeout")
```

`fut.cancel()`은 이미 실행 중인 스레드에는 효과가 없다. `_reset_llm_executor()`에서 `old.shutdown(wait=False, cancel_futures=True)`를 시도하지만, 실행 중인 HTTP 연결은 실제로 끊기지 않는다. **근본 해결을 위해 provider HTTP 클라이언트에 timeout을 직접 설정해야 한다.**

```python
# 개선안: LLMJudge.__init__에서 클라이언트 레벨 timeout 설정
self._anthropic = anthropic.Anthropic(
    api_key=self.anthropic_key,
    timeout=anthropic.Timeout(
        connect=5.0,
        read=float(llm_timeout_sec),
        write=5.0,
        pool=5.0,
    )
)
```

현재 `_call_provider()`에서 `timeout` kwarg를 지원하는 방식이 구현되어 있으나, SDK 버전에 따라 동작이 다를 수 있다.

---

#### 🟡 COST-01 — LLM 호출 빈도 모니터링 부재

캐싱은 구현되어 있으나 일일 호출 횟수 추적이 없다. `_metrics`에 `llm_calls_today`를 추가하고 임계값 초과 시 경고를 권장한다.

```python
# 권장 추가
self._metrics["llm_calls_today"] = int(self._metrics.get("llm_calls_today") or 0) + 1
if int(self._metrics["llm_calls_today"]) >= 400:
    logger.warning("[LLM_COST] daily LLM calls >= 400, consider increasing llm_min_interval_sec")
```

---

## 8. context_builder.py — LLM 컨텍스트

### 현황

`build_llm_context()`는 PIPELINE_INPUT / ORDERBOOK_SUMMARY_LAST_60S / OPTIONS_SNAPSHOT / ADAPTIVE_INDICATORS 섹션으로 구조화되어 있다. `_summarize_orderbook()`이 mean + delta를 계산하여 LLM에게 추세 방향성을 전달한다.

### 발견된 문제 및 개선점

---

#### 🟡 FEAT-03 — system prompt 단순

```python
# 현재 코드 (context_builder.py)
system = (
    "당신은 파생상품 트레이딩 리스크 분석 전문가입니다. "
    "제공된 입력을 바탕으로 {m}분 관점에서 전략 판단을 내리세요. "
    "반드시 JSON 단일 객체로만 응답하며, 마크다운/설명 텍스트를 절대 포함하지 마세요."
)
```

페르소나 강화(리스크 관리 우선, 과신 금지), 만기일·서킷브레이커 특수 상황 지침, 불확실성 표현 방법 등을 추가하면 응답 품질이 개선된다.

```python
# 개선안
system = (
    "당신은 KP200 파생상품 트레이딩 리스크 관리 전문가입니다. "
    "리스크 관리를 수익 추구보다 우선시하며, 불확실한 상황에서는 HOLD를 선호합니다. "
    "제공된 데이터를 바탕으로 {m}분 관점에서 전략 판단을 내리세요. "
    "만기일, 서킷브레이커 발동 중, 이상 스프레드 상황에서는 특히 보수적으로 판단하세요. "
    "반드시 JSON 단일 객체로만 응답하며, 마크다운/설명 텍스트를 절대 포함하지 마세요."
)
```

---

#### 🔵 FEAT-04 — 최근 예측 히스토리 미포함

현재 컨텍스트는 현재 시점 스냅샷만 전달한다. 지난 3~5회 예측 결과(signal, confidence, 실제 가격 변화)를 포함하면 LLM이 최근 예측 품질과 트렌드 지속성을 판단하는 데 유용하다.

---

#### 🔵 FEAT-05 — 거시 시장 맥락 구조화 미흡

`t2101`, `t2301`, `ij_` 스냅샷이 `set_market_snapshots()`으로 수집되나, LLM 컨텍스트에 구조화된 형태로 전달되지 않는다. 야간선물 대비 괴리, 전일 종가 대비 변동률 등을 요약해 포함하면 판단 품질이 향상된다.

---

## 9. option_features.py — 옵션 피처

### 현황

PCR, IV skew, max pain, ATM microstructure, **GEX(Gamma Exposure)** 까지 구현되어 있다. `calc_gex()`가 실제 코드에 존재하며 `build_option_snapshot()`에서 호출된다.

### 발견된 문제 및 개선점

---

#### ✅ FIXED — GEX 구현 완료

```python
# 현재 코드 (option_features.py)
def calc_gex(calls, puts, underlying_price, ...) -> Dict[str, float]:
    ...
    gex = float(gex_calls) - float(gex_puts)
    return {"gex": gex, "gex_calls": gex_calls, "gex_puts": gex_puts, ...}
```

GEX가 `build_option_snapshot()`에서 계산되어 snap에 포함된다. **조치 완료.**

---

#### ✅ FIXED — IV Skew ATM 허용 오차 추가

```python
# 현재 코드 (option_features.py)
atm_anchor = round(float(upx) * 2.0) / 2.0   # 0.5pt 단위 반올림
tol = 2.5
candidates = [float(s) for s in all_strikes if abs(float(s) - float(atm_anchor)) <= float(tol)]
if not candidates:
    return empty  # ATM 근처 행사가 없으면 중립값 반환
```

0.5pt 반올림 + ±2.5pt 허용 오차가 구현되어 있다. **조치 완료.**

---

#### 🟡 MODEL-05 — GEX `default_days_to_expiry` 고정값

```python
# 현재 코드 (option_features.py)
def calc_gex(..., default_days_to_expiry: float = 7.0, ...):
    T = float(default_days_to_expiry) / 365.0
```

만기까지 남은 일수가 실제로 계산되지 않고 7일로 고정되어 있다. GEX는 잔존 만기(T)에 민감하므로, `weights_selector.py`의 `get_expiry_week_info()`에서 실제 만기일을 가져와 T를 동적으로 계산하는 것을 권장한다.

```python
# 개선안
from utils import get_expiry_week_info
info = get_expiry_week_info(now)
expiry_dt = info.get("expiry_second_thursday")
if expiry_dt:
    days_to_expiry = max(0.5, (expiry_dt - now).total_seconds() / 86400.0)
else:
    days_to_expiry = 7.0
```

---

#### 🔵 MODEL-06 — `calc_gex()` gamma 데이터 없을 때 BS 공식 의존

```python
# 현재 코드 (option_features.py)
def _gamma_from_opt(v):
    g = float(v.get("gamma") or 0.0)
    if g > 0.0:
        return g              # 직접 제공된 gamma 우선
    # gamma 없으면 BS 공식으로 계산 (iv 필요)
    K = float(v.get("strike") or 0.0)
    iv = float(v.get("iv") or ...)
    return _bs_gamma(K=K, iv=iv)
```

eBest API가 gamma를 직접 제공하지 않으면 iv 기반 BS gamma를 계산한다. iv가 없을 때 0.0을 반환하므로 OI가 있는 행사가가 GEX 계산에서 누락된다. iv 데이터 커버리지를 확인하고, 누락 시 fallback 처리가 필요하다.

---

## 10. weights_selector.py — 가중치 선택

### 현황

만기주 동결 로직이 구현되어 있으며, `_DATE_PAT`으로 날짜 버전 가중치 자동 선택, KP200 두 번째 목요일 만기 처리가 잘 되어 있다.

### 발견된 문제 및 개선점

---

#### 🔵 ARCH-05 — 자동 롤백 메커니즘 부재

특정 날짜 모델의 OOS 성능이 급격히 저하될 때 이전 버전으로 자동 롤백하는 메커니즘이 없다. 최근 N일 OOS 정확도를 추적하다가 임계값 이하로 떨어지면 이전 버전으로 롤백하는 로직을 권장한다.

```python
# 개선안 (개념 코드)
def select_weights_with_performance_guard(
    *,
    now: datetime,
    min_accuracy: float = 0.52,
    accuracy_window: int = 50,
    ...
) -> WeightSelection:
    base = select_weights_for_datetime(now=now, ...)
    recent_acc = load_recent_accuracy(window=accuracy_window)
    if recent_acc < min_accuracy:
        logger.warning("OOS accuracy %.3f < %.3f, rolling back", recent_acc, min_accuracy)
        return select_weights_rollback(now=now, ...)
    return base
```

---

#### 🟡 ARCH-06 — 가중치 파일 존재 여부만 확인

```python
# 현재 코드 (weights_selector.py)
transformer_path=str(t_path) if t_path.exists() else None,
```

파일 존재 여부만 확인하고 파일 무결성(sha256, 파일 크기 최솟값 등)은 검증하지 않는다. 불완전하게 저장된 `.pt` 파일이 있으면 predictor 초기화 시 오류가 발생한다.

---

## 11. 우선순위별 개선 로드맵

### Phase 1 — 즉시 권장 (단기 운영 리스크)

| ID | 파일 | 내용 | 심각도 |
|----|------|------|--------|
| ARCH-04 | `pipeline.py` / `llm_judge.py` | LLM HTTP 클라이언트 레벨 timeout 설정 (SDK timeout 파라미터) | 🟠 High |
| ARCH-01 | `predictor.py` | `_rule_based()` spread penalty 스케일 파라미터 명시적 설정 | 🟠 High |
| FEAT-02 | `predictor.py` | `feedback_snapshot_required=True` 또는 `get_price_at()` 구현 여부 확인 | 🟠 High |
| MODEL-05 | `option_features.py` | GEX `days_to_expiry` 실제 만기일 기반 동적 계산 | 🟡 Medium |

### Phase 2 — 단기 개선 (1~2주)

| ID | 파일 | 내용 | 기대 효과 |
|----|------|------|-----------|
| FEAT-01 | `features.py` / `pipeline.py` | OFI 피처를 `ob_records` per-record에 첨부 | 시계열 OFI 반영 |
| FEAT-03 | `context_builder.py` | system prompt 강화 (리스크 우선, 만기일 지침) | LLM 판단 품질 향상 |
| FEAT-04 | `context_builder.py` | 최근 3~5회 예측 히스토리 LLM 컨텍스트 추가 | 트렌드 지속성 판단 |
| COST-01 | `pipeline.py` | 일일 LLM 호출 횟수 카운터 + 임계값 경고 | 비용 폭증 방지 |
| OPS-01 | `pipeline.py` | `prediction_failures` 연속 임계값 Telegram 알림 | 운영 안정성 |

### Phase 3 — 중기 아키텍처 (1개월)

| ID | 파일 | 내용 | 기대 효과 |
|----|------|------|-----------|
| ARCH-02 | `predictor.py` | TFT `model_kwargs` 체크포인트 저장으로 `_infer_vars()` 대체 | 호환성 안전 |
| ARCH-03 | `pipeline.py` | `get_prediction()` 더 단순한 흐름으로 리팩토링 | 유지보수성 |
| PERF-01 | `features.py` | `build_sequence()` `pd.merge_asof()` vectorize | CPU 병목 해소 |
| ARCH-05 | `weights_selector.py` | OOS 정확도 기반 자동 롤백 | 모델 드리프트 방지 |
| FEAT-05 | `context_builder.py` | 거시 시장 맥락(야간선물 괴리, 전일 변동률) 구조화 | LLM 판단 품질 |

### Phase 4 — 장기 연구 과제

| ID | 내용 |
|----|------|
| MODEL-02 | RoPE 또는 Relative PE 실험: KP200 주기적 패턴 최적화 |
| MODEL-04 | TFT LSTM + Attention 이중 처리 Ablation study |
| ARCH-06 | 가중치 파일 무결성 검증 (sha256, 최솟값) |

---

## 총평

이전 리뷰 대비 **대부분의 Critical/High 버그가 수정**되어 있다.

- `ad` 변수명 오류, `_fo0_schema_logged` 속성명 불일치, `_adaptive_last_features` 초기화 누락 모두 해결
- `disagreement_hold` 임계값 방향 수정
- `EnsemblePredictor._classify()` confidence 파라미터 전달
- OBI 클램핑, slope 계산 개선, GEX 구현, IV Skew ATM 허용 오차 추가
- `_compute_regime()` 클래스 메서드 분리, LLM 캐싱 구현, JSON 파싱 강화
- 피드백 루프(`_maybe_process_feedback()`)가 실제로 `update_adaptive_weights()`를 호출

**현재 가장 주의해야 할 영역:**

1. **LLM zombie 스레드** (ARCH-04) — HTTP 클라이언트 레벨 timeout이 설정되지 않으면 `_reset_llm_executor()` 이후에도 이전 스레드가 API 응답을 기다린다
2. **피드백 label 오염** (FEAT-02) — `get_price_at()` 미구현 시 `current_price`로 대체되어 5분 후 실제 가격이 아닌 현재 가격으로 정확도가 계산됨
3. **GEX 만기일 고정값** (MODEL-05) — 만기 직전 GEX 민감도가 크게 다름에도 7일 고정값 사용

Phase 1 항목을 우선 처리하고, 예측 품질 모니터링 인프라(`feedback_evaluations`, `feedback_weight_updates` 메트릭 로깅)를 갖춘 후 Phase 2~3를 진행하는 것을 권장한다.
