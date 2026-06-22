# Transformer 예측 알고리즘 상세 리뷰 및 개선 제안

> 작성일: 2026-03-02  
> 대상 코드: `prediction/` 패키지 전체 (pipeline.py, predictor.py, features.py, model.py, tft_model.py, llm_judge.py, context_builder.py, option_features.py, weights_selector.py)

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
  ├─ 1. AdaptiveIndicatorManager  (SuperTrend + ZigZag)
  ├─ 2. build_sequence()          (OB + Candle + Option + Adaptive + Time)
  ├─ 3. NumericPredictor.predict()
  │    ├─ TransformerPredictor  (PriceTransformer)
  │    └─ TFTPredictor          (TemporalFusionTransformer)
  │         └─ EnsemblePredictor (weighted avg)
  ├─ 4. 가드레일 (Basis / Option)
  ├─ 5. LLMJudge.judge()         (Claude / GPT / Gemini)
  └─ 6. 최종 결과 dict 반환
```

**Feature Vector 구성 (per time-step)**

| 블록 | 키 수 | 설명 |
|------|-------|------|
| OB_KEYS | 10 | OBI, spread, level1_ratio, slopes, totrem |
| CD_KEYS | 5 | ret1, ret3, slope3, vol_accel, range_pct |
| OPT_KEYS_V1 / V2 | 7 / 16 | PCR, IV skew, max pain, ATM features |
| ADAPT_KEYS | 28 | AST 9 + AZZ 19 + Cross 4 |
| Time (FUTURE_KNOWN_DIM) | N | sin/cos 인코딩 |

---

## 2. 핵심 컴포넌트별 상세 리뷰

### 2.1 특징 추출 — `features.py`

#### 현황

`calc_orderbook_features()`는 FH0 스키마 다양성을 잘 처리하며, 별칭(alias) 폴백, L1 수량 보완, 가격 역전 방어 등 현실적인 방어 코드가 잘 구현되어 있다.

`build_sequence()`는 타임스탬프 기반 분봉 매핑과 선형 매핑 폴백을 지원하며, 옵션/적응형 피처를 per-record 방식으로 붙인다.

#### 발견된 문제점 및 개선점

**[BUG-01] `bid_slope` / `offer_slope` 계산 오류**

```python
# 현재 코드
bid_slope = (bid_rems[-1] - bid_rems[0]) / 4.0
offer_slope = (offer_rems[-1] - offer_rems[0]) / 4.0
```

`bid_rems[-1]`은 5호가 수량(deepest), `bid_rems[0]`은 1호가 수량(nearest)이다. 이 차이를 4로 나누면 "단순 기울기"가 되는데, 5단계 모두 같은 수량이면 slope = 0.0이 되어 **정보 손실**이 발생한다. 또한 실제 시장에서 slope의 의미는 거리에 따른 유동성 감쇠율인데, 가격 레벨 간격이 포함되지 않아 **차원이 틀린** 피처다.

```python
# 개선안: 유동성 감쇠율 (깊이 가중 OBI slope)
prices = [bid_hoX, bid_ho2, ...]  # 가격 레벨 포함
bid_slope = np.polyfit(range(5), bid_rems, 1)[0] / (bid_rems[0] + 1e-9)
```

**[BUG-02] OBI 계산의 총량 정규화 불일치**

```python
total = total_offer + total_bid or 1.0
obi = (total_bid - total_offer) / total
```

`or 1.0`은 Python에서 `(total_offer + total_bid) or 1.0`으로 평가되므로, total = 0.0일 때만 1.0이 된다. 그런데 `total_bid` 또는 `total_offer`가 각각 0이고 합계는 양수인 경우(한쪽 호가만 있는 서킷브레이커 상황), OBI는 +1 또는 -1 극단값을 반환한다. 이 경우를 명시적으로 클램핑해야 한다.

```python
# 개선안
obi = np.clip((total_bid - total_offer) / max(total, 1e-9), -1.0, 1.0)
```

**[PERF-01] `build_sequence()` inner loop 속도**

현재 코드는 seq_len(60) 루프 × (opt_keys + adapt_keys) 딕셔너리 조회를 3개 배열에 걸쳐 반복한다. `_ts_epoch` 기반 분봉 매핑 루프가 가장 무겁다. 초당 호출 횟수가 많아지면 CPU 병목이 될 수 있다.

```python
# 개선안: numpy vectorized lookup 사용
# ob_arr은 이미 잘 구성되어 있지만, cd_arr 매핑은 searchsorted → iloc 조합으로 최적화 가능
# pd.merge_asof() 사용 시 전체 루프를 vectorize 가능
```

**[FEAT-01] 주문 흐름 불균형(OFI) 피처 부재**

Order Flow Imbalance는 고빈도 시장 미시구조 연구에서 단기 방향성 예측력이 OBI보다 높다고 알려져 있다. 현재 `obi_delta1`, `obi_delta5`로 간접 근사하고 있으나, 순수 OFI(volume-at-bid vs volume-at-ask 체결 불균형) 피처는 `tick_processor`에서 FC0 체결 데이터로 계산 가능하다.

```python
# 추가 제안 피처
"ofi_1s"   # 최근 1초 체결 OFI
"ofi_5s"   # 최근 5초 체결 OFI
"vwap_dev" # 현재가 vs 단기 VWAP 편차
```

---

### 2.2 수치 예측 — `predictor.py`

#### 현황

`TransformerPredictor`는 weights 없을 때 `_rule_based()`로 fallback하며, `TFTPredictor`와 `EnsemblePredictor`가 잘 구조화되어 있다. `AdaptiveEnsembleWeightTracker`는 20-bar 이동 정확도로 가중치를 업데이트한다.

#### 발견된 문제점 및 개선점

**[BUG-03] Rule-based fallback의 spread penalty 왜곡**

```python
# 현재 코드
spread_penalty = max(0.0, min(0.25, float(spread) / 5.0 * 0.25))
raw = 0.5 + max(-0.48, min(0.48, pressure)) - spread_penalty
```

`spread`가 KP200 선물의 호가 단위(0.05pt)이면 penalty는 0.0025로 사실상 무의미하다. 반대로 스프레드를 0~5 범위로 가정한 매직 넘버 `5.0`은 **실제 tick 단위와 무관**하다. 이 값은 constants에 명시하거나 ATM_PRICE 기반 상대 스프레드 비율로 대체해야 한다.

```python
# 개선안
spread_rel = spread / max(current_price * 0.001, 1e-9)  # 0.1% 기준 정규화
spread_penalty = max(0.0, min(0.25, spread_rel * 0.25))
```

**[BUG-04] EnsemblePredictor의 `disagreement_hold` 로직 반전**

```python
# 현재 코드
if (tft_prob is not None) and (not agreement) and self._disagreement_hold:
    prob_diff = abs(t_res.prob - tft_prob)
    if prob_diff < self._disagreement_hold_prob_diff_max:  # < 0.1 이면 HOLD
        signal = "HOLD"
```

설계 의도는 "두 모델이 방향이 다를 때 차이가 작으면 HOLD"인데, 실제로는 **방향이 다르면서 확률 차이가 작은 경우**는 두 모델 모두 중립에 가깝다는 뜻이므로 HOLD가 맞다. 그런데 `not agreement`를 먼저 체크하고 `prob_diff < 0.1`을 체크하므로, 방향은 다르나 확률 차이가 0.15처럼 큰 경우(한 모델은 0.65 BUY, 다른 모델은 0.50)는 오히려 신호가 그대로 통과된다. 임계값 방향성 재검토가 필요하다.

```python
# 개선안: prob_diff >= threshold일 때만 disagreement hold 적용
if (tft_prob is not None) and (not agreement) and self._disagreement_hold:
    prob_diff = abs(t_res.prob - tft_prob)
    if prob_diff >= self._disagreement_hold_prob_diff_max:  # 차이가 클 때 HOLD
        signal = "HOLD"
        confidence = "LOW"
```

**[FEAT-02] AdaptiveEnsembleWeightTracker 피드백 루프 미완성**

`update_adaptive_weights()` 메서드가 정의되어 있지만, `pipeline.py`에서 실제 정답 레이블을 기반으로 호출하는 코드가 없다. 즉 adaptive weight tracker는 항상 초기 `transformer_weight=0.5`로 고정 운영된다. 실제 예측 후 N분 뒤 결과를 검증하는 feedback loop를 연결해야 의미가 있다.

**[ARCH-01] `_classify()` 함수의 `spread` 파라미터 inconsistency**

`EnsemblePredictor.predict()`에서 confidence_high_margin, confidence_mid_margin을 전달하지 않고 `_classify()` 기본값을 사용한다. 반면 `TransformerPredictor._classify()`는 인스턴스 설정값을 사용한다. 동일한 임계값 설정이 Ensemble 경로에서는 무시된다.

```python
# 수정: EnsemblePredictor.predict()의 _classify 호출에 인스턴스 값 전달
signal, confidence = _classify(
    prob=float(ens_prob),
    spread=float(spread),
    buy_threshold=float(self._buy_threshold),
    sell_threshold=float(self._sell_threshold),
    confidence_high_margin=float(self._confidence_high_margin),  # 추가
    confidence_mid_margin=float(self._confidence_mid_margin),     # 추가
)
```

---

### 2.3 파이프라인 오케스트레이션 — `pipeline.py`

#### 현황

`PredictionPipeline`은 매우 풍부한 기능을 갖추고 있으며, 가드레일(basis, option), 듀얼 LLM, disagreement hold, FO0 stale 경고, adaptive 워밍업 등이 구현되어 있다.

#### 발견된 문제점 및 개선점

**[BUG-05] `adaptive_indicator` 설정 변수명 오류**

```python
# 현재 코드 (pipeline.py, ~line 200)
symbol = str(ad.get("symbol") or "KOSPI200 선물")
```

`ad` 변수가 정의되지 않고 있다. `adaptive_indicator` 파라미터를 `ad`로 rename하는 코드가 누락되어 있다. 실제로는 `ad = adaptive_indicator if isinstance(adaptive_indicator, dict) else {}` 라인이 있어야 한다. 현재 코드에서 `adaptive_indicator` 파라미터가 None이 아닌 값으로 들어오면 `NameError: name 'ad' is not defined`가 발생하며, except로 묵살되어 `_adaptive_mgr = None`이 된다.

```python
# 수정
ad = adaptive_indicator if isinstance(adaptive_indicator, dict) else {}
symbol = str(ad.get("symbol") or "KOSPI200 선물")
```

**[BUG-06] `_fo0_schema_logged` 속성명 오류**

`__init__`에서는 `self._last_fo0_schema_logged = False`로 초기화하지만, `add_realtime_tick()`에서는 `self._fo0_schema_logged`를 참조한다. 속성명이 다르기 때문에 첫 FO0 수신 시 `AttributeError`가 발생하거나 항상 schema 로그가 출력된다.

```python
# 수정: __init__에서 통일
self._fo0_schema_logged = False
```

**[BUG-07] `_adaptive_last_features` 속성 초기화 누락**

`_compute_adaptive_bundle()` 내부에서 `self._adaptive_last_features = dict(adaptive_features or {})` 를 설정하지만, `__init__`에서 초기화되지 않는다. `add_realtime_tick()` 내에서 `self._adaptive_last_features`를 참조하는 코드가 있어 초기화 전에 접근하면 `AttributeError`가 발생한다.

```python
# __init__에 추가
self._adaptive_last_features: Dict[str, float] = {}
self._adaptive_last_context: str = ""
```

**[PERF-02] `get_prediction()` 내 중복 `build_option_snapshot()` 호출**

`add_realtime_tick()` 내에서 이미 per-second 옵션 스냅샷을 `ob["_opt_features"]`에 캐시하고 있음에도, `get_prediction()`에서 `build_option_snapshot()`을 한 번 더 명시적으로 호출한다. 예측 호출 주기(약 1분)와 틱 주기(1초)가 다르므로 중복 호출이 낭비는 아니지만, 캐시된 최신값과 별도 계산값 간 미묘한 불일치가 생길 수 있다.

```python
# 개선안: _last_opt_features가 있으면 재사용
if self._last_opt_features and time.time() - (self._last_opt_sec_key or 0) < 5:
    opt_snap = {k: self._last_opt_features.get(k, 0.0) for k in self._opt_keys}
else:
    opt_snap = build_option_snapshot(...)
```

**[ARCH-02] `_compute_regime()` 함수가 `get_prediction()` 내 중첩 정의**

시장 국면 분류 로직이 `get_prediction()` 함수 내부 closure로 정의되어 있다. 이는 테스트 불가, 재사용 불가, 로직 추적 어려움으로 이어진다. 별도 함수로 분리해야 한다.

```python
# 개선안
def _compute_regime(
    adaptive_features: Optional[Dict], 
    adaptive_supertrend_state: Any
) -> Optional[str]:
    ...  # 클래스 메서드 또는 모듈 함수로 분리
```

**[ARCH-03] `get_prediction()` 함수의 단일 책임 원칙 위반**

현재 `get_prediction()`은 약 250줄에 달하며:
1. 데이터 준비 (가격, 분봉 df)
2. Adaptive indicator 계산
3. 옵션 스냅샷 계산
4. Numeric prediction
5. 가드레일 적용
6. LLM 컨텍스트/프롬프트 빌드
7. LLM 판단 실행
8. 최종 결과 dict 조립

이 8단계를 하나의 함수에서 처리한다. 각 단계는 이미 `_compute_adaptive_bundle()`, `_build_and_predict_numeric()` 등으로 일부 분리되어 있으나 3~8단계는 아직 인라인이다. 전체를 일관되게 분리하면 가독성과 테스트성이 크게 향상된다.

---

### 2.4 Transformer 모델 — `model.py`

#### 현황

`PriceTransformer`는 CLS 토큰 기반 분류 Transformer로 간결하게 구현되어 있다. `recency_weighted` pooling 옵션도 있다. `load()` 시 feature_dim mismatch를 사전 검사하는 방어 로직이 있다.

#### 발견된 문제점 및 개선점

**[MODEL-01] 모델 크기 대비 feature_dim 불균형**

KP200 선물의 feature_dim이 최대 10 + 5 + 16 + 28 + N ≈ 60+이고, `d_model=64`, `n_layers=2`, `n_heads=4`의 소형 모델이다. 이 크기는 빠른 추론에 적합하지만, 28개 adaptive 피처 + 16개 V2 옵션 피처처럼 피처 수가 급증한 경우 표현력이 부족할 수 있다. `d_model=128`, `n_layers=3`으로 실험 필요.

**[MODEL-02] CLS 토큰 초기화**

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, int(d_model)))
```

zero 초기화는 학습 초기 그라디언트가 CLS 토큰으로 전달되지 않을 수 있다. Xavier/Normal 초기화가 권장된다.

```python
nn.init.normal_(self.cls_token, std=0.02)
```

**[MODEL-03] Positional Encoding이 가격 시계열에 최적이 아닐 수 있음**

기존 sinusoidal PE는 절대 위치 기반이다. KP200 선물의 1Hz OB 시퀀스는 시장 개폐장 전환, 점심 시간 유동성 저하 등 **주기적 패턴**이 존재한다. 상대 위치 인코딩(Relative PE, RoPE) 또는 시간 특징을 별도 임베딩으로 처리하는 방식이 더 적합할 수 있다.

---

### 2.5 TFT 모델 — `tft_model.py`

#### 현황

VSN + GRN + LSTM encoder + multi-head attention 구조로 TFT 핵심을 구현. binary classification head로 마무리.

#### 발견된 문제점 및 개선점

**[MODEL-04] VSN에서 num_vars 추론 방식의 취약성**

`TFTPredictor.__init__()`에서 체크포인트 호환성 확인 시:

```python
def _infer_vars(prefix):
    w = state.get(f"{prefix}.weight_grn.fc1.weight")
    ...
    if in_dim % out_dim != 0:
        return None
    return in_dim // out_dim
```

이 추론은 `in_features = num_vars * d_model` 가정에 기반하는데, context_dim을 포함하면 `in_features = num_vars * d_model + context_dim`이 되어 잘못된 추론이 발생할 수 있다. 체크포인트에 `model_kwargs`를 저장하는 방식이 더 안전하다.

**[MODEL-05] LSTM과 Transformer Encoder 중복 시퀀스 처리**

TFT는 LSTM으로 시퀀스를 처리한 후, multi-head attention을 다시 적용한다. KOSPI200 60초 시퀀스에서 이 이중 구조가 과적합을 유발할 수 있다. Ablation study를 통해 LSTM만 사용하거나 Transformer 인코더만 사용했을 때와 비교 검증이 필요하다.

---

### 2.6 LLM 판단 — `llm_judge.py`

#### 현황

3개 프로바이더(Claude, GPT, Gemini) 지원, fallback 체인, 재시도 로직이 잘 구현되어 있다.

#### 발견된 문제점 및 개선점

**[BUG-08] Dual LLM 모드에서 futures.cancel() 효과 없음**

```python
try:
    fut.cancel()
except Exception:
    pass
```

`ThreadPoolExecutor`의 `Future.cancel()`은 이미 실행 중인 작업에는 효과가 없다. 타임아웃 후 해당 스레드는 LLM API 응답을 계속 기다리게 되고, `_reset_llm_executor()`로 새 executor를 만들어도 이전 executor의 스레드가 zombie 상태로 남는다. 근본 해결을 위해 LLM HTTP 요청에 실제 timeout을 설정해야 한다.

```python
# 개선안: httpx timeout 또는 requests timeout 설정
self._anthropic = anthropic.Anthropic(
    api_key=self.anthropic_key,
    timeout=self._llm_timeout_sec
)
```

**[ARCH-04] LLM 응답 JSON 파싱의 취약성**

현재 JSON 파싱은 `json.loads(raw)`를 시도하고, 실패 시 마크다운 코드 블록 제거 후 재시도하는 구조이다. 그러나 LLM이 JSON 앞뒤에 설명 텍스트를 추가하는 경우, 정규식 기반 추출이 필요하다. 특히 Gemini는 `{...}` 사이에 유효한 JSON을 포함하더라도 앞뒤에 텍스트를 추가하는 경향이 있다.

```python
# 개선안: JSON 추출 강화
import re
def _extract_json(raw: str) -> dict:
    # 1) direct parse
    # 2) strip markdown
    # 3) regex search for {...}
    m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError("no JSON found")
```

**[COST-01] LLM 호출 빈도 제어 없음**

`get_prediction()`은 호출될 때마다 LLM을 실행한다. 5분 예측 주기에서 1분마다 호출하면 하루 약 300~400회 LLM 호출이 발생한다. Rate limit 도달 또는 비용 폭증을 방지하기 위한 최소 호출 간격(throttle) 또는 신호 변화 시에만 LLM 재호출하는 캐싱 로직이 없다.

```python
# 개선안: LLM 결과 캐싱
_last_llm_call_epoch: float = 0.0
_last_llm_result: Optional[LLMJudgment] = None
_llm_min_interval_sec: float = 30.0  # 최소 30초 간격

if time.time() - self._last_llm_call_epoch < self._llm_min_interval_sec:
    return self._last_llm_result  # 이전 결과 재사용
```

---

### 2.7 컨텍스트 빌더 — `context_builder.py`

#### 현황

LLM에 전달하는 컨텍스트를 JSON + 섹션 태그로 구성하는 간결한 구조이다.

#### 발견된 문제점 및 개선점

**[FEAT-03] 최근 예측 히스토리 미포함**

LLM에게 현재 시점의 스냅샷만 전달하고 있다. 지난 3~5회 예측 결과(signal, confidence, 실제 가격 변화)를 포함하면, LLM이 최근 예측 품질과 트렌드 지속성을 판단하는 데 유용한 컨텍스트가 된다.

**[FEAT-04] 시장 맥락(거시 정보) 부재**

배경 스냅샷(`t2101`, `t2301`, `ij_`)이 있음에도, 컨텍스트에 포함되는 내용이 단순 JSON 덤프 수준이다. 야간선물 대비 현재 괴리, 전일 종가 대비 변동률, 옵션 만기까지 남은 일수 등을 구조화된 텍스트로 요약하면 LLM의 판단 품질이 향상된다.

**[ARCH-05] system prompt가 단일 문장으로 너무 단순**

```python
system = (
    "당신은 파생상품 트레이딩 리스크 분석 전문가입니다. "
    "제공된 입력을 바탕으로 {m}분 관점에서 전략 판단을 내리세요. "
    "반드시 JSON 단일 객체로만 응답하며, 마크다운/설명 텍스트를 절대 포함하지 마세요."
)
```

System prompt에 페르소나 강화(리스크 관리 우선, 과신 금지), Chain-of-Thought 억제(이미 JSON만 요청하지만 불충분), 특수 시장 상황(만기일, 서킷브레이커 발동 중)에 대한 지침 등을 추가하면 응답 품질이 개선된다.

---

### 2.8 옵션 피처 — `option_features.py`

#### 현황

PCR, IV skew, max pain, ATM microstructure 등 주요 옵션 지표가 구현되어 있다.

#### 발견된 문제점 및 개선점

**[BUG-09] IV Skew 계산의 ATM 정의 불일치**

`calc_iv_skew()`는 현재가에 가장 가까운 행사가를 ATM으로 정의하지만, KOSPI200 옵션의 표준 ATM 행사가는 **0.5pt 단위로 반올림**된 값이다. 현재가가 380.25이면 ATM은 380.0이나 380.5여야 하는데, `min(strikes, key=lambda s: abs(s - upx))`는 390.0 같은 먼 행사가도 선택할 수 있다. ATM 허용 오차(예: ±2.5pt)를 두고 그 범위 밖이면 IV skew = 1.0(중립)으로 처리해야 한다.

**[FEAT-05] Gamma Exposure (GEX) 미산출**

옵션 마켓메이커의 헤지 방향을 나타내는 GEX(= Σ gamma × OI × 100)는 단기 방향성 및 변동성 예측에 매우 유용하다. 특히 GEX < 0(음의 감마 환경)에서 시장은 방향성을 증폭하는 경향이 있다. IV와 OI 데이터가 이미 있으므로 구현 가능하다.

---

### 2.9 가중치 선택 — `weights_selector.py`

#### 현황

만기주 동결 로직이 구현되어 있으며, 날짜 파싱과 KOSPI200 두 번째 목요일 만기 처리가 잘 되어 있다.

#### 발견된 문제점 및 개선점

**[ARCH-06] 가중치 버저닝 및 롤백 메커니즘 부재**

현재 `transformer_5m_YYYYMMDD.pt` 형식으로 날짜 버전 가중치가 지원되지만, 특정 날짜 모델의 성능이 급격히 저하될 때 자동 롤백하는 메커니즘이 없다. 최근 N일 OOS 정확도를 추적하다가 임계값 이하로 떨어지면 이전 버전으로 롤백하는 로직 추가를 권장한다.

---

## 3. 우선순위별 개선 로드맵

### Phase 1: 즉시 수정 필요 (버그)

| ID | 파일 | 내용 | 심각도 |
|----|------|------|--------|
| BUG-05 | pipeline.py | `ad` 변수명 오류 → `_adaptive_mgr = None` 항상 | 🔴 Critical |
| BUG-06 | pipeline.py | `_fo0_schema_logged` 속성명 불일치 | 🟠 High |
| BUG-07 | pipeline.py | `_adaptive_last_features` 초기화 누락 | 🟠 High |
| BUG-04 | predictor.py | disagreement_hold 임계값 방향 반전 | 🟠 High |
| ARCH-01 | predictor.py | EnsemblePredictor confidence 파라미터 미전달 | 🟠 High |

### Phase 2: 단기 개선 (1~2주)

| ID | 파일 | 내용 | 기대 효과 |
|----|------|------|-----------|
| BUG-01 | features.py | slope 계산 수정 + 가격 레벨 포함 | 피처 품질 향상 |
| BUG-02 | features.py | OBI 클램핑 | 극단값 방지 |
| BUG-09 | option_features.py | ATM 허용 오차 추가 | IV skew 정확도 향상 |
| ARCH-04 | llm_judge.py | JSON 추출 강화 | LLM 파싱 실패율 감소 |
| FEAT-02 | predictor.py | AdaptiveWeight feedback loop 연결 | 앙상블 적응성 |

### Phase 3: 중기 아키텍처 개선 (1개월)

| ID | 파일 | 내용 | 기대 효과 |
|----|------|------|-----------|
| COST-01 | pipeline.py | LLM 호출 throttle / 캐싱 | API 비용 절감 |
| ARCH-02 | pipeline.py | `_compute_regime()` 분리 | 테스트 가능성 향상 |
| ARCH-03 | pipeline.py | `get_prediction()` 단계별 분리 | 유지보수성 향상 |
| FEAT-01 | features.py | OFI 피처 추가 | 예측 정확도 향상 |
| FEAT-05 | option_features.py | GEX 산출 | 변동성 국면 예측 |
| MODEL-02 | model.py | CLS 토큰 초기화 개선 | 학습 안정성 향상 |

### Phase 4: 장기 연구 과제

| ID | 내용 |
|----|------|
| MODEL-03 | Relative PE 또는 RoPE 적용 실험 |
| MODEL-04 | TFT vs Transformer Ablation study |
| FEAT-03 | 예측 히스토리 LLM 컨텍스트 포함 |
| FEAT-04 | 거시 시장 맥락 구조화 |
| ARCH-06 | 자동 가중치 롤백 메커니즘 |

---

## 4. 추가 제안: 예측 품질 모니터링

현재 시스템에는 예측 품질을 측정하는 코드가 없다. 다음 지표를 런타임에 수집하는 것을 강력히 권장한다.

```python
# 제안: PredictionTracker 클래스
class PredictionTracker:
    """예측 후 N분 뒤 실제 가격 변화를 추적하여 정확도를 계산."""
    
    def record_prediction(self, signal, confidence, current_price, timestamp):
        """예측 기록."""
        ...
    
    def record_outcome(self, timestamp, final_price):
        """N분 뒤 결과 기록."""
        ...
    
    def get_accuracy(self, window=100) -> Dict[str, float]:
        """최근 N회 예측의 방향 정확도, confidence calibration 반환."""
        return {
            "directional_accuracy": ...,  # 방향 맞춤 비율
            "high_confidence_accuracy": ...,  # HIGH confidence만 필터링
            "signal_dist": ...,  # BUY/SELL/HOLD 분포
        }
```

이 트래커를 `AdaptiveEnsembleWeightTracker.update()`의 피드백 소스로 연결하면, BUG-04에서 제안한 adaptive weight feedback loop가 완성된다.

---

## 5. 보안 및 운영 관련 사항

**[SEC-01] config.secrets.json 파일 ZIP 포함**  
`config.secrets.json`이 코드베이스 ZIP에 포함되어 있다. API 키, 계좌 정보 등 민감 데이터가 포함될 경우 버전 관리에서 반드시 제외해야 한다. `.gitignore`에 추가하고 환경변수 또는 별도 vault를 사용할 것.

**[OPS-01] 예측 실패율 모니터링**  
`self._metrics["prediction_failures"]`가 수집되고 있으나, 이를 Telegram 알림이나 외부 모니터링(Prometheus 등)으로 내보내는 코드가 없다. 일정 임계값(예: 10분 내 3회 연속 실패) 도달 시 알림 발송을 권장한다.

---

## 6. 총평

Transformer 예측 시스템은 전반적으로 잘 설계된 구조를 가지고 있으며, 실시간 트레이딩 시스템에 필요한 방어 코드(fallback, timeout, stale 경고)가 풍부하다. Dual LLM, adaptive indicator, ensemble predictor 등 고급 기능도 갖추고 있다.

그러나 변수명 오류(BUG-05~07)처럼 즉시 수정이 필요한 버그들이 존재하며, 이는 adaptive indicator 기능 전체를 묵시적으로 비활성화시키는 중요한 문제다. 또한 LLM 호출 비용 제어, 예측 품질 피드백 루프, feature 계산의 수치적 정확성 개선이 시스템의 실전 성능을 크게 향상시킬 수 있는 핵심 영역이다.

Phase 1의 버그 수정을 우선 진행하고, 예측 품질 모니터링 인프라를 갖춘 후 Phase 2~3의 개선을 체계적으로 진행하는 것을 권장한다.
