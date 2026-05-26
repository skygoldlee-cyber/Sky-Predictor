# Transformer 예측 파이프라인 — 코드 리뷰 보고서

> 분석 대상: `Transformer.zip` (prediction/ + adaptive_indicator/ 패키지, 총 ~7,400줄)  
> 작성일: 2026-02-28

---

## 목차

1. [아키텍처 개요](#1-아키텍처-개요)
2. [모델 구조 리뷰 (model.py / tft_model.py)](#2-모델-구조-리뷰)
3. [피처 엔지니어링 리뷰 (features.py)](#3-피처-엔지니어링-리뷰)
4. [예측기 레이어 리뷰 (predictor.py)](#4-예측기-레이어-리뷰)
5. [파이프라인 리뷰 (pipeline.py)](#5-파이프라인-리뷰)
6. [LLM Judge 리뷰 (llm_judge.py)](#6-llm-judge-리뷰)
7. [데이터 빌더 리뷰 (data_builder.py)](#7-데이터-빌더-리뷰)
8. [Adaptive Indicator 통합 리뷰](#8-adaptive-indicator-통합-리뷰)
9. [잔존 버그 및 오류 가능성](#9-잔존-버그-및-오류-가능성)
10. [설계 개선 제안](#10-설계-개선-제안)
11. [우선순위 요약표](#11-우선순위-요약표)

---

## 1. 아키텍처 개요

```
FH0(호가)/FC0(체결)/OC0(옵션체결)
        ↓
  PredictionPipeline
  ├── RealTimeTickProcessor   (분봉 집계 + 옵션 스냅샷)
  ├── AdaptiveIndicatorManager (SuperTrend + ZigZag)
  ├── FeatureBuilder           (OB 7 + CD 5 + OPT 7~16 + ADAPT 28 = 최대 56차원)
  ├── NumericPredictor         (Transformer / TFT / Ensemble / RuleBased)
  │     └── EnsemblePredictor  → TransformerPredictor + TFTPredictor
  └── LLMJudge                 (Claude / GPT / Gemini → JSON 판단)
        ↓
  get_prediction() → Dict (prob, signal, llm_action, consensus, ...)
```

### 구성 요소별 역할

| 모듈 | 역할 | 줄 수 |
|------|------|-------|
| `model.py` | Transformer Encoder (CLS 토큰 기반 이진 분류) | 190 |
| `tft_model.py` | Temporal Fusion Transformer (LSTM + Attention) | 282 |
| `features.py` | OB/캔들/옵션/Adaptive 피처 빌드 | 352 |
| `predictor.py` | 예측기 레이어 (Transformer / TFT / Ensemble) | 627 |
| `pipeline.py` | 전체 파이프라인 오케스트레이터 | 1,354 |
| `llm_judge.py` | LLM API 호출 및 JSON 파싱 | 584 |
| `data_builder.py` | 오프라인 학습 데이터 생성 | 564 |
| `context_builder.py` | LLM 프롬프트 구성 | 141 |
| `indicator_integration.py` | AdaptiveIndicator 통합 관리 | 305 |

---

## 2. 모델 구조 리뷰

### 2-1. PriceTransformer (model.py)

```
Input (B, seq_len, feature_dim)
  → LinearProjection (feature_dim → d_model=64)
  → [CLS] 토큰 prepend
  → PositionalEncoding (sinusoidal, max_len=seq_len+1)
  → TransformerEncoder (Pre-LN, nhead=4, n_layers=2, d_ff=128)
  → CLS 토큰 추출 → LayerNorm → Linear(64→32) → GELU → Dropout → Linear(32→1) → Sigmoid
```

**잘 된 점:**
- `norm_first=True` (Pre-LayerNorm): 학습 안정성 향상, 최신 관행 반영
- CLS 토큰 방식: 전체 시퀀스를 하나의 벡터로 집약, 분류에 적합
- `weights_only=True`: 안전한 가중치 로드
- feature_dim 불일치 시 fallback graceful handling 구현

**개선 필요:**

```python
# ① d_ff=128은 d_model=64의 2배 — 표준(4배=256)의 절반
# 표현력 제한 가능성이 있음
# 권장: d_ff=256 (또는 config 파라미터화)
encoder_layer = nn.TransformerEncoderLayer(
    d_model=64, nhead=4, dim_feedforward=128,  # ← 256으로 올리는 것 검토
    ...
)
```

```python
# ② 현재 헤드: LayerNorm → Linear(64→32) → GELU → Dropout → Linear(32→1)
# 중간 차원 32도 다소 작음. 64 또는 제거하고 직결도 고려
self.head = nn.Sequential(
    nn.LayerNorm(64),
    nn.Linear(64, 1),  # 단순화 대안
    nn.Sigmoid(),
)
```

---

### 2-2. TemporalFusionTransformer (tft_model.py)

```
past_unknown (B, seq_len, pu_dim)
past_known   (B, seq_len, fk_dim)    → VSN → LSTM Encoder
future_known (B, horizon, fk_dim)   → VSN → LSTM Decoder
  → GRN Static Enrichment
  → TransformerEncoder (Pre-LN, n_layers=2)
  → PWFF (GRN)
  → first_future[seq_len] → head → Sigmoid
```

**잘 된 점:**
- `_GatedResidualNetwork`, `_VariableSelectionNetwork` 명시적 구현
- feature_dim 불일치를 state_dict에서 사전 검증 (`_infer_vars`)
- encoder/decoder LSTM 게이팅 (`gate_enc`, `gate_dec`)

**개선 필요:**

```python
# ③ LSTM multi-layer dropout 미설정
self.encoder_lstm = nn.LSTM(
    ..., num_layers=1, dropout=0.0  # num_layers>1이면 dropout 설정 필요
)
# 현재 num_layers=1 이라 문제없지만, 레이어 수 증가 시 명시 필요

# ④ future_known을 VSN에 두 번 통과 (vsn_past_known / vsn_future_known)
# 두 VSN이 같은 입력 차원(future_known_dim)을 받음
# 모델 저장 크기 증가 대비 실익 검토 필요
```

---

## 3. 피처 엔지니어링 리뷰

### 3-1. 피처 차원 구성

```
feature_dim = OB_KEYS(7) + CD_KEYS(5) + OPT_KEYS_v1(7) + ADAPT_KEYS(28) = 47
             = OB_KEYS(7) + CD_KEYS(5) + OPT_KEYS_v2(16) + ADAPT_KEYS(28) = 56
```

`PAST_UNKNOWN_DIM = 47`은 v1 + ADAPT 기준 상수이며, v2 전환 시 **재학습 필수**.

### 3-2. `calc_orderbook_features()` — 잘 된 점

- `_invalid=True` 플래그로 불완전 스냅샷 명시적 처리
- 다양한 키 별칭(alias) 처리 (`offerho`, `ask`, `bid` 등)
- 역전된 스프레드(`spread < 0`) 절댓값 처리

### 3-3. 개선 필요

```python
# ⑤ bid_slope / offer_slope 계산 방식
# slope = (rem[-1] - rem[0]) / 4.0  ← 단순 차이/4 (배열이 5개이므로 /4 는 맞음)
# 그러나 Depth가 없어 L1만 존재할 때 bid_rems[0]==bid_rems[4]==0 → slope=0
# 의미 없는 0이 아닌 NaN으로 마킹하거나 별도 validity 플래그가 더 정확
```

```python
# ⑥ build_sequence() 내 타임스탬프 불일치 시 silent pass
try:
    ...
except Exception:
    continue  # ← 어떤 예외가 발생했는지 전혀 기록되지 않음
# → (반영 완료) 예외 경로에 logger.debug를 추가하여(1회/콜 기준) 원인 추적 가능
```

```python
# ⑦ opt_arr / adapt_arr: 옵션/Adaptive 피처를 seq_len 전체에 tile
opt_arr = np.tile(opt_row, (seq_len, 1))  # 모든 timestep에 동일한 값
```
옵션 스냅샷은 1분 단위이므로 모든 timestep에 동일 값을 broadcast하는 것은 의도적이지만,
Transformer가 시간 변화 패턴을 학습할 수 없다는 제약이 있음.  
학습 시 이 구조를 명시적으로 문서화해야 함.

---

## 4. 예측기 레이어 리뷰

### 4-1. 신호 분류 로직 (`_classify()`)

```python
# ⑧ 신뢰도(confidence) 기준 하드코딩
margin = abs(p - 0.5)
if margin >= 0.15 and spread <= 1.0:  confidence = "HIGH"
elif margin >= 0.08:                   confidence = "MEDIUM"
else:                                  confidence = "LOW"
```

- (반영 완료) `0.15`, `0.08`, `1.0`은 `config.json`의 `prediction.confidence_*`로 이동하여 운영 중 튜닝 가능
- `spread <= 1.0`은 KP200 선물 호가단위(0.05)를 기준으로 상당히 큰 값  
  → 실제로 HIGH confidence 조건이 spread로 걸리는 빈도가 높을 수 있음

```python
@dataclass
class ClassifierConfig:
    high_margin: float = 0.15
    mid_margin: float = 0.08
    spread_max_for_high: float = 1.0
```

실제 런타임에서는 다음 키로 설정할 수 있습니다:

- `prediction.confidence_high_margin`
- `prediction.confidence_mid_margin`
- `prediction.confidence_spread_max_for_high`

### 4-2. `EnsemblePredictor.predict()` — disagreement_hold

```python
# ⑨ 두 모델이 HOLD를 출력하면 agreement=True → signal을 유지
# HOLD vs HOLD는 "동의"로 처리되어 올바른 동작이지만
# BUY+HOLD 조합은 disagreement → signal=HOLD로 강제됨
# BUY+HOLD가 실제 상승 신호를 잃어버리는 false negative 가능성 존재
# → threshold 기반 direction 체크 전에 one-sided HOLD 예외처리 고려
```

(반영 완료) disagreement 시 무조건 HOLD가 아니라, 확률 차이가 작을 때만 HOLD로 강제할 수 있도록
`prediction.disagreement_hold_prob_diff_max`(기본 0.1) 임계값을 추가했습니다.

### 4-3. `TFTPredictor._infer_vars()` — 차원 추론 로직

```python
# ⑩ VSN fc1 weight: in_features = num_vars * d_model
# in_dim % out_dim != 0 조건으로 num_vars를 추론
# 그런데 static_context_dim > 0 이면 in_features = num_vars*d_model + context_dim
# → context_dim이 0이 아닌 경우 나눗셈이 실패하여 inferred_pu=None 반환
# 현재 static_dim=0이 기본값이므로 문제없지만, static 추가 시 추론 실패 가능성 있음
```

---

## 5. 파이프라인 리뷰

### 5-1. get_prediction() 구조 분석

전체 흐름에서 중요한 설계 결정들:

```
1. FO0 1Hz 다운샘플링  → deque(maxlen=seq_len)
2. 분봉 df 취득        → tick_processor
3. Adaptive 업데이트   → warmup(최초1회) / incremental(이후)
4. 피처 빌드           → build_sequence()
5. 수치 예측           → numeric_predictor.predict()
6. 옵션 리퀴디티 가드  → wide spread → signal downgrade
7. 베이시스 가드       → |basis| > 2.5 → HOLD
8. LLM 판단            → ThreadPoolExecutor(max_workers=1) + timeout
9. 결과 조합            → consensus = signal == llm_action
```

**잘 된 점:**
- 베이시스 가드: 현선물 차이가 크면 자동 HOLD (안전 장치)
- FO0 staleness 경고: 60초 쿨다운 포함
- 메트릭 수집: latency_ms, prediction_failures 내부 집계

### 5-2. 개선 필요

```python
# ⑪ get_prediction() 함수 길이 ~700줄
# 단일 메서드가 너무 많은 책임을 갖고 있음
# 권장 분리:
# - _build_adaptive_features() → AdaptiveFeatureResult
# - _build_model_input()       → ModelInput
# - _apply_guardrails()        → GuardrailResult
# - _run_llm_judgment()        → LLMResult
```

(반영 완료) `PredictionPipeline.get_prediction()`은 여러 개의 private helper 메서드로 분해하여
가독성과 테스트 용이성을 개선했습니다(공개 API/동작은 유지).

```python
# ⑫ LLM 실행이 ThreadPoolExecutor(max_workers=1)에서 순차 처리
# dual_llm=True 시 GPT와 Gemini 호출이 순차 실행됨
# 개선: max_workers=2로 두 호출을 병렬 실행 가능
self._llm_executor = ThreadPoolExecutor(max_workers=2 if self._dual_llm else 1)
```

```python
# ⑬ adaptive_warmed 플래그가 인스턴스 레벨 단 하나 → 웜업 후 df 길이 변경 시 재웜업 없음
# df 길이가 warmup_bars 미만으로 떨어지는 케이스(데이터 부족 → 이전 상태 사용)는
# _adaptive_last_features 재활용으로 처리되지만 명시적 주석이 없음
```

(반영 완료) 분봉 DF가 리셋/되감기되는 상황을 감지(마지막 완성 분봉 timestamp 비교)하여
adaptive manager를 reset 및 재웜업하도록 보완했습니다.

```python
# ⑭ FO0 sig 중복 체크 시 _ts_epoch 동일 + sig 동일이어야 skip
# 그런데 _ts_epoch가 sec_key가 아닌 별도 값으로 설정됨
ob.setdefault("_ts_epoch", int(sec_key))  # ← 실제로는 sec_key와 동일하므로 문제없음
# 하지만 코드 이해 시 혼동 가능 → 명확히 통일 필요
```

---

## 6. LLM Judge 리뷰

### 6-1. 프롬프트 설계

```python
# context_builder.py
system = "당신은 파생상품 트레이딩 리스크 분석 전문가입니다. ... JSON 단일 객체로만 응답 ..."
user   = f"입력 데이터...\n{context}\n\n스키마:\n{json.dumps(schema)}\n\n출력은 반드시 JSON 단일 객체만."
```

**잘 된 점:**
- 시스템 프롬프트와 사용자 프롬프트 분리
- JSON 스키마를 명시적으로 첨부
- `LLM_OUTPUT_SCHEMA`를 constants에서 중앙 관리

**개선 필요:**

```python
# ⑮ JSON 파싱 실패 시 fallback이 있는지 llm_judge.py에서 확인 필요
# 현재: judge()가 None 반환 → pipeline이 t_res.signal fallback 처리
# 단, Gemini 빈 응답("empty_output") 재시도는 구현되어 있음

# ⑯ rationale/caution 필드를 LLM이 비우면 pipeline에서
rationale = str(judgment.rationale or "")  # → ""가 그대로 출력됨
# → 기본 fallback 메시지 설정 권장
```

### 6-2. Gemini 모델 자동 선택

```python
# ⑰ _select_gemini_model()이 models.list() 실패 시 desired 모델을 그대로 반환
# 만약 모델이 실제로 존재하지 않으면 첫 API 호출 시 에러
# → _is_gemini_model_error() 로 감지 후 fallback_models 순서대로 재시도하는 로직은 있으나
#    이 재시도 경로가 judge_provider()에서만 발동되므로
#    단독 judge() 호출 경로에서는 model 에러를 graceful하게 처리하는지 검토 필요
```

---

## 7. 데이터 빌더 리뷰

### 7-1. 설계 일관성

학습/런타임 피처 일관성 측면:

```python
# data_builder.py
OPT_KEYS = list(get_opt_keys(str(option_feature_set or "v1")))
# adaptive_enabled = cfg.adaptive_indicator.enabled

# pipeline.py (런타임)
self._opt_keys = list(get_opt_keys(str(self._option_feature_set)))
# adaptive_block_dim = len(ADAPT_KEYS) if self._adaptive_mgr is not None else 0
```

`config.json`의 `option_feature_set` 및 `adaptive_indicator.enabled`가 학습/런타임에서 **반드시 일치**해야 합니다.  
(반영 완료) dataset npz에 feature schema metadata를 함께 저장하고, `train.py`/`train_tft.py`에서
런타임 설정과의 불일치를 학습 시작 전에 검증하도록 추가했습니다.

### 7-2. 개선 필요

```python
# ⑱ build_dataset()에서 adaptive indicator 웜업 누락 시 silent skip
try:
    ...
    adaptive_features = {k: float(v) for k, v in tf.items() if v is not None}
except Exception:
    adaptive_features = {}  # ← 예외 시 zero feature로 학습 → 런타임 불일치 가능성
```

```python
# ⑲ 학습 데이터 레이블 생성 방식
# horizon_min=5분 후 close를 기준으로 up=1/down=0 이진 레이블
# "up" 조건이 단순히 price_t+5 > price_t이므로
# 수수료/슬리피지를 반영한 임계값 기반 레이블이 실전과 더 일치할 수 있음
# 예: abs(ret) >= 0.1% → label, else → skip (no-trade zone 설정)
```

```python
# ⑳ _restore_compact_prices()에서 x100 → /100.0 역변환
# offerho*/bidho*가 정수 cents인 경우에만 변환하는데
# 조건이 isinstance(v, int)만으로는 실제 가격(예: 34000)도 변환될 수 있음
# → 값 범위 검증 추가 필요 (예: v > 10000 이면 price, v < 1000 이면 cents 등)
```

---

## 8. Adaptive Indicator 통합 리뷰

### 8-1. AdaptiveIndicatorManager

```python
# ㉑ is_ready() 조건: len(_all_swings) >= 4
# ZigZag 스윙이 4개 이상 생성될 때까지 features를 None으로 반환
# 단기 변동성이 낮으면 웜업 120봉 이후에도 스윙이 4개 미만일 수 있음
# → 이 경우 prediction 전체가 adaptive_features=None으로 진행됨
# → fallback 동작(rule_based_only 모드)을 사용자에게 명확히 로깅 필요
```

```python
# ㉒ compute_from_df()의 column명 기본값 소문자 ('high', 'low', 'close')
# pipeline.py는 분봉 df에서 소문자 컬럼을 사용 (tick_processor 산출물)
# 그러나 UnifiedTA.py의 compute_from_df()는 기본이 대문자
# → 통합 코드베이스에서 혼재 가능성 있음 (통일 필요)
```

### 8-2. 피처 정규화

```python
# ㉓ ADAPT_KEYS 28개 피처 중 일부가 이미 정규화, 일부는 원시값
# 예: ast_trend_duration은 bars 단위 원시값 (0 ~ 수백)
#     ast_dist_pct는 퍼센트 (보통 -5 ~ +5)
#     ast_direction은 -1/0/1 이산값
# → Transformer 입력 전 batch normalization 또는 min-max 정규화 레이어 추가 검토
```

---

## 9. 잔존 버그 및 오류 가능성

### 🔴 Critical

#### 9-1. `validate_consistency()` 마지막 부분 미완성

```python
# indicator_integration.py 마지막 부분 (zip 내 파일 잘림 가능성 or 미완성)
keys = [k for k in tf.keys() if k in batch.columns]
if not keys:
    # ← 함수가 여기서 끊김 (return 문 없음)
```
`validate_consistency()`가 명시적 `return`도 없이 끝나면 항상 `None`을 반환합니다.  
호출부가 `bool(validate_consistency(...))`로 체크할 경우 항상 `False`.

---

#### 9-2. `PAST_UNKNOWN_DIM = 47` 상수와 실제 차원 불일치 위험

```python
# constants.py (추정)
PAST_UNKNOWN_DIM = 47  # OB7 + CD5 + OPT_v1(7) + ADAPT(28)

# pipeline.py 실제 계산
feature_dim = len(OB_KEYS) + len(CD_KEYS) + len(self._opt_keys) + adaptive_block_dim
# adaptive_mgr=None 이면 adaptive_block_dim=0 → feature_dim=19
# adaptive_mgr 있고 v2이면 → 56
```

**PAST_UNKNOWN_DIM 상수와 런타임 feature_dim이 일치하지 않는 3가지 조합:**

| adaptive | option_set | 실제 dim | PAST_UNKNOWN_DIM=47 |
|----------|-----------|---------|---------------------|
| None     | v1        | 19      | ❌ 불일치 |
| 있음     | v1        | 47      | ✅ |
| 있음     | v2        | 56      | ❌ 불일치 |

`TransformerPredictor`/`TFTPredictor`는 weights 로드 시 feature_dim을 state_dict에서 검증하므로 런타임 크래시는 방지되지만, **rule_based 모드 fallback 시 PAST_UNKNOWN_DIM을 직접 사용하는 경로가 있으면 Shape 오류 발생**.

---

#### 9-3. `_adaptive_warmed` 플래그 재설정 없음

```python
# pipeline.py
if not self._adaptive_warmed:
    # 전체 tail_bars를 순회하여 웜업
    ...
    self._adaptive_warmed = True
else:
    # 마지막 봉만 incremental update
    ...
```

`PredictionPipeline` 인스턴스가 재시작 없이 장중 수시간 운용될 때 `tick_processor`가 flush되어 df가 초기화되면, `_adaptive_warmed=True`인 채로 incremental update만 진행됩니다. AdaptiveIndicatorManager 내부 상태는 초기화되지 않으므로 이론적으로는 문제없지만 df 스냅샷과 indicator 내부 상태의 시간 정합성이 어긋날 수 있습니다.

---

### 🟡 Warning

#### 9-4. `build_sequence()` Fallback 분기에서 선형 매핑

```python
# Fallback (타임스탬프 없을 때): linear mapping
for row in range(seq_len):
    bar_idx = min(bars - 1, int(row * bars / seq_len))
    cd_arr[row] = cd_vals[bar_idx]
```

분봉 길이(bars)와 OB 버퍼 길이(seq_len)가 다를 때 캔들 피처를 시계열 비율로 매핑합니다. 분봉 1분 = OB 여러 tick이라 1:1 대응이 아니어서 **학습/런타임 타임스탬프 불일치 시 노이즈 삽입** 가능성이 있습니다.

---

#### 9-5. `_judge_provider_with_timeout()` 재시도 시 동일 executor 재사용

```python
for attempt in range(max_retries):
    fut = self._llm_executor.submit(...)
    judgment = fut.result(timeout=...)
```

`ThreadPoolExecutor(max_workers=1)` 기준으로 재시도 시 이전 실패한 태스크가 완료되기 전에 새 submit이 큐에 쌓일 수 있습니다. Timeout 후 `fut.cancel()`을 시도하지만 이미 실행 중인 태스크는 취소 불가 → 백그라운드 스레드가 지연될 수 있습니다.

(반영 완료) timeout/예외 발생 시 executor를 shutdown 후 재생성하여, 큐 적체가 누적되지 않도록 개선했습니다.

---

## 10. 설계 개선 제안

### 10-1. 피처 스키마 버전 관리 자동화

현재 피처 구성을 결정하는 상수들이 여러 파일에 분산되어 있습니다:

```python
# 현재 (분산)
# features.py: OB_KEYS, CD_KEYS, OPT_KEYS_V1/V2, ADAPT_KEYS
# constants.py: PAST_UNKNOWN_DIM = 47
# pipeline.py: feature_dim 런타임 계산
# data_builder.py: feature_dim 런타임 계산
```

**권장 방안:**

```python
# schema.py (신규)
@dataclass(frozen=True)
class FeatureSchema:
    ob_dim: int
    cd_dim: int
    opt_dim: int
    adapt_dim: int
    version: str

    @property
    def total_dim(self) -> int:
        return self.ob_dim + self.cd_dim + self.opt_dim + self.adapt_dim

    def assert_matches(self, actual_dim: int):
        if self.total_dim != actual_dim:
            raise ValueError(f"Schema mismatch: expected {self.total_dim}, got {actual_dim}")

SCHEMA_V1_ADAPT = FeatureSchema(ob_dim=7, cd_dim=5, opt_dim=7, adapt_dim=28, version="v1_adapt")
```

---

### 10-2. `get_prediction()` 메서드 분해

700줄 단일 메서드를 다음과 같이 분해하면 테스트 가능성과 가독성이 크게 향상됩니다:

```python
class PredictionPipeline:
    def get_prediction(self, **kwargs) -> Dict:
        ctx = self._prepare_context(**kwargs)         # 시장 데이터 수집
        features = self._build_features(ctx)          # 피처 빌딩
        numeric = self._run_numeric_predictor(features) # 수치 예측
        guarded = self._apply_guardrails(numeric, ctx)  # 가드레일
        llm = self._run_llm(ctx, guarded)              # LLM 판단
        return self._compose_output(ctx, guarded, llm) # 결과 조합
```

---

### 10-3. `disagreement_hold` 정책 개선

```python
# 현재: 두 모델 방향이 다르면 무조건 HOLD
if not agreement and self._disagreement_hold:
    signal = "HOLD"

# 개선: 확률 차이가 작을 때만 HOLD
prob_diff = abs(t_res.prob - f_res.prob)
if not agreement and self._disagreement_hold and prob_diff < 0.1:
    signal = "HOLD"
# 두 모델이 강하게 다른 방향을 가리키면 오히려 중요한 신호일 수 있음
```

---

### 10-4. 학습/런타임 정합성 자동 검증

```python
# data_builder.py에 메타데이터 저장 추가
metadata = {
    "feature_schema": "v1_adapt",
    "feature_dim": int(X.shape[-1]),
    "opt_keys": OPT_KEYS,
    "adapt_keys": ADAPT_KEYS if adaptive_enabled else [],
    "seq_len": seq_len,
    "horizon_min": horizon_min,
    "created_at": datetime.now().isoformat(),
}
np.savez(out_path, X=X, y=y, metadata=json.dumps(metadata))

# train.py에서 검증
loaded_meta = json.loads(str(ds["metadata"]))
assert loaded_meta["feature_dim"] == PAST_UNKNOWN_DIM, "Feature dim mismatch!"
```

(반영 완료) 위 메타데이터 저장/검증이 실제 코드에 반영되었습니다.

---

### 10-5. 가드레일 임계값 config 이동

```python
# 현재 하드코딩 (pipeline.py)
hold_thr = 2.5        # 베이시스 HOLD 임계값 (index points)
downgrade_thr = 1.5   # 베이시스 다운그레이드 임계값
atm_spread_pct_thr = 1.5  # ATM spread 임계값
atm_liq_log_thr = 2.0    # ATM 유동성 임계값
```

(반영 완료) 위 값들은 `config.json`의 `prediction.guard_*` 키로 이동되어 운영 중 조정할 수 있습니다.

---

## 11. 우선순위 요약표

| 우선순위 | 항목 | 파일 | 영향 |
|----------|------|------|------|
| 🔴 즉시 | `validate_consistency()` 반환값 누락 | `indicator_integration.py` | 기능 무효화 |
| 🔴 즉시 | `PAST_UNKNOWN_DIM` 상수 실제 dim 불일치 | `constants.py` / `pipeline.py` | Shape 오류 위험 |
| 🔴 즉시 | `_restore_compact_prices()` 범위 검증 누락 | `data_builder.py` | 학습 데이터 오염 |
| 🟡 단기 | `get_prediction()` LLM 순차 실행 (dual_llm) | `pipeline.py` | 레이턴시 2배 |
| 🟡 단기 | `_adaptive_warmed` 재설정 없음 | `pipeline.py` | (반영 완료) df rewind 감지 + reset/rewarm로 정합성 개선 |
| 🟡 단기 | `build_sequence()` 타임스탬프 fallback 노이즈 | `features.py` | 예측 품질 저하 |
| 🟡 단기 | `_classify()` confidence 파라미터 하드코딩 | `predictor.py` | (반영 완료) `prediction.confidence_*`로 튜닝 가능 |
| 🟡 단기 | 가드레일 임계값 하드코딩 | `pipeline.py` | (반영 완료) `prediction.guard_*`로 이동 |
| 🟢 중기 | `get_prediction()` 단일 메서드 분해 | `pipeline.py` | (반영 완료) helper 메서드로 분해 |
| 🟢 중기 | 피처 스키마 버전 관리 자동화 | 신규 `schema.py` | 학습/런타임 정합성 |
| 🟢 중기 | `disagreement_hold` 정책 개선 | `predictor.py` | 예측 품질 |
| 🟢 중기 | 학습/런타임 메타데이터 자동 검증 | `data_builder.py` / `train.py` | (반영 완료) metadata 저장 + 학습 전 검증 |
| 🔵 장기 | Transformer d_ff 256으로 증가 검토 | `model.py` | 표현력 향상 |
| 🔵 장기 | ADAPT 피처 정규화 레이어 추가 | `features.py` / `model.py` | 학습 안정성 |
| 🔵 장기 | 레이블 생성에 no-trade zone 도입 | `data_builder.py` | 실전 적합성 |

---

*본 문서는 정적 코드 분석 결과이며, 학습/백테스트 실험을 통한 실증 검증을 권장합니다.*
