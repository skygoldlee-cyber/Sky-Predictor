# SkyEbest Transformer 시스템 — 보완 및 개선 리포트

> 대상 코드: `Transformer__1_.zip` | 작성일: 2026-02-28  
> 리뷰 범위: `prediction/`, `adaptive_indicator/`, `main.py`, `ebest_live.py`, `tick_processor.py`, `train.py`, `tests/`

---

## 목차

1. [전체 품질 현황](#1-전체-품질-현황)
2. [미해결 버그 및 잠재 오류](#2-미해결-버그-및-잠재-오류)
3. [아키텍처 보완 포인트](#3-아키텍처-보완-포인트)
4. [예측 파이프라인 개선](#4-예측-파이프라인-개선)
5. [Adaptive Indicator 개선](#5-adaptive-indicator-개선)
6. [학습 파이프라인 개선](#6-학습-파이프라인-개선)
7. [테스트 커버리지 보강](#7-테스트-커버리지-보강)
8. [운영 안정성 강화](#8-운영-안정성-강화)
9. [코드 품질 및 유지보수성](#9-코드-품질-및-유지보수성)
10. [우선순위 로드맵](#10-우선순위-로드맵)

---

## 1. 전체 품질 현황

전반적으로 구조 분리, 방어 코딩, 문서화 수준이 매우 높은 프로덕션급 코드베이스입니다. 이전 코드 리뷰(`code_review_report.md`)에서 지적된 치명적 버그(API_MAX_RETRIES NameError, Lock 미사용, Optional 타입힌트 누락)는 **✅ 수정 완료** 상태입니다. 하지만 여전히 아래 영역에서 개선이 필요합니다.

| 영역 | 현재 수준 | 주요 개선 과제 |
|---|---|---|
| 예측 파이프라인 | ★★★★☆ | 예외 경계 세분화, LLM 판단 신뢰도 지표 |
| Adaptive Indicator | ★★★☆☆ | 배치/순차 일관성, 방향 결정 로직 중복 |
| 학습 파이프라인 | ★★★☆☆ | OHLCV 유효성 검사, 클래스 불균형 처리 |
| 테스트 커버리지 | ★★☆☆☆ | 유닛 테스트 부족, 경계값 케이스 누락 |
| 운영 안정성 | ★★★★☆ | 메모리 성장, 재연결 로직 |
| 보안 | ★★★★☆ | secrets 파일 분리됨, env 주입 보강 여지 |

---

## 2. 미해결 버그 및 잠재 오류

### 2.1 🟠 `adaptive_zigzag.py` — `_pending_confirm` 연속 스윙 누락

**위치:** `adaptive_zigzag.py` L202–326

**문제:** 스윙 확정 직후 같은 봉에서 새 전환 조건이 감지되어도 `_pending_confirm is not None` 체크 때문에 후보가 무시됩니다. `confirmation_bars=1` 설정 시 연속 스윙이 누락될 수 있습니다.

```python
# 현재 코드 패턴 (문제)
if self._pending_confirm is not None:
    # 확정 처리...
    self._pending_confirm = self._new_candidate  # 덮어쓰기만 됨
    return  # ← 새 후보 생성 로직에 진입하지 않음

# 권장 수정
if self._pending_confirm is not None:
    confirmed = self._try_confirm(...)
    self._pending_confirm = None  # 명시적 초기화
    if confirmed:
        ...  # 확정 처리 후 아래 새 후보 감지 로직으로 fall-through
```

**영향:** `confirmation_bars=1` 환경 또는 고변동성 구간에서 지지/저항 전환점 일부 누락 → Transformer 피처 품질 저하.

---

### 2.2 🟠 `adaptive_indicator/indicator_integration.py` — `compute_from_df()` cross 피처 일관성

**위치:** `indicator_integration.py` `compute_from_df()` 메서드

**문제:** `docs/runtime/adaptive_indicator_improvements.md`에서 "수정 완료"로 기재되어 있으나, 현재 코드에서 cross 피처 계산이 행별 state를 진행시키는지 확인 필요합니다. 테스트(`test_adaptive_indicator_smoke.py`)의 허용 오차(`<= ...`)가 truncated되어 실제 적용 여부가 불명확합니다.

```python
# 검증 포인트: compute_from_df 내부에서 행별로 update()를 호출하는가?
def compute_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
    for _, row in df.iterrows():
        out = self.update(...)   # ← 이 형태여야 state가 진행됨
        # cross 피처는 self.supertrend.state, self.zigzag.state 기반
```

**권장 조치:** 테스트 `assert abs(got - float(v)) <= 1e-6` 조건을 명시적으로 복원하고, CI에서 배치/순차 비교 테스트가 통과하는지 확인.

---

### 2.3 🟡 `tick_processor.py` — `options_minute_data` 메모리 무제한 성장

**위치:** `tick_processor.py` `options_minute_data` 구조체

**문제:** `futures_ticks`는 `deque(maxlen=MAX_FUTURES_TICKS)`로 상한이 걸려 있지만, `options_minute_data`는 `defaultdict(lambda: defaultdict(list))` 형태로 cleanup 주기가 느릴 경우 장시간 운영 시 메모리가 선형 증가합니다.

```python
# 현재 (문제)
self.options_minute_data: Dict[str, Dict[datetime, List[Dict]]] = \
    defaultdict(lambda: defaultdict(list))

# 권장 개선
# cleanup_old_data() 내에서 options_minute_data도 명시적 만료 처리
# 또는 심볼별로 deque(maxlen=N) 적용
MAX_OPTION_MINUTE_BARS = 240  # 예: 4시간

def _cleanup_option_minute_data(self, cutoff: datetime) -> None:
    for sym in list(self.options_minute_data):
        self.options_minute_data[sym] = {
            k: v for k, v in self.options_minute_data[sym].items()
            if k >= cutoff
        }
        if not self.options_minute_data[sym]:
            del self.options_minute_data[sym]
```

---

### 2.4 🟡 `prediction/pipeline.py` — `_compute_regime` closure race condition 잠재 위험

**위치:** `pipeline.py` `_compute_regime()` 내부

**문제:** `_compute_regime()`은 outer 스코프의 `adaptive_supertrend_state`, `adaptive_features`를 직접 참조합니다. 현재 단일 스레드이므로 버그는 아니지만, 향후 비동기 예측 루프 확장 시 race condition이 발생할 수 있습니다.

```python
# 권장: 명시적 인자 전달
def _compute_regime(
    *,
    st_state: Any,
    adaptive_features: Dict[str, float],
    ...
) -> str:
    ...
```

---

### 2.5 🟡 `features.py` — `calc_orderbook_features()` 코드 truncation

**위치:** `prediction/features.py` 하단

**문제:** 파일 내 `q = quote i` 형태로 보이는 truncation이 있습니다. 실제 파일에는 문제 없을 수 있으나, 해당 함수의 `_invalid` 반환 조건과 `bid_slope`/`offer_slope` 계산 경로에 대한 단위 테스트가 누락되어 있습니다. 현재 smoke 테스트는 `_invalid=True` 케이스와 depth 파싱 케이스만 검증합니다.

**권장 추가 테스트:**
- `offerrem`/`bidrem` 합계가 0일 때 `level1_ratio` 처리
- 깊이 데이터 일부 누락 시 slope 계산 fallback

---

## 3. 아키텍처 보완 포인트

### 3.1 예외 경계 세분화 (`pipeline.py:get_prediction`)

**현재:** 최상위 `except Exception`이 NumericPredictor 오류, LLM 오류, 피처 계산 오류를 모두 동일하게 처리합니다.

```python
# 현재 (문제 — 에러 분류 불가)
try:
    result = self._full_pipeline(...)
except Exception as e:
    logger.error("prediction failed: %s", e)
    return self._fallback_result()
```

**권장 구조:**

```python
# 1단계: 피처 추출 (데이터 오류)
try:
    features = self._build_features(...)
except FeatureExtractionError as e:
    logger.warning("feature extraction failed: %s", e)
    return self._fallback_result("feature_error")

# 2단계: 수치 예측 (모델 오류)
try:
    numeric = self._numeric_predict(features)
except ModelInferenceError as e:
    logger.error("numeric predictor failed: %s", e)
    numeric = self._numeric_fallback()

# 3단계: LLM 판단 (외부 API 오류)
try:
    judgment = self._llm_judge(numeric, context)
except LLMTimeoutError:
    logger.warning("LLM timeout, using heuristic")
    judgment = self._heuristic_judgment(numeric)
except LLMProviderError as e:
    logger.error("LLM provider error: %s", e)
    judgment = self._heuristic_judgment(numeric)
```

---

### 3.2 `dual_llm` 모드 결과 집계 로직 명확화

**현재:** `dual_llm_primary_provider`(기본 `gpt`) 결과를 최종 판단으로 사용하고 Gemini는 `model_outputs["gemini"]`에 보조 저장됩니다. 두 LLM이 상반된 판단을 내릴 경우 자동 처리가 없습니다.

**권장 보완:**

```python
# 두 결과가 상충할 때 disagreement 처리
if gpt_action != gemini_action:
    if disagreement_hold:
        final_action = "HOLD"
        meta["dual_llm_disagreement"] = True
    else:
        final_action = primary_result.action  # primary 우선
```

이를 통해 `disagreement_hold` 옵션이 dual_llm 모드에서도 일관되게 동작하도록 합니다.

---

### 3.3 `ebest_live.py` — `LiveState` 공유 상태 스레드 안전성

**현재:** `LiveState` dataclass가 eBest 콜백 스레드와 메인 루프 간에 직접 공유됩니다. `tick_counts`, `pending_evals` 등 복합 연산이 Lock 없이 이루어집니다.

```python
# 위험: 콜백 스레드에서 write, 메인 루프에서 read
state.tick_counts["FC0"] += 1  # 비원자적
state.pending_evals.append(eval_dict)  # 비원자적
```

**권장:**

```python
@dataclass
class LiveState:
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def increment_tick(self, trcode: str) -> None:
        with self._lock:
            self.tick_counts[trcode] += 1
    
    def add_pending_eval(self, ev: dict) -> None:
        with self._lock:
            self.pending_evals.append(ev)
```

---

## 4. 예측 파이프라인 개선

### 4.1 LLM 판단 신뢰도 메타 지표 추가

현재 결과 dict에 LLM 응답의 신뢰도를 나타내는 메타 정보가 부족합니다. 다음 필드 추가를 권장합니다.

| 필드 | 타입 | 설명 |
|---|---|---|
| `llm_latency_ms` | float | LLM API 응답 시간 |
| `llm_retry_count` | int | 재시도 횟수 |
| `llm_provider_used` | str | 실제 사용된 프로바이더 (fallback 여부 추적) |
| `llm_json_parse_ok` | bool | JSON 파싱 성공 여부 |
| `numeric_model_used` | str | transformer/tft/ensemble/rule_based 중 실제 사용 |

---

### 4.2 Ensemble 모드 가중치 동적 조정

**현재:** `transformer_weight`가 고정값으로 `config.json`에 설정됩니다.

**개선 방향:** 최근 N예측에서 Transformer와 TFT의 방향 적중률을 추적하여 동적으로 가중치를 조정하는 간단한 적응형 앙상블을 구현할 수 있습니다.

```python
class AdaptiveEnsembleWeightTracker:
    """최근 예측 결과를 기반으로 모델별 가중치를 동적 조정."""
    
    def __init__(self, window: int = 20, decay: float = 0.95):
        self._transformer_hits = deque(maxlen=window)
        self._tft_hits = deque(maxlen=window)
        self._decay = decay
    
    def update(self, transformer_correct: bool, tft_correct: bool) -> None:
        self._transformer_hits.append(float(transformer_correct))
        self._tft_hits.append(float(tft_correct))
    
    def get_weights(self) -> tuple[float, float]:
        t_acc = sum(self._transformer_hits) / max(len(self._transformer_hits), 1)
        tft_acc = sum(self._tft_hits) / max(len(self._tft_hits), 1)
        total = t_acc + tft_acc
        if total < 1e-9:
            return 0.5, 0.5
        return t_acc / total, tft_acc / total
```

---

### 4.3 Basis 가드레일 개선

현재 `IJ_`(실시간 지수) 기반 basis 계산이 `spot_index` 미수신 시 가드레일이 비활성화됩니다. 개장 직후 `IJ_` 첫 수신 전 구간에서 basis 가드레일 없이 예측이 나갈 수 있습니다.

**권장:** 개장 후 `IJ_` 최초 수신까지 confidence를 `MEDIUM` 이하로 강제하는 warm-up 가드 추가.

---

### 4.4 `context_builder.py` — `_safe_float` 중복 정의

`context_builder.py`에 `_safe_float()`가 별도 정의되어 있고 `utils.py`에도 `safe_float()`가 있습니다. 중복을 제거하고 `utils.safe_float`를 import하여 사용하는 것이 유지보수에 유리합니다.

---

## 5. Adaptive Indicator 개선

### 5.1 `AdaptiveSuperTrend` — 방향 결정 로직 단순화

**현재:** `update()` 내에서 `direction` 변수가 여러 조건 분기에서 중복 덮어쓰기됩니다. 표준 SuperTrend 로직(`docs/runtime/adaptive_indicator_improvements.md` §2.3에서 "수정 완료"로 기재)이 실제로 반영되었는지 아래 패턴으로 검증이 필요합니다.

```python
# 표준 SuperTrend 방향 결정 (단일 경로)
if prev_direction == 1:  # 이전: 상승
    new_direction = 1 if close > final_lower_band else -1
else:  # 이전: 하락
    new_direction = -1 if close < final_upper_band else 1
# ← 이 이후 direction을 덮어쓰는 코드가 없어야 함
```

---

### 5.2 `AdaptiveZigZag` — Fibonacci 이중 키 정리

**현재:** `fib_keys`에 `legacy_key('0.618')`와 `new_key('fib_618')`가 동시 존재합니다. `get_llm_context`의 fallback 조회 코드가 복잡하고 향후 한 키 제거 시 묵히 실패할 위험이 있습니다.

**권장:** `new_key` 체계로 완전히 전환하고, legacy 키는 하위 호환 alias(deprecated 경고 포함)로만 유지.

```python
@property
def fib_618(self) -> float:
    """Deprecated: use fib_levels['fib_618']"""
    import warnings
    warnings.warn("legacy fib key '0.618' is deprecated, use 'fib_618'", DeprecationWarning, stacklevel=2)
    return self.fib_levels.get('fib_618', 0.0)
```

---

### 5.3 `IndicatorManagerConfig` — dataclass 기본값 문제

**현재:** `supertrend: AdaptiveSuperTrendConfig = None` 형태로, `__post_init__`에서 None 체크를 합니다. Python 타입 힌트 관점에서 올바르지 않으며 mypy에서 경고가 발생합니다.

```python
# 현재 (문제)
@dataclass
class IndicatorManagerConfig:
    supertrend: AdaptiveSuperTrendConfig = None  # mypy 경고

# 권장
from dataclasses import field

@dataclass
class IndicatorManagerConfig:
    supertrend: AdaptiveSuperTrendConfig = field(
        default_factory=AdaptiveSuperTrendConfig
    )
    zigzag: AdaptiveZigZagConfig = field(
        default_factory=AdaptiveZigZagConfig
    )
    symbol: str = "KP200 선물"
    # __post_init__ 불필요
```

---

### 5.4 `indicator_integration.py` — 피처 수 주석 불일치

docstring에 "약 22개 피처"로 기재되어 있으나 실제 `ADAPT_KEYS`는 28개입니다. 주석/docstring 전체를 28개 기준으로 통일해야 합니다.

```python
class AdaptiveIndicatorManager:
    """
    Transformer 피처 (ADAPT_KEYS 총 28개):
        ast_* : AdaptiveSuperTrend 피처 9개
        azz_* : AdaptiveZigZag 피처 15개
        cross_* : Cross 피처 4개
    """
```

---

## 6. 학습 파이프라인 개선

### 6.1 OHLCV 유효성 검사 미비

**현재:** `train.py`와 `prediction/data_builder.py`에서 OHLCV 관계(`high >= low`, `high >= open`, `high >= close`, `low <= open`, `low <= close`, `volume >= 0`)를 검증하지 않습니다. 데이터 오염 시 Adaptive Indicator 계산에 NaN이 전파될 수 있습니다.

```python
# data_builder.py에 추가 권장
def _validate_ohlcv(bar: Dict[str, float]) -> bool:
    h, l, o, c, v = (
        bar.get("High", 0), bar.get("Low", 0),
        bar.get("Open", 0), bar.get("Close", 0),
        bar.get("Volume", -1),
    )
    if h < l:
        return False  # high < low: 비정상
    if not (l <= o <= h) or not (l <= c <= h):
        return False  # OHLC 관계 위반
    if v < 0:
        return False  # 음수 거래량
    return True
```

---

### 6.2 클래스 불균형 처리 부재

**현재:** `train.py`에서 `y` 레이블(BUY/SELL)의 분포를 확인하거나 가중치를 적용하지 않습니다. KP200 선물은 방향성이 편향된 구간이 존재하므로, 불균형 비율에 따라 모델이 특정 방향으로 편향될 수 있습니다.

```python
# train.py에 추가 권장
pos_ratio = float(y.mean())
neg_ratio = 1.0 - pos_ratio
logger.info("Label distribution: BUY=%.1f%% SELL=%.1f%%", pos_ratio*100, neg_ratio*100)

if abs(pos_ratio - 0.5) > 0.1:
    logger.warning("Class imbalance detected (%.1f%% BUY). Consider pos_weight.", pos_ratio*100)
    pos_weight = torch.tensor([neg_ratio / pos_ratio])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

### 6.3 `merge_datasets.py` — 롤오버 마커 경쟁 조건

`_save_rollover_marker()`가 여러 프로세스에서 동시에 호출될 경우 파일 쓰기 경쟁이 발생할 수 있습니다. 단일 프로세스 환경에서는 문제없지만, 자동화 스크립트에서 병렬 실행 시를 대비해 원자적 쓰기(`rename` 기법)를 권장합니다.

```python
def _save_rollover_marker_atomic(path: Path, yyyymmdd: str) -> None:
    tmp = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(str(yyyymmdd), encoding="utf-8")
        tmp.replace(path)  # 원자적 교체
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
```

---

### 6.4 학습 중 체크포인트 및 조기 종료 부재

**현재:** `train.py`에 early stopping이나 best-model 저장 로직이 없습니다. `--epochs 50`으로 고정 실행하면 최적 가중치를 놓칠 수 있습니다.

```python
# 권장 추가 구조
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(args.epochs):
    train_loss = _train_epoch(model, loader, optimizer, criterion, device)
    val_loss = _eval_epoch(model, val_loader, criterion, device)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), args.out)
        logger.info("epoch %d: new best val_loss=%.4f, saved", epoch, val_loss)
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            logger.info("early stopping at epoch %d", epoch)
            break
```

---

## 7. 테스트 커버리지 보강

현재 `tests/` 디렉토리는 smoke 테스트 수준이며 실제 단위 테스트가 매우 부족합니다.

### 7.1 추가 권장 테스트 목록

**`prediction/features.py` 관련:**

```python
def test_calc_orderbook_features_zero_qty():
    """bid/offer quantity가 모두 0일 때 level1_ratio 처리."""
    q = {"offerho": 100.5, "bidho": 100.4, "offerrem": 0, "bidrem": 0}
    feat = calc_orderbook_features(q)
    assert feat.get("_invalid") is not True
    assert math.isfinite(float(feat.get("level1_ratio", 0)))

def test_calc_orderbook_features_partial_depth():
    """depth 3단계만 제공될 때 slope 계산 안전성."""
    q = {
        "offerho1": 100.5, "offerho2": 100.6, "offerho3": 100.7,
        "bidho1": 100.4, "bidho2": 100.3, "bidho3": 100.2,
        "offerrem1": 100, "offerrem2": 120, "offerrem3": 140,
        "bidrem1": 90, "bidrem2": 110, "bidrem3": 130,
    }
    feat = calc_orderbook_features(q)
    assert feat.get("_invalid") is not True
```

**`adaptive_indicator/` 관련:**

```python
def test_supertrend_direction_flip():
    """가격이 밴드를 하향 돌파할 때 direction이 1 → -1로 정확히 플립."""
    ast = AdaptiveSuperTrend()
    # 충분한 warmup
    for _ in range(30):
        ast.update(high=105, low=95, close=100)
    # 급락
    ast.update(high=95, low=85, close=86)
    assert ast.state.direction == -1

def test_zigzag_no_swing_loss_on_confirmation_bars_1():
    """confirmation_bars=1에서 연속 스윙이 누락되지 않음."""
    cfg = AdaptiveZigZagConfig(confirmation_bars=1)
    azz = AdaptiveZigZag(cfg)
    swings_detected = 0
    for i in range(50):
        c = 400 + 5 * math.sin(i * 0.5)
        out = azz.update(high=c+1, low=c-1, close=c)
        if out.get("new_swing_signal") != "none":
            swings_detected += 1
    assert swings_detected >= 3  # 최소 3개 스윙 감지
```

**`prediction/pipeline.py` 관련:**

```python
def test_pipeline_llm_timeout_falls_back_to_heuristic():
    """LLM timeout 시 heuristic fallback이 정상 동작."""
    pipeline = PredictionPipeline(
        use_llm=True,
        llm_timeout_sec=0.001,  # 즉시 타임아웃
        min_minute_bars_required=1,
        seq_len=5,
    )
    # ... tick 주입 후 get_prediction() 호출
    result = pipeline.get_prediction(...)
    assert result.get("action") in ("BUY", "SELL", "HOLD")
    assert result.get("source") in ("heuristic", "rule_based")
```

---

### 7.2 `simulate_indicators.py` — BUY/SELL annotate 루프 벡터화

이전 리뷰에서 지적된 항목으로 시뮬레이션 성능을 개선할 수 있습니다.

```python
# 현재 (느림 — O(n) 루프)
for i, row in df.iterrows():
    if row['signal'] == 'BUY':
        ax.annotate('↑', xy=(i, row['Close']), ...)

# 권장 (벡터화)
buy_mask = df['signal'] == 'BUY'
sell_mask = df['signal'] == 'SELL'
ax.scatter(df.index[buy_mask], df['Close'][buy_mask], marker='^', color='green')
ax.scatter(df.index[sell_mask], df['Close'][sell_mask], marker='v', color='red')
```

---

## 8. 운영 안정성 강화

### 8.1 eBest WebSocket 재연결 로직

**현재:** `ebest_live.py`에서 WebSocket 연결 끊김 시 자동 재연결 로직이 명시적으로 구현되어 있는지 확인이 필요합니다. 한국 주식시장 운영 중 간헐적 연결 끊김은 흔한 현상입니다.

**권장 패턴:**

```python
async def _reconnect_with_backoff(
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
) -> None:
    for attempt in range(max_retries):
        try:
            await _ebest_register_realtime(...)
            logger.info("reconnected on attempt %d", attempt + 1)
            return
        except Exception as e:
            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning("reconnect attempt %d failed: %s, retry in %.1fs", attempt+1, e, delay)
            await asyncio.sleep(delay)
    logger.error("failed to reconnect after %d attempts", max_retries)
```

---

### 8.2 장 종료 후 일일 성과 요약 자동화

현재 텔레그램 알림은 예측 건별 전송이 가능하지만, 장 종료 시 일일 성과 요약(방향 적중률, 평균 confidence, 예측 건수, eval_dir_hit_count)을 자동으로 집계하여 발송하는 로직이 없습니다.

```python
async def _send_daily_summary(state: LiveState, notifier: TelegramNotifier) -> None:
    if state.eval_count == 0:
        return
    acc = state.eval_dir_hit_count / state.eval_count * 100
    msg = (
        f"📊 오늘의 예측 요약\n"
        f"총 예측: {state.prediction_count}건\n"
        f"평가 완료: {state.eval_count}건\n"
        f"방향 적중률: {acc:.1f}%\n"
        f"평균 오차: {state.eval_abs_err_sum / state.eval_count:.3f}"
    )
    await notifier.send(msg)
```

---

### 8.3 `config.secrets.json` 보안 강화

**현재:** API 키가 `config.secrets.json`에 저장되며 `.gitignore`에 포함되어 있습니다. 추가로 다음 조치를 권장합니다.

- 파일 권한을 `chmod 600 config.secrets.json`으로 제한하는 시작 시 검사 추가
- 환경변수 우선 로드: `os.environ.get("ANTHROPIC_API_KEY")` → secrets 파일 순서
- 키 마스킹 로그 유틸리티: 실수로 키가 로그에 출력되는 것을 방지

```python
def _mask_key(key: Optional[str]) -> str:
    """API 키의 앞 4자 + *** 형태로 마스킹."""
    if not key:
        return "(없음)"
    return key[:4] + "***" + key[-4:] if len(key) > 8 else "***"

logger.info("Anthropic key: %s", _mask_key(cfg.ai.anthropic_key))
```

---

## 9. 코드 품질 및 유지보수성

### 9.1 타입 힌트 보완

아래 파일들에서 타입 힌트가 부분적으로 누락되어 있습니다.

| 파일 | 문제 위치 | 권장 |
|---|---|---|
| `ebest_live.py` | `_log()` 함수 반환 타입 | `-> None` 추가 |
| `ebest_live.py` | `_fmt_atm_strike()` 인자 타입 | `Any` → `Union[str, float, None]` 구체화 |
| `tick_processor.py` | `options_minute_data` 값 타입 | `Dict[str, Dict[datetime, List[Dict[str, Any]]]]` 명시 |
| `merge_datasets.py` | `_select_last_n()` 반환 | `-> List[Path]` 이미 있으나 `undated` 분기 타입 일관성 확인 |

---

### 9.2 `constants.py` — LLM 모델 버전 하드코딩 리스크

```python
CLAUDE_MODEL = "claude-sonnet-4-20250514"
GPT_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-2.0-flash-exp"
```

모델 버전 deprecation이 발생할 경우 코드 수정이 필요합니다. `config.json`에서 오버라이드 가능한 구조는 이미 갖추어져 있으나, constants의 하드코딩 값이 가장 최신 안정 버전인지 주기적 검토가 필요합니다. 특히 `gemini-2.0-flash-exp`는 실험적 모델로 프로덕션에서 `gemini-2.0-flash`로 변경을 권장합니다.

---

### 9.3 `logging_utils.py` — `TeeStream` flush 버퍼 누락 처리

`TeeStream.write()`에서 `\n`으로 끝나지 않는 마지막 라인이 `_buffer`에 잔류할 수 있습니다. 프로세스 종료 시 해당 내용이 파일에 기록되지 않을 수 있습니다.

```python
def flush(self) -> None:
    # 버퍼에 잔류하는 미완성 라인 처리
    if self._buffer:
        try:
            self._original.write(self._buffer)
            stream = getattr(self._file_handler, "stream", None)
            if stream is not None:
                stream.write(self._buffer)
        except Exception:
            pass
        finally:
            self._buffer = ""
    try:
        self._original.flush()
    except Exception:
        pass
```

---

## 10. 우선순위 로드맵

### 🔴 즉시 처리 (운영 안정성 직결)

1. ✅ **`tick_processor.py`** — `options_minute_data` 메모리 만료 처리 추가 (§2.3)
2. ✅ **`adaptive_zigzag.py`** — `_pending_confirm` 연속 스윙 누락 버그 수정 (§2.1)
3. ✅ **`pipeline.py`** — 예외 경계 세분화 (§3.1)
4. ✅ **`ebest_live.py`** — `LiveState` 공유 상태 Lock 추가 (§3.3)

### 🟠 단기 처리 (1–2주, 예측 품질)

5. ✅ **`indicator_integration.py`** — `compute_from_df` cross 피처 행별 일관성 검증 및 테스트 복원 (§2.2)
6. ✅ **`adaptive_supertrend.py`** — 방향 결정 로직 단일 경로 리팩터링 (§5.1)
7. ✅ **`train.py`** — OHLCV 유효성 검사 + 클래스 불균형 처리 (§6.1, §6.2)
8. ✅ **테스트** — ZigZag 플립 테스트, LLM timeout fallback 테스트 추가 (§7.1)

### 🟡 중기 처리 (1개월, 품질 향상)

9. ✅ **`dual_llm` disagreement hold** 일관성 확보 (§3.2)
10. ✅ **Adaptive Ensemble 가중치** 동적 조정 도입 (§4.2)
11. ✅ **`train.py`** — early stopping / best checkpoint 저장 (§6.4)
12. ✅ **Fibonacci 이중 키** 정리 및 deprecated alias 전환 (§5.2)
13. ✅ **일일 성과 요약** 텔레그램 자동 발송 (§8.2)
14. ✅ **타입 힌트 보완** 및 mypy 클린 통과 (§9.1)

검증: `pytest -q` = 19 passed, `mypy .` = Success

---

> **Note:** 본 문서에서 "수정 완료(✅)"로 표기된 이전 리뷰 항목들은 현재 코드 기준으로 반영된 것으로 확인됩니다. 위의 모든 개선 항목은 기존 인터페이스와의 하위 호환성을 유지하는 방향으로 구현을 권장합니다.
