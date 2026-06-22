# Transformer 프로젝트 — 보완 및 개선점 종합 보고서

> **기준일**: 2026-03-06  
> **대상 코드베이스**: KP200 선물 Transformer + TFT + LLM 예측 파이프라인  
> **파일 수**: 173개 (Python 소스 기준 ~25개 핵심 모듈, ~300K 라인)

---

## 목차

1. [즉시 수정 필요 (Critical)](#1-즉시-수정-필요-critical)
2. [구조·설계 개선 (Architecture)](#2-구조설계-개선-architecture)
3. [피처 엔지니어링 보완 (Features)](#3-피처-엔지니어링-보완-features)
4. [학습 파이프라인 개선 (Training)](#4-학습-파이프라인-개선-training)
5. [LLM 판단 레이어 개선 (LLM Judge)](#5-llm-판단-레이어-개선-llm-judge)
6. [운영·신뢰성 개선 (Ops)](#6-운영신뢰성-개선-ops)
7. [테스트 커버리지 확대 (Tests)](#7-테스트-커버리지-확대-tests)
8. [코드 품질·정리 (Code Quality)](#8-코드-품질정리-code-quality)
9. [우선순위 요약 매트릭스](#9-우선순위-요약-매트릭스)

---

## 1. 즉시 수정 필요 (Critical)

### 1-1. `option_flow_features.py` — 완전 미연결 모듈 🔴

**파일**: `prediction/option_flow_features.py`

**문제**: 모듈이 구현되어 있지만 `pipeline.py`, `predictor.py`, `features.py`, `ebest_live.py`, `main.py` 어디에서도 `import`되거나 호출되지 않는다. ATM 옵션의 실시간 흐름 피처(`optm_call_ret`, `optm_straddle_ret` 등)가 실제 예측에 전혀 기여하지 않는 상태이다.

**영향**: OPT_KEYS_V2 확장 피처셋(`option_feature_set: "v2"`)을 설정해도 `option_flow_features` 계산 결과가 파이프라인에 주입되지 않아 v2 피처의 일부가 항상 0으로 채워질 가능성이 있다.

**수정 방향**:
```python
# prediction/pipeline.py - _build_option_snapshot_safe() 내부
from .option_flow_features import build_option_flow_features

if self._option_feature_set == "v2":
    flow = build_option_flow_features(
        calls=tp.call_options,
        puts=tp.put_options,
        underlying_price=current_price,
        call_minute_df=tp.get_option_minute_df(atm_call_sym),
        put_minute_df=tp.get_option_minute_df(atm_put_sym),
    )
    opt_snap.update(flow)
```

---

### 1-2. `dual_llm` 설정의 default 불일치 🔴

**파일**: `prediction/pipeline.py` L119, `main.py` L456

**문제**: `PredictionPipeline.__init__`의 기본값은 `dual_llm=False`이지만, `config.json`은 `"dual_llm": true`로 설정되어 있다. `main.py`의 두 번째 `PredictionPipeline` 생성 경로(L3750)에서 `dual_llm` 인자가 누락되어 있어 항상 `False`로 실행된다.

**수정 방향**:
```python
# main.py L3750 근처 — 두 번째 파이프라인 생성 블록에 추가
dual_llm=bool(getattr(config.prediction, "dual_llm", False)),
dual_llm_primary_provider=str(getattr(config.prediction, "dual_llm_primary_provider", "gpt")),
```

---

### 1-3. `data_builder.py` — 전역 키 재정의 (모듈 오염) 🟠

**파일**: `prediction/data_builder.py` L46-48

**문제**: `features.py`에서 import한 `OB_KEYS`, `CD_KEYS`, `ADAPT_KEYS`를 곧바로 `list()`로 재정의하여 모듈 스코프를 오염시킨다. 이 파일을 다른 모듈이 import할 때 의도치 않은 심볼 충돌이 발생할 수 있다.

```python
# 현재 — 문제 있음
OB_KEYS = list(OB_KEYS)   # 모듈 스코프 재정의
CD_KEYS = list(CD_KEYS)
ADAPT_KEYS = list(ADAPT_KEYS)
```

**수정 방향**: 지역변수나 명시적 별칭으로 처리한다.
```python
_OB_KEYS = list(OB_KEYS)
_CD_KEYS = list(CD_KEYS)
_ADAPT_KEYS = list(ADAPT_KEYS)
```

---

### 1-4. `weights_selector.py` — 만기일 이후 freeze 해제 누락 🟠

**파일**: `prediction/weights_selector.py`

**문제**: 만기 주(目) Mon~Thu 동안만 freeze하도록 `in_freeze_window` 조건을 두었지만, **만기일(두 번째 목요일) 당일**은 `expiry_dt.date()`와 동일하여 freeze 상태가 유지된다. 만기 당일 오후 장마감 이후(15:30~)에는 freeze를 해제해야 하지만 시간 단위 체크가 없다.

**수정 방향**:
```python
# 만기일 15:30 이후 freeze 해제
if now.date() == expiry_dt.date() and now.hour >= 15 and now.minute >= 30:
    in_freeze_window = False
```

---

## 2. 구조·설계 개선 (Architecture)

### 2-1. `pipeline.py` — 2800라인 God Method `get_prediction()` 분리

**파일**: `prediction/pipeline.py` L2414-끝

**현황**: `get_prediction()`은 데이터 수집 → 피처 빌드 → 수치 예측 → 가드레일 → LLM 판단 → 결과 조립까지 단일 메서드에서 담당한다. 이미 내부 `_prepare_prediction_inputs()`, `_run_numeric_prediction_and_guardrails()` 등으로 일부 위임되었지만 여전히 `get_prediction()` 자체가 200라인 이상이다.

**개선 방향**: 아래 단계로 최종 조립만 남기고 각 단계를 전담 메서드로 완전 분리한다.

```
get_prediction()
├── _prepare_inputs()         → (df, adaptive, regime)
├── _run_numeric()            → (t_res, prob, signal)
├── _run_guardrails()         → (signal, confidence)
├── _run_llm()                → LLMJudgment
└── _assemble_output()        → Dict[str, Any]
```

---

### 2-2. `main.py` — 3750라인 단일 파일 모듈화

**파일**: `main.py`

**현황**: 3750라인 이상의 단일 파일에 CLI 파싱, GUI 로직, 파이프라인 팩토리, 실행 루프, 리플레이 로직이 혼재한다. 주석에도 "분리를 권장"(MNT-01)이라고 명시되어 있지만 아직 실행되지 않았다.

**개선 방향**:
```
main.py           → 진입점 (50라인 이내)
app_factory.py    → PredictionPipeline 팩토리
cli_runner.py     → CLI 모드 루프
replay_runner.py  → 리플레이 모드
gui_runner.py     → GUI 모드 (PySide6 관련)
```

---

### 2-3. `train.py` / `train_tft.py` — `set_seed()` 중복 제거

**파일**: `train.py`, `train_tft.py`

**현황**: 두 파일 모두 동일한 `set_seed()` 함수를 정의하고 있다.

**개선 방향**: `utils.py`에 `set_seed()`를 추가하고 두 파일에서 `from utils import set_seed`로 공유한다.

---

### 2-4. `NumericPredictor` 인터페이스 — 앙상블 가중치 외부 노출

**파일**: `prediction/predictor.py`

**현황**: `AdaptiveEnsembleWeightTracker`가 `NumericPredictor` 내부에 private하게 관리되어 있어 외부에서 현재 가중치 상태를 모니터링하기 어렵다. Telegram 알림이나 대시보드에서 "Transformer 55% / TFT 45%"와 같은 정보를 표시할 수 없다.

**개선 방향**: `NumericPredictor.get_ensemble_weights() -> dict` 메서드를 추가하고 `PredictionPipeline.get_metrics()`에 포함시킨다.

---

### 2-5. `kospi_indicators` 라이브러리 — 버전 동기화 체크 부재

**파일**: `kospi_indicators/pyproject.toml`, `prediction/pipeline.py`

**현황**: `kospi_indicators` 패키지가 별도 pyproject.toml을 가진 로컬 패키지이므로 버전이 맞지 않아도 런타임에 조용히 오작동할 수 있다. 특히 `ADAPT_KEYS` 목록이 지표 라이브러리와 파이프라인 간에 다를 경우 피처 차원 불일치가 발생한다.

**개선 방향**: 파이프라인 초기화 시점에 버전 체크를 수행한다.
```python
# prediction/pipeline.py __init__
import kospi_indicators
from kospi_indicators import EXPECTED_ADAPT_DIM
assert len(ADAPT_KEYS) == EXPECTED_ADAPT_DIM, \
    f"kospi_indicators 버전 불일치: 예상 {EXPECTED_ADAPT_DIM}, 실제 {len(ADAPT_KEYS)}"
```

---

## 3. 피처 엔지니어링 보완 (Features)

### 3-1. `calc_orderbook_features()` — `bid_slope` / `offer_slope` 계산 보완

**파일**: `prediction/features.py`

**현황**: `bid_slope`와 `offer_slope`는 호가 잔량의 1~5레벨 선형 기울기를 계산하지만, 레벨 데이터가 없는 경우(FH0 스키마 단순 버전) 0으로 fallback된다. 0이 "기울기 없음"과 "데이터 없음"을 구분하지 못한다.

**개선 방향**: 데이터 불충분 시 `NaN`을 사용하고, `build_sequence()`에서 `NaN`을 전 단계 유효값으로 forward-fill한다.

---

### 3-2. `build_time_features()` — 세션 외 시간대 피처 처리

**파일**: `prediction/time_features.py`

**현황**: `tod_sin`/`tod_cos`가 09:00~15:30 기준으로만 계산되므로, 장 전 사전 예측이나 리플레이 시 세션 밖 시각이 들어오면 `frac`이 음수 또는 1.0 초과가 되어 `max(0, min(1, ...))` clamp로 왜곡된 값이 생긴다.

**개선 방향**: 세션 외 시각은 `is_session = 0`으로 마스킹하고, 세션 내에서만 sin/cos를 계산한다.

---

### 3-3. `features.py` — `OPT_KEYS` 버전 분기 일관성

**파일**: `prediction/features.py`, `prediction/pipeline.py`

**현황**: `OPT_KEYS = OPT_KEYS_V1`로 backward-compat alias가 설정되어 있지만, `pipeline.py`에서 `opt_feature_set`에 따라 `get_opt_keys("v2")`를 호출하는 경로와 직접 `OPT_KEYS`를 참조하는 경로가 혼재하여 런타임 피처 차원이 달라질 수 있다.

**수정 방향**: `OPT_KEYS` 전역 변수 참조를 모두 `get_opt_keys(self._option_feature_set)`으로 교체하고 전역 alias를 제거한다.

---

### 3-4. `AdaptiveZigZag` — `_find_nearest_sr()` 0.0 반환 처리 불일치

**파일**: `kospi_indicators/kospi_indicators/adaptive_zigzag.py`

**현황**: 스윙이 없을 때 `support_dist_pct`와 `res_dist_pct`가 0.0으로 반환된다. downstream에서 `> 0` 체크로 처리하기로 했지만, `features.py`의 `ADAPT_KEYS` 조립 코드가 0.0을 그대로 사용하여 "지지선 0%"로 오해될 수 있다.

**개선 방향**: 0.0 반환 시 `NaN`으로 처리하고 `build_sequence()`에서 별도 마스크 피처를 추가하거나, `float('nan')`을 -1.0으로 convention화하여 모델이 학습할 수 있게 한다.

---

## 4. 학습 파이프라인 개선 (Training)

### 4-1. `data_builder.py` — 레이블 누수(Label Leakage) 위험

**파일**: `prediction/data_builder.py`

**현황**: 5분 후 가격을 레이블로 사용할 때, feature 시퀀스의 마지막 봉과 레이블 계산 기준 봉이 겹칠 수 있다. 특히 `horizon_sec=300`(5분)이고 `seq_len=60`(60분 봉)이면 시퀀스 안에 이미 레이블 구간의 일부가 포함된다.

**개선 방향**: feature 시퀀스의 마지막 타임스탬프와 레이블 기준 타임스탬프 사이에 최소 `horizon_sec` 간격을 강제로 삽입한다.
```python
assert label_ts >= seq_end_ts + timedelta(seconds=HORIZON_SEC), \
    f"Label leakage detected: seq_end={seq_end_ts}, label={label_ts}"
```

---

### 4-2. `train.py` — Focal Loss `pos_weight` 자동 계산 부재

**파일**: `train.py`

**현황**: `--pos-weight` 인자가 기본값 `1.0`으로 고정되어 있다. KP200 선물 방향 예측에서 상승/하락 비율이 50:50이 아닌 경우(불균형 레이블) Focal Loss의 `pos_weight`를 데이터에서 자동 계산해야 한다.

**개선 방향**:
```python
# train.py - run() 초반
if args.pos_weight is None or args.pos_weight == 1.0:
    pos_ratio = float(y_np.mean())
    if 0 < pos_ratio < 1:
        auto_pw = (1.0 - pos_ratio) / pos_ratio
        logger.info("Auto pos_weight=%.3f (pos_ratio=%.3f)", auto_pw, pos_ratio)
        args.pos_weight = auto_pw
```

---

### 4-3. `train_tft.py` — Validation Split 무작위 분할 문제

**파일**: `train_tft.py`

**현황**: 학습/검증 데이터를 `random_split` 또는 비율 슬라이싱으로 나누는 구조이므로 시계열 순서가 보장되지 않는다. 미래 데이터가 학습에 포함되어 과적합이 발생할 수 있다.

**개선 방향**: 시간 순서를 유지하는 walk-forward split을 사용한다.
```python
split_idx = int(len(X) * (1 - val_ratio))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]
```

---

### 4-4. `merge_datasets.py` — 스키마 불일치 병합 경고 부재

**파일**: `merge_datasets.py`

**현황**: 다른 날짜의 `.npz` 파일을 병합할 때 `schema_version` 필드를 확인하지 않는다. v1 피처셋으로 만든 데이터와 v2 피처셋 데이터를 병합하면 `X` 차원이 달라져 학습이 실패하거나 잘못 진행된다.

**개선 방향**: 병합 전 `metadata.schema_version`을 검증하고 불일치 시 에러를 발생시킨다.

---

### 4-5. `run_experiments.py` — wandb / MLflow 연동 부재

**파일**: `run_experiments.py`

**현황**: 실험 결과를 로컬 JSON으로만 저장한다. 여러 하이퍼파라미터 실험을 비교하거나 학습 곡선을 추적하기 어렵다.

**개선 방향**: `wandb` 또는 `mlflow` 의존성을 선택적으로 추가하여 실험 추적 기능을 옵션으로 제공한다.

---

## 5. LLM 판단 레이어 개선 (LLM Judge)

### 5-1. `llm_judge.py` — Gemini Fallback 모델 목록 미완성

**파일**: `prediction/llm_judge.py`, `constants.py`

**현황**: `GEMINI_FALLBACK_MODELS = ("gemini-2.5-flash", "gemini-2.0-flash")`만 있고 `gemini-1.5-pro`, `gemini-1.5-flash` 등 하위 호환 모델이 없다. 새 모델 API가 점검 중일 때 fallback 실패 시 LLM 판단이 완전히 누락된다.

**개선 방향**:
```python
GEMINI_FALLBACK_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
)
```

---

### 5-2. `context_builder.py` — `adaptive_context` 구조화 부재

**파일**: `prediction/context_builder.py`

**현황**: `adaptive_context`가 단순 문자열로 LLM 프롬프트에 주입된다. SuperTrend 방향, ZigZag 스윙 위치 등의 중요 신호가 구조화되지 않아 LLM이 파싱하기 어렵다.

**개선 방향**: `adaptive_features` dict를 JSON 블록으로 직렬화하여 프롬프트에 포함한다.
```python
"ADAPTIVE_SIGNALS": {
    "supertrend_direction": "UP",
    "dist_from_band_pct": 0.35,
    "zigzag_last_swing": "HIGH",
    "support_dist_pct": 1.2,
    "trend_agreement": 1
}
```

---

### 5-3. `pipeline.py` — LLM 응답 캐시 만료 정책 부재

**파일**: `prediction/pipeline.py`

**현황**: LLM 응답이 캐시되지만 캐시 만료 로직이 별도로 없고 `llm_min_interval_sec`에 의존한다. 시장이 급변(전일 대비 2% 이상 갭)할 때도 동일 캐시가 사용될 수 있다.

**개선 방향**: 가격 변화율이 임계치를 초과하면 캐시를 강제 무효화하는 로직을 추가한다.
```python
if abs(current_price - self._last_llm_price) / self._last_llm_price > 0.005:
    self._last_llm_ts = None  # cache invalidate
```

---

### 5-4. `dual_llm` 모드 — `disagreement_hold` 로직 강화 필요

**파일**: `prediction/pipeline.py` L2354-2368

**현황**: `disagreement_hold`가 활성화되어 있으면 GPT와 Gemini 의견이 다를 때 HOLD로 처리한다. 그러나 양측이 모두 확률 0.5에 수렴하는 "불확실 동의" 상황도 강한 신호로 처리될 수 있다.

**개선 방향**: 두 모델의 `prob` 평균이 `[0.45, 0.55]` 구간에 있으면 신호 강도를 LOW로 강등한다.

---

## 6. 운영·신뢰성 개선 (Ops)

### 6-1. `ebest_live.py` — `LiveState` 락 범위 과다

**파일**: `ebest_live.py`

**현황**: `LiveState._lock`이 단일 전역 락으로 tick_counts, realtime_response_count, option 관련 상태 등 모든 필드를 직렬화한다. 고빈도 틱 처리 상황에서 lock contention이 발생할 수 있다.

**개선 방향**: 필드 그룹별로 락을 분리한다 (tick 카운터 전용 `_tick_lock`, 옵션 상태 전용 `_option_lock`).

---

### 6-2. `tick_processor.py` — `options_minute_data` 메모리 정리 주기

**파일**: `tick_processor.py`

**현황**: `_options_minute_sweep_counter`를 이용한 정리 로직이 있지만 sweeping 주기(counter 기반)가 틱 빈도에 종속된다. 장중 옵션 틱이 적을 때는 메모리가 장시간 축적될 수 있다.

**개선 방향**: `time.time()` 기반 최소 sweep 간격을 추가한다.
```python
if time.time() - self._options_minute_last_sweep_epoch > 300:  # 5분마다
    self._sweep_old_options_minute()
```

---

### 6-3. `telegram_notifier.py` — 메시지 전송 실패 시 재시도 큐 부재

**파일**: `telegram_notifier.py`

**현황**: Telegram API 호출 실패 시 `logger.warning`만 하고 메시지를 버린다. 네트워크 순단 상황에서 예측 결과 알림이 유실된다.

**개선 방향**: 실패한 메시지를 `deque(maxlen=5)` 재시도 큐에 보관하고, 다음 전송 성공 시 큐를 드레인한다.

---

### 6-4. `logging_utils.py` — `TeeStream` 멀티스레드 안전성

**파일**: `logging_utils.py`

**현황**: `TeeStream._buffer`가 인스턴스 변수이지만 lock 없이 접근된다. `ebest_live.py`의 복수 스레드에서 `print()` 호출 시 buffer가 손상될 수 있다.

**개선 방향**: `threading.Lock()`으로 `_buffer` 접근을 직렬화한다.

---

### 6-5. `config.py` — `config.secrets.json` 없을 때 API 키 검증 경고 부재

**파일**: `config.py`

**현황**: `config.secrets.json`이 없어도 조용히 빈 딕셔너리로 처리된다. API 키가 없는 상태로 실행되면 LLM 예측이 전부 fallback(heuristic)으로 처리되지만 사용자는 이유를 모른다.

**개선 방향**: 초기화 시 API 키가 하나도 없으면 `WARNING: LLM 판단이 비활성화됩니다. config.secrets.json에 AI 키를 설정하세요.`를 출력한다.

---

## 7. 테스트 커버리지 확대 (Tests)

### 7-1. `option_flow_features.py` 전용 테스트 없음

**현황**: `option_flow_features.py`에 대한 테스트가 전혀 없다. 모듈 자체도 미연결 상태이므로 연결 후 단위 테스트가 필수이다.

**추가 필요**: `tests/test_option_flow_features.py`
- ATM 심볼 선택 로직 검증
- 분봉 df가 None일 때 default 반환 검증
- 정상 케이스 수치 검증

---

### 7-2. `weights_selector.py` — 만기주 경계 케이스 테스트 없음

**현황**: `test_config.py`에 일부 설정 테스트가 있지만 `weights_selector`의 만기일/만기주 전환 경계 케이스가 없다.

**추가 필요**: `tests/test_weights_selector.py`
- 만기 당일 09:00 (freeze)
- 만기 당일 15:35 (해제)
- 만기 다음 날 (정상)
- 만기주 밖 (정상)

---

### 7-3. `merge_datasets.py` — 스키마 불일치 병합 테스트 없음

**추가 필요**: `tests/test_merge_datasets.py`
- 동일 스키마 병합 성공 테스트
- 다른 스키마(v1 vs v2) 병합 실패 테스트

---

### 7-4. `dual_llm` 모드 통합 테스트 없음

**현황**: `test_llm_fallback.py`는 단일 LLM 경로만 테스트한다. `dual_llm=True` 시 두 LLM이 동시에 호출되고 결과가 올바르게 병합되는지 확인하는 테스트가 없다.

**추가 필요**: `tests/test_dual_llm.py`
- agree(BUY+BUY) → BUY
- disagree(BUY+SELL) + `disagreement_hold=True` → HOLD
- one timeout → single result

---

### 7-5. `data_builder.py` — 레이블 누수 검증 테스트 없음

**추가 필요**: `tests/test_data_builder_label_leak.py`
- 모든 샘플에서 `label_ts >= seq_end_ts + horizon` 조건 검증

---

## 8. 코드 품질·정리 (Code Quality)

### 8-1. `model.py` — docstring 자동 생성 잔재 제거

**파일**: `prediction/model.py`

**현황**: `_PositionalEncoding.__init__`, `PriceTransformer.__init__` 등의 docstring이 자동 생성 형식(`Args:\n    d_model:\n    max_len:`)으로 내용 없이 작성되어 있다. 실질적 정보가 없으면 오히려 가독성을 해친다.

**개선 방향**: 의미 있는 설명을 추가하거나 빈 docstring을 제거한다.

---

### 8-2. `ebest_live.py` — `_log()` 헬퍼 함수 불필요

**파일**: `ebest_live.py`

**현황**: `_log(msg, *args, level="info")` 헬퍼가 정의되어 있지만, 모듈 전체에서 `logger.info()`를 직접 호출하는 코드와 혼용된다. 일관성이 없다.

**개선 방향**: `_log()` 헬퍼를 제거하고 표준 `logger.info/warning/error()`를 통일하여 사용한다.

---

### 8-3. `constants.py` — `CLAUDE_MODEL` 버전 고착화

**파일**: `constants.py`

**현황**: `CLAUDE_MODEL = "claude-sonnet-4-20250514"`로 하드코딩되어 있다. Anthropic API가 새 모델을 출시하면 constants.py를 직접 수정해야 한다.

**개선 방향**: `config.json`의 `ai_providers.anthropic.model` 필드로 오버라이드할 수 있도록 `config.py`에서 환경 변수 → config → constants 순서로 해석한다.

---

### 8-4. `option_flow_features.py` — `_last_minute_bar_features()` iloc 접근 방식 불안전

**파일**: `prediction/option_flow_features.py`

**현황**:
```python
o = _safe_float(getattr(row, "get", lambda k, d=None: row[k])("open", 0.0))
```
이 패턴은 `pandas.Series`의 `get` 메서드를 우회하는 복잡한 코드이다. `pd.Series`에는 `.get()`이 있으므로 직접 사용하면 된다.

**개선 방향**:
```python
o = _safe_float(row.get("open", 0.0) if hasattr(row, "get") else row["open"])
```

---

### 8-5. `requirements.txt` — torch 버전 미고정

**파일**: `requirements.txt`

**현황**: `torch` 버전이 고정되어 있지 않다. PyTorch 2.x에서 `TransformerEncoderLayer` API가 변경되었으며, 버전에 따라 `batch_first` 기본값이 다르다.

**개선 방향**: `torch>=2.0.0,<3.0.0`으로 범위를 고정하고, `torchvision`, `torchaudio` 버전도 맞춘다.

---

## 9. 우선순위 요약 매트릭스

| # | 파일 / 위치 | 이슈 | 심각도 | 난이도 | 효과 |
|---|---|---|---|---|---|
| 1-1 | `option_flow_features.py` | v2 피처 미연결 | 🔴 Critical | 중 | 예측 품질 직접 향상 |
| 1-2 | `main.py` L3750 | dual_llm 인자 누락 | 🔴 Critical | 하 | 설정 의도 실현 |
| 1-3 | `data_builder.py` L46-48 | 전역 키 재정의 | 🟠 High | 하 | 부작용 방지 |
| 1-4 | `weights_selector.py` | 만기일 오후 freeze 미해제 | 🟠 High | 하 | 만기일 예측 안정성 |
| 4-1 | `data_builder.py` | 레이블 누수 위험 | 🟠 High | 중 | 학습 신뢰도 |
| 4-3 | `train_tft.py` | 시계열 무작위 split | 🟠 High | 하 | 과적합 방지 |
| 5-1 | `constants.py` | Gemini fallback 미완성 | 🟡 Medium | 하 | LLM 가용성 |
| 5-3 | `pipeline.py` | LLM 캐시 만료 정책 부재 | 🟡 Medium | 중 | 급변 대응 |
| 6-1 | `ebest_live.py` | 단일 락 contention | 🟡 Medium | 중 | 틱 처리 성능 |
| 4-2 | `train.py` | pos_weight 자동 계산 | 🟡 Medium | 하 | 불균형 레이블 처리 |
| 2-2 | `main.py` | 3750라인 모듈화 | 🟡 Medium | 고 | 장기 유지보수 |
| 3-1 | `features.py` | OPT_KEYS 버전 분기 | 🟡 Medium | 중 | 피처 일관성 |
| 7-1~5 | `tests/` | 테스트 추가 5종 | 🟡 Medium | 중 | 회귀 방지 |
| 8-5 | `requirements.txt` | torch 버전 미고정 | 🟡 Medium | 하 | 재현성 |
| 2-3 | `train.py/train_tft.py` | set_seed 중복 | 🟢 Low | 하 | 코드 정리 |
| 8-1 | `model.py` | 빈 docstring | 🟢 Low | 하 | 가독성 |
| 8-2 | `ebest_live.py` | _log() 혼용 | 🟢 Low | 하 | 일관성 |

---

## 수정 파일 목록 (실제 코드 변경 필요)

아래 파일들은 본 보고서의 권고사항을 반영하여 수정이 필요하다.

1. `prediction/pipeline.py` — option_flow 연결, LLM 캐시 무효화, disagreement 강화
2. `prediction/data_builder.py` — 전역 키 재정의 제거, 레이블 누수 가드
3. `prediction/features.py` — OPT_KEYS 분기 통일
4. `prediction/option_flow_features.py` — `_last_minute_bar_features` 접근 방식 수정
5. `prediction/weights_selector.py` — 만기일 오후 freeze 해제 추가
6. `prediction/time_features.py` — 세션 외 마스킹
7. `prediction/context_builder.py` — adaptive_context 구조화
8. `constants.py` — Gemini fallback 모델 확대
9. `train.py` — pos_weight 자동 계산, set_seed → utils
10. `train_tft.py` — walk-forward split, set_seed → utils
11. `main.py` — dual_llm 인자 누락 보완
12. `logging_utils.py` — TeeStream lock 추가
13. `requirements.txt` — torch 버전 범위 고정
14. `utils.py` — set_seed 추가

---

*본 보고서는 소스코드 정적 분석 기반으로 작성되었으며, 실제 런타임 동작과 차이가 있을 수 있습니다.*
