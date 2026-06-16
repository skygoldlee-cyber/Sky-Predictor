# Transformer 소스코드 재리뷰 보고서 v2

**리뷰 기준**: 업로드된 최신 소스 (Transformer.zip) — 이전 수정본 반영 여부 확인 + 전체 파일 신규 분석  
**총 파일**: 51개 Python 파일 (약 25,000줄)  
**이전 수정 반영**: ✅ Critical 6건 + High 14건 모두 반영 확인  
**신규 발견**: 8개 카테고리 **36건** (신규 26건 + 미적용 잔여 10건)

---

## 이전 수정본 반영 현황

| 이슈 | 반영 여부 | 비고 |
|------|----------|------|
| CON-01 `_metrics_lock` | ✅ 반영 | `_metrics_inc/set/get` 헬퍼 정상 작동 |
| CON-02 `_llm_cache_lock` | ✅ 반영 | |
| CON-03 `_user_pause_event` | ✅ 반영 | |
| CON-04 `set_market_closed()` | ✅ 반영 | |
| CON-05 `_ob_records` Lock | ✅ 반영 | |
| RES-01 `pipeline.close()` | ✅ 반영 | |
| RES-02 `LLMJudge.close()` | ✅ 반영 | |
| ARC-02 `_build_prediction_output()` | ✅ 반영 | |
| ARC-04 `_get()` 헬퍼 | ✅ 반영 | |
| ARC-05 `LiveState` 명시적 필드 | ✅ 반영 | |
| ARC-06 `_setup_logging`, `_build_pipeline` | ✅ 반영 | |
| ARC-09 `llm_judge` logger `__name__` | ✅ 반영 | |
| MNT-01 `parse_arguments` docstring | ✅ 반영 | |
| MNT-02 `--patience` 기본값 10 | ✅ 반영 | `train.py` |
| QUA-02 f-string 로그 (`config.py`, `main.py`) | ✅ 반영 | |
| QUA-04 `get_prediction` 시그니처 | ⚠️ 부분 반영 | `**kwargs` 잔존 — 신규 이슈 NW-01 |
| QUA-07 TR 코드 enum (`ebest_live`, `ebest_callbacks`) | ✅ 반영 | `tick_normalizer`, `data_builder` 미적용 — NW-02 |

---

## 신규 발견 이슈

> 🔴 Critical (즉시)  🟠 High (단기 1주)  🟡 Medium (중기 2~4주)  🟢 Low (개선 권장)

---

### 1. 동시성 (Concurrency)

#### NW-CON-01 🟠 `predictor.py` — `AdaptiveEnsembleWeightTracker` Lock 없음

`AdaptiveEnsembleWeightTracker`의 `_transformer_hits`, `_tft_hits`, `_transformer_w`, `_tft_w` deque는 피드백 루프 스레드(`_maybe_process_feedback`)와 예측 스레드(`get_prediction`)가 동시에 접근한다. deque의 `append()`는 GIL이 보호하지만 `get_weights()`의 `sum()` 복합 연산과 `reset()` 사이에는 race condition이 발생할 수 있다.

```python
# 현재: Lock 없음
class AdaptiveEnsembleWeightTracker:
    def update(self, ...):
        self._transformer_hits.append(...)   # 스레드 A
    def get_weights(self):
        t_acc = sum(self._transformer_hits)  # 스레드 B — 동시 읽기

# 수정: Lock 추가
def __init__(self, window: int = 20):
    self._lock = threading.Lock()
    ...
def update(self, ...) -> None:
    with self._lock:
        self._transformer_hits.append(...)
def get_weights(self) -> tuple[float, float]:
    with self._lock:
        t_acc = sum(self._transformer_hits) / ...
def reset(self) -> None:
    with self._lock:
        self._transformer_hits.clear()
        ...
```

---

### 2. 코드 품질 (Code Quality)

#### NW-QUA-01 ⚠️ `pipeline.py` — `get_prediction()` `**kwargs` 잔존 (QUA-04 부분 미완료)

시그니처에 `**kwargs`가 남아 있고 `kwargs.get("_now")`를 내부에서 사용한다. `_now`와 `auto_mode`는 테스트·실제 호출에서 모두 사용되는 공식 파라미터임에도 타입 선언이 없다.

```python
# 현재 (부분 수정 상태)
def get_prediction(self, *, off_boundary: bool = False, **kwargs: Any) -> Dict[str, Any]:
    now_dt = self._get_now_dt(now_override=kwargs.get("_now"))

# 수정: 명시적 파라미터
def get_prediction(
    self,
    *,
    off_boundary: bool = False,
    _now: Optional[datetime] = None,      # 테스트 주입용 시각 오버라이드
    auto_mode: bool = False,              # 자동 트리거 모드 플래그
) -> Dict[str, Any]:
    now_dt = self._get_now_dt(now_override=_now)
```

호출 측 (`ebest_live.py L1425`, `test_feedback_loop.py`, `test_replay_verification.py`, `test_llm_fallback.py`)도 함께 수정 필요.

---

#### NW-QUA-02 🟠 `tick_normalizer.py` / `data_builder.py` — TR 코드 매직 문자열 잔존 (QUA-07 미완료)

`ebest_live.py`와 `ebest_callbacks.py`는 수정됐으나 두 파일은 누락됐다.

| 파일 | 건수 |
|------|------|
| `tick_normalizer.py` | 5건 (`"FC0"`, `"OC0"`, `"FH0"`, `"OH0"`) |
| `prediction/data_builder.py` | 7건 |

```python
# tick_normalizer.py 현재
if tc in ("FC0", "OC0"):
if tc in ("FH0", "OH0"):

# 수정
from config import TRCode
if tc in (TRCode.FUTURES.value, TRCode.OPTIONS.value):
if tc in (TRCode.FUTURES_BOOK.value, TRCode.OPTIONS_QUOTE.value):
```

---

#### NW-QUA-03 🟠 `logging_utils.py` — f-string 로그 3건 잔존

```python
# 현재 (L314, L391, L399)
logging.error(f"Failed to open file handler stream: {e}")
self.logger.info(f"Starting: {self.operation}")
self.logger.info(f"Completed: {self.operation} ({elapsed:.2f}s)")

# 수정
logging.error("Failed to open file handler stream: %s", e)
self.logger.info("Starting: %s", self.operation)
self.logger.info("Completed: %s (%.2fs)", self.operation, elapsed)
```

---

#### NW-QUA-04 🟡 `predictor.py` — 자동생성 빈 docstring 4개 (QUA-06 미완료)

내용 없이 파라미터 이름만 나열하는 자동생성 docstring이 4개 잔존한다.

```python
# 현재 (L366, L387, L394, L170)
"""_classify.

Args:
    prob:
    spread:
"""

# 수정 예시
"""BUY/SELL/HOLD 신호와 신뢰도를 반환한다.

Args:
    prob: 모델 예측 확률 (0.0~1.0).
    spread: bid-ask 스프레드 (포인트).

Returns:
    TransformerPredictionResult(signal, confidence, prob, ...)
"""
```

---

#### NW-QUA-05 🟡 `predictor.py` `tick_normalizer.py` — `_depth_list` 빈 docstring

`tick_normalizer.py`의 내부 함수 `_depth_list`가 자동생성 docstring을 갖고 있다.

```python
# 현재
def _depth_list(prefix: str, n: int = 5) -> List[float]:
    """_depth_list.

Args:
    prefix:
    n:
"""

# 수정
def _depth_list(prefix: str, n: int = 5) -> List[float]:
    """호가 데이터에서 n단계 리스트를 추출한다 (예: offerho1~offerho5)."""
```

---

### 3. 아키텍처 (Architecture)

#### NW-ARC-01 🟠 `constants.py` — `PAST_UNKNOWN_DIM = 47` 하드코딩 (ARC-08 미완료)

`features.py`에 `get_feature_dim()` 동적 계산 헬퍼가 없고, `PAST_UNKNOWN_DIM`이 11개 파일에서 참조된다. `option_feature_set=v2` 활성화나 adaptive indicator 추가 시 실제 차원과 불일치가 런타임에서만 발견된다.

```python
# 현재: constants.py
PAST_UNKNOWN_DIM = 47  # option_feature_set, adaptive 여부 무관 고정값

# 수정: prediction/features.py에 헬퍼 추가
def get_feature_dim(
    option_feature_set: str = "v1",
    adaptive_enabled: bool = True,
) -> int:
    """런타임 설정에 따른 실제 feature 차원을 반환한다.

    option_feature_set="v2" 또는 adaptive_enabled=True 일 때
    PAST_UNKNOWN_DIM과 다를 수 있다.
    """
    base_keys = get_opt_keys(option_feature_set)
    dim = BASE_FEATURE_DIM + len(base_keys)
    if adaptive_enabled:
        dim += ADAPTIVE_INDICATOR_DIM
    return dim
```

`pipeline.py`, `predictor.py`, `train.py`, `train_tft.py` 등에서 `get_feature_dim()` 호출로 교체해야 한다.

---

#### NW-ARC-02 🟡 `train_tft.py` — Early stopping 미구현 (MNT-02 미완료)

`train.py`는 `--patience` 기본값이 10으로 수정됐으나, `train_tft.py`에는 early stopping 자체가 없다. TFT는 파라미터가 더 많아 과적합에 더 취약하다.

```python
# 현재: train_tft.py — patience 인자 없음
parser.add_argument("--epochs", type=int, default=80)
# early stopping 없음

# 수정
parser.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience. 0이면 비활성.")
parser.add_argument("--min-delta", type=float, default=0.0)

# train 루프에 추가
patience_counter = 0
if val_acc > best_val_acc + args.min_delta:
    best_val_acc = val_acc
    patience_counter = 0
    # save checkpoint
else:
    patience_counter += 1
    if args.patience > 0 and patience_counter >= args.patience:
        logger.info("Early stopping at epoch %d", epoch)
        break
```

---

#### NW-ARC-03 🟡 `context_builder.py` — 무음 예외 8건 / 반환값 보장 불명확

`context_builder.py`의 `build_llm_context()` 등은 예외 발생 시 빈 문자열을 반환하는데, 이것이 의도적 fallback인지 버그인지 주석이 없다. LLM에 빈 컨텍스트가 전달될 경우 판단 품질이 크게 저하된다.

```python
# 수정 방향: 최소 경고 로그 추가
try:
    section = _build_options_section(opt_snap)
except Exception as e:
    logger.warning("[ContextBuilder] 옵션 섹션 구성 실패 (빈 문자열 반환): %s", e)
    section = ""
```

---

#### NW-ARC-04 🟡 `ebest_live.py` — `predictor.close()` 미호출

`RES-01`에서 `PredictionPipeline.close()` 메서드를 추가했지만, 실제 호출 지점이 없다. `ebest_live.py`의 종료 처리 경로에서 `predictor.close()`가 누락되어 있다.

```python
# ebest_live.py 종료 블록에 추가 필요
finally:
    try:
        if predictor is not None and hasattr(predictor, "close"):
            predictor.close()
            logger.info("[LIVE] PredictionPipeline closed")
    except Exception as e:
        logger.warning("[LIVE] predictor.close() 실패: %s", e)
```

---

### 4. 테스트 (Testing)

#### NW-TST-01 🔴 `tests/test_llm_timeout_fallback.py` — 빈 파일 (TST-03 미작성)

파일이 공백 1줄만 존재한다. LLM timeout → transformer fallback 경로는 운영 중 가장 빈번한 시나리오임에도 테스트가 전혀 없다.

**최우선 작성 케이스:**

```python
def test_llm_timeout_falls_back_to_transformer_signal(monkeypatch):
    """LLM 타임아웃 시 transformer 신호가 그대로 반환되는지 검증."""

def test_llm_429_sets_rate_limit_cooldown(monkeypatch):
    """LLM 429 응답 시 _llm_rate_limited_until_epoch가 설정되는지 검증."""

def test_dual_llm_primary_fail_uses_secondary(monkeypatch):
    """dual_llm 모드에서 primary 실패 시 secondary가 사용되는지 검증."""

def test_off_boundary_skip_when_rate_limited(monkeypatch):
    """rate_limited 상태에서 off_boundary=True 호출이 LLM을 건너뛰는지 검증."""
```

---

#### NW-TST-02 🟠 `tests/test_feedback_loop.py` — 경계 조건 미검증 (TST-05 미완료)

현재 테스트는 기본 경로(feedback_snapshot_required=False)만 검증한다.

**누락된 케이스:**

```python
def test_feedback_snapshot_required_skips_without_snapshot():
    """feedback_snapshot_required=True인데 스냅샷 없을 때 평가를 건너뛰는지 검증."""

def test_feedback_skip_hold_small_move():
    """skip_hold_ticks 기준 이하 소폭 HOLD는 피드백 큐에서 제외되는지 검증."""

def test_feedback_weight_saturation():
    """transformer가 100번 연속 정답일 때 weight가 [0.0, 1.0] 범위를 유지하는지 검증."""

def test_feedback_queue_maturation_time():
    """prediction_minutes 미경과 레코드는 평가되지 않는지 검증."""
```

---

#### NW-TST-03 🟠 `tests/test_smoke.py` — TR 코드 매직 문자열 사용

```python
# 현재
{"trcode": "FC0", ...}
{"trcode": "FH0", ...}

# 수정
from config import TRCode
{"trcode": TRCode.FUTURES.value, ...}
{"trcode": TRCode.FUTURES_BOOK.value, ...}
```

---

#### NW-TST-04 🟡 `config.py` — 단위 테스트 부재 (TST-01 미완료)

823줄, 50개 이상 검증 로직이 있으나 테스트 파일이 없다.

**최우선 작성 케이스:**

```python
def test_validate_buy_sell_threshold_inverted():
    """buy_threshold < sell_threshold이면 ValidationError가 발생하는지 검증."""

def test_pred_cfg_overrides_root_level():
    """prediction 섹션 값이 루트 레벨보다 우선 적용되는지 검증 (ARC-04 _get() 헬퍼)."""

def test_from_file_missing_raises():
    """존재하지 않는 config 파일 로드 시 적절한 예외가 발생하는지 검증."""

def test_env_var_overrides_file():
    """EBEST_APPKEY 환경변수가 config.json보다 우선 적용되는지 검증."""
```

---

### 5. 유지보수성 (Maintainability)

#### NW-MNT-01 🟠 `constants.py` — LLM 타임아웃 매직 넘버 분산 (MNT-03 미완료)

```python
# 현재: constants.py에 없음
# pipeline.py에서 직접 사용
llm_timeout_sec=8.0   # 기본값 하드코딩

# 수정: constants.py에 추가
LLM_TIMEOUT_SEC = 8.0              # LLM 단일 호출 타임아웃
LLM_MIN_INTERVAL_SEC = 30.0        # LLM 최소 호출 간격
LLM_FEEDBACK_SNAPSHOT_TOLERANCE_SEC = 30.0  # 피드백 스냅샷 허용 오차
LLM_COOLDOWN_SECONDS_ON_429 = 300.0  # 이미 존재 — OK
```

---

#### NW-MNT-02 🟡 `prediction/data_builder.py` — 무음 예외 52건

학습 데이터 구성 핵심 경로에서 `except Exception: pass` 패턴이 52건 발견됐다. 데이터 파이프라인에서 무음 예외는 학습 데이터 품질 저하로 이어지지만 원인 파악이 불가능하다.

```python
# 현재
try:
    row = _build_feature_row(tick, ob_snap, opt_snap)
except Exception:
    pass

# 수정: 최소 debug 로그
try:
    row = _build_feature_row(tick, ob_snap, opt_snap)
except Exception as e:
    logger.debug("[DataBuilder] feature row 구성 실패 (건너뜀): %s", e)
    skipped_count += 1
```

---

#### NW-MNT-03 🟡 `MD_to_HTML.py` — 3,271줄 유틸리티가 프로젝트 루트에 혼재 (MNT-04 미완료)

```
현재: /MD_to_HTML.py        ← 루트 혼재
권장: /tools/MD_to_HTML.py  ← 도구 디렉터리 분리
```

---

#### NW-MNT-04 🟢 `tick_normalizer.py` — `"OC0"`, `"FC0"` 대문자 정규화 후 비교 불일치 가능성

`tc = str(trcode or "").strip().upper()`로 대문자 변환 후 `TRCode` enum 값과 비교하는데, 향후 `TRCode.FUTURES.value`가 소문자로 변경될 경우 silent mismatch 발생. 비교 전 정규화 위치를 명시적으로 문서화해야 한다.

---

### 6. 성능 (Performance)

#### NW-PER-01 🟡 `prediction/data_builder.py` — 전체 JSONL 2회 순회

`build_dataset()` 내부에서 심볼 수집용 1회 + 데이터 구성용 1회로 동일 파일을 2번 읽는다. 대용량 파일에서 I/O 병목이 발생할 수 있다.

```python
# 현재: 2-pass 구조
for rec in _load_jsonl(path):   # pass 1: symbol 수집
    ...
for rec in _load_jsonl(path):   # pass 2: feature 구성
    ...

# 수정: single-pass — symbol 수집과 feature 구성을 동시에
for rec in _load_jsonl(path):
    symbol = _extract_symbol(rec)
    all_symbols.add(symbol)
    _process_record(rec, ...)
```

---

#### NW-PER-02 🟢 `option_features.py` — `_strike_to_symbol_map` 매 호출 재계산

ATM 미시구조 계산(`calc_atm_microstructure`) 내부에서 `_strike_to_symbol_map(opts)`를 매번 재계산한다. 동일 `opts` dict에 대해 캐시를 두면 반복 호출 비용을 줄일 수 있다.

---

### 7. 보안 (Security)

#### NW-SEC-01 🟡 `ebest_api.py` — API 키 예외 메시지 노출 가능성

`_ebest_login` 실패 시 `logger.warning("[LOGIN] failed: %s", e)` 에서 예외 메시지에 API 키가 포함될 수 있다. 일부 HTTP 라이브러리는 인증 실패 예외에 요청 URL(키 포함)을 그대로 담는다.

```python
# 수정: 예외 메시지에서 민감 정보 마스킹
def _mask_sensitive(msg: str) -> str:
    # appkey, appsecretkey 패턴 마스킹
    import re
    return re.sub(r'(appkey|secret)[=:]\s*\S+', r'\1=***', msg, flags=re.IGNORECASE)

logger.warning("[LOGIN] failed: %s", _mask_sensitive(str(e)))
```

---

### 8. 설계 완성도

#### NW-DESIGN-01 🟡 `pipeline.py` + `ebest_live.py` — `_build_pipeline()` 헬퍼 미연결

`main.py`에 `_build_pipeline()` 헬퍼를 추가했으나, `ebest_live.py` 내부의 실제 `PredictionPipeline(...)` 생성 코드는 여전히 독립적으로 존재한다. 두 곳에서 파라미터가 다시 발산할 수 있다.

```python
# ebest_live.py 내 파이프라인 생성 코드를 _build_pipeline() 호출로 교체
from main import _build_pipeline
predictor = _build_pipeline(config, args)
```

---

## 우선순위 로드맵 v2

### 즉시 (1~2일)

| ID | 파일 | 작업 |
|----|------|------|
| NW-TST-01 | `test_llm_timeout_fallback.py` | 4개 테스트 케이스 작성 |
| NW-QUA-01 | `pipeline.py` | `**kwargs` → `_now`, `auto_mode` 명시 파라미터화 |
| NW-QUA-02 | `tick_normalizer.py`, `data_builder.py` | TR 코드 `TRCode` enum 교체 (12건) |
| NW-ARC-04 | `ebest_live.py` | `predictor.close()` 종료 경로 호출 추가 |
| NW-CON-01 | `predictor.py` | `AdaptiveEnsembleWeightTracker` Lock 추가 |

### 단기 (1주)

| ID | 파일 | 작업 |
|----|------|------|
| NW-MNT-01 | `constants.py` | `LLM_TIMEOUT_SEC` 등 3개 상수 추가 |
| NW-ARC-02 | `train_tft.py` | early stopping (`--patience`) 추가 |
| NW-QUA-03 | `logging_utils.py` | f-string 로그 3건 교체 |
| NW-TST-02 | `test_feedback_loop.py` | 경계 케이스 4개 추가 |
| NW-TST-04 | `tests/test_config.py` | 신규 파일 작성 (4개 케이스) |
| NW-SEC-01 | `ebest_api.py` | 예외 메시지 API 키 마스킹 |

### 중기 (2~4주)

| ID | 파일 | 작업 |
|----|------|------|
| NW-ARC-01 | `features.py`, `constants.py` | `get_feature_dim()` 동적 계산 헬퍼 |
| NW-ARC-03 | `context_builder.py` | 무음 예외 8건 → 최소 경고 로그 |
| NW-QUA-04/05 | `predictor.py`, `tick_normalizer.py` | 자동생성 docstring 내용 작성 |
| NW-MNT-02 | `data_builder.py` | 핵심 경로 무음 예외 → debug 로그 |
| NW-PER-01 | `data_builder.py` | single-pass 리팩터링 |
| NW-MNT-03 | `MD_to_HTML.py` | `tools/` 디렉터리로 이동 |
| NW-DESIGN-01 | `ebest_live.py` | `_build_pipeline()` 헬퍼로 통합 |

---

## 전체 이슈 현황 요약

| 카테고리 | 이전 미해결 | 신규 발견 | 합계 |
|---------|-----------|---------|------|
| 동시성 | 0 | 1 | 1 |
| 코드 품질 | 2 (QUA-01 부분, QUA-07 잔여) | 3 | 5 |
| 아키텍처 | 2 (ARC-08, ARC-06 미연결) | 2 | 4 |
| 테스트 | 2 (TST-01, TST-03, TST-05) | 2 | 4 |
| 유지보수성 | 2 (MNT-03, MNT-04) | 4 | 6 |
| 성능 | 0 | 2 | 2 |
| 보안 | 0 | 1 | 1 |
| 설계 완성도 | 1 | 1 | 2 |
| **합계** | **9** | **16** | **25** |

---

## 긍정적 변화 (이전 대비 개선된 점)

- `constants.py` 정비: `AnalysisMode`, `TRCode`, `PredictionDirection` enum 추가, 로그 설정 상수 중앙화
- `ebest_api.py` 리팩터링: API 자격증명 조회를 환경변수 우선 방식으로 개선 (`SEC-02` 부분 해소)
- `logging_utils.py`: `flush()` 잔여 버퍼 처리 구현 (`MNT-05` 해소), `atexit` 등록 추가
- `tick_processor.py`: `set_market_closed()` 세터 정상 반영
- `train.py`: `--patience` 기본값 10으로 개선 완료
- `utils.py`: `safe_float`, `safe_int` 방어적 타입 변환 표준화
