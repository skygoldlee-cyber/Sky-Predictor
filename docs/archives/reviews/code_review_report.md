# KOSPI200 Transformer 시스템 코드 리뷰
**종합 코드 & 문서 품질 분석 보고서**

> 대상: `Transformer/`, `adaptive_indicator/`, `prediction/` | 리뷰어: Claude | 일자: 2026-02-23

---

## 1. 전체 시스템 평가

전체적으로 매우 높은 수준의 프로덕션 코드입니다. 방어적 코딩(try/except), 타입 명시, 모듈 분리, 문서화 수준이 인상적입니다.

| 평가 영역 | 점수 | 코멘트 |
|---|---|---|
| 코드 구조 / 모듈화 | ★★★★★ | `prediction/`, `adaptive_indicator/` 역할 분리 명확 |
| 방어적 코딩 | ★★★★☆ | 거의 모든 경로에 try/except, 단 일부 예외 묵살 |
| 문서화 품질 | ★★★★★ | MD 가이드 7종 + docstring 체계적 |
| 버그 밀도 | ★★★★☆ | 치명적 버그 1건(NameError), 나머지 경계 이슈 |
| 테스트 커버리지 | ★★☆☆☆ | `tests/` 존재하나 유닛 테스트 매우 제한적 |
| 운영 안정성 | ★★★★☆ | FO0 stale 감지, LLM timeout, fallback 모두 구현 |

---

## 2. 발견된 이슈 전체 목록

심각도: 🔴 버그(즉시 수정) / 🟠 경고(운영 위험) / 🟡 개선(품질 향상) / 🟢 잔존(이전 리뷰 미반영)

| 심각도 | 위치 | 이슈 | 설명 / 권장 조치 |
|---|---|---|---|
| 🔴 버그 | `pipeline.py:336-338` | API_MAX_RETRIES NameError | `dual_llm` 모드에서 `_judge_provider_with_timeout()` 호출 시 constants에서 임포트되지 않은 `API_MAX_RETRIES` 등을 참조. `try/except`로 감싸져 있어 조용히 fallback되지만, 재시도 로직이 완전히 무력화됨. → constants import에 `API_MAX_RETRIES`, `API_RETRY_DELAY_SECONDS`, `API_BACKOFF_MULTIPLIER` 추가 (**✅ 수정됨**) |
| 🟠 경고 | `pipeline.py:_compute_regime` | closure에서 outer 변수 직접 참조 | `_compute_regime()`는 `adaptive_supertrend_state`, `adaptive_features`를 closure로 캡처. 병렬/비동기 확장 시 race condition 위험. 현재는 단일 스레드이므로 버그는 아님. → 인자로 명시적 전달 권장 |
| 🟠 경고 | `pipeline.py:get_prediction` | 예외 범위가 너무 넓음 | 최상위 `except Exception`이 numeric predictor/LLM 오류를 모두 동일하게 처리. 에러 종류별 분류가 어렵고 silent failure 가능성. → NumericPredictor 오류와 LLM 오류를 개별 try/except로 분리 |
| 🟠 경고 | `prediction/pipeline.py` | `_ob_records` 쓰기에 Lock 없음 | `add_realtime_tick()`은 ebest 콜백 스레드에서 호출, `get_prediction()`은 예측 루프에서 호출. deque는 GIL 보호를 받지만 복합 read-modify-write(`_last_fo0_second`, `_ob_records[-1]=ob`)는 원자적이지 않음. → `threading.Lock`으로 보호 (**✅ 수정됨**) |
| 🟡 개선 | `adaptive_zigzag.py:fib_keys` | Fibonacci 이중 키 복잡도 | `legacy_key('0.618')`와 `new_key('fib_618')`를 동시 저장. `get_llm_context`의 fallback 조회 코드가 복잡하고 향후 한 키만 제거 시 묵히 실패 위험. → 내부적으로 `new_key`만 사용 |
| 🟡 개선 | `adaptive_zigzag.py:L202-326` | `_pending_confirm` 처리 순서 | 스윙 확정 직후 같은 봉에서 새 전환 조건이 감지되면 `_pending_confirm != None`이어서 후보가 무시됨. `confirmation_bars=1`일 때 연속 스윙 누락 가능. → 확정 처리 후 `_pending_confirm = None` 확인 후 새 후보 생성 |
| 🟡 개선 | `prediction/predictor.py` | `ModelInput.feature_snapshot` 타입 힌트 | `feature_snapshot: Dict[str, Any] = None`으로 선언되어 있어 타입 힌트가 `Optional`이 아닌데 `None`을 기본값으로 사용. mypy 경고 발생. → `Optional[Dict[str, Any]] = None`으로 수정 (**✅ 수정됨**) |
| 🟡 개선 | `tick_processor.py` | `option_minute_data` 메모리 성장 | `options_minute_data`가 cleanup 주기가 긴 경우 장기 실행 시 메모리 증가 가능. 선물 대비 옵션 데이터 만료 처리가 상대적으로 약함. → cleanup 주기 검토, `deque(maxlen)` 활용 고려 |
| 🟢 잔존 | `simulate_indicators.py` | BUY/SELL annotate 루프 성능 | 이전 리뷰에서 지적된 annotate 루프가 미반영. 시뮬레이션용이어서 프로덕션 영향 없음. → 벡터화 연산으로 전환 (`Series.mask/where` 활용) |
| 🟢 잔존 | `config.py` / `train.py` | OHLC 관계 위반 미검증 | `open > high` 또는 `low > close` 같은 비정상 OHLCV 가드 없음. 데이터 오염 시 지표 계산 오류 전파 가능. → 학습 데이터 로드 시 OHLCV 유효성 검사 추가 |

---

## 3. 핵심 버그 수정 가이드

### 3-1. API_MAX_RETRIES NameError (즉시 수정 필요)

`dual_llm` 모드 활성화 시 `_judge_provider_with_timeout()`에서 `API_MAX_RETRIES` 등이 임포트되지 않아 `NameError` 발생. `try/except`로 묵살되어 재시도 로직이 완전 무력화됨.

**수정 방법** — `prediction/pipeline.py` 상단 import 수정:

```python
# 기존
from constants import FUTURE_KNOWN_DIM, HORIZON_SEC, TRCode

# 수정
from config import (
    FUTURE_KNOWN_DIM, HORIZON_SEC, TRCode,
    API_MAX_RETRIES, API_RETRY_DELAY_SECONDS, API_BACKOFF_MULTIPLIER
)
```

### 3-2. `_ob_records` 스레드 안전성

`add_realtime_tick()`과 `get_prediction()`이 서로 다른 스레드에서 `_ob_records`와 `_last_fo0_second`를 동시에 읽고 씀. 현재 구조상 충돌 확률은 낮지만 잠재적 race condition 존재.

**수정 예시:**

```python
import threading

# __init__에 추가
self._ob_lock = threading.Lock()

# add_realtime_tick의 OB 처리 구간
with self._ob_lock:
    if self._last_fo0_second == sec_key and self._ob_records:
        self._ob_records[-1] = ob
    else:
        self._ob_records.append(ob)
    self._last_fo0_second = sec_key
    self._last_fo0_sig = sig
    self._last_ob_snapshot = dict(ob)

# get_prediction에서 ob_records 읽기
with self._ob_lock:
    ob_records_snapshot = list(self._ob_records)
    last_ob = dict(self._last_ob_snapshot)
```

### 3-3. `_compute_regime` closure 리팩터링

```python
# 기존 — outer scope 암묵적 캡처
def _compute_regime() -> Optional[str]:
    ...
    st = adaptive_supertrend_state  # closure 캡처
    ...

# 권장 — 인자로 명시적 전달
def _compute_regime(
    features: Optional[Dict[str, float]],
    supertrend_state: Any,
) -> Optional[str]:
    ...

# 호출부
regime = _compute_regime(adaptive_features, adaptive_supertrend_state)
```

---

## 4. Markdown 문서 리뷰

| 문서 | 품질 | 내용 |
|---|---|---|
| `README.md` | ★★★★★ | 파일 구조, feature_dim 계산, LLM 정책, dual_llm 설명 완비. 신규 팀원 온보딩에 충분. |
| `TRANSFORMER_GUIDE.md` | ★★★★★ | 입력 피처 스키마, 학습/추론 파이프라인, look-ahead bias 방지 등 핵심 내용 상세 기술. |
| `TFT_DUAL_MODEL_DESIGN_GUIDE.md` | ★★★★☆ | TFT 아키텍처 설계 의도 명확. 단, `tft_model.py` 구현과 일부 파라미터 불일치 가능성 검증 필요. |
| `ADAPTIVE_INDICATOR_GUIDE.md` | ★★★★★ | 수식 포함, config 파라미터 설명, 신호 해석법까지 완비. 매우 상세. |
| `DAILY_TICK_TRAINING_RUNBOOK.md` | ★★★★★ | 운영 런북으로 단계별 명령어, 체크포인트, 자동화 팁까지 포함. |
| `code_review.md` | ★★★★☆ | 1차 리뷰 반영 현황 추적 표 포함. 단, `dist_abs` 버그는 이미 수정됐음에도 미반영으로 표시됨 — 현재 코드와 싱크 필요. |
| `Architecture.md` | ★★★★☆ | 시스템 아키텍처 전반 설명. 최신 dual_llm/adaptive_indicator 내용 반영 여부 확인 권장. |
| `LLM_INPUT_TABLE.md` | ★★★★☆ | LLM 입력 필드 스키마 테이블 정리. `context_builder.py` 변경 시 동기화 필요. |

---

## 5. 아키텍처 강점 (잘 된 점)

| 영역 | 내용 |
|---|---|
| **역할 분리** | `PredictionPipeline`(오케스트레이션) / `NumericPredictor`(수치) / `LLMJudge`(판단) / `RealTimeTickProcessor`(데이터)가 명확히 분리됨. 각 계층이 독립적으로 테스트/교체 가능. |
| **LLM Failover** | `preferred_provider` 설정 → 실패 시 자동 fallback, Gemini/Claude model fallback list, timeout 격리(`ThreadPoolExecutor`), `dual_llm` 동시 호출 지원까지 운영 견고성 우수. |
| **Feature 일관성** | `OB_KEYS`, `CD_KEYS`, `ADAPT_KEYS`, `OPT_KEYS`를 constants로 관리하여 train-serve feature drift 방지. `schema_version` 태깅으로 가중치 호환성 검증. |
| **Adaptive Indicator** | ER 기반 ATR 기간 + ADX 기반 multiplier 동적 조정 설계가 교과서적. `WilderRMA` 분리, config dataclass 분리로 재사용성 높음. |
| **가드레일 체계** | ATM spread/liquidity 필터, basis 과도 시 HOLD 강제, LLM empty output 감지, FO0 stale 경고 등 다층 안전장치. |
| **방어적 초기화** | 모든 optional 의존성(torch, anthropic, openai, google-genai)이 `try/except`로 graceful degradation. 패키지 미설치 환경에서도 rule-based로 동작. |

---

## 6. 우선순위별 액션 아이템

### P0 — 즉시 (이번 주)

- `pipeline.py`: `API_MAX_RETRIES` import 추가 → dual_llm 재시도 로직 복원 (**✅ 완료**)
- `code_review.md`: `dist_abs` 버그 항목을 ✅ 수정됨으로 갱신 (이미 수정된 상태)

### P1 — 단기 (1~2주)

- `pipeline.py`: `_ob_records` 복합 연산에 `threading.Lock` 추가 (**✅ 완료**)
- `pipeline.py`: `_compute_regime`를 명시적 인자 함수로 리팩터링
- `predictor.py`: `ModelInput.feature_snapshot` 타입 힌트 `Optional`로 수정 (**✅ 완료**)
- `tests/`: `add_realtime_tick → get_prediction` 통합 테스트 1건 추가

### P2 — 중기 (1달)

- `adaptive_zigzag.py`: Fibonacci 키 정리 (`new_key` 단일화), `_pending_confirm` 경계 수정
- `tick_processor.py`: `options_minute_data` 메모리 cleanup 주기 검토
- `Architecture.md`: dual_llm / adaptive_indicator 최신 내용 반영 확인
- 전체 테스트 커버리지 목표 설정 (현재 매우 낮음)

---

> 이 리뷰는 정적 코드 분석 기반이며, 런타임 동작은 실제 eBest API 연동 환경에서 별도 검증이 필요합니다.
