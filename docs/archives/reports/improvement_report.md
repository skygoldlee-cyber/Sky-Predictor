# SkyEbest/Transformer 코드베이스 보완·개선 보고서

> 분석 기준: `Transformer.zip` (2026-03-05 업로드)  
> 분석 범위: `prediction/pipeline.py`, `prediction/llm_judge.py`, `ebest_live.py`, `constants.py`, `config.py`, `tick_processor.py`, `logging_utils.py` 및 전체 구조

---

## 1. 버그 (즉시 수정 필요)

### 1-1. `numeric_res` 미정의 변수 참조 — `pipeline.py`

**위치:** `get_prediction()` → feedback_queue append 블록 (약 2393행)

```python
# ❌ 버그: numeric_res는 정의된 적 없는 변수
if hasattr(numeric_res, "transformer_prob") and hasattr(numeric_res, "tft_prob"):
    ...
    "transformer_prob": float(getattr(numeric_res, "transformer_prob")),
    "tft_prob":         float(getattr(numeric_res, "tft_prob")),
```

`_run_numeric_prediction_and_guardrails()`는 `t_res`를 반환하는데 이 블록에서만 `numeric_res`로 참조합니다.  
`try/except Exception: pass`로 감싸져 있어 **피드백 루프가 조용히 전혀 작동하지 않는** 상태입니다.  
앙상블 가중치 자동 학습(`update_adaptive_weights`)이 완전히 비활성화됩니다.

**수정:**
```python
# ✅ t_res로 교체
if hasattr(t_res, "transformer_prob") and hasattr(t_res, "tft_prob"):
    ...
    "transformer_prob": float(getattr(t_res, "transformer_prob")),
    "tft_prob":         float(getattr(t_res, "tft_prob")),
```

---

### 1-2. `dual_llm` 경로에서 provider별 rate limit 무시 — `pipeline.py`

**위치:** `_run_llm_judgment()` → `if self._dual_llm:` 블록

```python
# ❌ _judge_provider_direct는 _provider_rate_limited_until 체크 없음
fut_gpt = self._llm_executor.submit(
    self._judge_provider_direct, provider="gpt", ...
)
```

`_judge_provider_direct`는 단순히 `judge.judge_provider()`를 호출하며,  
`_provider_rate_limited_until` 딕셔너리를 전혀 확인하지 않습니다.  
GPT가 429 쿨다운 중임에도 `dual_llm` 모드에서 계속 호출됩니다.

**수정:** `_judge_provider_direct` 내부에 rate limit 체크 추가

```python
def _judge_provider_direct(self, *, provider: str, system: str, user: str):
    if self.judge is None:
        return None
    # provider별 쿨다운 체크
    try:
        prov = str(provider or "").strip().lower()
        prl_until = float((self._provider_rate_limited_until or {}).get(prov, 0.0) or 0.0)
        if prl_until > 0.0 and float(time.time()) < prl_until:
            return None  # skip — rate limited
    except Exception:
        pass
    try:
        return self.judge.judge_provider(str(provider), system, user, timeout=float(self._llm_timeout_sec))
    except Exception:
        return None
```

---

### 1-3. `_adaptive_mgr` / `_adaptive_enabled` 이중 초기화 — `pipeline.py`

**위치:** `__init__()` 약 298~308행

```python
self._adaptive_mgr = None      # ← 1차 할당
self._adaptive_warmed = False
self._adaptive_last_minute_ts: Optional[datetime] = None
self._adaptive_enabled = False  # ← 1차 할당
self._adaptive_mgr = None      # ← ❌ 불필요한 재할당 (동일 값)
self._adaptive_last_minute_ts = None  # ← ❌ 불필요한 재할당

self._adaptive_enabled = True   # ← 이후 True로 덮어씀
```

`_adaptive_enabled = False` 설정이 바로 아래에서 `True`로 덮어써집니다.  
config에서 `adaptive_indicator` 비활성화 옵션을 전달해도 무조건 `True`가 됩니다.

**수정:** 중복 할당 3행 제거, `_adaptive_enabled` 초기값을 최종 의도에 맞게 단일 설정

---

### 1-4. `_judge_with_timeout`에서 429 감지 후 rate limit 미설정 — `pipeline.py`

**위치:** `_judge_with_timeout()` (단일 provider 모드 경로)

```python
except Exception as e:
    return None, False, str(e)  # ← 429도 동일하게 처리, 쿨다운 설정 없음
```

`_judge_provider_with_timeout()`에는 429 감지→쿨다운 설정 로직이 있지만,  
단일 provider 모드(비 dual_llm)가 사용하는 `_judge_with_timeout()`에는 없습니다.

**수정:**
```python
except Exception as e:
    s = str(e or "").lower()
    if "429" in s or "too many requests" in s:
        cd = float(LLM_COOLDOWN_SECONDS_ON_429 or 0.0)
        if cd > 0.0:
            self._llm_rate_limited_until_epoch = float(time.time()) + cd
            logger.warning("[LLM_429] single_provider cooldown=%.0fs", cd)
    return None, False, str(e)
```

---

## 2. 설계 개선 (중요도 높음)

### 2-1. `LLM_COOLDOWN_SECONDS_ON_429`를 config에서 오버라이드할 수 없음

**현황:** `constants.py`에 `LLM_COOLDOWN_SECONDS_ON_429 = 300.0` (하드코딩)  
`config.py`의 `PredictionConfig`에 해당 필드가 없어 **운영 중 조정 불가**입니다.

**개선:**
```python
# config.py PredictionConfig에 추가
llm_cooldown_sec_on_429: float = 300.0

# pipeline.py에서 constants 대신 instance 변수 사용
self._llm_cooldown_sec_on_429 = float(llm_cooldown_sec_on_429 or LLM_COOLDOWN_SECONDS_ON_429)
```

---

### 2-2. `_judge_provider_direct` 예외 삼킴으로 429 원인 추적 불가

**현황:**
```python
def _judge_provider_direct(self, *, provider: str, system: str, user: str):
    ...
    except Exception:
        return None  # ← 어떤 에러인지 전혀 기록 안 함
```

429, timeout, 인증 오류 등 모든 실패가 `None` 반환으로 처리되어  
`gpt_err`에 저장되지 않고 로그에도 남지 않습니다.

**개선:** 예외를 `raise`하거나 최소한 `logger.debug()` 기록

```python
except Exception as e:
    logger.debug("[LLM_DIRECT_FAIL] provider=%s err=%s", provider, e)
    raise  # 호출부에서 fut.result() 시 except로 전달
```

---

### 2-3. LLM 캐시 키가 프롬프트 전체를 포함 — 메모리 낭비

**현황:**
```python
cache_key = "|".join([
    "dual" if bool(self._dual_llm) else "single",
    str(self._dual_llm_primary_provider or ""),
    str(system or ""),   # ← 수백 바이트
    str(user or ""),     # ← 수천 바이트
])
```

프롬프트 전체를 문자열로 이어 붙여 캐시 키를 만들어  
`_last_llm_cache_key`에 수 KB가 매 예측마다 저장됩니다.

**개선:** 해시 사용
```python
import hashlib
_raw = f"dual={self._dual_llm}|prov={self._dual_llm_primary_provider}|{system}|{user}"
cache_key = hashlib.md5(_raw.encode(), usedforsecurity=False).hexdigest()
```

---

### 2-4. `off_boundary_trigger` 시 rate limit 체크 후 즉시 return — LLM 프롬프트 불필요 빌드

**현황:** rate-limited 상태에서 off_boundary라면 LLM을 건너뛰지만,  
그 전에 이미 `_build_llm_prompt()` (옵션 스냅샷 + OBI + 컨텍스트 빌드)를 전부 실행합니다.

**개선:** rate limit 체크를 프롬프트 빌드보다 앞으로 이동

```python
# off_boundary + rate_limited → 프롬프트 빌드 전에 early return
if bool(off_boundary) and bool(in_rl):
    return _make_skip_result(...)  # snapshot, prompt 빌드 생략

system, user = self._build_llm_prompt(...)  # 여기서 실행
```

---

### 2-5. `constants.py`의 `API_MAX_RETRIES = 2`가 `max_retries=0`과 충돌

**현황:** `llm_judge.py`에서 `openai.OpenAI(max_retries=0)`으로 SDK retry를 껐지만,  
`_judge_provider_with_timeout()`에서 `API_MAX_RETRIES = 2` 기반의 루프가 여전히 2회 실행됩니다.  
이로 인해 GPT가 429를 반환하면 SDK 재시도는 없지만 **Python 레벨에서 2회 더 호출**합니다.

**개선:** 429 감지 시 `break` 처리는 이미 적용됐지만, GPT 전용으로 `max_retries = 1`로 고정하거나  
constants에 `GPT_MAX_RETRIES = 1`을 별도 정의하는 것이 명확합니다.

---

## 3. 구조적 개선 (중요도 보통)

### 3-1. `get_prediction()`이 너무 큰 단일 함수

`get_prediction()`는 약 200행으로, 내부에서 다음을 모두 처리합니다:
- `off_boundary` early return (약 50행 중복 결과 조립)
- 정상 경로 결과 조립 (약 50행 중복)

결과 dict 생성 코드가 두 경로에서 **동일하게 복사**되어 있어 유지보수 위험이 높습니다.

**개선:** 결과 dict 조립을 `_build_result_dict()` 헬퍼로 분리

```python
def _build_result_dict(self, *, now_dt, current_price, prob, signal, ...) -> Dict[str, Any]:
    return {"prediction_time": now_dt.isoformat(), ...}
```

---

### 3-2. `adaptive_indicator` 설정 비활성화 경로가 없음

`config.json`에서 `adaptive_indicator` 섹션을 아예 생략해도  
`pipeline.py`에서 `_adaptive_enabled = True`가 강제 설정되어  
kospi_indicators import 시도 후 실패하면 `_adaptive_mgr = None`이 됩니다.  
**의도적으로 비활성화하는 명시적 방법이 없습니다.**

**개선:**
```python
# config.json
"adaptive_indicator": {"enabled": false}

# pipeline.py
self._adaptive_enabled = bool(ad.get("enabled", True))
```

---

### 3-3. Transformer 가중치 없을 때 rule-based fallback의 spread 계산 불안정

**위치:** `prediction/predictor.py` `_rule_based()` 메서드

`spread` 값이 feature_snapshot에 없으면 `0.0`을 사용해 spread 페널티가 무효화됩니다.  
실시간 거래에서 유동성 경계를 초과하는 신호가 고신뢰도로 나올 수 있습니다.

**개선:** `spread`가 없거나 0일 때 기본값을 `0.05` (1 tick)으로 설정해  
spread 페널티가 보수적으로 작동하도록 처리

---

### 3-4. `ebest_live.py`의 `_append_eval_metrics`가 signal=HOLD도 기록

**위치:** `_append_eval_metrics()` (약 821행)

HOLD 신호도 동일하게 eval_metrics에 기록되어  
승률/정확도 지표가 HOLD 결과로 희석됩니다.  
HOLD는 판단 보류이므로 정확도 계산 분모에서 제외하는 것이 일반적입니다.

---

### 3-5. `telegram_notifier.py`의 `_should_send` 로직이 단순 신호 필터만 지원

현재 `_should_send(signal)`은 BUY/SELL/HOLD 기준으로만 발송 여부를 결정합니다.  
동일 방향 신호가 연속으로 오면 **반복 알림**이 발송됩니다.  
이전 방향과 비교하는 deduplication 로직이 없습니다.

**개선:** 직전 발송 신호 저장 후 동일 방향 연속 발송 억제 (최소 간격 또는 방향 전환 시에만 발송)

---

## 4. 코드 품질 개선

### 4-1. 과도한 `try/except Exception: pass` 사용

`pipeline.py` 전체에서 빈 `except Exception: pass` 블록이 수십 곳 사용됩니다.  
장애 원인 추적을 어렵게 하며, 특히 `__init__()` 내에서 발생하는 설정 오류를 조용히 삼킵니다.

**개선 방향:**
- 초기화 코드(`__init__`)의 critical path에서는 예외를 최소한 `logger.warning()`으로 기록
- `# Best-effort` 주석이 없는 블록에서는 최소한 debug 로그 추가

---

### 4-2. `constants.py` 모델명이 실제 API와 불일치 가능성

```python
CLAUDE_MODEL = "claude-sonnet-4-20250514"  # 실제 모델: claude-sonnet-4-5-20251022
```

모델명은 API 업데이트 주기가 짧아 하드코딩이 위험합니다.  
`CLAUDE_FALLBACK_MODELS`의 `"claude-3-7-sonnet-latest"` 등 `-latest` suffix를 기본값으로 사용하는 것이 안전합니다.

---

### 4-3. `train.py` / `data_builder.py`의 레이블 생성이 lookahead bias에 노출

**위치:** `data_builder.py`

미래 가격으로 레이블을 생성할 때 `.shift(-horizon)` 계산이  
인덱스 정렬 없이 수행되면 **리샘플링 경계에서 미래 봉 데이터가 섞이는** 위험이 있습니다.  
백테스트 과적합의 주요 원인입니다.

**개선:** 레이블 생성 직전 `df = df.sort_index()` 명시 및 단위 테스트 추가

---

### 4-4. `config.py`의 `validate()` 메서드가 LLM 설정 일부를 검증하지 않음

`llm_min_interval_sec`, `dual_llm_primary_provider`, `llm_timeout_sec` 등의 범위 검증이  
`validate()` 내에서 이루어지지 않아 잘못된 config로 시작해도 오류가 예측 시점에 발생합니다.

---

## 5. 성능 최적화

### 5-1. OBI delta 계산 시 전체 `_ob_records` 복사 비용

```python
with self._ob_lock:
    hist = list(self._ob_records)  # 최대 seq_len(60)개 전체 복사
```

초당 1회 실행되므로 매초 deque 전체를 list로 복사합니다.  
`obi_5s_ago` 계산에는 마지막 5개 항목만 필요하므로  
`itertools.islice(reversed(self._ob_records), 10)` 등으로 부분 접근이 효율적입니다.

---

### 5-2. `_compute_flow_features`에서 `list(ticks)` 전체 복사

```python
for rec in reversed(list(ticks)):  # 최대 100,000개 틱 전체를 list로 변환
```

`futures_ticks`는 `deque(maxlen=100_000)`입니다.  
`reversed()`는 deque를 직접 지원하므로 `list()` 변환 없이 반복 가능합니다.

```python
for rec in reversed(ticks):  # list() 불필요
```

---

## 6. 미완성 기능 / TODO

| 위치 | 내용 |
|------|------|
| `prediction/option_flow_features.py` | `calc_option_minute_micro_features()` 구현됐지만 pipeline에서 미사용 |
| `prediction/context_builder.py` | `build_llm_context()`의 `ob_records` 시계열 요약이 평균값만 사용 (표준편차, 추세 미반영) |
| `adaptive_indicator/simulate_indicators.py` | 시뮬레이션 도구는 있지만 백테스트 리포트 출력 미완성 |
| `prediction/weights_selector.py` | 가중치 파일 자동 선택 로직 존재하나 TFT 가중치 버전 관리 미비 |
| `tests/` | `test_feedback_loop.py`가 `numeric_res` 버그로 인해 피드백 적용 여부를 실제로 검증하지 못함 |

---

## 우선순위 요약

| 우선순위 | 항목 | 영향 |
|---------|------|------|
| 🔴 즉시 | `numeric_res` → `t_res` 변수명 수정 (§1-1) | 피드백 루프 전혀 작동 안 함 |
| 🔴 즉시 | `_judge_provider_direct` rate limit 체크 (§1-2) | dual_llm 모드에서 429 반복 발생 |
| 🟠 중요 | `_judge_with_timeout` 429 쿨다운 설정 (§1-4) | 단일 provider 모드 429 재발 가능 |
| 🟠 중요 | `_adaptive_enabled` 이중 초기화 정리 (§1-3) | config 비활성화 옵션 무효 |
| 🟡 개선 | LLM cooldown을 config에서 오버라이드 (§2-1) | 운영 중 파라미터 조정 불가 |
| 🟡 개선 | 캐시 키 해시화 (§2-3) | 메모리 낭비 |
| 🟡 개선 | `_compute_flow_features` list 변환 제거 (§5-2) | 성능 |
| 🟢 권장 | 결과 dict 중복 코드 헬퍼 분리 (§3-1) | 유지보수성 |
| 🟢 권장 | Telegram 중복 알림 억제 (§3-5) | UX |

