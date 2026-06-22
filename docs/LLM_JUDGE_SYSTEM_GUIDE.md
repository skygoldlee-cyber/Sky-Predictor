# LLM 판단 시스템 가이드

Claude, GPT, Gemini 등 다양한 LLM 제공자를 사용하여 시장 방향을 판단하는 시스템 가이드.

## 개요

LLM 판단 시스템은 대형 언어 모델을 사용하여 시장 데이터를 분석하고 BUY/SELL/HOLD 신호를 생성합니다. 다중 제공자 지원과 fallback 메커니즘을 통해 안정성을 높입니다.

### 목적

- 시장 데이터의 자연어 해석
- 기술적 지표 기반 판단
- 다중 LLM 제공자 지원
- 안정적인 fallback 메커니즘

### 대상 독자

- 시스템 운영자
- LLM 통합 개발자
- 트레이딩 전략 개발자

## 핵심 개념

### LLM 판단 흐름

```
┌─────────────────────────────────────────────────────────┐
│                   LLM 판단 시스템                        │
│                                                          │
│  1. 컨텍스트 빌더 (context_builder.py)                  │
│     ↓                                                   │
│  2. LLM 판단 (llm_judge.py)                            │
│     ↓                                                   │
│  3. 판단 결과 통합 (prediction_mixin.py)                 │
│     ↓                                                   │
│  4. 최종 신호 생성                                     │
└─────────────────────────────────────────────────────────┘
```

### 지원되는 LLM 제공자

| 제공자 | 모델 | 특징 |
|--------|------|------|
| Anthropic | Claude | 높은 추론 능력, 긴 컨텍스트 |
| OpenAI | GPT-4 | 강력한 일반 능력 |
| Google | Gemini | 다중 모달 지원 |

### dual_llm 모드

두 개의 LLM을 동시에 사용하여 판단의 신뢰도를 높입니다:

```
LLM 1 (Primary Provider)
    ↓
LLM 2 (Secondary Provider)
    ↓
일치 여부 확인
    ↓
일치: 신뢰도 부스트
불일치: HOLD 또 중간 확신도
```

## 설정

### config.json 설정

```json
{
  "use_llm": true,
  "preferred_provider": "gemini",
  "prediction": {
    "dual_llm": false,
    "dual_llm_primary_provider": "gemini",
    "llm_timeout_sec": 20.0,
    "gemini_timeout_sec": 45.0,
    "llm_min_interval_sec": 30.0,
    "llm_provider_cooldown_on_timeout_sec": 90.0
  }
}
```

### 파라미터 설명

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| use_llm | bool | true | LLM 사용 여부 |
| preferred_provider | string | "gemini" | 선호 LLM 제공자 |
| dual_llm | bool | false | 이중 LLM 모드 |
| dual_llm_primary_provider | string | "gemini" | 이중 LLM 주 제공자 |
| llm_timeout_sec | float | 20.0 | LLM 타임아웃 (초) |
| gemini_timeout_sec | float | 45.0 | Gemini 전용 타임아웃 |
| llm_min_interval_sec | float | 30.0 | LLM 호출 최소 간격 |
| llm_provider_cooldown_on_timeout_sec | float | 90.0 | 타임아웃 시 제공자 쿨다운 |

## 사용 방법

### 기본 사용법

```python
from prediction.pipeline import PredictionPipeline

# 파이프라인 초기화 (LLM 활성화)
pipeline = PredictionPipeline(
    use_llm=True,
    preferred_provider="gemini",
)

# 예측 실행 (LLM 판단 포함)
result = pipeline.predict(
    df=minute_df,
    now_dt=current_time,
    current_price=current_price,
)

# LLM 판단 결과 확인
llm_action = result.get("llm_action")  # "BUY", "SELL", "HOLD"
llm_confidence = result.get("llm_confidence")
```

### dual_llm 모드 사용

```json
{
  "dual_llm": true,
  "dual_llm_primary_provider": "gemini"
}
```

```python
# dual_llm 모드에서는 두 개의 LLM이 판단
# 두 판단이 일치하면 신뢰도 부스트
# 불일치하면 HOLD 또는 중간 확신도
```

### LLM 입력 컨텍스트

LLM 입력 컨텍스트는 `prediction/context_builder.py`에서 생성됩니다:

```python
from prediction.context_builder import build_llm_context

context = build_llm_context(
    adaptive_features=adapt_feats,
    st_state=st_state,
    zz_state=zz_state,
    option_snapshot=opt_snap,
    market_state=market_state,
)

# context는 JSON으로 직렬화되어 LLM 프롬프트에 포함
```

### Provider Fallback

LLM 제공자가 실패하면 자동으로 다른 제공자로 fallback:

```
Gemini 실패 → Claude → GPT → 휴리스틱 fallback
```

## 판단 로직

### LLM 판단 카테고리

| 카테고리 | 설명 | 조건 |
|----------|------|------|
| BUY | 상승 신호 | 확신도 높음, 상승 요인 우세 |
| SELL | 하락 신호 | 확신도 높음, 하락 요인 우세 |
| HOLD | 중립 신호 | 확신도 낮음, 요인 혼재 |

### 확신도 레벨

| 레벨 | 범위 | 해석 |
|------|------|------|
| HIGH | 0.7 ~ 1.0 | 높은 확신도 |
| MEDIUM | 0.5 ~ 0.7 | 중간 확신도 |
| LOW | 0.0 ~ 0.5 | 낮은 확신도 |

## 캐싱 및 Rate Limiting

### 캐싱

동일한 입력에 대해 LLM을 반복 호출하지 않도록 캐싱:

```python
# 캐시 키 생성 (dual_llm 모드)
cache_key = hashlib.md5(
    f"dual={dual}|prov={primary}|{system}|{user}".encode()
).hexdigest()
```

### Rate Limiting

LLM 호출 간격 제한:

```python
# llm_min_interval_sec: 최소 호출 간격 (기본 30초)
# 타임아웃 시 제공자 쿨다운 (기본 90초)
```

## 주의사항

### 일반적인 주의사항

1. **API Key 관리**: LLM 제공자 API Key는 환경변수 또는 `config.secrets.json`에 저장
2. **타임아웃 설정**: 네트워크 지연을 고려하여 적절한 타임아웃 설정
3. **비용 관리**: LLM 호출은 비용이 발생하므로 호출 빈도 조절
4. **캐싱 활용**: 중복 호출을 방지하기 위해 캐싱 적극 활용

### 에러 처리

```python
try:
    llm_action = llm_judge.judge(context)
except TimeoutError:
    # 타임아웃 시 fallback
    llm_action = heuristic_fallback(context)
except APIError:
    # API 오류 시 provider fallback
    llm_action = try_other_provider(context)
```

## 관련 문서

- [LLM 입력 테이블](./LLM_INPUT_TABLE.md)
- [예측 알고리즘](./Prediction_Algorithm.md)
- [런타임 API 레퍼런스](./runtime/README.md)

---

**문서 버전**: 1.0  
**작성일**: 2026-04-25  
**마지막 수정**: 2026-04-25
