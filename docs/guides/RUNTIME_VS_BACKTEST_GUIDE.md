# 실시간 vs 백테스트 파이프라인 차이 가이드

실시간 파이프라인과 백테스트 파이프라인의 차이점을 설명하는 가이드.

## 개요

SkyPredictor는 실시간 트레이딩과 백테스트를 위한 두 가지 파이프라인을 제공합니다. 각 파이프라인의 차이점을 이해하고 적절한 상황에 사용하는 것이 중요합니다.

### 목적

- 두 파이프라인의 차이점 이해
- 적절한 사용 시나리오 파악
- 데이터 소스 차이 이해
- 타이밍 차이 이해

### 대상 독자

- 시스템 운영자
- 트레이더
- 백테스트 엔지니어

## 핵심 개념

### 파이프라인 비교

| 특성 | 실시간 파이프라인 | 백테스트 파이프라인 |
|------|------------------|-------------------|
| 데이터 소스 | eBest 실시간 틱 | 저장된 히스토리 데이터 |
| 타이밍 | 실시간 (분봉마다) | 순차적 (전체 데이터) |
| LLM 사용 | 사용 (dual_llm 가능) | 사용 안 함 (속도) |
| 피드백 적용 | 실시간 적용 | 적용 안 함 |
| 가드레일 | 활성화 | 선택적 |
| 목적 | 실시간 트레이딩 | 전략 검증 |

### 실시간 파이프라인 구조

```
┌─────────────────────────────────────────────────────────┐
│              실시간 파이프라인 (main.py)                  │
│                                                          │
│  1. eBest API 연결                                      │
│     ↓                                                   │
│  2. 실시간 틱 수집                                     │
│     ↓                                                   │
│  3. 분봉 집계                                           │
│     ↓                                                   │
│  4. 예측 모델 추론                                     │
│     ↓                                                   │
│  5. LLM 판단 (선택적)                                  │
│     ↓                                                   │
│  6. 가드레일 검사                                      │
│     ↓                                                   │
│  7. 피드백 적용                                        │
│     ↓                                                   │
│  8. 트레이딩 신호 전송                                 │
└─────────────────────────────────────────────────────────┘
```

### 백테스트 파이프라인 구조

```
┌─────────────────────────────────────────────────────────┐
│            백테스트 파이프라인 (data_builder.py)          │
│                                                          │
│  1. 히스토리 데이터 로드                               │
│     ↓                                                   │
│  2. 데이터 전처리                                       │
│     ↓                                                   │
│  3. 순차적 예측                                       │
│     ↓                                                   │
│  4. 예측 모델 추론                                     │
│     ↓                                                   │
│  5. 가드레일 검사 (선택적)                             │
│     ↓                                                   │
│  6. 성능 평가                                         │
│     ↓                                                   │
│  7. 결과 저장                                           │
└─────────────────────────────────────────────────────────┘
```

## 주요 차이점

### 1. 데이터 소스

**실시간 파이프라인**:
```python
from ebestapi.live import EbestLive

ebest = EbestLive()
ebest.connect()
ticks = ebest.get_realtime_ticks()
```

**백테스트 파이프라인**:
```python
import pandas as pd

# 저장된 히스토리 데이터 로드
df = pd.read_pickle("data/futures_minute_history.pkl")
```

### 2. 타이밍

**실시간 파이프라인**:
- 매 분마다 실행
- 실시간 이벤트 기반
- 지연 허용 (1~5초)

**백테스트 파이프라인**:
- 전체 데이터 순차적 처리
- 배치 처리
- 지연 없음

### 3. LLM 사용

**실시간 파이프라인**:
```python
# LLM 사용 가능
use_llm = True
dual_llm = True
llm_action = llm_judge.judge(context)
```

**백테스트 파이프라인**:
```python
# LLM 사용 안 함 (속도)
use_llm = False
# 휴리스틱 fallback 사용
```

### 4. 피드백 적용

**실시간 파이프라인**:
```python
# 실시간 피드백 적용
update_adaptive_weights(trade_result)
transformer_weight = get_transformer_weight()
```

**백테스트 파이프라인**:
```python
# 피드백 적용 안 함
transformer_weight = 0.5  # 고정
```

### 5. 가드레일

**실시간 파이프라인**:
```python
# 가드레일 필수 활성화
guard_basis_hold_thr = 2.5
guard_atm_spread_pct_thr = 1.5
```

**백테스트 파이프라인**:
```python
# 가드레일 선택적
guardrail_enabled = False  # 비활성화 가능
```

## 사용 시나리오

### 실시간 파이프라인 사용

**적합한 상황**:
- 실시간 트레이딩
- LLM 판단 필요
- 피드백 기반 최적화
- 실시간 리스크 관리

**실행 방법**:
```bash
python main.py
```

### 백테스트 파이프라인 사용

**적합한 상황**:
- 전략 백테스트
- 모델 성능 평가
- 파라미터 최적화
- 대량 데이터 처리

**실행 방법**:
```python
from prediction.data_builder import build_adaptive_features

# 백테스트 실행
features, context = build_adaptive_features(
    minute_ohlcv=historical_data,
    adaptive_config=config,
)
```

## 설정 차이

### 실시간 설정

```json
{
  "prediction": {
    "use_llm": true,
    "dual_llm": true,
    "feedback_threshold_ticks": 10
  },
  "trade_gate": {
    "enabled": true,
    "guard_basis_hold_thr": 2.5
  }
}
```

### 백테스트 설정

```json
{
  "prediction": {
    "use_llm": false,
    "dual_llm": false,
    "feedback_threshold_ticks": 0
  },
  "trade_gate": {
    "enabled": false
  }
}
```

## 주의사항

### 일반적인 주의사항

1. **데이터 일치**: 백테스트 데이터는 실시간 데이터와 최대한 일치해야 합니다.
2. **로직 일치**: 두 파이프라인의 예측 로직은 일치해야 합니다.
3. **타이밍 차이**: 백테스트는 지연이 없으므로 과대 평가 가능성 있습니다.
4. **LLM 비용**: 실시간 파이프라인은 LLM 호출 비용이 발생합니다.

### 에러 처리

```python
# 실시간 파이프라인
try:
    signal = predict_realtime(tick_data)
except ConnectionError:
    # 연결 오류 시 재시도
    retry_connection()
except TimeoutError:
    # 타임아웃 시 HOLD
    signal = "HOLD"

# 백테스트 파이프라인
try:
    signal = predict_backtest(historical_data)
except DataError:
    # 데이터 오류 시 해당 봉 스킵
    skip_bar()
```

## 관련 문서

- [일일 틱 학습 런북](./DAILY_TICK_TRAINING_RUNBOOK.md)
- [예측 알고리즘](./Prediction_Algorithm.md)
- [런타임 레퍼런스](./runtime/README.md)

---

**문서 버전**: 1.0  
**작성일**: 2026-04-25  
**마지막 수정**: 2026-04-25
