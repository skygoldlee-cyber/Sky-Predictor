# 트레이딩 시그널 생성 가이드

수치 예측, LLM 판단, 가드레일 등을 통합하여 최종 트레이딩 시그널을 생성하는 프로세스 가이드.

## 개요

트레이딩 시그널 생성 시스템은 다양한 예측 모델과 판단 시스템을 통합하여 BUY/SELL/HOLD 신호를 생성합니다. 신뢰도 기반 필터링과 가드레일을 통해 리스크를 관리합니다.

### 목적

- 다중 예측 소스 통합
- 신뢰도 기반 필터링
- 리스크 관리 (가드레일)
- 최종 트레이딩 신호 생성

### 대상 독자

- 트레이더
- 시스템 운영자
- 전략 개발자

## 핵심 개념

### 시그널 생성 흐름

```
┌─────────────────────────────────────────────────────────┐
│              트레이딩 시그널 생성 프로세스                  │
│                                                          │
│  1. 수치 예측 (Transformer/TFT/Mamba)                   │
│     ↓                                                   │
│  2. LLM 판단 (Claude/GPT/Gemini)                        │
│     ↓                                                   │
│  3. 앙상블 통합                                        │
│     ↓                                                   │
│  4. 가드레일 검사                                      │
│     ↓                                                   │
│  5. 최종 신호 생성 (BUY/SELL/HOLD)                      │
└─────────────────────────────────────────────────────────┘
```

### 신호 카테고리

| 신호 | 설명 | 조건 |
|------|------|------|
| BUY | 매수 신호 | 확신도 높음, 상승 요인 |
| SELL | 매도 신호 | 확신도 높음, 하락 요인 |
| HOLD | 유지 신호 | 확신도 낮음, 불확실 |

## 설정

### config.json 설정

```json
{
  "prediction": {
    "buy_threshold": 0.64,
    "sell_threshold": 0.36,
    "transformer_weight": 0.5,
    "disagreement_hold": true,
    "disagreement_hold_prob_diff_max": 0.08,
    "ensemble_agreement_confidence_boost": true,
    "ensemble_agreement_prob_diff_max": 0.06
  },
  "trade_gate": {
    "enabled": true,
    "min_confidence": "MEDIUM",
    "min_prob_buy": 0.62,
    "max_prob_sell": 0.38,
    "require_consensus": true,
    "target_profit_pt": 2.0,
    "stop_loss_pt": 1.0
  }
}
```

### 파라미터 설명

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| buy_threshold | float | 0.64 | BUY 신호 임계값 |
| sell_threshold | float | 0.36 | SELL 신호 임계값 |
| transformer_weight | float | 0.5 | Transformer 가중치 |
| disagreement_hold | bool | true | 모델 불일치 시 HOLD |
| disagreement_hold_prob_diff_max | float | 0.08 | 불일치 허용 확률 차이 |
| ensemble_agreement_confidence_boost | bool | true | 앙상블 일치 시 신뢰도 부스트 |
| ensemble_agreement_prob_diff_max | float | 0.06 | 일치 허용 확률 차이 |
| min_confidence | string | "MEDIUM" | 최소 신뢰도 레벨 |
| min_prob_buy | float | 0.62 | 최소 매수 확률 |
| max_prob_sell | float | 0.38 | 최대 매도 확률 |
| require_consensus | bool | true | 합의 요구 여부 |

## 사용 방법

### 기본 사용법

```python
from prediction.pipeline import PredictionPipeline

# 파이프라인 초기화
pipeline = PredictionPipeline(
    buy_threshold=0.64,
    sell_threshold=0.36,
)

# 예측 실행
result = pipeline.predict(
    df=minute_df,
    now_dt=current_time,
    current_price=current_price,
)

# 최종 신호 확인
signal = result.get("signal")  # "BUY", "SELL", "HOLD"
confidence = result.get("confidence")  # "HIGH", "MEDIUM", "LOW"
```

### 신뢰도 레벨

| 레벨 | 범위 | 해석 |
|------|------|------|
| HIGH | 0.7 ~ 1.0 | 높은 확신도 |
| MEDIUM | 0.5 ~ 0.7 | 중간 확신도 |
| LOW | 0.0 ~ 0.5 | 낮은 확신도 |

### 앙상블 통합

```python
# Transformer와 TFT 앙상블
transformer_prob = transformer_predictor.predict(features)
tft_prob = tft_predictor.predict(features)

# 가중 평균
ensemble_prob = (
    transformer_weight * transformer_prob +
    (1 - transformer_weight) * tft_prob
)
```

### Disagreement Hold

모델 간 의견 불일치 시 HOLD:

```python
if disagreement_hold:
    prob_diff = abs(transformer_prob - tft_prob)
    if prob_diff > disagreement_hold_prob_diff_max:
        signal = "HOLD"  # 불일치 시 HOLD
```

### Ensemble Agreement Boost

앙상블 일치 시 신뢰도 부스트:

```python
if ensemble_agreement_confidence_boost:
    prob_diff = abs(transformer_prob - tft_prob)
    if prob_diff < ensemble_agreement_prob_diff_max:
        confidence = boost_confidence(confidence)
```

## LLM Action과 수치 예측 통합

```python
# 수치 예측
numeric_prob = ensemble_prob

# LLM 판단
llm_action = llm_judge.judge(context)
llm_confidence = llm_judge.get_confidence()

# 통합
if llm_action == "BUY" and numeric_prob > buy_threshold:
    final_signal = "BUY"
elif llm_action == "SELL" and numeric_prob < sell_threshold:
    final_signal = "SELL"
else:
    final_signal = "HOLD"
```

## Trade Gate와 연동

```python
from trade_gate import TradeGate

gate = TradeGate(
    enabled=True,
    min_confidence="MEDIUM",
    min_prob_buy=0.62,
    max_prob_sell=0.38,
)

# 시그널 필터링
if gate.check(signal, confidence, prob):
    # 시그너 승인
    execute_trade(signal)
else:
    # 시그널 거부
    log_rejection(reason)
```

## 주의사항

### 일반적인 주의사항

1. **임계값 설정**: buy_threshold와 sell_threshold는 시장 상황에 따라 조절 필요
2. **신뢰도 검증**: 실제 트레이딩 전에 백테스트로 신뢰도 검증
3. **가드레일 활용**: 리스크 관리를 위해 가드레일 필수 활용
4. **합의 요구**: require_consensus는 신호 수를 줄이지만 정확도 높임

### 에러 처리

```python
try:
    signal = generate_signal(features, context)
except ModelError:
    # 모델 오류 시 휴리스틱 fallback
    signal = heuristic_signal(features)
except GuardrailViolation:
    # 가드레일 위반 시 HOLD
    signal = "HOLD"
```

## 관련 문서

- [머신러닝 엔진 개요](./ML_ENGINE_OVERVIEW.md)
- [LLM 판단 시스템 가이드](./LLM_JUDGE_SYSTEM_GUIDE.md)
- [가드레일 시스템 가이드](./GUARDRAIL_SYSTEM_GUIDE.md)

---

**문서 버전**: 1.0  
**작성일**: 2026-04-25  
**마지막 수정**: 2026-04-25
