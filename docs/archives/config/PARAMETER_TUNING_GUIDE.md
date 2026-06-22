# 파라미터 튜닝 가이드

> **백테스트 기반 파라미터 최적화 가이드**
> 작성일: 2026-04-26

---

## 개요

본 가이드는 `scripts/parameter_tuner.py`를 사용하여 TradeExecutionGate의 파라미터를 백테스트 기반으로 최적화하는 방법을 설명합니다.

## 목차

1. 개요
2. 튜닝 가능한 파라미터
3. 튜닝 방법
4. 사용 예시
5. 결과 해석
6. 최적 파라미터 적용

---

## 2. 튜닝 가능한 파라미터

### 2.1 진입 관련 파라미터

| 파라미터 | 설명 | 기본값 | 후보값 |
|----------|------|--------|--------|
| `min_confidence` | 최소 신뢰도 | MEDIUM | LOW, MEDIUM, HIGH |
| `min_prob_buy` | 매수 최소 확률 | 0.62 | 0.60, 0.62, 0.65 |
| `max_prob_sell` | 매도 최대 확률 | 0.38 | 0.35, 0.38, 0.40 |
| `min_consecutive_signals` | 최소 연속 신호 | 2 | 1, 2, 3 |

### 2.2 청산 관련 파라미터

| 파라미터 | 설명 | 기본값 | 후보값 |
|----------|------|--------|--------|
| `target_profit_pt` | 목표수익 (pt) | 2.0 | 1.5, 2.0, 2.5, 3.0 |
| `stop_loss_pt` | 손절 (pt) | 1.0 | 0.8, 1.0, 1.2, 1.5 |

### 2.3 Trailing Stop 관련 파라미터

| 파라미터 | 설명 | 기본값 | 후보값 |
|----------|------|--------|--------|
| `trailing_stop_enabled` | Trailing Stop 활성화 | false | true, false |
| `trailing_stop_activation_pt` | Trailing 시작 이익 (pt) | 1.0 | 0.8, 1.0, 1.2 |
| `trailing_stop_distance_pt` | Trailing 거리 (pt) | 0.5 | 0.3, 0.5, 0.7 |

### 2.4 리스크 관리 파라미터

| 파라미터 | 설명 | 기본값 | 후보값 |
|----------|------|--------|--------|
| `max_consecutive_losses` | 최대 연속 손실 | 3 | 2, 3, 4 |
| `max_daily_loss_pt` | 일일 최대 손실 (pt) | 5.0 | 3.0, 5.0, 7.0 |

---

## 3. 튜닝 방법

### 3.1 Grid Search

모든 파라미터 조합을 체계적으로 탐색합니다.

**장점**: 최적해를 찾을 확률 높음
**단점**: 조합 수가 많으면 시간 오래 걸림

**사용 조건**: 파라미터 후보값이 적을 때 (조합 수 < 10,000)

### 3.2 Random Search

랜덤하게 파라미터 조합을 선택하여 탐색합니다.

**장점**: 빠름, 고차원 공간에서 효율적
**단점**: 최적해를 놓칠 수 있음

**사용 조건**: 파라미터 후보값이 많을 때, 빠른 탐색 필요 시

---

## 4. 사용 예시

### 4.1 Random Search (기본)

```bash
# 기본 Random Search (50회 반복)
python scripts/parameter_tuner.py --log-dir trade_history/

# 100회 반복
python scripts/parameter_tuner.py --log-dir trade_history/ --n-iterations 100

# 결과 파일 지정
python scripts/parameter_tuner.py --log-dir trade_history/ --output my_results.json
```

### 4.2 Grid Search

```bash
# 전체 Grid Search
python scripts/parameter_tuner.py --log-dir trade_history/ --method grid

# 특정 파라미터만 Grid Search
python scripts/parameter_tuner.py \
    --log-dir trade_history/ \
    --method grid \
    --target-profit 1.5 2.0 2.5 \
    --stop-loss 0.8 1.0 1.2

# Trailing Stop 파라미터 Grid Search
python scripts/parameter_tuner.py \
    --log-dir trade_history/ \
    --method grid \
    --trailing-activation 0.8 1.0 1.2 \
    --trailing-distance 0.3 0.5 0.7
```

### 4.3 결과 출력

```bash
# 상위 20개 결과 출력
python scripts/parameter_tuner.py --log-dir trade_history/ --top-n 20
```

---

## 5. 결과 해석

### 5.1 출력 예시

```
================================================================================
파라미터 튜닝 결과 (상위 10개)
================================================================================

#1 (Score: 0.4523)
--------------------------------------------------------------------------------
파라미터:
  target_profit_pt: 2.5
  stop_loss_pt: 1.0
  trailing_stop_enabled: true
  trailing_stop_activation_pt: 1.0
  trailing_stop_distance_pt: 0.5
  max_consecutive_losses: 3
  max_daily_loss_pt: 5.0
  min_confidence: MEDIUM
  min_prob_buy: 0.62
  max_prob_sell: 0.38
  min_consecutive_signals: 2

성능:
  총 거래: 45
  승률: 62.22%
  총 수익: 15.30%
  평균 수익/거래: 0.34%
  최대 낙폭: 8.50%
  Sharpe Ratio: 1.85
```

### 5.2 점수 계산

종합 점수는 다음 가중치로 계산됩니다:

- 승률: 30%
- 총 수익: 40%
- 최대 낙폭 (MDD): -20%
- Sharpe Ratio: 10%

```
score = win_rate * 0.3 + (total_profit_pct / 100) * 0.4 - (max_drawdown_pct / 100) * 0.2 + min(sharpe_ratio / 2, 1) * 0.1
```

### 5.3 결과 선택 기준

1. **총 거래 수**: 너무 적으면 신뢰도 낮음 (최소 20거래 권장)
2. **승률**: 50% 이상 권장
3. **총 수익**: 양수여야 함
4. **최대 낙폭**: 20% 이하 권장
5. **Sharpe Ratio**: 1.0 이상 권장

---

## 6. 최적 파라미터 적용

### 6.1 config.json 업데이트

튜닝 결과로 나온 최적 파라미터를 `config.json`에 적용합니다.

```json
{
  "trade_gate": {
    "target_profit_pt": 2.5,
    "stop_loss_pt": 1.0,
    "trailing_stop_enabled": true,
    "trailing_stop_activation_pt": 1.0,
    "trailing_stop_distance_pt": 0.5,
    "max_consecutive_losses": 3,
    "max_daily_loss_pt": 5.0,
    "min_confidence": "MEDIUM",
    "min_prob_buy": 0.62,
    "max_prob_sell": 0.38,
    "min_consecutive_signals": 2
  }
}
```

### 6.2 백엔드 재시작

파라미터 변경 후 백엔드를 재시작하여 새 설정을 적용합니다.

```bash
# 백엔드 재시작
python main.py
```

### 6.3 모니터링

새 파라미터로 운영 시 실제 성능을 모니터링합니다.

- 일일 승률
- 일일 손익
- 리스크 한도 준수 여부
- Trailing Stop 작동 여부

---

## 7. 고급 사용법

### 7.1 커스텀 파라미터 공간

`scripts/parameter_tuner.py`의 `ParameterSpace` 클래스를 수정하여 커스텀 파라미터 공간을 정의할 수 있습니다.

```python
@dataclass
class ParameterSpace:
    target_profit_pt: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    stop_loss_pt: List[float] = field(default_factory=lambda: [0.5, 0.8, 1.0, 1.2, 1.5])
    # ... 다른 파라미터
```

### 7.2 커스텀 점수 함수

`_simulate_with_params` 메서드의 점수 계산 로직을 수정하여 커스텀 점수 함수를 사용할 수 있습니다.

```python
# 종합 점수 (가중평균)
# 승률 40%, 수익 30%, MDD -20%, Sharpe 10%
score = (
    win_rate * 0.4 +
    (total_profit_pct / 100) * 0.3 -
    (max_drawdown_pct / 100) * 0.2 +
    min(sharpe_ratio / 2, 1) * 0.1
)
```

### 7.3 OHLCV 데이터 활용

OHLCV 데이터를 제공하면 더 정확한 시뮬레이션이 가능합니다.

```bash
python scripts/parameter_tuner.py \
    --log-dir trade_history/ \
    --data data/ohlcv.csv \
    --method random \
    --n-iterations 100
```

---

## 8. 주의사항

### 8.1 과적합 방지

- 과거 데이터에 과도하게 최적화된 파라미터는 미래 성과를 보장하지 않음
- 훈련 데이터와 테스트 데이터 분리 권장
- 월간/분기별 재튜닝 권장

### 8.2 데이터 충분성

- 최소 100거래 이상의 로그 데이터 권장
- 다양한 시장 상황 포함 권장 (상승/하락/횡보)

### 8.3 리스크 관리

- 수익률만 보지 말고 MDD도 고려
- 리스크 한도 파라미터는 보수적으로 설정 권장

---

## 9. 문제 해결

### 9.1 조합 수가 너무 많음

```
[TUNER] 전체 조합 수: 50000
[TUNER] 조합 수가 너무 많습니다. Random Search를 권장합니다.
```

**해결**: Random Search 사용 또는 파라미터 후보값 줄이기

### 9.2 로그 파일 없음

```
[TUNER] 로드된 로그: 0개 (파일: 0개)
```

**해결**: `--log-dir` 경로 확인, 로그 파일 존재 확인

### 9.3 결과가 모두 0점

**원인**: 필터링 조건이 너무 엄격하여 거래가 없음

**해결**: 파라미터 공간 완화 (min_confidence 낮추기 등)

---

**문서 버전**: 1.0  
**최종 갱신**: 2026-04-26
