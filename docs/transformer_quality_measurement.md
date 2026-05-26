# 트랜스포머 예측 품질 측정 로직

> **작성일**: 2026-05-26  
> **대상 프로젝트**: SkyPredictor (KOSPI200 선물 방향 예측 파이프라인)

---

## 1. 배경 — 기존 품질 측정의 한계

### 기존에 있던 것

| 항목 | 파일 | 상태 |
|------|------|------|
| Brier Score 기반 앙상블 가중치 조정 | `prediction/predictor.py` `AdaptiveEnsembleWeightTracker` | 동작 중 |
| 피드백 큐 평가 카운터 | `prediction/mixins/feedback_mixin.py` | 동작 중 |
| 오프라인 Brier Score + ECE 계산 | `prediction/calibration_metrics.py` | 수동 실행 전용 |
| GUI 메트릭 표시 | `gui/controller_rt_helpers.py` | `transformer_weight` 1개만 |

### 기존 코드의 문제

`get_metrics()`가 반환하는 품질 관련 수치는 `feedback_transformer_weight` 하나뿐이었다.
현재 가중치가 0.6이라는 숫자만 있고, "오늘 몇 건 중 몇 건이 맞았는가"라는 실제 정확도는
어디에도 기록되지 않았다.

**누락된 4가지**:

1. 일별 방향 정확도 이력 — 오늘 모델이 어제보다 나빠졌는지 알 수 없음
2. ECE 장중 측정 — `calibration_metrics.py`가 존재하나 파이프라인에 연결 안 됨
3. LLM 개별 정확도 — GPT·Gemini·Claude 합산만 있고 개별 비교 불가
4. 신뢰도 등급별 적중률 — HIGH 신호가 실제로 더 맞는지 검증 코드 없음
5. 품질 저하 자동 알림 — 가중치 급락 시 Telegram 미알림

---

## 2. 추가된 파일 목록

| 파일 | 변경 유형 | 역할 |
|------|-----------|------|
| `prediction/transformer_quality_tracker.py` | **신규** | 품질 측정 핵심 모듈 |
| `prediction/mixins/feedback_mixin.py` | 수정 | tracker 호출 연결 |
| `prediction/pipeline.py` | 수정 | tracker 초기화 + `get_metrics()` 확장 |
| `scripts/run_daily_backtest.py` | 수정 | 장마감 후 품질 요약 출력 |

---

## 3. `TransformerQualityTracker` 상세

### 3-1. 파일 위치

```
prediction/
└── transformer_quality_tracker.py   ← 신규
```

같은 폴더의 `calibration_metrics.py`, `calibration_report.py`와 동일 성격의
품질 측정 모듈군으로 배치한다.

### 3-2. 클래스 구조

```python
class TransformerQualityTracker:
    """트랜스포머 예측 품질 실시간 추적기 (thread-safe)."""

    def __init__(
        self,
        notifier=None,                   # TelegramNotifier (선택)
        ece_window: int = 50,            # ECE 슬라이딩 윈도우
        daily_history_days: int = 30,   # 일별 이력 보관 일수
        alert_accuracy_threshold: float = 0.45,  # 정확도 경보 임계값
        alert_ece_threshold: float = 0.15,        # ECE 경보 임계값
        alert_cooldown_sec: float = 1800.0,       # 알림 최소 간격 (30분)
        min_samples_for_alert: int = 10,          # 최소 평가 건수
    ): ...
```

### 3-3. 내부 데이터 구조

```python
@dataclass
class DailyAccuracy:
    """하루치 방향 예측 정확도."""
    date_str:  str
    hits:      int   = 0      # 맞춘 건수
    total:     int   = 0      # 전체 평가 건수
    brier_sum: float = 0.0    # Brier Score 누적합

    @property
    def accuracy(self) -> float: ...   # hits / total
    @property
    def mean_brier(self) -> float: ... # brier_sum / total


@dataclass
class ConfidenceBucket:
    """HIGH / MEDIUM / LOW 신뢰도 등급별 집계."""
    label: str
    hits:  int = 0
    total: int = 0


@dataclass
class LLMAccuracy:
    """GPT·Gemini·Claude 개별 방향 정확도."""
    name:  str
    hits:  int = 0
    total: int = 0
```

### 3-4. 핵심 메서드

#### `record_evaluation()` — 평가 1건 기록

```python
tracker.record_evaluation(
    correct=True,           # Transformer 예측이 실제와 일치했는가
    prob=0.73,              # Transformer 출력 확률 (0~1)
    confidence="HIGH",      # 신뢰도 등급
    actual_direction="BUY", # 실제 시장 방향
    llm_actions={           # LLM별 예측 action
        "gpt":    "BUY",
        "gemini": "BUY",
        "claude": "HOLD",
    },
    transformer_weight=0.62,
)
```

내부에서 5가지를 동시에 갱신한다:

1. `_daily[today]` — 일별 hits/total/brier 누적
2. `_ece_probs / _ece_labels` — ECE 슬라이딩 윈도우 추가
3. `_confidence_buckets[conf]` — 등급별 hits/total 누적
4. `_llm_accuracy[name]` — LLM별 hits/total 누적
5. `_maybe_fire_alert()` — 경보 조건 확인 후 Telegram 전송

#### `get_metrics_dict()` — `get_metrics()`에 병합

```python
{
    # 오늘 일별 정확도
    "quality_today_accuracy":   0.623,
    "quality_today_total":      48,
    "quality_today_hits":       30,
    "quality_today_brier":      0.1823,

    # ECE (최근 50건 슬라이딩)
    "quality_ece":              0.0821,

    # 신뢰도 등급별
    "quality_high_accuracy":    0.714,
    "quality_high_total":       21,
    "quality_medium_accuracy":  0.600,
    "quality_medium_total":     20,
    "quality_low_accuracy":     0.571,
    "quality_low_total":        7,

    # LLM 개별
    "quality_llm_gpt_accuracy":    0.667,
    "quality_llm_gpt_total":       48,
    "quality_llm_gemini_accuracy": 0.625,
    "quality_llm_gemini_total":    48,
    "quality_llm_claude_accuracy": 0.583,
    "quality_llm_claude_total":    48,
}
```

#### `log_daily_summary()` — 장마감 후 요약 로그

```
=======================================================
  트랜스포머 예측 품질 일별 요약
=======================================================
  오늘 방향 정확도:  62.3%  (48건)
  오늘 Brier Score:  0.1823
  ECE (최근 50건): 0.0821
  Transformer 가중치: 0.638
-------------------------------------------------------
  신뢰도 등급별 적중률
    HIGH    : 71.4%  (21건)
    MEDIUM  : 60.0%  (20건)
    LOW     : 57.1%  (7건)
-------------------------------------------------------
  LLM 개별 방향 적중률
    gpt     : 66.7%  (48건)
    gemini  : 62.5%  (48건)
    claude  : 58.3%  (48건)
=======================================================
```

---

## 4. 판정 임계값

| 지표 | 임계값 | 방향 | 비고 |
|------|--------|------|------|
| 방향 정확도 | `< 0.45` | 낮을수록 나쁨 | 랜덤(0.5) 이하 = 경보 |
| ECE | `> 0.15` | 높을수록 나쁨 | 과신 경고 |
| Telegram 쿨다운 | `1800초` | — | 30분 이내 중복 억제 |
| 최소 평가 건수 | `10건` | — | 미만이면 경보 억제 |

임계값은 `TransformerQualityTracker` 생성자 파라미터로 조정 가능하다.

---

## 5. Telegram 경보 메시지 형식

```
⚠ 트랜스포머 품질 저하 감지

경고: 방향 정확도 저하: 41.2% < 45.0% (34건)

오늘 현황
  방향 정확도: 41.2% (34건)
  Brier Score: 0.2134
  ECE: 0.0912
  Transformer 가중치: 0.521

신뢰도별 적중률
  HIGH: 38.5% (13건)
  MEDIUM: 44.4% (18건)
  LOW: 33.3% (3건)

LLM별 적중률
  gpt: 47.1% (34건)
  gemini: 44.1% (34건)
  claude: 41.2% (34건)
```

---

## 6. 파이프라인 연결 구조

```
[실시간 틱 수신]
      ↓
[get_prediction()]          prediction/mixins/prediction_mixin.py
  └─ 예측 결과를 feedback_queue에 enqueue
      ↓
[prediction_minutes 경과 후]
      ↓
[_maybe_process_feedback()]  prediction/mixins/feedback_mixin.py
  ├─ 실제 방향 판정 (BUY / SELL / HOLD)
  ├─ AdaptiveEnsembleWeightTracker.update()   ← 기존
  └─ TransformerQualityTracker.record_evaluation()  ← 신규 추가
      ↓
[get_metrics()]              prediction/pipeline.py
  └─ quality_* 키 14개 포함해 반환  ← 신규 확장
      ↓
[15:45 장마감 → JIF 이벤트]
      ↓
[run_daily_backtest_with_ohlcv()]
  └─ pipeline.log_quality_summary()  ← 신규 추가
```

---

## 7. Telegram notifier 연결 방법

`PipelineTelegramBridge.start()` 호출 직후 한 줄을 추가한다.

```python
bridge.start()
pipeline.set_quality_notifier(notifier)   # ← 이 한 줄
```

`notifier`가 `None`이면 알림 없이 로그만 출력된다.
파이프라인 시작 시점에 notifier가 아직 없어도 안전하게 동작한다.

---

## 8. 신뢰도 등급 자동 추정

`feedback_mixin`에서 `rec`에 `confidence` 키가 없을 때
Transformer 출력 확률의 중심(0.5)으로부터의 거리로 자동 추정한다.

```python
margin = abs(prob - 0.5)

if margin >= 0.25:   → HIGH    (예: prob = 0.75 이상 또는 0.25 이하)
elif margin >= 0.10: → MEDIUM  (예: prob = 0.60~0.75)
else:                → LOW     (예: prob = 0.50~0.60, 애매한 구간)
```

---

## 9. LLM 개별 집계 조건

`rec` 딕셔너리에 `llm_actions` 키가 있을 때만 집계한다.
`actual_direction`이 `"HOLD"`인 경우에는 LLM 집계를 건너뛴다
(방향성이 없는 구간에서의 action 비교는 의미가 없기 때문이다).

`llm_actions`를 피드백 큐에 저장하려면 `prediction_mixin.py`의
`feedback_queue.append()` 시점에 LLM action 정보를 함께 저장해야 한다.
현재 큐에는 `transformer_prob`, `tft_prob`만 포함되어 있으므로
`llm_action` 필드를 추가하는 별도 작업이 필요하다.

---

## 10. 관련 파일 요약

```
prediction/
├── transformer_quality_tracker.py   신규 — 품질 측정 핵심 모듈
├── calibration_metrics.py           기존 — 오프라인 Brier / ECE 함수
├── calibration_report.py            기존 — 오프라인 검증 리포트
├── pipeline.py                      수정 — tracker 초기화, get_metrics 확장
└── mixins/
    └── feedback_mixin.py            수정 — tracker.record_evaluation() 연결

scripts/
└── run_daily_backtest.py            수정 — log_quality_summary() 호출
```
