# 옵션 센티먼트 분석기 통합 가이드

## 개요

옵션 센티먼트 분석기는 Skew, Volume PCR, OI PCR을 조합하여 장의 방향성을 진단하는 모듈입니다. 기존 파이프라인에 통합되어 옵션 스냅샷 생성 시 자동으로 센티먼트 분석을 수행하며, 주요 이벤트 발생 시 로그 및 텔레그램 알림을 전송합니다.

## 구조

### 1. 핵심 클래스

#### `OptionSentimentAnalyzer` (`indicators/option_sentiment.py`)

옵션 센티먼트 분석을 수행하는 메인 클래스입니다.

```python
class OptionSentimentAnalyzer:
    def __init__(
        self,
        config: Optional[OptionSentimentConfig] = None,
        event_callback: Optional[Callable[[SentimentSignal], None]] = None,
    ) -> None:
```

- `config`: 분석 설정 (임계값, 가중치, 알림 설정)
- `event_callback`: 이벤트 발생 시 호출되는 콜백 함수

#### `SentimentSignal`

분석 결과를 담는 데이터클래스입니다.

```python
@dataclass
class SentimentSignal:
    direction: MarketDirection           # 상승/하락/중립
    confidence: float                      # 신뢰도 (0.0 ~ 1.0)
    skew: float                           # Skew 값
    volume_pcr: float                      # Volume PCR
    oi_pcr: float                         # OI PCR
    skew_signal: str                       # 개별 지표 신호
    volume_pcr_signal: str
    oi_pcr_signal: str
    event_type: str                        # 이벤트 유형
    prev_direction: Optional[MarketDirection]  # 이전 방향성
    prev_confidence: Optional[float]       # 이전 신뢰도
    details: Dict[str, Any]
```

### 2. 통합 지점

#### `OptionMixin` (`prediction/option_mixin.py`)

옵션 관련 기능을 담당하는 Mixin에 센티먼트 분석기가 통합되었습니다.

- `_init_option_sentiment_analyzer()`: 분석기 초기화
- `_analyze_option_sentiment()`: 센티먼트 분석 수행
- `_build_option_snapshot_safe()`: 옵션 스냅샷 생성 시 센티먼트 분석 호출

#### `PredictionPipeline` (`prediction/pipeline.py`)

파이프라인 초기화 시 센티먼트 분석기가 자동으로 초기화됩니다.

```python
# __init__() 끝부분
try:
    self._init_option_sentiment_analyzer(config_path=str(self._config_path))
except Exception:
    self._option_sentiment_analyzer = None
```

## 설정 (config.json)

`option_sentiment` 섹션에 분석 설정을 추가합니다.

```json
{
  "option_sentiment": {
    "skew_bullish_threshold": 0.05,
    "skew_bearish_threshold": -0.05,
    "volume_pcr_bullish_threshold": 0.8,
    "volume_pcr_bearish_threshold": 1.2,
    "oi_pcr_bullish_threshold": 0.9,
    "oi_pcr_bearish_threshold": 1.1,
    "skew_weight": 0.3,
    "volume_pcr_weight": 0.35,
    "oi_pcr_weight": 0.35,
    "min_confidence": 0.6,
    "direction_change_alert": true,
    "confidence_spike_alert": true,
    "confidence_spike_threshold": 0.8
  }
}
```

### 설정 파라미터

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `skew_bullish_threshold` | Skew가 이 값 이상이면 강세 신호 (비율 기준) | 0.05 |
| `skew_bearish_threshold` | Skew가 이 값 이하이면 약세 신호 (비율 기준) | -0.05 |
| `volume_pcr_bullish_threshold` | Volume PCR이 이 값 이하이면 강세 신호 | 0.8 |
| `volume_pcr_bearish_threshold` | Volume PCR이 이 값 이상이면 약세 신호 | 1.2 |
| `oi_pcr_bullish_threshold` | OI PCR이 이 값 이하이면 강세 신호 | 0.9 |
| `oi_pcr_bearish_threshold` | OI PCR이 이 값 이상이면 약세 신호 | 1.1 |
| `skew_weight` | Skew 가중치 | 0.3 |
| `volume_pcr_weight` | Volume PCR 가중치 | 0.35 |
| `oi_pcr_weight` | OI PCR 가중치 | 0.35 |
| `min_confidence` | 최소 신뢰도 (이하이면 중립) | 0.5 |
| `direction_change_alert` | 방향성 전환 알림 활성화 | true |
| `confidence_spike_alert` | 신뢰도 급증 알림 활성화 | true |
| `confidence_spike_threshold` | 신뢰도 급증 임계값 | 0.8 |
| `require_neutral_transit` | 중립 경유 필수 (허위 알림 억제) | false |

**참고**: 가중치 합계가 1.0이 아닐 경우 자동 정규화되며 경고 로그가 출력됩니다.

## 통합 방식

### 1. 초기화

파이프라인 초기화 시 `_init_option_sentiment_analyzer()`가 호출됩니다.

```python
def _init_option_sentiment_analyzer(self, config_path: Optional[str] = None) -> None:
    """옵션 센티먼트 분석기를 초기화한다."""
    from indicators.option_sentiment import OptionSentimentAnalyzer, load_config_from_dict
    import json

    cfg_path = str(config_path or getattr(self, "_config_path", "config.json"))
    config_dict = {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f) or {}
    except Exception:
        config_dict = {}

    sentiment_config = load_config_from_dict(config_dict)

    # 콜백 함수: 이벤트 발생 시 텔레그램 전송
    def on_sentiment_event(signal):
        notifier = getattr(self, "_notifier", None)
        if notifier is None:
            return

        signal_dict = {
            "direction": signal.direction.value,
            "confidence": signal.confidence,
            "skew": signal.skew,
            "volume_pcr": signal.volume_pcr,
            "oi_pcr": signal.oi_pcr,
            "event_type": signal.event_type,
            "prev_direction": signal.prev_direction.value if signal.prev_direction else None,
            "prev_confidence": signal.prev_confidence,
        }

        current_price = float(getattr(self, "_last_underlying_price", 0.0) or 0.0)
        if current_price <= 0.0:
            logger.warning(
                "[OptionSentiment] 기초자산 가격 미수신으로 알림 생략 (event=%s, direction=%s)",
                signal.event_type, signal.direction.value
            )
            return

        notifier.send_option_sentiment_alert(signal_dict, current_price)

    self._option_sentiment_analyzer = OptionSentimentAnalyzer(
        sentiment_config, event_callback=on_sentiment_event
    )
```

### 2. 분석 수행

옵션 스냅샷 생성 시 `_analyze_option_sentiment()`가 호출됩니다.

```python
def _analyze_option_sentiment(
    self, opt_snap: Dict[str, Any], current_price: float
) -> Optional[Dict[str, Any]]:
    """옵션 스냅샷에서 센티먼트를 분석한다."""
    analyzer = getattr(self, "_option_sentiment_analyzer", None)
    if analyzer is None:
        return None

    # 옵션 스냅샷에서 필요한 데이터 추출
    # None 체크와 0 체크를 분리하여 유효한 0값이 손실되지 않도록 함
    iv_skew_raw = opt_snap.get("iv_skew")
    iv_skew = float(iv_skew_raw) if iv_skew_raw is not None else 1.0
    # iv_skew = put_iv / call_iv (비율)
    # 우리가 필요한 것은 call_iv - put_iv (차이)
    # 따라서 skew = 1 - iv_skew로 근사 계산
    #   예: put_iv=20%, call_iv=18% → iv_skew=1.11 → skew=-0.11 (약세)
    #   예: put_iv=18%, call_iv=20% → iv_skew=0.90 → skew=+0.10 (강세)
    # 주의: 1 - ratio는 퍼센트가 아니라 비율 차이이므로 임계값(0.05)은 비율 기준임
    skew = 1.0 - iv_skew

    pcr_volume_raw = opt_snap.get("pcr_volume")
    pcr_volume = float(pcr_volume_raw) if pcr_volume_raw is not None else 1.0

    pcr_oi_raw = opt_snap.get("pcr_oi")
    pcr_oi = float(pcr_oi_raw) if pcr_oi_raw is not None else 1.0

    # 센티먼트 분석
    signal = analyzer.analyze(skew=skew, volume_pcr=pcr_volume, oi_pcr=pcr_oi)

    return {
        "direction": signal.direction.value,
        "confidence": signal.confidence,
        "skew": signal.skew,
        "volume_pcr": signal.volume_pcr,
        "oi_pcr": signal.oi_pcr,
        "skew_signal": signal.skew_signal,
        "volume_pcr_signal": signal.volume_pcr_signal,
        "oi_pcr_signal": signal.oi_pcr_signal,
        "event_type": signal.event_type,
        "prev_direction": signal.prev_direction.value if signal.prev_direction else None,
        "prev_confidence": signal.prev_confidence,
    }
```

### 3. 스냅샷에 결과 저장

```python
def _build_option_snapshot_safe(self, *, current_price: float, update_prev: bool = True) -> Dict[str, Any]:
    snap = build_option_snapshot(...)

    # 옵션 센티먼트 분석
    try:
        sentiment_result = self._analyze_option_sentiment(snap, float(current_price))
        if sentiment_result:
            snap["_sentiment"] = sentiment_result
    except Exception:
        pass

    return snap
```

## 데이터 흐름

```
옵션 틱 데이터
    ↓
build_option_snapshot()
    ↓
_build_option_snapshot_safe()
    ↓
_analyze_option_sentiment()
    ↓
OptionSentimentAnalyzer.analyze()
    ├─ 개별 지표 신호 분류 (skew, volume_pcr, oi_pcr)
    ├─ 가중 평균 점수 계산
    ├─ 방향성 결정 (bullish/bearish/neutral)
    ├─ 신뢰도 계산
    ├─ 이벤트 감지 (방향성 전환, 신뢰도 급증)
    └─ 이벤트 발생 시 콜백 호출
        ↓
로그 출력
    ↓
텔레그램 알림 (send_option_sentiment_alert)
    ↓
결과를 opt_snap["_sentiment"]에 저장
```

## 이벤트 감지

### 1. 방향성 전환 (direction_change)

이전 방향성과 다르고, 중립이 아닌 방향으로 변경될 때 감지됩니다.

```python
if cfg.direction_change_alert and prev_direction is not None:
    changed = prev_direction != direction
    not_neutral = direction != MarketDirection.NEUTRAL

    if cfg.require_neutral_transit:
        # 중립을 경유한 전환만 인정
        via_neutral = prev_direction == MarketDirection.NEUTRAL
        if changed and not_neutral and via_neutral:
            events.append("direction_change")
    else:
        # 직전환도 허용 (기본)
        if changed and not_neutral:
            events.append("direction_change")
```

### 2. 신뢰도 급증 (confidence_spike)

신뢰도가 임계값 이상으로 상승할 때 감지됩니다.

```python
if cfg.confidence_spike_alert and prev_confidence is not None:
    if confidence >= cfg.confidence_spike_threshold and prev_confidence < cfg.confidence_spike_threshold:
        events.append("confidence_spike")
```

### 3. 복합 이벤트

두 이벤트가 동시에 발생하면 "+"로 연결하여 반환합니다 (예: "direction_change+confidence_spike"). 텔레그램 알림에서는 두 이벤트를 모두 표시합니다.

## 텔레그램 알림

### `send_option_sentiment_alert()` (`telegram/notifier.py`)

이벤트 발생 시 텔레그램으로 알림을 전송합니다.

```python
def send_option_sentiment_alert(
    self,
    signal: Dict[str, Any],
    current_price: float,
    *,
    cooldown_sec: Optional[float] = None,
    force: bool = False,
) -> bool:
```

### 알림 포맷

```
📈 옵션 센티먼트 이벤트 ⚠️
종합 방향: 상승 (신뢰도: 85.0%)
현재가: 350.00
━━━━━━━━━━━━
Skew: +3.50%
Volume PCR: 0.75
OI PCR: 0.85
⚠️ 이벤트: 방향성 전환 (중립 → 상승)
```

### 쿨다운

중복 알림을 방지하기 위해 쿨다운 기능이 있습니다 (기본 300초).

**이벤트 타입별 쿨다운**: 각 이벤트 타입(direction_change, confidence_spike)별로 별도의 쿨다운 타이머를 관리합니다. 방향성 전환 알림 후 50초 뒤 신뢰도 급증이 발생해도 각각 독립적으로 쿨다운이 적용됩니다.

```python
# 이벤트 타입별 마지막 알림 시간 관리
_last_sentiment_alert_epochs = {
    "direction_change": 0.0,
    "confidence_spike": 0.0,
}

# 쿨다운 체크
last_epoch = last_alerts.get(event_type, 0.0)
elapsed = float(time.time()) - last_epoch
if elapsed < _cooldown:
    logger.debug("[TG][SENTIMENT] 쿨다운 중 (event_type=%s) — 전송 생략", event_type)
    return False
```

## 사용 예시

### 1. 독립 사용

```python
from indicators.option_sentiment import OptionSentimentAnalyzer, load_config_from_dict
import json

# 설정 로드
with open("config.json") as f:
    config_dict = json.load(f)

sentiment_config = load_config_from_dict(config_dict)
analyzer = OptionSentimentAnalyzer(sentiment_config)

# 분석 실행
signal = analyzer.analyze(
    skew=0.03,        # 콜 IV - 풋 IV
    volume_pcr=0.85,  # 풀 볼륨 / 콜 볼륨
    oi_pcr=0.95       # 풀 OI / 콜 OI
)

print(f"방향성: {signal.direction}")
print(f"신뢰도: {signal.confidence}")
print(analyzer.get_llm_context(signal))
```

### 2. 콜백 등록

```python
from indicators.option_sentiment import OptionSentimentAnalyzer, load_config_from_dict
from telegram.notifier import create_notifier_from_config
import json

# 설정 로드
with open("config.json") as f:
    config_dict = json.load(f)

sentiment_config = load_config_from_dict(config_dict)
notifier = create_notifier_from_config()

# 콜백 함수
def on_sentiment_event(signal):
    signal_dict = {
        "direction": signal.direction.value,
        "confidence": signal.confidence,
        "skew": signal.skew,
        "volume_pcr": signal.volume_pcr,
        "oi_pcr": signal.oi_pcr,
        "event_type": signal.event_type,
        "prev_direction": signal.prev_direction.value if signal.prev_direction else None,
        "prev_confidence": signal.prev_confidence,
    }
    current_price = 350.0
    notifier.send_option_sentiment_alert(signal_dict, current_price)

# 분석기 생성 (콜백 등록)
analyzer = OptionSentimentAnalyzer(sentiment_config, event_callback=on_sentiment_event)

# 분석 실행 (이벤트 발생 시 로그 + 텔레그램 전송)
signal = analyzer.analyze(skew=0.03, volume_pcr=0.85, oi_pcr=0.95)
```

### 3. 파이프라인 통합 (자동)

파이프라인에서는 자동으로 초기화 및 분석이 수행됩니다. 별도의 코드 추가가 필요 없습니다.

```python
# 파이프라인 초기화 시 자동으로 센티먼트 분석기 초기화
pipeline = PredictionPipeline(
    config_path="config.json",
    notifier=notifier,
    ...
)

# 옵션 스냅샷 생성 시 자동으로 센티먼트 분석 수행
opt_snap = pipeline._build_option_snapshot_safe(current_price=350.0)

# 분석 결과 확인
sentiment = opt_snap.get("_sentiment")
if sentiment:
    print(f"방향성: {sentiment['direction']}")
    print(f"신뢰도: {sentiment['confidence']}")
```

## 로그 예시

### 방향성 전환

```
[OptionSentiment] 방향성 전환: neutral -> bullish (신뢰도: 0.75)
[TG][SENTIMENT] 옵션 센티먼트 알림 전송 (event=direction_change, direction=bullish, confidence=0.75)
```

### 신뢰도 급증

```
[OptionSentiment] 신뢰도 급증: 0.65 -> 0.82 (임계값: 0.80)
[TG][SENTIMENT] 옵션 센티먼트 알림 전송 (event=confidence_spike, direction=bullish, confidence=0.82)
```

## LLM 컨텍스트

`get_llm_context()` 메서드를 사용하여 LLM에 전달할 수 있는 형식의 텍스트를 생성할 수 있습니다.

### 메서드 시그니처

```python
def get_llm_context(self, signal: SentimentSignal) -> str:
    """LLM 프롬프트에 삽입할 센티먼트 요약 문자열 반환.

    Args:
        signal: SentimentSignal 객체

    Returns:
        str: 센티먼트 요약 문자열 (한국어)
    """
```

### 사용 예시

```python
context = analyzer.get_llm_context(signal)
print(context)
```

출력 예시:

```
[옵션 센티먼트]
종합 방향: 상승 (신뢰도: 75.0%)
  - Skew: +3.50% (강세)
  - Volume PCR: 0.75 (강세)
  - OI PCR: 0.85 (강세)
⚠️ 이벤트: 방향성 전환 (중립 → 상승)
```

## 주의사항

1. **Skew 계산**: 옵션 스냅샷의 `iv_skew`는 `put_iv / call_iv` 형태입니다. 센티먼트 분석기에서는 이를 `1 - iv_skew`로 변환하여 사용합니다. 주의: 1 - ratio는 퍼센트가 아니라 비율 차이이므로 임계값(0.05)은 비율 기준입니다.
   - 예: call_iv=20%, put_iv=18% → skew=+0.10 (강세)
   - 예: call_iv=18%, put_iv=20% → skew=-0.11 (약세)

2. **쿨다운**: 텔레그램 알림은 이벤트 타입별로 별도 쿨다운이 적용됩니다 (기본 300초). `force=True`로 우회할 수 있습니다.

3. **초기화 실패**: 센티먼트 분석기 초기화 실패 시 파이프라인은 정상 동작하지만 센티먼트 분석은 수행되지 않습니다.

4. **복합 이벤트**: 방향성 전환과 신뢰도 급증이 동시에 발생하면 두 이벤트가 모두 감지되며 텔레그램 알림에도 모두 표시됩니다 (예: "direction_change+confidence_spike").

5. **가중치 정규화**: 가중치 합계가 1.0이 아닐 경우 자동 정규화되며 경고 로그가 출력됩니다.

6. **데이터 무결성**: `or 1.0` 폴백 대신 `None 체크와 0 체크 분리`를 사용하여 유효한 0값이 손실되지 않도록 합니다.

7. **가격 미수신**: 기초자산 가격이 0 이하일 경우 알림이 생략되며 경고 로그가 출력됩니다.

## 관련 파일

- `indicators/option_sentiment.py`: 옵션 센티먼트 분석기 구현
- `prediction/option_mixin.py`: 파이프라인 통합 (초기화, 분석 수행)
- `prediction/pipeline.py`: 파이프라인 메인 클래스
- `telegram/notifier.py`: 텔레그램 알림 전송
- `config.json`: 설정 파일
