# Market Regime Classifier 가이드

## 개요

MarketRegimeClassifier는 시장 상태를 변동성과 방향성 기준으로 분류하여 적합한 전략을 선택하도록 지원합니다. 옵션 센티먼트와 기술적 지표를 결합하여 분류 정확도를 높입니다.

## 주요 기능

- **변동성 상태 판단**: ATR, 표준편차를 사용하여 고변동/저변동/정상 상태 분류
- **방향성 판단**: ADX + DI+/DI- + 이동평균 기울기 + Market Structure 조합으로 상승/하락/횡보 추세 분류
- **시장 레짐 분류**: 변동성과 방향성을 조합하여 11가지 레짐 분류
- **옵션 센티먼트 통합**: 옵션 센티먼트 신호로 기술적 분류 신뢰도 보정
- **읽기 전용 레짐 레이블**: 레짐은 ZigZag 파라미터를 직접 제어하지 않고 LLM 컨텍스트와 알림에만 사용

## 시장 레짐 종류

| 레짐 | 설명 | 적합 전략 |
|------|------|----------|
| `HIGH_VOL_NO_DIRECTION` | 고변동+무방향 (흔들기) | 짧은 스캘핑 (Short Scalping) |
| `HIGH_VOL_UP` | 고변동+상승 (강한 상승 추세) | 돌파추종 (Breakout Following) |
| `HIGH_VOL_DOWN` | 고변동+하락 (강한 하락 추세) | 공매도 돌파추종 (Short Breakout Following) |
| `NORMAL_VOL_NO_DIRECTION` | 정상 변동성+무방향 | 표준 스윙 트레이딩 |
| `NORMAL_VOL_UP` | 정상 변동성+상승 | 스윙 트레이딩 (Swing Trading) |
| `NORMAL_VOL_DOWN` | 정상 변동성+하락 | 스윙 숏 (Swing Short) |
| `LOW_VOL_NO_DIRECTION` | 저변동+무방향 (횡보) | Mean Reversion |
| `LOW_VOL_UP` | 저변동+상승 (느린 상승) | 스윙 트레이딩 (Swing Trading) |
| `LOW_VOL_DOWN` | 저변동+하락 (느린 하락) | 스윙 숏 (Swing Short) |
| `OPENING_EVENT` | 장초반 이벤트장 | 장초반 스캘핑 (Opening Scalping) |
| `NEWS_EVENT` | 뉴스 급등락장 | 뉴스 트레이딩 (News Trading) 또는 관망 |

## 기술적 지표

### ATR (Average True Range)
- **목적**: 변동성 측정
- **계산**: Wilder's RMA 방식 사용
- **임계값** (KP200 선물 기준):
  - 고변동: ATR/가격 ≥ 0.5%
  - 저변동: ATR/가격 ≤ 0.15%
  - 정상: 그 사이

### ADX (Average Directional Index)
- **목적**: 추세 강도 측정
- **임계값** (KP200 선물 기준):
  - 강한 추세: ADX ≥ 20
  - 약한 추세: ADX ≤ 12
  - 횡보: 그 사이

### DI+/DI- (Directional Indicators)
- **목적**: 추세 방향 측정
- **해석**:
  - +DI > -DI: 상승 우세
  - -DI > +DI: 하락 우세

### 표준편차
- **목적**: 가격 변동성 측정
- **계산 기간**: 20봉

## 향상된 방향성 분석 (Enhanced Trend Analysis)

### 개요

ADX만으로는 방향성 판단에 충분하지 않습니다. 따라서 3축 조합으로 방향성을 판단합니다:

1. **ADX + DI+/DI-**: 추세 강도 + 방향
2. **이동평균 기울기**: MA20, MA60 기울기
3. **Market Structure**: Higher High/Lower Low 구조

### 이동평균 기울기

- **MA20**: 단기 추세
- **MA60**: 장기 추세
- **해석**:
  - MA20 기울기 > 0 AND MA60 기울기 > 0: 상승
  - MA20 기울기 < 0 AND MA60 기울기 < 0: 하락
  - 그 외: 횡보

### Market Structure

- **피벗 구조 분석**: 로컬 고점/저점 감지
- **해석**:
  - Higher High + Higher Low: 상승 구조 (uptrend)
  - Lower High + Lower Low: 하락 구조 (downtrend)
  - 그 외: 횡보 구조 (ranging)

### 투표 방식 결정

| 축 | 신호 | 가중치 |
|----|------|--------|
| DI | +DI > -DI → UP, 그 외 → DOWN | 1 |
| MA 기울기 | 양수 → UP, 음수 → DOWN, 혼합 → NEUTRAL | 1 |
| Market Structure | uptrend → UP, downtrend → DOWN, ranging → NEUTRAL | 1 |

**결정 규칙**:
- UP 표표 > DOWN 표표: 상승
- DOWN 표표 > UP 표표: 하락
- 동수일 경우: DI 우선

### 활성화/비활성화

config.json에서 제어:
```json
{
  "market_regime": {
    "enable_enhanced_trend": true,
    "ma_short_period": 20,
    "ma_long_period": 60
  }
}
```

### 로그 예시

```
[MarketRegime] 향상된 추세 분석: up (DI: up, MA: up, Structure: uptrend, Votes: ['up', 'up', 'up'])
[MarketRegime] 향상된 추세 분석: down (DI: down, MA: neutral, Structure: ranging, Votes: ['down', 'neutral', 'ranging'])
```

## 옵션 센티먼트 통합

### 개요

옵션 센티먼트는 파생상품 시장의 심리를 반영하는 선행 지표입니다. 기술적 지표와 결합하여 시장 레짐 분류 정확도를 높입니다.

### 옵션 지표

| 지표 | 설명 | 해석 |
|------|------|------|
| **Skew** | call_iv - put_iv | 양수: 콜 프리미엄 비쌈 → 강세<br>음수: 풋 프리미엄 비쌈 → 약세 |
| **Volume PCR** | put_volume / call_volume | 낮을수록 강세, 높을수록 약세 |
| **OI PCR** | put_OI / call_OI | 낮을수록 강세, 높을수록 약세 |

### 신호 결합 로직 (다단계 필터링)

기술적 신호와 옵션 센티먼트가 일치하면 신뢰도를 상향, 불일치하면 하향합니다.

| 기술적 신호 | 옵션 센티먼트 | 신뢰도 보정 |
|-------------|---------------|-------------|
| 상승 | 강세 (skew>0, PCR<1) | +0.2 |
| 하락 | 약세 (skew<0, PCR>1) | +0.2 |
| 상승 | 약세 (skew<0, PCR>1) | -0.2 |
| 하락 | 강세 (skew>0, PCR<1) | -0.2 |
| 기타 (중립, 혼합) | - | 유지 |

### 옵션 센티먼트 판단 규칙

```python
# 강세 조건
sentiment_bullish = (skew > 0) and (volume_pcr < 1.0) and (oi_pcr < 1.0)

# 약세 조건
sentiment_bearish = (skew < 0) and (volume_pcr > 1.0) and (oi_pcr > 1.0)
```

## 설정

### config.json

```json
{
  "market_regime": {
    "enable_option_sentiment": false,
    "sentiment_confidence_boost": 0.2,
    "sentiment_confidence_penalty": 0.2,
    "enable_enhanced_trend": true,
    "ma_short_period": 20,
    "ma_long_period": 60
  }
}
```

### 파라미터 설명

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `enable_option_sentiment` | boolean | false | 옵션 센티먼트 활성화 여부 |
| `sentiment_confidence_boost` | float | 0.2 | 센티먼트 일치 시 신뢰도 상향 폭 (0.0 ~ 1.0) |
| `sentiment_confidence_penalty` | float | 0.2 | 센티먼트 불일치 시 신뢰도 하향 폭 (0.0 ~ 1.0) |
| `enable_enhanced_trend` | boolean | true | 향상된 추세 분석 활성화 여부 |
| `ma_short_period` | int | 20 | 단기 이동평균 기간 |
| `ma_long_period` | int | 60 | 장기 이동평균 기간 |

### MarketRegimeClassifier 초기화 파라미터

```python
MarketRegimeClassifier(
    atr_period=14,                          # ATR 계산 기간
    adx_period=14,                          # ADX 계산 기간
    std_period=20,                          # 표준편차 계산 기간
    vol_high_threshold=0.005,               # 고변동 기준 (ATR/가격 비율, KP200 기준)
    vol_low_threshold=0.0015,               # 저변동 기준 (ATR/가격 비율, KP200 기준)
    adx_trend_threshold=20,                 # 추세 기준 (ADX, KP200 기준)
    adx_weak_threshold=12,                  # 횡보 기준 (ADX, KP200 기준)
    opening_minutes=30,                     # 장초반 기준 (분)
    market_open_hour=8,                     # 장 시작 시간 (시) - KP200 선물 08:45
    market_open_minute=45,                  # 장 시작 시간 (분) - KP200 선물 08:45
    enable_option_sentiment=False,          # 옵션 센티먼트 활성화 여부
    sentiment_confidence_boost=0.2,          # 센티먼트 일치 시 신뢰도 상향 폭
    sentiment_confidence_penalty=0.2,        # 센티먼트 불일치 시 신뢰도 하향 폭
    ma_short_period=20,                     # 단기 이동평균 기간
    ma_long_period=60,                      # 장기 이동평균 기간
    enable_enhanced_trend=True,             # 향상된 추세 분석 활성화 여부
)
```

## 사용 예시

### 기본 사용 (기술적 지표만)

```python
from services.market_regime_classifier import MarketRegimeClassifier
import pandas as pd

# 분류기 초기화
classifier = MarketRegimeClassifier()

# 시장 상태 분류
market_state = classifier.classify(df)

if market_state:
    print(f"레짐: {market_state.regime.value}")
    print(f"신뢰도: {market_state.confidence:.2%}")
    print(f"변동성: {market_state.volatility_state.value}")
    print(f"추세: {market_state.trend_direction.value}")
```

### 옵션 센티먼트 활성화

```python
# 분류기 초기화 (옵션 센티먼트 활성화)
classifier = MarketRegimeClassifier(
    enable_option_sentiment=True,
    sentiment_confidence_boost=0.2,
    sentiment_confidence_penalty=0.2,
)

# 옵션 데이터와 함께 분류
market_state = classifier.classify(
    df,
    skew=0.1,           # call_iv - put_iv
    volume_pcr=0.85,   # put_volume / call_volume
    oi_pcr=0.9,        # put_OI / call_OI
)
```

### config.json에서 설정 로드

```python
# chart_viewer.py에서 자동으로 config 로드
# config.json의 market_regime 섹션에서 설정 읽기
```

## 로그 예시

### 옵션 센티먼트 비활성화 시
```
[ChartViewerWidget] 시장 레짐 분류기 초기화 완료 (옵션 센티먼트: False, boost: 0.20, penalty: 0.20)
[ChartViewer] 시장 레짐: LOW_VOL_UP (신뢰도: 0.65)
```

### 옵션 센티먼트 활성화 시
```
[ChartViewerWidget] 시장 레짐 분류기 초기화 완료 (옵션 센티먼트: True, boost: 0.20, penalty: 0.20)
[ChartViewer] 옵션 데이터 추출: skew=0.1234, volume_pcr=0.85, oi_pcr=0.92
[MarketRegime] 기술적 상승 + 옵션 강세: 신뢰도 0.65 → 0.85
[ChartViewer] 시장 레짐: LOW_VOL_UP (신뢰도: 0.85)
```

### 옵션 센티먼트 불일치 시
```
[ChartViewer] 옵션 데이터 추출: skew=-0.0567, volume_pcr=1.15, oi_pcr=1.08
[MarketRegime] 기술적 상승 + 옵션 약세: 신뢰도 0.65 → 0.45 (보정)
[ChartViewer] 시장 레짐: LOW_VOL_UP (신뢰도: 0.45)
```

## 파라미터 튜닝 가이드

### sentiment_confidence_boost/penalty 조정

| 시나리오 | boost | penalty | 설명 |
|----------|-------|---------|------|
| 보수적 | 0.1 | 0.1 | 옵션 센티먼트 영향을 최소화 |
| 기본 | 0.2 | 0.2 | 균형 잡힌 영향 |
| 공격적 | 0.3 | 0.3 | 옵션 센티먼트 영향을 최대화 |

### 임계값 조정

```python
# 고변동 기준 조정 (변동성이 큰 시장)
classifier = MarketRegimeClassifier(
    vol_high_threshold=0.03,  # 2% → 3%
    vol_low_threshold=0.01,   # 0.5% → 1%
)

# 추세 기준 조정 (추세가 뚜렷한 시장)
classifier = MarketRegimeClassifier(
    adx_trend_threshold=20,  # 25 → 20 (더 민감)
    adx_weak_threshold=10,   # 15 → 10 (더 민감)
)
```

## 백테스트 가이드

### 1. 기술적 분류만 사용 (베이스라인)
```python
classifier = MarketRegimeClassifier(enable_option_sentiment=False)
# 과거 데이터로 백테스트
# 정확도 기록
```

### 2. 옵션 센티먼트 활성화
```python
classifier = MarketRegimeClassifier(enable_option_sentiment=True)
# 동일한 과거 데이터로 백테스트
# 정확도 기록 후 비교
```

### 3. 파라미터 튜닝
```python
# boost/penalty 값을 변경하며 최적값 찾기
for boost in [0.1, 0.2, 0.3]:
    for penalty in [0.1, 0.2, 0.3]:
        classifier = MarketRegimeClassifier(
            enable_option_sentiment=True,
            sentiment_confidence_boost=boost,
            sentiment_confidence_penalty=penalty,
        )
        # 백테스트 수행
```

## 주의사항

### 데이터 가용성
- 옵션 데이터 (skew, PCR)가 항상 존재하지 않을 수 있음
- 옵션 데이터 없으면 기존 기술적 분류만 수행 (fallback)

### 시간 동기화
- 옵션 데이터와 OHLC 데이터의 타임스탬프 불일치 가능
- 최신 옵션 데이터를 캐싱하여 사용

### 과적합 위험
- 너무 많은 지표를 결합하면 과적합 가능성
- 백테스트를 통한 효과 검증 필수

### KOSPI200 옵션 특성
- 통상 풋 skew(음수 방향)가 형성됨
- iv_skew = put_iv / call_iv 비율 사용 시 변환 필요:
  ```python
  skew = 1.0 - iv_skew
  ```

## 참고 문서

- [option_sentiment.py](../indicators/option_sentiment.py) - 옵션 센티먼트 분석기
- [regime_param_mapper.py](../indicators/regime_param_mapper.py) - 레짐 분류 및 레이블 제공 (읽기 전용)
- [adaptive_zigzag.py](../indicators/adaptive_zigzag.py) - ZigZag 파라미터 결정 (Layer A × Layer B)

## 레짐 역할 분리 (3-Layer 구조)

### Layer A: 세션 시간대 테이블
- **목적**: 무지연, 순환 없는 파라미터 결정
- **구현**: `session_min_wave_atr_ratio_table`, `session_min_wave_bars_table`
- **특징**: KP200 선물 일중 변동성 패턴 직접 인코딩

### Layer B: ATR 백분위 배율
- **목적**: 상대적 변동성 판단 (1봉 지연)
- **구현**: `AdaptiveParameterAdjuster._calc_atr_percentile()`, `get_vol_ratio()`
- **특징**: 절대값 임계값 없이 당일 ATR 분포 기준 백분위 사용

### Layer C: 레짐 레이블 (읽기 전용)
- **목적**: 시장 상태 설명 및 LLM/알림용
- **구현**: `MarketRegimeClassifier`, `RegimeParamMapper.classify()`
- **특징**: ZigZag 파라미터에 직접 관여하지 않음

### 구조적 이점
- **순환 의존 제거**: 레짐이 ZigZag 파라미터를 조정하지 않음 → 자기 강화 피드백 제거
- **지연 최소화**: 세션 시간대 테이블 (무지연) + ATR 백분위 (1봉 지연) → 기존 13분 지연 제거
- **역할 분리**: 레짐은 "설명 도구", 파라미터는 "반응 도구"로 분리

## 변경 이력

- **2026-05-10**: 3-Layer 역할 분리 구조 리팩터링
  - 레짐 파라미터 직접 제어 제거 → 읽기 전용 레이블로 변경
  - Layer A (세션 시간대 테이블) + Layer B (ATR 백분위)로 파라미터 결정
  - 순환 의존 제거, 지연 최소화 (13분 → 0~1봉)
  - NORMAL_VOL_* 레짐 추가 (정상 변동성 분류 개선)
  - 임계값 조정 (KP200 선물 현실에 맞게)
- **2026-05-09**: 향상된 추세 분석 기능 추가
  - 이동평균 기울기 분석 추가 (MA20, MA60)
  - Market Structure 분석 추가 (Higher High/Lower Low)
  - 3축 투표 방식으로 방향성 판단 강화
  - config.json 제어 기능 추가 (enable_enhanced_trend, ma_short_period, ma_long_period)
- **2026-05-09**: 옵션 센티먼트 통합 기능 추가
  - config.json 제어 기능 추가
  - 다단계 필터링 로직 구현
  - 신뢰도 보정 기능 추가
