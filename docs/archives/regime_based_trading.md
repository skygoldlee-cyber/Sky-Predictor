# 레짐 기반 매매 신호 시스템 가이드

## 개요

레짐 기반 매매 신호 시스템은 시장 상태를 변동성과 방향성 기준으로 분류하여 각 레짐에 최적화된 매매 전략을 자동으로 선택합니다. 기술적 지표와 옵션 센티먼트를 결합하여 레짐 분류 정확도를 높이고, **레짐 레이블은 LLM 컨텍스트와 알림에만 사용**하며 ZigZag 파라미터는 세션 시간대 테이블과 ATR 백분위로 결정합니다.

## 시스템 아키텍처 (3-Layer 역할 분리)

```
┌─────────────────────────────────────────────────────────────────┐
│                    OHLCV 데이터 입력                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              MarketRegimeClassifier (Layer C)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  기술적 분석                                              │  │
│  │  - ATR (변동성)                                          │  │
│  │  - ADX + DI+/DI- (추세 강도/방향)                        │  │
│  │  - MA 기울기 (이동평균)                                   │  │
│  │  - Market Structure (피벗 구조)                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  옵션 센티먼트 (선택적)                                   │  │
│  │  - Skew (IV Skew)                                        │  │
│  │  - Volume PCR                                            │  │
│  │  - OI PCR                                                │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
                    MarketState
                    (레짐, 신뢰도, 지표들)
                           │
                           ├──────────────┬──────────────┐
                           ▼              ▼              ▼
                   LLM 컨텍스트     텔레그램 알림    (파라미터 제어 없음)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│        Layer A: 세션 시간대 테이블 (무지연)                       │
│  - session_min_wave_atr_ratio_table                             │
│  - session_min_wave_bars_table                                   │
│  - KP200 선물 일중 변동성 패턴 직접 인코딩                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│        Layer B: ATR 백분위 배율 (1봉 지연)                       │
│  - AdaptiveParameterAdjuster._calc_atr_percentile()             │
│  - get_vol_ratio() (0.85/1.0/1.25)                              │
│  - 절대값 임계값 없이 상대적 변동성 판단                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              AdaptiveZigZag                                     │
│  - Layer A × Layer B 런타임 파라미터로 피봇 감지                │
│  - 피봇 후보 이벤트 발생                                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
                    매매 신호 출력
```

## 시장 레짐 분류

### 레짐 종류

| 레짐 코드 | 설명 | 변동성 | 방향성 | 적합 전략 |
|----------|------|--------|--------|----------|
| `HIGH_VOL_NO_DIRECTION` | 고변동 횡보 (흔들기) | 고 | 무방향 | 짧은 스캘핑 |
| `HIGH_VOL_UP` | 고변동 상승 (강한 상승 추세) | 고 | 상승 | 돌파추종 |
| `HIGH_VOL_DOWN` | 고변동 하락 (강한 하락 추세) | 고 | 하락 | 공매도 돌파추종 |
| `NORMAL_VOL_NO_DIRECTION` | 정상 변동성 횡보 | 중 | 무방향 | 표준 스윙 트레이딩 |
| `NORMAL_VOL_UP` | 정상 변동성 상승 | 중 | 상승 | 스윙 트레이딩 |
| `NORMAL_VOL_DOWN` | 정상 변동성 하락 | 중 | 하락 | 스윙 숏 |
| `LOW_VOL_NO_DIRECTION` | 저변동 횡보 (조용한 횡보) | 저 | 무방향 | Mean Reversion |
| `LOW_VOL_UP` | 저변동 상승 (느린 상승) | 저 | 상승 | 스윙 트레이딩 |
| `LOW_VOL_DOWN` | 저변동 하락 (느린 하락) | 저 | 하락 | 스윙 숏 |
| `OPENING_EVENT` | 장초반 이벤트장 | 고 | - | 장초반 스캘핑 |
| `NEWS_EVENT` | 뉴스 급등락장 | 매우 고 | - | 뉴스 트레이딩 또는 관망 |

### 레짐 결정 로직

```python
# 1. 장초반 이벤트장 우선
if is_opening_session and atr_ratio > vol_high_threshold:
    return MarketRegime.OPENING_EVENT

# 2. 뉴스 급등락장 (매우 높은 변동성)
if atr_ratio > vol_high_threshold * 2:
    return MarketRegime.NEWS_EVENT

# 3. 변동성 상태와 추세 방향 조합
if vol_state == VolatilityState.HIGH:
    if trend_dir == TrendDirection.UP:
        return MarketRegime.HIGH_VOL_UP
    elif trend_dir == TrendDirection.DOWN:
        return MarketRegime.HIGH_VOL_DOWN
    else:
        return MarketRegime.HIGH_VOL_NO_DIRECTION
elif vol_state == VolatilityState.NORMAL:
    if trend_dir == TrendDirection.UP:
        return MarketRegime.NORMAL_VOL_UP
    elif trend_dir == TrendDirection.DOWN:
        return MarketRegime.NORMAL_VOL_DOWN
    else:
        return MarketRegime.NORMAL_VOL_NO_DIRECTION
elif vol_state == VolatilityState.LOW:
    if trend_dir == TrendDirection.UP:
        return MarketRegime.LOW_VOL_UP
    elif trend_dir == TrendDirection.DOWN:
        return MarketRegime.LOW_VOL_DOWN
    else:
        return MarketRegime.LOW_VOL_NO_DIRECTION
```

## 기술적 지표

### 1. 변동성 지표

#### ATR (Average True Range)
- **목적**: 변동성 측정
- **계산**: Wilder's RMA 방식 (기본 기간: 14봉)
- **임계값** (KP200 선물 기준):
  - 고변동: ATR/가격 ≥ 0.5%
  - 정상: 0.15% < ATR/가격 < 0.5%
  - 저변동: ATR/가격 ≤ 0.15%

#### 표준편차
- **목적**: 가격 변동성 보조 지표
- **계산 기간**: 20봉
- **사용**: ATR과 함께 변동성 상태 판단

### 2. 방향성 지표

#### ADX (Average Directional Index)
- **목적**: 추세 강도 측정
- **임계값** (KP200 선물 기준):
  - 강한 추세: ADX ≥ 20
  - 횡보: 12 < ADX < 20
  - 약한 추세: ADX ≤ 12

#### DI+/DI- (Directional Indicators)
- **목적**: 추세 방향 측정
- **해석**:
  - +DI > -DI: 상승 우세
  - -DI > +DI: 하락 우세

### 3. 향상된 추세 분석 (Enhanced Trend Analysis)

ADX만으로는 방향성 판단에 충분하지 않으므로 3축 조합으로 방향성을 판단합니다.

#### 3축 구성

| 축 | 지표 | 신호 | 가중치 |
|----|------|------|--------|
| **DI** | ADX + DI+/DI- | +DI > -DI → UP<br>-DI > +DI → DOWN | 1 |
| **MA 기울기** | MA20, MA60 기울기 | 양수 → UP<br>음수 → DOWN<br>혼합 → NEUTRAL | 1 |
| **Market Structure** | 피벗 구조 | Higher High/Low → UP<br>Lower High/Low → DOWN<br>기타 → NEUTRAL | 1 |

#### 결정 규칙
- UP 표표 > DOWN 표표: 상승
- DOWN 표표 > UP 표표: 하락
- 동수일 경우: DI 우선

#### 활성화/비활성화
```json
{
  "market_regime": {
    "enable_enhanced_trend": true,
    "ma_short_period": 20,
    "ma_long_period": 60
  }
}
```

## 옵션 센티먼트 통합

### 옵션 지표

| 지표 | 설명 | 해석 |
|------|------|------|
| **Skew** | call_iv - put_iv | 양수: 콜 프리미엄 비쌈 → 강세<br>음수: 풋 프리미엄 비쌈 → 약세 |
| **Volume PCR** | put_volume / call_volume | 낮을수록 강세, 높을수록 약세 |
| **OI PCR** | put_OI / call_OI | 낮을수록 강세, 높을수록 약세 |

### 신호 결합 로직

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

## 레짐별 매매 전략

### 1. HIGH_VOL_NO_DIRECTION (고변동 횡보)

**특징**: 변동성이 크지만 명확한 방향성 없음, 흔들기 심함

**전략**: 짧은 스캘핑 (Short Scalping)
- 빠른 진입/진출
- 타이트한 스톱로스
- 작은 수익 목표

**ZigZag 파라미터**:
```python
atr_multiplier: 3.0 ~ 4.5        # 임계값 최대화
confirmation_bars: 2 ~ 3          # 허위 피봇 억제 (실시간 확정 지연 방지)
min_wave_atr_ratio: 2.5 ~ 3.5     # 파동 크기 크게 (KP200 기준)
pivot_threshold_min_pct: 0.40% ~ 0.50%    # 최소 임계값 높게
```

### 2. HIGH_VOL_UP (고변동 상승)

**특징**: 강한 상승 추세, 변동성 큼

**전략**: 돌파추종 (Breakout Following)
- 저항선 돌파 시 진입
- 추세 추종
- 넓은 스톱로스

**ZigZag 파라미터**:
```python
atr_multiplier: 2.0 ~ 3.0        # 중간 임계값
confirmation_bars: 1              # 빠른 확정
min_wave_atr_ratio: 2.0 ~ 3.0    # 중간 파동 크기 (KP200 기준)
pivot_threshold_min_pct: 0.30% ~ 0.40%   # 중간 임계값
```

### 3. HIGH_VOL_DOWN (고변동 하락)

**특징**: 강한 하락 추세, 변동성 큼

**전략**: 공매도 돌파추종 (Short Breakout Following)
- 지지선 하향 돌파 시 진입
- 하락 추세 추종
- 넓은 스톱로스

**ZigZag 파라미터**:
```python
atr_multiplier: 2.0 ~ 3.0
confirmation_bars: 1
min_wave_atr_ratio: 2.0 ~ 3.0    # KP200 기준
pivot_threshold_min_pct: 0.30% ~ 0.40%
```

### 4. LOW_VOL_NO_DIRECTION (저변동 횡보)

**특징**: 조용한 횡보, 변동성 작음

**전략**: Mean Reversion
- 지지/저항 밴드에서 반전 매매
- KP200 선물 기준 의미 있는 반전 포착 (틱 노이즈 제거)
- 타이트한 스톱로스

**ZigZag 파라미터**:
```python
atr_multiplier: 0.8 ~ 1.2        # 낮은 임계값
confirmation_bars: 2 ~ 3          # 확실한 반전 확인 (실시간 확정 지연 방지)
min_wave_atr_ratio: 1.5 ~ 2.5    # KP200 선물 기준 반전 포착
pivot_threshold_min_pct: 0.25% ~ 0.35%   # KP200 기준
```

### 5. LOW_VOL_UP (저변동 상승)

**특징**: 느린 상승 추세, 변동성 작음

**전략**: 스윙 트레이딩 (Swing Trading)
- 중기 스윙 포착
- 지지선에서 매수
- 중간 수익 목표

**ZigZag 파라미터**:
```python
atr_multiplier: 1.0 ~ 1.8        # 중간 임계값
confirmation_bars: 2 ~ 3         # 중간 확정
min_wave_atr_ratio: 1.5 ~ 2.5     # 중간 파동 크기 (KP200 기준)
pivot_threshold_min_pct: 0.25% ~ 0.35%   # 중간 임계값
```

### 6. LOW_VOL_DOWN (저변동 하락)

**특징**: 느린 하락 추세, 변동성 작음

**전략**: 스윙 숏 (Swing Short)
- 중기 스윙 포착
- 저항선에서 매도
- 중간 수익 목표

**ZigZag 파라미터**:
```python
atr_multiplier: 1.0 ~ 1.8
confirmation_bars: 2 ~ 3
min_wave_atr_ratio: 1.5 ~ 2.5     # KP200 기준
pivot_threshold_min_pct: 0.25% ~ 0.35%
```

### 7. OPENING_EVENT (장초반 이벤트장)

**특징**: 장초반 변동성 확대, 노이즈 많음

**전략**: 장초반 스캘핑 (Opening Scalping)
- 장초반 30분 집중
- 빠른 진입/진출
- 보수적 접근

**ZigZag 파라미터**:
```python
atr_multiplier: 4.0 ~ 8.0        # 매우 높은 임계값
confirmation_bars: 1 ~ 2         # 빠른 확정
min_wave_atr_ratio: 2.5 ~ 4.0     # 큰 파동만 (KP200 기준)
pivot_threshold_min_pct: 0.40% ~ 0.50%   # 높은 최소 임계값
```

### 8. NEWS_EVENT (뉴스 급등락장)

**특징**: 급격한 변동, 예측 불가능

**전략**: 뉴스 트레이딩 또는 관망
- 뉴스 기반 방향성 확인
- 관망 권장 (변동성 너무 큼)
- 진입 시 보수적 접근

**ZigZag 파라미터**:
```python
atr_multiplier: 5.0 ~ 8.0        # 최대 임계값
confirmation_bars: 1             # 빠른 확정
min_wave_atr_ratio: 3.0 ~ 4.5     # 매우 큰 파동만 (KP200 기준)
pivot_threshold_min_pct: 0.50% ~ 0.60%   # 최대 최소 임계값
```

## ZigZag 파라미터 결정 (3-Layer 구조)

### Layer A: 세션 시간대 테이블

세션 시간대별 변동성 패턴을 직접 인코딩하여 무지연 파라미터 결정:

```json
{
  "adaptive_indicator": {
    "zigzag": {
      "session_min_wave_atr_ratio_table": [
        ["08:45", "09:30", 3.5],   // 장초반: 고변동
        ["09:30", "10:30", 2.0],   // 오전 활성: 중간
        ["10:30", "13:00", 1.5],   // 중반 안정: 낮음
        ["13:00", "14:30", 1.2],   // 점심 저변동: 최저
        ["14:30", "15:20", 1.8],   // 마감 전 활성: 중간
        ["15:20", "15:31", 2.5]    // 동시호가: 높음
      ]
    }
  }
}
```

### Layer B: ATR 백분위 배율

당일 ATR 분포 기준 상대적 변동성 판단:

```python
# 상위 25%: 고변동 → 배율 1.25
# 하위 25%: 저변동 → 배율 0.85
# 중간 50%: 중간 → 배율 1.0
vol_ratio = adaptive_adjuster.get_vol_ratio()
```

### Layer C: 레짐 레이블 (읽기 전용)

레짐은 ZigZag 파라미터를 직접 제어하지 않고 다음 용도로만 사용:

- **LLM 컨텍스트 빌드**: 시장 상태 설명
- **텔레그램 알림**: 레짐 변화 알림

```python
market_state = regime_mapper.classify(df)
# 레짐 레이블만 반환, 파라미터 제어 없음
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
    
    # 적합 전략 조회
    strategy = classifier.get_suitable_strategy(market_state.regime)
    print(f"적합 전략: {strategy}")
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

### ZigZag 레짐 통합 사용

```python
from indicators.adaptive_zigzag_regime_integration import build_regime_aware_zigzag

# 레짐 기반 ZigZag 생성
zz, mapper = build_regime_aware_zigzag(
    symbol="futures",
    classify_interval_bars=10,
)

# 실시간 봉 처리
for bar in live_feed:
    state = zz.update(bar.high, bar.low, bar.close, bar_time=bar.time)
    
    # 현재 레짐 확인
    current_regime = mapper.stable_regime
    print(f"레짐: {current_regime.value}")
    
    # 피봇 후보 이벤트 처리
    if state.new_swing_signal != "none":
        print(f"피봇 후보: {state.new_swing_signal}")
```

## 로그 예시

### 레짐 분류 로그

```
[ChartViewerWidget] 시장 레짐 분류기 초기화 완료 (옵션 센티먼트: False, boost: 0.20, penalty: 0.20, 향상된 추세: True, MA: 20/60, ADX: 25/15, VOL: 0.020/0.005)
[ChartViewer] 시장 레짐: LOW_VOL_UP (신뢰도: 0.65)
[ChartViewer] 레짐 변경 감지: None → LOW_VOL_UP
[ChartViewer] 레짐 변경 통계: 2026-05-10 - 1번째 변경 (오늘 총 1회)
```

### 향상된 추세 분석 로그

```
[MarketRegime] 향상된 추세 분석: up (DI: up, MA: up, Structure: uptrend, Votes: ['up', 'up', 'up'])
[MarketRegime] 향상된 추세 분석: down (DI: down, MA: neutral, Structure: ranging, Votes: ['down', 'neutral', 'ranging'])
```

### 옵션 센티먼트 로그

```
[ChartViewer] 옵션 데이터 추출: skew=0.1234, volume_pcr=0.85, oi_pcr=0.92
[MarketRegime] 기술적 상승 + 옵션 강세: 신뢰도 0.65 → 0.85
[ChartViewer] 시장 레짐: LOW_VOL_UP (신뢰도: 0.85)
```

### 레짐 파라미터 적용 로그

```
[RegimeParamMapper] 파라미터 적용: regime=high_vol_up | 고변동 상승: 돌파추종 — 빠른 확정, 중간 임계값
  atr_mult=2.500  conf_bars=1  min_wave=1.000  thr_min=0.2000%
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

### 레짐 프로파일 튜닝

```python
# 특정 레짐의 파라미터 조정
from indicators.regime_param_mapper import REGIME_PROFILES

REGIME_PROFILES[MarketRegime.HIGH_VOL_UP] = RegimeProfile(
    atr_multiplier_min=2.5, atr_multiplier_max=3.5,  # 더 보수적
    confirmation_bars_min=1, confirmation_bars_max=2,
    min_wave_atr_ratio_min=1.0, min_wave_atr_ratio_max=1.5,
    pivot_threshold_min_pct_min=0.25, pivot_threshold_min_pct_max=0.25,
    description="커스텀: 고변동 상승 — 더 보수적",
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

### 3. 향상된 추세 분석 비교

```python
# 기본 ADX만 사용
classifier_basic = MarketRegimeClassifier(enable_enhanced_trend=False)

# 향상된 추세 분석 사용
classifier_enhanced = MarketRegimeClassifier(enable_enhanced_trend=True)

# 동일 데이터로 백테스트 후 비교
```

### 4. 레짐별 수익률 분석

```python
# 레짐별로 수익률 분리
regime_returns = {}
for regime in MarketRegime:
    regime_returns[regime] = calculate_returns_for_regime(regime)

# 레짐별 최적 파라미터 찾기
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

### KP200 선물 특성 (파라미터 튜닝 시 고려)
- ATR 수준: KP200 선물은 약 1.5pt, KOSPI 현금은 약 8pt
- 틱 단위: 0.05pt (틱 노이즈와 실제 반전 구분 필요)
- 파라미터 설계: KOSPI 현금 기준 파라미터를 KP200에 적용 시 피봇 급증 가능성
  - 예: min_wave_atr_ratio=0.4~0.7 × ATR=1.5pt = 0.6~1.0pt (틱 노이즈 수준)
  - KP200 기준으로는 min_wave_atr_ratio를 1.5~2.5 수준으로 상향 필요

## 참고 문서

- [MarketRegimeClassifier 가이드](./architecture/market_regime_classifier.md) - 시장 레짐 분류기 상세 문서
- [regime_param_mapper.py](../indicators/regime_param_mapper.py) - 레짐 분류 및 레이블 제공 (읽기 전용)
- [adaptive_zigzag.py](../indicators/adaptive_zigzag.py) - ZigZag 파라미터 결정 (Layer A × Layer B)
- [adaptive_parameter_adjuster.py](../indicators/adaptive_parameter_adjuster.py) - ATR 백분위 기반 변동성 조정
- [option_sentiment.py](../indicators/option_sentiment.py) - 옵션 센티먼트 분석기

## 구조적 이점 (3-Layer 역할 분리)

### 기존 구조의 문제점
- **순환 의존**: 레짐이 ZigZag 파라미터를 조정 → ZigZag 결과가 다시 레짐에 영향 → 자기 강화 피드백
- **누적 지연**: 레짐 분류 (10봉) + 히스테리시스 (3봉) → 총 13분 지연
- **이중 증폭**: ER과 레짐이 동시에 파라미터에 영향 → 과도한 민감도

### 새로운 구조의 이점
- **순환 의존 제거**: 레짐이 파라미터를 직접 제어하지 않음
- **지연 최소화**: 세션 시간대 테이블 (무지연) + ATR 백분위 (1봉 지연) → 0~1봉 지연
- **역할 분리**: 레짐은 "설명 도구", 파라미터는 "반응 도구"로 분리

## 변경 이력

- **2026-05-10**: 3-Layer 역할 분리 구조 리팩터링
  - 레짐 파라미터 직접 제어 제거 → 읽기 전용 레이블로 변경
  - Layer A (세션 시간대 테이블) + Layer B (ATR 백분위)로 파라미터 결정
  - 순환 의존 제거, 지연 최소화 (13분 → 0~1봉)
  - NORMAL_VOL_* 레짐 추가 (정상 변동성 분류 개선)
  - 임계값 조정 (KP200 선물 현실에 맞게)
- **2026-05-10**: 레짐 기반 매매 신호 시스템 문서 작성
  - 시스템 아키텍처 추가
  - 레짐별 매매 전략 상세화
  - 파라미터 튜닝 가이드 추가
  - 백테스트 가이드 추가
- **2026-05-10**: ZigZag 파라미터 적정성 검토 후 수정
  - PARAM_RANGES 상한을 KP200 config 기준으로 상향 (pivot_threshold_min_pct: 0.3%→0.6%, min_wave_atr_ratio: 3.5→5.0)
  - REGIME_PROFILES 파라미터를 KP200 선물 기준으로 상향 조정 (6개 레짐 피봇 급증 방지)
  - confirmation_bars 조정 (HIGH_VOL_NO_DIR, LOW_VOL_NO_DIR: 3~4→2~3)
  - LOW_VOL_NO_DIRECTION 전략 설명을 KP200 선물에 맞게 수정
  - KP200 선물 특성 주의사항 추가
