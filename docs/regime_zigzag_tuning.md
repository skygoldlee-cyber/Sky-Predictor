# Regime 기반 ZigZag 파라미터 튜닝 시스템

> 기존 파일(`market_regime_classifier.py`, `adaptive_zigzag.py`, `adaptive_parameter_adjuster.py`)과
> 신규 파일(`regime_param_mapper.py`, `adaptive_zigzag_regime_integration.py`)의
> 연관관계 및 알고리즘 전체 설계 문서

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [파일 구조 및 역할 분담](#2-파일-구조-및-역할-분담)
3. [기존 파일 분석](#3-기존-파일-분석)
4. [신규 파일 설계](#4-신규-파일-설계)
5. [파일 간 연관관계 (의존성 그래프)](#5-파일-간-연관관계-의존성-그래프)
6. [파라미터 결정 알고리즘 상세](#6-파라미터-결정-알고리즘-상세)
7. [레짐별 파라미터 프로파일 설계 근거](#7-레짐별-파라미터-프로파일-설계-근거)
8. [히스테리시스 및 안정화 메커니즘](#8-히스테리시스-및-안정화-메커니즘)
9. [피드백 루프 설계](#9-피드백-루프-설계)
10. [통합 방법 (기존 코드 변경 최소화)](#10-통합-방법-기존-코드-변경-최소화)
11. [데이터 흐름 전체 시퀀스](#11-데이터-흐름-전체-시퀀스)
12. [파라미터 튜닝 의사결정 트리](#12-파라미터-튜닝-의사결정-트리)
13. [주요 설계 트레이드오프](#13-주요-설계-트레이드오프)
14. [향후 확장 포인트](#14-향후-확장-포인트)

---

## 1. 시스템 개요

### 문제 정의

기존 `AdaptiveParameterAdjuster`는 변동성·추세·거래량·시간대를 개별적으로 고려해
ZigZag 파라미터를 ±30% 조정한다. 그러나 **시장 레짐(Market Regime)을 모른 채**
미세조정하므로 다음 문제가 발생한다.

| 상황 | 기존 동작 | 실제 필요 |
|------|-----------|-----------|
| 고변동 흔들기(whipsaw) | 변동성↑ → atr_mult 소폭 증가 | atr_mult 3~4.5, conf_bars 3~4로 대폭 억제 |
| 장초반 갭 이후 | 시간대 조정(09:00~10:00)만 적용 | 이벤트 레짐으로 분류 → 임계값 최대화 |
| 저변동 횡보 | 변동성↓ → atr_mult 소폭 감소 | min_wave 0.4~0.7로 낮춰 작은 반전 포착 |

### 해결 방향

```
레짐 인식(Regime Detection)
         ↓
레짐별 최적 파라미터 범위 선택(Profile Selection)
         ↓
실시간 미세조정(Real-time Fine-tuning)
         ↓
ZigZag 적용
```

**핵심 원칙:** 레짐 프로파일이 `base_params`를 재정의하고,
기존 `AdaptiveParameterAdjuster`가 그 위에서 ±30% 미세조정한다.

---

## 2. 파일 구조 및 역할 분담

```
kospi_indicators/
├── market_regime_classifier.py       ← [기존] 레짐 분류기
├── adaptive_zigzag.py                ← [기존] ZigZag 지표 (수정 없음)
├── adaptive_parameter_adjuster.py    ← [기존] 실시간 미세조정기 (수정 없음)
│
├── regime_param_mapper.py            ← [신규] 레짐→파라미터 변환 핵심 모듈
└── adaptive_zigzag_regime_integration.py  ← [신규] 통합 패치 및 헬퍼
```

### 역할 요약

| 파일 | 계층 | 책임 |
|------|------|------|
| `market_regime_classifier.py` | 인식 계층 | ATR/ADX/시간 → `MarketState` 출력 |
| `adaptive_parameter_adjuster.py` | 조정 계층 | 실시간 신호 → ±30% 미세조정 |
| `adaptive_zigzag.py` | 실행 계층 | 피봇 탐색·확정·S/R·피보나치 |
| **`regime_param_mapper.py`** | **매핑 계층** | **레짐 → 프로파일 → base_params 재정의** |
| **`adaptive_zigzag_regime_integration.py`** | **통합 계층** | **팩토리·패치·스트림 헬퍼** |

---

## 3. 기존 파일 분석

### 3.1 `market_regime_classifier.py`

#### 주요 클래스

```python
class MarketRegimeClassifier:
    def classify(df, current_idx=-1) -> Optional[MarketState]
```

#### 분류 흐름

```
df (OHLCV)
  → ATR 계산 (Wilder RMA)  → atr_ratio = ATR / price
  → ADX 계산 (+DI, -DI)    → trend direction
  → 표준편차 계산           → std_ratio
  → 장초반 여부             → is_opening_session
          ↓
  _classify_volatility() → VolatilityState (HIGH/NORMAL/LOW)
  _classify_trend()      → TrendDirection (UP/DOWN/NEUTRAL)
          ↓
  _determine_regime()    → MarketRegime (8종)
  _calculate_confidence() → float [0, 1]
          ↓
  MarketState (dataclass)
```

#### 신뢰도 계산 로직

```python
confidence = 0.5  # 기본값
if vol_state == HIGH and atr_ratio > threshold * 1.5:
    confidence += 0.2
if trend_dir != NEUTRAL:
    if adx > threshold * 1.5: confidence += 0.3
    elif adx > threshold:     confidence += 0.15
```

#### RegimeParamMapper와의 연결점

`MarketState.confidence`와 `MarketState.regime`이 `RegimeParamMapper._classify()`에서
직접 소비된다. `RegimeParamMapper`는 `MarketRegimeClassifier`를 **내부에 보유**하며
`classify_interval_bars`마다 호출한다.

---

### 3.2 `adaptive_parameter_adjuster.py`

#### 주요 클래스

```python
class AdaptiveParameterAdjuster:
    def get_adaptive_params(df, recent_lags, success_rates, current_time, ...) -> dict
```

#### 조정 전략 (Strategy 패턴)

| 전략 | 조건 | 조정 방향 |
|------|------|-----------|
| `VolatilityStrategy` | current_ATR / ATR_MA > 1.5 | atr_mult↑, thr↓, conf_bars-1 |
| `TrendStrategy` | trend_strength > 0.02 | conf_bars-1, min_wave↓ |
| `VolumeStrategy` | vol_current / vol_avg > 2.0 | min_wave↑, thr↓ |
| `TimeStrategy` | 09:00~10:00 / 12:00~13:00 / 15:00~15:30 | 시간대별 |
| `OneWayStrategy` | ROC > 0.5% or disparity > 3% | thr↓↓, conf_bars=1 |

#### 가중치 계산

```python
weights = {
    'volatility': vol_weight,         # futures: 0.50, kospi: 0.35
    'trend':      remaining / 2,
    'volume':     remaining / 3,
    'time':       remaining / 6,
}
# 원웨이 감지 시: oneway 파라미터 50% 추가 결합
```

#### RegimeParamMapper와의 연결점

`RegimeParamMapper`는 프로파일에서 계산한 파라미터로 `self._adjuster.base_params`를
**동적으로 재정의**한 뒤 `get_adaptive_params()`를 호출한다.
조정기 내부의 `_get_base_params()`가 재정의된 값을 사용하므로
**조정기 코드 수정 없이** 레짐별 기준점이 달라진다.

---

### 3.3 `adaptive_zigzag.py`

#### 파라미터 조정 적용 위치 (`update()` 내부)

```python
# adaptive_zigzag.py line ~480
if self._adaptive_enabled and self._param_adjuster is not None and n >= 50:
    _do_adjust = (self._last_adjustment_bar_idx < 0
                  or (self._bar_idx - self._last_adjustment_bar_idx) >= self._min_adjustment_interval)
    if _do_adjust:
        adjusted_params = self._param_adjuster.get_adaptive_params(
            recent_df, current_time=bar_time
        )
        if adjusted_params:
            self.config.atr_multiplier      = adjusted_params.get('atr_multiplier', ...)
            self.config.confirmation_bars   = adjusted_params.get('confirmation_bars', ...)
            self.config.pivot_threshold_min_pct = adjusted_params.get('pivot_threshold_min_pct', ...)
            ...
```

`self._param_adjuster`가 `RegimeParamMapper`로 교체되면 위 코드가
**그대로** 레짐 기반 파라미터를 적용한다. 인터페이스(`.get_adaptive_params()`)가
동일하므로 `adaptive_zigzag.py`를 한 줄도 수정하지 않는다.

---

## 4. 신규 파일 설계

### 4.1 `regime_param_mapper.py`

#### 핵심 데이터 구조

```python
@dataclass
class RegimeProfile:
    atr_multiplier_min: float       # confidence=0 → 보수적 끝
    atr_multiplier_max: float       # confidence=1 → 공격적 끝
    confirmation_bars_min: int
    confirmation_bars_max: int
    min_wave_atr_ratio_min: float
    min_wave_atr_ratio_max: float
    pivot_threshold_min_pct_min: float
    pivot_threshold_min_pct_max: float

    def interpolate(confidence: float) -> Dict[str, float]:
        # t = clip(confidence, 0, 1)
        # param = min + t * (max - min)
```

#### 레짐 프로파일 테이블

```python
REGIME_PROFILES: Dict[MarketRegime, RegimeProfile] = {
    MarketRegime.HIGH_VOL_UP:            RegimeProfile(2.0, 3.0, 1, 1, 0.8, 1.2, 0.20, 0.20),
    MarketRegime.HIGH_VOL_DOWN:          RegimeProfile(2.0, 3.0, 1, 1, 0.8, 1.2, 0.20, 0.20),
    MarketRegime.HIGH_VOL_NO_DIRECTION:  RegimeProfile(3.0, 4.5, 3, 4, 1.5, 2.5, 0.35, 0.35),
    MarketRegime.LOW_VOL_NO_DIRECTION:   RegimeProfile(0.8, 1.2, 3, 4, 0.4, 0.7, 0.10, 0.10),
    MarketRegime.LOW_VOL_UP:             RegimeProfile(1.0, 1.8, 2, 3, 0.5, 0.8, 0.12, 0.12),
    MarketRegime.LOW_VOL_DOWN:           RegimeProfile(1.0, 1.8, 2, 3, 0.5, 0.8, 0.12, 0.12),
    MarketRegime.OPENING_EVENT:          RegimeProfile(4.0, 8.0, 1, 2, 1.5, 3.0, 0.40, 0.40),
    MarketRegime.NEWS_EVENT:             RegimeProfile(5.0, 8.0, 1, 1, 2.0, 4.0, 0.50, 0.50),
}
```

#### `RegimeParamMapper.get_adaptive_params()` 내부 흐름

```python
def get_adaptive_params(df, current_time, recent_lags, success_rates) -> dict:

    # Step 1: 분류 (10봉 주기 제한)
    if self._should_classify():
        self._classify(df)   # MarketRegimeClassifier.classify() 호출

    # Step 2: 프로파일 보간
    profile_params = self._get_profile_params()
    #   → REGIME_PROFILES[stable_regime].interpolate(confidence)

    # Step 3: 조정기 base_params 재정의
    self._adjuster.base_params = AdaptiveParams(**profile_params)

    # Step 4: 기존 조정기 실행 (변동성·추세·거래량·시간대 미세조정)
    adjuster_params = self._adjuster.get_adaptive_params(
        df, recent_lags=recent_lags, success_rates=success_rates,
        current_time=current_time,
    )

    # Step 5: 가중 합산 (profile 65% : adjuster 35%)
    merged = self._merge(profile_params, adjuster_params)

    # Step 6: 클램핑 (AdaptiveParameterAdjuster.PARAM_RANGES 기준)
    return self._clamp(merged)
```

---

### 4.2 `adaptive_zigzag_regime_integration.py`

#### `build_regime_aware_zigzag()` 팩토리

```python
zz, mapper = build_regime_aware_zigzag(symbol="futures")
# 내부 동작:
# 1. AdaptiveZigZagConfig 심볼 기본값 설정
# 2. AdaptiveZigZag 생성
# 3. MarketRegimeClassifier 생성
# 4. patch_zigzag_with_regime() → zz._param_adjuster = mapper
```

#### `RegimeAwareZigZagRunner` 스트림 헬퍼

```python
runner = RegimeAwareZigZagRunner(symbol="futures")
runner.seed(open_price=360.0)      # 장 시작 앵커

state = runner.on_bar(high, low, close, bar_time=ts)
# 내부:
# 1. OHLCV 버퍼 갱신 (200봉 순환)
# 2. zz.update() 호출 → mapper.get_adaptive_params() 자동 호출
# 3. 피봇 품질 피드백 수집 (lag_history, success_history)
# 4. ZigZagState 반환
```

---

## 5. 파일 간 연관관계 (의존성 그래프)

```
                        ┌─────────────────────────────────┐
                        │         실시간 봉 데이터          │
                        │    (high, low, close, volume)    │
                        └────────────┬────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────┐
                    │       RegimeAwareZigZagRunner          │
                    │  (adaptive_zigzag_regime_integration)  │
                    └──────┬──────────────────┬─────────────┘
                           │                  │
              ┌────────────▼──────┐   ┌───────▼──────────────┐
              │  AdaptiveZigZag   │   │   RegimeParamMapper   │
              │ (adaptive_zigzag) │◄──│  (regime_param_mapper)│
              └────────────┬──────┘   └──────┬───────┬────────┘
                           │                 │       │
                  ZigZagState 출력    ┌───────▼──┐  ┌▼─────────────────────┐
                                     │Classifier│  │AdaptiveParameterAdj. │
                                     │(MRC.py)  │  │(adaptive_param_adj.) │
                                     └──────────┘  └──────────────────────┘
```

### 호출 방향 상세

```
on_bar(h, l, c, t)
  └─► AdaptiveZigZag.update(h, l, c, t)
        └─► [10봉마다] self._param_adjuster.get_adaptive_params(recent_df, t)
              = RegimeParamMapper.get_adaptive_params(...)
                  ├─► [10봉마다] MarketRegimeClassifier.classify(df)
                  │     └─► MarketState { regime, confidence, atr, adx }
                  ├─► REGIME_PROFILES[stable_regime].interpolate(confidence)
                  │     └─► profile_params { atr_mult, conf_bars, ... }
                  ├─► self._adjuster.base_params = AdaptiveParams(**profile_params)
                  └─► AdaptiveParameterAdjuster.get_adaptive_params(df, ...)
                        ├─► VolatilityStrategy.adjust()
                        ├─► TrendStrategy.adjust()
                        ├─► VolumeStrategy.adjust()
                        ├─► TimeStrategy.adjust()
                        └─► OneWayStrategy.adjust()
              └─► _merge(profile 65%, adjuster 35%) → _clamp() → dict
        └─► config.atr_multiplier = merged['atr_multiplier']
        └─► config.confirmation_bars = merged['confirmation_bars']
        └─► ... (이하 기존 ZigZag 피봇 탐색)
```

---

## 6. 파라미터 결정 알고리즘 상세

### Step 1: 레짐 분류 (MarketRegimeClassifier)

**입력:** 최근 N봉 OHLCV (N ≥ max(atr_period, adx_period, std_period) + 1 = 21)

**ATR 계산 (Wilder RMA):**
```
TR(t) = max(H-L, |H-C(t-1)|, |L-C(t-1)|)
ATR(t) = (1/n) * TR(t) + (1 - 1/n) * ATR(t-1)
atr_ratio = ATR / Close
```

**ADX 계산:**
```
+DM = max(H - H_prev, 0) if H-H_prev > L_prev-L
-DM = max(L_prev - L, 0) if L_prev-L > H-H_prev
+DI = 100 * RMA(+DM) / ATR
-DI = 100 * RMA(-DM) / ATR
DX  = 100 * |+DI - -DI| / (+DI + -DI)
ADX = RMA(DX)
```

**변동성 분류:**
```python
combined_vol = (atr_ratio + std_ratio) / 2
if combined_vol >= 0.02:  → HIGH
elif combined_vol <= 0.005: → LOW
else:                      → NORMAL
```

**추세 방향 분류:**
```python
if adx < 15:       → NEUTRAL
elif +DI > -DI:    → UP
else:               → DOWN
```

**레짐 결정 우선순위:**
```
1. is_opening_session AND atr_ratio > vol_high_threshold → OPENING_EVENT
2. atr_ratio > vol_high_threshold * 2                    → NEWS_EVENT
3. (vol_state, trend_dir) 조합 매핑                      → 6종 레짐
```

---

### Step 2: 히스테리시스 필터 (RegimeParamMapper)

```
classify() 결과 → MarketState(regime, confidence)
                              ↓
         confidence ≥ 0.55?
          ├─ NO  → stable_regime 유지 (이전 레짐)
          └─ YES → 다수결 검사
                    최근 5봉 중 현재 regime이 ≥ 3번?
                     ├─ NO  → stable_regime 유지
                     └─ YES → stable_regime = current_regime (전환!)
```

**히스테리시스 효과:**
- 단발성 오분류(1~2봉)는 무시
- 레짐 전환 시 파라미터가 점진적으로 이동 (confidence 보간)

---

### Step 3: 프로파일 보간

```python
profile = REGIME_PROFILES[stable_regime]
t = clip(confidence, 0.0, 1.0)

atr_multiplier       = profile.atr_mult_min + t * (atr_mult_max - atr_mult_min)
confirmation_bars    = round(profile.conf_min + t * (conf_max - conf_min))
min_wave_atr_ratio   = profile.wave_min + t * (wave_max - wave_min)
pivot_threshold_min  = profile.thr_min + t * (thr_max - thr_min)
```

**confidence=0.55 (최소 통과):** 파라미터 = min + 0.55 * (max - min) ← 보수적
**confidence=1.00 (완전 확신):** 파라미터 = max ← 공격적

---

### Step 4: 기존 조정기 미세조정

`AdaptiveParameterAdjuster`가 프로파일 기준값으로 ±30% 추가 조정한다.

**최종 조정 범위 예시 (HIGH_VOL_UP, confidence=1.0):**

| 파라미터 | 프로파일 기준 | 조정 범위 | 실제 가능 범위 |
|----------|--------------|-----------|---------------|
| atr_multiplier | 3.0 | ±30% | 2.1 ~ 3.9 |
| confirmation_bars | 1 | ±1 | 1 ~ 2 |
| min_wave_atr_ratio | 1.2 | ±30% | 0.84 ~ 1.56 |
| pivot_threshold_min | 0.20% | ±30% | 0.14% ~ 0.26% |

---

### Step 5: 가중 합산

```python
PROFILE_WEIGHT  = 0.65   # 레짐 프로파일 (거시 구조)
ADJUSTER_WEIGHT = 0.35   # 실시간 조정기 (미시 신호)

merged[k] = 0.65 * profile[k] + 0.35 * adjuster[k]
```

**설계 의도:**
- 레짐이 정확히 분류됐을 때 레짐 특성이 우세
- 실시간 신호(거래량 급증, 원웨이 추세)가 나머지 35%를 점유
- 합이 100%이므로 클램핑 전 예측 가능한 범위 내 유지

---

## 7. 레짐별 파라미터 프로파일 설계 근거

### HIGH_VOL_UP / HIGH_VOL_DOWN (돌파추종)

```
atr_mult: 2.0~3.0   ← 노이즈 차단하면서도 진입 신호 포착
conf_bars: 1         ← 지연 없이 즉시 확정 (모멘텀 손실 방지)
min_wave: 0.8~1.2   ← 중간 크기 파동만 인식
thr_min: 0.20%      ← 낮은 임계값으로 빠른 반응
```

추세 레짐에서 `conf_bars=1`이 중요하다. `conf_bars=2`이면 돌파 후
1봉 더 대기하므로 진입 가격이 불리해진다.

---

### HIGH_VOL_NO_DIRECTION (흔들기)

```
atr_mult: 3.0~4.5   ← 작은 반전을 피봇으로 오인 방지
conf_bars: 3~4       ← 허위 확정 억제 (3봉 연속 방향 유지 필요)
min_wave: 1.5~2.5   ← 대형 파동만 인식
thr_min: 0.35%      ← 높은 임계값
```

이 레짐은 고변동이지만 방향성 없음 → ZigZag가 계속 반전해
"피봇 쏟아짐" 현상 발생. `conf_bars=3`과 높은 `min_wave`로
이를 차단한다.

---

### LOW_VOL_NO_DIRECTION (Mean Reversion)

```
atr_mult: 0.8~1.2   ← 낮은 임계값으로 민감도 높임
conf_bars: 3~4       ← 허위 전환 억제
min_wave: 0.4~0.7   ← 작은 파동도 인식 (좁은 레인지 매매)
thr_min: 0.10%      ← 최저 임계값
```

좁은 레인지 내 작은 고점/저점을 피봇으로 잡아야 하므로
`atr_mult`와 `min_wave`를 낮춘다. `conf_bars`는 유지해
허위 반전 방지.

---

### OPENING_EVENT (장초반)

```
atr_mult: 4.0~8.0   ← 극대화 (기존 early_session_atr_multiplier_max와 동일 논리)
conf_bars: 1~2       ← 빠른 확정 (방향 잡힌 후 신속 추종)
min_wave: 1.5~3.0   ← 큰 파동만 인식
thr_min: 0.40%      ← 장초반 갭/급변동 필터
```

`adaptive_zigzag.py`의 `early_session_atr_multiplier_max=8.0` 설정과
동일한 논리를 레짐 레벨로 승격한 것.

---

### NEWS_EVENT (뉴스 급등락)

```
atr_mult: 5.0~8.0   ← 최대 보수적
conf_bars: 1         ← 방향 확정 시 신속 진입 (관망 후 편입)
min_wave: 2.0~4.0   ← 뉴스 방향성 파동만 인식
thr_min: 0.50%      ← 가장 높은 임계값
```

뉴스 레짐에서 ZigZag를 사용하는 목적: 뉴스 방향성이 확립된 후
첫 되돌림 저점/고점을 잡기 위함. 소음 피봇 완전 차단이 우선.

---

## 8. 히스테리시스 및 안정화 메커니즘

### 레짐 전환 조건 (이중 게이트)

```
Gate 1: confidence ≥ 0.55
Gate 2: 최근 5봉 중 같은 레짐 ≥ 3번 (60% 다수결)

→ 두 조건 모두 충족해야 stable_regime 전환
```

**시나리오 분석:**

| 상황 | Gate 1 | Gate 2 | 결과 |
|------|--------|--------|------|
| 단발 노이즈 오분류 | ✓ | ✗ (1/5) | 유지 |
| 2봉 오분류 | ✓ | ✗ (2/5) | 유지 |
| 3봉 이상 지속 | ✓ | ✓ (3/5) | **전환** |
| 낮은 신뢰도 | ✗ | N/A | 유지 |

### 파라미터 점진 이동

레짐이 전환되더라도 `confidence`가 보간값으로 작용해 급격한 점프를 완화한다.

```
레짐 A (confidence 0.9) → 레짐 B (confidence 0.6)
  파라미터 변화 폭 = (B_max - B_min) * 0.6 + B_min
                   < (A_max - A_min) * 0.9 + A_min  ← 완전 전환값보다 작음
```

첫 전환 봉에서 confidence가 낮을수록 파라미터 이동폭이 작아
시장 참여자가 느끼는 ZigZag 변화가 자연스럽다.

---

## 9. 피드백 루프 설계

### 피봇 품질 지표

`RegimeAwareZigZagRunner`가 수집하는 두 가지 피드백:

**1. lag_history (확정 지연 봉수)**
```python
lag = state.last_swing_high_lag_bars  # 피봇봉 → 확정봉 경과 봉수
# lag 크면 → conf_bars 낮춰 빠른 확정 필요
```

**2. success_history (레짐 방향 일치 여부)**
```python
# HIGH_VOL_UP 레짐에서 new_high 피봇 → 성공 (추세 방향 일치)
# HIGH_VOL_UP 레짐에서 new_low  피봇 → 실패 (역방향 피봇 = 허위 신호 가능성)
```

### 피드백 적용

```python
AdaptiveParameterAdjuster.adjust_for_lag_feedback(
    recent_lags=lag_history,      # 지연 보정
    success_rates=success_history # 정확도 보정
)
# feedback_score = lag_score * 0.4 + accuracy_score * 0.6
# score > 0.6 → 공격적 조정 (conf_bars-1, thr↓)
# score < 0.3 → 보수적 조정 (conf_bars+1, min_wave↑)
```

피드백은 `get_adaptive_params()` 호출 시 `recent_lags`/`success_rates` 인자로
`AdaptiveParameterAdjuster`에 전달되며, 최종 합산에서 30% 비중을 차지한다.

---

## 10. 통합 방법 (기존 코드 변경 최소화)

### 방법 A: 팩토리 함수 (권장, 신규 인스턴스)

```python
from adaptive_zigzag_regime_integration import build_regime_aware_zigzag

zz, mapper = build_regime_aware_zigzag(
    symbol="futures",              # 또는 "kospi"
    classify_interval_bars=10,     # 레짐 재분류 주기 (봉)
)
zz.set_adaptive_enabled(True)      # Adaptive 체크박스 ON (기본값)

# 이후 기존과 동일하게 사용
state = zz.update(high, low, close, bar_time=ts)
print(mapper.stable_regime)        # 현재 레짐 조회
```

### 방법 B: 기존 인스턴스에 패치 (최소 변경)

```python
from regime_param_mapper import patch_zigzag_with_regime

# 기존 코드에서 생성된 zz 인스턴스에 주입
mapper = patch_zigzag_with_regime(
    zigzag=self._zigzag,
    symbol="futures",
)
# zz._param_adjuster가 mapper로 교체됨 (update() 코드 무변경)
```

### 방법 C: 스트림 헬퍼 (실시간 운용 최적)

```python
from adaptive_zigzag_regime_integration import RegimeAwareZigZagRunner

runner = RegimeAwareZigZagRunner(
    symbol="futures",
    classify_interval_bars=10,
    ohlcv_buffer_size=200,
)
runner.seed(open_price=360.0, swing_type="low")  # 장 시작

for tick in live_feed:
    state = runner.on_bar(
        tick.high, tick.low, tick.close,
        bar_time=tick.time,
        open_=tick.open,
        volume=tick.volume,
    )
    if state.new_swing_signal != "none":
        handle_pivot(state, runner.current_regime)
```

### SkyPredictor 기존 코드 연동 포인트

기존 `SkyPredictor` 코드에서 `AdaptiveZigZag`를 생성하는 위치:

```python
# [기존] training/adaptive_indicator_manager.py 또는 유사 위치
self._zigzag = AdaptiveZigZag(config=cfg)

# [변경] 한 줄 추가로 레짐 기반 파라미터 활성화
from regime_param_mapper import patch_zigzag_with_regime
self._regime_mapper = patch_zigzag_with_regime(self._zigzag, symbol="futures")
```

---

## 11. 데이터 흐름 전체 시퀀스

```
봉 N 도착
│
├── RegimeAwareZigZagRunner.on_bar(h, l, c, t)
│     │
│     ├── OHLCV 버퍼 push (최근 200봉 유지)
│     │
│     └── AdaptiveZigZag.update(h, l, c, t)
│           │
│           ├── [매봉] ATR 계산, 임계값 계산, 피봇 탐색
│           │
│           └── [50봉 이상, 50봉 간격] _param_adjuster.get_adaptive_params()
│                 = RegimeParamMapper.get_adaptive_params(recent_50봉, t)
│                       │
│                       ├── [10봉 간격] MarketRegimeClassifier.classify(df)
│                       │     → MarketState { regime, confidence, atr, adx }
│                       │
│                       ├── 히스테리시스 필터
│                       │     → stable_regime 결정
│                       │
│                       ├── REGIME_PROFILES[stable_regime].interpolate(confidence)
│                       │     → profile_params { atr_mult, conf_bars, ... }
│                       │
│                       ├── adjuster.base_params = AdaptiveParams(**profile_params)
│                       │
│                       ├── adjuster.get_adaptive_params(df, lags, rates, t)
│                       │     ├── VolatilityStrategy → vol_params
│                       │     ├── TrendStrategy     → trend_params
│                       │     ├── VolumeStrategy    → vol_params
│                       │     ├── TimeStrategy      → time_params
│                       │     └── OneWayStrategy    → oneway_params
│                       │     → _weighted_average() → adjuster_params
│                       │
│                       └── _merge(profile 65%, adjuster 35%)
│                             → _clamp(PARAM_RANGES)
│                             → { atr_multiplier, confirmation_bars, ... }
│
│           config.atr_multiplier      = merged_params['atr_multiplier']
│           config.confirmation_bars   = merged_params['confirmation_bars']
│           config.min_wave_atr_ratio  = merged_params['min_wave_atr_ratio']
│           config.pivot_threshold_min_pct = merged_params['pivot_threshold_min_pct']
│
│     피봇 탐색 / 확정 / S/R / 피보나치 계산
│     └── ZigZagState 반환
│
└── RegimeAwareZigZagRunner._collect_feedback(state)
      ├── lag_history.append(state.last_swing_*_lag_bars)
      └── success_history.append(레짐 방향 일치 여부)
          (다음 호출 시 adjuster에 피드백으로 전달)
```

---

## 12. 파라미터 튜닝 의사결정 트리

```
현재 레짐은?
│
├── HIGH_VOL_UP / HIGH_VOL_DOWN
│     "추세 방향이 명확하고 변동성이 큼"
│     → atr_mult 2.0~3.0 | conf_bars 1 | min_wave 0.8~1.2
│     목표: 돌파 피봇을 지연 없이 빠르게 확정
│
├── HIGH_VOL_NO_DIRECTION
│     "변동성은 크지만 방향이 없음 (흔들기)"
│     → atr_mult 3.0~4.5 | conf_bars 3~4 | min_wave 1.5~2.5
│     목표: 허위 피봇 최대 억제, 진짜 반전만 인식
│
├── LOW_VOL_NO_DIRECTION
│     "변동성도 낮고 방향도 없음 (좁은 횡보)"
│     → atr_mult 0.8~1.2 | conf_bars 3~4 | min_wave 0.4~0.7
│     목표: 좁은 레인지 내 고/저점 감지 (Mean Reversion)
│
├── LOW_VOL_UP / LOW_VOL_DOWN
│     "느린 추세"
│     → atr_mult 1.0~1.8 | conf_bars 2~3 | min_wave 0.5~0.8
│     목표: 스윙 고/저점 추종, 중간 민감도
│
├── OPENING_EVENT
│     "장초반 갭/급변동"
│     → atr_mult 4.0~8.0 | conf_bars 1~2 | min_wave 1.5~3.0
│     목표: 장초반 노이즈 차단, 주방향 확립 후 진입
│
└── NEWS_EVENT
      "뉴스 급등락"
      → atr_mult 5.0~8.0 | conf_bars 1 | min_wave 2.0~4.0
      목표: 극단적 보수 → 뉴스 방향성 대파동만 인식
```

---

## 13. 주요 설계 트레이드오프

### 분류 주기 (classify_interval_bars)

| 값 | 장점 | 단점 |
|----|------|------|
| 1봉 | 즉각 반응 | CPU 부하 증가, 레짐 깜빡임 위험 |
| 10봉 (기본) | 균형 | 레짐 전환에 최대 10봉 지연 |
| 30봉 | 안정적 | 급변 시 대응 늦음 |

**권장:** `classify_interval_bars=10`. 레짐 전환 평균 지연 = 10봉 × 다수결 3봉 = 30봉 최대. 1분봉 기준 30분 이내 전환.

---

### 가중치 (65% : 35%)

| 상황 | 65% 프로파일 | 35% 조정기 |
|------|-------------|-----------|
| 레짐이 정확한 경우 | 최적 파라미터 제공 | 미세조정 |
| 레짐이 경계선인 경우 | 중간값 제공 | 실시간 신호로 보정 |
| 레짐 오분류 시 | 잘못된 기준 → 위험 | 35%만 올바름 |

**오분류 위험 완화:** 히스테리시스 이중 게이트 + confidence 보간으로
오분류 시에도 급격한 파라미터 변화 방지.

---

### conf_bars 정수 문제

`confirmation_bars`는 반드시 정수다. `interpolate()`에서 `round()`로 처리하지만
confidence 0.49 (→ 정수 내림)와 0.50 (→ 정수 올림) 경계에서 conf_bars가
1봉 점프할 수 있다. 실운용 시 영향:

- HIGH_VOL_UP: conf_min=conf_max=1 → 항상 1 (문제 없음)
- LOW_VOL_NO_DIRECTION: conf_min=3, conf_max=4 → confidence 0.5 기준 3↔4 전환

이 전환이 피봇 확정 속도에 영향 주지만, 히스테리시스가 안정화한다.

---

## 14. 향후 확장 포인트

### 1. ML 기반 confidence 보정

현재 `MarketRegimeClassifier._calculate_confidence()`는 규칙 기반이다.
향후 과거 레짐 판정의 실제 정확도를 학습한 ML 모델로 교체 가능:

```python
class MLConfidenceEstimator:
    def predict(self, market_state) -> float:
        # XGBoost 또는 LSTM 기반
        return calibrated_confidence
```

`RegimeParamMapper._classify()`에서 `state.confidence`를 이 값으로 Override.

---

### 2. 레짐별 피드백 분리

현재 `success_history`가 레짐 무관하게 누적된다. 레짐별로 분리하면
특정 레짐에서의 파라미터 성능을 독립적으로 추적할 수 있다:

```python
self._success_by_regime: Dict[MarketRegime, Deque] = defaultdict(lambda: deque(maxlen=20))
```

---

### 3. 레짐 전환 이벤트 텔레그램 알림

```python
# RegimeParamMapper._log_if_regime_changed() 확장
if self._telegram_callback and self._stable_regime != self._last_logged_regime:
    msg = f"📊 레짐 전환: {prev} → {curr}\n파라미터: atr={atr:.2f} conf={conf}"
    self._telegram_callback(msg)
```

---

### 4. REGIME_PROFILES 외부 설정 파일화

현재 프로파일이 코드에 하드코딩되어 있다. 향후:

```json
// config.json
"regime_profiles": {
    "high_vol_up": {
        "atr_multiplier_min": 2.0,
        "atr_multiplier_max": 3.0,
        ...
    }
}
```

`RegimeParamMapper.__init__()`에서 `config["regime_profiles"]`를 읽어
`REGIME_PROFILES`를 동적 생성하도록 확장.

---

*문서 버전: 1.0 | 작성 기준 파일: market_regime_classifier.py, adaptive_zigzag.py, adaptive_parameter_adjuster.py, regime_param_mapper.py, adaptive_zigzag_regime_integration.py*
