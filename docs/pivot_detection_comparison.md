# SkyPredictor 변곡점 탐지 시스템: 기존 ZigZag vs 신규 4-Layer 설계 비교

> 작성 기준: SkyPredictor Step 1~3 구현 완료 후 (2026-05)  
> 대상 독자: SkyPredictor 개발자 (KOSPI200 선물·옵션 실시간 매매 시스템)

---

## 목차

1. [시스템 개요 비교](#1-시스템-개요-비교)
2. [기존 방식: AdaptiveZigZag](#2-기존-방식-adaptivezigzag)
3. [신규 Step 1: ATR Adaptive Pivot + Fractal](#3-신규-step-1-atr-adaptive-pivot--fractal)
4. [신규 Step 2: Market Structure Break + OI Gate](#4-신규-step-2-market-structure-break--oi-gate)
5. [신규 Step 3: Kalman Turning Point + PivotScoreIntegrator](#5-신규-step-3-kalman-turning-point--pivotscoreintegrator)
6. [알고리즘 핵심 수식 비교](#6-알고리즘-핵심-수식-비교)
7. [Transformer Feature 비교](#7-transformer-feature-비교)
8. [실전 동작 비교 (KOSPI200 1일 시뮬레이션)](#8-실전-동작-비교-kospi200-1일-시뮬레이션)
9. [파이프라인 데이터 흐름 비교](#9-파이프라인-데이터-흐름-비교)
10. [설정 파라미터 비교](#10-설정-파라미터-비교)
11. [모델 학습 영향 및 마이그레이션](#11-모델-학습-영향-및-마이그레이션)
12. [한계 및 주의사항](#12-한계-및-주의사항)

---

## 1. 시스템 개요 비교

### 기존 방식 (단일 지표)

```
Raw OHLC
  │
  ▼
AdaptiveZigZag ──────────────────────────────► azz_* feature 14개
  │  (ATR + ER 기반 threshold)                  → PriceTransformer
  │  (pending_confirm 창 2봉)
  ▼
ZigZagState.new_swing_signal
("new_high" / "new_low" / "none")
```

- **단일 지표**: 변곡점 탐지를 ZigZag 알고리즘 하나에 의존
- **고정 임계값 구조**: `pivot_threshold_min_pct ~ max_pct` 범위 내 ER 보간
- **확정 지연**: `confirmation_bars=2` — 평균 2~6봉 후 신호 확정
- **출력**: boolean 수준 신호 (`new_high`/`new_low`/`none`)

### 신규 방식 (4-Layer 통합)

```
Raw OHLC
  ├─► ATRAdaptivePivot ──► aap_* (3개) ──────────┐
  ├─► FractalConfirmation ──► (보조 확증)          │
  ├─► MarketStructureBreak ──► msb_* (8개) ───────┤
  ├─► OIStructureGate ──► oi_* (6개) ─────────────┤ PivotScoreIntegrator
  ├─► KalmanTurningPoint ──► kf_* (8개) ──────────┤   ▼
  │                                                └► ps_* (6개)
  ▼                                                   ▼
  PivotScoreIntegrator ────────────────────────────► ps_total_score [0,1]
    (4-Layer 가중합)                                  ps_signal: long/short/none
```

- **4개 독립 레이어**: 각기 다른 관점(변동성·구조·OI·추세)에서 변곡 강도 산출
- **연속 점수**: [0, 1] 범위의 `total_score`로 신호 강도를 표현
- **입력 피처 25개 추가**: 기존 38개 → 63개 (ADAPT_KEYS 기준)
- **방향 다수결**: 3개 레이어의 방향 투표로 long/short 결정

---

## 2. 기존 방식: AdaptiveZigZag

### 2.1 알고리즘 원리

ZigZag는 가격이 직전 극값에서 `threshold` 이상 되돌아왔을 때 방향 전환을 판정합니다.

```
방향 = 상승 탐색 중:
  pending_high 갱신(매 봉 고가 추적)
  if (pending_high - current_low) >= threshold:
    → 고점 후보 등록 (pending_confirm)
    → confirmation_bars 봉 유지 확인 후 확정
```

#### 임계값 계산 (AdaptiveZigZag 핵심)

```python
# ER(Efficiency Ratio) 기반 동적 배수
er = |close[-1] - close[-period]| / Σ|close[i] - close[i-1]|

# 배수: ER ↑(추세) → mult 크게(노이즈 차단), ER ↓(횡보) → mult 작게
mult = atr_multiplier_min + er × (atr_multiplier_max - atr_multiplier_min)
# 기본: 1.0 + er × (4.0 - 1.0)

# 임계값 (% 기준)
thr_pct = ATR × mult / close × 100
thr_pct = clip(thr_pct, pivot_threshold_min_pct=0.3, pivot_threshold_max_pct=3.0)
```

### 2.2 기존 방식의 구조적 문제

#### 문제 1: 확정 지연 (Lag)

```
실제 고점 발생 봉: T
후보 등록 봉:     T+N  (threshold 돌파 시)
확정 봉:          T+N+confirmation_bars  (기본 2봉)
총 지연:          N+2 봉 이상

예) KOSPI200 1분봉: 평균 5~8봉 지연 = 5~8분 후 신호
```

`confirmation_bars=2` 설정이지만, ranging/unknown 구간에서는 `_calc_confirmation_bars()`가 3~4봉으로 자동 상향합니다. 소파동 보정까지 더해지면 8봉 이상 지연도 빈번합니다.

#### 문제 2: Fake Pivot 취소 로직의 양면성

```python
# pending_confirm 취소 조건 (기존)
if 대기 중 반대 방향 threshold 돌파:
    → 기존 후보 취소 + 반대 방향 후보 등록
```

추세가 강할 때는 바람직하지만, 급등락 후 빠른 되돌림 시장에서는 매 봉마다 후보가 교체되어 신호가 발생하지 않는 현상이 생깁니다. 실제 로그에서 `[ZZ][취소] ... 이유=반대후보교체` 가 연속으로 나타나는 구간이 이에 해당합니다.

#### 문제 3: 단일 신호의 이진성

```python
state.new_swing_signal  # "new_high" | "new_low" | "none"
```

신호 강도 정보 없음. 강한 추세 반전과 잡음 수준의 작은 반전이 동일한 `"new_high"` 로 표현됩니다. PriceTransformer 가 강도를 학습할 수 없습니다.

#### 문제 4: Regime 무대응

ZigZag는 `AdaptiveZigZagConfig.er_period` 기반 ER 계산으로 어느 정도 적응하지만, 장중 레짐 전환(추세→횡보→추세)을 실시간으로 반영하는 별도 레이어가 없습니다. `structure_majority_threshold=0.7` 로 구조 판정은 하지만 임계값 자체를 바꾸지는 않습니다.

### 2.3 기존 ZigZag가 생성하는 Feature 목록

| 키 | 설명 | 범위 |
|----|------|------|
| `azz_direction` | 현재 탐색 방향 | {-1, 0, +1} |
| `azz_wave_size_pct` | 파동 크기 (정규화) | [0, 1] |
| `azz_support_dist_pct` | 지지선까지 거리 | [0, 1] |
| `azz_res_dist_pct` | 저항선까지 거리 | [0, 1] |
| `azz_bars_since_swing` | 마지막 피봇 이후 경과 봉 | [0, 1] |
| `azz_fib618_dist` | Fib 0.618 거리 | [-1, 1] |
| `azz_fib382_dist` | Fib 0.382 거리 | [-1, 1] |
| `azz_higher_highs` | HH 여부 | {0, 1} |
| `azz_lower_lows` | LL 여부 | {0, 1} |
| `azz_new_swing` | 신규 스윙 신호 | {-1, 0, +1} |
| `azz_swing_recency` | 스윙 신선도 (지수 감쇠) | [0, 1] |
| `azz_threshold_pct` | 현재 threshold % | [0, 1] |
| `azz_structure_up/down/ranging` | 구조 판정 | {0, 1} |
| `azz_micro_up/down/ranging` | 단기 구조 | {0, 1} |
| `azz_structure_conf` | 구조 신뢰도 | [0, 1] |
| `azz_pend_sr_dist` | 잠정 S/R 거리 | [-1, 1] |
| `azz_pending_type/dist/urgency/age/prob` | 후보 상태 피처 | 다양 |

---

## 3. 신규 Step 1: ATR Adaptive Pivot + Fractal

### 3.1 ATRAdaptivePivot — 기존과 무엇이 다른가

#### 핵심 차이: threshold 계산 방식

```python
# 기존 ZigZag
thr_pct = ATR × (min + er × (max - min)) / close × 100
# clip → 0.3% ~ 3.0% 범위 강제

# 신규 ATRAdaptivePivot
mult = base_multiplier × session_scale(bar_time)  # 시간대 배율 추가
thr_abs = ATR × mult  # 절대값 threshold (포인트 단위)
# clip 없음 — ATR 자체가 변동성 반영
```

기존은 `threshold_min/max_pct`로 **백분율 강제 클리핑**이 있어 저변동 구간에서도 최소 0.3%를 요구합니다. 신규는 ATR에 배수만 곱하므로 저변동 구간에서 threshold가 자연스럽게 작아지고, 고변동 구간에서 커집니다.

#### 세션 시간대 배율 (KOSPI200 특화)

```python
# config.json 예시
"session_multiplier_table": [
    ["09:00", "09:30", 1.5],   # 장초반: 배율 확대 → 노이즈 차단
    ["09:30", "10:30", 1.0],   # 오전 정규: 기본
    ["10:30", "13:00", 1.2],   # 점심 전: 약간 상향
    ["14:30", "15:20", 0.8],   # 마감 전: 배율 축소 → 민감도 증가
    ["15:20", "15:31", 1.3],   # 동시호가: 확대
]
```

ZigZag의 `session_min_wave_bars_table`/`early_session_atr_multiplier_max`는 **봉 수 제한** 방식인 반면, 신규는 **배율 직접 조정** 방식이라 즉각 반응합니다.

#### 확정 로직 단순화

```python
# 기존 ZigZag: pending_confirm dict + remaining 카운터
self._pending_confirm = {"type": "high", "price": ..., "remaining": 2}
# → 매 봉마다 remaining 감소, 0이 되면 확정
# → freeze_on_confirm: True/False 분기, max_wait_bars 처리 등

# 신규 ATRAdaptivePivot: 동일 구조이나 단순화
# + confirmation_bars=1 (기본) → ZigZag 대비 1봉 빠름
# + freeze 없음 — 더 극단적인 값이 오면 항상 갱신
```

#### 가짜 피봇(Fake Pivot) 필터: 되돌림 취소

```python
# 신규: 되돌림 비율이 threshold의 30% 미만이면 후보 취소
if pt == "high" and (pp - low) < thr * 0.3:
    self._pending_confirm = None  # 취소
```

ZigZag는 `반대후보교체` 방식이라 취소 즉시 반대 후보를 등록합니다. 신규는 취소 후 방향 탐색을 재개하여 과도한 신호 교체를 줄입니다.

### 3.2 FractalConfirmation — 확증 레이어

빌 윌리엄스 프랙탈의 핵심:

```
고점 프랙탈 조건 (lookback=2):
  high[i] > high[i-2], high[i-1], high[i+1], high[i+2]
  + volume[i] >= avg_volume(i-lookback~i-1) × volume_spike_ratio
  + 이전 프랙탈과 min_bar_gap 봉 이상 간격
```

ZigZag는 **단일 조건(가격 되돌림)** 으로 피봇을 판정하지만, Fractal은 **좌우 극값 비교 + 거래량 증가** 두 가지를 동시 요구합니다. 거래량 급증 없는 가격 극값은 프랙탈로 인정하지 않아 기관 개입 없는 노이즈 피봇을 차단합니다.

**ZigZag와의 관계**: FractalConfirmation은 ATRAdaptivePivot의 신호를 **이중 확증**하는 보조 레이어입니다. 두 지표가 동시에 신호를 낼 때 TradeExecutionGate 진입 조건을 강화하는 방식으로 활용합니다.

---

## 4. 신규 Step 2: Market Structure Break + OI Gate

### 4.1 MarketStructureBreak — ZigZag와 근본적으로 다른 개념

ZigZag는 **"충분히 되돌아왔는가?"** 를 묻습니다.  
MSB는 **"이전에 저항으로 작용했던 레벨을 돌파했는가?"** 를 묻습니다.

```
ZigZag 관점:
  현재가 → 직전 극값에서 threshold 이상 하락했는가?
  Yes → 고점 피봇 후보 등록

MSB 관점:
  현재 고가 → 직전 swing high (저항선) 보다 높은가?
  Yes → BOS_UP 발생 (구조 변화)
  현재 구조가 DOWNTREND + BOS_UP → CHoCH_UP 발생 (추세 반전 신호)
```

#### BOS vs CHoCH

| 신호 | 의미 | 강도 |
|------|------|------|
| `BOS_UP` | 상승 구조 내 swing high 돌파 → 구조 계속 | 보통 (0.3) |
| `BOS_DOWN` | 하락 구조 내 swing low 돌파 → 구조 계속 | 보통 (0.3) |
| `CHoCH_UP` | **하락** 구조 중 swing high 돌파 → **반전 신호** | 강함 (0.4) |
| `CHoCH_DOWN` | **상승** 구조 중 swing low 돌파 → **반전 신호** | 강함 (0.4) |

CHoCH(Change of Character)는 기존 추세에서 처음으로 반대 방향 구조 붕괴가 발생하는 시점입니다. 프로 트레이더들이 "추세 전환의 1차 증거"로 보는 신호로, ZigZag로는 포착하기 어렵습니다.

#### MSB Score 계산

```python
msb_score = (
    bos_weight[bos_signal]        # 0.0 ~ 0.4 (신호 강도)
  + structure_consistency_score   # 0.0 ~ 0.25 (구조 유지/전환)
  + hh_ll_continuity_score        # 0.0 ~ 0.35 (HH/LL 연속성)
)
# 결과: [0.0, 1.0]
```

ZigZag의 `azz_structure_conf`는 **현재 구조의 일관성**만 측정하지만, MSB Score는 **BOS 발생 여부 + 구조 방향성 + 연속성** 세 가지를 종합합니다.

### 4.2 OIStructureGate — KOSPI200 특화 교차 분석

```python
# OI peak 근접 + BOS/CHoCH 동시 발생 시 신호 강화
near_call = |close - call_oi_peak| / close <= oi_proximity_pct / 100

if CHoCH_UP and near_call:
    score = msb_score × choch_oi_boost  # 기본 1.7배
elif BOS_UP and near_call:
    score = msb_score × bos_oi_boost    # 기본 1.4배
else:
    score = msb_score  # 부스트 없음
```

기존 파이프라인의 `guardrail_mixin.py`는 OI 레벨을 **진입 금지 구역** 으로만 활용합니다(`_guardrail_oi_enabled`). OIStructureGate는 OI 레벨을 **신호 강화 트리거** 로 활용하는 다른 관점입니다.

**왜 OI peak 근처의 BOS가 강력한가?**  
Call OI peak = 딜러 헤징 매도 집중 = 상단 저항. 이 레벨을 돌파하는 BOS는 딜러 헤징 물량을 소화한 것이므로 추세 지속 가능성이 높습니다.

---

## 5. 신규 Step 3: Kalman Turning Point + PivotScoreIntegrator

### 5.1 KalmanTurningPoint — EMA/ZigZag 대비 원리적 차이

#### 기존: 가격 직접 비교

```
ZigZag: close[t] vs pending_high/low → 되돌림 거리 계산
EMA derivative: (EMA[t] - EMA[t-1]) / EMA[t-1] → 기울기
```

#### 신규: Kalman Filter — 상태 추정

가격 시계열을 "노이즈가 섞인 관측값"으로 보고, 실제 추세 상태를 추정합니다.

```
State vector:  x = [[price_est], [slope]]    (2×1 column vector)
Transition:    F = [[1, 1], [0, 1]]          (등속 운동 모델)
Observation:   H = [[1, 0]]                  (가격만 관측)

매 봉 처리:
  1. Predict: x_pred = F @ x,  P_pred = F @ P @ F.T + Q
  2. Innovation: z - H @ x_pred  (관측-예측 차이)
  3. Kalman Gain: K = P_pred @ H.T / (H @ P_pred @ H.T + R)
  4. Update: x = x_pred + K × innovation

핵심 파라미터:
  Q (process noise): 클수록 slope가 빠르게 변함 (기본: 0.01)
  R (observation noise): 클수록 관측을 덜 신뢰 (기본: 2.0)
```

Adaptive Q (`adaptive_q=True`):

```python
# ATR에 비례해 Q를 동적 조정
atr = mean(tr_buffer[-14:])
q_adaptive = Q × atr²  # 변동성 큰 봉 → Q 증가 → slope 빠르게 추적
```

#### Slope Flip = 변곡점

```python
# slope 부호 전환 + 최소 크기 조건
flip = (slope > 0) != (prev_slope > 0)  # 부호 전환
flip = flip and abs(slope) >= slope_flip_min  # 노이즈 플립 제거
# slope: 양→음 → turning_signal = "down" (고점 변곡)
# slope: 음→양 → turning_signal = "up"  (저점 변곡)
```

ZigZag가 **되돌림 거리**로 변곡을 판단한다면, Kalman은 **추세 기울기 방향 전환**으로 판단합니다. Kalman은 노이즈를 분리했기 때문에 ZigZag보다 lag가 작고 허위 전환(노이즈 플립)이 적습니다.

#### EMA derivative vs Kalman slope 비교

| 특성 | EMA derivative | Kalman slope |
|------|----------------|--------------|
| 노이즈 민감도 | 높음 | 낮음 (필터링) |
| 반응 속도 | 빠름 (EMA 기간에 비례) | Q, R 조합으로 조정 |
| 이론적 근거 | 단순 이동 평균 | 최소분산 추정기 (베이즈 최적) |
| Lag | EMA 기간의 절반 | Q/R 비율에 따라 가변 |
| 적응적 변동성 | 없음 | Adaptive Q로 가능 |

### 5.2 PivotScoreIntegrator — 4개 레이어 통합

#### 통합 공식

```python
# 활성 레이어 가중치 정규화
total_w_active = Σ w_i  (활성 레이어만)
norm_w_i = w_i × (1 + total_w_inactive / total_w_active)

# 가중합
total_score = Σ (score_i × norm_w_i)

# 레짐 조정
if regime == "uptrend" or "downtrend":
    adjusted_score = clip(total_score × 1.15, 0, 1)
elif regime == "ranging":
    adjusted_score = clip(total_score × 0.85, 0, 1)

# 방향 다수결
votes = {
    "aap": "long" if new_pivot=="new_low" else "short",
    "msb": "long" if bos_signal in {BOS_UP, CHOCH_UP} else "short",
    "kf":  "long" if turning_signal=="up" else "short",
}
signal = "long" if long_votes > short_votes else "short"

# 임계값 필터
if adjusted_score < entry_threshold (0.55):
    signal = "none"
```

기존 파이프라인의 `TradeExecutionGate`는 확률 예측값(`prob_buy > 0.62`)과 신뢰도(`confidence == "HIGH"`) 조건을 사용합니다. `PivotScoreIntegrator`는 이 조건에 독립적으로 추가되어 "구조적 변곡점 강도" 조건을 AND로 부여합니다.

---

## 6. 알고리즘 핵심 수식 비교

### 변곡점 판정 조건

```
기존 ZigZag:
  reversal_abs = |pending_high - current_low|  또는  |current_high - pending_low|
  reversal_pct = reversal_abs / close × 100
  조건: reversal_pct >= thr_pct  (ER 기반 동적 임계값)

신규 ATRAdaptivePivot:
  reversal_abs = |pending_high - current_low|  또는  |current_high - pending_low|
  조건: reversal_abs >= ATR × base_multiplier × session_scale(t)
  (백분율 변환 없음 — 포인트 단위 직접 비교)

신규 MSB (BOS):
  조건: current_high > ref_swing_high × (1 + bos_buffer_pct/100)
  또는: current_low  < ref_swing_low  × (1 - bos_buffer_pct/100)

신규 Kalman:
  slope_t = x[1, 0]  (Kalman 추정 velocity)
  조건: (slope_t > 0) ≠ (slope_{t-1} > 0) AND |slope_t| >= slope_flip_min
```

### 신호 강도 표현

```
기존: categorical  →  "new_high" | "new_low" | "none"  (3가지)

신규: continuous   →  PivotScore ∈ [0.0, 1.0]          (연속값)
                   →  SignalStrength: "STRONG" | "MODERATE" | "WEAK" | "none"
                   →  Signal: "long" | "short" | "none"
```

---

## 7. Transformer Feature 비교

### 기존 ADAPT_KEYS (38개)

```python
# Adaptive SuperTrend (9개)
["ast_direction", "ast_dist_pct", ..., "ast_signal"]

# Adaptive ZigZag (24개)
["azz_direction", "azz_wave_size_pct", ..., "azz_pending_prob"]

# Cross Features (4개)
["cross_trend_agreement", "cross_at_support", ...]
```

모든 피처가 ZigZag의 `ZigZagState` 또는 SuperTrend의 `SuperTrendState` 에서 파생됩니다. 단일 알고리즘의 서로 다른 측면을 표현하므로 **피처 간 상관관계가 높습니다**.

### 신규 추가 피처 (25개 추가 → 총 63개)

```python
# ATR Adaptive Pivot (3개) — 새로운 관점
["aap_atr", "aap_threshold_pct", "aap_pivot_score"]
# → 변동성 자체와 동적 threshold의 현재값을 직접 노출

# Market Structure Break (8개) — 구조적 관점
["msb_bos_signal", "msb_structure", "msb_hh_ratio",
 "msb_ll_ratio", "msb_sh_dist", "msb_sl_dist", "msb_score", "msb_choch"]
# → 가격 구조(HH/LL 패턴)와 구조 붕괴를 별도로 표현

# Kalman Turning Point (8개) — 수학적 관점
["kf_slope_norm", "kf_slope_flip", "kf_slope_surge",
 "kf_turning_signal", "kf_score", "kf_dev_norm",
 "kf_innovation", "kf_gain"]
# → 노이즈 제거 후 순수 추세 기울기 정보

# 통합 PivotScore (6개) — 메타 피처
["ps_total_score", "ps_adjusted_score", "ps_signal",
 "ps_strength", "ps_long", "ps_short"]
# → 4개 레이어의 통합 의견을 단일 값으로 압축
```

**설계 의도**: 각 레이어가 서로 다른 관점에서 같은 현상을 설명하므로, PriceTransformer는 이들 사이의 합의/불일치를 학습할 수 있습니다.

---

## 8. 실전 동작 비교 (KOSPI200 1일 시뮬레이션)

> 시뮬레이션 조건: 08:45~15:45 기준 411봉, 360pt 시작 후 완만한 상승 추세, 랜덤 노이즈 σ=0.4pt

### 신호 발생 수 비교

| 지표 | 파라미터 | 411봉 중 신호 수 | 특이사항 |
|------|---------|-----------------|---------|
| ZigZag (`confirmation_bars=2`) | `atr_mult=1.5` | 3개 | 웜업 6봉 내 첫 신호 (초기범위 확정) |
| ATRAdaptivePivot (`confirmation_bars=2`) | `base_mult=3.0` | 6개 | 웜업 20봉 후 첫 신호 |
| ATRAdaptivePivot (`confirmation_bars=1`) | `base_mult=2.0` | 29개 | 소파동도 탐지 |
| MarketStructureBreak | 기본 설정 | 317개 | `bos_buffer_pct=0.05%` 너무 낮음 ⚠️ |
| KalmanTurningPoint | `q=0.01, r=2.0` | 3개 | slope_flip 기준 |

**MSB 과다 신호 문제**: `bos_buffer_pct=0.05%`는 KOSPI200 1분봉 노이즈 대비 너무 낮습니다. 실전에서는 `0.15~0.30%` 권장. ATRAdaptivePivot 피봇을 외부 주입(`pivot_points`)할 때 스윙 레벨이 충분히 확정된 후에 MSB 신호를 평가해야 합니다.

### 신호 시점 비교 예시 (시뮬레이션 중 첫 하락→상승 변곡)

```
실제 저점 발생: bar 35 (09:35, 358.41pt)

ZigZag 확정:   bar 40 (09:40, 358.78pt) — lag 5봉 = +5분
ATRAdaptivePivot: bar 45 (09:45, ...) — lag 10봉 = +10분
Kalman slope_flip: bar 50경 (slope 음→양 전환 시점)
```

이 예시에서는 ZigZag가 빠릅니다. 그러나 ZigZag는 **"되돌아왔는가"** 기준이라 threshold가 클수록 늦어지고, ATRAdaptivePivot은 파라미터 조정으로 빠르게 만들 수 있습니다. Kalman은 노이즈 제거 때문에 slope 변화가 누적되어야 전환을 인식합니다.

---

## 9. 파이프라인 데이터 흐름 비교

### 기존 흐름

```
ebestapi/live.py (FH0/OH0 틱)
  │
  ▼
tick_processor.py → 1분봉 OHLCV 집계
  │
  ▼
adaptive_mixin.py
  │
  ├─ _adaptive_mgr.update(H, L, C, open, bar_time)
  │    └─ AdaptiveIndicatorManager
  │         ├─ AdaptiveSuperTrend.update()
  │         └─ AdaptiveZigZag.update()
  │              └─ ZigZagState → {"transformer": azz_* dict}
  │
  └─ adaptive_features = res["transformer"]  # azz_* 38개
       │
       ▼
  prediction_mixin.py → PriceTransformer 입력
```

### 신규 흐름 (추가된 부분)

```
adaptive_mixin.py
  │
  ├─ _adaptive_mgr.update()  [기존 — 변경 없음]
  │    └─ adaptive_features = res["transformer"]  # azz_* 38개
  │
  └─ _merge_step123_features(self, H, L, C, bar_time, features=adaptive_features)
       │
       ├─ self._aap.update()
       │    └─ features.update(aap.get_transformer_features())  # aap_* 3개
       │
       ├─ self._msb.update(pivot_points=aap.confirmed_pivots)
       │    └─ features.update(msb.get_transformer_features())  # msb_* 8개
       │
       ├─ self._kf.update()
       │    └─ features.update(kf.get_transformer_features())   # kf_* 8개
       │
       ├─ oi_score = self._oi_gate.score(msb_state, close,
       │             self._last_opt_snap["_oi_levels"])
       │    └─ features.update(oi_gate.get_transformer_features())  # oi_* 6개
       │
       └─ result = self._integrator.compute(aap, msb, oi, kf scores + signals)
            └─ features.update(integrator.get_transformer_features())  # ps_* 6개
                 │
                 ▼
            adaptive_features  # 총 63개 키
                 │
                 ▼
            prediction_mixin.py → PriceTransformer 입력
```

**중요**: `_merge_step123_features`는 기존 `adaptive_features` dict를 **in-place 수정**합니다. 따라서 ZigZag/SuperTrend 피처는 그대로 유지되고 신규 피처가 추가됩니다. 어느 한 레이어가 예외를 발생시켜도 `try/except`로 흡수하므로 기존 파이프라인은 영향 없습니다.

---

## 10. 설정 파라미터 비교

### 기존 ZigZag 핵심 파라미터

```json
"adaptive_indicator": {
  "zigzag": {
    "atr_multiplier": 1.5,
    "atr_multiplier_min": 1.0,
    "atr_multiplier_max": 4.0,
    "confirmation_bars": 2,
    "confirmation_bars_ranging": 2,
    "confirmation_bars_unknown": 3,
    "pivot_threshold_min_pct": 0.3,
    "pivot_threshold_max_pct": 3.0,
    "min_wave_pct": 0.25,
    "min_wave_atr_ratio": 0.5,
    "use_atr_based_filtering": true
  }
}
```

### 신규 4-Layer 파라미터 (config.json에 추가)

```json
"adaptive_indicator": {
  "atr_pivot": {
    "atr_period": 14,
    "base_multiplier": 2.0,
    "multiplier_min": 1.2,
    "multiplier_max": 3.5,
    "er_period": 10,
    "confirmation_bars": 1,
    "min_wave_atr_ratio": 0.5,
    "warmup_bars": 20
  },
  "msb": {
    "swing_lookback": 3,
    "bos_buffer_pct": 0.20,
    "structure_lookback_pivots": 6,
    "choch_enabled": true
  },
  "kalman": {
    "q": 0.01,
    "r": 2.0,
    "warmup_bars": 15,
    "slope_flip_min": 0.005,
    "adaptive_q": true
  },
  "integrator": {
    "w_aap": 0.30,
    "w_msb": 0.30,
    "w_oi": 0.20,
    "w_kf": 0.20,
    "entry_threshold": 0.55,
    "strong_threshold": 0.72,
    "regime_boost": 1.15,
    "regime_suppress": 0.85
  },
  "oi_proximity_pct": 0.3
}
```

### 파라미터 조정 가이드

| 상황 | 권장 조정 |
|------|---------|
| 신호 너무 많음 | `base_multiplier` ↑ (2.0→2.5), `entry_threshold` ↑ (0.55→0.65) |
| 신호 너무 적음 | `base_multiplier` ↓ (2.0→1.5), `confirmation_bars` 0으로 |
| Lag 줄이고 싶음 | `confirmation_bars` 0 또는 1, `kalman.q` ↑ (0.01→0.05) |
| 장초반 노이즈 많음 | `session_multiplier_table` 에서 09:00~09:30 배율 ↑ |
| MSB 신호 과다 | `bos_buffer_pct` ↑ (0.05→0.20) |
| Kalman 반응 느림 | `q` ↑ (0.01→0.05), `r` ↓ (2.0→0.5) |

---

## 11. 모델 학습 영향 및 마이그레이션

### ADAPT_KEYS 변경의 영향

```
기존: ADAPT_KEYS 38개 → adaptive_block_dim = 38
신규: ADAPT_KEYS 63개 → adaptive_block_dim = 63
```

**기존 학습된 모델은 그대로 사용 불가** — 입력 차원이 달라집니다.

#### 안전한 마이그레이션 방법

**방법 A — 신규 키 제로 패딩 (단기 운용용)**

```python
# pipeline.py에서 adaptive_block_dim 계산 전
_known_dim = 38  # 기존 모델 학습 차원
if len(ADAPT_KEYS) > _known_dim:
    # 신규 키는 제로 초기화
    for k in ADAPT_KEYS[_known_dim:]:
        adaptive_features[k] = 0.0
```

**방법 B — 신규 피처로 재학습 (권장)**

1. `prediction/features/features.py` 수정 완료된 상태에서 데이터 수집
2. `prediction/data_builder.py` 에서 63개 차원으로 새 학습 데이터 생성
3. `training/train.py` 실행 → PriceTransformer 재학습

재학습 없이 신규 피처를 평가하려면 `prediction/features/features.py`에서 기존 38개 순서를 유지한 채로 63개를 사용합니다. 기존 모델은 앞 38개 입력만 사용하므로 추가된 25개는 모델이 볼 수 없지만, 파이프라인 코드는 변경 없이 유지됩니다.

---

## 12. 한계 및 주의사항

### 12.1 신규 방식의 알려진 한계

**ATRAdaptivePivot**
- `confirmation_bars=0` 설정 시 repaint 발생 가능 (ZigZag와 동일 문제)
- warmup 20봉 미만 데이터에서는 신호 없음
- `session_multiplier_table` 미설정 시 KOSPI200 장초반 노이즈에 취약

**MarketStructureBreak**
- 스윙 레벨이 충분히 쌓이기 전 (피봇 2개 미만) BOS 판정 불안정
- `bos_buffer_pct` 너무 작으면 매 봉 BOS 신호 → MSB 단독 사용 금지
- 반드시 ATRAdaptivePivot의 `confirmed_pivots`를 외부 주입해야 의미 있음

**KalmanTurningPoint**
- `q=0.01, r=2.0` 기본값은 추세가 약한 날 slope_flip이 거의 없음
- 고변동 구간에서 Adaptive Q가 너무 높아지면 slope 진동 증가
- `slope_flip_min` 너무 작으면 노이즈 플립 급증 → 0.01pt 이상 권장

**PivotScoreIntegrator**
- MSB BOS 과다 신호 환경에서 msb_score 가중치(0.30)가 total_score를 왜곡
- `w_msb`를 0.20으로 낮추거나 MSB bos_buffer 조정 필요

### 12.2 기존 ZigZag 완전 대체 전 고려사항

신규 4-Layer는 기존 `AdaptiveZigZag`를 **대체**하는 것이 아니라 **병렬로 추가**하는 방식으로 구현했습니다. `azz_*` 피처는 그대로 유지되고 `aap_*/msb_*/kf_*/ps_*` 가 추가됩니다.

완전 대체(ZigZag → ATRAdaptivePivot)를 원한다면:
1. `AdaptiveIndicatorManager`의 ZigZag를 ATRAdaptivePivot으로 교체
2. `prediction/mixins/adaptive_mixin.py`의 `zigzag_state` 참조 코드 수정
3. `gui/engines/chart_engine.py`의 ZigZag 렌더링 코드 교체
4. `prediction/features/features.py`에서 `azz_*` 키를 `aap_*`로 교체

단계적 마이그레이션을 권장합니다.

---

## 요약 비교표

| 항목 | 기존 AdaptiveZigZag | 신규 4-Layer |
|------|---------------------|-------------|
| **변곡점 판정** | 되돌림 거리 ≥ ATR×배수 | 4개 관점 종합 점수 |
| **신호 형식** | categorical (3종) | continuous [0,1] + categorical |
| **Lag** | 평균 5~8봉 | Layer별 다름 (KF 최소) |
| **노이즈 필터** | ER 기반 동적 배수 | ATR 절대값 + 거래량 + Kalman |
| **구조 인식** | ZigZag 내부 구조 분석 | MSB — 구조 붕괴 직접 탐지 |
| **OI 연동** | guardrail (진입 금지만) | OIStructureGate (신호 강화) |
| **Feature 수** | azz_* 24개 | aap_3 + msb_8 + kf_8 + ps_6 = 25개 추가 |
| **재학습 필요** | — | 63차원 재학습 필요 |
| **설정 복잡도** | 중간 | 높음 (4개 레이어 개별 설정) |
| **실시간 안전성** | 검증됨 | 신규 — 장 외 테스트 후 적용 권장 |

---

*이 문서는 SkyPredictor Step 1~3 구현 완료 시점(2026-05) 기준으로 작성되었습니다.*  
*Step 4 (Volume Profile, EMD/Wavelet 기반 변곡 탐지) 구현 시 갱신 예정입니다.*
