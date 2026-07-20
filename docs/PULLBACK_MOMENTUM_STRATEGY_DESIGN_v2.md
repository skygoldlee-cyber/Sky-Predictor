# Pullback Momentum Strategy Design (v2)

> 기존 KDE mean-reversion pipeline을 대체하는, KP200 선물 5분봉 기반 알고리즘 설계.
> **성능 목표는 가설이며, baseline backtest 실측 전까지 성능표는 채우지 않는다.**

**가설 목표 (검증 전, 참고용):** Sharpe ≥ 2.0, 승률 55~70%, PF ≥ 1.8, Calmar ≥ 1.5.
KP200 5분봉 단일 방향 pullback + rule-based로 비용 반영 후 Sharpe 2.0은 도전적 목표임을 전제로 한다. 실측이 이보다 낮게 나오는 것이 정상적인 출발점이다.

---

## 1. 핵심 아이디어: Regime-Conditional Pullback Momentum

KOSPI200 선물은 추세장에서 **추세 방향으로의 짧은 평균회귀(pullback)** 진입이 가장 안정적인 edge를 제공한다는 가설.

- **추세 장**: 단기 조정(EMA20 / 볼린저 중심)에서 추세 방향 진입
- **횡보/고변동성 장**: 거래 최소화 또는 관망
- **Risk/Reward ≥ 1:1.5** 유지
- **ATR 기반 손절/익절 + trailing** → 변동성 정규화

---

## 2. Regime 분류 (역할: "언제 거래할지")

Regime은 **거래 허용 여부(on/off)와 방향**만 결정한다. 개별 봉 진입 판정은 3번 entry rule이 담당한다.
→ regime과 entry가 같은 피처(EMA 등)를 이중 필터로 쓰지 않도록, regime 층은 **더 느린 프레임(EMA60/ADX/ATR percentile)** 만 사용하고, entry 층은 **더 빠른 프레임(EMA20/RSI/BB/volume)** 만 사용한다.

| Regime | 조건 (느린 프레임만) | 거래 |
|--------|------|------|
| **Bull trend** | close > EMA60, ADX > 25, ATR percentile < 70% | Long entry 허용 |
| **Bear trend** | close < EMA60, ADX > 25, ATR percentile < 70% | Short entry 허용 |
| **Volatile** | ATR percentile ≥ 70% | 관망 (또는 사이즈 0.5×, 별도 검증 후) |
| **Neutral/chop** | ADX < 20 | 거래 금지 |

- EMA60(추세 방향), ADX(14), ATR(14) 60일 percentile
- **주의**: EMA20은 regime 판정에서 제외 → entry 층 전용

---

## 3. Entry Rule (역할: "어떤 봉에서 진입할지")

> **Look-ahead 방지 원칙 (rule-based 포함):** 모든 지표는 **직전 봉(t-1) 확정 종가**로 계산한 값으로 t 시점 신호를 판정한다. 진행 중인 봉의 미확정 종가로 신호를 만들지 않는다. basis 등 외부 데이터도 t-1 시점 스냅샷만 사용한다.

### Long (Bull trend regime일 때만)

```text
조건 1: EMA20 > EMA60                              (빠른 추세 정렬)
조건 2: 직전 봉에서 close가 EMA20 아래 터치 후 재상승
        or  low <= lower BB * 1.01                 (풀백)
조건 3: RSI(14) ∈ [30, 50]
조건 4: volume(t-1) > 20-period volume mean(t-1)   (확인)
조건 5: basis(t-1) >= -0.05%                       (선물 과다 디스카운트 회피)
```

### Short (Bear trend regime일 때만)

```text
조건 1: EMA20 < EMA60
조건 2: 직전 봉에서 close가 EMA20 위 터치 후 재하락
        or  high >= upper BB * 0.99
조건 3: RSI(14) ∈ [50, 70]
조건 4: volume(t-1) > mean
조건 5: basis(t-1) <= +0.05%
```

### Session/Time Filter

- 진입 금지: 개장 직후 20분, 마감 직전 30분
- 월요일/공휴일 직후 보수적 진입

---

## 4. Exit Rule

| 항목 | 규칙 |
|------|------|
| **Stop loss** | 진입가 ± 1.0 × ATR(14) |
| **Take profit** | 진입가 ± 1.5 × ATR(14) → R/R = 1:1.5 |
| **Trailing stop** | 최초 1.0 ATR 수익 발생 후, **0.75 ATR** trailing (이익 구간 잠금) |
| **Time stop** | 진입 후 16봉(80분) 내 미청산 시 현가 청산 |
| **Session close** | 미청산 포지션은 마감 직전 강제 청산 |

> **변경**: trailing 폭을 1.5→0.75 ATR로 축소. 1.5 ATR로 두면 1.0 ATR 도달 후 되돌림이 손실 구간까지 허용되어 실질 R/R이 문서상 1:1.5보다 크게 나빠진다. trailing과 time/session stop 우선순위는 손절 > trailing > time > session 순으로 명시.

---

## 5. Position Sizing

```python
risk_per_trade = 0.004   # 계좌 자본의 0.4% (일일/주간 한도와 정합)
atr = current_ATR
stop_distance = atr * tick_value           # tick_value = KP200 계약 승수 명시 필수
contracts = floor(risk_per_trade * capital / stop_distance)
```

- **변경**: 트레이드당 리스크 2% → 0.4%. 5분봉 전략은 하루 다수 진입을 전제하는데, 2%면 첫 손절 한 번으로 일일 -2% 한도에 도달해 당일 종료된다(6번과 모순). 0.3~0.5% 구간에서 일일/주간 한도와 함께 성립하는지 스프레드시트로 사전 검증.
- 계약 기준(일반/미니) 및 `tick_value` 명시 필요. 소액 계좌에서 `floor()`가 0/1로 수렴하는 구간을 별도 확인.
- Volatile regime: `contracts * 0.5` (검증 후 적용)
- 연속 손실 3회 시 다음 6봉 진입 금지 (cooldown)

---

## 6. Risk Management

| 항목 | 규칙 |
|------|------|
| 일일 손실 한도 | 계좌 **-2%** 도달 시 당일 중단 (≈ 트레이드 5회분 리스크) |
| 주간 MDD | **-4%** 도달 시 주간 중단 |
| 최대 노출 | 동시 1 direction, 1 position |
| 슬리피지 | baseline 1 tick/side, 민감도 테스트 2·3 tick/side |
| 수수료 | 브로커 실제 수수료 반영 |

> risk_per_trade(0.4%) × 5회 ≈ 일일 한도(2%). 두 수치가 서로 정합적인지가 sizing 설계의 핵심.

---

## 7. ML 적용 방식 (보조 필터/스코어)

KDE는 1차 신호가 아니라 필터/스코어로만 사용.

| 단계 | 역할 |
|------|------|
| Regime classifier | LightGBM으로 trend/volatility regime 예측 |
| Entry scorer | XGBoost가 기술적 조건 + KDE tail + 거시 피처로 진입 확률 산출 |
| Position sizer | 예상 R-multiple에 비례해 size 조정 |
| Label | 승/패가 아닌 **R-multiple** (`net_pnl / stop_distance`) |

### 피처
- OHLC: EMA20/60, ADX, ATR, BB width, RSI, MACD, session time
- KDE: return의 regime별 tail 확률(z-score) — **walk-forward로 과거 구간만으로 fit** (전체 기간 fit은 look-ahead)
- 거시/수급: KP200 spot, basis, VIX, USD/KRW, 거래량/미결
- 시차: 모든 피처 t-1 lag

---

## 8. 성능 기록 (baseline backtest 후 채운다)

> 검증 전 예상 범위를 기입하지 않는다. 사전 기입은 이후 튜닝·피처선택·결과 취사선택을 목표치 방향으로 편향시킨다.

| 지표 | 가설 목표 | 실측 (baseline) | 실측 (ML) |
|------|------|------|------|
| Sharpe | ≥ 2.0 | **-17.04** | _TBD_ |
| 승률 | 55~70% | **15.14%** | _TBD_ |
| Profit Factor | ≥ 1.8 | **0.04** | _TBD_ |
| Calmar | ≥ 1.5 | **-1.01** | _TBD_ |
| 연간 거래 수 | ≥ 500 | **~36** | _TBD_ |
| MDD/연수익 | 낮음 | **53.55%** | _TBD_ |

---

## 8.1 Baseline 및 랜덤 대조군 결과 분석

### Baseline (rule-based signal)

`pullback_momentum_backtest.py`, 2019-2026, 1 tick/side slippage:

- Sharpe: **-17.04**, 승률: **15.14%**, PF: **0.04**, MDD: **53.55%**, 거래 수: 251

### Random control (동일 exit/sizing, 진입만 무작위)

- 확률 `p=0.0025`로 진입 시: Sharpe **-12.49**, 승률 **18.50%**, PF **0.11**, 거래 수: 535
- 확률 `p=0.05`로 진입 시: Sharpe **-26.45**, 승률 **24.13%**, PF **0.21**, 거래 수: 4,500

### 해석

- **Random control도 비슷하게 큰 손실**. 이는 entry 신호만의 문제가 아니라 **exit/sizing 규칙 자체가 기대값을 음수로 만듦**.
- 1 ATR stop / 1.5 ATR target + 1 tick/side 슬리피지 조합에서는 무작위 진입으로도 장기 손실이 나며, signal-based 결과가 random보다 더 나쁘기도 함.
- 따라서 **진입 신호 개선만으로는 목표에 도달할 수 없음**. exit/sizing 구조(예: ATR 배수, 손익비, trailing, time stop) 자체를 먼저 random control에서 생존 가능한 수준으로 재설계해야 함.
- 다음 단계: random control에서 손실이 최소화되는 exit parameter 공간을 찾고, 그 위에 entry scorer를 얹는다.

## 9. 검증 로드맵

1. **Baseline 구현**: rule-based로 2019–2026 backtest (t-1 확정봉 신호 엄수)
2. **랜덤 대조군**: pullback 신호를 랜덤 진입으로 치환. edge가 **신호**에서 오는지 **exit/사이징**에서 오는지 분리. (성능 대부분이 ATR exit에서 나오는 경우가 흔함)
3. **Cost sensitivity**: 1/2/3 tick slippage별 Sharpe/PF
4. **파라미터 민감도**: EMA(20/60), RSI 밴드, ATR 배수(1.0/1.5/0.75), time stop 봉수에 대한 Sharpe surface. 소폭 변경에 성능 붕괴 시 과최적화 판정
5. **Regime별 분해**: bull/bear/neutral/volatile PnL 확인
6. **ML enhancement**: LightGBM regime + XGB entry scorer
7. **Walk-forward + paper trading**: 2027 데이터 확보 후 OOS

---

## 10. 요약

KDE 중심 mean-reversion에서 **추세 방향 pullback entry + ATR 기반 exit + regime 사이징**으로 전환. v1 대비 수정점:

1. **성능표 사전 기입 삭제** → baseline 실측 후 기록
2. **regime/entry 역할 분리** (느린 프레임 vs 빠른 프레임, 이중 필터 제거)
3. **sizing vs 일일 한도 정합** (트레이드당 2%→0.4%)
4. **trailing 폭 축소** (1.5→0.75 ATR)
5. **로드맵에 랜덤 대조군 + 파라미터 민감도 추가**, look-ahead 원칙을 rule-based에도 명시

ML은 보조 필터/스코어로만 활용해 해석 가능성을 유지한다.
