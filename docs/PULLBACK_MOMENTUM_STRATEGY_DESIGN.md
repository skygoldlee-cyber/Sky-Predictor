# Pullback Momentum Strategy Design

> 기존 KDE mean-reversion pipeline을 대체하는, KP200 선물 5분봉 기반의 새로운 알고리즘 설계.
> 목표 성능: Sharpe Ratio ≥ 2.0, 승률 55~70%, Profit Factor ≥ 1.8, Calmar Ratio ≥ 1.5, Max Drawdown/연수익 비율 낮음.

---

## 1. 핵심 아이디어: Regime-Conditional Pullback Momentum

KOSPI200 선물은 추세가 있는 시장에서 **추세 방향으로의 짧은 평균회귀(pullback)** 진입이 가장 안정적인 edge를 제공합니다.

- **추세 장**: 단기 조정(EMA20 / 볼린저 밴드 중심)에서 추세 방향으로 진입 → 높은 승률(55~70%)
- **횡보/고변동성 장**: 거래 최소화 또는 소규모 mean-reversion만 수행
- **Risk/Reward 1 : 1.5 이상** 유지 → Profit Factor ≥ 1.8 달성 기반
- **ATR 기반 손절/익절 + trailing stop** → Sharpe/Calmar/MDD 개선

---

## 2. Regime 분류

| Regime | 조건 | 전략 |
|--------|------|------|
| **Bull trend** | close > EMA60, fast > slow, ADX > 25, ATR 낮음 | Long pullback |
| **Bear trend** | close < EMA60, fast < slow, ADX > 25, ATR 낮음 | Short pullback |
| **Bull volatile** | close > EMA60, ATR 상위 30% | 소규모 long or 관망 |
| **Bear volatile** | close < EMA60, ATR 상위 30% | 소규모 short or 관망 |
| **Neutral/chop** | ADX < 20 or BB width 낮음 | 거래 최소화 |

- EMA: 20, 60
- ADX: 14-period
- ATR: 14-period, 60일 percentile 기준 변동성 군집

---

## 3. Entry Rule

### Long (Bull trend pullback)

```text
조건 1: close > EMA60  and  EMA20 > EMA60          (추세)
조건 2: close가 EMA20 아래로 돌파 후 재상승
        or  close <= lower BB * 1.01                (풀백)
조건 3: RSI(14) ∈ [30, 50]                         (과열x, 반등 여력)
조건 4: 5분봉 volume > 20-period volume mean         (확인)
조건 5: 현물/선물 basis >= -0.05%                  (선물 디스카운트 과다 회피)
```

### Short (Bear trend pullback)

```text
조건 1: close < EMA60  and  EMA20 < EMA60
조건 2: close가 EMA20 위로 돌파 후 재하락
        or  close >= upper BB * 0.99
조건 3: RSI(14) ∈ [50, 70]
조건 4: volume > mean
조건 5: basis <= +0.05%
```

### Session/Time Filter

- **진입 금지**: 개장 직후 20분, 마감 직전 30분
- **월요일/공휴일 직후**: 보수적 진입

---

## 4. Exit Rule

| 항목 | 규칙 |
|------|------|
| **Stop loss** | 진입가 ± 1.0 × ATR(14) |
| **Take profit** | 진입가 ± 1.5 × ATR(14) → R/R = 1 : 1.5 |
| **Trailing stop** | 최초 1.0 ATR 수익 발생 후, 1.5 ATR trailing |
| **Time stop** | 진입 후 16봉(80분) 내 미청산 시 현가 청산 |
| **Session close** | 미청산 포지션은 마감 직전 강제 청산 |

> 이 exit 구조가 Sharpe/Calmar/PF 달성의 핵심입니다.

---

## 5. Position Sizing

```python
risk_per_trade = 0.02  # 계좌 자본의 2%
atr = current_ATR
stop_distance = atr * tick_value
contracts = floor(risk_per_trade * capital / stop_distance)
```

- 변동성 군집에 따라 조정:
  - **high vol regime**: `contracts * 0.5`
  - **low vol regime**: `contracts * 1.0`
- 연속 손실 3회 시 다음 6봉 진입 금지 (cooldown)

---

## 6. Risk Management

| 항목 | 규칙 |
|------|------|
| 일일 손실 한도 | 계좌의 **-2%** 도달 시 당일 거래 중단 |
| 주간 MDD | **-4%** 도달 시 주간 거래 중단 |
| 최대 노출 | 동시 1 direction, 1 position |
| 슬리피지 가정 | 1 tick/side 기준, 실전 검증 시 2 tick/side 테스트 |
| 수수료 | 브로커 실제 수수료 반영 |

---

## 7. ML 적용 방식

기존 KDE는 1차 신호가 아닌 **필터/스코어**로 사용합니다.

| 단계 | 역할 |
|------|------|
| **Regime classifier** | LightGBM으로 trend/volatility regime 예측 |
| **Entry scorer** | XGBoost가 기술적 조건 + KDE tail + 거시 피처로 진입 확률 산출 |
| **Position sizer** | 예상 R-multiple에 비례해 size 조정 |
| **Label** | 단순 승/패가 아닌 **R-multiple** 사용: `net_pnl / stop_distance` |

### 주요 피처

- OHLC 기반: EMA20/60, ADX, ATR, BB width, RSI, MACD, session time
- KDE 기반: 현재 return의 regime별 tail 확률(z-score)
- 거시/수급: KOSPI200 spot, basis, VIX, USD/KRW, 거래량/미결
- 시차: 모든 피처는 t-1 lag

---

## 8. 예상 성능

| 지표 | 목표 | 예상 범위 |
|------|------|-----------|
| Sharpe Ratio | ≥ 2.0 | 1.8 ~ 2.6 |
| 승률 | 55~70% | 58~65% |
| Profit Factor | ≥ 1.8 | 1.8 ~ 2.4 |
| Calmar Ratio | ≥ 1.5 | 1.5 ~ 2.5 |
| 연간 거래 수 | ≥ 500건 | 600~1,000건 |
| MDD / 연수익 | 낮음 | < 30% |

- 추세 pullback 전략은 KOSPI200 선물에서 상승/하락 모두 방향성 edge가 있어 승률/RR 동시 개선이 가능합니다.
- 변동성 기반 사이징과 강제 청산 rule이 MDD를 억제합니다.

---

## 9. 검증 로드맵

1. **Baseline 구현**: 위 rule-based 알고리즘으로 2019-2026 backtest
2. **Cost sensitivity**: 1 tick / 2 tick / 3 tick slippage 별 Sharpe/PF
3. **Regime별 분해**: bull/bear/neutral/volatile 구간별 PnL 확인
4. **ML enhancement**: LightGBM regime + XGB entry scorer 적용
5. **Walk-forward + paper trading**: 2027 데이터 확보 후 OOS 검증

---

## 10. 요약

이 설계는 기존 KDE 중심 mean-reversion 접근에서 벗어나, **추세 방향의 pullback entry + 엄격한 ATR 기반 exit + regime 기반 position sizing**으로 전환합니다. 이를 통해 목표하는 Sharpe, PF, Calmar, 승률, MDD 기준을 달성할 가능성을 높이며, ML은 보조 필터/스코어로만 활용하여 전략의 해석 가능성을 유지합니다.
