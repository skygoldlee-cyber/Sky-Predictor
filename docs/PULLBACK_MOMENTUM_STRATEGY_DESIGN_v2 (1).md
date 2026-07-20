# Pullback Momentum Strategy Design (v2)

> 기존 KDE mean-reversion pipeline을 대체하는, KP200 선물 5분봉 기반 알고리즘 설계.
> **성능 목표는 가설이며, baseline backtest 실측 전까지 성능표는 채우지 않는다.**

## 최우선 목표: 복리 수익률(CAGR), 통제 대상: 드로다운(MAR)

이 시스템의 최적화 타깃은 **복리로 계좌에 실제 남는 수익률(CAGR)** 이며, 그 CAGR을 만드는 제약이 **드로다운**이다. 이유:

- **드로다운 회복 비대칭**: −20%는 +25%, −50%는 +100%를 벌어야 본전. 깊은 DD 한 번이 복리를 파괴한다. 낮은 평균수익이 아니라 깊은 DD가 장기 수익률의 최대 적.
- **변동성 드래그**: 기하평균 ≈ 산술평균 − 분산/2. 평균수익이 같아도 변동성이 크면 복리 수익이 깎인다 → vol targeting은 Sharpe뿐 아니라 실현 CAGR을 올린다.
- **켈리 상한**: 양의 edge라도 사이즈를 최적점 이상 키우면 CAGR이 오히려 하락하고 결국 0으로 수렴. "수익률 극대화"가 사이즈 확대로 흐르는 것이 가장 흔한 파산 경로.

**1차 목표 (최대화):** CAGR, **MAR = CAGR / MaxDrawdown ≥ 0.5** (실전 관점 우수 구간).
**제약 (통제):** MaxDrawdown 상한을 사이징으로 관리. 사이즈 스윕으로 CAGR 정점(≈켈리 최적점)을 찾고, **정점의 1/4~1/2(fractional Kelly)만 사용**한다.
**부차 지표 (진단용, 목표 아님):** Sharpe, 승률(55~70%), PF(≥1.8), Calmar. Sharpe·Calmar는 CAGR/MAR을 만드는 과정에서 딸려오는 결과로 취급하며, 이 값들을 직접 겨냥해 튜닝하지 않는다.

> 주의: 아래 승률·PF는 pullback 설계상 자연히 충족 가능한 정합적 조합(승률 60% + R/R 1.5 → PF 2.25)이라 목표라기보다 "무너지지 않았는지 확인하는 하한선"으로 읽는다.

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
> 표는 **1차 지표(CAGR/MAR/MDD)** 와 **부차 지표(진단용)** 로 분리한다. 최적화·튜닝 판단은 1차 지표로만 내린다.

### 1차 지표 (최대화·통제 대상)

| 지표 | 방향 | 실측 (baseline) | 실측 (ML) |
|------|------|------|------|
| **CAGR** | 최대화 | **-2.12% ~ -4.51%** (음수) | _TBD_ |
| **MAR = CAGR/MDD** | ≥ 0.5 (높을수록) | **-0.12 ~ -0.27** | _TBD_ |
| **MaxDrawdown** | 사이징으로 통제 | **14% ~ 38%** (사이즈에 민감) | _TBD_ |
| **최적 사이즈(fractional Kelly)** | CAGR 정점의 1/4~1/2 | **존재하지 않음** (CAGR 음수) | _TBD_ |

### 부차 지표 (진단용, 직접 튜닝 대상 아님)

| 지표 | 하한 확인선 | 실측 (baseline) | 실측 (ML) |
|------|------|------|------|
| Sharpe | 참고 | **-2.5 ~ -8.2** | _TBD_ |
| 승률 | 55~70% | **38 ~ 44%** | _TBD_ |
| Profit Factor | ≥ 1.8 | **0.20 ~ 0.66** | _TBD_ |
| Calmar | 참고 | **-0.99 ~ -1.01** | _TBD_ |
| 연간 거래 수 | ≥ 500 | **~36 (signal) / ~63 (random p=0.0025)** | _TBD_ |

---

## 8.1 사이즈 스윕 결과 (CAGR/MDD 곡선)

Exit 조건: `stop=1.0 ATR, target=2.0 ATR, trailing=0, time_stop=32봉, long_only=True`.

### Random control (진입 p=0.0025)

| risk_per_trade | 거래 수 | 승률 | PF | Sharpe | CAGR | MDD | MAR |
|------|--------|------|----|--------|------|-----|-----|
| 0.1% | 440 | 38.2% | 0.47 | -5.29 | -6.14% | 42.14% | -0.15 |
| 0.2% | 432 | 42.4% | 0.57 | -3.59 | -4.51% | 37.76% | -0.12 |
| 0.4% | 442 | 41.2% | 0.40 | -6.44 | -12.16% | 62.09% | -0.20 |
| 0.8% | 456 | 39.3% | 0.29 | -9.48 | -39.24% | 95.82% | -0.41 |
| 1.5% | 444 | 43.5% | 0.41 | -6.36 | -100% | 101.54% | -0.98 |

### Rule-based signal (동일 exit)

| risk_per_trade | 거래 수 | 승률 | PF | Sharpe | CAGR | MDD | MAR |
|------|--------|------|----|--------|------|-----|-----|
| 0.1% | 148 | 39.9% | 0.29 | -6.39 | -2.12% | 14.05% | -0.15 |
| 0.2% | 148 | 39.9% | 0.27 | -6.85 | -2.43% | 15.73% | -0.15 |
| 0.4% | 148 | 39.9% | 0.22 | -8.20 | -5.39% | 31.10% | -0.17 |
| 0.8% | 148 | 39.9% | 0.21 | -8.18 | -11.26% | 54.45% | -0.21 |
| 1.5% | 148 | 39.9% | 0.19 | -7.73 | -20.20% | 75.97% | -0.27 |

### 결론

- **CAGR가 모든 사이즈에서 음수**. 즉, 현재 exit/sizing 조합은 기대값이 0보다 작음.
- 사이즈를 줄이면 손실 절대액은 줄지만, MAR은 개선되지 않음(MAR ≈ -0.12 ~ -0.27).
- **fractional Kelly 최적점이 존재하지 않음**. 켈리 관점에서 "edge가 없는 전략에 사이즈를 줄이는 것"은 손실 속도만 늦출 뿐.
- **entry 신호(rule-based)도 random control 대비 우위가 거의 없음**. 먼저 exit/sizing이 random에서도 생존 가능해야 의미 있는 entry 실험 가능.

## 9. 검증 로드맵

1. **Baseline 구현**: rule-based로 2019–2026 backtest (t-1 확정봉 신호 엄수)
2. **랜덤 대조군**: pullback 신호를 랜덤 진입으로 치환. edge가 **신호**에서 오는지 **exit/사이징**에서 오는지 분리. (성능 대부분이 ATR exit에서 나오는 경우가 흔함)
3. **사이즈 스윕 → CAGR/MDD 곡선 (최우선 실험)**: risk_per_trade를 0.2%·0.4%·0.8%·1.5%…로 스윕해 **CAGR vs MaxDrawdown 곡선**을 그린다. CAGR 정점(≈켈리 최적점)을 찾고, 실전 사이즈는 그 정점의 **1/4~1/2**로 확정. 정점을 넘어서면 CAGR이 하락하므로 절대 초과 금지.
4. **Cost sensitivity**: 1/2/3 tick slippage별 CAGR/MAR (부차로 PF도 기록)
5. **파라미터 민감도**: EMA(20/60), RSI 밴드, ATR 배수(1.0/1.5/0.75), time stop 봉수에 대한 **MAR surface**. 소폭 변경에 MAR 붕괴 시 과최적화 판정
6. **Regime별 분해**: bull/bear/neutral/volatile PnL·DD 기여 확인
7. **ML enhancement**: LightGBM regime + XGB entry scorer (평가도 CAGR/MAR 기준)
8. **Walk-forward + paper trading**: 2027 데이터 확보 후 OOS

---

## 10. 요약

KDE 중심 mean-reversion에서 **추세 방향 pullback entry + ATR 기반 exit + regime 사이징**으로 전환. v1 대비 수정점:

1. **성능표 사전 기입 삭제** → baseline 실측 후 기록
2. **regime/entry 역할 분리** (느린 프레임 vs 빠른 프레임, 이중 필터 제거)
3. **sizing vs 일일 한도 정합** (트레이드당 2%→0.4%)
4. **trailing 폭 축소** (1.5→0.75 ATR)
5. **로드맵에 랜덤 대조군 + 파라미터 민감도 추가**, look-ahead 원칙을 rule-based에도 명시
6. **목표 지표를 Sharpe 중심 → CAGR/MAR 중심으로 재정의**: 최대화 대상은 복리 수익률(CAGR), 통제 대상은 드로다운. Sharpe·Calmar·승률·PF는 부차 진단 지표로 강등. 사이즈 스윕으로 CAGR 정점을 찾아 fractional Kelly로 운용하는 것을 최우선 실험으로 배치.

ML은 보조 필터/스코어로만 활용해 해석 가능성을 유지한다. **최적화·튜닝의 모든 의사결정은 1차 지표(CAGR/MAR)로만 내린다.**
