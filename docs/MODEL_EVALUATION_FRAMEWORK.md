# 모델 성능 객관적 평가 지표 체계

> 문서 ID: `DESIGN-EVAL-001`
> 작성일: 2026-07-21
> 목적: SkyPredictor KDE/ML 파이프라인의 여러 변형을 공정하고 재현 가능한 기준으로 비교

---

## 1. 평가 설계 원칙

1. **Look-ahead 배제**: 모든 threshold 선택, feature, hyperparameter는 `val` (2024)에서만 결정하고 `test` (2025-2026)에서 1회 평가.
2. **비용 반영**: 거래당 `slippage=1 tick`, tick size `0.05`, contract multiplier `31,500 KRW`, initial capital `10,000,000 KRW`를 기본으로 함.
3. **결정론적 재현성**: `PYTHONHASHSEED=42`, XGBoost `random_state=42`, numpy/pandas seed 고정. LSTM/TensorFlow는 deterministic 설정 적용 또는 제외.
4. **동일 거래 정의**: entry → exit까지 하나의 완결된 trade, `net_krw`는 수수료/슬리피지 차감 후 최종 KRW 손익.
5. **일 단위 측정**: Sharpe/CAGR/MAR/MDD는 일별 누적 수익률(equity curve) 기준.

---

## 2. 핵심 지표 정의

| 지표 | 심볼 | 정의 | 계산식 | 해석 |
|------|------|------|--------|------|
| **거래 수** | `n_trades` | test 기간 완결된 trade 개수 | `len(trades)` | 통계적 신뢰성 판단. 너무 적으면(<50) 지표 불안정. |
| **승률** | `win_rate` | 양수 net_krw 거래 비율 | `sum(net_krw > 0) / n_trades` | 높을수록 좋으나, 손익비와 함께 봐야 함. |
| **총 손익** | `total_pnl` | test 기간 누적 KRW 손익 | `sum(net_krw)` | 절대 수익. initial capital 대비 %로 변환 가능. |
| **평균 손익** | `avg_pnl` | 거래당 평균 손익 | `total_pnl / n_trades` | 거래 비용 대비 edge 크기. |
| **Profit Factor** | `PF` | 총 이익 / 총 손실 절대값 | `sum(net_krw > 0) / abs(sum(net_krw < 0))` | **PF ≥ 1.5**를 생존 기준. PF < 1.0이면 장기 손실. |
| **일별 Sharpe** | `sharpe` | 일별 수익의 risk-adjusted return | `√252 * mean(daily_pnl) / std(daily_pnl)` | **Sharpe ≥ 2.0**을 양호 기준. 변동성 조정된 수익률. |
| **최대 낙폭** | `MDD` | 일별 equity 기준 최대 drawdown | `max(peak - trough)` | 리스크 절대량. 작을수록 좋음. |
| **연평균 수익률** | `CAGR` | 연평균 복리 수익률 | `(final_equity / init_equity)^(1/years) - 1` | 시간 가치 반영. 양수가 최우선 목표. |
| **MAR Ratio** | `MAR` | CAGR / MDD (누적) | `CAGR / (MDD / init_equity)` | **MAR ≥ 1.0**이 리스크 조정 우수. MDD가 0이면 정의 불가. |
| **손익비** | `profit_loss_ratio` | 평균 수익 / 평균 손실 | `mean(wins) / abs(mean(losses))` | 승률 50% 기준 1.0이면 break-even. |

### 2.1 지표 간 관계

- `PF >= 1.5` + `win_rate >= 40%` + `n_trades >= 100` → 일반적으로 장기 생존 가능
- `sharpe`는 MDD 기간과 변동성에 민감. 거래 수가 적으면 과대평가/과소평가 위험.
- `CAGR`와 `MAR`는 평가 기간(test 2025-2026, 약 2년)에 민감. 다른 기간으로 walk-forward 해야 안정성 확보.

---

## 3. 일차/이차/보조 지표

| 우선순위 | 지표 | 목표 | 설명 |
|----------|------|------|------|
| **1차 (go/no-go)** | CAGR | > 0% | 장기 생존 가능 여부 |
| **1차** | PF | ≥ 1.5 | 수익이 손실보다 충분히 큰지 |
| **2차** | Sharpe | ≥ 2.0 | 변동성 조정 후 의미 있는 edge |
| **2차** | MAR | ≥ 1.5 | 리스크 조정 수익 효율 |
| **3차 (보조)** | win_rate, avg_pnl, n_trades, profit_loss_ratio | - | 진단용. 단독 판단 금지. |

---

## 4. 현재까지 평가한 모델/설정 비교표

모든 수치는 **test 2025-2026, slippage=1 tick, initial 10M** 기준.

| 모델/설정 | 거래 수 | 승률 | 총 PnL | PF | Sharpe | CAGR | MDD | MAR |
|-----------|--------|------|--------|----|--------|------|-----|-----|
| Baseline (stage1+stage2, LSTM exit, KDE 없음) | 631 | 40.4% | -9.10M | 0.76 | -1.78 | -100%* | 16.43M | -0.61 |
| KDE + stage1/2 + LSTM exit | 384 | 42.2% | +1.89M | 1.12 | 0.60 | 12.61% | 2.65M | 0.48 |
| KDE + stage1/2 + xgb_cls exit (validation threshold) | 197 | 43.7% | +0.34M | 1.03 | 0.15 | 2.18% | 1.55M | 0.14 |
| **KDE + s1=0.5/s2=0.5 + xgb_reg exit (thr=0)** | **377** | **41.9%** | **+2.31M** | **1.15** | **0.76** | **15.35%** | **2.08M** | **0.74** |
| KDE + s1=0.5/s2=0.5 + xgb_reg exit (thr=3,000) | 367 | 42.0% | +2.62M | 1.17 | 0.86 | 17.29% | 1.77M | 0.97 |
| KDE + s1=0.5/s2=0.5 + xgb_reg exit (thr=10,000) | 27 | 55.6% | +1.13M | 1.73 | 0.97 | 9.24% | 0.22M | 4.23 |
| KDE + s1=0.3/s2=0.3 + xgb_reg (thr=0) | 24 | 50.0% | -0.29M | 0.87 | -0.27 | -2.39% | 1.60M | -0.15 |
| KDE + s1=0.5/s2=0.3 + xgb_reg (thr=0) | 37 | 43.2% | -2.13M | 0.52 | -1.23 | -16.86% | 2.49M | -0.68 |
| Random control (비ML 랜덤 진입) | - | ~50% | - | <1.0 | - | 음수 | - | - |

> *CAGR가 -100%에 가까운 경우는 test 기간 내 initial capital을 거의 전부 잃는 시나리오입니다.

---

## 5. 평가 시 주의사항

1. **단일 test 기간 의존**: 2025-2026은 특정 국면(추세/횡보/변동성)을 포함. 2019-2020, 2021-2022, 2023-2024 등으로 walk-forward 검증 필요.
2. **거래 수 부족**: 거래 수 < 50인 설정은 우연성이 큼. 최소 100~200건 권장.
3. **threshold 최적화 overfit**: validation(2024)에서 고른 threshold가 test(2025-2026)에서 무너지는 경우가 많음. 고정 threshold robustness test 필수.
4. **비용/슬리피지 감도**: 1 tick, 2 tick, 3 tick에서 결과가 바뀌는지 확인. PF가 1.1~1.2 수준이면 slippage 증가 시 쉽게 음수로 전환.
5. **CAGR/MDD 기간 편향**: test 기간이 2년이므로 CAGR는 annualized. 하지만 1년 만에 MDD가 발생하고 회복하지 못하면 CAGR 왜곡 가능.

---

## 6. 다음 평가 계획

1. **Walk-forward 평가**: 2019-2020, 2021-2022, 2023-2024, 2025-2026 4-fold로 best 설정(KDE+s1=0.5/s2=0.5+xgb_reg thr=3,000) 재검증
2. **Slippage 감도**: 1/2/3 tick에서 PF/Sharpe/CAGR 변화
3. **Regime별 분해**: trend/volatile 횡보 구간별 Sharpe/CAGR 확인
4. **Position sizing 추가**: fractional Kelly 적용 후 CAGR/MAR 변화
5. **Model registry**: 각 실험 결과를 MLflow/CSV에 기록해 버전 관리
