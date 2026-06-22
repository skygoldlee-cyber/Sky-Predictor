# 숏-전용 피봇 + 하락장 필터 파라미터 재최적화

## 개요
- **목표**: 숏-전용 피봇 반전에 맞춘 하락장 필터와 피봇 파라미터를 동시에 최적화
- **Train 기간**: 2025-06-25 ~ 2025-12-31
- **Test 기간 (OOS)**: 2026-01-01 ~ 2026-06-19
- **추가 OOS 기간**: 2025-10-01 ~ 2026-01-31 (train과 일부 겹침)

- **전체 거래일**: 240일
- **Train 거래일**: 128일
- **Test 거래일**: 112일

## 하락장 필터 정의

| 필터 | 조건 | 전체 일수 | train | test |
|---|---|---:|---:|---:|
| MA20_down | CLOSE < MA20 and return < 0 | 28 | 15 | 13 |
| ADX_bear | ADX > 25 and CLOSE < MA20 | 19 | 8 | 11 |
| consecutive_2down | today and yesterday returns < 0 | 29 | 13 | 16 |
| volatility_exp | ATR14 > 1.2 * ATR20_MA and return < 0 | 30 | 10 | 20 |
| MA20_or_strong | CLOSE < MA20 or return < -1% | 73 | 37 | 36 |
| MA20_down_and_ADX_bear | MA20_down AND ADX_bear | 11 | 4 | 7 |
| MA20_down_and_2down | MA20_down AND consecutive_2down | 16 | 8 | 8 |
| MA20_down_or_ADX_bear | MA20_down OR ADX_bear | 36 | 19 | 17 |
| any2_of_3 | at least 2 of MA20_down/ADX_bear/consecutive_2down | 22 | 11 | 11 |
| strong_or_MA20_not_vol | (MA20_down or strong_down) AND NOT volatility_exp | 51 | 29 | 22 |

## 필터별 최적 파라미터 (train 기준)

### MA20_down

**Train 최고 Sharpe**
- params={'base_pct': 0.5, 'base_multiplier': 2.5, 'atr_weight': 0.3, 'confirmation_bars': 1, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=  11 | 승률= 45.45% | PnL=     -963,078 | Sharpe= -1.413 | MaxDD=   -2,140,471
- test:  거래=  12 | 승률= 66.67% | PnL=   19,666,146 | Sharpe=  7.804 | MaxDD=   -3,719,107
- request OOS:  거래=   6 | 승률= 50.00% | PnL=      884,399 | Sharpe=  1.951 | MaxDD=   -2,140,471

**Train 최고 PnL**
- params={'base_pct': 0.5, 'base_multiplier': 2.5, 'atr_weight': 0.3, 'confirmation_bars': 1, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=  11 | 승률= 45.45% | PnL=     -963,078 | Sharpe= -1.413 | MaxDD=   -2,140,471
- test:  거래=  12 | 승률= 66.67% | PnL=   19,666,146 | Sharpe=  7.804 | MaxDD=   -3,719,107
- request OOS:  거래=   6 | 승률= 50.00% | PnL=      884,399 | Sharpe=  1.951 | MaxDD=   -2,140,471

### ADX_bear

**Train 최고 Sharpe**
- params={'base_pct': 0.5, 'base_multiplier': 2.5, 'atr_weight': 0.3, 'confirmation_bars': 1, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=   6 | 승률=100.00% | PnL=    4,625,831 | Sharpe= 22.186 | MaxDD=            0
- test:  거래=  10 | 승률= 70.00% | PnL=   16,075,301 | Sharpe=  6.261 | MaxDD=   -4,169,731
- request OOS:  거래=   6 | 승률=100.00% | PnL=    4,625,831 | Sharpe= 22.186 | MaxDD=            0

**Train 최고 PnL**
- params={'base_pct': 0.5, 'base_multiplier': 2.5, 'atr_weight': 0.3, 'confirmation_bars': 1, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=   6 | 승률=100.00% | PnL=    4,625,831 | Sharpe= 22.186 | MaxDD=            0
- test:  거래=  10 | 승률= 70.00% | PnL=   16,075,301 | Sharpe=  6.261 | MaxDD=   -4,169,731
- request OOS:  거래=   6 | 승률=100.00% | PnL=    4,625,831 | Sharpe= 22.186 | MaxDD=            0

### consecutive_2down

**Train 최고 Sharpe**
- params={'base_pct': 0.5, 'base_multiplier': 1.5, 'atr_weight': 0.3, 'confirmation_bars': 2, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=  10 | 승률= 60.00% | PnL=      737,997 | Sharpe=  1.104 | MaxDD=   -1,685,204
- test:  거래=  19 | 승률= 52.63% | PnL=   22,576,635 | Sharpe=  5.331 | MaxDD=   -6,570,026
- request OOS:  거래=   4 | 승률= 50.00% | PnL=      916,749 | Sharpe=  2.598 | MaxDD=   -1,685,204

**Train 최고 PnL**
- params={'base_pct': 0.5, 'base_multiplier': 1.5, 'atr_weight': 0.3, 'confirmation_bars': 2, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=  10 | 승률= 60.00% | PnL=      737,997 | Sharpe=  1.104 | MaxDD=   -1,685,204
- test:  거래=  19 | 승률= 52.63% | PnL=   22,576,635 | Sharpe=  5.331 | MaxDD=   -6,570,026
- request OOS:  거래=   4 | 승률= 50.00% | PnL=      916,749 | Sharpe=  2.598 | MaxDD=   -1,685,204

### volatility_exp

**Train 최고 Sharpe**
- params={'base_pct': 0.5, 'base_multiplier': 1.5, 'atr_weight': 0.3, 'confirmation_bars': 1, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=   9 | 승률= 55.56% | PnL=   -2,574,625 | Sharpe= -2.205 | MaxDD=   -7,232,834
- test:  거래=  25 | 승률= 36.00% | PnL=   -1,887,061 | Sharpe= -0.447 | MaxDD=  -13,753,829
- request OOS:  거래=   9 | 승률= 55.56% | PnL=   -2,574,625 | Sharpe= -2.205 | MaxDD=   -7,232,834

**Train 최고 PnL**
- params={'base_pct': 0.5, 'base_multiplier': 2.0, 'atr_weight': 0.3, 'confirmation_bars': 2, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=   7 | 승률= 42.86% | PnL=   -2,387,854 | Sharpe= -2.298 | MaxDD=   -7,012,444
- test:  거래=  23 | 승률= 30.43% | PnL=    9,442,707 | Sharpe=  2.144 | MaxDD=  -15,162,006
- request OOS:  거래=   7 | 승률= 42.86% | PnL=   -2,387,854 | Sharpe= -2.298 | MaxDD=   -7,012,444

### MA20_or_strong

**Train 최고 Sharpe**
- params={'base_pct': 0.5, 'base_multiplier': 2.5, 'atr_weight': 0.3, 'confirmation_bars': 1, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=  27 | 승률= 48.15% | PnL=   -4,634,047 | Sharpe= -2.089 | MaxDD=   -8,525,287
- test:  거래=  32 | 승률= 62.50% | PnL=   44,642,498 | Sharpe=  6.153 | MaxDD=   -5,977,719
- request OOS:  거래=  19 | 승률= 57.89% | PnL=   -2,417,324 | Sharpe= -1.364 | MaxDD=   -7,370,313

**Train 최고 PnL**
- params={'base_pct': 0.5, 'base_multiplier': 2.5, 'atr_weight': 0.3, 'confirmation_bars': 1, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=  27 | 승률= 48.15% | PnL=   -4,634,047 | Sharpe= -2.089 | MaxDD=   -8,525,287
- test:  거래=  32 | 승률= 62.50% | PnL=   44,642,498 | Sharpe=  6.153 | MaxDD=   -5,977,719
- request OOS:  거래=  19 | 승률= 57.89% | PnL=   -2,417,324 | Sharpe= -1.364 | MaxDD=   -7,370,313

### MA20_down_and_ADX_bear

**Train 최고 Sharpe**
- params={'base_pct': 0.5, 'base_multiplier': 2.5, 'atr_weight': 0.3, 'confirmation_bars': 1, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=   3 | 승률=100.00% | PnL=    3,024,870 | Sharpe= 21.445 | MaxDD=            0
- test:  거래=   7 | 승률= 71.43% | PnL=   15,296,936 | Sharpe=  8.607 | MaxDD=   -3,360,521
- request OOS:  거래=   3 | 승률=100.00% | PnL=    3,024,870 | Sharpe= 21.445 | MaxDD=            0

**Train 최고 PnL**
- params={'base_pct': 0.5, 'base_multiplier': 2.5, 'atr_weight': 0.3, 'confirmation_bars': 1, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=   3 | 승률=100.00% | PnL=    3,024,870 | Sharpe= 21.445 | MaxDD=            0
- test:  거래=   7 | 승률= 71.43% | PnL=   15,296,936 | Sharpe=  8.607 | MaxDD=   -3,360,521
- request OOS:  거래=   3 | 승률=100.00% | PnL=    3,024,870 | Sharpe= 21.445 | MaxDD=            0

### MA20_down_and_2down

**Train 최고 Sharpe**
- params={'base_pct': 0.5, 'base_multiplier': 2.0, 'atr_weight': 0.3, 'confirmation_bars': 2, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 10}
- train: 거래=   7 | 승률= 42.86% | PnL=   -1,655,304 | Sharpe= -3.424 | MaxDD=   -1,822,683
- test:  거래=  14 | 승률= 64.29% | PnL=   11,019,398 | Sharpe=  6.000 | MaxDD=   -7,596,312
- request OOS:  거래=   3 | 승률= 33.33% | PnL=     -315,369 | Sharpe= -1.023 | MaxDD=   -1,822,683

**Train 최고 PnL**
- params={'base_pct': 0.5, 'base_multiplier': 2.0, 'atr_weight': 0.3, 'confirmation_bars': 2, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 10}
- train: 거래=   7 | 승률= 42.86% | PnL=   -1,655,304 | Sharpe= -3.424 | MaxDD=   -1,822,683
- test:  거래=  14 | 승률= 64.29% | PnL=   11,019,398 | Sharpe=  6.000 | MaxDD=   -7,596,312
- request OOS:  거래=   3 | 승률= 33.33% | PnL=     -315,369 | Sharpe= -1.023 | MaxDD=   -1,822,683

### MA20_down_or_ADX_bear

**Train 최고 Sharpe**
- params={'base_pct': 0.5, 'base_multiplier': 2.5, 'atr_weight': 0.3, 'confirmation_bars': 1, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=  14 | 승률= 57.14% | PnL=      637,884 | Sharpe=  0.799 | MaxDD=   -2,140,471
- test:  거래=  15 | 승률= 66.67% | PnL=   20,444,512 | Sharpe=  6.320 | MaxDD=   -4,169,731
- request OOS:  거래=   9 | 승률= 66.67% | PnL=    2,485,361 | Sharpe=  4.510 | MaxDD=   -2,140,471

**Train 최고 PnL**
- params={'base_pct': 0.5, 'base_multiplier': 2.5, 'atr_weight': 0.3, 'confirmation_bars': 1, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=  14 | 승률= 57.14% | PnL=      637,884 | Sharpe=  0.799 | MaxDD=   -2,140,471
- test:  거래=  15 | 승률= 66.67% | PnL=   20,444,512 | Sharpe=  6.320 | MaxDD=   -4,169,731
- request OOS:  거래=   9 | 승률= 66.67% | PnL=    2,485,361 | Sharpe=  4.510 | MaxDD=   -2,140,471

### any2_of_3

**Train 최고 Sharpe**
- params={'base_pct': 0.5, 'base_multiplier': 2.5, 'atr_weight': 0.3, 'confirmation_bars': 1, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=   9 | 승률= 55.56% | PnL=     -250,231 | Sharpe= -0.407 | MaxDD=   -1,972,661
- test:  거래=  11 | 승률= 72.73% | PnL=   20,213,836 | Sharpe=  8.749 | MaxDD=   -3,719,107
- request OOS:  거래=   5 | 승률= 60.00% | PnL=    1,052,209 | Sharpe=  2.513 | MaxDD=   -1,972,661

**Train 최고 PnL**
- params={'base_pct': 0.5, 'base_multiplier': 2.5, 'atr_weight': 0.3, 'confirmation_bars': 1, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=   9 | 승률= 55.56% | PnL=     -250,231 | Sharpe= -0.407 | MaxDD=   -1,972,661
- test:  거래=  11 | 승률= 72.73% | PnL=   20,213,836 | Sharpe=  8.749 | MaxDD=   -3,719,107
- request OOS:  거래=   5 | 승률= 60.00% | PnL=    1,052,209 | Sharpe=  2.513 | MaxDD=   -1,972,661

### strong_or_MA20_not_vol

**Train 최고 Sharpe**
- params={'base_pct': 0.5, 'base_multiplier': 2.5, 'atr_weight': 0.3, 'confirmation_bars': 1, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=  19 | 승률= 42.11% | PnL=   -2,177,581 | Sharpe= -2.535 | MaxDD=   -3,392,335
- test:  거래=  17 | 승률= 70.59% | PnL=   26,250,622 | Sharpe=  6.782 | MaxDD=   -5,960,633
- request OOS:  거래=  11 | 승률= 54.55% | PnL=       39,143 | Sharpe=  0.068 | MaxDD=   -3,571,944

**Train 최고 PnL**
- params={'base_pct': 0.5, 'base_multiplier': 2.5, 'atr_weight': 0.3, 'confirmation_bars': 1, 'min_wave_pct': 0.3, 'min_pivot_interval_bars': 20}
- train: 거래=  19 | 승률= 42.11% | PnL=   -2,177,581 | Sharpe= -2.535 | MaxDD=   -3,392,335
- test:  거래=  17 | 승률= 70.59% | PnL=   26,250,622 | Sharpe=  6.782 | MaxDD=   -5,960,633
- request OOS:  거래=  11 | 승률= 54.55% | PnL=       39,143 | Sharpe=  0.068 | MaxDD=   -3,571,944

## 종합 비교 (test 기준)

| 필터 | 최적기준 | 거래 | 승률 | PnL | Sharpe | MaxDD |
|---|---|---:|---:|---:|---:|---:|
| MA20_down | train Sharpe | 12 | 66.67% | 19,666,146 | 7.804 | -3,719,107 |
| ADX_bear | train Sharpe | 10 | 70.00% | 16,075,301 | 6.261 | -4,169,731 |
| consecutive_2down | train Sharpe | 19 | 52.63% | 22,576,635 | 5.331 | -6,570,026 |
| volatility_exp | train Sharpe | 25 | 36.00% | -1,887,061 | -0.447 | -13,753,829 |
| MA20_or_strong | train Sharpe | 32 | 62.50% | 44,642,498 | 6.153 | -5,977,719 |
| MA20_down_and_ADX_bear | train Sharpe | 7 | 71.43% | 15,296,936 | 8.607 | -3,360,521 |
| MA20_down_and_2down | train Sharpe | 14 | 64.29% | 11,019,398 | 6.000 | -7,596,312 |
| MA20_down_or_ADX_bear | train Sharpe | 15 | 66.67% | 20,444,512 | 6.320 | -4,169,731 |
| any2_of_3 | train Sharpe | 11 | 72.73% | 20,213,836 | 8.749 | -3,719,107 |
| strong_or_MA20_not_vol | train Sharpe | 17 | 70.59% | 26,250,622 | 6.782 | -5,960,633 |
| 기준 숏-전용(무조건) | default | 112 | 46.43% | 60,653,846 | 2.434 | -16,857,321 |
| 롱-또는-플랫 | default | 81 | 58.02% | 48,125,318 | 1.482 | -31,050,193 |

## 해석
- **train Sharpe 기준**으로 고른 파라미터가 test에서도 우수한지 확인.
- **test 기준**으로 보는 것이 가장 신뢰할 만한 OOS 평가.
- 필터별 trade 빈도가 적으면(예: ADX_bear 13일) 통계적 검증력이 낮음.
- **요청 OOS 기간(2025-10~2026-01)**은 train과 겹치므로 참고용으로만 사용.