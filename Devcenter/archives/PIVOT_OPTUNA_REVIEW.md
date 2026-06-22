# PIVOT_OPTUNA_REVIEW.md

**대상**: `Devcenter/48. 피봇탐색_성능검증.py`, `indicators/hybrid_adaptive_pivot.py`
**참조 문서**: `OPTUNA_OPTIMIZATION_GUIDE.md`
**수정 구현**: `Devcenter/pivot_optuna_v2.py`

---

## 0. 핵심 결론

가이드 문서는 `optuna 70%` vs `전체 백테스트 43.57%(순손실 -141)` 격차를 **"과적합"** 으로
진단했다. 그러나 실제 코드를 보면 격차의 1차 원인은 과적합이 아니라 **목적함수 설계 결함**이다.

- 최적화 목표가 **승률(win rate)** 이며, 이는 영구 반전(reversal) 시스템에서 **수익성과 무관**하다.
- **거래비용·슬리피지·계약승수**가 전부 빠져 있다.
- 청크 분할은 교차검증이 아니라 **in-sample 평균**이다.

이 세 가지를 고치기 전에는 Walk-Forward·다목적·Pruner를 붙여도 *잘못된 목표를 더 정교하게
최적화*할 뿐이다. 아래에서 심각도순으로 정리한다.

---

## 1. 치명적 — 격차의 직접 원인

### [C-1] 목적함수가 '승률'이다 — 수익성과 디커플링
- **위치**: `optuna_objective` L1426, L1442 (`win_rate = backtest_result['overall_win_rate']`)
- `run_backtest`는 한 피봇에서 다음 피봇까지 보유하는 **항상 시장에 있는 반전 시스템**이라
  거래마다 보유시간·손익 크기가 제각각이다. 승률 70%여도 *작은 익절 다수 + 드문 큰 손절*이면
  기대값이 음수가 된다. `total_profit: -141`이 정확히 그 증상이다.
- Optuna는 정직하게 "승률을 잘못 끌어올리는 파라미터"를 찾았다. **잘못된 것을 최적화**한 결과이며,
  과적합은 부차적이다.

### [C-2] 거래비용·슬리피지·계약승수 전부 누락
- **위치**: `run_backtest` L354, L385 (`profit = exit_price - entry_price`)
- 순수 지수 포인트 차이만 계산. 반전 시스템은 매 피봇마다 청산+신규(2체결)이고 수백 회 거래한다.
  수수료·호가스프레드·슬리피지를 반영하면 실거래 손익은 -141보다 **훨씬 악화**된다.
- KOSPI200 선물 **승수(25만 원/pt)** 미반영 → 원화/R-multiple 손익 산정 불가.
- 비용을 넣으면 최적화기가 긁어모으던 "미세 익절"이 자동 소거된다.

### [C-3] 청크 백테스트는 교차검증이 아니다 (in-sample 평균)
- **위치**: `optuna_objective` L1395~1436
- 학습 데이터를 *연속 5등분*해 평균낸 것뿐이다. **hold-out fold가 없어 과적합 방지 효과 0**.
  가이드의 "장점: 과적합 방지"는 사실과 반대.
- 부작용:
  - 청크마다 `chunk_df.copy()` 후 ATR/ADX/SuperTrend 재계산 → 각 청크 앞 10~14봉이 워밍업 NaN
    → P5/P10 필터가 `pd.notna()` 가드로 **조용히 스킵**(L257, L264). 청크별 필터 강도가 달라짐.
  - 청크마다 `HybridAdaptivePivot`를 새로 만들어 `direction=0`부터 시작 → 경계 pending 피봇 폐기.

---

## 2. 구조적 결함

### [S-4] 피봇 가격을 '확정봉 종가'로 기록 — 실제 극점 폐기
- **위치**: `run_pivot_detection` L272~273 (`'timestamp': idx, 'price': row['CLOSE']`)
- 검출기 state는 실제 극점 `last_high`/`last_low`(+`*_idx`)를 노출하는데 이를 무시하고
  확정봉 종가를 피봇 가격으로 저장한다. 확정봉은 `thr`만큼 되돌린 뒤이므로 기록값이 실제 고점보다 낮다.
- 결과: P1 `wave_size_pct`(L245)가 *확정봉 종가 기준*으로 계산돼 필터가 측정하는 거리 자체가 무의미.

### [S-5] 전역 가변 필터 파라미터
- **위치**: `run_pivot_detection` L217, `optuna_objective` L1413 (`global _min_wave_pct ...`)
- 코드에도 "임시 해결책"이라 명시. `optuna_objective`↔`run_pivot_detection` 숨은 결합이 생기고,
  **`n_jobs>1` 병렬 최적화 시 trial 간 값이 오염**된다(병렬화 불가 → 최적화 느림).

### [S-6] Optuna 설정이 기본값 그대로
- **위치**: `run_optuna_optimization` L1471 (`optuna.create_study(direction='maximize')`)
- seed 없음(재현 불가), pruner 없음, multivariate sampler 없음.
- 약 **12개 하이퍼파라미터를 50 trial로 탐색** → TPE 수렴 표본 절대 부족 → "best"는 상당 부분 운.
  재실행하면 best가 출렁인다(이것이 "70%"의 실체).
- 페널티(L1430~1433)도 승률에 0.7/0.9를 곱하는 **불연속·임의 방식**이라 목적 지형을 왜곡.

### [S-7] 백테스트가 O(N×P)
- **위치**: `run_backtest` L341 (`pivots[pivots['timestamp'] == idx]` 를 per-bar 루프 내 호출)
- 매 봉마다 전체 피봇 DataFrame 스캔. 240일 분봉 × 5청크 × 50 trial이면 비현실적으로 느림
  → trial 수를 못 늘림 → [S-6] 악화.

### [S-8] 데이터 로더의 `days*500` 휴리스틱
- **위치**: `load_data_from_duckdb` L154 (`limit_count = days * 500`)
- "하루 최대 500봉 가정"은 야간세션 유무로 틀어져 "30일"이 실제 며칠인지 보장 안 됨.

### [S-9] 모델 선택용 검증셋 부재 / 경계 강제청산 편향
- **위치**: main L615~672, `run_backtest` L382~397
- train(75%)에서만 파라미터를 고르고 test는 1회 리포트일 뿐(선택 기준이 test에 반영 안 됨).
- 데이터/청크 끝의 강제청산 거래(L382~397)는 경계 위치에 따라 손익이 달라져 청크별 승률을 흔든다.
- `is_win = profit > 0`(L404)은 break-even을 패배로 카운트.

> **참고(오해 방지)**: 지표(ATR/ADX/SuperTrend)는 `shift(1)`·trailing rolling 기반 **인과적**이라
> look-ahead가 없다. 진입도 확정봉 종가라 인과적으로 valid하다(단 같은 봉 종가 진입은 1봉 낙관 편향이라
> '다음봉 시가' 진입이 더 정확). 즉 **이 시스템의 문제는 look-ahead가 아니라 위 항목들**이다.

---

## 3. 개선 매핑 (→ `pivot_optuna_v2.py`)

| # | 문제 | 개선 | v2 구현 |
|---|------|------|---------|
| 1 | 승률 최적화 | 비용차감 후 위험조정 수익(Sharpe/Expectancy/PF) 최대화, MaxDD 산출 | `BacktestResult`, `make_objective(metric=...)` |
| 2 | 비용 누락 | 수수료+슬리피지+승수 모델 | `BacktestConfig.round_trip_cost_pts`, `backtest()` |
| 3 | in-sample 청크 | Purged/Embargo Walk-Forward, 구간 일관성 보상 `mean−λ·std` | `purged_walkforward_folds`, `make_objective` |
| 4 | 지표 재계산/상태 리셋 | 전체 1회 계산 후 슬라이스, 검출기 상태 연속 | `compute_indicators`, `detect_pivots` |
| 5 | 확정봉 종가 기록 | 실제 극점(`last_high/low`) 기록 + '다음봉 시가' 진입 | `detect_pivots`, `BacktestConfig.entry_on` |
| 6 | 전역변수 | `FilterConfig`/`BacktestConfig` 주입 → 병렬 가능 | dataclass 주입 |
| 7 | 기본 Optuna | seed+multivariate TPE+MedianPruner+제약(min_trades)+중요도 | `optimize`, `_constraints` |
| 8 | O(N×P) | 이벤트 드리븐 O(피봇수) | `backtest()` |
| 9 | row-count 로더 | 날짜 범위 쿼리 | `load_data_by_date` |

### 권장 운영 플로우
```python
import pivot_optuna_v2 as pv

# 1) 날짜 범위로 정확히 로드 (#9)
df = pv.load_data_by_date(DB, "futures_1min", start="2025-06-24", end="2026-06-18")

# 2) 시간순 단일 홀드아웃: 과거→최적화, 미래→최종 1회 평가
train, test = pv.time_split(df, train_frac=0.75)

# 3) 지표 1회 계산 (#4) — 누수 방지 위해 train/test 각각 계산
train_i, test_i = pv.compute_indicators(train), pv.compute_indicators(test)

bt = pv.BacktestConfig(multiplier=250_000, commission_pct_per_side=3e-5,
                       slippage_ticks_per_side=1.0, entry_on="next_open")

# 4) Walk-Forward + 위험조정 목적함수로 파라미터 선택 (#1,#3,#7)
res = pv.optimize(train_i, n_trials=300, seed=42, bt_cfg=bt,
                  n_splits=4, embargo_bars=30, min_total_trades=60,
                  metric="sharpe", robustness_lambda=0.5,
                  output_dir="data/backtest_results")

# 5) 미확정 파라미터를 test 에 단 한 번 적용 (#9 검증)
p = res["best_params"]
pcfg = pv.HybridAdaptivePivotConfig(base_pct=p["base_pct"],
        base_multiplier=p["base_multiplier"], atr_weight=p["atr_weight"],
        confirmation_bars=p["confirmation_bars"])
fcfg = pv.FilterConfig(min_wave_pct=p["min_wave_pct"],
        min_pivot_interval_bars=p["min_pivot_interval_bars"],
        st_distance_threshold=p["st_distance_threshold"],
        adx_hold_threshold=p["adx_hold_threshold"])
pivots = pv.detect_pivots(test_i, pcfg, fcfg)
print(pv.backtest(test_i, pivots, bt).as_dict())   # Sharpe/PF/MaxDD/PnL(원)
```

---

## 4. 검증

`pivot_optuna_v2.py`는 합성 분봉(3일)로 엔드투엔드 동작을 확인했다.
- `compute_indicators`: 워밍업 NaN이 **시리즈 시작부에만** 발생(ATR 13 / ADX 26 / ST 9), fold 경계 NaN 없음.
- `detect_pivots`: 실제 극점 가격/시각으로 피봇 기록.
- `backtest`: `n_trades / win_rate / total_pnl_pts / total_pnl_krw / expectancy / profit_factor / sharpe_daily / max_drawdown_krw` 산출.
- `purged_walkforward_folds`: embargo로 fold 앞단 트리밍 확인(`[270, 250, 250, 250]`).
- `optimize`: seed 고정 + multivariate TPE + MedianPruner + 거래수 제약 + 파라미터 중요도 정상 출력.

> 합성 사인파 데이터라 승률이 100%로 나오는 것은 정상(결정적 사이클). 핵심은 **비용·승수가 반영된
> Sharpe/PF/MaxDD/원화 손익이 계산된다**는 점이며, 실데이터에서 이 지표들이 의미를 갖는다.

---

## 5. 추가 권고 (선택)
- **다목적**: `metric="sharpe"` 단일 대신 NSGA-II로 `(Sharpe, −MaxDD)` 파레토 탐색.
- **확정봉 1봉 낙관 제거**: `entry_on="next_open"`을 기본 유지(이미 적용).
- **거래수 하한 동적화**: fold별 최소 거래수도 제약에 추가해 표본부족 fold의 Sharpe 노이즈 차단.
- **워밍업 처리**: `min_total_trades` 외에 fold당 `n_trades>=k` 미만이면 해당 fold score 제외.
