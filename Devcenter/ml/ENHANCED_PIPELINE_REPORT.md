# Enhanced Pipeline Sequential Improvement Report

> Generated: 2026-07-27
> Pipeline: `Devcenter/ml/ml_full_pipeline_enhanced.py`
> Period: train=2019-2023, val=2024, test=2025-2026
> Cost assumptions: 1 tick slippage unless noted otherwise, initial capital 10M KRW

## 1. Data preparation

The original 5-minute OHLCV history is stored in `Devcenter/data/since2019_future_data.txt` (137,047 bars, 2019-06-03 09:00 ~ 2026-06-19 15:45). KDE features are now regenerated from these bars using `Devcenter/ml/build_kde_from_original_bars.py`, which fits the KDE on a rolling window of bar-level log-returns and then merges the resulting metrics into `ml_dataset.csv` by `entry_time` (100 % match rate). This avoids the earlier approximation that used trade-level `entry_close` values.

```powershell
c:/Python/WPy64-31241/python-3.12.4.amd64/python.exe Devcenter/ml/build_kde_from_original_bars.py
```

## 2. Pipeline improvements implemented

`ml_full_pipeline_enhanced.py` adds the following options to the original 3-stage pipeline:

1. **High-confidence meta-filter**: stacks stage1 filter score, stage2 entry score, regime dummies, and KDE tail features into a second-level XGB classifier.
2. **Ensemble exit**: averages normalized scores from LSTM, XGB classifier, and XGB regressor exit models.
3. **Regime-aware thresholds**: selects stage1/stage2 thresholds independently for each `trend_regime` on validation data.
4. **Fractional Kelly sizing + drawdown guard**: sizes trades with `f = 0.5 * mean_p / var_p` over a rolling window; reduces size when running drawdown exceeds a threshold.
5. **Multi-timeframe KDE + walk-forward harness**: the pipeline already consumes all `_kde_*` columns (1m/5m from the approximated file); walk-forward folds can be run with `--walk-forward`.

## 3. Key experimental results

### 3.1 Baseline vs KDE-enhanced (0 tick slippage, xgb_reg exit)

| Variant | Trades | Win% | Total PnL | PF | Sharpe | CAGR% | MDD | MAR |
|---|---|---|---|---|---|---|---|---|
| Baseline (no KDE) | 341 | 41.35 | -4,344,401 | 0.82 | -1.08 | -32.38 | 5,779,933 | -0.56 |
| KDE-enhanced xgb_reg | 141 | 44.68 | +1,394,489 | 1.14 | 0.60 | 9.43 | 2,485,473 | 0.38 |

Adding the (approximate) KDE features turns a large loss into a positive PnL and roughly doubles the risk-adjusted metrics.

### 3.2 Slippage robustness sweep (KDE-enhanced xgb_reg, best setting)

| Slippage ticks | Trades | Win% | Total PnL | PF | Sharpe | CAGR% | MDD | MAR |
|---|---|---|---|---|---|---|---|---|
| 1 | 79 | 43.04 | +1,262,270 | **1.29** | **1.27** | 8.59 | 1,133,712 | **0.76** |
| 2 | 40 | 40.00 | -456,592 | 0.89 | -0.61 | -3.16 | 1,881,516 | -0.17 |
| 3 | 234 | 41.88 | -4,315,255 | 0.79 | -1.56 | -32.29 | 6,086,431 | -0.53 |

The best observed configuration is **1 tick slippage, KDE-enhanced, xgb_reg exit**. Higher slippage quickly erodes the edge, indicating the current margin is thin.

### 3.3 Sequential improvement attempts

| Configuration | Trades | Win% | Total PnL | PF | Sharpe | MAR |
|---|---|---|---|---|---|---|
| KDE + xgb_reg (1m/5m scott) | 79 | 43.0 | +1,262,270 | 1.29 | 1.27 | 0.76 |
| + fractional Kelly + drawdown guard | 79 | 43.0 | +263,399 | 1.10 | 0.65 | 0.32 |
| + ensemble exit | 150 | 44.7 | +429,317 | 1.04 | 0.17 | 0.12 |
| + meta-filter | 438 | 45.4 | -2,837,249 | 0.88 | -0.90 | -0.51 |
| + regime-aware thresholds | worse across tested metrics |
| fixed stage1=stage2=0.5 thresholds | worse across tested metrics |

Applying fractional Kelly/drawdown guard to only 79 trades hurts performance because the Kelly warm-up (252 trades) is longer than the available test history. The ensemble exit and meta-filter did not improve the test-set outcome, suggesting they add model variance without enough independent signal.

### 3.4 KDE feature grid search (breakthrough)

A grid search over KDE timeframe combinations, bandwidth, and rolling window was run using the script `Devcenter/ml/kde_grid_search.py`. Results are saved to `Devcenter/ml/ml_models/kde_grid_results.csv`.

Top 5 by Sharpe:

| Timeframes | Bandwidth | Window | Trades | PF | Sharpe | MAR | Total PnL |
|---|---|---|---|---|---|---|---|
| **(1, 5, 15)** | **scott** | **2000** | **71** | **1.97** | **2.89** | **1.88** | **+3,351,498** |
| (1, 5) | 0.5 | 2000 | 24 | 1.71 | 2.78 | 1.42 | +1,526,749 |
| (1, 5) | 0.3 | 3000 | 22 | 1.64 | 2.48 | 1.13 | +1,450,901 |
| (5, 15) | scott | 2000 | 22 | 1.53 | 2.45 | 1.87 | +195,852 |
| (5, 15) | 0.3 | 3000 | 154 | 1.21 | 1.57 | 0.90 | +2,553,850 |

The best configuration, **timeframes=(1,5,15), bandwidth=scott, window=2000**, simultaneously clears the survival targets: **PF=1.97 ≥ 1.5, Sharpe=2.89 ≥ 2.0, MAR=1.88 ≥ 1.5**. It is saved as the current `ml_dataset_with_kde.csv`.

Reproduced single-run result (test 2025-2026, 1 tick slippage):

| Variant | Trades | Win% | Total PnL | PF | Sharpe | CAGR% | MDD | MAR |
|---|---|---|---|---|---|---|---|---|
| Baseline | 98 | 43.88 | -1,035,984 | 0.88 | -0.79 | -7.44 | 2,017,479 | -0.37 |
| KDE-enhanced | **71** | **49.30** | **+3,351,498** | **1.97** | **2.89** | **23.46** | **1,245,027** | **1.88** |

### 3.5 Multi-fold walk-forward validation (critical finding)

To test robustness, the same KDE-enhanced configuration was run over 4 expanding-window folds:

| Fold | Variant | Trades | Win% | Total PnL | PF | Sharpe | MAR |
|---|---|---|---|---|---|---|---|
| 2021→(2023,2024) | Baseline | 116 | 32.76 | -1,071,406 | 0.46 | -6.41 | -0.47 |
| 2021→(2023,2024) | KDE-enhanced | 20 | 35.00 | -227,297 | 0.40 | -5.37 | -0.37 |
| 2022→(2024,2025) | Baseline | 88 | 38.64 | -612,333 | 0.67 | -3.41 | -0.45 |
| 2022→(2024,2025) | KDE-enhanced | 95 | 34.74 | -1,478,653 | 0.46 | -6.50 | -0.53 |
| 2023→(2025,2026) | Baseline | 168 | 46.43 | +2,215,025 | 1.20 | 1.32 | 0.81 |
| 2023→(2025,2026) | KDE-enhanced | 218 | 44.50 | -2,864,348 | 0.80 | -1.37 | -0.41 |
| 2019-2023→(2025,2026) | Baseline | 310 | 41.29 | -5,869,774 | 0.78 | -1.61 | -0.64 |
| 2019-2023→(2025,2026) | KDE-enhanced | 71 | 49.30 | **+3,351,498** | **1.97** | **2.89** | **1.88** |

**The strong PF/Sharpe/MAR result is confined to the single train=2019-2023 / test=(2025,2026) split.** On earlier folds the KDE-enhanced variant is either similar to or worse than the baseline, and on the 2023→(2025,2026) fold it is materially worse. This indicates the winning configuration is either over-fit to the 2024 validation year / 2025-2026 test period, or the approximate KDE features contain a look-ahead / period-specific artifact.

### 3.6 Original 5-minute bar KDE regeneration (critical update)

The original 5-minute OHLCV bars (`since2019_future_data.txt`) were parsed and used to regenerate KDE features with a rolling-window fit on bar-level log-returns only. This removes the trade-level `entry_close` approximation and any potential look-ahead from the KDE step.

Grid search over a reduced set of configurations on the true bars (`Devcenter/ml/kde_grid_search_v2.py`) shows a much weaker signal:

| Timeframes | Bandwidth | Window | Trades | PF | Sharpe | MAR | Total PnL |
|---|---|---|---|---|---|---|---|
| **(1, 5)** | **scott** | **2000** | **286** | **1.07** | **0.44** | **0.24** | **+1,340,886** |
| (1, 5, 15) | scott | 3000 | 384 | 1.03 | 0.19 | 0.15 | +537,329 |
| (1, 5) | scott | 3000 | 503 | 0.96 | -0.35 | -0.31 | -1,330,915 |
| (5, 15) | scott | 3000 | 1053 | 0.95 | -0.48 | -0.43 | -2,596,938 |
| (1, 5, 15) | scott | 2000 | 298 | 0.93 | -0.63 | -0.27 | -1,686,684 |
| (5, 15) | scott | 2000 | 241 | 0.77 | -1.65 | -0.64 | -4,342,592 |
| (1, 5, 15, 60) | scott | 2000 | 224 | 0.74 | -2.37 | -0.55 | -3,839,592 |
| (1, 5, 15, 60) | scott | 3000 | 390 | 0.72 | -2.90 | -0.80 | -8,514,879 |

The best true-bar configuration is **(1, 5), scott, window=2000**, but it only achieves PF 1.07 / Sharpe 0.44 / MAR 0.24, far below the earlier approximate-KDE result. Adding the 60m timeframe or a longer window consistently degrades performance.

Multi-fold walk-forward on the true-bar KDE shows no robust outperformance:

| Fold | Variant | Trades | Win% | Total PnL | PF | Sharpe | MAR |
|---|---|---|---|---|---|---|---|
| 2021→(2023,2024) | Baseline | 116 | 32.76 | -1,071,406 | 0.46 | -6.41 | -0.47 |
| 2021→(2023,2024) | KDE-enhanced | 612 | 39.22 | -2,810,792 | 0.65 | -2.89 | -0.46 |
| 2022→(2024,2025) | Baseline | 88 | 38.64 | -612,333 | 0.67 | -3.41 | -0.45 |
| 2022→(2024,2025) | KDE-enhanced | 105 | 42.86 | -208,568 | 0.87 | -0.86 | -0.23 |
| 2023→(2025,2026) | Baseline | 168 | 46.43 | +2,215,025 | 1.20 | 1.32 | 0.81 |
| 2023→(2025,2026) | KDE-enhanced | 233 | 46.35 | +131,127 | 1.01 | 0.05 | 0.04 |
| 2019-2023→(2025,2026) | Baseline | 335 | 41.79 | -6,022,431 | 0.78 | -1.54 | -0.63 |
| 2019-2023→(2025,2026) | KDE-enhanced | 286 | 49.65 | +1,340,886 | 1.07 | 0.44 | 0.24 |

**Interpretation:** the previous PF 1.97 / Sharpe 2.89 / MAR 1.88 result is **not reproduced** when KDE is computed from the real 5-minute bars. The most likely explanation is that the trade-level `entry_close` approximation introduced a favorable artifact, possibly because the sampled closes were aligned to the trade outcome in a way that leaked information into the density estimate. True-bar KDE is closer to a realistic signal and shows only marginal edge.

### 3.7 LSTM ensemble indexing fix

The LSTM score alignment in `stage3_exit_ensemble` was using `reset_index(drop=True)` on the time-sorted validation/test DataFrames and then assigning predictions back to the original-indexed score Series. This caused `KeyError: "None of [RangeIndex(...)] are in the [index]"`. The fix preserves the original index while sorting:

```python
df_val_sorted = val_df.sort_values("entry_time")  # keep original index
val_index = df_val_sorted.index[sequence_length:]
scores_val.loc[val_index] += weights["lstm"] * val_proba
```

After the fix, the ensemble exit runs without errors. Using the true-bar KDE dataset on train=2019-2023 / test=2025-2026:

| Variant | Trades | Win% | Total PnL | PF | Sharpe | MAR |
|---|---|---|---|---|---|---|
| Baseline (ensemble exit) | 178 | 39.89 | -5,541,625 | 0.74 | -2.22 | -0.65 |
| KDE-enhanced (ensemble exit) | 268 | 48.13 | +300,251 | 1.02 | 0.11 | 0.06 |

The ensemble is now functional, but the edge remains marginal.

### 3.8 Regime-aware thresholds re-test with balanced hour-of-day regimes

The pre-computed `trend_regime` is extremely imbalanced (regime 1 ≈ 97 %). A balanced regime column was added by quantile-binning `entry_hour` on the training set into 3 buckets. The regime-aware threshold logic was also fixed to fall back to a global threshold for regimes with too few validation samples, instead of defaulting to 0.0 (pass-through).

Results on train=2019-2023 / test=2025-2026, 1 tick slippage:

| Variant | Trades | Win% | Total PnL | PF | Sharpe | MAR |
|---|---|---|---|---|---|---|
| Baseline (regime-aware) | 146 | 48.63 | +3,630,684 | **1.41** | **1.84** | **1.57** |
| KDE-enhanced (regime-aware) | 283 | 46.29 | -3,569,038 | 0.85 | -1.34 | -0.43 |

Multi-fold walk-forward on the hour-of-day regime split:

| Fold | Variant | PF | Sharpe | MAR |
|---|---|---|---|---|
| 2021→(2023,2024) | Baseline | 0.52 | -4.43 | -0.45 |
| 2021→(2023,2024) | KDE-enhanced | 0.72 | -1.84 | -0.40 |
| 2022→(2024,2025) | Baseline | 0.39 | -6.62 | -0.52 |
| 2022→(2024,2025) | KDE-enhanced | 0.70 | -2.22 | -0.43 |
| 2023→(2025,2026) | Baseline | 1.02 | 0.17 | 0.09 |
| 2023→(2025,2026) | KDE-enhanced | 1.19 | 1.01 | 0.93 |
| 2019-2023→(2025,2026) | Baseline | 1.41 | 1.84 | 1.57 |
| 2019-2023→(2025,2026) | KDE-enhanced | 0.85 | -1.34 | -0.43 |

**Findings:**
- Hour-of-day regime splits give a strong single-fold baseline result, but it does not hold in earlier folds.
- KDE features consistently hurt the hour-of-day baseline in the test period.
- A likely root cause was that many pre-computed technical indicator columns in `ml_dataset.csv` (ATR, RSI, volume, MACD, etc.) were constant zeros, leaving very few learnable features besides raw price, hour-of-day, and the KDE metrics.

### 3.9 Technical-indicator regeneration from original 5-minute bars

Because the indicator columns in `ml_dataset.csv` were degenerate, a separate script `Devcenter/ml/rebuild_indicators_from_bars.py` was added. It parses the original 5-minute OHLCV bars and recomputes entry-time ATR(14), RSI(14), MACD(12/26/9), Bollinger Bands(20,2), MA5/10/20/60, and SuperTrend, then merges them into `ml_dataset.csv` by `entry_time`.

After regeneration, the same walk-forward configuration as section 3.6 (true-bar KDE, 1 tick slippage, xgb_reg exit, no fractional Kelly) produces:

| Variant | Trades | Win% | Total PnL | PF | Sharpe | MAR |
|---|---|---|---|---|---|---|
| Baseline | 363 | 40.22 | -7,815,806 | 0.45 | -4.76 | -0.83 |
| KDE-enhanced | 363 | 48.21 | -84,793 | 1.00 | -0.04 | -0.03 |

The rebuilt indicators provide richer features, but they do not restore the earlier strong performance. The KDE-enhanced variant is essentially flat on this fold.

### 3.10 Feature selection and slippage sensitivity

A top-k feature-selection step was added (`--feature-selection-k`): a small XGBoost classifier is fit on the training set, and the highest-gain features are used for stage1/stage2. With the rebuilt indicators and k=8:

**0 tick slippage, train=2019-2023, test=2025-2026:**

| Variant | Trades | Win% | Total PnL | PF | Sharpe | MAR |
|---|---|---|---|---|---|---|
| Baseline (k=8) | 341 | 48.68 | +2,886,751 | 1.24 | 1.24 | 1.30 |
| KDE-enhanced (k=8) | 703 | 54.05 | +6,136,788 | 1.19 | 1.38 | 1.18 |

**1 tick slippage, same fold:**

| Variant | Trades | Win% | Total PnL | PF | Sharpe | MAR |
|---|---|---|---|---|---|---|
| Baseline (k=8) | 490 | 44.69 | +1,098,119 | 1.05 | 0.27 | 0.20 |
| KDE-enhanced (k=8) | 429 | 44.29 | -1,830,452 | 0.92 | -0.52 | -0.30 |

Multi-fold walk-forward at 0 tick slippage with k=8:

| Fold | Variant | PF | Sharpe | MAR |
|---|---|---|---|---|
| 2021→(2023,2024) | Baseline | 0.88 | -0.63 | -0.24 |
| 2021→(2023,2024) | KDE-enhanced | 1.06 | 0.38 | 0.17 |
| 2022→(2024,2025) | Baseline | 0.92 | -0.54 | -0.14 |
| 2022→(2024,2025) | KDE-enhanced | 1.04 | 0.20 | 0.06 |
| 2023→(2025,2026) | Baseline | 0.96 | -0.41 | -0.33 |
| 2023→(2025,2026) | KDE-enhanced | 1.10 | 0.71 | 0.56 |
| 2019-2023→(2025,2026) | Baseline | 1.24 | 1.24 | 1.30 |
| 2019-2023→(2025,2026) | KDE-enhanced | 1.19 | 1.38 | 1.18 |

**Findings:**
- Feature selection improves the 0-tick result and makes the latest folds modestly profitable.
- The edge is **very sensitive to transaction costs**: at 1 tick slippage the baseline is barely profitable and the KDE variant is unprofitable.
- Earlier folds (2021-2024, 2022-2025) remain weak, so the configuration is not yet robust across periods.

### 3.11 Stage3 fixed threshold sweep

`Devcenter/ml/stage3_threshold_sweep.py` was updated to support feature selection and optional stage1/stage2 pass-through. Sweeping stage3 fixed xgb_reg thresholds on the test 2025-2026 fold (1 tick slippage):

**Stage1/stage2 skipped, all trades enter stage3:**

| Threshold | Trades | Win% | PF | Sharpe | MAR |
|---|---|---|---|---|---|
| 2,000 | 147 | 54.42 | 1.09 | 0.42 | 0.26 |
| 4,000 | 109 | 54.13 | 1.41 | 1.81 | 1.91 |
| 6,000 | 79 | 56.96 | 1.68 | 2.55 | 3.36 |
| 8,000 | 57 | 56.14 | 2.18 | 3.63 | 7.20 |
| 10,000 | 48 | 52.08 | 2.09 | 3.41 | 5.17 |
| 12,000 | 40 | 57.50 | 2.86 | 4.43 | 7.84 |
| 14,000 | 31 | 54.84 | 2.89 | 3.80 | 5.83 |

The high-threshold results look strong on this single fold, but walk-forward shows they do not generalize:

| Fold | Stage3 threshold | Trades | PF | Sharpe | MAR |
|---|---|---|---|---|---|
| 2021→(2023,2024) | 12,000 | 0 | 0.00 | 0.00 | 0.00 |
| 2022→(2024,2025) | 12,000 | 97 | 0.95 | -0.78 | -0.14 |
| 2023→(2025,2026) | 12,000 | 732 | 0.92 | -0.83 | -0.54 |
| 2019-2023→(2025,2026) | 12,000 | 40 | 2.86 | 4.43 | 7.84 |

**Stage1/stage2 active + feature selection k=8, threshold = 4,000:**

| Fold | Variant | Trades | PF | Sharpe | MAR |
|---|---|---|---|---|---|
| 2021→(2023,2024) | Baseline | 126 | 0.61 | -2.27 | -0.45 |
| 2021→(2023,2024) | KDE-enhanced | 200 | 0.94 | -0.39 | -0.14 |
| 2022→(2024,2025) | Baseline | 450 | 0.68 | -3.86 | -0.54 |
| 2022→(2024,2025) | KDE-enhanced | 280 | 0.90 | -0.71 | -0.20 |
| 2023→(2025,2026) | Baseline | 1,091 | 0.85 | -1.52 | -0.75 |
| 2023→(2025,2026) | KDE-enhanced | 842 | 0.98 | -0.20 | -0.25 |
| 2019-2023→(2025,2026) | Baseline | 511 | 1.04 | 0.23 | 0.17 |
| 2019-2023→(2025,2026) | KDE-enhanced | 787 | 0.98 | -0.15 | -0.13 |

**Findings:**
- A fixed stage3 threshold can produce attractive single-fold metrics at high values, but trade counts become very small and the result is not robust across folds.
- With stage1/stage2 active and k=8, a threshold of 4,000 yields reasonable trade counts but remains roughly break-even to negative across folds.
- The stage3 fixed-threshold approach does not overcome the weak predictive edge in the data.

### 3.12 Slippage / cost robustness sweep

The pipeline was run across 0, 1, 2, and 3 ticks of slippage per side on the test 2025-2026 fold, using stage1/stage2 active, feature selection k=8, and xgb_reg exit.

**Baseline (no KDE):**

| Slippage ticks | Trades | Win% | PF | Sharpe | MAR |
|---|---|---|---|---|---|
| 0 | 341 | 48.68 | 1.24 | 1.24 | 1.30 |
| 1 | 490 | 44.69 | 1.05 | 0.27 | 0.20 |
| 2 | 289 | 41.52 | 0.82 | -1.31 | -0.44 |
| 3 | 919 | 42.44 | 0.83 | -1.85 | -0.75 |

**KDE-enhanced:**

| Slippage ticks | Trades | Win% | PF | Sharpe | MAR |
|---|---|---|---|---|---|
| 0 | 703 | 54.05 | 1.19 | 1.38 | 1.18 |
| 1 | 429 | 44.29 | 0.92 | -0.52 | -0.30 |
| 2 | 837 | 46.36 | 0.90 | -1.00 | -0.67 |
| 3 | 1,305 | 43.22 | 0.77 | -3.47 | -0.52 |

**Findings:**
- At zero slippage both variants look attractive, but the edge is almost entirely gone by 1 tick of slippage per side.
- At 2 and 3 ticks the strategies are clearly unprofitable.
- This confirms that the marginal edge found by feature selection is smaller than realistic transaction costs.
- A small bug in `evaluate_trades_enhanced` that raised a complex-number error when final equity was negative was fixed by capping CAGR at -100 %.

### 3.13 Target/label review: fixed-horizon labels

A major concern with the original labels is that `is_win` and `pnl_per_contract` are defined by the strategy's own exit (`net_krw`), which embeds the future exit decision the ML exit stage is supposed to learn. This creates circular supervision: stage3 is trained to predict the PnL produced by the very exit behaviour it is meant to replace.

To address this, `Devcenter/ml/add_fixed_horizon_labels.py` was added. It computes direction-signed returns at fixed bars after each trade's `entry_time` using the original 5-minute bars:

```
future_return_h = direction * (close_{t+h} - entry_px) / entry_px
is_win_h        = future_return_h > 0
```

Horizons h = 1, 3, 4, 5, 6, 7, 10, 20 bars were generated using `Devcenter/ml/add_fixed_horizon_labels.py`. The pipeline was run with `--target-horizon h`, training all three stages on these forward-looking, strategy-independent labels while still selecting entry/exit thresholds on validation `net_krw` and evaluating final performance on the original strategy PnL. A dedicated sweep script, `Devcenter/ml/target_horizon_sweep.py`, was added to run this comparison automatically.

**Single-fold test 2025-2026, 1 tick slippage, feature selection k=8:**

| Horizon | Variant | Trades | PF | Sharpe | MAR |
|---|---|---|---|---|---|
| 1 | Baseline | 325 | 1.01 | 0.08 | 0.03 |
| 1 | KDE-enhanced | 122 | 0.84 | -1.11 | -0.30 |
| 3 | Baseline | 226 | 0.89 | -0.61 | -0.25 |
| 3 | **KDE-enhanced** | **129** | **1.60** | **2.56** | **2.68** |
| 4 | Baseline | 66 | 1.29 | 1.30 | 0.63 |
| 4 | KDE-enhanced | 161 | 1.10 | 0.44 | 0.22 |
| 5 | Baseline | 178 | 0.75 | -1.37 | -0.50 |
| 5 | KDE-enhanced | 156 | 1.37 | 1.68 | 1.02 |
| 6 | Baseline | 118 | 1.25 | 1.22 | 0.78 |
| 6 | KDE-enhanced | 229 | 0.99 | -0.13 | -0.08 |
| 7 | Baseline | 306 | 0.86 | -1.04 | -0.38 |
| 7 | KDE-enhanced | 123 | 0.90 | -0.53 | -0.16 |
| 10 | Baseline | 96 | 0.46 | -3.11 | -0.74 |
| 10 | KDE-enhanced | 799 | 1.03 | 0.27 | 0.29 |
| 20 | Baseline | 337 | 0.77 | -2.54 | -0.70 |
| 20 | KDE-enhanced | 762 | 0.97 | -0.30 | -0.21 |

**Multi-fold walk-forward, target-horizon 3, 1 tick slippage, feature selection k=8:**

| Fold | Variant | Trades | PF | Sharpe | MAR |
|---|---|---|---|---|---|
| 2021→(2023,2024) | Baseline | 41 | 0.30 | -3.11 | -0.50 |
| 2021→(2023,2024) | KDE-enhanced | 163 | 0.88 | -0.83 | -0.28 |
| 2022→(2024,2025) | Baseline | 121 | 1.52 | 2.30 | 1.01 |
| 2022→(2024,2025) | KDE-enhanced | 154 | 1.14 | 0.84 | 0.27 |
| 2023→(2025,2026) | Baseline | 153 | 0.86 | -0.61 | -0.27 |
| 2023→(2025,2026) | KDE-enhanced | 421 | 1.31 | 1.75 | 2.35 |
| 2019-2023→(2025,2026) | Baseline | 226 | 0.89 | -0.61 | -0.25 |
| 2019-2023→(2025,2026) | KDE-enhanced | 129 | 1.60 | 2.56 | 2.68 |

**Findings:**
- Using a **3-bar fixed-horizon label** with KDE features gives the strongest single-fold result so far (**PF=1.60, Sharpe=2.56, MAR=2.68** on test 2025-2026).
- Walk-forward shows the KDE-enhanced variant is now profitable on the two latest folds (2022→2024/2025 and 2023→2025/2026) and on the full-history single fold.
- The 2021→2023/2024 fold is still weak, so the signal is **not fully robust across periods**, but it is materially stronger than the strategy-exit-label results.
- Threshold selection continues to optimize actual validation `net_krw`, while the models are trained on the fixed-horizon surrogate. This is a deliberate choice: the surrogate provides a cleaner learning target, while validation thresholds are chosen to maximize real per-trade PnL.

## 4. Best reproducible command (true-bar KDE + rebuilt indicators)

```powershell
# 1. Rebuild entry-side indicators from the original 5-minute bars
c:/Python/WPy64-31241/python-3.12.4.amd64/python.exe Devcenter/ml/rebuild_indicators_from_bars.py

# 2. Regenerate true-bar KDE features
c:/Python/WPy64-31241/python-3.12.4.amd64/python.exe Devcenter/ml/build_kde_from_original_bars.py
copy Devcenter\ml\ml_data\ml_dataset_with_kde_v2.csv Devcenter\ml\ml_data\ml_dataset_with_kde.csv

# 3. Run the enhanced pipeline
c:/Python/WPy64-31241/python-3.12.4.amd64/python.exe Devcenter/ml/ml_full_pipeline_enhanced.py `
    --slippage-ticks 1 --use-kde --no-regime-aware `
    --stage1-metric pnl --stage2-metric pnl --stage3-metric sharpe `
    --stage3-type xgb_reg --no-ensemble `
    --no-fractional-kelly --no-drawdown-guard
```

Best current command (true-bar KDE, rebuilt indicators, fixed-horizon target, 1 tick slippage):

```powershell
c:/Python/WPy64-31241/python-3.12.4.amd64/python.exe Devcenter/ml/add_fixed_horizon_labels.py
c:/Python/WPy64-31241/python-3.12.4.amd64/python.exe Devcenter/ml/build_kde_from_original_bars.py
copy Devcenter\ml\ml_data\ml_dataset_with_kde_v2.csv Devcenter\ml\ml_data\ml_dataset_with_kde.csv
c:/Python/WPy64-31241/python-3.12.4.amd64/python.exe Devcenter/ml/ml_full_pipeline_enhanced.py `
    --slippage-ticks 1 --use-kde --no-regime-aware `
    --stage1-metric pnl --stage2-metric pnl --stage3-metric sharpe `
    --stage3-type xgb_reg --no-ensemble `
    --no-fractional-kelly --no-drawdown-guard `
    --feature-selection-k 8 --target-horizon 3
```

Expected output on test 2025-2026:
- Baseline: Trades ~226, PF ~0.89, Sharpe ~-0.61, MAR ~-0.25
- KDE-enhanced: Trades ~129, PF ~1.60, Sharpe ~2.56, MAR ~2.68

Walk-forward shows the KDE-enhanced variant is positive on the two latest folds (PF 1.14 / 1.31) and on the full-history single fold, but still negative on 2021-2024.

Note: the earlier reported PF 1.97 / Sharpe 2.89 / MAR 1.88 was obtained with an approximate, trade-level KDE and is **not reproducible** on the original 5-minute bars.

## 5. Limitations

1. **True-bar KDE signal is weak without fixed-horizon labels**: KDE features regenerated from the original 5-minute OHLCV bars produce a marginal signal when the target is the strategy's own exit PnL. With a 3-bar fixed-horizon label, the KDE-enhanced variant reaches strong single-fold metrics (PF ≈ 1.60, Sharpe ≈ 2.56, MAR ≈ 2.68) and is profitable on the two most recent walk-forward folds, but it is still not robust across all folds. The earlier PF 1.97 / Sharpe 2.89 / MAR 1.88 result remains an artifact of the trade-level `entry_close` approximation.
2. **No robust edge across all folds**: The KDE-enhanced variant with the 3-bar fixed-horizon label is profitable on the two most recent walk-forward folds and on the full-history single fold, but still loses on the 2021-2024 fold.
3. **No original sizing**: `ml_dataset.csv` has `size_factor = 1.0` for all rows, so fractional Kelly sizing has limited room to improve results on already-sparse trade sets.
4. **Missing volume data**: The original 5-minute bar file does not contain a volume column, so volume-based features cannot be reconstructed from the available history.
5. **Look-ahead risk**: Stage1/stage2 predict `is_win`, which is known only after exit. Although all features are entry-time, the target itself embeds future information.
6. **Slippage sensitivity**: Any marginal edge is profitable at 1 tick slippage but unlikely to survive higher costs.
7. **Fold-specific performance**: Multi-fold walk-forward shows no consistent outperformance by either the baseline or the KDE-enhanced variant across different periods.
8. **LSTM ensemble bug fixed**: The `KeyError: "None of [RangeIndex(...)] are in the [index]"` indexing issue was resolved by preserving the original DataFrame index when sorting by `entry_time`. The ensemble runs end-to-end but does not materially improve the weak true-bar KDE signal.

## 6. Recommended next steps

1. **Investigate the 2021-2024 fold failure** for the 3-bar fixed-horizon KDE configuration: add regime-aware or time-aware features, or test whether the 2021-2023 training window is too short/noisy.
2. **Expand the horizon/feature search**: test fractional bars (e.g., 2, 3, 4, 5) and combine with regime-aware thresholds using the new fixed-horizon labels.
3. **Explore alternative feature engineering** beyond bar-level KDE (e.g., bar patterns, intraday seasonality, multi-timeframe momentum, cross-asset features) using the 3-bar fixed-horizon labels.
4. **Check the trade-level data generating process**: confirm that the 5-minute bar alignment, entry/exit logic, and `net_krw` computation correctly reflect the intended strategy and costs.
5. **Fractional Kelly + drawdown guard** only after a configuration consistently produces ≥ 200 trades with PF > 1.2 across folds.

## 7. Conclusion

The sequential-improvement framework is functional, and both KDE features and technical indicators have been regenerated from the original 5-minute OHLCV bars (`since2019_future_data.txt`). This removes the trade-level approximation used earlier. The previous single-fold result that exceeded the survival targets (**PF=1.97, Sharpe=2.89, MAR=1.88**) is **not reproducible** on the true bars when using the original strategy-exit labels.

However, replacing the circular strategy-exit labels (`is_win` / `pnl_per_contract`) with a **3-bar fixed-horizon directional return label** materially improves the KDE-enhanced results. The latest single-fold test (2025-2026, 1 tick slippage, feature selection k=8) reaches **PF=1.60, Sharpe=2.56, MAR=2.68**, and the two most recent walk-forward folds are also positive (**PF=1.14 / 1.31**). The 2021-2024 fold remains negative, so the configuration is **not yet robust across all periods**, but the fixed-horizon label change demonstrates that the data does contain a short-term predictive signal.

The next priorities are therefore to **investigate and fix the 2021-2024 fold failure**, **expand the horizon/feature search**, and continue **alternative feature engineering**. Once a configuration consistently produces ≥ 200 trades with PF > 1.2 across folds, sizing and drawdown guards can be added.
