# -*- coding: utf-8 -*-
"""
pivot_optuna_v2 실제 데이터 테스트 스크립트 (보강판)

업로드본 대비 추가/수정:
  (a) 검출기 WARNING 로그 + Optuna 로그 억제   → 콘솔 스팸/속도저하 제거
  (b) 거래수 제약 충족 여부 확인               → best_total_trades >= min_total_trades
  (c) '과적합' 진단을 승률이 아니라 비용차감 후
      train↔test Sharpe/Expectancy/PF/MaxDD 비교로 수행 (핵심)
  (d) 롱/숏 분해는 net_pts(비용반영) 기준으로 산출
"""
import sys
import logging

sys.path.append('c:/Project/SkyPredictor')
sys.path.append('c:/Project/SkyPredictor/Devcenter')

import pandas as pd
import pivot_optuna_v2 as pv
import regime_intraday_v2 as rg

# ── (a) 로그 억제: 이게 없으면 300 trial 동안 [HAP][확정] WARNING 이 수십만 줄 출력됨 ──
logging.getLogger('indicators.hybrid_adaptive_pivot').setLevel(logging.ERROR)
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    pass

DB_PATH = "c:/Project/SkyPredictor/Devcenter/data/duckdb/market_data.duckdb"
OUTPUT_DIR = "c:/Project/SkyPredictor/Devcenter/data/backtest_results"

MIN_TOTAL_TRADES = 60   # 최적화 제약값과 동일하게 둔다
DAILY_RESET = True      # True: 거래일별로 검출기 상태 리셋(전일→당일 갭 누수 차단)

print("=" * 80)
print("pivot_optuna_v2 실제 데이터 테스트 (보강판)")
print("=" * 80)

# 1) 날짜 범위 로드
print("\n[1] 데이터 로드...")
df = pv.load_data_by_date(DB_PATH, "futures_1min", start="2025-06-24", end="2026-06-18")
print(f"로드 완료: {len(df)} rows | {df.index.min()} ~ {df.index.max()}")

# 1-b) 주간세션(08:45~15:45)만 사용 — 야간세션 제거, 오버나잇 미보유
df = pv.filter_day_session(df, start="08:45", end="15:45")
print(f"주간세션 필터 후: {len(df)} rows | {df.index.min()} ~ {df.index.max()}")

# 2) 시간순 단일 홀드아웃
print("\n[2] train/test 분할...")
train, test = pv.time_split(df, train_frac=0.75)
print(f"Train: {len(train)} ({len(train)/len(df)*100:.1f}%) | Test: {len(test)} ({len(test)/len(df)*100:.1f}%)")

# 3) 지표 1회 계산 (train/test 각각 → 누수 없음)
print("\n[3] 지표 계산...")
train_i = pv.compute_indicators(train)
test_i = pv.compute_indicators(test)
print(f"Train {train_i.shape} | Test {test_i.shape}")

# 4) 비용/승수 모델
bt = pv.BacktestConfig(
    multiplier=250_000,
    commission_pct_per_side=3e-5,
    slippage_ticks_per_side=1.0,
    tick_size=0.05,
    entry_on="next_open",
    annualization=252.0,
    intraday_only=True,            # 당일청산: 오버나잇 보유 금지(장마감 강제청산)
    session_boundary_hour=8,       # 주간세션만 쓰면 사실상 달력일 기준
    direction_mode="both",         # 'both' | 'long_only' | 'short_only'
)
print(f"\n[4] 백테스트 설정: 승수={bt.multiplier:,}, 수수료편도={bt.commission_pct_per_side*100:.4f}%, "
      f"슬리피지={bt.slippage_ticks_per_side}틱, 진입={bt.entry_on}, 방향={bt.direction_mode}")

# 5) Optuna 최적화 (위험조정 + Walk-Forward + 제약)
print("\n[5] Optuna 최적화: n_trials=300, n_splits=4, embargo=30, metric=sharpe, "
      f"daily_reset={DAILY_RESET}")
res = pv.optimize(
    train_i, n_trials=300, seed=42, bt_cfg=bt,
    n_splits=4, embargo_bars=30, min_total_trades=MIN_TOTAL_TRADES,
    metric="sharpe", robustness_lambda=0.5, daily_reset=DAILY_RESET,
    output_dir=OUTPUT_DIR,
)

print("\n" + "=" * 80)
print("최적화 결과")
print("=" * 80)
print(f"최적 {res['best_metric']}: {res['best_value']:.4f}")
print(f"총 거래수(WF 합산): {res['best_total_trades']}")
print(f"시도 횟수: {res['n_trials']} | trial 상태: {res.get('trial_states')}")
imp = res['param_importances']
if imp:
    print(f"파라미터 중요도: { {k: round(float(v),3) for k,v in imp.items()} }")
else:
    print(f"파라미터 중요도: (계산 불가) 사유 → {res.get('param_importance_error')}")
print(f"최적 파라미터: {res['best_params']}")

# (b) 제약 충족 확인: 거래수가 하한 미만이면 best 는 '차악 infeasible' 일 수 있음
feasible = res.get("constraint_satisfied", (res["best_total_trades"] or 0) >= MIN_TOTAL_TRADES)
print(f"\n[제약] 거래수 하한 {MIN_TOTAL_TRADES} 충족: {feasible} (best 거래수={res['best_total_trades']})")
if not feasible:
    print("  ⚠ 제약을 만족하는 trial 이 없어 값 기준 차선책으로 폴백했습니다.")
    print("    필터를 완화하거나 min_total_trades 를 낮추거나 데이터 구간/ n_trials 를 늘리세요.")

# 6) best 파라미터로 train / test 각각 백테스트
p = res["best_params"]
pcfg = pv.HybridAdaptivePivotConfig(
    base_pct=p["base_pct"], base_multiplier=p["base_multiplier"],
    atr_weight=p["atr_weight"], confirmation_bars=p["confirmation_bars"],
)
fcfg = pv.FilterConfig(
    min_wave_pct=p["min_wave_pct"], min_pivot_interval_bars=p["min_pivot_interval_bars"],
    st_distance_threshold=p["st_distance_threshold"], adx_hold_threshold=p["adx_hold_threshold"],
)

def _detect(d):
    if DAILY_RESET:
        return pv.detect_pivots_daily(d, pcfg, fcfg, bt.session_boundary_hour)
    return pv.detect_pivots(d, pcfg, fcfg)

train_res = pv.backtest(train_i, _detect(train_i), bt)
test_res = pv.backtest(test_i, _detect(test_i), bt)

# (c) 핵심: 과적합 진단은 '승률'이 아니라 비용차감 후 위험조정 지표의 train↔test 일치도로 본다
print("\n" + "=" * 80)
print("과적합 진단 (train ↔ test, 비용 반영)")
print("=" * 80)
hdr = f"{'지표':<22}{'train':>16}{'test':>16}"
print(hdr); print("-" * len(hdr))
def _row(name, a, b, fmt="{:.4f}"):
    print(f"{name:<22}{fmt.format(a):>16}{fmt.format(b):>16}")
_row("n_trades", train_res.n_trades, test_res.n_trades, "{:d}")
_row("sharpe_daily", train_res.sharpe_daily, test_res.sharpe_daily)
_row("expectancy_pts", train_res.expectancy_pts, test_res.expectancy_pts)
_row("profit_factor", min(train_res.profit_factor, 99), min(test_res.profit_factor, 99))
_row("total_pnl_krw", train_res.total_pnl_krw, test_res.total_pnl_krw, "{:,.0f}")
_row("max_drawdown_krw", train_res.max_drawdown_krw, test_res.max_drawdown_krw, "{:,.0f}")
_row("win_rate(참고용)", train_res.win_rate, test_res.win_rate, "{:.2f}")

# 판정: 승률이 떨어졌는지가 아니라, test Sharpe/PF/PnL 이 살아있는지가 기준
ok = (test_res.sharpe_daily > 0) and (test_res.total_pnl_krw > 0) and (test_res.profit_factor > 1.0)
print(f"\n[판정] test 구간 위험조정 수익 유지: {ok}")
print("  · 승률 하락 자체는 과적합 근거가 아님(비대칭 손익 반전계는 승률<50% 에서도 수익 가능).")
print("  · train 대비 test 의 Sharpe/PF 가 크게 무너지면 그때가 과적합이다.")

# (d) 롱/숏 분해 (비용 반영 net 기준)
if test_res.trades is not None and len(test_res.trades):
    t = test_res.trades
    for d, label in [(1, "롱"), (-1, "숏")]:
        sub = t[t["direction"] == d]
        if len(sub):
            wr = (sub["net_pts"] > 0).mean() * 100
            print(f"  [{label}] {len(sub)}건 | 승률 {wr:.2f}% | 순손익 {sub['net_krw'].sum():,.0f}원")
    # 청산 사유 분해: 당일청산이 제대로 걸리는지 확인
    if "exit_reason" in t.columns:
        rc = t["exit_reason"].value_counts().to_dict()
        print(f"  [청산사유] {rc}  (eod=장마감 강제청산, pivot=반전, final=구간끝)")

print("\n" + "=" * 80)
print("구간별 진단 (전체 구간, 고정 best 파라미터) - 단일 홀드아웃이 '운'인지 확인")
print("=" * 80)
print("  · test Sharpe > train Sharpe 이고 PnL 이 롱에만 몰리면, test 구간이 상승 레짐이었을 가능성.")
print("  · 아래 표에서 숏 손익이 '일부 구간만' 0/음수면 레짐 의존, '전 구간' 음수면 구조적 결함.")
full_i = pv.compute_indicators(df)   # df = 주간세션 필터된 전체 구간
pv.diagnose_windows(full_i, pcfg, fcfg, bt, n_windows=6, daily_reset=DAILY_RESET)

# 베타 점검: 롱 leg 가 피봇 타이밍의 알파인지, 그냥 '장중 상시 롱' 베타인지
print("\n[베타 점검] 전략 vs 장중 상시 롱 베이스라인 (전체 구간)")
strat = pv.backtest(full_i, _detect(full_i), bt)
base = pv.intraday_long_baseline(full_i, bt)
print(f"  전략(both)      : Sharpe {strat.sharpe_daily:>7.3f} | PnL {strat.total_pnl_krw:>14,.0f} | 거래 {strat.n_trades}")
print(f"  장중 상시 롱     : Sharpe {base.sharpe_daily:>7.3f} | PnL {base.total_pnl_krw:>14,.0f} | 일수 {base.n_trades}")
print("  · 베이스라인 Sharpe 가 전략과 비슷/우위면, 롱 수익은 피봇 알파가 아니라 구간 상승(베타)일 뿐.")

# long_only 모드 테스트: 숏 제거 시 성능 개선 확인
print("\n[long_only 모드 테스트] 숏 제거 시 성능 변화")
bt_long = pv.BacktestConfig(
    multiplier=bt.multiplier,
    commission_pct_per_side=bt.commission_pct_per_side,
    slippage_ticks_per_side=bt.slippage_ticks_per_side,
    tick_size=bt.tick_size,
    entry_on=bt.entry_on,
    annualization=bt.annualization,
    intraday_only=bt.intraday_only,
    session_boundary_hour=bt.session_boundary_hour,
    direction_mode="long_only",  # 롱만
)
strat_long = pv.backtest(full_i, _detect(full_i), bt_long)
print(f"  전략(long_only) : Sharpe {strat_long.sharpe_daily:>7.3f} | PnL {strat_long.total_pnl_krw:>14,.0f} | 거래 {strat_long.n_trades}")
print(f"  전략(both)      : Sharpe {strat.sharpe_daily:>7.3f} | PnL {strat.total_pnl_krw:>14,.0f} | 거래 {strat.n_trades}")
pnl_improvement = strat_long.total_pnl_krw - strat.total_pnl_krw
print(f"  · 숏 제거 시 PnL 개선: {pnl_improvement:,.0f}원 ({pnl_improvement/strat.total_pnl_krw*100:+.1f}%)")
print("  · long_only가 both보다 우위면 숏 제거가 통계적으로 우월함.")

# 트렌드 팔로우 테스트: 피봇 반전 vs 트렌드 팔로우 비교 (당일청산 유지 + 단기 MA)
print("\n[트렌드 팔로우 테스트] 피봇 반전 vs 트렌드 팔로우 (당일청산 유지 + 단기 MA)")

# 테스트할 파라미터 조합 (단기 MA - 당일청산에 적합)
trend_configs = [
    ("MA 5/15 (초단기)", pv.TrendConfig(method="ma_crossover", short_ma=5, long_ma=15, adx_threshold=25.0, breakout_period=20, atr_multiplier=2.0)),
    ("MA 10/30 (단기)", pv.TrendConfig(method="ma_crossover", short_ma=10, long_ma=30, adx_threshold=25.0, breakout_period=20, atr_multiplier=2.0)),
    ("MA 15/45 (중단기)", pv.TrendConfig(method="ma_crossover", short_ma=15, long_ma=45, adx_threshold=25.0, breakout_period=20, atr_multiplier=2.0)),
    ("MA 20/60 (기본)", pv.TrendConfig(method="ma_crossover", short_ma=20, long_ma=60, adx_threshold=25.0, breakout_period=20, atr_multiplier=2.0)),
    ("ADX 필터 25 (5/15)", pv.TrendConfig(method="adx_trend", short_ma=5, long_ma=15, adx_threshold=25.0, breakout_period=20, atr_multiplier=2.0)),
    ("ADX 필터 30 (10/30)", pv.TrendConfig(method="adx_trend", short_ma=10, long_ma=30, adx_threshold=30.0, breakout_period=20, atr_multiplier=2.0)),
    ("브레이크아웃 30", pv.TrendConfig(method="breakout", short_ma=20, long_ma=60, adx_threshold=25.0, breakout_period=30, atr_multiplier=2.0)),
    ("브레이크아웃 60", pv.TrendConfig(method="breakout", short_ma=20, long_ma=60, adx_threshold=25.0, breakout_period=60, atr_multiplier=2.0)),
]

print(f"  피봇 반전(both)   : Sharpe {strat.sharpe_daily:>7.3f} | PnL {strat.total_pnl_krw:>14,.0f} | 거래 {strat.n_trades}")

for name, cfg in trend_configs:
    try:
        signals = pv.detect_trend_signals(full_i, cfg)
        if len(signals) == 0:
            print(f"  {name:<20} : 신호 없음")
            continue

        bt_trend = pv.BacktestConfig(
            multiplier=bt.multiplier,
            commission_pct_per_side=bt.commission_pct_per_side,
            slippage_ticks_per_side=bt.slippage_ticks_per_side,
            tick_size=bt.tick_size,
            entry_on=bt.entry_on,
            annualization=bt.annualization,
            intraday_only=bt.intraday_only,  # 당일청산 유지
            session_boundary_hour=bt.session_boundary_hour,
            direction_mode="both",
        )
        result = pv.backtest(full_i, signals, bt_trend)
        print(f"  {name:<20} : Sharpe {result.sharpe_daily:>7.3f} | PnL {result.total_pnl_krw:>14,.0f} | 거래 {result.n_trades}")
    except Exception as e:
        print(f"  {name:<20} : 오류 - {str(e)}")

print("  · 트렌드 팔로우가 피봇 반전보다 우위면 추세 지속 전략이 적합함.")

# 장중 상시 롤 + 필터 최적화 테스트
print("\n[장중 상시 롤 + 필터 최적화] 매일 신호 + 승률 개선")
print(f"  장중 상시 롤(기본) : Sharpe {base.sharpe_daily:>7.3f} | PnL {base.total_pnl_krw:>14,.0f} | 거래 {base.n_trades}")

# 필터 조합 테스트
filter_configs = [
    ("필터 없음", False, 0.0),
    ("전일 음봉 필터", True, 0.0),
    ("ATR 필터 0.5", False, 0.5),
    ("전일 음봉 + ATR 0.5", True, 0.5),
    ("전일 음봉 + ATR 0.3", True, 0.3),
    ("전일 음봉 + ATR 0.7", True, 0.7),
]

for name, filter_candle, filter_atr in filter_configs:
    try:
        result = pv.enhanced_intraday_long(
            full_i,
            bt,
            filter_previous_candle=filter_candle,
            filter_atr_ratio=filter_atr,
        )
        print(f"  {name:<20} : Sharpe {result.sharpe_daily:>7.3f} | PnL {result.total_pnl_krw:>14,.0f} | 거래 {result.n_trades}")
    except Exception as e:
        print(f"  {name:<20} : 오류 - {str(e)}")

print("  · 필터 조합이 기본보다 우위면 필터 추가가 유효함.")

# 레짐 기반 당일 매매 테스트: 상승장 롱, 하락장 숏, 횡보장 스킵
print("\n[레짐 기반 당일 매매] 상승장 롱, 하락장 숏, 횡보장 스킵")
print(f"  피봇 반전(both)   : Sharpe {strat.sharpe_daily:>7.3f} | PnL {strat.total_pnl_krw:>14,.0f} | 거래 {strat.n_trades:>3} | 승률 {strat.win_rate:>5.1f}%")

# 레짐 파라미터 조합
regime_configs = [
    ("MA 20/60", "ma", 20, 60, 25.0, 0.0, 0.0),
    ("MA 50/200", "ma", 50, 200, 25.0, 0.0, 0.0),
    ("ADX 25 (20/60)", "adx", 20, 60, 25.0, 0.0, 0.0),
    ("ADX 30 (20/60)", "adx", 20, 60, 30.0, 0.0, 0.0),
    ("MA 20/60 + ATR 0.5", "ma", 20, 60, 25.0, 0.5, 0.0),
    ("MA 50/200 + ATR 0.5", "ma", 50, 200, 25.0, 0.5, 0.0),
]

for name, method, ma_short, ma_long, adx_thresh, atr_ratio, gap_thresh in regime_configs:
    try:
        result = pv.regime_based_intraday(
            full_i,
            bt,
            regime_method=method,
            ma_short=ma_short,
            ma_long=ma_long,
            adx_threshold=adx_thresh,
            filter_atr_ratio=atr_ratio,
            gap_threshold=gap_thresh,
        )
        print(f"  {name:<20} : Sharpe {result.sharpe_daily:>7.3f} | PnL {result.total_pnl_krw:>14,.0f} | 거래 {result.n_trades:>3} | 승률 {result.win_rate:>5.1f}%")
    except Exception as e:
        print(f"  {name:<20} : 오류 - {str(e)}")

print("  · 레짐 기반이 피봇 반전보다 우위면 레짐 전략이 적합함.")

# 레짐 MA 20/60 기본 버전 (갭 필터 없음) 계산
regime_result = pv.regime_based_intraday(
    full_i,
    bt,
    regime_method="ma",
    ma_short=20,
    ma_long=60,
    adx_threshold=25.0,
    filter_atr_ratio=0.0,
    gap_threshold=0.0,
)

# 갭 필터 테스트
print("\n[갭 필터 테스트] 갭 크기 필터 효과")
print(f"  레짐 MA 20/60 (갭 필터 없음) : Sharpe {regime_result.sharpe_daily:>7.3f} | PnL {regime_result.total_pnl_krw:>14,.0f} | 거래 {regime_result.n_trades:>3}")

gap_configs = [
    ("갭 필터 0.5%", 0.005),
    ("갭 필터 1.0%", 0.01),
    ("갭 필터 1.5%", 0.015),
    ("갭 필터 2.0%", 0.02),
]

for name, gap_thresh in gap_configs:
    try:
        result = pv.regime_based_intraday(
            full_i,
            bt,
            regime_method="ma",
            ma_short=20,
            ma_long=60,
            adx_threshold=25.0,
            filter_atr_ratio=0.0,
            gap_threshold=gap_thresh,
        )
        print(f"  {name:<20} : Sharpe {result.sharpe_daily:>7.3f} | PnL {result.total_pnl_krw:>14,.0f} | 거래 {result.n_trades:>3}")
    except Exception as e:
        print(f"  {name:<20} : 오류 - {str(e)}")

print("  · 갭 필터가 기본보다 우위면 갭 필터 추가가 유효함.")

# 당일 중간 레짐 역전 감지 테스트
print("\n[당일 중간 레짐 역전 감지] 포지션 전환 효과")
print(f"  레짐 MA 20/60 (기본) : Sharpe {regime_result.sharpe_daily:>7.3f} | PnL {regime_result.total_pnl_krw:>14,.0f} | 거래 {regime_result.n_trades:>3}")

try:
    reversal_result = pv.regime_based_intraday(
        full_i,
        bt,
        regime_method="ma",
        ma_short=20,
        ma_long=60,
        adx_threshold=25.0,
        filter_atr_ratio=0.0,
        gap_threshold=0.0,
        intraday_reversal=True,
    )
    print(f"  레짐 MA 20/60 (역전) : Sharpe {reversal_result.sharpe_daily:>7.3f} | PnL {reversal_result.total_pnl_krw:>14,.0f} | 거래 {reversal_result.n_trades:>3}")
    print("  · 역전 감지가 기본보다 우위면 역전 감지 추가가 유효함.")
except Exception as e:
    print(f"  레짐 MA 20/60 (역전) : 오류 - {str(e)}")

# 레짐 MA 20/60 상세 수익성 분석
print("\n[레짐 MA 20/60 상세 수익성 분석]")
print(f"  Sharpe            : {regime_result.sharpe_daily:>7.3f}")
print(f"  총 수익 (원)      : {regime_result.total_pnl_krw:>14,.0f}")
print(f"  총 수익 (pt)      : {regime_result.total_pnl_pts:>10.2f}")
print(f"  거래 수           : {regime_result.n_trades:>3}")
print(f"  승률 (%)          : {regime_result.win_rate:>5.1f}%")
print(f"  기대값 (pt/거래)  : {regime_result.expectancy_pts:>8.3f}")
print(f"  최대 드로다운(원) : {regime_result.max_drawdown_krw:>14,.0f}")

# 롱/숏 분리 분석 (거래 내역 확인)
if len(regime_result.trades) > 0:
    trades_df = regime_result.trades.copy()
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_time']).dt.date
    trades_df['is_win'] = trades_df['net_pts'] > 0

    # 일별 통계
    daily_stats = trades_df.groupby('exit_date').agg({
        'is_win': ['sum', 'count'],
        'net_pts': 'sum',
        'net_krw': 'sum'
    }).reset_index()
    daily_stats.columns = ['date', 'wins', 'total_trades', 'total_pts', 'total_krw']
    daily_stats['win_rate'] = (daily_stats['wins'] / daily_stats['total_trades'] * 100).round(2)

    print(f"\n  일별 승률 분포:")
    print(f"    평균 일별 승률   : {daily_stats['win_rate'].mean():.2f}%")
    print(f"    최고 일별 승률   : {daily_stats['win_rate'].max():.2f}%")
    print(f"    최저 일별 승률   : {daily_stats['win_rate'].min():.2f}%")
    print(f"    승률 50% 이상 날 : {(daily_stats['win_rate'] >= 50).sum()}일 / {len(daily_stats)}일")

    # 수익 구간 분석
    print(f"\n  수익 구간 분석:")
    print(f"    수익 일 수        : {(daily_stats['total_krw'] > 0).sum()}일")
    print(f"    손실 일 수        : {(daily_stats['total_krw'] < 0).sum()}일")
    print(f"    최고 일 수익      : {daily_stats['total_krw'].max():,.0f}원")
    print(f"    최대 일 손실      : {daily_stats['total_krw'].min():,.0f}원")

print("\n" + "=" * 80)
print("테스트 완료")
print("=" * 80)

# 인과적 레짐 전략 비교 (look-ahead 제거 버전)
print("\n" + "=" * 80)
print("인과적 레짐 전략 비교 (look-ahead 제거, train/test 분리)")
print("=" * 80)
print("  · 기존 레짐 테스트는 look-ahead 결함이 있어 신뢰할 수 없음")
print("  · regime_intraday_v2.py의 인과적 버전으로 재테스트")
print("  · train/test 분리로 out-of-sample 평가")

rg.compare_strategies(train_i, test_i, pcfg, fcfg, bt, daily_reset=DAILY_RESET)

# 방법론적 결함 수정: 워밍업 보존 + train기준 선택 + Sharpe SE
print("\n" + "=" * 80)
print("방법론적 결함 수정: 워밍업 보존 + train기준 선택 + Sharpe SE")
print("=" * 80)
print("  · test-set 선택 금지: TRAIN Sharpe로 MA 선택")
print("  · 워밍업 절단 제거: 전체 기간 일봉에서 1회 계산 후 슬라이스")
print("  · Sharpe 표준오차: 짧은 test에서 차이가 노이즈인지 판정")

full_i = pv.compute_indicators(df)  # df = 주간세션 필터된 전체
test_start = test_i.index[0]

res = rg.select_regime_oos(full_i, test_start, bt,
                           ma_configs=((5,15),(10,30),(20,60),(50,200)))

# 선택된 조합의 구간별 안정성
b = res["best"]
if b:
    print("\n" + "=" * 80)
    print("선택된 조합의 구간별 안정성 (6분할)")
    print("=" * 80)
    rg.regime_window_dispersion(full_i, bt, b["ma_short"], b["ma_long"], n_windows=6)

# 베타 구간 분석 (6분할) - 무방비 롤 노출의 구간별 위험 확인
print("\n" + "=" * 80)
print("베타 구간 분석 (6분할) - 무방비 롤 노출의 구간별 위험")
print("=" * 80)
print("  · 베타 베이스라인은 '전략'이 아니라 무방비 롤 노출")
print("  · 상승 구간에서는 수익, 하락 구간에서는 그대로 손실")
print("  · 최악 구간의 MaxDD가 감내 가능한지 확인 필요")

import numpy as np
bounds = np.array_split(np.arange(len(full_i)), 6)
for i, b in enumerate(bounds, 1):
    seg = full_i.iloc[b[0]:b[-1]+1]
    r = pv.intraday_long_baseline(seg, bt)
    print(f"win{i} 거래{r.n_trades:>3} Sharpe{r.sharpe_daily:>7.3f} "
          f"PnL{r.total_pnl_krw:>13,.0f} MaxDD{r.max_drawdown_krw:>13,.0f}")

# 롱/플랫 변형 테스트 - 숏 절대 안 함
print("\n" + "=" * 80)
print("롱/플랫 변형 테스트 - 숏 절대 안 함")
print("=" * 80)
print("  · bull -> 롱, 그 외 -> 플랫(현금), 숏 절대 안 함")
print("  · 상승 구간에서는 상시 롱과 거의 동일")
print("  · 하락 구간에서 손실 회피로 MaxDD 감소 목적")

res_longonly = rg.select_regime_oos(full_i, test_start, bt,
                                     ma_configs=((5,15),(10,30),(20,60),(50,200)),
                                     allow_short=False)

# 선택된 조합의 구간별 안정성 (롱/플랫)
b_longonly = res_longonly["best"]
if b_longonly:
    print("\n" + "=" * 80)
    print("선택된 조합의 구간별 안정성 (롱/플랫, 6분할)")
    print("=" * 80)
    rg.regime_window_dispersion(full_i, bt, b_longonly["ma_short"], b_longonly["ma_long"],
                                 allow_short=False, n_windows=6)

# 손절매/변동성 필터 테스트
print("\n" + "=" * 80)
print("손절매/변동성 필터 테스트")
print("=" * 80)
print("  · 손절매: 진입 후 특정 손실률 도달 시 강제 청산")
print("  · 변동성 필터: ATR 기반 변동성이 너무 낮으면 진입 스킵")
print("  · 목표: win5 (-3,553만원 MaxDD) 방지")

# 다양한 손절매 비율 테스트
stoploss_configs = [0.01, 0.02, 0.03]  # 1%, 2%, 3%
volatility_configs = [True, False]

print(f"{'전략':<30}{'TRAIN Sharpe':>14}{'TEST Sharpe':>14}{'TEST PnL':>16}"
      f"{'TEST MaxDD':>16}{'TEST거래':>9}")
print("-" * 100)

# 베타 기준
base_tr = pv.intraday_long_baseline(train_i, bt)
base_te = pv.intraday_long_baseline(test_i, bt)
print(f"{'장중상시롱(베타)':<30}{base_tr.sharpe_daily:>14.3f}{base_te.sharpe_daily:>14.3f}"
      f"{base_te.total_pnl_krw:>16,.0f}{base_te.max_drawdown_krw:>16,.0f}{base_te.n_trades:>9}")

# 손절매/변동성 필터 조합 테스트
for sl in stoploss_configs:
    for vf in volatility_configs:
        mode = "손절" if vf else "손절만"
        name = f"손절{int(sl*100)}%({mode})"
        tr = pv.intraday_long_with_stoploss(train_i, bt, stoploss_pct=sl, volatility_filter=vf)
        te = pv.intraday_long_with_stoploss(test_i, bt, stoploss_pct=sl, volatility_filter=vf)
        print(f"{name:<30}{tr.sharpe_daily:>14.3f}{te.sharpe_daily:>14.3f}"
              f"{te.total_pnl_krw:>16,.0f}{te.max_drawdown_krw:>16,.0f}{te.n_trades:>9}")

# 최적 조합 구간 분석
print("\n" + "=" * 80)
print("최적 조합 구간 분석 (손절2% + 변동성필터)")
print("=" * 80)

best_sl = 0.02
best_vf = True
bounds = np.array_split(np.arange(len(full_i)), 6)
for i, b in enumerate(bounds, 1):
    seg = full_i.iloc[b[0]:b[-1]+1]
    r = pv.intraday_long_with_stoploss(seg, bt, stoploss_pct=best_sl, volatility_filter=best_vf)
    print(f"win{i} 거래{r.n_trades:>3} Sharpe{r.sharpe_daily:>7.3f} "
          f"PnL{r.total_pnl_krw:>13,.0f} MaxDD{r.max_drawdown_krw:>13,.0f}")

# 롱-또는-플랫 구간 분석 (다운사이드 보호 확인)
print("\n" + "=" * 80)
print("롱-또는-플랫 구간 분석 (다운사이드 보호 확인)")
print("=" * 80)
print("  · bull/중립 -> 롱, bear -> 플랫(현금)")
print("  · 하락 구간 MaxDD 감소 효과 확인")

daily_full = rg.to_daily(full_i, bt.session_boundary_hour)
signal_lf = rg.daily_regime_signal(daily_full, regime_method="ma", ma_short=20, ma_long=60,
                                   adx_threshold=25.0, allow_short=False)

bounds = np.array_split(np.arange(len(full_i)), 6)
for i, b in enumerate(bounds, 1):
    seg = full_i.iloc[b[0]:b[-1]+1]
    r = rg.regime_intraday_daily(seg, bt, regime_method="ma", ma_short=20, ma_long=60,
                                 adx_threshold=25.0, signal=signal_lf)
    print(f"win{i} 거래{r.n_trades:>3} Sharpe{r.sharpe_daily:>7.3f} "
          f"PnL{r.total_pnl_krw:>13,.0f} MaxDD{r.max_drawdown_krw:>13,.0f}")
