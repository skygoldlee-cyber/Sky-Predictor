# -*- coding: utf-8 -*-
"""
pivot_walkforward_v3.py
=======================

`pivot_optuna_v2.make_objective` 의 **진짜 train→test 분리형** 리팩토링.

기존 `make_objective` 의 한계
-----------------------------
- `purged_walkforward_folds` 로 구간을 나누지만, 목적함수가 *모든 fold* 에서
  backtest 를 돌려 평균을 내고 Optuna 가 그 평균을 최대화한다.
- 즉 파라미터를 선택하는 그 데이터로 동시에 평가한다 → 전부 **in-sample**.
  purge/embargo 는 fold 경계의 상태 누수만 막을 뿐, 선택편향을 막지 못한다.
- 결과적으로 "최적 Sharpe 1.9~3.1" 같은 수치는 선택잡음으로 부풀려진다.

이 모듈이 바꾸는 것
-------------------
1. **최종 홀드아웃 분리**: 거래일을 시간순으로 잘라 마지막 `test_frac` 구간을
   Optuna 가 **한 번도** 보지 않는 테스트 셋으로 떼어둔다(사이 embargo).
2. **전방(forward) 평가 블록**: 최적화 구간 안을 비중첩 전방 윈도우로 나눠
   각 블록에서만 metric 을 측정하고 mean−λ·std 로 '구간 일관성'을 본다.
   (검출은 거래일별 리셋 + 지표는 전체 1회 계산 후 슬라이스 → 블록 간 누수 없음)
3. **편향 없는 최종 수치**: study 종료 후 best 파라미터를 홀드아웃에서 **딱 한 번**
   평가한 값을 OOS 추정치로 보고한다. 베타(장중 상시 롱)/롱-또는-플랫과 같은
   홀드아웃에서 비교하고 Sharpe 표준오차까지 같이 낸다.

지표 계산·검출·백테스트·비용모델은 모두 `pivot_optuna_v2` 의 인과적 구현을 그대로
재사용한다(이 파일은 '평가 프로토콜'만 교체한다).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import pivot_optuna_v2 as pv

try:
    import optuna
except ImportError:
    optuna = None

try:
    import regime_intraday_v2 as rg
except ImportError:
    rg = None


# ════════════════════════════════════════════════════════════════════════════
# 거래일 단위 분할 유틸 (모든 윈도우 경계를 '거래일'에 정렬 → 하루가 쪼개지지 않음)
# ════════════════════════════════════════════════════════════════════════════
def unique_trading_days(df: pd.DataFrame, boundary_hour: int = 8) -> np.ndarray:
    """등장 순서를 보존한 거래일 키 배열(중복 제거)."""
    tday = pv.trading_day_key(df.index, boundary_hour)
    return pd.unique(tday)


def slice_by_days(df: pd.DataFrame, day_keys, boundary_hour: int = 8) -> pd.DataFrame:
    """주어진 거래일 키 집합에 해당하는 봉만 추출(시간순 보존)."""
    tday = pv.trading_day_key(df.index, boundary_hour)
    mask = np.isin(tday, np.asarray(list(day_keys)))
    return df.iloc[mask]


def split_opt_holdout(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    embargo_days: int = 2,
    boundary_hour: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """거래일을 시간순으로 [최적화 구간 | embargo | 홀드아웃 테스트] 로 분할.

    embargo 거래일은 어느 쪽에도 포함하지 않아 경계 인접일이 선택·평가에
    동시에 쓰이지 않게 한다.
    Returns: (opt_df, test_df, opt_days, test_days)
    """
    days = unique_trading_days(df, boundary_hour)
    n = len(days)
    n_test = max(1, int(round(n * test_frac)))
    n_opt = n - n_test - embargo_days
    if n_opt < 1:
        raise ValueError(f"데이터가 너무 짧습니다: 총 {n}일, test {n_test}, embargo {embargo_days}")
    opt_days = days[:n_opt]
    test_days = days[n_opt + embargo_days:]
    opt_df = slice_by_days(df, opt_days, boundary_hour)
    test_df = slice_by_days(df, test_days, boundary_hour)
    return opt_df, test_df, opt_days, test_days


def forward_blocks(
    opt_days: np.ndarray, val_block_days: int = 21, step_days: Optional[int] = None
) -> List[np.ndarray]:
    """최적화 구간을 비중첩(기본) 전방 평가 블록들로 나눈다."""
    step = step_days or val_block_days
    blocks: List[np.ndarray] = []
    i = 0
    while i + val_block_days <= len(opt_days):
        blocks.append(opt_days[i:i + val_block_days])
        i += step
    # 마지막 자투리가 충분히 길면 한 블록으로 흡수
    if i < len(opt_days) and (len(opt_days) - i) >= max(5, val_block_days // 2):
        blocks.append(opt_days[i:])
    return blocks


# ════════════════════════════════════════════════════════════════════════════
# 파라미터 dict → 설정 객체 / 평가
# ════════════════════════════════════════════════════════════════════════════
def cfgs_from_params(params: Dict) -> Tuple[pv.HybridAdaptivePivotConfig, pv.FilterConfig]:
    pcfg = pv.HybridAdaptivePivotConfig(
        base_pct=params["base_pct"],
        base_multiplier=params["base_multiplier"],
        atr_weight=params["atr_weight"],
        confirmation_bars=params["confirmation_bars"],
    )
    fcfg = pv.FilterConfig(
        enabled=True,
        min_wave_pct=params["min_wave_pct"],
        min_pivot_interval_bars=params["min_pivot_interval_bars"],
        st_distance_threshold=params["st_distance_threshold"],
        adx_hold_threshold=params["adx_hold_threshold"],
    )
    return pcfg, fcfg


def evaluate_on(
    df_block: pd.DataFrame,
    pcfg: pv.HybridAdaptivePivotConfig,
    fcfg: pv.FilterConfig,
    bt: pv.BacktestConfig,
    daily_reset: bool = True,
) -> pv.BacktestResult:
    """한 구간(이미 슬라이스된 df)에서 검출+백테스트.

    df_block 은 compute_indicators 가 끝난 '전체 df' 에서 슬라이스된 것이어야 한다
    (지표 워밍업 NaN 이 구간 앞에서 다시 생기지 않게).
    """
    if df_block is None or len(df_block) == 0:
        return pv.BacktestResult(trades=pd.DataFrame())
    if daily_reset:
        pivots = pv.detect_pivots_daily(df_block, pcfg, fcfg, bt.session_boundary_hour)
    else:
        pivots = pv.detect_pivots(df_block, pcfg, fcfg)
    return pv.backtest(df_block, pivots, bt)


def _metric_value(res: pv.BacktestResult, metric: str) -> float:
    if metric == "sharpe":
        return res.sharpe_daily
    if metric == "expectancy":
        return res.expectancy_pts
    if metric == "profit_factor":
        pf = res.profit_factor
        return min(pf, 10.0) if math.isfinite(pf) else 10.0
    raise ValueError(metric)


# ════════════════════════════════════════════════════════════════════════════
# 전방 블록 기반 목적함수 (선택 기준) — 단, 최종 OOS 추정치는 홀드아웃에서 따로
# ════════════════════════════════════════════════════════════════════════════
def make_walkforward_objective(
    df_opt_ind: pd.DataFrame,
    blocks: List[np.ndarray],
    bt: pv.BacktestConfig,
    metric: str = "sharpe",
    robustness_lambda: float = 0.5,
    min_total_trades: int = 40,
    min_block_trades: int = 3,
    no_trade_penalty: float = -1.0,
    daily_reset: bool = True,
    boundary_hour: int = 8,
) -> Callable:
    """전방 평가 블록들에서 metric 의 mean−λ·std 를 반환하는 Optuna 목적함수.

    주의: 이 값은 '최적화 구간 안의 구간 일관성' 선택 기준이지 OOS 추정치가 아니다.
    편향 없는 OOS 추정은 nested_walkforward_optimize 가 홀드아웃에서 따로 계산한다.
    """
    # 블록별 df 를 미리 슬라이스(반복 계산 회피)
    block_dfs = [slice_by_days(df_opt_ind, b, boundary_hour) for b in blocks]

    def objective(trial) -> float:
        params = dict(
            base_pct=trial.suggest_float("base_pct", 0.05, 2.0, log=True),
            base_multiplier=trial.suggest_float("base_multiplier", 0.5, 10.0),
            atr_weight=trial.suggest_float("atr_weight", 0.0, 1.0),
            confirmation_bars=trial.suggest_int("confirmation_bars", 1, 10),
            min_wave_pct=trial.suggest_float("min_wave_pct", 0.05, 2.0, log=True),
            min_pivot_interval_bars=trial.suggest_int("min_pivot_interval_bars", 1, 30),
            st_distance_threshold=trial.suggest_float("st_distance_threshold", 0.01, 1.0, log=True),
            adx_hold_threshold=trial.suggest_float("adx_hold_threshold", 5.0, 50.0),
        )
        pcfg, fcfg = cfgs_from_params(params)

        scores: List[float] = []
        total_trades = 0
        for step, bdf in enumerate(block_dfs):
            res = evaluate_on(bdf, pcfg, fcfg, bt, daily_reset)
            total_trades += res.n_trades
            scores.append(_metric_value(res, metric)
                          if res.n_trades >= min_block_trades else no_trade_penalty)
            trial.report(float(np.mean(scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("constraints", (float(min_total_trades - total_trades),))
        trial.set_user_attr("total_trades", total_trades)
        trial.set_user_attr("n_blocks", len(scores))
        if not scores:
            return -1e9
        mean_s = float(np.mean(scores))
        std_s = float(np.std(scores)) if len(scores) > 1 else 0.0
        return mean_s - robustness_lambda * std_s

    return objective


# ════════════════════════════════════════════════════════════════════════════
# Nested 최적화: 선택은 opt 구간, 평가는 홀드아웃(한 번도 안 본 구간)
# ════════════════════════════════════════════════════════════════════════════
def _sharpe_se(res: pv.BacktestResult, annualization: float = 252.0) -> float:
    if rg is not None:
        return rg.annualized_sharpe_se(res, annualization)
    if res.trades is None or len(res.trades) < 2:
        return float("nan")
    n = pd.to_datetime(res.trades["exit_time"]).dt.date.nunique()
    if n < 2:
        return float("nan")
    sr = res.sharpe_daily
    return float(math.sqrt((1.0 + 0.5 * sr ** 2 / annualization) / n) * math.sqrt(annualization))


def baselines_on(
    test_df: pd.DataFrame,
    bt: pv.BacktestConfig,
    df_full: Optional[pd.DataFrame] = None,
) -> Dict[str, pv.BacktestResult]:
    """홀드아웃에서의 공정 비교 기준선.

    df_full 을 주면 롱-또는-플랫 레짐 신호를 '전체 기간' 일봉에서 1회 계산해
    홀드아웃 구간으로 슬라이스한다. (짧은 홀드아웃에서 MA60 워밍업으로 신호가
    0이 되는 문제 방지)
    """
    out: Dict[str, pv.BacktestResult] = {}
    try:
        out["beta_long"] = pv.intraday_long_baseline(test_df, bt)
    except Exception:
        pass
    if rg is not None:
        try:
            signal = None
            if df_full is not None:
                daily_full = rg.to_daily(df_full, bt.session_boundary_hour)
                if len(daily_full) >= 62:
                    signal = rg.daily_regime_signal(daily_full, "adx", 20, 60, 25.0)
            out["long_or_flat"] = rg.regime_intraday_daily(
                test_df, bt, regime_method="adx", ma_short=20, ma_long=60,
                adx_threshold=25.0, signal=signal,
            )
        except Exception:
            pass
    return out


def nested_walkforward_optimize(
    df_full_ind: pd.DataFrame,
    bt: pv.BacktestConfig,
    n_trials: int = 200,
    seed: int = 42,
    test_frac: float = 0.2,
    embargo_days: int = 2,
    val_block_days: int = 21,
    metric: str = "sharpe",
    robustness_lambda: float = 0.5,
    min_total_trades: int = 40,
    daily_reset: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict:
    """진짜 train→test 분리형 최적화.

    df_full_ind 는 compute_indicators 가 끝난 '전체' 5분봉 프레임(세션필터 후).
    """
    if optuna is None:
        raise RuntimeError("optuna 가 설치되어 있지 않습니다.")

    boundary = bt.session_boundary_hour
    opt_df, test_df, opt_days, test_days = split_opt_holdout(
        df_full_ind, test_frac, embargo_days, boundary
    )
    blocks = forward_blocks(opt_days, val_block_days)
    if len(blocks) < 2:
        raise ValueError(f"전방 블록이 {len(blocks)}개뿐입니다. val_block_days 를 줄이세요.")

    sampler = optuna.samplers.TPESampler(
        multivariate=True, group=True, seed=seed, constraints_func=pv._constraints
    )
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    objective = make_walkforward_objective(
        opt_df, blocks, bt, metric=metric, robustness_lambda=robustness_lambda,
        min_total_trades=min_total_trades, daily_reset=daily_reset, boundary_hour=boundary,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # feasible best (제약 충족) 우선, 없으면 완료 trial 중 최고값으로 폴백
    feasible = True
    try:
        best = study.best_trial
    except ValueError:
        feasible = False
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if not completed:
            raise RuntimeError("완료된 trial 이 없습니다.")
        best = max(completed, key=lambda t: t.value)

    # ── 편향 없는 최종 평가: 홀드아웃에서 딱 한 번 ──────────────────────────
    pcfg, fcfg = cfgs_from_params(best.params)
    holdout = evaluate_on(test_df, pcfg, fcfg, bt, daily_reset)
    bases = baselines_on(test_df, bt, df_full=df_full_ind)

    def _summ(res: pv.BacktestResult) -> Dict:
        return {
            "n_trades": res.n_trades,
            "win_rate": round(res.win_rate, 2),
            "pnl_krw": round(res.total_pnl_krw, 0),
            "sharpe": round(res.sharpe_daily, 3),
            "sharpe_se": round(_sharpe_se(res, bt.annualization), 3),
            "max_dd_krw": round(res.max_drawdown_krw, 0),
            "profit_factor": round(res.profit_factor, 3) if math.isfinite(res.profit_factor) else None,
        }

    beta = bases.get("beta_long")
    beats_beta_significant = None
    if beta is not None and beta.trades is not None and len(beta.trades) >= 2:
        diff = holdout.sharpe_daily - beta.sharpe_daily
        se = _sharpe_se(holdout, bt.annualization)
        beats_beta_significant = bool(math.isfinite(se) and diff > 2.0 * se)

    result = {
        "protocol": "nested_walkforward (select on opt, evaluate once on untouched holdout)",
        "metric": metric,
        "n_opt_days": int(len(opt_days)),
        "n_test_days": int(len(test_days)),
        "n_forward_blocks": len(blocks),
        "embargo_days": embargo_days,
        "best_params": best.params,
        "selection_score_opt": best.value,         # ← in-sample 선택 기준(OOS 아님)
        "selection_total_trades": best.user_attrs.get("total_trades"),
        "constraint_satisfied": feasible,
        "n_trials": len(study.trials),
        "holdout": _summ(holdout),                 # ← 편향 없는 OOS 추정치
        "baselines_holdout": {k: _summ(v) for k, v in bases.items()},
        "holdout_beats_beta_2se": beats_beta_significant,
        "backtest_cfg": asdict(bt),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "walkforward_v3_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    return result, holdout, bases


# ════════════════════════════════════════════════════════════════════════════
# 데모 실행
# ════════════════════════════════════════════════════════════════════════════
def _load_5min(db_path: str) -> pd.DataFrame:
    df1 = pv.load_data_by_date(db_path, "futures_1min", start="2025-06-24", end="2026-06-19")
    df1 = pv.filter_day_session(df1, start="08:45", end="15:45")
    df5 = df1.resample("5min").agg({
        "OPEN": "first", "HIGH": "max", "LOW": "min", "CLOSE": "last", "VOLUME": "sum",
    }).dropna()
    return pv.compute_indicators(df5)


def main():
    import logging
    logging.getLogger("indicators.hybrid_adaptive_pivot").setLevel(logging.ERROR)
    if optuna is not None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    db = "data/duckdb/market_data.duckdb"
    bt = pv.BacktestConfig(
        multiplier=250_000, commission_pct_per_side=0.00015, slippage_ticks_per_side=1.0,
        tick_size=0.05, entry_on="next_open", annualization=252.0,
        intraday_only=True, session_boundary_hour=8, direction_mode="both",
    )
    df = _load_5min(db)
    print(f"5분봉: {len(df)}봉 | {df.index[0]} ~ {df.index[-1]}")
    result, holdout, bases = nested_walkforward_optimize(
        df, bt, n_trials=60, val_block_days=21, test_frac=0.2,
        output_dir=Path("data/backtest_results"),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
