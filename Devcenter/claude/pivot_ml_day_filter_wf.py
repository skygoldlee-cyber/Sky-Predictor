# -*- coding: utf-8 -*-
"""
pivot_ml_day_filter_wf.py — '신호일 강마감 확률' ML 필터 (순수 Walk-Forward)

배경 (pivot_intraday_edge_validation.py 결과)
  인트라데이 피봇 엣지의 본질은 '일 선택': 필터된 저점 피봇이 BULL 레짐에서
  확정되는 날은 강마감 경향 (test 2026 무작위 BULL일 대비 p=0.0002).
  → ML 필터의 목표를 거래별 승패가 아니라 "이 신호일이 강마감할 확률" 예측으로
    재정의하고, 저확률 신호일을 스킵해 기대값을 높인다.

프로토콜 (누수 방지 — ml/ 폴더 기존 파이프라인의 실패 원인 차단)
  1. 피처는 전부 '진입봉 시가 시점'까지 정보만 사용 (확정봉 종가 이하).
  2. 확장 윈도우 연도별 walk-forward: Y년 예측 시 ≤Y-1년 거래로만 학습.
     (2021~2026 예측. 2019-2020은 초기 학습 구간)
  3. threshold는 각 fold의 '학습 데이터 내' 기대값 최대화로 결정 후 고정.
     OOS 성과를 보고 threshold를 바꾸는 행위 금지.
  4. 주 모델: 정규화 로지스틱 회귀 (표본 55~173건 → 복잡 모델 금지).
     참고 모델: 소형 RandomForest (과적합 대조용).
  5. 판정: 풀링 OOS(2021-2026)에서 '필터 통과 거래 기대값 > 전체 기대값'이
     부트스트랩으로 유의한가. 승률 95% 같은 수치가 나오면 누수를 의심할 것.

사용
  python pivot_ml_day_filter_wf.py
"""
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import pivot_bull_gated as gb

RNG = np.random.default_rng(42)
N_BOOT = 10_000
PRED_YEARS = list(range(2021, 2027))
THRESH_GRID = np.arange(0.30, 0.71, 0.05)
MIN_KEEP_FRAC = 0.30          # threshold가 train 거래의 30% 미만만 남기면 무효


# ──────────────────────────────────────────────────────────────────────────────
def build_features(trades: pd.DataFrame, df: pd.DataFrame,
                   regime: np.ndarray, dfirst: np.ndarray, dlast: np.ndarray) -> pd.DataFrame:
    """진입봉 시가 시점까지의 정보만으로 피처 생성."""
    op = df["OPEN"].to_numpy(); cl = df["CLOSE"].to_numpy()
    hi = df["HIGH"].to_numpy(); lo = df["LOW"].to_numpy()
    atr = df["ATR"].to_numpy(); adx = df["ADX"].to_numpy(); st = df["ST"].to_numpy()

    # 일봉 시계열 (레짐 강도/BULL 경과일/전일 수익률 계산용, 모두 D-1까지)
    tlast = np.unique(dlast).astype(int)
    d_close = pd.Series(cl[tlast])
    ma20 = d_close.rolling(20).mean().shift(1)      # D일 기준 D-1까지
    ma60 = d_close.rolling(60).mean().shift(1)
    strength = ((ma20 - ma60) / ma60).to_numpy()
    prev_ret = d_close.pct_change().shift(1).to_numpy()
    prev_close = d_close.shift(1).to_numpy()
    bull = (ma20 > ma60).to_numpy()
    # BULL 연속 경과일
    age = np.zeros(len(bull))
    for i in range(1, len(bull)):
        age[i] = age[i - 1] + 1 if bull[i] else 0
    day_of_pos = {int(p): i for i, p in enumerate(tlast)}   # day_last pos → 일 인덱스

    rows = []
    for _, r in trades.iterrows():
        e = int(r["entry_pos"]); c = e - 1                 # 확정봉
        f = int(dfirst[e]); l = int(dlast[e])
        di = day_of_pos[l]
        a = float(atr[c]) if not math.isnan(atr[c]) else np.nan
        # 확정봉이 전일 마지막 봉인 경우(익일 첫 봉 진입): 당일 시가=진입봉 시가,
        # '일중 구조' 세그먼트는 확정봉까지의 직전 구간으로 대체
        day_open = op[f] if f <= c else op[e]
        seg0 = min(f, c)
        low_so_far = lo[seg0:c + 1].min()
        high_so_far = hi[seg0:c + 1].max()
        pc = prev_close[di]
        rows.append({
            # 레짐/일봉 맥락
            "regime_strength": strength[di],
            "bull_age": age[di],
            "prev_day_ret": prev_ret[di],
            "gap_atr": (day_open - pc) / a if (a and not math.isnan(pc)) else np.nan,
            # 일중 구조 (확정 시점까지)
            "pos_vs_open_atr": (cl[c] - day_open) / a if a else np.nan,
            "dist_from_daylow_atr": (cl[c] - low_so_far) / a if a else np.nan,
            "range_used_atr": (high_so_far - low_so_far) / a if a else np.nan,
            # 시간
            "mins_since_open": (e - f) * 5.0,
            "bars_to_close": float(l - e),
            # 지표
            "atr_norm": a / cl[c] * 100 if a else np.nan,
            "adx": float(adx[c]) if not math.isnan(adx[c]) else np.nan,
            "st_dist_pct": abs(cl[c] - st[c]) / cl[c] * 100 if not math.isnan(st[c]) else np.nan,
            # 메타/타깃
            "year": pd.Timestamp(r["entry_time"]).year,
            "net_pts": r["net_pts"],
            "y": int(r["net_pts"] > 0),
        })
    X = pd.DataFrame(rows)
    return X.fillna(X.median(numeric_only=True))


FEATS = ["regime_strength", "bull_age", "prev_day_ret", "gap_atr",
         "pos_vs_open_atr", "dist_from_daylow_atr", "range_used_atr",
         "mins_since_open", "bars_to_close", "atr_norm", "adx", "st_dist_pct"]


def pick_threshold(p_train: np.ndarray, net_train: np.ndarray) -> float:
    """train 내부에서만 threshold 결정: 통과 거래 기대값 최대화 (최소 유지비율 제약)."""
    best_t, best_e = 0.5, -1e9
    n = len(p_train)
    for t in THRESH_GRID:
        keep = p_train >= t
        if keep.sum() < max(10, MIN_KEEP_FRAC * n):
            continue
        e = net_train[keep].mean()
        if e > best_e:
            best_e, best_t = e, float(t)
    return best_t


def run_wf(X: pd.DataFrame, model_name: str):
    """확장 윈도우 walk-forward. 반환: OOS 예측/선택 마스크가 붙은 DataFrame."""
    out = []
    for Y in PRED_YEARS:
        tr = X[X["year"] < Y]
        te = X[X["year"] == Y]
        if len(te) == 0 or len(tr) < 40 or tr["y"].nunique() < 2:
            continue
        if model_name == "logit":
            mdl = make_pipeline(StandardScaler(),
                                LogisticRegression(C=0.5, max_iter=2000))
        else:
            mdl = RandomForestClassifier(
                n_estimators=200, max_depth=3, min_samples_leaf=10,
                random_state=42)
        mdl.fit(tr[FEATS], tr["y"])
        p_tr = mdl.predict_proba(tr[FEATS])[:, 1]
        thr = pick_threshold(p_tr, tr["net_pts"].to_numpy())
        p_te = mdl.predict_proba(te[FEATS])[:, 1]
        g = te.copy()
        g["p"] = p_te
        g["keep"] = p_te >= thr
        g["thr"] = thr
        out.append(g)
    return pd.concat(out) if out else pd.DataFrame()


def boot_diff_pvalue(all_net: np.ndarray, kept_net: np.ndarray) -> float:
    """귀무: 필터는 무작위 부분집합 선택과 다르지 않다.
    → 같은 크기의 무작위 부분집합 기대값 분포에서 kept 기대값의 p-value."""
    k = len(kept_net)
    if k == 0 or k == len(all_net):
        return 1.0
    obs = kept_net.mean()
    null = np.array([RNG.choice(all_net, k, replace=False).mean() for _ in range(N_BOOT)])
    return float((null >= obs).mean())


def report(oos: pd.DataFrame, label: str):
    if len(oos) == 0:
        print(f"[{label}] OOS 없음")
        return
    print(f"\n{'─' * 96}\n[{label}] 풀링 OOS {PRED_YEARS[0]}-{PRED_YEARS[-1]}")
    base_n, base_e = len(oos), oos["net_pts"].mean()
    base_wr = (oos["net_pts"] > 0).mean() * 100
    kept = oos[oos["keep"]]
    kept_e = kept["net_pts"].mean() if len(kept) else float("nan")
    kept_wr = (kept["net_pts"] > 0).mean() * 100 if len(kept) else float("nan")
    p = boot_diff_pvalue(oos["net_pts"].to_numpy(), kept["net_pts"].to_numpy())
    print(f"  베이스라인(전 거래)  : n={base_n:>3} | 기대값={base_e:+.3f}pt | 승률={base_wr:5.2f}% | PnL={oos['net_pts'].sum()*250000:+,.0f}")
    print(f"  필터 통과            : n={len(kept):>3} ({len(kept)/base_n*100:.0f}%) | 기대값={kept_e:+.3f}pt | 승률={kept_wr:5.2f}% | PnL={kept['net_pts'].sum()*250000:+,.0f}")
    print(f"  무작위 동수 선택 대비 p-value = {p:.4f}")
    print("  연도별 (기대값 전체→통과 | 유지율):")
    for y, g in oos.groupby("year"):
        k = g[g["keep"]]
        ke = k["net_pts"].mean() if len(k) else float("nan")
        print(f"    {y}: {g['net_pts'].mean():+.3f} → {ke:+.3f}pt | {len(k)}/{len(g)} (thr={g['thr'].iloc[0]:.2f})")


def main():
    df = gb.load_full_data()
    regime, dfirst, dlast = gb.build_regime_arrays(df)
    tpath = Path(__file__).parent / "data" / "backtest_results" / "gated_intraday_next_entry_trades.csv"
    trades = pd.read_csv(tpath, parse_dates=["entry_time", "exit_time"])
    X = build_features(trades, df, regime, dfirst, dlast)
    print(f"거래 {len(X)}건, 피처 {len(FEATS)}개 | 연도 분포: "
          + " ".join(f"{y}:{n}" for y, n in X['year'].value_counts().sort_index().items()))

    oos_l = run_wf(X, "logit")
    report(oos_l, "로지스틱 회귀 (주 모델)")
    oos_r = run_wf(X, "rf")
    report(oos_r, "RandomForest depth3 (대조)")

    # 해석용: 전체 데이터 로지스틱 계수 (참고 — 예측엔 사용 안 함)
    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    mdl.fit(X[FEATS], X["y"])
    coefs = pd.Series(mdl.named_steps["logisticregression"].coef_[0], index=FEATS)
    print(f"\n[해석] 로지스틱 계수 (표준화, 전체 적합 — 참고용):")
    for k, v in coefs.sort_values(key=abs, ascending=False).items():
        print(f"    {k:>22}: {v:+.3f}")

    out = Path(__file__).parent / "data" / "backtest_results"
    if len(oos_l):
        oos_l.to_csv(out / "ml_day_filter_oos_logit.csv", index=False)
    print(f"\n저장: {out / 'ml_day_filter_oos_logit.csv'}")
    print("\n※ 이 OOS 결과를 보고 threshold/피처/모델을 바꿔 재실행하면 그때부터 OOS가 아닙니다.")
    print("   개선 아이디어는 train(≤2025) 안에서 내부 CV로만 검증 후, 최종 1회 재평가하세요.")


if __name__ == "__main__":
    main()
