# -*- coding: utf-8 -*-
"""추가 순열검정: 구현 가능한 대조군(피봇 확정 '이후' 무작위 시점 진입) + 진단"""
import sys, math
sys.path.append('.'); sys.path.append('..')
import numpy as np, pandas as pd
import pivot_bull_gated as gb

RNG = np.random.default_rng(7); N=5000
df = gb.load_full_data()
regime, dfirst, dlast = gb.build_regime_arrays(df)
op, cl = df["OPEN"].to_numpy(), df["CLOSE"].to_numpy()
cfg = gb.BT

t = pd.read_csv("data/backtest_results/gated_intraday_next_entry_trades.csv", parse_dates=["entry_time","exit_time"])
t["y"] = t.entry_time.dt.year

for y0,y1,label in [(2019,2025,"train"),(2026,2026,"test 2026 ★")]:
    g = t[(t.y>=y0)&(t.y<=y1)]
    e = g.entry_pos.astype(int).to_numpy(); L = dlast[e].astype(int)
    actual = np.array([(cl[x]-op[i]) - cfg.round_trip_cost_pts(op[i],cl[x]) for i,x in zip(e,L)])
    # 귀무1b: 같은 날, '확정 이후' [entry, last) 무작위 진입 → EOD (구현 가능 대조군)
    nulls=[]
    for _ in range(N):
        ent = np.array([RNG.integers(i, x) if x>i else i for i,x in zip(e,L)])
        nulls.append(np.mean((cl[L]-op[ent]) - np.array([cfg.round_trip_cost_pts(op[a],cl[x]) for a,x in zip(ent,L)])))
    nulls=np.array(nulls)
    p = float((nulls>=actual.mean()).mean())
    # 진단: 신호일의 시가→종가 (비구현, 참고용) / 확정시점 이후 잔여 드리프트
    oc = np.mean(cl[L]-op[dfirst[e].astype(int)])
    print(f"[{label}] n={len(g)}")
    print(f"  실제(확정+1 즉시 진입→EOD)      = {actual.mean():+.3f}pt")
    print(f"  귀무1b(확정 이후 무작위 진입→EOD)= {nulls.mean():+.3f}pt (p5~p95 [{np.percentile(nulls,5):+.3f},{np.percentile(nulls,95):+.3f}]) → p={p:.4f}")
    print(f"  참고: 신호일 시가→종가(비구현)   = {oc:+.3f}pt")
