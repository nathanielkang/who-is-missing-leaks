"""Three release regimes for census allocation flags.

(a) published: release joint allocation pattern; level = two-sided pure level of group laws
(b) hidden: suppress flags (release only completed values); pattern channel closed → level 0
(c) erased: apply zero-leakage transport (Thm transport) and robust LP at budgets E

Also audits a Laplace mechanism on the pattern histogram (ε-DP synthetic counts) under
published vs erased kernels. Writes results/acs_flag_regimes.json (+ .csv summary).
"""
from __future__ import annotations

import csv
import json
import math
import sys

import numpy as np
import pandas as pd

from _path_bootstrap import ROOT, acs_csv, out_path

sys.path.insert(0, str(ROOT))
from v4_lib import (
    kernel_from_dict,
    patterns,
    released_law,
    robust_faithful_lp,
    transport_policy,
    two_sided_level,
)

rng = np.random.default_rng(20261011)
_csv = acs_csv()
d = pd.read_csv(_csv)
d = d[d.AGEP >= 18].copy()
d["H"] = (d.CIT == 5).astype(int)
FIELDS = [("wage", "FWAGP"), ("pinc", "FPINCP"), ("school", "FSCHLP"), ("hins1", "FHINS1P")]
for name, col in FIELDS:
    d[f"R_{name}"] = (d[col] == 0).astype(int)
K = 4
R = patterns(K)
wts = np.array([8, 4, 2, 1])
Rm = np.c_[[d[f"R_{n}"] for n, _ in FIELDS]].T
d["pat"] = Rm @ wts
nR = 16

k1_raw = np.bincount(d.loc[d.H == 1, "pat"], minlength=nR).astype(float)
k0_raw = np.bincount(d.loc[d.H == 0, "pat"], minlength=nR).astype(float)
# Jeffreys-smoothed population laws (same as NHANES protection)
truth = {
    1: (k1_raw + 0.5) / (k1_raw.sum() + 8.0),
    0: (k0_raw + 0.5) / (k0_raw.sum() + 8.0),
}
g = {
    h: np.array([sum(truth[h][i] for i, r in enumerate(R) if r[k]) for k in range(K)])
    for h in (0, 1)
}

published = float(two_sided_level(truth[0], truth[1]))
hidden = 0.0  # flags not released

# erased: zero-leakage transport
Q_tr, retained_tr = transport_policy(truth[0], truth[1], K)
q0_tr = released_law(truth[0], Q_tr[0])
q1_tr = released_law(truth[1], Q_tr[1])
erased_transport = float(two_sided_level(q0_tr, q1_tr))

# robust product LP at budgets
budgets = [0.5, 1.0]
lp_rows = []
for E in budgets:
    # lower rates and concentration factors from empirical g
    l0 = np.maximum(g[0] * 0.85, 1e-3)
    l1 = np.maximum(g[1] * 0.85, 1e-3)
    c = np.maximum(np.maximum(g[0], g[1]) / np.minimum(l0, l1), 1.05)
    val, Qd = robust_faithful_lp(K, E, l0=l0, l1=l1, c=c, mode="rect", blind=False)
    row = {"E": E, "lp_retained_fields": None if val is None else float(val)}
    if Qd is not None:
        Q = kernel_from_dict(Qd, K)
        qq0 = released_law(truth[0], Q[0])
        qq1 = released_law(truth[1], Q[1])
        row["realized_level"] = float(two_sided_level(qq0, qq1))
    lp_rows.append(row)

# Laplace DP audit on pattern counts (published histogram)
eps_grid = [0.5, 1.0, 2.0]
dp_rows = []
n0, n1 = float(k0_raw.sum()), float(k1_raw.sum())
for eps in eps_grid:
    # sensitivity 1 for count histograms; scale = 1/eps per cell (independent Laplace)
    reps = 40
    lev_pub, lev_erase = [], []
    for _ in range(reps):
        noise0 = rng.laplace(0.0, 1.0 / eps, size=nR)
        noise1 = rng.laplace(0.0, 1.0 / eps, size=nR)
        c0 = np.maximum(k0_raw + noise0, 0.0) + 0.5
        c1 = np.maximum(k1_raw + noise1, 0.0) + 0.5
        p0, p1 = c0 / c0.sum(), c1 / c1.sum()
        lev = two_sided_level(p0, p1)
        if not np.isfinite(lev):
            continue
        lev_pub.append(lev)
        # erased: apply transport to noisy laws then measure
        Qe, _ = transport_policy(p0, p1, K)
        e0 = released_law(p0, Qe[0])
        e1 = released_law(p1, Qe[1])
        lev_erase.append(two_sided_level(e0, e1))
    dp_rows.append(
        {
            "eps": eps,
            "published_level_mean": float(np.mean(lev_pub)) if lev_pub else None,
            "published_level_median": float(np.median(lev_pub)) if lev_pub else None,
            "erased_level_mean": float(np.mean(lev_erase)) if lev_erase else None,
            "erased_level_median": float(np.median(lev_erase)) if lev_erase else None,
            "reps": len(lev_pub),
        }
    )

ledger = {
    "source": str(_csv),
    "n": int(len(d)),
    "noncitizen_share": float(d.H.mean()),
    "field_response_rates": {name: float(d[f"R_{name}"].mean()) for name, _ in FIELDS},
    "regimes": {
        "published": {"level": published, "note": "joint allocation pattern released"},
        "hidden": {"level": hidden, "note": "flags suppressed; pattern channel closed"},
        "erased_transport": {
            "level": erased_transport,
            "retained_fields": float(retained_tr),
            "note": "zero-leakage meet transport (Thm transport)",
        },
    },
    "robust_lp": lp_rows,
    "laplace_pattern_dp": dp_rows,
    "raw_counts": {"n0": int(n0), "n1": int(n1)},
}

_out = out_path("acs_flag_regimes.json")
json.dump(ledger, open(_out, "w"), indent=1)
_csv_out = out_path("acs_flag_regimes.csv")
with open(_csv_out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["regime", "metric", "value"])
    w.writerow(["published", "level", published])
    w.writerow(["hidden", "level", hidden])
    w.writerow(["erased_transport", "level", erased_transport])
    w.writerow(["erased_transport", "retained_fields", retained_tr])
    for r in dp_rows:
        w.writerow([f"laplace_eps_{r['eps']}", "published_level_mean", r["published_level_mean"]])
        w.writerow([f"laplace_eps_{r['eps']}", "erased_level_mean", r["erased_level_mean"]])
print(json.dumps(ledger["regimes"], indent=2))
print("DP:", dp_rows)
print("wrote", _out, "and", _csv_out)
