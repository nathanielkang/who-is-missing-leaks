"""ACS CA 2021: allocation-flag leakage about non-citizenship (parallel to NHANES Table I).

R_k = 1 if Census left the field unallocated (as-reported); 0 if allocated (edited/imputed).
Driver H = CIT==5 (not a U.S. citizen). Writes results/acs_accounting.json.
"""
from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from _path_bootstrap import acs_csv, out_path, ROOT
import sys

sys.path.insert(0, str(ROOT))
from v4_lib import patterns, two_sided_level

warnings.filterwarnings("ignore")
rng = np.random.default_rng(20261011)

_csv = acs_csv()
d = pd.read_csv(_csv)
d = d[d.AGEP >= 18].copy()
d["H"] = (d.CIT == 5).astype(int)
# as-reported = not allocated
FIELDS = [
    ("wage", "FWAGP"),
    ("pinc", "FPINCP"),
    ("school", "FSCHLP"),
    ("hins1", "FHINS1P"),
]
for name, col in FIELDS:
    d[f"R_{name}"] = (d[col] == 0).astype(int)
d["age3"] = pd.cut(d.AGEP, [17, 39, 59, 99], labels=["18-39", "40-59", "60+"]).astype(str)
K = 4
R = patterns(K)
wts = np.array([8, 4, 2, 1])
Rm = np.c_[[d[f"R_{n}"] for n, _ in FIELDS]].T
d["pat"] = Rm @ wts
out = {
    "source": str(_csv),
    "n": int(len(d)),
    "noncitizen_share": float(d.H.mean()),
    "fields": [],
}


def branch(g1, g0):
    e1 = abs(math.log(max(g1, 1e-12) / max(g0, 1e-12)))
    e0 = abs(math.log(max(1 - g1, 1e-12) / max(1 - g0, 1e-12)))
    return e1, e0


print("n=%d  noncitizen=%.3f" % (len(d), d.H.mean()))
X = pd.get_dummies(d[["age3", "SEX", "RAC1P"]].astype(str), drop_first=True).astype(float)
for name, _col in FIELDS:
    Rcol = f"R_{name}"
    g1 = float(d.loc[d.H == 1, Rcol].mean())
    g0 = float(d.loc[d.H == 0, Rcol].mean())
    e1, e0 = branch(g1, g0)
    m = sm.Logit(d[Rcol], sm.add_constant(pd.concat([d[["H"]], X], axis=1))).fit(disp=0)
    gam = float(m.params["H"])
    by_age = {}
    for a, sub in d.groupby("age3"):
        if sub.H.nunique() < 2:
            continue
        a1 = float(sub.loc[sub.H == 1, Rcol].mean())
        a0 = float(sub.loc[sub.H == 0, Rcol].mean())
        ae1, ae0 = branch(a1, a0)
        by_age[str(a)] = [a1, a0, ae1, ae0]
    row = {
        "field": name,
        "rate": float(d[Rcol].mean()),
        "g1": g1,
        "g0": g0,
        "adj_logOR": gam,
        "e1": e1,
        "e0": e0,
        "by_age": by_age,
    }
    out["fields"].append(row)
    print(
        "%s: rate=%.3f g1=%.3f g0=%.3f adj_logOR=%.3f e_resp=%.3f e_nonresp=%.3f"
        % (name, row["rate"], g1, g0, gam, e1, e0)
    )

# joint disclosed level by age band
joint_by_age = {}
band = 0.0
for a, sub in d.groupby("age3"):
    if sub.H.nunique() < 2 or len(sub) < 50:
        continue
    k1 = np.bincount(sub.loc[sub.H == 1, "pat"], minlength=16).astype(float)
    k0 = np.bincount(sub.loc[sub.H == 0, "pat"], minlength=16).astype(float)
    k1 = (k1 + 0.5) / (k1.sum() + 8.0)
    k0 = (k0 + 0.5) / (k0.sum() + 8.0)
    lev = float(two_sided_level(k0, k1))
    joint_by_age[str(a)] = {
        "n1": int((sub.H == 1).sum()),
        "n0": int((sub.H == 0).sum()),
        "joint_disclosed": lev,
    }
    band = max(band, lev)

k1 = np.bincount(d.loc[d.H == 1, "pat"], minlength=16).astype(float)
k0 = np.bincount(d.loc[d.H == 0, "pat"], minlength=16).astype(float)
k1 = (k1 + 0.5) / (k1.sum() + 8.0)
k0 = (k0 + 0.5) / (k0.sum() + 8.0)
out["joint_by_age"] = joint_by_age
out["joint_band"] = float(band)
out["joint_level"] = float(two_sided_level(k0, k1))
# independence product of field rates
g1 = np.array([out["fields"][i]["g1"] for i in range(K)])
g0 = np.array([out["fields"][i]["g0"] for i in range(K)])
from v4_lib import product_law

out["independence_level"] = float(two_sided_level(product_law(g0), product_law(g1)))
_out = out_path("acs_accounting.json")
json.dump(out, open(_out, "w"), indent=1)
print("joint_band=%.4f joint_level=%.4f indep=%.4f" % (out["joint_band"], out["joint_level"], out["independence_level"]))
print("wrote", _out)
