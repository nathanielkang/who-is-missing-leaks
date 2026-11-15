"""NHANES 2015-2016 public extract: leakage of four availability indicators about citizenship (Section 3.2).
Fields: family income ratio (INDFMPIR), alcohol questionnaire (ALQ101), systolic blood pressure (BPXSY1),
waist circumference (BMXWAIST). Driver: not a U.S. citizen (DMDCITZN=2), observed for everyone.
Run: python3 -I nhanes_accounting.py /path/to/n1516.csv
"""
import sys, json, math, itertools, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from _path_bootstrap import nhanes_csv, out_path, ROOT
sys.path.insert(0, str(ROOT))
from core_lib import patterns
from scipy.optimize import linprog
warnings.filterwarnings("ignore")
rng = np.random.default_rng(2026)

_csv = nhanes_csv(sys.argv[1] if len(sys.argv) > 1 else None)
d = pd.read_csv(_csv); d = d[d.RIDAGEYR >= 20].copy()
d = d[d.SMQ020.isin([1, 2]) & d.DMDCITZN.isin([1, 2]) & d.DMDEDUC2.isin([1, 2, 3, 4, 5])].copy()
d["H"] = (d.DMDCITZN == 2).astype(int)
F = ["income", "alcohol", "bp", "waist"]
d["R_income"] = d.INDFMPIR.notna().astype(int); d["R_alcohol"] = d.ALQ101.isin([1, 2]).astype(int)
d["R_bp"] = d.BPXSY1.notna().astype(int); d["R_waist"] = d.BMXWAIST.notna().astype(int)
d["age3"] = pd.cut(d.RIDAGEYR, [19, 39, 59, 80], labels=["20-39", "40-59", "60+"]).astype(str)
d["ageb"] = pd.cut(d.RIDAGEYR, [19, 29, 39, 49, 59, 69, 80]).astype(str)
d["psu"] = d.SDMVSTRA * 10 + d.SDMVPSU
K = 4; R = patterns(K); wts = np.array([8, 4, 2, 1])
out = {"n": int(len(d)), "noncitizen_share": float(d.H.mean())}


def branch(g1, g0):
    return abs(math.log(g1 / g0)), abs(math.log((1 - g1) / (1 - g0)))


def Phi(g, gp, ea):
    return math.log(max((1 + g * (math.exp(ea) - 1)) / (1 + gp * (math.exp(ea) - 1)),
                        (1 + g * (math.exp(-ea) - 1)) / (1 + gp * (math.exp(-ea) - 1))))


def joint_law(df, h):
    pat = df.loc[df.H == h, ["R_" + f for f in F]].values @ wts
    return np.bincount(pat, minlength=16) / max(1, len(pat)), np.bincount(pat, minlength=16)


def disclosed_joint(k1, k0):
    m = (k1 > 0) & (k0 > 0)
    lr = np.log(k1[m] / k0[m])
    return float(np.abs(lr).max()), int(np.argmax(np.abs(np.log(np.where(m, k1, 1) / np.where(m, k0, 1)))))


def hidden_lp(k1, k0, es, ea):
    """exact hidden-pattern level (Theorem 1) for a change of H, which is always observed; additive metric."""
    states = [(1, r) for r in R] + [(0, r) for r in R]
    P = np.r_[k1, np.zeros(16)]; Q = np.r_[np.zeros(16), k0]
    keep = (P > 0) | (Q > 0); S = [s for s, kk in zip(states, keep) if kk]; P = P[keep]; Q = Q[keep]
    n = len(S)
    def dist(u, v):
        return es * (u[0] != v[0]) + ea * sum(a != b for a, b in zip(u[1], v[1]))
    rows = []
    for i in range(n):
        for j in range(n):
            if i != j:
                r = np.zeros(n); r[i] = 1; r[j] = -math.exp(dist(S[i], S[j])); rows.append(r)
    def solve(P_, Q_):
        res = linprog(-P_, A_ub=np.array(rows), b_ub=np.zeros(len(rows)), A_eq=Q_[None, :], b_eq=[1], bounds=[(0, None)] * n, method="highs")
        return -res.fun
    return math.log(max(solve(P, Q), solve(Q, P)))


# ---------------------------------------------------------------- per-field pooled leakage with design bootstrap
X = pd.get_dummies(d[["ageb", "RIAGENDR", "RIDRETH1"]].astype(str), drop_first=True).astype(float)


def field_stats(df, Xm, f):
    g1 = df.loc[df.H == 1, "R_" + f].mean(); g0 = df.loc[df.H == 0, "R_" + f].mean()
    m = sm.Logit(df["R_" + f], sm.add_constant(pd.concat([df[["H"]], Xm.loc[df.index]], axis=1))).fit(disp=0)
    return g1, g0, m.params["H"]


strata = d.groupby("SDMVSTRA").psu.unique()
boot = []
for b in range(400):
    parts = [d[d.psu == p] for s, ps in strata.items() for p in rng.choice(ps, len(ps))]
    bd = pd.concat(parts, ignore_index=True)
    bX = pd.get_dummies(bd[["ageb", "RIAGENDR", "RIDRETH1"]].astype(str), drop_first=True).astype(float)
    row = []
    for f in F:
        try: row += list(field_stats(bd, bX, f))
        except Exception: row += [np.nan] * 3
    k1, _ = joint_law(bd, 1); k0, _ = joint_law(bd, 0)
    row += [hidden_lp(k1, k0, 0.5, 0.5)]
    boot.append(row)
boot = np.array(boot)
rows = []
for i, f in enumerate(F):
    g1, g0, lor = field_stats(d, X, f)
    e1, e0 = branch(g1, g0)
    bg1, bg0, blor = boot[:, 3 * i], boot[:, 3 * i + 1], boot[:, 3 * i + 2]
    be1 = np.abs(np.log(bg1 / bg0)); be0 = np.abs(np.log((1 - bg1) / (1 - bg0)))
    hid = 0.5 + max(Phi(g1, g0, 0.5), Phi(g0, g1, 0.5))
    strat = {}
    for a in ["20-39", "40-59", "60+"]:
        s = d[d.age3 == a]; s1, s0 = s.loc[s.H == 1, "R_" + f].mean(), s.loc[s.H == 0, "R_" + f].mean()
        strat[a] = (s1, s0) + branch(s1, s0)
    rows.append(dict(field=f, rate=float(d["R_" + f].mean()), g1=g1, g0=g0, adj_logOR=lor,
                     adj_logOR_ci=np.nanpercentile(blor, [2.5, 97.5]).tolist(), e1=e1,
                     e1_ci=np.nanpercentile(be1, [2.5, 97.5]).tolist(), e0=e0, e0_ci=np.nanpercentile(be0, [2.5, 97.5]).tolist(),
                     hidden_item_level=hid, by_age=strat))
    print("%-8s rate %.3f g1 %.3f g0 %.3f | adj logOR %+.2f [%+.2f,%+.2f] | e1 %.2f [%.2f,%.2f] e0 %.2f [%.2f,%.2f] | hidden(.5) %.3f"
          % (f, d["R_" + f].mean(), g1, g0, lor, *rows[-1]["adj_logOR_ci"], e1, *rows[-1]["e1_ci"], e0, *rows[-1]["e0_ci"], hid))
    print("          by age:", {a: "g1=%.3f g0=%.3f e1=%.2f e0=%.2f" % v for a, v in strat.items()})
out["fields"] = rows

# ---------------------------------------------------------------- joint pattern
k1, c1 = joint_law(d, 1); k0, c0 = joint_law(d, 0)
jl, jarg = disclosed_joint(k1, k0)
g1v = np.array([r["g1"] for r in rows]); g0v = np.array([r["g0"] for r in rows])
plus = sum(max(math.log(a / b), math.log((1 - a) / (1 - b))) for a, b in zip(g1v, g0v))
minus = sum(max(math.log(b / a), math.log((1 - b) / (1 - a))) for a, b in zip(g1v, g0v))
prod_level = max(plus, minus)
hid_joint = hidden_lp(k1, k0, 0.5, 0.5)
hid_prod = 0.5 + max(sum(Phi(a, b, 0.5) for a, b in zip(g1v, g0v)), sum(Phi(b, a, 0.5) for a, b in zip(g1v, g0v)))
print("joint disclosed level (cells with both counts>0) %.3f at pattern %s (counts %d / %d); independence prediction %.3f"
      % (jl, R[jarg], c1[jarg], c0[jarg], prod_level))
print("hidden-pattern level, item calibration .5/.5: joint LP %.4f [%.4f, %.4f], independence formula %.4f; sum of per-field excesses %.4f"
      % (hid_joint, *np.percentile(boot[:, -1], [2.5, 97.5]), hid_prod, sum(r["hidden_item_level"] - 0.5 for r in rows)))
# dependence: joint vs product for each group
for h, kk in ((1, k1), (0, k0)):
    g = g1v if h == 1 else g0v
    prod = np.array([np.prod([g[k] if r[k] else 1 - g[k] for k in range(K)]) for r in R])
    tv = 0.5 * np.abs(kk - prod).sum()
    print("group %d: TV(joint, product of marginals) %.4f ; P(all four present) %.4f vs product %.4f ; P(bp & waist both missing) %.4f vs product %.4f"
          % (h, tv, kk[15], prod[15], kk[[i for i, r in enumerate(R) if r[2] == 0 and r[3] == 0]].sum(),
             prod[[i for i, r in enumerate(R) if r[2] == 0 and r[3] == 0]].sum()))
    out["tv_dependence_%d" % h] = tv
# stratum-specific joint levels and flag levels
out["joint_by_age"] = {}
for a in ["20-39", "40-59", "60+"]:
    s = d[d.age3 == a]; s1, n1 = joint_law(s, 1); s0, n0 = joint_law(s, 0)
    jl_a = float(disclosed_joint(s1, s0)[0])
    out["joint_by_age"][a] = {"n1": int(n1.sum()), "n0": int(n0.sum()), "joint_disclosed": jl_a}
    print("age %s: n=(%d,%d) joint disclosed level %.3f ; income e0 %.3f" % (a, n1.sum(), n0.sum(), jl_a,
          [r for r in rows if r["field"] == "income"][0]["by_age"][a][3]))
out["joint_band"] = float(max(v["joint_disclosed"] for v in out["joint_by_age"].values()))
out.update(joint_level=jl, joint_pattern=R[jarg], joint_counts=(int(c1[jarg]), int(c0[jarg])), independence_level=prod_level,
           hidden_joint=hid_joint, hidden_joint_ci=np.percentile(boot[:, -1], [2.5, 97.5]).tolist(), hidden_independent=hid_prod)
# share of noncitizens among those with each pattern of missing income
for f in F:
    nr = d["R_" + f] == 0
    print("noncitizen share among %s nonrespondents %.3f (n=%d)" % (f, d.loc[nr, "H"].mean(), nr.sum()))
allmiss = (d[["R_" + f for f in F]].sum(1) <= 1)
print("noncitizen share among those missing at least three of four fields %.3f (n=%d)" % (d.loc[allmiss, "H"].mean(), allmiss.sum()))
_out = out_path("nhanes_accounting.json")
json.dump(out, open(_out, "w"), indent=1, default=float)
print("wrote", _out, flush=True)
