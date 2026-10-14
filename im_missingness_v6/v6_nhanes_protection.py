"""NHANES-calibrated protection experiment (Section 3.3).
Truth: for each age band x in {20-39, 40-59, 60+} and citizenship group h, the joint law of the four availability
indicators (income, alcohol, blood pressure, waist) estimated from the 5,449 examined adults (Jeffreys smoothing).
A calibration sample of size m * n_{x,h} is drawn from the truth; simultaneous Clopper-Pearson intervals at total
level beta = 0.05 give (a) per-field rectangles (product model) and (b) cellwise boxes for the joint law.
Policies are designed from the calibration sample only and evaluated exactly under the truth.
Version 6 adds the four-field joint task (exact program with joint boxes), the known-law optimum at the same budget
for both objectives, and the complete-case rate. The random draws, and hence the version-4 rows, are unchanged.
Run: python3 -I v6_nhanes_protection.py /home/claude/nhanes/n1516.csv REPS
"""
import sys, json, math, time
import numpy as np, pandas as pd
from _path_bootstrap import nhanes_csv, out_path, ROOT
sys.path.insert(0, str(ROOT))
from v6_lib import (patterns, robust_faithful_lp, kernel_from_dict, product_kernel, released_law, two_sided_level,
                    transport_policy, balanced_allocation, cp_interval, np_power, task_lp)

reps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
rng = np.random.default_rng(777)
_csv = nhanes_csv(sys.argv[1] if len(sys.argv) > 1 else None)
d = pd.read_csv(_csv); d = d[d.RIDAGEYR >= 20].copy()
d = d[d.SMQ020.isin([1, 2]) & d.DMDCITZN.isin([1, 2]) & d.DMDEDUC2.isin([1, 2, 3, 4, 5])].copy()
d["H"] = (d.DMDCITZN == 2).astype(int)
Rm = np.c_[d.INDFMPIR.notna(), d.ALQ101.isin([1, 2]), d.BPXSY1.notna(), d.BMXWAIST.notna()].astype(int)
d["pat"] = Rm @ np.array([8, 4, 2, 1])
d["age3"] = pd.cut(d.RIDAGEYR, [19, 39, 59, 80], labels=["20-39", "40-59", "60+"]).astype(str)
X = ["20-39", "40-59", "60+"]; K = 4; R = patterns(K); nR = 16
n_xh = {(x, h): int(((d.age3 == x) & (d.H == h)).sum()) for x in X for h in (0, 1)}
N = len(d); Px = {x: (d.age3 == x).mean() for x in X}; Ph_x = {(x, h): n_xh[(x, h)] / (d.age3 == x).sum() for x in X for h in (0, 1)}
truth = {}
for x in X:
    for h in (0, 1):
        cnt = np.bincount(d.loc[(d.age3 == x) & (d.H == h), "pat"], minlength=16)
        truth[(x, h)] = (cnt + 0.5) / (cnt.sum() + 8.0)
# income means among income respondents (for the composition/bias metric)
mu = {(x, h): d.loc[(d.age3 == x) & (d.H == h) & d.INDFMPIR.notna(), "INDFMPIR"].mean() for x in X for h in (0, 1)}
sd_inc = d.INDFMPIR.std()
beta = 0.05; a_marg = beta / (len(X) * 2 * K); a_cell = beta / (len(X) * 2 * nR)


def marg(law):  # response probability of each field under a joint law
    return np.array([sum(law[i] for i, r in enumerate(R) if r[k]) for k in range(K)])


def evaluate(policy):
    """policy: dict x -> {0: Q0, 1: Q1} (nR x nR). Returns metrics under the truth."""
    lev, ret, U, shift, comp = 0.0, 0.0, 0.0, 0.0, 0.0
    null, alt = [], []
    for x in X:
        q = {h: released_law(truth[(x, h)], policy[x][h]) for h in (0, 1)}
        lev = max(lev, two_sided_level(q[0], q[1]))
        pk = {h: np.array([sum(q[h][j] for j, t in enumerate(R) if t[k]) for k in range(K)]) for h in (0, 1)}
        ret += Px[x] * sum(Ph_x[(x, h)] * pk[h].sum() for h in (0, 1))
        comp += Px[x] * sum(Ph_x[(x, h)] * q[h][nR - 1] for h in (0, 1))
        U += Px[x] * np.minimum(pk[0], pk[1]).sum()
        a1, a0 = Ph_x[(x, 1)] * pk[1][0], Ph_x[(x, 0)] * pk[0][0]
        if a1 + a0 > 1e-12: shift = max(shift, abs(a1 / (a1 + a0) - Ph_x[(x, 1)]))
        null.append(Ph_x[(x, 0)] * Px[x] * q[0] / sum(Ph_x[(y, 0)] * Px[y] for y in X))
        alt.append(Ph_x[(x, 1)] * Px[x] * q[1] / sum(Ph_x[(y, 1)] * Px[y] for y in X))
    power = np_power(np.concatenate(null), np.concatenate(alt), 0.01)
    return dict(level=lev, retained=ret, U=U, shift=shift, tpr=power, complete=comp)


_known = {}


def known(x, E):
    """optimum when the truth is known, at the same budget E: field objective and four-field task."""
    if (x, E) not in _known:
        cls = {h: ("finite", [truth[(x, h)]]) for h in (0, 1)}
        _, Qf = task_lp(K, E, cls, [(k,) for k in range(K)])
        _, Qt = task_lp(K, E, cls, [tuple(range(K))] + [(k,) for k in range(K)], weights=[1.0] + [0.01] * K)
        _known[(x, E)] = (kernel_from_dict(Qf, K), kernel_from_dict(Qt, K))
    return _known[(x, E)]


def erase_all():
    Z = np.zeros((nR, nR)); Z[:, 0] = 1.0
    return {0: Z, 1: Z.copy()}


def design(m, E):
    pols = {k: {} for k in ("raw", "point", "prodbal", "blind", "lp_rect", "lp_box", "blk_box", "known_field", "known_task", "oracle")}
    guar = {"lp_box": 0.0, "blk_box": 0.0}
    for x in X:
        cnt, ghat, lo_m, hi_m, lo_c, hi_c = {}, {}, {}, {}, {}, {}
        for h in (0, 1):
            n = m * n_xh[(x, h)]
            cnt[h] = rng.multinomial(n, truth[(x, h)])
            ghat[h] = marg(cnt[h] / n)
            mc = np.array([sum(cnt[h][i] for i, r in enumerate(R) if r[k]) for k in range(K)])
            iv = [cp_interval(int(mc[k]), n, a_marg) for k in range(K)]
            lo_m[h], hi_m[h] = np.array([v[0] for v in iv]), np.array([v[1] for v in iv])
            ivc = [cp_interval(int(cnt[h][i]), n, a_cell) for i in range(nR)]
            lo_c[h], hi_c[h] = np.array([v[0] for v in ivc]), np.array([v[1] for v in ivc])
        # conversion to common relative width (Proposition 2)
        c = np.maximum(hi_m[0] / np.maximum(lo_m[0], 1e-12), hi_m[1] / np.maximum(lo_m[1], 1e-12))
        l = {h: hi_m[h] / c for h in (0, 1)}
        I = np.eye(nR)
        pols["raw"][x] = {0: I, 1: I}
        tp = {h: np.minimum(ghat[0], ghat[1]) / ghat[h] for h in (0, 1)}
        pols["point"][x] = {h: product_kernel(tp[h], K) for h in (0, 1)}
        if np.all(np.isfinite(c)):
            Ub, tau = balanced_allocation(c, np.minimum(l[0], l[1]), E)
            pols["prodbal"][x] = {h: product_kernel(np.where(tau > 0, tau / l[h], 0.0), K) for h in (0, 1)}
            v, Qd = robust_faithful_lp(K, E, l[0], l[1], c)
            pols["lp_rect"][x] = kernel_from_dict(Qd, K) if Qd else erase_all()
            vb, Qb = robust_faithful_lp(K, E, l[0], l[1], c, blind=True)
            pols["blind"][x] = kernel_from_dict(Qb, K) if Qb else erase_all()
        vx, Qx = robust_faithful_lp(K, E, mode="box", boxes={0: (lo_c[0], hi_c[0]), 1: (lo_c[1], hi_c[1])})
        pols["lp_box"][x] = kernel_from_dict(Qx, K) if Qx else erase_all()
        boxcls = {0: ("box", lo_c[0], hi_c[0]), 1: ("box", lo_c[1], hi_c[1])}
        # four-field task first, field retention as a secondary objective (weights 1 and 0.01)
        vt, Qtk = task_lp(K, E, boxcls, [tuple(range(K))] + [(k,) for k in range(K)], weights=[1.0] + [0.01] * K)
        pols["blk_box"][x] = kernel_from_dict(Qtk, K) if Qtk else erase_all()
        # guaranteed four-field rate over the certified box: only the all-present input can release all four fields
        kmin = {h: max(lo_c[h][nR - 1], 1 - (hi_c[h].sum() - hi_c[h][nR - 1])) for h in (0, 1)}
        for name in ("lp_box", "blk_box"):
            guar[name] += Px[x] * min(pols[name][x][h][nR - 1, nR - 1] * kmin[h] for h in (0, 1))
        pols["known_field"][x], pols["known_task"][x] = known(x, E)
        Qt, _ = transport_policy(truth[(x, 0)], truth[(x, 1)], K)
        pols["oracle"][x] = Qt
    return pols, guar


if __name__ == "__main__":
    res = {}
    t0 = time.time()
    raw_metrics = evaluate({x: {0: np.eye(nR), 1: np.eye(nR)} for x in X})
    print("raw:", {k: round(v, 4) for k, v in raw_metrics.items()})
    for m in (1, 4):
        for E in (0.5, 1.0):
            acc = {}
            for rep in range(reps):
                pols, guar = design(m, E)
                for name, pol in pols.items():
                    met = evaluate(pol)
                    met["guaranteed_complete"] = guar.get(name, float("nan"))
                    acc.setdefault(name, []).append(met)
            summ = {}
            for name, lst in acc.items():
                lv = np.array([z["level"] for z in lst])
                summ[name] = {k: float(np.mean([z[k] for z in lst])) for k in lst[0]}
                summ[name]["level_median"] = float(np.median(lv))
                summ[name]["violation"] = float(np.mean(lv > E + 1e-9))
            res["m=%d,E=%.1f" % (m, E)] = summ
            print("\n=== calibration multiplier %d, mask budget E=%.1f (%d replications, %.0fs) ===" % (m, E, reps, time.time() - t0))
            for name in ("raw", "blind", "point", "prodbal", "lp_rect", "lp_box", "blk_box", "known_field", "known_task", "oracle"):
                s = summ[name]
                print("%-11s level mean %.3f median %.3f viol %.2f | kept %.3f | complete %.3f (guaranteed %.3f) | U %.3f | shift %.4f | TPR@.01 %.4f"
                      % (name, s["level"], s["level_median"], s["violation"], s["retained"], s["complete"], s["guaranteed_complete"], s["U"], s["shift"], s["tpr"]), flush=True)
    res["population_share"] = float(sum(Px[x] * Ph_x[(x, 1)] for x in X))
    res["raw"] = raw_metrics
    _out = out_path("v6_nhanes_protection.json")
    json.dump(res, open(_out, "w"), indent=1)
    print("wrote", _out, flush=True)
