"""Additional numerical checks for version 6 (joint tasks, two response models, risk).
The checks of version 4 (v4_theory_checks.py) cover Proposition 1, Theorems 1-5, Corollary 1 and Proposition 3.
Run: python3 v6_theory_checks.py
"""
import itertools, json, math, sys
import numpy as np
from scipy.stats import binom
from _path_bootstrap import ROOT, out_path
sys.path.insert(0, str(ROOT))
from v6_lib import (patterns, product_law, rect_vertices, task_lp, full_record_level, F_front, robust_faithful_lp,
                    kernel_from_dict, released_law, two_sided_level, product_kernel)

rng = np.random.default_rng(20261011)
RES, NUM = {}, {}


def check(name, cond, info=""):
    RES[name] = bool(cond)
    print(("[PASS] " if cond else "[FAIL] ") + name + ("  " + info if info else ""))


def exact_product_loss(t0, t1, l0, l1, c):
    K = len(t0); best = 0.0
    for tm in patterns(K):
        def qv(h, g):
            th = t0 if h == 0 else t1
            return np.prod([g[k] * th[k] if tm[k] else 1 - g[k] * th[k] for k in range(K)])
        V0 = [np.array(v) for v in itertools.product(*[(l0[k], c[k] * l0[k]) for k in range(K)])]
        V1 = [np.array(v) for v in itertools.product(*[(l1[k], c[k] * l1[k]) for k in range(K)])]
        mx0, mn0 = max(qv(0, g) for g in V0), min(qv(0, g) for g in V0)
        mx1, mn1 = max(qv(1, g) for g in V1), min(qv(1, g) for g in V1)
        for a, b in ((mx0, mn1), (mx1, mn0)):
            if a > 0: best = max(best, math.log(a / b) if b > 0 else np.inf)
    return best


# 1. the task program with singleton tasks reproduces the version-4 program
ok = True
for it in range(15):
    K = 3; c = rng.uniform(1.02, 1.4, K); l0 = np.array([rng.uniform(0.3, 1 / x) for x in c]); l1 = np.array([rng.uniform(0.3, 1 / x) for x in c])
    E = rng.uniform(0.05, 1.2)
    v_old, _ = robust_faithful_lp(K, E, l0, l1, c)
    v_new, _ = task_lp(K, E, {0: ("finite", rect_vertices(l0, c)), 1: ("finite", rect_vertices(l1, c))}, [(k,) for k in range(K)])
    ok &= abs(v_old - v_new) < 1e-7
check("task program with singleton tasks equals the version-4 program", ok)

# 2. version-5 finite examples
V = {0: ("finite", rect_vertices(np.full(3, .8), np.full(3, 1.1))), 1: ("finite", rect_vertices(np.full(3, .7), np.full(3, 1.1)))}
port = {}
for E in (0.2, 0.3, 0.5):
    f, _ = task_lp(3, E, V, [(0,), (1,), (2,)])
    p, _ = task_lp(3, E, V, [(0, 1), (0, 2), (1, 2)], objective="min")
    t, _ = task_lp(3, E, V, [(0, 1, 2)])
    port[E] = (f, p, max(t, 0.0))
    print("   E=%.1f field sum %.6f  min pair %.6f  triple %.6f" % (E, f, p, max(t, 0.0)))
NUM["portfolio_K3"] = port
check("portfolio values (.999162/.157535/0, 1.494230/.381810/.343, 2.062815/.49/.343)",
      all(abs(a - b) < 2e-6 for a, b in zip(port[0.2] + port[0.3] + port[0.5],
                                               (.999162, .157535, 0, 1.494230, .381810, .343, 2.062815, .49, .343))))
# bundle mixture over the three pairs and the triple
mix = {}
for E in (0.2, 0.3, 0.5):
    u_pair = F_front(E, 0.7 ** 2, 1.1 ** 2); u_tri = F_front(E, 0.7 ** 3, 1.1 ** 3)
    # maximise min-pair coverage: alpha_p each pair, alpha_t triple, 3 alpha_p + alpha_t <= 1
    best = max(a * u_pair + (1 - 3 * a) * u_tri for a in np.linspace(0, 1 / 3, 3001))
    mix[E] = best / port[E][1]
    print("   E=%.1f bundle mixture %.6f = %.1f%% of exact min-pair" % (E, best, 100 * mix[E]))
NUM["mixture_share"] = mix
check("bundle mixture shares 98.0/89.8/70.0 %", all(abs(100 * mix[E] - v) < 0.06 for E, v in ((0.2, 98.0), (0.3, 89.8), (0.5, 70.0))))

# 3. block optimality over all faithful policies, both models, common widths
ok_blk = ok_thr = True
for it in range(60):
    K = 3; c = rng.uniform(1.02, 1.35, K)
    l0 = np.array([rng.uniform(0.3, 1 / x) for x in c]); l1 = np.array([rng.uniform(0.3, 1 / x) for x in c])
    S = tuple(sorted(rng.choice(K, size=int(rng.integers(1, K + 1)), replace=False)))
    E = rng.uniform(0.0, 1.0)
    cls = {0: ("finite", rect_vertices(l0, c)), 1: ("finite", rect_vertices(l1, c))}
    pred = F_front(E, min(np.prod(l0[list(S)]), np.prod(l1[list(S)])), np.prod(c[list(S)]))
    for within in (False, True):
        v, _ = task_lp(K, E, cls, [S], within=within)
        ok_blk &= abs(max(v, 0) - pred) < 1e-6
# unequal widths: thresholds 1/2 log(C0 C1) (model D) and max_h log C_h (model C)
for it in range(60):
    K = 2; c0 = rng.uniform(1.02, 1.5, K); c1 = rng.uniform(1.02, 1.5, K)
    l0 = np.array([rng.uniform(0.3, 1 / x) for x in c0]); l1 = np.array([rng.uniform(0.3, 1 / x) for x in c1])
    cls = {0: ("finite", rect_vertices(l0, c0)), 1: ("finite", rect_vertices(l1, c1))}
    S = (0, 1); tD = 0.5 * math.log(np.prod(c0) * np.prod(c1)); tC = max(math.log(np.prod(c0)), math.log(np.prod(c1)))
    for E, within, thr in ((tD * 0.97, False, tD), (tD * 1.03, False, tD), (tC * 0.97, True, tC), (tC * 1.03, True, tC)):
        v, _ = task_lp(K, E, cls, [S], within=within)
        ok_thr &= (v > 1e-8) == (E > thr)
check("Thm6(ii): block frontier F(E, L_S, C_S) equals the optimum over all faithful policies in models D and C", ok_blk)
check("Thm6(i): activation thresholds 1/2 log(C_0S C_1S) in model D and max_h log C_hS in model C", ok_thr)

# 4. best product policy for a joint task: [F(E/s, L, c)]^s
ok_prod = True; worst = 0
for it in range(4000):
    s = int(rng.integers(2, 4)); c = rng.uniform(1.02, 1.4); l0 = rng.uniform(0.3, 1 / c); l1 = rng.uniform(0.3, 1 / c)
    t0, t1 = rng.uniform(0.02, 1, s), rng.uniform(0.02, 1, s)
    E = exact_product_loss(t0, t1, np.full(s, l0), np.full(s, l1), np.full(s, c))
    rate = min(np.prod(l0 * t0), np.prod(l1 * t1))
    bound = F_front(E / s, min(l0, l1), c) ** s
    ok_prod &= rate <= bound + 1e-12
    worst = max(worst, rate / bound if bound > 0 else 0)
for it in range(100):
    s = int(rng.integers(2, 4)); c = rng.uniform(1.02, 1.4); l0 = rng.uniform(0.3, 1 / c); l1 = rng.uniform(0.3, 1 / c); E = rng.uniform(s * math.log(c), 1.5)
    z = F_front(E / s, min(l0, l1), c)
    lev = exact_product_loss(np.full(s, z / l0), np.full(s, z / l1), np.full(s, l0), np.full(s, l1), np.full(s, c))
    ok_prod &= lev <= E + 1e-9
check("Thm6(iii): no product policy beats [F(E/s,L,c)]^s for the joint task; balanced rates attain it", ok_prod, "max ratio %.4f" % worst)
tab = []
for s, E in ((3, .2), (3, .3), (3, .5), (3, 1.0), (5, .5), (10, 1.0)):
    p = F_front(E / s, .7, 1.1) ** s; b = F_front(E, .7 ** s, 1.1 ** s); tab.append((s, E, p, b))
    print("   s=%2d E=%.1f best product %.6f  block %.6f  ratio %s" % (s, E, p, b, ("%.2f" % (b / p)) if p > 0 else "-"))
NUM["block_vs_product"] = tab

# 5. parity construction: proper marginals equal, interaction +-1
ok_par = True
for s in (2, 3, 4, 5):
    X = np.array(list(itertools.product((0, 1), repeat=s)))
    even = X[X.sum(1) % 2 == 0]; odd = X[X.sum(1) % 2 == 1]
    for sub in itertools.chain.from_iterable(itertools.combinations(range(s), m) for m in range(1, s)):
        for vals in itertools.product((0, 1), repeat=len(sub)):
            pe = np.mean(np.all(even[:, list(sub)] == vals, axis=1)); po = np.mean(np.all(odd[:, list(sub)] == vals, axis=1))
            ok_par &= abs(pe - po) < 1e-12
    pe = np.mean(np.prod(2 * even - 1, axis=1)); po = np.mean(np.prod(2 * odd - 1, axis=1))
    ok_par &= abs(abs(pe - po) - 2) < 1e-12
check("Prop2: parity classes share every proper marginal and have interaction +1 and -1", ok_par)

# 6. latent within-group example (model C)
L = {0: [product_law([.2, .6, .7]), product_law([.8, .6, .7])], 1: [product_law([.5, .6, .7])]}
lat = []
for E, within in ((0.8, False), (0.8, True), (1.0, False), (1.0, True), (1.5, True)):
    v, Qd = task_lp(3, E, {0: ("finite", L[0]), 1: ("finite", L[1])}, [(0,), (1,), (2,)], within=within)
    lev = full_record_level(Qd, 3, L, within=True)
    Q = kernel_from_dict(Qd, 3)
    f1 = min(sum(released_law(kap, Q[h])[j] for j, t in enumerate(patterns(3)) if t[0]) for h in L for kap in L[h])
    lat.append((E, within, v, f1, lev))
    print("   latent E=%.1f %-12s fields %.6f field-1 rate %.6f full-record level %.6f" % (E, "all pairs" if within else "cross-group", v, f1, lev))
NUM["latent"] = lat
check("latent example: cross-group design leaks log 4 within group; all-pairs design erases field 1 below log 4",
      abs(lat[0][4] - math.log(4)) < 1e-6 and abs(lat[1][2] - 1.3) < 1e-6 and lat[1][4] < 1e-9 and abs(lat[4][3] - 0.2) < 1e-6)

# 7. misspecification bound
ok_mis = True; ntested = 0
for it in range(400):
    K = 2; R = patterns(K)
    kb = {h: rng.dirichlet(np.ones(4)) for h in (0, 1)}; rho = rng.uniform(0, 0.5, 2)
    Qp = {h: np.zeros((4, 4)) for h in (0, 1)}
    for h in (0, 1):
        for i, r in enumerate(R):
            js = [j for j, t in enumerate(R) if all(t[k] <= r[k] for k in range(K))]
            Qp[h][i, js] = rng.dirichlet(np.ones(len(js)))
    E0 = two_sided_level(kb[0] @ Qp[0], kb[1] @ Qp[1], 1e-14)
    laws = {}
    for h in (0, 1):
        laws[h] = []
        for m in range(4):
            k = kb[h] * np.exp(rng.uniform(-rho[h] / 2, rho[h] / 2, 4) if m < 3 else rho[h] / 2 * np.sign(rng.normal(size=4)))
            # renormalise inside the multiplicative band by bisection on a common tilt
            k = k / k.sum()
            if np.all(k <= np.exp(rho[h]) * kb[h] + 1e-12) and np.all(k >= np.exp(-rho[h]) * kb[h] - 1e-12): laws[h].append(k)
    ntested += len(laws[0]) + len(laws[1])
    outs = [(h, k @ Qp[h]) for h in laws for k in laws[h]]
    lev = max([two_sided_level(a, b, 1e-14) for (h, a) in outs for (g, b) in outs] + [0])
    ok_mis &= lev <= max(E0 + rho[0] + rho[1], 2 * rho[0], 2 * rho[1]) + 1e-9
check("Remark: full-record level <= max(E0+rho0+rho1, 2rho0, 2rho1) under multiplicative misspecification", ok_mis and ntested > 3000, "%d perturbed laws" % ntested)

# 8. Chernoff tail used in Theorem 8
ok_ch = True
for it in range(2000):
    n = int(rng.integers(5, 3000)); tau = rng.uniform(0.01, 0.9); p = rng.uniform(tau, 1)
    k = math.ceil(n * tau / 2) - 1
    pr = binom.cdf(k, n, p) if k >= 0 else 0.0
    ok_ch &= pr <= math.exp(-n * tau / 8) + 1e-12
check("Thm8: P(N_S < n tau/2) <= exp(-n tau/8) for binomial N_S with mean >= n tau", ok_ch)

print("\n%d/%d checks passed" % (sum(RES.values()), len(RES)))
_out = out_path("v6_theory_checks.json")
json.dump({"results": RES, "numbers": NUM}, open(_out, "w"), indent=1, default=float)
print("wrote", _out, flush=True)
sys.exit(0 if all(RES.values()) else 1)
