"""Numerical checks for Research Note 2, version 4 (missingness_note_v4.tex).

Every theorem, corollary and proposition of Section 2 is checked on random finite
instances, together with the worked examples and table values quoted in the note.
Run:  python3 v4_theory_checks.py   (numpy, scipy)
"""
import itertools, json, math, sys
import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm
from _path_bootstrap import out_path, ROOT
sys.path.insert(0, str(ROOT))

rng = np.random.default_rng(20261010)
RES, NUM = {}, {}


def check(name, cond, info=""):
    RES[name] = bool(cond)
    print(("[PASS] " if cond else "[FAIL] ") + name + ("  " + info if info else ""))


def lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return r


# ---------------------------------------------------------------- Theorem 1 (pure)
def pure_lp(P, Q, D):
    n = len(P)
    rows, b = [], []
    for u in range(n):
        for v in range(n):
            if u != v and np.isfinite(D[u, v]):
                r = np.zeros(n); r[u] = 1; r[v] = -math.exp(D[u, v]); rows.append(r); b.append(0.0)
    r = lp(-P, np.array(rows), b, Q[None, :], [1.0], [(0, None)] * n)
    return -r.fun, r.x


def pure_dual(P, Q, D):
    n = len(P)
    pairs = [(u, v) for u in range(n) for v in range(n) if u != v and np.isfinite(D[u, v])]
    A = np.zeros((n, 1 + len(pairs))); A[:, 0] = -Q
    for j, (u, v) in enumerate(pairs):
        A[u, 1 + j] -= 1.0; A[v, 1 + j] += math.exp(D[u, v])
    c = np.zeros(1 + len(pairs)); c[0] = 1
    r = lp(c, A, -P, bounds=[(0, None)] * (1 + len(pairs)))
    return r.fun, pairs, r.x


def flow_certificate(P, Q, D, Dl, G):
    """cheapest flow feasible at capacity G: min sum pi_uv delta_uv."""
    n = len(P)
    pairs = [(u, v) for u in range(n) for v in range(n) if u != v and np.isfinite(D[u, v])]
    A = np.zeros((n, len(pairs)))
    for j, (u, v) in enumerate(pairs):
        A[u, j] -= 1.0; A[v, j] += math.exp(D[u, v])
    c = np.array([Dl[u, v] for (u, v) in pairs])
    r = lp(c, A, G * Q - P, bounds=[(0, None)] * len(pairs))
    return r.fun if r.status == 0 else np.inf


def approx_exact(P, Q, D, Dl, gamma, complement=True):
    n = len(P); rows, b = [], []
    for u in range(n):
        for v in range(n):
            if u != v and np.isfinite(D[u, v]):
                e = math.exp(D[u, v])
                r = np.zeros(n); r[u] = 1; r[v] = -e; rows.append(r); b.append(Dl[u, v])
                if complement:
                    r2 = np.zeros(n); r2[u] = -1; r2[v] = e; rows.append(r2); b.append(Dl[u, v] - 1 + e)
    r = lp(-(P - gamma * Q), np.array(rows), b, bounds=[(0, 1)] * n)
    return -r.fun, r.x


def rand_metric(n, m=3, scale=0.6):
    X = rng.normal(size=(n, m))
    return scale * np.abs(X[:, None, :] - X[None, :, :]).sum(2), X


ok_dual = ok_attain = ok_flow = ok_exactflow = ok_eventonly = ok_purecons = True
for it in range(150):
    n = int(rng.integers(3, 8))
    D, X = rand_metric(n + 4)
    Dw = D[:n, :n]
    P = rng.dirichlet(np.ones(n)); Q = rng.dirichlet(np.ones(n))
    G, a = pure_lp(P, Q, Dw)
    Gd, _, _ = pure_dual(P, Q, Dw)
    ok_dual &= abs(G - Gd) < 1e-6 * max(1, G)
    # McShane extension on W plus 4 extra points, binary kernel a/(3m)
    m = a.max()
    ext = np.array([min(m, min(a[u] * math.exp(D[z, u]) for u in range(n))) for z in range(n + 4)])
    ok_attain &= np.allclose(ext[:n], a, rtol=1e-7, atol=1e-9)
    p1 = ext / (3 * m)
    for z in range(n + 4):
        for z2 in range(n + 4):
            e = math.exp(D[z, z2])
            ok_attain &= p1[z] <= e * p1[z2] + 1e-10 and (1 - p1[z]) <= e * (1 - p1[z2]) + 1e-10
    ok_attain &= abs((P @ p1[:n]) / (Q @ p1[:n]) - G) < 1e-6 * G
    # approximate: exact program vs flow certificate vs event-only relaxation
    Dl = rng.uniform(0, 0.05, size=(n, n)); np.fill_diagonal(Dl, 0)
    for gam in (1.0, 0.5 * (1 + G), G):
        ex, _ = approx_exact(P, Q, Dw, Dl, gam)
        ev, _ = approx_exact(P, Q, Dw, Dl, gam, complement=False)
        fl = flow_certificate(P, Q, Dw, Dl, gam)
        ok_exactflow &= ex <= fl + 1e-8
        ok_eventonly &= ev >= ex - 1e-9
    # pure consistency: exact approximate program with delta=0 vanishes iff gamma >= G*
    Z = np.zeros((n, n))
    lo, _ = approx_exact(P, Q, Dw, Z, G * (1 - 1e-3))
    hi, _ = approx_exact(P, Q, Dw, Z, G * (1 + 1e-6))
    ok_purecons &= lo > 1e-9 and abs(hi) < 1e-6
check("Thm1(i-iii): primal LP = transport dual (150 random instances)", ok_dual)
check("Thm1(ii): McShane extension + binary kernel a/(3m) is d-private and attains G*", ok_attain)
check("Thm1(iv)<=(v): exact event/complement program <= cheapest-flow certificate", ok_exactflow)
check("Thm1(iv): event-only program is an upper relaxation", ok_eventonly)
check("Thm1(iv): with delta=0 the exact program vanishes iff gamma >= G*", ok_purecons)

# counterexample: complement constraint is needed
P2 = np.array([1.0, 0.0]); Q2 = np.array([0.0, 1.0]); D2 = np.array([[0, math.log(2)], [math.log(2), 0]])
ex, _ = approx_exact(P2, Q2, D2, np.zeros((2, 2)), 1.0)
ev, _ = approx_exact(P2, Q2, D2, np.zeros((2, 2)), 1.0, complement=False)
check("Thm1(iv) example: exact 1/3, event-only 1/2", abs(ex - 1 / 3) < 1e-9 and abs(ev - 0.5) < 1e-9,
      f"exact={ex:.6f} event-only={ev:.6f}")


# ---------------------------------------------------------------- Corollary 1 (frame-level calibration)
def hs(P, Q, G):
    return np.clip(P - G * Q, 0, None).sum()


def fixed_point(P, Q, e0):
    lo, hi = 1.0, math.exp(e0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1 - mid + (math.exp(e0) - 1) * hs(P, Q, mid) > 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


ok_cor = True
for it in range(200):
    n = int(rng.integers(2, 9)); e0 = rng.uniform(0.05, 3)
    P = rng.dirichlet(np.ones(n) * 0.7); Q = rng.dirichlet(np.ones(n) * 0.7)
    D = np.full((n, n), e0); np.fill_diagonal(D, 0)
    G, _ = pure_lp(P, Q, D)
    best = max((1 + (math.exp(e0) - 1) * P[list(S)].sum()) / (1 + (math.exp(e0) - 1) * Q[list(S)].sum())
               for k in range(n + 1) for S in itertools.combinations(range(n), k))
    fp = fixed_point(P, Q, e0)
    ok_cor &= abs(G - best) < 1e-7 * G and abs(G - fp) < 1e-7 * G
check("Cor1: LP = max over subsets = root of G-1=(e^eps-1)H_G(P||Q)", ok_cor)
ok_self = True
for it in range(300):
    p, q, e0 = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.01, 4)
    P = np.array([p, 0, 1 - p]); Q = np.array([0, q, 1 - q])
    D = np.full((3, 3), e0); np.fill_diagonal(D, 0)
    g1, _ = pure_lp(P, Q, D); g2, _ = pure_lp(Q, P, D)
    ok_self &= abs(math.log(max(g1, g2)) - math.log(1 + max(p, q) * (math.exp(e0) - 1))) < 1e-7
check("Cor1: one self-masking item, two-sided level log(1+max(p,q)(e^eps-1))", ok_self)


# ---------------------------------------------------------------- Theorem 2 (closed form)
def Lam(p, pp, es, ea):
    return math.log(max((1 + p * (math.exp(ea) - 1)) / (1 + pp * (math.exp(ea - es) - 1)),
                        (1 + p * (math.exp(es - ea) - 1)) / (1 + pp * (math.exp(-ea) - 1))))


def Phi(g, gp, ea):
    return Lam(g, gp, 0.0, ea)


def additive_d(u, v, es, ea):
    s = 0.0
    for a, b in zip(u, v):
        if a == "x" and b == "x": continue
        if (a == "x") != (b == "x"): s += ea
        elif a != b: s += es
    return s


ok_closed = True
for it in range(250):
    K = int(rng.integers(1, 4)); ea = rng.uniform(0.05, 1.5); es = rng.uniform(0, 2 * ea)
    p, pp = rng.uniform(0.02, 0.98, 2); g = rng.uniform(0.02, 0.98, K - 1); gp = rng.uniform(0.02, 0.98, K - 1)
    PP, QQ = {}, {}
    for r in itertools.product((0, 1), repeat=K):
        su = tuple(["a" if r[0] else "x"] + [("v%d" % k if r[k] else "x") for k in range(1, K)])
        sv = tuple(["b" if r[0] else "x"] + [("v%d" % k if r[k] else "x") for k in range(1, K)])
        wp = (p if r[0] else 1 - p) * np.prod([g[k - 1] if r[k] else 1 - g[k - 1] for k in range(1, K)])
        wq = (pp if r[0] else 1 - pp) * np.prod([gp[k - 1] if r[k] else 1 - gp[k - 1] for k in range(1, K)])
        PP[su] = PP.get(su, 0) + wp; QQ[sv] = QQ.get(sv, 0) + wq
    W = sorted(set(PP) | set(QQ))
    P = np.array([PP.get(w, 0) for w in W]); Q = np.array([QQ.get(w, 0) for w in W])
    D = np.array([[additive_d(u, v, es, ea) for v in W] for u in W])
    G, _ = pure_lp(P, Q, D)
    pred = Lam(p, pp, es, ea) + sum(Phi(g[k], gp[k], ea) for k in range(K - 1))
    ok_closed &= abs(math.log(G) - pred) < 1e-7
check("Thm2: log G* = Lambda + sum Phi for conditionally independent items (250 cases)", ok_closed)
# continuum: Phi_a -> disclosed flag level as a -> infinity, 0 at a = 0
g, gp = 0.853, 0.907
lev = lambda a: max(Phi(g, gp, a), Phi(gp, g, a))
disc = max(abs(math.log(g / gp)), abs(math.log((1 - g) / (1 - gp))))
check("Thm2 consequence: Phi_a continuum 0 at a=0 and -> disclosed level", abs(lev(1e-9)) < 1e-8 and abs(lev(40) - disc) < 1e-6,
      f"disclosed={disc:.4f}, a=.5:{lev(0.5):.4f}, a=1:{lev(1):.4f}, a=2:{lev(2):.4f}")
NUM["Phi_continuum_income_noncit"] = {"a=0.5": lev(0.5), "a=1": lev(1.0), "a=2": lev(2.0), "disclosed": disc}

# ---------------------------------------------------------------- Theorem 3(i): agnostic diameter
ok_diam = ok_kernel = True
for it in range(60):
    K = int(rng.integers(1, 4)); ea = rng.uniform(0.05, 1.5); es = rng.uniform(0, 2 * ea)
    states = list(itertools.product(*[("a%d" % k, "b%d" % k, "x") for k in range(K)]))
    y = tuple("a%d" % k for k in range(K)); y1 = tuple(["b0"] + ["a%d" % k for k in range(1, K)]); y2 = tuple("b%d" % k for k in range(K))
    mask = lambda yy: [tuple(yy[k] if r[k] else "x" for k in range(K)) for r in itertools.product((0, 1), repeat=K)]
    Dm = max(additive_d(u, v, es, ea) for u in mask(y) for v in mask(y1))
    Dw = max(additive_d(u, v, es, ea) for u in mask(y) for v in mask(y2))
    ok_diam &= abs(Dm - (max(es, ea) + (K - 1) * ea)) < 1e-12 and abs(Dw - K * max(es, ea)) < 1e-12
    ustar = max(((u, v) for u in mask(y) for v in mask(y1)), key=lambda uv: additive_d(uv[0], uv[1], es, ea))
    A1 = {z: 0.5 * math.exp(-additive_d(z, ustar[0], es, ea)) for z in states}
    for z in states:
        for z2 in states:
            e = math.exp(additive_d(z, z2, es, ea))
            ok_kernel &= A1[z] <= e * A1[z2] + 1e-12 and 1 - A1[z] <= e * (1 - A1[z2]) + 1e-12
    ok_kernel &= abs(math.log(A1[ustar[0]] / A1[ustar[1]]) - Dm) < 1e-9
check("Thm3(i): agnostic diameter max(es,ea)+(K-1)ea (item), K max(es,ea) (unit)", ok_diam)
check("Thm3(i): kernel A_z(1)=exp(-d(z,u*))/2 is d-private and attains the diameter", ok_kernel)

# ---------------------------------------------------------------- Theorem 3(iv): sensitivity bounds
ok_sens = True; worst1 = worst0 = 0
for it in range(4000):
    rho, Gb = rng.uniform(0.05, 0.95), rng.uniform(0.01, 3)
    prev = rng.uniform(0.001, 0.999); gap = rng.uniform(0, Gb)
    # find base logit b with prev*expit(b+gap)+(1-prev)*expit(b)=rho
    lo, hi = -40, 40
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = prev / (1 + math.exp(-(mid + gap))) + (1 - prev) / (1 + math.exp(-mid)) - rho
        lo, hi = (mid, hi) if f < 0 else (lo, mid)
    p1, p0 = 1 / (1 + math.exp(-(lo + gap))), 1 / (1 + math.exp(-lo))
    e1, e0 = abs(math.log(p1 / p0)), abs(math.log((1 - p1) / (1 - p0)))
    b1, b0 = math.log(rho + (1 - rho) * math.exp(Gb)), math.log(1 - rho + rho * math.exp(Gb))
    ok_sens &= e1 <= b1 + 1e-9 and e0 <= b0 + 1e-9
    if gap == gap: worst1 = max(worst1, e1 / b1); worst0 = max(worst0, e0 / b0)
check("Thm3(iv): sensitivity bounds hold (4000 binary models)", ok_sens, f"max ratio to bound: {worst1:.4f}, {worst0:.4f}")


# ---------------------------------------------------------------- Theorem 4: zero-leakage transport
def patterns(K):
    return list(itertools.product((0, 1), repeat=K))


def coupling_meet(k0, k1, K, w):
    R = patterns(K); n = len(R)
    c = -np.array([sum(w[k] * min(r[k], s[k]) for k in range(K)) for r in R for s in R])
    Aeq = np.zeros((2 * n, n * n))
    for i in range(n):
        for j in range(n):
            Aeq[i, i * n + j] = 1; Aeq[n + j, i * n + j] = 1
    r = lp(c, A_eq=Aeq, b_eq=np.r_[k0, k1], bounds=[(0, None)] * (n * n))
    return -r.fun, r.x.reshape(n, n)


def w1(k0, k1, K, w):
    R = patterns(K); n = len(R)
    c = np.array([sum(w[k] * abs(r[k] - s[k]) for k in range(K)) for r in R for s in R])
    Aeq = np.zeros((2 * n, n * n))
    for i in range(n):
        for j in range(n):
            Aeq[i, i * n + j] = 1; Aeq[n + j, i * n + j] = 1
    return lp(c, A_eq=Aeq, b_eq=np.r_[k0, k1], bounds=[(0, None)] * (n * n)).fun


def faithful_equalizing(k0, k1, K, w):
    R = patterns(K); idx = {}
    for h in (0, 1):
        for r in R:
            for t in R:
                if all(t[k] <= r[k] for k in range(K)): idx[(h, r, t)] = len(idx)
    nv = len(idx); c = np.zeros(nv); Aeq, beq = [], []
    for (h, r, t), j in idx.items():
        if h == 0: c[j] = -k0[R.index(r)] * sum(w[k] * t[k] for k in range(K))
    for h in (0, 1):
        for r in R:
            row = np.zeros(nv)
            for t in R:
                if (h, r, t) in idx: row[idx[(h, r, t)]] = 1
            Aeq.append(row); beq.append(1)
    for t in R:
        row = np.zeros(nv)
        for r in R:
            if (0, r, t) in idx: row[idx[(0, r, t)]] += k0[R.index(r)]
            if (1, r, t) in idx: row[idx[(1, r, t)]] -= k1[R.index(r)]
        Aeq.append(row); beq.append(0)
    return -lp(c, A_eq=np.array(Aeq), b_eq=beq, bounds=[(0, None)] * nv).fun


ok_tr = True
for it in range(40):
    K = int(rng.integers(1, 4)); w = rng.uniform(0.2, 2, K)
    k0 = rng.dirichlet(np.ones(2 ** K) * 0.8); k1 = rng.dirichlet(np.ones(2 ** K) * 0.8)
    um, _ = coupling_meet(k0, k1, K, w)
    Ew0 = sum(k0[i] * sum(w[k] * r[k] for k in range(K)) for i, r in enumerate(patterns(K)))
    Ew1 = sum(k1[i] * sum(w[k] * r[k] for k in range(K)) for i, r in enumerate(patterns(K)))
    ok_tr &= abs(um - 0.5 * (Ew0 + Ew1 - w1(k0, k1, K, w))) < 1e-7 and abs(um - faithful_equalizing(k0, k1, K, w)) < 1e-7
check("Thm4: max over all faithful equalizing policies = max meet coupling = (E w0+E w1-W1)/2", ok_tr)
ok_prod = True
for it in range(30):
    K = int(rng.integers(1, 4)); w = rng.uniform(0.2, 2, K); g0, g1 = rng.uniform(0, 1, K), rng.uniform(0, 1, K)
    pk = lambda g: np.array([np.prod([g[k] if r[k] else 1 - g[k] for k in range(K)]) for r in patterns(K)])
    um, _ = coupling_meet(pk(g0), pk(g1), K, w)
    ok_prod &= abs(um - sum(w * np.minimum(g0, g1))) < 1e-7
check("Thm4: independent responses give sum_k w_k min(g0k,g1k)", ok_prod)
th = 0.8
par0 = np.array([(1 + th * (-1) ** sum(r)) / 8 for r in patterns(3)]); par1 = np.array([(1 - th * (-1) ** sum(r)) / 8 for r in patterns(3)])
um, _ = coupling_meet(par0, par1, 3, np.ones(3))
k0 = np.array([0.5, 0, 0, 0.5]); k1 = np.array([0, 0.5, 0.5, 0])
u2, _ = coupling_meet(k0, k1, 2, np.ones(2))
check("Thm4 examples: parity (3-theta)/2 = 1.1, two-field 1/2", abs(um - 1.1) < 1e-9 and abs(u2 - 0.5) < 1e-9,
      f"parity raw level log9={math.log(9):.4f}")


# ---------------------------------------------------------------- Theorem 5: product frontier
def scalar_lp(l0, l1, c0, c1, eta):
    G = math.exp(eta)
    # variables a, b, tau ; maximize tau
    A = [[-1, 0, 1], [0, -1, 1], [c0, -G, 0], [-G, c1, 0], [-1, G * c1, 0], [G * c0, -1, 0]]
    b = [0, 0, 0, 0, G - 1, G - 1]
    r = lp([0, 0, -1], np.array(A, float), b, bounds=[(0, l0), (0, l1), (0, None)])
    return -r.fun


def tau_star(l0, l1, c, eta):
    if eta < math.log(c) - 1e-12: return 0.0
    return min(l0, l1, (math.exp(eta) - 1) / (c * math.exp(eta) - 1))


ok_sc = ok_act = True
for it in range(400):
    c = rng.uniform(1.01, 2.5); l0 = rng.uniform(0.05, 1 / c); l1 = rng.uniform(0.05, 1 / c); eta = rng.uniform(0, 1.5)
    ok_sc &= abs(scalar_lp(l0, l1, c, c, eta) - tau_star(l0, l1, c, eta)) < 1e-7
    c0, c1 = rng.uniform(1.01, 2.0, 2); l0 = rng.uniform(0.05, 1 / c0); l1 = rng.uniform(0.05, 1 / c1)
    v = scalar_lp(l0, l1, c0, c1, eta)
    ok_act &= (v > 1e-9) == (eta >= 0.5 * math.log(c0 * c1) - 1e-9) or abs(eta - 0.5 * math.log(c0 * c1)) < 1e-6
check("Thm5(i): scalar frontier tau*(eta) equals the robust LP (400 cases)", ok_sc)
check("Thm5(i): activation threshold (1/2)log(c0 c1) for unequal widths", ok_act)


def exact_product_loss(t0, t1, l0, l1, c):
    """robust two-sided level of a product erasure policy by brute force over masks and rectangle vertices."""
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


def phi(c, tau):
    if tau <= 0: return 0.0
    return max(math.log(c), math.log((1 - tau) / (1 - c * tau)))


ok_bal = True
for it in range(300):
    K = int(rng.integers(1, 4)); c = rng.uniform(1.02, 1.6, K)
    l0 = np.array([rng.uniform(0.05, 1 / ck) for ck in c]); l1 = np.array([rng.uniform(0.05, 1 / ck) for ck in c])
    t0, t1 = rng.uniform(0.05, 1, K), rng.uniform(0.05, 1, K)
    tau = np.minimum(l0 * t0, l1 * t1)
    E_any = exact_product_loss(t0, t1, l0, l1, c)
    E_bal = exact_product_loss(tau / l0, tau / l1, l0, l1, c)
    lb = sum(phi(c[k], tau[k]) for k in range(K))
    ok_bal &= E_any >= lb - 1e-9 and abs(E_bal - lb) < 1e-9
check("Thm5(ii): any product policy pays >= sum phi(tau_k); balancing attains it (300 cases)", ok_bal)


def homog_U(K, E, c, L):
    best = 0.0
    for m in range(1, K + 1):
        if m * math.log(c) <= E + 1e-12:
            best = max(best, m * min(L, (math.exp(E / m) - 1) / (c * math.exp(E / m) - 1)))
    return best


ok_hom = True
for it in range(25):
    c = rng.uniform(1.05, 1.5); L = rng.uniform(0.2, 1 / c); E = rng.uniform(0.05, 1.2)
    grid = np.linspace(0, L, 121); costs = np.array([phi(c, x) for x in grid])
    best = 0.0
    for i in range(len(grid)):
        for j in range(len(grid)):
            rem = E - costs[i] - costs[j]
            if rem < -1e-12: continue
            kk = np.searchsorted(-costs[::-1], -rem)  # largest grid tau with cost <= rem
            ok3 = np.where(costs <= rem + 1e-12)[0]
            best = max(best, grid[i] + grid[j] + (grid[ok3.max()] if len(ok3) else 0))
    ok_hom &= best <= homog_U(3, E, c, L) + 1e-9 and best >= homog_U(3, E, c, L) - 2 * L / 120 - 1e-9
check("Thm5(iii): homogeneous allocation formula vs grid search (K=3)", ok_hom)

# GPT-v3 synthetic table: balanced (Thm 5(iii)) and blind (exact product family, equal keeping prob.)
def blind_U(K, E, l0, l1, c):
    best = 0.0
    for m in range(1, K + 1):
        if m * math.log(c * max(l0, l1) / min(l0, l1)) > E + 1e-12: continue
        lo_t, hi_t = 0.0, 1.0
        for _ in range(100):
            mid = 0.5 * (lo_t + hi_t)
            lev = exact_product_loss(np.full(m, mid), np.full(m, mid), np.full(m, l0), np.full(m, l1), np.full(m, c)) if m <= 3 else \
                m * max(math.log(c * l0 / l1), math.log((1 - l1 * mid) / (1 - c * l0 * mid)))
            lo_t, hi_t = (mid, hi_t) if lev <= E + 1e-12 else (lo_t, mid)
        best = max(best, m * min(l0, l1) * lo_t)
    return best


tab = []
for reg, (l0, l1) in (("separated", (0.8, 0.4)), ("nearby", (0.8, 0.7))):
    for K, E in ((3, 0.5), (3, 1.0), (5, 0.5), (5, 1.0), (10, 0.5), (10, 1.0)):
        tab.append((reg, K, E, homog_U(K, E, 1.1, min(l0, l1)), blind_U(K, E, l0, l1, 1.1)))
for row in tab: print("   %-9s K=%2d E=%.1f balanced U=%.4f blind U=%.4f" % row)
NUM["synthetic_table"] = tab
exp = {("separated", 3, 0.5): (1.2, 0.0), ("separated", 3, 1.0): (1.2, 0.345), ("separated", 10, 1.0): (4.0, 0.345),
       ("nearby", 3, 0.5): (1.8167, 0.9249), ("nearby", 3, 1.0): (2.1, 1.5730), ("nearby", 10, 0.5): (2.4380, 0.9249)}
ok_tab = all(abs(dict(((r, K, E), (b, bl)) for r, K, E, b, bl in tab)[k][0] - v[0]) < 6e-4 and
             abs(dict(((r, K, E), (b, bl)) for r, K, E, b, bl in tab)[k][1] - v[1]) < 6e-4 for k, v in exp.items())
check("Thm5 synthetic table values reproduced", ok_tab)


# ---------------------------------------------------------------- Theorem 6: cardinality bound and exact robust LP
from v4_lib import robust_faithful_lp, balanced_allocation, transport_policy, product_kernel, two_sided_level, released_law


def policy_q(Qp, h, kap, K):
    R = patterns(K); q = np.zeros(len(R))
    for (hh, i, j), v in Qp.items():
        if hh == h: q[j] += kap[i] * v
    return q


def robust_level_rect(Qp, K, l0, l1, c):
    R = patterns(K)
    V = {h: [np.array([np.prod([v[k] if r[k] else 1 - v[k] for k in range(K)]) for r in R])
             for v in itertools.product(*[((l0 if h == 0 else l1)[k], c[k] * (l0 if h == 0 else l1)[k]) for k in range(K)])] for h in (0, 1)}
    qs = {h: np.array([policy_q(Qp, h, kap, K) for kap in V[h]]) for h in (0, 1)}
    lev = 0.0; supp = []
    for j in range(len(R)):
        for h in (0, 1):
            mx, mn = qs[h][:, j].max(), qs[1 - h][:, j].min()
            if mx > 1e-10: lev = max(lev, math.log(mx / mn) if mn > 1e-14 else np.inf); supp.append(R[j])
    return lev, set(supp)


ok_card = ok_lp_ge = True; gains = []
for reg, (L0, L1) in (("separated", (0.8, 0.4)), ("nearby", (0.8, 0.7))):
    for E in (0.1, 0.2, 0.3, 0.5, 1.0):
        K = 3; c = np.full(K, 1.1); l0 = np.full(K, L0); l1 = np.full(K, L1)
        val, Qp = robust_faithful_lp(K, E, l0, l1, c)
        lev, supp = robust_level_rect(Qp, K, l0, l1, c)
        ok_card &= lev <= E + 1e-6 and all(sum(t) * math.log(1.1) <= E + 1e-6 for t in supp)
        prod = homog_U(K, E, 1.1, min(L0, L1))
        ok_lp_ge &= val >= prod - 1e-7
        gains.append((reg, K, E, prod, val))
        print("   %-9s K=3 E=%.1f  product-balanced U=%.4f  exact faithful LP U=%.4f  certified level %.4f" % (reg, E, prod, val, lev))
NUM["correlated_gain_K3"] = gains
check("Thm6: exact LP policy has robust level <= E and every released pattern obeys sum log c <= E", ok_card)
check("Thm6: exact faithful optimum >= product-balanced optimum", ok_lp_ge)
check("Thm6: strict gain from correlated erasure in at least one setting", any(v > p + 1e-3 for _, _, _, p, v in gains))

# random check of the cardinality bound for arbitrary (random) faithful policies
ok_cb = True
for it in range(300):
    K = 2; c = rng.uniform(1.02, 1.6, K); l0 = np.array([rng.uniform(0.05, 1 / ck) for ck in c]); l1 = np.array([rng.uniform(0.05, 1 / ck) for ck in c])
    Qp = {}
    for h in (0, 1):
        for i, r in enumerate(patterns(K)):
            ts = [j for j, t in enumerate(patterns(K)) if all(t[k] <= r[k] for k in range(K))]
            p = rng.dirichlet(np.ones(len(ts)))
            for jj, j in enumerate(ts): Qp[(h, i, j)] = p[jj]
    lev, supp = robust_level_rect(Qp, K, l0, l1, c)
    for t in supp:
        ok_cb &= lev >= sum(math.log(c[k]) for k in range(K) if t[k]) - 1e-9
check("Thm6: random faithful policies: level >= max over released patterns of sum log c_k", ok_cb)

# box-simplex (dependence-robust) LP sanity: with tiny boxes around a product law it approaches the known-law transport value
K = 2; g0 = np.array([0.9, 0.85]); g1 = np.array([0.8, 0.75])
pk = lambda g: np.array([np.prod([g[k] if r[k] else 1 - g[k] for k in range(K)]) for r in patterns(K)])
k0, k1 = pk(g0), pk(g1)
val_box, Qb = robust_faithful_lp(K, 0.05, None, None, None, mode="box", boxes={0: (k0 - 1e-7, k0 + 1e-7), 1: (k1 - 1e-7, k1 + 1e-7)})
check("Thm6: box LP with vanishing boxes recovers the cap sum_k min(g0k,g1k)", abs(val_box - np.minimum(g0, g1).sum()) < 1e-4,
      f"{val_box:.5f} vs {np.minimum(g0, g1).sum():.5f}")



# shared-library checks used by the NHANES experiment
ok_lib = True
for it in range(40):
    K = 3; c = rng.uniform(1.02, 1.4, K); l0 = np.array([rng.uniform(0.3, 1 / ck) for ck in c]); l1 = np.array([rng.uniform(0.3, 1 / ck) for ck in c])
    E = rng.uniform(0.05, 1.5)
    U, tau = balanced_allocation(c, np.minimum(l0, l1), E)
    lev = exact_product_loss(tau / l0, tau / l1, l0, l1, c) if U > 0 else 0.0
    ok_lib &= lev <= E + 1e-7
    # homogeneous special case against the closed form
    ch = np.full(K, c[0]); Lh = np.full(K, min(l0[0], l1[0]))
    Uh, _ = balanced_allocation(ch, Lh, E)
    ok_lib &= abs(Uh - homog_U(K, E, c[0], Lh[0])) < 1e-6
    vr, Qr = robust_faithful_lp(K, E, l0, l1, c)
    vb, Qb2 = robust_faithful_lp(K, E, l0, l1, c, blind=True)
    ok_lib &= vr >= U - 1e-7 and vr >= vb - 1e-7
    k0 = rng.dirichlet(np.ones(8)); k1 = rng.dirichlet(np.ones(8))
    Qt, ut = transport_policy(k0, k1, K)
    ok_lib &= two_sided_level(released_law(k0, Qt[0]), released_law(k1, Qt[1])) < 1e-7
    ok_lib &= all(Qt[h][i, j] < 1e-12 for h in (0, 1) for i, r in enumerate(patterns(K)) for j, t in enumerate(patterns(K)) if any(t[k] > r[k] for k in range(K)))
check("Library: heterogeneous water-filling is feasible and matches the homogeneous formula; LP >= product, blind; transport policy is faithful with zero leakage", ok_lib)

# ---------------------------------------------------------------- Theorem 7: privacy and ignorability
ok_ign = True; maxratio = 0
for it in range(3000):
    E = rng.uniform(0.01, 3); nH, nV = int(rng.integers(2, 5)), int(rng.integers(2, 6))
    pj = rng.dirichlet(np.ones(nH * nV)).reshape(nH, nV)
    qh = np.exp(rng.uniform(0, E, nH)) * rng.uniform(0.01, 1)       # ratios within e^E
    qh = qh / qh.max() * rng.uniform(0.01, 1)
    f = rng.uniform(0, 1, (nH, nV))
    post = pj * qh[:, None]; post /= post.sum()
    bias = abs((post * f).sum() - (pj * f).sum())
    ok_ign &= bias <= math.tanh(E / 4) + 1e-12
    maxratio = max(maxratio, bias / math.tanh(E / 4))
# sharpness: binary H, prevalence 1/(1+sqrt(c)) on the high-weight group, f = indicator
E = 0.7; cc = math.exp(E); pi = 1 / (1 + math.sqrt(cc))
pj = np.array([[pi], [1 - pi]]); qh = np.array([cc, 1.0]); post = pj[:, 0] * qh / (pj[:, 0] * qh).sum()
sharp = abs(post[0] - pi)
check("Thm7: |E[f|T=t]-E f| <= tanh(E/4) (3000 random laws) and sharp", ok_ign and abs(sharp - math.tanh(E / 4)) < 1e-12,
      f"max ratio {maxratio:.4f}; sharp case {sharp:.6f}=tanh(E/4)={math.tanh(E/4):.6f}")

# ---------------------------------------------------------------- Proposition 2: certification
ok_hoef = True
for it in range(200):
    K = int(rng.integers(1, 12)); beta = rng.uniform(0.01, 0.2); E = rng.uniform(0.1, 3); gmin = rng.uniform(0.3, 0.99)
    N = math.ceil(math.log(4 * K / beta) / (2 * gmin ** 2 * math.tanh(E / (2 * K)) ** 2))
    bw = math.sqrt(math.log(4 * K / beta) / (2 * N))
    ghat = rng.uniform(gmin, 1, K)
    lo, hi = np.maximum(ghat - bw, 0), np.minimum(ghat + bw, 1)
    cK = hi / lo
    ok_hoef &= np.all(K * np.log(cK) <= E + 1e-9)
    ell = hi / cK.max(); ok_hoef &= np.all(ell <= lo + 1e-12)
check("Prop2: N >= log(4K/beta)/(2 gmin^2 tanh^2(E/2K)) activates all K fields; conversion contains intervals", ok_hoef)
NUM["N_example"] = {"K=4,beta=.05,E=1,gmin=.85": math.ceil(math.log(16 / 0.05) / (2 * 0.85 ** 2 * math.tanh(1 / 8) ** 2))}


# ---------------------------------------------------------------- Proposition 1: Gaussian profile numbers
def h_mu(mu, t):
    if mu == 0: return max(0.0, 1 - math.exp(t))
    return norm.cdf(mu / 2 - t / mu) - math.exp(t) * norm.cdf(-mu / 2 - t / mu)


def eps_at(mus, p, q, delta):
    def dl(eps, P_, Q_):
        return sum(P_[r] * h_mu(mus[r], eps - math.log(P_[r] / Q_[r])) for r in range(len(P_)) if P_[r] > 0)
    def solve(P_, Q_):
        lo, hi = 0.0, 30.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if dl(mid, P_, Q_) > delta else (lo, mid)
        return hi
    return max(solve(p, q), solve(q, p))


g1, g0 = 0.853, 0.907
gauss = {}
for mu in (0.5, 1.0):
    gauss[mu] = (eps_at([mu], [1.0], [1.0], 1e-5), eps_at([mu, mu], [g1, 1 - g1], [g0, 1 - g0], 1e-5),
                 eps_at([mu, 0.0], [g1, 1 - g1], [g0, 1 - g0], 1e-5))
    print("   mu=%.1f value-only %.3f both cells %.3f responders only %.3f" % ((mu,) + gauss[mu]))
NUM["gaussian"] = {str(k): v for k, v in gauss.items()}
check("Prop1 Gaussian profile values (1.993/2.218/2.043; 4.377/4.492/4.416)",
      all(abs(a - b) < 1.5e-3 for a, b in zip(gauss[0.5] + gauss[1.0], (1.993, 2.218, 2.043, 4.377, 4.492, 4.416))))

print("\n%d/%d checks passed" % (sum(RES.values()), len(RES)))
_out = out_path("v4_theory_checks.json")
json.dump({"results": RES, "numbers": NUM}, open(_out, "w"), indent=1, default=float)
print("wrote", _out, flush=True)
sys.exit(0 if all(RES.values()) else 1)
