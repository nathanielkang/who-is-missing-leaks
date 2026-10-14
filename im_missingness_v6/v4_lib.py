"""Shared routines for the v4 note: faithful erasure policies, robust programs, evaluation."""
import itertools, math
import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta as betadist


def patterns(K):
    return list(itertools.product((0, 1), repeat=K))


def lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")


def product_law(g):
    K = len(g)
    return np.array([np.prod([g[k] if r[k] else 1 - g[k] for k in range(K)]) for r in patterns(K)])


# ------------------------------------------------------------------ exact robust program (Theorem 6)
def robust_faithful_lp(K, E, l0=None, l1=None, c=None, w=None, mode="rect", boxes=None, blind=False):
    """Maximise sum_k w_k tau_k over all faithful policies Q_h(t|r) (t <= r), where tau_k is the worst-case
    probability, over the uncertainty set and both groups, that field k is released; subject to robust two-sided
    pure level E for the released mask.
    mode='rect': independent responses with g_hk in [l_hk, c_k l_hk]  (constraints at rectangle vertices)
    mode='box' : joint pattern laws in {kappa: lo<=kappa<=hi, sum kappa=1}  (LP duality)
    blind=True adds Q_0 = Q_1 (attribute-blind policies)."""
    w = np.ones(K) if w is None else np.asarray(w, float)
    R = patterns(K); nR = len(R)
    idx = {}
    for h in (0, 1):
        for i, r in enumerate(R):
            for j, t in enumerate(R):
                if all(t[k] <= r[k] for k in range(K)):
                    idx[(h, i, j)] = len(idx)
    nQ = len(idx); extra = {}

    def newvar(name):
        extra[name] = nQ + len(extra)
        return extra[name]

    cons = []

    def qform(h, j, kap):
        return {idx[(h, i, j)]: kap[i] for i in range(nR) if (h, i, j) in idx and kap[i] > 0}

    def retform(h, k, kap):
        row = {}
        for i in range(nR):
            if kap[i] <= 0: continue
            for j, t in enumerate(R):
                if t[k] == 1 and (h, i, j) in idx:
                    row[idx[(h, i, j)]] = row.get(idx[(h, i, j)], 0) + kap[i]
        return row

    if mode == "rect":
        L = {0: np.asarray(l0, float), 1: np.asarray(l1, float)}
        V = {h: [product_law(np.array(v)) for v in itertools.product(*[(L[h][k], c[k] * L[h][k]) for k in range(K)])]
             for h in (0, 1)}
        for j in range(nR):
            for h in (0, 1):
                M = newvar(("M", h, j)); m = newvar(("m", h, j))
                for kap in V[h]:
                    f = qform(h, j, kap)
                    d1 = dict(f); d1[M] = -1.0; cons.append(d1)
                    d2 = {a: -b for a, b in f.items()}; d2[m] = 1.0; cons.append(d2)
            for h in (0, 1):
                cons.append({extra[("M", h, j)]: 1.0, extra[("m", 1 - h, j)]: -math.exp(E)})
        taus = [newvar(("tau", k)) for k in range(K)]
        for k in range(K):
            for h in (0, 1):
                for kap in V[h]:
                    f = retform(h, k, kap); d = {a: -b for a, b in f.items()}; d[taus[k]] = 1.0; cons.append(d)
    else:
        def max_rep(h, coeff, ub):
            lo, hi = boxes[h]
            al = [newvar(("al", h, ub, i)) for i in range(nR)]; be = [newvar(("be", h, ub, i)) for i in range(nR)]
            lam = newvar(("lam", h, ub))
            for i in range(nR):
                d = dict(coeff[i]); d[al[i]] = -1.0; d[be[i]] = 1.0; d[lam] = -1.0; cons.append(d)
            d = {al[i]: hi[i] for i in range(nR)}
            for i in range(nR): d[be[i]] = -lo[i]
            d[lam] = 1.0; d[ub] = -1.0; cons.append(d)

        def min_rep(h, coeff, lb):
            lo, hi = boxes[h]
            al = [newvar(("al2", h, lb, i)) for i in range(nR)]; be = [newvar(("be2", h, lb, i)) for i in range(nR)]
            lam = newvar(("lam2", h, lb))
            for i in range(nR):
                d = {a: -b for a, b in coeff[i].items()}; d[lam] = 1.0; d[be[i]] = 1.0; d[al[i]] = d.get(al[i], 0) - 1.0
                cons.append(d)
            d = {be[i]: -lo[i] for i in range(nR)}
            for i in range(nR): d[al[i]] = d.get(al[i], 0) + hi[i]
            d[lam] = -1.0; d[lb] = 1.0; cons.append(d)

        for j in range(nR):
            Ms = {h: newvar(("M", h, j)) for h in (0, 1)}; ms = {h: newvar(("m", h, j)) for h in (0, 1)}
            for h in (0, 1):
                coeff = [({idx[(h, i, j)]: 1.0} if (h, i, j) in idx else {}) for i in range(nR)]
                max_rep(h, coeff, Ms[h]); min_rep(h, coeff, ms[h])
            for h in (0, 1):
                cons.append({Ms[h]: 1.0, ms[1 - h]: -math.exp(E)})
        taus = [newvar(("tau", k)) for k in range(K)]
        for k in range(K):
            for h in (0, 1):
                coeff = []
                for i in range(nR):
                    dd = {}
                    for j, t in enumerate(R):
                        if t[k] == 1 and (h, i, j) in idx: dd[idx[(h, i, j)]] = 1.0
                    coeff.append(dd)
                min_rep(h, coeff, taus[k])
    nv = nQ + len(extra)
    A = np.zeros((len(cons), nv))
    for i, d in enumerate(cons):
        for a, b in d.items(): A[i, a] += b
    eqs = []
    for h in (0, 1):
        for i in range(nR):
            row = np.zeros(nv)
            for j in range(nR):
                if (h, i, j) in idx: row[idx[(h, i, j)]] = 1.0
            eqs.append((row, 1.0))
    if blind:
        for (h, i, j), col in idx.items():
            if h == 0:
                row = np.zeros(nv); row[col] = 1.0; row[idx[(1, i, j)]] = -1.0; eqs.append((row, 0.0))
    Aeq = np.array([e[0] for e in eqs]); beq = np.array([e[1] for e in eqs])
    cobj = np.zeros(nv)
    for k in range(K): cobj[extra[("tau", k)]] = -w[k]
    bounds = [(0, None)] * nQ + [((None, None) if nm[0] in ("lam", "lam2") else (0, None)) for nm in extra]
    r = lp(cobj, A, np.zeros(len(cons)), Aeq, beq, bounds)
    if r.status != 0:
        return None, None
    Q = {key: max(0.0, r.x[col]) for key, col in idx.items()}
    return -r.fun, Q


# ------------------------------------------------------------------ policies as kernels Q[h] : (nR x nR) matrices
def kernel_from_dict(Qd, K, clean=1e-9):
    """LP solutions carry solver noise of order 1e-10; entries below `clean` are set to zero before
    renormalising, so that a pattern is either released with non-negligible probability or not at all."""
    nR = 2 ** K; Q = {h: np.zeros((nR, nR)) for h in (0, 1)}
    for (h, i, j), v in Qd.items(): Q[h][i, j] = v if v > clean else 0.0
    for h in (0, 1):
        s = Q[h].sum(1, keepdims=True); s[s == 0] = 1; Q[h] = Q[h] / s
    return Q


def product_kernel(t, K):
    """t: keep probability per field for one group. Returns nR x nR matrix Q(t_out | r)."""
    R = patterns(K); nR = len(R); M = np.zeros((nR, nR))
    for i, r in enumerate(R):
        for j, s in enumerate(R):
            if any(s[k] > r[k] for k in range(K)): continue
            M[i, j] = np.prod([(t[k] if s[k] else 1 - t[k]) if r[k] else 1.0 for k in range(K)])
    return M


def released_law(kappa, Qh):
    return kappa @ Qh


def two_sided_level(q0, q1, tol=1e-8):
    """two-sided pure level of two laws on the same finite space; masses below tol in both laws are solver noise."""
    lev = 0.0
    for a, b in zip(q0, q1):
        if max(a, b) < tol: continue
        if min(a, b) < tol: return np.inf
        lev = max(lev, abs(math.log(a / b)))
    return lev


def transport_policy(k0, k1, K, w=None):
    """Optimal zero-leakage faithful policy (Theorem 4) from a meet-maximising coupling."""
    w = np.ones(K) if w is None else w
    R = patterns(K); n = len(R)
    meet = np.array([[R.index(tuple(min(r[k], s[k]) for k in range(K))) for s in R] for r in R])
    c = -np.array([sum(w[k] * min(r[k], s[k]) for k in range(K)) for r in R for s in R])
    Aeq = np.zeros((2 * n, n * n))
    for i in range(n):
        for j in range(n):
            Aeq[i, i * n + j] = 1; Aeq[n + j, i * n + j] = 1
    res = lp(c, A_eq=Aeq, b_eq=np.r_[k0, k1], bounds=[(0, None)] * (n * n))
    pi = res.x.reshape(n, n)
    Q = {h: np.zeros((n, n)) for h in (0, 1)}
    for i in range(n):
        for j in range(n):
            Q[0][i, meet[i, j]] += pi[i, j]; Q[1][j, meet[i, j]] += pi[i, j]
    for h, kap in ((0, k0), (1, k1)):
        for i in range(n):
            s = kap[i]
            if s > 1e-15: Q[h][i] /= s
            else: Q[h][i] = 0; Q[h][i, 0] = 1.0   # erase everything at zero-probability inputs
    return Q, -res.fun


# ------------------------------------------------------------------ product balancing (Theorem 5) with heterogeneous fields
def f_front(c, eta):
    return (math.exp(eta) - 1) / (c * math.exp(eta) - 1)


def fprime(c, eta):
    x = math.exp(eta)
    return (c - 1) * x / (c * x - 1) ** 2


def balanced_allocation(c, Lcap, E):
    """max sum_k tau_k s.t. sum_k phi_{c_k}(tau_k) <= E, 0<=tau_k<=Lcap_k. Enumerates active sets, water-fills."""
    K = len(c); best = (0.0, np.zeros(K))
    for A in itertools.product((0, 1), repeat=K):
        A = [k for k in range(K) if A[k]]
        if not A: continue
        base = sum(math.log(c[k]) for k in A)
        if base > E + 1e-12: continue
        lo_eta = {k: math.log(c[k]) for k in A}
        cap_eta = {}
        for k in A:
            if Lcap[k] <= 1 / (c[k] + 1): cap_eta[k] = lo_eta[k]
            else: cap_eta[k] = math.log((1 - Lcap[k]) / (1 - c[k] * Lcap[k]))
        def etas(lam):
            out = {}
            for k in A:
                if cap_eta[k] <= lo_eta[k] + 1e-15 or fprime(c[k], lo_eta[k] + 1e-12) <= lam: out[k] = lo_eta[k]; continue
                if fprime(c[k], cap_eta[k]) >= lam: out[k] = cap_eta[k]; continue
                a, b = lo_eta[k], cap_eta[k]
                for _ in range(80):
                    mid = 0.5 * (a + b)
                    a, b = (mid, b) if fprime(c[k], mid) >= lam else (a, mid)
                out[k] = 0.5 * (a + b)
            return out
        if sum(cap_eta.values()) <= E: e = cap_eta
        else:
            la, lb = 1e-12, 1e6
            for _ in range(200):
                mid = math.sqrt(la * lb)
                if sum(etas(mid).values()) > E: la = mid
                else: lb = mid
            e = etas(lb)
        tau = np.zeros(K)
        for k in A: tau[k] = min(Lcap[k], f_front(c[k], e[k]))
        if tau.sum() > best[0]: best = (tau.sum(), tau)
    return best


# ------------------------------------------------------------------ simultaneous Clopper-Pearson intervals
def cp_interval(x, n, alpha):
    lo = 0.0 if x == 0 else betadist.ppf(alpha / 2, x, n - x + 1)
    hi = 1.0 if x == n else betadist.ppf(1 - alpha / 2, x + 1, n - x)
    return lo, hi


# ------------------------------------------------------------------ Neyman-Pearson power at a fixed FPR (discrete laws)
def np_power(null, alt, fpr):
    """null, alt: arrays of masses over the same finite outcome space. Randomised LR test."""
    lr = np.where(null > 0, alt / np.maximum(null, 1e-300), np.inf)
    order = np.argsort(-lr)
    spent, power = 0.0, 0.0
    for i in order:
        if null[i] <= 0:
            power += alt[i]; continue
        if spent + null[i] <= fpr:
            spent += null[i]; power += alt[i]
        else:
            frac = (fpr - spent) / null[i]; power += frac * alt[i]; break
    return power
