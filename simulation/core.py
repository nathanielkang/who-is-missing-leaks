"""Core routines for 'Who Is Missing Leaks' experiments ."""
import numpy as np
from scipy.optimize import linprog
from scipy import sparse as sp
from scipy.stats import beta as _beta

K = 4
NP = 1 << K
PAT = np.array([[(r >> k) & 1 for k in range(K)] for r in range(NP)])  # PAT[r,k]
SIZE = PAT.sum(1)

def level(p, q):
    """two-sided privacy level of two finite distributions (inf if supports differ)."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    m = (p > 0) | (q > 0)
    if np.any((p[m] == 0) | (q[m] == 0)):
        return np.inf
    return float(np.max(np.abs(np.log(p[m] / q[m])))) if m.any() else 0.0

def level_occupied(p, q):
    m = (p > 0) & (q > 0)
    return float(np.max(np.abs(np.log(p[m] / q[m]))))

def occ_levels(p0, p1):
    """joint level and product-of-margins level of two pattern distributions, both taken over the
    patterns observed in both groups (the convention of Tables leak and acs_leak)."""
    m = (p0 > 0) & (p1 > 0)
    j = float(np.max(np.abs(np.log(p0[m] / p1[m]))))
    q0 = prod_from_margins(margins(p0)); q1 = prod_from_margins(margins(p1))
    return j, float(np.max(np.abs(np.log(q0[m] / q1[m]))))

def jeffreys(c):
    c = np.asarray(c, float)
    return (c + 0.5) / (c.sum() + 0.5 * len(c))

def margins(p):
    return PAT.T @ p  # g_k

def prod_from_margins(g):
    out = np.ones(NP)
    for k in range(K):
        out *= np.where(PAT[:, k] == 1, g[k], 1 - g[k])
    return out

# ---------- Section III: closed forms ----------
def Lambda(g, gp, es, ea):
    a = (1 + g * (np.exp(ea) - 1)) / (1 + gp * (np.exp(ea - es) - 1))
    b = (1 + g * (np.exp(es - ea) - 1)) / (1 + gp * (np.exp(-ea) - 1))
    return np.log(np.maximum(a, b))

def Phi(g, gp, ea):
    return Lambda(g, gp, 0.0, ea)

def hidden_lp(P, Q, d):
    """G* of eq. (lp): max P.a s.t. Q.a=1, a_u<=exp(d_uv) a_v. returns log G*."""
    n = len(P)
    rows = []; cols = []; vals = []; m = 0
    for u in range(n):
        for v in range(n):
            if u != v and np.isfinite(d[u, v]):
                rows += [m, m]; cols += [u, v]; vals += [1.0, -np.exp(d[u, v])]; m += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(m, n))
    res = linprog(-np.asarray(P), A_ub=A, b_ub=np.zeros(m), A_eq=np.asarray(Q)[None, :], b_eq=[1.0],
                  bounds=[(0, None)] * n, method='highs')
    if res.status == 3:
        return np.inf
    assert res.status == 0, res.message
    return float(np.log(-res.fun))

def hidden_joint_level(k0, k1, es, ea):
    """two-sided worst-case hidden level; neighbours differ in the always-observed protected item only."""
    n = 2 * NP
    d = np.zeros((n, n))
    ham = (PAT[:, None, :] != PAT[None, :, :]).sum(2)
    for h in range(2):
        for hp in range(2):
            d[h * NP:(h + 1) * NP, hp * NP:(hp + 1) * NP] = es * (h != hp) + ea * ham
    P = np.concatenate([k0, np.zeros(NP)]); Q = np.concatenate([np.zeros(NP), k1])
    return max(hidden_lp(P, Q, d), hidden_lp(Q, P, d))

def hidden_closed(g0, g1, es, ea):
    a = es + sum(Phi(g0[k], g1[k], ea) for k in range(len(g0)))
    b = es + sum(Phi(g1[k], g0[k], ea) for k in range(len(g0)))
    return max(a, b)

# ---------- Section IV: policies ----------
# faithful policy Q_h(t|r), t<=r : index list
SUB = [(r, t) for r in range(NP) for t in range(NP) if (t & r) == t]
NSUB = len(SUB)
SUBIDX = {rt: i for i, rt in enumerate(SUB)}
# M_T[t, j] = 1 if SUB[j] has released pattern t ; R_of[j] = r
R_OF = np.array([r for r, t in SUB]); T_OF = np.array([t for r, t in SUB])

def released(kappa, Qv):
    """q(t) = sum_r kappa(r) Q(t|r); Qv is a vector over SUB."""
    q = np.zeros(NP)
    np.add.at(q, T_OF, kappa[R_OF] * Qv)
    return q

def rates(q):
    """per-field release rates and all-four rate from released-pattern distribution."""
    return PAT.T @ q, q[NP - 1]

def identity_policy():
    Qv = np.zeros(NSUB)
    for r in range(NP):
        Qv[SUBIDX[(r, r)]] = 1.0
    return Qv

def product_policy(theta):
    """release each observed field k independently w.p. theta[k]."""
    Qv = np.zeros(NSUB)
    for j, (r, t) in enumerate(SUB):
        p = 1.0
        for k in range(K):
            if (r >> k) & 1:
                p *= theta[k] if (t >> k) & 1 else (1 - theta[k])
        Qv[j] = p
    return Qv

def block_policy(prob):
    """release all K fields w.p. prob if all observed, nothing otherwise."""
    Qv = np.zeros(NSUB)
    for r in range(NP):
        if r == NP - 1:
            Qv[SUBIDX[(r, r)]] = prob; Qv[SUBIDX[(r, 0)]] = 1 - prob
        else:
            Qv[SUBIDX[(r, 0)]] = 1.0
    return Qv

def F(E, L, C):
    if C <= 1 + 1e-15:
        return L
    if E < np.log(C) - 1e-12:
        return 0.0
    return min(L, (np.exp(E) - 1) / (C * np.exp(E) - 1))

def phi_c(c, tau):
    if tau <= 0:
        return 0.0
    if c * tau >= 1:
        return np.inf if c > 1 else 0.0
    return max(np.log(c), np.log((1 - tau) / (1 - c * tau)))

def f_c(c, eta):
    return (np.exp(eta) - 1) / (c * np.exp(eta) - 1)

def balanced_alloc(E, c, L):
    """maximise sum tau_k s.t. sum phi_{c_k}(tau_k) <= E, tau_k <= L_k (Theorem joint (iii), balanced policy).
    returns tau (len K). Enumerates active sets; within a set equalises marginal gains by bisection."""
    Kk = len(c); best = np.zeros(Kk); bestv = 0.0
    c = np.asarray(c, float); L = np.asarray(L, float)
    lc = np.log(c)
    for mask in range(1, 1 << Kk):
        A = [k for k in range(Kk) if (mask >> k) & 1]
        if lc[A].sum() > E + 1e-12:
            continue
        # eta cap: f_c(eta)=L  -> eta = log((1-L)/(1-cL)) (if cL<1)
        cap = np.array([np.log((1 - L[k]) / (1 - c[k] * L[k])) if c[k] * L[k] < 1 else np.inf for k in A])
        cap = np.maximum(cap, lc[A])
        lo_eta = lc[A].copy()
        if c[A].max() <= 1 + 1e-15 and False:
            pass
        cA = c[A]
        def eta_of(lam):
            # f'(eta) = x (c-1)/(c x -1)^2 = lam with x=e^eta ; larger root of lam c^2 x^2 - (2 lam c + c - 1) x + lam = 0
            with np.errstate(divide='ignore', invalid='ignore'):
                bq = 2 * lam * cA + cA - 1
                x = (bq + np.sqrt(np.maximum(bq * bq - 4 * lam * lam * cA * cA, 0.0))) / (2 * lam * cA * cA)
                eta = np.where(cA > 1 + 1e-15, np.log(x), lo_eta)
            return np.clip(eta, lo_eta, np.minimum(cap, 50.0))
        capfin = np.minimum(cap, 50.0)
        if capfin.sum() <= E:
            eta = capfin
        else:
            a, b = 0.0, 1e6
            for _ in range(100):
                m = 0.5 * (a + b)
                if eta_of(m).sum() > E: a = m
                else: b = m
            eta = eta_of(b)
        tau = np.array([min(L[k], f_c(c[k], e)) if c[k] > 1 + 1e-15 else L[k] for k, e in zip(A, eta)])
        if tau.sum() > bestv + 1e-12:
            bestv = tau.sum(); best = np.zeros(Kk); best[A] = tau
    return best

def cp_interval(x, n, alpha):
    lo = _beta.ppf(alpha / 2, x, n - x + 1) if x > 0 else 0.0
    hi = _beta.ppf(1 - alpha / 2, x + 1, n - x) if x < n else 1.0
    return lo, hi
