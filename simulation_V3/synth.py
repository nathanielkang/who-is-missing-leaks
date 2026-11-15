import os
HERE = os.path.dirname(os.path.abspath(__file__))
"""Synthetic boxes (Table 'synth'), monotone-MAR attrition, self-masking illustration."""
import sys, json, numpy as np
from scipy.optimize import linprog
from scipy import sparse as sp
out = {}
def F(E, L, C):
    if C <= 1: return L
    if E < np.log(C) - 1e-12: return 0.0
    return min(L, (np.exp(E) - 1) / (C * np.exp(E) - 1))

def exact_box(K, lo, hi, E):
    """max sum_k tau_k over faithful policies of level E over the box of independent indicators (generic K)."""
    NP = 1 << K
    PAT = np.array([[(r >> k) & 1 for k in range(K)] for r in range(NP)])
    SUB = [(r, t) for r in range(NP) for t in range(NP) if (t & r) == t]; ns = len(SUB)
    R_OF = np.array([r for r, t in SUB]); T_OF = np.array([t for r, t in SUB])
    nv = 2 * ns + 4 * NP + K
    Q = [np.arange(ns), ns + np.arange(ns)]; U = [2 * ns + np.arange(NP), 2 * ns + NP + np.arange(NP)]
    Lw = [2 * ns + 2 * NP + np.arange(NP), 2 * ns + 3 * NP + np.arange(NP)]; tau = 2 * ns + 4 * NP + np.arange(K)
    rows = []; cols = []; vals = []; m = 0
    def add(c, v):
        nonlocal m
        rows.append(np.full(len(c), m)); cols.append(np.asarray(c)); vals.append(np.asarray(v, float)); m += 1
    for h in range(2):
        for vm in range(NP):
            g = np.where([(vm >> k) & 1 for k in range(K)], hi[h], lo[h])
            kap = np.ones(NP)
            for k in range(K): kap *= np.where(PAT[:, k] == 1, g[k], 1 - g[k])
            w = kap[R_OF]
            for t in range(NP):
                js = np.where(T_OF == t)[0]
                add(np.concatenate([Q[h][js], [U[h][t]]]), np.concatenate([w[js], [-1.0]]))
                add(np.concatenate([Q[h][js], [Lw[h][t]]]), np.concatenate([-w[js], [1.0]]))
            for k in range(K):
                js = np.where(PAT[T_OF, k] == 1)[0]
                add(np.concatenate([[tau[k]], Q[h][js]]), np.concatenate([[1.0], -w[js]]))
    eE = np.exp(E)
    for t in range(NP):
        add([U[0][t], Lw[1][t]], [1.0, -eE]); add([U[1][t], Lw[0][t]], [1.0, -eE])
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m, nv))
    er = []; ec = []; me = 0
    for h in range(2):
        for r in range(NP):
            js = np.where(R_OF == r)[0]; er += [me] * len(js); ec += list(Q[h][js]); me += 1
    Ae = sp.csr_matrix((np.ones(len(er)), (er, ec)), shape=(me, nv))
    c = np.zeros(nv); c[tau] = -1
    res = linprog(c, A_ub=A, b_ub=np.zeros(m), A_eq=Ae, b_eq=np.ones(me), bounds=[(0, None)] * nv, method='highs',
                  options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
    assert res.status == 0
    return -res.fun

c = 1.1; A = {}
for (l0, l1) in [(0.8, 0.4), (0.8, 0.7)]:
    for K in (3, 4):
        for E in (0.1, 0.2, 0.5, 1.0):
            L = min(l0, l1)
            bal = max(m * F(E / m, L, c) for m in range(1, K + 1))
            lo = np.array([[l0] * K, [l1] * K]); hi = c * lo
            ex = exact_box(K, lo, hi, E)
            A[f'({l0},{l1}),K={K},E={E}'] = (round(bal, 4), round(ex, 4), round(ex - bal, 4), K * L)
out['A'] = A
Bt = {}
for (s, E) in [(3, 0.3), (3, 0.5), (3, 1.0), (5, 0.5), (10, 1.0)]:
    L = 0.7
    prod = F(E / s, L, c) ** s; block = F(E, L ** s, c ** s)
    Bt[f'({s},{E})'] = (prod, block, block / prod if prod > 0 else None, s * np.log(c))
out['B'] = Bt
# ---- monotone MAR attrition (Theorem cert (i)) ----
def attr_level(l1, l0, T):
    j = np.arange(T + 1)
    p1 = np.where(j < T, l1 ** j * (1 - l1), l1 ** T); p0 = np.where(j < T, l0 ** j * (1 - l0), l0 ** T)
    return float(np.max(np.abs(np.log(p1 / p0))))
def attr_formula(l1, l0, T):
    return max(np.log((1 - l1) / (1 - l0)), T * np.log(l0 / l1))
rng = np.random.default_rng(20261115); REPS = int(sys.argv[1]); N = 2000
att = {}
for T in (4, 8, 12):
    true = attr_level(0.8, 0.9, T); est = np.empty(REPS)
    # plug-in from hazards: lambda_m estimated among units at risk; common hazard per wave is NOT assumed
    lev = np.empty(REPS)
    for i in range(0, REPS, 5000):
        nb = min(5000, REPS - i)
        hz = []
        for lam in (0.8, 0.9):
            # dropout wave ~ geometric truncated: number of waves observed J in {0..T}
            u = rng.random((nb, N)); J = np.minimum(np.floor(np.log(u) / np.log(lam)), T).astype(int)
            cnt = np.stack([(J == j).sum(1) for j in range(T + 1)], 1)       # nb x (T+1)
            atrisk = np.cumsum(cnt[:, ::-1], 1)[:, ::-1]                    # units with J>=j
            lamhat = atrisk[:, 1:] / np.maximum(atrisk[:, :-1], 1)          # hazard of continuing at wave m=1..T
            # pattern probs from hazards, eq. (hazard)
            cp = np.cumprod(np.concatenate([np.ones((nb, 1)), lamhat], 1), 1)  # prod_{m<=j} lambda_m
            p = cp.copy(); p[:, :T] = cp[:, :T] * (1 - lamhat)
            hz.append(p)
        with np.errstate(divide='ignore', invalid='ignore'):
            lev[i:i + nb] = np.nanmax(np.abs(np.log(hz[0] / hz[1])), 1)
    att[T] = dict(true=true, formula=attr_formula(0.8, 0.9, T), mean=float(np.mean(lev[np.isfinite(lev)])), sd=float(np.std(lev[np.isfinite(lev)])), se=float(np.std(lev[np.isfinite(lev)]) / np.sqrt(REPS)), finite=float(np.isfinite(lev).mean()))
out['attrition'] = att
# ---- self-masking (Theorem cert (ii)-(iii)) ----
rho = 0.8; pB = 0.5; sm = {}
for G in (0, 1, 2, 4):
    # observed data: P(R=1)=rho, P(Y in B | R=1)=pB.  tilted nonrespondent law: P(Y in B | R=0) = pB e^{-G}/(pB e^{-G}+1-pB)
    qB = pB * np.exp(-G) / (pB * np.exp(-G) + 1 - pB)
    PB = rho * pB + (1 - rho) * qB; gB = rho * pB / PB; gBc = rho * (1 - pB) / (1 - PB)
    Gam = abs(np.log(gB / (1 - gB)) - np.log(gBc / (1 - gBc)))
    e0 = abs(np.log((1 - gB) / (1 - gBc))); e1 = abs(np.log(gB / gBc))
    sm[G] = dict(Gamma=Gam, e0=e0, e1=e1, bound0=np.log(1 + rho * (np.exp(G) - 1)), bound1=np.log(1 + (1 - rho) * (np.exp(G) - 1)), gB=gB, gBc=gBc)
out['selfmask'] = sm
out['income_bounds'] = dict(e0=np.log(1 + 0.898 * (np.e - 1)), e1=np.log(1 + 0.102 * (np.e - 1)))
print(json.dumps(out, indent=1, default=float)); json.dump(out, open(HERE + '/results/out_synth.json', 'w'), indent=1, default=float)
