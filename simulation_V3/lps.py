"""Exact programs of Section IV-C (faithful policies of level E over a box / joint box)."""
import numpy as np
from scipy.optimize import linprog
from scipy import sparse as sp
from core import *

# index helpers
J_BY_T = [np.where(T_OF == t)[0] for t in range(NP)]
J_FIELD = [np.where(PAT[T_OF, k] == 1)[0] for k in range(K)]
J_ALL = np.where(T_OF == NP - 1)[0]

# ---- helper: sparse builder ----
class LP:
    def __init__(self):
        self.nv = 0; self.lb = []; self.ub = []
        self.r = []; self.c = []; self.v = []; self.b = []; self.m = 0
        self.er = []; self.ec = []; self.ev = []; self.eb = []; self.me = 0
    def var(self, n, lb=0.0, ub=None):
        i = self.nv; self.nv += n; self.lb += [lb] * n; self.ub += [ub] * n; return np.arange(i, i + n)
    def le(self, cols, vals, rhs=0.0):
        self.r.append(np.full(len(cols), self.m)); self.c.append(np.asarray(cols)); self.v.append(np.asarray(vals, float)); self.b.append(rhs); self.m += 1
    def le_many(self, rows_local, cols, vals, nrows, rhs=0.0):
        self.r.append(np.asarray(rows_local) + self.m); self.c.append(np.asarray(cols)); self.v.append(np.asarray(vals, float))
        self.b += [rhs] * nrows if np.isscalar(rhs) else list(rhs); self.m += nrows
    def eq(self, cols, vals, rhs):
        self.er.append(np.full(len(cols), self.me)); self.ec.append(np.asarray(cols)); self.ev.append(np.asarray(vals, float)); self.eb.append(rhs); self.me += 1
    def solve(self, obj):
        A = sp.csr_matrix((np.concatenate(self.v), (np.concatenate(self.r), np.concatenate(self.c))), shape=(self.m, self.nv))
        Ae = sp.csr_matrix((np.concatenate(self.ev), (np.concatenate(self.er), np.concatenate(self.ec))), shape=(self.me, self.nv))
        res = linprog(-obj, A_ub=A, b_ub=np.asarray(self.b, float), A_eq=Ae, b_eq=np.asarray(self.eb, float),
                      bounds=list(zip(self.lb, self.ub)), method='highs',
                      options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
        return res
    def add_le_row_obj(self, obj, rhs):
        """add constraint obj.x >= rhs"""
        nz = np.nonzero(obj)[0]; self.le(nz, -obj[nz], -rhs)

def _policy_vars(lp):
    Q = [lp.var(NSUB, 0.0, 1.0), lp.var(NSUB, 0.0, 1.0)]
    for h in range(2):
        for r in range(NP):
            js = np.where(R_OF == r)[0]
            lp.eq(Q[h][js], np.ones(len(js)), 1.0)
    return Q


def clean_pair(Q0, Q1, tol=1e-6):
    """remove solver noise from a pair of faithful policies: a released pattern t that one group
    (numerically) never uses is dropped in both groups; the freed mass goes to the largest entry of the row."""
    Qs = [np.clip(Q0, 0, 1).copy(), np.clip(Q1, 0, 1).copy()]
    for t in range(NP):
        js = J_BY_T[t]
        if min(Qs[0][js].max(), Qs[1][js].max()) < tol:
            Qs[0][js] = 0.0; Qs[1][js] = 0.0
    for Qv in Qs:
        for r in range(NP):
            js = np.where(R_OF == r)[0]; j = js[np.argmax(Qv[js])]
            Qv[j] += 1.0 - Qv[js].sum()
    return Qs[0], Qs[1]

def exact_product_box(lo, hi, E, objective='fields'):
    """lo,hi: arrays [h,k] of the box for response probabilities (independent indicators)."""
    lp = LP(); Q = _policy_vars(lp)
    U = [lp.var(NP, 0.0, None), lp.var(NP, 0.0, None)]
    Lw = [lp.var(NP, 0.0, None), lp.var(NP, 0.0, None)]
    tau = lp.var(K, 0.0, None); tauS = lp.var(1, 0.0, None)
    eE = np.exp(E)
    for h in range(2):
        for vmask in range(1 << K):
            g = np.where([(vmask >> k) & 1 for k in range(K)], hi[h], lo[h])
            kap = prod_from_margins(g)
            w = kap[R_OF]
            # q(t) - U(t) <= 0 ; Lw(t) - q(t) <= 0
            rows = T_OF
            lp.le_many(np.concatenate([rows, np.arange(NP)]), np.concatenate([Q[h], U[h]]), np.concatenate([w, -np.ones(NP)]), NP)
            lp.le_many(np.concatenate([rows, np.arange(NP)]), np.concatenate([Q[h], Lw[h]]), np.concatenate([-w, np.ones(NP)]), NP)
            for k in range(K):
                js = J_FIELD[k]
                lp.le(np.concatenate([[tau[k]], Q[h][js]]), np.concatenate([[1.0], -w[js]]))
            lp.le(np.concatenate([tauS, Q[h][J_ALL]]), np.concatenate([[1.0], -w[J_ALL]]))
    for t in range(NP):
        lp.le([U[0][t], Lw[1][t]], [1.0, -eE]); lp.le([U[1][t], Lw[0][t]], [1.0, -eE])
    return _solve_lex(lp, Q, tau, tauS, objective)

def _solve_lex(lp, Q, tau, tauS, objective):
    o_f = np.zeros(lp.nv); o_f[tau] = 1.0
    o_s = np.zeros(lp.nv); o_s[tauS] = 1.0
    if objective == 'fields':
        r1 = lp.solve(o_f); assert r1.status == 0, r1.message
        lp.add_le_row_obj(o_f, -r1.fun - 1e-9)
        res = lp.solve(o_s)
    else:
        r1 = lp.solve(o_s); assert r1.status == 0, r1.message
        lp.add_le_row_obj(o_s, -r1.fun - 1e-9)
        res = lp.solve(o_f)
    assert res.status == 0, res.message
    x = res.x
    Qa, Qb = clean_pair(x[Q[0]], x[Q[1]])
    return Qa, Qb, x[tau].copy(), float(x[tauS][0])

def _robust_max(lp, lo, hi, cols, coefs_by_r, target_col, sign=+1):
    """adds:  max_{kappa in joint box} sum_r kappa_r a_r  <= target   (sign=+1)
       or     min_{kappa} sum_r kappa_r a_r >= target                 (sign=-1)
       where a_r = sum over (cols[r], coefs_by_r[r]) of policy variables, dualised.
       Cells with a_r = 0 are merged into one cell with bounds [sum lo, sum hi]; this is the exact
       projection of {lo<=kappa<=hi, sum kappa=1}.
       max: mu + hi.al - lo.be <= target,  mu + al_r - be_r >= a_r
       min: nu - hi.ga + lo.de >= target,  nu - ga_r + de_r <= a_r"""
    act = [r for r in range(NP) if len(cols[r]) > 0]
    ina = [r for r in range(NP) if len(cols[r]) == 0]
    los = list(lo[act]); his = list(hi[act]); cc = [cols[r] for r in act]; cf = [coefs_by_r[r] for r in act]
    if ina:
        los.append(lo[ina].sum()); his.append(min(1.0, hi[ina].sum())); cc.append(np.array([], int)); cf.append(np.array([]))
    los = np.array(los); his = np.array(his); n = len(los)
    mu = lp.var(1, None, None); al = lp.var(n, 0.0, None); be = lp.var(n, 0.0, None)
    if sign > 0:
        lp.le(np.concatenate([mu, al, be, [target_col]]), np.concatenate([[1.0], his, -los, [-1.0]]))
        for i in range(n):
            lp.le(np.concatenate([cc[i], mu, [al[i]], [be[i]]]), np.concatenate([cf[i], [-1.0, -1.0, 1.0]]))
    else:
        lp.le(np.concatenate([[target_col], mu, al, be]), np.concatenate([[1.0], [-1.0], his, -los]))
        for i in range(n):
            lp.le(np.concatenate([mu, [al[i]], [be[i]], cc[i]]), np.concatenate([[1.0, -1.0, 1.0], -np.asarray(cf[i])]))

def exact_joint_box(lo, hi, E, objective='fields'):
    """lo,hi: arrays [h, 16] of intervals for the pattern probabilities of each group."""
    lp = LP(); Q = _policy_vars(lp)
    U = [lp.var(NP, 0.0, None), lp.var(NP, 0.0, None)]
    Lw = [lp.var(NP, 0.0, None), lp.var(NP, 0.0, None)]
    tau = lp.var(K, 0.0, None); tauS = lp.var(1, 0.0, None)
    eE = np.exp(E)
    for h in range(2):
        for t in range(NP):
            cols = []; cf = []
            for r in range(NP):
                if (t & r) == t:
                    cols.append(np.array([Q[h][SUBIDX[(r, t)]]])); cf.append(np.array([1.0]))
                else:
                    cols.append(np.array([], int)); cf.append(np.array([]))
            _robust_max(lp, lo[h], hi[h], cols, cf, U[h][t], +1)
            _robust_max(lp, lo[h], hi[h], cols, cf, Lw[h][t], -1)
        for k in range(K):
            cols = []; cf = []
            for r in range(NP):
                js = [SUBIDX[(r, t)] for t in range(NP) if (t & r) == t and (t >> k) & 1]
                cols.append(Q[h][js] if js else np.array([], int)); cf.append(np.ones(len(js)))
            _robust_max(lp, lo[h], hi[h], cols, cf, tau[k], -1)
        cols = [np.array([Q[h][SUBIDX[(r, NP - 1)]]]) if r == NP - 1 else np.array([], int) for r in range(NP)]
        cf = [np.ones(len(c)) for c in cols]
        _robust_max(lp, lo[h], hi[h], cols, cf, tauS[0], -1)
    for t in range(NP):
        lp.le([U[0][t], Lw[1][t]], [1.0, -eE]); lp.le([U[1][t], Lw[0][t]], [1.0, -eE])
    return _solve_lex(lp, Q, tau, tauS, objective)

def transport_zero(k0, k1, w=None):
    """Theorem transport: max E w(R0 ^ R1) over couplings; lexicographic second objective: P(R0^R1 = all)."""
    w = np.ones(K) if w is None else w
    n = NP * NP
    gain = np.array([[w @ PAT[r & rp] for rp in range(NP)] for r in range(NP)]).ravel()
    gs = np.array([[float((r & rp) == NP - 1) for rp in range(NP)] for r in range(NP)]).ravel()
    Ae = np.zeros((2 * NP, n))
    for r in range(NP):
        Ae[r, r * NP:(r + 1) * NP] = 1
        Ae[NP + r, r::NP] = 1
    be = np.concatenate([k0, k1])
    r1 = linprog(-gain, A_eq=Ae, b_eq=be, bounds=[(0, None)] * n, method='highs'); assert r1.status == 0
    r2 = linprog(-gs, A_ub=-gain[None, :], b_ub=[r1.fun + 1e-10], A_eq=Ae, b_eq=be, bounds=[(0, None)] * n, method='highs'); assert r2.status == 0
    pi = r2.x.reshape(NP, NP)
    q = np.zeros(NP)
    for r in range(NP):
        for rp in range(NP):
            q[r & rp] += pi[r, rp]
    return -r1.fun, q, pi
