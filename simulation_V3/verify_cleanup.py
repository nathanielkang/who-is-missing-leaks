"""Certificate: robust privacy level of raw LP solutions and of clean_pair output.

Independent of lps._robust_max.
Joint box: for every group and released pattern, min and max of q(t) over
{lo <= kappa <= hi, sum kappa = 1} are primal LPs solved with HiGHS
(the solver inside scipy.optimize.linprog, method='highs'). The binding
event of every instance, and every event on the NHANES grid, is recomputed
with scipy.optimize.linprog and the absolute gap is stored.
Product box: q(t) is multilinear in the rates, so extrema are the 16 vertices.

Does not modify manuscript TeX. Reads data/nhanes_counts.npy. Writes results_V3/.
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import csv
import json
import math
import sys
import time
import traceback
import multiprocessing as mp

import numpy as np
from scipy.optimize import linprog
import scipy.optimize._highspy._core as hs

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import (  # noqa: E402
    K, NP, PAT, SUBIDX, R_OF, T_OF,
    jeffreys, margins, prod_from_margins, released, cp_interval,
    identity_policy, product_policy, block_policy, balanced_alloc, F,
)
import lps  # noqa: E402

FEAS = 1e-9  # lps.LP.solve primal/dual feasibility tolerance
CLEAN_TOL = 1e-6  # lps.clean_pair default
BETA = 0.05
ES = (0.5, 1.0)
MULTS = (1, 4)
POL_LP = ('exprod', 'exjoint_f', 'exjoint_t')
# claimed to meet E on the box used to build them
CLAIMED = {'exprod', 'exjoint_f', 'exjoint_t', 'balprod', 'block'}
FIELDS = ('family_income', 'alcohol', 'systolic_bp', 'waist')
BANDS = ('20-39', '40-59', '60+')
DATE = '2026-11-15'
OUTDIR = os.path.join(HERE, 'results_V3')
OPT_LP = {'primal_feasibility_tolerance': FEAS, 'dual_feasibility_tolerance': FEAS}
OPTIMAL = int(hs.HighsModelStatus.kOptimal)

J_BY_T = [np.where(T_OF == t)[0] for t in range(NP)]
ROW = [np.where(R_OF == r)[0] for r in range(NP)]
SUPPORT = []
for _t in range(NP):
    _rs = np.array([r for r in range(NP) if (_t & r) == _t], dtype=int)
    _js = np.array([SUBIDX[(int(r), _t)] for r in _rs], dtype=int)
    SUPPORT.append((_rs, _js))

_C = np.load(os.path.join(HERE, 'data', 'nhanes_counts.npy'))
TOT = _C.sum(0)  # band, h, pattern
KAP = np.array([[jeffreys(TOT[b, h]) for h in range(2)] for b in range(3)])
NBH = TOT.sum(2).astype(int)

COLS = [
    'setting', 'seed', 'mult', 'band', 'E', 'policy', 'box', 'claimed', 'checker',
    'raw_level', 'clean_level', 'raw_excess', 'clean_excess', 'raw_abs', 'clean_abs',
    'max_l1_row', 'n_zeroed', 'n_zeroed_pos', 'mass_zeroed', 'max_mass_added',
    'box_ok', 'linprog_diff', 'worst_h', 'worst_t', 'worst_maxq', 'worst_minq',
    'tau_sum', 'tau_S', 'solve_ok', 'error',
]


def _fmt(x):
    if x is None:
        return ''
    if isinstance(x, (float, np.floating)):
        xf = float(x)
        if math.isnan(xf):
            return 'nan'
        if math.isinf(xf):
            return 'inf' if xf > 0 else '-inf'
        return repr(xf)
    if isinstance(x, (bool, np.bool_)):
        return '1' if bool(x) else '0'
    return str(x)


class JointBox:
    """Primal HiGHS model for one group's joint box. Objective is swapped per query."""

    def __init__(self, lo, hi):
        self.lo = np.ascontiguousarray(lo, dtype=np.float64)
        self.hi = np.ascontiguousarray(hi, dtype=np.float64)
        self.ok = bool(
            self.lo.shape == (NP,)
            and np.all(np.isfinite(self.lo))
            and np.all(np.isfinite(self.hi))
            and np.all(self.lo <= self.hi + 1e-12)
            and float(self.lo.sum()) <= 1.0 + 1e-8
            and float(self.hi.sum()) >= 1.0 - 1e-8
        )
        self.model = None
        self.idx = np.arange(NP, dtype=np.int32)
        if not self.ok:
            return
        # scipy.linprog's HiGHS model leaves the process-wide scheduler unusable
        # for a later _Highs instance (run() returns status Not Set). Reset first.
        hs._Highs().resetGlobalScheduler(True)
        m = hs._Highs()
        m.setOptionValue('output_flag', False)
        m.setOptionValue('log_to_console', False)
        m.setOptionValue('primal_feasibility_tolerance', FEAS)
        m.setOptionValue('dual_feasibility_tolerance', FEAS)
        m.setOptionValue('threads', 1)
        z = np.zeros(NP, dtype=np.float64)
        rc = m.addCols(
            NP, z, self.lo, self.hi,
            0, np.zeros(1, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64),
        )
        if int(rc) != 0:
            self.ok = False
            return
        rc = m.addRow(1.0, 1.0, NP, self.idx, np.ones(NP, dtype=np.float64))
        if int(rc) != 0:
            self.ok = False
            return
        self.model = m

    def minmax(self, a):
        if not self.ok:
            return None
        a = np.ascontiguousarray(a, dtype=np.float64)
        self.model.changeColsCost(NP, self.idx, a)
        self.model.run()
        if int(self.model.getModelStatus()) != OPTIMAL:
            return None
        mn = float(self.model.getObjectiveValue())
        self.model.changeColsCost(NP, self.idx, -a)
        self.model.run()
        if int(self.model.getModelStatus()) != OPTIMAL:
            return None
        mx = -float(self.model.getObjectiveValue())
        return mn, mx


def linprog_minmax(a, lo, hi):
    a = np.ascontiguousarray(a, dtype=np.float64)
    bounds = list(zip(lo.tolist(), hi.tolist()))
    A = np.ones((1, a.size))
    rmin = linprog(a, A_eq=A, b_eq=[1.0], bounds=bounds, method='highs', options=OPT_LP)
    rmax = linprog(-a, A_eq=A, b_eq=[1.0], bounds=bounds, method='highs', options=OPT_LP)
    if rmin.status != 0 or rmax.status != 0:
        return None
    return float(rmin.fun), float(-rmax.fun)


def pattern_coef(Q, t):
    rs, js = SUPPORT[t]
    a = np.zeros(NP, dtype=np.float64)
    a[rs] = np.asarray(Q, dtype=np.float64)[js]
    return a


def joint_extrema(box, Q):
    if box is None or not box.ok:
        return None
    maxq = np.zeros(NP)
    minq = np.zeros(NP)
    for t in range(NP):
        mm = box.minmax(pattern_coef(Q, t))
        if mm is None:
            return None
        minq[t], maxq[t] = mm
    return maxq, minq


def joint_linprog_gap(box, Q, maxq, minq, only_t=None):
    """Max |HiGHS - scipy.linprog| over patterns (or one pattern)."""
    if box is None or not box.ok:
        return float('nan')
    ts = range(NP) if only_t is None else (only_t,)
    gap = 0.0
    worst_fail = False
    for t in ts:
        mm = linprog_minmax(pattern_coef(Q, t), box.lo, box.hi)
        if mm is None:
            worst_fail = True
            continue
        gap = max(gap, abs(mm[0] - float(minq[t])), abs(mm[1] - float(maxq[t])))
    if worst_fail and gap == 0.0:
        return float('nan')
    return gap


def product_extrema(Q, lo, hi):
    maxq = np.full(NP, -np.inf)
    minq = np.full(NP, np.inf)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    for mask in range(1 << K):
        g = np.where([(mask >> k) & 1 for k in range(K)], hi, lo)
        q = released(prod_from_margins(g), Q)
        np.maximum(maxq, q, out=maxq)
        np.minimum(minq, q, out=minq)
    return maxq, minq


def _clip_tiny_neg(x):
    if -1e-8 < x < 0.0:
        return 0.0
    return x


def robust_level(maxs, mins):
    level = 0.0
    worst = (-1, -1, 0.0, 0.0)
    for h in range(2):
        for t in range(NP):
            mx = _clip_tiny_neg(float(maxs[h, t]))
            mn = _clip_tiny_neg(float(mins[1 - h, t]))
            if mx < 0.0:
                mx = 0.0
            if mn < 0.0:
                mn = 0.0
            # Floor only a pair that is absent on both sides. A positive min, even
            # near 1e-9, keeps the likelihood ratio; min==0 and max>FEAS is infinite.
            if mx <= FEAS and mn <= FEAS:
                contrib = 0.0
            elif mn <= 0.0:
                contrib = math.inf if mx > FEAS else 0.0
            else:
                contrib = math.log(mx / mn)
            if contrib > level:
                level = contrib
                worst = (h, t, mx, mn)
    return level, worst


def abs_slack(maxs, mins, E):
    eE = math.exp(E)
    slack = 0.0
    for h in range(2):
        for t in range(NP):
            mx = _clip_tiny_neg(float(maxs[h, t]))
            mn = _clip_tiny_neg(float(mins[1 - h, t]))
            if mx < 0:
                mx = 0.0
            if mn < 0:
                mn = 0.0
            if mx <= FEAS and mn <= FEAS:
                s = 0.0
            else:
                s = mx - eE * mn
            if s > slack:
                slack = s
    return slack


def stack_extrema(fn0, fn1):
    maxs = np.zeros((2, NP))
    mins = np.zeros((2, NP))
    e0 = fn0()
    e1 = fn1()
    if e0 is None or e1 is None:
        return None
    maxs[0], mins[0] = e0
    maxs[1], mins[1] = e1
    return maxs, mins


def cleanup_stats(raw0, raw1, clean0, clean1, tol=CLEAN_TOL):
    z0 = np.clip(np.asarray(raw0, float), 0.0, 1.0)
    z1 = np.clip(np.asarray(raw1, float), 0.0, 1.0)
    n_zero = 0
    n_pos = 0
    mass = 0.0
    for t in range(NP):
        js = J_BY_T[t]
        if min(float(z0[js].max()), float(z1[js].max())) < tol:
            mass_t = float(z0[js].sum() + z1[js].sum())
            n_zero += 1
            if mass_t > 0.0:
                n_pos += 1
                mass += mass_t
            z0[js] = 0.0
            z1[js] = 0.0
    max_l1 = 0.0
    max_added = 0.0
    for raw, cln, z in ((raw0, clean0, z0), (raw1, clean1, z1)):
        raw = np.asarray(raw, float)
        cln = np.asarray(cln, float)
        for r in range(NP):
            js = ROW[r]
            max_l1 = max(max_l1, float(np.abs(cln[js] - np.clip(raw[js], 0.0, 1.0)).sum()))
            max_added = max(max_added, float(1.0 - z[js].sum()))
    return {
        'max_l1_row': max_l1,
        'n_zeroed': n_zero,
        'n_zeroed_pos': n_pos,
        'mass_zeroed': mass,
        'max_mass_added': max_added,
    }


def solve_lp(kind, lo, hi, E, objective):
    holder = {}
    real = lps.clean_pair

    def wrap(Q0, Q1, tol=CLEAN_TOL):
        holder['raw'] = (np.array(Q0, dtype=float, copy=True), np.array(Q1, dtype=float, copy=True))
        out = real(Q0, Q1, tol=tol)
        return out

    lps.clean_pair = wrap
    try:
        fn = lps.exact_joint_box if kind == 'joint' else lps.exact_product_box
        Q0, Q1, tau, tS = fn(lo, hi, E, objective)
    finally:
        lps.clean_pair = real
    if 'raw' in holder:
        raw = holder['raw']
    elif getattr(lps, '_last_raw', None) is not None:
        raw = lps._last_raw
    else:
        raise RuntimeError('neither clean_pair nor _last_raw captured the solver vector')
    return raw, (np.asarray(Q0, float), np.asarray(Q1, float)), np.asarray(tau, float), float(tS)


def empty_row(**kw):
    row = {c: '' for c in COLS}
    row.update(kw)
    return row


def _pair_metrics(maxs, mins, E):
    level, worst = robust_level(maxs, mins)
    slack = abs_slack(maxs, mins, E)
    excess = level - E if math.isfinite(level) else math.inf
    return level, excess, slack, worst


def certify_pair(setting, seed, mult, band, E, policy, box_name, claimed, checker,
                 raw, clean, maxs_raw, mins_raw, maxs_cl, mins_cl, stats, linprog_diff,
                 tau_sum, tau_S, box_ok, solve_ok=1, error=''):
    if maxs_raw is None:
        rl = cl = re = ce = ra = ca = math.nan
        worst = (-1, -1, math.nan, math.nan)
    else:
        rl, re, ra, _ = _pair_metrics(maxs_raw, mins_raw, E)
        cl, ce, ca, worst = _pair_metrics(maxs_cl, mins_cl, E)
    return {
        'setting': setting, 'seed': seed, 'mult': mult, 'band': band, 'E': E,
        'policy': policy, 'box': box_name, 'claimed': int(claimed), 'checker': checker,
        'raw_level': rl, 'clean_level': cl, 'raw_excess': re, 'clean_excess': ce,
        'raw_abs': ra, 'clean_abs': ca,
        'max_l1_row': stats.get('max_l1_row', 0.0),
        'n_zeroed': stats.get('n_zeroed', 0),
        'n_zeroed_pos': stats.get('n_zeroed_pos', 0),
        'mass_zeroed': stats.get('mass_zeroed', 0.0),
        'max_mass_added': stats.get('max_mass_added', 0.0),
        'box_ok': int(box_ok), 'linprog_diff': linprog_diff,
        'worst_h': worst[0], 'worst_t': worst[1],
        'worst_maxq': worst[2], 'worst_minq': worst[3],
        'tau_sum': tau_sum, 'tau_S': tau_S, 'solve_ok': solve_ok, 'error': error,
    }


def _no_move():
    return {'max_l1_row': 0.0, 'n_zeroed': 0, 'n_zeroed_pos': 0, 'mass_zeroed': 0.0, 'max_mass_added': 0.0}


def certify_calibration(setting, seed, mult, band, cal, full_linprog=False, only=''):
    rows = []
    jlo, jhi, blo, bhi, ghat = cp_margins_and_joint(cal)
    boxes = [JointBox(jlo[h], jhi[h]) for h in range(2)]
    joint_ok = all(b.ok for b in boxes)
    prod_ok = bool(np.all(np.isfinite(blo)) and np.all(np.isfinite(bhi)) and np.all(blo <= bhi + 1e-15))
    with np.errstate(divide='ignore', invalid='ignore'):
        ck = (bhi / blo).max(0)
        ell = bhi / ck
    common_ok = bool(prod_ok and np.all(blo > 0) and np.all(np.isfinite(ck)) and np.all(np.isfinite(ell)) and np.all(ell > 0))

    for E in ES:
        # --- LP policies ---
        jobs = (
            ('exprod', 'product', blo, bhi, 'fields'),
            ('exjoint_f', 'joint', jlo, jhi, 'fields'),
            ('exjoint_t', 'joint', jlo, jhi, 'task'),
        )
        solved = {}
        for name, kind, lo, hi, obj in jobs:
            if only == 'joint' and kind != 'joint':
                continue
            try:
                raw, clean, tau, tS = solve_lp(kind, lo, hi, E, obj)
                solved[name] = (raw, clean, tau, tS, '')
            except Exception as exc:
                rows.append(empty_row(
                    setting=setting, seed=seed, mult=mult, band=band, E=E, policy=name,
                    box=kind if kind == 'joint' else 'product_cp', claimed=1, checker='fail',
                    solve_ok=0, box_ok=0, error=f'{type(exc).__name__}: {exc}',
                    raw_level=float('nan'), clean_level=float('nan'),
                    raw_excess=float('nan'), clean_excess=float('nan'),
                    raw_abs=float('nan'), clean_abs=float('nan'), linprog_diff=float('nan'),
                    tau_sum=float('nan'), tau_S=float('nan'),
                    max_l1_row=0, n_zeroed=0, n_zeroed_pos=0, mass_zeroed=0, max_mass_added=0,
                    worst_h=-1, worst_t=-1, worst_maxq=float('nan'), worst_minq=float('nan'),
                ))
        # Rebuild after the policy LPs. A product-box linprog leaves the shared
        # HiGHS scheduler unable to run a model that was created earlier.
        boxes = [JointBox(jlo[h], jhi[h]) for h in range(2)]
        joint_ok = all(b.ok for b in boxes)
        for name, kind, lo, hi, obj in jobs:
            if name not in solved:
                continue
            raw, clean, tau, tS, _err = solved[name]
            stats = cleanup_stats(raw[0], raw[1], clean[0], clean[1])
            if kind == 'joint':
                er = stack_extrema(
                    lambda Q=raw[0]: joint_extrema(boxes[0], Q),
                    lambda Q=raw[1]: joint_extrema(boxes[1], Q),
                )
                ec = stack_extrema(
                    lambda Q=clean[0]: joint_extrema(boxes[0], Q),
                    lambda Q=clean[1]: joint_extrema(boxes[1], Q),
                )
                gap = float('nan')
                if er is not None and ec is not None:
                    if full_linprog:
                        g0 = joint_linprog_gap(boxes[0], raw[0], er[0][0], er[1][0])
                        g1 = joint_linprog_gap(boxes[1], raw[1], er[0][1], er[1][1])
                        g2 = joint_linprog_gap(boxes[0], clean[0], ec[0][0], ec[1][0])
                        g3 = joint_linprog_gap(boxes[1], clean[1], ec[0][1], ec[1][1])
                        gap = np.nanmax([g0, g1, g2, g3])
                    else:
                        # binding event of the cleaned policy, both sides
                        _lvl, worst = robust_level(ec[0], ec[1])
                        h, t = worst[0], worst[1]
                        if h >= 0:
                            g_max = joint_linprog_gap(boxes[h], clean[h], ec[0][h], ec[1][h], only_t=t)
                            g_min = joint_linprog_gap(boxes[1 - h], clean[1 - h], ec[0][1 - h], ec[1][1 - h], only_t=t)
                            gap = np.nanmax([g_max, g_min])
                box_ok = joint_ok and er is not None and ec is not None
                rows.append(certify_pair(
                    setting, seed, mult, band, E, name, 'joint', True, 'highs-primal',
                    raw, clean,
                    None if er is None else er[0], None if er is None else er[1],
                    None if ec is None else ec[0], None if ec is None else ec[1],
                    stats, float(gap) if gap == gap else float('nan'),
                    float(np.sum(tau)), tS, box_ok,
                ))
            else:
                def _pe(Q, lo=lo, hi=hi):
                    return product_extrema(Q, lo, hi)

                # lo, hi are (2, K)
                er = stack_extrema(lambda: _pe(raw[0], lo[0], hi[0]), lambda: _pe(raw[1], lo[1], hi[1]))
                ec = stack_extrema(lambda: _pe(clean[0], lo[0], hi[0]), lambda: _pe(clean[1], lo[1], hi[1]))
                rows.append(certify_pair(
                    setting, seed, mult, band, E, name, 'product_cp', True, 'vertex',
                    raw, clean, er[0], er[1], ec[0], ec[1], stats, 0.0,
                    float(np.sum(tau)), tS, prod_ok,
                ))

        # --- closed forms (no clean_pair) ---
        if only == 'joint':
            continue
        try:
            gm = ghat.min(0)
            fitted = (product_policy(gm / ghat[0]), product_policy(gm / ghat[1]))
            fit_ok = bool(np.all(np.isfinite(fitted[0])) and np.all(np.isfinite(fitted[1])) and np.all(ghat > 0))
        except Exception:
            fitted = None
            fit_ok = False
        if common_ok:
            tau_b = balanced_alloc(E, ck, ell.min(0))
            bal = (product_policy(tau_b / ell[0]), product_policy(tau_b / ell[1]))
            aS = ell.prod(1)
            Fv = F(E, float(aS.min()), float(ck.prod()))
            blk = (block_policy(Fv / aS[0]), block_policy(Fv / aS[1]))
            bal_ok = bool(np.all(np.isfinite(bal[0])) and np.all(np.isfinite(bal[1])))
            blk_ok = bool(np.all(np.isfinite(blk[0])) and np.all(np.isfinite(blk[1])))
        else:
            tau_b = None
            bal = blk = None
            bal_ok = blk_ok = False
            Fv = float('nan')
        ident = (identity_policy(), identity_policy())

        closed = [
            ('none', ident, True, 'product_cp', False, prod_ok, float('nan'), float('nan')),
            ('fitted', fitted if fit_ok else None, fit_ok, 'product_cp', False, prod_ok and fit_ok, float('nan'), float('nan')),
            ('balprod', bal if bal_ok else None, bal_ok, 'common_width', True, common_ok and bal_ok,
             float(np.sum(tau_b)) if tau_b is not None else float('nan'), float('nan')),
            ('block', blk if blk_ok else None, blk_ok, 'common_width', True, common_ok and blk_ok,
             float('nan'), float(Fv) if blk_ok else float('nan')),
        ]
        for name, pair, ok, box_name, claimed, box_ok, tau_sum, tau_S in closed:
            if not ok or pair is None:
                rows.append(empty_row(
                    setting=setting, seed=seed, mult=mult, band=band, E=E, policy=name,
                    box=box_name, claimed=int(claimed), checker='vertex', solve_ok=0, box_ok=0,
                    error='closed-form undefined (zero margin or non-finite box)',
                    raw_level=float('nan'), clean_level=float('nan'),
                    raw_excess=float('nan'), clean_excess=float('nan'),
                    raw_abs=float('nan'), clean_abs=float('nan'), linprog_diff=0.0,
                    tau_sum=tau_sum, tau_S=tau_S,
                    max_l1_row=0, n_zeroed=0, n_zeroed_pos=0, mass_zeroed=0, max_mass_added=0,
                    worst_h=-1, worst_t=-1, worst_maxq=float('nan'), worst_minq=float('nan'),
                ))
                continue
            if box_name == 'common_width':
                er = stack_extrema(
                    lambda Q=pair[0]: product_extrema(Q, ell[0], ell[0] * ck),
                    lambda Q=pair[1]: product_extrema(Q, ell[1], ell[1] * ck),
                )
            else:
                er = stack_extrema(
                    lambda Q=pair[0]: product_extrema(Q, blo[0], bhi[0]),
                    lambda Q=pair[1]: product_extrema(Q, blo[1], bhi[1]),
                )
            rows.append(certify_pair(
                setting, seed, mult, band, E, name, box_name, claimed, 'vertex',
                pair, pair, er[0], er[1], er[0], er[1], _no_move(), 0.0,
                tau_sum, tau_S, box_ok,
            ))
    return rows


def cp_margins_and_joint(cal):
    cal = np.asarray(cal)
    n = cal.sum(1)
    jlo = np.zeros((2, NP))
    jhi = np.zeros((2, NP))
    blo = np.zeros((2, K))
    bhi = np.zeros((2, K))
    ghat = np.zeros((2, K))
    for h in range(2):
        for r in range(NP):
            jlo[h, r], jhi[h, r] = cp_interval(int(cal[h, r]), int(n[h]), BETA / 96.0)
        x = PAT.T @ np.asarray(cal[h], float)
        ghat[h] = x / float(n[h])
        for k in range(K):
            blo[h, k], bhi[h, k] = cp_interval(int(round(float(x[k]))), int(n[h]), BETA / 24.0)
    return jlo, jhi, blo, bhi, ghat


def mc_calibrations(seed):
    rng = np.random.default_rng([20261115, int(seed)])
    out = {}
    for mult in MULTS:
        for b in range(3):
            cal = np.array([rng.multinomial(int(mult * NBH[b, h]), KAP[b, h]) for h in range(2)])
            out[(mult, b)] = cal
    return out


def worker(seed):
    try:
        only = os.environ.get('VERIFY_ONLY', '')
        cals = mc_calibrations(seed)
        rows = []
        for mult in MULTS:
            for b in range(3):
                rows.extend(certify_calibration(
                    'mc', int(seed), mult, b, cals[(mult, b)],
                    full_linprog=False, only=only,
                ))
        return rows
    except Exception:
        return [empty_row(
            setting='mc', seed=int(seed), mult=-1, band=-1, E=float('nan'), policy='ALL',
            box='', claimed=0, checker='fail', solve_ok=0, box_ok=0,
            error=traceback.format_exc(limit=4),
            raw_level=float('nan'), clean_level=float('nan'),
            raw_excess=float('nan'), clean_excess=float('nan'),
            raw_abs=float('nan'), clean_abs=float('nan'), linprog_diff=float('nan'),
            tau_sum=float('nan'), tau_S=float('nan'),
            max_l1_row=0, n_zeroed=0, n_zeroed_pos=0, mass_zeroed=0, max_mass_added=0,
            worst_h=-1, worst_t=-1, worst_maxq=float('nan'), worst_minq=float('nan'),
        )]


def nhanes_cal(mult, band):
    return np.rint(mult * TOT[band]).astype(int)


def pattern_258():
    """Occupied-support joint level by age band, from raw NHANES counts."""
    bands = []
    for b in range(3):
        n0 = int(TOT[b, 0].sum())
        n1 = int(TOT[b, 1].sum())
        patterns = []
        for r in range(NP):
            c0 = int(TOT[b, 0, r])
            c1 = int(TOT[b, 1, r])
            if c0 > 0 and c1 > 0:
                level = abs(math.log((c0 / n0) / (c1 / n1)))
                patterns.append({
                    'pattern': r,
                    'bits_income_alcohol_sbp_waist': ''.join(str(int(PAT[r, k])) for k in range(K)),
                    'fields_observed': [FIELDS[k] for k in range(K) if PAT[r, k] == 1],
                    'fields_missing': [FIELDS[k] for k in range(K) if PAT[r, k] == 0],
                    'count_citizen_h0': c0,
                    'count_noncitizen_h1': c1,
                    'level': level,
                })
        patterns.sort(key=lambda d: -d['level'])
        best = patterns[0]
        bands.append({
            'band_index': b,
            'age': BANDS[b],
            'n_citizen': n0,
            'n_noncitizen': n1,
            'max_level': best['level'],
            'argmax_pattern': best,
            'second_level': patterns[1]['level'] if len(patterns) > 1 else None,
            'n_occupied_patterns': len(patterns),
        })
    # manuscript sentence
    b1 = bands[1]
    arg = b1['argmax_pattern']
    literal = (
        b1['age'] == '40-59'
        and arg['count_citizen_h0'] == 1
        and arg['count_noncitizen_h1'] == 3
        and abs(b1['max_level'] - 2.58) < 0.005
        and (b1['second_level'] is None or b1['max_level'] > b1['second_level'] + 1e-9)
    )
    return {
        'source': 'data/nhanes_counts.npy summed over PSU',
        'coding': {
            'h0': 'citizen (DMDCITZN==1)',
            'h1': 'noncitizen (DMDCITZN==2)',
            'band_index': {'0': '20-39', '1': '40-59', '2': '60+'},
            'bits': 'bit0 family income, bit1 alcohol, bit2 systolic BP, bit3 waist; 1=observed',
            'level': 'abs log of raw proportions on patterns with both counts positive (level_occupied)',
        },
        'bands': bands,
        'manuscript_sentence': (
            'Within one age band, the level reaches 2.58, on a pattern shown by '
            'one citizen and three non-citizens aged 40 to 59.'
        ),
        'sentence_literally_correct': bool(literal),
        'focus': b1,
    }


class Agg:
    def __init__(self):
        self.cells = {}
        self.n_rows = 0
        self.n_solve_fail = 0
        self.max_linprog_diff = 0.0

    def add(self, row):
        self.n_rows += 1
        if int(row.get('solve_ok') or 0) != 1:
            self.n_solve_fail += 1
        gap = row.get('linprog_diff')
        if isinstance(gap, float) and math.isfinite(gap):
            self.max_linprog_diff = max(self.max_linprog_diff, gap)
        key = f"{row['setting']}|{row['mult']}|{row['E']}|{row['policy']}"
        cell = self.cells.get(key)
        if cell is None:
            cell = {
                'setting': row['setting'], 'mult': row['mult'], 'E': row['E'], 'policy': row['policy'],
                'box': row['box'], 'claimed': row['claimed'], 'checker': row['checker'],
                'n': 0, 'n_box_fail': 0, 'n_solve_fail': 0,
                'max_excess_raw': None, 'max_excess_clean': None,
                'n_raw_gt_1e-7': 0, 'n_raw_gt_1e-4': 0, 'n_raw_gt_1e-2': 0,
                'n_clean_gt_1e-7': 0, 'n_clean_gt_1e-4': 0, 'n_clean_gt_1e-2': 0,
                'max_l1_row': 0.0, 'max_n_zeroed': 0, 'max_n_zeroed_pos': 0,
                'max_mass_zeroed': 0.0, 'max_mass_added': 0.0,
                'max_abs_raw': 0.0, 'max_abs_clean': 0.0,
                'max_linprog_diff': 0.0,
                'worst_clean_seed': None, 'worst_clean_band': None,
                'worst_clean_level': None, 'worst_clean_abs': None,
                'worst_h': None, 'worst_t': None, 'worst_maxq': None, 'worst_minq': None,
            }
            self.cells[key] = cell
        cell['n'] += 1
        if int(row.get('box_ok') or 0) != 1:
            cell['n_box_fail'] += 1
        if int(row.get('solve_ok') or 0) != 1:
            cell['n_solve_fail'] += 1
        self._max_excess(cell, 'max_excess_raw', row.get('raw_excess'))
        self._max_excess(cell, 'max_excess_clean', row.get('clean_excess'))
        for name, thr, bucket in (
            ('raw_excess', 1e-7, 'n_raw_gt_1e-7'),
            ('raw_excess', 1e-4, 'n_raw_gt_1e-4'),
            ('raw_excess', 1e-2, 'n_raw_gt_1e-2'),
            ('clean_excess', 1e-7, 'n_clean_gt_1e-7'),
            ('clean_excess', 1e-4, 'n_clean_gt_1e-4'),
            ('clean_excess', 1e-2, 'n_clean_gt_1e-2'),
        ):
            v = row.get(name)
            if isinstance(v, float) and v > thr:
                cell[bucket] += 1
        for src, dst in (
            ('max_l1_row', 'max_l1_row'),
            ('mass_zeroed', 'max_mass_zeroed'),
            ('max_mass_added', 'max_mass_added'),
            ('raw_abs', 'max_abs_raw'),
            ('clean_abs', 'max_abs_clean'),
        ):
            v = row.get(src)
            if isinstance(v, float) and math.isfinite(v) and v > cell[dst]:
                cell[dst] = v
        if isinstance(gap, float) and math.isfinite(gap) and gap > cell['max_linprog_diff']:
            cell['max_linprog_diff'] = gap
        for src, dst in (('n_zeroed', 'max_n_zeroed'), ('n_zeroed_pos', 'max_n_zeroed_pos')):
            v = row.get(src)
            if isinstance(v, (int, float)) and int(v) > cell[dst]:
                cell[dst] = int(v)
        ce = row.get('clean_excess')
        cur = cell['max_excess_clean']
        if isinstance(ce, float) and not math.isnan(ce) and cur is not None and ce == cur:
            cell['worst_clean_seed'] = row['seed']
            cell['worst_clean_band'] = row['band']
            cell['worst_clean_level'] = row.get('clean_level')
            cell['worst_clean_abs'] = row.get('clean_abs')
            cell['worst_h'] = row.get('worst_h')
            cell['worst_t'] = row.get('worst_t')
            cell['worst_maxq'] = row.get('worst_maxq')
            cell['worst_minq'] = row.get('worst_minq')

    @staticmethod
    def _max_excess(cell, key, v):
        if not isinstance(v, float) or (math.isnan(v)):
            return
        cur = cell[key]
        if cur is None or v > cur:
            cell[key] = v

    def as_dict(self):
        out = {}
        for k, cell in sorted(self.cells.items(), key=lambda kv: kv[0]):
            out[k] = _jsonify(cell)
        return out


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        xf = float(obj)
        if math.isnan(xf) or math.isinf(xf):
            return None if math.isnan(xf) else ('inf' if xf > 0 else '-inf')
        return xf
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def self_test(log):
    rng = np.random.default_rng(0)
    lo = rng.uniform(0.01, 0.04, size=NP)
    hi = lo + rng.uniform(0.05, 0.12, size=NP)
    # force a nonempty simplex slice
    lo = lo / lo.sum() * 0.5
    hi = np.minimum(0.4, np.maximum(hi, lo + 0.01))
    if hi.sum() < 1:
        hi = hi / hi.sum() * 1.5
    box = JointBox(lo, hi)
    if not box.ok:
        raise RuntimeError('self-test box failed to build')
    gap = 0.0
    for i in range(12):
        a = rng.normal(size=NP)
        mm = box.minmax(a)
        lp = linprog_minmax(a, box.lo, box.hi)
        if mm is None or lp is None:
            raise RuntimeError(f'self-test solve failed at {i}')
        gap = max(gap, abs(mm[0] - lp[0]), abs(mm[1] - lp[1]))
    log(f'self-test HiGHS vs linprog max abs diff {gap:.3e} on 12 random objectives')
    if gap > 1e-8:
        raise RuntimeError(f'HiGHS model disagrees with scipy.linprog by {gap}')
    # cleanup stats vs clean_pair on a noisy policy
    Q = np.zeros(lps.NSUB)
    for r in range(NP):
        js = ROW[r]
        w = rng.random(len(js))
        w = w / w.sum()
        Q[js] = w
    Q[ROW[15][0]] = 1e-8
    c0, c1 = lps.clean_pair(Q, Q)
    st = cleanup_stats(Q, Q, c0, c1)
    log(f'self-test cleanup L1 {st["max_l1_row"]:.3e} zeroed {st["n_zeroed"]} pos {st["n_zeroed_pos"]} mass {st["mass_zeroed"]:.3e}')
    # NHANES band 1, mult 1, one joint policy: full linprog gap
    cal = nhanes_cal(1, 1)
    jlo, jhi, blo, bhi, _g = cp_margins_and_joint(cal)
    raw, clean, _tau, _tS = solve_lp('joint', jlo, jhi, 0.5, 'fields')
    boxes = [JointBox(jlo[h], jhi[h]) for h in range(2)]
    for h in range(2):
        ext = joint_extrema(boxes[h], clean[h])
        if ext is None:
            raise RuntimeError('NHANES smoke joint extrema failed')
        g = joint_linprog_gap(boxes[h], clean[h], ext[0], ext[1])
        log(f'self-test NHANES band1 mult1 exjoint clean group {h} linprog gap {g:.3e}')
        if not math.isfinite(g) or g > 1e-8:
            raise RuntimeError(f'NHANES linprog gap {g}')
    log('self-test PASS')


def write_rows(fp, writer, rows):
    for row in rows:
        writer.writerow([_fmt(row.get(c, '')) for c in COLS])
    fp.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-seeds', type=int, default=None, help='if set, run exactly this many MC seeds (0..n-1) and skip the 30-min rule')
    ap.add_argument('--nproc', type=int, default=12)
    ap.add_argument('--only', default='', choices=['', 'joint'])
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if args.only:
        os.environ['VERIFY_ONLY'] = args.only
    parts = [p for p in (args.only, args.tag) if p]
    tag = ('_' + '_'.join(parts)) if parts else ''
    log_path = os.path.join(OUTDIR, f'verify_cleanup{tag}_{DATE}.log')
    detail_path = os.path.join(OUTDIR, f'verify_cleanup{tag}_detail_{DATE}.csv')
    summary_path = os.path.join(OUTDIR, f'verify_cleanup{tag}_summary_{DATE}.json')
    pattern_path = os.path.join(OUTDIR, f'pattern_258_{DATE}.json')
    logf = open(log_path, 'w', encoding='utf-8', buffering=1)

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t_start = time.perf_counter()
    log(f'verify_cleanup {DATE}')
    log(f'python {sys.version.split()[0]} numpy {np.__version__} highs {hs.HIGHS_VERSION_MAJOR}.{hs.HIGHS_VERSION_MINOR}.{hs.HIGHS_VERSION_PATCH}')
    log(f'FEAS={FEAS} clean_tol={CLEAN_TOL} beta={BETA} E={ES} mult={MULTS}')
    log('joint checker: primal HiGHS on the simplex-box, cross-checked with scipy.linprog')
    log('product checker: 16 vertices of the rate box')
    log('nhanes calibration: observed counts times the multiplier (not a multinomial draw)')
    log('mc calibration: sim.py protocol, rng default_rng([20261115, seed]), Multinomial(mult*n, Jeffreys)')
    pat = pattern_258()
    with open(pattern_path, 'w', encoding='utf-8') as f:
        json.dump(_jsonify(pat), f, indent=1)
    focus = pat['focus']
    arg = focus['argmax_pattern']
    log(
        f"pattern 2.58: age {focus['age']} pattern {arg['pattern']} bits {arg['bits_income_alcohol_sbp_waist']} "
        f"citizen {arg['count_citizen_h0']} noncitizen {arg['count_noncitizen_h1']} "
        f"n {focus['n_citizen']}/{focus['n_noncitizen']} level {focus['max_level']:.10f} "
        f"literal={pat['sentence_literally_correct']}"
    )
    log('self-test')
    self_test(log)

    agg = Agg()
    n_total = 0
    nproc = max(1, min(int(args.nproc), os.cpu_count() or 1))
    log(f'nproc {nproc}')
    with open(detail_path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        writer.writerow(COLS)
        log('NHANES grid')
        t_nh = time.perf_counter()
        for mult in MULTS:
            for b in range(3):
                rows = certify_calibration('nhanes', -1, mult, b, nhanes_cal(mult, b), full_linprog=True, only=args.only)
                for row in rows:
                    agg.add(row)
                write_rows(fp, writer, rows)
                # headline for claimed LP
                for row in rows:
                    if row['policy'] in POL_LP:
                        log(
                            f"  nhanes mult {mult} band {b} E {row['E']} {row['policy']} "
                            f"raw_ex {row['raw_excess']} clean_ex {row['clean_excess']} "
                            f"abs {row['clean_abs']} L1 {row['max_l1_row']} zeroed_pos {row['n_zeroed_pos']} "
                            f"linprog {row['linprog_diff']}"
                        )
        log(f'NHANES seconds {time.perf_counter() - t_nh:.2f}')

        ctx = mp.get_context('spawn')
        if args.max_seeds is not None:
            n_total = int(args.max_seeds)
            log(f'max-seeds override {n_total}')
            t_mc = time.perf_counter()
            done = 0
            with ctx.Pool(min(nproc, max(1, n_total))) as pool:
                for rows in pool.imap_unordered(worker, range(n_total), chunksize=1):
                    for row in rows:
                        agg.add(row)
                    write_rows(fp, writer, rows)
                    done += 1
                    if done % 5 == 0 or done == n_total:
                        log(f'  mc {done}/{n_total} elapsed {time.perf_counter() - t_mc:.1f}s linprog_gap {agg.max_linprog_diff:.3e}')
            log(f'MC seconds {time.perf_counter() - t_mc:.2f} seeds {n_total}')
        else:
            n_pilot = max(8, nproc)
            log(f'pilot seeds 0..{n_pilot - 1}')
            t_mc = time.perf_counter()
            done = 0
            with ctx.Pool(nproc) as pool:
                for rows in pool.imap_unordered(worker, range(n_pilot), chunksize=1):
                    for row in rows:
                        agg.add(row)
                    write_rows(fp, writer, rows)
                    done += 1
                pilot_wall = time.perf_counter() - t_mc
                rate = pilot_wall / n_pilot
                elapsed = time.perf_counter() - t_start
                eta_5000 = elapsed + rate * (5000 - n_pilot)
                budget = 30 * 60
                if eta_5000 <= budget:
                    n_total = 5000
                    reason = f'eta_5000 {eta_5000:.0f}s <= {budget:.0f}s'
                else:
                    remain = max(0.0, budget - elapsed)
                    n_more = int(remain / rate) if rate > 0 else 0
                    n_total = min(5000, max(200, n_pilot + n_more))
                    reason = f'eta_5000 {eta_5000:.0f}s > {budget:.0f}s; rate {rate:.3f}s/seed wall; n_total {n_total}'
                log(f'pilot wall {pilot_wall:.2f}s rate {rate:.3f}s/seed {reason}')
                if n_total > n_pilot:
                    rest = range(n_pilot, n_total)
                    for rows in pool.imap_unordered(worker, rest, chunksize=4):
                        for row in rows:
                            agg.add(row)
                        write_rows(fp, writer, rows)
                        done += 1
                        if done % 100 == 0 or done == n_total:
                            # peek claimed cleaned max
                            peek = []
                            for key, cell in agg.cells.items():
                                if cell['setting'] == 'mc' and cell['policy'] in POL_LP:
                                    peek.append((cell['max_excess_clean'], key))
                            peek.sort(key=lambda z: -(z[0] if isinstance(z[0], float) else -1))
                            top = peek[0] if peek else None
                            log(
                                f'  mc {done}/{n_total} elapsed {time.perf_counter() - t_mc:.1f}s '
                                f'linprog_gap {agg.max_linprog_diff:.3e} top {top}'
                            )
            log(f'MC seconds {time.perf_counter() - t_mc:.2f} seeds {n_total}')

    # decision: any cleaned claimed policy with excess > 1e-7
    violations = []
    for key, cell in agg.cells.items():
        if cell['policy'] not in POL_LP:
            continue
        if cell['n_clean_gt_1e-7'] > 0:
            violations.append(key)
    summary = {
        'date': DATE,
        'feasibility_tolerance': FEAS,
        'clean_tol': CLEAN_TOL,
        'definition': {
            'robust_level': (
                'max over h in {0,1} and released patterns t of log(max_box q_h(t) / min_box q_{1-h}(t)). '
                'If both probabilities are <= 1e-9 (HiGHS feasibility tolerance), the contribution is 0. '
                'If the min is <= 1e-9 and the max is above it, the level is infinite.'
            ),
            'excess': 'robust_level - E',
            'abs_slack': 'max (max q_h(t) - exp(E) * min q_{1-h}(t)), with the same 1e-9 floor',
            'nhanes': 'Clopper-Pearson box on multiplier * observed integer counts',
            'mc': 'sim.py seeds, Jeffreys multinomial calibration',
            'boxes': {
                'exprod': 'product Clopper-Pearson box used by exact_product_box',
                'exjoint_f': 'joint Clopper-Pearson box',
                'exjoint_t': 'joint Clopper-Pearson box',
                'balprod': 'common-width product box of Theorem joint (contains the CP marginal box)',
                'block': 'common-width product box',
                'fitted': 'not claimed; diagnostic level on the CP product box',
                'none': 'not claimed; diagnostic level on the CP product box',
            },
        },
        'timing_seconds': time.perf_counter() - t_start,
        'n_mc_seeds': int(n_total),
        'n_rows': agg.n_rows,
        'n_solve_fail': agg.n_solve_fail,
        'max_linprog_diff': agg.max_linprog_diff,
        'cleaned_policies_exceeding_1e-7': violations,
        'fix_needed': bool(violations),
        'cells': agg.as_dict(),
        'files': {
            'detail_csv': detail_path,
            'log': log_path,
            'pattern': pattern_path,
            'summary': summary_path,
        },
    }
    # n_total is assigned inside the with-block; bind it
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(_jsonify(summary), f, indent=1)
    log(f'rows {agg.n_rows} solve_fail {agg.n_solve_fail} max_linprog_diff {agg.max_linprog_diff:.3e}')
    log(f'fix_needed {bool(violations)} keys {violations}')
    log(f'total seconds {time.perf_counter() - t_start:.2f}')
    log(f'wrote {detail_path}')
    log(f'wrote {summary_path}')
    logf.close()


if __name__ == '__main__':
    main()
