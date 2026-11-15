"""Matched-budget optional-value simulation for the NHANES survey-calibrated law.

Does not modify core.py, lps.py, or sim.py. Calibration multinomials use the same
seed stream as sim.one_rep (Generator seed [20261115, seed], multipliers 1 then 4).
Value draws use a separate stream [20261115, seed, 99].

ACS: acs.py has the pattern counts and a transport oracle, but not the
Clopper-Pearson calibration loop, so this script runs NHANES only.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import hashlib
import json
import sys
import time
import argparse
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from sim import (
    KAP, NB, NBH, WB, PI, BETA, MULTS, ES, EPSV, NP, K, PAT,
    evalpol, cp_interval, exact_joint_box, transport_zero, level,
    identity_policy, block_policy, F,
)

# Optional source for complete-respondent CSVs. Prefer files already under data/.
ATT_DIR = os.environ.get('WM1_COMPLETE_VALUES_DIR', '')
DATA_DIR = os.path.join(HERE, 'data')
OUT_DIR = os.path.join(HERE, 'results')
LPS_PATH = os.path.join(HERE, 'lps.py')

# Array fields written per (replication, slot, band). Keep in sync with the meta list.
FIELDS = ['NS', 'm0', 'tau', 'level', 'cc0', 'cc1', 'est0', 'est1', 'rel', 'a0', 'a1']
FNS, FM0, FTAU, FLEV, FCC0, FCC1, FEST0, FEST1, FREL, FA0, FA1 = range(len(FIELDS))

# Nominal pattern budget E_T is 0 when the whole total budget is spent on values.
CFGS = [
    {'policy': 'none', 'E_T': 0.0, 'eps_V': 1.0, 'E_tot': 1.0, 'bound_source': 'true_level'},
    {'policy': 'none', 'E_T': 0.0, 'eps_V': 1.5, 'E_tot': 1.5, 'bound_source': 'true_level'},
    {'policy': 'block', 'E_T': 0.5, 'eps_V': 0.5, 'E_tot': 1.0, 'bound_source': 'nominal'},
    {'policy': 'block', 'E_T': 0.5, 'eps_V': 1.0, 'E_tot': 1.5, 'bound_source': 'nominal'},
    {'policy': 'block', 'E_T': 1.0, 'eps_V': 0.5, 'E_tot': 1.5, 'bound_source': 'nominal'},
    {'policy': 'exjoint_t', 'E_T': 0.5, 'eps_V': 0.5, 'E_tot': 1.0, 'bound_source': 'nominal'},
    {'policy': 'exjoint_t', 'E_T': 0.5, 'eps_V': 1.0, 'E_tot': 1.5, 'bound_source': 'nominal'},
    {'policy': 'exjoint_t', 'E_T': 1.0, 'eps_V': 0.5, 'E_tot': 1.5, 'bound_source': 'nominal'},
    {'policy': 'oracle', 'E_T': 0.0, 'eps_V': 1.0, 'E_tot': 1.0, 'bound_source': 'nominal'},
    {'policy': 'oracle', 'E_T': 0.0, 'eps_V': 1.5, 'E_tot': 1.5, 'bound_source': 'nominal'},
]
NCFG = len(CFGS)
NSLOT = len(MULTS) * NCFG

_READY = False
POOL = None
MU = None
NCOMP = None
ORACLE_A = None
DISC = None
CELL_MEAN = None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def m0_of(n, tau):
    """sim.py rule: ceil(n * tau / 2) when tau > 0, else 1, then at least 1."""
    if tau > 0:
        m0 = int(np.ceil(n * tau / 2.0))
    else:
        m0 = 1
    return max(m0, 1)


def certified_atom_min(lo, hi, i=15):
    """Lower bound on coordinate i over the probability simplex cut by [lo, hi]."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    other_hi = float(np.sum(hi) - hi[i])
    raw = max(float(lo[i]), 1.0 - other_hi)
    if raw > float(hi[i]) + 1e-8 or float(np.sum(lo)) > 1.0 + 1e-8:
        raw = float(lo[i])
    return float(min(1.0, max(0.0, raw)))


def as_prob(x, what):
    x = float(x)
    if not np.isfinite(x):
        raise RuntimeError('nonfinite probability for %s' % what)
    if x < -1e-8 or x > 1.0 + 1e-8:
        raise RuntimeError('probability for %s is %s' % (what, x))
    return float(min(1.0, max(0.0, x)))


def draw_calibration(rng, b, mult):
    """Same Clopper-Pearson boxes as sim.one_rep. Consumes two multinomials."""
    cal = np.array([rng.multinomial(int(mult * NBH[b, h]), KAP[b, h]) for h in range(2)])
    n = cal.sum(1)
    jlo = np.zeros((2, NP))
    jhi = np.zeros((2, NP))
    blo = np.zeros((2, K))
    bhi = np.zeros((2, K))
    ghat = np.zeros((2, K))
    for h in range(2):
        for r in range(NP):
            jlo[h, r], jhi[h, r] = cp_interval(cal[h, r], n[h], BETA / 96)
        x = PAT.T @ cal[h]
        ghat[h] = x / n[h]
        for k in range(K):
            blo[h, k], bhi[h, k] = cp_interval(x[k], n[h], BETA / 24)
    return cal, jlo, jhi, blo, bhi, ghat


def block_from_box(blo, bhi, E):
    """Product-box block policy, copied from sim.one_rep."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ck = (bhi / blo).max(0)
    ell = bhi / ck
    aS = ell.prod(1)
    Fv = F(E, aS.min(), ck.prod())
    return block_policy(Fv / aS[0]), block_policy(Fv / aS[1]), float(Fv)


def copy_inputs():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in ('nhanes_complete_values.csv', 'acs_complete_values.csv'):
        dst = os.path.join(DATA_DIR, name)
        if ATT_DIR:
            src = os.path.join(ATT_DIR, name)
            if os.path.isfile(src):
                data = open(src, 'rb').read()
                # Rewrite only when the bytes differ.
                old = open(dst, 'rb').read() if os.path.isfile(dst) else None
                if old != data:
                    with open(dst, 'wb') as f:
                        f.write(data)
        if not os.path.isfile(dst):
            raise FileNotFoundError(
                f'missing {dst}; place the complete-respondent CSV under data/ '
                f'or set WM1_COMPLETE_VALUES_DIR'
            )


def _candidate_report(stored, pred):
    d = np.abs(np.asarray(pred, float) - np.asarray(stored, float))
    return {'max_abs': float(np.nanmax(d)), 'n_gt_1e-8': int(np.sum(d > 1e-8))}


def formula_check():
    """Confirm stored f0, f1 against the inferred maps, including the unit-interval clip."""
    nh = pd.read_csv(os.path.join(DATA_DIR, 'nhanes_complete_values.csv'))
    acs = pd.read_csv(os.path.join(DATA_DIR, 'acs_complete_values.csv'))
    if not (nh['pattern'] == 15).all() or not (acs['pattern'] == 15).all():
        raise RuntimeError('complete-value files are not all pattern 15')

    nh_f0_raw = (nh['INDFMPIR'] / 5.0) * ((nh['BPXSY1'] - 70.0) / 130.0)
    nh_f0_fac = (nh['INDFMPIR'] / 5.0) * np.clip((nh['BPXSY1'] - 70.0) / 130.0, 0.0, 1.0)
    nh_f0_clip = np.clip(nh_f0_raw, 0.0, 1.0)
    nh_f1_raw = (nh['ALQ101'] == 1).astype(float) * ((nh['BMXWAIST'] - 50.0) / 100.0)
    nh_f1_clip = np.clip(nh_f1_raw, 0.0, 1.0)
    acs_f0_raw = (acs['SCHL'] >= 21).astype(float) * acs['WAGP'] / 200000.0
    acs_f1_raw = (acs['HINS1'] == 1).astype(float) * acs['PINCP'] / 200000.0

    def pack(df, means_note):
        rows = []
        for b in range(3):
            for h in range(2):
                m = (df['band'].to_numpy() == b) & (df['h'].to_numpy() == h)
                rows.append({
                    'band': b, 'h': h, 'n': int(m.sum()),
                    'f0_mean': float(df.loc[m, 'f0'].mean()),
                    'f1_mean': float(df.loc[m, 'f1'].mean()),
                    'f0_min': float(df.loc[m, 'f0'].min()),
                    'f0_max': float(df.loc[m, 'f0'].max()),
                    'f1_min': float(df.loc[m, 'f1'].min()),
                    'f1_max': float(df.loc[m, 'f1'].max()),
                })
        return {
            'n': int(len(df)),
            'f0_outside_01': int(((df['f0'] < 0) | (df['f0'] > 1)).sum()),
            'f1_outside_01': int(((df['f1'] < 0) | (df['f1'] > 1)).sum()),
            'mean_by_band_h': rows,
            'note': means_note,
        }

    nh_rep = pack(nh, 'means of stored f0, f1 on complete respondents')
    acs_rep = pack(acs, 'means of stored f0, f1 on complete respondents')
    nh_rep['candidates'] = {
        'f0_inferred_unclipped': _candidate_report(nh['f0'], nh_f0_raw),
        'f0_clip_product': _candidate_report(nh['f0'], nh_f0_clip),
        'f0_clip_sbp_factor': _candidate_report(nh['f0'], nh_f0_fac),
        'f1_inferred_unclipped': _candidate_report(nh['f1'], nh_f1_raw),
        'f1_clip_product': _candidate_report(nh['f1'], nh_f1_clip),
    }
    acs_rep['candidates'] = {
        'f0_inferred_unclipped': _candidate_report(acs['f0'], acs_f0_raw),
        'f0_clip_product': _candidate_report(acs['f0'], np.clip(acs_f0_raw, 0.0, 1.0)),
        'f1_inferred_unclipped': _candidate_report(acs['f1'], acs_f1_raw),
        'f1_clip_product': _candidate_report(acs['f1'], np.clip(acs_f1_raw, 0.0, 1.0)),
    }
    nh_rep['matched_formula'] = {
        'f0': '(INDFMPIR/5)*clip((BPXSY1-70)/130, 0, 1)',
        'f1': 'clip(1{ALQ101=1}*(BMXWAIST-50)/100, 0, 1)',
    }
    acs_rep['matched_formula'] = {
        'f0': 'clip(1{SCHL>=21}*WAGP/200000, 0, 1)',
        'f1': 'clip(1{HINS1=1}*PINCP/200000, 0, 1)',
    }
    if nh_rep['candidates']['f0_clip_sbp_factor']['max_abs'] > 1e-8:
        raise RuntimeError('NHANES f0 formula did not match')
    if nh_rep['candidates']['f1_clip_product']['max_abs'] > 1e-8:
        raise RuntimeError('NHANES f1 formula did not match')
    if acs_rep['candidates']['f0_clip_product']['max_abs'] > 1e-8:
        raise RuntimeError('ACS f0 formula did not match')
    if acs_rep['candidates']['f1_clip_product']['max_abs'] > 1e-8:
        raise RuntimeError('ACS f1 formula did not match')
    if nh_rep['f0_outside_01'] or nh_rep['f1_outside_01']:
        raise RuntimeError('NHANES stored f outside [0,1]')
    if acs_rep['f0_outside_01'] or acs_rep['f1_outside_01']:
        raise RuntimeError('ACS stored f outside [0,1]')
    return {'nhanes': nh_rep, 'acs': acs_rep}


def init_pools():
    global _READY, POOL, MU, NCOMP, ORACLE_A, DISC, CELL_MEAN
    if _READY:
        return
    path = os.path.join(DATA_DIR, 'nhanes_complete_values.csv')
    df = pd.read_csv(path, usecols=['band', 'h', 'f0', 'f1'])
    POOL = [[None, None] for _ in range(3)]
    NCOMP = np.zeros((3, 2), dtype=int)
    CELL_MEAN = np.zeros((3, 2, 2))
    MU = np.zeros((3, 2))
    for b in range(3):
        for h in range(2):
            m = (df['band'].to_numpy() == b) & (df['h'].to_numpy() == h)
            arr = np.column_stack([
                df.loc[m, 'f0'].to_numpy(float),
                df.loc[m, 'f1'].to_numpy(float),
            ])
            if len(arr) == 0:
                raise RuntimeError('empty complete-value cell band %s h %s' % (b, h))
            POOL[b][h] = np.ascontiguousarray(arr)
            NCOMP[b, h] = len(arr)
            CELL_MEAN[b, h] = arr.mean(axis=0)
        for j in range(2):
            MU[b, j] = (1.0 - PI[b]) * CELL_MEAN[b, 0, j] + PI[b] * CELL_MEAN[b, 1, j]
    ORACLE_A = np.zeros(3)
    DISC = np.zeros(3)
    for b in range(3):
        _, q, pi = transport_zero(KAP[b, 0], KAP[b, 1])
        if abs(float(q[15]) - float(pi[15, 15])) > 1e-6:
            raise RuntimeError('oracle task probability mismatch on band %s' % b)
        if level(q, q) > 1e-8:
            raise RuntimeError('oracle released laws are not identical on band %s' % b)
        ORACLE_A[b] = float(q[15])
        lev, _, _, _, a0, a1 = evalpol(b, identity_policy(), identity_policy())
        if abs(float(a0) - float(KAP[b, 0, 15])) > 1e-10 or abs(float(a1) - float(KAP[b, 1, 15])) > 1e-10:
            raise RuntimeError('identity task rate is not the all-four cell')
        DISC[b] = float(lev)
    _READY = True


def draw_cc(vrng, b, a0, a1):
    """Binomial complete-case counts, then values with replacement inside (band, h)."""
    n = int(NB[b])
    n1 = int(vrng.binomial(n, float(PI[b])))
    s1 = int(vrng.binomial(n1, a1))
    s0 = int(vrng.binomial(n - n1, a0))
    ns = s0 + s1
    if ns == 0:
        return 0, float('nan'), float('nan')
    chunks = []
    if s0:
        chunks.append(POOL[b][0][vrng.integers(0, POOL[b][0].shape[0], size=s0)])
    if s1:
        chunks.append(POOL[b][1][vrng.integers(0, POOL[b][1].shape[0], size=s1)])
    vals = np.concatenate(chunks, axis=0)
    return ns, float(vals[:, 0].mean()), float(vals[:, 1].mean())


def write_slot(out, mi, ci, b, ns, m0, tau, lev, c0, c1, eps, vrng, a0, a1):
    sl = mi * NCFG + ci
    rel = ns >= m0 and ns > 0
    out[sl, b, FNS] = ns
    out[sl, b, FM0] = m0
    out[sl, b, FTAU] = tau
    out[sl, b, FLEV] = lev
    out[sl, b, FCC0] = c0
    out[sl, b, FCC1] = c1
    out[sl, b, FA0] = a0
    out[sl, b, FA1] = a1
    if rel:
        scale = 1.0 / (m0 * eps)
        out[sl, b, FEST0] = c0 + float(vrng.laplace(0.0, scale))
        out[sl, b, FEST1] = c1 + float(vrng.laplace(0.0, scale))
        out[sl, b, FREL] = 1.0
    else:
        out[sl, b, FEST0] = np.nan
        out[sl, b, FEST1] = np.nan
        out[sl, b, FREL] = 0.0


def emit(out, vrng, mi, b, policy, ET, a0, a1, m0, tau, lev):
    a0s = as_prob(a0, '%s a0' % policy)
    a1s = as_prob(a1, '%s a1' % policy)
    ns, c0, c1 = draw_cc(vrng, b, a0s, a1s)
    hit = False
    for ci, cfg in enumerate(CFGS):
        if cfg['policy'] != policy or abs(cfg['E_T'] - ET) > 1e-12:
            continue
        hit = True
        write_slot(
            out, mi, ci, b, ns, m0, tau, lev, c0, c1, cfg['eps_V'], vrng, float(a0), float(a1),
        )
    if not hit:
        raise RuntimeError('no config for %s E_T=%s' % (policy, ET))


def _one_rep(seed):
    init_pools()
    rng = np.random.default_rng([20261115, int(seed)])
    vrng = np.random.default_rng([20261115, int(seed), 99])
    out = np.empty((NSLOT, 3, len(FIELDS)), np.float64)
    for mi, mult in enumerate(MULTS):
        for b in range(3):
            _cal, jlo, jhi, blo, bhi, _ghat = draw_calibration(rng, b, mult)
            tau_n = min(certified_atom_min(jlo[h], jhi[h]) for h in range(2))
            m0_n = m0_of(NB[b], tau_n)
            a0_n = float(KAP[b, 0, 15])
            a1_n = float(KAP[b, 1, 15])
            saved_b = {}
            saved_e = {}
            for E in ES:
                Qb0, Qb1, Fv = block_from_box(blo, bhi, E)
                if not np.isfinite(Fv):
                    raise RuntimeError('nonfinite block Fv band %s mult %s E %s' % (b, mult, E))
                lev_b, _, _, _, a0b, a1b = evalpol(b, Qb0, Qb1)
                saved_b[E] = (float(a0b), float(a1b), m0_of(NB[b], float(Fv)), float(Fv), float(lev_b))
                Q0, Q1, _, tS = exact_joint_box(jlo, jhi, E, 'task')
                if not np.isfinite(tS):
                    raise RuntimeError('nonfinite tauS band %s mult %s E %s' % (b, mult, E))
                lev, _, _, _, a0, a1 = evalpol(b, Q0, Q1)
                # Same threshold and the same Generator calls as sim.py lines 61-65.
                m0 = int(np.ceil(NB[b] * tS / 2)) if tS > 0 else 1
                n1 = rng.binomial(NB[b], PI[b])
                s1 = rng.binomial(n1, a1)
                s0 = rng.binomial(NB[b] - n1, a0)
                if s0 + s1 >= max(m0, 1):
                    rng.laplace(0, 1.0 / (max(m0, 1) * EPSV))
                saved_e[E] = (float(a0), float(a1), max(m0, 1), float(tS), float(lev))
            emit(out, vrng, mi, b, 'none', 0.0, a0_n, a1_n, m0_n, tau_n, float(DISC[b]))
            for E in ES:
                emit(out, vrng, mi, b, 'block', float(E), *saved_b[E])
            for E in ES:
                emit(out, vrng, mi, b, 'exjoint_t', float(E), *saved_e[E])
            emit(
                out, vrng, mi, b, 'oracle', 0.0,
                float(ORACLE_A[b]), float(ORACLE_A[b]), m0_of(NB[b], float(ORACLE_A[b])),
                float(ORACLE_A[b]), 0.0,
            )
    return out


def one_rep(seed):
    try:
        return _one_rep(seed)
    except Exception as exc:
        raise RuntimeError('value_sim seed %s failed: %s' % (seed, exc)) from exc


def _worker_init(q):
    q.put(sha256_file(LPS_PATH))


class _RecordingGenerator:
    """Proxy so selfcheck can see multinomial draws without editing sim.py."""

    def __init__(self, rng):
        self._rng = rng
        self.multi = []

    def multinomial(self, n, pvals, size=None):
        out = self._rng.multinomial(n, pvals, size=size)
        self.multi.append(np.array(out, copy=True))
        return out

    def __getattr__(self, name):
        return getattr(self._rng, name)


def selfcheck():
    """Determinism, and calibration multinomials identical to sim.one_rep(0)."""
    import sim as simmod
    init_pools()
    real = np.random.default_rng
    made = []

    def factory(seed=None):
        rec = _RecordingGenerator(real(seed))
        made.append(rec)
        return rec

    np.random.default_rng = factory
    t0 = time.time()
    try:
        simmod.one_rep(0)
        if len(made) != 1:
            raise RuntimeError('sim.one_rep created %s generators' % len(made))
        calls_sim = made[0].multi
        _one_rep(0)
    finally:
        np.random.default_rng = real
    if len(made) != 3:
        raise RuntimeError('expected sim rng plus value rng and value vrng, got %s' % len(made))
    calls_mine = made[1].multi
    if made[2].multi:
        raise RuntimeError('value stream called multinomial')
    if len(calls_sim) != len(calls_mine):
        raise RuntimeError('multinomial count sim %s vs value_sim %s' % (len(calls_sim), len(calls_mine)))
    for i, (a, b) in enumerate(zip(calls_sim, calls_mine)):
        if a.shape != b.shape or not np.array_equal(a, b):
            raise RuntimeError('calibration multinomial mismatch at call %s' % i)
    a = _one_rep(0)
    b = _one_rep(0)
    if not np.allclose(a, b, equal_nan=True):
        raise RuntimeError('seed 0 is not deterministic')
    w_task = float(sum(WB[b] * min(KAP[b, 0, 15], KAP[b, 1, 15]) for b in range(3)))
    w_orc = float(sum(WB[b] * ORACLE_A[b] for b in range(3)))
    print(
        'selfcheck PASS multinomials %d in %.1fs | disclosed %s | identity task_min %.4f | oracle task %.4f'
        % (len(calls_sim), time.time() - t0, np.round(DISC, 5).tolist(), w_task, w_orc),
        flush=True,
    )


def run(reps, seed0, procs, tag, do_selfcheck):
    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, tag + '.log')
    orig_out, orig_err = sys.stdout, sys.stderr
    log_f = open(log_path, 'w', encoding='utf-8', newline='\n')

    class Tee:
        def write(self, s):
            orig_out.write(s)
            orig_out.flush()
            log_f.write(s)
            log_f.flush()
        def flush(self):
            orig_out.flush()
            log_f.flush()

    tee = Tee()
    sys.stdout = tee
    sys.stderr = tee
    t_all = time.time()
    try:
        print('value_sim tag %s reps %s seed0 %s procs %s' % (tag, reps, seed0, procs), flush=True)
        copy_inputs()
        formulas = formula_check()
        init_pools()
        print('NHANES f0/f1 inside [0,1]; matched formulas:', flush=True)
        print('  f0 ' + formulas['nhanes']['matched_formula']['f0'], flush=True)
        print('  f1 ' + formulas['nhanes']['matched_formula']['f1'], flush=True)
        print('ACS matched formulas:', flush=True)
        print('  f0 ' + formulas['acs']['matched_formula']['f0'], flush=True)
        print('  f1 ' + formulas['acs']['matched_formula']['f1'], flush=True)
        hash_before = sha256_file(LPS_PATH)
        print('lps.py sha256 before selfcheck %s' % hash_before, flush=True)
        t_sc = 0.0
        if do_selfcheck:
            t1 = time.time()
            selfcheck()
            t_sc = time.time() - t1
        hash_pool = sha256_file(LPS_PATH)
        print('lps.py sha256 before pool %s' % hash_pool, flush=True)
        data = np.empty((reps, NSLOT, 3, len(FIELDS)), np.float64)
        t_rep = time.time()
        seeds = list(range(seed0, seed0 + reps))
        if procs <= 1:
            worker_hash = hash_pool
            for i, seed in enumerate(seeds):
                data[i] = one_rep(seed)
                if (i + 1) % 10 == 0 or i + 1 == reps:
                    el = time.time() - t_rep
                    print('reps %d / %d elapsed_s %.1f per_rep %.3f' % (i + 1, reps, el, el / (i + 1)), flush=True)
        else:
            ctx = mp.get_context('spawn')
            hash_q = ctx.Queue()
            worker_hashes = []
            with ctx.Pool(procs, initializer=_worker_init, initargs=(hash_q,)) as pool:
                for i, arr in enumerate(pool.imap(one_rep, seeds, chunksize=5)):
                    data[i] = arr
                    if (i + 1) % 50 == 0 or i + 1 == reps:
                        el = time.time() - t_rep
                        print(
                            'reps %d / %d elapsed_s %.1f per_rep %.3f eta_s %.0f'
                            % (i + 1, reps, el, el / (i + 1), el / (i + 1) * (reps - i - 1)),
                            flush=True,
                        )
                for _ in range(procs):
                    worker_hashes.append(hash_q.get(timeout=120))
            worker_hash = sorted(set(worker_hashes))
        t_reps = time.time() - t_rep
        hash_end = sha256_file(LPS_PATH)
        print('lps.py sha256 after pool %s' % hash_end, flush=True)
        print('worker import hashes %s' % worker_hash, flush=True)
        mu_rows = []
        for b in range(3):
            mu_rows.append({
                'band': b,
                'n_survey': int(NB[b]),
                'n_survey_h0': int(NBH[b, 0]),
                'n_survey_h1': int(NBH[b, 1]),
                'survey_share_h1': float(PI[b]),
                'sample_share': float(WB[b]),
                'n_complete_h0': int(NCOMP[b, 0]),
                'n_complete_h1': int(NCOMP[b, 1]),
                'mean_f0_h0': float(CELL_MEAN[b, 0, 0]),
                'mean_f0_h1': float(CELL_MEAN[b, 1, 0]),
                'mean_f1_h0': float(CELL_MEAN[b, 0, 1]),
                'mean_f1_h1': float(CELL_MEAN[b, 1, 1]),
                'mu_f0': float(MU[b, 0]),
                'mu_f1': float(MU[b, 1]),
                'disclosed_level': float(DISC[b]),
                'oracle_task': float(ORACLE_A[b]),
            })
        meta = {
            'survey': 'NHANES',
            'acs': 'not run; acs.py has no Clopper-Pearson calibration or policy replication loop',
            'tag': tag,
            'reps': int(reps),
            'seed0': int(seed0),
            'seeds': 'inclusive range [seed0, seed0+reps)',
            'rng_calibration': '[20261115, seed] multinomials then the sim.py exjoint binomial/Laplace sync',
            'rng_values': '[20261115, seed, 99]',
            'procs': int(procs),
            'fields': FIELDS,
            'configs': CFGS,
            'mults': [int(m) for m in MULTS],
            'L': 1.0,
            'm0_rule': (
                'max{1, ceil(n_b*tau/2)} with sim.py branch (ceil if tau>0 else 1). '
                'exjoint_t tau = LP tauS; block tau = product-box Fv; '
                'none tau = min over groups of the simplex lower bound on the all-four cell; '
                'oracle tau = known-law transport task probability.'
            ),
            'bound_rule': (
                'mean over released replications of '
                'sqrt(tanh(E/4)^2 + 1/(4*NS) + 2/(m0^2*eps_V^2)), L=1. '
                'E is the band true pattern level for no suppression and the nominal E_T otherwise. '
                'mean_bound_true_level always plugs in the realised band level.'
            ),
            'rmse_rule': 'sqrt of the mean squared error on replications with NS >= m0; non-release is a separate rate',
            'sel_bias_rule': 'absolute difference between mu_b and the mean of complete-case means over replications with NS >= 1',
            'total_level_rule': 'mean over replications of (worst-band true pattern level) + eps_V',
            'pct_over_rule': '100 * fraction of replications with worst-band pattern level > E_T + 1e-4',
            'aggregation': (
                'weighted = sample-share (WB) average of band metrics. '
                'RMSE, absolute bias, selection bias, and the analytical bound renormalize those shares '
                'over bands with at least one release (n_bands_rmse). '
                'release_rate includes bands that never release. max = maximum across bands.'
            ),
            'lps_sha256_before_selfcheck': hash_before,
            'lps_sha256_before_pool': hash_pool,
            'lps_sha256_worker_at_first_result': worker_hash,
            'lps_sha256_after_pool': hash_end,
            'nhanes_csv_sha256': sha256_file(os.path.join(DATA_DIR, 'nhanes_complete_values.csv')),
            'acs_csv_sha256': sha256_file(os.path.join(DATA_DIR, 'acs_complete_values.csv')),
            'elapsed_selfcheck_sec': t_sc,
            'elapsed_reps_sec': t_reps,
            'elapsed_total_sec': time.time() - t_all,
            'formula_check': formulas,
            'estimand': mu_rows,
            'per_rep_sec': t_reps / reps,
            'extrapolated_5000_sec': (t_reps / reps) * 5000.0,
        }
        npz_path = os.path.join(OUT_DIR, tag + '_reps.npz')
        meta_bytes = json.dumps(meta).encode('utf-8')
        np.savez_compressed(
            npz_path,
            data=data,
            mu=MU,
            wb=np.asarray(WB, float),
            nb=np.asarray(NB, float),
            meta=np.frombuffer(meta_bytes, dtype=np.uint8).copy(),
        )
        print('wrote %s' % npz_path, flush=True)
        print(
            'timing reps %.1fs (%.3f s/rep) extrapolate_5000 %.0fs selfcheck %.1fs'
            % (t_reps, t_reps / reps, (t_reps / reps) * 5000.0, t_sc),
            flush=True,
        )
        import value_summary
        value_summary.summarize(npz_path, os.path.join(OUT_DIR, tag + '.csv'), os.path.join(OUT_DIR, tag + '.json'))
        meta['elapsed_total_sec'] = time.time() - t_all
        print('done total_s %.1f' % meta['elapsed_total_sec'], flush=True)
        return meta
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err
        log_f.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reps', type=int, default=5000)
    p.add_argument('--seed0', type=int, default=0)
    p.add_argument('--procs', type=int, default=0)
    p.add_argument('--tag', default='value_matched_budget_2026-11-15')
    p.add_argument('--skip-selfcheck', action='store_true')
    args = p.parse_args()
    procs = args.procs if args.procs > 0 else max(1, (os.cpu_count() or 2) - 2)
    run(args.reps, args.seed0, procs, args.tag, do_selfcheck=not args.skip_selfcheck)


if __name__ == '__main__':
    mp.freeze_support()
    main()
