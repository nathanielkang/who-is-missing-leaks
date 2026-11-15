import os
HERE = os.path.dirname(os.path.abspath(__file__))
"""Survey-calibrated certification simulation (Table 'prot')."""
import sys, time, json, numpy as np, multiprocessing as mp
sys.path.insert(0, HERE)
from core import *; from lps import *

C = np.load(HERE + '/data/nhanes_counts.npy'); TOT = C.sum(0)
KAP = np.array([[jeffreys(TOT[b, h]) for h in range(2)] for b in range(3)])
NBH = TOT.sum(2).astype(int); NB = NBH.sum(1); WB = NB / NB.sum(); PI = NBH[:, 1] / NB
BETA = 0.05; ES = (0.5, 1.0); MULTS = (1, 4); EPSV = 0.5
POL = ['none', 'fitted', 'balprod', 'block', 'exprod', 'exjoint_f', 'exjoint_t']

def evalpol(b, Q0, Q1):
    q0 = released(KAP[b, 0], Q0); q1 = released(KAP[b, 1], Q1)
    r0, a0 = rates(q0); r1, a1 = rates(q1)
    lev = level(q0, q1)
    share = PI[b] * a1 / (PI[b] * a1 + (1 - PI[b]) * a0) if (a0 + a1) > 0 else np.nan
    return lev, np.minimum(r0, r1).sum(), min(a0, a1), share - PI[b], a0, a1

def one_rep(seed):
    rng = np.random.default_rng([20261115, seed])
    out = {}
    for mi, mult in enumerate(MULTS):
        cov_j = True; cov_p = True
        per = {(p, ei): [] for p in POL for ei in range(2)}
        hoef = [True, True]; cc = {}
        for b in range(3):
            cal = np.array([rng.multinomial(int(mult * NBH[b, h]), KAP[b, h]) for h in range(2)])
            n = cal.sum(1)
            jlo = np.zeros((2, NP)); jhi = np.zeros((2, NP)); blo = np.zeros((2, K)); bhi = np.zeros((2, K)); ghat = np.zeros((2, K))
            for h in range(2):
                for r in range(NP):
                    jlo[h, r], jhi[h, r] = cp_interval(cal[h, r], n[h], BETA / 96)
                x = PAT.T @ cal[h]; ghat[h] = x / n[h]
                for k in range(K):
                    blo[h, k], bhi[h, k] = cp_interval(x[k], n[h], BETA / 24)
            cov_j &= bool(np.all((jlo <= KAP[b]) & (KAP[b] <= jhi)))
            gtrue = np.array([margins(KAP[b, h]) for h in range(2)])
            cov_p &= bool(np.all((blo <= gtrue) & (gtrue <= bhi)))
            # Theorem joint box in the common-width form
            ck = (bhi / blo).max(0); ell = bhi / ck
            # Hoeffding box of Proposition cert, failure prob beta/3 per band
            bh = np.sqrt(np.log(4 * K / (BETA / 3)) / (2 * n))[:, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                chk = np.where(ghat - bh > 0, (ghat + bh) / (ghat - bh), np.inf).max(0)
            for ei, E in enumerate(ES):
                hoef[ei] &= bool(np.log(chk).sum() <= E)
                pols = {}
                pols['none'] = (identity_policy(), identity_policy())
                gm = ghat.min(0); pols['fitted'] = (product_policy(gm / ghat[0]), product_policy(gm / ghat[1]))
                tau = balanced_alloc(E, ck, ell.min(0)); pols['balprod'] = (product_policy(tau / ell[0]), product_policy(tau / ell[1]))
                aS = ell.prod(1); Fv = F(E, aS.min(), ck.prod()); pols['block'] = (block_policy(Fv / aS[0]), block_policy(Fv / aS[1]))
                Q0, Q1, _, _ = exact_product_box(blo, bhi, E, 'fields'); pols['exprod'] = (Q0, Q1)
                Q0, Q1, _, _ = exact_joint_box(jlo, jhi, E, 'fields'); pols['exjoint_f'] = (Q0, Q1)
                Q0, Q1, _, tS = exact_joint_box(jlo, jhi, E, 'task'); pols['exjoint_t'] = (Q0, Q1)
                for p in POL:
                    per[(p, ei)].append(evalpol(b, *pols[p]))
                # private complete-case estimate under the joint-box task program (Theorem ign (ii))
                lev, _, _, _, a0, a1 = per[('exjoint_t', ei)][-1]
                m0 = int(np.ceil(NB[b] * tS / 2)) if tS > 0 else 1
                n1 = rng.binomial(NB[b], PI[b]); s1 = rng.binomial(n1, a1); s0 = rng.binomial(NB[b] - n1, a0)
                NS = s0 + s1
                if NS >= max(m0, 1):
                    est = s1 / NS + rng.laplace(0, 1.0 / (max(m0, 1) * EPSV)); cc[(b, ei)] = (est - PI[b], NS, m0, tS)
                else:
                    cc[(b, ei)] = (np.nan, NS, m0, tS)
        res = {'cov_j': cov_j, 'cov_p': cov_p, 'hoef': hoef}
        for p in POL:
            for ei in range(2):
                arr = np.array(per[(p, ei)])  # band x (lev, fields, task, dev, a0, a1)
                res[f'{p}|{ei}'] = [float(arr[:, 0].max()), float(WB @ arr[:, 1]), float(WB @ arr[:, 2]), float(np.nanmax(np.abs(arr[:, 3]))) if np.isfinite(arr[:, 3]).any() else None]
        res['cc'] = {f'{b}|{ei}': [None if (isinstance(x, float) and np.isnan(x)) else float(x) for x in cc[(b, ei)]] for b in range(3) for ei in range(2)}
        out[str(mult)] = res
    return out

if __name__ == '__main__':
    R0, R1, nproc, outf = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    t0 = time.time()
    with mp.Pool(nproc) as pool:
        results = []
        for i, r in enumerate(pool.imap(one_rep, range(R0, R1), chunksize=5)):
            results.append(r)
            if (i + 1) % 100 == 0:
                json.dump(results, open(outf, 'w')); print(i + 1, round(time.time() - t0), flush=True)
    json.dump(results, open(outf, 'w')); print('done', len(results), round(time.time() - t0), flush=True)
