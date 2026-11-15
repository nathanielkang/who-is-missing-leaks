import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys, numpy as np, pandas as pd
sys.path.insert(0, HERE)
from core import *
d = pd.read_csv(sys.argv[1])
a = d[(d.RIDAGEYR >= 20) & d.DMDCITZN.isin([1, 2]) & d.DMDEDUC2.isin([1, 2, 3, 4, 5]) & d.SMQ020.isin([1, 2])].copy()
a['h'] = (a.DMDCITZN == 2).astype(int)
a['band'] = np.where(a.RIDAGEYR < 40, 0, np.where(a.RIDAGEYR < 60, 1, 2))
R = np.c_[a.INDFMPIR.notna(), a.ALQ101.isin([1, 2]), a.BPXSY1.notna(), a.BMXWAIST.notna()].astype(int)
a['pat'] = (R * (1 << np.arange(4))).sum(1)
a['psu'] = a.SDMVSTRA.astype(int) * 10 + a.SDMVPSU.astype(int)
print('n', len(a), 'share', a.h.mean(), 'strata', a.SDMVSTRA.nunique(), 'psu', a.psu.nunique())
# count array: C[psu, band, h, pat]
psus = np.sort(a.psu.unique()); pi = {p: i for i, p in enumerate(psus)}
C = np.zeros((len(psus), 3, 2, 16))
for p, b, h, r in zip(a.psu, a.band, a.h, a.pat):
    C[pi[p], b, h, r] += 1
np.save(HERE + '/data/nhanes_counts.npy', C)
np.save(HERE + '/data/nhanes_psu.npy', psus)
tot = C.sum(0)
print('group sizes band x h:\n', tot.sum(2))
def stats(tot):
    pooled = tot.sum(0)  # h x pat
    out = {}
    n = pooled.sum(1)
    g = np.array([margins(pooled[h] / n[h]) for h in range(2)])  # h x k
    out['rate'] = margins(pooled.sum(0) / n.sum())
    out['g1'] = g[1]; out['g0'] = g[0]
    out['e1'] = np.abs(np.log(g[1] / g[0])); out['e0'] = np.abs(np.log((1 - g[1]) / (1 - g[0])))
    gb = np.array([[margins(tot[b, h] / tot[b, h].sum()) for h in range(2)] for b in range(3)])  # b,h,k
    out['band_e0'] = np.abs(np.log((1 - gb[:, 1]) / (1 - gb[:, 0]))).max(0)
    out['band_e1'] = np.abs(np.log(gb[:, 1] / gb[:, 0])).max(0)
    out['band_max'] = np.maximum(out['band_e0'], out['band_e1'])
    k0 = jeffreys(pooled[0]); k1 = jeffreys(pooled[1])
    out['joint_occ'], out['prod_occ'] = occ_levels(pooled[0] / n[0], pooled[1] / n[1])   # values reported in Table leak
    out['band_joint_occ'] = [occ_levels(tot[b, 0] / tot[b, 0].sum(), tot[b, 1] / tot[b, 1].sum())[0] for b in range(3)]
    out['band_hidden_lp'] = [hidden_joint_level(tot[b, 0] / tot[b, 0].sum(), tot[b, 1] / tot[b, 1].sum(), .5, .5) for b in range(3)]
    out['band_joint_jeffreys'] = [level(jeffreys(tot[b, 0]), jeffreys(tot[b, 1])) for b in range(3)]   # simulation truth of Section V-C
    out['joint'] = level(k0, k1)
    out['prod'] = level(prod_from_margins(margins(k0)), prod_from_margins(margins(k1)))
    out['prod_raw'] = level(prod_from_margins(g[0]), prod_from_margins(g[1]))
    out['joint_band_smooth'] = max(level(jeffreys(tot[b, 0]), jeffreys(tot[b, 1])) for b in range(3))
    out['joint_band_occ'] = max(level_occupied(tot[b, 0] / tot[b, 0].sum(), tot[b, 1] / tot[b, 1].sum()) for b in range(3))
    out['hid1'] = np.array([hidden_closed([g[0][k]], [g[1][k]], .5, .5) for k in range(4)])
    out['hid_closed'] = hidden_closed(g[0], g[1], .5, .5)
    out['hid_closed_s'] = hidden_closed(margins(k0), margins(k1), .5, .5)
    return out, k0, k1, pooled
o, k0, k1, pooled = stats(tot)
for k, v in o.items(): print(k, np.round(v, 4))
print('hidden LP joint (raw, reported)', hidden_joint_level(pooled[0] / pooled[0].sum(), pooled[1] / pooled[1].sum(), .5, .5), ' Jeffreys:', hidden_joint_level(k0, k1, .5, .5))
print('hidden LP joint max over bands', [round(hidden_joint_level(jeffreys(tot[b,0]), jeffreys(tot[b,1]), .5, .5),4) for b in range(3)])
print('pooled counts h0', pooled[0].astype(int)); print('pooled counts h1', pooled[1].astype(int))
r = np.argmax(np.abs(np.log(k0 / k1))); print('argmax pattern', r, PAT[r], pooled[:, r])
