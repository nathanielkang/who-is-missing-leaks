"""Aggregate value_sim replications into the matched-budget CSV and JSON.

Every printed table is built from the CSV after it is written.
"""
import csv
import json
import math
import os
import sys

import numpy as np

COLS = [
    'survey', 'policy', 'E_tot', 'E_T', 'eps_V', 'mult', 'f', 'agg',
    'n_reps', 'n_released', 'n_complete_case',
    'rmse', 'abs_bias', 'sel_bias', 'sel_bias_signed',
    'release_rate', 'nonrelease_rate',
    'pattern_level', 'pattern_level_worst', 'pattern_level_worst_maxrep',
    'total_level', 'pct_over',
    'mean_bound', 'mean_bound_true_level',
    'mean_NS', 'mean_m0', 'mean_tau', 'mean_task_min', 'mean_incl',
    'n_bands_rmse',
]


def _wmean_defined(vals, w):
    """Sample-share mean, renormalized over bands that have a finite value."""
    mask = []
    for v in vals:
        ok = v is not None and isinstance(v, (int, float, np.floating)) and math.isfinite(float(v))
        mask.append(ok)
    if not any(mask):
        return None
    ww = np.array([float(w[i]) if mask[i] else 0.0 for i in range(len(vals))])
    ww = ww / ww.sum()
    return float(sum(ww[i] * float(vals[i]) for i in range(len(vals)) if mask[i]))


def _load(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(bytes(z['meta']).decode('utf-8'))
    return z['data'], z['mu'], z['wb'], z['nb'], meta


def _finite_mean(x):
    """Mean that keeps a positive infinity instead of dropping it."""
    x = np.asarray(x, float)
    if x.size == 0 or np.all(np.isnan(x)):
        return None
    if np.isposinf(x).any():
        return float('inf')
    if np.isneginf(x).any():
        return float('-inf')
    return float(np.nanmean(x))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if math.isnan(x):
            return None
        if math.isinf(x):
            return 'inf' if x > 0 else '-inf'
        return x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _num(x):
    if x is None:
        return None
    x = float(x)
    if math.isnan(x):
        return None
    return x


def _root_bound(ET, m, m0, eps):
    th = np.tanh(np.asarray(ET, float) / 4.0)
    return np.sqrt(th ** 2 + 1.0 / (4.0 * m) + 2.0 / (np.square(m0) * eps ** 2))


def _band_metrics(data, slot, b, f, mu_bf, E_T, eps, nb_b):
    """data: (reps, slot, band, field). f is 0 or 1."""
    cc = 4 + f
    est = 6 + f
    NS = data[:, slot, b, 0]
    rel = data[:, slot, b, 8] > 0.5
    cc_mean = data[:, slot, b, cc]
    estv = data[:, slot, b, est]
    m0 = data[:, slot, b, 1]
    tau = data[:, slot, b, 2]
    lev = data[:, slot, b, 3]
    a0 = data[:, slot, b, 9]
    a1 = data[:, slot, b, 10]
    sel = NS >= 1
    out = {
        'n_reps': int(data.shape[0]),
        'n_released': int(rel.sum()),
        'n_complete_case': int(sel.sum()),
        'release_rate': float(rel.mean()),
        'nonrelease_rate': float((~rel).mean()),
        'pattern_level': _finite_mean(lev),
        'mean_NS': float(np.mean(NS)),
        'mean_m0': float(np.mean(m0)),
        'mean_tau': _finite_mean(tau),
        'mean_task_min': _finite_mean(np.minimum(a0, a1)),
        'mean_incl': float(np.mean(NS) / nb_b),
    }
    if rel.any():
        err = estv[rel] - mu_bf
        out['rmse'] = float(np.sqrt(np.mean(err ** 2)))
        out['abs_bias'] = float(abs(np.mean(err)))
        m = NS[rel]
        m0r = m0[rel]
        out['mean_bound'] = float(np.mean(_root_bound(E_T, m, m0r, eps)))
        out['mean_bound_true_level'] = float(np.mean(_root_bound(lev[rel], m, m0r, eps)))
    else:
        out['rmse'] = None
        out['abs_bias'] = None
        out['mean_bound'] = None
        out['mean_bound_true_level'] = None
    if sel.any() and np.isfinite(cc_mean[sel]).any():
        signed = float(np.mean(cc_mean[sel][np.isfinite(cc_mean[sel])]) - mu_bf)
        out['sel_bias_signed'] = signed
        out['sel_bias'] = abs(signed)
    else:
        out['sel_bias_signed'] = None
        out['sel_bias'] = None
    return out


def _wmean(vals, w):
    if any(v is None or not math.isfinite(v) for v in vals):
        return None
    return float(np.dot(w, np.asarray(vals, float)))


def _wmax(vals):
    fin = [v for v in vals if v is not None and math.isfinite(v)]
    if not fin:
        return None
    return float(max(fin))


def build_rows(data, mu, wb, nb, meta):
    fields = meta['fields']
    if fields[:11] != ['NS', 'm0', 'tau', 'level', 'cc0', 'cc1', 'est0', 'est1', 'rel', 'a0', 'a1']:
        raise RuntimeError('unexpected field list %s' % fields)
    rows = []
    nrep = data.shape[0]
    for mi, mult in enumerate(meta['mults']):
        for ci, cfg in enumerate(meta['configs']):
            slot = mi * len(meta['configs']) + ci
            lev = data[:, slot, :, 3]
            worst = np.nanmax(lev, axis=1)
            pct = float(100.0 * np.mean(worst > (cfg['E_T'] + 1e-4)))
            worst_mean = _finite_mean(worst)
            worst_max = None if not np.isfinite(worst).any() else float(np.nanmax(worst))
            if worst_mean is None:
                total = None
            elif math.isinf(worst_mean):
                total = float('inf')
            else:
                total = worst_mean + cfg['eps_V']
            for f, fname in enumerate(('f0', 'f1')):
                band = [
                    _band_metrics(data, slot, b, f, float(mu[b, f]), cfg['E_T'], cfg['eps_V'], float(nb[b]))
                    for b in range(3)
                ]
                if cfg['bound_source'] == 'true_level':
                    for b in range(3):
                        band[b]['mean_bound'] = band[b]['mean_bound_true_level']
                shared = {
                    'survey': meta['survey'],
                    'policy': cfg['policy'],
                    'E_tot': cfg['E_tot'],
                    'E_T': cfg['E_T'],
                    'eps_V': cfg['eps_V'],
                    'mult': int(mult),
                    'f': fname,
                    'n_reps': nrep,
                    'pattern_level_worst': worst_mean,
                    'pattern_level_worst_maxrep': worst_max,
                    'total_level': total,
                    'pct_over': pct,
                }
                for b, bm in enumerate(band):
                    row = dict(shared)
                    row['agg'] = 'band%d' % b
                    row.update(bm)
                    row['pattern_level_worst'] = bm['pattern_level']
                    if bm['pattern_level'] is None:
                        row['total_level'] = None
                    else:
                        row['total_level'] = bm['pattern_level'] + cfg['eps_V']
                    over_b = data[:, slot, b, 3] > (cfg['E_T'] + 1e-4)
                    row['pct_over'] = float(100.0 * np.mean(over_b))
                    row['n_bands_rmse'] = 1 if bm['rmse'] is not None else 0
                    rows.append(row)
                for agg, reduce in (('weighted', lambda vs: _wmean(vs, wb)), ('max', _wmax)):
                    row = dict(shared)
                    row['agg'] = agg
                    for key in (
                        'n_released', 'n_complete_case',
                        'release_rate', 'nonrelease_rate', 'pattern_level',
                        'mean_NS', 'mean_m0', 'mean_tau', 'mean_task_min', 'mean_incl',
                    ):
                        row[key] = reduce([bm[key] for bm in band])
                    # RMSE, bias, and the bound exist only on bands that release.
                    # Weights are renormalized over those bands; release_rate still includes the rest.
                    score = _wmean_defined if agg == 'weighted' else _wmax
                    for key in ('rmse', 'abs_bias', 'sel_bias', 'sel_bias_signed', 'mean_bound', 'mean_bound_true_level'):
                        row[key] = score([bm[key] for bm in band], wb) if agg == 'weighted' else score([bm[key] for bm in band])
                    row['n_bands_rmse'] = int(sum(bm['rmse'] is not None for bm in band))
                    rows.append(row)
    return rows


def _fmt(x):
    if x is None:
        return ''
    if isinstance(x, str):
        return x
    if isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)):
        return str(int(x))
    x = float(x)
    if math.isnan(x):
        return 'nan'
    if math.isinf(x):
        return 'inf' if x > 0 else '-inf'
    return format(x, '.16g')


def write_csv(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({k: _fmt(row.get(k)) for k in COLS})


def _parse(s):
    if s is None or s == '' or s == 'nan':
        return None
    if s == 'inf':
        return float('inf')
    if s == '-inf':
        return float('-inf')
    if s in ('f0', 'f1', 'none', 'block', 'exjoint_t', 'oracle', 'weighted', 'max') or s.startswith('band'):
        return s
    if s == 'NHANES':
        return s
    try:
        if '.' in s or 'e' in s or 'E' in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def read_csv(path):
    with open(path, encoding='utf-8', newline='') as f:
        return [{k: _parse(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _cell(x, nd):
    if x is None:
        return ' ' * (nd + 4)
    if not isinstance(x, (int, float)) or (isinstance(x, float) and not math.isfinite(x)):
        return str(x)
    return f'{float(x):.{nd}f}'


def _pick(rows, agg):
    out = {}
    for r in rows:
        if r['agg'] != agg:
            continue
        key = (r['policy'], float(r['E_T']), float(r['eps_V']), float(r['E_tot']))
        out.setdefault(key, {}).setdefault(int(r['mult']), {})[r['f']] = r
    return out


def print_tables(rows):
    """Print tables from CSV rows. Numbers are the CSV values rounded for display."""
    order = []
    for r in rows:
        key = (r['policy'], float(r['E_T']), float(r['eps_V']), float(r['E_tot']))
        if key not in order and r['agg'] == 'weighted':
            order.append(key)
    for agg, title in (('weighted', 'TABLE weighted (sample-share mean of band metrics; total level is worst band)'),
                       ('max', 'TABLE max (maximum across bands; total level is still the worst-band mean)')):
        bucket = _pick(rows, agg)
        print('===== %s =====' % title, flush=True)
        hdr = (
            f"{'policy':<10} {'ET':>4} {'eps':>4} {'Etot':>4} | "
            f"{'rmse f0/f1 1x':>22} {'rmse f0/f1 4x':>22} | "
            f"{'rel 1x/4x':>16} | "
            f"{'sel f0/f1 1x':>22} {'sel f0/f1 4x':>22} | "
            f"{'tot 1x (%over)':>18} {'tot 4x (%over)':>18} | "
            f"{'bound 1x/4x':>18} | "
            f"{'abs f0/f1 1x':>22} | "
            f"{'bands':>5}"
        )
        print(hdr, flush=True)
        for key in order:
            g = bucket[key]
            def pair(mult, field, nd):
                a = g[mult]['f0'][field]
                b = g[mult]['f1'][field]
                return '%s/%s' % (_cell(a, nd).strip(), _cell(b, nd).strip())
            def one(mult, field, nd):
                return _cell(g[mult]['f0'][field], nd).strip()
            # Release, total level, percent over, and the bound do not depend on f.
            for field in ('release_rate', 'total_level', 'pct_over', 'mean_bound'):
                for mult in (1, 4):
                    x = g[mult]['f0'][field]
                    y = g[mult]['f1'][field]
                    if x is None and y is None:
                        continue
                    if x is None or y is None or abs(float(x) - float(y)) > 1e-9:
                        raise RuntimeError('f0/f1 mismatch on %s %s mult %s' % (key, field, mult))
            line = (
                f"{key[0]:<10} {key[1]:4.1f} {key[2]:4.1f} {key[3]:4.1f} | "
                f"{pair(1, 'rmse', 4):>22} {pair(4, 'rmse', 4):>22} | "
                f"{one(1, 'release_rate', 4) + '/' + one(4, 'release_rate', 4):>16} | "
                f"{pair(1, 'sel_bias', 4):>22} {pair(4, 'sel_bias', 4):>22} | "
                f"{(one(1, 'total_level', 3) + ' (' + one(1, 'pct_over', 2) + ')'):>18} "
                f"{(one(4, 'total_level', 3) + ' (' + one(4, 'pct_over', 2) + ')'):>18} | "
                f"{(one(1, 'mean_bound', 4) + '/' + one(4, 'mean_bound', 4)):>18} | "
                f"{pair(1, 'abs_bias', 4):>22} | "
                f"{str(g[1]['f0']['n_bands_rmse']) + '/' + str(g[4]['f0']['n_bands_rmse']):>5}"
            )
            print(line, flush=True)
        print('===== SANITY %s task_min / mean_tau / mean_incl / mean_m0 =====' % agg, flush=True)
        for key in order:
            g = bucket[key]
            print(
                '%s ET=%s eps=%s | 1x task_min %s tau %s incl %s m0 %s | 4x task_min %s tau %s incl %s m0 %s'
                % (
                    key[0], key[1], key[2],
                    one_s(g, 1, 'mean_task_min'), one_s(g, 1, 'mean_tau'),
                    one_s(g, 1, 'mean_incl'), one_s(g, 1, 'mean_m0'),
                    one_s(g, 4, 'mean_task_min'), one_s(g, 4, 'mean_tau'),
                    one_s(g, 4, 'mean_incl'), one_s(g, 4, 'mean_m0'),
                ),
                flush=True,
            )


def one_s(g, mult, field):
    x = g[mult]['f0'][field]
    if x is None:
        return 'nan'
    return f'{float(x):.4f}'


def summarize(npz_path, csv_path, json_path):
    data, mu, wb, nb, meta = _load(npz_path)
    rows = build_rows(data, mu, wb, nb, meta)
    write_csv(csv_path, rows)
    reread = read_csv(csv_path)
    print_tables(reread)
    # JSON records are the CSV values, so the two files cannot disagree.
    payload = dict(meta)
    payload['csv'] = os.path.abspath(csv_path)
    payload['records'] = _jsonable(reread)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=1, allow_nan=False)
    print('wrote %s' % csv_path, flush=True)
    print('wrote %s' % json_path, flush=True)
    return reread


if __name__ == '__main__':
    summarize(sys.argv[1], sys.argv[2], sys.argv[3])
