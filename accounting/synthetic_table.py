"""Synthetic stress test (Section 3.4): product balancing vs exact faithful program vs attribute-blind exact program."""
import sys, json, math
import numpy as np
from _path_bootstrap import out_path, ROOT
sys.path.insert(0, str(ROOT))
from core_lib import robust_faithful_lp, balanced_allocation
rows = []
for reg, (L0, L1) in (("separated", (0.8, 0.4)), ("nearby", (0.8, 0.7))):
    for K in (3, 4):
        for E in (0.1, 0.2, 0.5, 1.0):
            c = np.full(K, 1.1); l0 = np.full(K, L0); l1 = np.full(K, L1)
            Up, _ = balanced_allocation(c, np.minimum(l0, l1), E)
            Ux, _ = robust_faithful_lp(K, E, l0, l1, c)
            Ub, _ = robust_faithful_lp(K, E, l0, l1, c, blind=True)
            rows.append((reg, K, E, Up, Ux, Ub))
            print("%-9s K=%d E=%.1f product %.4f exact %.4f blind-exact %.4f" % rows[-1], flush=True)
_out = out_path("synthetic_table.json")
json.dump(rows, open(_out, "w"), indent=1)
print("wrote", _out, flush=True)
