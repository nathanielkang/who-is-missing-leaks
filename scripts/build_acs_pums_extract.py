"""Build a California 2021 adult extract with allocation flags from Census PUMS.

Place psam_p06.csv at data/psam_p06.csv, or pass its path as the first argument.
Output: data/acs_ca2021_adults.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_CANDIDATES = [ROOT / "data" / "psam_p06.csv"]
if len(sys.argv) > 1:
    SRC_CANDIDATES.insert(0, Path(sys.argv[1]))
OUT = ROOT / "data" / "acs_ca2021_adults.csv"
N_KEEP = 40000
USECOLS = [
    "SERIALNO",
    "SPORDER",
    "AGEP",
    "SEX",
    "CIT",
    "NATIVITY",
    "RAC1P",
    "SCHL",
    "FSCHLP",
    "WAGP",
    "FWAGP",
    "PINCP",
    "FPINCP",
    "HINS1",
    "FHINS1P",
    "ESR",
    "FESRP",
]


def main() -> int:
    src = next((p for p in SRC_CANDIDATES if p.is_file()), None)
    if src is None:
        raise SystemExit("missing psam_p06.csv — pass its path or place it at data/psam_p06.csv")
    print("[STATUS] read", src, flush=True)
    raw = pd.read_csv(src, usecols=USECOLS, low_memory=False)
    d = raw.loc[raw["AGEP"] >= 18].copy()
    d = d[d["CIT"].isin([1, 2, 3, 4, 5])]
    # Subsample for regenerable runtime; stratified by non-citizen.
    d["H"] = (d["CIT"] == 5).astype(int)
    rng = np.random.default_rng(20261011)
    if len(d) > N_KEEP:
        # keep all non-citizens up to half, fill with citizens
        h1 = d[d["H"] == 1]
        h0 = d[d["H"] == 0]
        n1 = min(len(h1), N_KEEP // 4)
        n0 = N_KEEP - n1
        take1 = h1.sample(n=n1, random_state=int(rng.integers(1e9)))
        take0 = h0.sample(n=min(n0, len(h0)), random_state=int(rng.integers(1e9)))
        d = pd.concat([take0, take1], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(
        "[DONE] wrote",
        OUT,
        "n=",
        len(d),
        "noncitizen=",
        float(d["H"].mean()),
        "FWAGP_alloc=",
        float((d["FWAGP"] == 1).mean()),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
