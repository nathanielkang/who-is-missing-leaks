"""Understanding Society (UKHLS) consent + multi-wave attrition leakage.

Expects a UKDS End User Licence extract (SN 6614) dropped at:
  2_Code/data/ukhls/   (folder of Stata/CSV wave files)
or a single flat CSV at 2_Code/data/ukhls_extract.csv
Env override: WM1_UKHLS

Variables (EUL names; script tolerates case / wave prefixes a_, b_, ...):
  - consent to health linkage (or any lac* / cons* health flag present)
  - sex, age, ethnic group (or racel_dv)
  - wave participation indicators across available waves

Writes results/ukhls_consent_attrition.json and .csv
Exits 2 with a clear message if no extract is present.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _path_bootstrap import ROOT, out_path, ukhls_dir_or_csv

sys.path.insert(0, str(ROOT))
from core_lib import two_sided_level


def _find_col(cols, *needles):
    low = {c.lower(): c for c in cols}
    for n in needles:
        if n.lower() in low:
            return low[n.lower()]
    for c in cols:
        cl = c.lower()
        if any(n.lower() in cl for n in needles):
            return c
    return None


def _load_table(path: Path) -> pd.DataFrame:
    if path.is_file():
        if path.suffix.lower() in {".csv", ".tsv"}:
            return pd.read_csv(path, low_memory=False)
        if path.suffix.lower() in {".dta"}:
            return pd.read_stata(path, convert_categoricals=False)
        raise SystemExit(f"unsupported file type: {path}")
    # directory: prefer a flat extract, else first indresp
    for name in ("ukhls_extract.csv", "indresp.csv", "xwaveid.csv"):
        p = path / name
        if p.is_file():
            return _load_table(p)
    dta = sorted(path.rglob("*indresp*.dta"))
    if dta:
        # merge wave participation from multiple indresp if present
        frames = []
        for f in dta[:16]:
            w = f.stem[:1] if f.stem[0].isalpha() else "w"
            df = pd.read_stata(f, convert_categoricals=False)
            pid = _find_col(df.columns, "pidp", "pid")
            if pid is None:
                continue
            df = df.rename(columns={pid: "pidp"})
            df["wave_file"] = f.name
            frames.append(df)
        if not frames:
            raise SystemExit(f"no usable indresp under {path}")
        # keep widest frame; mark wave presence
        base = frames[0]
        for fr in frames[1:]:
            base = base.merge(fr[["pidp"]].drop_duplicates().assign(**{fr["wave_file"].iloc[0]: 1}), on="pidp", how="outer")
        return base
    raise SystemExit(
        f"UKHLS folder {path} has no indresp/CSV. Put SN 6614 files under 2_Code/data/ukhls/"
    )


def main() -> int:
    try:
        src = ukhls_dir_or_csv()
    except FileNotFoundError as e:
        print("UKHLS_BLOCKED:", e)
        stub = {
            "status": "blocked_no_extract",
            "need": "UKDS EUL SN 6614 under 2_Code/data/ukhls/ or WM1_UKHLS",
            "note": "Cannot download without UKDS registration/licence acceptance.",
        }
        _out = out_path("ukhls_consent_attrition.json")
        json.dump(stub, open(_out, "w"), indent=1)
        print("wrote stub", _out)
        return 2

    d = _load_table(src)
    cols = list(d.columns)
    pid = _find_col(cols, "pidp", "pid")
    sex = _find_col(cols, "sex_dv", "sex")
    age = _find_col(cols, "age_dv", "dvage", "age")
    ethn = _find_col(cols, "racel_dv", "ethn_dv", "race")
    cons = _find_col(
        cols,
        "conshealth",
        "healthconsent",
        "lac_consent",
        "consent_health",
        "cdrug_consent",
        "nhs_consent",
    )
    # fallback: any column with both consent and health
    if cons is None:
        for c in cols:
            cl = c.lower()
            if "cons" in cl and ("health" in cl or "nhs" in cl or "link" in cl):
                cons = c
                break

    ledger = {
        "status": "ok",
        "source": str(src),
        "n_rows": int(len(d)),
        "columns_used": {"pid": pid, "sex": sex, "age": age, "ethn": ethn, "consent": cons},
    }
    if cons is None:
        ledger["status"] = "missing_consent_column"
        ledger["hint"] = "Need a health/admin linkage consent flag in the EUL extract."
        json.dump(ledger, open(out_path("ukhls_consent_attrition.json"), "w"), indent=1)
        print(json.dumps(ledger, indent=2))
        return 3

    # binary consent: positive codes often 1; treat >0 as consent if mostly 0/1
    cser = pd.to_numeric(d[cons], errors="coerce")
    R = (cser == 1).astype(int)
    # driver: ethnic minority proxy if available else sex
    if ethn is not None:
        es = pd.to_numeric(d[ethn], errors="coerce")
        # UKHLS racel_dv: 1 often White British — treat !=1 as H=1 when coded that way
        H = (es.fillna(1) != 1).astype(int)
        driver = "ethn_not_1"
    elif sex is not None:
        H = (pd.to_numeric(d[sex], errors="coerce") == 2).astype(int)
        driver = "sex_female"
    else:
        ledger["status"] = "missing_driver"
        json.dump(ledger, open(out_path("ukhls_consent_attrition.json"), "w"), indent=1)
        return 3

    ok = R.notna() & H.notna() if hasattr(R, "notna") else slice(None)
    # R is already 0/1; rebuild from aligned frame
    df = pd.DataFrame({"R": R, "H": H})
    if age is not None:
        df["age"] = pd.to_numeric(d[age], errors="coerce")
        df = df[df["age"].isna() | (df["age"] >= 16)]
    df = df.dropna()
    g1 = float(df.loc[df.H == 1, "R"].mean()) if (df.H == 1).any() else float("nan")
    g0 = float(df.loc[df.H == 0, "R"].mean()) if (df.H == 0).any() else float("nan")
    e1 = abs(math.log(max(g1, 1e-12) / max(g0, 1e-12)))
    e0 = abs(math.log(max(1 - g1, 1e-12) / max(1 - g0, 1e-12)))
    # two-point laws on consent
    q0 = np.array([1 - g0, g0])
    q1 = np.array([1 - g1, g1])
    level = float(two_sided_level(q0, q1))

    # attrition: columns that look like wave indicators
    wave_cols = [c for c in df.columns if False]  # placeholder
    wave_cols = [c for c in d.columns if "wave" in c.lower() or c.lower().endswith("indresp")]
    attrition = {}
    if len(wave_cols) >= 2:
        present = d[wave_cols].notna().astype(int)
        n_waves = present.sum(axis=1)
        leavers = (n_waves < present.shape[1]).astype(int)
        # disclosed level of leaver vs completer by driver H on original d
        tmp = pd.DataFrame({"H": H, "leaver": leavers}).dropna()
        for label, mask in ("leavers", tmp.leaver == 1), ("completers", tmp.leaver == 0):
            sub = tmp[mask]
            if sub.H.nunique() < 2:
                continue
            p1 = float((sub.H == 1).mean())
            # use group composition difference as a coarse signal
            attrition[label] = {"n": int(len(sub)), "driver_share": p1}

    ledger.update(
        {
            "n": int(len(df)),
            "driver": driver,
            "consent_rate": float(df.R.mean()),
            "driver_share": float(df.H.mean()),
            "g1": g1,
            "g0": g0,
            "eps_resp": e1,
            "eps_nonresp": e0,
            "consent_level": level,
            "attrition": attrition,
        }
    )
    _out = out_path("ukhls_consent_attrition.json")
    json.dump(ledger, open(_out, "w"), indent=1)
    _csv = out_path("ukhls_consent_attrition.csv")
    pd.DataFrame(
        [
            {"metric": "consent_level", "value": level},
            {"metric": "g1", "value": g1},
            {"metric": "g0", "value": g0},
            {"metric": "eps_nonresp", "value": e0},
        ]
    ).to_csv(_csv, index=False)
    print(json.dumps({k: ledger[k] for k in ("n", "driver", "g1", "g0", "consent_level")}, indent=2))
    print("wrote", _out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
