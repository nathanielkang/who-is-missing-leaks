"""Insert this package dir on sys.path; resolve OUTPUT_DIR / NHANES from env or cwd."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = Path(os.environ.get("WM1_OUT", ROOT / ".." / "results")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_CANDIDATES = [
    Path(os.environ["WM1_NHANES"]) if os.environ.get("WM1_NHANES") else None,
    ROOT / ".." / "data" / "n1516.csv",
    ROOT / "n1516.csv",
    Path("/home/claude/nhanes/n1516.csv"),
]


def nhanes_csv(argv1: str | None = None) -> Path:
    if argv1:
        p = Path(argv1)
        if p.is_file():
            return p.resolve()
    for c in DATA_CANDIDATES:
        if c is not None and c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        "NHANES CSV not found. Set WM1_NHANES or place data/n1516.csv"
    )


def out_path(name: str) -> Path:
    return OUTPUT_DIR / name


ACS_CANDIDATES = [
    Path(os.environ["WM1_ACS"]) if os.environ.get("WM1_ACS") else None,
    ROOT / ".." / "data" / "acs_ca2021_adults.csv",
    ROOT / "acs_ca2021_adults.csv",
]


def acs_csv(argv1: str | None = None) -> Path:
    if argv1:
        p = Path(argv1)
        if p.is_file():
            return p.resolve()
    for c in ACS_CANDIDATES:
        if c is not None and c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        "ACS CSV not found. Set WM1_ACS or run scripts/build_acs_pums_extract.py"
    )


UKHLS_CANDIDATES = [
    Path(os.environ["WM1_UKHLS"]) if os.environ.get("WM1_UKHLS") else None,
    ROOT / ".." / "data" / "ukhls",
    ROOT / ".." / "data" / "ukhls_extract.csv",
]


def ukhls_dir_or_csv() -> Path:
    for c in UKHLS_CANDIDATES:
        if c is not None and (c.is_file() or c.is_dir()):
            return c.resolve()
    raise FileNotFoundError(
        "UKHLS extract not found. Set WM1_UKHLS to a UKDS EUL folder or CSV "
        "(SN 6614). Drop files under 2_Code/data/ukhls/"
    )
