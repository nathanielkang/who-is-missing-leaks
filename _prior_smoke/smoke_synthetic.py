"""Synthetic closed-form smoke for the product and block policies.

No NHANES download. Validates F(E,L,C) retention and block vs product task rates
used in the note's synthetic stress section.
"""
from __future__ import annotations

import math
import sys


def F(E: float, L: float, C: float) -> float:
    if C <= 1.0 + 1e-15:
        return L
    if E < math.log(C) - 1e-12:
        return 0.0
    return min(L, (math.exp(E) - 1.0) / (C * math.exp(E) - 1.0))


def product_task(s: int, E: float, L: float, c: float) -> float:
    return F(E / s, L, c) ** s


def block_task(E: float, L: float, C: float) -> float:
    return F(E, L, C)


def main() -> int:
    c = 1.1
    # Panel B: (ell0,ell1)=(0.8,0.7) per field → L_S = 0.7^s, C_S = c^s (Thm joint)
    checks = [
        ("block s=3 E=0.3", block_task(0.3, 0.7**3, c**3), 0.343, 0.005),
        ("product s=3 E=0.3", product_task(3, 0.3, 0.7, c), 0.116, 0.01),
        ("block s=10 E=1", block_task(1.0, 0.7**10, c**10), 0.028, 0.005),
        ("product s=10 E=1", product_task(10, 1.0, 0.7, c), 0.00076, 0.0002),
        ("activation log c", math.log(c), 0.0953, 0.001),
    ]
    failed = []
    for name, got, want, tol in checks:
        if abs(got - want) > tol:
            failed.append(f"{name}: got {got:.6g}, want {want} ± {tol}")
    if failed:
        print("SMOKE FAIL")
        for line in failed:
            print(" ", line)
        return 1
    print("SMOKE PASS - synthetic closed forms match Research Note regimes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
