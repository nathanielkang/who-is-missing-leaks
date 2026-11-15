#!/usr/bin/env bash
# Survey protection suite: accounting, theory checks, and the synthetic grid.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE="$ROOT/accounting"
export WM1_OUT="$ROOT/results"
export WM1_NHANES="$ROOT/data/n1516.csv"
mkdir -p "$WM1_OUT" "$ROOT/logs"
cd "$CODE"
LOG="$ROOT/logs/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "WM1_OUT=$WM1_OUT"
echo "WM1_NHANES=$WM1_NHANES"
echo "START $(date -Is)"

run() {
  local name="$1"; shift
  echo ""
  echo "===== $name ====="
  local t0=$SECONDS
  python3 -u "$@"
  echo "===== DONE $name ($((SECONDS - t0))s) ====="
}

run protection_theory_checks protection_theory_checks.py
run theory_checks theory_checks.py
run synthetic_table synthetic_table.py
run nhanes_accounting nhanes_accounting.py
run nhanes_flag_analysis nhanes_flag_analysis.py
REPS="${WM1_REPS:-100}"
run nhanes_protection nhanes_protection.py "$WM1_NHANES" "$REPS"
run make_table2 make_table2.py
echo "ALL_DONE $(date -Is)"
ls -la "$WM1_OUT"
