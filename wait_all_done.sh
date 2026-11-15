#!/usr/bin/env bash
# Wait for run_all.sh to finish (ALL_DONE marker).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="$ROOT/logs/nohup_outer.log"
for i in $(seq 1 180); do
  if grep -q ALL_DONE "$LOG" 2>/dev/null; then
    echo GATE_DONE
    tail -60 "$LOG"
    ls -la "$ROOT/results/"
    exit 0
  fi
  if ! pgrep -f 'run_all\.sh' >/dev/null; then
    echo GATE_DIED
    tail -80 "$LOG"
    exit 1
  fi
  echo "POLL $i"
  tail -5 "$LOG"
  sleep 60
done
echo GATE_TIMEOUT
tail -40 "$LOG"
exit 2
