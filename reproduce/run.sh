#!/bin/bash
# Run the reproduction queues in order (default: all of reproduce/queues/*.q).
#   FDU_RUNS=/path/to/runs FDU_DATA=/path/to/data PY=python bash reproduce/run.sh [queue.q ...]
# Each queue line is run once; finished lines are recorded in <queue>.done and skipped on a restart, failed lines
# (rc != 0) are retried up to twice. Logs: <queue>.log next to the queue file.
set -u
cd "$(dirname "$0")/.."
: "${FDU_RUNS:?set FDU_RUNS (results directory)}"
: "${FDU_DATA:?set FDU_DATA (datasets, see README)}"
export FDU_RUNS FDU_DATA
PY=${PY:-python}
if [ $# -eq 0 ]; then set -- reproduce/queues/*.q; fi
for q in "$@"; do
  q=$(readlink -f "$q")
  for pass in 1 2 3; do
    if [ $pass -gt 1 ]; then
      grep -q " rc=[1-9]" "$q.done" 2>/dev/null || break
      sed -i "/ rc=[1-9]/d" "$q.done"
      echo "[retry pass $pass] $q"
    fi
    bash scripts/queue.sh "$q" "$PY" 2>&1 | tee -a "${q%.q}.log"
  done
done
