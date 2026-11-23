#!/bin/bash
# Usage: scripts/queue.sh <queue_file> <python>
# Runs each non-empty, non-comment line of <queue_file> sequentially (re-read every time, so lines can
# be appended while running). A line is skipped if its hash is in <queue_file>.done.
Q="$1"; PY="$2"; cd "$(dirname "$0")/.."
DONE="$Q.done"; touch "$DONE"
while true; do
  next=""
  while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    h=$(echo -n "$line" | md5sum | cut -c1-12)
    grep -q "^$h" "$DONE" && continue
    next="$line"; break
  done < "$Q"
  [[ -z "$next" ]] && { echo "[$(date '+%m-%d %H:%M')] queue empty"; break; }
  echo "[$(date '+%m-%d %H:%M')] START $next"
  eval "$PY $next"; rc=$?
  echo "$h rc=$rc $(date '+%m-%d %H:%M') $next" >> "$DONE"
  echo "[$(date '+%m-%d %H:%M')] END rc=$rc"
done
