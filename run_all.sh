#!/usr/bin/env bash
# Runs every experiment headlessly and regenerates all outputs under experiments/*/out/.
set -uo pipefail
cd "$(dirname "$0")"
for d in experiments/[0-9][0-9]_*/; do
  if [ -f "$d/run.py" ]; then
    py=.venv/bin/python
    [ -f "$d/.uses_meep" ] && py=.meep/bin/python
    echo "=== $d ($py)"
    ( cd "$d" && "../../$py" run.py ) || echo "!!! $d failed"
  fi
done
