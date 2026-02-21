#!/usr/bin/env bash
set -euo pipefail
J1="$1"
J2="$2"
while true; do
  s1=$(qstat "$J1" 2>/dev/null | awk 'NR==3{print $5}') || s1=""
  s2=$(qstat "$J2" 2>/dev/null | awk 'NR==3{print $5}') || s2=""

  if [[ "$s1" == "R" && "$s2" == "Q" ]]; then
    qdel "$J2" || true
    exit 0
  fi
  if [[ "$s2" == "R" && "$s1" == "Q" ]]; then
    qdel "$J1" || true
    exit 0
  fi
  if [[ "$s1" == "R" && "$s2" == "R" ]]; then
    # fallback: keep earlier submitted job id (smaller numeric prefix)
    n1=${J1%%.*}
    n2=${J2%%.*}
    if [[ "$n1" -le "$n2" ]]; then
      qdel "$J2" || true
    else
      qdel "$J1" || true
    fi
    exit 0
  fi
  if [[ -z "$s1" || -z "$s2" ]]; then
    # one job disappeared (finished/cancelled); stop watcher
    exit 0
  fi
  sleep 30
done
