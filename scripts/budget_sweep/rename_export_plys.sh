#!/usr/bin/env bash
# Rename sweep export PLYs to the original runs/ convention:
#   runs/<scene>/<run_name>/export_last.ply  ->  runs/<scene>/<run_name>/<run_name>.ply
# Only touches runs whose export already exists, so it is safe to run while the
# sweep is still going (the in-flight run has no export_last.ply yet).
set -uo pipefail
cd "$(dirname "$0")/../.."

n=0
for ply in runs/*/*_mcmc*/export_last.ply; do
  [ -e "$ply" ] || continue
  dir=$(dirname "$ply")
  mv -- "$ply" "$dir/$(basename "$dir").ply"
  n=$((n + 1))
done
echo "renamed $n ply files"
