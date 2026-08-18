#!/usr/bin/env bash
# Copy finished sweep PLYs into the fig_templates assets folder (flat, same as
# the existing *_30k.ply files). Runs rename_export_plys.sh first so any new
# export_last.ply gets its <run_name>.ply name before copying.
# Safe to re-run anytime; only missing/updated files are copied.
set -euo pipefail
cd "$(dirname "$0")/../.."

ASSETS_DIR=${ASSETS_DIR:-"/d/fig_templates/assets"}

bash scripts/budget_sweep/rename_export_plys.sh
cp -u runs/*/*_mcmc*/*.ply "$ASSETS_DIR/"
echo "assets now has $(ls "$ASSETS_DIR" | grep -c mcmc) mcmc ply files"
