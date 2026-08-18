#!/usr/bin/env bash
# Gaussian-count budget sweep for the trained runs in runs/.
#
# For each scene x render method x budget, trains with the MCMC strategy capped
# at the budget (strategy.add.max_n_gaussians). All other parameters are kept
# identical to the original runs (apps/<dataset>_<method> configs + the same
# CLI overrides found in the original parsed.yaml files):
#   num_workers=8, val_frequency=999999 (from the apps configs), export_ply on.
# Differences vs the originals, by request:
#   - strategy=mcmc + budget cap (GS strategy has no count cap)
#   - test_last=true + compute_extra_metrics=true (per-budget PSNR/SSIM/LPIPS)
#   - for budgets below the original init count: init from a subsampled point
#     cloud (data/init_clouds/<scene>_<N>.ply) / fewer random points for lego,
#     because MCMC only *adds* gaussians up to the cap.
#
# Meant to run INSIDE the 3dgrut:cuda12 container (see run_sweep_in_docker.sh).
# Safe to re-run: runs that already produced a ckpt_last.pt are skipped.

set -uo pipefail
cd "$(dirname "$0")/../.."  # repo root

BUDGETS=(500000 250000 125000 62500 31250 15625 7812 3906 1953 976)
METHODS=(3dgrt 3dgut)

# scene|dataset_path|config_prefix|downsample_factor|original_init_count
SCENES=(
  "bicycle|data/mipnerf360/bicycle|colmap|2|54275"
  "bonsai|data/mipnerf360/bonsai|colmap|2|206613"
  "garden|data/mipnerf360/garden|colmap|2|138766"
  "drjohnson|data/db/drjohnson|colmap|1|80861"
  "playroom|data/db/playroom|colmap|1|37005"
  "train|data/tandt/train|colmap|1|182686"
  "truck|data/tandt/truck|colmap|1|136029"
  "lego|data/nerf_synthetic/lego|nerf_synthetic|0|100000"
)

TOTAL=$(( ${#BUDGETS[@]} * ${#SCENES[@]} * ${#METHODS[@]} ))
DONE=0

for N in "${BUDGETS[@]}"; do
  for entry in "${SCENES[@]}"; do
    IFS='|' read -r scene dpath cfgprefix df ninit <<< "$entry"
    for method in "${METHODS[@]}"; do
      DONE=$((DONE + 1))
      run_name="${scene}_${method}_mcmc${N}"
      out_dir="runs/${scene}/${run_name}"
      if ls "${out_dir}"/*/ckpt_last.pt >/dev/null 2>&1; then
        echo "[skip] (${DONE}/${TOTAL}) ${run_name} (already done)"
        continue
      fi

      args=(
        --config-name "apps/${cfgprefix}_${method}"
        "path=${dpath}"
        "experiment_name="
        "out_dir=${out_dir}"
        "num_workers=8"
        "test_last=true"
        "compute_extra_metrics=true"
        "export_ply.enabled=true"
        "export_ply.path=${out_dir}/export_last.ply"
        "strategy=mcmc"
        "strategy.add.max_n_gaussians=${N}"
      )
      if [ "$cfgprefix" = "colmap" ]; then
        args+=("dataset.downsample_factor=${df}")
        if [ "$N" -lt "$ninit" ]; then
          args+=(
            "initialization=fused_point_cloud"
            "initialization.fused_point_cloud_path=data/init_clouds/${scene}_${N}.ply"
            "initialization.use_observation_points=true"
          )
        fi
      elif [ "$N" -lt "$ninit" ]; then
        args+=("initialization.num_gaussians=${N}")
      fi

      log="${out_dir}.log"
      echo "[run ] (${DONE}/${TOTAL}) ${run_name}  start=$(date '+%F %T')"
      mkdir -p "${out_dir}"
      python train.py "${args[@]}" > "${log}" 2>&1
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "[FAIL] ${run_name} rc=${rc} (log: ${log})" | tee -a runs/budget_sweep_failures.txt
      else
        echo "[ ok ] ${run_name}  end=$(date '+%F %T')"
      fi
    done
  done
done
echo "SWEEP DONE $(date '+%F %T')"
