#!/bin/bash
#SBATCH --job-name=boot_mmresp
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --array=0-44

set -euo pipefail

PROJECT_ROOT="/work/users/e/m/emainas/projects/biliresp/biliresp"
cd "$PROJECT_ROOT"

# Activate env
module load anaconda || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh || true
conda activate biliresp

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

REP_ID=$(printf "%d" "$SLURM_ARRAY_TASK_ID")
BOOT_DIR="$PROJECT_ROOT/scripts/bootstrap_hist/rep_$(printf '%03d' "$REP_ID")"

# Build bootstrap matrices and manifest
python scripts/bootstrap_hist.py \
  --replicate "$REP_ID" \
  --base-dir "$PROJECT_ROOT/data/microstates" \
  --output-dir "$PROJECT_ROOT/scripts/bootstrap_hist" \
  --manifest-template "$PROJECT_ROOT/configs/manifest_hist.yaml"

# Run mmresp for this replicate with a small ridge; retry with larger ridge if needed
set +e
python -m biliresp.multimoleculeresp.mmresp \
  --manifest "$BOOT_DIR/manifest.yaml" \
  --output "$BOOT_DIR/mmresp/charges.npz" \
  --ridge 1e-6
rc=$?
if [ $rc -ne 0 ]; then
  echo "Retrying with ridge=1e-4 (prev rc=$rc)"
  python -m biliresp.multimoleculeresp.mmresp \
    --manifest "$BOOT_DIR/manifest.yaml" \
    --output "$BOOT_DIR/mmresp/charges.npz" \
    --ridge 1e-4
  rc=$?
fi
set -e
exit $rc
