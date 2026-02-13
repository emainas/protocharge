#!/bin/bash
#SBATCH --job-name=boot_mmresp_rand
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --array=1-200

set -euo pipefail

PROJECT_ROOT="/work/users/e/m/emainas/projects/biliresp/biliresp"
cd "$PROJECT_ROOT"

module load anaconda || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" || true
conda activate biliresp

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

RID=${SLURM_ARRAY_TASK_ID}
RID3=$(printf "%03d" "$RID")
REP_DIR="$PROJECT_ROOT/scripts/bootstrap_hist_random/rep_${RID3}"

# Build bootstrap matrices and manifest (with replacement, all 10 blocks)
python scripts/bootstrap_hist_random.py \
  --replicate "$RID" \
  --base-dir "$PROJECT_ROOT/data/microstates" \
  --output-dir "$PROJECT_ROOT/scripts/bootstrap_hist_random" \
  --manifest-template "$PROJECT_ROOT/configs/manifest_hist.yaml"

# Run mmresp with ridge/lstsq safety net
python -m biliresp.multimoleculeresp.mmresp \
  --manifest "$REP_DIR/manifest.yaml" \
  --output "$REP_DIR/mmresp/charges.npz" \
