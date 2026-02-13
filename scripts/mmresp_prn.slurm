#!/bin/bash
#SBATCH --job-name=mmresp_PRN
#SBATCH --partition=hov
#SBATCH --output=slurm_logs/mmresp.%j.out
#SBATCH --error=slurm_logs/mmresp.%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=emainas@unc.edu

set -euo pipefail

PROJECT_DIR=/work/users/e/m/emainas/projects/biliresp/biliresp
MANIFEST=$PROJECT_DIR/configs/manifest_prn.yaml  # adjust if elsewhere

# Thread hygiene
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

module purge
module load anaconda || true
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate biliresp
fi

cd "$PROJECT_DIR"
mkdir -p slurm_logs

srun python -m biliresp.multimoleculeresp.mmresp --manifest "$MANIFEST" \
	--output /work/users/e/m/emainas/projects/biliresp/biliresp/scripts/mmresp_prn/charges.npz
