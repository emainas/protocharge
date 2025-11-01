#!/bin/bash
#SBATCH --job-name=mcresp_APP           # Update APP with your microstate label
#SBATCH --output=logs/mcresp.%j.out
#SBATCH --error=logs/mcresp.%j.err
#SBATCH --time=12:00:00                 # Adjust wall-clock limit
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8               # NumPy/SciPy will use these threads
#SBATCH --mem=160G                      # Plenty of headroom for large Coulomb matrices
#SBATCH --mail-type=FAIL                # Email only on failure
#SBATCH --mail-user=YOUR_EMAIL@DOMAIN   # <-- set to your institutional address

set -euo pipefail

# Update this path to the project checkout on the cluster filesystem.
PROJECT_DIR="$HOME/projects/biliresp"
MICROSTATE="APP"  # Set to the microstate you want to fit

module purge
# Load whatever Python/Conda module your cluster provides.
module load python/3.10 || true

# Activate the Conda environment that contains biliresp.
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate biliresp
fi

cd "$PROJECT_DIR"
mkdir -p logs

python -m multiconfresp.mcresp \
    --microstate "$MICROSTATE" \
    --load-and-resp \
    --maxiter 400
