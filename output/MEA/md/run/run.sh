#!/usr/bin/env bash
set -euo pipefail

module purge
module load amber/25-CUDA_12.6-GCC12.2.0



echo "==> Running MD stages with pmemd.cuda"

if [[ ! -f min.out ]]; then
  echo "  min..."
  pmemd.cuda -O \
    -i min.in \
    -p MEA.parm7 \
    -c MEA.rst7 \
    -r min.rst7 \
    -o min.out \
    -inf min.info
else
  echo "  Skipping min (min.out exists)"
fi

if [[ ! -f heat.out ]]; then
  echo "  heat..."
  pmemd.cuda -O \
    -i heat.in \
    -p MEA.parm7 \
    -c min.rst7 \
    -r heat.rst7 \
    -o heat.out \
    -inf heat.info \
    -x heat.nc
else
  echo "  Skipping heat (heat.out exists)"
fi

if [[ ! -f equil-nvt.out ]]; then
  echo "  equil-nvt..."
  pmemd.cuda -O \
    -i equil-nvt.in \
    -p MEA.parm7 \
    -c heat.rst7 \
    -r equil-nvt.rst7 \
    -o equil-nvt.out \
    -inf equil-nvt.info \
    -x equil-nvt.nc
else
  echo "  Skipping equil-nvt (equil-nvt.out exists)"
fi

if [[ ! -f equil-npt.out ]]; then
  echo "  equil-npt..."
  pmemd.cuda -O \
    -i equil-npt.in \
    -p MEA.parm7 \
    -c equil-nvt.rst7 \
    -r equil-npt.rst7 \
    -o equil-npt.out \
    -inf equil-npt.info \
    -x equil-npt.nc
else
  echo "  Skipping equil-npt (equil-npt.out exists)"
fi

