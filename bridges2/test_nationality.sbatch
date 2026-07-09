#!/bin/bash
#SBATCH --job-name=caarma-nationality-test
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=62000M
#SBATCH --output=caarma-nationality-test-%j.out

set -euo pipefail

: "${PROJECT:?PROJECT is not set; select the correct Bridges-2 allocation before submitting}"
: "${AI_MODULE:?Set AI_MODULE to the current PSC module reported by 'module spider AI'}"
: "${CAARMA_CHECKPOINT:?Set CAARMA_CHECKPOINT to the checkpoint .ckpt to evaluate}"

REPO_ROOT="${CAARMA_REPO_ROOT:-${PROJECT}/CAARMA_reprod}"
CONFIG_PATH="${CAARMA_CONFIG:-${REPO_ROOT}/config_nationality_bridges2.yaml}"
ENV_ROOT="${HOME}/.venvs/caarma"
TEST_SPLIT="${CAARMA_TEST_SPLIT:-test}"
SCORE_PREFIX="${CAARMA_SCORE_PREFIX:-caarma_nationality_${TEST_SPLIT}}"

for conda_hook in \
    /opt/packages/anaconda3/etc/profile.d/conda.sh \
    /opt/packages/AI/anaconda3/etc/profile.d/conda.sh \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -r "${conda_hook}" ]]; then
        source "${conda_hook}"
        break
    fi
done

module purge
module load "${AI_MODULE}"

if [[ -n "${ENV:-}" && -r "${ENV}/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="${ENV}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

PYTHON_BIN="${ENV_ROOT}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing environment: ${ENV_ROOT}" >&2
    exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"
mkdir -p "${PROJECT}/caarma-output/nationality"

"${PYTHON_BIN}" -c 'import os, sys; print("Python:", sys.executable); print("AI ENV:", os.environ.get("ENV")); print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", ""))'
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not visible"; print(torch.cuda.get_device_name(0))'
srun "${PYTHON_BIN}" -u mixup_nationality.py \
    --config "${CONFIG_PATH}" \
    --mode test \
    --checkpoint-path "${CAARMA_CHECKPOINT}" \
    --validation-split "${TEST_SPLIT}" \
    --score-output-prefix "${SCORE_PREFIX}" \
    --nationality-mixup
