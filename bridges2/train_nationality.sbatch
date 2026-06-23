#!/bin/bash
#SBATCH --job-name=caarma-nationality
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=caarma-nationality-%j.out

set -euo pipefail

: "${PROJECT:?PROJECT is not set; select the correct Bridges-2 allocation before submitting}"
: "${AI_MODULE:?Set AI_MODULE to the current PSC module reported by 'module spider AI'}"

REPO_ROOT="${CAARMA_REPO_ROOT:-${PROJECT}/CAARMA_reprod}"
CONFIG_PATH="${CAARMA_CONFIG:-${REPO_ROOT}/config_nationality_bridges2.yaml}"
ENV_ROOT="${HOME}/.venvs/caarma"

module purge
module load "${AI_MODULE}"

PYTHON_BIN="${ENV_ROOT}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing environment: ${ENV_ROOT}" >&2
    echo "Create it on a compute node after loading the same PSC modules." >&2
    exit 2
fi
if [[ "$(stat -c '%U' "${ENV_ROOT}")" != "${USER}" ]]; then
    echo "Refusing to execute an environment not owned by ${USER}: ${ENV_ROOT}" >&2
    exit 2
fi
if [[ "$(stat -c '%U' "${HOME}")" != "${USER}" ]] || [[ -n "$(find "${HOME}" -maxdepth 0 -perm /022 -print -quit)" ]]; then
    echo "Refusing to execute from a HOME directory not private to ${USER}: ${HOME}" >&2
    exit 2
fi
if [[ -n "$(find "${ENV_ROOT}" -xdev \( -type f -o -type d \) ! -user "${USER}" -print -quit)" ]]; then
    echo "Refusing to execute an environment containing files not owned by ${USER}" >&2
    exit 2
fi
if [[ -n "$(find "${ENV_ROOT}" \( -type f -o -type d \) -perm /022 -print -quit)" ]]; then
    echo "Refusing to execute a group- or world-writable environment: ${ENV_ROOT}" >&2
    exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"
mkdir -p "${PROJECT}/caarma-output/nationality"

"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not visible"; print(torch.cuda.get_device_name(0))'
srun "${PYTHON_BIN}" -u mixup_nationality.py \
    --config "${CONFIG_PATH}" \
    --mode train \
    --nationality-mixup
