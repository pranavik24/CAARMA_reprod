#!/bin/bash
#SBATCH --job-name=caarma-gender
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=62000M
#SBATCH --output=caarma-diffusion-%j.out

set -euo pipefail

: "${PROJECT:?PROJECT is not set; select the correct Bridges-2 allocation before submitting}"
: "${AI_MODULE:?Set AI_MODULE to the current PSC module reported by 'module spider AI'}"

REPO_ROOT="${CAARMA_REPO_ROOT:-${PROJECT}/CAARMA_reprod}"
CONFIG_PATH="${CAARMA_CONFIG:-${REPO_ROOT}/configs/gender_bridges2.yaml}"
ENV_ROOT="${HOME}/.venvs/caarma"
SAVE_DIR="${CAARMA_SAVE_DIR:-${PROJECT}/caarma-output/gender}"
TEST_SPLIT="${CAARMA_TEST_SPLIT:-test}"
SCORE_PREFIX="${CAARMA_SCORE_PREFIX:-caarma_gender_${TEST_SPLIT}}"

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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"
mkdir -p "${SAVE_DIR}"

"${PYTHON_BIN}" -c 'import os, sys; print("Python:", sys.executable); print("AI ENV:", os.environ.get("ENV")); print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", "")); print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", ""))'
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not visible"; print(torch.cuda.get_device_name(0))'
JOB_START_TIME="$(date +%s)"
srun "${PYTHON_BIN}" -u mixup_gender.py \
    --config "${CONFIG_PATH}" \
    --mode train \
    --sl-mixup

export CAARMA_SAVE_DIR="${SAVE_DIR}"
export CAARMA_JOB_START_TIME="${JOB_START_TIME}"
BEST_CKPT="$("${PYTHON_BIN}" - <<'PY'
import os
import re
from pathlib import Path

save_dir = Path(os.environ["CAARMA_SAVE_DIR"])
job_start_time = float(os.environ["CAARMA_JOB_START_TIME"])
all_checkpoints = [
    path for path in save_dir.glob("*.ckpt")
    if path.stat().st_mtime >= job_start_time - 5
]
if not all_checkpoints:
    all_checkpoints = list(save_dir.glob("*.ckpt"))

checkpoints = [path for path in all_checkpoints if "cosine_eer=" in path.name]
if not checkpoints:
    checkpoints = all_checkpoints
if not checkpoints:
    raise SystemExit(f"No checkpoints found in {save_dir}")

def checkpoint_key(path):
    match = re.search(r"cosine_eer=([0-9]+(?:\.[0-9]+)?)", path.name)
    return (float(match.group(1)) if match else float("inf"), -path.stat().st_mtime)

print(min(checkpoints, key=checkpoint_key))
PY
)"

echo "Testing best checkpoint: ${BEST_CKPT}"
srun "${PYTHON_BIN}" -u mixup_gender.py \
    --config "${CONFIG_PATH}" \
    --mode test \
    --checkpoint-path "${BEST_CKPT}" \
    --validation-split "${TEST_SPLIT}" \
    --score-output-prefix "${SCORE_PREFIX}" \
    --sl-mixup
