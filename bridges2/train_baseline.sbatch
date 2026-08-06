#!/bin/bash
#SBATCH --job-name=caarma-baseline-mfa
#SBATCH --output=caarma-baseline-mfa-%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=62000M
#SBATCH --time=48:00:00

set -euo pipefail

: "${PROJECT:?PROJECT is not set}"
: "${AI_MODULE:?Set AI_MODULE to the Bridges-2 AI module, e.g. AI/pytorch_23.02-1.13.1-py3}"

REPO_ROOT="${CAARMA_REPO_ROOT:-${PROJECT}/CAARMA_reprod}"
CONFIG_PATH="${CAARMA_CONFIG:-${REPO_ROOT}/configs/baseline_mfa_conformer_bridges2.yaml}"
TEST_TRIAL_PATH="${CAARMA_TEST_TRIAL_PATH:-${REPO_ROOT}/data/veri_test.txt}"
TEST_SPLIT="${CAARMA_TEST_SPLIT:-test}"
ENV_ROOT="${HOME}/.venvs/caarma"
PYTHON_BIN="${ENV_ROOT}/bin/python"

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

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing environment: ${ENV_ROOT}" >&2
    exit 2
fi
if [[ ! -s "${TEST_TRIAL_PATH}" ]]; then
    echo "Missing final test trial protocol: ${TEST_TRIAL_PATH}" >&2
    exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-5}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"

mapfile -t CONFIG_VALUES < <("${PYTHON_BIN}" - "${CONFIG_PATH}" "${TEST_SPLIT}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1]).expanduser().resolve()
test_split = sys.argv[2]
with config_path.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

def resolve(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return str(path)
    return str((config_path.parent / path).resolve())

print(resolve(config["save_dir"]))
print(f"{config.get('score_output_prefix', config.get('title', 'caarma_baseline'))}_{test_split}")
PY
)
SAVE_DIR="${CAARMA_SAVE_DIR:-${CONFIG_VALUES[0]}}"
SCORE_PREFIX="${CAARMA_SCORE_PREFIX:-${CONFIG_VALUES[1]}}"
mkdir -p "${SAVE_DIR}"

"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not visible"; print(torch.cuda.get_device_name(0))'

JOB_START_TIME="$(date +%s)"
srun "${PYTHON_BIN}" -u train_baseline.py \
    --config "${CONFIG_PATH}" \
    --mode train

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
srun "${PYTHON_BIN}" -u train_baseline.py \
    --config "${CONFIG_PATH}" \
    --mode test \
    --checkpoint-path "${BEST_CKPT}" \
    --trial-path "${TEST_TRIAL_PATH}" \
    --score-output-prefix "${SCORE_PREFIX}"
