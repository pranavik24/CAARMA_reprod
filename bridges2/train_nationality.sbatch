#!/bin/bash
#SBATCH --job-name=caarma-nationality
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=62000M
#SBATCH --output=caarma-nationality-%j.out

: "${PROJECT:?PROJECT is not set; select the correct Bridges-2 allocation before submitting}"
REPO_ROOT="${CAARMA_REPO_ROOT:-${PROJECT}/CAARMA_reprod}"
export CAARMA_CONFIG="${CAARMA_CONFIG:-${REPO_ROOT}/configs/nationality_mixup_bridges2.yaml}"
export CAARMA_ENTRYPOINT="${CAARMA_ENTRYPOINT:-train.py}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_experiment_body.sh"
