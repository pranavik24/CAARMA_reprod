#!/bin/bash
#SBATCH --job-name=caarma-experiment
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=62000M
#SBATCH --output=caarma-experiment-%j.out

: "${PROJECT:?PROJECT is not set; select the correct Bridges-2 allocation before submitting}"
REPO_ROOT="${CAARMA_REPO_ROOT:-${PROJECT}/CAARMA_reprod}"

source "${REPO_ROOT}/bridges2/run_experiment_body.sh"
