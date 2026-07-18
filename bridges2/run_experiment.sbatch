#!/bin/bash
#SBATCH --job-name=caarma-experiment
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=62000M
#SBATCH --output=caarma-experiment-%j.out

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_experiment_body.sh"
