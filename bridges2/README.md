# Running nationality-conditioned CAARMA on Bridges-2

Keep the repository, environment, datasets, and checkpoints under the persistent
`$PROJECT` filesystem. Keep the executable virtual environment in your private
`$HOME` by default. Bridges-2 deletes `$LOCAL` after a job, so the supplied launcher
does not use it for persistent data.

## 1. Prepare the environment

PSC updates its module versions over time. Discover the current supported module
instead of hardcoding one:

```bash
module spider AI
module load <module-name-reported-by-PSC>
python -m venv --system-site-packages "$HOME/.venvs/caarma"
source "$HOME/.venvs/caarma/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements_bridges2.txt
python -m pip check
```

Use the same module when submitting by setting `AI_MODULE`; the launcher requires
it so the environment always receives a PSC-supported CUDA-enabled PyTorch build.

## 2. Stage data

The default config expects:

```text
$PROJECT/caarma-data/
├── voxceleb_full.csv
├── voxceleb1/vox1_dev_wav/wav/idNNNNN/.../*.wav
└── voxceleb1_test/wav/idNNNNN/.../*.wav
```

The training CSV must contain `utt_spk_int_labels`, plus a VoxCeleb id in either
`utt_spk_id` or `utt_paths`. The notebook can rebuild this CSV from the training
tree. The repository's `vox1_meta.csv` supplies the `Nationality` values.

## 3. Run

For an interactive run, open an Open OnDemand Jupyter session with a GPU and run
`CAARMA_nationality_bridges2.ipynb` from the repository root.

For a detached run:

```bash
export CAARMA_REPO_ROOT="$PROJECT/CAARMA_reprod"
export AI_MODULE=<module-name-reported-by-PSC>
sbatch -A <your-allocation> bridges2/train_nationality.sbatch
```

The launcher uses the private `$HOME/.venvs/caarma` environment and requests one
V100-32 GPU in `GPU-shared`. This is intentional:
pairing is local to each process's batch, so starting with one GPU avoids reducing
the number of same-nationality candidates across DDP ranks. Adjust the GPU type,
memory, time, or batch size to match your allocation and measured usage.
