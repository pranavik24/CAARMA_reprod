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

The default config uses the shared server-staged VoxCeleb1 dataset directly:

```text
/ocean/projects/cis220031p/shared/raw/data/VoxCeleb1/
├── wav/idNNNNN/.../*.wav
└── data/
    ├── vox1_train.csv
    ├── vox1_val.csv
    └── vox1_test.csv
```

The split CSVs contain `VoxCeleb1_ID,VGGFace1_ID,Gender,Nationality,Set`. At
runtime the dataloader expands the active speaker split into the existing
`utt_spk_int_labels`, `utt_spk_id`, and `utt_paths` training format without
redownloading or copying audio. Set `active_split` in the config to choose a
different split.

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
V100-32 GPU, 5 CPU cores, and 60GB RAM in `GPU-shared`. This is intentional:
pairing is local to each process's batch, so starting with one GPU avoids reducing
the number of same-nationality candidates across DDP ranks. Adjust the GPU type,
memory, time, or batch size to match your allocation and measured usage.
