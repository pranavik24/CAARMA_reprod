# Running modular CAARMA experiments on Bridges-2

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
the selected config through `train.py` from the repository root.

For a detached run:

```bash
export CAARMA_REPO_ROOT="$PROJECT/CAARMA_reprod"
export AI_MODULE=<module-name-reported-by-PSC>
sbatch -A <your-allocation> bridges2/train_base_diffusion.sbatch
```

Use `bridges2/train_gender.sbatch` or `bridges2/train_nationality.sbatch` for
the conditioned mixup variants. All three scripts set a default config, then call
the shared `bridges2/run_experiment_body.sh` logic.

The launcher uses the private `$HOME/.venvs/caarma` environment and requests one
V100-32 GPU, 4 CPU cores, and 62GB RAM in `GPU-shared`. Each train job now runs
training, selects the best checkpoint by lowest validation `cosine_eer`, and then
tests that checkpoint on `CAARMA_TEST_SPLIT`, defaulting to `test`.

To get mail from Slurm, add these flags to `sbatch`:

```bash
--mail-user=<your-email> --mail-type=BEGIN,END,FAIL
```
