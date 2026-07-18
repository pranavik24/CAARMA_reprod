# CAARMA Reproduction Agent Notes

## Persistent User Context

The user asked to keep the following plan as context and not implement it until
explicitly instructed.

## Modular CAARMA Experiment Refactor Plan

Refactor the experiment code so gender mixup, nationality mixup, and base
diffusion share the same training/evaluation pipeline. The goal is to stop
duplicating logic across `train.py`, `mixup_gender.py`, and
`mixup_nationality.py`, because that duplication is causing mismatched configs,
misleading W&B project names, checkpoint confusion, and inconsistent test
behavior.

Experiment switching should be config-driven:

```yaml
experiment_type: base | gender | nationality
synthetic_strategy: none | avg | diffusion
condition_attribute: none | gender | nationality
adversarial_enabled: true | false
```

### Key Changes To Implement Later

- Create a shared experiment layer:
  - One config loader/resolver that expands paths, validates required fields,
    derives `num_spk` from the active train split, and normalizes flags.
  - One Lightning task/training loop used by all experiment types.
  - One validation/test path that computes EER/minDCF identically for base,
    gender, and nationality.
  - One checkpoint policy: monitor `cosine_eer`, `mode: min`, save top-k,
    save last.
- Modularize synthetic generation:
  - Keep average mixup and diffusion as separate strategy helpers behind a
    common interface.
  - Add `condition_attribute` only where needed, so gender/nationality
    conditioning is an optional input, not a separate training script.
  - Base diffusion uses `condition_attribute: none`,
    `adversarial_enabled: false`, and writes to
    `caarma-output/base-diffusion`.
- Clean config organization:
  - `configs/base_clean_bridges2.yaml`
  - `configs/base_diffusion_bridges2.yaml`
  - `configs/gender_mixup_bridges2.yaml`
  - `configs/nationality_mixup_bridges2.yaml`
  - Each config has explicit `wandb_project`, `title`, `save_dir`, and
    `score_output_prefix` so W&B/project naming cannot accidentally say gender
    for base diffusion.
- Keep compatibility:
  - Existing `mixup_gender.py`, `mixup_nationality.py`, and `train.py` remain
    as thin wrappers for now.
  - New shared logic lives underneath them, so old commands still work while
    the implementation becomes modular.

### Experiment Defaults

- Base diffusion:
  - `num_spk`: derived from train split, expected `942`
  - `lambda_syn: 0.01`
  - `diffusion_fake_fraction: 0.25`
  - `diffusion_t_min: 0`
  - `diffusion_t_max: 3`
  - `diffusion_embedding_noise: 0.0`
  - `adversarial_enabled: false`
- Gender/nationality mixup:
  - Use the same shared trainer/evaluator.
  - Set only `condition_attribute: gender` or
    `condition_attribute: nationality`.
  - Use separate W&B projects and output dirs.

### Job Scripts

- Add one generic Bridges job script:
  - Reads `CAARMA_CONFIG`.
  - Trains using the shared entrypoint.
  - Selects the best checkpoint from the current job by lowest `cosine_eer`.
  - Tests the selected checkpoint on `CAARMA_TEST_SPLIT=test`.
  - Supports email via command-line `sbatch --mail-user ... --mail-type=BEGIN,END,FAIL`.
- Keep role-specific convenience scripts:
  - `train_base_diffusion.sbatch`
  - `train_gender.sbatch`
  - `train_nationality.sbatch`
  - These only set default config/output names, then call the generic logic.

### Tests

- Unit tests:
  - Config validation rejects mismatched experiment names/output dirs.
  - `num_spk` is derived deterministically from the active train split.
  - Synthetic strategy routing selects diffusion, average mixup, or none
    correctly.
  - Gender/nationality metadata is loaded only when `condition_attribute`
    requires it.
  - EER/minDCF computation is shared across experiment types.
- Script/config tests:
  - Base diffusion never logs to `caarma-gender`.
  - Gender and nationality configs use separate output dirs.
  - Combined Slurm job trains then tests the best `cosine_eer` checkpoint.
  - `python3 -m unittest discover -s tests` passes.

### Assumptions

- The intended main experiment is non-conditional base diffusion.
- Gender and nationality experiments should stay runnable for comparison.
- No dataset download or copy is needed.
- Final test should run once after training using the best validation
  checkpoint.

## Required Workflow When User Asks To Implement

When the user explicitly asks to implement any part of this plan:

1. Make the code changes locally.
2. Run focused tests first, then run:

   ```bash
   python3 -m unittest discover -s tests
   ```

3. Review the diff and avoid touching unrelated user changes.
4. Commit the completed work to Git with a conventional commit message.
5. Push the commit to GitHub.
6. Give the user the exact Bridges-2 commands below, adjusted only for the
   config/script that was implemented.

## Bridges-2 Runbook To Give User After Implementation

From the local machine, before pulling new GitHub changes into a dirty Bridges-2
checkout, preserve any uncommitted Bridges-2 work:

```bash
cd "$PROJECT/CAARMA_reprod"
git status
git stash push -u -m "bridges2 local work before CAARMA update"
git pull --rebase
git stash list
```

If the stash is needed after the pull:

```bash
git stash pop
```

Set the Bridges-2 environment variables:

```bash
cd "$PROJECT/CAARMA_reprod"
export CAARMA_REPO_ROOT="$PROJECT/CAARMA_reprod"
export AI_MODULE=<module-name-reported-by-PSC-module-spider-AI>
export CAARMA_TEST_SPLIT=test
```

Run base diffusion, the main intended experiment:

```bash
sbatch -A <your-allocation> \
  --mail-user=pkondapa@andrew.cmu.edu \
  --mail-type=BEGIN,END,FAIL \
  bridges2/train_base_diffusion.sbatch
```

Run gender mixup comparison:

```bash
sbatch -A <your-allocation> \
  --mail-user=pkondapa@andrew.cmu.edu \
  --mail-type=BEGIN,END,FAIL \
  bridges2/train_gender.sbatch
```

Run nationality mixup comparison:

```bash
sbatch -A <your-allocation> \
  --mail-user=pkondapa@andrew.cmu.edu \
  --mail-type=BEGIN,END,FAIL \
  bridges2/train_nationality.sbatch
```

Run the generic script with an explicit config:

```bash
export CAARMA_CONFIG="$CAARMA_REPO_ROOT/configs/base_diffusion_bridges2.yaml"
sbatch -A <your-allocation> \
  --mail-user=pkondapa@andrew.cmu.edu \
  --mail-type=BEGIN,END,FAIL \
  bridges2/run_experiment.sbatch
```

Monitor the submitted job:

```bash
squeue -u "$USER"
tail -f caarma-base-diffusion-<job-id>.out
```

Remember: if a Slurm script sources shared shell logic, it should source it from
`"${REPO_ROOT}/bridges2/..."`, not from `${BASH_SOURCE[0]}`, because Slurm may
execute a spool copy under `/var/spool/slurm/...`.
