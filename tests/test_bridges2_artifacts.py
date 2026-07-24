import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Bridges2ArtifactTests(unittest.TestCase):
    def test_notebook_is_cluster_oriented_and_runs_nationality_entrypoint(self):
        notebook = json.loads((ROOT / "CAARMA_nationality_bridges2.ipynb").read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("mixup_nationality.py", source)
        self.assertIn("SLURM_CPUS_PER_TASK", source)
        self.assertIn("PROJECT", source)
        self.assertIn("torch.cuda.is_available", source)
        self.assertNotIn("google.colab", source)
        self.assertNotIn("/content", source)

    def test_shared_slurm_body_trains_then_tests(self):
        source = (ROOT / "bridges2" / "run_experiment_body.sh").read_text()

        self.assertIn("conda.sh", source)
        self.assertIn("module load", source)
        self.assertIn("srun", source)
        self.assertIn("--mode train", source)
        self.assertIn("--mode test", source)
        self.assertIn("BEST_CKPT", source)
        self.assertIn("cosine_eer=", source)
        self.assertIn("TEST_SPLIT=\"${CAARMA_TEST_SPLIT:-test}\"", source)
        self.assertNotIn("#SBATCH --account=", source)

    def test_train_nationality_script_sets_modular_defaults(self):
        source = (ROOT / "bridges2" / "train_nationality.sbatch").read_text()

        self.assertIn("#SBATCH --partition=GPU-shared", source)
        self.assertIn("#SBATCH --gpus=v100-32:1", source)
        self.assertIn("#SBATCH --cpus-per-task=4", source)
        self.assertIn("#SBATCH --mem=62000M", source)
        self.assertIn("configs/nationality_mixup_bridges2.yaml", source)
        self.assertIn("CAARMA_ENTRYPOINT=\"${CAARMA_ENTRYPOINT:-train.py}\"", source)
        self.assertIn("run_experiment_body.sh", source)
        self.assertNotIn("#SBATCH --account=", source)

    def test_test_slurm_script_runs_nationality_test_mode(self):
        source = (ROOT / "bridges2" / "test_nationality.sbatch").read_text()

        self.assertIn("#SBATCH --gpus=v100-32:1", source)
        self.assertIn("CAARMA_CHECKPOINT", source)
        self.assertIn("train.py", source)
        self.assertIn("configs/nationality_mixup_bridges2.yaml", source)
        self.assertIn("--mode test", source)
        self.assertIn("--validation-split", source)
        self.assertIn("TEST_SPLIT=\"${CAARMA_TEST_SPLIT:-test}\"", source)
        self.assertIn("srun", source)
        self.assertNotIn("#SBATCH --account=", source)

    def test_gender_slurm_script_sets_modular_defaults(self):
        source = (ROOT / "bridges2" / "train_gender.sbatch").read_text()

        self.assertIn("#SBATCH --gpus=v100-32:1", source)
        self.assertIn("#SBATCH --cpus-per-task=4", source)
        self.assertIn("#SBATCH --mem=62000M", source)
        self.assertIn("configs/gender_mixup_bridges2.yaml", source)
        self.assertIn("CAARMA_ENTRYPOINT=\"${CAARMA_ENTRYPOINT:-train.py}\"", source)
        self.assertIn("run_experiment_body.sh", source)
        self.assertNotIn("#SBATCH --account=", source)

    def test_test_slurm_script_runs_gender_test_mode(self):
        source = (ROOT / "bridges2" / "test_gender.sbatch").read_text()

        self.assertIn("#SBATCH --gpus=v100-32:1", source)
        self.assertIn("CAARMA_CHECKPOINT", source)
        self.assertIn("train.py", source)
        self.assertIn("configs/gender_mixup_bridges2.yaml", source)
        self.assertIn("--mode test", source)
        self.assertIn("--validation-split", source)
        self.assertIn("--score-output-prefix", source)
        self.assertIn("TEST_SPLIT=\"${CAARMA_TEST_SPLIT:-test}\"", source)
        self.assertIn("srun", source)
        self.assertNotIn("#SBATCH --account=", source)

    def test_base_diffusion_slurm_script_sets_modular_defaults(self):
        source = (ROOT / "bridges2" / "train_base_diffusion.sbatch").read_text()

        self.assertIn("#SBATCH --job-name=caarma-base-diffusion", source)
        self.assertIn("#SBATCH --output=caarma-base-diffusion-%j.out", source)
        self.assertIn("#SBATCH --gpus=v100-32:1", source)
        self.assertIn("configs/base_diffusion_bridges2.yaml", source)
        self.assertIn("CAARMA_ENTRYPOINT=\"${CAARMA_ENTRYPOINT:-train.py}\"", source)
        self.assertIn("run_experiment_body.sh", source)
        self.assertNotIn("--sl-mixup", source)
        self.assertNotIn("#SBATCH --account=", source)

    def test_training_slurm_scripts_source_body_from_repo_root(self):
        for script_name in (
            "run_experiment.sbatch",
            "train_base_diffusion.sbatch",
            "train_gender.sbatch",
            "train_nationality.sbatch",
        ):
            with self.subTest(script_name=script_name):
                source = (ROOT / "bridges2" / script_name).read_text()

                self.assertIn('source "${REPO_ROOT}/bridges2/run_experiment_body.sh"', source)
                self.assertNotIn("BASH_SOURCE", source)

    def test_nationality_config_is_explicit(self):
        source = (ROOT / "configs" / "nationality_mixup_bridges2.yaml").read_text()

        self.assertIn("Nationality-conditioned average-mixup config", source)
        self.assertIn("experiment_type: nationality", source)
        self.assertIn("condition_attribute: nationality", source)
        self.assertIn("synthetic_strategy: avg", source)
        self.assertIn("epochs: 20", source)
        self.assertIn("num_workers: 4", source)
        self.assertIn("val_num_workers: 4", source)
        self.assertIn("train_drop_last: true", source)
        self.assertIn("USE_WANDB: true", source)
        self.assertIn("discriminator: simple", source)
        self.assertIn("generate_validation_trials: true", source)
        self.assertIn("validation_split: val", source)
        self.assertIn("validate_during_train: true", source)

    def test_base_diffusion_config_is_explicit(self):
        source = (ROOT / "configs" / "base_diffusion_bridges2.yaml").read_text()

        self.assertIn("Base diffusion-mixup config", source)
        self.assertIn("experiment_type: base", source)
        self.assertIn("condition_attribute: none", source)
        self.assertIn("synthetic_strategy: diffusion", source)
        self.assertIn("adversarial_enabled: false", source)
        self.assertIn("wandb_project: caarma-diffusion", source)
        self.assertIn("diffusion_timesteps: 100", source)
        self.assertIn("diffusion_t_min: 0", source)
        self.assertIn("diffusion_t_max: 3", source)
        self.assertIn("diffusion_fake_fraction: 0.25", source)
        self.assertIn("lambda_syn: 0.01", source)
        self.assertIn("batch_size: 100", source)
        self.assertIn("num_workers: 4", source)
        self.assertIn("generate_validation_trials: true", source)
        self.assertIn("validation_split: val", source)
        self.assertIn("validation_pos_pairs_per_speaker: 2", source)
        self.assertIn("validation_neg_pairs_per_speaker: 5", source)
        self.assertIn("num_spk: 942", source)
        self.assertIn("save_dir: ${PROJECT}/caarma-output/base-diffusion", source)
        self.assertNotIn("caarma-gender", source)

    def test_base_diffusion_2to1_config_sets_half_fake_fraction(self):
        source = (ROOT / "configs" / "base_diffusion_2to1_bridges2.yaml").read_text()

        self.assertIn("2:1 real-to-fake speaker ratio", source)
        self.assertIn("experiment_type: base", source)
        self.assertIn("condition_attribute: none", source)
        self.assertIn("synthetic_strategy: diffusion", source)
        self.assertIn("adversarial_enabled: false", source)
        self.assertIn("diffusion_fake_fraction: 0.5", source)
        self.assertIn("save_dir: ${PROJECT}/caarma-output/base-diffusion-2to1", source)
        self.assertIn("title: caarma_base_diffusion_2to1", source)
        self.assertNotIn("caarma-gender", source)

    def test_base_diffusion_2to1_low_lambda_config_sets_half_fake_fraction(self):
        source = (ROOT / "configs" / "base_diffusion_2to1_lambdasyn005_bridges2.yaml").read_text()

        self.assertIn("2:1 real-to-fake speaker ratio", source)
        self.assertIn("lower synthetic loss weight", source)
        self.assertIn("experiment_type: base", source)
        self.assertIn("condition_attribute: none", source)
        self.assertIn("synthetic_strategy: diffusion", source)
        self.assertIn("adversarial_enabled: false", source)
        self.assertIn("diffusion_fake_fraction: 0.5", source)
        self.assertIn("lambda_syn: 0.005", source)
        self.assertIn("save_dir: ${PROJECT}/caarma-output/base-diffusion-2to1-lambdasyn005", source)
        self.assertIn("title: caarma_base_diffusion_2to1_lambdasyn005", source)
        self.assertNotIn("caarma-gender", source)

    def test_base_diffusion_3to1_config_sets_third_fake_fraction(self):
        source = (ROOT / "configs" / "base_diffusion_3to1_bridges2.yaml").read_text()

        self.assertIn("3:1 real-to-fake speaker ratio", source)
        self.assertIn("experiment_type: base", source)
        self.assertIn("condition_attribute: none", source)
        self.assertIn("synthetic_strategy: diffusion", source)
        self.assertIn("adversarial_enabled: false", source)
        self.assertIn("diffusion_fake_fraction: 0.3333333333333333", source)
        self.assertIn("save_dir: ${PROJECT}/caarma-output/base-diffusion-3to1", source)
        self.assertIn("title: caarma_base_diffusion_3to1", source)
        self.assertNotIn("caarma-gender", source)

    def test_gender_config_uses_server_splits_and_intermediate_discriminator(self):
        source = (ROOT / "configs" / "gender_mixup_bridges2.yaml").read_text()

        self.assertIn("Gender-conditioned average-mixup config", source)
        self.assertIn("experiment_type: gender", source)
        self.assertIn("condition_attribute: gender", source)
        self.assertIn("wandb_project: caarma-gender", source)
        self.assertIn("init_lr: 0.0003", source)
        self.assertIn("epochs: 30", source)
        self.assertIn("batch_size: 100", source)
        self.assertIn("sl_mixup: true", source)
        self.assertIn("synthetic_strategy: avg", source)
        self.assertIn("discriminator: intermediate", source)
        self.assertIn("discriminator_hidden_dim: 512", source)
        self.assertIn("discriminator_mid_dim: 384", source)
        self.assertIn("discriminator_head_dim: 128", source)
        self.assertIn("discriminator_dropout: 0.15", source)
        self.assertIn("discriminator_lr: 0.0001", source)
        self.assertIn("lambda_adv: 0.02", source)
        self.assertIn("lambda_adv_min: 0.001", source)
        self.assertIn("lambda_adv_max: 0.05", source)
        self.assertIn("lambda_syn: 0.05", source)
        self.assertIn("pretrain_eps: 5", source)
        self.assertIn("log_training_steps: false", source)
        self.assertIn("save_top_k: 3", source)
        self.assertIn("generate_validation_trials: true", source)
        self.assertIn("validation_max_speakers: 1000", source)
        self.assertIn("validation_pos_pairs_per_speaker: 2", source)
        self.assertIn("validation_neg_pairs_per_speaker: 5", source)
        self.assertIn("validate_during_train: true", source)
        self.assertIn("save_dir: ${PROJECT}/caarma-output/gender", source)


if __name__ == "__main__":
    unittest.main()
