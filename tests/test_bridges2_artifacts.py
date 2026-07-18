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

    def test_slurm_script_requests_one_gpu_and_uses_srun(self):
        source = (ROOT / "bridges2" / "train_nationality.sbatch").read_text()

        self.assertIn("#SBATCH --partition=GPU-shared", source)
        self.assertIn("#SBATCH --gpus=v100-32:1", source)
        self.assertIn("#SBATCH --cpus-per-task=4", source)
        self.assertIn("#SBATCH --mem=62000M", source)
        self.assertIn("conda.sh", source)
        self.assertIn("srun", source)
        self.assertIn("mixup_nationality.py", source)
        self.assertIn("configs/nationality_bridges2.yaml", source)
        self.assertNotIn("#SBATCH --account=", source)

    def test_test_slurm_script_runs_nationality_test_mode(self):
        source = (ROOT / "bridges2" / "test_nationality.sbatch").read_text()

        self.assertIn("#SBATCH --gpus=v100-32:1", source)
        self.assertIn("CAARMA_CHECKPOINT", source)
        self.assertIn("--mode test", source)
        self.assertIn("--validation-split", source)
        self.assertIn("TEST_SPLIT=\"${CAARMA_TEST_SPLIT:-test}\"", source)
        self.assertIn("srun", source)
        self.assertNotIn("#SBATCH --account=", source)

    def test_gender_slurm_script_runs_gender_entrypoint(self):
        source = (ROOT / "bridges2" / "train_gender.sbatch").read_text()

        self.assertIn("#SBATCH --gpus=v100-32:1", source)
        self.assertIn("#SBATCH --cpus-per-task=4", source)
        self.assertIn("#SBATCH --mem=62000M", source)
        self.assertIn("conda.sh", source)
        self.assertIn("mixup_gender.py", source)
        self.assertIn("configs/gender_bridges2.yaml", source)
        self.assertIn("--mode train", source)
        self.assertIn("--mode test", source)
        self.assertIn("BEST_CKPT", source)
        self.assertIn("cosine_eer=", source)
        self.assertIn("TEST_SPLIT=\"${CAARMA_TEST_SPLIT:-test}\"", source)
        self.assertIn("SCORE_PREFIX=\"${CAARMA_SCORE_PREFIX:-caarma_gender_${TEST_SPLIT}}\"", source)
        self.assertIn("--sl-mixup", source)
        self.assertIn("srun", source)
        self.assertNotIn("#SBATCH --account=", source)

    def test_test_slurm_script_runs_gender_test_mode(self):
        source = (ROOT / "bridges2" / "test_gender.sbatch").read_text()

        self.assertIn("#SBATCH --gpus=v100-32:1", source)
        self.assertIn("CAARMA_CHECKPOINT", source)
        self.assertIn("mixup_gender.py", source)
        self.assertIn("configs/gender_bridges2.yaml", source)
        self.assertIn("--mode test", source)
        self.assertIn("--validation-split", source)
        self.assertIn("--score-output-prefix", source)
        self.assertIn("TEST_SPLIT=\"${CAARMA_TEST_SPLIT:-test}\"", source)
        self.assertIn("srun", source)
        self.assertNotIn("#SBATCH --account=", source)

    def test_base_diffusion_slurm_script_runs_base_entrypoint(self):
        source = (ROOT / "bridges2" / "train_base_diffusion.sbatch").read_text()

        self.assertIn("#SBATCH --job-name=caarma-base-diffusion", source)
        self.assertIn("#SBATCH --output=caarma-base-diffusion-%j.out", source)
        self.assertIn("#SBATCH --gpus=v100-32:1", source)
        self.assertIn("configs/base_diffusion_bridges2.yaml", source)
        self.assertIn("train.py", source)
        self.assertIn("srun", source)
        self.assertNotIn("--sl-mixup", source)
        self.assertNotIn("#SBATCH --account=", source)

    def test_static_config_does_not_oversubscribe_one_gpu_cpu_allocation(self):
        source = (ROOT / "configs" / "nationality_bridges2.yaml").read_text()

        self.assertIn("epochs: 20", source)
        self.assertIn("Nationality mixup config", source)
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
        self.assertIn("synthetic_strategy: diffusion", source)
        self.assertIn("diffusion_timesteps: 100", source)
        self.assertIn("diffusion_t_min: 1", source)
        self.assertIn("diffusion_t_max: 20", source)
        self.assertIn("batch_size: 100", source)
        self.assertIn("num_workers: 4", source)
        self.assertIn("generate_validation_trials: true", source)
        self.assertIn("validation_split: val", source)
        self.assertIn("num_spk: 1211", source)
        self.assertIn("save_dir: \"${PROJECT}/caarma-output/base-diffusion\"", source)

    def test_gender_config_uses_server_splits_and_intermediate_discriminator(self):
        source = (ROOT / "configs" / "gender_bridges2.yaml").read_text()

        self.assertIn("USE_WANDB: true", source)
        self.assertIn("Gender diffusion-mixup config", source)
        self.assertIn("wandb_project: caarma-gender", source)
        self.assertIn("init_lr: 0.0003", source)
        self.assertIn("epochs: 30", source)
        self.assertIn("batch_size: 100", source)
        self.assertIn("sl_mixup: true", source)
        self.assertIn("synthetic_strategy: diffusion", source)
        self.assertIn("diffusion_timesteps: 100", source)
        self.assertIn("diffusion_t_min: 1", source)
        self.assertIn("diffusion_t_max: 5", source)
        self.assertIn("diffusion_beta_start: 0.0001", source)
        self.assertIn("diffusion_beta_end: 0.02", source)
        self.assertIn("diffusion_embedding_noise: 0.005", source)
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
        self.assertIn("validate_during_train: true", source)
        self.assertIn("save_dir: ${PROJECT}/caarma-output/gender", source)


if __name__ == "__main__":
    unittest.main()
