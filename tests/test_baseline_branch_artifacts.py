import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BaselineBranchArtifactTests(unittest.TestCase):
    def test_baseline_config_matches_mfa_conformer_no_adversarial_setup(self):
        source = (ROOT / "configs" / "baseline_mfa_conformer_bridges2.yaml").read_text()

        self.assertIn("model: MFA-CONFORMER", source)
        self.assertIn("features: Fbank", source)
        self.assertIn("criterion: AMSoftmax", source)
        self.assertIn("adversarial_enabled: false", source)
        self.assertIn("mixup: false", source)
        self.assertIn("num_blocks: 6", source)
        self.assertIn("encoder_dim: 256", source)
        self.assertIn("attention_heads: 4", source)
        self.assertIn("conformer_kernel_size: 15", source)
        self.assertIn("embedding_dim: 192", source)
        self.assertIn("active_split: train", source)
        self.assertIn("num_spk: 942", source)
        self.assertIn("trial_path: null", source)
        self.assertIn("validation_split: val", source)
        self.assertIn("save_dir: ${PROJECT}/caarma-output/baseline-mfa-conformer-clean", source)
        self.assertNotIn("voxceleb_full.csv", source)
        self.assertNotIn("AMSoftmaxGAN", source)

    def test_baseline_runner_uses_single_gpu_no_ddp_no_discriminator(self):
        source = (ROOT / "train_baseline.py").read_text()

        self.assertIn("class AMSoftmax", source)
        self.assertIn("class BaselineTask", source)
        self.assertIn("def build_trainer", source)
        self.assertIn("devices=int(config.get(\"devices\", 1))", source)
        self.assertIn("strategy=\"auto\"", source)
        self.assertIn("monitor=\"cosine_eer\"", source)
        self.assertIn("mode=\"min\"", source)
        self.assertNotIn("DDPStrategy", source)
        self.assertNotIn("MixupDiscriminator", source)
        self.assertNotIn("BCEWithLogitsLoss", source)
        self.assertNotIn("flagSyn", source)

    def test_baseline_slurm_trains_then_tests_best_checkpoint(self):
        source = (ROOT / "bridges2" / "train_baseline.sbatch").read_text()

        self.assertIn("#SBATCH --job-name=caarma-baseline-mfa", source)
        self.assertIn("#SBATCH --gpus=v100-32:1", source)
        self.assertIn("#SBATCH --cpus-per-task=5", source)
        self.assertIn("CAARMA_CONFIG", source)
        self.assertIn("configs/baseline_mfa_conformer_bridges2.yaml", source)
        self.assertIn("--mode train", source)
        self.assertIn("--mode test", source)
        self.assertIn("BEST_CKPT", source)
        self.assertIn("cosine_eer=", source)
        self.assertIn("--trial-path \"${TEST_TRIAL_PATH}\"", source)
        self.assertNotIn("#SBATCH --account=", source)


if __name__ == "__main__":
    unittest.main()
