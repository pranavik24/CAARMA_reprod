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
        self.assertNotIn("#SBATCH --account=", source)

    def test_static_config_does_not_oversubscribe_one_gpu_cpu_allocation(self):
        source = (ROOT / "config_nationality_bridges2.yaml").read_text()

        self.assertIn("num_workers: 4", source)


if __name__ == "__main__":
    unittest.main()
