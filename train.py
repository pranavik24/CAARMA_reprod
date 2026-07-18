"""Compatibility wrapper for the modular CAARMA experiment runner."""

from experiments.caarma import cli_main


if __name__ == "__main__":
    cli_main(
        default_config="configs/base_clean_bridges2.yaml",
        default_experiment_type="base",
        description="Run a modular CAARMA base experiment",
    )
