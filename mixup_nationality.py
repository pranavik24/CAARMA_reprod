"""Compatibility wrapper for nationality-conditioned CAARMA experiments."""

from experiments.caarma import (
    AMSoftmaxGANNationality,
    Task,
    TestOnlyDataModule,
    _as_bool,
    _ensure_root_suffix,
    _nearest_candidate,
    _none_like,
    _resolve_path,
    accuracy,
    build_nationality_criterion,
    build_task,
    build_trainer,
    cli_main as _shared_cli_main,
    load_config,
    mixup_data_euc_avg_nationality,
    parse_experiment_args,
    prepare_experiment_config,
)
from functions.nationality import (
    build_label_nationality_map,
    canonical_speaker_pair,
    eligible_peer_count,
    load_vox1_nationality_by_speaker,
    same_nationality_candidates,
)


def parse_args():
    return parse_experiment_args(
        "configs/nationality_mixup_bridges2.yaml",
        "CAARMA training/testing with nationality-conditioned mixup",
    )


def prepare_config(config, config_path, args):
    return prepare_experiment_config(
        config,
        config_path,
        args,
        default_experiment_type="nationality",
    )


def cli_main() -> None:
    _shared_cli_main(
        default_config="configs/nationality_mixup_bridges2.yaml",
        default_experiment_type="nationality",
        description="CAARMA training/testing with nationality-conditioned mixup",
    )


if __name__ == "__main__":
    cli_main()
