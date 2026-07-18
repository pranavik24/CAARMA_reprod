"""Compatibility wrapper for gender-conditioned CAARMA experiments."""

from experiments.caarma import (
    AMSoftmaxGANGender,
    LegacyMixupDiscriminator as MixupDiscriminator,
    ResidualBlock,
    Task,
    TestOnlyDataModule,
    _as_bool,
    _ensure_root_suffix,
    _nearest_candidate,
    _none_like,
    _resolve_path,
    accuracy,
    build_embedding_discriminator,
    build_gender_criterion,
    build_label_gender_map,
    build_trainer,
    cli_main as _shared_cli_main,
    extract_speaker_id,
    load_config,
    load_vox1_gender_by_speaker,
    mixup_data_euc_avg_gender,
    parse_experiment_args,
    prepare_experiment_config,
)


def parse_args():
    return parse_experiment_args(
        "configs/gender_mixup_bridges2.yaml",
        "CAARMA training/testing with gender-conditioned mixup",
    )


def prepare_config(config, config_path, args):
    return prepare_experiment_config(
        config,
        config_path,
        args,
        default_experiment_type="gender",
    )


def cli_main() -> None:
    _shared_cli_main(
        default_config="configs/gender_mixup_bridges2.yaml",
        default_experiment_type="gender",
        description="CAARMA training/testing with gender-conditioned mixup",
    )


if __name__ == "__main__":
    cli_main()
