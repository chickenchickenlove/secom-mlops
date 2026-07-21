"""Train a candidate model from a verified, versioned training Dataset."""

import argparse

import numpy as np

from secom_mlops.datasets.training_dataset_loader import (
    load_training_dataset,
)
from secom_mlops.training.candidate_registration import register_candidate, CandidateRegistrationConfig
from secom_mlops.training.training_data_preparation import (
    prepare_training_dataset,
    PreparedTrainingData, validate_training_data
)
from secom_mlops.training.random_forest_training import (
    RandomForestTrainingConfig,
    train_random_forest,
)
from secom_mlops.training.candidate_tracking import (
    CandidateSplitStats,
    CandidateTrackingContext,
    build_candidate_tracking_record,
)
from secom_mlops_common.cli.validators import (
    positive_int,
    positive_int_list,
    probability_list,
)
from secom_mlops_common.config.mlflow import (
    DEFAULT_CANDIDATE_ALIAS,
    ENV_ML_CANDIDATE_GROUP,
    ENV_ML_TRAINING_JOB_ID,
    MODEL_ROLE_CANDIDATE,
    MODEL_ROLES,
    get_env_value,
    resolve_model_alias,
    resolve_model_name,
    resolve_model_role,
    resolve_tracking_uri,
)

POSITIVE_CLASS = 1

DEFAULT_N_ESTIMATORS = "100,300"
DEFAULT_MIN_SAMPLES_LEAF = "1,3"
DEFAULT_THRESHOLDS = "0.1,0.2,0.3,0.4,0.5"
MAX_DEVELOPMENT_SAMPLES = 1000
VALIDATION_SIZE = 0.2
DEFAULT_MIN_LABEL_COVERAGE = 0.95
DEVELOPMENT_SAMPLE_SELECTION = "latest_labeled_snapshot_available_at"
GATE_SOURCE = "serving_feature_snapshots"


def coverage_float(raw_value: str) -> float:
    value = float(raw_value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError(
            "value must be finite and between 0 and 1"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--model-name", default=resolve_model_name())
    parser.add_argument(
        "--model-alias",
        default=resolve_model_alias(default=DEFAULT_CANDIDATE_ALIAS),
    )
    parser.add_argument("--model-role", default=resolve_model_role())
    parser.add_argument("--candidate-group", default=get_env_value(ENV_ML_CANDIDATE_GROUP))
    parser.add_argument("--training-job-id", default=get_env_value(ENV_ML_TRAINING_JOB_ID))

    parser.add_argument("--dataset-id", required=True)

    parser.add_argument("--min-label-coverage", type=coverage_float, default=DEFAULT_MIN_LABEL_COVERAGE,)
    parser.add_argument("--min-fail-samples", type=positive_int, default=20)
    parser.add_argument("--min-pass-samples", type=positive_int, default=20)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--n-estimators", default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--min-samples-leaf", default=DEFAULT_MIN_SAMPLES_LEAF)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.model_role not in MODEL_ROLES:
        raise ValueError("model_role must be one of: candidate, champion")

    if args.model_role == MODEL_ROLE_CANDIDATE:
        missing = []
        if not args.candidate_group:
            missing.append("--candidate-group or ML_CANDIDATE_GROUP")
        if not args.training_job_id:
            missing.append("--training-job-id or ML_TRAINING_JOB_ID")
        if missing:
            raise ValueError("candidate training requires " + ", ".join(missing))


def train_and_register(args: argparse.Namespace) -> None:
    tracking_uri = resolve_tracking_uri(args.tracking_uri)

    # 1. Get the dataset from MLflow.
    loaded_dataset = load_training_dataset(
        args.dataset_id,
        tracking_uri=tracking_uri,
    )

    training_data: PreparedTrainingData = prepare_training_dataset(
        loaded_dataset
    )

    # 2. Validate prepared dataset.
    validate_training_data(
        metadata=training_data.metadata,
        min_samples=MAX_DEVELOPMENT_SAMPLES,
        min_label_coverage=args.min_label_coverage,
        min_fail_samples=args.min_fail_samples,
        min_pass_samples=args.min_pass_samples,
    )

    # 3. Train models.
    training_config = RandomForestTrainingConfig(
        n_estimators=positive_int_list(args.n_estimators, "--n-estimators"),
        min_samples_leaf=positive_int_list(args.min_samples_leaf, "--min-samples-leaf"),
        thresholds=probability_list(args.thresholds, "--thresholds"),
        validation_size=VALIDATION_SIZE,
        random_state=args.random_state,
    )

    training_result = train_random_forest(
        training_data.features,
        training_data.targets,
        training_config,
    )

    train_targets = training_data.targets.iloc[
        training_result.train_indices
    ]
    validation_targets = training_data.targets.iloc[
        training_result.validation_indices
    ]

    # 4. build tracking metadata.
    tracking = build_candidate_tracking_record(
        context=CandidateTrackingContext(
            tracking_uri=tracking_uri,
            model_name=args.model_name,
            model_alias=args.model_alias,
            model_role=args.model_role,
            candidate_group=args.candidate_group,
            training_job_id=args.training_job_id,
            random_state=args.random_state,
            min_label_coverage=args.min_label_coverage,
            validation_size=VALIDATION_SIZE,
            gate_source=GATE_SOURCE,
        ),
        metadata=training_data.metadata,
        best_row=training_result.best_row,
        split_stats=CandidateSplitStats(
            training_sample_count=len(training_result.train_indices),
            validation_sample_count=len(training_result.validation_indices),
            training_fail_count=int((train_targets == POSITIVE_CLASS).sum()),
            validation_fail_count=int((validation_targets == POSITIVE_CLASS).sum()),
        ),
    )

    if args.dry_run:
        print(tracking.dry_run_message())
        return

    # Write to the MLflow.
    registration_config = CandidateRegistrationConfig(
        tracking_uri=tracking_uri,
        experiment_name="secom-fail-detection",
        model_name=args.model_name,
        model_alias=args.model_alias,
    )

    registered = register_candidate(
        training_data=training_data,
        training_result=training_result,
        tracking=tracking,
        config=registration_config,
    )

    print(
        tracking.registered_message(
            model_version=registered.model_version,
            run_id=registered.run_id,
        )
    )

def main() -> None:
    args = parse_args()
    validate_args(args)
    train_and_register(args)


if __name__ == "__main__":
    main()
