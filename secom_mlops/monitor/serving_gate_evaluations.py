"""Durable MLflow records for serving-gate evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import mlflow

DEFAULT_EXPERIMENT_NAME = "secom-serving-gate-evaluations"
EVALUATION_SCHEMA_VERSION = "serving_gate_evaluation.v1"
COMPARISON_TYPE = "serving_gate_dataset_candidate_vs_champion"
LATEST_EVALUATION_RUN_ID_TAG = "candidate_serving_gate_latest_evaluation_run_id"
EVALUATION_ARTIFACT_PATH = "evaluation/evaluation.json"


@dataclass(frozen=True)
class ServingGateEvaluationRecord:
    evaluation_run_id: str
    evaluation_status: str
    model_name: str
    dataset_id: str
    dataset_manifest_hash: str
    dataset_artifact_sha256: str
    dataset_mlflow_run_id: str
    candidate_model_version: str
    candidate_model_run_id: str
    champion_model_version: str
    champion_model_run_id: str
    summary: dict[str, Any]


def _required_param(params: dict[str, str], name: str) -> str:
    value = params.get(name)
    if value is None or not str(value).strip():
        raise RuntimeError(f"serving-gate evaluation parameter is missing: {name}")
    return str(value)


def _assert_summary_value(
        summary: dict[str, Any],
        path: tuple[str, ...],
        expected: str,
) -> None:
    value: Any = summary
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise RuntimeError(
                "serving-gate evaluation summary field is missing: "
                f"{'.'.join(path)}"
            )
        value = value[key]
    if str(value) != expected:
        raise RuntimeError(
            "serving-gate evaluation summary mismatch: "
            f"field={'.'.join(path)} expected={expected} actual={value}"
        )


def parse_evaluation_run(run: Any, summary: dict[str, Any]) -> ServingGateEvaluationRecord:
    run_id = str(run.info.run_id)
    if str(run.info.status) != "FINISHED":
        raise RuntimeError(
            "serving-gate evaluation run is not complete: "
            f"evaluation_run_id={run_id} status={run.info.status}"
        )

    params = {str(key): str(value) for key, value in run.data.params.items()}
    schema_version = _required_param(params, "evaluation_schema_version")
    if schema_version != EVALUATION_SCHEMA_VERSION:
        raise RuntimeError(
            "unsupported serving-gate evaluation schema: "
            f"evaluation_run_id={run_id} schema={schema_version}"
        )
    comparison_type = _required_param(params, "comparison_type")
    if comparison_type != COMPARISON_TYPE:
        raise RuntimeError(
            "unexpected serving-gate comparison type: "
            f"evaluation_run_id={run_id} comparison_type={comparison_type}"
        )

    record = ServingGateEvaluationRecord(
        evaluation_run_id=run_id,
        evaluation_status=_required_param(params, "evaluation_status"),
        model_name=_required_param(params, "model_name"),
        dataset_id=_required_param(params, "dataset_id"),
        dataset_manifest_hash=_required_param(params, "dataset_manifest_hash"),
        dataset_artifact_sha256=_required_param(params, "dataset_artifact_sha256"),
        dataset_mlflow_run_id=_required_param(params, "dataset_mlflow_run_id"),
        candidate_model_version=_required_param(params, "candidate_model_version"),
        candidate_model_run_id=_required_param(params, "candidate_model_run_id"),
        champion_model_version=_required_param(params, "champion_model_version"),
        champion_model_run_id=_required_param(params, "champion_model_run_id"),
        summary=summary,
    )

    expected_summary_values = {
        ("evaluation_run_id",): record.evaluation_run_id,
        ("comparison_type",): COMPARISON_TYPE,
        ("eval_status",): record.evaluation_status,
        ("dataset", "dataset_id"): record.dataset_id,
        ("dataset", "manifest_hash"): record.dataset_manifest_hash,
        ("dataset", "artifact_sha256"): record.dataset_artifact_sha256,
        ("dataset", "mlflow_run_id"): record.dataset_mlflow_run_id,
        ("candidate", "model_version"): record.candidate_model_version,
        ("candidate", "model_run_id"): record.candidate_model_run_id,
        ("champion", "model_version"): record.champion_model_version,
        ("champion", "model_run_id"): record.champion_model_run_id,
        ("gate_policy", "primary_metric"): _required_param(params, "primary_metric"),
        ("gate_policy", "min_primary_delta"): _required_param(
            params, "min_primary_delta"
        ),
        ("gate_policy", "min_recall_delta"): _required_param(
            params, "min_recall_delta"
        ),
        ("gate_policy", "min_precision_delta"): _required_param(
            params, "min_precision_delta"
        ),
    }
    for path, expected in expected_summary_values.items():
        _assert_summary_value(summary, path, expected)
    return record


def load_evaluation_run(
        client: Any,
        evaluation_run_id: str,
) -> ServingGateEvaluationRecord:
    run_id = evaluation_run_id.strip()
    if not run_id:
        raise ValueError("evaluation_run_id is required")
    run = client.get_run(run_id)
    summary = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/{EVALUATION_ARTIFACT_PATH}"
    )
    if not isinstance(summary, dict):
        raise RuntimeError(
            f"serving-gate evaluation summary is invalid: evaluation_run_id={run_id}"
        )
    return parse_evaluation_run(run, summary)


def evaluation_reason_json(reasons: list[str]) -> str:
    return json.dumps(reasons, ensure_ascii=False, separators=(",", ":"))
