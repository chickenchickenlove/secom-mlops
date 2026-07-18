"""Shared local artifact serialization for immutable datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

MLFLOW_DATASET_DIGEST_LENGTH = 36


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:v1:{digest.hexdigest()}"


def write_artifacts(
        output_dir: Path,
        frame: pd.DataFrame,
        manifest: dict[str, Any],
        stats: dict[str, Any],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.parquet"
    manifest_path = output_dir / "manifest.json"
    stats_path = output_dir / "stats.json"

    frame.to_parquet(dataset_path, index=False, engine="pyarrow")
    persisted_frame = pd.read_parquet(dataset_path, engine="pyarrow")
    if list(persisted_frame.columns) != list(frame.columns):
        raise RuntimeError("persisted dataset schema does not match the source frame")
    if len(persisted_frame) != len(frame):
        raise RuntimeError("persisted dataset row count does not match the source frame")

    artifact_sha256 = sha256_file(dataset_path)
    persisted_manifest = {
        **manifest,
        "artifact": {
            "path": "data/dataset.parquet",
            "sha256": artifact_sha256,
            "row_count": len(frame),
            "columns": list(frame.columns),
        },
    }
    manifest_path.write_text(
        json.dumps(
            persisted_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact_sha256
