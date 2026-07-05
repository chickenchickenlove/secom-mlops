#!/usr/bin/env bash
set -euo pipefail

BASELINE_START=""
BASELINE_END=""
BASELINE_NAME="fixed-reference-champion-manual"
MIN_SAMPLES="500"
MIN_FEATURE_NON_NULL="30"
MIN_FEATURE_SAMPLES="30"
PSI_BIN_COUNT="10"
MIN_PSI_FEATURE_NON_NULL="30"
NOTES="manual fixed-reference baseline"
CREATED_BY="${USER:-airflow}"
RETIRE_EXISTING_ACTIVE="True"
DRY_RUN="True"

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://mlflow:5100}"
MODEL_NAME="${ML_MODEL_NAME:-secom-fail-detector}"
MODEL_ALIAS="${ML_MODEL_ALIAS:-champion}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --baseline-start) BASELINE_START="${2:-}"; shift 2 ;;
    --baseline-end) BASELINE_END="${2:-}"; shift 2 ;;
    --baseline-name) BASELINE_NAME="${2:-}"; shift 2 ;;
    --min-samples) MIN_SAMPLES="${2:-}"; shift 2 ;;
    --min-feature-non-null) MIN_FEATURE_NON_NULL="${2:-}"; shift 2 ;;
    --min-feature-samples) MIN_FEATURE_SAMPLES="${2:-}"; shift 2 ;;
    --psi-bin-count) PSI_BIN_COUNT="${2:-}"; shift 2 ;;
    --min-psi-feature-non-null) MIN_PSI_FEATURE_NON_NULL="${2:-}"; shift 2 ;;
    --notes) NOTES="${2:-}"; shift 2 ;;
    --created-by) CREATED_BY="${2:-}"; shift 2 ;;
    --retire-existing-active) RETIRE_EXISTING_ACTIVE="${2:-True}"; shift 2 ;;
    --dry-run) DRY_RUN="${2:-True}"; shift 2 ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

is_blank() {
  case "${1:-}" in
    ""|"None"|"none"|"NONE"|"Null"|"null"|"NULL"|"Nil"|"nil"|"NIL")
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_truthy() {
  case "${1:-}" in
    "1"|"true"|"True"|"TRUE"|"yes"|"Yes"|"YES"|"y"|"Y"|"on"|"On"|"ON")
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_python() {
  if [ -n "${PYTHON_BIN:-}" ] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    printf "%s\n" "${PYTHON_BIN}"
  elif command -v python >/dev/null 2>&1; then
    printf "%s\n" "python"
  elif command -v python3 >/dev/null 2>&1; then
    printf "%s\n" "python3"
  else
    echo "python executable not found" >&2
    exit 1
  fi
}

normalize_epoch_time() {
  local value="$1"
  local name="$2"
  "${PYTHON}" -c '
import math
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

raw_value = sys.argv[1].strip()
name = sys.argv[2]

try:
    parsed = float(raw_value)
except ValueError:
    normalized = raw_value
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed_datetime = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SystemExit(f"{name} must be epoch seconds or ISO datetime: {raw_value}") from exc
    if parsed_datetime.tzinfo is None:
        timezone_name = os.getenv("AIRFLOW_INPUT_TIMEZONE", "Asia/Seoul")
        parsed_datetime = parsed_datetime.replace(tzinfo=ZoneInfo(timezone_name))
    parsed = parsed_datetime.timestamp()

if parsed < 0.0 or not math.isfinite(parsed):
    raise SystemExit(f"{name} must be finite and >= 0")

print(f"{parsed:.6f}")
' "${value}" "${name}"
}

PYTHON="$(resolve_python)"

if is_blank "${BASELINE_START}"; then
  echo "baseline_start is required" >&2
  exit 1
fi

if is_blank "${BASELINE_END}"; then
  echo "baseline_end is required" >&2
  exit 1
fi

BASELINE_START="$(normalize_epoch_time "${BASELINE_START}" "baseline_start")"
BASELINE_END="$(normalize_epoch_time "${BASELINE_END}" "baseline_end")"

CHAMPION_RUN_ID="$(
  "${PYTHON}" scripts/utility/resolve_mlflow_champion_run_id.py \
    --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
    --model-name "${MODEL_NAME}" \
    --model-alias "${MODEL_ALIAS}"
)"

THRESHOLD="$(
  "${PYTHON}" scripts/utility/resolve_mlflow_run_threshold.py \
    --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}" \
    --run-id "${CHAMPION_RUN_ID}"
)"

ARGS=(
  "${PYTHON}"
  scripts/monitoring/create_drift_reference_baseline.py
  --baseline-name "${BASELINE_NAME}"
  --model-run-id "${CHAMPION_RUN_ID}"
  --threshold "${THRESHOLD}"
  --source-start "${BASELINE_START}"
  --source-end "${BASELINE_END}"
  --min-samples "${MIN_SAMPLES}"
  --min-feature-non-null "${MIN_FEATURE_NON_NULL}"
  --min-feature-samples "${MIN_FEATURE_SAMPLES}"
  --psi-bin-count "${PSI_BIN_COUNT}"
  --min-psi-feature-non-null "${MIN_PSI_FEATURE_NON_NULL}"
  --notes "${NOTES}"
  --created-by "${CREATED_BY}"
)

if is_truthy "${RETIRE_EXISTING_ACTIVE}"; then
  ARGS+=(--retire-existing-active)
fi

if is_truthy "${DRY_RUN}"; then
  ARGS+=(--dry-run)
fi

echo "fixed_reference_baseline_create_command model_name=${MODEL_NAME} model_alias=${MODEL_ALIAS}"
echo "champion_run_id=${CHAMPION_RUN_ID} threshold=${THRESHOLD}"
echo "baseline_start=${BASELINE_START} baseline_end=${BASELINE_END} min_samples=${MIN_SAMPLES}"
echo "baseline_name=${BASELINE_NAME} retire_existing_active=${RETIRE_EXISTING_ACTIVE} dry_run=${DRY_RUN}"

"${ARGS[@]}"
