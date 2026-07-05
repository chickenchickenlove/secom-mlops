#!/usr/bin/env bash
set -euo pipefail

TRAINER_SCRIPT=""
POINT_TIME_START=""
POINT_TIME=""
TRACKING_URI=""
MODEL_NAME=""
MODEL_ALIAS=""
MODEL_ROLE=""
CANDIDATE_GROUP=""
TRAINING_JOB_ID=""
SIMULATION_RUN_ID=""
DRIFT_SEGMENT=""
MIN_SAMPLES="500"
MIN_FAIL_SAMPLES="20"
MIN_PASS_SAMPLES="20"
TEST_SIZE="0.2"
RANDOM_STATE="42"
N_ESTIMATORS="100,300,500,700"
MIN_SAMPLES_LEAF="1,3,5,7"
THRESHOLDS="0.1,0.2,0.3,0.4,0.5"
REFIT_ON_ALL_DATA="False"
DRY_RUN="False"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --trainer-script) TRAINER_SCRIPT="${2:-}"; shift 2 ;;
    --point-time-start) POINT_TIME_START="${2:-}"; shift 2 ;;
    --point-time) POINT_TIME="${2:-}"; shift 2 ;;
    --tracking-uri) TRACKING_URI="${2:-}"; shift 2 ;;
    --model-name) MODEL_NAME="${2:-}"; shift 2 ;;
    --model-alias) MODEL_ALIAS="${2:-}"; shift 2 ;;
    --model-role) MODEL_ROLE="${2:-}"; shift 2 ;;
    --candidate-group) CANDIDATE_GROUP="${2:-}"; shift 2 ;;
    --training-job-id) TRAINING_JOB_ID="${2:-}"; shift 2 ;;
    --simulation-run-id) SIMULATION_RUN_ID="${2:-}"; shift 2 ;;
    --drift-segment) DRIFT_SEGMENT="${2:-}"; shift 2 ;;
    --min-samples) MIN_SAMPLES="${2:-}"; shift 2 ;;
    --min-fail-samples) MIN_FAIL_SAMPLES="${2:-}"; shift 2 ;;
    --min-pass-samples) MIN_PASS_SAMPLES="${2:-}"; shift 2 ;;
    --test-size) TEST_SIZE="${2:-}"; shift 2 ;;
    --random-state) RANDOM_STATE="${2:-}"; shift 2 ;;
    --n-estimators) N_ESTIMATORS="${2:-}"; shift 2 ;;
    --min-samples-leaf) MIN_SAMPLES_LEAF="${2:-}"; shift 2 ;;
    --thresholds) THRESHOLDS="${2:-}"; shift 2 ;;
    --refit-on-all-data)
      if [ "$#" -ge 2 ] && [ "${2#--}" = "$2" ]; then
        REFIT_ON_ALL_DATA="${2}"
        shift 2
      else
        REFIT_ON_ALL_DATA="True"
        shift 1
      fi
      ;;
    --dry-run)
      if [ "$#" -ge 2 ] && [ "${2#--}" = "$2" ]; then
        DRY_RUN="${2}"
        shift 2
      else
        DRY_RUN="True"
        shift 1
      fi
      ;;
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

resolve_trainer_script() {
  if ! is_blank "${TRAINER_SCRIPT}"; then
    if [ -f "${TRAINER_SCRIPT}" ]; then
      printf "%s\n" "${TRAINER_SCRIPT}"
      return
    fi
    echo "trainer script does not exist: ${TRAINER_SCRIPT}" >&2
    exit 1
  fi

  for candidate in \
    "scripts/training/train_candidate_from_offline_point_in_time_features.py"
  do
    if [ -f "${candidate}" ]; then
      printf "%s\n" "${candidate}"
      return
    fi
  done

  echo "trainer script was not found" >&2
  exit 1
}

PYTHON="$(resolve_python)"

if is_blank "${POINT_TIME_START}"; then
  echo "point_time_start is required" >&2
  exit 1
fi

if is_blank "${POINT_TIME}"; then
  echo "point_time is required" >&2
  exit 1
fi

POINT_TIME_START="$(normalize_epoch_time "${POINT_TIME_START}" "point_time_start")"
POINT_TIME="$(normalize_epoch_time "${POINT_TIME}" "point_time")"

"${PYTHON}" -c '
import math
import sys

point_time_start = float(sys.argv[1])
point_time = float(sys.argv[2])

if point_time_start < 0.0 or not math.isfinite(point_time_start):
    raise SystemExit("point_time_start must be finite and >= 0")
if point_time <= point_time_start or not math.isfinite(point_time):
    raise SystemExit("point_time must be finite and greater than point_time_start")
' "${POINT_TIME_START}" "${POINT_TIME}"

if is_blank "${CANDIDATE_GROUP}"; then
  CANDIDATE_GROUP="$("${PYTHON}" -c '
from datetime import datetime, timezone
import sys

point_time = float(sys.argv[1])
suffix = datetime.fromtimestamp(point_time, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
print(f"airflow_retrain_{suffix}")
' "${POINT_TIME}")"
fi

if is_blank "${TRAINING_JOB_ID}"; then
  TRAINING_JOB_ID="${CANDIDATE_GROUP}"
fi

TRAINER_SCRIPT="$(resolve_trainer_script)"

COMMAND=(
  "${PYTHON}"
  "${TRAINER_SCRIPT}"
  --point-time-start "${POINT_TIME_START}"
  --point-time "${POINT_TIME}"
  --candidate-group "${CANDIDATE_GROUP}"
  --training-job-id "${TRAINING_JOB_ID}"
  --min-samples "${MIN_SAMPLES}"
  --min-fail-samples "${MIN_FAIL_SAMPLES}"
  --min-pass-samples "${MIN_PASS_SAMPLES}"
  --test-size "${TEST_SIZE}"
  --random-state "${RANDOM_STATE}"
  --n-estimators "${N_ESTIMATORS}"
  --min-samples-leaf "${MIN_SAMPLES_LEAF}"
  --thresholds "${THRESHOLDS}"
)

append_optional() {
  local flag="$1"
  local value="$2"
  if ! is_blank "${value}"; then
    COMMAND+=("${flag}" "${value}")
  fi
}

append_optional --tracking-uri "${TRACKING_URI}"
append_optional --model-name "${MODEL_NAME}"
append_optional --model-alias "${MODEL_ALIAS}"
append_optional --model-role "${MODEL_ROLE}"
append_optional --simulation-run-id "${SIMULATION_RUN_ID}"
append_optional --drift-segment "${DRIFT_SEGMENT}"

if is_truthy "${REFIT_ON_ALL_DATA}"; then
  COMMAND+=(--refit-on-all-data)
fi

if is_truthy "${DRY_RUN}"; then
  COMMAND+=(--dry-run)
fi

echo "offline_point_time_candidate_retraining_command trainer_script=${TRAINER_SCRIPT} point_time_start=${POINT_TIME_START} point_time=${POINT_TIME} candidate_group=${CANDIDATE_GROUP} training_job_id=${TRAINING_JOB_ID}"

"${COMMAND[@]}"
