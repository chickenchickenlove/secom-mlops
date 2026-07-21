#!/usr/bin/env bash
set -euo pipefail

TRAINER_SCRIPT=""
DATASET_ID=""
TRACKING_URI=""
MODEL_NAME=""
MODEL_ALIAS=""
MODEL_ROLE=""
CANDIDATE_GROUP=""
TRAINING_JOB_ID=""
MIN_LABEL_COVERAGE="0.95"
MIN_FAIL_SAMPLES="20"
MIN_PASS_SAMPLES="20"
RANDOM_STATE="42"
N_ESTIMATORS="100,300,500,700"
MIN_SAMPLES_LEAF="1,3,5,7"
THRESHOLDS="0.1,0.2,0.3,0.4,0.5"
DRY_RUN="False"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --trainer-script) TRAINER_SCRIPT="${2:-}"; shift 2 ;;
    --dataset-id) DATASET_ID="${2:-}"; shift 2 ;;
    --tracking-uri) TRACKING_URI="${2:-}"; shift 2 ;;
    --model-name) MODEL_NAME="${2:-}"; shift 2 ;;
    --model-alias) MODEL_ALIAS="${2:-}"; shift 2 ;;
    --model-role) MODEL_ROLE="${2:-}"; shift 2 ;;
    --candidate-group) CANDIDATE_GROUP="${2:-}"; shift 2 ;;
    --training-job-id) TRAINING_JOB_ID="${2:-}"; shift 2 ;;
    --min-label-coverage) MIN_LABEL_COVERAGE="${2:-}"; shift 2 ;;
    --min-fail-samples) MIN_FAIL_SAMPLES="${2:-}"; shift 2 ;;
    --min-pass-samples) MIN_PASS_SAMPLES="${2:-}"; shift 2 ;;
    --random-state) RANDOM_STATE="${2:-}"; shift 2 ;;
    --n-estimators) N_ESTIMATORS="${2:-}"; shift 2 ;;
    --min-samples-leaf) MIN_SAMPLES_LEAF="${2:-}"; shift 2 ;;
    --thresholds) THRESHOLDS="${2:-}"; shift 2 ;;
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

if is_blank "${DATASET_ID}"; then
  echo "dataset_id is required" >&2
  exit 1
fi

RUN_TOKEN=""
if is_blank "${CANDIDATE_GROUP}" || is_blank "${TRAINING_JOB_ID}"; then
  RUN_TOKEN="$("${PYTHON}" -c '
from datetime import datetime, timezone
from uuid import uuid4

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
print(f"{timestamp}_{uuid4().hex[:8]}")
')"
fi

if is_blank "${CANDIDATE_GROUP}"; then
  CANDIDATE_GROUP="retrain_${RUN_TOKEN}"
fi

if is_blank "${TRAINING_JOB_ID}"; then
  TRAINING_JOB_ID="train_${RUN_TOKEN}"
fi

TRAINER_SCRIPT="$(resolve_trainer_script)"

COMMAND=(
  "${PYTHON}"
  "${TRAINER_SCRIPT}"
  --dataset-id "${DATASET_ID}"
  --candidate-group "${CANDIDATE_GROUP}"
  --training-job-id "${TRAINING_JOB_ID}"
  --min-label-coverage "${MIN_LABEL_COVERAGE}"
  --min-fail-samples "${MIN_FAIL_SAMPLES}"
  --min-pass-samples "${MIN_PASS_SAMPLES}"
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

if is_truthy "${DRY_RUN}"; then
  COMMAND+=(--dry-run)
fi

echo "training_dataset_candidate_retraining_command trainer_script=${TRAINER_SCRIPT} dataset_id=${DATASET_ID} candidate_group=${CANDIDATE_GROUP} training_job_id=${TRAINING_JOB_ID}"

"${COMMAND[@]}"
