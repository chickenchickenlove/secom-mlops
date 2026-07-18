#!/usr/bin/env bash
set -euo pipefail

DATASET_ID=""
TRACKING_URI=""
MODEL_NAME=""
CANDIDATE_ALIAS="candidate"
CHAMPION_ALIAS="champion"
CANDIDATE_VERSION=""
CHAMPION_VERSION=""
PRIMARY_METRIC="fail_f1"
MIN_PRIMARY_DELTA="0.0"
MIN_RECALL_DELTA="-0.02"
MIN_PRECISION_DELTA="-0.05"
FAIL_ON_GATE_FAILURE="False"
DRY_RUN="True"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset-id) DATASET_ID="${2:-}"; shift 2 ;;
    --tracking-uri) TRACKING_URI="${2:-}"; shift 2 ;;
    --model-name) MODEL_NAME="${2:-}"; shift 2 ;;
    --candidate-alias) CANDIDATE_ALIAS="${2:-}"; shift 2 ;;
    --champion-alias) CHAMPION_ALIAS="${2:-}"; shift 2 ;;
    --candidate-version) CANDIDATE_VERSION="${2:-}"; shift 2 ;;
    --champion-version) CHAMPION_VERSION="${2:-}"; shift 2 ;;
    --primary-metric) PRIMARY_METRIC="${2:-}"; shift 2 ;;
    --min-primary-delta) MIN_PRIMARY_DELTA="${2:-}"; shift 2 ;;
    --min-recall-delta) MIN_RECALL_DELTA="${2:-}"; shift 2 ;;
    --min-precision-delta) MIN_PRECISION_DELTA="${2:-}"; shift 2 ;;
    --fail-on-gate-failure) FAIL_ON_GATE_FAILURE="${2:-True}"; shift 2 ;;
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

normalize_model_version() {
  local value="$1"
  if is_blank "${value}"; then
    printf "\n"
    return
  fi
  case "${value}" in
    v[0-9]*|V[0-9]*) printf "%s\n" "${value#?}" ;;
    *) printf "%s\n" "${value}" ;;
  esac
}

if is_truthy "${DRY_RUN}"; then
  echo "candidate_champion_serving_compare_skipped reason=dry_run"
  exit 0
fi
if is_blank "${DATASET_ID}"; then
  echo "dataset_id is required" >&2
  exit 1
fi

PYTHON="$(resolve_python)"
CANDIDATE_VERSION="$(normalize_model_version "${CANDIDATE_VERSION}")"
CHAMPION_VERSION="$(normalize_model_version "${CHAMPION_VERSION}")"

COMMAND=(
  "${PYTHON}"
  scripts/monitoring/compare_candidate_with_champion_serving.py
  --dataset-id "${DATASET_ID}"
  --candidate-alias "${CANDIDATE_ALIAS}"
  --champion-alias "${CHAMPION_ALIAS}"
  --primary-metric "${PRIMARY_METRIC}"
  --min-primary-delta "${MIN_PRIMARY_DELTA}"
  --min-recall-delta "${MIN_RECALL_DELTA}"
  --min-precision-delta "${MIN_PRECISION_DELTA}"
  --set-tags
)

if ! is_blank "${TRACKING_URI}"; then COMMAND+=(--tracking-uri "${TRACKING_URI}"); fi
if ! is_blank "${MODEL_NAME}"; then COMMAND+=(--model-name "${MODEL_NAME}"); fi
if ! is_blank "${CANDIDATE_VERSION}"; then COMMAND+=(--candidate-version "${CANDIDATE_VERSION}"); fi
if ! is_blank "${CHAMPION_VERSION}"; then COMMAND+=(--champion-version "${CHAMPION_VERSION}"); fi
if is_truthy "${FAIL_ON_GATE_FAILURE}"; then COMMAND+=(--fail-on-gate-failure); fi

echo "candidate_champion_serving_compare_command dataset_id=${DATASET_ID} candidate_alias=${CANDIDATE_ALIAS} champion_alias=${CHAMPION_ALIAS} dry_run=${DRY_RUN}"
"${COMMAND[@]}"
