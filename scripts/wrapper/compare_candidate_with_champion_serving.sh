#!/usr/bin/env bash
set -euo pipefail

COHORT_START_TIME=""
CUTOFF_TIME=""
LABEL_MATURITY_SECONDS=""
TRACKING_URI=""
MODEL_NAME=""
CANDIDATE_ALIAS="candidate"
CHAMPION_ALIAS="champion"
CANDIDATE_VERSION=""
CHAMPION_VERSION=""
MAX_DECISIONS="1000"
MIN_DECISIONS="500"
MIN_LABEL_COVERAGE="0.95"
MIN_FAIL_SAMPLES="20"
MIN_PASS_SAMPLES="20"
PRIMARY_METRIC="fail_f1"
MIN_PRIMARY_DELTA="0.0"
MIN_RECALL_DELTA="-0.02"
MIN_PRECISION_DELTA="-0.05"
FAIL_ON_GATE_FAILURE="False"
DRY_RUN="True"
RECORD_DEPLOYMENT_REQUEST="False"
DEPLOYMENT_APPROVAL_STATUS="approved"
DEPLOYMENT_NOTES=""
DEPLOYMENT_REQUESTED_BY="${USER:-airflow}"
DEPLOYMENT_APPROVED_BY=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --cohort-start-time) COHORT_START_TIME="${2:-}"; shift 2 ;;
    --cutoff-time) CUTOFF_TIME="${2:-}"; shift 2 ;;
    --label-maturity-seconds) LABEL_MATURITY_SECONDS="${2:-}"; shift 2 ;;
    --tracking-uri) TRACKING_URI="${2:-}"; shift 2 ;;
    --model-name) MODEL_NAME="${2:-}"; shift 2 ;;
    --candidate-alias) CANDIDATE_ALIAS="${2:-}"; shift 2 ;;
    --champion-alias) CHAMPION_ALIAS="${2:-}"; shift 2 ;;
    --candidate-version) CANDIDATE_VERSION="${2:-}"; shift 2 ;;
    --champion-version) CHAMPION_VERSION="${2:-}"; shift 2 ;;
    --max-decisions|--limit) MAX_DECISIONS="${2:-}"; shift 2 ;;
    --min-decisions) MIN_DECISIONS="${2:-}"; shift 2 ;;
    --min-label-coverage) MIN_LABEL_COVERAGE="${2:-}"; shift 2 ;;
    --min-fail-samples) MIN_FAIL_SAMPLES="${2:-}"; shift 2 ;;
    --min-pass-samples) MIN_PASS_SAMPLES="${2:-}"; shift 2 ;;
    --primary-metric) PRIMARY_METRIC="${2:-}"; shift 2 ;;
    --min-primary-delta) MIN_PRIMARY_DELTA="${2:-}"; shift 2 ;;
    --min-recall-delta) MIN_RECALL_DELTA="${2:-}"; shift 2 ;;
    --min-precision-delta) MIN_PRECISION_DELTA="${2:-}"; shift 2 ;;
    --fail-on-gate-failure) FAIL_ON_GATE_FAILURE="${2:-True}"; shift 2 ;;
    --dry-run) DRY_RUN="${2:-True}"; shift 2 ;;
    --record-deployment-request) RECORD_DEPLOYMENT_REQUEST="${2:-True}"; shift 2 ;;
    --deployment-approval-status) DEPLOYMENT_APPROVAL_STATUS="${2:-approved}"; shift 2 ;;
    --deployment-notes) DEPLOYMENT_NOTES="${2:-}"; shift 2 ;;
    --deployment-requested-by) DEPLOYMENT_REQUESTED_BY="${2:-}"; shift 2 ;;
    --deployment-approved-by) DEPLOYMENT_APPROVED_BY="${2:-}"; shift 2 ;;
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

normalize_non_negative_seconds() {
  local value="$1"
  local name="$2"
  "${PYTHON}" -c '
import math
import sys

raw_value = sys.argv[1].strip()
name = sys.argv[2]

try:
    parsed = float(raw_value)
except ValueError as exc:
    raise SystemExit(f"{name} must be numeric seconds: {raw_value}") from exc

if parsed < 0.0 or not math.isfinite(parsed):
    raise SystemExit(f"{name} must be finite and >= 0")

print(f"{parsed:.6f}")
' "${value}" "${name}"
}

normalize_model_version() {
  local value="$1"
  if is_blank "${value}"; then
    printf "\n"
    return
  fi

  case "${value}" in
    v[0-9]*|V[0-9]*)
      printf "%s\n" "${value#?}"
      ;;
    *)
      printf "%s\n" "${value}"
      ;;
  esac
}

PYTHON="$(resolve_python)"

if is_truthy "${DRY_RUN}"; then
  echo "candidate_champion_serving_compare_skipped reason=dry_run"
  exit 0
fi

if is_blank "${COHORT_START_TIME}"; then
  echo "cohort_start_time is required" >&2
  exit 1
fi

if is_blank "${CUTOFF_TIME}"; then
  echo "cutoff_time is required" >&2
  exit 1
fi

if is_blank "${LABEL_MATURITY_SECONDS}"; then
  echo "label_maturity_seconds is required" >&2
  exit 1
fi

COHORT_START_TIME="$(normalize_epoch_time "${COHORT_START_TIME}" "cohort_start_time")"
CUTOFF_TIME="$(normalize_epoch_time "${CUTOFF_TIME}" "cutoff_time")"
LABEL_MATURITY_SECONDS="$(normalize_non_negative_seconds "${LABEL_MATURITY_SECONDS}" "label_maturity_seconds")"
CANDIDATE_VERSION="$(normalize_model_version "${CANDIDATE_VERSION}")"
CHAMPION_VERSION="$(normalize_model_version "${CHAMPION_VERSION}")"

PYTHON="$(resolve_python)"

COMMAND=(
  "${PYTHON}"
  scripts/monitoring/compare_candidate_with_champion_serving.py
  --cohort-start-time "${COHORT_START_TIME}"
  --cutoff-time "${CUTOFF_TIME}"
  --label-maturity-seconds "${LABEL_MATURITY_SECONDS}"
  --candidate-alias "${CANDIDATE_ALIAS}"
  --champion-alias "${CHAMPION_ALIAS}"
  --max-decisions "${MAX_DECISIONS}"
  --min-decisions "${MIN_DECISIONS}"
  --min-label-coverage "${MIN_LABEL_COVERAGE}"
  --min-fail-samples "${MIN_FAIL_SAMPLES}"
  --min-pass-samples "${MIN_PASS_SAMPLES}"
  --primary-metric "${PRIMARY_METRIC}"
  --min-primary-delta "${MIN_PRIMARY_DELTA}"
  --min-recall-delta "${MIN_RECALL_DELTA}"
  --min-precision-delta "${MIN_PRECISION_DELTA}"
  --set-tags
)

if ! is_blank "${TRACKING_URI}"; then
  COMMAND+=(--tracking-uri "${TRACKING_URI}")
fi

if ! is_blank "${MODEL_NAME}"; then
  COMMAND+=(--model-name "${MODEL_NAME}")
fi

if ! is_blank "${CANDIDATE_VERSION}"; then
  COMMAND+=(--candidate-version "${CANDIDATE_VERSION}")
fi

if ! is_blank "${CHAMPION_VERSION}"; then
  COMMAND+=(--champion-version "${CHAMPION_VERSION}")
fi

if is_truthy "${FAIL_ON_GATE_FAILURE}"; then
  COMMAND+=(--fail-on-gate-failure)
fi

if is_truthy "${RECORD_DEPLOYMENT_REQUEST}"; then
  COMMAND+=(
    --record-deployment-request
    --deployment-approval-status "${DEPLOYMENT_APPROVAL_STATUS}"
  )

  if ! is_blank "${DEPLOYMENT_NOTES}"; then
    COMMAND+=(--deployment-notes "${DEPLOYMENT_NOTES}")
  fi

  if ! is_blank "${DEPLOYMENT_REQUESTED_BY}"; then
    COMMAND+=(--deployment-requested-by "${DEPLOYMENT_REQUESTED_BY}")
  fi

  if ! is_blank "${DEPLOYMENT_APPROVED_BY}"; then
    COMMAND+=(--deployment-approved-by "${DEPLOYMENT_APPROVED_BY}")
  fi
fi

echo "candidate_champion_serving_compare_command cohort_start_time=${COHORT_START_TIME} cutoff_time=${CUTOFF_TIME} label_maturity_seconds=${LABEL_MATURITY_SECONDS} max_decisions=${MAX_DECISIONS} candidate_alias=${CANDIDATE_ALIAS} champion_alias=${CHAMPION_ALIAS} dry_run=${DRY_RUN}"

"${COMMAND[@]}"
