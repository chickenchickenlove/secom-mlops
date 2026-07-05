#!/usr/bin/env bash
set -euo pipefail

POINT_TIME=""
RECENT_MINUTES="10"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --point-time)
      POINT_TIME="${2:-}"
      shift 2
      ;;
    --recent-minutes)
      RECENT_MINUTES="${2:-}"
      shift 2
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

if is_blank "${POINT_TIME}"; then
  echo "point_time is required" >&2
  exit 2
fi

PYTHON="$(resolve_python)"
POINT_TIME="$(normalize_epoch_time "${POINT_TIME}" "point_time")"

"${PYTHON}" -c '
import math
import sys

point_time = float(sys.argv[1])
recent_minutes = float(sys.argv[2])

if point_time < 0.0 or not math.isfinite(point_time):
    raise SystemExit("point_time must be finite and >= 0")
if recent_minutes <= 0.0 or not math.isfinite(recent_minutes):
    raise SystemExit("recent_minutes must be finite and > 0")

point_time_start = point_time - recent_minutes * 60.0
if point_time_start < 0.0:
    raise SystemExit("resolved point_time_start is negative")

print(f"{point_time_start:.6f}")
' "${POINT_TIME}" "${RECENT_MINUTES}"
