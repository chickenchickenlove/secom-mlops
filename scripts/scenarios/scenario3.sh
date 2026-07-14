#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

pids=()

kill_tree() {
  local pid="$1"
  local signal="${2:-TERM}"
  local child

  if ! kill -0 "$pid" 2>/dev/null; then
    return
  fi

  while read -r child; do
    if [ -n "$child" ]; then
      kill_tree "$child" "$signal"
    fi
  done < <(pgrep -P "$pid" 2>/dev/null || true)
  kill "-$signal" "$pid" 2>/dev/null || true
}

cleanup() {
  trap - EXIT INT TERM

  echo
  echo "Stopping scenario2 background processes..."

  for pid in "${pids[@]}"; do
    kill_tree "$pid" TERM
  done

  sleep 2

  for pid in "${pids[@]}"; do
    kill_tree "$pid" KILL
  done

  wait 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# Reset watermark.
mkdir -p ./runtime
printf '{"next_feature_index": 0}\n' > ./runtime/online_workload_next_feature_early_state.json
printf '{"next_feature_index": 0}\n' > ./runtime/online_workload_next_feature_middle_state.json
printf '{"next_feature_index": 0}\n' > ./runtime/online_workload_next_feature_late_state.json
printf '{"next_label_index": 0}\n' > ./runtime/online_workload_next_label_state.json
printf '{"next_predict_index": 0}\n' > ./runtime/online_workload_next_predict_state.json

send_features() {
  local group="$1"
  local drift_segment="$2"

  uv run python scripts/workload/send_feature_events_from_cursor.py \
    --feature-group "${group}" \
    --max-samples 6000 \
    --batch-size 200 \
    --sleep-seconds 5

  uv run python scripts/workload/send_feature_events_from_cursor.py \
    --feature-group "${group}" \
    --max-samples 60000 \
    --batch-size 200 \
    --sleep-seconds 5 \
    --drift-segment "${drift_segment}" \
    --feature-offset-action "59,+,100.0,1" \
    --feature-offset-action "103,+,0.2,1" \
    --feature-offset-action "33,+,10.0,1" \
    --feature-offset-action "31,+,5.0,1" \
    --feature-offset-action "477,+,80.0,1"

  uv run python scripts/workload/send_feature_events_from_cursor.py \
    --feature-group "${group}" \
    --max-samples 6000 \
    --batch-size 200 \
    --sleep-seconds 5
}

send_labels() {
  sleep 60
  uv run python scripts/workload/send_label_events_from_cursor.py \
    --max-samples 72000 \
    --batch-size 200 \
    --sleep-seconds 5
}

send_predicts() {
  sleep 30
  uv run python scripts/workload/request_predictions_from_cursor.py \
    --max-samples 72000 \
    --batch-size 200 \
    --sleep-seconds 9 \
    --concurrency 100 \
    --print-failures
}

send_features early "scenario3" &
pids+=("$!")

send_features middle "scenario3" &
pids+=("$!")

send_features late "scenario3" &
pids+=("$!")

send_labels &
pids+=("$!")

send_predicts &
pids+=("$!")

echo "Started scenario3 background processes:"
echo "  early features pid: ${pids[0]}"
echo "  middle features pid: ${pids[1]}"
echo "  late features pid: ${pids[2]}"
echo "  labels   pid: ${pids[3]}"
echo "  predicts pid: ${pids[4]}"
echo
echo "Press Ctrl+C to stop all."

wait
