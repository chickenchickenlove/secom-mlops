# Local Monitoring Stack

This stack runs the local SECOM/FDC online path, offline/raw store sync, model serving, and Grafana monitoring.

## Components

- PostgreSQL stores `feature_events`, `actual_labels`, `prediction_logs`, offline snapshots, and monitoring metrics.
- Kafka carries feature patch, feature state update, label event, and prediction event topics.
- Prometheus scrapes Kafka broker JMX metrics exposed by the JMX exporter.
- Valkey stores the latest online feature snapshot per `sample_id`.
- MLflow stores experiments, model registry metadata, and model artifacts.
- Grafana reads PostgreSQL through the provisioned `Monitoring Postgres` datasource and Kafka broker metrics through the `Prometheus` datasource.
- Java daemons materialize Kafka streams into PostgreSQL and Valkey.
- Python scripts generate workload events, prediction requests, and monitoring metrics.

## Runtime Flow

```text
Feature online path:
secom-feature-patches
-> feature-assembler
-> secom-feature-state-updates
-> feature-materializer
-> Valkey online_feature_snapshot:{sample_id}
-> serving-api /predict-by-id
-> model-gateway
-> model-server-release
-> secom-prediction-events
-> prediction-log-archiver
-> prediction_logs

Feature offline/raw store:
secom-feature-patches
-> feature-raw-archiver
-> PostgreSQL feature_events

Label evaluation store:
secom-label-events
-> label-archiver
-> PostgreSQL actual_labels

Prediction evidence store:
serving-api
-> secom-prediction-events
-> prediction-log-archiver
-> PostgreSQL prediction_logs

Metrics:
prediction_logs + actual_labels
-> metrics-evaluator
-> model_metrics / prediction_window_metrics
-> Grafana

Quality windows:
prediction_logs + actual_labels
-> quality-window-evaluator
-> model_quality_windows
-> Grafana

Drift metrics:
prediction_logs
-> drift-metrics-evaluator
-> drift_metrics
-> Grafana
```

## Start

Run from this directory.

```bash
docker-compose up -d --build
docker-compose ps
```

Use `down -v` only when you intentionally want to delete PostgreSQL, Kafka, Valkey, Grafana, and MLflow local volumes.

```bash
docker-compose down
docker-compose down -v
```

## Kafka Metrics

The Kafka container is built with the Prometheus JMX exporter javaagent and
exposes broker metrics on `kafka:7071`. The stack also runs Kafka Exporter on
`kafka-exporter:9308` for consumer group lag metrics.

Prometheus scrapes both endpoints and is available locally at:

```text
http://localhost:9090
```

Grafana provisions the Prometheus datasource with uid `prometheus`.

Useful starter PromQL queries for broker/topic throughput:

```promql
rate(kafka_server_brokertopicmetrics_messagesinpersec_total[1m])
rate(kafka_server_brokertopicmetrics_bytesinpersec_total[1m])
rate(kafka_server_brokertopicmetrics_bytesoutpersec_total[1m])
kafka_server_replicamanager_underreplicatedpartitions
kafka_controller_kafkacontroller_offlinepartitionscount
```

Useful starter PromQL queries for consumer lag:

```promql
sum by (consumergroup, topic) (kafka_consumergroup_lag)
max by (consumergroup, topic) (kafka_consumergroup_lag)
sum by (consumergroup) (kafka_consumergroup_lag)
```

## Services

Core platform services:

```text
postgres
kafka
kafka-init
valkey
grafana
mlflow
model-trainer
model-server-release
model-gateway
serving-api
metrics-evaluator
quality-window-evaluator
drift-metrics-evaluator
```

Stream daemons:

```text
feature-raw-archiver
feature-assembler
feature-materializer
label-archiver
prediction-log-archiver
```

## Kafka Topics

```text
secom-feature-patches
secom-feature-state-updates
secom-label-events
secom-prediction-events
```

Host clients usually use:

```text
127.0.0.1:9092
```

Compose services use:

```text
kafka:29092
```

## Official Workload Scripts

Run these from the project root.

Feature event workload:

```bash
uv run python scripts/workload/send_feature_events_from_cursor.py \
  --feature-group early \
  --max-samples 30000 \
  --batch-size 300 \
  --sleep-seconds 10
```

Label event workload:

```bash
uv run python scripts/workload/send_label_events_from_cursor.py \
  --max-samples 30000 \
  --batch-size 300 \
  --sleep-seconds 10 \
  --label-delay-seconds 300
```

Prediction request workload:

```bash
uv run python scripts/workload/request_predictions_from_cursor.py \
  --max-samples 30000 \
  --batch-size 300 \
  --sleep-seconds 15 \
  --concurrency 16 \
  --print-failures
```

The cursor state file is:

```text
runtime/online_workload_state.json
```

## Verify Stores

Feature offline/raw store:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  COUNT(*) AS feature_event_count,
  COUNT(DISTINCT sample_id) AS feature_sample_count,
  MIN(sample_id) AS first_sample_id,
  MAX(sample_id) AS last_sample_id
FROM feature_events;
"
```

Complete feature samples:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  e.sample_id,
  COUNT(*) AS patch_count,
  array_agg(e.feature_group ORDER BY e.feature_group) AS feature_groups,
  SUM((
    SELECT COUNT(*)
    FROM jsonb_object_keys(e.features_json)
  )) AS observed_feature_count
FROM feature_events e
GROUP BY e.sample_id
ORDER BY e.sample_id DESC
LIMIT 20;
"
```

Label store:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  COUNT(*) AS label_count,
  COUNT(*) FILTER (WHERE actual_label = 'fail') AS fail_count,
  COUNT(*) FILTER (WHERE actual_label = 'pass') AS pass_count,
  MIN(sample_id) AS first_sample_id,
  MAX(sample_id) AS last_sample_id
FROM actual_labels;
"
```

Feature-label coverage:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
WITH complete_feature_samples AS (
  SELECT sample_id
  FROM feature_events
  GROUP BY sample_id
  HAVING COUNT(*) = 3
     AND SUM((SELECT COUNT(*) FROM jsonb_object_keys(features_json))) = 590
)
SELECT
  COUNT(*) AS complete_feature_samples,
  COUNT(*) FILTER (WHERE a.sample_id IS NOT NULL) AS samples_with_label,
  COUNT(*) FILTER (WHERE a.sample_id IS NULL) AS samples_without_label,
  MIN(f.sample_id) AS first_feature_sample_id,
  MAX(f.sample_id) AS last_feature_sample_id
FROM complete_feature_samples f
LEFT JOIN actual_labels a ON a.sample_id = f.sample_id;
"
```

## Verify Online Predictions

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  COUNT(*) AS prediction_count,
  COUNT(*) FILTER (WHERE predicted_label = 'fail') AS predicted_fail_count,
  COUNT(*) FILTER (WHERE predicted_label = 'pass') AS predicted_pass_count,
  MIN(sample_id) AS first_sample_id,
  MAX(sample_id) AS last_sample_id
FROM prediction_logs;
"
```

## Model Retraining / Deployment Workflow

This local workflow uses MLflow model registry aliases as deployment control
pointers.

```text
candidate
  next model version under review

champion
  approved model version for serving
```

The model server loads the configured MLflow model only at startup. After
promoting a new `champion`, recreate `model-server-release` so the serving process
loads the promoted model version.

### End-To-End Runbook

Run from `container` unless a command says otherwise.

Start the stack:

```bash
docker-compose up -d --build
docker-compose ps
```

Candidate retraining now uses on-demand point-in-time reconstruction from
`feature_events`. Materialized `offline_feature_snapshots` utilities remain
available for smoke tests/backfills, but they are not required for the
operational retraining DAG.

Trigger the Airflow training DAG:

```text
train_candidate_from_offline_point_in_time_features
```

Then trigger the serving snapshot gate DAG on the intended independent gate
window:

```text
evaluate_candidate_serving_snapshot_gate
```

Both DAGs take a required `point_time` datetime and a `recent_minutes` window.

For direct debugging from `container`, run the current trainer:

```bash
docker-compose run --rm \
  -e MONITORING_DATABASE_URL=postgresql://mlops:mlops@postgres:5432/monitoring \
  -e MLFLOW_TRACKING_URI=http://mlflow:5100 \
  model-trainer \
  python scripts/training/train_candidate_from_offline_point_in_time_features.py \
    --point-time-start 9999999399 \
    --point-time 9999999999 \
    --candidate-group retrain_20260701_001 \
    --training-job-id retrain_20260701_001 \
    --min-samples 500 \
    --min-fail-samples 20 \
    --min-pass-samples 20
```

The required hard gate is serving snapshot evaluation. The retraining DAG runs
this as its final task:

```bash
docker-compose run --rm \
  -e MONITORING_DATABASE_URL=postgresql://mlops:mlops@postgres:5432/monitoring \
  model-trainer \
  python scripts/monitoring/compare_candidate_with_champion_serving.py \
    --point-time-start 9999999399 \
    --point-time 9999999999 \
    --set-tags
```

Promote only after the approved request exists:

```bash
docker-compose run --rm \
  -e MONITORING_DATABASE_URL=postgresql://mlops:mlops@postgres:5432/monitoring \
  model-trainer \
  python scripts/deployment/promote_model_alias.py \
    --source-alias candidate \
    --target-alias champion \
    --require-approved-request
```

Check the registry/request state:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  request_id,
  source_version,
  target_alias,
  gate_status,
  approval_status,
  to_timestamp(requested_at) AS requested_at,
  to_timestamp(deployed_at) AS deployed_at
FROM model_deployment_requests
ORDER BY requested_at DESC
LIMIT 5;
"
```

If you need the online serving process to load the promoted alias, recreate the
model server and verify `/metadata`:

```bash
docker-compose up -d --no-deps --force-recreate model-server-release
curl http://127.0.0.1:28091/metadata
```

### Train Candidate From Offline Point-in-Time Features

Run from `container`.

```bash
docker-compose run --rm \
  -e MONITORING_DATABASE_URL=postgresql://mlops:mlops@postgres:5432/monitoring \
  -e MLFLOW_TRACKING_URI=http://mlflow:5100 \
  model-trainer \
  python scripts/training/train_candidate_from_offline_point_in_time_features.py \
    --point-time-start 9999999399 \
    --point-time 9999999999 \
    --candidate-group retrain_20260701_001 \
    --training-job-id retrain_20260701_001 \
    --min-samples 500 \
    --min-fail-samples 20 \
    --min-pass-samples 20
```

Candidate training requires:

```text
ML_CANDIDATE_GROUP
ML_TRAINING_JOB_ID
```

The point-in-time trainer uses `serving_feature_snapshots` as the sample/time
spine, reconstructs features from `feature_events`, joins labels by
`sample_point_time <= labeled_at`, trains a RandomForest
candidate, registers a new `secom-fail-detector` model version, points the
`candidate` alias to it, and writes model version tags such as:

```text
role=candidate
candidate_group=...
training_job_id=...
train_source=offline_feature_store_point_in_time
training_spine=serving_feature_snapshots
training_point_time=serving_feature_snapshots.snapshot_time
point_time_start=...
point_time=...
gate_status=pending
```

The older default `model-trainer` command still trains from raw SECOM CSV files
and is useful for bootstrap/champion seeding. For retraining from operational
data, prefer `scripts/training/train_candidate_from_offline_point_in_time_features.py`.

### Airflow Candidate Retraining DAGs

The DAG `train_candidate_from_offline_point_in_time_features` trains and registers a
candidate from offline point-in-time reconstructed features.

Trigger it manually from the Airflow UI, or with config such as:

```json
{
  "point_time": "2026-07-05T13:00:00+09:00",
  "recent_minutes": 10,
  "min_samples": 500,
  "min_fail_samples": 20,
  "min_pass_samples": 20
}
```

After training succeeds, run `evaluate_candidate_serving_snapshot_gate` on the
chosen serving snapshot gate window. If the gate fails, run
`cleanup_failed_candidate_alias`. If the gate passes, create or approve the
deployment request before canary/release deployment.

### Fixed Reference Drift Baseline

The DAG `create_fixed_reference_drift_baseline` creates an active fixed
reference baseline for the current MLflow champion from a selected
`prediction_logs` window.

Run it manually before `fixed_reference_drift_evaluator` is expected to produce
fixed-reference drift rows. Use a stable champion traffic window:

```json
{
  "baseline_start": "2026-07-05T17:45:00+09:00",
  "baseline_end": "2026-07-05T17:55:00+09:00",
  "baseline_name": "fixed-reference-champion-manual",
  "min_samples": 500,
  "retire_existing_active": true,
  "dry_run": false
}
```

### Optional Fast Sanity Check

Fast comparison uses MLflow training-run metrics. It is useful as a cheap
candidate sanity check, but it is not the required promotion gate because it
does not re-evaluate candidate and champion on the same offline dataset.

```bash
docker-compose run --rm model-trainer \
  python scripts/monitoring/compare_candidate_with_champion.py \
    --set-tags
```

The comparison reads MLflow run metrics for the `candidate` and `champion`
aliases and checks the configured gate.

Default metrics:

```text
f1_1
recall_1
precision_1
pr_auc
balanced_accuracy
accuracy
```

A stricter example:

```bash
docker-compose run --rm model-trainer \
  python scripts/monitoring/compare_candidate_with_champion.py \
    --min-primary-delta 0.01 \
    --min-recall-delta 0.0 \
    --min-precision-delta -0.02 \
    --set-tags
```

### Required Offline Candidate Gate

Offline comparison using the same labeled snapshot window for both models:

```bash
docker-compose run --rm \
  -e MONITORING_DATABASE_URL=postgresql://mlops:mlops@postgres:5432/monitoring \
  model-trainer \
  python scripts/monitoring/compare_candidate_with_champion_offline.py \
    --build-cutoff-time 9999999999 \
    --limit 10000 \
    --set-tags \
    --record-deployment-request \
    --deployment-approval-status approved \
    --deployment-notes "offline gate passed; ready for manual promote"
```

Use `--limit 0` to evaluate all labeled snapshots for the cutoff.

When `--record-deployment-request` is set, a passed offline gate writes a
`model_deployment_requests` row with the source version, previous champion
version, gate status, and metric summary. If the gate fails, the deployment
request is skipped and `candidate` should be cleared.

`scripts/deployment/record_model_deployment.py` remains available for manual backfill or
exceptional records.

### Promote Candidate To Champion

Preferred path after a passed gate:

```bash
docker-compose run --rm model-trainer \
  python scripts/deployment/promote_model_alias.py \
    --source-alias candidate \
    --target-alias champion \
    --require-approved-request
```

When promotion uses `--source-alias`, the source alias is cleared by default
after successful promotion. This keeps `candidate` as a review pointer rather
than leaving both `candidate` and `champion` on the same model version. Use
`--keep-source-alias` only when you intentionally want to keep both aliases.

`--require-approved-request` requires a matching `model_deployment_requests`
row created by the offline gate:

```text
source_version = candidate version
target_alias = champion
gate_status = passed
approval_status = approved
```

After successful promotion, the matching request is marked `deployed`.

You can also promote a concrete version. This does not clear `candidate`
because no source alias was provided. Use this path mainly for rollback or
manual override:

```bash
docker-compose run --rm model-trainer \
  python scripts/deployment/promote_model_alias.py \
    --model-version 5 \
    --target-alias champion
```

### Reject Candidate

If the comparison gate fails, keep `champion` unchanged and clear the review
pointer:

```bash
docker-compose run --rm model-trainer \
  python scripts/deployment/clear_model_alias.py \
    --alias candidate
```

Alias invariant:

```text
champion
  always points to the currently approved serving model

candidate
  exists only while a candidate is pending review
  should be cleared after promotion or rejection
```

### Recreate Model Server

The model server loads the MLflow model during process startup. Recreate it
after alias promotion.

```bash
docker-compose up -d --no-deps --force-recreate model-server-release
```

Verify the loaded runtime model:

```bash
curl http://127.0.0.1:28091/metadata
```

Expected fields:

```text
model_name=secom-fail-detector
model_alias=champion
model_version=<promoted version>
model_run_id=<promoted run id>
```

### Optional Smoke Test

Run from the project root.

```bash
uv run python scripts/workload/smoke_predict_by_id.py \
  --start-index 0 \
  --count 20 \
  --concurrency 4 \
  --print-failures
```

Verify model traffic in PostgreSQL:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  model_name,
  model_version,
  model_alias,
  model_run_id,
  threshold,
  COUNT(*) AS prediction_count
FROM prediction_logs
GROUP BY model_name, model_version, model_alias, model_run_id, threshold
ORDER BY prediction_count DESC;
"
```

### Rollback

Rollback means moving `champion` back to a known previous model version, then
recreating the model server so runtime serving reloads that version.

First find the latest deployed request and its previous champion:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  request_id,
  source_version AS deployed_version,
  previous_version AS rollback_version,
  target_alias,
  approval_status,
  to_timestamp(deployed_at) AS deployed_at
FROM model_deployment_requests
WHERE target_alias = 'champion'
  AND approval_status = 'deployed'
ORDER BY deployed_at DESC
LIMIT 5;
"
```

Then promote the rollback version back to `champion`. Concrete-version
promotion intentionally does not require a fresh approved request because it is
an emergency/manual recovery action.

```bash
docker-compose run --rm model-trainer \
  python scripts/deployment/promote_model_alias.py \
    --model-version <rollback_version> \
    --target-alias champion
```

Mark the deployment request being rolled back:

```bash
docker-compose run --rm \
  -e MONITORING_DATABASE_URL=postgresql://mlops:mlops@postgres:5432/monitoring \
  model-trainer \
  python scripts/deployment/mark_model_deployment_rolled_back.py \
    --request-id <request_id> \
    --expected-source-version <deployed_version> \
    --notes "rolled back to previous champion"
```

Reload serving and verify the runtime model:

```bash
docker-compose up -d --no-deps --force-recreate model-server-release

curl http://127.0.0.1:28091/metadata
```

Verify rollback history:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  request_id,
  source_version AS deployed_version,
  previous_version AS rollback_version,
  approval_status,
  to_timestamp(deployed_at) AS deployed_at,
  to_timestamp(rolled_back_at) AS rolled_back_at,
  notes
FROM model_deployment_requests
ORDER BY requested_at DESC
LIMIT 5;
"
```

### Notes

In this local workflow, `candidate` is a single review pointer. Future
production-style extensions can use `candidate_group` to compare multiple
candidate versions from the same retraining job, then promote the selected
version to `champion`.

The registry alias is the control-plane pointer. The serving process should be
verified through `/metadata` because changing an MLflow alias does not hot-reload
an already running model server.

## Verify Quality Windows

`quality-window-evaluator` builds non-overlapping label-backed quality windows
from `prediction_logs` joined to `actual_labels`.

```text
window type = non_overlapping_labeled_predictions
window size = 500 labeled predictions
partition = model_run_id + threshold
ordering = predicted_at, prediction_id
```

Status summary:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  evaluation_status,
  COUNT(*) AS window_count,
  MIN(to_timestamp(window_start)) AS first_window,
  MAX(to_timestamp(window_end)) AS last_window
FROM model_quality_windows
GROUP BY 1
ORDER BY 1;
"
```

Latest windows:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  window_id,
  evaluation_status,
  n_samples,
  n_fail_samples,
  accuracy,
  fail_precision,
  fail_recall,
  fail_f1,
  pr_auc,
  true_negative,
  false_positive,
  false_negative,
  true_positive,
  to_timestamp(window_start) AS window_start,
  to_timestamp(window_end) AS window_end
FROM model_quality_windows
ORDER BY window_id DESC
LIMIT 10;
"
```

Use only `evaluation_status = 'ok'` windows for quality trend and alert
decisions. `insufficient_samples` is normally the latest in-progress window.

## Verify Drift Metrics

`drift-metrics-evaluator` computes recent-vs-previous drift metrics from
`prediction_logs`.

```text
window type = recent_vs_previous_time_window
default window = latest 5 minutes vs previous 5 minutes
partition = model_run_id + threshold
```

Metric families:

```text
output
  prediction_count
  predicted_fail_ratio
  fail_probability_avg
  fail_probability_p50
  fail_probability_p95

input
  missing_count_avg
  missing_count_p95
  missing_count_max

feature
  feature_mean_standardized_delta
  feature_missing_ratio_abs_delta
```

Status summary:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  metric_family,
  metric_name,
  COUNT(*) AS row_count,
  MAX(to_timestamp(computed_at)) AS latest_computed_at
FROM drift_metrics
GROUP BY metric_family, metric_name
ORDER BY metric_family, metric_name;
"
```

Latest evaluation:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  metric_family,
  metric_name,
  feature_name,
  metric_value,
  baseline_value,
  current_value,
  delta_value,
  baseline_samples,
  current_samples,
  to_timestamp(current_start) AS current_start,
  to_timestamp(current_end) AS current_end
FROM drift_metrics
WHERE evaluation_id = (
  SELECT evaluation_id
  FROM drift_metrics
  ORDER BY computed_at DESC
  LIMIT 1
)
ORDER BY metric_family, metric_name, metric_value DESC NULLS LAST
LIMIT 80;
"
```

Grafana drift panels should read from `drift_metrics`. The older dashboard
queries that expand `prediction_logs.features_json` directly are useful for
debugging but should not be the primary monitoring path.

Latest model traffic:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  model_name,
  model_version,
  model_alias,
  model_run_id,
  threshold,
  COUNT(*) AS prediction_count
FROM prediction_logs
GROUP BY model_name, model_version, model_alias, model_run_id, threshold
ORDER BY prediction_count DESC;
"
```

## Grafana

Open:

```text
http://localhost:3000
```

Login:

```text
anonymous admin access is enabled locally
```

Provisioned files:

```text
grafana/datasource.yml
grafana/dashboard-provider.yml
grafana/dashboards/dashboard.json
```

Recommended dashboard grouping:

```text
Labeled Prediction Quality Windows
  source: model_quality_windows
  purpose: primary label-backed model quality trend

Output Drift
  source: drift_metrics
  purpose: model output distribution changes

Input / Feature Drift
  source: drift_metrics
  purpose: missing-count and top feature shift monitoring

Rolling / Latest Metrics
  source: model_metrics / prediction_window_metrics
  purpose: smoke visibility and latest-state debugging
```

## Script Boundary

Kafka-to-store materialization is handled by Java Docker Compose daemons:

```text
feature-raw-archiver
label-archiver
feature-assembler
feature-materializer
```

Current Python workload/source scripts:

```text
scripts/workload/send_feature_events_from_cursor.py
scripts/workload/send_label_events_from_cursor.py
scripts/workload/request_predictions_from_cursor.py
scripts/workload/smoke_predict_by_id.py
```

Offline/manual jobs:

```text
scripts/utility/build_offline_feature_snapshots.py
scripts/utility/reconstruct_offline_features.py
scripts/monitoring/predict_offline_feature_snapshots.py
scripts/monitoring/evaluate_offline_model_metrics.py
```

Runtime monitoring evaluators:

```text
scripts/monitoring/run_metrics_evaluator_loop.py
scripts/monitoring/evaluate_model_quality_windows.py
scripts/monitoring/evaluate_drift_metrics.py
```
