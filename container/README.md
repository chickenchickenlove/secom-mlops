# Local Monitoring Stack

This stack runs the local SECOM/FDC online path, offline/raw store sync, model serving, and Grafana monitoring.

## Components

- PostgreSQL stores `feature_events`, `serving_feature_snapshots`, append-only `label_events`, `prediction_logs`, offline data, and monitoring metrics.
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
   -> PostgreSQL serving_feature_snapshots
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
-> PostgreSQL append-only label_events

Prediction evidence store:
serving-api /predict-by-id
-> secom-prediction-events
-> prediction-log-archiver
-> PostgreSQL prediction_logs

Prediction window metrics:
prediction_logs
-> metrics-evaluator
-> prediction_window_metrics
-> Grafana

Live label-backed model quality:
prediction_logs + label_events
-> metrics-evaluator / evaluate_live_model_quality.py
-> live_model_quality_evaluations
-> Grafana

Drift metrics:
prediction_logs
  + serving_feature_snapshots
    logical identity = serving_snapshot_id + sample_id + snapshot_version
    content check = feature_hash
-> drift-metrics-evaluator
-> drift_metrics
-> Grafana
```

Operational prediction evidence is emitted only by `/predict-by-id`. Each row
records the `serving_snapshot_id`, sample-local `snapshot_version`, and
`feature_hash` used for inference. `/predict` accepts caller-supplied feature vectors and is a debug
endpoint, so it does not emit operational prediction evidence.

`label_events` is an append-only correction history. `measured_at` is measurement
metadata, `available_at` is the time PostgreSQL can observe the label, and a
higher `label_revision` supersedes an earlier revision for the same sample.
Cutoff selection first applies `available_at <= T` and then chooses the highest
available revision.

The live quality evaluator captures one database cutoff `T` per cycle and uses
`S = T-L-W`, `E = T-L`, and the half-open decision window `[S,E)`. It selects
the global-first prediction for each semantic decision, left-joins labels at
`T`, and stores coverage, status, confusion counts, and eligible scalar metrics
in one `live_model_quality_evaluations` row per model run and threshold.
`model_metrics` is reserved for offline evaluation; its label-history migration
is still pending.

The three feature-vector consumers
`evaluate_drift_metrics.py`, `create_drift_reference_baseline.py`, and
`evaluate_fixed_reference_drift_metrics.py` join `prediction_logs` to
`serving_feature_snapshots` on the logical identity `serving_snapshot_id +
sample_id + snapshot_version`, require matching `feature_hash`, and read
`serving_feature_snapshots.features_json`.
Prediction events and `prediction_logs` do not duplicate the full feature
vector. The small `missing_count` scalar remains in prediction logs. The
snapshot reference is intentionally logical; there is no foreign key.

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
  --sleep-seconds 10
```

The producer sets `measured_at` immediately before publishing. The label
archiver records `available_at` with the PostgreSQL clock when it first inserts
the append-only event. To simulate delayed labels, start the label workload
later instead of passing a synthetic availability timestamp.

Prediction request workload:

```bash
uv run python scripts/workload/request_predictions_from_cursor.py \
  --max-samples 30000 \
  --batch-size 300 \
  --sleep-seconds 15 \
  --concurrency 16 \
  --print-failures
```

This workload calls `/predict-by-id`, so successful requests create
snapshot-backed operational prediction evidence. Direct `/predict` calls are
debug-only and do not create rows in `prediction_logs`.

The default cursor state files are:

```text
runtime/online_workload_next_feature_{feature_group}_state.json
runtime/online_workload_next_label_state.json
runtime/online_workload_next_predict_state.json
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

Recent raw feature patch aggregation:

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

Append-only label event history:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  COUNT(*) AS label_event_count,
  COUNT(DISTINCT sample_id) AS labeled_sample_count,
  COUNT(*) FILTER (WHERE actual_label = 'fail') AS fail_count,
  COUNT(*) FILTER (WHERE actual_label = 'pass') AS pass_count,
  MIN(label_revision) AS min_revision,
  MAX(label_revision) AS max_revision,
  MIN(to_timestamp(available_at)) AS first_available_at,
  MAX(to_timestamp(available_at)) AS last_available_at
FROM label_events;
"
```

Labels visible at the current cutoff `T`:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
WITH params AS (
  SELECT EXTRACT(EPOCH FROM statement_timestamp())::DOUBLE PRECISION AS cutoff_time
),
ranked_labels AS (
  SELECT
    le.*,
    ROW_NUMBER() OVER (
      PARTITION BY le.sample_id
      ORDER BY
        le.label_revision DESC,
        le.available_at DESC,
        le.label_event_id DESC
    ) AS label_rank
  FROM label_events le
  CROSS JOIN params p
  WHERE le.available_at <= p.cutoff_time
)
SELECT
  COUNT(*) AS labeled_sample_count,
  COUNT(*) FILTER (WHERE actual_label = 'fail') AS fail_count,
  COUNT(*) FILTER (WHERE actual_label = 'pass') AS pass_count
FROM ranked_labels
WHERE label_rank = 1;
"
```

First-complete serving snapshot and label coverage at cutoff `T`:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
WITH params AS (
  SELECT EXTRACT(EPOCH FROM statement_timestamp())::DOUBLE PRECISION AS cutoff_time
),
ranked_complete_snapshots AS (
  SELECT
    s.sample_id,
    s.available_at,
    ROW_NUMBER() OVER (
      PARTITION BY s.sample_id
      ORDER BY
        s.available_at ASC,
        s.snapshot_version ASC,
        s.serving_snapshot_id ASC
    ) AS snapshot_rank
  FROM serving_feature_snapshots s
  WHERE s.snapshot_status = 'complete'
    AND s.is_complete = TRUE
),
first_complete_snapshots AS (
  SELECT sample_id, available_at
  FROM ranked_complete_snapshots
  WHERE snapshot_rank = 1
),
ranked_labels AS (
  SELECT
    le.sample_id,
    ROW_NUMBER() OVER (
      PARTITION BY le.sample_id
      ORDER BY
        le.label_revision DESC,
        le.available_at DESC,
        le.label_event_id DESC
    ) AS label_rank
  FROM label_events le
  CROSS JOIN params p
  WHERE le.available_at <= p.cutoff_time
),
labels_at_cutoff AS (
  SELECT sample_id
  FROM ranked_labels
  WHERE label_rank = 1
)
SELECT
  COUNT(*) AS first_complete_samples,
  COUNT(l.sample_id) AS samples_with_label,
  COUNT(*) - COUNT(l.sample_id) AS samples_without_label,
  MIN(f.sample_id) AS first_feature_sample_id,
  MAX(f.sample_id) AS last_feature_sample_id
FROM first_complete_snapshots f
CROSS JOIN params p
LEFT JOIN labels_at_cutoff l
  ON l.sample_id = f.sample_id
WHERE f.available_at <= p.cutoff_time;
"
```

## Verify Online Predictions

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  COUNT(*) AS prediction_count,
  COUNT(DISTINCT serving_snapshot_id) AS serving_snapshot_count,
  MIN(snapshot_version) AS min_snapshot_version,
  MAX(snapshot_version) AS max_snapshot_version,
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

Candidate retraining reads immutable online-serving history directly from
`serving_feature_snapshots`. It uses each sample's first complete snapshot and
the snapshot's Valkey-confirmed `available_at` as the training decision time.
It does not reconstruct the training vector from `feature_events`.

Trigger the Airflow training DAG:

```text
train_candidate_from_offline_point_in_time_features
```

Then trigger the serving prediction decision gate DAG on its intended
independent cohort:

```text
evaluate_candidate_serving_snapshot_gate
```

Both DAGs use `cohort_start_time`, `cutoff_time`, and
`label_maturity_seconds`, but their decision spines are different. Training
uses first-complete snapshot `available_at`; the serving gate uses champion
prediction `predicted_at`.

For direct debugging from `container`, run the current trainer:

```bash
docker-compose run --rm \
  -e MONITORING_DATABASE_URL=postgresql://mlops:mlops@postgres:5432/monitoring \
  -e MLFLOW_TRACKING_URI=http://mlflow:5100 \
  model-trainer \
  python scripts/training/train_candidate_from_offline_point_in_time_features.py \
    --cohort-start-time 9999999399 \
    --cutoff-time 9999999999 \
    --label-maturity-seconds 0 \
    --candidate-group retrain_20260701_001 \
    --training-job-id retrain_20260701_001 \
    --min-samples 500 \
    --min-label-coverage 0.95 \
    --min-fail-samples 20 \
    --min-pass-samples 20
```

The required hard gate evaluates candidate and champion on the same cohort of
actual champion prediction decisions. For direct debugging from `container`,
run:

```bash
docker-compose run --rm \
  -e MONITORING_DATABASE_URL=postgresql://mlops:mlops@postgres:5432/monitoring \
  model-trainer \
  python scripts/monitoring/compare_candidate_with_champion_serving.py \
    --cohort-start-time 1783764800 \
    --cutoff-time 1783765500 \
    --label-maturity-seconds 0 \
    --max-decisions 1000 \
    --min-decisions 500 \
    --min-label-coverage 0.95 \
    --min-fail-samples 20 \
    --min-pass-samples 20 \
    --set-tags \
    --fail-on-gate-failure
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
  source_run_id,
  target_alias,
  eval_type,
  eval_status,
  approval_status,
  rollout_status,
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

### Train Candidate From Serving Snapshot History

Run from `container`.

```bash
docker-compose run --rm \
  -e MONITORING_DATABASE_URL=postgresql://mlops:mlops@postgres:5432/monitoring \
  -e MLFLOW_TRACKING_URI=http://mlflow:5100 \
  model-trainer \
  python scripts/training/train_candidate_from_offline_point_in_time_features.py \
    --cohort-start-time 9999999399 \
    --cutoff-time 9999999999 \
    --label-maturity-seconds 0 \
    --candidate-group retrain_20260701_001 \
    --training-job-id retrain_20260701_001 \
    --min-samples 500 \
    --min-label-coverage 0.95 \
    --min-fail-samples 20 \
    --min-pass-samples 20
```

Candidate training requires:

```text
ML_CANDIDATE_GROUP
ML_TRAINING_JOB_ID
```

The trainer defines `S = cohort_start_time`, `T = cutoff_time`, and
`L = label_maturity_seconds`, then derives `cohort_end_time = T - L`. It first
selects each sample's earliest complete `serving_feature_snapshots` row across
the full snapshot history and includes it when its `available_at` is in
`[S, T-L]`. The stored `features_json` is the training vector; `snapshot_time`
remains event-time metadata and is not the training cutoff.

After applying the first-complete and time filters, the trainer orders eligible
snapshots by `available_at` descending and selects at most 1,000 rows before
joining labels.

Labels are selected from append-only `label_events`: only rows with
`available_at <= T` are visible, and the highest `label_revision` for each
sample wins. Labels are `LEFT JOIN`ed to the selected snapshot cohort so
unlabeled rows remain part of the `label_coverage` denominator. Only labeled
rows are used for model development.

The main Airflow defaults require 1,000 labeled development rows. The Candidate
uses a stratified 80% selection-training source and 20% validation source to
select hyperparameters and the threshold, then refits the registered model on
the complete labeled development source. With the default full cohort, this is
800 training rows, 200 validation rows, and a final fit on all 1,000 rows.

The bootstrap Champion uses the first 1,000 raw SECOM rows and follows the same
800/200 selection and full-1,000 refit procedure. Candidate and Champion use
the same development size and selection procedure, not the same source rows.

The trainer registers a new `secom-fail-detector` model version, points the
`candidate` alias to it, and writes model version tags such as:

```text
role=candidate
candidate_group=...
training_job_id=...
train_source=serving_feature_snapshot_history
training_spine=serving_feature_snapshots
training_decision_time=serving_feature_snapshots.available_at
snapshot_selection=first_complete
label_selection=available_at_lte_cutoff_then_max_revision
cohort_start_time=...
cohort_end_time=...
cutoff_time=...
label_maturity_seconds=...
gate_status=pending
```

The older default `model-trainer` command still trains from raw SECOM CSV files
and is useful for bootstrap/champion seeding. For retraining from operational
data, prefer `scripts/training/train_candidate_from_offline_point_in_time_features.py`.

### Airflow Candidate Retraining DAGs

The DAG `train_candidate_from_offline_point_in_time_features` trains and
registers a candidate from first-complete serving snapshot history and the
label history visible at `cutoff_time`.

Trigger it manually from the Airflow UI, or with config such as:

```json
{
  "cohort_start_time": "2026-07-05T12:50:00+09:00",
  "cutoff_time": "2026-07-05T13:00:00+09:00",
  "label_maturity_seconds": 0,
  "min_samples": 1000,
  "min_label_coverage": 0.95,
  "min_fail_samples": 20,
  "min_pass_samples": 20
}
```

After training succeeds, trigger `evaluate_candidate_serving_snapshot_gate`
with an independently selected prediction decision cohort:

```json
{
  "cohort_start_time": "2026-07-11T19:45:00+09:00",
  "cutoff_time": "2026-07-11T20:00:00+09:00",
  "label_maturity_seconds": 0,
  "candidate_version": null,
  "max_decisions": 1000,
  "min_decisions": 500,
  "min_label_coverage": 0.95,
  "min_fail_samples": 20,
  "min_pass_samples": 20,
  "dry_run": false
}
```

The gate selects the global-first decision for each semantic identity, filters
the requested time cohort, and then keeps the latest `max_decisions` rows for
the current Champion model run. It verifies the exact serving snapshot identity
and Feature hash and uses the highest label revision with
`available_at <= cutoff_time`.

Candidate and Champion are replayed on the same labeled Feature vectors. The
Airflow task succeeds only when the Gate status is `passed`; `failed`,
`insufficient_data`, and integrity errors fail the task.

The current Gate filters the Champion `model_run_id` but does not yet enforce
`runtime_slot='release'` or the actual release threshold.

If the gate fails, run `cleanup_failed_candidate_alias`. If the gate passes,
create or approve the deployment request before canary/release deployment.

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

### Record Deployment Request After the Serving Gate

The required promotion gate is
`evaluate_candidate_serving_snapshot_gate`, described above. A passed Gate
records its result in the Candidate model version tags but does not
automatically create a deployment request.

After the Gate passes, use the Airflow DAG:

```text
record_serving_candidate_deployment_request
```

For direct debugging from `container`, the equivalent command is:

```bash
docker-compose run --rm model-trainer \
  python scripts/deployment/record_serving_candidate_deployment_request.py \
    --approval-status approved \
    --notes "serving prediction decision gate passed"
```

This command verifies the Candidate's serving Gate tags and writes a
`model_deployment_requests` row. Promotion with
`--require-approved-request` requires this approved request.

The legacy `compare_candidate_with_champion_offline.py` is not part of the
current promotion path. It still depends on the removed `actual_labels` table
and must not be used with the current schema.

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
row created after the Serving Gate:

```text
source_version = candidate version
source_run_id = candidate run id
target_alias = champion
eval_type = serving_snapshot
eval_status = passed
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

## Verify Live Model Quality

`metrics-evaluator` runs `evaluate_live_model_quality.py` every 30 seconds. For
each cycle it captures one database cutoff `T`, sets `E = T-L` and
`S = E-W`, and evaluates decisions in the sliding time window `[S,E)`.

The current Docker Compose defaults are:

```text
L = 0 seconds
W = 600 seconds
minimum decisions = 500
minimum label coverage = 0.95
minimum fail labels = 20
minimum pass labels = 20
partition = model_run_id + threshold
```

Before applying the time window, repeated predictions are reduced to the
global-first row for each semantic decision:

```text
model_run_id + threshold + sample_id
  + serving_snapshot_id + snapshot_version
```

The evaluator joins the highest `label_revision` visible under
`label_events.available_at <= T`. It writes one
`live_model_quality_evaluations` row per non-empty model/threshold cohort. A
cycle with no decisions writes no row.

Status summary:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  evaluation_status,
  COUNT(*) AS evaluation_count,
  MIN(to_timestamp(cutoff_time)) AS first_cutoff,
  MAX(to_timestamp(cutoff_time)) AS latest_cutoff
FROM live_model_quality_evaluations
GROUP BY 1
ORDER BY 1;
"
```

Latest evaluations:

```bash
docker-compose exec postgres psql -U mlops -d monitoring -c "
SELECT
  evaluation_id,
  model_run_id,
  threshold,
  evaluation_status,
  n_decisions,
  n_samples,
  n_pass_samples,
  n_fail_samples,
  label_coverage,
  accuracy,
  fail_precision,
  fail_recall,
  fail_f1,
  fail_average_precision,
  true_negative,
  false_positive,
  false_negative,
  true_positive,
  to_timestamp(cutoff_time) AS cutoff_time,
  to_timestamp(window_start) AS window_start,
  to_timestamp(window_end) AS window_end
FROM live_model_quality_evaluations
ORDER BY cutoff_time DESC, model_run_id, threshold
LIMIT 10;
"
```

Use only `evaluation_status = 'ok'` rows for scalar quality trends and alerts.
The evaluator still stores cohort counts and the confusion matrix for non-`ok`
rows, but stores `accuracy`, fail precision/recall/F1, and fail average
precision as `NULL`. Status checks run in this order: decisions, label coverage,
fail labels, and pass labels.

## Verify Drift Metrics

`drift-metrics-evaluator` computes recent-vs-previous drift metrics from
`prediction_logs`. Feature-vector metrics read
`serving_feature_snapshots.features_json` through the logical triple join on
`serving_snapshot_id`, `sample_id`, and `snapshot_version`; input missing-count
metrics continue to read `prediction_logs.missing_count`.

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

Grafana drift panels should read from `drift_metrics`. Prediction logs do not
store a duplicate feature vector; feature drift consumers resolve it from
`serving_feature_snapshots` through the logical triple join.

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
Live Sliding Model Quality
  source: live_model_quality_evaluations
  purpose: label coverage, status, confusion matrix, and live quality trends

Output Drift
  source: drift_metrics
  purpose: model output distribution changes

Input / Feature Drift
  source: drift_metrics
  purpose: missing-count and top feature shift monitoring

Prediction Traffic / Operational Windows
  source: prediction_logs / prediction_window_metrics
  purpose: request volume, latency, prediction mix, and latest-state debugging

Offline Model Evaluation
  source: model_metrics
  purpose: manual/offline evaluation; label-history migration is still pending
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

Offline snapshot utilities:

```text
scripts/utility/build_offline_feature_snapshots.py
scripts/utility/reconstruct_offline_features.py
scripts/monitoring/predict_offline_feature_snapshots.py
```

Legacy Label-backed offline evaluators:

```text
scripts/monitoring/evaluate_offline_model_metrics.py
scripts/monitoring/compare_candidate_with_champion_offline.py
```

The legacy evaluators still query the removed `actual_labels` table. They are
not part of the current training, monitoring, or promotion path and must be
migrated to `label_events` before they can be used with the current schema.

Runtime monitoring evaluators:

```text
scripts/monitoring/run_metrics_evaluator_loop.py
  -> scripts/monitoring/evaluate_prediction_window_metrics.py
  -> scripts/monitoring/evaluate_live_model_quality.py
scripts/monitoring/evaluate_drift_metrics.py
```
