# PostgreSQL Monitoring Schema

## Data Tables

| Category | Tables |
| --- | --- |
| Operational evidence | `feature_events`, `serving_feature_snapshots`, `label_events`, `prediction_logs` |
| Optional offline utilities | `offline_feature_snapshots`, `offline_prediction_logs` |
| Live monitoring | `live_model_quality_evaluations`, `prediction_window_metrics`, `drift_metrics` |
| Offline evaluation | `model_metrics` |
| Drift reference | `drift_reference_baselines`, `drift_reference_stats` |
| Deployment state | `model_deployment_requests`, `model_runtime_deployment_state`, `model_runtime_reload_events` |

`serving_feature_snapshots` stores immutable online-serving Feature history
with Valkey-confirmed `available_at`. `label_events` stores append-only Label
revision history with PostgreSQL-assigned `available_at`.
