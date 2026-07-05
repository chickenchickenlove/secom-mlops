CREATE TABLE IF NOT EXISTS prediction_logs (
  prediction_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  sample_id TEXT NOT NULL,
  model_run_id TEXT NOT NULL,
  model_name TEXT,
  model_version TEXT,
  model_alias TEXT,
  model_uri TEXT,
  runtime_slot TEXT NOT NULL DEFAULT 'unknown',
  predicted_at DOUBLE PRECISION NOT NULL,
  fail_probability DOUBLE PRECISION NOT NULL,
  predicted_value INTEGER NOT NULL,
  predicted_label TEXT NOT NULL,
  threshold DOUBLE PRECISION NOT NULL,
  features_json JSONB NOT NULL,
  missing_count INTEGER NOT NULL,
  latency_ms DOUBLE PRECISION NOT NULL,
  CONSTRAINT chk_prediction_fail_probability
    CHECK (fail_probability >= 0.0 AND fail_probability <= 1.0),
  CONSTRAINT chk_prediction_value
    CHECK (predicted_value IN (-1, 1)),
  CONSTRAINT chk_prediction_label
    CHECK (predicted_label IN ('pass', 'fail')),
  CONSTRAINT chk_prediction_threshold
    CHECK (threshold >= 0.0 AND threshold <= 1.0),
  CONSTRAINT chk_prediction_missing_count
    CHECK (missing_count >= 0 AND missing_count <= 590)
);

  CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_time
    ON prediction_logs (model_run_id, predicted_at);

  CREATE INDEX IF NOT EXISTS idx_prediction_logs_sample_id
    ON prediction_logs (sample_id);

  CREATE TABLE IF NOT EXISTS actual_labels (
    sample_id TEXT PRIMARY KEY,
    actual_value INTEGER NOT NULL,
    actual_label TEXT NOT NULL,
    labeled_at DOUBLE PRECISION NOT NULL,
    CONSTRAINT chk_actual_value
      CHECK (actual_value IN (-1, 1)),
    CONSTRAINT chk_actual_label
      CHECK (actual_label IN ('pass', 'fail'))
  );

  CREATE TABLE IF NOT EXISTS model_metrics (
    id BIGSERIAL PRIMARY KEY,
    evaluation_id TEXT NOT NULL,
    computed_at DOUBLE PRECISION NOT NULL,
    model_run_id TEXT NOT NULL,
    threshold DOUBLE PRECISION,
    window_type TEXT NOT NULL,
    window_size INTEGER,
    window_start DOUBLE PRECISION,
    window_end DOUBLE PRECISION,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    n_samples INTEGER NOT NULL,
    n_fail_samples INTEGER NOT NULL,
    positive_class INTEGER NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    CONSTRAINT chk_model_metrics_positive_class
      CHECK (positive_class IN (-1, 1)),
    CONSTRAINT chk_model_metrics_counts
      CHECK (n_samples >= 0 AND n_fail_samples >= 0)
  );

  CREATE INDEX IF NOT EXISTS idx_model_metrics_model_time
    ON model_metrics (model_run_id, computed_at);

  CREATE INDEX IF NOT EXISTS idx_model_metrics_model_metric_time
    ON model_metrics (model_run_id, metric_name, computed_at);

  CREATE INDEX IF NOT EXISTS idx_model_metrics_evaluation_id
    ON model_metrics (evaluation_id);

  CREATE TABLE IF NOT EXISTS prediction_window_metrics (
    id BIGSERIAL PRIMARY KEY,
    evaluation_id TEXT NOT NULL,
    computed_at DOUBLE PRECISION NOT NULL,
    model_run_id TEXT NOT NULL,
    threshold DOUBLE PRECISION,
    window_type TEXT NOT NULL,
    window_size INTEGER,
    window_start DOUBLE PRECISION,
    window_end DOUBLE PRECISION,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    n_predictions INTEGER NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    CONSTRAINT chk_prediction_window_metrics_count
      CHECK (n_predictions >= 0)
  );

  CREATE INDEX IF NOT EXISTS idx_prediction_window_metrics_model_time
    ON prediction_window_metrics (model_run_id, computed_at);

  CREATE INDEX IF NOT EXISTS idx_prediction_window_metrics_model_metric_time
    ON prediction_window_metrics (model_run_id, metric_name, computed_at);

  CREATE INDEX IF NOT EXISTS idx_prediction_window_metrics_evaluation_id
    ON prediction_window_metrics (evaluation_id);


-- For Feature Store
  CREATE TABLE IF NOT EXISTS feature_events (
    event_id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL,
    event_time DOUBLE PRECISION NOT NULL,
    feature_group TEXT NOT NULL,
    features_json JSONB NOT NULL,
    simulation_run_id TEXT,
    drift_segment TEXT,
    created_at DOUBLE PRECISION NOT NULL,
    CONSTRAINT chk_feature_events_event_time
      CHECK (event_time >= 0.0),
    CONSTRAINT chk_feature_events_feature_group
      CHECK (length(feature_group) > 0),
    CONSTRAINT chk_feature_events_features_json
      CHECK (jsonb_typeof(features_json) = 'object')
  );

  CREATE INDEX IF NOT EXISTS idx_feature_events_sample_time
    ON feature_events (sample_id, event_time);

  CREATE INDEX IF NOT EXISTS idx_feature_events_run_time
    ON feature_events (simulation_run_id, event_time);

  CREATE INDEX IF NOT EXISTS idx_feature_events_group_time
    ON feature_events (feature_group, event_time);


  CREATE TABLE IF NOT EXISTS serving_feature_snapshots (
    serving_snapshot_id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL,
    snapshot_time DOUBLE PRECISION NOT NULL,
    window_start DOUBLE PRECISION NOT NULL,
    window_end DOUBLE PRECISION NOT NULL,
    snapshot_status TEXT NOT NULL,
    feature_count INTEGER NOT NULL,
    missing_count INTEGER NOT NULL,
    is_complete BOOLEAN NOT NULL,
    features_json JSONB NOT NULL,
    simulation_run_id TEXT,
    drift_segment TEXT,
    created_at DOUBLE PRECISION NOT NULL,
    CONSTRAINT chk_serving_feature_snapshots_time
      CHECK (
        snapshot_time >= 0.0
        AND window_start >= 0.0
        AND window_end >= window_start
      ),
    CONSTRAINT chk_serving_feature_snapshots_status
      CHECK (snapshot_status IN ('partial', 'timed_out', 'complete', 'late_update')),
    CONSTRAINT chk_serving_feature_snapshots_counts
      CHECK (
        feature_count >= 0
        AND feature_count <= 590
        AND missing_count >= 0
        AND missing_count <= 590
      ),
    CONSTRAINT chk_serving_feature_snapshots_complete
      CHECK (
        is_complete = FALSE
        OR (
          snapshot_status = 'complete'
          AND feature_count = 590
        )
      ),
    CONSTRAINT chk_serving_feature_snapshots_features_json
      CHECK (jsonb_typeof(features_json) = 'object')
  );

  CREATE INDEX IF NOT EXISTS idx_serving_feature_snapshots_sample_time
    ON serving_feature_snapshots (sample_id, snapshot_time DESC);

  CREATE INDEX IF NOT EXISTS idx_serving_feature_snapshots_run_time
    ON serving_feature_snapshots (simulation_run_id, snapshot_time);

  CREATE INDEX IF NOT EXISTS idx_serving_feature_snapshots_status_time
    ON serving_feature_snapshots (snapshot_status, snapshot_time);


  CREATE TABLE IF NOT EXISTS offline_feature_snapshots (
    offline_snapshot_id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL,
    build_cutoff_time DOUBLE PRECISION NOT NULL,
    feature_count INTEGER NOT NULL,
    missing_count INTEGER NOT NULL,
    is_complete BOOLEAN NOT NULL,
    features_json JSONB NOT NULL,
    source_event_count INTEGER NOT NULL,
    max_event_time DOUBLE PRECISION,
    simulation_run_id TEXT,
    drift_segment TEXT,
    created_at DOUBLE PRECISION NOT NULL,
    CONSTRAINT chk_offline_feature_snapshots_time
      CHECK (
        build_cutoff_time >= 0.0
        AND (
          max_event_time IS NULL
          OR max_event_time <= build_cutoff_time
        )
      ),
    CONSTRAINT chk_offline_feature_snapshots_counts
      CHECK (
        feature_count >= 0
        AND feature_count <= 590
        AND missing_count >= 0
        AND missing_count <= 590
        AND source_event_count >= 0
      ),
    CONSTRAINT chk_offline_feature_snapshots_complete
      CHECK (
        is_complete = FALSE
        OR feature_count = 590
      ),
    CONSTRAINT chk_offline_feature_snapshots_features_json
      CHECK (jsonb_typeof(features_json) = 'object')
  );

  CREATE INDEX IF NOT EXISTS idx_offline_feature_snapshots_sample_cutoff
    ON offline_feature_snapshots (sample_id, build_cutoff_time);

  CREATE INDEX IF NOT EXISTS idx_offline_feature_snapshots_run_cutoff
    ON offline_feature_snapshots (simulation_run_id, build_cutoff_time);

  CREATE INDEX IF NOT EXISTS idx_offline_feature_snapshots_max_event_time
    ON offline_feature_snapshots (max_event_time);


  CREATE TABLE IF NOT EXISTS offline_prediction_logs (
    offline_prediction_id TEXT PRIMARY KEY,
    offline_snapshot_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    build_cutoff_time DOUBLE PRECISION NOT NULL,
    model_run_id TEXT NOT NULL,
    predicted_at DOUBLE PRECISION NOT NULL,
    fail_probability DOUBLE PRECISION NOT NULL,
    predicted_value INTEGER NOT NULL,
    predicted_label TEXT NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    missing_count INTEGER NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    CONSTRAINT fk_offline_prediction_snapshot
      FOREIGN KEY (offline_snapshot_id)
      REFERENCES offline_feature_snapshots (offline_snapshot_id),
    CONSTRAINT uq_offline_prediction_snapshot_model_threshold
      UNIQUE (offline_snapshot_id, model_run_id, threshold),
    CONSTRAINT chk_offline_prediction_fail_probability
      CHECK (fail_probability >= 0.0 AND fail_probability <= 1.0),
    CONSTRAINT chk_offline_prediction_value
      CHECK (predicted_value IN (-1, 1)),
    CONSTRAINT chk_offline_prediction_label
      CHECK (predicted_label IN ('pass', 'fail')),
    CONSTRAINT chk_offline_prediction_threshold
      CHECK (threshold >= 0.0 AND threshold <= 1.0),
    CONSTRAINT chk_offline_prediction_missing_count
      CHECK (missing_count >= 0 AND missing_count <= 590),
    CONSTRAINT chk_offline_prediction_latency
      CHECK (latency_ms >= 0.0)
  );

  CREATE INDEX IF NOT EXISTS idx_offline_prediction_logs_sample_time
    ON offline_prediction_logs (sample_id, predicted_at);

  CREATE INDEX IF NOT EXISTS idx_offline_prediction_logs_model_time
    ON offline_prediction_logs (model_run_id, predicted_at);

  CREATE INDEX IF NOT EXISTS idx_offline_prediction_logs_cutoff
    ON offline_prediction_logs (build_cutoff_time);


  CREATE TABLE IF NOT EXISTS label_events (
    label_event_id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL,
    source_row_index INTEGER NOT NULL,
    event_time DOUBLE PRECISION NOT NULL,
    label_available_time DOUBLE PRECISION NOT NULL,
    actual_value INTEGER NOT NULL,
    actual_label TEXT NOT NULL,
    simulation_run_id TEXT,
    drift_segment TEXT,
    created_at DOUBLE PRECISION NOT NULL,
    CONSTRAINT chk_label_events_source_row_index
      CHECK (source_row_index >= 0),
    CONSTRAINT chk_label_events_time
      CHECK (
        event_time >= 0.0
        AND label_available_time >= event_time
      ),
    CONSTRAINT chk_label_events_actual_value
      CHECK (actual_value IN (-1, 1)),
    CONSTRAINT chk_label_events_actual_label
      CHECK (actual_label IN ('pass', 'fail'))
  );

  CREATE INDEX IF NOT EXISTS idx_label_events_sample_time
    ON label_events (sample_id, label_available_time);

  CREATE INDEX IF NOT EXISTS idx_label_events_run_time
    ON label_events (simulation_run_id, label_available_time);


  CREATE TABLE IF NOT EXISTS model_quality_windows (
    id BIGSERIAL PRIMARY KEY,
    window_type TEXT NOT NULL DEFAULT 'non_overlapping_labeled_predictions',
    window_size INTEGER NOT NULL,
    window_id INTEGER NOT NULL,
    window_start DOUBLE PRECISION NOT NULL,
    window_end DOUBLE PRECISION NOT NULL,
    computed_at DOUBLE PRECISION NOT NULL,

    model_name TEXT,
    model_version TEXT,
    model_alias TEXT,
    model_run_id TEXT NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,

    n_samples INTEGER NOT NULL,
    n_fail_samples INTEGER NOT NULL,
    evaluation_status TEXT NOT NULL CHECK (
      evaluation_status IN ('ok', 'insufficient_samples', 'insufficient_fail_labels')
    ),

    accuracy DOUBLE PRECISION,
    fail_precision DOUBLE PRECISION,
    fail_recall DOUBLE PRECISION,
    fail_f1 DOUBLE PRECISION,
    pr_auc DOUBLE PRECISION,

    true_negative INTEGER NOT NULL,
    false_positive INTEGER NOT NULL,
    false_negative INTEGER NOT NULL,
    true_positive INTEGER NOT NULL,

    created_at DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW()),
    updated_at DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW()),

    UNIQUE (model_run_id, threshold, window_type, window_size, window_id)
  );

  CREATE INDEX IF NOT EXISTS idx_model_quality_windows_window_end
    ON model_quality_windows (window_end);

  CREATE INDEX IF NOT EXISTS idx_model_quality_windows_status
    ON model_quality_windows (evaluation_status, window_end);

CREATE TABLE IF NOT EXISTS drift_reference_baselines (
    baseline_id TEXT PRIMARY KEY,

    baseline_name TEXT NOT NULL,
    baseline_type TEXT NOT NULL DEFAULT 'fixed_reference',

    source_table TEXT NOT NULL,
    source_start DOUBLE PRECISION NOT NULL,
    source_end DOUBLE PRECISION NOT NULL,
    source_filter_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    model_name TEXT,
    model_version TEXT,
    model_alias TEXT,
    model_run_id TEXT NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,

    feature_count INTEGER NOT NULL DEFAULT 590,
    sample_count INTEGER NOT NULL,

    status TEXT NOT NULL DEFAULT 'active',
    notes TEXT,

    created_by TEXT,
    created_at DOUBLE PRECISION NOT NULL,

    CONSTRAINT chk_drift_reference_baselines_time
      CHECK (
        source_start >= 0.0
        AND source_end >= source_start
      ),

    CONSTRAINT chk_drift_reference_baselines_threshold
      CHECK (threshold >= 0.0 AND threshold <= 1.0),

    CONSTRAINT chk_drift_reference_baselines_counts
      CHECK (
        feature_count >= 0
        AND sample_count >= 0
      ),

    CONSTRAINT chk_drift_reference_baselines_status
      CHECK (status IN ('active', 'retired'))
  );

  CREATE TABLE IF NOT EXISTS drift_reference_stats (
    id BIGSERIAL PRIMARY KEY,

    baseline_id TEXT NOT NULL
      REFERENCES drift_reference_baselines (baseline_id)
      ON DELETE CASCADE,

    metric_family TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    feature_name TEXT,

    metric_value DOUBLE PRECISION,

    sample_count INTEGER NOT NULL,
    non_null_count INTEGER,
    null_count INTEGER,

    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at DOUBLE PRECISION NOT NULL,

    CONSTRAINT chk_drift_reference_stats_family
      CHECK (metric_family IN ('output', 'input', 'feature')),

    CONSTRAINT chk_drift_reference_stats_counts
      CHECK (
        sample_count >= 0
        AND (non_null_count IS NULL OR non_null_count >= 0)
        AND (null_count IS NULL OR null_count >= 0)
      )
  );


  CREATE TABLE IF NOT EXISTS drift_metrics (
    id BIGSERIAL PRIMARY KEY,
    evaluation_id TEXT NOT NULL,

    reference_baseline_id TEXT
      REFERENCES drift_reference_baselines (baseline_id),

    computed_at DOUBLE PRECISION NOT NULL,
    window_type TEXT NOT NULL,
    window_minutes INTEGER NOT NULL,

    baseline_start DOUBLE PRECISION NOT NULL,
    baseline_end DOUBLE PRECISION NOT NULL,
    current_start DOUBLE PRECISION NOT NULL,
    current_end DOUBLE PRECISION NOT NULL,

    model_name TEXT,
    model_version TEXT,
    model_alias TEXT,
    model_run_id TEXT NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,

    metric_family TEXT NOT NULL CHECK (metric_family IN ('output', 'input', 'feature')),
    metric_name TEXT NOT NULL,
    feature_name TEXT,

    metric_value DOUBLE PRECISION,
    baseline_value DOUBLE PRECISION,
    current_value DOUBLE PRECISION,
    delta_value DOUBLE PRECISION,

    baseline_samples INTEGER NOT NULL,
    current_samples INTEGER NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
  );

  CREATE INDEX IF NOT EXISTS idx_drift_metrics_computed_at
    ON drift_metrics (computed_at);

  CREATE INDEX IF NOT EXISTS idx_drift_metrics_window_end
    ON drift_metrics (current_end);

  CREATE INDEX IF NOT EXISTS idx_drift_metrics_family_name_time
    ON drift_metrics (metric_family, metric_name, computed_at);

  CREATE INDEX IF NOT EXISTS idx_drift_metrics_feature_time
    ON drift_metrics (feature_name, computed_at);

  CREATE UNIQUE INDEX IF NOT EXISTS ux_drift_metrics_window_metric
    ON drift_metrics (
      model_run_id,
      threshold,
      window_type,
      window_minutes,
      current_end,
      metric_family,
      metric_name,
      COALESCE(feature_name, '')
    );


  CREATE TABLE IF NOT EXISTS model_deployment_requests (
    request_id TEXT PRIMARY KEY,

    model_name TEXT NOT NULL,
    target_alias TEXT NOT NULL DEFAULT 'champion',

    source_alias TEXT,
    source_version TEXT NOT NULL,
    source_run_id TEXT NOT NULL,

    previous_version TEXT,
    previous_run_id TEXT,

    eval_type TEXT NOT NULL DEFAULT 'unknown',
    eval_status TEXT NOT NULL DEFAULT 'unknown' CHECK (
      eval_status IN ('passed', 'failed', 'insufficient_data', 'skipped', 'unknown')
    ),
    eval_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    approval_status TEXT NOT NULL DEFAULT 'pending' CHECK (
      approval_status IN ('pending', 'approved', 'rejected')
    ),

    rollout_status TEXT NOT NULL DEFAULT 'not_started' CHECK (
      rollout_status IN (
        'not_started',
        'promoted',
        'canary_reloading',
        'canary_ready',
        'release_reloading',
        'deployed',
        'failed',
        'rolled_back',
        'superseded'
      )
    ),

    runtime_target TEXT NOT NULL DEFAULT 'release',

    notes TEXT,
    requested_by TEXT,
    approved_by TEXT,

    requested_at DOUBLE PRECISION NOT NULL,
    approved_at DOUBLE PRECISION,
    promoted_at DOUBLE PRECISION,
    canary_reload_started_at DOUBLE PRECISION,
    canary_ready_at DOUBLE PRECISION,
    release_reload_started_at DOUBLE PRECISION,
    deployed_at DOUBLE PRECISION,
    failed_at DOUBLE PRECISION,
    rolled_back_at DOUBLE PRECISION,

    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
  );

  CREATE TABLE IF NOT EXISTS model_runtime_reload_events (
    event_id TEXT PRIMARY KEY,

    request_id TEXT,

    service_name TEXT NOT NULL,
    runtime_slot TEXT NOT NULL,

    model_name TEXT NOT NULL,

    previous_model_version TEXT,
    previous_model_run_id TEXT,
    previous_threshold DOUBLE PRECISION,

    new_model_version TEXT NOT NULL,
    new_model_run_id TEXT NOT NULL,
    new_threshold DOUBLE PRECISION NOT NULL,

    reload_status TEXT NOT NULL DEFAULT 'started' CHECK (
      reload_status IN ('started', 'succeeded', 'failed')
    ),

    error_message TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    started_at DOUBLE PRECISION NOT NULL,
    completed_at DOUBLE PRECISION,
    created_at DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())
  );

  CREATE TABLE IF NOT EXISTS model_runtime_deployment_state (
    model_name TEXT NOT NULL,
    runtime_slot TEXT NOT NULL,
    target_alias TEXT NOT NULL DEFAULT 'champion',

    active_request_id TEXT,
    active_model_version TEXT NOT NULL,
    active_model_run_id TEXT NOT NULL,
    active_threshold DOUBLE PRECISION NOT NULL,

    previous_request_id TEXT,
    previous_model_version TEXT,
    previous_model_run_id TEXT,
    previous_threshold DOUBLE PRECISION,

    last_operation TEXT NOT NULL,
    last_operation_request_id TEXT,

    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,

    PRIMARY KEY (model_name, runtime_slot)
  );
