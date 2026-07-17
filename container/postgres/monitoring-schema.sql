CREATE TABLE IF NOT EXISTS prediction_logs (
  prediction_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  sample_id TEXT NOT NULL,
  serving_snapshot_id TEXT NOT NULL,
  snapshot_version BIGINT NOT NULL,
  feature_hash TEXT NOT NULL,
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
  CONSTRAINT chk_prediction_snapshot_version
    CHECK (snapshot_version > 0),
  CONSTRAINT chk_prediction_feature_hash
    CHECK (feature_hash ~ '^sha256:v1:[0-9a-f]{64}$'),
  CONSTRAINT chk_prediction_missing_count
    CHECK (missing_count >= 0 AND missing_count <= 590)
);

  CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_time
    ON prediction_logs (model_run_id, predicted_at);

  CREATE INDEX IF NOT EXISTS idx_prediction_logs_sample_id
    ON prediction_logs (sample_id);

  CREATE INDEX IF NOT EXISTS idx_prediction_logs_serving_snapshot
    ON prediction_logs (serving_snapshot_id, sample_id, snapshot_version, feature_hash);

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

  CREATE TABLE IF NOT EXISTS live_model_quality_evaluations (
    evaluation_id TEXT PRIMARY KEY,
    computed_at DOUBLE PRECISION NOT NULL,
    model_run_id TEXT NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    window_type TEXT NOT NULL DEFAULT 'sliding_time',

    cutoff_time DOUBLE PRECISION NOT NULL,
    label_maturity_seconds DOUBLE PRECISION NOT NULL,
    monitoring_window_seconds DOUBLE PRECISION NOT NULL,
    window_start DOUBLE PRECISION NOT NULL,
    window_end DOUBLE PRECISION NOT NULL,

    n_decisions INTEGER NOT NULL,
    n_samples INTEGER NOT NULL,
    n_pass_samples INTEGER NOT NULL,
    n_fail_samples INTEGER NOT NULL,
    label_coverage DOUBLE PRECISION NOT NULL,

    min_decisions INTEGER NOT NULL,
    min_label_coverage DOUBLE PRECISION NOT NULL,
    min_pass_samples INTEGER NOT NULL,
    min_fail_samples INTEGER NOT NULL,
    evaluation_status TEXT NOT NULL,

    accuracy DOUBLE PRECISION,
    fail_precision DOUBLE PRECISION,
    fail_recall DOUBLE PRECISION,
    fail_f1 DOUBLE PRECISION,
    fail_average_precision DOUBLE PRECISION,
    true_negative INTEGER NOT NULL,
    false_positive INTEGER NOT NULL,
    false_negative INTEGER NOT NULL,
    true_positive INTEGER NOT NULL,

    created_at DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW()),

    CONSTRAINT chk_live_model_quality_threshold
      CHECK (threshold >= 0.0 AND threshold <= 1.0),
    CONSTRAINT chk_live_model_quality_window_type
      CHECK (window_type = 'sliding_time'),
    CONSTRAINT chk_live_model_quality_times
      CHECK (
        computed_at >= 0.0
        AND computed_at < 'Infinity'::DOUBLE PRECISION
        AND cutoff_time >= 0.0
        AND cutoff_time < 'Infinity'::DOUBLE PRECISION
        AND label_maturity_seconds >= 0.0
        AND label_maturity_seconds < 'Infinity'::DOUBLE PRECISION
        AND monitoring_window_seconds > 0.0
        AND monitoring_window_seconds < 'Infinity'::DOUBLE PRECISION
        AND cutoff_time >= label_maturity_seconds + monitoring_window_seconds
        AND window_start >= 0.0
        AND window_start < 'Infinity'::DOUBLE PRECISION
        AND window_end >= window_start
        AND window_end < 'Infinity'::DOUBLE PRECISION
      ),
    CONSTRAINT chk_live_model_quality_counts
      CHECK (
        n_decisions >= 0
        AND n_samples >= 0
        AND n_samples <= n_decisions
        AND n_pass_samples >= 0
        AND n_fail_samples >= 0
        AND n_pass_samples + n_fail_samples = n_samples
      ),
    CONSTRAINT chk_live_model_quality_label_coverage
      CHECK (label_coverage >= 0.0 AND label_coverage <= 1.0),
    CONSTRAINT chk_live_model_quality_minimums
      CHECK (
        min_decisions > 0
        AND min_label_coverage >= 0.0
        AND min_label_coverage <= 1.0
        AND min_pass_samples >= 0
        AND min_fail_samples >= 0
      ),
    CONSTRAINT chk_live_model_quality_status_value
      CHECK (
        evaluation_status IN (
          'ok',
          'insufficient_decisions',
          'insufficient_label_coverage',
          'insufficient_fail_labels',
          'insufficient_pass_labels'
        )
      ),
    CONSTRAINT chk_live_model_quality_metric_ranges
      CHECK (
        (accuracy IS NULL OR (accuracy >= 0.0 AND accuracy <= 1.0))
        AND (fail_precision IS NULL OR (fail_precision >= 0.0 AND fail_precision <= 1.0))
        AND (fail_recall IS NULL OR (fail_recall >= 0.0 AND fail_recall <= 1.0))
        AND (fail_f1 IS NULL OR (fail_f1 >= 0.0 AND fail_f1 <= 1.0))
        AND (
          fail_average_precision IS NULL
          OR (fail_average_precision >= 0.0 AND fail_average_precision <= 1.0)
        )
      ),
    CONSTRAINT chk_live_model_quality_confusion_matrix
      CHECK (
        true_negative >= 0
        AND false_positive >= 0
        AND false_negative >= 0
        AND true_positive >= 0
        AND true_negative + false_positive + false_negative + true_positive = n_samples
        AND true_positive + false_negative = n_fail_samples
        AND true_negative + false_positive = n_pass_samples
      )
  );

  CREATE INDEX IF NOT EXISTS idx_live_model_quality_model_cutoff
    ON live_model_quality_evaluations (model_run_id, threshold, cutoff_time DESC);

  CREATE INDEX IF NOT EXISTS idx_live_model_quality_status_cutoff
    ON live_model_quality_evaluations (evaluation_status, cutoff_time DESC);

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
    snapshot_version BIGINT NOT NULL,
    feature_hash TEXT NOT NULL,
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
    available_at DOUBLE PRECISION NOT NULL,

    -- (sample, snapshot_version) 묶음은 유일해야 함.
    CONSTRAINT uq_serving_feature_snapshots_sample_version
      UNIQUE (sample_id, snapshot_version),
    CONSTRAINT chk_serving_feature_snapshots_version
      CHECK (snapshot_version > 0),
    CONSTRAINT chk_serving_feature_snapshots_feature_hash
      CHECK (feature_hash ~ '^sha256:v1:[0-9a-f]{64}$'),
    CONSTRAINT chk_serving_feature_snapshots_time
      CHECK (
        snapshot_time >= 0.0
        AND window_start >= 0.0
        AND window_end >= window_start
      ),
    CONSTRAINT chk_serving_feature_snapshots_available_at
      CHECK (available_at >= 0.0),
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

  CREATE INDEX IF NOT EXISTS idx_serving_feature_snapshots_sample_available
    ON serving_feature_snapshots (sample_id, available_at DESC, snapshot_version DESC);

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
    label_revision BIGINT NOT NULL,
    measured_at DOUBLE PRECISION NOT NULL,
    -- Available_at은 DB에 insert 하는 시점에 DB 시간으로 정리
    -- 실제 SECOM Ops 시스템이 확인할 수 있는 시점이 이 시점이기 때문.
    available_at DOUBLE PRECISION NOT NULL
      DEFAULT EXTRACT(EPOCH FROM clock_timestamp()),
    actual_value INTEGER NOT NULL,
    actual_label TEXT NOT NULL,
    CONSTRAINT uq_label_events_sample_revision
      UNIQUE (sample_id, label_revision),
    CONSTRAINT chk_label_events_revision
      CHECK (label_revision > 0),
    CONSTRAINT chk_label_events_measured_at
      CHECK (measured_at >= 0.0),
    CONSTRAINT chk_label_events_available_at
      CHECK (available_at >= 0.0),
    CONSTRAINT chk_label_events_actual_value
      CHECK (actual_value IN (-1, 1)),
    CONSTRAINT chk_label_events_actual_label
      CHECK (actual_label IN ('pass', 'fail'))
  );

  CREATE INDEX IF NOT EXISTS idx_label_events_available_at
    ON label_events (available_at);

  CREATE INDEX IF NOT EXISTS idx_label_events_sample_available_revision
    ON label_events (sample_id, available_at, label_revision DESC);

  CREATE TABLE IF NOT EXISTS dataset_builds (
    dataset_id TEXT PRIMARY KEY,
    dataset_type TEXT NOT NULL,
    dataset_schema_version TEXT NOT NULL,
    selector_version TEXT NOT NULL,
    status TEXT NOT NULL,
    cohort_start_time DOUBLE PRECISION NOT NULL,
    cutoff_time DOUBLE PRECISION NOT NULL,
    label_maturity_seconds DOUBLE PRECISION NOT NULL,
    manifest_hash TEXT NOT NULL,
    mlflow_run_id TEXT,
    artifact_uri TEXT,
    artifact_sha256 TEXT,
    eligible_sample_count BIGINT NOT NULL,
    labeled_sample_count BIGINT NOT NULL,
    unlabeled_sample_count BIGINT NOT NULL,
    label_coverage DOUBLE PRECISION NOT NULL,
    fail_count BIGINT NOT NULL,
    pass_count BIGINT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
      DEFAULT EXTRACT(EPOCH FROM clock_timestamp()),
    updated_at DOUBLE PRECISION NOT NULL
      DEFAULT EXTRACT(EPOCH FROM clock_timestamp()),
    ready_at DOUBLE PRECISION,
    error_message TEXT,
    UNIQUE (dataset_type, manifest_hash)
  );

  CREATE MATERIALIZED VIEW IF NOT EXISTS label_maturity_cohort_age_metrics AS
  WITH refresh_clock AS MATERIALIZED (
    SELECT
      EXTRACT(EPOCH FROM clock_timestamp())::DOUBLE PRECISION AS computed_at
  ),
  age_horizons AS (
    SELECT
      generate_series(0, 10)::INTEGER AS age_minute
  ),
  ranked_complete_snapshots AS (
    SELECT
      s.serving_snapshot_id,
      s.sample_id,
      s.snapshot_version,
      s.available_at AS anchor_at,
      ROW_NUMBER() OVER (
        PARTITION BY s.sample_id
        ORDER BY
          s.available_at ASC,
          s.snapshot_version ASC,
          s.serving_snapshot_id ASC
      ) AS snapshot_rank
    FROM serving_feature_snapshots s
    WHERE s.is_complete = TRUE
      AND s.snapshot_status = 'complete'
      AND s.available_at >= 0.0
      AND s.available_at < 'Infinity'::DOUBLE PRECISION
  ),
  offline_anchors AS (
    SELECT
      'offline'::TEXT AS maturity_scope,
      sample_id AS observation_id,
      sample_id,
      anchor_at,
      NULL::TEXT AS model_run_id
    FROM ranked_complete_snapshots
    WHERE snapshot_rank = 1
  ),
  ranked_release_predictions AS (
    SELECT
      p.prediction_id,
      p.sample_id,
      p.model_run_id,
      p.predicted_at AS anchor_at,
      ROW_NUMBER() OVER (
        PARTITION BY
          p.model_run_id,
          p.threshold,
          p.sample_id,
          p.serving_snapshot_id,
          p.snapshot_version
        ORDER BY
          p.predicted_at ASC,
          p.prediction_id ASC
      ) AS prediction_rank
    FROM prediction_logs p
    WHERE p.runtime_slot = 'release'
      AND p.predicted_at >= 0.0
      AND p.predicted_at < 'Infinity'::DOUBLE PRECISION
  ),
  serving_anchors AS (
    SELECT
      'serving'::TEXT AS maturity_scope,
      prediction_id AS observation_id,
      sample_id,
      anchor_at,
      model_run_id
    FROM ranked_release_predictions
    WHERE prediction_rank = 1
  ),
  first_label_arrivals AS (
    SELECT
      le.sample_id,
      MIN(le.available_at) AS first_label_available_at
    FROM label_events le
    CROSS JOIN refresh_clock r
    WHERE le.available_at <= r.computed_at
      AND le.available_at >= 0.0
      AND le.available_at < 'Infinity'::DOUBLE PRECISION
    GROUP BY le.sample_id
  ),
  all_anchors AS (
    SELECT * FROM offline_anchors
    UNION ALL
    SELECT * FROM serving_anchors
  ),
  anchored_observations AS (
    SELECT
      a.maturity_scope,
      a.observation_id,
      a.sample_id,
      a.anchor_at,
      a.model_run_id,
      f.first_label_available_at,
      FLOOR((a.anchor_at - 60.0) / 600.0) * 600.0 + 60.0 AS cohort_start
    FROM all_anchors a
    CROSS JOIN refresh_clock r
    LEFT JOIN first_label_arrivals f
      ON f.sample_id = a.sample_id
    WHERE a.anchor_at <= r.computed_at
  ),
  cohort_age_counts AS (
    SELECT
      a.maturity_scope,
      a.cohort_start,
      a.cohort_start + 600.0 AS cohort_end,
      h.age_minute,
      COUNT(*) AS cohort_size_seen_at_refresh,
      COUNT(*) FILTER (
        WHERE a.first_label_available_at
          <= a.anchor_at + h.age_minute * 60.0
      ) AS labeled_count,
      NULLIF(COUNT(DISTINCT a.model_run_id), 0) AS model_run_count,
      r.computed_at,
      r.computed_at >= a.cohort_start + 600.0 + h.age_minute * 60.0
        AS is_observable,
      CASE
        WHEN r.computed_at < a.cohort_start + 600.0 THEN 'open'
        WHEN r.computed_at < a.cohort_start + 1200.0 THEN 'observing'
        ELSE 'complete'
      END AS cohort_status
    FROM anchored_observations a
    CROSS JOIN age_horizons h
    CROSS JOIN refresh_clock r
    GROUP BY
      a.maturity_scope,
      a.cohort_start,
      h.age_minute,
      r.computed_at
  )
  SELECT
    maturity_scope,
    cohort_start,
    cohort_end,
    age_minute,
    cohort_end + age_minute * 60.0 AS observable_at,
    computed_at >= cohort_end AS cohort_closed,
    cohort_size_seen_at_refresh,
    CASE
      WHEN computed_at >= cohort_end THEN cohort_size_seen_at_refresh
      ELSE NULL
    END AS cohort_size,
    CASE
      WHEN is_observable THEN labeled_count
      ELSE NULL
    END AS labeled_count,
    CASE
      WHEN is_observable THEN cohort_size_seen_at_refresh - labeled_count
      ELSE NULL
    END AS unlabeled_count,
    CASE
      WHEN is_observable THEN
        ROUND(
          labeled_count::NUMERIC / NULLIF(cohort_size_seen_at_refresh, 0),
          6
        )
      ELSE NULL
    END AS label_coverage,
    model_run_count,
    is_observable,
    cohort_status,
    computed_at
  FROM cohort_age_counts
  WITH DATA;

  CREATE UNIQUE INDEX IF NOT EXISTS uq_label_maturity_scope_cohort_age
    ON label_maturity_cohort_age_metrics (
      maturity_scope,
      cohort_start,
      age_minute
    );

  CREATE INDEX IF NOT EXISTS idx_label_maturity_scope_recent
    ON label_maturity_cohort_age_metrics (
      maturity_scope,
      cohort_start DESC,
      age_minute
    );

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
