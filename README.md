# SECOM / FDC MLOps Platform

반도체 공정 이상 탐지를 대상으로 만든 end-to-end MLOps 플랫폼입니다. 
Kafka 기반 이벤트 파이프라인, Valkey online feature store, FastAPI serving, MLflow model registry, Airflow control plane, Prometheus/Grafana monitoring을 로컬 Docker Compose 환경에서 재현 가능하게 구성했습니다.
이 프로젝트는 ML LifeCycle 관점에서 **feature 수집 -> online serving -> prediction logging -> monitoring -> candidate 학습 -> canary 배포 -> promotion/rollback**까지 이어지는 운영형 ML workflow를 구현하는 것을 목표로 합니다.  

## 핵심 요약

- Kafka topic을 feature patch, feature state update, label event, prediction event로 분리했습니다.
- Kafka streams app과 archiver가 raw feature 저장, feature assembly, snapshot 저장, Valkey materialization을 담당합니다.
- `serving-api`는 Valkey에서 최신 online feature snapshot을 읽고 `model-gateway`를 통해 model runtime을 호출합니다.
- 예측 결과는 Kafka prediction event로 발행되고 `prediction-log-archiver`가 PostgreSQL `prediction_logs`에 저장합니다.
- Airflow DAG가 candidate 학습, gate 평가, canary 배포, traffic split, release promotion, rollback을 제어합니다.
- Candidate 학습은 PostgreSQL evidence table에서 point-in-time feature를 on demand로 재구성하여 학습합니다.
- Candidate Gate는 serving feature snapshot과 actual labels를 읽어서 champion과 비교합니다.
- Grafana는 PostgreSQL metric table과 Prometheus Kafka metric을 함께 시각화합니다.

## 기술 스택

| 영역 | 사용 기술                                                   |
| --- |---------------------------------------------------------|
| Streaming | Kafka                                                   |
| Stream app | Java, Kafka Streams, Kafka Consumer                     |
| Online feature store | Valkey                                                  |
| Orchestration | Airflow                                                 |
| Model registry | MLflow                                                  |
| Serving | FastAPI, Uvicorn, Nginx model-gateway                   |
| Storage | PostgreSQL                                              |
| Monitoring | Prometheus, Grafana, Kafka JMX exporter, kafka-exporter |
| ML / Python | Python 3.11, uv, pandas, scikit-learn, MLflow           |

## 아키텍처

아키텍처는 세 개의 관점으로 나누어 설명합니다.

| 관점 | 설명 |
| --- | --- |
| Online Serving Path | feature event부터 online prediction과 prediction logging까지의 runtime 경로 |
| MLOps Control Plane | Airflow 기반 candidate 학습, canary 배포, promotion, rollback 흐름 |
| Monitoring Feedback Loop | prediction, label, drift, Kafka metric이 Grafana로 모이는 흐름  |

### Tutoreial
실행 방법은 [HOW_TO.md](./HOW_TO.md)를 참고하세요.

## Online Serving Path
![online-serving-path](docs/images/online-serving-path.png)

Feature online path는 Kafka event를 기준으로 구성됩니다.

```text
Workload scripts
-> secom-feature-patches
-> feature-assembler
-> secom-feature-state-updates
-> feature-materializer
-> Valkey online_feature_snapshot:{sample_id}
-> serving-api /predict-by-id
-> model-gateway
-> model-server-release / canary 
```
- Prediction evidence는 serving API가 직접 DB에 쓰지 않고 Kafka event로 남깁니다.
- 현재 Shadow로 트래픽을 전송하여, Shadow로부터 오는 예측 결과를 클라이언트에게 응답하지 않고 평가에만 반영하는 것은 미구현 상태입니다. 


```text
serving-api
-> secom-prediction-events
-> prediction-log-archiver
-> PostgreSQL prediction_logs
```

- Feature와 label evidence는 별도 archiver가 PostgreSQL에 저장합니다.

```text
secom-feature-patches -> feature-raw-archiver -> PostgreSQL feature_events
secom-label-events    -> label-archiver       -> PostgreSQL actual_labels
```

## MLOps Control Plane
![mlops-control-plane](docs/images/mlops-control-plane.png)
- Airflow는 모델 lifecycle을 제어합니다.

```text
Airflow
-> point-in-time feature 재구성
-> candidate model 학습
-> MLflow candidate 등록
-> candidate gate 평가
-> canary slot 배포
-> model-gateway canary traffic 조정
-> release promotion 또는 rollback
```

현재 candidate 학습 경로의 중요한 특징은 **on-demand point-in-time 재구성**입니다.

```text
PostgreSQL evidence tables
  serving_feature_snapshots
  actual_labels
  feature_events

-> point-in-time features 재구성
-> candidate model 학습
```

현재 main Airflow training path는 `offline_feature_snapshots`를 source로 읽지 않습니다. 
별도 utility로 `offline_feature_snapshots`를 저장할 수는 있지만, 실제 candidate 학습 DAG는 PostgreSQL evidence table에서 필요한 시점의 feature를 즉석으로 재구성합니다.

주요 DAG:

```text
train_candidate_from_offline_point_in_time_features
record_serving_candidate_deployment_request
inspect_deployment_requests
evaluate_candidate_serving_snapshot_gate
deploy_candidate_to_canary
set_model_gateway_canary_traffic
promote_candidate_to_release
rollback_candidate_canary
rollback_release_deployment
cleanup_failed_candidate_alias
create_fixed_reference_drift_baseline
fixed_reference_drift_evaluator
```

## Monitoring Feedback Loop
![monitoring-feedback-loop.png](docs/images/monitoring-feedback-loop.png)
- Monitoring은 PostgreSQL evidence table과 Kafka operational metric을 함께 사용합니다.
```text
prediction_logs + actual_labels
-> metrics-evaluator
-> model_metrics / prediction_window_metrics

prediction_logs + actual_labels
-> quality-window-evaluator
-> model_quality_windows

prediction_logs.features_json
-> drift-metrics-evaluator
-> drift_metrics

prediction_logs
-> fixed-reference drift baseline / evaluator DAGs
-> drift_reference_baselines / drift_reference_stats / drift_metrics
```
- Kafka 운영 지표는 Prometheus를 거쳐 Grafana로 들어갑니다.

## Runtime Services

로컬 stack은 `container/docker-compose.yml`에 정의되어 있습니다.

| 서비스 | 역할 | 로컬 포트 |
| --- | --- | --- |
| PostgreSQL | monitoring DB, MLflow DB, Airflow DB | `5432` |
| Kafka | event backbone | `9092` |
| Valkey | online feature store | `6379` |
| MLflow | tracking server / registry | `5100` |
| Airflow webserver | control plane UI | `8081` |
| serving-api | online prediction API | `8080` |
| model-gateway | model runtime gateway | `8090` |
| model-gateway admin | traffic/reload admin API | `18080` |
| model-server-release | release runtime | `28091` |
| model-server-canary | canary runtime | `28092` |
| model-server-shadow | shadow runtime | `28093` |
| Prometheus | metric store | `9090` |
| Grafana | dashboard | `3000` |
| kafka-exporter | Kafka consumer lag metric | `9308` |

Stream app과 daemon:

```text
feature-raw-archiver
feature-assembler
feature-snapshot-archiver
feature-materializer
label-archiver
prediction-log-archiver
metrics-evaluator
quality-window-evaluator
drift-metrics-evaluator
```

## 주요 데이터

PostgreSQL 주요 table:

| 분류 | Tables |
| --- | --- |
| Evidence | `feature_events`, `serving_feature_snapshots`, `actual_labels`, `prediction_logs` |
| Optional offline utility | `offline_feature_snapshots`, `offline_prediction_logs` |
| Monitoring | `model_metrics`, `prediction_window_metrics`, `model_quality_windows`, `drift_metrics` |
| Drift baseline | `drift_reference_baselines`, `drift_reference_stats` |
| Deployment state | `model_deployment_requests`, `model_runtime_deployment_state`, `model_runtime_reload_events` |

Kafka topics:

```text
secom-feature-patches
secom-feature-state-updates
secom-label-events
secom-prediction-events
```

## 설계 포인트

### Event evidence 중심 설계
Feature patch, label, prediction을 event evidence로 남긴 뒤, monitoring과 candidate 학습이 이를 읽는 구조입니다. 
Serving 결과를 재현하고, 특정 시점의 feature 상태를 다시 구성할 수 있습니다.

### Online store와 durable store 분리
Valkey는 online serving에 필요한 최신 feature snapshot만 보관합니다. 
PostgreSQL은 feature, label, prediction, metric, deployment state를 보관하는 durable evidence store 역할을 합니다.

### On-demand point-in-time feature 재구성
Candidate 학습은 현재 `offline_feature_snapshots`를 source로 삼지 않고 아래 table을 읽어 point-in-time feature를 즉석에서 재구성합니다.
이 방식은 training 시점의 feature 누수를 줄이고, online serving에서 실제로 관측된 snapshot을 기준으로 candidate를 평가하기 좋습니다.
```text
serving_feature_snapshots
actual_labels
feature_events
```

### Gateway 기반 release control
Airflow는 model runtime을 직접 조작하지 않고 `model-gateway` admin API를 호출합니다. 
Gateway는 release, canary, shadow runtime으로 traffic을 분리하고 reload를 제어합니다.
그러나 현재 shadow는 upstream으로 등록되어있으나 shadow 평가 경로가 미구현되어, 실제 트래픽 조절은 release / canary 사이에서만 처리되고 있습니다.

### Monitoring feedback
Prediction/label evidence에서 model metric, quality window, drift metric을 계산하고, Kafka JMX/exporter metric과 함께 Grafana에서 확인합니다. 
이 시그널은 canary promotion과 rollback 판단 자료로 활용할 수 있습니다.

## Repository Map

```text
airflow/dags/                         Airflow DAG definitions
container/docker-compose.yml           Local MLOps stack
container/nginx/                       model-gateway config and admin API
container/postgres/monitoring-schema.sql
                                      PostgreSQL monitoring schema
feature-raw-archiver/                  Kafka -> feature_events
feature-snapshot-archiver/             Kafka -> serving_feature_snapshots
fdc-feature-assembler/                 feature patch aggregation
fdc-feature-materializer/              Kafka -> Valkey online store
label-archiver/                        Kafka -> actual_labels
prediction-log-archiver/               Kafka -> prediction_logs
secom_mlops/serving/                   serving API and model runtime
scripts/training/                      candidate training scripts
scripts/monitoring/                    metric, quality, drift evaluators
scripts/deployment/                    deployment and rollback helpers
scripts/workload/                      local workload generators
```

## 현재 범위
이 프로젝트는 cloud production deployment가 아니라, 로컬에서 재현 가능한 포트폴리오용 MLOps platform입니다. 

