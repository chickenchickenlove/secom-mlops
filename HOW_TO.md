## Prerequisites
```shell
$ mise trust
$ mise install python

# install uv
$ brew install uv

# Install venv and dependencies
$ uv sync
```
- docker desktop or rancher desktop

## Tutorials
### 0. Download `SECOM` data
```shell
$  scripts/utility/download_secom.sh
```


### 1. Start docker-compose
```shell
$ cd container
$ docker-compose build
$ docker-compose up -d
```

### 2. Produce feature, labels, predict requests
```shell
$ cd ..
$ pwd
~/secom-mlops

$ ./scripts/scenarios/scenario2.sh
```
- early, middle, late feature producer, label producer, prediction request producer를 실행합니다.
  - feature drift를 유발하기 위해 초기 record에는 offset이 주어집니다.
- Prediction producer는 complete snapshot이 쌓일 시간을 주기 위해 약 30초 뒤 시작됩니다.
- 중지하려면 터미널에서 Ctrl+C를 누릅니다.

#### Scenario documentation

각 Drift 시나리오의 목적과 실행 방법은 [SCENARIO.md](./SCENARIO.md)를 참고하세요.

### 3. Access the dashboard.
- http://localhost:3000에 접속하셔서 `Monitoring` dashboard로 접근해주세요.
- Kafka로 메세지는 공급되고 있습니다. 
- 다음 영역에서 라벨 도착 속도를 확인할 수 있습니다.
  - `Label Maturity / Fixed 10-Min Cohorts / Offline Training`
  - `Label Maturity / Fixed 10-Min Cohorts / Serving Gate`

### 4. Check runtime evidence
- Grafana에서 prediction count, latency, model metrics, Kafka lag가 증가하는지 확인합니다.
- Airflow의 `refresh_label_maturity_metrics` 작업이 1분마다 자동으로 결과를 갱신하므로 수동으로 실행할 필요는 없습니다.
- 10분 구간은 관측 대상을 모으는 시간입니다. 구간이 끝난 뒤에는 마지막 대상의 라벨도 최대 10분을 기다려야 하므로 관측이 계속됩니다.
  - `open`: 10분 구간에 관측 대상을 모으고 있습니다.
  - `observing`: 대상 수는 확정됐고 라벨 도착을 기다리고 있습니다.
  - `complete`: 마지막 대상까지 10분 관측을 마쳤습니다.
- 예를 들어 `00:01~00:10` 구간은 `00:11`부터 age 0분을 확인하고 `00:21`에 age 10분 관측을 마칩니다.
- Useful URLs:
  - Grafana: http://localhost:3000
  - Airflow: http://localhost:8081
  - MLflow: http://localhost:5100
  - prediction gateway health: http://localhost:8080/health
  - prediction gateway admin health: http://localhost:18080/health

### 5. Train a candidate model
- 라벨이 충분히 공급된 이후 작업이 필요합니다. `runtime/online_workload_next_feature_*_state.json`, `runtime/online_workload_next_label_state.json`의 index가 1000 이상인 경우 시도해주세요.  
- Airflow UI에서 `train_candidate_from_offline_point_in_time_features` DAG를 실행합니다.
- Important params:
  - `dry_run`: `False`
  - `cohort_start_time`, `cutoff_time`, `label_maturity_seconds`: first-complete snapshot 학습 범위와 label cutoff
  - `min_samples`: labeled 개발 원본의 최소 크기입니다. 기본값은 1,000이며 Candidate가 선택하는 최대 eligible snapshot 수도 1,000입니다.
  - `min_label_coverage`: 최신 eligible snapshot을 먼저 선택한 뒤 계산하는 최소 label coverage입니다. 기본값은 `0.95`입니다.
  - `min_fail_samples`, `min_pass_samples`: 각 label class의 최소 학습 표본입니다.
- 완료 시, MLflow에 모델이 등록됩니다.
![mlflow.png](docs/images/mlflow.png)

### 6. Evaluate candidate gate
- Airflow UI에서 `evaluate_candidate_serving_snapshot_gate` DAG를 실행합니다.
- Important params:
  - `dry_run`: `False`
  - `candidate_version`: empty이면 MLflow `candidate` alias를 사용합니다.
  - `cohort_start_time`, `cutoff_time`, `label_maturity_seconds`: Champion model run의 prediction decision 평가 범위와 label cutoff
  - `max_decisions`: 최신 canonical Champion decision을 선택하는 최대 개수입니다. 기본값은 1,000입니다.
  - `min_decisions`, `min_label_coverage`: quick demo에서는 실제 데이터 양에 맞게 낮출 수 있습니다.
  - `min_fail_samples`, `min_pass_samples`: 각 label class의 최소 평가 표본입니다.
- Gate는 Champion model run의 실제 prediction decision이 참조한 exact snapshot/version/hash를 복원하고, `cutoff_time`까지 도착한 최신 label revision을 사용합니다.
- 현재 Gate는 `runtime_slot='release'`와 실제 release threshold를 아직 강제하지 않습니다.
- `passed`일 때만 DAG가 성공합니다. `failed`, `insufficient_data`, snapshot/hash integrity 오류는 DAG 실패로 끝납니다.
![passed.png](docs/images/passed.png)


### 7. Record deployment request
- Gate가 통과한 뒤 `record_serving_candidate_deployment_request` DAG를 실행합니다.
- FYI. kubernetes 환경에서는 ArgoCD 등을 이용한 GitOps로 전환 가능합니다.
- Important params:
  - `dry_run`: `False`
  - `approval_status`: `approved`

### 8. Inspect deployment requests
- `record_serving_candidate_deployment_request` DAG가 성공하면 `model_deployment_requests`에 request가 저장됩니다.
- Airflow UI에서 `inspect_deployment_requests` DAG를 실행하거나, 로컬에서 아래 명령으로 확인합니다.
```shell
$ uv run python scripts/utility/inspect_deployment_requests.py
```
- 출력에서 배포할 request_id를 확인합니다.
```shell
  request_id=abcdefg
  source_version=...
  approval_status=approved
  rollout_status=not_started
```
- Airflow UI 실행 결과는 다음과 같습니다. 이곳에서 request id `915a7016-ae32-4ee7-ae08-9a7e1becbe7d`를 얻습니다. 이후에 사용하는 값입니다.
![airflow.png](docs/images/airflow.png)


### 9. Deploy candidate to canary
- Airflow UI에서 `deploy_candidate_to_canary` DAG를 실행합니다.
- Important params:
  - `dry_run`: `False`
  - `request_id`: 앞 단계에서 확인한 request_id. 이 값은 필수입니다.

### 10. Shift canary traffic
- Canary runtime에 candidate가 올라간 뒤, `set_model_gateway_canary_traffic` DAG를 실행합니다.
- Recommended demo values:
  - `canary_percent`: `1`, `5`, `10`, `50`, `100`
  - `dry_run`: `False`
- 처음에는 `canary_percent=10` 정도로 시작하고, Grafana에서 prediction volume, latency, model quality, drift 지표를 확인합니다.
- 현재 gateway traffic split은 release/canary 사이에서만 동작합니다. Shadow runtime은 별도 route는 있지만 자동 shadow evaluation path는 아직 미구현입니다.
- Gateway admin API로 결과를 직접 확인할 수도 있습니다.
```shell
$ curl http://localhost:18080/admin/traffic-policy
```


### 11. Inspect traffic policy and deployment state
- Traffic 변경 후 `inspect_deployment_requests` DAG를 다시 실행합니다.
- 또는 로컬에서 아래 명령으로 확인합니다.
```shell
$ uv run python scripts/utility/inspect_deployment_requests.py
```
- 확인할 항목:
  - request_id
  - rollout_status
  - runtime_slot=canary
  - active canary model version / run id
  - gateway canary_percent

### 12. Promote candidate to release
- Canary 결과가 괜찮으면 `promote_candidate_to_release` DAG를 실행합니다.
- Important params:
  - `dry_run`: `False`
  - `request_id`: 앞에서 확인한 request_id
  - `reset_canary_traffic`: `True`
- 이 DAG는 candidate를 release runtime으로 reload하고, MLflow champion alias를 candidate version으로 이동한 뒤, canary traffic을 0%로 되돌립니다.

### 13. Verify release promotion
- Promotion 후 상태를 다시 확인합니다.
```shell
$ uv run python scripts/utility/inspect_deployment_requests.py
```
- 확인할 항목:
  - rollout_status=deployed
  - runtime_slot=release
  - release active model version / run id
  - gateway canary_percent=0
- MLflow UI에서도 champion alias가 새 candidate version으로 이동했는지 확인합니다.
  - http://localhost:5100

### 14. Rollback if needed
- Canary 단계에서 문제가 있으면 `rollback_candidate_canary` DAG를 실행합니다.
- Important params:
  - `dry_run`: `False`
  - `request_id`: `rollback`할 deployment request id
- Release promotion 이후 문제가 있으면 `rollback_release_deployment` DAG를 실행합니다.
- Important params:
  - `dry_run`: `False`
  - `reset_canary_traffic`: `True`
- Rollback 후 다시 확인합니다.
```shell
$ uv run python scripts/utility/inspect_deployment_requests.py
$ curl http://localhost:18080/admin/traffic-policy
```
