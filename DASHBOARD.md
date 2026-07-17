# Fail Recall - Active Release (패널 43)

현재 Active Release가 실제 Fail을 얼마나 놓치지 않고 탐지하는지 보여주는 슬라이딩 품질 지표입니다.

```text
Fail Recall = True Positive / (True Positive + False Negative)
```

- 값이 높을수록 실제 Fail을 더 많이 탐지합니다.
- 값이 낮아지면 실제 Fail을 Pass로 잘못 판단한 False Negative가 증가했음을 의미합니다.
- Precision, F1, Confusion Matrix와 함께 확인해야 오탐과 미탐을 구분할 수 있습니다.

## 시간 기준

패널의 각 점은 evaluator가 `cutoff_time = T`에 계산하여 `live_model_quality_evaluations`에 저장한 Recall입니다.

현재 설정은 다음과 같습니다.

```text
label maturity(L)    = 60초
monitoring window(W) = 600초
prediction window    = [T-L-W, T-L)
                     = [T-11분, T-1분)
```

예를 들어 cutoff 시각이 10:00이면 09:49 이상 09:59 미만에 발생한 prediction을, 10:00까지 도착한 라벨과 결합하여 평가합니다. 따라서 이 패널은 현재 시각보다 최소 1분 성숙한 최근 10분 prediction window의 품질을 보여줍니다.

Grafana의 `__from`과 `__to`는 이미 계산되어 저장된 evaluation 중 화면에 표시할 `cutoff_time` 범위만 선택합니다. Prediction window나 label maturity를 변경하지 않습니다.

## 유효성 조건

Label maturity 60초는 라벨 커버리지 100%를 보장하지 않고, 각 prediction에 최소 60초의 라벨 도착 시간을 제공합니다.

현재 evaluator는 다음 기준을 모두 통과한 경우에만 Recall 값을 기록합니다.

```text
최소 decision 수       = 500
최소 label coverage    = 0.95
최소 Fail 라벨 수      = 20
최소 Pass 라벨 수      = 20
```

Recall 값이 없으면 `Active Release Evaluated Samples` 또는 `Active Release Latest Labeled Quality`에서 `evaluation_status`, `label_coverage`, 표본 수를 먼저 확인합니다.
