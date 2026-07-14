#### Scenario 4: 전체 Feature 관계 반전

Candidate 재학습과 Serving Gate를 강하게 검증하려면 Scenario 4를 사용할 수 있습니다.

```shell
$ ./scripts/scenarios/scenario4.sh
```

Scenario 4는 cursor를 초기화한 뒤 다음 순서로 데이터를 발행합니다.

```text
Baseline   6,000 samples
Drift     60,000 samples
Recovery   6,000 samples
```

Drift 구간에서는 590개 Feature의 non-null 값을 Feature별 raw SECOM
median을 기준으로 반전합니다.

```text
x' = 2 × median - x
```

`NaN`은 그대로 유지합니다. Label workload는 60초 뒤, prediction workload는
90초 뒤 시작하여 complete serving snapshot이 먼저 생성되도록 합니다.

이 시나리오는 Candidate가 변화한 Feature/Label 관계를 다시 학습하고 기존
Champion과 비교되는 흐름을 확인하기 위한 강한 stress scenario입니다.
현실적인 반도체 공정 drift의 크기나 분포를 재현하는 시나리오는 아닙니다.