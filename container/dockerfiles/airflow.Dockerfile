FROM apache/airflow:2.10.5-python3.11

USER airflow

RUN pip install --no-cache-dir \
    "psycopg2-binary>=2.9,<3" \
    "psycopg[binary]>=3.2,<4" \
    "httpx>=0.28,<1" \
    "mlflow==3.14.0" \
    "numpy==1.26.0" \
    "pandas>=2.2,<3" \
    "scikit-learn==1.9.0"

ENV PYTHONPATH=/opt/airflow/mlops:/opt/airflow/mlops/scripts
