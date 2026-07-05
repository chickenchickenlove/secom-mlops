FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY pyproject.toml ./
COPY secom_mlops ./secom_mlops
COPY secom_mlops_common ./secom_mlops_common
COPY scripts ./scripts

RUN pip install --upgrade pip && pip install .

EXPOSE 8091
