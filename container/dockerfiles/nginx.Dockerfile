FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install "fastapi>=0.115,<1" "uvicorn[standard]>=0.34,<1"

WORKDIR /opt/model-gateway-admin

COPY nginx/model-gateway.conf /etc/nginx/nginx.conf
COPY nginx/traffic-policies /etc/nginx/traffic-policies
COPY nginx/model-gateway-reload-admin.py ./model_gateway_reload_admin.py
COPY nginx/start-model-gateway.sh /usr/local/bin/start-model-gateway.sh

RUN chmod +x /usr/local/bin/start-model-gateway.sh

CMD ["/usr/local/bin/start-model-gateway.sh"]
