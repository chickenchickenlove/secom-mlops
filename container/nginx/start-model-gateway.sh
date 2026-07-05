#!/bin/sh
set -eu

runtime_conf="${NGINX_RUNTIME_CONF:-/etc/nginx/runtime/model-production-upstream.conf}"
default_policy="${NGINX_DEFAULT_POLICY_CONF:-/etc/nginx/traffic-policies/canary-000.conf}"
admin_host="${NGINX_RELOAD_ADMIN_HOST:-0.0.0.0}"
admin_port="${NGINX_RELOAD_ADMIN_PORT:-18080}"

mkdir -p "$(dirname "$runtime_conf")"

if [ ! -f "$runtime_conf" ]; then
  cp "$default_policy" "$runtime_conf"
  echo "model_gateway_runtime_conf_initialized source=$default_policy target=$runtime_conf"
fi

nginx -t

cd /opt/model-gateway-admin
python3 -m uvicorn model_gateway_reload_admin:app --host "$admin_host" --port "$admin_port" &

exec nginx -g "daemon off;"
