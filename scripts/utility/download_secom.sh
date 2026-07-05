#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"
SECOM_URL="https://archive.ics.uci.edu/static/public/179/secom.zip"

mkdir -p "${RAW_DIR}"

curl -L "${SECOM_URL}" -o "${RAW_DIR}/secom.zip"
unzip -o "${RAW_DIR}/secom.zip" -d "${RAW_DIR}"

echo "Downloaded SECOM dataset to ${RAW_DIR}"
