#!/bin/bash

# ==============================================================================
# Query the ALCHEMI BGR NIM server running on a GPU node from any login node.
#
# Usage:
#   bash query_nim_from_login.sh <gpu-node-hostname> [port]
#
# Examples:
#   bash query_nim_from_login.sh gpu-node-01
#   bash query_nim_from_login.sh gpu-node-01 8001
# 
# Check the compute nodes running the job by: squeue -u $USER
# ==============================================================================

set -euo pipefail

GPU_NODE="${1:?Usage: $0 <gpu-node-hostname> [port]}"
PORT="${2:-8000}"
BASE_URL="http://${GPU_NODE}:${PORT}"

# ---------- Health check ----------
echo "[INFO] Checking NIM health at: ${BASE_URL}/v1/health/ready"

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/v1/health/ready" 2>/dev/null || echo "000")

if [ "${HTTP_CODE}" = "200" ]; then
    echo "[INFO] NIM service is healthy and ready!"
else
    echo "[WARNING] Health check returned HTTP ${HTTP_CODE}"
    echo "[WARNING] The NIM container may still be starting up."
    echo "[INFO] Check the Slurm log: tail -f nim-mace-server-<jobid>.out"
    exit 1
fi

# ---------- Test inference ----------
echo ""
echo "[INFO] Sending test water molecule for geometry relaxation..."
echo ""

RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/infer" \
  -H 'Content-Type: application/json' \
  -d '{
    "atoms": [
      {
        "structure_id": "test_water_molecule",
        "coord": [
          0.0, 0.0, 0.0,
          0.0, 0.8, 0.6,
          0.0, -0.8, 0.6
        ],
        "numbers": [8, 1, 1],
        "pbc": [false, false, false]
      }
    ]
  }')

echo "Response:"
echo "${RESPONSE}" | python3 -m json.tool 2>/dev/null || echo "${RESPONSE}"

echo ""
echo "[INFO] Query complete."
echo "[INFO] To use the Python client instead:"
echo "  python3 query_mace_nim.py --url ${BASE_URL}"
