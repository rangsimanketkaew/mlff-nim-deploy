#!/bin/bash

# ==============================================================================
# Launch the NVIDIA ALCHEMI BGR NIM container interactively on a GPU node.
#
# This script should be executed AFTER you have:
#   1. Run setup_nim_env.sh to pull the container image
#   2. Obtained a GPU allocation via srun, e.g.:
#      srun -p gpu --mem=32G -c 8 -G 1 -t 4:00:00 --pty /bin/bash
#
# Usage (on the GPU node):
#   bash launch_nim_interactive.sh
#
# Options (via environment variables):
#   SCRATCH_BASE       - Base scratch directory (default: /scratch/users/$USER)
#   NIM_PORT           - Port to serve on (default: 8000)
#   CUSTOM_MODEL_PATH  - Path to custom MACE .model file (optional)
# ==============================================================================

set -euo pipefail

# ---------- Configuration ----------
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/users/$USER}"
NIM_ROOT="${SCRATCH_BASE}/nim_mace"
SIF_FILE="${NIM_ROOT}/alchemi-bgr.sif"
NGC_KEY_FILE="${HOME}/.secrets/ngc_api_key"
NIM_PORT="${NIM_PORT:-8000}"

# ---------- Validate prerequisites ----------
if [ ! -f "${SIF_FILE}" ]; then
    echo "[ERROR] SIF file not found: ${SIF_FILE}"
    echo "[INFO]  Run setup_nim_env.sh first to pull the container image."
    exit 1
fi

if [ ! -f "${NGC_KEY_FILE}" ]; then
    echo "[ERROR] NGC API key not found at: ${NGC_KEY_FILE}"
    exit 1
fi

export NGC_API_KEY=$(cat "${NGC_KEY_FILE}")

# ---------- Build Singularity command ----------
SING_CMD=(
    singularity run --nv --cleanenv
    --writable-tmpfs
    --bind "${NIM_ROOT}/cache:/opt/nim/.cache"
    --bind "${NIM_ROOT}/work/nginx:/opt/nim/nginx"
    --bind "${NIM_ROOT}/work/configs:/opt/nim/generated_configs"
    --bind "${NIM_ROOT}/triton:/tmp/triton_cache"
    --env TRITON_CACHE_DIR="/tmp/triton_cache"
    --env NGC_API_KEY="${NGC_API_KEY}"
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    --env NIM_HTTP_API_PORT="${NIM_PORT}"
)

# ---------- Custom model support ----------
if [ -n "${CUSTOM_MODEL_PATH:-}" ]; then
    if [ ! -f "${CUSTOM_MODEL_PATH}" ]; then
        echo "[ERROR] Custom model file not found: ${CUSTOM_MODEL_PATH}"
        exit 1
    fi

    echo "[INFO] Deploying custom MACE model: ${CUSTOM_MODEL_PATH}"
    SING_CMD+=(
        --bind "${CUSTOM_MODEL_PATH}:/opt/nim/.cache/mace.model:ro"
        --env NIM_DISABLE_MODEL_DOWNLOAD=true
        --env ALCHEMI_NIM_MODEL_TYPE="mace"
        --env ALCHEMI_NIM_MODEL_PATH="/opt/nim/.cache/mace.model"
    )
else
    echo "[INFO] Deploying default MACE-MP-0 model"
    SING_CMD+=(
        --env ALCHEMI_NIM_MODEL_TYPE="mace"
    )
fi

SING_CMD+=("${SIF_FILE}")

# ---------- Launch ----------
echo "[INFO] Starting ALCHEMI BGR NIM container..."
echo "[INFO] Serving on port: ${NIM_PORT}"
echo "[INFO] GPU devices: ${CUDA_VISIBLE_DEVICES:-0}"
echo "[INFO] Press Ctrl+C to stop the server."
echo ""

"${SING_CMD[@]}"
