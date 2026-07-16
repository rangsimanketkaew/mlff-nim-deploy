#!/bin/bash

# ==============================================================================
# Slurm script to start the NVIDIA ALCHEMI BGR NIM server on a GPU node
# The container runs as a long-lived service for the duration of the allocation
#
# Usage:
#   sbatch run_nim_server.slurm
#
# After submission:
#   - Check logs:       tail -f nim-mace-server-<jobid>.out
#   - Query the server: curl http://<gpu-node>:8000/v1/health/ready
#   - Cancel the job:   scancel <jobid>
#
# To deploy a custom model instead of the default MACE-MP-0:
#   export CUSTOM_MODEL_PATH="/path/to/your/custom.model"
#   sbatch run_nim_server.slurm
# ==============================================================================

#SBATCH -J nim-mace-server
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o nim-mace-server-%j.out
#SBATCH -e nim-mace-server-%j.err

# Uncomment and set the following for email notifications:
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=your_email@example.com

# Uncomment to constrain to a specific GPU model (if your cluster supports it):
# #SBATCH -C "GPU_MODEL:A100"

set -euo pipefail

# ---------- Configuration ----------
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/users/$USER}"
NIM_ROOT="${SCRATCH_BASE}/nim_mace"
NIM_IMAGE="nvcr.io/nim/nvidia/alchemi-bgr:1.0.0"
SIF_FILE="${NIM_ROOT}/alchemi-bgr.sif"
NGC_KEY_FILE="${HOME}/.secrets/ngc_api_key"
NIM_PORT="${NIM_PORT:-8000}"

# ---------- Load modules ----------
echo "[INFO] Loading Singularity module..."
if command -v module &>/dev/null; then
    module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true
fi

# ---------- Read NGC API key ----------
if [ ! -f "${NGC_KEY_FILE}" ]; then
    echo "[ERROR] NGC API key not found at: ${NGC_KEY_FILE}"
    echo "[ERROR] Run setup_nim_env.sh first."
    exit 1
fi

export NGC_API_KEY=$(cat "${NGC_KEY_FILE}")
export SINGULARITY_CACHEDIR="${NIM_ROOT}/singularity_cache"
export SINGULARITY_TMPDIR="${NIM_ROOT}/tmp"
export SINGULARITY_CONFIGDIR="${NIM_ROOT}/singularity_config"
export SINGULARITY_DOCKER_USERNAME='$oauthtoken'
export SINGULARITY_DOCKER_PASSWORD="${NGC_API_KEY}"

# ---------- Setup directories ----------
mkdir -p "${NIM_ROOT}"/{cache,work/{nginx,configs},tmp,singularity_cache,singularity_config,triton}
chmod -R 700 "${NIM_ROOT}"

# ---------- Pull SIF if needed ----------
if [ ! -f "${SIF_FILE}" ]; then
    echo "[INFO] SIF file not found. Downloading container image..."
    singularity pull "${SIF_FILE}" "docker://${NIM_IMAGE}"
else
    echo "[INFO] Using existing SIF file: ${SIF_FILE}"
fi

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

# ---------- Print job info ----------
echo "============================================================"
echo " ALCHEMI BGR NIM Server - Slurm Batch Job"
echo ""
echo " Job ID    : ${SLURM_JOB_ID}"
echo " Node      : $(hostname)"
echo " GPUs      : ${CUDA_VISIBLE_DEVICES:-0}"
echo " Port      : ${NIM_PORT}"
echo " NIM_ROOT  : ${NIM_ROOT}"
echo ""
echo " Query with:"
echo "   curl http://$(hostname):${NIM_PORT}/v1/health/ready"
echo "============================================================"

# ---------- Launch NIM ----------

"${SING_CMD[@]}"
