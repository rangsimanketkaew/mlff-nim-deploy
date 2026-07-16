#!/bin/bash

# ==============================================================================
# One-time environment setup for running NVIDIA ALCHEMI BGR NIM on a Slurm
# cluster using Singularity/Apptainer containers.
#
# This script:
#   1. Loads the Singularity module
#   2. Creates secure scratch directories for cache, work, and temp files
#   3. Authenticates with the NGC registry
#   4. Pulls the ALCHEMI BGR container image as a .sif file
#
# Usage:
#   source setup_nim_env.sh
#
# Prerequisites:
#   - NGC API key saved in ~/.secrets/ngc_api_key
#   - Singularity/Apptainer available as a module on the cluster
#   - Scratch filesystem available (e.g. /scratch, /tmp, or similar)
# ==============================================================================

set -euo pipefail

# ---------- Configuration ----------
# Adjust SCRATCH_BASE to your cluster's scratch/fast-storage filesystem.
# Common choices: /scratch/users/$USER, /scratch/shared/$USER, /tmp/$USER
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/users/$USER}"
NIM_ROOT="${SCRATCH_BASE}/nim_mace"
NIM_IMAGE="nvcr.io/nim/nvidia/alchemi-bgr:1.0.0"
SIF_FILE="${NIM_ROOT}/alchemi-bgr.sif"
NGC_KEY_FILE="${HOME}/.secrets/ngc_api_key"

# ---------- Step 1: Load Singularity ----------
echo "[INFO] Loading Singularity module..."
if command -v module &>/dev/null; then
    module load singularity 2>/dev/null || module load apptainer 2>/dev/null || {
        echo "[WARNING] Could not load singularity or apptainer module."
        echo "[WARNING] Assuming Singularity/Apptainer is already in PATH."
    }
fi

if ! command -v singularity &>/dev/null && ! command -v apptainer &>/dev/null; then
    echo "[ERROR] Neither singularity nor apptainer found in PATH."
    echo "[ERROR] Please install or load the appropriate module."
    exit 1
fi

# ---------- Step 2: Set up directories ----------
echo "[INFO] Setting up NIM directories under: ${NIM_ROOT}"
export NIM_ROOT
export SINGULARITY_CACHEDIR="${NIM_ROOT}/singularity_cache"
export SINGULARITY_TMPDIR="${NIM_ROOT}/tmp"
export SINGULARITY_CONFIGDIR="${NIM_ROOT}/singularity_config"

mkdir -p "${NIM_ROOT}"/{cache,work/{nginx,configs},tmp,singularity_cache,singularity_config,triton}
chmod -R 700 "${NIM_ROOT}"

echo "[INFO] NIM_ROOT          = ${NIM_ROOT}"
echo "[INFO] SINGULARITY_CACHE = ${SINGULARITY_CACHEDIR}"

# ---------- Step 3: Read NGC API key ----------
if [ ! -f "${NGC_KEY_FILE}" ]; then
    echo "[ERROR] NGC API key file not found at: ${NGC_KEY_FILE}"
    echo "[INFO]  Create it with:"
    echo "          mkdir -p ~/.secrets"
    echo "          echo 'your-ngc-api-key' > ~/.secrets/ngc_api_key"
    echo "          chmod 600 ~/.secrets/ngc_api_key"
    exit 1
fi

export NGC_API_KEY=$(cat "${NGC_KEY_FILE}")
export SINGULARITY_DOCKER_USERNAME='$oauthtoken'
export SINGULARITY_DOCKER_PASSWORD="${NGC_API_KEY}"

echo "[INFO] NGC API key loaded from ${NGC_KEY_FILE}"

# ---------- Step 4: Pull the container image ----------
if [ -f "${SIF_FILE}" ]; then
    echo "[INFO] SIF file already exists: ${SIF_FILE}"
    echo "[INFO] Skipping download. Delete the file to force re-download."
else
    echo "[INFO] Pulling ALCHEMI BGR NIM container..."
    echo "[INFO] Image: ${NIM_IMAGE}"
    echo "[INFO] Target: ${SIF_FILE}"
    echo "[INFO] This may take several minutes depending on network speed..."
    singularity pull "${SIF_FILE}" "docker://${NIM_IMAGE}"
    echo "[INFO] Container image saved to: ${SIF_FILE}"
fi

echo ""
echo "============================================================"
echo " Environment setup complete!"
echo ""
echo " NIM_ROOT : ${NIM_ROOT}"
echo " SIF file : ${SIF_FILE}"
echo ""
echo " Next steps:"
echo "   - Interactive: srun to request a GPU, then run launch_nim_interactive.sh"
echo "   - Batch:       sbatch run_nim_server.slurm"
echo "============================================================"
