# Deploy Pretrained, Optimized MLFF Model with NVIDIA ALCHEMI NIM Models

## Q & A

<details>
<summary>What is NVIDIA NIM?</summary>
NVIDIA NIM (NVIDIA Inference Microservices) is a container-based AI models that were already pretrained and optimized, and ready to be used for high-performance inference at scale. It provides a standardized way to deploy and serve AI models, including LLMs, vision models, and also scientific ML models like machine learning force field (MLFF) for quantum chemistry purpose. 
</details>

<details>
<summary>What is NVIDIA ALCHEMI NIM?</summary>
NVIDIA ALCHEMI NIM is a NIM that is specialized for MLFF models. That is, it is a high-level API wrapper for running MLFF models for quantum chemistry purpose.
</details>

<details>
<summary>Limitation of NVIDIA NIM</summary>
The model must use an architecture supported by the underlying engines (TensorRT-LLM or vLLM) optimized by NVIDIA team. If you have designed a completely proprietary, non-standard neural network architecture from scratch that isn't supported by these frameworks, you won't be able to run it inside the standard NIM LLM container without deep custom development. But for 99% of custom, fine-tuned, or open-weight models, NIM can run them smoothly.  
</details>

<details>
<summary>I build my own MLFF from scratch, can I deplot it with NVIDIA NIM?</summary>
The answer is NO. you cannot deploy a completely custom, from-scratch MLFF (Machine Learning Force Field) / MLIP (Machine Learning Interatomic Potential) with a completely new architecture directly inside the standard NVIDIA NIM container. Nonetheless, you could do so but you need a deep, low-level customization.
</details>

<details>
<summary>How to get NVIDIA API key?</summary>
Go to https://build.nvidia.com/explore/discover to generate your API key.
</details>

## Table of Contents

- [Q & A](#q--a)
- [1. Deploying NVIDIA NIM MACE model](#1-deploying-nvidia-nim-mace-model)
  - [Prerequisites](#prerequisites)
  - [Step 1: Deploy the NIM Container](#step-1-deploy-the-nim-container)
  - [Step 2: Verify Status](#step-2-verify-status)
  - [Step 3: Run Inference (Geometry Relaxation)](#step-3-run-inference-geometry-relaxation)
  - [Step 4: Python Query Client](#step-4-python-query-client)
- [2. Deploying model-free NVIDIA NIM MACE models](#2-deploying-model-free-nvidia-nim-mace-models)
  - [Step 1: Fine-Tuning a MACE Model](#step-1-fine-tuning-a-mace-model)
  - [Step 2: Deploying the Custom MACE Model](#step-2-deploying-the-custom-mace-model)
  - [Step 3: Deployment Automation Script](#step-3-deployment-automation-script)
- [3. Self-Hosting and deploying NVIDIA ALCHEMI NIM models on Local HPC Cluster](#3-self-hosting-and-deploying-nvidia-alchemi-nim-models-on-local-hpc-cluster)
  - [Directory Structure of Scripts](#directory-structure-of-scripts)
  - [Step 1: Environment and Security Setup](#step-1-environment-and-security-setup)
  - [Step 2: Running NIM Interactively](#step-2-running-nim-interactively)
  - [Step 3: Deploying as a Slurm Batch Job](#step-3-deploying-as-a-slurm-batch-job)
  - [Step 4: Querying the NIM from a Login Node](#step-4-querying-the-nim-from-a-login-node)


## 1. Deploying NVIDIA NIM MACE model

NVIDIA ALCHEMI NIM provides optimized containers for MLFF models. The primary service for geometry optimization is the **Batched Geometry Relaxation (BGR)** NIM, which supports the foundational **MACE-MP-0** interatomic potential model by default.

#### Prerequisites

- NVIDIA GPU: Compute Capability 8.0+ (Ampere, Ada Lovelace, Hopper, or newer)
- NVIDIA Container Toolkit: Installed and configured for Docker/Podman
- NGC API Key: Required to authenticate with the NVIDIA GPU Cloud (NGC) registry and download the MACE model weights

#### Step 1: Deploy the NIM Container

Run the following Docker command to pull and run the ALCHEMI BGR NIM container:

```bash
export NGC_API_KEY="your_ngc_api_key_here"

docker run --gpus all \
  --rm \
  -p 8000:8000 \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e ALCHEMI_NIM_MODEL_TYPE="mace" \
  --shm-size=8g \
  nvcr.io/nim/nvidia/alchemi-bgr:1.0.0
```

*Note: Replace `1.0.0` with the target release version of the BGR container. Setting `ALCHEMI_NIM_MODEL_TYPE="mace"` configures the container to use the MACE-MP-0 model.*

#### Step 2: Verify Status

Once launched, the container performs an automatic batch-size check and downloads the MACE weights. Check the readiness status of the NIM using:

```bash
curl -i http://localhost:8000/v1/health/ready
```

When it returns `HTTP/1.1 200 OK`, the microservice is fully ready for inference.

#### Step 3: Run Inference (Geometry Relaxation)

The container exposes a `POST /v1/infer` REST API endpoint that accepts batch molecular data. Below is the API request/response format:

##### Request Schema (`POST http://localhost:8000/v1/infer`)
```json
{
  "atoms": [
    {
      "structure_id": "distorted_water",
      "coord": [
        0.0000,  0.0000,  0.0000,
        0.0000,  0.8000,  0.6000,
        0.0000, -0.8000,  0.6000
      ],
      "numbers": [8, 1, 1],
      "pbc": [false, false, false]
    }
  ]
}
```

##### Response Schema
```json
{
  "atoms": [
    {
      "structure_id": "distorted_water",
      "coord": [0.0, 0.0, 0.08, 0.0, 0.76, 0.58, 0.0, -0.76, 0.58],
      "forces": [0.0, 0.0, -0.01, 0.0, 0.005, 0.005, 0.0, -0.005, 0.005],
      "energy": -14.285,
      "converged": true,
      "optimizer_nsteps": 14
    }
  ]
}
```

#### Step 4: Python Query Client

We provide a Python client utility, `query_mace_nim.py`, to programmatically build requests, handle optional integrations with the **Atomic Simulation Environment (ASE)**, and parse/format the relaxed coordinates and forces returned by the container.

To install the dependencies and run the client:
1. Install requirements using the `pyproject.toml` file:
   ```bash
   pip install .
   ```
   *Alternatively, you can install the libraries individually: `pip install requests ase`.*
2. Run the script:
   ```bash
   python3 query_mace_nim.py --url http://localhost:8000
   ```

---

## 2. Deploying model-free NVIDIA NIM MACE models

When you want to deploy a fine-tuned or custom MACE model (instead of the standard MACE-MP-0 model), you can run the NIM container in **Model-free** mode. In this mode, the container mounts your custom weights file directly and configures the interatomic potential service to load your model.

#### Step 1: Fine-Tuning a MACE Model

Before deployment, you can fine-tune MACE foundation weights using your custom atomistic simulation datasets (DFT energy/forces labels). We provide a training script, `train_mace.py`, which:
1. Generates a synthetic Extended XYZ training and validation dataset (`train.xyz` and `valid.xyz`) representing water molecule deformations.
2. Configures training parameters and runs the `mace.cli.run_train` module.

##### Prerequisites for Fine-Tuning

All training dependencies (including `mace-torch`) can be installed directly via `pyproject.toml` using:
```bash
pip install .
```
*Alternatively, you can install the training package individually: `pip install mace-torch`.*

##### Run the Fine-Tuning Script

To generate synthetic data and execute a fast fine-tuning demo:
```bash
python3 train_mace.py --epochs 5 --device cuda --name mace_custom_water
```
This generates your fine-tuned model file, e.g., `mace_custom_water.model`.

---

#### Step 2: Deploying the Custom MACE Model

To deploy the custom model, we disable the default model downloader, volume-mount the `.model` file to the container's internal cache path, and point the environment variables to the mounted weights.

##### Docker deployment command:

```bash
export NGC_API_KEY="your_ngc_api_key_here"
export CUSTOM_MODEL_PATH="/absolute/path/to/mace_custom_water.model"

docker run --gpus all \
  --rm \
  -p 8000:8000 \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_DISABLE_MODEL_DOWNLOAD=true \
  -e ALCHEMI_NIM_MODEL_TYPE="mace" \
  -e ALCHEMI_NIM_MODEL_PATH="/opt/nim/.cache/mace.model" \
  -v "${CUSTOM_MODEL_PATH}:/opt/nim/.cache/mace.model:ro" \
  --shm-size=8g \
  nvcr.io/nim/nvidia/alchemi-bgr:1.0.0
```

---

#### Step 3: Deployment Automation Script

We provide a helper script, `deploy_custom_mace.py`, to manage model validation, start the Docker container, poll the readiness endpoint `/v1/health/ready` until it's online, and submit a test molecular structure to verify predictions are working on your custom model.

##### To deploy via Python:
```bash
python3 deploy_custom_mace.py \
  --model-path /absolute/path/to/mace_custom_water.model \
  --ngc-key your_ngc_api_key_here
```

##### Command options:

- `--model-path`: (Required) Path to the custom `.model` file.
- `--ngc-key`: NGC API Key (can also be loaded via `NGC_API_KEY` env var).
- `--port`: The local port to expose (default `8000`).
- `--container-name`: Custom name for the docker container (default `alchemi-bgr-custom`).
- `--no-verify`: Skip checking health and running test queries after container start.

## 3. Self-Hosting and deploying NVIDIA ALCHEMI NIM models on Local HPC Cluster

NVIDIA NIM containers traditionally rely on Docker, which requires root privileges. On shared High-Performance Computing (HPC) clusters managed by Slurm, root access is restricted, so we use **Singularity** or **Apptainer** to run containers securely as non-root users.

We provide a collection of helper scripts and a Slurm batch template in the [scripts_NIM_slurm_cluster](file:///home/cds/rketkaew/github/mlff-nim-deploy/scripts_NIM_slurm_cluster) folder to automate the deployment process.

#### Directory Structure of Scripts

- [`setup_nim_env.sh`](file:///home/cds/rketkaew/github/mlff-nim-deploy/scripts_NIM_slurm_cluster/setup_nim_env.sh): Performs environment setup, directory creation under a fast scratch filesystem, and pulls the ALCHEMI BGR container image.
- [`launch_nim_interactive.sh`](file:///home/cds/rketkaew/github/mlff-nim-deploy/scripts_NIM_slurm_cluster/launch_nim_interactive.sh): Interactive deployment utility to run NIM on a GPU node inside an active `srun` allocation.
- [`run_nim_server.slurm`](file:///home/cds/rketkaew/github/mlff-nim-deploy/scripts_NIM_slurm_cluster/run_nim_server.slurm): Slurm batch script to submit a background server job (`sbatch`).
- [`query_nim_from_login.sh`](file:///home/cds/rketkaew/github/mlff-nim-deploy/scripts_NIM_slurm_cluster/query_nim_from_login.sh): Tests health check and submits validation inference to your running container on a compute node.

---

#### Step 1: Environment and Security Setup

Before requesting GPU resources, initialize your workspace on any login node. Because container images and model caches can exceed tens of gigabytes, configure the workspace under a fast local scratch filesystem (e.g., `/scratch/users/$USER` or `/tmp`).

##### 1. Save your NGC API Key Securely
Never hardcode API keys or commit them to version control. Create a restricted-permission file in your home directory:
```bash
mkdir -p ~/.secrets
echo "your_ngc_api_key_here" > ~/.secrets/ngc_api_key
chmod 600 ~/.secrets/ngc_api_key
```

##### 2. Run Environment Setup
Execute the setup script on the login node to load the Singularity/Apptainer modules, configure path environments, and build the Singularity Image Format (`.sif`) container file:
```bash
cd scripts_NIM_slurm_cluster
source setup_nim_env.sh
```
*Note: This script pulls `nvcr.io/nim/nvidia/alchemi-bgr:1.0.0` from NGC. The downloaded SIF image is saved to your scratch workspace (`$NIM_ROOT/alchemi-bgr.sif`) for reuse across job sessions.*

---

#### Step 2: Running NIM Interactively

For testing, debugging, or prototyping, you can spin up the microservice interactively.

##### 1. Request an Interactive GPU Node
Use `srun` to request GPU resources. The MACE-MP-0 model has a lightweight footprint and easily fits on a single modern GPU (e.g., A100, H100, H200, etc.):
```bash
srun -p gpu --mem=32G -c 8 -G 1 -t 2:00:00 --pty /bin/bash
```

##### 2. Launch the NIM Container
Once Slurm grants the allocation and launches your shell on the compute node, run:
```bash
# Launch BGR NIM with default MACE model
bash launch_nim_interactive.sh
```
*To use a different port (e.g., 8001) or deploy a custom model file:*
```bash
NIM_PORT=8001 CUSTOM_MODEL_PATH="/path/to/mace_custom_water.model" bash launch_nim_interactive.sh
```

---

#### Step 3: Deploying as a Slurm Batch Job

For production pipelines or prolonged runtime sessions, submit the container server as a background Slurm job.

##### 1. Submit the Job
Submit the batch script:
```bash
sbatch run_nim_server.slurm
```
*To serve a custom model instead of the default MACE-MP-0 model:*
```bash
export CUSTOM_MODEL_PATH="/path/to/your/custom.model"
sbatch run_nim_server.slurm
```

##### 2. Monitor Logs
Check the output logs to watch model weights load and wait until the Triton server is ready for inference:
```bash
tail -f nim-mace-server-<jobid>.out
```

---

#### Step 4: Querying the NIM from a Login Node

Once the service is healthy, you can query it from any login node or client node within the cluster network.

##### 1. Run the Query Helper Script
Specify the GPU compute node host and the port (default `8000`):
```bash
# Example: compute node hostname is compute-gpu4
bash query_nim_from_login.sh compute-gpu4 8000
```

##### 2. Querying with Python client
You can also point the Python client (`query_mace_nim.py`) at the host:
```bash
python3 query_mace_nim.py --url http://compute-gpu4:8000
```

##### 3. Terminating the Server
Once you are done with inference, free cluster resources by canceling the Slurm job:
```bash
scancel <jobid>
```
