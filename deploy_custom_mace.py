"""
Deploy a custom or fine-tuned MACE model via model-free NVIDIA ALCHEMI BGR NIM container
"""

import sys
import os
import time
import subprocess
import argparse
import requests

DEFAULT_IMAGE = "nvcr.io/nim/nvidia/alchemi-bgr:1.0.0"
DEFAULT_PORT = 8000
DEFAULT_CONTAINER_NAME = "alchemi-bgr-custom"

def check_health(url, timeout_sec=300):
    """Polls the readiness health endpoint until the NIM container is ready."""

    health_url = f"{url.rstrip('/')}/v1/health/ready"
    print(f"[INFO] Waiting for NIM container to start up and load custom MACE weights...")
    print(f"[INFO] Polling health check endpoint: {health_url}")
    
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print(f"\n[INFO] NIM service is healthy and ready for inference!")
                return True
        except requests.exceptions.RequestException:
            # Container starting up
            pass
        
        elapsed = int(time.time() - start_time)
        # Dynamic inline printing
        sys.stdout.write(f"\r[INFO] Waiting... {elapsed}s elapsed (timeout: {timeout_sec}s)")
        sys.stdout.flush()
        time.sleep(5)
        
    print(f"\n[ERROR] Health check timed out after {timeout_sec} seconds. Check docker container logs.")

    return False

def verify_inference(url):
    """Sends a sample water structure to verify that inference on the deployed model works."""

    infer_url = f"{url.rstrip('/')}/v1/infer"
    
    payload = {
        "atoms": [
            {
                "structure_id": "test_water_molecule",
                "coord": [
                    0.0, 0.0, 0.0,
                    0.0, 0.8, 0.6,
                    0.0, -0.8, 0.6
                ],
                "numbers": [8, 1, 1],
                "pbc": [False, False, False]
            }
        ]
    }
    
    print(f"[INFO] Submitting test molecule to check deployment inference...")
    try:
        response = requests.post(infer_url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if "atoms" in result and len(result["atoms"]) > 0:
            opt = result["atoms"][0]
            print("\n================ Verification Inference Results ================")
            print(f"Structure ID    : {opt.get('structure_id')}")
            print(f"Convergence     : {opt.get('converged')}")
            print(f"Steps taken     : {opt.get('optimizer_nsteps')}")
            print(f"Potential Energy: {opt.get('energy')} eV")
            print("================================================================")
            print("[INFO] Custom model deployment successfully verified!")
            return True
        else:
            print("[ERROR] Response did not contain expected atoms structures.")
            return False
            
    except Exception as e:
        print(f"[ERROR] Verification request failed: {e}")
        
        return False

def deploy_container(image, model_path, ngc_key, port, name):
    """Launches the Docker container with the custom MACE weights mounted."""
    # Absolute local path validation
    abs_model_path = os.path.abspath(model_path)
    if not os.path.exists(abs_model_path):
        print(f"[ERROR] The model path '{abs_model_path}' does not exist.")
        sys.exit(1)
        
    print(f"[INFO] Deploying custom model from: {abs_model_path}")
    print(f"[INFO] Internal container model path: /opt/nim/.cache/mace.model")
    
    # Build docker run command
    docker_cmd = [
        "docker", "run", "-d",
        "--name", name,
        "--gpus", "all",
        "-p", f"{port}:8000",
        "-e", f"NGC_API_KEY={ngc_key}",
        "-e", "NIM_DISABLE_MODEL_DOWNLOAD=true",
        "-e", "ALCHEMI_NIM_MODEL_TYPE=mace",
        "-e", "ALCHEMI_NIM_MODEL_PATH=/opt/nim/.cache/mace.model",
        "-v", f"{abs_model_path}:/opt/nim/.cache/mace.model:ro",
        "--shm-size=8g",
        image
    ]
    
    print(f"[INFO] Executing Docker command:\n{' '.join(docker_cmd)}\n")
    
    try:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        result = subprocess.run(docker_cmd, capture_output=True, text=True, check=True)
        container_id = result.stdout.strip()
        print(f"[INFO] Container launched successfully. Container ID: {container_id[:12]}")
        return container_id
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to start docker container: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("[ERROR] Docker is not installed or not in system PATH.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Deploy custom MACE model via NVIDIA BGR NIM container.")
    parser.add_argument("--model-path", required=True, help="Path to your custom/fine-tuned MACE model (.model file)")
    parser.add_argument("--ngc-key", default=os.getenv("NGC_API_KEY"), help="NGC API Key (can also be set via NGC_API_KEY environment variable)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Host port to expose BGR service (default: {DEFAULT_PORT})")
    parser.add_argument("--container-name", default=DEFAULT_CONTAINER_NAME, help=f"Name of the docker container (default: {DEFAULT_CONTAINER_NAME})")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help=f"Docker image for BGR NIM (default: {DEFAULT_IMAGE})")
    parser.add_argument("--no-verify", action="store_true", help="Skip health check and verification queries after deployment")
    parser.add_argument("--timeout", type=int, default=300, help="Max timeout (seconds) to wait for service readiness (default: 300)")
    
    args = parser.parse_args()

    if not args.ngc_key:
        print("[ERROR] NGC API Key must be provided either via --ngc-key or the NGC_API_KEY env variable.")
        sys.exit(1)

    container_id = deploy_container(
        image=args.image,
        model_path=args.model_path,
        ngc_key=args.ngc_key,
        port=args.port,
        name=args.container_name
    )

    url = f"http://localhost:{args.port}"

    if args.no_verify:
        print("[INFO] --no-verify flag set. Deployment completed. Exiting.")
        return

    ready = check_health(url, timeout_sec=args.timeout)
    if ready:
        verify_inference(url)
    else:
        print(f"\n[WARNING] Container failed health check. Check logs with: docker logs {args.container_name}")
        sys.exit(1)

    print(f"\n[INFO] To monitor logs:  docker logs -f {args.container_name}")
    print(f"[INFO] To stop service:  docker rm -f {args.container_name}")

if __name__ == "__main__":
    main()
