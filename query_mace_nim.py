"""
Query client for NVIDIA ALCHEMI Batched Geometry Relaxation (BGR) NIM
"""

import json
import sys
import argparse
import requests
import ase
from ase.atoms import Atoms

DEFAULT_URL = "http://localhost:8000"


def check_health(base_url):
    """Checks if the NIM container is ready for inference"""

    health_url = f"{base_url.rstrip('/')}/v1/health/ready"
    try:
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200:
            print("[INFO] NIM service is ready.")
            return True
        else:
            print(f"[WARNING] NIM service health check returned status code {response.status_code}.")
            return False
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Could not connect to NIM health endpoint: {e}")
        return False

def relax_structure(base_url, structures, opttol=None, maxsteps=None):
    """
    Sends atomic structures to the BGR NIM for geometry relaxation
    
    Parameters:
        base_url (str): Base URL of the NIM service.
        structures (list): List of dicts, each with 'coord' (flat list) and 'numbers' (list of ints).
                           Optional fields include 'cell', 'pbc', and 'structure_id'.
        opttol (float): Force optimization tolerance (eV/Å) override.
        maxsteps (int): Maximum number of optimizer steps override.
    """

    infer_url = f"{base_url.rstrip('/')}/v1/infer"
    
    payload = {
        "atoms": structures
    }
    
    if opttol is not None:
        payload["opttol"] = opttol
    if maxsteps is not None:
        payload["maxsteps"] = maxsteps

    print(f"[INFO] Sending request to {infer_url} with {len(structures)} structure(s)...")
    
    try:
        response = requests.post(infer_url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP Error: {response.text}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        sys.exit(1)

def print_results(response_data):
    """Parses and prints the relaxed structure results."""

    if "atoms" not in response_data:
        print("[ERROR] Invalid response format (missing 'atoms' key).")
        print(json.dumps(response_data, indent=2))

    for idx, opt_struct in enumerate(response_data["atoms"]):
        struct_id = opt_struct.get("structure_id", f"Structure_{idx}")
        print(f"\n================ Results for: {struct_id} ================")
        print(f"Convergence Status : {opt_struct.get('converged', 'N/A')}")
        print(f"Optimization Steps : {opt_struct.get('optimizer_nsteps', 'N/A')}")
        print(f"Potential Energy   : {opt_struct.get('energy', 'N/A')} eV")
        
        coords = opt_struct.get("coord", [])
        forces = opt_struct.get("forces", [])
        numbers = opt_struct.get("numbers", [])
        
        if coords:
            print("\nOptimized Coordinates (Å):")
            num_atoms = len(coords) // 3
            for i in range(num_atoms):
                atom_num = numbers[i] if i < len(numbers) else "?"
                x, y, z = coords[3*i : 3*i+3]
                fx, fy, fz = forces[3*i : 3*i+3] if forces else (0.0, 0.0, 0.0)
                force_str = f" | Force (eV/Å): [{fx:8.4f}, {fy:8.4f}, {fz:8.4f}]" if forces else ""
                print(f"  Atom {i+1:2d} (Z={str(atom_num):>2s}): [{x:9.4f}, {y:9.4f}, {z:9.4f}]{force_str}")

def convert_ase_atoms(atoms_obj, struct_id="ASE_Structure"):
    """Converts an ASE Atoms object into the dict schema expected by ALCHEMI BGR"""

    structure = {
        "structure_id": struct_id,
        "coord": atoms_obj.positions.flatten().tolist(),
        "numbers": atoms_obj.numbers.tolist(),
        "pbc": atoms_obj.pbc.tolist()
    }

    if any(atoms_obj.pbc):
        structure["cell"] = [float(val) for row in atoms_obj.cell for val in row]

    return structure

def main():
    parser = argparse.ArgumentParser(description="Query client for NVIDIA ALCHEMI BGR NIM (MACE-MP-0 model)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"NIM base URL (default: {DEFAULT_URL})")
    parser.add_argument("--opttol", type=float, default=None, help="Force optimization tolerance (eV/Å)")
    parser.add_argument("--maxsteps", type=int, default=None, help="Maximum number of optimizer steps")
    parser.add_argument("--skip-health", action="store_true", help="Skip health check before sending requests")
    
    args = parser.parse_args()

    if not args.skip_health:
        if not check_health(args.url):
            print("[INFO] NIM service health check failed. Proceeding anyway...")

    water_coords = [
        0.0000,  0.0000,  0.0000,
        0.0000,  0.8000,  0.6000,
        0.0000, -0.8000,  0.6000 
    ]
    water_numbers = [8, 1, 1]

    print("[INFO] Creating structure using ase.Atoms...")
    atoms = Atoms(
        symbols="OH2",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.8, 0.6],
            [0.0, -0.8, 0.6]
        ]
    )
    structures = [convert_ase_atoms(atoms, "Distorted_Water_ASE")]

    print("[INFO] Submitting structure for geometry relaxation...")
    response = relax_structure(args.url, structures, opttol=args.opttol, maxsteps=args.maxsteps)
    print_results(response)

if __name__ == "__main__":
    main()
