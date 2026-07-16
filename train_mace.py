"""
Prepare data and run fine-tuning for a MACE model
"""

import sys
import subprocess
import argparse
import random

def generate_synthetic_xyz(filepath, num_structures=50):
    """
    Generates a synthetic dataset of distorted H2O configurations
    in the Extended XYZ format suitable for training/fine-tuning MACE
    """

    print(f"[INFO] Generating {num_structures} synthetic H2O configurations in {filepath}...")
    
    # We will perturb these coordinates of water geometry to simulate molecular configurations
    with open(filepath, "w") as f:
        for idx in range(num_structures):
            # Slightly perturb O position
            ox = random.uniform(-0.05, 0.05)
            oy = random.uniform(-0.05, 0.05)
            oz = random.uniform(-0.05, 0.05)
            
            # Slightly perturb O-H distance and bond angles
            d1 = random.uniform(0.90, 1.05)
            d2 = random.uniform(0.90, 1.05)
            
            h1x = ox
            h1y = oy + d1 * random.uniform(0.95, 1.05)
            h1z = oz + d1 * random.uniform(0.1, 0.3)
            
            h2x = ox
            h2y = oy - d2 * random.uniform(0.95, 1.05)
            h2z = oz + d2 * random.uniform(0.1, 0.3)
            
            # Mock properties: energy (eV) and forces (eV/A)
            # Energy scales roughly with distortion
            distortion = abs(d1 - 0.96) + abs(d2 - 0.96)
            energy = -14.285 + distortion * 5.0
            
            # Mock forces (restoring forces pointing back to standard bond length)
            fo_x, fo_y, fo_z = 0.0, 0.0, 0.0
            fh1_x = -(h1x - ox) * 0.1
            fh1_y = -(h1y - oy - 0.96) * 1.5
            fh1_z = -(h1z - oz) * 0.1
            fh2_x = -(h2x - ox) * 0.1
            fh2_y = -(h2y - oy + 0.96) * 1.5
            fh2_z = -(h2z - oz) * 0.1
            
            # Sum of forces should be zero (action-reaction)
            fo_x = -(fh1_x + fh2_x)
            fo_y = -(fh1_y + fh2_y)
            fo_z = -(fh1_z + fh2_z)
            
            f.write("3\n")
            f.write(f"Lattice=\"10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0\" Properties=species:S:1:pos:R:3:forces:R:3 energy={energy:.6f} pbc=\"T T T\"\n")
            f.write(f"O  {ox:10.6f} {oy:10.6f} {oz:10.6f}  {fo_x:10.6f} {fo_y:10.6f} {fo_z:10.6f}\n")
            f.write(f"H  {h1x:10.6f} {h1y:10.6f} {h1z:10.6f}  {fh1_x:10.6f} {fh1_y:10.6f} {fh1_z:10.6f}\n")
            f.write(f"H  {h2x:10.6f} {h2y:10.6f} {h2z:10.6f}  {fh2_x:10.6f} {fh2_y:10.6f} {fh2_z:10.6f}\n")
            
    print(f"[INFO] Successfully created {filepath}.")

def run_fine_tuning(train_file, valid_file, foundation_model, name, epochs, batch_size, device):
    """
    Invokes MACE training CLI programmatically using subprocess
    """
    print(f"[INFO] Starting MACE fine-tuning process...")
    print(f"[INFO] Base foundation model: {foundation_model}")
    print(f"[INFO] Target epochs: {epochs}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Device: {device}")
    
    cmd = [
        sys.executable, "-m", "mace.cli.run_train",
        "--name", name,
        "--train_file", train_file,
        "--valid_file", valid_file,
        "--energy_weight", "1.0",
        "--forces_weight", "10.0",
        "--lr", "0.005",
        "--batch_size", str(batch_size),
        "--max_num_epochs", str(epochs),
        "--device", device,
        "--default_dtype", "float32"
    ]
    
    if foundation_model:
        cmd.extend(["--foundation_model", foundation_model])
        
    print(f"[INFO] Running training command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end="")
            
        process.wait()
        
        if process.returncode == 0:
            print(f"[INFO] Fine-tuning finished successfully. Check files starting with '{name}_' for checkpoints and outputs.")
        else:
            print(f"[ERROR] Fine-tuning failed with return code {process.returncode}.")
            sys.exit(process.returncode)
            
    except FileNotFoundError:
        print("[ERROR] 'mace-torch' packages or 'mace' module not installed/runnable in the current environment.")
        print("Please install MACE via: pip install mace-torch")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Exception occurred during training: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic molecular datasets and fine-tune a MACE model.")
    parser.add_argument("--train-file", default="train.xyz", help="Path to training data XYZ file (default: train.xyz)")
    parser.add_argument("--valid-file", default="valid.xyz", help="Path to validation data XYZ file (default: valid.xyz)")
    parser.add_argument("--generate-only", action="store_true", help="Only generate synthetic data and exit without running training")
    parser.add_argument("--foundation-model", default="small", help="Foundation model name ('small', 'medium', 'large') or file path to base weights")
    parser.add_argument("--name", default="mace_finetuned_model", help="Prefix name for the output model weights and checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training (default: 4)")
    parser.add_argument("--device", default="cuda", help="Execution device: 'cuda', 'cpu', or 'mps' (default: cuda)")
    
    args = parser.parse_args()

    # Generate synthetic training/validation data
    generate_synthetic_xyz(args.train_file, num_structures=40)
    generate_synthetic_xyz(args.valid_file, num_structures=10)

    if args.generate_only:
        print("[INFO] --generate-only flag is set. Data generation complete. Exiting.")
        return

    run_fine_tuning(
        train_file=args.train_file,
        valid_file=args.valid_file,
        foundation_model=args.foundation_model,
        name=args.name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )

if __name__ == "__main__":
    main()
