"""
History:
- Rangsiman Ketkaew [15.04.2026]
"""

import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
from ase import Atoms
from ase.io import read

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import _write_neighbor_data_to_batch
from nvalchemi.models.base import NeighborListFormat
from ase.neighborlist import neighbor_list

# We import torch_tensorrt below in the code

"""
Run MACE-MH-1 model inference using TensorRT-optimized dynamic batching

Note: Here we use MACE model via nvalchemi package (not mace-torch package)
"""

def compute_neighbors_ase(batch: Batch, cutoff: float) -> None:
    """
    Computes neighbor lists for a batch of structures using ASE on CPU,
    and writes the formatted COO neighbors back to the nvalchemi Batch object
    """
    N = batch.num_nodes
    device = batch.positions.device
    B = batch.num_graphs
    ptr = batch.batch_ptr.tolist()
    
    all_neighbor_lists = []
    num_neighbors = torch.zeros(N, dtype=torch.int32, device=device)
    
    for b in range(B):
        start_idx = ptr[b]
        end_idx = ptr[b+1]
        
        pos = batch.positions[start_idx:end_idx].cpu().numpy()
        nums = batch.atomic_numbers[start_idx:end_idx].cpu().numpy()
        cell = batch.cell[b].cpu().numpy()
        pbc = batch.pbc[b].cpu().numpy()
        
        atoms = Atoms(numbers=nums, positions=pos, cell=cell, pbc=pbc)
        i, j, S = neighbor_list('ijS', atoms, cutoff)
        
        i_batch = i + start_idx
        j_batch = j + start_idx
        
        all_neighbor_lists.append((i_batch, j_batch, S))
        
        # Count neighbors per atom
        unique, counts = np.unique(i_batch, return_counts=True)
        for u, c in zip(unique, counts):
            num_neighbors[int(u)] = int(c)
            
    max_neighbors = int(num_neighbors.max()) if N > 0 else 0

    if max_neighbors == 0:
        max_neighbors = 1
        
    neighbor_matrix = torch.full((N, max_neighbors), N, dtype=torch.int32, device=device)
    neighbor_matrix_shifts = torch.zeros(N, max_neighbors, 3, dtype=torch.int32, device=device)
    current_slots = torch.zeros(N, dtype=torch.int32, device=device)
    
    for i_batch, j_batch, S in all_neighbor_lists:
        for idx_s, idx_r, shift in zip(i_batch, j_batch, S):
            s_idx = int(idx_s)
            slot = int(current_slots[s_idx])
            if slot < max_neighbors:
                neighbor_matrix[s_idx, slot] = int(idx_r)
                neighbor_matrix_shifts[s_idx, slot] = torch.tensor(shift, dtype=torch.int32, device=device)
                current_slots[s_idx] += 1
                
    _write_neighbor_data_to_batch(
        batch=batch,
        neighbor_matrix=neighbor_matrix,
        num_neighbors=num_neighbors,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        format=NeighborListFormat.COO,
        cutoff=cutoff
    )


class DynamicBatchEngine:
    """
    Dynamic batching inference engine optimized with Nvidia TensorRT.
    """
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        head: str = "matpes_r2scan",
        dtype: torch.dtype = torch.float32,
        compile_model: bool = True,
        enable_cueq: bool = False # to be implemented/used
    ):
        self.device = device
        self.head = head
        self.dtype = dtype
        self.queue = []
        self.structure_ids = []
        
        print(f"[INFO] Initializing Dynamic Batch Engine with TensorRT on {device}...")
        print(f"[INFO] Loading MACE model weights from {model_path}...")
        
        mace_model = torch.load(model_path, weights_only=False, map_location=device)
        
        if hasattr(mace_model, "heads"):
            print(f"[INFO] Available heads: {mace_model.heads}")
            if head in mace_model.heads:
                print(f"[INFO] Selecting model head: {head}")
                self.head_idx = mace_model.heads.index(head)
            else:
                self.head_idx = 0
                print(f"[WARNING] Head '{head}' not found. Defaulting to head 0: {mace_model.heads[0]}")
        else:
            self.head_idx = 0
            
        # Optimize model using PyTorch 2.0 compile on the inner model with TensorRT backend
        if compile_model:
            print("[INFO] Patching e3nn Irrep length for compilation compatibility...")
            try:
                from e3nn.o3 import Irrep
                if Irrep.__len__ is not tuple.__len__:
                    Irrep.__len__ = tuple.__len__
            except ImportError:
                pass
            
            mace_model.eval()

            for param in mace_model.parameters():
                param.requires_grad = False
                
            if device.type == "cuda":
                print("[INFO] Attempting to compile inner MACE model using Nvidia TensorRT backend...")
                try:
                    import torch_tensorrt

                    compiled_inner = torch.compile(mace_model, backend="tensorrt")
                    self.wrapper = MACEWrapper(compiled_inner)
                    print("[INFO] TensorRT compilation setup successfully.")
                except Exception as e:
                    print(f"[WARNING] Could not initialize TensorRT backend: {e}. Falling back to default inductor compiler backend.")
                    compiled_inner = torch.compile(mace_model)
                    self.wrapper = MACEWrapper(compiled_inner)
            else:
                print("[INFO] CUDA device not available. Compiling inner MACE model with default inductor backend on CPU...")
                compiled_inner = torch.compile(mace_model)
                self.wrapper = MACEWrapper(compiled_inner)
        else:
            self.wrapper = MACEWrapper(mace_model)
            
        self.wrapper.to(device=device, dtype=dtype)
        print("[INFO] Engine ready for inference.")

    def add_structure(self, atoms: Atoms, structure_id: str = None):
        """
        Dynamically queues a new atomic structure into the in-memory engine queue
        """
        struct_id = structure_id or f"struct_{len(self.queue)}"
        
        data = AtomicData(
            atomic_numbers=torch.tensor(atoms.numbers, dtype=torch.int64),
            positions=torch.tensor(atoms.positions, dtype=self.dtype),
            cell=torch.tensor(atoms.cell.array, dtype=self.dtype).unsqueeze(0),
            pbc=torch.tensor(atoms.pbc).unsqueeze(0)
        )
        
        self.queue.append(data)
        self.structure_ids.append(struct_id)
        print(f"[INFO] Queued structure '{struct_id}' ({len(atoms)} atoms) into dynamic memory queue.")

    def run_inference(self):
        if not self.queue:
            print("[WARNING] Memory queue is empty. No inference performed.")
            return {}
            
        num_structures = len(self.queue)
        print(f"\n[INFO] Starting batch inference on {num_structures} queued structure(s)...")
        
        # Collate list of AtomicData into a single Batch representation
        batch = Batch.from_data_list(self.queue, device=self.device)
        batch.head = torch.full((num_structures,), self.head_idx, dtype=torch.int64, device=self.device)
        
        compute_neighbors_ase(batch, cutoff=self.wrapper.cutoff)
        
        t0 = time.time()

        try:
            out = self.wrapper(batch)
        except Exception as e:
            print("ERROR RUNNING MODEL FORWARD PASS:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise e
            
        dt = time.time() - t0
        print(f"[INFO] Batch inference completed in {dt:.4f} seconds.")
        
        try:
            # Extract energy and forces with .detach() to prevent grad error
            energies = out["energy"].detach().cpu().numpy()
            forces = out["forces"].detach().cpu().numpy()
        except Exception as e:
            print("ERROR EXTRACTING OUTPUTS:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise e
        
        results = {}
        ptr = batch.batch_ptr.tolist()
        for i, struct_id in enumerate(self.structure_ids):
            start = ptr[i]
            end = ptr[i+1]
            results[struct_id] = {
                "energy": float(energies[i].item()),
                "forces": forces[start:end].tolist()
            }
            
        # Reset queue
        self.queue.clear()
        self.structure_ids.clear()
        return results

#-------- Main Code ---------

device = torch.device("cpu")
if torch.cuda.is_available():
    try:
        x = torch.zeros(1, device="cuda")
        device = torch.device("cuda")
        print("[INFO] CUDA is available and working.", flush=True)
    except Exception:
        print("[WARNING] CUDA is detected but failed to initialize. Falling back to CPU.", flush=True)
        
model_path = "./mace-mh-1.model"
head_name = "matpes_r2scan"

# Instantiate our Dynamic Batch Engine with TensorRT optimization
engine = DynamicBatchEngine(
    model_path=model_path,
    device=device,
    head=head_name,
    dtype=torch.float32,
    compile_model=True
)

print("\n[INFO] Loading Naphthalene initial configuration...", flush=True)
init_conf = read("Naphthalene.xyz")

print("[INFO] Generating distorted configurations for dynamic batching...", flush=True)

np.random.seed(42)
num_samples = 4

for idx in range(num_samples):
    struct = init_conf.copy()
    pos = struct.get_positions()
    # Add random coordinate perturbations (simulating thermal fluctuations/vibrations)
    distortion = np.random.normal(0, 0.05, size=pos.shape)
    struct.set_positions(pos + distortion)
    engine.add_structure(struct, structure_id=f"Naphthalene_Perturbed_{idx+1}")
    
results = engine.run_inference()

print("")
print("================ Energy & Force Prediction Results ================")

for struct_id, pred in results.items():
    print(f"\nStructure ID : {struct_id}", flush=True)
    print(f"Energy       : {pred['energy']:.6f} eV", flush=True)
    print("Forces (First 3 atoms, eV/A):", flush=True)
    for atom_idx in range(min(3, len(pred['forces']))):
        fx, fy, fz = pred['forces'][atom_idx]
        print(f"  Atom {atom_idx+1}: [{fx:10.5f}, {fy:10.5f}, {fz:10.5f}]", flush=True)

print("===================================================================", flush=True)
