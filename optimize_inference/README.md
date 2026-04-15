# MACE Model Inference Optimization Examples

This directory contains examples for running and optimizing Machine Learning Force Field (MLFF) inference on MACE models using the NVIDIA ALCHEMI framework and PyTorch optimization primitives.

## Code structure

- [**`1_simple_inference.py`**](1_simple_inference.py): A serial Langevin molecular dynamics (MD) simulation using the base `MACECalculator` from the MACE library on a Naphthalene molecule.
- [**`2_vllm_opt_inference.py`**](2_vllm_opt_inference.py): An optimized, high-throughput batch inference script using an in-memory `PagedAttentionEngine` for dynamic graph collation, `nvalchemi` wrapper structures, and `torch.compile` graph optimization.
- **`Naphthalene.xyz`**: Example molecule - Naphthalene molecule ($C_{10}H_8$)
- **`mace-mh-1.model`**: Weight of mace-mh-1 model

---

## 1. Simple Inference (`1_simple_inference.py`)

This script demonstrates basic Langevin MD simulations. It uses a single molecule in a serial loop to calculate energies and forces at each step:

```bash
python 1_simple_inference.py
```

---

## 2. Optimized Inference Engine (`2_vllm_opt_inference.py`)

This script implements `PagedAttentionEngine`, a dynamic batching and inference acceleration engine inspired by memory-paging architectures in LLM frameworks:

- **Dynamic Collation**: Incoming structures are paged into an in-memory queue and dynamically grouped into a unified batch (`Batch.from_data_list`) to maximize GPU/CPU thread occupancy.
- **Multi-Head Selection**: Points the ScaleShiftMACE model to the correct head (e.g. `matpes_r2scan`) by batching head indices.
- **Compiler Optimizations**: Patches `e3nn` sequence metadata and compiles the inner MACE model (`torch.compile`) to generate high-performance execution graphs.
- **CPU Fallback Neighbors**: Employs an ASE-based CPU neighbor list generator to bypass NVIDIA Warp GPU driver restrictions, making the optimization script runnable on local development nodes and clusters.

To run the optimized engine, use

```bash
python 2_vllm_opt_inference.py
```

### Output Verification Results

Executing `2_vllm_opt_inference.py` processes 4 distorted naphthalene configurations concurrently in a single batch. Below is the verified log output:

```
[INFO] Initializing PagedAttention Engine on cpu...
[INFO] Loading MACE model weights from ./mace-mh-1.model...
[INFO] Available heads: ['matpes_r2scan', 'mp_pbe_refit_add', 'spice_wB97M', 'oc20_usemppbe', 'omol', 'omat_pbe']
[INFO] Selecting model head: matpes_r2scan
[INFO] Patching e3nn Irrep length for compilation compatibility...
[INFO] Compiling inner MACE model with torch.compile()...
[INFO] Engine ready for inference.

[INFO] Loading Naphthalene initial configuration...
[INFO] Generating distorted configurations for dynamic batching...
[INFO] Paged structure 'Naphthalene_Perturbed_1' (18 atoms) into dynamic memory queue.
[INFO] Paged structure 'Naphthalene_Perturbed_2' (18 atoms) into dynamic memory queue.
[INFO] Paged structure 'Naphthalene_Perturbed_3' (18 atoms) into dynamic memory queue.
[INFO] Paged structure 'Naphthalene_Perturbed_4' (18 atoms) into dynamic memory queue.

[INFO] Starting batch inference on 4 paged structure(s)...
[INFO] Batch inference completed in 11.4391 seconds.

================ Energy & Force Prediction Results ================

Structure ID : Naphthalene_Perturbed_1
Energy       : -124.770142 eV
Forces (First 3 atoms, eV/A):
  Atom 1: [  -0.48989,    0.90810,   -0.32138]
  Atom 2: [  -4.90373,    4.11876,    0.25887]
  Atom 3: [  -2.34159,   -4.70596,   -0.16903]

Structure ID : Naphthalene_Perturbed_2
Energy       : -124.846573 eV
Forces (First 3 atoms, eV/A):
  Atom 1: [  -6.93765,   -3.50196,    1.44382]
  Atom 2: [   1.36569,   -0.17033,   -1.23582]
  Atom 3: [  -0.71332,    3.93664,    1.50115]

Structure ID : Naphthalene_Perturbed_3
Energy       : -123.915665 eV
Forces (First 3 atoms, eV/A):
  Atom 1: [  -1.91437,    1.38088,    2.91456]
  Atom 2: [   1.72376,   -1.31046,   -2.28917]
  Atom 3: [  -0.41426,    1.17349,    0.93960]

Structure ID : Naphthalene_Perturbed_4
Energy       : -125.243790 eV
Forces (First 3 atoms, eV/A):
  Atom 1: [  -1.40481,    2.47802,    0.76006]
  Atom 2: [  -0.32496,   -6.25894,   -1.51413]
  Atom 3: [   0.11754,    1.92750,    1.56766]
===================================================================
```

## History

"""
History:
- Rangsiman Ketkaew [15.04.2026]
"""
