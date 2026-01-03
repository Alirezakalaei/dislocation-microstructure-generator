# Dislocation Microstructure Generator (Parallel Slip Planes)

Python code to generate **dislocation loop microstructures** on **parallel slip planes** in a cubic simulation box and compute **internal stress** and density measures on a 2D grid (XY), with slip planes separated in Z.

This repository includes:
- `single_thread_active_all.py`: sample generation + saving results
- `density_extraction.py`: density extraction and internal stress computation kernels (Numba)

The code is designed for **multi-process parallel runs** (e.g., many independent samples in parallel). Inside each process, BLAS/Numba threads are forced to 1 to avoid oversubscription.

---
## What the code does

For each sample:

1. Randomly chooses a cubic box size `Lx = Ly = Lz = U(400, 600) * b` (where `b` is the Burgers magnitude for FCC Cu).
2. Creates dislocation loop “features” for multiple Burgers systems (3 in-plane Burgers vectors).
3. Each feature generates several concentric polygonal loops:
   - Loop plane is **forced parallel to the XY plane** (`normal = [0,0,1]`).
   - A random **in-plane rotation** is applied so loops are not axis-aligned.
   - Radii are chosen with a bounded eccentricity to avoid extreme aspect ratios.
4. Discretizes the box into an `(nx, ny)` grid and uses `dz` for the slip-plane spacing.
5. Computes:
   - Dislocation density/tangent fields (via `density_extraction.compute_dislocation_density_and_tangent`)
   - Internal shear stress map `tau_int` (via `compute_internal_tau_2d`)
   - Density distributions binned by Burgers/angle
   - Average stress statistics per Burgers system
   - SSD, GND, total density, and GND vector summaries

The script writes a compressed `npz` file per sample.

---

## Repository contents

- `src/single_thread_active_all.py`  
  Main executable script that generates microstructures and saves results.
- `density_extraction`  
  **Required dependency** providing density and stress routines (add it to `src/` or install separately).

---

## Installation

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # (Linux/macOS)
# .venv\Scripts\activate   # (Windows)

pip install -r requirements.txt
