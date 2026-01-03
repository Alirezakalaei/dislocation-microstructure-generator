# Dislocation Microstructure Generator (Parallel Slip Planes)

Python code to generate **dislocation loop microstructures** on **parallel slip planes** in a cubic simulation box and compute **internal stress** and density measures on a 2D grid (XY), with slip planes separated in Z.

This repository includes:
- `single_thread_active_all.py`: sample generation + saving results
- `density_extraction.py`: density extraction and internal stress computation kernels (Numba)

The code is designed for **multi-process parallel runs** (e.g., many independent samples in parallel). Inside each process, BLAS/Numba threads are forced to 1 to avoid oversubscription.

---

## Quick start

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
