"""
step1_extract_athena.py
=======================
Run this in WSL (needs only h5py + numpy, NO pytorch):

    cd "/mnt/c/Users/tirth/Documents/Projects/GRINN/Power spectrum perturbations"
    python3 step1_extract_athena.py

Reads Athena++ .athdf snapshots and saves each field as a .npy file
inside the athena_cache/ subfolder (on the Windows filesystem via /mnt/c,
so step2 can read them directly from Windows Python).
"""

import os
import sys
import numpy as np

ATHENA_DATA_DIR = "/home/Aboba/test/data_for_grinn/2d_supercritical/seed_4"
ATHENA_OUTPUT_ID = "external2d.out1"
TIME_POINTS = [0.00, 0.15, 0.30, 0.45, 0.60]
TIME_TOL = 0.2

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from numerical_solvers.athena_reader import build_athena_cache

cache = build_athena_cache(
    data_dir=ATHENA_DATA_DIR,
    output_id=ATHENA_OUTPUT_ID,
    time_points=TIME_POINTS,
    time_tol=TIME_TOL,
)

out_dir = os.path.join(PROJECT, "athena_cache")
os.makedirs(out_dir, exist_ok=True)

times = sorted(cache.keys())
np.save(os.path.join(out_dir, "times.npy"), np.array(times))

for t in times:
    tag = f"{t:.6f}".replace(".", "p")
    for field in ("x", "y", "rho", "vx", "vy"):
        np.save(os.path.join(out_dir, f"{field}_{tag}.npy"), cache[t][field])

print(f"\nSaved {len(times)} snapshots to {out_dir}")
print("Times:", [f"{t:.4f}" for t in times])
print("\nNow run step2_compare.py on Windows.")
