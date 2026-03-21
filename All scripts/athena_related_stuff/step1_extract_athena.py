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
from pathlib import Path

ATHENA_DATA_DIR = "/home/Aboba/test/data_for_grinn/2d_subcritical/seed_82"
ATHENA_OUTPUT_ID = "external2d.out1"
TIME_POINTS = [0.0, 0.75, 1.5, 2.25, 3.00]
TIME_TOL = 0.2

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from numerical_solvers.athena_reader import build_athena_cache

cache = build_athena_cache(
    data_dir=ATHENA_DATA_DIR,
    output_id=ATHENA_OUTPUT_ID,
    time_points=TIME_POINTS,
    time_tol=TIME_TOL,
)

out_dir = os.path.join(str(SCRIPT_DIR), "athena_cache")
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
