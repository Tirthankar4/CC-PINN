"""
step1_extract_athena.py
=======================
Run this in WSL (needs only h5py + numpy, NO pytorch):

    cd "/mnt/c/Users/tirth/Documents/Projects/GRINN/Power spectrum perturbations"
    python3 step1_extract_athena.py

Reads Athena++ .athdf snapshots and saves each field as a .npy file
inside the athena_cache/ subfolder (on the Windows filesystem via /mnt/c,
so step2 can read them directly from Windows Python).

Also saves rho_max_times.npy and rho_max_curve.npy spanning ALL available
snapshots, for the ln(rho_max) vs t growth-rate panel in step2.
"""

import os
import sys
import numpy as np
from pathlib import Path

ATHENA_DATA_DIR = "/home/Aboba/test/data_for_grinn/2d_supercritical/seed_4"
ATHENA_OUTPUT_ID = "external2d.out1"
TIME_POINTS = [0.00, 0.30, 0.60]
TIME_TOL = 0.1

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from numerical_solvers.athena_reader import build_athena_cache, discover_snapshots, load_athdf_2d

# ---------------------------------------------------------------------------
# 1. Build spatial-field cache for TIME_POINTS (existing behaviour)
# ---------------------------------------------------------------------------
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

print(f"\nSaved {len(times)} spatial snapshots to {out_dir}")
print("Times:", [f"{t:.4f}" for t in times])

# ---------------------------------------------------------------------------
# 2. Build rho_max curve across ALL available snapshots
#    Only rho_max is extracted (cheap) — no need to cache full fields.
# ---------------------------------------------------------------------------
print("\nBuilding rho_max curve across all snapshots...")

time_map = discover_snapshots(ATHENA_DATA_DIR, ATHENA_OUTPUT_ID)
all_times_sorted = sorted(time_map.keys())

rho_max_times = []
rho_max_vals  = []

for t in all_times_sorted:
    _, _, _, rho, _, _ = load_athdf_2d(time_map[t])
    rho_max = float(rho.max())
    rho_max_times.append(t)
    rho_max_vals.append(rho_max)
    print(f"  t={t:.4f}  rho_max={rho_max:.4f}")

rho_max_times = np.array(rho_max_times)
rho_max_vals  = np.array(rho_max_vals)

np.save(os.path.join(out_dir, "rho_max_times.npy"), rho_max_times)
np.save(os.path.join(out_dir, "rho_max_curve.npy"),  rho_max_vals)

print(f"\nSaved rho_max curve ({len(rho_max_times)} points) to {out_dir}")
print("\nNow run step2_compare.py on Windows.")
