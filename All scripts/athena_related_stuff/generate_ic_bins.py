"""
generate_ic_bins.py  –  Temporary helper script
================================================
Generates the power-spectrum velocity initial-condition fields (vx0, vy0)
via LAX_torch and saves them as raw float64 binary files:

    vx_ic.bin   –  x-velocity field  (nx × ny, float64, C-order)
    vy_ic.bin   –  y-velocity field  (nx × ny, float64, C-order)

Run from the project root:
    python generate_ic_bins.py
"""

import os
import sys
from pathlib import Path

# ── Resolve paths after moving this script under athena_related_stuff/ ───────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# ── Import config values ──────────────────────────────────────────────────────
from config import CONFIG

# ── Import the LAX_torch IC generator ────────────────────────────────────────
from numerical_solvers.LAX_torch import setup_power_spectrum_ic_torch

# =============================================================================
# Parameters  (edit here or override via CONFIG)
# =============================================================================
ps_cfg = CONFIG.power_spectrum
domain_cfg = CONFIG.domain

DIMENSION = 2  # 2-D IC fields

# Grid
N = ps_cfg.n_grid  # e.g. 400
Lx = domain_cfg.wave * domain_cfg.num_of_waves  # physical domain length

domain_params = {
    "Lx": Lx,
    "Ly": Lx,  # square domain
    "nx": N,
    "ny": N,
}

ps_params = {
    "rho_o": CONFIG.physics.rho_o,
    "power_index": float(ps_cfg.power_exponent),
    "amplitude": CONFIG.physics.a
    * CONFIG.physics.cs,  # must match train.py: v_1 = a*cs = 1.2
    "random_seed": ps_cfg.random_seed,
}

# Output directory (same folder as this script)
OUT_DIR = str(SCRIPT_DIR)

# =============================================================================
# Generate IC fields
# =============================================================================
print(f"Generating {DIMENSION}D power-spectrum IC fields …")
print(f"  Grid : {N} × {N}")
print(f"  Domain : Lx = Ly = {Lx:.3f}")
print(f"  Power index : {ps_params['power_index']}")
print(
    f"  Amplitude  : {ps_params['amplitude']}  (= a × cs = {CONFIG.physics.a} × {CONFIG.physics.cs})"
)
print(f"  Random seed : {ps_params['random_seed']}")

ic = setup_power_spectrum_ic_torch(domain_params, ps_params, dimension=DIMENSION)

vx0 = ic["vx"]
vy0 = ic["vy"]

print(f"\nvx0 shape : {tuple(vx0.shape)}  dtype : {vx0.dtype}")
print(f"vy0 shape : {tuple(vy0.shape)}  dtype : {vy0.dtype}")
print(f"vx0  min/max : {vx0.min().item():.6e} / {vx0.max().item():.6e}")
print(f"vy0  min/max : {vy0.min().item():.6e} / {vy0.max().item():.6e}")

# =============================================================================
# Save to binary files
# =============================================================================
vx_path = os.path.join(OUT_DIR, "vx_ic.bin")
vy_path = os.path.join(OUT_DIR, "vy_ic.bin")

vx0.cpu().numpy().astype(np.float64).tofile(vx_path)
vy0.cpu().numpy().astype(np.float64).tofile(vy_path)

print(f"\nSaved  →  {vx_path}")
print(f"Saved  →  {vy_path}")
print(f"\nTo reload in Python:")
print(f"  vx = np.fromfile('{vx_path}', dtype=np.float64).reshape({N}, {N})")
print(f"  vy = np.fromfile('{vy_path}', dtype=np.float64).reshape({N}, {N})")
print("\nDone.")
