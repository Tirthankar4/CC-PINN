"""
step2_compare.py
================
Run this on Windows (needs pytorch, which is already installed):

    cd "C:\\Users\\tirth\\Documents\\Projects\\GRINN\\Power spectrum perturbations"
    py step2_compare.py

Loads the Athena cache saved by step1, loads the trained PINN model,
and saves a comparison figure to the Desktop.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from torch.autograd import Variable

MODEL_PATH = r"C:\Users\tirth\OneDrive\Desktop\gravitational collapse results\With Athena\2D power spectrum\Supersonic\Seed 4\t = 0.6\Baseline\model.pth"
OUTPUT_DENSITY = r"C:\Users\tirth\OneDrive\Desktop\grinn_vs_athena_density_new.png"
OUTPUT_VELOCITY = r"C:\Users\tirth\OneDrive\Desktop\grinn_vs_athena_velocity_new.png"

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from config import (
    xmin, ymin, wave, a, cs, harmonics,
    DIMENSION, N_GRID, PERTURBATION_TYPE, RANDOM_SEED, num_of_waves,
)
from core.model_architecture import PINN
from core.initial_conditions import initialize_shared_velocity_fields
from visualization.Plotting import set_shared_velocity_fields

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# 1. Load Athena cache saved by step1
# ---------------------------------------------------------------------------
cache_dir = os.path.join(PROJECT, "athena_cache")
if not os.path.isdir(cache_dir):
    sys.exit("ERROR: athena_cache/ folder not found. Run step1_extract_athena.py in WSL first.")

times = np.load(os.path.join(cache_dir, "times.npy"))
cache = {}
for t in times:
    tag = f"{t:.6f}".replace(".", "p")
    cache[t] = {
        field: np.load(os.path.join(cache_dir, f"{field}_{tag}.npy"))
        for field in ("x", "y", "rho", "vx", "vy")
    }
print(f"Loaded Athena cache: {len(times)} snapshots")
print("Times:", [f"{t:.4f}" for t in times])

# ---------------------------------------------------------------------------
# 2. Load model
# ---------------------------------------------------------------------------
lam  = wave
xmax = xmin + lam * num_of_waves
ymax = ymin + lam * num_of_waves

if not os.path.exists(MODEL_PATH):
    sys.exit(f"ERROR: model not found at {MODEL_PATH}")

net = PINN(dimension=DIMENSION, n_harmonics=harmonics)
net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
net.set_domain(rmin=[xmin, ymin], rmax=[xmax, ymax], dimension=DIMENSION)
net = net.to(device).eval()
print(f"Loaded model: {MODEL_PATH}")

# ---------------------------------------------------------------------------
# 3. Shared velocity fields (power spectrum only)
# ---------------------------------------------------------------------------
if str(PERTURBATION_TYPE).lower() == "power_spectrum":
    v_1 = a * cs
    result = initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=RANDOM_SEED)
    if len(result) == 3:
        vx_np, vy_np, _ = result
        set_shared_velocity_fields(vx_np, vy_np)
    else:
        vx_np, vy_np, vz_np, _ = result
        set_shared_velocity_fields(vx_np, vy_np, vz_np)
    print("Velocity fields initialised.")

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
Q = N_GRID
xs = np.linspace(xmin, xmax, Q, endpoint=False)
ys = np.linspace(ymin, ymax, Q, endpoint=False)
tau, phi_grid = np.meshgrid(xs, ys, indexing="ij")
pts = np.column_stack([tau.ravel(), phi_grid.ravel()]).astype(np.float32)

n_times = len(times)

# ---------------------------------------------------------------------------
# Collect PINN + Athena data for all times first (reuse across both figures)
# ---------------------------------------------------------------------------
rows_data = []   # list of dicts, one per time

for t in times:
    t_arr = np.full((Q * Q, 1), t, dtype=np.float32)
    pt_x = Variable(torch.from_numpy(pts[:, 0:1]), requires_grad=True).to(device)
    pt_y = Variable(torch.from_numpy(pts[:, 1:2]), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_arr),       requires_grad=True).to(device)
    with torch.no_grad():
        out = net([pt_x, pt_y, pt_t])

    rho_pinn = out[:, 0].cpu().numpy().reshape(Q, Q)
    vx_pinn  = out[:, 1].cpu().numpy().reshape(Q, Q)
    vy_pinn  = out[:, 2].cpu().numpy().reshape(Q, Q)
    vmag_pinn = np.sqrt(vx_pinn**2 + vy_pinn**2)

    entry = cache[t]
    def _interp(field_key):
        return RegularGridInterpolator(
            (entry["x"], entry["y"]), entry[field_key],
            method="linear", bounds_error=False, fill_value=None,
        )(pts).reshape(Q, Q)

    rho_ath   = _interp("rho")
    vx_ath    = _interp("vx")
    vy_ath    = _interp("vy")
    vmag_ath  = np.sqrt(vx_ath**2 + vy_ath**2)

    # Error metric: RMS-normalised absolute error (expressed as a percentage).
    #
    #   ε = 100 × |GRINN - Athena| / RMS(Athena)
    #
    # RMS(Athena) = sqrt(mean(Athena²)) is a single global scale per timestep,
    # immune to local zeros (velocity stagnation points, low-density voids).
    #
    # WHY NOT local relative error 100×|diff|/|Athena_local|?
    #   Blows up wherever Athena passes through zero.
    #
    # WHY NOT SMAPE 200×|diff|/(GRINN+Athena)?
    #   Denominator includes GRINN's own prediction, making cross-run
    #   comparisons misleading (higher-amplitude models look artificially worse).
    #
    # WHY RMS normalisation?
    #   Independent of GRINN's output, immune to local zeros, and gives a
    #   physically meaningful scale: "fraction of the typical field magnitude".
    rho_rms  = np.sqrt(np.mean(rho_ath  ** 2)) + 1e-6
    vmag_rms = np.sqrt(np.mean(vmag_ath ** 2)) + 1e-6
    eps_rho  = 100.0 * np.abs(rho_pinn  - rho_ath)  / rho_rms
    eps_vmag = 100.0 * np.abs(vmag_pinn - vmag_ath)  / vmag_rms

    # Signed residual: GRINN − Athena (same RMS normalisation, no absolute value).
    # Positive (red in RdBu_r) = GRINN over-predicts; negative (blue) = under-predicts.
    # A dipole pattern (red on one side / blue on the other of a feature) is the
    # visual signature of a spatially phase-shifted prediction — this explains
    # how a model can show higher ε despite having a plausible amplitude range.
    diff_rho  = (rho_pinn  - rho_ath)  / rho_rms  * 100.0
    diff_vmag = (vmag_pinn - vmag_ath) / vmag_rms * 100.0

    rows_data.append(dict(
        t=t,
        rho_pinn=rho_pinn,   rho_ath=rho_ath,
        eps_rho=eps_rho,     diff_rho=diff_rho,
        vx_pinn=vx_pinn,     vx_ath=vx_ath,
        vy_pinn=vy_pinn,     vy_ath=vy_ath,
        vmag_pinn=vmag_pinn, vmag_ath=vmag_ath,
        eps_vmag=eps_vmag,   diff_vmag=diff_vmag,
    ))
    print(f"  t={t:.3f}  ρ: med ε={np.median(eps_rho):.1f}%  90th={np.percentile(eps_rho,90):.1f}%"
          f"   |v|: med ε={np.median(eps_vmag):.1f}%  90th={np.percentile(eps_vmag,90):.1f}%")


def _make_figure(rows_data, field_pinn_key, field_ath_key,
                 eps_key, diff_key, cmap, suptitle, outpath):
    # 4 columns: GRINN | Athena++ | |eps| (magnitude) | GRINN-Athena (signed)
    fig, axes = plt.subplots(n_times, 4, figsize=(20, 4 * n_times), constrained_layout=True)
    if n_times == 1:
        axes = axes.reshape(1, -1)

    # ------------------------------------------------------------------
    # Shared colour scales across ALL timestep rows.
    #
    # |eps| panel  -> sequential "YlOrRd", vmin=0, vmax=global 99th pct.
    #   Correct for a strictly non-negative quantity. Diverging maps
    #   (e.g. "coolwarm") waste the blue half and produce a grey panel
    #   when errors are near zero.
    #
    # signed residual panel -> diverging "RdBu_r", symmetric about zero.
    #   Red = GRINN over-predicts, blue = under-predicts.
    #   A dipole (red patch beside a blue patch on the same feature) is
    #   the visual signature of a spatially phase-shifted prediction --
    #   the "double-penalty" effect that explains how a model can have
    #   higher eps despite its amplitude range looking closer to Athena.
    # ------------------------------------------------------------------
    all_eps  = np.concatenate([d[eps_key ].ravel() for d in rows_data])
    all_diff = np.concatenate([d[diff_key].ravel() for d in rows_data])
    eps_vmax  = np.percentile(all_eps, 99)
    diff_vlim = np.percentile(np.abs(all_diff), 99)
    print(f"  |eps| scale  : 0 - {eps_vmax:.1f}%  (global 99th pct)")
    print(f"  diff scale : +/-{diff_vlim:.1f}%  (global 99th pct of |diff|)")

    for row, d in enumerate(rows_data):
        t         = d["t"]
        pinn_data = d[field_pinn_key]
        ath_data  = d[field_ath_key]
        eps       = d[eps_key]
        diff      = d[diff_key]
        med_eps   = np.median(eps)
        p90_eps   = np.percentile(eps, 90)

        skip = max(1, Q // 20)
        sk   = (slice(None, None, skip), slice(None, None, skip))

        cols_spec = [
            (pinn_data, f"GRINN  t={t:.2f}",           cmap,     {}),
            (ath_data,  f"Athena++  t={t:.2f}",         cmap,     {}),
            (eps,       f"|eps| (%)  t={t:.2f}",         "YlOrRd", {"vmin": 0,           "vmax": eps_vmax }),
            (diff,      f"GRINN-Athena (%)  t={t:.2f}", "RdBu_r", {"vmin": -diff_vlim,  "vmax": diff_vlim}),
        ]

        for col, (data, title, cm, kwargs) in enumerate(cols_spec):
            ax = axes[row, col]
            im = ax.pcolormesh(tau, phi_grid, data, shading="auto", cmap=cm, **kwargs)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            plt.colorbar(im, ax=ax, shrink=0.75)

            if col == 0:
                ax.quiver(tau[sk], phi_grid[sk], d["vx_pinn"][sk], d["vy_pinn"][sk],
                          color="k", alpha=0.5, width=0.003, headwidth=3)
            if col == 1:
                ax.quiver(tau[sk], phi_grid[sk], d["vx_ath"][sk], d["vy_ath"][sk],
                          color="k", alpha=0.5, width=0.003, headwidth=3)
            if col == 2:
                ax.text(0.03, 0.04,
                        f"med={med_eps:.1f}%\n90th={p90_eps:.1f}%",
                        transform=ax.transAxes, fontsize=8,
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.suptitle(suptitle, fontsize=13)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved -> {outpath}")
    plt.close(fig)


_make_figure(rows_data,
             "rho_pinn", "rho_ath", "eps_rho", "diff_rho",
             "YlOrBr",
             "GRINN vs Athena++  |  density  (arrows = velocity)",
             OUTPUT_DENSITY)

_make_figure(rows_data,
             "vmag_pinn", "vmag_ath", "eps_vmag", "diff_vmag",
             "viridis",
             "GRINN vs Athena++  |  velocity magnitude  (arrows = velocity)",
             OUTPUT_VELOCITY)
