"""
step2_compare.py
================
Run this on Windows (needs pytorch, which is already installed):

    cd "C:\\Users\\tirth\\Documents\\Projects\\GRINN\\Power spectrum perturbations"
    py step2_compare.py

Loads the Athena cache saved by step1, loads the trained PINN model,
and saves a comparison figure to the Desktop.

5-column layout per row:
  col 0  GRINN field
  col 1  Athena++ field
  col 2  |ε| map
  col 3  Power spectrum (log-log)
  col 4  ln(rho_max) vs t  [same curve every row, vertical marker at current t]
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from torch.autograd import Variable

MODEL_PATH = r"C:\Users\tirth\OneDrive\Desktop\gravitational collapse results\With Athena\2D power spectrum\Supersonic\Seed 4\t = 0.6\Adaptive Collocation\model.pth"
OUTPUT_DENSITY = r"C:\Users\tirth\OneDrive\Desktop\grinn_vs_athena_density.png"
OUTPUT_VELOCITY = r"C:\Users\tirth\OneDrive\Desktop\grinn_vs_athena_velocity.png"

# ---------------------------------------------------------------------------
# Radial (azimuthally-averaged) power spectrum of a 2-D field
# ---------------------------------------------------------------------------
def radial_power_spectrum(field_2d, dx):
    Nx, Ny  = field_2d.shape
    fft2    = np.fft.fft2(field_2d)
    power   = (np.abs(fft2) ** 2) * (dx ** 2) / (Nx * Ny)

    kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K       = np.sqrt(KX**2 + KY**2).ravel()
    power   = power.ravel()

    k_max   = 0.5 * (2 * np.pi / dx) * np.sqrt(2)
    n_bins  = Nx // 2
    edges   = np.linspace(0.0, k_max, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    ps_mean = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (K >= edges[i]) & (K < edges[i + 1])
        if mask.any():
            ps_mean[i] = power[mask].mean()

    nonzero = ps_mean > 0
    if nonzero.any():
        last    = np.where(nonzero)[0][-1] + 1
        centres = centres[:last]
        ps_mean = ps_mean[:last]

    return centres, ps_mean


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

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
cache_dir = os.path.join(str(SCRIPT_DIR), "athena_cache")
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
# 2. Load rho_max growth curve (all snapshots, saved by step1)
# ---------------------------------------------------------------------------
rho_max_times_path = os.path.join(cache_dir, "rho_max_times.npy")
rho_max_curve_path = os.path.join(cache_dir, "rho_max_curve.npy")

if os.path.exists(rho_max_times_path) and os.path.exists(rho_max_curve_path):
    rho_max_times_all = np.load(rho_max_times_path)
    rho_max_vals_all  = np.load(rho_max_curve_path)
    ln_rho_max_ath    = np.log(rho_max_vals_all)
    has_growth_curve  = True
    print(f"Loaded rho_max curve: {len(rho_max_times_all)} points, "
          f"t in [{rho_max_times_all[0]:.3f}, {rho_max_times_all[-1]:.3f}]")
else:
    has_growth_curve = False
    print("WARNING: rho_max_times.npy / rho_max_curve.npy not found. "
          "Re-run step1_extract_athena.py to generate them. "
          "Column 5 will be skipped.")

# ---------------------------------------------------------------------------
# 3. Load model
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
# 4. Shared velocity fields (power spectrum only)
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
# 5. Evaluate PINN + Athena at each snapshot time
# ---------------------------------------------------------------------------
Q = N_GRID
xs = np.linspace(xmin, xmax, Q, endpoint=False)
ys = np.linspace(ymin, ymax, Q, endpoint=False)
tau, phi_grid = np.meshgrid(xs, ys, indexing="ij")
pts = np.column_stack([tau.ravel(), phi_grid.ravel()]).astype(np.float32)

n_times = len(times)
rows_data = []

for t in times:
    t_arr = np.full((Q * Q, 1), t, dtype=np.float32)
    pt_x = Variable(torch.from_numpy(pts[:, 0:1]), requires_grad=True).to(device)
    pt_y = Variable(torch.from_numpy(pts[:, 1:2]), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_arr),       requires_grad=True).to(device)
    with torch.no_grad():
        out = net([pt_x, pt_y, pt_t])

    rho_pinn  = out[:, 0].cpu().numpy().reshape(Q, Q)
    vx_pinn   = out[:, 1].cpu().numpy().reshape(Q, Q)
    vy_pinn   = out[:, 2].cpu().numpy().reshape(Q, Q)
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

    rho_rms  = np.sqrt(np.mean(rho_ath  ** 2)) + 1e-6
    vmag_rms = np.sqrt(np.mean(vmag_ath ** 2)) + 1e-6
    eps_rho  = 100.0 * np.abs(rho_pinn  - rho_ath)  / rho_rms
    eps_vmag = 100.0 * np.abs(vmag_pinn - vmag_ath)  / vmag_rms

    dx = (xmax - xmin) / Q
    k_rho,   ps_rho_pinn  = radial_power_spectrum(rho_pinn,  dx)
    _,       ps_rho_ath   = radial_power_spectrum(rho_ath,   dx)
    k_vmag,  ps_vmag_pinn = radial_power_spectrum(vmag_pinn, dx)
    _,       ps_vmag_ath  = radial_power_spectrum(vmag_ath,  dx)

    rho_max_pinn_t = float(rho_pinn.max())

    rows_data.append(dict(
        t=t,
        rho_pinn=rho_pinn,   rho_ath=rho_ath,
        eps_rho=eps_rho,
        vx_pinn=vx_pinn,     vx_ath=vx_ath,
        vy_pinn=vy_pinn,     vy_ath=vy_ath,
        vmag_pinn=vmag_pinn, vmag_ath=vmag_ath,
        eps_vmag=eps_vmag,
        k_rho=k_rho,     ps_rho_pinn=ps_rho_pinn,   ps_rho_ath=ps_rho_ath,
        k_vmag=k_vmag,   ps_vmag_pinn=ps_vmag_pinn, ps_vmag_ath=ps_vmag_ath,
        rho_max_pinn=rho_max_pinn_t,
    ))
    print(f"  t={t:.3f}  rho_max_pinn={rho_max_pinn_t:.3f}  "
          f"rho: med ε={np.median(eps_rho):.1f}%  90th={np.percentile(eps_rho,90):.1f}%"
          f"   |v|: med ε={np.median(eps_vmag):.1f}%  90th={np.percentile(eps_vmag,90):.1f}%")

# ---------------------------------------------------------------------------
# 6. Precompute PINN growth curve points for overlay
#    If Athena full growth data is available, evaluate PINN on ALL of those times.
# ---------------------------------------------------------------------------
if has_growth_curve:
    print("Evaluating PINN rho_max over all Athena growth-curve times...")
    pinn_times_pts = np.array(rho_max_times_all)
    pinn_rho_max_pts = []

    with torch.no_grad():
        pt_x_curve = torch.from_numpy(pts[:, 0:1]).to(device)
        pt_y_curve = torch.from_numpy(pts[:, 1:2]).to(device)

        for t_curve in pinn_times_pts:
            t_arr_curve = np.full((Q * Q, 1), t_curve, dtype=np.float32)
            pt_t_curve = torch.from_numpy(t_arr_curve).to(device)
            out_curve = net([pt_x_curve, pt_y_curve, pt_t_curve])
            pinn_rho_max_pts.append(float(out_curve[:, 0].max().item()))

    pinn_rho_max_pts = np.array(pinn_rho_max_pts)
    print(f"Computed PINN rho_max curve: {len(pinn_times_pts)} points")
else:
    pinn_times_pts   = np.array([d["t"] for d in rows_data])
    pinn_rho_max_pts = np.array([d["rho_max_pinn"] for d in rows_data])

# ---------------------------------------------------------------------------
# 7. Figure-making function  (5-column layout)
# ---------------------------------------------------------------------------
def _make_figure(rows_data, field_pinn_key, field_ath_key,
                 eps_key, ps_pinn_key, ps_ath_key, k_key,
                 ps_ylabel, cmap, suptitle, outpath):
    """
    5-column layout per row:
      col 0  GRINN field          (pcolormesh)
      col 1  Athena++ field       (pcolormesh)
      col 2  |ε| map              (pcolormesh)
      col 3  Power spectrum       (log-log line plot)
      col 4  ln(rho_max) vs t     (line plot, vertical marker at current t)
    """
    n_cols = 5 if has_growth_curve else 4
    fig, axes = plt.subplots(n_times, n_cols,
                             figsize=(5 * n_cols, 4 * n_times),
                             constrained_layout=True)
    if n_times == 1:
        axes = axes.reshape(1, -1)

    for row, d in enumerate(rows_data):
        t         = d["t"]
        pinn_data = d[field_pinn_key]
        ath_data  = d[field_ath_key]
        eps       = d[eps_key]
        med_eps   = np.median(eps)
        p90_eps   = np.percentile(eps, 90)

        row_eps_vmax = np.percentile(eps, 99)
        skip = max(1, Q // 20)
        sk   = (slice(None, None, skip), slice(None, None, skip))

        # ── Columns 0-2: spatial maps ───────────────────────────────────
        map_cols = [
            (pinn_data, f"GRINN  t={t:.2f}",    cmap,       {}),
            (ath_data,  f"Athena++  t={t:.2f}",  cmap,       {}),
            (eps,       f"|ε| (%)  t={t:.2f}",   "coolwarm", {"vmin": 0, "vmax": row_eps_vmax}),
        ]
        for col, (data, title, cm, kwargs) in enumerate(map_cols):
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

        # ── Column 3: power spectrum ────────────────────────────────────
        ax_ps = axes[row, 3]
        k_arr   = np.asarray(d[k_key])
        ps_pinn = np.asarray(d[ps_pinn_key])
        ps_ath  = np.asarray(d[ps_ath_key])

        n_spec = min(len(k_arr), len(ps_pinn), len(ps_ath))
        if n_spec == 0:
            ax_ps.set_title(f"Power spectrum  t={t:.2f}", fontsize=10)
            ax_ps.set_xlabel(r"$k$  [rad / length]")
            ax_ps.set_ylabel(ps_ylabel)
            ax_ps.text(0.5, 0.5, "No spectrum data", transform=ax_ps.transAxes,
                       ha="center", va="center", fontsize=9)
            ax_ps.grid(True, which="both", ls=":", alpha=0.4)
        else:
            k_arr   = k_arr[:n_spec]
            ps_pinn = ps_pinn[:n_spec]
            ps_ath  = ps_ath[:n_spec]
            valid   = (ps_pinn > 0) & (ps_ath > 0)

            if np.any(valid):
                k_v, psp_v, psa_v = k_arr[valid], ps_pinn[valid], ps_ath[valid]
                ax_ps.loglog(k_v, psa_v, color="C1", lw=1.8, label="Athena++")
                ax_ps.loglog(k_v, psp_v, color="C0", lw=1.8, linestyle="--", label="GRINN")
                ax_ps.fill_between(k_v,
                                   np.minimum(psp_v, psa_v),
                                   np.maximum(psp_v, psa_v),
                                   alpha=0.18, color="grey", label="discrepancy")
                k_ref   = k_v[len(k_v) // 4]
                p_ref   = psa_v[len(k_v) // 4]
                k_guide = np.array([k_v[0], k_v[-1]])
                p_guide = p_ref * (k_guide / k_ref) ** (-4)
                ax_ps.loglog(k_guide, p_guide, "k:", lw=1.0, label=r"$k^{-4}$")
                ax_ps.legend(fontsize=7, loc="lower left")

            ax_ps.set_title(f"Power spectrum  t={t:.2f}", fontsize=10)
            ax_ps.set_xlabel(r"$k$  [rad / length]")
            ax_ps.set_ylabel(ps_ylabel)
            ax_ps.grid(True, which="both", ls=":", alpha=0.4)

        # ── Column 4: ln(rho_max) vs t ──────────────────────────────────
        if has_growth_curve:
            ax_g = axes[row, 4]

            # Athena++ full curve
            ax_g.plot(rho_max_times_all, ln_rho_max_ath,
                      color="C1", lw=1.8, label="Athena++")

            # PINN curve (full timeline if rho_max_times.npy is available)
            ax_g.plot(pinn_times_pts, np.log(np.maximum(pinn_rho_max_pts, 1e-12)),
                      color="C0", lw=1.5, linestyle="--", label="GRINN")

            # Vertical marker for this row's timestep
            y_lo = ln_rho_max_ath.min() - 0.2
            y_hi = ln_rho_max_ath.max() + 0.2
            ax_g.axvline(x=t, color="k", lw=1.2, linestyle=":", alpha=0.7)
            ax_g.text(t + 0.01 * (rho_max_times_all[-1] - rho_max_times_all[0]),
                      y_lo + 0.05 * (y_hi - y_lo),
                      f"t={t:.2f}", fontsize=8, va="bottom", color="k", alpha=0.8)
            ax_g.set_ylim(y_lo, y_hi)

            ax_g.set_xlabel(r"$t$")
            ax_g.set_ylabel(r"$\ln(\rho_{\mathrm{max}})$")
            ax_g.set_title(f"Growth rate  t={t:.2f}", fontsize=10)
            ax_g.legend(fontsize=7, loc="upper left")
            ax_g.grid(True, ls=":", alpha=0.4)

    fig.suptitle(suptitle, fontsize=13)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved -> {outpath}")
    plt.close(fig)


_make_figure(rows_data,
             field_pinn_key="rho_pinn",    field_ath_key="rho_ath",
             eps_key="eps_rho",
             ps_pinn_key="ps_rho_pinn",    ps_ath_key="ps_rho_ath",
             k_key="k_rho",
             ps_ylabel=r"$P_{\rho}(k)$",
             cmap="YlOrBr",
             suptitle="GRINN vs Athena++  |  density  (arrows = velocity)",
             outpath=OUTPUT_DENSITY)

_make_figure(rows_data,
             field_pinn_key="vmag_pinn",   field_ath_key="vmag_ath",
             eps_key="eps_vmag",
             ps_pinn_key="ps_vmag_pinn",   ps_ath_key="ps_vmag_ath",
             k_key="k_vmag",
             ps_ylabel=r"$P_{|v|}(k)$",
             cmap="viridis",
             suptitle="GRINN vs Athena++  |  velocity magnitude  (arrows = velocity)",
             outpath=OUTPUT_VELOCITY)
