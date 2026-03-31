import argparse
import numpy as np
import os
import sys

# Add parent directory to path for imports when running from utilities directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    cs,
    rho_o,
    const,
    G,
    wave,
    num_of_waves,
    tmax,
    DIMENSION,
    PERTURBATION_TYPE,
    KX,
    KY,
    KZ,
    RANDOM_SEED,
)
from numerical_solvers.LAX_torch import lax_solver_torch, has_gpu, device


def _default_grid_n(dimension):
    # Keep defaults modest to avoid heavy runs by default.
    if dimension == 3:
        return 48
    return 96


def _build_domain_params(dimension, lam, waves, grid_n):
    Lx = lam * waves
    if dimension == 1:
        return {"Lx": Lx, "nx": grid_n}
    if dimension == 2:
        return {"Lx": Lx, "Ly": Lx, "nx": grid_n, "ny": grid_n}
    if dimension == 3:
        return {"Lx": Lx, "Ly": Lx, "Lz": Lx, "nx": grid_n, "ny": grid_n, "nz": grid_n}
    raise ValueError(f"Unsupported DIMENSION={dimension}")


def _build_ic_params(ic_type, dimension, random_seed):
    if ic_type == "sinusoidal":
        params = {"KX": KX, "KY": KY}
        if dimension == 3:
            params["KZ"] = KZ
        return params
    if ic_type == "power_spectrum":
        # Use solver defaults for power_index/amplitude; keep only seed for reproducibility.
        return {"random_seed": random_seed}
    return None


def _compute_total_mass(rho, domain_params, dimension):
    cell_volume = domain_params["Lx"] / domain_params["nx"]
    if dimension >= 2:
        cell_volume *= domain_params["Ly"] / domain_params["ny"]
    if dimension == 3:
        cell_volume *= domain_params["Lz"] / domain_params["nz"]
    return float(np.sum(rho) * cell_volume)


def _compute_total_momentum(rho, velocities, domain_params, dimension):
    """Compute total momentum in each direction."""
    cell_volume = domain_params["Lx"] / domain_params["nx"]
    if dimension >= 2:
        cell_volume *= domain_params["Ly"] / domain_params["ny"]
    if dimension == 3:
        cell_volume *= domain_params["Lz"] / domain_params["nz"]
    
    momentum = []
    for vel in velocities:
        mom = float(np.sum(rho * vel) * cell_volume)
        momentum.append(mom)
    
    return momentum


def run_mass_conservation(time, grid_n, n_times, rho_1, nu, gravity, ic_type):
    dimension = DIMENSION
    lam = wave
    domain_params = _build_domain_params(dimension, lam, num_of_waves, grid_n)

    physics_params = {
        "c_s": cs,
        "rho_o": rho_o,
        "const": const,
        "G": G,
    }
    if ic_type == "sinusoidal":
        physics_params["rho_1"] = rho_1
        physics_params["lam"] = lam

    ic_params = _build_ic_params(ic_type, dimension, RANDOM_SEED)
    options = {"gravity": gravity, "nu": nu, "comparison": False, "isplot": False}

    save_times = np.linspace(0.0, time, n_times)
    results = lax_solver_torch(
        time,
        domain_params,
        physics_params,
        ic_type=ic_type,
        ic_params=ic_params,
        options=options,
        save_times=save_times,
    )

    masses = []
    momenta_x = []
    momenta_y = []
    momenta_z = [] if dimension == 3 else None
    
    for t_val in save_times:
        snapshot = results.get(t_val)
        if snapshot is None:
            nearest = min(results.keys(), key=lambda k: abs(k - t_val))
            snapshot = results[nearest]
        
        mass = _compute_total_mass(snapshot.density, domain_params, dimension)
        masses.append(mass)
        
        momentum = _compute_total_momentum(snapshot.density, snapshot.velocity_components, domain_params, dimension)
        momenta_x.append(momentum[0])
        if dimension >= 2:
            momenta_y.append(momentum[1])
        if dimension == 3:
            momenta_z.append(momentum[2])

    mass0 = masses[0]
    mass_drifts = [(m - mass0) / mass0 * 100.0 for m in masses]
    
    px0 = momenta_x[0]
    py0 = momenta_y[0] if dimension >= 2 else None
    pz0 = momenta_z[0] if dimension == 3 else None
    
    # Handle near-zero initial momentum
    if abs(px0) < 1e-10:
        px_drifts = [(px - px0) * 1000 for px in momenta_x]  # absolute change in milliunit
        px_label = "Momentum X abs change (×10⁻³)"
    else:
        px_drifts = [(px - px0) / abs(px0) * 100.0 for px in momenta_x]
        px_label = "Momentum X drift (%)"
    
    if dimension >= 2:
        if abs(py0) < 1e-10:
            py_drifts = [(py - py0) * 1000 for py in momenta_y]
            py_label = "Momentum Y abs change (×10⁻³)"
        else:
            py_drifts = [(py - py0) / abs(py0) * 100.0 for py in momenta_y]
            py_label = "Momentum Y drift (%)"
    
    if dimension == 3:
        if abs(pz0) < 1e-10:
            pz_drifts = [(pz - pz0) * 1000 for pz in momenta_z]
            pz_label = "Momentum Z abs change (×10⁻³)"
        else:
            pz_drifts = [(pz - pz0) / abs(pz0) * 100.0 for pz in momenta_z]
            pz_label = "Momentum Z drift (%)"

    print("\nLAX conservation diagnostics (GPU solver)")
    print(f"  Device: {device}")
    print(f"  Dimension: {dimension}D")
    print(f"  Grid: {grid_n} per axis")
    print(f"  IC type: {ic_type}")
    print(f"  Gravity: {gravity}")
    print(f"  Time: {time}")
    
    print("\n" + "="*80)
    print("MASS CONSERVATION")
    print("="*80)
    print(f"{'Time':>8s}  {'Total Mass':>14s}  {'Drift (%)':>12s}")
    for t_val, mass, drift in zip(save_times, masses, mass_drifts):
        print(f"{t_val:8.4f}  {mass: .8e}  {drift: .6e}")
    
    abs_mass_drifts = [abs(d) for d in mass_drifts]
    print(f"\nMass0: {mass0:.8e}")
    print(f"Max |mass drift| (%): {max(abs_mass_drifts):.6e}")
    
    print("\n" + "="*80)
    print("MOMENTUM CONSERVATION")
    print("="*80)
    print(f"{'Time':>8s}  {'Momentum X':>14s}  {px_label:>25s}")
    for t_val, px, drift in zip(save_times, momenta_x, px_drifts):
        print(f"{t_val:8.4f}  {px: .8e}  {drift: .6e}")
    
    abs_px_drifts = [abs(d) for d in px_drifts]
    print(f"\nMomentum X initial: {px0:.8e}")
    print(f"Max |momentum X drift|: {max(abs_px_drifts):.6e} {px_label.split()[-1]}")
    
    if dimension >= 2:
        print(f"\n{'Time':>8s}  {'Momentum Y':>14s}  {py_label:>25s}")
        for t_val, py, drift in zip(save_times, momenta_y, py_drifts):
            print(f"{t_val:8.4f}  {py: .8e}  {drift: .6e}")
        
        abs_py_drifts = [abs(d) for d in py_drifts]
        print(f"\nMomentum Y initial: {py0:.8e}")
        print(f"Max |momentum Y drift|: {max(abs_py_drifts):.6e} {py_label.split()[-1]}")
    
    if dimension == 3:
        print(f"\n{'Time':>8s}  {'Momentum Z':>14s}  {pz_label:>25s}")
        for t_val, pz, drift in zip(save_times, momenta_z, pz_drifts):
            print(f"{t_val:8.4f}  {pz: .8e}  {drift: .6e}")
        
        abs_pz_drifts = [abs(d) for d in pz_drifts]
        print(f"\nMomentum Z initial: {pz0:.8e}")
        print(f"Max |momentum Z drift|: {max(abs_pz_drifts):.6e} {pz_label.split()[-1]}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Check LAX (GPU) mass conservation.")
    parser.add_argument("--time", type=float, default=tmax, help="Final simulation time")
    parser.add_argument("--grid", type=int, default=_default_grid_n(DIMENSION), help="Grid resolution per axis")
    parser.add_argument("--n-times", type=int, default=10, help="Number of saved time points")
    parser.add_argument("--rho-1", type=float, default=0.1, help="Sinusoidal density perturbation amplitude")
    parser.add_argument("--nu", type=float, default=0.5, help="Courant number")
    parser.add_argument("--no-gravity", action="store_true", help="Disable self-gravity")
    parser.add_argument(
        "--ic-type",
        type=str,
        default=str(PERTURBATION_TYPE).lower(),
        choices=["sinusoidal", "power_spectrum", "warm_start"],
        help="Initial condition type",
    )
    args = parser.parse_args()

    if not has_gpu:
        print("Warning: GPU not available. Solver will run on CPU.")

    if args.ic_type == "warm_start":
        raise ValueError("warm_start requires provided fields; use sinusoidal or power_spectrum.")

    run_mass_conservation(
        time=args.time,
        grid_n=args.grid,
        n_times=args.n_times,
        rho_1=args.rho_1,
        nu=args.nu,
        gravity=not args.no_gravity,
        ic_type=args.ic_type,
    )


if __name__ == "__main__":
    main()
