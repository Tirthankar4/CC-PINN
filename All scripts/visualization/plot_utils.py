from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.autograd import Variable
import torch
import scipy
import os
import time
from numerical_solvers.LAX import lax_solver
from numerical_solvers.LAX_torch import lax_solver_torch
from core.device import has_cuda_gpu, has_mps_backend, get_compute_device, clear_cuda_cache

# Explicitly export functions for 'from .plot_utils import *'
__all__ = [
    # Common modules
    'np',
    'plt',
    'torch',
    'Variable',
    'animation',
    'os',
    # Functions
    '_clear_cuda_cache',
    'find_max_contrast_slice',
    'find_max_density_slice',
    'analyze_z_variation',
    '_timed_call',
    '_get_fd_nu',
    '_build_input_list',
    '_split_outputs',
    'set_shared_velocity_fields',
    'call_unified_3d_solver',
    'get_fd_default_params',
    'compute_fd_data_cache',
    # Config values (will be imported later in this file)
    'has_gpu',
    'has_mps',
    'device',
    'SAVE_STATIC_SNAPSHOTS',
    'SNAPSHOT_DIR',
    'PERTURBATION_TYPE',
    'cs',
    'const',
    'G',
    'rho_o',
    'TIMES_1D',
    'a',
    'KX',
    'KY',
    'KZ',
    'FD_N_1D',
    'FD_N_2D',
    'FD_N_3D',
    'FD_NU_SINUSOIDAL',
    'FD_NU_POWER',
    'POWER_EXPONENT',
    'N_GRID',
    'N_GRID_3D',
    'DIMENSION',
    'SLICE_Y',
    'SLICE_Z',
    'RANDOM_SEED',
    'SHOW_LINEAR_THEORY'
]

def _clear_cuda_cache():
    """
    Comprehensive GPU cache clearing before FD solver runs.
    
    This function does more than just empty_cache() - it also:
    1. Forces garbage collection
    2. Synchronizes CUDA operations
    3. Clears the cache allocator
    
    This helps prevent OOM errors when FD solver needs GPU memory.
    """
    clear_cuda_cache()

def find_max_contrast_slice(rho_volume, z_coords):
    """
    Find z-slice with maximum density contrast (range).
    
    Best for showing interesting physics - regions with both 
    high density (collapse) and low density (rarefaction).
    
    Args:
        rho_volume: (Nx, Ny, Nz) 3D density field
        z_coords: (Nz,) z-coordinate array
    
    Returns:
        z_idx: Index of optimal z-slice
        z_val: Value of z at that slice
        contrast: Density contrast at that slice
    """
    contrast_per_z = []
    for z_idx in range(rho_volume.shape[2]):
        # np.ptp = peak-to-peak (max - min)
        contrast = np.ptp(rho_volume[:, :, z_idx])
        contrast_per_z.append(contrast)
    
    best_z_idx = np.argmax(contrast_per_z)
    best_z_val = z_coords[best_z_idx]
    max_contrast = contrast_per_z[best_z_idx]
    
    return best_z_idx, best_z_val, max_contrast


def find_max_density_slice(rho_volume, z_coords):
    """
    Find z-slice with maximum integrated density.
    
    Good for showing where collapse is strongest.
    
    Args:
        rho_volume: (Nx, Ny, Nz) 3D density field
        z_coords: (Nz,) z-coordinate array
    
    Returns:
        z_idx: Index of optimal z-slice
        z_val: Value of z at that slice
        total_density: Integrated density at that slice
    """
    density_per_z = np.sum(rho_volume, axis=(0, 1))
    
    best_z_idx = np.argmax(density_per_z)
    best_z_val = z_coords[best_z_idx]
    max_density = density_per_z[best_z_idx]
    
    return best_z_idx, best_z_val, max_density


def analyze_z_variation(rho_volume, z_coords, t, save_dir=None):
    """
    Diagnostic tool: visualize how density varies across z.
    
    Call this once to understand your 3D data before choosing
    which selection method to use.
    
    Args:
        rho_volume: (Nx, Ny, Nz) 3D density field
        z_coords: (Nz,) z-coordinate array
        t: Current time (for labeling)
        save_dir: Optional directory to save figure
    
    Returns:
        fig: matplotlib figure
    """
    # Compute metrics for each z-slice
    density_per_z = np.sum(rho_volume, axis=(0, 1))
    contrast_per_z = [np.ptp(rho_volume[:, :, i]) for i in range(len(z_coords))]
    
    # Find optimal slices
    z_max_dens = z_coords[np.argmax(density_per_z)]
    z_max_cont = z_coords[np.argmax(contrast_per_z)]
    z_median = z_coords[np.argmin(np.abs(density_per_z - np.median(density_per_z)))]
    
    # Create diagnostic plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Integrated density vs z
    ax1.plot(z_coords, density_per_z, 'o-', linewidth=2, markersize=4)
    ax1.axvline(z_max_dens, color='red', linestyle='--', alpha=0.7, 
                label=f'Max density: z={z_max_dens:.3f}')
    ax1.axvline(z_median, color='green', linestyle='--', alpha=0.7,
                label=f'Median: z={z_median:.3f}')
    ax1.set_xlabel('z', fontsize=12)
    ax1.set_ylabel('Integrated Density', fontsize=12)
    ax1.set_title(f'Density Distribution at t={t:.2f}', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right: Contrast vs z
    ax2.plot(z_coords, contrast_per_z, 's-', color='orange', linewidth=2, markersize=4)
    ax2.axvline(z_max_cont, color='red', linestyle='--', alpha=0.7,
                label=f'Max contrast: z={z_max_cont:.3f}')
    ax2.set_xlabel('z', fontsize=12)
    ax2.set_ylabel('Density Contrast (max - min)', fontsize=12)
    ax2.set_title(f'Structure Distribution at t={t:.2f}', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Print summary
    print(f"\\n{'='*60}")
    print(f"Z-Slice Analysis at t={t:.2f}")
    print(f"{'='*60}")
    print(f"Max density slice:   z = {z_max_dens:.3f} (density = {np.max(density_per_z):.2e})")
    print(f"Max contrast slice:  z = {z_max_cont:.3f} (contrast = {np.max(contrast_per_z):.2e})")
    print(f"Median density slice: z = {z_median:.3f}")
    print(f"Density range across z: [{np.min(density_per_z):.2e}, {np.max(density_per_z):.2e}]")
    print(f"Contrast range across z: [{np.min(contrast_per_z):.2e}, {np.max(contrast_per_z):.2e}]")
    print(f"{'='*60}\\n")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        savepath = os.path.join(save_dir, f'z_variation_analysis_t{t:.2f}.png')
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Saved z-variation analysis to {savepath}")
    
    return fig


def _timed_call(label, fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"[Timing] {label} took {elapsed:.2f}s")
    return result

from config import (SAVE_STATIC_SNAPSHOTS, SNAPSHOT_DIR, PERTURBATION_TYPE, cs, const, G, rho_o,
                    TIMES_1D, a, KX, KY, KZ, FD_N_1D, FD_N_2D, FD_N_3D, FD_NU_SINUSOIDAL,
                    FD_NU_POWER, POWER_EXPONENT,
                    N_GRID, N_GRID_3D, DIMENSION, SLICE_Y, SLICE_Z, GRAVITY)
from config import RANDOM_SEED, SHOW_LINEAR_THEORY

# Global variable to store shared velocity fields for plotting
_shared_vx_np = None
_shared_vy_np = None
_shared_vz_np = None

def _get_fd_nu():
    return FD_NU_POWER if str(PERTURBATION_TYPE).lower() == "power_spectrum" else FD_NU_SINUSOIDAL

def _build_input_list(x_tensor, t_tensor, y_tensor=None, z_tensor=None):
    coords = [x_tensor]
    if DIMENSION >= 2:
        if y_tensor is None:
            y_tensor = torch.full_like(x_tensor, SLICE_Y)
        coords.append(y_tensor)
    if DIMENSION >= 3:
        if z_tensor is None:
            z_tensor = torch.full_like(x_tensor, SLICE_Z)
        coords.append(z_tensor)
    coords.append(t_tensor)
    return coords

def _split_outputs(outputs):
    rho = outputs[:, 0:1]
    vx = outputs[:, 1:2]
    vy = outputs[:, 2:3] if DIMENSION >= 2 else None
    if DIMENSION == 1:
        phi = outputs[:, 2:3] if GRAVITY else None
        vz = None
    elif DIMENSION == 2:
        phi = outputs[:, 3:4] if GRAVITY else None
        vz = None
    else:
        vz = outputs[:, 3:4]
        phi = outputs[:, 4:5] if GRAVITY else None
    return rho, vx, vy, vz, phi

def set_shared_velocity_fields(vx_np, vy_np, vz_np=None):
    """
    Set shared velocity fields for consistent FD plotting.
    
    Args:
        vx_np: X-velocity field (2D or 3D numpy array)
        vy_np: Y-velocity field (2D or 3D numpy array)
        vz_np: Z-velocity field (3D numpy array, optional for 2D)
    """
    global _shared_vx_np, _shared_vy_np, _shared_vz_np
    _shared_vx_np = vx_np
    _shared_vy_np = vy_np
    _shared_vz_np = vz_np

def call_unified_3d_solver(time, lam, num_of_waves, rho_1, nu=None,
                           use_velocity_ps=None, ps_index=None, vel_rms=None, random_seed=None,
                           save_times=None):
    """
    Helper function to call unified 3D LAX solver with proper IC type.
    
    Args:
        time: Final simulation time
        lam: Wavelength
        num_of_waves: Number of waves in domain
        rho_1: Density perturbation amplitude
        nu: Courant number (defaults to config based on perturbation type)
        use_velocity_ps: Whether to use power spectrum IC (defaults to config)
        ps_index: Power spectrum index (defaults to POWER_EXPONENT)
        vel_rms: Velocity RMS amplitude (defaults to a*cs)
        random_seed: Random seed (defaults to RANDOM_SEED)
        save_times: Optional list of times to save snapshots (default: None)
    
    Returns:
        If save_times is None: SimulationResult object
        If save_times is provided: Dict mapping time -> SimulationResult
    """
    # Use config defaults if not specified
    if use_velocity_ps is None:
        use_velocity_ps = (str(PERTURBATION_TYPE).lower() == "power_spectrum")
    if ps_index is None:
        ps_index = POWER_EXPONENT
    if vel_rms is None:
        vel_rms = a * cs
    if random_seed is None:
        random_seed = RANDOM_SEED
    
    # Set up domain parameters
    Lx = Ly = Lz = lam * num_of_waves
    Nx = Ny = Nz = FD_N_3D
    
    domain_params = {
        'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
        'nx': Nx, 'ny': Ny, 'nz': Nz
    }
    
    # Set up physics parameters
    physics_params = {
        'c_s': cs,
        'rho_o': rho_o,
        'const': const,
        'G': G,
        'rho_1': rho_1,
        'lam': lam
    }
    
    # Set up options
    if nu is None:
        nu = _get_fd_nu()
    options = {
        'gravity': GRAVITY,
        'nu': nu,
        'comparison': False,
        'isplot': False
    }
    
    # Set up IC parameters based on perturbation type
    if use_velocity_ps:
        ic_type = 'power_spectrum'
        ic_params = {
            'power_index': ps_index,
            'amplitude': vel_rms,
            'random_seed': random_seed,
            'vx0_shared': _shared_vx_np,
            'vy0_shared': _shared_vy_np,
            'vz0_shared': _shared_vz_np
        }
    else:
        ic_type = 'sinusoidal'
        ic_params = {
            'KX': KX,
            'KY': KY,
            'KZ': KZ
        }
    
    # Call unified solver (GPU or CPU)
    if torch.cuda.is_available():
        _clear_cuda_cache()
        result = _timed_call(
            "LAX 3D (unified torch)",
            lax_solver_torch,
            time=time, domain_params=domain_params,
            physics_params=physics_params,
            ic_type=ic_type, ic_params=ic_params,
            options=options, save_times=save_times
        )
    else:
        # CPU solver doesn't support save_times yet, so we'll need to handle it differently
        # For now, if save_times is provided, we'll use the torch version even if on CPU
        if save_times is not None:
            result = _timed_call(
                "LAX 3D (unified torch cpu)",
                lax_solver_torch,
                time=time, domain_params=domain_params,
                physics_params=physics_params,
                ic_type=ic_type, ic_params=ic_params,
                options=options, save_times=save_times
            )
        else:
            result = _timed_call(
                "LAX 3D (unified cpu)",
                lax_solver,
                time=time, domain_params=domain_params,
                physics_params=physics_params,
                ic_type=ic_type, ic_params=ic_params,
                options=options
            )
    
    return result

# Backward-compatible alias for any direct module references.
_call_unified_3d_solver = call_unified_3d_solver

def get_fd_default_params():
    """
    Get default FD parameters that match PINN training configuration.
    This ensures consistency between PINN and FD initial conditions.
    
    Returns:
        dict with keys: use_velocity_ps, ps_index, vel_rms, random_seed
    """
    return {
        'use_velocity_ps': (str(PERTURBATION_TYPE).lower() == "power_spectrum"),
        'ps_index': POWER_EXPONENT,
        'vel_rms': a * cs,
        'random_seed': RANDOM_SEED
    }

has_gpu = has_cuda_gpu()
has_mps = has_mps_backend()
device = get_compute_device()


def compute_fd_data_cache(initial_params, time_points, N=200, nu=None,
                          use_velocity_ps=None, ps_index=None, vel_rms=None, random_seed=None):
    """
    Compute and cache FD solver data for all time points to avoid redundant solver calls.
    
    Args:
        initial_params: (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_points: List of time points to compute
        N: Grid resolution for LAX solver
        nu: Courant number for LAX solver
        use_velocity_ps: Whether to use velocity power spectrum (defaults to config)
        ps_index: Power spectrum index (defaults to POWER_EXPONENT)
        vel_rms: Velocity RMS amplitude (defaults to a*cs)
        random_seed: Random seed (defaults to RANDOM_SEED)
    
    Returns:
        Dictionary mapping time -> FD data: {time: {'x': x_fd, 'y': y_fd, 'z': z_fd (if 3D),
                                                    'rho': rho_fd, 'vx': vx_fd, 'vy': vy_fd, 'phi': phi_fd}}
    """
    xmin, xmax, ymin, ymax, rho_1, _alpha, lam, _output_folder, _tmax = initial_params
    
    # Use config defaults if not specified
    if nu is None:
        nu = _get_fd_nu()
    if use_velocity_ps is None:
        use_velocity_ps = (str(PERTURBATION_TYPE).lower() == "power_spectrum")
    if ps_index is None:
        ps_index = POWER_EXPONENT
    if vel_rms is None:
        vel_rms = a * cs
    if random_seed is None:
        random_seed = RANDOM_SEED
    
    num_of_waves = (xmax - xmin) / lam
    
    # Decide grid resolution policy
    if str(PERTURBATION_TYPE).lower() == "power_spectrum":
        N_use = N_GRID
    else:
        N_use = FD_N_2D if N is None else N
    
    # Convert time_points to numpy array and sort
    time_points = np.array(time_points)
    time_points = np.sort(np.unique(time_points))
    max_time = float(np.max(time_points))
    
    print(f"Computing FD data cache for {len(time_points)} time points (optimized: single run to t={max_time:.2f})...")
    fd_cache = {}
    
    # OPTIMIZED: Run solver once to max_time with snapshots at all time points
    if DIMENSION == 3:
        # Use unified solver with snapshot support
        results_dict = call_unified_3d_solver(
            time=max_time, lam=lam, num_of_waves=num_of_waves, rho_1=rho_1, nu=nu,
            use_velocity_ps=use_velocity_ps, ps_index=ps_index,
            vel_rms=vel_rms, random_seed=random_seed,
            save_times=time_points.tolist()
        )
        
        # Select a fixed z-slice for all time points (Option 1: max density at final time)
        fixed_z_idx = None
        fixed_z_val = None
        if str(PERTURBATION_TYPE).lower() == 'power_spectrum':
            if max_time in results_dict:
                result_final = results_dict[max_time]
                z_fd_final = result_final.coordinates['z']
                rho_final = result_final.density
                max_idx = np.unravel_index(np.argmax(rho_final), rho_final.shape)
                fixed_z_idx = int(max_idx[2])
                fixed_z_val = z_fd_final[fixed_z_idx]
                print(f"  📌 Fixed z-slice (max density at t={max_time:.2f}): z={fixed_z_val:.3f}")
            else:
                print("  Warning: final-time snapshot missing; falling back to per-time z selection.")
        
        # Process each snapshot
        for t in time_points:
            if t not in results_dict:
                print(f"  Warning: snapshot at t={t:.2f} not found, skipping...")
                continue
            
            print(f"  Processing snapshot at t = {t:.2f}")
            result = results_dict[t]
            
            # Extract results from unified solver
            x_fd = result.coordinates['x']
            y_fd = result.coordinates['y']
            z_fd = result.coordinates['z']
            rho_vol = result.density
            vx_vol, vy_vol, vz_vol = result.velocity_components
            phi_vol = result.potential
            
            # Z-slice selection based on perturbation type
            if str(PERTURBATION_TYPE).lower() == 'power_spectrum':
                if fixed_z_idx is not None:
                    # Fixed z-slice based on max density at final time
                    z_idx = fixed_z_idx
                    z_val = fixed_z_val
                else:
                    # Fallback: choose slice with most structure
                    z_idx, z_val, contrast = find_max_contrast_slice(rho_vol, z_fd)
                    print(f"  📊 Auto-selected z={z_val:.3f} (contrast={contrast:.2e})")
                    # Optional: Save diagnostic plot first time
                    if t == time_points[0]:
                        analyze_z_variation(rho_vol, z_fd, t, save_dir=SNAPSHOT_DIR)
            else:  # sinusoidal
                # FIXED Z-SELECTION: Use config value for sinusoidal
                z_idx = np.argmin(np.abs(z_fd - SLICE_Z))
                z_val = z_fd[z_idx]
                print(f"  Using fixed z={z_val:.3f} (sinusoidal)")
            
            # Extract 2D slices
            rho_fd = rho_vol[:, :, z_idx]
            vx_fd = vx_vol[:, :, z_idx]
            vy_fd = vy_vol[:, :, z_idx]
            if GRAVITY:
                phi_fd = phi_vol[:, :, z_idx] if phi_vol is not None else np.zeros_like(rho_fd)
            else:
                phi_fd = None
            
            # Store selected z for later use
            fd_cache[t] = {
                'x': x_fd,
                'y': y_fd,
                'z': z_fd,
                'rho': rho_fd,
                'vx': vx_fd,
                'vy': vy_fd,
                'phi': phi_fd,
                'z_idx': z_idx,     # Store index
                'z_val': z_val,     # Store value
                'rho_vol': rho_vol,  # Keep volume for reference (optional)
                'vx_vol': vx_vol,
                'vy_vol': vy_vol,
                'phi_vol': phi_vol,
            }
    else:
        # 2D case - use optimized single-run approach
        # Set up domain and physics parameters for unified solver
        Lx = Ly = lam * num_of_waves
        domain_params = {'Lx': Lx, 'Ly': Ly, 'nx': N_use, 'ny': N_use}
        physics_params = {
            'c_s': cs,
            'rho_o': rho_o,
            'const': const,
            'G': G,
            'rho_1': rho_1,
            'lam': lam
        }
        options = {'gravity': GRAVITY, 'nu': nu, 'comparison': False, 'isplot': False}
        
        # Set up IC parameters
        if use_velocity_ps:
            ic_type = 'power_spectrum'
            ic_params = {
                'power_index': ps_index,
                'amplitude': vel_rms,
                'random_seed': random_seed,
                'vx0_shared': _shared_vx_np,
                'vy0_shared': _shared_vy_np
            }
        else:
            ic_type = 'sinusoidal'
            ic_params = {'KX': KX, 'KY': KY}
        
        # Run solver once with snapshots
        if torch.cuda.is_available() or (use_velocity_ps and _shared_vx_np is None):
            # Use torch solver (supports save_times)
            _clear_cuda_cache()
            results_dict = _timed_call(
                "LAX 2D (optimized torch)",
                lax_solver_torch,
                time=max_time, domain_params=domain_params,
                physics_params=physics_params,
                ic_type=ic_type, ic_params=ic_params,
                options=options, save_times=time_points.tolist()
            )
        else:
            # Fallback: use old approach for shared-field CPU case
            # Fallback: use unified solver without save_times (run multiple times)
            results_dict = {}
            for t in time_points:
                print(f"  Computing FD data at t = {t:.2f}")
                result = _timed_call(
                    "LAX 2D (unified cpu)",
                    lax_solver,
                    t, domain_params, physics_params,
                    ic_type=ic_type, ic_params=ic_params, options=options
                )
                # Result is already a SimulationResult from unified solver
                results_dict[t] = result
        
        # Process each snapshot for 2D
        for t in time_points:
            if t not in results_dict:
                print(f"  Warning: snapshot at t={t:.2f} not found, skipping...")
                continue
            
            print(f"  Processing snapshot at t = {t:.2f}")
            result = results_dict[t]
            
            x_fd = result.coordinates['x']
            y_fd = result.coordinates['y']
            rho_fd = result.density
            vx_fd, vy_fd = result.velocity_components
            if GRAVITY:
                phi_fd = result.potential if hasattr(result, 'potential') and result.potential is not None else np.zeros_like(rho_fd)
            else:
                phi_fd = None
            
            fd_cache[t] = {
                'x': x_fd,
                'y': y_fd,
                'rho': rho_fd,
                'vx': vx_fd,
                'vy': vy_fd,
                'phi': phi_fd
            }
    
    print(f"FD data cache computed for {len(fd_cache)} time points.")
    return fd_cache
