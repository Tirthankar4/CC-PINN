"""
Initial Conditions Module

This module contains all initial condition functions and power spectrum generation
for PINNs training. Extracted from solver.py for better code organization.

Functions:
- initialize_shared_velocity_fields: Setup shared velocity fields for PINN/FD consistency
- generate_power_spectrum_field: Generate vx component using power spectrum
- generate_power_spectrum_field_vy: Generate vy component using power spectrum
- fun_rho_0: Initial density condition
- fun_vx_0: Initial x-velocity condition
- fun_vy_0: Initial y-velocity condition
- func: Placeholder function for phi initial condition
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Callable
from config import (cs, rho_o, N_GRID, POWER_EXPONENT, 
                    PERTURBATION_TYPE, KX, KY, KZ, RANDOM_SEED,
                    DIMENSION, N_GRID_3D)
from numerical_solvers.LAX import generate_shared_velocity_field


@dataclass
class VelocityFieldInterpolators:
    """Container for velocity field interpolation functions."""
    vx_interp: Callable
    vy_interp: Callable
    vz_interp: Optional[Callable] = None  # None for 2D cases

def _ensure_column_tensor(tensor):
    return tensor if tensor.dim() > 1 else tensor.unsqueeze(-1)

def _extract_spatial_coords(coords):
    return coords[:-1] if len(coords) > 1 else coords


def _detect_dimension(x):
    """
    Detect spatial dimension from coordinate tensor.
    
    Args:
        x: Collocation coordinates [x, y, t] or [x, y, z, t] or [x, t]
    
    Returns:
        int: Spatial dimension (2 or 3)
    """
    # Extract spatial coordinates (exclude time dimension)
    spatial_coords = _extract_spatial_coords(x)
    
    # Check if z coordinate exists and has non-zero variation
    if len(spatial_coords) >= 3:
        z_coord = spatial_coords[2]
        # Check if z has meaningful variation (not all zeros or constant)
        if z_coord.numel() > 0:
            z_range = torch.max(z_coord) - torch.min(z_coord)
            if z_range > 1e-6:  # Non-trivial z variation indicates 3D
                return 3
    
    # Default to 2D if z is missing or constant
    return 2


def initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=None, dimension=None):
    """
    Initialize shared velocity fields for consistent PINN/FD initial conditions.
    This should be called once at the beginning of training.
    
    IMPORTANT: The parameters used here (POWER_EXPONENT, v_1=a*cs, seed) MUST match
    the parameters used in FD plotting functions to ensure identical initial conditions.
    All FD visualization functions should use the same defaults.
    
    Args:
        lam: Wavelength
        num_of_waves: Number of waves in domain
        v_1: Velocity amplitude
        seed: Random seed for reproducibility
        dimension: Spatial dimension (2 or 3). If None, uses DIMENSION from config.
    
    Returns:
        For 2D: Tuple (vx_np, vy_np, interpolators: VelocityFieldInterpolators)
        For 3D: Tuple (vx_np, vy_np, vz_np, interpolators: VelocityFieldInterpolators)
    """
    if seed is None:
        seed = RANDOM_SEED
    
    # Determine dimension
    if dimension is None:
        dimension = DIMENSION
    
    # Calculate domain size to match FD solver
    Lx = lam * num_of_waves
    Ly = lam * num_of_waves
    
    if dimension == 2:
        # 2D case
        Lz = None
        nz = None
        grid_size = N_GRID
        
        # Generate shared velocity fields for 2D
        vx_np, vy_np, vx_interp, vy_interp = generate_shared_velocity_field(
            grid_size, grid_size, Lx, Ly, 
            power_index=POWER_EXPONENT, 
            amplitude=v_1, 
            DIMENSION=2,
            random_seed=seed
        )
        
        # Return arrays and interpolators (no global state)
        interpolators = VelocityFieldInterpolators(vx_interp, vy_interp)
        return vx_np, vy_np, interpolators
    
    elif dimension == 3:
        # 3D case - use N_GRID_3D for all dimensions for consistency
        Lz = lam * num_of_waves
        nx = ny = nz = N_GRID_3D  # Use N_GRID_3D for all dimensions in 3D
        
        # Generate shared velocity fields for 3D
        vx_np, vy_np, vz_np, vx_interp, vy_interp, vz_interp = generate_shared_velocity_field(
            nx, ny, Lx, Ly,
            power_index=POWER_EXPONENT,
            amplitude=v_1,
            DIMENSION=3,
            random_seed=seed,
            nz=nz,
            Lz=Lz
        )
        
        # Return arrays and interpolators (no global state)
        interpolators = VelocityFieldInterpolators(vx_interp, vy_interp, vz_interp)
        return vx_np, vy_np, vz_np, interpolators
    
    else:
        raise ValueError(f"Unsupported dimension={dimension}. Use 2 or 3.")


def _interpolate_shared_field(x, field_interp):
    """
    Helper function to interpolate shared velocity field to collocation points.
    Supports both 2D and 3D coordinates.
    
    Args:
        x: Collocation coordinates [x, y, ...] for 2D or [x, y, z, ...] for 3D
        field_interp: Interpolation function from shared fields
    
    Returns:
        Interpolated field values as torch tensor
    """
    # Detect dimension from coordinates
    dim = _detect_dimension(x)
    
    # Convert tensor coordinates to numpy for interpolation
    x_np = x[0].detach().cpu().numpy()
    y_np = x[1].detach().cpu().numpy()
    
    if dim == 2:
        # 2D case: create coordinate pairs for interpolation
        coords = np.stack([x_np.flatten(), y_np.flatten()], axis=1)
    else:
        # 3D case: create coordinate triplets for interpolation
        z_np = x[2].detach().cpu().numpy()
        coords = np.stack([x_np.flatten(), y_np.flatten(), z_np.flatten()], axis=1)
    
    # Interpolate shared velocity field
    field_interp_values = field_interp(coords)
    
    # Convert back to tensor and reshape
    field_tensor = torch.from_numpy(field_interp_values).float().to(x[0].device)
    
    # Ensure correct shape [N, 1]
    if field_tensor.dim() == 1:
        return field_tensor.unsqueeze(-1)
    else:
        return field_tensor

def generate_power_spectrum_field(lam, v_1, x, seed=None, interpolators=None):
    """
    Generate vx component using interpolators.
    
    Args:
        lam: Wavelength
        v_1: Velocity amplitude
        x: Collocation coordinates [x, y, ...]
        seed: Random seed for reproducibility
        interpolators: VelocityFieldInterpolators instance (required for power spectrum ICs)
    
    Returns:
        vx component of velocity field
        
    Raises:
        ValueError: If interpolators are not provided for power spectrum perturbations
    """
    if seed is None:
        seed = RANDOM_SEED
    
    # Use interpolators if provided
    if interpolators is not None and interpolators.vx_interp is not None:
        return _interpolate_shared_field(x, interpolators.vx_interp)
    
    # Error if interpolators not provided - ensures PINN/LAX consistency
    raise ValueError(
        "VelocityFieldInterpolators required for power spectrum initial conditions. "
        "Call initialize_shared_velocity_fields() first and pass the returned interpolators."
    )


def generate_power_spectrum_field_vy(lam, v_1, x, seed=None, interpolators=None):
    """
    Generate vy component using interpolators.
    
    Args:
        lam: Wavelength
        v_1: Velocity amplitude
        x: Collocation coordinates [x, y, ...] for 2D or [x, y, z, ...] for 3D
        seed: Random seed for reproducibility
        interpolators: VelocityFieldInterpolators instance (required for power spectrum ICs)
    
    Returns:
        vy component of velocity field
        
    Raises:
        ValueError: If interpolators are not provided for power spectrum perturbations
    """
    if seed is None:
        seed = RANDOM_SEED
    
    # Use interpolators if provided
    if interpolators is not None and interpolators.vy_interp is not None:
        return _interpolate_shared_field(x, interpolators.vy_interp)
    
    # Error if interpolators not provided - ensures PINN/LAX consistency
    raise ValueError(
        "VelocityFieldInterpolators required for power spectrum initial conditions. "
        "Call initialize_shared_velocity_fields() first and pass the returned interpolators."
    )


def generate_power_spectrum_field_vz(lam, v_1, x, seed=None, interpolators=None):
    """
    Generate vz component using interpolators.
    
    Args:
        lam: Wavelength
        v_1: Velocity amplitude
        x: Collocation coordinates [x, y, z, ...] for 3D
        seed: Random seed for reproducibility
        interpolators: VelocityFieldInterpolators instance (required for power spectrum ICs)
    
    Returns:
        vz component of velocity field
        
    Raises:
        ValueError: If interpolators are not provided for power spectrum perturbations
    """
    if seed is None:
        seed = RANDOM_SEED
    
    # Use interpolators if provided
    if interpolators is not None and interpolators.vz_interp is not None:
        return _interpolate_shared_field(x, interpolators.vz_interp)
    
    # Error if interpolators not provided - ensures PINN/LAX consistency
    raise ValueError(
        "VelocityFieldInterpolators required for power spectrum initial conditions. "
        "Call initialize_shared_velocity_fields() first and pass the returned interpolators."
    )


def _compute_wave_phase(spatial_coords, lam):
    """
    Compute wave phase and wave-vector components for sinusoidal perturbations.
    """
    if not spatial_coords:
        raise ValueError("Spatial coordinates are required to compute wave phase.")
    
    x_coord = _ensure_column_tensor(spatial_coords[0])
    zeros = torch.zeros_like(x_coord)
    y_coord = _ensure_column_tensor(spatial_coords[1]) if len(spatial_coords) >= 2 else zeros
    z_coord = _ensure_column_tensor(spatial_coords[2]) if len(spatial_coords) >= 3 else zeros
    
    device = x_coord.device
    dtype = x_coord.dtype
    kx = torch.as_tensor(float(KX), device=device, dtype=dtype)
    ky = torch.as_tensor(float(KY), device=device, dtype=dtype)
    kz = torch.as_tensor(float(KZ), device=device, dtype=dtype)
    
    phase = kx * x_coord + ky * y_coord + kz * z_coord
    
    # Fallback to fundamental wavelength if wave-vector is zero (e.g., user-specified)
    if torch.allclose(kx.abs() + ky.abs() + kz.abs(), torch.tensor(0.0, device=device, dtype=dtype)):
        fundamental = torch.as_tensor(2 * np.pi / lam, device=device, dtype=dtype)
        phase = fundamental * x_coord
        kx, ky, kz = fundamental, torch.zeros_like(fundamental), torch.zeros_like(fundamental)
    
    return phase, kx, ky, kz, x_coord, y_coord, z_coord


def _coupled_velocity_components(coords, lam, jeans, v_1):
    """
    Generate coupled velocity components from the same wave pattern (supports 1D/2D/3D).
    """
    spatial_coords = _extract_spatial_coords(coords)
    phase, kx, ky, kz, _, _, _ = _compute_wave_phase(spatial_coords, lam)
    dtype = phase.dtype
    device = phase.device
    v_scale = torch.as_tensor(float(v_1), device=device, dtype=dtype)
    
    if lam > jeans:
        wave_field = -v_scale * torch.sin(phase)
    else:
        wave_field = v_scale * torch.cos(phase)
    
    k_mag = torch.sqrt(kx**2 + ky**2 + kz**2)
    if k_mag <= torch.tensor(1e-12, device=device, dtype=dtype):
        vx = wave_field
        vy = torch.zeros_like(wave_field)
        vz = torch.zeros_like(wave_field)
    else:
        inv_mag = 1.0 / k_mag
        vx = wave_field * (kx * inv_mag)
        vy = wave_field * (ky * inv_mag)
        vz = wave_field * (kz * inv_mag)
    
    return vx, vy, vz


def fun_rho_0(rho_1, lam, x):
    """
    Define initial condition for density.
    
    Args:
        rho_1: Perturbation amplitude
        lam: Wavelength
        x: Spatial coordinates [x, y, t] or [x, t]
    
    Returns:
        rho_0: Initial density field
    """
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        spatial_coords = _extract_spatial_coords(x)
        phase, *_ = _compute_wave_phase(spatial_coords, lam)
        rho_0 = rho_o + rho_1 * torch.cos(phase)
    else:
        # Power spectrum: uniform initial density
        rho_0 = torch.full_like(x[0], rho_o)
        # Ensure correct shape [N, 1]
        if rho_0.dim() == 1:
            rho_0 = rho_0.unsqueeze(-1)
    
    return rho_0


def fun_vx_0(lam, jeans, v_1, x, interpolators=None):
    """
    Initial condition for x-velocity.
    
    Args:
        lam: Wavelength
        jeans: Jeans length
        v_1: Velocity amplitude
        x: Spatial coordinates
        interpolators: VelocityFieldInterpolators (required for power spectrum perturbations)
    
    Returns:
        vx_0: Initial x-velocity field
    """
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        vx, _, _ = _coupled_velocity_components(x, lam, jeans, v_1)
        return vx
    else:
        # Power spectrum case - pass interpolators through
        return generate_power_spectrum_field(lam, v_1, x, seed=RANDOM_SEED, interpolators=interpolators)


def fun_vy_0(lam, jeans, v_1, x, interpolators=None):
    """
    Initial condition for y-velocity.
    
    Args:
        lam: Wavelength
        jeans: Jeans length
        v_1: Velocity amplitude
        x: Spatial coordinates
        interpolators: VelocityFieldInterpolators (required for power spectrum perturbations)
    
    Returns:
        vy_0: Initial y-velocity field
    """
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        _, vy, _ = _coupled_velocity_components(x, lam, jeans, v_1)
        return vy
    else:
        # Power spectrum case - pass interpolators through
        return generate_power_spectrum_field_vy(lam, v_1, x, seed=RANDOM_SEED, interpolators=interpolators)


def fun_vz_0(lam, jeans, v_1, x, interpolators=None):
    """
    Initial condition for z-velocity (used in 3D runs).
    
    Args:
        lam: Wavelength
        jeans: Jeans length
        v_1: Velocity amplitude
        x: Spatial coordinates
        interpolators: VelocityFieldInterpolators (required for power spectrum perturbations)
    
    Returns:
        vz_0: Initial z-velocity field
    """
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        _, _, vz = _coupled_velocity_components(x, lam, jeans, v_1)
        return vz
    else:
        # Power spectrum case - pass interpolators through
        return generate_power_spectrum_field_vz(lam, v_1, x, seed=RANDOM_SEED, interpolators=interpolators)


def func(x):
    """
    Placeholder function for phi initial condition (zero potential).
    
    Args:
        x: Spatial coordinates
    
    Returns:
        Zero tensor matching the shape of x[0]
    """
    return x[0] * 0