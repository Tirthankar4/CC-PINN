"""
field_generation.py
-------------------
Pure utility functions for synthesising turbulent velocity fields from an
isotropic power-law spectrum in k-space.

These functions have NO dependency on any numerical solver (LAX, athena++, etc.)
and are intentionally kept in the `core/` package so that:
  - `core/initial_conditions.py` can import them directly, and
  - `numerical_solvers/LAX.py` can also import them without creating a
    `core/ -> numerical_solvers/` reverse dependency.

Public API
----------
build_k_space_grid(shape, domain_lengths, dimension)
    Build the FFT wave-vector grid for a periodic domain.

generate_velocity_field_power_spectrum(nx, ny, Lx, Ly, ...)
    Generate 2-D or 3-D velocity components with P(k) ~ k^{power_index}.

generate_shared_velocity_field(nx, ny, Lx, Ly, ...)
    Generate the field once and return both numpy arrays (for FD solvers)
    and scipy interpolators (for PINN collocation points).
"""

import numpy as np
from numpy.fft import fftn, ifftn
from scipy.interpolate import RegularGridInterpolator

from config import RANDOM_SEED


# ============================================================================
# k-space grid builder
# ============================================================================

def build_k_space_grid(shape, domain_lengths, dimension):
    """
    Build k-space grid for FFT operations.

    Args:
        shape: Tuple of grid dimensions (nx, ny) or (nx, ny, nz)
        domain_lengths: Tuple of domain sizes (Lx, Ly) or (Lx, Ly, Lz)
        dimension: 2 or 3

    Returns:
        tuple:
            For 2D: (kx, ky, kx_mesh, ky_mesh)
            For 3D: (kx, ky, kz, kx_mesh, ky_mesh, kz_mesh)
    """
    if dimension == 2:
        nx, ny = shape
        Lx, Ly = domain_lengths
        dx = Lx / nx
        dy = Ly / ny
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
        kx_mesh, ky_mesh = np.meshgrid(kx, ky, indexing='ij')
        return kx, ky, kx_mesh, ky_mesh
    elif dimension == 3:
        nx, ny, nz = shape
        Lx, Ly, Lz = domain_lengths
        dx = Lx / nx
        dy = Ly / ny
        dz = Lz / nz
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
        kz = 2 * np.pi * np.fft.fftfreq(nz, d=dz)
        kx_mesh, ky_mesh, kz_mesh = np.meshgrid(kx, ky, kz, indexing='ij')
        return kx, ky, kz, kx_mesh, ky_mesh, kz_mesh
    else:
        raise ValueError(f"Unsupported dimension={dimension}. Use 2 or 3.")


# ============================================================================
# Velocity field synthesiser
# ============================================================================

def generate_velocity_field_power_spectrum(
        nx, ny, Lx, Ly,
        power_index=-3.0, amplitude=0.02,
        DIMENSION=2, random_seed=None,
        nz=None, Lz=None):
    """
    Generate 2-D or 3-D velocity components with an isotropic power-law
    spectrum P(k) ~ k^{power_index}.

    The field is generated directly at the target resolution (nx, ny[, nz])
    without any downsampling or cutoff strategies.

    Args:
        nx, ny: Grid dimensions (required).
        Lx, Ly: Physical domain sizes (required).
        power_index: Spectral index of the power law (default: -3.0).
        amplitude: Target per-component RMS amplitude = std(v_i)
                   (this is also the per-component RMS Mach number when
                    expressed in units of the sound speed). Default: 0.02.
        DIMENSION: 2 or 3 (default: 2).
        random_seed: Integer seed for reproducibility (default: None → random).
        nz, Lz: Grid points and domain size along z (required when DIMENSION=3).

    Returns:
        For DIMENSION=2: (vx, vy)   — each shape (nx, ny)
        For DIMENSION=3: (vx, vy, vz) — each shape (nx, ny, nz)
    """
    rng = np.random.default_rng(random_seed)

    # Determine shape and domain lengths based on dimension
    if DIMENSION == 2:
        shape = (nx, ny)
        domain_lengths = (Lx, Ly)
    elif DIMENSION == 3:
        if nz is None or Lz is None:
            raise ValueError("nz and Lz must be provided for 3D")
        shape = (nx, ny, nz)
        domain_lengths = (Lx, Ly, Lz)
    else:
        raise ValueError(f"Unsupported DIMENSION={DIMENSION}. Use 2 or 3.")

    def synthesize_component():
        # 1. Random white-noise field in real space
        field = rng.standard_normal(shape)
        F = fftn(field)

        # 2. k-space grid
        k_grid = build_k_space_grid(shape, domain_lengths, DIMENSION)

        # 3. |k| magnitude
        if DIMENSION == 2:
            kx, ky, kxg, kyg = k_grid
            kk = np.sqrt(kxg**2 + kyg**2)
        else:
            kx, ky, kz, kxg, kyg, kzg = k_grid
            kk = np.sqrt(kxg**2 + kyg**2 + kzg**2)

        # 4. Power-law filter: F(k) *= |k|^{power_index/2}
        filt = np.zeros_like(kk)
        mask = kk > 0
        filt[mask] = kk[mask] ** (power_index / 2.0)
        F_filtered = F * filt

        # 5. Back to real space
        comp = np.real(ifftn(F_filtered))

        # 6. Normalise so that std(comp) == amplitude
        comp -= np.mean(comp)
        std = np.std(comp)
        if std > 0:
            comp = comp * (amplitude / std)

        return comp

    vx0 = synthesize_component()
    vy0 = synthesize_component()

    if DIMENSION == 3:
        vz0 = synthesize_component()
        return vx0, vy0, vz0
    else:
        return vx0, vy0


# ============================================================================
# Shared-field generator (arrays + interpolators)
# ============================================================================

def generate_shared_velocity_field(
        nx, ny, Lx, Ly,
        power_index=-4.0, amplitude=0.01,
        DIMENSION=2, random_seed=None,
        nz=None, Lz=None):
    """
    Generate a shared velocity field and return both numpy arrays (for FD
    solvers) and scipy interpolators (for PINN collocation points).

    Generating the field once and sharing it ensures that the PINN and the
    reference FD solver start from identical initial conditions.

    Args:
        nx, ny: Grid dimensions (required).
        Lx, Ly: Physical domain sizes (required).
        power_index: Spectral index (default: -4.0).
        amplitude: Per-component RMS amplitude (default: 0.01).
        DIMENSION: 2 or 3 (default: 2).
        random_seed: Integer seed (default: falls back to config.RANDOM_SEED).
        nz, Lz: Required when DIMENSION=3.

    Returns:
        For DIMENSION=2: (vx_np, vy_np, vx_interp, vy_interp)
        For DIMENSION=3: (vx_np, vy_np, vz_np, vx_interp, vy_interp, vz_interp)
    """
    if random_seed is None:
        random_seed = RANDOM_SEED

    # Generate arrays
    if DIMENSION == 2:
        vx_np, vy_np = generate_velocity_field_power_spectrum(
            nx, ny, Lx, Ly,
            power_index=power_index, amplitude=amplitude,
            DIMENSION=2, random_seed=random_seed)
        velocity_arrays = [vx_np, vy_np]
    elif DIMENSION == 3:
        if nz is None or Lz is None:
            raise ValueError("nz and Lz must be provided for 3D")
        vx_np, vy_np, vz_np = generate_velocity_field_power_spectrum(
            nx, ny, Lx, Ly,
            power_index=power_index, amplitude=amplitude,
            DIMENSION=3, random_seed=random_seed, nz=nz, Lz=Lz)
        velocity_arrays = [vx_np, vy_np, vz_np]
    else:
        raise ValueError(f"Unsupported DIMENSION={DIMENSION}. Use 2 or 3.")

    # Build coordinate grids (periodic: exclude the right boundary)
    if DIMENSION == 2:
        coords = (
            np.linspace(0, Lx, nx, endpoint=False),
            np.linspace(0, Ly, ny, endpoint=False),
        )
    else:
        coords = (
            np.linspace(0, Lx, nx, endpoint=False),
            np.linspace(0, Ly, ny, endpoint=False),
            np.linspace(0, Lz, nz, endpoint=False),
        )

    # Create one interpolator per velocity component
    interpolators = [
        RegularGridInterpolator(
            coords, vel_array,
            method='linear', bounds_error=False, fill_value=0.0)
        for vel_array in velocity_arrays
    ]

    if DIMENSION == 2:
        return vx_np, vy_np, interpolators[0], interpolators[1]
    else:
        return vx_np, vy_np, vz_np, interpolators[0], interpolators[1], interpolators[2]
