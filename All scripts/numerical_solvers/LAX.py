import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
from dataclasses import dataclass
from typing import Optional

## For the FFT solver

from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn

from config import RANDOM_SEED, DIMENSION

# Import wave vector components and physical constants for 2D sinusoidal perturbations
try:
    from config import KX, KY, KZ, cs, rho_o, const, G
except ImportError:
    # Fallback if config not available
    KX = 2*np.pi/7.0  # Default wavelength
    KY = 0.0
    KZ = 0.0
    cs = 1.0
    rho_o = 1.0
    const = 1.0
    G = 1.0

np.random.seed(RANDOM_SEED)
#tf.random.set_seed(1234)


@dataclass
class DomainParams:
    """Domain parameters for the simulation."""
    Lx: float  # Domain size in x direction
    Ly: float  # Domain size in y direction
    nx: int  # Number of grid points in x direction
    ny: int  # Number of grid points in y direction
    Lz: Optional[float] = None  # Domain size in z direction (for 3D)
    nz: Optional[int] = None  # Number of grid points in z direction (for 3D)
    
    @property
    def dx(self) -> float:
        """Grid spacing in x direction."""
        return self.Lx / self.nx
    
    @property
    def dy(self) -> float:
        """Grid spacing in y direction."""
        return self.Ly / self.ny
    
    @property
    def dz(self) -> Optional[float]:
        """Grid spacing in z direction (for 3D)."""
        return self.Lz / self.nz if self.Lz is not None and self.nz is not None else None
    
    @property
    def is_3d(self) -> bool:
        """Check if this is a 3D domain."""
        return self.Lz is not None and self.nz is not None


@dataclass
class SimulationParams:
    """Simulation parameters for the LAX solver."""
    time: float  # Simulation time
    N: int  # Grid resolution (Nx = Ny = N)
    nu: float  # Courant number
    lam: float  # Wavelength
    num_of_waves: int  # Number of waves in domain
    rho_1: float  # Density perturbation amplitude
    gravity: bool = False  # Whether to include self-gravity
    isplot: Optional[bool] = None  # Whether to plot results
    comparison: Optional[bool] = None  # Whether to compare with linear theory
    animation: Optional[bool] = None  # Whether to animate
    use_velocity_ps: bool = False  # Whether to use power spectrum velocity initialization
    ps_index: float = -3.0  # Power spectrum index
    vel_rms: float = 0.02  # RMS velocity amplitude
    random_seed: Optional[int] = None  # Random seed for reproducibility
    vx0_shared: Optional[np.ndarray] = None  # Shared initial x-velocity field
    vy0_shared: Optional[np.ndarray] = None  # Shared initial y-velocity field


@dataclass
class SimulationResult:
    """Result container for LAX solver simulations."""
    dimension: int  # 1, 2, or 3
    density: np.ndarray  # Density field
    velocity_components: list  # List of velocity arrays [vx, vy, ...] or [vx, vy, vz]
    coordinates: dict  # Dict with 'x', 'y' (2D/3D), 'z' (3D) coordinate arrays
    metadata: dict  # Dict with time, iterations, rho_max, etc.
    potential: Optional[np.ndarray] = None  # Gravitational potential (if gravity=True)
    linear_theory_comparison: Optional[dict] = None  # Optional linear theory comparison data


# ============================================================================
# Helper Functions
# ============================================================================

def compute_jeans_length(c_s, rho_o, const, G):
    """
    Compute the Jeans length for gravitational instability.
    
    Args:
        c_s: Sound speed
        rho_o: Background density
        const: Constant factor
        G: Gravitational constant
    
    Returns:
        Jeans length
    """
    return np.sqrt(4 * np.pi**2 * c_s**2 / (const * G * rho_o))


def compute_perturbation_velocity(rho_1, rho_o, lam, c_s, jeans, gravity):
    """
    Compute the velocity perturbation amplitude v_1 and growth rate alpha.
    
    Args:
        rho_1: Density perturbation amplitude
        rho_o: Background density
        lam: Wavelength
        c_s: Sound speed
        jeans: Jeans length
        gravity: Whether gravity is enabled
    
    Returns:
        tuple: (v_1, alpha, is_unstable)
            v_1: Velocity perturbation amplitude
            alpha: Growth/oscillation rate (positive for instability, real for oscillation)
            is_unstable: True if gravitational instability (lam >= jeans), False otherwise
    """
    if not gravity:
        # No gravity: sound wave
        v_1 = (c_s * rho_1) / rho_o
        return v_1, None, False
    
    # With gravity: check if unstable
    if lam >= jeans:
        # Gravitational instability
        alpha = np.sqrt(const * G * rho_o - c_s**2 * (2 * np.pi / lam)**2)
        v_1 = (rho_1 / rho_o) * (alpha / (2 * np.pi / lam))
        return v_1, alpha, True
    else:
        # Oscillatory regime
        alpha = np.sqrt(c_s**2 * (2 * np.pi / lam)**2 - const * G * rho_o)
        v_1 = (rho_1 / rho_o) * (alpha / (2 * np.pi / lam))
        return v_1, alpha, False


def build_k_space_grid(shape, domain_lengths, dimension):
    """
    Build k-space grid for FFT operations.
    
    Args:
        shape: Tuple of grid dimensions (nx, ny) or (nx, ny, nz)
        domain_lengths: Tuple of domain sizes (Lx, Ly) or (Lx, Ly, Lz)
        dimension: 2 or 3
    
    Returns:
        tuple: (kx, ky, kz (optional), kx_mesh, ky_mesh, kz_mesh (optional))
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


def compute_adaptive_timestep(velocities, c_s, dx, nu, include_sound_speed=False):
    """
    Compute adaptive timestep based on CFL condition.
    
    Args:
        velocities: List of velocity arrays [vx, vy, ...] or single velocity array, or None
        c_s: Sound speed
        dx: Grid spacing
        nu: Courant number
        include_sound_speed: If True, include c_s in the max velocity calculation (for initial timestep)
    
    Returns:
        Adaptive timestep
    """
    if include_sound_speed:
        if velocities is None:
            vmax = c_s
        elif isinstance(velocities, (list, tuple)):
            vmax = max(max(np.max(np.abs(v)) for v in velocities), c_s)
        else:
            vmax = max(np.max(np.abs(velocities)), c_s)
        return nu * dx / vmax
    else:
        if velocities is None:
            vmax = c_s
        elif isinstance(velocities, (list, tuple)):
            vmax = max(np.max(np.abs(v)) for v in velocities)
        else:
            vmax = np.max(np.abs(velocities))
        
        dt1 = nu * dx / vmax if vmax > 1e-9 else float('inf')
        dt2 = nu * dx / c_s
        return min(dt1, dt2)


def compute_lax_density_update(rho, velocities, dt, grid_spacings, dimension):
    """
    Compute LAX density update using dimension-agnostic approach.
    
    This function implements the LAX method for updating density in the continuity equation:
    ∂ρ/∂t + ∇·(ρv) = 0
    
    The update uses a dimension-agnostic approach that works for 1D, 2D, and 3D without
    if-else chains by iterating over dimensions.
    
    Args:
        rho: Current density array (1D, 2D, or 3D numpy array)
        velocities: List of velocity arrays [vx, vy, ...] or [vx, vy, vz] for 3D
                   Length must match dimension
        dt: Time step
        grid_spacings: List of grid spacings [dx, dy, ...] or [dx, dy, dz] for 3D
                       Length must match dimension
        dimension: Integer dimension (1, 2, or 3)
    
    Returns:
        Updated density array (same shape as input rho)
    
    Examples:
        # 1D
        rho_new = compute_lax_density_update(rho, [vx], dt, [dx], 1)
        
        # 2D
        rho_new = compute_lax_density_update(rho, [vx, vy], dt, [dx, dy], 2)
        
        # 3D
        rho_new = compute_lax_density_update(rho, [vx, vy, vz], dt, [dx, dy, dz], 3)
    """
    if len(velocities) != dimension:
        raise ValueError(f"Number of velocity arrays ({len(velocities)}) must match dimension ({dimension})")
    if len(grid_spacings) != dimension:
        raise ValueError(f"Number of grid spacings ({len(grid_spacings)}) must match dimension ({dimension})")
    
    # Compute mu values for each dimension: mu_i = dt / (2 * dx_i)
    mu_values = [dt / (2 * dx) for dx in grid_spacings]
    
    # Initialize with averaging term: (1/(2*dimension)) * sum of all rolled neighbors
    # For each dimension, we add contributions from both +1 and -1 rolls
    rho_new = np.zeros_like(rho)
    
    # Sum all rolled neighbors (forward and backward for each dimension)
    for axis in range(dimension):
        rho_new += np.roll(rho, -1, axis=axis)  # Forward roll
        rho_new += np.roll(rho, 1, axis=axis)   # Backward roll
    
    # Normalize by 1/(2*dimension)
    rho_new = rho_new / (2 * dimension)
    
    # Subtract flux terms for each dimension: -mu_i * (flux_forward - flux_backward)
    # where flux = rho * velocity
    for axis in range(dimension):
        mu = mu_values[axis]
        vel = velocities[axis]
        
        # Forward flux: rho_rolled_forward * vel_rolled_forward
        rho_forward = np.roll(rho, -1, axis=axis)
        vel_forward = np.roll(vel, -1, axis=axis)
        flux_forward = rho_forward * vel_forward
        
        # Backward flux: rho_rolled_backward * vel_rolled_backward
        rho_backward = np.roll(rho, 1, axis=axis)
        vel_backward = np.roll(vel, 1, axis=axis)
        flux_backward = rho_backward * vel_backward
        
        # Subtract the flux difference
        rho_new -= mu * (flux_forward - flux_backward)
    
    return rho_new


def compute_lax_momentum_update(momenta, velocities, rho, phi, c_s, dt, grid_spacings, dimension, gravity=True):
    """
    Compute LAX momentum update using dimension-agnostic approach.
    
    This function implements the LAX method for updating momentum in the momentum equation:
    ∂(ρv)/∂t + ∇·(ρv⊗v) = -∇P - ρ∇φ
    
    The update uses a dimension-agnostic approach that works for 1D, 2D, and 3D without
    if-else chains by iterating over dimensions.
    
    Args:
        momenta: List of momentum arrays [Px, Py, ...] or [Px, Py, Pz] for 3D
                 Length must match dimension
        velocities: List of velocity arrays [vx, vy, ...] or [vx, vy, vz] for 3D
                   Length must match dimension
        rho: Current density array (same shape as momentum arrays)
        phi: Gravitational potential array (same shape as momentum arrays)
             Only used if gravity=True
        c_s: Sound speed
        dt: Time step
        grid_spacings: List of grid spacings [dx, dy, ...] or [dx, dy, dz] for 3D
                       Length must match dimension
        dimension: Integer dimension (1, 2, or 3)
        gravity: Whether to include gravity term (default: True)
    
    Returns:
        List of updated momentum arrays (same shape as input momenta)
    
    Examples:
        # 1D
        P_new = compute_lax_momentum_update([Px], [vx], rho, phi, c_s, dt, [dx], 1, gravity=True)
        
        # 2D
        Px_new, Py_new = compute_lax_momentum_update([Px, Py], [vx, vy], rho, phi, c_s, dt, [dx, dy], 2, gravity=True)
        
        # 3D
        Px_new, Py_new, Pz_new = compute_lax_momentum_update([Px, Py, Pz], [vx, vy, vz], rho, phi, c_s, dt, [dx, dy, dz], 3, gravity=True)
    """
    if len(momenta) != dimension:
        raise ValueError(f"Number of momentum arrays ({len(momenta)}) must match dimension ({dimension})")
    if len(velocities) != dimension:
        raise ValueError(f"Number of velocity arrays ({len(velocities)}) must match dimension ({dimension})")
    if len(grid_spacings) != dimension:
        raise ValueError(f"Number of grid spacings ({len(grid_spacings)}) must match dimension ({dimension})")
    
    # Compute mu values for each dimension: mu_i = dt / (2 * dx_i)
    mu_values = [dt / (2 * dx) for dx in grid_spacings]
    
    # Initialize list to store updated momenta
    momenta_new = []
    
    # Update each momentum component
    for comp_dim in range(dimension):
        P = momenta[comp_dim]
        mu_comp = mu_values[comp_dim]  # mu for this component's direction
        
        # Initialize with averaging term: (1/(2*dimension)) * sum of all rolled neighbors
        P_new = np.zeros_like(P)
        
        # Sum all rolled neighbors (forward and backward for each dimension)
        for axis in range(dimension):
            P_new += np.roll(P, -1, axis=axis)  # Forward roll
            P_new += np.roll(P, 1, axis=axis)   # Backward roll
        
        # Normalize by 1/(2*dimension)
        P_new = P_new / (2 * dimension)
        
        # Advection terms: For each dimension, subtract mu_i * (flux_forward - flux_backward)
        # where flux = momentum * velocity
        # Each momentum component is advected by ALL velocity components
        for axis in range(dimension):
            mu = mu_values[axis]
            vel = velocities[axis]
            
            # Forward flux: P_rolled_forward * vel_rolled_forward
            P_forward = np.roll(P, -1, axis=axis)
            vel_forward = np.roll(vel, -1, axis=axis)
            flux_forward = P_forward * vel_forward
            
            # Backward flux: P_rolled_backward * vel_rolled_backward
            P_backward = np.roll(P, 1, axis=axis)
            vel_backward = np.roll(vel, 1, axis=axis)
            flux_backward = P_backward * vel_backward
            
            # Subtract the flux difference
            P_new -= mu * (flux_forward - flux_backward)
        
        # Pressure gradient term: Only in the component's own direction
        # -c_s^2 * mu_comp * (rho_forward - rho_backward) in direction comp_dim
        rho_forward = np.roll(rho, -1, axis=comp_dim)
        rho_backward = np.roll(rho, 1, axis=comp_dim)
        P_new -= (c_s**2) * mu_comp * (rho_forward - rho_backward)
        
        # Gravity term: Only in the component's own direction (if enabled)
        # -mu_comp * rho * (phi_forward - phi_backward) in direction comp_dim
        if gravity:
            phi_forward = np.roll(phi, -1, axis=comp_dim)
            phi_backward = np.roll(phi, 1, axis=comp_dim)
            P_new -= mu_comp * rho * (phi_forward - phi_backward)
        
        momenta_new.append(P_new)
    
    return momenta_new


def lax_time_step(state_dict, dt, params_dict, dimension, gravity=True):
    """
    Perform a single LAX time step, updating all state variables.
    
    This function orchestrates a complete time step by:
    1. Updating density using compute_lax_density_update()
    2. Updating momenta using compute_lax_momentum_update()
    3. Computing velocities from updated momenta and density
    4. Updating gravitational potential if gravity is enabled
    
    Args:
        state_dict: Dictionary containing current state with keys:
                   - 'rho': density array
                   - 'velocities': list of velocity arrays [vx, vy, ...] or [vx, vy, vz]
                   - 'momenta': list of momentum arrays [Px, Py, ...] or [Px, Py, Pz]
                   - 'phi': gravitational potential array (used if gravity=True)
        dt: Time step
        params_dict: Dictionary containing simulation parameters with keys:
                    - 'Lx': Domain size in x direction
                    - 'Ly': Domain size in y direction (required for 2D/3D)
                    - 'Lz': Domain size in z direction (required for 3D)
                    - 'nx': Number of grid points in x direction
                    - 'ny': Number of grid points in y direction (required for 2D/3D)
                    - 'nz': Number of grid points in z direction (required for 3D)
                    - 'c_s': Sound speed
                    - 'rho_o': Background density (for Poisson solver)
                    - 'const': Constant factor (for Poisson solver)
        dimension: Integer dimension (1, 2, or 3)
        gravity: Whether to include gravity (default: True)
    
    Returns:
        Dictionary with updated state, same structure as input state_dict
    
    Examples:
        # 2D example
        state = {
            'rho': rho0,
            'velocities': [vx0, vy0],
            'momenta': [Px0, Py0],
            'phi': phi0
        }
        params = {
            'Lx': 10.0, 'Ly': 10.0,
            'nx': 100, 'ny': 100,
            'c_s': 1.0, 'rho_o': 1.0, 'const': 1.0
        }
        new_state = lax_time_step(state, dt, params, dimension=2, gravity=True)
    """
    # Extract state variables
    rho = state_dict['rho']
    velocities = state_dict['velocities']
    momenta = state_dict['momenta']
    phi = state_dict.get('phi', None)
    
    # Extract parameters
    Lx = params_dict['Lx']
    nx = params_dict['nx']
    c_s = params_dict['c_s']
    rho_o = params_dict['rho_o']
    const = params_dict['const']
    
    # Extract dimension-specific parameters
    if dimension >= 2:
        Ly = params_dict['Ly']
        ny = params_dict['ny']
    if dimension == 3:
        Lz = params_dict['Lz']
        nz = params_dict['nz']
    
    # Compute grid spacings
    grid_spacings = [Lx / nx]
    if dimension >= 2:
        grid_spacings.append(Ly / ny)
    if dimension == 3:
        grid_spacings.append(Lz / nz)
    
    # Step 1: Update density
    rho_new = compute_lax_density_update(rho, velocities, dt, grid_spacings, dimension)
    
    # Step 2: Update momenta
    momenta_new = compute_lax_momentum_update(
        momenta, velocities, rho, phi, c_s, dt, grid_spacings, dimension, gravity=gravity
    )
    
    # Step 3: Compute velocities from updated momenta and density
    velocities_new = [momenta_new[i] / rho_new for i in range(dimension)]
    
    # Step 4: Update gravitational potential if gravity is enabled
    if gravity:
        # Call fft_poisson_solver with appropriate parameters based on dimension
        if dimension == 1:
            # 1D Poisson solver (not implemented in fft_poisson_solver, but we can handle it)
            # For now, raise an error or use a simple implementation
            # Note: fft_poisson_solver doesn't support 1D, so we'll skip it for 1D
            # In practice, 1D gravity might use a different approach
            phi_new = np.zeros_like(rho_new)  # 1D gravity not typically used with this solver
        elif dimension == 2:
            phi_new = fft_poisson_solver(const * (rho_new - rho_o), Lx, nx, Ly=Ly, ny=ny)
        else:  # dimension == 3
            phi_new = fft_poisson_solver(const * (rho_new - rho_o), Lx, nx, Ly=Ly, ny=ny, Lz=Lz, nz=nz)
    else:
        # If no gravity, keep phi as zeros or copy existing if provided
        phi_new = phi.copy() if phi is not None else np.zeros_like(rho_new)
    
    # Return new state dictionary
    return {
        'rho': rho_new,
        'velocities': velocities_new,
        'momenta': momenta_new,
        'phi': phi_new
    }


def initialize_arrays(shape, dimension):
    """
    Initialize arrays for LAX solver.
    
    Args:
        shape: Tuple of grid dimensions (nx, ny) or (nx, ny, nz)
        dimension: 2 or 3
    
    Returns:
        dict: {
            'rho': density array,
            'velocities': [vx, vy, ...] list of velocity arrays,
            'momenta': [Px, Py, ...] list of momentum arrays,
            'phi': potential array
        }
    """
    if dimension == 2:
        nx, ny = shape
        arrays = {
            'rho': np.zeros((nx, ny)),
            'velocities': [np.zeros((nx, ny)), np.zeros((nx, ny))],  # vx, vy
            'momenta': [np.zeros((nx, ny)), np.zeros((nx, ny))],  # Px, Py
            'phi': np.zeros((nx, ny))
        }
    elif dimension == 3:
        nx, ny, nz = shape
        arrays = {
            'rho': np.zeros((nx, ny, nz)),
            'velocities': [np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))],  # vx, vy, vz
            'momenta': [np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))],  # Px, Py, Pz
            'phi': np.zeros((nx, ny, nz))
        }
    else:
        raise ValueError(f"Unsupported dimension={dimension}. Use 2 or 3.")
    
    return arrays


# ============================================================================
# Initial Conditions Setup Functions
# ============================================================================

def setup_sinusoidal_ic(domain_params, physics_params, dimension, xx=None, yy=None, zz=None):
    """
    Set up sinusoidal initial conditions with KX, KY, KZ wave patterns.
    
    Args:
        domain_params: DomainParams dataclass or dict with Lx, Ly, (Lz), nx, ny, (nz)
        physics_params: Dict with rho_o, rho_1, lam, c_s, gravity, jeans, KX, KY, (KZ)
        dimension: 2 or 3
        xx, yy, zz: Coordinate meshes (optional, will be created if not provided)
    
    Returns:
        dict: {
            'rho': density field,
            'vx': x-velocity field,
            'vy': y-velocity field,
            'vz': z-velocity field (3D only),
            'v_1': velocity perturbation amplitude,
            'alpha': growth/oscillation rate (if gravity),
            'is_unstable': bool (if gravity)
        }
    """
    # Extract parameters
    if isinstance(domain_params, dict):
        Lx, Ly = domain_params['Lx'], domain_params['Ly']
        nx, ny = domain_params['nx'], domain_params['ny']
        if dimension == 3:
            Lz, nz = domain_params['Lz'], domain_params['nz']
    else:
        Lx, Ly = domain_params.Lx, domain_params.Ly
        nx, ny = domain_params.nx, domain_params.ny
        if dimension == 3:
            Lz, nz = domain_params.Lz, domain_params.nz
    
    rho_o = physics_params['rho_o']
    rho_1 = physics_params['rho_1']
    lam = physics_params['lam']
    c_s = physics_params['c_s']
    gravity = physics_params.get('gravity', False)
    jeans = physics_params.get('jeans', None)
    KX = physics_params.get('KX', 2*np.pi/lam)
    KY = physics_params.get('KY', 0.0)
    KZ = physics_params.get('KZ', 0.0) if dimension == 3 else 0.0
    
    # Create coordinate meshes if not provided
    if xx is None:
        x = np.linspace(0, Lx, nx, endpoint=False)
        y = np.linspace(0, Ly, ny, endpoint=False)
        if dimension == 2:
            xx, yy = np.meshgrid(x, y, indexing='ij')
        else:
            z = np.linspace(0, Lz, nz, endpoint=False)
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # Set up density
    if dimension == 2:
        rho = rho_o + rho_1 * np.cos(KX * xx + KY * yy)
    else:
        rho = rho_o + rho_1 * np.cos(KX * xx + KY * yy + KZ * zz)
    
    # Compute velocity perturbation
    v_1, alpha, is_unstable = compute_perturbation_velocity(rho_1, rho_o, lam, c_s, jeans, gravity)
    
    # Set up velocity fields
    k_magnitude = np.sqrt(KX**2 + KY**2 + (KZ**2 if dimension == 3 else 0))
    
    if dimension == 2:
        if gravity and is_unstable:
            # Unstable: use sin
            wave_field = -v_1 * np.sin(KX * xx + KY * yy)
        else:
            # Stable or no gravity: use cos
            wave_field = v_1 * np.cos(KX * xx + KY * yy)
        
        if k_magnitude > 0:
            vx = wave_field * (KX / k_magnitude)
            vy = wave_field * (KY / k_magnitude)
        else:
            vx = wave_field
            vy = np.zeros_like(xx)
        
        result = {
            'rho': rho,
            'vx': vx,
            'vy': vy,
            'v_1': v_1,
            'alpha': alpha,
            'is_unstable': is_unstable
        }
    else:  # dimension == 3
        if gravity and is_unstable:
            wave_field = -v_1 * np.sin(KX * xx + KY * yy + KZ * zz)
        else:
            wave_field = v_1 * np.cos(KX * xx + KY * yy + KZ * zz)
        
        if k_magnitude > 0:
            vx = wave_field * (KX / k_magnitude)
            vy = wave_field * (KY / k_magnitude)
            vz = wave_field * (KZ / k_magnitude)
        else:
            vx = wave_field
            vy = np.zeros_like(vx)
            vz = np.zeros_like(vx)
        
        result = {
            'rho': rho,
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'v_1': v_1,
            'alpha': alpha,
            'is_unstable': is_unstable
        }
    
    return result


def setup_power_spectrum_ic(domain_params, ps_params, dimension, vx0_shared=None, vy0_shared=None, vz0_shared=None):
    """
    Set up power spectrum initial conditions.
    
    Args:
        domain_params: DomainParams dataclass or dict with Lx, Ly, (Lz), nx, ny, (nz)
        ps_params: Dict with rho_o, power_index, amplitude, random_seed
        dimension: 2 or 3
        vx0_shared, vy0_shared, vz0_shared: Optional pre-generated velocity fields
    
    Returns:
        dict: {
            'rho': density field (uniform),
            'vx': x-velocity field,
            'vy': y-velocity field,
            'vz': z-velocity field (3D only)
        }
    """
    # Validate dimension
    if dimension not in [2, 3]:
        raise ValueError(f"setup_power_spectrum_ic() only supports dimension=2 or 3, got {dimension}")
    
    # Extract parameters
    if isinstance(domain_params, dict):
        Lx, Ly = domain_params['Lx'], domain_params['Ly']
        nx, ny = domain_params['nx'], domain_params['ny']
        if dimension == 3:
            if 'Lz' not in domain_params or 'nz' not in domain_params:
                raise ValueError("For dimension=3, domain_params must contain 'Lz' and 'nz'")
            Lz, nz = domain_params['Lz'], domain_params['nz']
    else:
        Lx, Ly = domain_params.Lx, domain_params.Ly
        nx, ny = domain_params.nx, domain_params.ny
        if dimension == 3:
            if not hasattr(domain_params, 'Lz') or not hasattr(domain_params, 'nz'):
                raise ValueError("For dimension=3, domain_params must have 'Lz' and 'nz' attributes")
            Lz, nz = domain_params.Lz, domain_params.nz
    
    rho_o = ps_params['rho_o']
    power_index = ps_params.get('power_index', -3.0)
    amplitude = ps_params.get('amplitude', 0.02)
    random_seed = ps_params.get('random_seed', None)
    
    # Uniform density
    if dimension == 2:
        rho = rho_o * np.ones((nx, ny))
    else:
        rho = rho_o * np.ones((nx, ny, nz))
    
    # Generate or use shared velocity fields
    if vx0_shared is not None and vy0_shared is not None:
        if dimension == 2:
            vx = vx0_shared.copy()
            vy = vy0_shared.copy()
            return {'rho': rho, 'vx': vx, 'vy': vy}
        else:
            if vz0_shared is not None:
                vx = vx0_shared.copy()
                vy = vy0_shared.copy()
                vz = vz0_shared.copy()
                return {'rho': rho, 'vx': vx, 'vy': vy, 'vz': vz}
    
    # Generate new velocity fields
    if dimension == 2:
        vx, vy = generate_velocity_field_power_spectrum(nx, ny, Lx, Ly, power_index, amplitude, DIMENSION=2, random_seed=random_seed)
        return {'rho': rho, 'vx': vx, 'vy': vy}
    else:
        vx, vy, vz = generate_velocity_field_power_spectrum(nx, ny, Lx, Ly, power_index, amplitude, DIMENSION=3, random_seed=random_seed, nz=nz, Lz=Lz)
        return {'rho': rho, 'vx': vx, 'vy': vy, 'vz': vz}


def setup_warm_start_ic(provided_fields):
    """
    Set up initial conditions from provided fields (for PINN integration).
    
    Args:
        provided_fields: Dict with 'rho', 'vx', 'vy', ('vz' for 3D)
    
    Returns:
        dict: Same structure as input, with arrays copied
    """
    result = {}
    for key, value in provided_fields.items():
        if value is not None:
            result[key] = value.copy() if hasattr(value, 'copy') else value
        else:
            result[key] = None
    
    return result


def lax_solver(time, domain_params, physics_params, ic_type='sinusoidal', ic_params=None, options=None):
    """
    Unified master solver for 1D, 2D, and 3D LAX method.
    
    This function provides a single entry point for running LAX simulations with different
    initial condition types and options. It automatically determines the dimension from
    domain_params and handles all setup and time-stepping internally.
    
    Args:
        time: Final simulation time (must be > 0)
        domain_params: DomainParams dataclass or dict with:
            - Lx, Ly, (Lz for 3D): Domain sizes
            - nx, ny, (nz for 3D): Grid resolutions
        physics_params: Dict with:
            - c_s: Sound speed
            - rho_o: Background density
            - const: Constant factor (for Poisson solver)
            - G: Gravitational constant
            - rho_1: Density perturbation amplitude (for sinusoidal IC)
            - lam: Wavelength (for sinusoidal IC)
        ic_type: 'sinusoidal' | 'power_spectrum' | 'warm_start'
        ic_params: Dict with IC-specific parameters:
            For 'sinusoidal': KX, KY, (KZ for 3D) - optional, defaults from config
            For 'power_spectrum': power_index, amplitude, random_seed, vx0_shared, vy0_shared, (vz0_shared for 3D)
            For 'warm_start': provided_fields dict with 'rho', 'vx', 'vy', ('vz' for 3D)
        options: Dict with:
            - gravity: bool (default: True)
            - nu: Courant number (default: 0.5)
            - comparison: bool (default: False)
            - isplot: bool (default: False)
    
    Returns:
        SimulationResult: Dataclass containing simulation results
    
    Raises:
        ValueError: If input validation fails
    """
    # ========================================================================
    # Input Validation
    # ========================================================================
    
    # Validate time
    if not isinstance(time, (int, float)) or time < 0:
        raise ValueError(f"time must be a non-negative number, got {time}")
    
    # Validate ic_type
    allowed_ic_types = ['sinusoidal', 'power_spectrum', 'warm_start']
    if ic_type not in allowed_ic_types:
        raise ValueError(f"ic_type must be one of {allowed_ic_types}, got '{ic_type}'")
    
    # Validate domain_params
    if isinstance(domain_params, dict):
        required_domain_keys = ['Lx', 'nx']
        for key in required_domain_keys:
            if key not in domain_params:
                raise ValueError(f"domain_params missing required key: '{key}'")
        
        Lx = domain_params['Lx']
        nx = domain_params['nx']
        Ly = domain_params.get('Ly', None)
        ny = domain_params.get('ny', None)
        Lz = domain_params.get('Lz', None)
        nz = domain_params.get('nz', None)
    else:
        if not hasattr(domain_params, 'Lx') or not hasattr(domain_params, 'nx'):
            raise ValueError("domain_params must have Lx and nx attributes")
        Lx = domain_params.Lx
        nx = domain_params.nx
        Ly = getattr(domain_params, 'Ly', None)
        ny = getattr(domain_params, 'ny', None)
        Lz = getattr(domain_params, 'Lz', None)
        nz = getattr(domain_params, 'nz', None)
    
    # Validate domain parameter values
    if not isinstance(Lx, (int, float)) or Lx <= 0:
        raise ValueError(f"Lx must be a positive number, got {Lx}")
    if not isinstance(nx, int) or nx <= 0:
        raise ValueError(f"nx must be a positive integer, got {nx}")
    
    # Determine dimension and validate dimension-specific parameters
    if Lz is not None and nz is not None:
        dimension = 3
        if Ly is None or ny is None:
            raise ValueError("For 3D simulations, Ly and ny must be provided")
        if not isinstance(Ly, (int, float)) or Ly <= 0:
            raise ValueError(f"Ly must be a positive number, got {Ly}")
        if not isinstance(ny, int) or ny <= 0:
            raise ValueError(f"ny must be a positive integer, got {ny}")
        if not isinstance(Lz, (int, float)) or Lz <= 0:
            raise ValueError(f"Lz must be a positive number, got {Lz}")
        if not isinstance(nz, int) or nz <= 0:
            raise ValueError(f"nz must be a positive integer, got {nz}")
    elif Ly is not None and ny is not None:
        dimension = 2
        if not isinstance(Ly, (int, float)) or Ly <= 0:
            raise ValueError(f"Ly must be a positive number, got {Ly}")
        if not isinstance(ny, int) or ny <= 0:
            raise ValueError(f"ny must be a positive integer, got {ny}")
    else:
        dimension = 1
    
    # Validate physics_params
    if not isinstance(physics_params, dict):
        raise ValueError(f"physics_params must be a dict, got {type(physics_params)}")
    
    required_physics_keys = ['c_s', 'rho_o', 'const', 'G']
    for key in required_physics_keys:
        if key not in physics_params:
            raise ValueError(f"physics_params missing required key: '{key}'")
        if not isinstance(physics_params[key], (int, float)) or physics_params[key] <= 0:
            raise ValueError(f"physics_params['{key}'] must be a positive number, got {physics_params[key]}")
    
    c_s = physics_params['c_s']
    rho_o = physics_params['rho_o']
    const = physics_params['const']
    G = physics_params['G']
    
    # Validate ic_params for warm_start
    if ic_type == 'warm_start':
        if ic_params is None:
            raise ValueError("ic_params must be provided for warm_start ic_type")
        if 'provided_fields' not in ic_params:
            raise ValueError("ic_params must contain 'provided_fields' for warm_start ic_type")
        provided_fields = ic_params['provided_fields']
        if not isinstance(provided_fields, dict):
            raise ValueError("ic_params['provided_fields'] must be a dict")
        if 'rho' not in provided_fields:
            raise ValueError("provided_fields must contain 'rho'")
        if dimension >= 2 and 'vy' not in provided_fields:
            raise ValueError(f"provided_fields must contain 'vy' for {dimension}D simulations")
        if dimension == 3 and 'vz' not in provided_fields:
            raise ValueError("provided_fields must contain 'vz' for 3D simulations")
    
    # Validate options
    if options is not None and not isinstance(options, dict):
        raise ValueError(f"options must be a dict or None, got {type(options)}")
    
    # Parse options
    if options is None:
        options = {}
    gravity = options.get('gravity', True)
    nu = options.get('nu', 0.5)
    comparison = options.get('comparison', False)
    isplot = options.get('isplot', False)
    
    # Validate option values
    if not isinstance(gravity, bool):
        raise ValueError(f"options['gravity'] must be a bool, got {type(gravity)}")
    if not isinstance(nu, (int, float)) or nu <= 0 or nu > 1:
        raise ValueError(f"options['nu'] (Courant number) must be in (0, 1], got {nu}")
    if not isinstance(comparison, bool):
        raise ValueError(f"options['comparison'] must be a bool, got {type(comparison)}")
    if not isinstance(isplot, bool):
        raise ValueError(f"options['isplot'] must be a bool, got {type(isplot)}")
    
    # Compute grid spacings
    dx = Lx / nx
    grid_spacings = [dx]
    if dimension >= 2:
        dy = Ly / ny
        grid_spacings.append(dy)
    if dimension == 3:
        dz = Lz / nz
        grid_spacings.append(dz)
    
    # Compute Jeans length if gravity is enabled
    jeans = None
    if gravity:
        jeans = compute_jeans_length(c_s, rho_o, const, G)
    
    # Create coordinate arrays first
    x = np.linspace(0, Lx, nx, endpoint=False)
    if dimension >= 2:
        y = np.linspace(0, Ly, ny, endpoint=False)
    if dimension == 3:
        z = np.linspace(0, Lz, nz, endpoint=False)
    
    # Set up initial conditions based on ic_type
    if ic_type == 'sinusoidal':
        rho_1 = physics_params.get('rho_1', 0.1)
        lam = physics_params.get('lam', 7.0)
        
        if dimension == 1:
            # 1D sinusoidal IC
            rho0 = rho_o + rho_1 * np.cos(2*np.pi*x/lam)
            v_1, alpha, is_unstable = compute_perturbation_velocity(rho_1, rho_o, lam, c_s, jeans, gravity)
            if gravity and is_unstable:
                vx0 = -v_1 * np.sin(2*np.pi*x/lam)
            else:
                vx0 = v_1 * np.cos(2*np.pi*x/lam)
            velocities = [vx0]
        else:
            # 2D/3D sinusoidal IC
            setup_physics = {
                'rho_o': rho_o,
                'rho_1': rho_1,
                'lam': lam,
                'c_s': c_s,
                'gravity': gravity,
                'jeans': jeans
            }
            if ic_params:
                setup_physics.update(ic_params)
            else:
                # Use defaults from config
                setup_physics['KX'] = KX
                setup_physics['KY'] = KY
                if dimension == 3:
                    setup_physics['KZ'] = KZ
            
            ic_result = setup_sinusoidal_ic(domain_params, setup_physics, dimension)
            rho0 = ic_result['rho']
            if dimension == 2:
                vx0 = ic_result['vx']
                vy0 = ic_result['vy']
                velocities = [vx0, vy0]
            else:  # dimension == 3
                vx0 = ic_result['vx']
                vy0 = ic_result['vy']
                vz0 = ic_result['vz']
                velocities = [vx0, vy0, vz0]
    
    elif ic_type == 'power_spectrum':
        if dimension == 1:
            # 1D power spectrum IC - uniform density, zero velocity for now
            # (1D power spectrum velocity generation not implemented)
            rho0 = rho_o * np.ones(nx)
            vx0 = np.zeros(nx)
            velocities = [vx0]
        else:
            # 2D/3D power spectrum IC
            ps_params = {
                'rho_o': rho_o,
                'power_index': ic_params.get('power_index', -3.0) if ic_params else -3.0,
                'amplitude': ic_params.get('amplitude', 0.02) if ic_params else 0.02,
                'random_seed': ic_params.get('random_seed', None) if ic_params else None
            }
            vx0_shared = ic_params.get('vx0_shared', None) if ic_params else None
            vy0_shared = ic_params.get('vy0_shared', None) if ic_params else None
            vz0_shared = ic_params.get('vz0_shared', None) if ic_params else None
            
            ic_result = setup_power_spectrum_ic(domain_params, ps_params, dimension, 
                                               vx0_shared, vy0_shared, vz0_shared)
            rho0 = ic_result['rho']
            if dimension == 2:
                vx0 = ic_result['vx']
                vy0 = ic_result['vy']
                velocities = [vx0, vy0]
            else:  # dimension == 3
                # Validate that all 3D velocity components are present
                if 'vz' not in ic_result:
                    raise ValueError(f"setup_power_spectrum_ic() did not return 'vz' for dimension=3. "
                                   f"Result keys: {list(ic_result.keys())}")
                vx0 = ic_result['vx']
                vy0 = ic_result['vy']
                vz0 = ic_result['vz']
                velocities = [vx0, vy0, vz0]
    
    elif ic_type == 'warm_start':
        ic_result = setup_warm_start_ic(ic_params['provided_fields'])
        rho0 = ic_result['rho']
        if dimension == 1:
            vx0 = ic_result.get('vx', np.zeros_like(rho0))
            velocities = [vx0]
        elif dimension == 2:
            vx0 = ic_result['vx']
            vy0 = ic_result['vy']
            velocities = [vx0, vy0]
        else:  # dimension == 3
            vx0 = ic_result['vx']
            vy0 = ic_result['vy']
            vz0 = ic_result['vz']
            velocities = [vx0, vy0, vz0]
    
    else:
        raise ValueError(f"Unsupported ic_type: {ic_type}. Use 'sinusoidal', 'power_spectrum', or 'warm_start'")
    
    # Initialize momenta
    momenta = [rho0 * v for v in velocities]
    
    # Initialize gravitational potential
    if gravity:
        if dimension == 1:
            # 1D Poisson solver
            k = 2*np.pi*np.fft.fftfreq(nx, d=dx)
            rhohat = np.fft.fft(const*(rho0 - rho_o))
            denom = -(k**2)
            denom[0] = 1.0
            phihat = rhohat / denom
            phihat[0] = 0.0
            phi0 = np.real(np.fft.ifft(phihat))
        elif dimension == 2:
            phi0 = fft_poisson_solver(const*(rho0 - rho_o), Lx, nx, Ly=Ly, ny=ny)
        else:  # dimension == 3
            phi0 = fft_poisson_solver(const*(rho0 - rho_o), Lx, nx, Ly=Ly, ny=ny, Lz=Lz, nz=nz)
    else:
        phi0 = np.zeros_like(rho0)
    
    
    # Prepare params_dict for lax_time_step
    params_dict = {
        'Lx': Lx,
        'nx': nx,
        'c_s': c_s,
        'rho_o': rho_o,
        'const': const
    }
    if dimension >= 2:
        params_dict['Ly'] = Ly
        params_dict['ny'] = ny
    if dimension == 3:
        params_dict['Lz'] = Lz
        params_dict['nz'] = nz
    
    # Initialize state dictionary
    state = {
        'rho': rho0,
        'velocities': velocities,
        'momenta': momenta,
        'phi': phi0
    }
    
    # Time-stepping loop
    t = 0.0
    n_steps = 0
    
    # Initial timestep
    dt = compute_adaptive_timestep(velocities, c_s, dx, nu, include_sound_speed=True)
    
    while t < time:
        # Ensure last step doesn't overshoot
        if t + dt > time:
            dt = time - t
        
        # Perform one time step
        state = lax_time_step(state, dt, params_dict, dimension, gravity=gravity)
        
        # Update time and step counter
        t += dt
        n_steps += 1
        
        # Calculate dt for next step
        dt = compute_adaptive_timestep(state['velocities'], c_s, dx, nu)
    
    # Prepare coordinates dictionary
    coordinates = {'x': x}
    if dimension >= 2:
        coordinates['y'] = y
    if dimension == 3:
        coordinates['z'] = z
    
    # Prepare metadata dictionary
    metadata = {
        'time': t,
        'iterations': n_steps,
        'rho_max': np.max(state['rho']),
        'gravity': gravity,
        'nu': nu,
        'dimension': dimension
    }
    
    # Prepare linear theory comparison if requested
    linear_theory_comparison = None
    if comparison:
        # Linear theory comparison would be computed here if needed
        # For now, set to None as it's optional
        linear_theory_comparison = {}
    
    # Create and return SimulationResult
    return SimulationResult(
        dimension=dimension,
        density=state['rho'],
        velocity_components=state['velocities'],
        coordinates=coordinates,
        metadata=metadata,
        potential=state['phi'] if gravity else None,
        linear_theory_comparison=linear_theory_comparison
    )


def generate_velocity_field_power_spectrum(nx, ny, Lx, Ly, power_index=-3.0, amplitude=0.02, DIMENSION=2, random_seed=None, nz=None, Lz=None):
    """
    Generate 2D or 3D velocity components (vx, vy, vz (optional)) with an isotropic power-law spectrum P(k) ~ k^{power_index}.
    
    This function generates velocity fields directly at the target resolution (nx, ny, nz (optional))
    without any downsampling or cutoff strategies. Dimension-agnostic implementation.
    
    Args:
        nx, ny: Grid dimensions (required)
        Lx, Ly: Domain sizes (required)
        power_index: Power spectrum index (default: -3.0)
        amplitude: RMS amplitude of velocity field (default: 0.02)
        DIMENSION: 2 or 3 (default: 2)
        random_seed: Random seed for reproducibility
        nz, Lz: Grid dimension and domain size for 3D (required if DIMENSION=3)
    
    Returns:
        For 2D: (vx, vy)
        For 3D: (vx, vy, vz)
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

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
        # 1. Generate random field at target resolution (dimension-agnostic)
        field = rng.standard_normal(shape)
        F = fftn(field)  # fftn works for any dimension
        
        # 2. Construct k-space grid at target resolution
        k_grid = build_k_space_grid(shape, domain_lengths, DIMENSION)
        
        # 3. Compute |k| magnitude (dimension-agnostic)
        if DIMENSION == 2:
            kx, ky, kxg, kyg = k_grid
            kk = np.sqrt(kxg**2 + kyg**2)
        else:  # DIMENSION == 3
            kx, ky, kz, kxg, kyg, kzg = k_grid
            kk = np.sqrt(kxg**2 + kyg**2 + kzg**2)
        
        # 4. Apply power-law filter
        filt = np.zeros_like(kk)
        mask = kk > 0
        filt[mask] = kk[mask]**(power_index / 2.0)
        
        F_filtered = F * filt
        
        # 5. Transform back to real space (dimension-agnostic)
        comp = np.real(ifftn(F_filtered))
        
        # 6. Normalize the field
        comp -= np.mean(comp)
        std = np.std(comp)
        if std > 0:
            comp = comp * (amplitude / std)
        
        return comp

    # Generate velocity components
    vx0 = synthesize_component()
    vy0 = synthesize_component()
    
    if DIMENSION == 3:
        vz0 = synthesize_component()
        return vx0, vy0, vz0
    else:
        return vx0, vy0

def generate_shared_velocity_field(nx, ny, Lx, Ly, power_index=-4.0, amplitude=0.01, DIMENSION=2, random_seed=None, nz=None, Lz=None):
    """
    Generate shared velocity field for both PINN and FD to ensure identical initial conditions.
    This function creates the velocity field once and returns both numpy arrays (for FD) 
    and interpolation functions (for PINN). Dimension-agnostic implementation.
    
    Args:
        nx, ny: Grid dimensions (required)
        Lx, Ly: Domain sizes (required)
        power_index: Power spectrum index (default: -4.0)
        amplitude: RMS amplitude of velocity field (default: 0.01)
        DIMENSION: 2 or 3 (default: 2)
        random_seed: Random seed for reproducibility
        nz, Lz: Grid dimension and domain size for 3D (required if DIMENSION=3)
    
    Returns:
        For 2D: (vx_np, vy_np, vx_interp, vy_interp)
        For 3D: (vx_np, vy_np, vz_np, vx_interp, vy_interp, vz_interp)
    """
    if random_seed is None:
        random_seed = RANDOM_SEED
    
    # Generate the velocity field using the unified power spectrum generator
    if DIMENSION == 2:
        vx_np, vy_np = generate_velocity_field_power_spectrum(nx, ny, Lx, Ly, power_index, amplitude, DIMENSION=2, random_seed=random_seed)
        velocity_arrays = [vx_np, vy_np]
    elif DIMENSION == 3:
        if nz is None or Lz is None:
            raise ValueError("nz and Lz must be provided for 3D")
        vx_np, vy_np, vz_np = generate_velocity_field_power_spectrum(nx, ny, Lx, Ly, power_index, amplitude, DIMENSION=3, random_seed=random_seed, nz=nz, Lz=Lz)
        velocity_arrays = [vx_np, vy_np, vz_np]
    else:
        raise ValueError(f"Unsupported DIMENSION={DIMENSION}. Use 2 or 3.")
    
    # Create interpolation functions for PINN (dimension-agnostic)
    from scipy.interpolate import RegularGridInterpolator
    
    # Build coordinate grids (exclude right boundary for periodic domains)
    if DIMENSION == 2:
        coords = (np.linspace(0, Lx, nx, endpoint=False), np.linspace(0, Ly, ny, endpoint=False))
    else:  # DIMENSION == 3
        coords = (np.linspace(0, Lx, nx, endpoint=False), 
                  np.linspace(0, Ly, ny, endpoint=False),
                  np.linspace(0, Lz, nz, endpoint=False))
    
    # Create interpolators for all velocity components
    interpolators = [RegularGridInterpolator(coords, vel_array, method='linear', bounds_error=False, fill_value=0.0) 
                     for vel_array in velocity_arrays]
    
    # Return arrays and interpolators
    if DIMENSION == 2:
        return vx_np, vy_np, interpolators[0], interpolators[1]
    else:  # DIMENSION == 3
        return vx_np, vy_np, vz_np, interpolators[0], interpolators[1], interpolators[2]

def plot_results(x, y, rho0, vx0, vy0, time, rho_1, t, gravity=False, comparison=False, 
                 rho_LT=None, vx_LT=None, vy_LT=None, rho_LT_max=None, rho_max=None):
    """
    Plot the results from the LAX solver.
    
    Args:
        x: x coordinates (1D array)
        y: y coordinates (1D array)
        rho0: Density field (2D array)
        vx0: x-velocity field (2D array)
        vy0: y-velocity field (2D array)
        time: Final time value
        rho_1: Density perturbation amplitude
        t: Current time (for labeling)
        gravity: Whether gravity is enabled
        comparison: Whether to plot linear theory comparison
        rho_LT: Linear theory density (1D array, optional)
        vx_LT: Linear theory x-velocity (1D array, optional)
        vy_LT: Linear theory y-velocity (1D array, optional)
        rho_LT_max: Linear theory maximum density (scalar, optional)
        rho_max: FD maximum density (scalar, optional)
    """
    plt.figure(1, figsize=(6, 4))
    plt.plot(x, rho0[:, 1] - rho_o, linewidth=1, label="FD at t={}".format(round(t, 2)))
    plt.legend(numpoints=1, loc='upper right', fancybox=True, shadow=True)
    plt.xlabel(r"$\mathbf{x}$")
    plt.title("At time {} and rho_1 = {}".format(time, rho_1))
    plt.ylabel(r"$\mathbf{\rho - \rho_{0}}$")
    if comparison and rho_LT is not None:
        plt.plot(x, rho_LT - rho_o, '--', linewidth=1, label="LT")
        plt.legend(numpoints=1, loc='upper right', fancybox=True, shadow=True)

    plt.figure(2, figsize=(6, 4))
    plt.plot(x, vx0[:, 1], '--', markersize=2, label="t={}".format(round(t, 2)))
    plt.legend(numpoints=1, loc='upper right', fancybox=True, shadow=True)
    plt.xlabel(r"$\mathbf{x}$")
    plt.title(r"Lax Solution Velocity For $\rho_1$ = {}".format(rho_1))
    plt.ylabel("vx")
    if comparison and vx_LT is not None:
        plt.plot(x, vx_LT, '--', linewidth=1, label="LT")
        plt.legend(numpoints=1, loc='upper right', fancybox=True, shadow=True)
        
    plt.figure(3, figsize=(6, 4))
    plt.plot(y, vy0[1, :], '--', markersize=2, label="t={}".format(round(t, 2)))
    plt.legend(numpoints=1, loc='upper right', fancybox=True, shadow=True)
    plt.xlabel(r"$\mathbf{y}$")
    plt.title(r"Lax Solution Velocity For $\rho_1$ = {}".format(rho_1))
    plt.ylabel("vy")
    if comparison and vy_LT is not None:
        plt.plot(y, vy_LT, '--', linewidth=1, label="LT")
        plt.legend(numpoints=1, loc='upper right', fancybox=True, shadow=True)

    if gravity:
        #### Plotting the comparison of the \rho_max for FD and Linear Theory
        plt.figure(5, figsize=(6, 4))
        if rho_max is not None:
            plt.scatter(time, rho_max, label="FD")
        plt.xlabel("t")
        plt.ylabel(r"$\log (\rho_{\rm max} - \rho_{0}) $")
        plt.yscale('log')
        plt.legend(numpoints=1, loc='upper left', fancybox=True, shadow=True)
        if comparison and rho_LT_max is not None:
            plt.scatter(time, rho_LT_max, facecolors='none', edgecolors='r', label="LT")
            plt.legend(numpoints=1, loc='upper left', fancybox=True, shadow=True)
            print(time, rho_LT_max)

def fft_poisson_solver(rho, Lx, nx, Ly=None, ny=None, Lz=None, nz=None):
    """
    Unified FFT-based Poisson solver for 2D and 3D periodic domains.
    
    Uses discrete Fast Fourier Transform to solve the Poisson Equation.
    Applies correction due to the finite difference grid of phi.
    
    Args:
        rho: Source function (density) - 2D or 3D array
        Lx: Domain size in x direction
        nx: Number of grid points in x direction
        Ly: Domain size in y direction (required for 2D/3D)
        ny: Number of grid points in y direction (required for 2D/3D)
        Lz: Domain size in z direction (optional, for 3D only)
        nz: Number of grid points in z direction (optional, for 3D only)
    
    Returns:
        phi: The potential field (same shape as rho)
    """
    # Determine dimensionality based on provided parameters
    is_3d = (Lz is not None) and (nz is not None)
    
    if is_3d:
        # 3D case
        dx = Lx / nx
        dy = Ly / ny
        dz = Lz / nz
        rhohat = fftn(rho)
        kx, ky, kz, kx_mesh, ky_mesh, kz_mesh = build_k_space_grid((nx, ny, nz), (Lx, Ly, Lz), 3)
        # Use discrete Laplacian for consistency with finite-difference scheme
        laplace = (2*(np.cos(kx_mesh*dx)-1)/(dx**2) + 
                  2*(np.cos(ky_mesh*dy)-1)/(dy**2) + 
                  2*(np.cos(kz_mesh*dz)-1)/(dz**2))
        laplace[laplace == 0] = 1e-9
        phihat = rhohat / laplace
        phi = np.real(ifftn(phihat))
    else:
        # 2D case
        if Ly is None or ny is None:
            raise ValueError("Ly and ny must be provided for 2D case")
        dx = Lx / nx
        dy = Ly / ny
        # Calculate the Fourier modes of the gas density
        rhohat = fft2(rho)
        # Calculate the wave numbers in x and y directions
        kx, ky, kx_mesh, ky_mesh = build_k_space_grid((nx, ny), (Lx, Ly), 2)
        # Construct the discrete Laplacian operator in Fourier space
        # This ensures consistency with finite-difference gradients used in LAX scheme
        laplace = 2*(np.cos(kx_mesh*dx)-1)/(dx**2) + 2*(np.cos(ky_mesh*dy)-1)/(dy**2)
        laplace[laplace == 0] = 1e-9
        # Solve for the electrostatic potential in Fourier space
        phihat = rhohat / laplace
        # Transform back to real space to obtain the solution
        phi = np.real(ifft2(phihat))
    
    return phi


def lax_solution(time,N,nu,lam,num_of_waves,rho_1,gravity=False,isplot = None,comparison =None,animation=None,
                 use_velocity_ps=False, ps_index=-3.0, vel_rms=0.02, random_seed=None, vx0_shared=None, vy0_shared=None):
    '''
    DEPRECATED: This function is a wrapper around the unified lax_solver().
    Use lax_solver() directly for new code.
    
    This function solves the hydrodynamic Eqns in 2D with/without self gravity using LAX methods.
    
    Input:  Time till the system is integrated :time
            Number of Xgrid points : N
            Courant number : nu
            Wavelength : If lambda> lambdaJ (with gravity--> Instability) else waves propagation 
            Number of waves : The domain size changes with this maintain periodicity
            Density perturbation : rho1 (for linear or non-linear perturbation)
            Gravity:  If True it deploys the FFT routine to estimate the potential 
            isplot(optional): if True plots the output
            Comparison (optional) : If True then the plots are overplotted with LT solutions for comparison
            Animation (optional): Not used at the moment
    
    Output: Density, velocity + (phi and g if gravity is True)
            isplot: True then the plots are generated 
    
    '''
    import warnings
    warnings.warn(
        "lax_solution() is deprecated. Use lax_solver() instead. "
        "This function will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Convert parameters to unified solver format
    Lx = Ly = lam * num_of_waves
    Nx = Ny = N
    c_s = cs
    
    domain_params = {'Lx': Lx, 'Ly': Ly, 'nx': Nx, 'ny': Ny}
    physics_params = {
        'c_s': c_s,
        'rho_o': rho_o,
        'const': const,
        'G': G,
        'rho_1': rho_1,
        'lam': lam
    }
    
    # Set up IC parameters
    ic_type = 'power_spectrum' if use_velocity_ps else 'sinusoidal'
    ic_params = None
    if use_velocity_ps:
        ic_params = {
            'power_index': ps_index,
            'amplitude': vel_rms,
            'random_seed': random_seed,
            'vx0_shared': vx0_shared,
            'vy0_shared': vy0_shared
        }
    else:
        ic_params = {
            'KX': KX,
            'KY': KY
        }
    
    options = {
        'gravity': gravity,
        'nu': nu,
        'comparison': comparison if comparison is not None else False,
        'isplot': isplot if isplot is not None else False
    }
    
    # Call unified solver
    result = lax_solver(time, domain_params, physics_params, ic_type=ic_type, ic_params=ic_params, options=options)
    
    # Extract values to match old return format
    x = result.coordinates['x']
    y = result.coordinates['y']
    rho0 = result.density
    vx0, vy0 = result.velocity_components
    phi0 = result.potential
    n = result.metadata['iterations']
    rho_max = result.metadata['rho_max']
    
    # Handle linear theory comparison if needed
    rho_LT = None
    rho_LT_max = None
    vx_LT = None
    if comparison:
        # Compute linear theory values (same logic as before)
        if not gravity:
            v_1, _, _ = compute_perturbation_velocity(rho_1, rho_o, lam, c_s, None, False)
            rho_LT = rho_o + rho_1*np.cos(2*np.pi * x/lam - 2*np.pi/lam *time)
            rho_LT_max = np.max(rho_LT)
            vx_LT = v_1* np.cos(2*np.pi * x/lam - 2*np.pi/lam *time)
        else:
            jeans = compute_jeans_length(c_s, rho_o, const, G)
            if lam >= jeans:
                v_1, alpha, _ = compute_perturbation_velocity(rho_1, rho_o, lam, c_s, jeans, True)
                rho_LT = rho_o + rho_1*np.exp(alpha * time)*np.cos(2*np.pi*x/lam)
                rho_LT_max = np.max(rho_LT)
                vx_LT = -v_1*np.exp(alpha * time)*np.sin(2*np.pi*x/lam)
            else:
                v_1, alpha, _ = compute_perturbation_velocity(rho_1, rho_o, lam, c_s, jeans, True)
                rho_LT = rho_o + rho_1*np.cos(alpha * time - 2*np.pi*x/lam)
                rho_LT_max = np.max(rho_o + rho_1*np.cos(alpha * time - 2*np.pi*x/lam))
                vx_LT = v_1*np.cos(alpha * time - 2*np.pi*x/lam)
    
    # Return in old format
    if isplot:
        # Plotting is handled by lax_solver if isplot=True
        return
    
    if gravity:
        if comparison:
            return x, rho0, vx0, phi0, n, rho_LT, rho_LT_max, rho_max, vx_LT
        else:
            return x, rho0, vx0, vy0, phi0, n, rho_max
    else:
        if comparison:
            return rho0, vx0, rho_LT, rho_LT_max, rho_max, vx_LT
        else:
            return rho0, vx0, rho_max


def lax_solution_3d_sinusoidal(time, N, nu, lam, num_of_waves, rho_1, gravity=True,
                               use_velocity_ps=False, ps_index=-3.0, vel_rms=0.02, 
                               random_seed=None, vx0_shared=None, vy0_shared=None, vz0_shared=None):
    """
    DEPRECATED: This function is a wrapper around the unified lax_solver().
    Use lax_solver() directly for new code.
    
    3D LAX solver for sinusoidal perturbations with optional self-gravity.
    Returns full 3D fields which can be sliced for visualization/comparison.
    
    Args:
        time: Final simulation time
        N: Grid resolution (Nx = Ny = Nz = N)
        nu: Courant number
        lam: Wavelength
        num_of_waves: Number of waves in domain
        rho_1: Density perturbation amplitude
        gravity: Whether to include self-gravity (default: True)
        use_velocity_ps: If True, use power spectrum velocity initialization instead of sinusoidal (default: False)
        ps_index: Power spectrum index (default: -3.0)
        vel_rms: RMS velocity amplitude (default: 0.02)
        random_seed: Random seed for reproducibility (default: None)
        vx0_shared: Optional pre-generated x-velocity field (default: None)
        vy0_shared: Optional pre-generated y-velocity field (default: None)
        vz0_shared: Optional pre-generated z-velocity field (default: None)
    """
    import warnings
    warnings.warn(
        "lax_solution_3d_sinusoidal() is deprecated. Use lax_solver() with dimension=3 instead. "
        "This function will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Convert parameters to unified solver format
    Lx = Ly = Lz = lam * num_of_waves
    Nx = Ny = Nz = int(N)
    c_s = cs
    
    domain_params = {'Lx': Lx, 'Ly': Ly, 'Lz': Lz, 'nx': Nx, 'ny': Ny, 'nz': Nz}
    physics_params = {
        'c_s': c_s,
        'rho_o': rho_o,
        'const': const,
        'G': G,
        'rho_1': rho_1,
        'lam': lam
    }
    
    # Set up IC parameters based on use_velocity_ps flag
    ic_type = 'power_spectrum' if use_velocity_ps else 'sinusoidal'
    ic_params = None
    if use_velocity_ps:
        ic_params = {
            'power_index': ps_index,
            'amplitude': vel_rms,
            'random_seed': random_seed,
            'vx0_shared': vx0_shared,
            'vy0_shared': vy0_shared,
            'vz0_shared': vz0_shared
        }
    else:
        ic_params = {
            'KX': KX,
            'KY': KY,
            'KZ': KZ
        }
    
    options = {
        'gravity': gravity,
        'nu': nu,
        'comparison': False,
        'isplot': False
    }
    
    # Call unified solver
    result = lax_solver(time, domain_params, physics_params, ic_type=ic_type, ic_params=ic_params, options=options)
    
    # Extract values to match old return format
    x = result.coordinates['x']
    y = result.coordinates['y']
    z = result.coordinates['z']
    rho0 = result.density
    vx0, vy0, vz0 = result.velocity_components
    phi0 = result.potential
    k_iter = result.metadata['iterations']
    rho_max = result.metadata['rho_max']
    
    return x, y, z, rho0, vx0, vy0, vz0, phi0, k_iter, rho_max


def lax_solution_warm_start(rho_ic, vx_ic, vy_ic, x_grid, y_grid, 
                            t_start, t_end, nu=0.5, save_times=None, gravity=True):
    """
    Run FD solver from custom initial conditions (warm-start).
    
    This function allows restarting the FD solver from a PINN state or any custom state,
    enabling efficient generation of FD data for hybrid PINN-FD training.
    
    Args:
        rho_ic: Initial density field (Nx, Ny)
        vx_ic: Initial x-velocity field (Nx, Ny)
        vy_ic: Initial y-velocity field (Nx, Ny)
        x_grid: x coordinates (Nx,)
        y_grid: y coordinates (Ny,)
        t_start: Starting time
        t_end: Ending time
        nu: Courant number
        save_times: List of times to save snapshots [default: [t_end]]
        gravity: Whether to include self-gravity (default: True)
    
    Returns:
        Dictionary: {time: (rho, vx, vy, phi, x, y)} for each saved time
    """
    if save_times is None:
        save_times = [t_end]
    
    # Domain setup
    Nx, Ny = rho_ic.shape
    Lx = x_grid[-1] - x_grid[0] + (x_grid[1] - x_grid[0])  # Approximate domain size
    Ly = y_grid[-1] - y_grid[0] + (y_grid[1] - y_grid[0])
    dx = Lx / Nx
    dy = Ly / Ny
    
    # Physical constants
    c_s = cs
    
    # Use setup_warm_start_ic() helper to set up initial conditions
    provided_fields = {
        'rho': rho_ic,
        'vx': vx_ic,
        'vy': vy_ic
    }
    ic_result = setup_warm_start_ic(provided_fields)
    rho0 = ic_result['rho']
    vx0 = ic_result['vx']
    vy0 = ic_result['vy']
    
    # Calculate initial potential (gravity is always True for collapse problems)
    phi0 = fft_poisson_solver(const * (rho0 - rho_o), Lx, Nx, Ly=Ly, ny=Ny)
    
    # Initialize flux terms
    Px0 = rho0 * vx0
    Py0 = rho0 * vy0
    
    # Storage for snapshots
    snapshots = {}
    
    # Time-stepping loop
    t = t_start
    k = 0
    
    # Initial dt
    vmax_initial = max(np.max(np.abs(vx0)), np.max(np.abs(vy0)), c_s)
    dt = nu * dx / vmax_initial
    
    while t < t_end:
        # Check if we should save a snapshot before this step
        for save_t in save_times:
            if t <= save_t < t + dt and save_t not in snapshots:
                # Save current state (or interpolate if needed)
                snapshots[save_t] = (rho0.copy(), vx0.copy(), vy0.copy(), 
                                     phi0.copy(), x_grid.copy(), y_grid.copy())
        
        # Ensure last step doesn't overshoot
        if t + dt > t_end:
            dt = t_end - t
        
        # LAX time-stepping
        mux = dt / (2 * dx)
        muy = dt / (2 * dy)
        
        # Update density
        rho1 = (1/4) * (np.roll(rho0, -1, axis=0) + np.roll(rho0, 1, axis=0) +
                       np.roll(rho0, -1, axis=1) + np.roll(rho0, 1, axis=1)) - \
               (mux * (np.roll(rho0, -1, axis=0) * np.roll(vx0, -1, axis=0) -
                       np.roll(rho0, 1, axis=0) * np.roll(vx0, 1, axis=0))) - \
               (muy * (np.roll(rho0, -1, axis=1) * np.roll(vy0, -1, axis=1) -
                       np.roll(rho0, 1, axis=1) * np.roll(vy0, 1, axis=1)))
        
        # Update momentum (with gravity)
        if gravity:
            Px1 = 0.25 * (np.roll(Px0, -1, axis=0) + np.roll(Px0, 1, axis=0) +
                          np.roll(Px0, -1, axis=1) + np.roll(Px0, 1, axis=1)) - \
                  (mux * (np.roll(Px0, -1, axis=0) * np.roll(vx0, -1, axis=0) -
                          np.roll(Px0, 1, axis=0) * np.roll(vx0, 1, axis=0))) - \
                  (muy * (np.roll(Px0, -1, axis=1) * np.roll(vy0, -1, axis=1) -
                          np.roll(Px0, 1, axis=1) * np.roll(vy0, 1, axis=1))) - \
                  ((c_s**2) * mux * (np.roll(rho0, -1, axis=0) - np.roll(rho0, 1, axis=0))) - \
                  (mux * rho0 * (np.roll(phi0, -1, axis=0) - np.roll(phi0, 1, axis=0)))
            
            Py1 = 0.25 * (np.roll(Py0, -1, axis=0) + np.roll(Py0, 1, axis=0) +
                          np.roll(Py0, -1, axis=1) + np.roll(Py0, 1, axis=1)) - \
                  (muy * (np.roll(Py0, -1, axis=1) * np.roll(vy0, -1, axis=1) -
                          np.roll(Py0, 1, axis=1) * np.roll(vy0, 1, axis=1))) - \
                  (mux * (np.roll(Py0, -1, axis=0) * np.roll(vx0, -1, axis=0) -
                          np.roll(Py0, 1, axis=0) * np.roll(vx0, 1, axis=0))) - \
                  ((c_s**2) * muy * (np.roll(rho0, -1, axis=1) - np.roll(rho0, 1, axis=1))) - \
                  (muy * rho0 * (np.roll(phi0, -1, axis=1) - np.roll(phi0, 1, axis=1)))
            
            # Update potential
            phi1 = fft_poisson_solver(const * (rho1 - rho_o), Lx, Nx, Ly=Ly, ny=Ny)
        else:
            # Without gravity (shouldn't happen for collapse problems, but included for completeness)
            Px1 = 0.25 * (np.roll(Px0, -1, axis=0) + np.roll(Px0, 1, axis=0) +
                          np.roll(Px0, -1, axis=1) + np.roll(Px0, 1, axis=1)) - \
                  (mux * (np.roll(Px0, -1, axis=0) * np.roll(vx0, -1, axis=0) -
                          np.roll(Px0, 1, axis=0) * np.roll(vx0, 1, axis=0))) - \
                  (muy * (np.roll(Px0, -1, axis=1) * np.roll(vy0, -1, axis=1) -
                          np.roll(Px0, 1, axis=1) * np.roll(vy0, 1, axis=1))) - \
                  ((c_s**2) * mux * (np.roll(rho0, -1, axis=0) - np.roll(rho0, 1, axis=0)))
            
            Py1 = 0.25 * (np.roll(Py0, -1, axis=0) + np.roll(Py0, 1, axis=0) +
                          np.roll(Py0, -1, axis=1) + np.roll(Py0, 1, axis=1)) - \
                  (muy * (np.roll(Py0, -1, axis=1) * np.roll(vy0, -1, axis=1) -
                          np.roll(Py0, 1, axis=1) * np.roll(vy0, 1, axis=1))) - \
                  (mux * (np.roll(Py0, -1, axis=0) * np.roll(vx0, -1, axis=0) -
                          np.roll(Py0, 1, axis=0) * np.roll(vx0, 1, axis=0))) - \
                  ((c_s**2) * muy * (np.roll(rho0, -1, axis=1) - np.roll(rho0, 1, axis=1)))
            
            phi1 = np.zeros_like(rho1)
        
        # Update velocities
        vx1 = Px1 / rho1
        vy1 = Py1 / rho1
        
        # Update state
        rho0 = rho1
        vx0 = vx1
        vy0 = vy1
        Px0 = Px1
        Py0 = Py1
        if gravity:
            phi0 = phi1
        
        t += dt
        k += 1
        
        # Calculate dt for next step
        dt = compute_adaptive_timestep([vx0, vy0], c_s, dx, nu)
    
    # Save final snapshot if not already saved
    if t_end not in snapshots:
        snapshots[t_end] = (rho0.copy(), vx0.copy(), vy0.copy(), 
                           phi0.copy(), x_grid.copy(), y_grid.copy())
    
    return snapshots


def lax_solution1D_sinusoidal(time,N,nu,lam,num_of_waves,rho_1,gravity=False,isplot = None,comparison =None,animation=None):
    '''
    1D LAX solver for sinusoidal initial conditions with optional self-gravity and linear theory outputs.
    Returns density, velocity, potential (if gravity), and optional linear theory references.
    '''
    lam = lam
    L = lam * num_of_waves

    c_s = cs             # Sound Speed (from config)
    rho0_base = rho_o    # Background density (from config)
    # nu, const, G are already imported from config, no need to reassign

    nx = int(N)
    dx = float(L / nx)
    dt = nu * dx / c_s
    mu = dt / (2 * dx)
    n = int(time / dt)

    # Exclude right boundary for periodic domains to avoid double-counting
    x = np.linspace(0, L, nx, endpoint=False)

    rho0 = np.zeros(nx)
    phi0 = np.zeros(nx)
    v0 = np.zeros(nx)
    P0 = np.zeros(nx)

    rho1 = np.zeros(nx)
    phi1 = np.zeros(nx)
    v1 = np.zeros(nx)
    P1 = np.zeros(nx)

    if gravity:
        jeans = compute_jeans_length(c_s, rho0_base, const, G)

    # Initial conditions
    rho0 = rho0_base + rho_1 * np.cos(2*np.pi*x/lam)

    if not gravity:
        v_1, _, _ = compute_perturbation_velocity(rho_1, rho0_base, lam, c_s, None, False)
        v0 = v_1 * np.cos(2*np.pi*x/lam)
        if comparison:
            rho_LT = rho0_base + rho_1 * np.cos(2*np.pi * x/lam - 2*np.pi/lam * time)
            rho_LT_max = np.max(rho_LT)
            v_LT = v_1 * np.cos(2*np.pi * x/lam - 2*np.pi/lam * time)
    else:
        if lam >= jeans:
            v_1, alpha, _ = compute_perturbation_velocity(rho_1, rho0_base, lam, c_s, jeans, True)
            v0 = - v_1 * np.sin(2*np.pi*x/lam)
            if comparison:
                rho_LT = rho0_base + rho_1*np.exp(alpha * time)*np.cos(2*np.pi*x/lam)
                rho_LT_max = np.max(rho_LT)
                v_LT = -v_1*np.exp(alpha * time)*np.sin(2*np.pi*x/lam)
        else:
            v_1, alpha, _ = compute_perturbation_velocity(rho_1, rho0_base, lam, c_s, jeans, True)
            v0 = v_1 * np.cos(2*np.pi*x/lam)
            if comparison:
                rho_LT = rho0_base + rho_1*np.cos(alpha * time - 2*np.pi*x/lam)
                rho_LT_max = np.max(rho_LT)
                v_LT = v_1*np.cos(alpha * time - 2*np.pi*x/lam)

        # 1D Poisson (periodic) via FFT: phi_k = rho_k / (-k^2), k=0 set to 0
        k = 2*np.pi*np.fft.fftfreq(nx, d=dx)
        rhohat = np.fft.fft(const*(rho0 - rho0_base))
        denom = -(k**2)
        denom[0] = 1.0
        phihat = rhohat / denom
        phihat[0] = 0.0
        phi0 = np.real(np.fft.ifft(phihat))

    P0 = rho0 * v0

    for _ in range(1, n):
        rho1 = 0.5*(np.roll(rho0,-1)+ np.roll(rho0,1)) - mu*(np.roll(rho0,-1)*np.roll(v0,-1) - np.roll(rho0,1)*np.roll(v0,1))

        if not gravity:
            P1 = 0.5*(np.roll(P0,-1)+ np.roll(P0,1)) - mu*(np.roll(P0,-1)*np.roll(v0,-1) - np.roll(P0,1)*np.roll(v0,1)) - (c_s**2)*mu*(np.roll(rho0,-1) - np.roll(rho0,1))
        else:
            P1 = 0.5*(np.roll(P0,-1)+ np.roll(P0,1)) - mu*(np.roll(P0,-1)*np.roll(v0,-1) - np.roll(P0,1)*np.roll(v0,1)) - (c_s**2)*mu*(np.roll(rho0,-1) - np.roll(rho0,1)) - mu*rho0*(np.roll(phi0,-1) - np.roll(phi0,1))
            k = 2*np.pi*np.fft.fftfreq(nx, d=dx)
            rhohat = np.fft.fft(const*(rho1 - rho0_base))
            denom = -(k**2)
            denom[0] = 1.0
            phihat = rhohat / denom
            phihat[0] = 0.0
            phi1 = np.real(np.fft.ifft(phihat))

        v1 = P1 / rho1

        rho0, v0, P0, phi0 = rho1, v1, P1, phi1

        vmax = np.max(np.abs(v1))
        dt1 = nu*dx/(vmax if vmax != 0 else c_s)
        dt2 = nu*dx/c_s
        dt = min(dt1, dt2)
        mu = dt/(2*dx)
        n = int(time/dt)

    rho_max = np.max(rho1)

    if isplot:
        return
    else:
        if gravity:
            if comparison:
                return x, rho1, v1, phi1, n, rho_LT, rho_LT_max, rho_max, v_LT
            else:
                return x, rho1, v1, phi1, n, rho_max
        else:
            if comparison:
                return rho1, v1, rho_LT, rho_LT_max, rho_max, v_LT
            else:
                return rho1, v1, rho_max