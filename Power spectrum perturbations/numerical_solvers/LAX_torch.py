import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional
from config import cs, rho_o, const, G, KX, KY, KZ

# Device setup - check at module import
has_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if has_gpu else "cpu")
dtype = torch.float64
if has_gpu:
    print(f"LAX_torch: GPU available, using device: {device}")
    print(f"  GPU name: {torch.cuda.get_device_name(0)}")
else:
    print(f"LAX_torch: No GPU available, using device: {device}")


@dataclass
class SimulationResult:
    """Result container for LAX solver simulations (PyTorch version)."""
    dimension: int  # 1, 2, or 3
    density: np.ndarray  # Density field (converted to numpy)
    velocity_components: list  # List of velocity arrays [vx, vy, ...] or [vx, vy, vz] (numpy)
    coordinates: dict  # Dict with 'x', 'y' (2D/3D), 'z' (3D) coordinate arrays (numpy)
    metadata: dict  # Dict with time, iterations, rho_max, etc.
    potential: Optional[np.ndarray] = None  # Gravitational potential (if gravity=True, numpy)
    linear_theory_comparison: Optional[dict] = None  # Optional linear theory comparison data


# ============================================================================
# Helper Functions
# ============================================================================

def compute_jeans_length_torch(c_s, rho_o, const, G):
    """
    Compute the Jeans length for gravitational instability (PyTorch version).
    
    Args:
        c_s: Sound speed
        rho_o: Background density
        const: Constant factor
        G: Gravitational constant
    
    Returns:
        Jeans length as torch tensor
    """
    return torch.sqrt(torch.tensor(4 * np.pi**2 * c_s**2 / (const * G * rho_o), device=device, dtype=dtype))


def compute_perturbation_velocity_torch(rho_1, rho_o, lam, c_s, jeans, gravity):
    """
    Compute the velocity perturbation amplitude v_1 and growth rate alpha (PyTorch version).
    
    Args:
        rho_1: Density perturbation amplitude
        rho_o: Background density
        lam: Wavelength
        c_s: Sound speed
        jeans: Jeans length (torch tensor or float)
        gravity: Whether gravity is enabled
    
    Returns:
        tuple: (v_1, alpha, is_unstable)
            v_1: Velocity perturbation amplitude (torch tensor)
            alpha: Growth/oscillation rate (torch tensor or None)
            is_unstable: True if gravitational instability (lam >= jeans), False otherwise
    """
    if not gravity:
        # No gravity: sound wave
        v_1 = torch.tensor((c_s * rho_1) / rho_o, device=device, dtype=dtype)
        return v_1, None, False
    
    # With gravity: check if unstable
    jeans_val = jeans.item() if isinstance(jeans, torch.Tensor) else jeans
    if lam >= jeans_val:
        # Gravitational instability
        alpha = torch.sqrt(torch.tensor(const * G * rho_o - c_s**2 * (2 * np.pi / lam)**2, device=device, dtype=dtype))
        v_1 = torch.tensor((rho_1 / rho_o) * (alpha.item() / (2 * np.pi / lam)), device=device, dtype=dtype)
        return v_1, alpha, True
    else:
        # Oscillatory regime
        alpha = torch.sqrt(torch.tensor(c_s**2 * (2 * np.pi / lam)**2 - const * G * rho_o, device=device, dtype=dtype))
        v_1 = torch.tensor((rho_1 / rho_o) * (alpha.item() / (2 * np.pi / lam)), device=device, dtype=dtype)
        return v_1, alpha, False


def build_k_space_grid_torch(shape, domain_lengths, dimension):
    """
    Build k-space grid for FFT operations (PyTorch version).
    
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
        kx = 2 * np.pi * torch.fft.fftfreq(nx, d=dx).to(device)
        ky = 2 * np.pi * torch.fft.fftfreq(ny, d=dy).to(device)
        kx_mesh, ky_mesh = torch.meshgrid(kx, ky, indexing='ij')
        return kx, ky, kx_mesh, ky_mesh
    elif dimension == 3:
        nx, ny, nz = shape
        Lx, Ly, Lz = domain_lengths
        dx = Lx / nx
        dy = Ly / ny
        dz = Lz / nz
        kx = 2 * np.pi * torch.fft.fftfreq(nx, d=dx).to(device)
        ky = 2 * np.pi * torch.fft.fftfreq(ny, d=dy).to(device)
        kz = 2 * np.pi * torch.fft.fftfreq(nz, d=dz).to(device)
        kx_mesh, ky_mesh, kz_mesh = torch.meshgrid(kx, ky, kz, indexing='ij')
        return kx, ky, kz, kx_mesh, ky_mesh, kz_mesh
    else:
        raise ValueError(f"Unsupported dimension={dimension}. Use 2 or 3.")


def compute_adaptive_timestep_torch(velocities, c_s, dx, nu, include_sound_speed=False):
    """
    Compute adaptive timestep based on CFL condition (PyTorch version).
    
    Args:
        velocities: List of velocity tensors [vx, vy, ...] or single velocity tensor, or None
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
            vmax = max(max(torch.max(torch.abs(v)).item() for v in velocities), c_s)
        else:
            vmax = max(torch.max(torch.abs(velocities)).item(), c_s)
        return nu * dx / vmax
    else:
        if velocities is None:
            vmax = c_s
        elif isinstance(velocities, (list, tuple)):
            vmax = max(torch.max(torch.abs(v)).item() for v in velocities)
        else:
            vmax = torch.max(torch.abs(velocities)).item()
        
        dt1 = nu * dx / vmax if vmax > 1e-9 else float('inf')
        dt2 = nu * dx / c_s
        return min(dt1, dt2)


def compute_lax_density_update_torch(rho, velocities, dt, grid_spacings, dimension):
    """
    Compute LAX density update using dimension-agnostic approach (PyTorch version).
    
    This function implements the LAX method for updating density in the continuity equation:
    ∂ρ/∂t + ∇·(ρv) = 0
    
    The update uses a dimension-agnostic approach that works for 1D, 2D, and 3D without
    if-else chains by iterating over dimensions.
    
    Args:
        rho: Current density tensor (1D, 2D, or 3D torch tensor)
        velocities: List of velocity tensors [vx, vy, ...] or [vx, vy, vz] for 3D
                   Length must match dimension
        dt: Time step
        grid_spacings: List of grid spacings [dx, dy, ...] or [dx, dy, dz] for 3D
                       Length must match dimension
        dimension: Integer dimension (1, 2, or 3)
    
    Returns:
        Updated density tensor (same shape as input rho)
    
    Examples:
        # 1D
        rho_new = compute_lax_density_update_torch(rho, [vx], dt, [dx], 1)
        
        # 2D
        rho_new = compute_lax_density_update_torch(rho, [vx, vy], dt, [dx, dy], 2)
        
        # 3D
        rho_new = compute_lax_density_update_torch(rho, [vx, vy, vz], dt, [dx, dy, dz], 3)
    """
    if len(velocities) != dimension:
        raise ValueError(f"Number of velocity tensors ({len(velocities)}) must match dimension ({dimension})")
    if len(grid_spacings) != dimension:
        raise ValueError(f"Number of grid spacings ({len(grid_spacings)}) must match dimension ({dimension})")
    
    # Compute mu values for each dimension: mu_i = dt / (2 * dx_i)
    mu_values = [dt / (2 * dx) for dx in grid_spacings]
    
    # Initialize with averaging term: (1/(2*dimension)) * sum of all rolled neighbors
    # For each dimension, we add contributions from both +1 and -1 rolls
    rho_new = torch.zeros_like(rho)
    
    # Sum all rolled neighbors (forward and backward for each dimension)
    # Note: PyTorch uses 'dims' parameter instead of 'axis'
    for dim in range(dimension):
        rho_new += torch.roll(rho, -1, dims=dim)  # Forward roll
        rho_new += torch.roll(rho, 1, dims=dim)   # Backward roll
    
    # Normalize by 1/(2*dimension)
    rho_new = rho_new / (2 * dimension)
    
    # Subtract flux terms for each dimension: -mu_i * (flux_forward - flux_backward)
    # where flux = rho * velocity
    for dim in range(dimension):
        mu = mu_values[dim]
        vel = velocities[dim]
        
        # Forward flux: rho_rolled_forward * vel_rolled_forward
        rho_forward = torch.roll(rho, -1, dims=dim)
        vel_forward = torch.roll(vel, -1, dims=dim)
        flux_forward = rho_forward * vel_forward
        
        # Backward flux: rho_rolled_backward * vel_rolled_backward
        rho_backward = torch.roll(rho, 1, dims=dim)
        vel_backward = torch.roll(vel, 1, dims=dim)
        flux_backward = rho_backward * vel_backward
        
        # Subtract the flux difference
        rho_new -= mu * (flux_forward - flux_backward)
    
    return rho_new


def compute_lax_momentum_update_torch(momenta, velocities, rho, phi, c_s, dt, grid_spacings, dimension, gravity=True):
    """
    Compute LAX momentum update using dimension-agnostic approach (PyTorch version).
    
    This function implements the LAX method for updating momentum in the momentum equation:
    ∂(ρv)/∂t + ∇·(ρv⊗v) = -∇P - ρ∇φ
    
    The update uses a dimension-agnostic approach that works for 1D, 2D, and 3D without
    if-else chains by iterating over dimensions.
    
    Args:
        momenta: List of momentum tensors [Px, Py, ...] or [Px, Py, Pz] for 3D
                 Length must match dimension
        velocities: List of velocity tensors [vx, vy, ...] or [vx, vy, vz] for 3D
                   Length must match dimension
        rho: Current density tensor (same shape as momentum tensors)
        phi: Gravitational potential tensor (same shape as momentum tensors)
             Only used if gravity=True
        c_s: Sound speed
        dt: Time step
        grid_spacings: List of grid spacings [dx, dy, ...] or [dx, dy, dz] for 3D
                       Length must match dimension
        dimension: Integer dimension (1, 2, or 3)
        gravity: Whether to include gravity term (default: True)
    
    Returns:
        List of updated momentum tensors (same shape as input momenta)
    
    Examples:
        # 1D
        P_new = compute_lax_momentum_update_torch([Px], [vx], rho, phi, c_s, dt, [dx], 1, gravity=True)
        
        # 2D
        Px_new, Py_new = compute_lax_momentum_update_torch([Px, Py], [vx, vy], rho, phi, c_s, dt, [dx, dy], 2, gravity=True)
        
        # 3D
        Px_new, Py_new, Pz_new = compute_lax_momentum_update_torch([Px, Py, Pz], [vx, vy, vz], rho, phi, c_s, dt, [dx, dy, dz], 3, gravity=True)
    """
    if len(momenta) != dimension:
        raise ValueError(f"Number of momentum tensors ({len(momenta)}) must match dimension ({dimension})")
    if len(velocities) != dimension:
        raise ValueError(f"Number of velocity tensors ({len(velocities)}) must match dimension ({dimension})")
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
        P_new = torch.zeros_like(P)
        
        # Sum all rolled neighbors (forward and backward for each dimension)
        for dim in range(dimension):
            P_new += torch.roll(P, -1, dims=dim)  # Forward roll
            P_new += torch.roll(P, 1, dims=dim)   # Backward roll
        
        # Normalize by 1/(2*dimension)
        P_new = P_new / (2 * dimension)
        
        # Advection terms: For each dimension, subtract mu_i * (flux_forward - flux_backward)
        # where flux = momentum * velocity
        # Each momentum component is advected by ALL velocity components
        for dim in range(dimension):
            mu = mu_values[dim]
            vel = velocities[dim]
            
            # Forward flux: P_rolled_forward * vel_rolled_forward
            P_forward = torch.roll(P, -1, dims=dim)
            vel_forward = torch.roll(vel, -1, dims=dim)
            flux_forward = P_forward * vel_forward
            
            # Backward flux: P_rolled_backward * vel_rolled_backward
            P_backward = torch.roll(P, 1, dims=dim)
            vel_backward = torch.roll(vel, 1, dims=dim)
            flux_backward = P_backward * vel_backward
            
            # Subtract the flux difference
            P_new -= mu * (flux_forward - flux_backward)
        
        # Pressure gradient term: Only in the component's own direction
        # -c_s^2 * mu_comp * (rho_forward - rho_backward) in direction comp_dim
        rho_forward = torch.roll(rho, -1, dims=comp_dim)
        rho_backward = torch.roll(rho, 1, dims=comp_dim)
        P_new -= (c_s**2) * mu_comp * (rho_forward - rho_backward)
        
        # Gravity term: Only in the component's own direction (if enabled)
        # -mu_comp * rho * (phi_forward - phi_backward) in direction comp_dim
        if gravity:
            phi_forward = torch.roll(phi, -1, dims=comp_dim)
            phi_backward = torch.roll(phi, 1, dims=comp_dim)
            P_new -= mu_comp * rho * (phi_forward - phi_backward)
        
        momenta_new.append(P_new)
    
    return momenta_new


def lax_time_step_torch(state_dict, dt, params_dict, dimension, gravity=True):
    """
    Perform a single LAX time step, updating all state variables (PyTorch version).
    
    This function orchestrates a complete time step by:
    1. Updating density using compute_lax_density_update_torch()
    2. Updating momenta using compute_lax_momentum_update_torch()
    3. Computing velocities from updated momenta and density
    4. Updating gravitational potential if gravity is enabled
    
    Args:
        state_dict: Dictionary containing current state with keys:
                   - 'rho': density tensor
                   - 'velocities': list of velocity tensors [vx, vy, ...] or [vx, vy, vz]
                   - 'momenta': list of momentum tensors [Px, Py, ...] or [Px, Py, Pz]
                   - 'phi': gravitational potential tensor (used if gravity=True)
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
        new_state = lax_time_step_torch(state, dt, params, dimension=2, gravity=True)
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
    rho_new = compute_lax_density_update_torch(rho, velocities, dt, grid_spacings, dimension)
    
    # Step 2: Update momenta
    momenta_new = compute_lax_momentum_update_torch(
        momenta, velocities, rho, phi, c_s, dt, grid_spacings, dimension, gravity=gravity
    )
    
    # Step 3: Compute velocities from updated momenta and density
    velocities_new = [momenta_new[i] / rho_new for i in range(dimension)]
    
    # Step 4: Update gravitational potential if gravity is enabled
    if gravity:
        # Call appropriate fft solver based on dimension
        if dimension == 1:
            # 1D Poisson solver (not implemented in fft_solver_torch, but we can handle it)
            # For now, skip it for 1D
            phi_new = torch.zeros_like(rho_new)  # 1D gravity not typically used with this solver
        elif dimension == 2:
            phi_new = fft_solver_torch(const * (rho_new - rho_o), Lx, nx, Ly, ny)
        else:  # dimension == 3
            phi_new = fft_solver_torch_3d(const * (rho_new - rho_o), Lx, nx, Ly, ny, Lz, nz)
    else:
        # If no gravity, keep phi as zeros or copy existing if provided
        phi_new = phi.clone() if phi is not None else torch.zeros_like(rho_new)
    
    # Return new state dictionary
    return {
        'rho': rho_new,
        'velocities': velocities_new,
        'momenta': momenta_new,
        'phi': phi_new
    }


def initialize_arrays_torch(shape, dimension):
    """
    Initialize arrays for LAX solver (PyTorch version).
    
    Args:
        shape: Tuple of grid dimensions (nx, ny) or (nx, ny, nz)
        dimension: 2 or 3
    
    Returns:
        dict: {
            'rho': density tensor,
            'velocities': [vx, vy, ...] list of velocity tensors,
            'momenta': [Px, Py, ...] list of momentum tensors,
            'phi': potential tensor
        }
    """
    if dimension == 2:
        nx, ny = shape
        arrays = {
            'rho': torch.zeros((nx, ny), device=device, dtype=dtype),
            'velocities': [torch.zeros((nx, ny), device=device, dtype=dtype), 
                          torch.zeros((nx, ny), device=device, dtype=dtype)],  # vx, vy
            'momenta': [torch.zeros((nx, ny), device=device, dtype=dtype), 
                       torch.zeros((nx, ny), device=device, dtype=dtype)],  # Px, Py
            'phi': torch.zeros((nx, ny), device=device, dtype=dtype)
        }
    elif dimension == 3:
        nx, ny, nz = shape
        arrays = {
            'rho': torch.zeros((nx, ny, nz), device=device, dtype=dtype),
            'velocities': [torch.zeros((nx, ny, nz), device=device, dtype=dtype),
                          torch.zeros((nx, ny, nz), device=device, dtype=dtype),
                          torch.zeros((nx, ny, nz), device=device, dtype=dtype)],  # vx, vy, vz
            'momenta': [torch.zeros((nx, ny, nz), device=device, dtype=dtype),
                       torch.zeros((nx, ny, nz), device=device, dtype=dtype),
                       torch.zeros((nx, ny, nz), device=device, dtype=dtype)],  # Px, Py, Pz
            'phi': torch.zeros((nx, ny, nz), device=device, dtype=dtype)
        }
    else:
        raise ValueError(f"Unsupported dimension={dimension}. Use 2 or 3.")
    
    return arrays


# ============================================================================
# Initial Conditions Setup Functions
# ============================================================================

def setup_sinusoidal_ic_torch(domain_params, physics_params, dimension, xx=None, yy=None, zz=None):
    """
    Set up sinusoidal initial conditions with KX, KY, KZ wave patterns (PyTorch version).
    
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
        x = torch.linspace(0, Lx, nx+1, device=device, dtype=dtype)[:-1]
        y = torch.linspace(0, Ly, ny+1, device=device, dtype=dtype)[:-1]
        if dimension == 2:
            xx, yy = torch.meshgrid(x, y, indexing='ij')
        else:
            z = torch.linspace(0, Lz, nz+1, device=device, dtype=dtype)[:-1]
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    
    # Set up density
    KX_tensor = torch.tensor(KX, device=device, dtype=dtype)
    KY_tensor = torch.tensor(KY, device=device, dtype=dtype)
    if dimension == 2:
        rho = rho_o + rho_1 * torch.cos(KX_tensor * xx + KY_tensor * yy)
    else:
        KZ_tensor = torch.tensor(KZ, device=device, dtype=dtype)
        rho = rho_o + rho_1 * torch.cos(KX_tensor * xx + KY_tensor * yy + KZ_tensor * zz)
    
    # Compute velocity perturbation
    v_1, alpha, is_unstable = compute_perturbation_velocity_torch(rho_1, rho_o, lam, c_s, jeans, gravity)
    
    # Set up velocity fields
    if dimension == 3:
        KZ_tensor = torch.tensor(KZ, device=device, dtype=dtype)
        k_magnitude = torch.sqrt(KX_tensor**2 + KY_tensor**2 + KZ_tensor**2)
    else:
        k_magnitude = torch.sqrt(KX_tensor**2 + KY_tensor**2)
    
    if dimension == 2:
        if gravity and is_unstable:
            # Unstable: use sin
            wave_field = -v_1 * torch.sin(KX_tensor * xx + KY_tensor * yy)
        else:
            # Stable or no gravity: use cos
            wave_field = v_1 * torch.cos(KX_tensor * xx + KY_tensor * yy)
        
        if k_magnitude > 0:
            vx = wave_field * (KX_tensor / k_magnitude)
            vy = wave_field * (KY_tensor / k_magnitude)
        else:
            vx = wave_field
            vy = torch.zeros_like(xx)
        
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
            wave_field = -v_1 * torch.sin(KX_tensor * xx + KY_tensor * yy + KZ_tensor * zz)
        else:
            wave_field = v_1 * torch.cos(KX_tensor * xx + KY_tensor * yy + KZ_tensor * zz)
        
        if k_magnitude > 0:
            vx = wave_field * (KX_tensor / k_magnitude)
            vy = wave_field * (KY_tensor / k_magnitude)
            vz = wave_field * (KZ_tensor / k_magnitude)
        else:
            vx = wave_field
            vy = torch.zeros_like(vx)
            vz = torch.zeros_like(vx)
        
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


def setup_power_spectrum_ic_torch(domain_params, ps_params, dimension, vx0_shared=None, vy0_shared=None, vz0_shared=None):
    """
    Set up power spectrum initial conditions (PyTorch version).
    
    Args:
        domain_params: DomainParams dataclass or dict with Lx, Ly, (Lz), nx, ny, (nz)
        ps_params: Dict with rho_o, power_index, amplitude, random_seed
        dimension: 2 or 3
        vx0_shared, vy0_shared, vz0_shared: Optional pre-generated velocity fields (numpy arrays)
    
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
        raise ValueError(f"setup_power_spectrum_ic_torch() only supports dimension=2 or 3, got {dimension}")
    
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
        rho = rho_o * torch.ones((nx, ny), device=device, dtype=dtype)
    else:
        rho = rho_o * torch.ones((nx, ny, nz), device=device, dtype=dtype)
    
    # Generate or use shared velocity fields
    if vx0_shared is not None and vy0_shared is not None:
        vx = torch.from_numpy(vx0_shared.copy()).to(device=device, dtype=dtype)
        vy = torch.from_numpy(vy0_shared.copy()).to(device=device, dtype=dtype)
        if dimension == 2:
            return {'rho': rho, 'vx': vx, 'vy': vy}
        else:
            if vz0_shared is not None:
                vz = torch.from_numpy(vz0_shared.copy()).to(device=device, dtype=dtype)
                return {'rho': rho, 'vx': vx, 'vy': vy, 'vz': vz}
    
    # Generate new velocity fields
    if dimension == 2:
        vx, vy = generate_velocity_field_power_spectrum_torch(nx, ny, Lx, Ly, power_index, amplitude, DIMENSION=2, random_seed=random_seed)
        return {'rho': rho, 'vx': vx, 'vy': vy}
    else:
        vx, vy, vz = generate_velocity_field_power_spectrum_torch(nx, ny, Lx, Ly, power_index, amplitude, DIMENSION=3, random_seed=random_seed, nz=nz, Lz=Lz)
        return {'rho': rho, 'vx': vx, 'vy': vy, 'vz': vz}


def setup_warm_start_ic_torch(provided_fields):
    """
    Set up initial conditions from provided fields (for PINN integration, PyTorch version).
    
    Args:
        provided_fields: Dict with 'rho', 'vx', 'vy', ('vz' for 3D) - can be numpy arrays or torch tensors
    
    Returns:
        dict: Same structure as input, with arrays converted to torch tensors
    """
    result = {}
    for key, value in provided_fields.items():
        if value is not None:
            if isinstance(value, torch.Tensor):
                result[key] = value.clone()
            elif isinstance(value, np.ndarray):
                result[key] = torch.from_numpy(value.copy()).to(device=device, dtype=dtype)
            else:
                result[key] = value
        else:
            result[key] = None
    
    return result


def lax_solver_torch(time, domain_params, physics_params, ic_type='sinusoidal', ic_params=None, options=None, save_times=None):
    """
    Unified master solver for 1D, 2D, and 3D LAX method (PyTorch version).
    
    This function provides a single entry point for running LAX simulations with different
    initial condition types and options. It automatically determines the dimension from
    domain_params and handles all setup and time-stepping internally.
    
    Args:
        time: Final simulation time
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
        save_times: Optional list of times to save snapshots during integration.
                   If provided, returns dict {time: SimulationResult} instead of single SimulationResult.
                   If None, returns single SimulationResult at final time.
    
    Returns:
        If save_times is None: SimulationResult at final time
        If save_times is provided: Dict mapping time -> SimulationResult for each saved time
    
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
    
    # Validate save_times
    if save_times is not None:
        if not isinstance(save_times, (list, tuple, np.ndarray)):
            raise ValueError(f"save_times must be a list, tuple, or numpy array, got {type(save_times)}")
        save_times = np.array(save_times)
        if len(save_times) == 0:
            raise ValueError("save_times cannot be empty")
        if np.any(save_times < 0):
            raise ValueError("save_times must contain non-negative values")
        if np.any(save_times > time):
            raise ValueError(f"save_times contains values greater than final time={time}")
        save_times = np.sort(np.unique(save_times))  # Sort and remove duplicates
        snapshots = {}  # Dictionary to store snapshots
    
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
        jeans = compute_jeans_length_torch(c_s, rho_o, const, G)
    
    # Create coordinate arrays first
    x = torch.linspace(0, Lx, nx+1, device=device, dtype=dtype)[:-1]
    if dimension >= 2:
        y = torch.linspace(0, Ly, ny+1, device=device, dtype=dtype)[:-1]
    if dimension == 3:
        z = torch.linspace(0, Lz, nz+1, device=device, dtype=dtype)[:-1]
    
    # Set up initial conditions based on ic_type
    if ic_type == 'sinusoidal':
        rho_1 = physics_params.get('rho_1', 0.1)
        lam = physics_params.get('lam', 7.0)
        
        if dimension == 1:
            # 1D sinusoidal IC
            rho0 = rho_o + rho_1 * torch.cos(2*np.pi*x/lam)
            v_1, alpha, is_unstable = compute_perturbation_velocity_torch(rho_1, rho_o, lam, c_s, jeans, gravity)
            if gravity and is_unstable:
                vx0 = -v_1 * torch.sin(2*np.pi*x/lam)
            else:
                vx0 = v_1 * torch.cos(2*np.pi*x/lam)
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
            
            ic_result = setup_sinusoidal_ic_torch(domain_params, setup_physics, dimension)
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
            rho0 = rho_o * torch.ones(nx, device=device, dtype=dtype)
            vx0 = torch.zeros(nx, device=device, dtype=dtype)
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
            
            ic_result = setup_power_spectrum_ic_torch(domain_params, ps_params, dimension, 
                                                     vx0_shared, vy0_shared, vz0_shared)
            rho0 = ic_result['rho']
            if dimension == 2:
                vx0 = ic_result['vx']
                vy0 = ic_result['vy']
                velocities = [vx0, vy0]
            else:  # dimension == 3
                # Validate that all 3D velocity components are present
                if 'vz' not in ic_result:
                    raise ValueError(f"setup_power_spectrum_ic_torch() did not return 'vz' for dimension=3. "
                                   f"Result keys: {list(ic_result.keys())}")
                vx0 = ic_result['vx']
                vy0 = ic_result['vy']
                vz0 = ic_result['vz']
                velocities = [vx0, vy0, vz0]
    
    elif ic_type == 'warm_start':
        ic_result = setup_warm_start_ic_torch(ic_params['provided_fields'])
        rho0 = ic_result['rho']
        if dimension == 1:
            vx0 = ic_result.get('vx', torch.zeros_like(rho0))
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
            k = 2*np.pi*torch.fft.fftfreq(nx, d=dx).to(device)
            rhohat = torch.fft.fft(const*(rho0 - rho_o))
            denom = -(k**2)
            denom[0] = 1.0
            phihat = rhohat / denom
            phihat[0] = 0.0
            phi0 = torch.real(torch.fft.ifft(phihat))
        elif dimension == 2:
            phi0 = fft_solver_torch(const*(rho0 - rho_o), Lx, nx, Ly, ny)
        else:  # dimension == 3
            phi0 = fft_solver_torch_3d(const*(rho0 - rho_o), Lx, nx, Ly, ny, Lz, nz)
    else:
        phi0 = torch.zeros_like(rho0)
    
    
    # Prepare params_dict for lax_time_step_torch
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
    dt = compute_adaptive_timestep_torch(velocities, c_s, dx, nu, include_sound_speed=True)
    
    # Save initial state if t=0 is in save_times
    if save_times is not None and len(save_times) > 0 and save_times[0] == 0.0:
        # Convert initial state to SimulationResult
        density_np = state['rho'].cpu().numpy()
        velocity_components_np = [v.cpu().numpy() for v in state['velocities']]
        coordinates = {'x': x.cpu().numpy()}
        if dimension >= 2:
            coordinates['y'] = y.cpu().numpy()
        if dimension == 3:
            coordinates['z'] = z.cpu().numpy()
        potential_np = state['phi'].cpu().numpy() if gravity else None
        snapshots[0.0] = SimulationResult(
            dimension=dimension,
            density=density_np,
            velocity_components=velocity_components_np,
            coordinates=coordinates,
            metadata={'time': 0.0, 'iterations': 0, 'rho_max': torch.max(state['rho']).item(),
                     'gravity': gravity, 'nu': nu, 'dimension': dimension},
            potential=potential_np,
            linear_theory_comparison=None
        )
    
    while t < time:
        # Ensure last step doesn't overshoot
        if t + dt > time:
            dt = time - t
        
        # Perform one time step
        state = lax_time_step_torch(state, dt, params_dict, dimension, gravity=gravity)
        
        # Update time and step counter
        t += dt
        n_steps += 1
        
        # Check if we should save a snapshot after this step
        if save_times is not None:
            for save_t in save_times:
                # Save if we've passed this time point (with small tolerance for floating point)
                if abs(t - save_t) < 1e-9 or (t > save_t and save_t not in snapshots):
                    # Save current state after stepping
                    density_np = state['rho'].cpu().numpy()
                    velocity_components_np = [v.cpu().numpy() for v in state['velocities']]
                    coordinates = {'x': x.cpu().numpy()}
                    if dimension >= 2:
                        coordinates['y'] = y.cpu().numpy()
                    if dimension == 3:
                        coordinates['z'] = z.cpu().numpy()
                    potential_np = state['phi'].cpu().numpy() if gravity else None
                    snapshots[save_t] = SimulationResult(
                        dimension=dimension,
                        density=density_np,
                        velocity_components=velocity_components_np,
                        coordinates=coordinates,
                        metadata={'time': t, 'iterations': n_steps, 'rho_max': torch.max(state['rho']).item(),
                                 'gravity': gravity, 'nu': nu, 'dimension': dimension},
                        potential=potential_np,
                        linear_theory_comparison=None
                    )
        
        # Calculate dt for next step
        dt = compute_adaptive_timestep_torch(state['velocities'], c_s, dx, nu)
    
    # Convert tensors to numpy for return
    density_np = state['rho'].cpu().numpy()
    velocity_components_np = [v.cpu().numpy() for v in state['velocities']]
    
    # Prepare coordinates dictionary
    coordinates = {'x': x.cpu().numpy()}
    if dimension >= 2:
        coordinates['y'] = y.cpu().numpy()
    if dimension == 3:
        coordinates['z'] = z.cpu().numpy()
    
    # Prepare metadata dictionary
    metadata = {
        'time': t,
        'iterations': n_steps,
        'rho_max': torch.max(state['rho']).item(),
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
    
    # Prepare potential (convert to numpy if present)
    potential_np = None
    if gravity:
        potential_np = state['phi'].cpu().numpy()
    
    # Create final SimulationResult
    final_result = SimulationResult(
        dimension=dimension,
        density=density_np,
        velocity_components=velocity_components_np,
        coordinates=coordinates,
        metadata=metadata,
        potential=potential_np,
        linear_theory_comparison=linear_theory_comparison
    )
    
    # Return snapshots dict if save_times was provided, otherwise return single result
    if save_times is not None:
        # Ensure final time is saved if it's in save_times
        if time in save_times and time not in snapshots:
            snapshots[time] = final_result
        # Also ensure we have the final result even if not explicitly requested
        if time not in snapshots:
            snapshots[time] = final_result
        return snapshots
    else:
        return final_result


def fft_solver_torch(rho, Lx, nx, Ly, ny):
    """
    PyTorch FFT solver for Poisson equation (gravitational potential).
    """
    dx = Lx / nx
    dy = Ly / ny
    
    # Calculate the Fourier modes of the gas density
    rhohat = torch.fft.fft2(rho)
    
    # Calculate the wave numbers in x and y directions
    kx = 2 * np.pi * torch.fft.fftfreq(nx, d=dx).to(device)
    ky = 2 * np.pi * torch.fft.fftfreq(ny, d=dy).to(device)
    
    # Construct the discrete Laplacian operator in Fourier space
    # This ensures consistency with finite-difference gradients used in LAX scheme
    kx_mesh, ky_mesh = torch.meshgrid(kx, ky, indexing='xy')
    # Transpose to match FFT2 output layout (nx, ny)
    laplace = (2*(torch.cos(kx_mesh.T*dx)-1)/(dx**2) + 
               2*(torch.cos(ky_mesh.T*dy)-1)/(dy**2))
    
    # Handle zero mode (k=0) - set to small value to avoid division by zero
    laplace = torch.where(laplace == 0, torch.tensor(1e-9, device=device, dtype=dtype), laplace)
    
    # Solve for the potential in Fourier space
    phihat = rhohat / laplace
    
    # Transform back to real space
    phi = torch.real(torch.fft.ifft2(phihat))
    
    return phi

def fft_solver_torch_3d(rho, Lx, nx, Ly, ny, Lz, nz):
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz
    rhohat = torch.fft.fftn(rho)
    kx = 2 * np.pi * torch.fft.fftfreq(nx, d=dx).to(device)
    ky = 2 * np.pi * torch.fft.fftfreq(ny, d=dy).to(device)
    kz = 2 * np.pi * torch.fft.fftfreq(nz, d=dz).to(device)
    kx_mesh, ky_mesh, kz_mesh = torch.meshgrid(kx, ky, kz, indexing='ij')
    # Use discrete Laplacian for consistency with finite-difference scheme
    laplace = (2*(torch.cos(kx_mesh*dx)-1)/(dx**2) + 
               2*(torch.cos(ky_mesh*dy)-1)/(dy**2) + 
               2*(torch.cos(kz_mesh*dz)-1)/(dz**2))
    laplace = torch.where(laplace == 0, torch.tensor(1e-9, device=device, dtype=dtype), laplace)
    phihat = rhohat / laplace
    phi = torch.real(torch.fft.ifftn(phihat))
    return phi

def generate_velocity_field_power_spectrum_torch(nx, ny, Lx, Ly, power_index=-3.0, amplitude=0.02, DIMENSION=2, random_seed=None, nz=None, Lz=None):
    """
    PyTorch implementation for generating 2D or 3D velocity fields with a power-law spectrum.
    Dimension-agnostic implementation using fftn/ifftn.
    
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
    # Use NumPy's random generator for consistency with CPU solver
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
        # 1. Generate random field at target resolution using NumPy (for consistency)
        field_np = rng.standard_normal(shape)
        field = torch.from_numpy(field_np).to(device=device, dtype=dtype)
        F = torch.fft.fftn(field)  # fftn works for any dimension
        
        # 2. Construct k-space grid at target resolution
        k_grid = build_k_space_grid_torch(shape, domain_lengths, DIMENSION)
        
        # 3. Compute |k| magnitude (dimension-agnostic)
        if DIMENSION == 2:
            kx, ky, kxg, kyg = k_grid
            kk = torch.sqrt(kxg**2 + kyg**2)
        else:  # DIMENSION == 3
            kx, ky, kz, kxg, kyg, kzg = k_grid
            kk = torch.sqrt(kxg**2 + kyg**2 + kzg**2)
        
        # 4. Apply power-law filter (avoid computing power when kk == 0)
        filt = torch.zeros_like(kk)
        mask = kk > 0
        filt[mask] = kk[mask]**(power_index / 2.0)
        
        F_filtered = F * filt
        
        # 5. Transform back to real space (dimension-agnostic)
        comp = torch.real(torch.fft.ifftn(F_filtered))
        
        # 6. Normalize the field
        comp -= torch.mean(comp)
        std = torch.std(comp)
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

def lax_solution_torch(time_val, N, nu, lam, num_of_waves, rho_1, gravity=False, use_velocity_ps=False, 
                         ps_index=-3.0, vel_rms=0.02, random_seed=None, vx0_shared=None, vy0_shared=None):
    """
    DEPRECATED: This function is a wrapper around the unified lax_solver_torch().
    Use lax_solver_torch() directly for new code.
    
    PyTorch implementation of the LAX method for solving hydrodynamic equations.
    This version is designed to run on a GPU for accelerated computation.
    
    Args:
        vx0_shared: Optional pre-generated vx velocity field (numpy array) for consistent ICs
        vy0_shared: Optional pre-generated vy velocity field (numpy array) for consistent ICs
    """
    import warnings
    warnings.warn(
        "lax_solution_torch() is deprecated. Use lax_solver_torch() instead. "
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
        'comparison': False,
        'isplot': False
    }
    
    # Call unified solver
    result = lax_solver_torch(time_val, domain_params, physics_params, ic_type=ic_type, ic_params=ic_params, options=options)
    
    # Extract values to match old return format
    x = result.coordinates['x']
    y = result.coordinates['y']
    rho0 = result.density
    vx0, vy0 = result.velocity_components
    n = result.metadata['iterations']
    rho_max = result.metadata['rho_max']
    
    return x, rho0, vx0, vy0, None, n, rho_max


def lax_solution_3d_sinusoidal_torch(time_val, N, nu, lam, num_of_waves, rho_1, gravity=True,
                                     use_velocity_ps=False, ps_index=-3.0, vel_rms=0.02, 
                                     random_seed=None, vx0_shared=None, vy0_shared=None, vz0_shared=None):
    """
    DEPRECATED: This function is a wrapper around the unified lax_solver_torch().
    Use lax_solver_torch() with dimension=3 instead.
    
    3D LAX solver for sinusoidal perturbations with optional self-gravity (PyTorch version).
    Returns full 3D fields which can be sliced for visualization/comparison.
    
    Args:
        time_val: Final simulation time
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
        vx0_shared: Optional pre-generated x-velocity field (numpy array, default: None)
        vy0_shared: Optional pre-generated y-velocity field (numpy array, default: None)
        vz0_shared: Optional pre-generated z-velocity field (numpy array, default: None)
    """
    import warnings
    warnings.warn(
        "lax_solution_3d_sinusoidal_torch() is deprecated. Use lax_solver_torch() with dimension=3 instead. "
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
    result = lax_solver_torch(time_val, domain_params, physics_params, ic_type=ic_type, ic_params=ic_params, options=options)
    
    # Extract values to match old return format
    x = result.coordinates['x']
    y = result.coordinates['y']
    z = result.coordinates['z']
    rho0 = result.density
    vx0, vy0, vz0 = result.velocity_components
    phi0 = result.potential
    k_iter = result.metadata['iterations']
    rho_max = result.metadata['rho_max']
    
    return (x, y, z, rho0, vx0, vy0, vz0, phi0, k_iter, rho_max)

def lax_solution_warm_start_torch(rho_ic, vx_ic, vy_ic, x_grid, y_grid, 
                                   t_start, t_end, nu=0.5, save_times=None, gravity=True):
    """
    PyTorch implementation: Run FD solver from custom initial conditions (warm-start).
    
    This function allows restarting the FD solver from a PINN state or any custom state,
    enabling efficient generation of FD data for hybrid PINN-FD training.
    
    Args:
        rho_ic: Initial density field (Nx, Ny) - can be numpy array or torch tensor
        vx_ic: Initial x-velocity field (Nx, Ny) - can be numpy array or torch tensor
        vy_ic: Initial y-velocity field (Nx, Ny) - can be numpy array or torch tensor
        x_grid: x coordinates (Nx,) - can be numpy array or torch tensor
        y_grid: y coordinates (Ny,) - can be numpy array or torch tensor
        t_start: Starting time
        t_end: Ending time
        nu: Courant number
        save_times: List of times to save snapshots [default: [t_end]]
        gravity: Whether to include self-gravity (default: True)
    
    Returns:
        Dictionary: {time: (rho, vx, vy, phi, x, y)} for each saved time (all as numpy arrays)
    """
    if save_times is None:
        save_times = [t_end]
    
    # Domain setup (get shape from original input, before any conversion)
    if isinstance(rho_ic, np.ndarray):
        Nx, Ny = rho_ic.shape
    else:
        Nx, Ny = rho_ic.shape
    
    # Convert coordinate grids to torch tensors if needed (for domain setup and later use)
    if isinstance(x_grid, np.ndarray):
        x_grid = torch.from_numpy(x_grid).to(device=device, dtype=dtype)
    elif not isinstance(x_grid, torch.Tensor):
        x_grid = torch.tensor(x_grid, device=device, dtype=dtype)
    if isinstance(y_grid, np.ndarray):
        y_grid = torch.from_numpy(y_grid).to(device=device, dtype=dtype)
    elif not isinstance(y_grid, torch.Tensor):
        y_grid = torch.tensor(y_grid, device=device, dtype=dtype)
    
    Lx = (x_grid[-1] - x_grid[0] + (x_grid[1] - x_grid[0])).item()  # Approximate domain size
    Ly = (y_grid[-1] - y_grid[0] + (y_grid[1] - y_grid[0])).item()
    dx = Lx / Nx
    dy = Ly / Ny
    
    # Physical constants
    c_s = cs
    
    # Use setup_warm_start_ic_torch() helper to set up initial conditions
    # Convert inputs to torch tensors if needed (setup_warm_start_ic_torch handles this)
    provided_fields = {
        'rho': rho_ic,
        'vx': vx_ic,
        'vy': vy_ic
    }
    ic_result = setup_warm_start_ic_torch(provided_fields)
    rho0 = ic_result['rho']
    vx0 = ic_result['vx']
    vy0 = ic_result['vy']
    
    # Calculate initial potential (gravity is always True for collapse problems)
    phi0 = fft_solver_torch(const * (rho0 - rho_o), Lx, Nx, Ly, Ny)
    
    # Initialize flux terms
    Px0 = rho0 * vx0
    Py0 = rho0 * vy0
    
    # Storage for snapshots
    snapshots = {}
    
    # Time-stepping loop
    t = t_start
    k = 0
    
    # Initial dt
    dt = compute_adaptive_timestep_torch([vx0, vy0], c_s, dx, nu, include_sound_speed=True)
    
    while t < t_end:
        # Check if we should save a snapshot before this step
        for save_t in save_times:
            if t <= save_t < t + dt and save_t not in snapshots:
                # Save current state (convert to numpy for consistency)
                snapshots[save_t] = (
                    rho0.cpu().numpy().copy(),
                    vx0.cpu().numpy().copy(),
                    vy0.cpu().numpy().copy(),
                    phi0.cpu().numpy().copy(),
                    x_grid.cpu().numpy().copy(),
                    y_grid.cpu().numpy().copy()
                )
        
        # Ensure last step doesn't overshoot
        if t + dt > t_end:
            dt = t_end - t
        
        # LAX time-stepping
        mux = dt / (2 * dx)
        muy = dt / (2 * dy)
        
        # Update density
        rho1 = (0.25) * (torch.roll(rho0, -1, dims=0) + torch.roll(rho0, 1, dims=0) +
                        torch.roll(rho0, -1, dims=1) + torch.roll(rho0, 1, dims=1)) - \
               (mux * (torch.roll(rho0, -1, dims=0) * torch.roll(vx0, -1, dims=0) -
                       torch.roll(rho0, 1, dims=0) * torch.roll(vx0, 1, dims=0))) - \
               (muy * (torch.roll(rho0, -1, dims=1) * torch.roll(vy0, -1, dims=1) -
                       torch.roll(rho0, 1, dims=1) * torch.roll(vy0, 1, dims=1)))
        
        # Update momentum (with gravity)
        if gravity:
            Px1 = (0.25) * (torch.roll(Px0, -1, dims=0) + torch.roll(Px0, 1, dims=0) +
                            torch.roll(Px0, -1, dims=1) + torch.roll(Px0, 1, dims=1)) - \
                  (mux * (torch.roll(Px0, -1, dims=0) * torch.roll(vx0, -1, dims=0) -
                          torch.roll(Px0, 1, dims=0) * torch.roll(vx0, 1, dims=0))) - \
                  (muy * (torch.roll(Px0, -1, dims=1) * torch.roll(vy0, -1, dims=1) -
                          torch.roll(Px0, 1, dims=1) * torch.roll(vy0, 1, dims=1))) - \
                  ((c_s**2) * mux * (torch.roll(rho0, -1, dims=0) - torch.roll(rho0, 1, dims=0))) - \
                  (mux * rho0 * (torch.roll(phi0, -1, dims=0) - torch.roll(phi0, 1, dims=0)))
            
            Py1 = (0.25) * (torch.roll(Py0, -1, dims=0) + torch.roll(Py0, 1, dims=0) +
                            torch.roll(Py0, -1, dims=1) + torch.roll(Py0, 1, dims=1)) - \
                  (muy * (torch.roll(Py0, -1, dims=1) * torch.roll(vy0, -1, dims=1) -
                          torch.roll(Py0, 1, dims=1) * torch.roll(vy0, 1, dims=1))) - \
                  (mux * (torch.roll(Py0, -1, dims=0) * torch.roll(vx0, -1, dims=0) -
                          torch.roll(Py0, 1, dims=0) * torch.roll(vx0, 1, dims=0))) - \
                  ((c_s**2) * muy * (torch.roll(rho0, -1, dims=1) - torch.roll(rho0, 1, dims=1))) - \
                  (muy * rho0 * (torch.roll(phi0, -1, dims=1) - torch.roll(phi0, 1, dims=1)))
            
            # Update potential
            phi1 = fft_solver_torch(const * (rho1 - rho_o), Lx, Nx, Ly, Ny)
        else:
            # Without gravity (shouldn't happen for collapse problems, but included for completeness)
            Px1 = (0.25) * (torch.roll(Px0, -1, dims=0) + torch.roll(Px0, 1, dims=0) +
                            torch.roll(Px0, -1, dims=1) + torch.roll(Px0, 1, dims=1)) - \
                  (mux * (torch.roll(Px0, -1, dims=0) * torch.roll(vx0, -1, dims=0) -
                          torch.roll(Px0, 1, dims=0) * torch.roll(vx0, 1, dims=0))) - \
                  (muy * (torch.roll(Px0, -1, dims=1) * torch.roll(vy0, -1, dims=1) -
                          torch.roll(Px0, 1, dims=1) * torch.roll(vy0, 1, dims=1))) - \
                  ((c_s**2) * mux * (torch.roll(rho0, -1, dims=0) - torch.roll(rho0, 1, dims=0)))
            
            Py1 = (0.25) * (torch.roll(Py0, -1, dims=0) + torch.roll(Py0, 1, dims=0) +
                            torch.roll(Py0, -1, dims=1) + torch.roll(Py0, 1, dims=1)) - \
                  (muy * (torch.roll(Py0, -1, dims=1) * torch.roll(vy0, -1, dims=1) -
                          torch.roll(Py0, 1, dims=1) * torch.roll(vy0, 1, dims=1))) - \
                  (mux * (torch.roll(Py0, -1, dims=0) * torch.roll(vx0, -1, dims=0) -
                          torch.roll(Py0, 1, dims=0) * torch.roll(vx0, 1, dims=0))) - \
                  ((c_s**2) * muy * (torch.roll(rho0, -1, dims=1) - torch.roll(rho0, 1, dims=1)))
            
            phi1 = torch.zeros_like(rho1)
        
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
        vmax = max(torch.max(torch.abs(vx0)).item(), torch.max(torch.abs(vy0)).item())
        dt1 = nu * dx / vmax if vmax > 1e-9 else float('inf')
        dt2 = nu * dx / c_s
        dt = min(dt1, dt2)
    
    # Save final snapshot if not already saved
    if t_end not in snapshots:
        snapshots[t_end] = (
            rho0.cpu().numpy().copy(),
            vx0.cpu().numpy().copy(),
            vy0.cpu().numpy().copy(),
            phi0.cpu().numpy().copy(),
            x_grid.cpu().numpy().copy(),
            y_grid.cpu().numpy().copy()
        )
    
    return snapshots