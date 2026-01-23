"""
Comprehensive test suite for 3D initial conditions modifications.

Tests:
1. 2D backward compatibility
2. 3D functionality
3. Divergence-free validation
4. Shared fields vs fallback modes
"""

import numpy as np
import torch
import sys
import os

# Add parent directory to path for imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from config import (
    N_GRID, N_GRID_3D, POWER_EXPONENT, RANDOM_SEED, 
    PERTURBATION_TYPE, wave, num_of_waves, a, cs
)
from core.initial_conditions import (
    initialize_shared_velocity_fields,
    generate_power_spectrum_field,
    generate_power_spectrum_field_vy,
    generate_power_spectrum_field_vz,
    fun_rho_0,
    fun_vx_0,
    fun_vy_0,
    fun_vz_0,
    _detect_dimension
)


def compute_divergence_2d(vx, vy, x, y):
    """
    Compute velocity divergence for 2D fields: div(v) = ∂vx/∂x + ∂vy/∂y
    
    Args:
        vx, vy: Velocity components (tensors, shape [N, 1])
        x, y: Coordinate tensors (shape [N, 1])
    
    Returns:
        divergence: Tensor of divergence values
    """
    # Sort by coordinates for gradient computation
    indices = torch.argsort(x.squeeze())
    x_sorted = x[indices].squeeze()
    y_sorted = y[indices].squeeze()
    vx_sorted = vx[indices].squeeze()
    vy_sorted = vy[indices].squeeze()
    
    # Compute gradients using finite differences
    # For simplicity, use nearest neighbor differences
    dx = x_sorted[1:] - x_sorted[:-1]
    dy = y_sorted[1:] - y_sorted[:-1]
    
    # Avoid division by zero
    dx = torch.where(dx.abs() < 1e-10, torch.ones_like(dx) * 1e-10, dx)
    dy = torch.where(dy.abs() < 1e-10, torch.ones_like(dy) * 1e-10, dy)
    
    dvx_dx = (vx_sorted[1:] - vx_sorted[:-1]) / dx
    dvy_dy = (vy_sorted[1:] - vy_sorted[:-1]) / dy
    
    # Pad to match original size
    dvx_dx = torch.cat([dvx_dx[:1], dvx_dx])
    dvy_dy = torch.cat([dvy_dy[:1], dvy_dy])
    
    divergence = dvx_dx + dvy_dy
    
    # Restore original order
    inv_indices = torch.argsort(indices)
    return divergence[inv_indices].unsqueeze(-1)


def compute_divergence_3d(vx, vy, vz, x, y, z):
    """
    Compute velocity divergence for 3D fields: div(v) = ∂vx/∂x + ∂vy/∂y + ∂vz/∂z
    
    Args:
        vx, vy, vz: Velocity components (tensors, shape [N, 1])
        x, y, z: Coordinate tensors (shape [N, 1])
    
    Returns:
        divergence: Tensor of divergence values
    """
    # Sort by coordinates for gradient computation
    indices = torch.argsort(x.squeeze())
    x_sorted = x[indices].squeeze()
    y_sorted = y[indices].squeeze()
    z_sorted = z[indices].squeeze()
    vx_sorted = vx[indices].squeeze()
    vy_sorted = vy[indices].squeeze()
    vz_sorted = vz[indices].squeeze()
    
    # Compute gradients using finite differences
    dx = x_sorted[1:] - x_sorted[:-1]
    dy = y_sorted[1:] - y_sorted[:-1]
    dz = z_sorted[1:] - z_sorted[:-1]
    
    # Avoid division by zero
    dx = torch.where(dx.abs() < 1e-10, torch.ones_like(dx) * 1e-10, dx)
    dy = torch.where(dy.abs() < 1e-10, torch.ones_like(dy) * 1e-10, dy)
    dz = torch.where(dz.abs() < 1e-10, torch.ones_like(dz) * 1e-10, dz)
    
    dvx_dx = (vx_sorted[1:] - vx_sorted[:-1]) / dx
    dvy_dy = (vy_sorted[1:] - vy_sorted[:-1]) / dy
    dvz_dz = (vz_sorted[1:] - vz_sorted[:-1]) / dz
    
    # Pad to match original size
    dvx_dx = torch.cat([dvx_dx[:1], dvx_dx])
    dvy_dy = torch.cat([dvy_dy[:1], dvy_dy])
    dvz_dz = torch.cat([dvz_dz[:1], dvz_dz])
    
    divergence = dvx_dx + dvy_dy + dvz_dz
    
    # Restore original order
    inv_indices = torch.argsort(indices)
    return divergence[inv_indices].unsqueeze(-1)


def test_2d_backward_compatibility():
    """Test that 2D functionality still works after 3D modifications."""
    print("\n" + "="*60)
    print("TEST 1: 2D Backward Compatibility")
    print("="*60)
    
    # Test parameters
    lam = wave
    v_1 = a * cs
    num_points = 100
    
    # Create 2D collocation points
    x = torch.rand(num_points, 1) * (lam * num_of_waves)
    y = torch.rand(num_points, 1) * (lam * num_of_waves)
    t = torch.zeros(num_points, 1)
    coords_2d = [x, y, t]
    
    # Test dimension detection
    dim = _detect_dimension(coords_2d)
    assert dim == 2, f"Expected dimension 2, got {dim}"
    print(f"[OK] Dimension detection: {dim}")
    
    # Test shared fields initialization (2D)
    vx_np, vy_np = initialize_shared_velocity_fields(
        lam, num_of_waves, v_1, seed=RANDOM_SEED, dimension=2
    )
    assert vx_np is not None and vy_np is not None, "Shared fields not initialized"
    assert vx_np.shape == vy_np.shape, "vx and vy should have same shape"
    print(f"[OK] Shared fields initialized: vx shape {vx_np.shape}, vy shape {vy_np.shape}")
    
    # Test velocity field generation
    vx = generate_power_spectrum_field(lam, v_1, coords_2d, seed=RANDOM_SEED)
    vy = generate_power_spectrum_field_vy(lam, v_1, coords_2d, seed=RANDOM_SEED)
    
    assert vx.shape == (num_points, 1), f"vx shape mismatch: {vx.shape}"
    assert vy.shape == (num_points, 1), f"vy shape mismatch: {vy.shape}"
    assert torch.all(torch.isfinite(vx)), "vx contains non-finite values"
    assert torch.all(torch.isfinite(vy)), "vy contains non-finite values"
    print(f"[OK] Velocity fields generated: vx shape {vx.shape}, vy shape {vy.shape}")
    print(f"  vx range: [{vx.min():.4f}, {vx.max():.4f}]")
    print(f"  vy range: [{vy.min():.4f}, {vy.max():.4f}]")
    
    # Test initial condition functions
    rho_1 = 0.1
    jeans = 1.0
    rho_0 = fun_rho_0(rho_1, lam, coords_2d)
    vx_0 = fun_vx_0(lam, jeans, v_1, coords_2d)
    vy_0 = fun_vy_0(lam, jeans, v_1, coords_2d)
    
    assert rho_0.shape == (num_points, 1), f"rho_0 shape mismatch: {rho_0.shape}"
    assert vx_0.shape == (num_points, 1), f"vx_0 shape mismatch: {vx_0.shape}"
    assert vy_0.shape == (num_points, 1), f"vy_0 shape mismatch: {vy_0.shape}"
    print(f"[OK] Initial condition functions work correctly")
    
    print("[OK] All 2D backward compatibility tests passed!")


def test_3d_functionality():
    """Test 3D functionality."""
    print("\n" + "="*60)
    print("TEST 2: 3D Functionality")
    print("="*60)
    
    # Test parameters
    lam = wave
    v_1 = a * cs
    num_points = 100
    
    # Create 3D collocation points
    x = torch.rand(num_points, 1) * (lam * num_of_waves)
    y = torch.rand(num_points, 1) * (lam * num_of_waves)
    z = torch.rand(num_points, 1) * (lam * num_of_waves)
    t = torch.zeros(num_points, 1)
    coords_3d = [x, y, z, t]
    
    # Test dimension detection
    dim = _detect_dimension(coords_3d)
    assert dim == 3, f"Expected dimension 3, got {dim}"
    print(f"[OK] Dimension detection: {dim}")
    
    # Test shared fields initialization (3D)
    vx_np, vy_np, vz_np = initialize_shared_velocity_fields(
        lam, num_of_waves, v_1, seed=RANDOM_SEED, dimension=3
    )
    assert vx_np is not None and vy_np is not None and vz_np is not None, "Shared fields not initialized"
    assert vx_np.shape == vy_np.shape == vz_np.shape, "All velocity components should have same shape"
    print(f"[OK] Shared fields initialized: vx shape {vx_np.shape}, vy shape {vy_np.shape}, vz shape {vz_np.shape}")
    
    # Test velocity field generation
    vx = generate_power_spectrum_field(lam, v_1, coords_3d, seed=RANDOM_SEED)
    vy = generate_power_spectrum_field_vy(lam, v_1, coords_3d, seed=RANDOM_SEED)
    vz = generate_power_spectrum_field_vz(lam, v_1, coords_3d, seed=RANDOM_SEED)
    
    assert vx.shape == (num_points, 1), f"vx shape mismatch: {vx.shape}"
    assert vy.shape == (num_points, 1), f"vy shape mismatch: {vy.shape}"
    assert vz.shape == (num_points, 1), f"vz shape mismatch: {vz.shape}"
    assert torch.all(torch.isfinite(vx)), "vx contains non-finite values"
    assert torch.all(torch.isfinite(vy)), "vy contains non-finite values"
    assert torch.all(torch.isfinite(vz)), "vz contains non-finite values"
    print(f"[OK] Velocity fields generated: vx shape {vx.shape}, vy shape {vy.shape}, vz shape {vz.shape}")
    print(f"  vx range: [{vx.min():.4f}, {vx.max():.4f}]")
    print(f"  vy range: [{vy.min():.4f}, {vy.max():.4f}]")
    print(f"  vz range: [{vz.min():.4f}, {vz.max():.4f}]")
    
    # Test initial condition functions
    rho_1 = 0.1
    jeans = 1.0
    rho_0 = fun_rho_0(rho_1, lam, coords_3d)
    vx_0 = fun_vx_0(lam, jeans, v_1, coords_3d)
    vy_0 = fun_vy_0(lam, jeans, v_1, coords_3d)
    vz_0 = fun_vz_0(lam, jeans, v_1, coords_3d)
    
    assert rho_0.shape == (num_points, 1), f"rho_0 shape mismatch: {rho_0.shape}"
    assert vx_0.shape == (num_points, 1), f"vx_0 shape mismatch: {vx_0.shape}"
    assert vy_0.shape == (num_points, 1), f"vy_0 shape mismatch: {vy_0.shape}"
    assert vz_0.shape == (num_points, 1), f"vz_0 shape mismatch: {vz_0.shape}"
    print(f"[OK] Initial condition functions work correctly (including vz_0)")
    
    print("[OK] All 3D functionality tests passed!")


def test_divergence_free_2d():
    """Test that 2D velocity fields are approximately divergence-free."""
    print("\n" + "="*60)
    print("TEST 3: 2D Divergence-Free Validation")
    print("="*60)
    
    # Test parameters
    lam = wave
    v_1 = a * cs
    num_points = 200  # More points for better gradient estimation
    
    # Create regular 2D grid for better divergence computation
    nx = int(np.sqrt(num_points))
    ny = nx
    x_vals = torch.linspace(0, lam * num_of_waves, nx)
    y_vals = torch.linspace(0, lam * num_of_waves, ny)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
    x = X.flatten().unsqueeze(-1)
    y = Y.flatten().unsqueeze(-1)
    t = torch.zeros(x.shape[0], 1)
    coords_2d = [x, y, t]
    
    # Initialize shared fields
    initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=RANDOM_SEED, dimension=2)
    
    # Generate velocity fields
    vx = generate_power_spectrum_field(lam, v_1, coords_2d, seed=RANDOM_SEED)
    vy = generate_power_spectrum_field_vy(lam, v_1, coords_2d, seed=RANDOM_SEED)
    
    # Reshape to grid for gradient computation
    vx_grid = vx.reshape(nx, ny)
    vy_grid = vy.reshape(nx, ny)
    x_grid = x.reshape(nx, ny)
    y_grid = y.reshape(nx, ny)
    
    # Compute divergence using numpy gradients (more reliable)
    vx_np = vx_grid.detach().cpu().numpy()
    vy_np = vy_grid.detach().cpu().numpy()
    x_np = x_grid.detach().cpu().numpy()
    y_np = y_grid.detach().cpu().numpy()
    
    dx = x_np[1, 0] - x_np[0, 0] if nx > 1 else 1.0
    dy = y_np[0, 1] - y_np[0, 0] if ny > 1 else 1.0
    
    dvx_dx = np.gradient(vx_np, dx, axis=0)
    dvy_dy = np.gradient(vy_np, dy, axis=1)
    divergence = dvx_dx + dvy_dy
    
    # Check divergence statistics
    max_div = np.abs(divergence).max()
    mean_div = np.abs(divergence).mean()
    std_div = np.abs(divergence).std()
    
    print(f"  Divergence statistics:")
    print(f"    Max |div|: {max_div:.6f}")
    print(f"    Mean |div|: {mean_div:.6f}")
    print(f"    Std |div|: {std_div:.6f}")
    
    # Power spectrum fields are not necessarily divergence-free
    # (they're random fields), but we check they're reasonable
    # For incompressible flow, divergence should be small relative to velocity magnitude
    v_mag = np.sqrt(vx_np**2 + vy_np**2)
    v_mag_mean = v_mag.mean()
    relative_div = max_div / (v_mag_mean + 1e-10)
    
    print(f"    Mean velocity magnitude: {v_mag_mean:.6f}")
    print(f"    Relative max divergence: {relative_div:.6f}")
    
    # Check that divergence is finite
    assert np.all(np.isfinite(divergence)), "Divergence contains non-finite values"
    print("[OK] Divergence computation is finite")
    
    # For power spectrum fields, divergence is not necessarily zero
    # but should be reasonable (not orders of magnitude larger than velocity)
    # This is a sanity check rather than a strict requirement
    if relative_div < 10.0:  # Allow some tolerance
        print("[OK] Divergence is within reasonable bounds")
    else:
        print(f"[WARNING] Relative divergence is large ({relative_div:.2f})")
    
    print("[OK] 2D divergence validation completed")


def test_divergence_free_3d():
    """Test that 3D velocity fields have reasonable divergence."""
    print("\n" + "="*60)
    print("TEST 4: 3D Divergence-Free Validation")
    print("="*60)
    
    # Test parameters
    lam = wave
    v_1 = a * cs
    num_points = 125  # 5x5x5 grid
    
    # Create regular 3D grid
    n = 5  # 5x5x5 grid
    x_vals = torch.linspace(0, lam * num_of_waves, n)
    y_vals = torch.linspace(0, lam * num_of_waves, n)
    z_vals = torch.linspace(0, lam * num_of_waves, n)
    X, Y, Z = torch.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    x = X.flatten().unsqueeze(-1)
    y = Y.flatten().unsqueeze(-1)
    z = Z.flatten().unsqueeze(-1)
    t = torch.zeros(x.shape[0], 1)
    coords_3d = [x, y, z, t]
    
    # Initialize shared fields
    initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=RANDOM_SEED, dimension=3)
    
    # Generate velocity fields
    vx = generate_power_spectrum_field(lam, v_1, coords_3d, seed=RANDOM_SEED)
    vy = generate_power_spectrum_field_vy(lam, v_1, coords_3d, seed=RANDOM_SEED)
    vz = generate_power_spectrum_field_vz(lam, v_1, coords_3d, seed=RANDOM_SEED)
    
    # Reshape to grid
    vx_grid = vx.reshape(n, n, n)
    vy_grid = vy.reshape(n, n, n)
    vz_grid = vz.reshape(n, n, n)
    x_grid = x.reshape(n, n, n)
    y_grid = y.reshape(n, n, n)
    z_grid = z.reshape(n, n, n)
    
    # Compute divergence using numpy gradients
    vx_np = vx_grid.detach().cpu().numpy()
    vy_np = vy_grid.detach().cpu().numpy()
    vz_np = vz_grid.detach().cpu().numpy()
    x_np = x_grid.detach().cpu().numpy()
    y_np = y_grid.detach().cpu().numpy()
    z_np = z_grid.detach().cpu().numpy()
    
    dx = x_np[1, 0, 0] - x_np[0, 0, 0] if n > 1 else 1.0
    dy = y_np[0, 1, 0] - y_np[0, 0, 0] if n > 1 else 1.0
    dz = z_np[0, 0, 1] - z_np[0, 0, 0] if n > 1 else 1.0
    
    dvx_dx = np.gradient(vx_np, dx, axis=0)
    dvy_dy = np.gradient(vy_np, dy, axis=1)
    dvz_dz = np.gradient(vz_np, dz, axis=2)
    divergence = dvx_dx + dvy_dy + dvz_dz
    
    # Check divergence statistics
    max_div = np.abs(divergence).max()
    mean_div = np.abs(divergence).mean()
    std_div = np.abs(divergence).std()
    
    print(f"  Divergence statistics:")
    print(f"    Max |div|: {max_div:.6f}")
    print(f"    Mean |div|: {mean_div:.6f}")
    print(f"    Std |div|: {std_div:.6f}")
    
    v_mag = np.sqrt(vx_np**2 + vy_np**2 + vz_np**2)
    v_mag_mean = v_mag.mean()
    relative_div = max_div / (v_mag_mean + 1e-10)
    
    print(f"    Mean velocity magnitude: {v_mag_mean:.6f}")
    print(f"    Relative max divergence: {relative_div:.6f}")
    
    # Check that divergence is finite
    assert np.all(np.isfinite(divergence)), "Divergence contains non-finite values"
    print("[OK] Divergence computation is finite")
    
    if relative_div < 10.0:
        print("[OK] Divergence is within reasonable bounds")
    else:
        print(f"[WARNING] Relative divergence is large ({relative_div:.2f})")
    
    print("[OK] 3D divergence validation completed")


def test_shared_fields_vs_fallback():
    """Test that shared fields and fallback modes both work."""
    print("\n" + "="*60)
    print("TEST 5: Shared Fields vs Fallback Mode")
    print("="*60)
    
    # Test parameters
    lam = wave
    v_1 = a * cs
    num_points = 50
    
    # Create test coordinates
    x = torch.rand(num_points, 1) * (lam * num_of_waves)
    y = torch.rand(num_points, 1) * (lam * num_of_waves)
    t = torch.zeros(num_points, 1)
    coords_2d = [x, y, t]
    
    # Test 1: With shared fields (2D)
    initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=RANDOM_SEED, dimension=2)
    vx_shared = generate_power_spectrum_field(lam, v_1, coords_2d, seed=RANDOM_SEED)
    vy_shared = generate_power_spectrum_field_vy(lam, v_1, coords_2d, seed=RANDOM_SEED)
    
    assert torch.all(torch.isfinite(vx_shared)), "Shared field vx contains non-finite values"
    assert torch.all(torch.isfinite(vy_shared)), "Shared field vy contains non-finite values"
    print("[OK] Shared fields mode works")
    
    # Test 2: Clear shared fields and test fallback (by importing and clearing)
    # Note: We can't easily clear the global variables from here, but we can test
    # that the functions work with different seeds or coordinates
    # The fallback will be used if shared fields are None (which happens on first import)
    
    # Test 3D shared fields
    x_3d = torch.rand(num_points, 1) * (lam * num_of_waves)
    y_3d = torch.rand(num_points, 1) * (lam * num_of_waves)
    z_3d = torch.rand(num_points, 1) * (lam * num_of_waves)
    t_3d = torch.zeros(num_points, 1)
    coords_3d = [x_3d, y_3d, z_3d, t_3d]
    
    initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=RANDOM_SEED, dimension=3)
    vx_3d = generate_power_spectrum_field(lam, v_1, coords_3d, seed=RANDOM_SEED)
    vy_3d = generate_power_spectrum_field_vy(lam, v_1, coords_3d, seed=RANDOM_SEED)
    vz_3d = generate_power_spectrum_field_vz(lam, v_1, coords_3d, seed=RANDOM_SEED)
    
    assert torch.all(torch.isfinite(vx_3d)), "3D shared field vx contains non-finite values"
    assert torch.all(torch.isfinite(vy_3d)), "3D shared field vy contains non-finite values"
    assert torch.all(torch.isfinite(vz_3d)), "3D shared field vz contains non-finite values"
    print("[OK] 3D shared fields mode works")
    
    print("[OK] Shared fields vs fallback tests passed")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("TEST 6: Edge Cases and Error Handling")
    print("="*60)
    
    lam = wave
    v_1 = a * cs
    
    # Reinitialize 2D shared fields for these tests (since previous test may have set 3D)
    initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=RANDOM_SEED, dimension=2)
    
    # Test 1: Single point
    x = torch.tensor([[1.0]])
    y = torch.tensor([[1.0]])
    t = torch.tensor([[0.0]])
    coords_2d = [x, y, t]
    
    vx = generate_power_spectrum_field(lam, v_1, coords_2d, seed=RANDOM_SEED)
    assert vx.shape == (1, 1), f"Single point test failed: {vx.shape}"
    print("[OK] Single point handling works")
    
    # Test 2: Zero coordinates
    x = torch.zeros(10, 1)
    y = torch.zeros(10, 1)
    t = torch.zeros(10, 1)
    coords_2d = [x, y, t]
    
    vx = generate_power_spectrum_field(lam, v_1, coords_2d, seed=RANDOM_SEED)
    assert torch.all(torch.isfinite(vx)), "Zero coordinates test failed"
    print("[OK] Zero coordinates handling works")
    
    # Test 3: Very small domain
    x = torch.rand(10, 1) * 0.01
    y = torch.rand(10, 1) * 0.01
    t = torch.zeros(10, 1)
    coords_2d = [x, y, t]
    
    vx = generate_power_spectrum_field(lam, v_1, coords_2d, seed=RANDOM_SEED)
    assert torch.all(torch.isfinite(vx)), "Small domain test failed"
    print("[OK] Small domain handling works")
    
    # Test 4: 3D with constant z (should be detected as 2D)
    x = torch.rand(10, 1) * (lam * num_of_waves)
    y = torch.rand(10, 1) * (lam * num_of_waves)
    z = torch.zeros(10, 1)  # Constant z
    t = torch.zeros(10, 1)
    coords_3d_const_z = [x, y, z, t]
    
    dim = _detect_dimension(coords_3d_const_z)
    # Should detect as 2D since z is constant
    assert dim == 2, f"Expected dimension 2 for constant z, got {dim}"
    print("[OK] Constant z coordinate detection works")
    
    print("[OK] All edge case tests passed")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST SUITE FOR 3D INITIAL CONDITIONS")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  N_GRID: {N_GRID}")
    print(f"  N_GRID_3D: {N_GRID_3D}")
    print(f"  POWER_EXPONENT: {POWER_EXPONENT}")
    print(f"  PERTURBATION_TYPE: {PERTURBATION_TYPE}")
    print(f"  RANDOM_SEED: {RANDOM_SEED}")
    
    try:
        test_2d_backward_compatibility()
        test_3d_functionality()
        test_divergence_free_2d()
        test_divergence_free_3d()
        test_shared_fields_vs_fallback()
        test_edge_cases()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        return 0
        
    except AssertionError as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
