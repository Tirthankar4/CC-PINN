"""
Test script for integration with existing code patterns.
Tests backward compatibility with deprecated wrapper functions and integration patterns.
"""

import numpy as np
import sys
import traceback
from config import cs, rho_o, const, G, KX, KY, KZ, wave, num_of_waves

# Import both old wrapper functions and new unified solvers
from numerical_solvers.LAX import (
    lax_solver, lax_solution, lax_solution_3d_sinusoidal,
    generate_velocity_field_power_spectrum, generate_shared_velocity_field
)
try:
    from numerical_solvers.LAX_torch import (
        lax_solver_torch, lax_solution_torch,
        generate_velocity_field_power_spectrum_torch
    )
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch solver not available: {e}")
    TORCH_AVAILABLE = False

# Test parameters
TEST_TIME = 0.3  # Very short for quick tests
NU = 0.5
LAM = wave
RHO_1 = 0.1
N_2D = 64
N_3D = 32  # Low resolution for 3D
PS_INDEX = -3.0
VEL_RMS = 0.02
RANDOM_SEED = 42

def print_test_header(test_name):
    """Print a formatted test header."""
    print("\n" + "="*70)
    print(f"  {test_name}")
    print("="*70)

def print_test_result(success, message=""):
    """Print test result."""
    status = "[PASS]" if success else "[FAIL]"
    print(f"{status}: {message}")

def test_deprecated_wrapper_2d():
    """Test deprecated lax_solution() wrapper function (2D)."""
    print_test_header("Test 1: Deprecated lax_solution() Wrapper (2D)")
    
    try:
        # Test with sinusoidal IC
        result = lax_solution(
            time=TEST_TIME,
            N=N_2D,
            nu=NU,
            lam=LAM,
            num_of_waves=num_of_waves,
            rho_1=RHO_1,
            gravity=True,
            isplot=False,
            comparison=False,
            use_velocity_ps=False
        )
        
        # Should return tuple: x, rho, vx, vy, phi, n, rho_max (with gravity)
        assert len(result) == 7, f"Expected 7 return values, got {len(result)}"
        x, rho, vx, vy, phi, n, rho_max = result
        
        assert x.shape == (N_2D,), f"Wrong x shape: {x.shape}"
        assert rho.shape == (N_2D, N_2D), f"Wrong rho shape: {rho.shape}"
        assert vx.shape == (N_2D, N_2D), f"Wrong vx shape: {vx.shape}"
        assert vy.shape == (N_2D, N_2D), f"Wrong vy shape: {vy.shape}"
        assert phi.shape == (N_2D, N_2D), f"Wrong phi shape: {phi.shape}"
        assert isinstance(n, int) and n > 0, f"n should be positive integer, got {n}"
        assert rho_max > 0, f"rho_max should be positive, got {rho_max}"
        
        print_test_result(True, f"Deprecated wrapper works. Iterations: {n}, rho_max: {rho_max:.4f}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_deprecated_wrapper_2d_power_spectrum():
    """Test deprecated lax_solution() with power spectrum IC."""
    print_test_header("Test 2: Deprecated lax_solution() with Power Spectrum")
    
    try:
        result = lax_solution(
            time=TEST_TIME,
            N=N_2D,
            nu=NU,
            lam=LAM,
            num_of_waves=num_of_waves,
            rho_1=RHO_1,
            gravity=True,
            isplot=False,
            comparison=False,
            use_velocity_ps=True,
            ps_index=PS_INDEX,
            vel_rms=VEL_RMS,
            random_seed=RANDOM_SEED
        )
        
        x, rho, vx, vy, phi, n, rho_max = result
        
        # Check that velocity fields are non-zero
        vx_max = np.max(np.abs(vx))
        vy_max = np.max(np.abs(vy))
        assert vx_max > 0, "vx should be non-zero"
        assert vy_max > 0, "vy should be non-zero"
        
        print_test_result(True, f"Power spectrum wrapper works. Iterations: {n}, vx_max: {vx_max:.4f}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_deprecated_wrapper_3d():
    """Test deprecated lax_solution_3d_sinusoidal() wrapper function."""
    print_test_header("Test 3: Deprecated lax_solution_3d_sinusoidal() Wrapper")
    
    try:
        result = lax_solution_3d_sinusoidal(
            time=TEST_TIME,
            N=N_3D,
            nu=NU,
            lam=LAM,
            num_of_waves=num_of_waves,
            rho_1=RHO_1,
            gravity=True,
            use_velocity_ps=False
        )
        
        # Should return: x, y, z, rho, vx, vy, vz, phi, k_iter, rho_max
        assert len(result) == 10, f"Expected 10 return values, got {len(result)}"
        x, y, z, rho, vx, vy, vz, phi, k_iter, rho_max = result
        
        assert x.shape == (N_3D,), f"Wrong x shape: {x.shape}"
        assert y.shape == (N_3D,), f"Wrong y shape: {y.shape}"
        assert z.shape == (N_3D,), f"Wrong z shape: {z.shape}"
        assert rho.shape == (N_3D, N_3D, N_3D), f"Wrong rho shape: {rho.shape}"
        assert vx.shape == (N_3D, N_3D, N_3D), f"Wrong vx shape: {vx.shape}"
        assert vy.shape == (N_3D, N_3D, N_3D), f"Wrong vy shape: {vy.shape}"
        assert vz.shape == (N_3D, N_3D, N_3D), f"Wrong vz shape: {vz.shape}"
        assert phi.shape == (N_3D, N_3D, N_3D), f"Wrong phi shape: {phi.shape}"
        
        print_test_result(True, f"3D wrapper works. Iterations: {k_iter}, rho_max: {rho_max:.4f}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_deprecated_wrapper_3d_power_spectrum():
    """Test deprecated lax_solution_3d_sinusoidal() with power spectrum IC."""
    print_test_header("Test 4: Deprecated 3D Wrapper with Power Spectrum")
    
    try:
        result = lax_solution_3d_sinusoidal(
            time=TEST_TIME,
            N=N_3D,
            nu=NU,
            lam=LAM,
            num_of_waves=num_of_waves,
            rho_1=RHO_1,
            gravity=True,
            use_velocity_ps=True,
            ps_index=PS_INDEX,
            vel_rms=VEL_RMS,
            random_seed=RANDOM_SEED
        )
        
        x, y, z, rho, vx, vy, vz, phi, k_iter, rho_max = result
        
        # Check that all velocity components are non-zero
        vx_max = np.max(np.abs(vx))
        vy_max = np.max(np.abs(vy))
        vz_max = np.max(np.abs(vz))
        assert vx_max > 0, "vx should be non-zero"
        assert vy_max > 0, "vy should be non-zero"
        assert vz_max > 0, "vz should be non-zero"
        
        print_test_result(True, f"3D power spectrum wrapper works. Iterations: {k_iter}, vz_max: {vz_max:.4f}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_power_spectrum_generator_2d():
    """Test generate_velocity_field_power_spectrum() for 2D."""
    print_test_header("Test 5: Power Spectrum Generator (2D)")
    
    try:
        Lx = Ly = LAM * num_of_waves
        
        vx, vy = generate_velocity_field_power_spectrum(
            nx=N_2D, ny=N_2D, Lx=Lx, Ly=Ly,
            power_index=PS_INDEX, amplitude=VEL_RMS,
            DIMENSION=2, random_seed=RANDOM_SEED
        )
        
        assert vx.shape == (N_2D, N_2D), f"Wrong vx shape: {vx.shape}"
        assert vy.shape == (N_2D, N_2D), f"Wrong vy shape: {vy.shape}"
        
        # Check RMS amplitude
        vx_rms = np.sqrt(np.mean(vx**2))
        vy_rms = np.sqrt(np.mean(vy**2))
        
        # Should be approximately VEL_RMS (within 20% tolerance)
        assert abs(vx_rms - VEL_RMS) < 0.2 * VEL_RMS, f"vx RMS should be ~{VEL_RMS}, got {vx_rms}"
        assert abs(vy_rms - VEL_RMS) < 0.2 * VEL_RMS, f"vy RMS should be ~{VEL_RMS}, got {vy_rms}"
        
        print_test_result(True, f"2D power spectrum generator works. vx_rms: {vx_rms:.4f}, vy_rms: {vy_rms:.4f}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_power_spectrum_generator_3d():
    """Test generate_velocity_field_power_spectrum() for 3D."""
    print_test_header("Test 6: Power Spectrum Generator (3D)")
    
    try:
        Lx = Ly = Lz = LAM * num_of_waves
        
        vx, vy, vz = generate_velocity_field_power_spectrum(
            nx=N_3D, ny=N_3D, Lx=Lx, Ly=Ly,
            power_index=PS_INDEX, amplitude=VEL_RMS,
            DIMENSION=3, random_seed=RANDOM_SEED,
            nz=N_3D, Lz=Lz
        )
        
        assert vx.shape == (N_3D, N_3D, N_3D), f"Wrong vx shape: {vx.shape}"
        assert vy.shape == (N_3D, N_3D, N_3D), f"Wrong vy shape: {vy.shape}"
        assert vz.shape == (N_3D, N_3D, N_3D), f"Wrong vz shape: {vz.shape}"
        
        # Check RMS amplitude
        vx_rms = np.sqrt(np.mean(vx**2))
        vy_rms = np.sqrt(np.mean(vy**2))
        vz_rms = np.sqrt(np.mean(vz**2))
        
        assert abs(vx_rms - VEL_RMS) < 0.2 * VEL_RMS, f"vx RMS should be ~{VEL_RMS}, got {vx_rms}"
        assert abs(vy_rms - VEL_RMS) < 0.2 * VEL_RMS, f"vy RMS should be ~{VEL_RMS}, got {vy_rms}"
        assert abs(vz_rms - VEL_RMS) < 0.2 * VEL_RMS, f"vz RMS should be ~{VEL_RMS}, got {vz_rms}"
        
        print_test_result(True, f"3D power spectrum generator works. vx_rms: {vx_rms:.4f}, vy_rms: {vy_rms:.4f}, vz_rms: {vz_rms:.4f}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_shared_velocity_field_2d():
    """Test generate_shared_velocity_field() for 2D."""
    print_test_header("Test 7: Shared Velocity Field Generator (2D)")
    
    try:
        Lx = Ly = LAM * num_of_waves
        
        vx_np, vy_np, vx_interp, vy_interp = generate_shared_velocity_field(
            nx=N_2D, ny=N_2D, Lx=Lx, Ly=Ly,
            power_index=PS_INDEX, amplitude=VEL_RMS,
            DIMENSION=2, random_seed=RANDOM_SEED
        )
        
        assert vx_np.shape == (N_2D, N_2D), f"Wrong vx_np shape: {vx_np.shape}"
        assert vy_np.shape == (N_2D, N_2D), f"Wrong vy_np shape: {vy_np.shape}"
        assert callable(vx_interp), "vx_interp should be callable"
        assert callable(vy_interp), "vy_interp should be callable"
        
        # Test interpolation
        test_points = np.array([[Lx/4, Ly/4], [Lx/2, Ly/2]])
        vx_interp_vals = vx_interp(test_points)
        vy_interp_vals = vy_interp(test_points)
        
        assert vx_interp_vals.shape == (2,), f"Wrong interpolation shape: {vx_interp_vals.shape}"
        assert vy_interp_vals.shape == (2,), f"Wrong interpolation shape: {vy_interp_vals.shape}"
        
        print_test_result(True, "Shared velocity field generator works")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_torch_wrapper():
    """Test deprecated lax_solution_torch() wrapper function."""
    if not TORCH_AVAILABLE:
        print_test_header("Test 8: Deprecated lax_solution_torch() Wrapper")
        print_test_result(False, "PyTorch not available, skipping")
        return False
    
    print_test_header("Test 8: Deprecated lax_solution_torch() Wrapper")
    
    try:
        result = lax_solution_torch(
            time_val=TEST_TIME,
            N=N_2D,
            nu=NU,
            lam=LAM,
            num_of_waves=num_of_waves,
            rho_1=RHO_1,
            gravity=True,
            use_velocity_ps=False
        )
        
        # Should return: x, rho, vx, vy, phi, n, rho_max
        assert len(result) == 7, f"Expected 7 return values, got {len(result)}"
        x, rho, vx, vy, phi, n, rho_max = result
        
        # Convert to numpy if needed
        if hasattr(x, 'cpu'):
            x = x.cpu().numpy()
        if hasattr(rho, 'cpu'):
            rho = rho.cpu().numpy()
        
        assert x.shape == (N_2D,), f"Wrong x shape: {x.shape}"
        assert rho.shape == (N_2D, N_2D), f"Wrong rho shape: {rho.shape}"
        assert isinstance(n, int) and n > 0, f"n should be positive integer, got {n}"
        
        print_test_result(True, f"PyTorch wrapper works. Iterations: {n}, rho_max: {rho_max:.4f}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_torch_power_spectrum_generator():
    """Test generate_velocity_field_power_spectrum_torch() for 3D."""
    if not TORCH_AVAILABLE:
        print_test_header("Test 9: PyTorch Power Spectrum Generator (3D)")
        print_test_result(False, "PyTorch not available, skipping")
        return False
    
    print_test_header("Test 9: PyTorch Power Spectrum Generator (3D)")
    
    try:
        Lx = Ly = Lz = LAM * num_of_waves
        
        vx, vy, vz = generate_velocity_field_power_spectrum_torch(
            nx=N_3D, ny=N_3D, Lx=Lx, Ly=Ly,
            power_index=PS_INDEX, amplitude=VEL_RMS,
            DIMENSION=3, random_seed=RANDOM_SEED,
            nz=N_3D, Lz=Lz
        )
        
        # Convert to numpy for shape checking
        if hasattr(vx, 'shape'):
            vx_shape = vx.shape
            vy_shape = vy.shape
            vz_shape = vz.shape
        else:
            vx_shape = np.array(vx).shape
            vy_shape = np.array(vy).shape
            vz_shape = np.array(vz).shape
        
        assert vx_shape == (N_3D, N_3D, N_3D), f"Wrong vx shape: {vx_shape}"
        assert vy_shape == (N_3D, N_3D, N_3D), f"Wrong vy shape: {vy_shape}"
        assert vz_shape == (N_3D, N_3D, N_3D), f"Wrong vz shape: {vz_shape}"
        
        print_test_result(True, "PyTorch 3D power spectrum generator works")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("  LAX Solver Integration Test Suite")
    print("  Testing backward compatibility and integration patterns")
    print("="*70)
    
    results = []
    
    # Test deprecated wrappers
    results.append(("Deprecated 2D Wrapper", test_deprecated_wrapper_2d()))
    results.append(("Deprecated 2D Power Spectrum Wrapper", test_deprecated_wrapper_2d_power_spectrum()))
    results.append(("Deprecated 3D Wrapper", test_deprecated_wrapper_3d()))
    results.append(("Deprecated 3D Power Spectrum Wrapper", test_deprecated_wrapper_3d_power_spectrum()))
    
    # Test utility functions
    results.append(("Power Spectrum Generator (2D)", test_power_spectrum_generator_2d()))
    results.append(("Power Spectrum Generator (3D)", test_power_spectrum_generator_3d()))
    results.append(("Shared Velocity Field (2D)", test_shared_velocity_field_2d()))
    
    # Test PyTorch wrappers
    results.append(("PyTorch Wrapper", test_torch_wrapper()))
    results.append(("PyTorch Power Spectrum Generator (3D)", test_torch_power_spectrum_generator()))
    
    # Summary
    print("\n" + "="*70)
    print("  Integration Test Summary")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {test_name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  [SUCCESS] All integration tests passed!")
        return 0
    else:
        print(f"\n  [FAILED] {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

