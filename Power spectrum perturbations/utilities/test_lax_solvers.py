"""
Test script for refactored LAX solvers (CPU and PyTorch versions).
Tests 2D and 3D power spectrum support, sinusoidal ICs, and warm start functionality.
Uses low resolution for 3D to avoid resource issues.
"""

import numpy as np
import sys
import traceback
from config import cs, rho_o, const, G, KX, KY, KZ, wave, num_of_waves

# Import solvers
from numerical_solvers.LAX import lax_solver, DomainParams, SimulationResult as SimulationResultCPU
try:
    from numerical_solvers.LAX_torch import lax_solver_torch, SimulationResult as SimulationResultTorch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch solver not available: {e}")
    TORCH_AVAILABLE = False

# Test parameters
TEST_TIME = 0.5  # Short simulation time for quick tests
NU = 0.5  # Courant number
LAM = wave  # Wavelength from config
RHO_1 = 0.1  # Density perturbation amplitude

# Low resolution for 3D tests
N_2D = 64  # 2D resolution
N_3D = 32  # 3D resolution (low to save resources)

# Power spectrum parameters
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

def test_2d_sinusoidal_cpu():
    """Test 2D sinusoidal initial conditions with CPU solver."""
    print_test_header("Test 1: 2D Sinusoidal IC (CPU)")
    
    try:
        domain_params = {
            'Lx': LAM * num_of_waves,
            'Ly': LAM * num_of_waves,
            'nx': N_2D,
            'ny': N_2D
        }
        
        physics_params = {
            'c_s': cs,
            'rho_o': rho_o,
            'const': const,
            'G': G,
            'rho_1': RHO_1,
            'lam': LAM
        }
        
        ic_params = {
            'KX': KX,
            'KY': KY
        }
        
        options = {
            'gravity': True,
            'nu': NU,
            'comparison': False,
            'isplot': False
        }
        
        result = lax_solver(
            TEST_TIME, domain_params, physics_params,
            ic_type='sinusoidal', ic_params=ic_params, options=options
        )
        
        # Validate result
        assert isinstance(result, SimulationResultCPU), "Result should be SimulationResult"
        assert result.dimension == 2, f"Expected dimension=2, got {result.dimension}"
        assert result.density.shape == (N_2D, N_2D), f"Wrong density shape: {result.density.shape}"
        assert len(result.velocity_components) == 2, "Should have 2 velocity components"
        assert result.velocity_components[0].shape == (N_2D, N_2D), "Wrong vx shape"
        assert result.velocity_components[1].shape == (N_2D, N_2D), "Wrong vy shape"
        assert result.potential is not None, "Potential should not be None (gravity=True)"
        assert result.potential.shape == (N_2D, N_2D), "Wrong potential shape"
        assert 'x' in result.coordinates and 'y' in result.coordinates, "Missing coordinates"
        assert result.metadata['iterations'] > 0, "Should have performed iterations"
        
        print_test_result(True, f"2D sinusoidal CPU solver works. Iterations: {result.metadata['iterations']}, rho_max: {result.metadata['rho_max']:.4f}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_2d_power_spectrum_cpu():
    """Test 2D power spectrum initial conditions with CPU solver."""
    print_test_header("Test 2: 2D Power Spectrum IC (CPU)")
    
    try:
        domain_params = {
            'Lx': LAM * num_of_waves,
            'Ly': LAM * num_of_waves,
            'nx': N_2D,
            'ny': N_2D
        }
        
        physics_params = {
            'c_s': cs,
            'rho_o': rho_o,
            'const': const,
            'G': G
        }
        
        ic_params = {
            'power_index': PS_INDEX,
            'amplitude': VEL_RMS,
            'random_seed': RANDOM_SEED
        }
        
        options = {
            'gravity': True,
            'nu': NU,
            'comparison': False,
            'isplot': False
        }
        
        result = lax_solver(
            TEST_TIME, domain_params, physics_params,
            ic_type='power_spectrum', ic_params=ic_params, options=options
        )
        
        # Validate result
        assert isinstance(result, SimulationResultCPU), "Result should be SimulationResult"
        assert result.dimension == 2, f"Expected dimension=2, got {result.dimension}"
        assert result.density.shape == (N_2D, N_2D), f"Wrong density shape: {result.density.shape}"
        assert len(result.velocity_components) == 2, "Should have 2 velocity components"
        assert result.potential is not None, "Potential should not be None"
        
        # Check that velocity fields are non-zero (power spectrum should generate perturbations)
        vx_max = np.max(np.abs(result.velocity_components[0]))
        vy_max = np.max(np.abs(result.velocity_components[1]))
        assert vx_max > 0, "vx should be non-zero"
        assert vy_max > 0, "vy should be non-zero"
        
        print_test_result(True, f"2D power spectrum CPU solver works. Iterations: {result.metadata['iterations']}, vx_max: {vx_max:.4f}, vy_max: {vy_max:.4f}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_3d_sinusoidal_cpu():
    """Test 3D sinusoidal initial conditions with CPU solver (low resolution)."""
    print_test_header("Test 3: 3D Sinusoidal IC (CPU, Low Resolution)")
    
    try:
        domain_params = {
            'Lx': LAM * num_of_waves,
            'Ly': LAM * num_of_waves,
            'Lz': LAM * num_of_waves,
            'nx': N_3D,
            'ny': N_3D,
            'nz': N_3D
        }
        
        physics_params = {
            'c_s': cs,
            'rho_o': rho_o,
            'const': const,
            'G': G,
            'rho_1': RHO_1,
            'lam': LAM
        }
        
        ic_params = {
            'KX': KX,
            'KY': KY,
            'KZ': KZ
        }
        
        options = {
            'gravity': True,
            'nu': NU,
            'comparison': False,
            'isplot': False
        }
        
        result = lax_solver(
            TEST_TIME, domain_params, physics_params,
            ic_type='sinusoidal', ic_params=ic_params, options=options
        )
        
        # Validate result
        assert isinstance(result, SimulationResultCPU), "Result should be SimulationResult"
        assert result.dimension == 3, f"Expected dimension=3, got {result.dimension}"
        assert result.density.shape == (N_3D, N_3D, N_3D), f"Wrong density shape: {result.density.shape}"
        assert len(result.velocity_components) == 3, "Should have 3 velocity components"
        assert result.velocity_components[0].shape == (N_3D, N_3D, N_3D), "Wrong vx shape"
        assert result.velocity_components[1].shape == (N_3D, N_3D, N_3D), "Wrong vy shape"
        assert result.velocity_components[2].shape == (N_3D, N_3D, N_3D), "Wrong vz shape"
        assert result.potential is not None, "Potential should not be None"
        assert result.potential.shape == (N_3D, N_3D, N_3D), "Wrong potential shape"
        assert 'z' in result.coordinates, "Missing z coordinate"
        
        print_test_result(True, f"3D sinusoidal CPU solver works. Iterations: {result.metadata['iterations']}, rho_max: {result.metadata['rho_max']:.4f}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_3d_power_spectrum_cpu():
    """Test 3D power spectrum initial conditions with CPU solver (low resolution)."""
    print_test_header("Test 4: 3D Power Spectrum IC (CPU, Low Resolution)")
    
    try:
        domain_params = {
            'Lx': LAM * num_of_waves,
            'Ly': LAM * num_of_waves,
            'Lz': LAM * num_of_waves,
            'nx': N_3D,
            'ny': N_3D,
            'nz': N_3D
        }
        
        physics_params = {
            'c_s': cs,
            'rho_o': rho_o,
            'const': const,
            'G': G
        }
        
        ic_params = {
            'power_index': PS_INDEX,
            'amplitude': VEL_RMS,
            'random_seed': RANDOM_SEED
        }
        
        options = {
            'gravity': True,
            'nu': NU,
            'comparison': False,
            'isplot': False
        }
        
        result = lax_solver(
            TEST_TIME, domain_params, physics_params,
            ic_type='power_spectrum', ic_params=ic_params, options=options
        )
        
        # Validate result
        assert isinstance(result, SimulationResultCPU), "Result should be SimulationResult"
        assert result.dimension == 3, f"Expected dimension=3, got {result.dimension}"
        assert result.density.shape == (N_3D, N_3D, N_3D), f"Wrong density shape: {result.density.shape}"
        assert len(result.velocity_components) == 3, f"Should have 3 velocity components, got {len(result.velocity_components)}"
        
        # Check that velocity fields are non-zero
        vx_max = np.max(np.abs(result.velocity_components[0]))
        vy_max = np.max(np.abs(result.velocity_components[1]))
        vz_max = np.max(np.abs(result.velocity_components[2]))
        assert vx_max > 0, "vx should be non-zero"
        assert vy_max > 0, "vy should be non-zero"
        assert vz_max > 0, "vz should be non-zero"
        
        print_test_result(True, f"3D power spectrum CPU solver works. Iterations: {result.metadata['iterations']}, vx_max: {vx_max:.4f}, vy_max: {vy_max:.4f}, vz_max: {vz_max:.4f}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_warm_start_cpu():
    """Test warm start functionality with CPU solver."""
    print_test_header("Test 5: Warm Start (CPU)")
    
    try:
        # First, run a short simulation to get initial state
        domain_params = {
            'Lx': LAM * num_of_waves,
            'Ly': LAM * num_of_waves,
            'nx': N_2D,
            'ny': N_2D
        }
        
        physics_params = {
            'c_s': cs,
            'rho_o': rho_o,
            'const': const,
            'G': G,
            'rho_1': RHO_1,
            'lam': LAM
        }
        
        ic_params = {
            'KX': KX,
            'KY': KY
        }
        
        options = {
            'gravity': True,
            'nu': NU,
            'comparison': False,
            'isplot': False
        }
        
        # Run initial simulation
        result1 = lax_solver(
            TEST_TIME / 2, domain_params, physics_params,
            ic_type='sinusoidal', ic_params=ic_params, options=options
        )
        
        # Use result as warm start
        provided_fields = {
            'rho': result1.density,
            'vx': result1.velocity_components[0],
            'vy': result1.velocity_components[1]
        }
        
        ic_params_warm = {
            'provided_fields': provided_fields
        }
        
        # Continue simulation from warm start
        result2 = lax_solver(
            TEST_TIME / 2, domain_params, physics_params,
            ic_type='warm_start', ic_params=ic_params_warm, options=options
        )
        
        # Validate result
        assert isinstance(result2, SimulationResultCPU), "Result should be SimulationResult"
        assert result2.dimension == 2, "Should be 2D"
        assert result2.density.shape == (N_2D, N_2D), "Wrong density shape"
        
        print_test_result(True, f"Warm start CPU solver works. Total iterations: {result1.metadata['iterations']} + {result2.metadata['iterations']}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_2d_sinusoidal_torch():
    """Test 2D sinusoidal initial conditions with PyTorch solver."""
    if not TORCH_AVAILABLE:
        print_test_header("Test 6: 2D Sinusoidal IC (PyTorch)")
        print_test_result(False, "PyTorch not available, skipping")
        return False
    
    print_test_header("Test 6: 2D Sinusoidal IC (PyTorch)")
    
    try:
        domain_params = {
            'Lx': LAM * num_of_waves,
            'Ly': LAM * num_of_waves,
            'nx': N_2D,
            'ny': N_2D
        }
        
        physics_params = {
            'c_s': cs,
            'rho_o': rho_o,
            'const': const,
            'G': G,
            'rho_1': RHO_1,
            'lam': LAM
        }
        
        ic_params = {
            'KX': KX,
            'KY': KY
        }
        
        options = {
            'gravity': True,
            'nu': NU,
            'comparison': False,
            'isplot': False
        }
        
        result = lax_solver_torch(
            TEST_TIME, domain_params, physics_params,
            ic_type='sinusoidal', ic_params=ic_params, options=options
        )
        
        # Validate result
        assert isinstance(result, SimulationResultTorch), "Result should be SimulationResultTorch"
        assert result.dimension == 2, f"Expected dimension=2, got {result.dimension}"
        assert result.density.shape == (N_2D, N_2D), f"Wrong density shape: {result.density.shape}"
        assert len(result.velocity_components) == 2, "Should have 2 velocity components"
        assert result.potential is not None, "Potential should not be None"
        
        print_test_result(True, f"2D sinusoidal PyTorch solver works. Iterations: {result.metadata['iterations']}, rho_max: {result.metadata['rho_max']:.4f}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_2d_power_spectrum_torch():
    """Test 2D power spectrum initial conditions with PyTorch solver."""
    if not TORCH_AVAILABLE:
        print_test_header("Test 7: 2D Power Spectrum IC (PyTorch)")
        print_test_result(False, "PyTorch not available, skipping")
        return False
    
    print_test_header("Test 7: 2D Power Spectrum IC (PyTorch)")
    
    try:
        domain_params = {
            'Lx': LAM * num_of_waves,
            'Ly': LAM * num_of_waves,
            'nx': N_2D,
            'ny': N_2D
        }
        
        physics_params = {
            'c_s': cs,
            'rho_o': rho_o,
            'const': const,
            'G': G
        }
        
        ic_params = {
            'power_index': PS_INDEX,
            'amplitude': VEL_RMS,
            'random_seed': RANDOM_SEED
        }
        
        options = {
            'gravity': True,
            'nu': NU,
            'comparison': False,
            'isplot': False
        }
        
        result = lax_solver_torch(
            TEST_TIME, domain_params, physics_params,
            ic_type='power_spectrum', ic_params=ic_params, options=options
        )
        
        # Validate result
        assert isinstance(result, SimulationResultTorch), "Result should be SimulationResultTorch"
        assert result.dimension == 2, f"Expected dimension=2, got {result.dimension}"
        assert result.density.shape == (N_2D, N_2D), f"Wrong density shape: {result.density.shape}"
        assert len(result.velocity_components) == 2, "Should have 2 velocity components"
        
        vx_max = np.max(np.abs(result.velocity_components[0]))
        vy_max = np.max(np.abs(result.velocity_components[1]))
        assert vx_max > 0, "vx should be non-zero"
        assert vy_max > 0, "vy should be non-zero"
        
        print_test_result(True, f"2D power spectrum PyTorch solver works. Iterations: {result.metadata['iterations']}, vx_max: {vx_max:.4f}, vy_max: {vy_max:.4f}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_3d_power_spectrum_torch():
    """Test 3D power spectrum initial conditions with PyTorch solver (low resolution)."""
    if not TORCH_AVAILABLE:
        print_test_header("Test 8: 3D Power Spectrum IC (PyTorch)")
        print_test_result(False, "PyTorch not available, skipping")
        return False
    
    print_test_header("Test 8: 3D Power Spectrum IC (PyTorch, Low Resolution)")
    
    try:
        domain_params = {
            'Lx': LAM * num_of_waves,
            'Ly': LAM * num_of_waves,
            'Lz': LAM * num_of_waves,
            'nx': N_3D,
            'ny': N_3D,
            'nz': N_3D
        }
        
        physics_params = {
            'c_s': cs,
            'rho_o': rho_o,
            'const': const,
            'G': G
        }
        
        ic_params = {
            'power_index': PS_INDEX,
            'amplitude': VEL_RMS,
            'random_seed': RANDOM_SEED
        }
        
        options = {
            'gravity': True,
            'nu': NU,
            'comparison': False,
            'isplot': False
        }
        
        result = lax_solver_torch(
            TEST_TIME, domain_params, physics_params,
            ic_type='power_spectrum', ic_params=ic_params, options=options
        )
        
        # Validate result
        assert isinstance(result, SimulationResultTorch), "Result should be SimulationResultTorch"
        assert result.dimension == 3, f"Expected dimension=3, got {result.dimension}"
        assert result.density.shape == (N_3D, N_3D, N_3D), f"Wrong density shape: {result.density.shape}"
        assert len(result.velocity_components) == 3, "Should have 3 velocity components"
        
        vx_max = np.max(np.abs(result.velocity_components[0]))
        vy_max = np.max(np.abs(result.velocity_components[1]))
        vz_max = np.max(np.abs(result.velocity_components[2]))
        assert vx_max > 0, "vx should be non-zero"
        assert vy_max > 0, "vy should be non-zero"
        assert vz_max > 0, "vz should be non-zero"
        
        print_test_result(True, f"3D power spectrum PyTorch solver works. Iterations: {result.metadata['iterations']}, vx_max: {vx_max:.4f}, vy_max: {vy_max:.4f}, vz_max: {vz_max:.4f}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_no_gravity():
    """Test solver without gravity."""
    print_test_header("Test 9: No Gravity (CPU)")
    
    try:
        domain_params = {
            'Lx': LAM * num_of_waves,
            'Ly': LAM * num_of_waves,
            'nx': N_2D,
            'ny': N_2D
        }
        
        physics_params = {
            'c_s': cs,
            'rho_o': rho_o,
            'const': const,
            'G': G,
            'rho_1': RHO_1,
            'lam': LAM
        }
        
        ic_params = {
            'KX': KX,
            'KY': KY
        }
        
        options = {
            'gravity': False,  # No gravity
            'nu': NU,
            'comparison': False,
            'isplot': False
        }
        
        result = lax_solver(
            TEST_TIME, domain_params, physics_params,
            ic_type='sinusoidal', ic_params=ic_params, options=options
        )
        
        # Validate result
        assert isinstance(result, SimulationResultCPU), "Result should be SimulationResult"
        assert result.dimension == 2, "Should be 2D"
        # Potential can be None or zeros when gravity=False
        print_test_result(True, f"No gravity test works. Iterations: {result.metadata['iterations']}")
        return True
        
    except Exception as e:
        print_test_result(False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  LAX Solver Test Suite")
    print("  Testing refactored solvers with 2D/3D power spectrum support")
    print("="*70)
    
    results = []
    
    # CPU tests
    results.append(("2D Sinusoidal (CPU)", test_2d_sinusoidal_cpu()))
    results.append(("2D Power Spectrum (CPU)", test_2d_power_spectrum_cpu()))
    results.append(("3D Sinusoidal (CPU)", test_3d_sinusoidal_cpu()))
    results.append(("3D Power Spectrum (CPU)", test_3d_power_spectrum_cpu()))
    results.append(("Warm Start (CPU)", test_warm_start_cpu()))
    results.append(("No Gravity (CPU)", test_no_gravity()))
    
    # PyTorch tests
    results.append(("2D Sinusoidal (PyTorch)", test_2d_sinusoidal_torch()))
    results.append(("2D Power Spectrum (PyTorch)", test_2d_power_spectrum_torch()))
    results.append(("3D Power Spectrum (PyTorch)", test_3d_power_spectrum_torch()))
    
    # Summary
    print("\n" + "="*70)
    print("  Test Summary")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {test_name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  [SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n  [FAILED] {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

