import os
import sys
import shutil
import numpy as np
import time
from typing import Tuple
import torch
import torch.nn as nn

# ==================== PyTorch Performance Optimizations ====================
# Enable cuDNN autotuner - finds optimal convolution algorithms
# This helps when input sizes are consistent (which they are in PINN training)
torch.backends.cudnn.benchmark = True

# Enable TF32 on Ampere GPUs (A100, RTX 3090, RTX 4090, etc.) for 20-30% speedup
# TF32 provides faster matrix multiplication with minimal accuracy loss
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Prefer explicit device/dtype settings rather than deprecated tensor-type override
    torch.set_default_dtype(torch.float32)
    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cuda")
    print("PyTorch performance optimizations enabled (cuDNN benchmark, TF32)")

from core.data_generator import input_taker, req_consts_calc
from training.trainer import train
from core.initial_conditions import initialize_shared_velocity_fields
from config import BATCH_SIZE, NUM_BATCHES, N_0, N_r, DIMENSION
from config import a, wave, cs, xmin, ymin, zmin, tmin, tmax as TMAX_CFG, iteration_adam_2D, iteration_lbgfs_2D, harmonics, PERTURBATION_TYPE, rho_o, SLICE_Y
from config import num_neurons, num_layers, num_of_waves
from config import RANDOM_SEED, STARTUP_DT
from config import ENABLE_TRAINING_DIAGNOSTICS
from core.losses import ASTPN
from core.model_architecture import PINN
from visualization.Plotting import create_2d_animation
from visualization.Plotting import create_1d_cross_sections_sinusoidal
from visualization.Plotting import create_density_growth_plot
from config import PLOT_DENSITY_GROWTH, GROWTH_PLOT_TMAX, GROWTH_PLOT_DT
from config import FD_N_2D


def _save_trained_models(net):
    """Persist trained model immediately after training completes."""
    try:
        from config import SNAPSHOT_DIR
        model_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(net.state_dict(), model_path)
        print(f"Saved model to {model_path}")
    except Exception as e:
        print(f"Warning: failed to save model: {e}")


def _comprehensive_gpu_cleanup(nets, optimizers=None, collocation_data=None, 
                                cached_ic_values=None, keep_models_for_vis=True, 
                                target_device=None):
    """
    Comprehensive GPU memory cleanup after PINN training.
    
    This function properly frees GPU memory by:
    1. Moving models to CPU (if keep_models_for_vis=True) or deleting them
    2. Deleting optimizer state (Adam momentum buffers, LBFGS history)
    3. Deleting collocation points and related tensors
    4. Deleting cached IC values
    5. Clearing PyTorch's cache allocator
    
    Args:
        nets: List of neural networks (or single network in a list)
        optimizers: List of optimizers to clean up (optional)
        collocation_data: Dict or list of collocation tensors to delete (optional)
        cached_ic_values: List of cached IC value dicts to delete (optional)
        keep_models_for_vis: If True, move models to CPU instead of deleting (default: True)
        target_device: Target device for models if keep_models_for_vis=True (default: 'cpu')
    
    Returns:
        None (modifies nets in-place)
    """
    if not torch.cuda.is_available():
        return
    
    if target_device is None:
        target_device = 'cpu'
    
    print("Starting comprehensive GPU memory cleanup...")
    
    # 1. Set models to eval mode (reduces memory usage) and optionally move to CPU
    if nets is not None:
        for i, net in enumerate(nets):
            if net is not None:
                # Set to eval mode to disable gradient tracking and reduce memory
                net.eval()
                # Clear any cached gradients
                for param in net.parameters():
                    if param.grad is not None:
                        param.grad = None
                
                if keep_models_for_vis:
                    # Move to target device (usually CPU) for visualization
                    # Models can be moved back to GPU later if needed
                    net = net.to(target_device)
                    nets[i] = net
                    print(f"  Moved network {i} to {target_device} (eval mode)")
                else:
                    # Delete model entirely
                    del net
                    nets[i] = None
                    print(f"  Deleted network {i}")
    
    # 2. Delete optimizer state (can be large, especially LBFGS history)
    if optimizers is not None:
        if isinstance(optimizers, (list, tuple)):
            for i, opt in enumerate(optimizers):
                if opt is not None:
                    # Clear optimizer state (this frees momentum buffers, LBFGS history, etc.)
                    opt.state.clear()
                    opt.param_groups.clear()
                    del opt
                    print(f"  Deleted optimizer {i} state")
        else:
            if optimizers is not None:
                optimizers.state.clear()
                optimizers.param_groups.clear()
                del optimizers
                print("  Deleted optimizer state")
    
    # 3. Delete collocation points (these can be very large)
    if collocation_data is not None:
        if isinstance(collocation_data, dict):
            for key, value in collocation_data.items():
                if value is not None:
                    if isinstance(value, (list, tuple)):
                        for item in value:
                            if isinstance(item, torch.Tensor) and item.is_cuda:
                                del item
                    elif isinstance(value, torch.Tensor) and value.is_cuda:
                        del value
            collocation_data.clear()
            print("  Deleted collocation data dict")
        elif isinstance(collocation_data, (list, tuple)):
            for item in collocation_data:
                if item is not None:
                    if isinstance(item, (list, tuple)):
                        for subitem in item:
                            if isinstance(subitem, torch.Tensor) and subitem.is_cuda:
                                del subitem
                    elif isinstance(item, torch.Tensor) and item.is_cuda:
                        del item
            print("  Deleted collocation data list")
        elif isinstance(collocation_data, torch.Tensor) and collocation_data.is_cuda:
            del collocation_data
            print("  Deleted collocation tensor")
    
    # 4. Delete cached IC values
    if cached_ic_values is not None:
        if isinstance(cached_ic_values, list):
            for i, cached_dict in enumerate(cached_ic_values):
                if cached_dict is not None and isinstance(cached_dict, dict):
                    for key, value in cached_dict.items():
                        if isinstance(value, torch.Tensor) and value.is_cuda:
                            del value
                    cached_dict.clear()
            cached_ic_values.clear()
            print("  Deleted cached IC values")
    
    # 5. Force garbage collection
    import gc
    gc.collect()
    
    # 6. Clear PyTorch's cache allocator and synchronize
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    print("GPU memory cleanup completed")
    
    # Note: If visualization needs models on GPU, they can be moved back with:
    # for net in nets:
    #     if net is not None:
    #         net = net.to(device)


def _move_models_to_device(nets, target_device):
    """
    Move models to target device (e.g., back to GPU for visualization).
    
    Args:
        nets: List of neural networks
        target_device: Target device string (e.g., 'cuda:0' or 'cpu')
    
    Returns:
        None (modifies nets in-place)
    """
    if nets is not None:
        for i, net in enumerate(nets):
            if net is not None:
                net = net.to(target_device)
                nets[i] = net


def clean_pycache(root_dir: str) -> Tuple[int, int]:
    """Remove all __pycache__ directories and .pyc files under ``root_dir``."""
    removed_dirs = 0
    removed_files = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                print(f"Removed: {pycache_path}")
                removed_dirs += 1
            except Exception as exc:  # pragma: no cover - cleanup best effort
                print(f"Error removing {pycache_path}: {exc}")

        for filename in filenames:
            if filename.endswith(".pyc"):
                pyc_path = os.path.join(dirpath, filename)
                try:
                    os.remove(pyc_path)
                    print(f"Removed: {pyc_path}")
                    removed_files += 1
                except Exception as exc:  # pragma: no cover - cleanup best effort
                    print(f"Error removing {pyc_path}: {exc}")

    print("=" * 60)
    print("Cleanup complete!")
    print(f"  Removed {removed_dirs} __pycache__ directories")
    print(f"  Removed {removed_files} .pyc files")
    print("=" * 60)

    return removed_dirs, removed_files


if "--clean-pycache" in sys.argv or "--clean-cache" in sys.argv:
    flag = "--clean-pycache" if "--clean-pycache" in sys.argv else "--clean-cache"
    script_root = os.path.dirname(os.path.abspath(__file__))
    print(f"Cleaning Python cache files from: {script_root}")
    clean_pycache(script_root)
    # Prevent the rest of the training script from running in cleanup-only mode
    sys.exit(0)

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"

# Clear GPU memory if using CUDA
if device.startswith('cuda'):
    torch.cuda.empty_cache()


lam, rho_1, num_of_waves, tmax, _, _, _ = input_taker(wave, a, num_of_waves, TMAX_CFG, N_0, 0, N_r)

jeans, alpha = req_consts_calc(lam, rho_1)
# Set initial velocity amplitude per perturbation type
if str(PERTURBATION_TYPE).lower() == "sinusoidal":
    k = 2*np.pi/lam
    v_1 = (rho_1 / (rho_o if rho_o != 0 else 1.0)) * (alpha / k)
else:
    v_1 = a * cs

xmax = xmin + lam * num_of_waves
ymax = ymin + lam * num_of_waves
zmax = zmin + lam * num_of_waves
#zmax = 1.0

# Initialize shared velocity fields for consistent PINN/FD initial conditions
vx_np, vy_np, vz_np = None, None, None  # Default values for sinusoidal case
if str(PERTURBATION_TYPE).lower() == "power_spectrum":
    if DIMENSION == 3:
        vx_np, vy_np, vz_np = initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=RANDOM_SEED, dimension=3)
    else:
        vx_np, vy_np = initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=RANDOM_SEED, dimension=2)
    
    # Set shared velocity fields for plotting (only for 2D visualization)
    if DIMENSION == 2:
        from visualization.Plotting import set_shared_velocity_fields
        set_shared_velocity_fields(vx_np, vy_np)

# ==================== STANDARD PINN MODE ====================
print("Running in standard PINN mode (single network)...")

net = PINN(n_harmonics=harmonics)
net = net.to(device)
mse_cost_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
optimizerL = torch.optim.LBFGS(net.parameters(), line_search_fn='strong_wolfe')

if DIMENSION == 1:
    astpn_rmin = [xmin, tmin]
    astpn_rmax = [xmax, tmax]
elif DIMENSION == 2:
    astpn_rmin = [xmin, ymin, tmin]
    astpn_rmax = [xmax, ymax, tmax]
elif DIMENSION == 3:
    astpn_rmin = [xmin, ymin, zmin, tmin]
    astpn_rmax = [xmax, ymax, zmax, tmax]
else:
    raise ValueError(f"Unsupported DIMENSION={DIMENSION}")
collocation_model = ASTPN(rmin=astpn_rmin, rmax=astpn_rmax, N_0=N_0, N_b=0, N_r=N_r, dimension=DIMENSION)

# Set domain on the network so periodic embeddings enforce hard BCs
spatial_rmin = [xmin]
spatial_rmax = [xmax]
if DIMENSION >= 2:
    spatial_rmin.append(ymin)
    spatial_rmax.append(ymax)
if DIMENSION >= 3:
    spatial_rmin.append(zmin)
    spatial_rmax.append(zmax)
net.set_domain(rmin=spatial_rmin, rmax=spatial_rmax, dimension=DIMENSION)

# IC collocation stays at t=0 throughout
collocation_IC = collocation_model.geo_time_coord(option="IC")

# Generate extra collocation points at t=0 for Poisson enforcement (Option 3)
from core.data_generator import generate_poisson_ic_points
from config import N_POISSON_IC
collocation_poisson_ic = generate_poisson_ic_points(
    rmin=collocation_model.rmin,
    rmax=collocation_model.rmax,
    n_points=N_POISSON_IC,
    dimension=DIMENSION,
    device=device
)
print(f"Generated {N_POISSON_IC} extra collocation points at t=0 for Poisson enforcement")

start_time = time.time()

# Standard training
print("Using standard training...")
collocation_domain = collocation_model.geo_time_coord(option="Domain")

training_diagnostics = train(
    net=net,
    model=collocation_model,
    collocation_domain=collocation_domain,
    collocation_IC=collocation_IC,
    collocation_poisson_ic=collocation_poisson_ic,
    optimizer=optimizer,
    optimizerL=optimizerL,
    closure=None,
    mse_cost_function=mse_cost_function,
    iteration_adam=iteration_adam_2D,
    iterationL=iteration_lbgfs_2D,
    rho_1=rho_1,
    lam=lam,
    jeans=jeans,
    v_1=v_1,
    device=device
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"[Timing] PINN training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

# Save model immediately after training completes
_save_trained_models(net)

# ==================== POST-TRAINING COMPREHENSIVE DIAGNOSTICS ====================
# Run comprehensive diagnostics for high-tmax failure analysis
# This generates 4 additional critical plots beyond the training diagnostics
if ENABLE_TRAINING_DIAGNOSTICS and training_diagnostics is not None:
    print("\n" + "="*70)
    print("Running post-training comprehensive diagnostics...")
    print("="*70)
    
    try:
        # Run comprehensive diagnostics - generates 4 additional plots:
        # 1. PDE Residual Heatmaps - WHERE/WHEN physics breaks
        # 2. Conservation Laws - Mass/momentum conservation
        # 3. Spectral Evolution - Frequency content analysis
        # 4. Temporal Statistics - Error accumulation tracking
        training_diagnostics.run_comprehensive_diagnostics(
            model=net,
            dimension=DIMENSION,
            tmax=tmax
        )
        
        print("\n" + "="*70)
        print("All diagnostic plots generated successfully!")
        print("Check ./diagnostics/ folder for:")
        print("  1. training_diagnostics.png - Training convergence")
        print("  2. residual_heatmaps.png - Spatiotemporal PDE violations")
        print("  3. conservation_laws.png - Physical consistency")
        print("  4. spectral_evolution.png - Frequency content")
        print("  5. temporal_statistics.png - Field evolution")
        print("="*70 + "\n")
    except Exception as e:
        print(f"[WARNING] Post-training diagnostics failed: {e}")
        print("Training completed successfully, but diagnostic plots may be incomplete.")
        import traceback
        traceback.print_exc()

# Comprehensive GPU memory cleanup after training
# This properly frees model parameters, optimizer state, collocation points, etc.
if device.startswith('cuda'):
    # Collect all training-related data for cleanup
    cleanup_optimizers = [optimizer, optimizerL] if 'optimizer' in locals() and 'optimizerL' in locals() else None
    
    # Try to get collocation data references
    try:
        cleanup_collocation = {
            'domain': collocation_domain if 'collocation_domain' in locals() else None,
            'IC': collocation_IC if 'collocation_IC' in locals() else None,
            'poisson_IC': collocation_poisson_ic if 'collocation_poisson_ic' in locals() else None
        }
    except:
        cleanup_collocation = None
    
    # Perform comprehensive cleanup
    # This frees GPU memory by deleting optimizer state and collocation data
    # Models are temporarily moved to CPU during cleanup, then moved back to GPU for visualization
    _comprehensive_gpu_cleanup(
        nets=[net],
        optimizers=cleanup_optimizers,
        collocation_data=cleanup_collocation,
        cached_ic_values=None,
        keep_models_for_vis=True,  # Keep models for visualization
        target_device='cpu'  # Temporarily move models to CPU during cleanup
    )
    
    # Move model back to GPU for visualization (visualization code expects model on GPU)
    # The FD solver will clear cache before it runs, so this is safe
    if device.startswith('cuda'):
        _move_models_to_device([net], device)
        print("Model moved back to GPU for visualization")

# ==================== VISUALIZATION ====================
initial_params = (xmin, xmax, ymin, ymax, rho_1, alpha, lam, "temp", tmax)

anim_density = create_2d_animation(net, initial_params, which="density", fps=10, verbose=False)
anim_velocity = create_2d_animation(net, initial_params, which="velocity", fps=10, verbose=False)

if str(PERTURBATION_TYPE).lower() == "sinusoidal":
    create_1d_cross_sections_sinusoidal(net, initial_params, time_points=None, y_fixed=SLICE_Y, N_fd=600, nu_fd=0.5)

if PLOT_DENSITY_GROWTH:
    try:
        tmax_growth = float(GROWTH_PLOT_TMAX)
    except Exception:
        tmax_growth = float(TMAX_CFG)
    dt_growth = float(GROWTH_PLOT_DT)
    create_density_growth_plot(net, initial_params, tmax=tmax_growth, dt=dt_growth)

# ==================== FINAL CACHE CLEANUP ====================
script_root = os.path.dirname(os.path.abspath(__file__))
print("Performing final Python cache cleanup...")
clean_pycache(script_root)

