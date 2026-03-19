# Standard library imports
import os
import sys
import shutil
import time
import hashlib
from datetime import datetime
from typing import Tuple, Optional, List
from dataclasses import dataclass

# Third-party imports
import numpy as np
import torch
import torch.nn as nn

# Local imports - Core modules
from core.device import has_cuda_gpu, has_mps_backend, get_compute_device, clear_cuda_cache
from core.data_generator import input_taker, req_consts_calc
from core.initial_conditions import initialize_shared_velocity_fields, VelocityFieldInterpolators
from core.losses import ASTPN
from core.model_architecture import PINN

# Local imports - Training modules
from training.trainer import train

# Local imports - Configuration
from config import (
    BATCH_SIZE, NUM_BATCHES, N_0, N_r, DIMENSION,
    a, wave, cs, xmin, ymin, zmin, tmin,
    tmax as TMAX_CFG, iteration_adam_2D, iteration_lbgfs_2D,
    harmonics, PERTURBATION_TYPE, rho_o, SLICE_Y,
    num_neurons, num_layers, num_of_waves,
    RANDOM_SEED, STARTUP_DT,
    ENABLE_TRAINING_DIAGNOSTICS,
    PLOT_DENSITY_GROWTH, GROWTH_PLOT_TMAX, GROWTH_PLOT_DT,
    FD_N_2D,
    ENABLE_INTERACTIVE_3D, INTERACTIVE_3D_RESOLUTION, INTERACTIVE_3D_TIME_STEPS
)

from config import AdaptiveCollocationConfig

@dataclass
class Experiment:
    """Container for experiment state, replacing 16+ function arguments."""
    # Model and collocation
    net: PINN
    model: ASTPN
    collocation_domain: List[torch.Tensor]
    collocation_IC: List[torch.Tensor]

    # Optimizers
    optimizer: torch.optim.Adam
    optimizer_lbfgs: torch.optim.LBFGS

    # Physics parameters
    physics_params: dict  # {rho_1, lam, jeans, v_1, alpha, ...}

    # Device and loss
    device: str
    mse_cost_function: nn.MSELoss

    # Additional metadata
    initial_params: tuple  # (xmin, xmax, ymin, ymax, rho_1, alpha, lam, "temp", tmax)

    # Run tracking (Phase D)
    run_id: str  # Unique identifier for this run (timestamp_perturbation_dimension)
    run_dir: str  # Absolute path to run directory

    # Velocity field interpolators (Phase E) - required for power spectrum ICs
    interpolators: Optional[VelocityFieldInterpolators] = None

    training_diagnostics: Optional[object] = None  # Set after training completes

    adaptive_config: Optional[AdaptiveCollocationConfig] = None

def _generate_run_id(perturbation_type: str, dimension: int, include_hash: bool = True) -> str:
    """
    Generate a unique run ID for experiment tracking.

    Format: {timestamp}_{perturbation_type}_{dimension}D[_{config_hash}]
    Example: 20260215_214130_power_spectrum_2D_a3f5b2c1

    Args:
        perturbation_type: Type of perturbation (e.g., 'power_spectrum', 'sinusoidal')
        dimension: Spatial dimension (1, 2, or 3)
        include_hash: If True, append first 8 chars of config hash

    Returns:
        Unique run identifier string
    """
    from config import CONFIG

    # Timestamp: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Perturbation type: lowercase with underscores
    pert_type_clean = str(perturbation_type).lower().replace(" ", "_")

    # Base run ID
    run_id = f"{timestamp}_{pert_type_clean}_{dimension}D"

    # Optional: Add config hash for quick comparison
    if include_hash:
        config_dict = CONFIG.to_dict()
        config_str = str(sorted(config_dict.items()))
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        run_id += f"_{config_hash}"

    return run_id


def _create_run_directory(base_dir: str, run_id: str) -> str:
    """
    Create directory structure for a training run.

    Structure:
        base_dir/
        └── runs/
            └── {run_id}/
                ├── diagnostics/  (created here)
                ├── model.pth     (saved later)
                └── config.yaml   (saved later)

    Args:
        base_dir: Base output directory (from config)
        run_id: Unique run identifier

    Returns:
        Absolute path to the run directory
    """
    run_dir = os.path.join(base_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Create subdirectories
    diagnostics_dir = os.path.join(run_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)

    print(f"Created run directory: {run_dir}")
    return run_dir


def _configure_torch_runtime():
    """Set PyTorch runtime performance flags for training runs."""
    # Enable cuDNN autotuner - finds optimal convolution algorithms
    torch.backends.cudnn.benchmark = True

    # Enable TF32 on Ampere GPUs (A100, RTX 3090, RTX 4090, etc.)
    if has_cuda_gpu():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Prefer explicit device/dtype settings rather than deprecated tensor-type override
        torch.set_default_dtype(torch.float32)
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cuda")
        print("PyTorch performance optimizations enabled (cuDNN benchmark, TF32)")


def _save_trained_models(net, run_dir: str, adaptive_config=None):
    """Persist trained model, full config, and (if active) adaptive collocation config.

    Saves:
    - ``model.pth``                        — network state dict
    - ``config.yaml``                      — full SimulationConfig
    - ``adaptive_collocation_config.json`` — only when adaptive collocation is enabled

    Args:
        net: Trained PINN model
        run_dir: Absolute path to the run directory
        adaptive_config: AdaptiveCollocationConfig instance, or None
    """
    import json
    from dataclasses import asdict

    try:
        from config import CONFIG

        # Save model weights to run directory
        model_path = os.path.join(run_dir, "model.pth")
        torch.save(net.state_dict(), model_path)
        print(f"Saved model to {model_path}")

        # Save full config for reproducibility
        config_path = os.path.join(run_dir, "config.yaml")
        CONFIG.to_yaml(config_path)
        print(f"Saved config to {config_path}")

        # Save adaptive collocation config separately when it was active
        if adaptive_config is not None and adaptive_config.enabled:
            ac_path = os.path.join(run_dir, "adaptive_collocation_config.json")
            with open(ac_path, "w") as f:
                json.dump(asdict(adaptive_config), f, indent=2)
            print(f"Saved adaptive collocation config to {ac_path}")

    except Exception as e:
        print(f"Warning: failed to save model/config: {e}")


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


def _setup_device():
    """Setup and configure compute device (CUDA/MPS/CPU)."""
    has_gpu = has_cuda_gpu()
    has_mps = has_mps_backend()
    device = get_compute_device()
    print(f"Selected device: {device} (CUDA={has_gpu}, MPS={has_mps})")

    if device.startswith('cuda'):
        clear_cuda_cache()

    return device


def _compute_physics_parameters():
    """Compute domain extents and physical parameters."""
    lam, rho_1, num_waves, tmax, _, _, _ = input_taker(wave, a, num_of_waves, TMAX_CFG, N_0, 0, N_r)
    jeans, alpha = req_consts_calc(lam, rho_1)

    # Set initial velocity amplitude per perturbation type
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        k = 2 * np.pi / lam
        v_1 = (rho_1 / (rho_o if rho_o != 0 else 1.0)) * (alpha / k)
    else:
        v_1 = a * cs

    xmax = xmin + lam * num_waves
    ymax = ymin + lam * num_waves
    zmax = zmin + lam * num_waves

    return {
        'lam': lam,
        'rho_1': rho_1,
        'num_waves': num_waves,
        'tmax': tmax,
        'jeans': jeans,
        'alpha': alpha,
        'v_1': v_1,
        'xmax': xmax,
        'ymax': ymax,
        'zmax': zmax,
    }


def _initialize_velocity_fields(lam, num_waves, v_1):
    """Initialize velocity field interpolators for power spectrum perturbations."""
    interpolators = None

    if str(PERTURBATION_TYPE).lower() == "power_spectrum":
        if DIMENSION == 3:
            vx_np, vy_np, vz_np, interpolators = initialize_shared_velocity_fields(
                lam, num_waves, v_1, seed=RANDOM_SEED, dimension=3
            )
        else:
            vx_np, vy_np, interpolators = initialize_shared_velocity_fields(
                lam, num_waves, v_1, seed=RANDOM_SEED, dimension=2
            )

            # Set shared velocity fields for 2D plotting
            if DIMENSION == 2:
                from visualization.Plotting import set_shared_velocity_fields
                set_shared_velocity_fields(vx_np, vy_np)

    return interpolators


def _create_model_and_optimizers(device):
    """Create PINN model and optimizers."""
    print("Running in standard PINN mode (single network)...")

    net = PINN(dimension=DIMENSION, n_harmonics=harmonics)
    net = net.to(device)
    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    optimizerL = torch.optim.LBFGS(net.parameters(), line_search_fn='strong_wolfe')

    return net, mse_cost_function, optimizer, optimizerL


def _generate_collocation_points(params, device):
    """Generate collocation points for IC, domain, and optional Poisson enforcement."""
    # Build dimension-specific domain bounds
    if DIMENSION == 1:
        astpn_rmin = [xmin, tmin]
        astpn_rmax = [params['xmax'], params['tmax']]
        spatial_rmin, spatial_rmax = [xmin], [params['xmax']]
    elif DIMENSION == 2:
        astpn_rmin = [xmin, ymin, tmin]
        astpn_rmax = [params['xmax'], params['ymax'], params['tmax']]
        spatial_rmin, spatial_rmax = [xmin, ymin], [params['xmax'], params['ymax']]
    elif DIMENSION == 3:
        astpn_rmin = [xmin, ymin, zmin, tmin]
        astpn_rmax = [params['xmax'], params['ymax'], params['zmax'], params['tmax']]
        spatial_rmin = [xmin, ymin, zmin]
        spatial_rmax = [params['xmax'], params['ymax'], params['zmax']]
    else:
        raise ValueError(f"Unsupported DIMENSION={DIMENSION}")

    # Create collocation model
    collocation_model = ASTPN(
        rmin=astpn_rmin,
        rmax=astpn_rmax,
        N_0=N_0,
        N_b=0,
        N_r=N_r,
        dimension=DIMENSION
    )

    # Generate collocation points
    collocation_IC = collocation_model.geo_time_coord(option="IC")
    collocation_domain = collocation_model.geo_time_coord(option="Domain")

    return {
        'model': collocation_model,
        'IC': collocation_IC,
        'domain': collocation_domain,
        'spatial_bounds': (spatial_rmin, spatial_rmax),
    }


def build_experiment() -> Experiment:
    """
    Build complete experiment by coordinating setup of all components.

    Returns:
        Experiment: Container with all experiment state
    """
    # Device setup
    device = _setup_device()

    # Physics parameters and domain bounds
    params = _compute_physics_parameters()

    # Velocity field interpolators (power spectrum only)
    interpolators = _initialize_velocity_fields(params['lam'], params['num_waves'], params['v_1'])

    # Model and optimizers
    net, mse_cost_function, optimizer, optimizerL = _create_model_and_optimizers(device)

    # Collocation points
    collocation = _generate_collocation_points(params, device)

    # Set periodic boundary conditions via domain
    spatial_rmin, spatial_rmax = collocation['spatial_bounds']
    net.set_domain(rmin=spatial_rmin, rmax=spatial_rmax, dimension=DIMENSION)

    # Run tracking
    from config import SNAPSHOT_DIR, CONFIG
    run_id = _generate_run_id(PERTURBATION_TYPE, DIMENSION, include_hash=True)
    run_dir = _create_run_directory(SNAPSHOT_DIR, run_id)
    print(f"Run ID: {run_id}")

    # Adaptive collocation config (driven entirely from config.yaml / CONFIG)
    adaptive_config = CONFIG.adaptive_collocation if CONFIG.adaptive_collocation.enabled else None
    if adaptive_config is not None:
        print(f"[AdaptiveColloc] Enabled — resampling every {adaptive_config.resample_every_n} Adam iterations")

    # Build experiment container
    return Experiment(
        net=net,
        model=collocation['model'],
        collocation_domain=collocation['domain'],
        collocation_IC=collocation['IC'],
        optimizer=optimizer,
        optimizer_lbfgs=optimizerL,
        physics_params={
            'rho_1': params['rho_1'],
            'lam': params['lam'],
            'jeans': params['jeans'],
            'v_1': params['v_1'],
            'alpha': params['alpha'],
            'tmax': params['tmax'],
        },
        device=device,
        mse_cost_function=mse_cost_function,
        initial_params=(xmin, params['xmax'], ymin, params['ymax'], params['rho_1'], params['alpha'], params['lam'], "temp", params['tmax']),
        run_id=run_id,
        run_dir=run_dir,
        interpolators=interpolators,
        adaptive_config=adaptive_config,
    )


def run_training(experiment: Experiment) -> None:
    """
    Run training: calls the training function (currently train()).

    Args:
        experiment: Experiment container with all necessary state
    """
    print("Using standard training...")
    start_time = time.time()

    training_diagnostics = train(
        experiment=experiment,
        iteration_adam=iteration_adam_2D,
        iterationL=iteration_lbgfs_2D
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[Timing] PINN training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Store training diagnostics in experiment for post-training use
    experiment.training_diagnostics = training_diagnostics


def post_training(experiment: Experiment) -> None:
    """
    Post-training: model saving, GPU cleanup, diagnostics, visualization.

    Args:
        experiment: Experiment container with all necessary state
    """
    # ==================== MODEL SAVING ====================
    _save_trained_models(experiment.net, experiment.run_dir, experiment.adaptive_config)

    # ==================== GPU CLEANUP ====================
    # Comprehensive GPU memory cleanup after training
    # This properly frees model parameters, optimizer state, collocation points, etc.
    if experiment.device.startswith('cuda'):
        cleanup_optimizers = [experiment.optimizer, experiment.optimizer_lbfgs]

        try:
            cleanup_collocation = {
                'domain': experiment.collocation_domain,
                'IC': experiment.collocation_IC
            }
        except Exception:
            cleanup_collocation = None

        _comprehensive_gpu_cleanup(
            nets=[experiment.net],
            optimizers=cleanup_optimizers,
            collocation_data=cleanup_collocation,
            cached_ic_values=None,
            keep_models_for_vis=True,  # Keep models for visualization
            target_device='cpu'  # Temporarily move models to CPU during cleanup
        )

        # Move model back to GPU for visualization (visualization code expects model on GPU)
        _move_models_to_device([experiment.net], experiment.device)
        print("Model moved back to GPU for visualization")

    # ==================== POST-TRAINING COMPREHENSIVE DIAGNOSTICS ====================
    # Run comprehensive diagnostics for high-tmax failure analysis
    # This generates 4 additional critical plots beyond the training diagnostics
    if ENABLE_TRAINING_DIAGNOSTICS and experiment.training_diagnostics is not None:
        print("\n" + "=" * 70)
        print("Running post-training comprehensive diagnostics...")
        print("=" * 70)

        try:
            experiment.net.eval()
            # Run comprehensive diagnostics - generates 4 additional plots:
            # 1. PDE Residual Heatmaps - WHERE/WHEN physics breaks
            # 2. Conservation Laws - Mass/momentum conservation
            # 3. Spectral Evolution - Frequency content analysis
            # 4. Temporal Statistics - Error accumulation tracking
            experiment.training_diagnostics.run_comprehensive_diagnostics(
                model=experiment.net,
                dimension=DIMENSION,
                tmax=experiment.physics_params['tmax']
            )

            print("\n" + "=" * 70)
            print("All diagnostic plots generated successfully!")
            diagnostics_path = os.path.join(experiment.run_dir, "diagnostics")
            print(f"Check {diagnostics_path} folder for:")
            print("  1. training_diagnostics.png - Training convergence")
            print("  2. residual_heatmaps.png - Spatiotemporal PDE violations")
            print("  3. conservation_laws.png - Physical consistency")
            print("  4. spectral_evolution.png - Frequency content")
            print("  5. temporal_statistics.png - Field evolution")
            print("=" * 70 + "\n")
        except Exception as e:
            print(f"[WARNING] Post-training diagnostics failed: {e}")
            print("Training completed successfully, but diagnostic plots may be incomplete.")
            import traceback
            traceback.print_exc()

    from config import TRAINING_ONLY
    if TRAINING_ONLY:
        print("\n" + "=" * 70)
        print("training_only=True: skipping all post-training visualization.")
        print("Model and config have been saved. Download from run directory.")
        print("=" * 70 + "\n")
        # Still do final cache cleanup
        script_root = os.path.dirname(os.path.abspath(__file__))
        print("Performing final Python cache cleanup...")
        clean_pycache(script_root)
        return

    # ==================== VISUALIZATION ====================
    from visualization.Plotting import (
        create_2d_animation,
        create_1d_cross_sections_sinusoidal,
        create_density_growth_plot,
    )

    create_2d_animation(experiment.net, experiment.initial_params, which="density", fps=10, verbose=False)
    create_2d_animation(experiment.net, experiment.initial_params, which="velocity", fps=10, verbose=False)

    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        create_1d_cross_sections_sinusoidal(experiment.net, experiment.initial_params, time_points=None, y_fixed=SLICE_Y, N_fd=600, nu_fd=0.5)

    if PLOT_DENSITY_GROWTH:
        try:
            tmax_growth = float(GROWTH_PLOT_TMAX)
        except Exception:
            tmax_growth = float(TMAX_CFG)
        dt_growth = float(GROWTH_PLOT_DT)
        create_density_growth_plot(experiment.net, experiment.initial_params, tmax=tmax_growth, dt=dt_growth)

    # ==================== 3D INTERACTIVE VISUALIZATION ====================
    if DIMENSION == 3 and ENABLE_INTERACTIVE_3D:
        try:
            from visualization.Interactive import create_interactive_3d_plot
            from config import SNAPSHOT_DIR

            print("\n" + "=" * 70)
            print("Generating 3D interactive plot...")
            print("=" * 70)

            save_path = os.path.join(SNAPSHOT_DIR, "interactive_3d_plot.html")
            create_interactive_3d_plot(
                net=experiment.net,
                initial_params=experiment.initial_params,
                time_range=(tmin, experiment.physics_params['tmax']),
                time_steps=INTERACTIVE_3D_TIME_STEPS,
                resolution=INTERACTIVE_3D_RESOLUTION,
                save_path=save_path,
            )

            print(f"Interactive 3D plot saved to: {save_path}")
            print("=" * 70 + "\n")
        except Exception as e:
            print(f"[WARNING] 3D interactive plot generation failed: {e}")
            import traceback
            traceback.print_exc()

    # ==================== FINAL CACHE CLEANUP ====================
    script_root = os.path.dirname(os.path.abspath(__file__))
    print("Performing final Python cache cleanup...")
    clean_pycache(script_root)


def _run_training_pipeline():
    """Training pipeline: build experiment, run training, post-process results."""
    experiment = build_experiment()
    run_training(experiment)
    post_training(experiment)



def main(argv=None):
    """Entry point for train script execution."""
    if argv is None:
        argv = sys.argv[1:]

    if "--clean-pycache" in argv or "--clean-cache" in argv:
        script_root = os.path.dirname(os.path.abspath(__file__))
        print(f"Cleaning Python cache files from: {script_root}")
        clean_pycache(script_root)
        return

    _configure_torch_runtime()
    _run_training_pipeline()


if __name__ == "__main__":
    main()
