"""
Training loop management for Physics-Informed Neural Networks (PINNs).

This module handles the optimization loops (Adam and L-BFGS) for PINN training,
including batched collocation point sampling and training diagnostics.
"""
import os
import numpy as np
import torch
from training.training_diagnostics import TrainingDiagnostics, _compute_ic_component_losses
from config import BATCH_SIZE, NUM_BATCHES, ENABLE_TRAINING_DIAGNOSTICS
from training.physics import closure_batched, closure_batched_cached, _random_batch_indices, _make_batch_tensors


def train(experiment, iteration_adam, iterationL, data_terms=None):
    """
    Standard training loop for single PINN.
    
    Manages Adam and L-BFGS optimization phases with cosine scheduling for
    continuity weight and startup_dt parameters.
    
    Args:
        experiment: Experiment dataclass containing all training state and parameters
        iteration_adam: Number of Adam iterations
        iterationL: Number of L-BFGS iterations
        data_terms: Optional list of additional supervised datasets with weights
    
    Returns:
        TrainingDiagnostics: Diagnostics object with training history
    """
    # Unpack experiment fields for convenience
    model = experiment.model
    net = experiment.net
    collocation_domain = experiment.collocation_domain
    collocation_IC = experiment.collocation_IC
    optimizer = experiment.optimizer
    optimizerL = experiment.optimizer_lbfgs
    mse_cost_function = experiment.mse_cost_function
    rho_1 = experiment.physics_params['rho_1']
    lam = experiment.physics_params['lam']
    jeans = experiment.physics_params['jeans']
    v_1 = experiment.physics_params['v_1']
    device = experiment.device
    diagnostics_dir = os.path.join(experiment.run_dir, "diagnostics")
    interpolators = experiment.interpolators
    
    bs = int(BATCH_SIZE)
    nb = int(NUM_BATCHES)

    # Import config values for diagnostics
    from config import PERTURBATION_TYPE
    
    # Create diagnostics with correct dimension, perturbation type, and save directory
    diagnostics = TrainingDiagnostics(
        save_dir=diagnostics_dir,
        dimension=model.dimension,
        perturbation_type=PERTURBATION_TYPE
    ) if ENABLE_TRAINING_DIAGNOSTICS else None

    # ----- Adam optimization phase -----
    for i in range(iteration_adam):
        loss, loss_breakdown = optimizer.step(lambda: closure_batched(
            model, net, mse_cost_function, collocation_domain, collocation_IC, 
            optimizer, rho_1, lam, jeans, v_1, bs, nb, update_tracker=True, 
            iteration=i, use_fft_poisson=True, data_terms=data_terms, 
            interpolators=interpolators
        ))

        # Adaptive collocation: resample domain points periodically
        if experiment.adaptive_config is not None and i > 0 \
                and i % experiment.adaptive_config.resample_every_n == 0:
            from methods.adaptive_collocation import resample_collocation
            collocation_domain = resample_collocation(
                net, collocation_domain, model, experiment.adaptive_config
            )
            experiment.collocation_domain = collocation_domain  # keep experiment in sync

        with torch.autograd.no_grad():
            _log_diagnostics(diagnostics, i, net, loss, loss_breakdown, collocation_domain, 
                           collocation_IC, rho_1, lam, jeans, v_1, model.dimension, mse_cost_function)
            _print_progress(i, loss, loss_breakdown, model.dimension, optimizer_type="Adam", print_every=100)

    # Resample once with the fully-Adam-trained net before L-BFGS cached indexing locks in
    if experiment.adaptive_config is not None:
        from methods.adaptive_collocation import resample_collocation
        collocation_domain = resample_collocation(
            net, collocation_domain, model, experiment.adaptive_config
        )
        experiment.collocation_domain = collocation_domain

    # ----- L-BFGS optimization phase -----
    for i in range(iterationL):
        # Cache batch indices for L-BFGS (ensures consistent loss landscape during line search)
        cached_dom_indices, cached_ic_indices = _generate_cached_indices(
            collocation_domain, collocation_IC, bs, nb
        )
        
        # L-BFGS closure: returns only scalar loss, stores breakdown in holder
        loss_breakdown_holder = [None]
        
        def lbfgs_closure():
            optimizerL.zero_grad()
            loss, loss_breakdown = closure_batched_cached(
                model, net, mse_cost_function, collocation_domain, collocation_IC,
                optimizerL, rho_1, lam, jeans, v_1, bs, nb,
                cached_dom_indices, cached_ic_indices,
                data_terms=data_terms,
                interpolators=interpolators
            )
            loss_breakdown_holder[0] = loss_breakdown
            return loss
        
        loss = optimizerL.step(lbfgs_closure)
        loss_breakdown = loss_breakdown_holder[0]

        with torch.autograd.no_grad():
            global_step = iteration_adam + i
            _log_diagnostics(diagnostics, global_step, net, loss, loss_breakdown, collocation_domain,
                           collocation_IC, rho_1, lam, jeans, v_1, model.dimension, mse_cost_function,
                           is_lbfgs=True)
            _print_progress(i, loss, loss_breakdown, model.dimension, optimizer_type="LBFGS", print_every=50)
    
    # ----- Final diagnostics -----
    _run_final_diagnostics(diagnostics, net, model, collocation_IC, rho_1, lam, jeans, v_1, 
                          iteration_adam, iterationL)
    
    return diagnostics


def _log_diagnostics(diagnostics, iteration, net, loss, loss_breakdown, collocation_domain,
                     collocation_IC, rho_1, lam, jeans, v_1, dimension, mse_cost_function, 
                     is_lbfgs=False):
    """Log diagnostics every 50 iterations."""
    if iteration % 50 != 0 or diagnostics is None:
        return
    
    try:
        # Compute IC component losses for 3D power spectrum tracking
        ic_component_losses = None
        if diagnostics.is_3d_power_spectrum:
            ic_component_losses = _compute_ic_component_losses(
                net, collocation_IC, rho_1, lam, jeans, v_1, dimension, mse_cost_function
            )
        
        diagnostics.log_iteration(
            iteration=iteration,
            model=net,
            loss_dict={
                'total': loss.item() if hasattr(loss, 'item') else float(loss),
                'pde': float(loss_breakdown.get('PDE', 0.0)),
                'ic': float(loss_breakdown.get('IC', 0.0))
            },
            geomtime_col=collocation_domain,
            ic_component_losses=ic_component_losses
        )
    except Exception as _diag_err:
        phase = "LBFGS" if is_lbfgs else "Adam"
        print(f"[WARN] Diagnostics logging ({phase}) failed at {iteration}: {_diag_err}")


def _print_progress(iteration, loss, loss_breakdown, dimension, optimizer_type="Adam", print_every=100):
    """Print training progress at specified intervals."""
    if iteration % print_every != 0:
        return
    
    loss_value = loss.item() if hasattr(loss, 'item') else float(loss)
    print(f"Training Loss at {iteration} for {optimizer_type} (batched) in {dimension}D system = {loss_value:.2e}", flush=True)
    
    # Print loss breakdown
    breakdown_str = " | ".join([f"{k}: {v:.2e}" for k, v in loss_breakdown.items() if v > 0])
    if breakdown_str:
        print(f"  Loss breakdown: {breakdown_str}", flush=True)


def _generate_cached_indices(collocation_domain, collocation_IC, batch_size, num_batches):
    """Generate cached batch indices for L-BFGS optimization."""
    # Determine domain size and device
    if isinstance(collocation_domain, (list, tuple)):
        dom_n = collocation_domain[0].size(0)
        device = collocation_domain[0].device
    else:
        dom_n = collocation_domain.size(0)
        device = collocation_domain.device
    
    ic_n = collocation_IC[0].size(0)
    
    # Generate cached indices
    cached_dom_indices = [_random_batch_indices(dom_n, batch_size, device) for _ in range(int(max(1, num_batches)))]
    cached_ic_indices = [_random_batch_indices(ic_n, batch_size, device) for _ in range(int(max(1, num_batches)))]
    
    return cached_dom_indices, cached_ic_indices


def _run_final_diagnostics(diagnostics, net, model, collocation_IC, rho_1, lam, jeans, v_1, 
                           iteration_adam, iterationL):
    """Generate comprehensive diagnostic plots at the end of training."""
    if diagnostics is None:
        return
    
    try:
        final_iteration = iteration_adam + iterationL - 1
        
        print("\n[INFO] Generating unified training diagnostics...")
        from core.initial_conditions import fun_rho_0, fun_vx_0, fun_vy_0, fun_vz_0
        
        # Evaluate true ICs on the IC collocation points for comparison
        with torch.no_grad():
            true_ic_data = {
                'colloc_IC': collocation_IC,
                'rho': fun_rho_0(rho_1, lam, collocation_IC),
                'vx': fun_vx_0(lam, jeans, v_1, collocation_IC),
                'vy': fun_vy_0(lam, jeans, v_1, collocation_IC),
                'vz': fun_vz_0(lam, jeans, v_1, collocation_IC) if model.dimension == 3 else None,
            }
        
        diagnostics.run_unified_diagnostics(
            model=net, 
            true_ic_data=true_ic_data,
            final_iteration=final_iteration
        )
        
        print(f"\n[INFO] All diagnostics saved to {diagnostics.save_dir}")
    except Exception as _diag_err:
        import traceback
        print(f"[WARN] Final diagnostics plotting failed: {_diag_err}")
        traceback.print_exc()
