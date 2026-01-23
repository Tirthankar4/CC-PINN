"""
Training loop management for PINN and XPINN.

This module handles the optimization loops (Adam and L-BFGS) for both
single PINN and XPINN decomposition training.
"""
import numpy as np
import torch
from training.training_diagnostics import TrainingDiagnostics
from config import BATCH_SIZE, NUM_BATCHES, ENABLE_TRAINING_DIAGNOSTICS
from training.physics import closure_batched


def _compute_ic_component_losses(net, collocation_IC, rho_1, lam, jeans, v_1, dimension, mse_cost_function):
    """
    Compute individual IC component losses for 3D power spectrum diagnostics.
    
    Returns:
        Dict with component-wise losses: {'rho', 'vx', 'vy', 'vz', 'phi'}
    """
    from core.initial_conditions import fun_rho_0, fun_vx_0, fun_vy_0, fun_vz_0
    from config import rho_o
    
    with torch.no_grad():
        # Get true ICs
        rho_0 = fun_rho_0(rho_1, lam, collocation_IC)
        vx_0 = fun_vx_0(lam, jeans, v_1, collocation_IC)
        vy_0 = fun_vy_0(lam, jeans, v_1, collocation_IC)
        vz_0 = fun_vz_0(lam, jeans, v_1, collocation_IC)
        
        # Get predictions
        net_ic_out = net(collocation_IC)
        rho_ic_out = net_ic_out[:, 0:1]
        vx_ic_out = net_ic_out[:, 1:2]
        vy_ic_out = net_ic_out[:, 2:3]
        vz_ic_out = net_ic_out[:, 3:4]
        phi_ic_out = net_ic_out[:, 4:5]
        
        # Compute component losses
        ic_losses = {
            'rho': mse_cost_function(rho_ic_out, rho_0).item(),
            'vx': mse_cost_function(vx_ic_out, vx_0).item(),
            'vy': mse_cost_function(vy_ic_out, vy_0).item(),
            'vz': mse_cost_function(vz_ic_out, vz_0).item(),
            'phi': torch.mean(phi_ic_out ** 2).item(),  # phi should be close to 0 ideally
        }
    
    return ic_losses


def train(model, net, collocation_domain, collocation_IC, optimizer, optimizerL, iteration_adam, iterationL, mse_cost_function, closure, rho_1, lam, jeans, v_1, device, data_terms=None, collocation_poisson_ic=None):
    """
    Standard training loop for single PINN.
    
    Manages Adam and L-BFGS optimization phases with cosine scheduling for
    continuity weight and startup_dt parameters.
    
    Args:
        model: Collocation model
        net: Neural network
        collocation_domain: Domain collocation points
        collocation_IC: Initial condition points
        optimizer: Adam optimizer
        optimizerL: L-BFGS optimizer
        iteration_adam: Number of Adam iterations
        iterationL: Number of L-BFGS iterations
        mse_cost_function: MSE loss function
        closure: Closure function (unused, kept for compatibility)
        rho_1: Perturbation amplitude
        lam: Wavelength
        jeans: Jeans length
        v_1: Velocity amplitude
        device: PyTorch device
        data_terms: Optional list of additional supervised datasets with weights
        collocation_poisson_ic: Extra collocation points at t=0 for Poisson enforcement
    """
    bs = int(BATCH_SIZE)
    nb = int(NUM_BATCHES)

    # Import config values for diagnostics
    from config import PERTURBATION_TYPE, tmax
    
    # Create diagnostics with correct dimension and perturbation type
    diagnostics = TrainingDiagnostics(
        save_dir='./diagnostics/',
        dimension=model.dimension,
        perturbation_type=PERTURBATION_TYPE
    ) if ENABLE_TRAINING_DIAGNOSTICS else None

    for i in range(iteration_adam):
        optimizer.zero_grad()

        loss, loss_breakdown = optimizer.step(lambda: closure_batched(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer, rho_1, lam, jeans, v_1, bs, nb, update_tracker=True, iteration=i, use_fft_poisson=True, data_terms=data_terms, collocation_poisson_ic=collocation_poisson_ic))

        with torch.autograd.no_grad():
            # Diagnostics logging every 50 iterations
            if i % 50 == 0 and diagnostics is not None:
                try:
                    # Compute IC component losses for 3D power spectrum tracking
                    ic_component_losses = None
                    if diagnostics.is_3d_power_spectrum:
                        ic_component_losses = _compute_ic_component_losses(
                            net, collocation_IC, rho_1, lam, jeans, v_1, model.dimension, mse_cost_function
                        )
                    
                    diagnostics.log_iteration(
                        iteration=i,
                        model=net,
                        loss_dict={
                            'total': loss.item(),
                            'pde': float(loss_breakdown.get('PDE', 0.0)),
                            'ic': float(loss_breakdown.get('IC', 0.0))
                        },
                        geomtime_col=collocation_domain,
                        ic_component_losses=ic_component_losses
                    )
                except Exception as _diag_err:
                    # Keep training robust if diagnostics fail
                    print(f"[WARN] Diagnostics logging failed at {i}: {_diag_err}")

            if i % 100 == 0:
                print(f"Training Loss at {i} for Adam (batched) in {model.dimension}D system = {loss.item():.2e}", flush=True)
                # Print loss breakdown
                breakdown_str = " | ".join([f"{k}: {v:.2e}" for k, v in loss_breakdown.items() if v > 0])
                if breakdown_str:
                    print(f"  Loss breakdown: {breakdown_str}", flush=True)

    for i in range(iterationL):
        optimizer.zero_grad()
        global_step = iteration_adam + i

        # L-BFGS expects a closure that returns only scalar loss
        # Store loss_breakdown in a list so we can access it after the step
        loss_breakdown_holder = [None]
        
        def lbfgs_closure():
            loss, loss_breakdown = closure_batched(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizerL, rho_1, lam, jeans, v_1, bs, nb, update_tracker=False, iteration=global_step, use_fft_poisson=False, data_terms=data_terms, collocation_poisson_ic=collocation_poisson_ic)
            loss_breakdown_holder[0] = loss_breakdown
            return loss
        
        loss = optimizerL.step(lbfgs_closure)
        loss_breakdown = loss_breakdown_holder[0]

        with torch.autograd.no_grad():
            # Diagnostics logging every 50 iterations in LBFGS too
            if i % 50 == 0 and loss_breakdown is not None and diagnostics is not None:
                try:
                    # Compute IC component losses for 3D power spectrum tracking
                    ic_component_losses = None
                    if diagnostics.is_3d_power_spectrum:
                        ic_component_losses = _compute_ic_component_losses(
                            net, collocation_IC, rho_1, lam, jeans, v_1, model.dimension, mse_cost_function
                        )
                    
                    diagnostics.log_iteration(
                        iteration=iteration_adam + i,
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
                    print(f"[WARN] Diagnostics logging (LBFGS) failed at {i}: {_diag_err}")
            if i % 50 == 0:
                print(f"Training Loss at {i} for LBGFS (batched) in {model.dimension}D system = {loss.item():.2e}", flush=True)
                # Print loss breakdown
                breakdown_str = " | ".join([f"{k}: {v:.2e}" for k, v in loss_breakdown.items() if v > 0])
                if breakdown_str:
                    print(f"  Loss breakdown: {breakdown_str}", flush=True)
    
    # Generate diagnostic plots at the end of training
    if diagnostics is not None:
        try:
            final_iteration = iteration_adam + iterationL - 1
            
            # Generate unified diagnostics (5 plots for all cases)
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
            
            print(f"\n[INFO] All diagnostics saved to ./diagnostics/")
        except Exception as _diag_err:
            import traceback
            print(f"[WARN] Final diagnostics plotting failed: {_diag_err}")
            traceback.print_exc()
    
    # Return diagnostics object for potential reuse in post-training analysis
    return diagnostics