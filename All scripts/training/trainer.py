"""
Training loop management for PINN and XPINN.

This module handles the optimization loops (Adam and L-BFGS) for both
single PINN and XPINN decomposition training.
"""
import numpy as np
import torch
from training.training_diagnostics import TrainingDiagnostics
from config import BATCH_SIZE, NUM_BATCHES, ENABLE_TRAINING_DIAGNOSTICS
from training.physics import closure_batched, closure_batched_cached, _random_batch_indices, _make_batch_tensors


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
        global_step = iteration_adam + i

        # L-BFGS expects a closure that returns only scalar loss
        # Store loss_breakdown in a list so we can access it after the step
        loss_breakdown_holder = [None]
        
        # Cache batch indices for L-BFGS step (L-BFGS calls closure multiple times)
        # Using same indices ensures consistent loss landscape during line search
        if isinstance(collocation_domain, (list, tuple)):
            dom_n = collocation_domain[0].size(0)
            device = collocation_domain[0].device
        else:
            dom_n = collocation_domain.size(0)
            device = collocation_domain.device
        ic_n = collocation_IC[0].size(0)
        
        cached_dom_indices = [_random_batch_indices(dom_n, bs, device) for _ in range(int(max(1, nb)))]
        cached_ic_indices = [_random_batch_indices(ic_n, bs, device) for _ in range(int(max(1, nb)))]
        
        def lbfgs_closure():
            optimizerL.zero_grad()
            loss, loss_breakdown = closure_batched_cached(
                model, net, mse_cost_function, collocation_domain, collocation_IC,
                optimizerL, rho_1, lam, jeans, v_1, bs, nb,
                cached_dom_indices, cached_ic_indices,
                data_terms=data_terms, collocation_poisson_ic=collocation_poisson_ic
            )
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


def _select_batch(collocation, indices):
    if isinstance(collocation, (list, tuple)):
        return _make_batch_tensors(collocation, indices)
    # Single tensor format: split into [x, y, t]
    return [collocation[indices, i:i+1] for i in range(collocation.shape[1])]


def _phi_slice_index(dimension):
    if dimension == 1:
        return 2
    if dimension == 2:
        return 3
    return 4


def _compute_poisson_loss_from_ic(net, collocation_points, rho_1, lam, dimension):
    """
    Compute Poisson loss using IC density (Stage 1).
    
    The Poisson equation is: ∇²φ = const * (ρ - ρ₀)
    We use the true IC density ρ_ic from fun_rho_0 and the network's predicted φ.
    """
    from core.data_generator import diff
    from core.initial_conditions import fun_rho_0
    from config import const, rho_o
    
    # Ensure coords is a list and enable gradient tracking
    if isinstance(collocation_points, (list, tuple)):
        coords = []
        for coord in collocation_points:
            if not coord.requires_grad:
                coord = coord.requires_grad_(True)
            coords.append(coord)
    else:
        # Single tensor format: split into [x, y, (z), t] and enable gradients
        coords = []
        for i in range(collocation_points.shape[1]):
            coord = collocation_points[:, i:i+1]
            if not coord.requires_grad:
                coord = coord.requires_grad_(True)
            coords.append(coord)
    
    # Forward pass (network expects list format)
    outputs = net(coords)
    phi = outputs[:, _phi_slice_index(dimension):_phi_slice_index(dimension) + 1]

    if not torch.isfinite(phi).all():
        safe = torch.tensor(0.0, device=phi.device, requires_grad=True)
        return safe
    
    # Extract spatial coordinates
    x = coords[0]
    
    # Compute first derivative to detect blow-ups early
    phi_x = diff(phi, x, order=1)
    if torch.isinf(phi_x).any():
        safe = torch.tensor(1e10, device=phi.device, requires_grad=True)
        return safe

    # Check for NaN in phi before computing laplacian
    if torch.isnan(phi).any():
        print(f"[DEBUG] NaN detected in phi output!")
        print(f"  phi shape: {phi.shape}")
        print(f"  phi has NaN: {torch.isnan(phi).sum().item()} / {phi.numel()}")
        try:
            print(f"  phi range: [{phi.min().item():.2e}, {phi.max().item():.2e}]")
        except:
            print(f"  phi min/max computation failed")
    
    # Compute Laplacian and check for NaN
    if dimension == 1:
        laplacian = diff(phi, x, order=2)
    elif dimension == 2:
        y = coords[1]
        laplacian = diff(phi, x, order=2) + diff(phi, y, order=2)
    else:  # dimension == 3
        y = coords[1]
        z = coords[2]
        laplacian = diff(phi, x, order=2) + diff(phi, y, order=2) + diff(phi, z, order=2)
    
    if torch.isnan(laplacian).any():
        pass

    if torch.isinf(laplacian).any():
        laplacian = torch.clamp(laplacian, -1e10, 1e10)
    
    # Get true IC density
    rho_ic = fun_rho_0(rho_1, lam, coords)
    
    if torch.isnan(rho_ic).any():
        pass
    
    # Poisson residual: ∇²φ - const*(ρ - ρ₀)
    residual = laplacian - const * (rho_ic - rho_o)

    if torch.isinf(residual).any():
        # Clamp to avoid NaN in subsequent operations
        residual = torch.clamp(residual, -1e10, 1e10)
    
    # Check for NaN and print debug info
    if torch.isnan(residual).any():
        print(f"[DEBUG] NaN detected in Poisson residual!")
        print(f"  residual has NaN: {torch.isnan(residual).sum().item()} / {residual.numel()}")
        try:
            print(f"  laplacian range: [{laplacian.min().item():.2e}, {laplacian.max().item():.2e}]")
            print(f"  rho_ic range: [{rho_ic.min().item():.2e}, {rho_ic.max().item():.2e}]")
            print(f"  phi range: [{phi.min().item():.2e}, {phi.max().item():.2e}]")
        except Exception as e:
            print(f"  Could not compute ranges: {e}")
        print(f"  const: {const}, rho_o: {rho_o}")
        import sys
        sys.stdout.flush()  # Force flush to ensure output appears
    
    # Normalized loss
    scale = abs(const) * (abs(rho_o) if rho_o != 0 else 1.0)
    scale = max(scale, 1e-10)  # Avoid division by zero
    
    loss = torch.mean((residual / scale) ** 2)
    
    if torch.isnan(loss):
        pass
    if torch.isinf(loss):
        safe = torch.tensor(1e8, device=residual.device, requires_grad=True)
        return safe
    
    return loss


def _compute_ic_loss(net, collocation_ic, rho_1, lam, jeans, v_1, dimension, mse_cost_function):
    from core.initial_conditions import fun_rho_0, fun_vx_0, fun_vy_0, fun_vz_0
    from config import rho_o, PERTURBATION_TYPE, KX, KY

    rho_0 = fun_rho_0(rho_1, lam, collocation_ic)
    vx_0 = fun_vx_0(lam, jeans, v_1, collocation_ic)

    net_ic_out = net(collocation_ic)
    rho_ic_out = net_ic_out[:, 0:1]
    vx_ic_out = net_ic_out[:, 1:2]

    if dimension == 2:
        vy_0 = fun_vy_0(lam, jeans, v_1, collocation_ic)
        vy_ic_out = net_ic_out[:, 2:3]
    elif dimension == 3:
        vy_0 = fun_vy_0(lam, jeans, v_1, collocation_ic)
        vz_0 = fun_vz_0(lam, jeans, v_1, collocation_ic)
        vy_ic_out = net_ic_out[:, 2:3]
        vz_ic_out = net_ic_out[:, 3:4]

    is_sin = str(PERTURBATION_TYPE).lower() == "sinusoidal"
    if is_sin:
        x_ic_for_ic = collocation_ic[0]
        if len(collocation_ic) >= 2:
            y_ic_for_ic = collocation_ic[1]
            rho_ic_target = rho_o + rho_1 * torch.cos(KX * x_ic_for_ic + KY * y_ic_for_ic)
        else:
            rho_ic_target = rho_o + rho_1 * torch.cos(2 * np.pi * x_ic_for_ic / lam)
        mse_rho_ic = mse_cost_function(rho_ic_out, rho_ic_target)
    else:
        mse_rho_ic = 0.0 * torch.mean(rho_ic_out * 0)

    mse_vx_ic = mse_cost_function(vx_ic_out, vx_0)
    if dimension == 2:
        mse_vy_ic = mse_cost_function(vy_ic_out, vy_0)
    elif dimension == 3:
        mse_vy_ic = mse_cost_function(vy_ic_out, vy_0)
        mse_vz_ic = mse_cost_function(vz_ic_out, vz_0)

    ic_loss = mse_vx_ic + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)
    if dimension == 2:
        ic_loss = ic_loss + mse_vy_ic
    elif dimension == 3:
        ic_loss = ic_loss + mse_vy_ic + mse_vz_ic

    return ic_loss


def _compute_hydro_losses(model, net, batch_dom):
    from core.losses import pde_residue
    from config import cs, rho_o

    if model.dimension == 1:
        rho_r, vx_r, _phi_r = pde_residue(batch_dom, net, dimension=1)
    elif model.dimension == 2:
        rho_r, vx_r, vy_r, _phi_r = pde_residue(batch_dom, net, dimension=2)
    else:
        rho_r, vx_r, vy_r, vz_r, _phi_r = pde_residue(batch_dom, net, dimension=3)

    continuity_scale = abs(rho_o * cs)
    continuity_scale = continuity_scale if continuity_scale > 0 else 1.0
    momentum_scale = abs(cs ** 2) if cs != 0 else 1.0

    loss_cont = torch.mean((rho_r / continuity_scale) ** 2)
    loss_mom = torch.mean((vx_r / momentum_scale) ** 2)
    if model.dimension == 2:
        loss_mom = loss_mom + torch.mean((vy_r / momentum_scale) ** 2)
    elif model.dimension == 3:
        loss_mom = loss_mom + torch.mean((vy_r / momentum_scale) ** 2)
        loss_mom = loss_mom + torch.mean((vz_r / momentum_scale) ** 2)

    return loss_cont + loss_mom


def _compute_poisson_loss_from_pde(model, net, batch_dom):
    from core.losses import pde_residue
    from config import const, rho_o

    if model.dimension == 1:
        _rho_r, _vx_r, phi_r = pde_residue(batch_dom, net, dimension=1)
    elif model.dimension == 2:
        _rho_r, _vx_r, _vy_r, phi_r = pde_residue(batch_dom, net, dimension=2)
    else:
        _rho_r, _vx_r, _vy_r, _vz_r, phi_r = pde_residue(batch_dom, net, dimension=3)

    scale = abs(const) * (abs(rho_o) if rho_o != 0 else 1.0)
    scale = scale if scale > 0 else 1.0
    return torch.mean((phi_r / scale) ** 2)

