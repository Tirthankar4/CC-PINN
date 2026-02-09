import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from core.losses import ASTPN, pde_residue
from training.training_diagnostics import TrainingDiagnostics
from core.data_generator import diff
from core.model_architecture import PINN
from core.initial_conditions import (initialize_shared_velocity_fields, generate_power_spectrum_field, 
                                    generate_power_spectrum_field_vy, fun_rho_0, fun_vx_0, fun_vy_0, fun_vz_0)
from config import cs, const, G, rho_o, PERTURBATION_TYPE, KX, KY, BATCH_SIZE, NUM_BATCHES, RANDOM_SEED
from config import IC_WEIGHT, ENABLE_TRAINING_DIAGNOSTICS


# ==================== Physics Calculations and Loss Functions ====================

def closure(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer, rho_1, lam, jeans, v_1, data_terms=None):

    ############## Loss based on initial conditions ###############
    rho_0 = fun_rho_0(rho_1, lam, collocation_IC)
    vx_0  = fun_vx_0(lam, jeans, v_1, collocation_IC)

    if model.dimension == 2:
        vy_0  = fun_vy_0(lam, jeans, v_1, collocation_IC)

    elif model.dimension == 3:
        vy_0  = fun_vy_0(lam, jeans, v_1, collocation_IC)
        vz_0  = fun_vz_0(lam, jeans, v_1, collocation_IC)
    
    net_ic_out = net(collocation_IC)

    rho_ic_out = net_ic_out[:,0:1]
    vx_ic_out  = net_ic_out[:,1:2]

    if model.dimension == 2:
        vy_ic_out  = net_ic_out[:,2:3]
    elif model.dimension == 3:
        vy_ic_out  = net_ic_out[:,2:3]
        vz_ic_out  = net_ic_out[:,3:4]

    # For sinusoidal: enforce only explicit sinusoidal ICs; skip continuity seeding
    is_sin = str(PERTURBATION_TYPE).lower() == "sinusoidal"
    if is_sin:
        x_ic_for_ic = collocation_IC[0]
        if len(collocation_IC) >= 2:  # 2D case
            y_ic_for_ic = collocation_IC[1]
            rho_ic_target = rho_o + rho_1 * torch.cos(KX * x_ic_for_ic + KY * y_ic_for_ic)
        else:  # 1D case
            rho_ic_target = rho_o + rho_1 * torch.cos(2*np.pi*x_ic_for_ic/lam)
        mse_rho_ic = mse_cost_function(rho_ic_out, rho_ic_target)
    else:
        mse_rho_ic = 0.0 * torch.mean(rho_ic_out*0)

    mse_vx_ic  =  mse_cost_function(vx_ic_out, vx_0)

    if model.dimension == 2:
        mse_vy_ic  =  mse_cost_function(vy_ic_out, vy_0)

    elif model.dimension == 3:
        mse_vy_ic  =  mse_cost_function(vy_ic_out, vy_0)
        mse_vz_ic  =  mse_cost_function(vz_ic_out, vz_0)

    ############## Loss based on PDE ###################################
    
    # Apply startup time offset to PDE collocation time only (IC remains at t=0)
    if isinstance(collocation_domain, (list, tuple)):
        colloc_shifted = list(collocation_domain)
    else:
        # Single tensor format: split into [x, y, t]
        colloc_shifted = [collocation_domain[:, i:i+1] for i in range(collocation_domain.shape[1])]

    if model.dimension == 1:
        # time is at index 1
        # Note: Domain collocation points now start from STARTUP_DT (set in data_generator.py)
        rho_r,vx_r,phi_r = pde_residue(colloc_shifted, net, dimension = 1)

    elif model.dimension == 2:
        # time is at index 2
        # Note: Domain collocation points now start from STARTUP_DT (set in data_generator.py)
        rho_r,vx_r,vy_r,phi_r = pde_residue(colloc_shifted, net, dimension = 2)

    elif model.dimension == 3:
        # time is at index 3
        # Note: Domain collocation points now start from STARTUP_DT (set in data_generator.py)
        rho_r,vx_r,vy_r,vz_r,phi_r = pde_residue(colloc_shifted, net, dimension = 3)
    

    mse_rho  = torch.mean(rho_r ** 2)
    mse_velx = torch.mean(vx_r  ** 2)

    if model.dimension == 2:
        mse_vely = torch.mean(vy_r  ** 2)

    elif model.dimension == 3:
        mse_vely = torch.mean(vy_r  ** 2)
        mse_velz = torch.mean(vz_r  ** 2)
    
    mse_phi  = torch.mean(phi_r ** 2)

    ic_weight = IC_WEIGHT

    ################### Combining the loss functions ####################
    if model.dimension == 1:
        base = ic_weight * mse_vx_ic + mse_rho + mse_velx + mse_phi
        loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)

    elif model.dimension == 2:
        base = ic_weight * (mse_vx_ic + mse_vy_ic) + mse_rho + mse_velx + mse_vely + mse_phi
        loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)

    elif model.dimension == 3:
        base = ic_weight * (mse_vx_ic + mse_vy_ic + mse_vz_ic) + mse_rho + mse_velx + mse_vely + mse_velz + mse_phi
        loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)

    
        #loss = mse_rho_ic + mse_vx_ic + mse_vy_ic + mse_vz_ic + \
        #rhox_b + rhoy_b + rhoz_b + vx_xb + vx_yb + vx_zb +  vy_xb + vy_yb + vy_zb + vz_xb + vz_yb + vz_zb + \
        #phi_xb + phi_xx_b + phi_yb + phi_yy_b +  phi_zb + phi_zz_b + mse_rho + mse_velx +  mse_vely + mse_velz + mse_phi 

    data_loss_tensor, data_breakdown = _evaluate_data_terms(net, mse_cost_function, data_terms if data_terms else [])
    if data_loss_tensor is not None:
        loss = loss + data_loss_tensor

    optimizer.zero_grad()
    loss.backward()
    
    # Apply gradient clipping for training stability
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    
    # Create loss breakdown dictionary
    loss_breakdown = {}
    
    # IC losses (grouped together)
    ic_loss = mse_vx_ic.item()
    if isinstance(mse_rho_ic, torch.Tensor) and mse_rho_ic.item() > 0:
        ic_loss += mse_rho_ic.item()
    
    if model.dimension == 2:
        ic_loss += mse_vy_ic.item()
    elif model.dimension == 3:
        ic_loss += mse_vy_ic.item()
        ic_loss += mse_vz_ic.item()
    
    loss_breakdown['IC'] = ic_loss
    
    # PDE losses (grouped together)
    pde_loss = mse_rho.item() + mse_velx.item()
    
    if model.dimension == 2:
        pde_loss += mse_vely.item()
    elif model.dimension == 3:
        pde_loss += mse_vely.item()
        pde_loss += mse_velz.item()
    
    pde_loss += mse_phi.item()
    loss_breakdown['PDE'] = pde_loss

    if data_loss_tensor is not None:
        for label, value in data_breakdown.items():
            loss_breakdown[label] = value

    return loss, loss_breakdown

def _random_batch_indices(total_count, batch_size, device):
    actual = int(min(batch_size, total_count))
    return torch.randperm(total_count, device=device)[:actual]


def _make_batch_tensors(tensors_list, indices):
    """Create batch tensors by indexing. No cloning for speed."""
    return [t[indices] for t in tensors_list]


def _sample_data_term(dataset, batch_size):
    if dataset is None or dataset.get('x') is None or dataset.get('t') is None:
        return None
    count = int(dataset.get('count', dataset['x'].shape[0]))
    if count <= 0:
        return None
    device = dataset['x'].device
    bs = batch_size if batch_size is not None else count
    bs = max(1, min(int(bs), count))
    indices = _random_batch_indices(count, bs, device)

    def _slice(key):
        tensor = dataset.get(key)
        if tensor is None:
            return None
        return tensor[indices]

    return {
        'x': _slice('x'),
        'y': _slice('y'),
        'z': _slice('z'),
        't': _slice('t'),
        'rho': _slice('rho'),
        'vx': _slice('vx'),
        'vy': _slice('vy'),
        'phi': _slice('phi'),
    }


def _compute_data_loss_unweighted(net, batch, mse_cost_function):
    if batch is None or batch['x'] is None or batch['t'] is None:
        return None

    inputs = [batch['x']]
    if batch.get('y') is not None:
        inputs.append(batch['y'])
    if batch.get('z') is not None:
        inputs.append(batch['z'])
    inputs.append(batch['t'])

    outputs = net(inputs)
    num_outputs = outputs.shape[1]

    component_specs = [
        ('rho', 0),
        ('vx', 1),
        ('vy', 2),
        ('phi', 3),
    ]

    losses = []
    for key, idx in component_specs:
        target = batch.get(key)
        if target is None or idx >= num_outputs:
            continue
        pred = outputs[:, idx:idx+1]
        losses.append(mse_cost_function(pred, target))

    if not losses:
        return None
    return sum(losses) / len(losses)


def _evaluate_data_terms(net, mse_cost_function, data_terms):
    if not data_terms:
        return None, {}

    total_loss = None
    breakdown = {}

    for term in data_terms:
        dataset = term.get('dataset')
        weight = float(term.get('weight', 0.0) or 0.0)
        if dataset is None or weight <= 0:
            continue
        batch = _sample_data_term(dataset, term.get('batch_size'))
        data_loss = _compute_data_loss_unweighted(net, batch, mse_cost_function)
        if data_loss is None:
            continue
        weighted_loss = weight * data_loss
        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        label = term.get('label', 'DATA')
        breakdown[label] = breakdown.get(label, 0.0) + float(weighted_loss.detach().cpu())

    return total_loss, breakdown


def closure_batched(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer,
                    rho_1, lam, jeans, v_1, batch_size, num_batches, update_tracker=True, iteration=0, use_fft_poisson=None, data_terms=None, collocation_poisson_ic=None):
    """
    Batched closure function for training with proper gradient accumulation.
    
    This function computes losses over mini-batches and accumulates gradients
    (not tensors) to maintain constant memory usage regardless of num_batches.
    """
    # Determine counts and devices
    # Handle both formats: list of tensors [x, y, t] or single tensor [N, 3]
    if isinstance(collocation_domain, (list, tuple)):
        dom_n = collocation_domain[0].size(0)
        device = collocation_domain[0].device
    else:
        dom_n = collocation_domain.size(0)
        device = collocation_domain.device
    
    ic_n = collocation_IC[0].size(0)
    
    num_effective_batches = int(max(1, num_batches))
    
    # Zero gradients before accumulation loop
    optimizer.zero_grad()
    
    # Track scalar losses for logging (not tensors to avoid memory leak)
    total_loss_scalar = 0.0
    last_data_breakdown = {}
    
    # Variables to track last batch's breakdown (for logging)
    last_mse_vx_ic = 0.0
    last_mse_rho_ic = 0.0
    last_mse_vy_ic = 0.0
    last_mse_vz_ic = 0.0
    last_mse_rho = 0.0
    last_mse_velx = 0.0
    last_mse_vely = 0.0
    last_mse_velz = 0.0
    last_mse_phi = 0.0

    for batch_idx in range(num_effective_batches):
        dom_idx = _random_batch_indices(dom_n, batch_size, device)
        ic_idx = _random_batch_indices(ic_n, batch_size, device)

        # Handle both formats for collocation_domain
        if isinstance(collocation_domain, (list, tuple)):
            batch_dom = _make_batch_tensors(collocation_domain, dom_idx)
        else:
            # Single tensor format: split into [x, y, t]
            batch_dom = [collocation_domain[dom_idx, i:i+1] for i in range(collocation_domain.shape[1])]
        
        batch_ic = _make_batch_tensors(collocation_IC, ic_idx)

        # IC loss terms
        rho_0 = fun_rho_0(rho_1, lam, batch_ic)
        vx_0  = fun_vx_0(lam, jeans, v_1, batch_ic)

        net_ic_out = net(batch_ic)
        rho_ic_out = net_ic_out[:,0:1]
        vx_ic_out  = net_ic_out[:,1:2]

        if model.dimension == 2:
            vy_0 = fun_vy_0(lam, jeans, v_1, batch_ic)
            vy_ic_out = net_ic_out[:,2:3]
        elif model.dimension == 3:
            vy_0 = fun_vy_0(lam, jeans, v_1, batch_ic)
            vz_0 = fun_vz_0(lam, jeans, v_1, batch_ic)
            vy_ic_out = net_ic_out[:,2:3]
            vz_ic_out = net_ic_out[:,3:4]

        is_sin = str(PERTURBATION_TYPE).lower() == "sinusoidal"
        if is_sin:
            x_ic_for_ic = batch_ic[0]
            if len(batch_ic) >= 2:
                y_ic_for_ic = batch_ic[1]
                rho_ic_target = rho_o + rho_1 * torch.cos(KX * x_ic_for_ic + KY * y_ic_for_ic)
            else:
                rho_ic_target = rho_o + rho_1 * torch.cos(2*np.pi*x_ic_for_ic/lam)
            mse_rho_ic = mse_cost_function(rho_ic_out, rho_ic_target)
        else:
            mse_rho_ic = 0.0 * torch.mean(rho_ic_out*0)

        mse_vx_ic  = mse_cost_function(vx_ic_out, vx_0)
        if model.dimension == 2:
            mse_vy_ic = mse_cost_function(vy_ic_out, vy_0)
        elif model.dimension == 3:
            mse_vy_ic = mse_cost_function(vy_ic_out, vy_0)
            mse_vz_ic = mse_cost_function(vz_ic_out, vz_0)

        # PDE residuals on batched domain with startup shift
        if isinstance(batch_dom, (list, tuple)):
            colloc_shifted = list(batch_dom)
        else:
            # Single tensor format: split into [x, y, t]
            colloc_shifted = [batch_dom[:, i:i+1] for i in range(batch_dom.shape[1])]
        if model.dimension == 1:
            # Note: Domain collocation points already start from STARTUP_DT, no need to shift further
            rho_r, vx_r, phi_r = pde_residue(colloc_shifted, net, dimension=1)
        elif model.dimension == 2:
            # Note: Domain collocation points already start from STARTUP_DT, no need to shift further
            rho_r, vx_r, vy_r, phi_r = pde_residue(colloc_shifted, net, dimension=2)
        else:
            # Note: Domain collocation points already start from STARTUP_DT, no need to shift further
            rho_r, vx_r, vy_r, vz_r, phi_r = pde_residue(colloc_shifted, net, dimension=3)

        # Standard uniform weighting for PDE residuals
        mse_rho  = torch.mean(rho_r ** 2)
        mse_velx = torch.mean(vx_r  ** 2)
        if model.dimension == 2:
            mse_vely = torch.mean(vy_r  ** 2)
        elif model.dimension == 3:
            mse_vely = torch.mean(vy_r  ** 2)
            mse_velz = torch.mean(vz_r  ** 2)
        mse_phi  = torch.mean(phi_r ** 2)

        if model.dimension == 1:
            base = mse_vx_ic + mse_rho + mse_velx + mse_phi
            loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)
        elif model.dimension == 2:
            base = mse_vx_ic + mse_vy_ic + mse_rho + mse_velx + mse_vely + mse_phi
            loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)
        else:
            base = mse_vx_ic + mse_vy_ic + mse_vz_ic + mse_rho + mse_velx + mse_vely + mse_velz + mse_phi
            loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)

        if data_terms:
            data_loss_batch, batch_breakdown = _evaluate_data_terms(net, mse_cost_function, data_terms)
            if data_loss_batch is not None:
                loss = loss + data_loss_batch
                last_data_breakdown = batch_breakdown

        # Scale loss and backward immediately (gradient accumulation)
        # This frees the computational graph after each batch
        scaled_loss = loss / num_effective_batches
        scaled_loss.backward()
        
        # Track scalar loss for logging
        total_loss_scalar += loss.item()
        
        # Store last batch values for breakdown (as scalars)
        last_mse_vx_ic = mse_vx_ic.item()
        last_mse_rho_ic = mse_rho_ic.item() if isinstance(mse_rho_ic, torch.Tensor) else 0.0
        last_mse_rho = mse_rho.item()
        last_mse_velx = mse_velx.item()
        last_mse_phi = mse_phi.item()
        if model.dimension >= 2:
            last_mse_vy_ic = mse_vy_ic.item()
            last_mse_vely = mse_vely.item()
        if model.dimension == 3:
            last_mse_vz_ic = mse_vz_ic.item()
            last_mse_velz = mse_velz.item()
    
    # Add Poisson IC loss (Option 3: Pure ML approach for initial phi)
    # This enforces Poisson equation at t=0 with extra collocation points
    # Note: This is added outside the batch loop as it uses separate collocation points
    mse_poisson_ic = 0.0
    mse_phi_mean = 0.0
    if collocation_poisson_ic is not None:
        from core.losses import poisson_residue_only
        from config import POISSON_IC_WEIGHT, PHI_MEAN_CONSTRAINT_WEIGHT
        
        poisson_loss = torch.tensor(0.0, device=device)
        
        # Only compute and add Poisson IC loss if weight is non-zero
        if POISSON_IC_WEIGHT > 0:
            # Enforce Poisson equation: ∇²φ = const*(ρ-ρ₀)
            phi_r_ic = poisson_residue_only(collocation_poisson_ic, net, dimension=model.dimension)
            mse_poisson_ic = torch.mean(phi_r_ic ** 2)
            poisson_loss = poisson_loss + POISSON_IC_WEIGHT * mse_poisson_ic
        
        # Only compute and add phi mean constraint if weight is non-zero
        if PHI_MEAN_CONSTRAINT_WEIGHT > 0:
            # Enforce mean(φ) = 0 at t=0 to fix gauge freedom (Option A)
            # This removes the arbitrary constant offset in φ
            net_output_ic = net(collocation_poisson_ic)
            phi_ic = net_output_ic[:, -1]  # Last output is phi
            mean_phi = torch.mean(phi_ic)
            mse_phi_mean = mean_phi ** 2
            poisson_loss = poisson_loss + PHI_MEAN_CONSTRAINT_WEIGHT * mse_phi_mean
        
        # Backward for Poisson IC loss (adds to accumulated gradients)
        if poisson_loss.requires_grad:
            poisson_loss.backward()
            total_loss_scalar += poisson_loss.item()

    # Apply gradient clipping for training stability
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    
    # Create loss breakdown dictionary (from last batch, as approximation)
    loss_breakdown = {}
    
    # IC losses (grouped together)
    ic_loss = last_mse_vx_ic
    if last_mse_rho_ic > 0:
        ic_loss += last_mse_rho_ic
    
    if model.dimension == 2:
        ic_loss += last_mse_vy_ic
    elif model.dimension == 3:
        ic_loss += last_mse_vy_ic
        ic_loss += last_mse_vz_ic
    
    loss_breakdown['IC'] = ic_loss
    
    # PDE losses (grouped together)
    pde_loss = last_mse_rho + last_mse_velx
    
    if model.dimension == 2:
        pde_loss += last_mse_vely
    elif model.dimension == 3:
        pde_loss += last_mse_vely
        pde_loss += last_mse_velz
    
    pde_loss += last_mse_phi
    loss_breakdown['PDE'] = pde_loss
    
    # Add Poisson IC loss to breakdown if it was computed
    if isinstance(mse_poisson_ic, torch.Tensor):
        loss_breakdown['Poisson_IC'] = mse_poisson_ic.item()
    if isinstance(mse_phi_mean, torch.Tensor):
        loss_breakdown['Phi_Mean'] = mse_phi_mean.item()
    
    for label, value in last_data_breakdown.items():
        loss_breakdown[label] = value

    # Return average loss as a tensor for compatibility
    avg_loss = torch.tensor(total_loss_scalar / num_effective_batches, device=device)
    return avg_loss, loss_breakdown


def closure_batched_cached(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer,
                           rho_1, lam, jeans, v_1, batch_size, num_batches,
                           cached_dom_indices, cached_ic_indices,
                           data_terms=None, collocation_poisson_ic=None):
    """
    Batched closure function for L-BFGS with cached indices.
    
    L-BFGS calls the closure multiple times per step for line search.
    Using cached indices ensures the loss landscape is consistent across calls,
    which is critical for L-BFGS convergence.
    
    Args:
        cached_dom_indices: List of pre-generated domain batch indices
        cached_ic_indices: List of pre-generated IC batch indices
    """
    # Determine device
    if isinstance(collocation_domain, (list, tuple)):
        device = collocation_domain[0].device
    else:
        device = collocation_domain.device
    
    num_effective_batches = len(cached_dom_indices)
    
    # Zero gradients before accumulation loop
    optimizer.zero_grad()
    
    # Track scalar losses for logging
    total_loss_scalar = 0.0
    last_data_breakdown = {}
    
    # Variables to track last batch's breakdown
    last_mse_vx_ic = 0.0
    last_mse_rho_ic = 0.0
    last_mse_vy_ic = 0.0
    last_mse_vz_ic = 0.0
    last_mse_rho = 0.0
    last_mse_velx = 0.0
    last_mse_vely = 0.0
    last_mse_velz = 0.0
    last_mse_phi = 0.0

    for batch_idx in range(num_effective_batches):
        dom_idx = cached_dom_indices[batch_idx]
        ic_idx = cached_ic_indices[batch_idx]

        # Handle both formats for collocation_domain
        if isinstance(collocation_domain, (list, tuple)):
            batch_dom = _make_batch_tensors(collocation_domain, dom_idx)
        else:
            batch_dom = [collocation_domain[dom_idx, i:i+1] for i in range(collocation_domain.shape[1])]
        
        batch_ic = _make_batch_tensors(collocation_IC, ic_idx)

        # IC loss terms
        rho_0 = fun_rho_0(rho_1, lam, batch_ic)
        vx_0  = fun_vx_0(lam, jeans, v_1, batch_ic)

        net_ic_out = net(batch_ic)
        rho_ic_out = net_ic_out[:,0:1]
        vx_ic_out  = net_ic_out[:,1:2]

        if model.dimension == 2:
            vy_0 = fun_vy_0(lam, jeans, v_1, batch_ic)
            vy_ic_out = net_ic_out[:,2:3]
        elif model.dimension == 3:
            vy_0 = fun_vy_0(lam, jeans, v_1, batch_ic)
            vz_0 = fun_vz_0(lam, jeans, v_1, batch_ic)
            vy_ic_out = net_ic_out[:,2:3]
            vz_ic_out = net_ic_out[:,3:4]

        is_sin = str(PERTURBATION_TYPE).lower() == "sinusoidal"
        if is_sin:
            x_ic_for_ic = batch_ic[0]
            if len(batch_ic) >= 2:
                y_ic_for_ic = batch_ic[1]
                rho_ic_target = rho_o + rho_1 * torch.cos(KX * x_ic_for_ic + KY * y_ic_for_ic)
            else:
                rho_ic_target = rho_o + rho_1 * torch.cos(2*np.pi*x_ic_for_ic/lam)
            mse_rho_ic = mse_cost_function(rho_ic_out, rho_ic_target)
        else:
            mse_rho_ic = 0.0 * torch.mean(rho_ic_out*0)

        mse_vx_ic  = mse_cost_function(vx_ic_out, vx_0)
        if model.dimension == 2:
            mse_vy_ic = mse_cost_function(vy_ic_out, vy_0)
        elif model.dimension == 3:
            mse_vy_ic = mse_cost_function(vy_ic_out, vy_0)
            mse_vz_ic = mse_cost_function(vz_ic_out, vz_0)

        # PDE residuals
        if isinstance(batch_dom, (list, tuple)):
            colloc_shifted = list(batch_dom)
        else:
            colloc_shifted = [batch_dom[:, i:i+1] for i in range(batch_dom.shape[1])]
            
        if model.dimension == 1:
            rho_r, vx_r, phi_r = pde_residue(colloc_shifted, net, dimension=1)
        elif model.dimension == 2:
            rho_r, vx_r, vy_r, phi_r = pde_residue(colloc_shifted, net, dimension=2)
        else:
            rho_r, vx_r, vy_r, vz_r, phi_r = pde_residue(colloc_shifted, net, dimension=3)

        # Standard uniform weighting for PDE residuals
        mse_rho  = torch.mean(rho_r ** 2)
        mse_velx = torch.mean(vx_r  ** 2)
        if model.dimension == 2:
            mse_vely = torch.mean(vy_r  ** 2)
        elif model.dimension == 3:
            mse_vely = torch.mean(vy_r  ** 2)
            mse_velz = torch.mean(vz_r  ** 2)
        mse_phi  = torch.mean(phi_r ** 2)

        if model.dimension == 1:
            base = mse_vx_ic + mse_rho + mse_velx + mse_phi
            loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)
        elif model.dimension == 2:
            base = mse_vx_ic + mse_vy_ic + mse_rho + mse_velx + mse_vely + mse_phi
            loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)
        else:
            base = mse_vx_ic + mse_vy_ic + mse_vz_ic + mse_rho + mse_velx + mse_vely + mse_velz + mse_phi
            loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)

        if data_terms:
            data_loss_batch, batch_breakdown = _evaluate_data_terms(net, mse_cost_function, data_terms)
            if data_loss_batch is not None:
                loss = loss + data_loss_batch
                last_data_breakdown = batch_breakdown

        # Scale loss and backward immediately (gradient accumulation)
        scaled_loss = loss / num_effective_batches
        scaled_loss.backward()
        
        # Track scalar loss for logging
        total_loss_scalar += loss.item()
        
        # Store last batch values for breakdown
        last_mse_vx_ic = mse_vx_ic.item()
        last_mse_rho_ic = mse_rho_ic.item() if isinstance(mse_rho_ic, torch.Tensor) else 0.0
        last_mse_rho = mse_rho.item()
        last_mse_velx = mse_velx.item()
        last_mse_phi = mse_phi.item()
        if model.dimension >= 2:
            last_mse_vy_ic = mse_vy_ic.item()
            last_mse_vely = mse_vely.item()
        if model.dimension == 3:
            last_mse_vz_ic = mse_vz_ic.item()
            last_mse_velz = mse_velz.item()
    
    # Add Poisson IC loss if needed
    mse_poisson_ic = 0.0
    mse_phi_mean = 0.0
    if collocation_poisson_ic is not None:
        from core.losses import poisson_residue_only
        from config import POISSON_IC_WEIGHT, PHI_MEAN_CONSTRAINT_WEIGHT
        
        poisson_loss = torch.tensor(0.0, device=device)
        
        if POISSON_IC_WEIGHT > 0:
            phi_r_ic = poisson_residue_only(collocation_poisson_ic, net, dimension=model.dimension)
            mse_poisson_ic = torch.mean(phi_r_ic ** 2)
            poisson_loss = poisson_loss + POISSON_IC_WEIGHT * mse_poisson_ic
        
        if PHI_MEAN_CONSTRAINT_WEIGHT > 0:
            net_output_ic = net(collocation_poisson_ic)
            phi_ic = net_output_ic[:, -1]
            mean_phi = torch.mean(phi_ic)
            mse_phi_mean = mean_phi ** 2
            poisson_loss = poisson_loss + PHI_MEAN_CONSTRAINT_WEIGHT * mse_phi_mean
        
        if poisson_loss.requires_grad:
            poisson_loss.backward()
            total_loss_scalar += poisson_loss.item()

    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    
    # Create loss breakdown dictionary
    loss_breakdown = {}
    
    ic_loss = last_mse_vx_ic
    if last_mse_rho_ic > 0:
        ic_loss += last_mse_rho_ic
    if model.dimension == 2:
        ic_loss += last_mse_vy_ic
    elif model.dimension == 3:
        ic_loss += last_mse_vy_ic
        ic_loss += last_mse_vz_ic
    loss_breakdown['IC'] = ic_loss
    
    pde_loss = last_mse_rho + last_mse_velx
    if model.dimension == 2:
        pde_loss += last_mse_vely
    elif model.dimension == 3:
        pde_loss += last_mse_vely
        pde_loss += last_mse_velz
    pde_loss += last_mse_phi
    loss_breakdown['PDE'] = pde_loss
    
    if isinstance(mse_poisson_ic, torch.Tensor):
        loss_breakdown['Poisson_IC'] = mse_poisson_ic.item()
    if isinstance(mse_phi_mean, torch.Tensor):
        loss_breakdown['Phi_Mean'] = mse_phi_mean.item()
    
    for label, value in last_data_breakdown.items():
        loss_breakdown[label] = value

    # Return average loss as tensor for L-BFGS compatibility
    avg_loss = torch.tensor(total_loss_scalar / max(1, num_effective_batches), device=device, requires_grad=False)
    return avg_loss, loss_breakdown
