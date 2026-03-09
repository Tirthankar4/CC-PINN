import numpy as np

import torch
from core.losses import pde_residue
from core.initial_conditions import fun_rho_0, fun_vx_0, fun_vy_0, fun_vz_0
from config import rho_o, PERTURBATION_TYPE, KX, KY, GRAVITY


# ==================== Physics Calculations and Loss Functions ====================


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
        'vz': _slice('vz'),
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

    if batch.get('z') is not None:  # 3D
        component_specs = [('rho', 0), ('vx', 1), ('vy', 2), ('vz', 3)]
        if GRAVITY:
            component_specs.append(('phi', 4))
    elif batch.get('y') is not None:  # 2D
        component_specs = [('rho', 0), ('vx', 1), ('vy', 2)]
        if GRAVITY:
            component_specs.append(('phi', 3))
    elif batch.get('x') is not None:  # 1D
        component_specs = [('rho', 0), ('vx', 1)]
        if GRAVITY:
            component_specs.append(('phi', 2))
    else:
        return None

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


# ==================== Shared batch-level physics (inner loop body) ====================

def _compute_ic_losses(net, batch_ic, model, rho_1, lam, jeans, v_1, mse_cost_function, interpolators=None):
    """
    Compute initial condition losses for density and velocity components.
    
    Returns:
        Dict with MSE losses for each IC component
    """
    # Get true IC values
    rho_0 = fun_rho_0(rho_1, lam, batch_ic)
    vx_0 = fun_vx_0(lam, jeans, v_1, batch_ic, interpolators=interpolators)
    
    # Network predictions at t=0
    net_ic_out = net(batch_ic)
    rho_ic_out = net_ic_out[:, 0:1]
    vx_ic_out = net_ic_out[:, 1:2]
    
    # Dimension-specific velocity components
    if model.dimension == 2:
        vy_0 = fun_vy_0(lam, jeans, v_1, batch_ic, interpolators=interpolators)
        vy_ic_out = net_ic_out[:, 2:3]
    elif model.dimension == 3:
        vy_0 = fun_vy_0(lam, jeans, v_1, batch_ic, interpolators=interpolators)
        vz_0 = fun_vz_0(lam, jeans, v_1, batch_ic, interpolators=interpolators)
        vy_ic_out = net_ic_out[:, 2:3]
        vz_ic_out = net_ic_out[:, 3:4]
    
    # Density IC loss (only for sinusoidal perturbations)
    is_sin = str(PERTURBATION_TYPE).lower() == "sinusoidal"
    if is_sin:
        x_ic = batch_ic[0]
        if model.dimension >= 2:
            y_ic = batch_ic[1]
            rho_ic_target = rho_o + rho_1 * torch.cos(KX * x_ic + KY * y_ic)
        else:
            rho_ic_target = rho_o + rho_1 * torch.cos(2 * np.pi * x_ic / lam)
        mse_rho_ic = mse_cost_function(rho_ic_out, rho_ic_target)
    else:
        mse_rho_ic = 0.0 * torch.mean(rho_ic_out * 0)
    
    # Velocity IC losses
    mse_vx_ic = mse_cost_function(vx_ic_out, vx_0)
    
    if model.dimension == 2:
        mse_vy_ic = mse_cost_function(vy_ic_out, vy_0)
        mse_vz_ic = None
    elif model.dimension == 3:
        mse_vy_ic = mse_cost_function(vy_ic_out, vy_0)
        mse_vz_ic = mse_cost_function(vz_ic_out, vz_0)
    else:
        mse_vy_ic = None
        mse_vz_ic = None
    
    return {
        'mse_rho_ic': mse_rho_ic,
        'mse_vx_ic': mse_vx_ic,
        'mse_vy_ic': mse_vy_ic,
        'mse_vz_ic': mse_vz_ic,
    }


def _compute_pde_losses(net, batch_dom, model, mse_cost_function):
    """
    Compute PDE residual losses for all equations (continuity, momentum, Poisson).
    
    Returns:
        Dict with MSE losses for each PDE component
    """
    # Ensure batch_dom is in list format for pde_residue
    colloc_shifted = list(batch_dom) if isinstance(batch_dom, (list, tuple)) else \
                     [batch_dom[:, i:i+1] for i in range(batch_dom.shape[1])]
    
    # Compute PDE residuals (dimension-specific returns)
    if model.dimension == 1:
        if GRAVITY:
            rho_r, vx_r, phi_r = pde_residue(colloc_shifted, net, dimension=1)
        else:
            rho_r, vx_r = pde_residue(colloc_shifted, net, dimension=1)
            phi_r = None
        vy_r = vz_r = None
    elif model.dimension == 2:
        if GRAVITY:
            rho_r, vx_r, vy_r, phi_r = pde_residue(colloc_shifted, net, dimension=2)
        else:
            rho_r, vx_r, vy_r = pde_residue(colloc_shifted, net, dimension=2)
            phi_r = None
        vz_r = None
    else:  # dimension == 3
        if GRAVITY:
            rho_r, vx_r, vy_r, vz_r, phi_r = pde_residue(colloc_shifted, net, dimension=3)
        else:
            rho_r, vx_r, vy_r, vz_r = pde_residue(colloc_shifted, net, dimension=3)
            phi_r = None
    
    # Convert residuals to MSE losses
    mse_rho = torch.mean(rho_r ** 2)
    mse_velx = torch.mean(vx_r ** 2)
    mse_phi = torch.mean(phi_r ** 2) if phi_r is not None else None
    
    mse_vely = torch.mean(vy_r ** 2) if vy_r is not None else None
    mse_velz = torch.mean(vz_r ** 2) if vz_r is not None else None
    
    return {
        'mse_rho': mse_rho,
        'mse_velx': mse_velx,
        'mse_vely': mse_vely,
        'mse_velz': mse_velz,
        'mse_phi': mse_phi,
    }


def _aggregate_losses(ic_losses, pde_losses, model):
    """
    Aggregate IC and PDE losses into total loss based on dimension.
    
    Returns:
        Total loss tensor
    """
    # Density IC term (0 for power spectrum)
    rho_ic_term = ic_losses['mse_rho_ic'] if isinstance(ic_losses['mse_rho_ic'], torch.Tensor) else 0.0
    
    # Base losses (all dimensions)
    loss = ic_losses['mse_vx_ic'] + pde_losses['mse_rho'] + pde_losses['mse_velx'] + rho_ic_term
    
    # Add dimension-specific components
    if model.dimension >= 2:
        loss = loss + ic_losses['mse_vy_ic'] + pde_losses['mse_vely']
    if model.dimension == 3:
        loss = loss + ic_losses['mse_vz_ic'] + pde_losses['mse_velz']
    if pde_losses['mse_phi'] is not None:
        loss = loss + pde_losses['mse_phi']
    
    return loss


def _compute_single_batch_losses(net, mse_cost_function, batch_dom, batch_ic,
                                  model, rho_1, lam, jeans, v_1,
                                  num_effective_batches, data_terms, interpolators=None):
    """
    Computes IC loss + PDE residuals + loss aggregation for one mini-batch,
    immediately calls scaled_loss.backward(), and returns scalar tracking values.

    Shared by both closure_batched (Adam) and closure_batched_cached (L-BFGS).
    The only thing that differs between those two closures is *how* batch_dom
    and batch_ic were obtained (fresh random indices vs cached indices).

    Returns a dict of plain Python floats suitable for logging/breakdown, plus
    the unscaled scalar loss value for this batch.
    """
    # Compute IC losses
    ic_losses = _compute_ic_losses(net, batch_ic, model, rho_1, lam, jeans, v_1, mse_cost_function, interpolators)
    
    # Compute PDE losses
    pde_losses = _compute_pde_losses(net, batch_dom, model, mse_cost_function)
    
    # Aggregate into total loss
    loss = _aggregate_losses(ic_losses, pde_losses, model)
    
    # Add optional supervised data terms
    data_breakdown = {}
    if data_terms:
        data_loss_batch, data_breakdown = _evaluate_data_terms(net, mse_cost_function, data_terms)
        if data_loss_batch is not None:
            loss = loss + data_loss_batch

    # Backward pass (gradient accumulation — graph freed immediately)
    scaled_loss = loss / num_effective_batches
    scaled_loss.backward()

    # Collect scalar tracking values (detached, no graph refs)
    scalars = {
        'loss': loss.item(),
        'mse_vx_ic': ic_losses['mse_vx_ic'].item(),
        'mse_rho_ic': ic_losses['mse_rho_ic'].item() if isinstance(ic_losses['mse_rho_ic'], torch.Tensor) else 0.0,
        'mse_rho': pde_losses['mse_rho'].item(),
        'mse_velx': pde_losses['mse_velx'].item(),
        'mse_phi': pde_losses['mse_phi'].item() if pde_losses['mse_phi'] is not None else 0.0,
        'mse_vy_ic': 0.0,
        'mse_vely': 0.0,
        'mse_vz_ic': 0.0,
        'mse_velz': 0.0,
        'data_breakdown': data_breakdown,
    }
    if model.dimension >= 2:
        scalars['mse_vy_ic'] = ic_losses['mse_vy_ic'].item()
        scalars['mse_vely'] = pde_losses['mse_vely'].item()
    if model.dimension == 3:
        scalars['mse_vz_ic'] = ic_losses['mse_vz_ic'].item()
        scalars['mse_velz'] = pde_losses['mse_velz'].item()

    return scalars



def _build_loss_breakdown(last_scalars, model, last_data_breakdown):
    """
    Constructs the loss breakdown dict from the scalar values tracked across batches.
    Shared by both closure variants.
    """
    breakdown = {}

    ic_loss = last_scalars['mse_vx_ic']
    if last_scalars['mse_rho_ic'] > 0:
        ic_loss += last_scalars['mse_rho_ic']
    if model.dimension >= 2:
        ic_loss += last_scalars['mse_vy_ic']
    if model.dimension == 3:
        ic_loss += last_scalars['mse_vz_ic']
    breakdown['IC'] = ic_loss

    pde_loss = last_scalars['mse_rho'] + last_scalars['mse_velx']
    if model.dimension >= 2:
        pde_loss += last_scalars['mse_vely']
    if model.dimension == 3:
        pde_loss += last_scalars['mse_velz']
    if GRAVITY:
        pde_loss += last_scalars['mse_phi']
    breakdown['PDE'] = pde_loss

    for label, value in last_data_breakdown.items():
        breakdown[label] = value

    return breakdown


# ==================== Public closure functions ====================

def closure_batched(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer,
                    rho_1, lam, jeans, v_1, batch_size, num_batches,
                    update_tracker=True, iteration=0, use_fft_poisson=None,
                    data_terms=None, interpolators=None):
    """
    Batched closure for Adam.

    Generates fresh random batch indices on every call. Adam only calls the
    closure once per step, so a shifting loss landscape is fine.

    Gradients are accumulated across mini-batches (not tensors) so peak GPU
    memory stays constant regardless of num_batches.
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    if isinstance(collocation_domain, (list, tuple)):
        dom_n  = collocation_domain[0].size(0)
        device = collocation_domain[0].device
    else:
        dom_n  = collocation_domain.size(0)
        device = collocation_domain.device

    ic_n = collocation_IC[0].size(0)
    num_effective_batches = int(max(1, num_batches))

    optimizer.zero_grad()

    total_loss_scalar = 0.0
    scalar_keys = ['mse_vx_ic', 'mse_rho_ic', 'mse_vy_ic', 'mse_vz_ic',
                   'mse_rho', 'mse_velx', 'mse_vely', 'mse_velz', 'mse_phi']
    accum_scalars = {k: 0.0 for k in scalar_keys}
    accum_data_breakdown = {}

    # ── Mini-batch loop (fresh random indices each iteration) ─────────────────
    for _ in range(num_effective_batches):
        dom_idx = _random_batch_indices(dom_n, batch_size, device)
        ic_idx  = _random_batch_indices(ic_n,  batch_size, device)

        if isinstance(collocation_domain, (list, tuple)):
            batch_dom = _make_batch_tensors(collocation_domain, dom_idx)
        else:
            batch_dom = [collocation_domain[dom_idx, i:i+1] for i in range(collocation_domain.shape[1])]

        batch_ic = _make_batch_tensors(collocation_IC, ic_idx)

        scalars = _compute_single_batch_losses(
            net, mse_cost_function, batch_dom, batch_ic,
            model, rho_1, lam, jeans, v_1,
            num_effective_batches, data_terms, interpolators=interpolators
        )

        total_loss_scalar += scalars['loss']
        for key in scalar_keys:
            accum_scalars[key] += scalars[key]
        for label, value in scalars['data_breakdown'].items():
            accum_data_breakdown[label] = accum_data_breakdown.get(label, 0.0) + value

    # ── Gradient clipping ─────────────────────────────────────────────────────
    # Measure gradient norm before clipping for monitoring
    grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float('inf'))
    # Now actually clip
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

    avg_scalars = {k: v / num_effective_batches for k, v in accum_scalars.items()}
    avg_data_breakdown = {
        label: value / num_effective_batches
        for label, value in accum_data_breakdown.items()
    }

    loss_breakdown = _build_loss_breakdown(avg_scalars, model, avg_data_breakdown)
    loss_breakdown['grad_norm'] = grad_norm_before_clip.item()

    avg_loss_scalar = total_loss_scalar / num_effective_batches
    avg_loss = torch.tensor(avg_loss_scalar, device=device)
    return avg_loss, loss_breakdown


def closure_batched_cached(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer,
                           rho_1, lam, jeans, v_1, batch_size, num_batches,
                           cached_dom_indices, cached_ic_indices,
                           data_terms=None, interpolators=None):
    """
    Batched closure for L-BFGS.

    L-BFGS performs a line search and calls the closure multiple times per
    optimizer step. Using pre-generated cached indices ensures the loss
    landscape is consistent across those calls, which is critical for
    L-BFGS convergence. Fresh random indices (as in closure_batched) would
    make the landscape shift between line-search evaluations and break
    convergence.
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    if isinstance(collocation_domain, (list, tuple)):
        device = collocation_domain[0].device
    else:
        device = collocation_domain.device

    num_effective_batches = len(cached_dom_indices)

    optimizer.zero_grad()

    total_loss_scalar = 0.0
    scalar_keys = ['mse_vx_ic', 'mse_rho_ic', 'mse_vy_ic', 'mse_vz_ic',
                   'mse_rho', 'mse_velx', 'mse_vely', 'mse_velz', 'mse_phi']
    accum_scalars = {k: 0.0 for k in scalar_keys}
    accum_data_breakdown = {}

    # ── Mini-batch loop (cached indices — consistent across L-BFGS line search) ──
    for batch_idx in range(num_effective_batches):
        dom_idx  = cached_dom_indices[batch_idx]
        ic_idx   = cached_ic_indices[batch_idx]

        if isinstance(collocation_domain, (list, tuple)):
            batch_dom = _make_batch_tensors(collocation_domain, dom_idx)
        else:
            batch_dom = [collocation_domain[dom_idx, i:i+1] for i in range(collocation_domain.shape[1])]

        batch_ic = _make_batch_tensors(collocation_IC, ic_idx)

        scalars = _compute_single_batch_losses(
            net, mse_cost_function, batch_dom, batch_ic,
            model, rho_1, lam, jeans, v_1,
            num_effective_batches, data_terms, interpolators=interpolators
        )

        total_loss_scalar += scalars['loss']
        for key in scalar_keys:
            accum_scalars[key] += scalars[key]
        for label, value in scalars['data_breakdown'].items():
            accum_data_breakdown[label] = accum_data_breakdown.get(label, 0.0) + value

    # ── Gradient clipping ─────────────────────────────────────────────────────
    # Measure gradient norm before clipping for monitoring
    grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float('inf'))
    # Now actually clip
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

    avg_scalars = {k: v / max(1, num_effective_batches) for k, v in accum_scalars.items()}
    avg_data_breakdown = {
        label: value / max(1, num_effective_batches)
        for label, value in accum_data_breakdown.items()
    }

    loss_breakdown = _build_loss_breakdown(avg_scalars, model, avg_data_breakdown)
    loss_breakdown['grad_norm'] = grad_norm_before_clip.item()

    # requires_grad=False: returned tensor is used by L-BFGS only for logging,
    # not for further differentiation
    avg_loss_scalar = total_loss_scalar / max(1, num_effective_batches)
    avg_loss = torch.tensor(avg_loss_scalar,
                            device=device, requires_grad=False)
    return avg_loss, loss_breakdown
