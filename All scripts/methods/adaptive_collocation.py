import torch
import torch.nn as nn
import numpy as np
from config import AdaptiveCollocationConfig  # noqa: F401  (re-exported for callers)

def compute_pointwise_residuals(net, collocation_domain, dimension):
    """Score each collocation point by total PDE residual magnitude."""
    from core.losses import pde_residue
    
    net.eval()
    # pde_residue needs grad because diff() uses torch.autograd.grad
    # But we don't want to accumulate parameter gradients
    
    # Correct approach: enable grad for inputs, but don't track param grads
    for p in net.parameters():
        p.requires_grad_(False)
    
    try:
        if dimension == 1:
            rho_r, vx_r, phi_r = pde_residue(collocation_domain, net, dimension=1)
            residual = (rho_r**2 + vx_r**2 + phi_r**2).detach()
        elif dimension == 2:
            rho_r, vx_r, vy_r, phi_r = pde_residue(collocation_domain, net, dimension=2)
            residual = (rho_r**2 + vx_r**2 + vy_r**2 + phi_r**2).detach()
        elif dimension == 3:
            rho_r, vx_r, vy_r, vz_r, phi_r = pde_residue(collocation_domain, net, dimension=3)
            residual = (rho_r**2 + vx_r**2 + vy_r**2 + vz_r**2 + phi_r**2).detach()
        else:
            raise ValueError("Invalid dimension")
    finally:
        for p in net.parameters():
            p.requires_grad_(True)
    
    net.train()
    return residual

def generate_candidate_points(rmin, rmax, n_candidates, dimension, device):
    """
    Generate a fresh set of candidate collocation points.
    Same format as col_gen.geo_time_coord("Domain") but standalone.
    """
    from config import STARTUP_DT
    coords = []
    # Spatial dimensions
    for i in range(dimension):
        coords.append(
            torch.empty(n_candidates, 1, device=device, dtype=torch.float32)
                  .uniform_(rmin[i], rmax[i])
                  .requires_grad_(True)
        )
    # Time dimension — respect STARTUP_DT
    coords.append(
        torch.empty(n_candidates, 1, device=device, dtype=torch.float32)
              .uniform_(max(rmin[dimension], STARTUP_DT), rmax[dimension])
              .requires_grad_(True)
    )
    return coords

def resample_collocation(net, current_pool, model, config):
    """Replace collocation pool: keep high-residual + add fresh uniform."""
    # 1. Score existing points
    residuals = compute_pointwise_residuals(net, current_pool, model.dimension)
    
    # 2. Sort by residual magnitude
    n_keep = int(config.keep_fraction * current_pool[0].size(0))
    _, top_indices = torch.topk(residuals.squeeze(), n_keep)
    
    # 3. Keep top-residual points  
    kept = [tensor[top_indices].detach().requires_grad_(True) for tensor in current_pool]
    
    # 4. Generate fresh uniform points for coverage
    n_fresh = current_pool[0].size(0) - n_keep
    device = current_pool[0].device
    
    fresh = generate_candidate_points(model.rmin, model.rmax, n_fresh, model.dimension, device)
    
    # 5. Concatenate
    new_pool = [torch.cat([k, f], dim=0) for k, f in zip(kept, fresh)]

    # Informative summary so resampling events are visible in the logs
    kept_residuals = residuals.squeeze()[top_indices]
    print(
        f"[AdaptiveCollocation] Resampled: kept {n_keep} high-residual pts "
        f"(residual range [{kept_residuals.min():.3e}, {kept_residuals.max():.3e}]) "
        f"+ {n_fresh} fresh uniform pts → pool size {new_pool[0].size(0)}",
        flush=True,
    )

    return new_pool