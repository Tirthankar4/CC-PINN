"""
Causal Training for PINNs — GRINN implementation.

Reference: Wang et al. (2022) "Respecting causality is all you need for
training physics-informed neural networks."
https://arxiv.org/abs/2203.07404

Core idea
---------
Standard PINNs evaluate residuals at all time points simultaneously,
allowing the optimizer to satisfy late-time residuals without first
establishing correct early-time behavior.  For Jeans instability this
means the low-lying rotating solution wins at large t_max: it has lower
integrated PDE residual across the whole domain and the optimizer commits
to it before the correct growing mode has a chance to emerge.

Causal weighting enforces temporal ordering of learning by assigning each
collocation point a weight that reflects how well the PDE is already
satisfied at all earlier times:

    w(t_i) = exp( -ε * Σ_{t_j < t_i} L(t_j) )

where L(t_j) is the mean squared PDE residual in a time bin preceding t_i
and ε > 0 controls sharpness.  A point at late time t_i receives a large
weight only once early residuals are already small — the network cannot
"cheat" by satisfying late-time residuals first.

Bin-based approximation
-----------------------
The continuous formula is approximated by partitioning [t_min, t_max] into
M ordered bins.  For each bin we compute the mean total squared residual
(summed across all PDE equations), take a right-shifted cumulative sum, and
map the result back to a per-point weight.  This is fully compatible with
the existing random-batch collocation strategy.

Integration
-----------
The module exposes a single public function:

    compute_causal_weights(residuals, t_colloc, config)
        residuals : list of [N,1] tensors — one per active PDE equation
        t_colloc  : [N,1] raw physical time tensor
        config    : CausalTrainingConfig

Returns a [N,1] weight tensor, normalised so the mean weight is 1.0 (keeps
loss magnitude comparable across different ε and time windows).

When causal training is disabled (`config.enabled = False`) the function
returns None and the caller falls back to plain MSE.
"""

import torch


def compute_causal_weights(residuals, t_colloc, config):
    """
    Compute per-point causal weights for PDE loss.

    Args:
        residuals   : list of torch.Tensor [N, 1], one per PDE equation.
                      May contain None entries (unused equations) — these
                      are silently skipped.
        t_colloc    : torch.Tensor [N, 1], raw physical time at each point.
        config      : CausalTrainingConfig instance.

    Returns:
        torch.Tensor [N, 1] of causal weights, or None if disabled.
    """
    if not config.enabled:
        return None

    device  = t_colloc.device
    epsilon = float(config.epsilon)
    n_bins  = int(config.n_time_bins)

    # ── Total pointwise squared residual (sum across all PDE equations) ──────
    valid = [r for r in residuals if r is not None]
    if not valid:
        return None
    total_r2 = sum(r.detach() ** 2 for r in valid)   # [N, 1]

    # ── Bin points by time ────────────────────────────────────────────────────
    t_flat = t_colloc.detach().squeeze()               # [N]
    t_min  = t_flat.min()
    t_max  = t_flat.max()

    if (t_max - t_min).item() < 1e-8:
        # Degenerate case: all points at same time → uniform weights
        return torch.ones_like(t_colloc)

    # bin_edges: (n_bins + 1,) — equally spaced between t_min and t_max
    bin_edges = torch.linspace(t_min.item(), t_max.item(), n_bins + 1,
                               device=device)

    # torch.bucketize assigns each value to a bin index in [0, n_bins-1].
    # boundaries = bin_edges[1:-1] (the interior edges, length n_bins-1).
    # Values below the first boundary → bin 0; above the last → bin n_bins-1.
    boundaries = bin_edges[1:-1]                       # (n_bins-1,)
    bin_idx    = torch.bucketize(t_flat, boundaries)   # [N], values 0..n_bins-1

    # ── Mean squared residual per bin ─────────────────────────────────────────
    r2_flat   = total_r2.squeeze()                     # [N]
    bin_losses = torch.zeros(n_bins, device=device)

    for b in range(n_bins):
        mask = (bin_idx == b)
        if mask.any():
            bin_losses[b] = r2_flat[mask].mean()

    # ── Right-shifted cumulative sum (causal: bin i uses losses from 0..i-1) ─
    # cumsum[i] = sum of bin_losses[0..i]
    # shifted[i] = sum of bin_losses[0..i-1]  →  shifted[0] = 0
    cumsum  = torch.cumsum(bin_losses, dim=0)           # (n_bins,)
    shifted = torch.cat([
        torch.zeros(1, device=device),
        cumsum[:-1]
    ])                                                  # (n_bins,)

    # ── Per-point weights ─────────────────────────────────────────────────────
    # w_i = exp( -ε * shifted[bin_idx_i] )
    point_cumsum = shifted[bin_idx]                     # [N]
    weights_flat = torch.exp(-epsilon * point_cumsum)   # [N]

    # Normalise so mean weight = 1 (keeps loss magnitude stable across ε)
    weights_flat = weights_flat / (weights_flat.mean() + 1e-12)

    return weights_flat.unsqueeze(1)                    # [N, 1]
