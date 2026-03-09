# Adaptive Collocation Method — GRINN

## Overview

Standard PINNs sample collocation points uniformly across the spatiotemporal domain. This is inefficient: the PDE residual is typically small in smooth regions and large near shocks, steep gradients, or late-time instabilities. Adaptive collocation **concentrates training effort where the physics is hardest to satisfy** by periodically resampling the collocation pool based on PDE residual magnitude.

The method implemented here is a **residual-based retain-and-refresh** strategy. At configurable intervals during Adam optimization, the current pool of collocation points is scored by pointwise PDE residual magnitude. The highest-residual points are kept (they mark regions the network still struggles with), while the lowest-residual points are discarded and replaced with fresh uniform samples (to maintain spatial coverage and avoid blind spots).

---

## Algorithm

The resampling procedure runs every `resample_every_n` Adam iterations. It does **not** run during L-BFGS optimization, since L-BFGS requires a consistent loss landscape across its line-search evaluations.

### Step-by-step

```
Given:
  - current collocation pool:  P = [x, y, (z), t],  each tensor of shape [N, 1]
  - trained network:           net
  - config:                    keep_fraction, n_candidates, resample_every_n

1. SCORE every point in P
   Compute the full PDE residual at each point:
     residual_i = ρ_r² + vx_r² + (vy_r²) + (vz_r²) + φ_r²
   where ρ_r, vx_r, ... are the continuity, momentum, and Poisson equation
   residuals respectively. This uses the same pde_residue() function as training.

2. RANK by residual magnitude (descending)

3. KEEP the top (keep_fraction × N) points
   These are the "hard" points where the network still has high PDE error.
   → Detach from the computation graph to prevent memory leaks.
   → Re-enable requires_grad for future autograd differentiation.

4. GENERATE (1 - keep_fraction) × N fresh points
   Uniformly sampled across the full domain [rmin, rmax] for each coordinate,
   respecting the STARTUP_DT offset for the time dimension.

5. CONCATENATE kept + fresh → new pool of size N (same total count)

6. REPLACE experiment.collocation_domain with the new pool
```

### Pseudocode

```python
def resample_collocation(net, current_pool, model, config):
    # 1. Score
    residuals = compute_pointwise_residuals(net, current_pool, model.dimension)
    
    # 2-3. Keep top-k
    n_keep = int(config.keep_fraction * N)
    _, top_indices = torch.topk(residuals, n_keep)
    kept = [tensor[top_indices].detach().requires_grad_(True) for tensor in current_pool]
    
    # 4. Fresh uniform points
    n_fresh = N - n_keep
    fresh = generate_candidate_points(rmin, rmax, n_fresh, dimension, device)
    
    # 5. New pool
    new_pool = [torch.cat([k, f], dim=0) for k, f in zip(kept, fresh)]
    return new_pool
```

---

## Configuration

All parameters live in `AdaptiveCollocationConfig` (defined in `config.py`):

| Parameter | Default | Description |
|---|---|---|
| `enabled` | `True` | Master toggle. When `False`, the entire method is a no-op. |
| `resample_every_n` | `50` | Resample every N Adam iterations. |
| `n_candidates` | `5000` | Number of candidate points generated at each resample (currently used as the fresh-point count, but designed to support future candidate-based selection). |
| `keep_fraction` | `0.8` | Fraction of the existing pool retained (highest-residual points). |
| `uniform_fraction` | `0.2` | Fraction replaced with fresh uniform samples. Equals `1 - keep_fraction`. |

These are set via the frozen dataclass system and serialized to YAML with every run:

```yaml
adaptive_collocation:
  enabled: true
  resample_every_n: 50
  n_candidates: 5000
  keep_fraction: 0.8
  uniform_fraction: 0.2
```

---

## Where It Runs in the Pipeline

```
build_experiment()
  └── creates AdaptiveCollocationConfig from CONFIG
        └── stored in experiment.adaptive_config

train(experiment)
  ├── Adam loop (iteration 0 … iteration_adam)
  │     └── every resample_every_n iterations:
  │           resample_collocation(net, collocation_domain, model, config)
  │           └── experiment.collocation_domain = new_pool
  │
  ├── One final resample after Adam completes (before L-BFGS)
  │
  └── L-BFGS loop (no resampling — cached indices ensure consistent landscape)
```

Key integration points:

- **`trainer.py` lines 70–76**: The `if` block inside the Adam loop that triggers resampling.
- **`trainer.py` lines 84–89**: The post-Adam, pre-L-BFGS resample to give L-BFGS the best starting pool.
- **`adaptive_collocation.py`**: The entire method implementation (~89 lines).

---

## Implementation Details

### Residual computation without parameter gradients

`compute_pointwise_residuals()` needs autograd to compute spatial/temporal derivatives of the network output (since `pde_residue()` calls `diff()` which uses `torch.autograd.grad`). However, we do **not** want to accumulate gradients on network parameters during scoring — that would corrupt the optimizer state.

The solution:

```python
# Disable parameter gradients (no .grad accumulation)
for p in net.parameters():
    p.requires_grad_(False)

# pde_residue() still works because input tensors have requires_grad=True
# torch.autograd.grad computes d(output)/d(input), not d(output)/d(params)
residuals = pde_residue(collocation_domain, net, dimension)

# Re-enable for subsequent training
for p in net.parameters():
    p.requires_grad_(True)
```

### Detach-and-regrade pattern

Kept points must be detached from the old computation graph to prevent memory leaks (the old graph would otherwise stay alive across resampling cycles). After detaching, `requires_grad_(True)` is called so that `diff()` can still compute spatial derivatives through these points during future training steps:

```python
kept = [tensor[top_indices].detach().requires_grad_(True) for tensor in current_pool]
```

### Pool size invariance

The pipeline discovers the pool size dynamically via `collocation_domain[0].size(0)` at every closure call. The resampling always produces `n_keep + n_fresh = N` (the original pool size), so the effective batch sizes and memory usage remain constant. The pipeline never needs to be told about the swap.

### L-BFGS compatibility

L-BFGS performs a line search that calls the closure multiple times per `optimizer.step()`. If the collocation pool changed between these calls (as it would with adaptive resampling), the loss landscape would shift and L-BFGS convergence would break. Therefore:

- Resampling is **disabled** during the L-BFGS phase.
- A single resample occurs **after Adam completes, before L-BFGS begins**, giving L-BFGS the best possible starting pool.
- L-BFGS uses **cached batch indices** (`_generate_cached_indices`), ensuring consistent subsampling across line-search evaluations.

---

## Intuition: Why It Works

### Concentrating effort where it matters

In a Jeans instability simulation, the PDE residual is not spatially uniform. Regions with steep density gradients, velocity divergences, or gravitational potential wells generate larger residuals. By keeping the high-residual points, we ensure the optimizer sees the hardest parts of the problem at every iteration — rather than spending gradient budget on regions the network has already learned.

### Maintaining exploration via fresh points

If we only kept high-residual points and never refreshed, the pool would progressively cluster in a shrinking region. The network might overfit to those regions while its accuracy silently degrades elsewhere. The `uniform_fraction` of fresh points prevents this by continually probing the entire domain.

### The 80/20 split

The default `keep_fraction=0.8` is a balance:
- **Too high** (e.g., 0.95): insufficient exploration, pool stagnates.
- **Too low** (e.g., 0.5): the network loses memory of hard regions, relearning them from scratch after each resample.
- **0.8**: retains the most informative points while injecting enough randomness to discover new problem areas.

---

## Logging

Every resampling event prints a summary:

```
[AdaptiveCollocation] Resampled: kept 56000 high-residual pts 
(residual range [1.234e-03, 8.567e+01]) + 14000 fresh uniform pts → pool size 70000
```

This shows:
- How many points were retained vs. refreshed
- The residual magnitude range of kept points (useful for tracking whether the network is improving — the max residual should generally decrease over training)
- The total pool size (should stay constant)

---

## Relation to Other Methods

| Method | Selection basis | When | Scope |
|---|---|---|---|
| **This method** (residual-based retain-and-refresh) | Pointwise PDE residual magnitude | Periodically during Adam | Collocation points only |
| **RAR** (Residual-based Adaptive Refinement) | Similar residual scoring, but *adds* points instead of replacing | Periodically | Collocation points only |
| **PINNACLE** (Adaptive Collocation + Experimental Points) | NTK-based interaction analysis across all point types | Every iteration | Collocation + experimental points jointly |
| **Uniform sampling** (baseline) | Random uniform | Once at initialization | Collocation points only |

The current implementation is closest to RAR in spirit but differs in that it maintains a **fixed pool size** rather than growing the pool. This keeps memory and computation costs constant throughout training.

---

## Files

| File | Role |
|---|---|
| [`adaptive_collocation.py`](file:///c:/Users/tirth/Documents/Projects/GRINN/Power%20spectrum%20perturbations/methods/adaptive_collocation.py) | Core implementation: `compute_pointwise_residuals`, `generate_candidate_points`, `resample_collocation` |
| [`config.py`](file:///c:/Users/tirth/Documents/Projects/GRINN/Power%20spectrum%20perturbations/config.py) (lines 128–136) | `AdaptiveCollocationConfig` dataclass |
| [`trainer.py`](file:///c:/Users/tirth/Documents/Projects/GRINN/Power%20spectrum%20perturbations/training/trainer.py) (lines 70–89) | Integration into the training loop |
| [`train.py`](file:///c:/Users/tirth/Documents/Projects/GRINN/Power%20spectrum%20perturbations/train.py) (line 74, 529) | `Experiment.adaptive_config` field and config wiring |
