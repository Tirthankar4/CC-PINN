# utilities/test_adaptive_collocation.py
# Run from project root: python -m utilities.test_adaptive_collocation
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.model_architecture import PINN
from core.losses import ASTPN
from methods.adaptive_collocation import (
    compute_pointwise_residuals, generate_candidate_points, resample_collocation,
    AdaptiveCollocationConfig
)
from config import DIMENSION, xmin, ymin, tmin, harmonics

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TEST = 500   # small for speed

# 1. Build a minimal ASTPN to get rmin/rmax/dimension
xmax, ymax, tmax = 14.0, 14.0, 3.0
if DIMENSION == 1:
    rmin, rmax = [xmin, tmin], [xmax, tmax]
elif DIMENSION == 2:
    rmin, rmax = [xmin, ymin, tmin], [xmax, ymax, tmax]

model = ASTPN(rmin=rmin, rmax=rmax, N_0=100, N_b=0, N_r=N_TEST, dimension=DIMENSION)
pool  = model.geo_time_coord("Domain")  # exact same structure as experiment.collocation_domain

# 2. Move pool to device
pool = [t.to(DEVICE) for t in pool]

# 3. Build an untrained net (random weights — residuals will be large but finite)
net = PINN(dimension=DIMENSION, n_harmonics=harmonics).to(DEVICE)
net.set_domain(rmin=rmin[:-1], rmax=rmax[:-1], dimension=DIMENSION)  # spatial only

# ── Test compute_pointwise_residuals ──
print("Testing compute_pointwise_residuals...")
residuals = compute_pointwise_residuals(net, pool, DIMENSION)
print(f"  shape:   {residuals.shape}")          # expect [N_TEST, 1]
print(f"  finite:  {torch.isfinite(residuals).all()}")  # expect True
print(f"  min/max: {residuals.min():.4f} / {residuals.max():.4f}")
print(f"  net still training mode: {net.training}")   # expect True

# ── Test generate_candidate_points ──
print("\nTesting generate_candidate_points...")
fresh = generate_candidate_points(rmin, rmax, 200, DIMENSION, DEVICE)
print(f"  n_tensors: {len(fresh)}")             # expect DIMENSION + 1
print(f"  shapes:    {[t.shape for t in fresh]}")
print(f"  requires_grad: {[t.requires_grad for t in fresh]}")

# ── Test resample_collocation ──
print("\nTesting resample_collocation...")
config = AdaptiveCollocationConfig(keep_fraction=0.8)
new_pool = resample_collocation(net, pool, model, config)
print(f"  n_tensors:  {len(new_pool)}")
print(f"  new sizes:  {[t.shape[0] for t in new_pool]}")   # should equal N_TEST
print(f"  old sizes:  {[t.shape[0] for t in pool]}")
print(f"  requires_grad: {[t.requires_grad for t in new_pool]}")
print("All tests passed.")