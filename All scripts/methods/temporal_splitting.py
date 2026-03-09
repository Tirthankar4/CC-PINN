from dataclasses import dataclass
import torch
from core.data_generator import col_gen
from config import GRAVITY


@dataclass
class TemporalSplittingConfig:
    enabled: bool = False
    n_splits: int = 3
    n_interface_points: int = 1000


def make_temporal_splits(tmin, tmax, n_splits):
    step = (tmax - tmin) / n_splits
    return [(tmin + i * step, tmin + (i + 1) * step) for i in range(n_splits)]


def generate_interface_ic_coords(t_interface, rmin_spatial, rmax_spatial, n_points, dimension, device):
    """
    Returns [x, (y, z), t] tensor list at t=t_interface.
    Mirrors col_gen._generate_* helpers — no ASTPN needed.
    """
    coords = []
    for lo, hi in zip(rmin_spatial, rmax_spatial):
        coords.append(
            torch.empty(n_points, 1, device=device, dtype=torch.float32)
                  .uniform_(lo, hi)
                  .requires_grad_(True)
        )
    t_tensor = (
        torch.ones(n_points, 1, device=device, dtype=torch.float32) * t_interface
    ).requires_grad_(True)
    coords.append(t_tensor)
    return coords  # [x, (y, z), t_interface] — same structure as geo_time_coord("IC")


def evaluate_frozen_network(prev_net, coords, dimension):
    """
    Run prev_net at coords with no gradient tracking.
    Returns a dict matching the data_terms schema.
    """
    prev_net.eval()
    with torch.no_grad():
        out = prev_net(coords)  # shape: [N, n_outputs]

    # Unpack by column position — preserving the output contract
    result = {
        'x': coords[0], 't': coords[-1],
        'rho': out[:, 0:1],
        'vx':  out[:, 1:2],
    }
    if dimension >= 2:
        result['y']  = coords[1]
        result['vy'] = out[:, 2:3]
    if dimension == 3:
        result['z']  = coords[2]
        result['vz'] = out[:, 3:4]
    if GRAVITY:
        result['phi'] = out[:, -1:]
    return result


def train_temporal_splitting(config):
    windows = make_temporal_splits(config.tmin, config.tmax, config.n_splits)
    trained_models = []

    for i, (t_low, t_high) in enumerate(windows):
        if i == 0:
            collocation_IC = col_gen.geo_time_coord(option="IC")
        else:
            collocation_IC = generate_interface_ic_coords(
                t_low,
                col_gen.rmin_spatial,
                col_gen.rmax_spatial,
                config.n_interface_points,
                col_gen.dimension,
                col_gen.device,
            )
