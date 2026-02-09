import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import (
    INTERACTIVE_3D_RESOLUTION,
    INTERACTIVE_3D_TIME_STEPS,
    tmin as TMIN_CFG,
    tmax as TMAX_CFG,
    zmin as ZMIN_CFG,
    num_of_waves,
)


def _get_device(net: torch.nn.Module) -> torch.device:
    try:
        return next(net.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _make_3d_grid(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(xmin, xmax, resolution, dtype=np.float32)
    ys = np.linspace(ymin, ymax, resolution, dtype=np.float32)
    zs = np.linspace(zmin, zmax, resolution, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return X.ravel(), Y.ravel(), Z.ravel()


def _query_pinn_3d_grid(
    net: torch.nn.Module,
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    z_flat: np.ndarray,
    t_val: float,
    device: torch.device,
    batch_size: int = 200_000,
) -> np.ndarray:
    total = x_flat.shape[0]
    outputs = np.empty((total, 5), dtype=np.float32)
    t_val = float(t_val)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            x = torch.from_numpy(x_flat[start:end]).to(device)
            y = torch.from_numpy(y_flat[start:end]).to(device)
            z = torch.from_numpy(z_flat[start:end]).to(device)
            t = torch.full_like(x, fill_value=t_val, device=device)

            pred = net([x, y, z, t])
            outputs[start:end, :] = pred.detach().cpu().numpy()

    return outputs


def _build_volume_trace(
    x_flat: np.ndarray,
    y_flat: np.ndarray,
    z_flat: np.ndarray,
    values: np.ndarray,
    field_label: str,
    colorscale: str,
    cmin: float,
    cmax: float,
    visible: bool,
):
    import plotly.graph_objects as go

    return go.Volume(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        value=values,
        opacity=0.1,
        surface_count=15,
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        visible=visible,
        colorbar={"title": field_label},
        showscale=visible,
    )


def create_interactive_3d_plot(
    net: torch.nn.Module,
    initial_params: Tuple[float, float, float, float, float, float, float, str, float],
    time_range: Optional[Tuple[float, float]] = None,
    time_steps: Optional[int] = None,
    resolution: Optional[int] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Create a 3D interactive Plotly volume visualization with time slider and field selector.

    Args:
        net: Trained PINN model.
        initial_params: Tuple (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax_train).
        time_range: (tmin, tmax) for slider; defaults to config and training tmax.
        time_steps: Number of time steps in slider.
        resolution: Grid resolution per axis.
        save_path: Output HTML path; defaults to SNAPSHOT_DIR/interactive_3d_plot.html.

    Returns:
        Absolute path to the saved HTML file.
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for interactive 3D plots. Install with: pip install plotly"
        ) from exc

    xmin, xmax, ymin, ymax, _rho_1, _alpha, lam, _output_folder, tmax_train = initial_params
    zmin = float(ZMIN_CFG)
    zmax = zmin + float(lam) * float(num_of_waves)

    if time_steps is None:
        time_steps = int(INTERACTIVE_3D_TIME_STEPS)
    if resolution is None:
        resolution = int(INTERACTIVE_3D_RESOLUTION)

    if time_range is None:
        t_start = float(TMIN_CFG)
        t_end = float(tmax_train) if tmax_train is not None else float(TMAX_CFG)
    else:
        t_start, t_end = time_range
        t_start = float(t_start)
        t_end = float(t_end)

    time_values = np.linspace(t_start, t_end, time_steps, dtype=np.float32)
    x_flat, y_flat, z_flat = _make_3d_grid(
        xmin, xmax, ymin, ymax, zmin, zmax, resolution
    )
    # Round coordinates to reduce JSON file size
    x_flat = np.round(x_flat, decimals=4)
    y_flat = np.round(y_flat, decimals=4)
    z_flat = np.round(z_flat, decimals=4)

    net.eval()
    device = _get_device(net)

    fields = ["density", "vx", "vy", "vz", "phi"]
    field_labels = {
        "density": "rho",
        "vx": "vx",
        "vy": "vy",
        "vz": "vz",
        "phi": "phi",
    }
    colorscales = {
        "density": "Viridis",
        "vx": "RdBu",
        "vy": "RdBu",
        "vz": "RdBu",
        "phi": "Plasma",
    }

    frames_data: List[Dict[str, np.ndarray]] = []
    global_min = {field: np.inf for field in fields}
    global_max = {field: -np.inf for field in fields}

    for t_val in time_values:
        outputs = _query_pinn_3d_grid(net, x_flat, y_flat, z_flat, t_val, device)
        # Round to 4 decimal places to reduce JSON file size
        outputs = np.round(outputs, decimals=4)
        field_values = {
            "density": outputs[:, 0],
            "vx": outputs[:, 1],
            "vy": outputs[:, 2],
            "vz": outputs[:, 3],
            "phi": outputs[:, 4],
        }

        for field in fields:
            field_min = float(np.min(field_values[field]))
            field_max = float(np.max(field_values[field]))
            global_min[field] = min(global_min[field], field_min)
            global_max[field] = max(global_max[field], field_max)

        frames_data.append(field_values)

    traces = []
    for idx, field in enumerate(fields):
        values = frames_data[0][field]
        traces.append(
            _build_volume_trace(
                x_flat=x_flat,
                y_flat=y_flat,
                z_flat=z_flat,
                values=values,
                field_label=field_labels[field],
                colorscale=colorscales[field],
                cmin=global_min[field],
                cmax=global_max[field],
                visible=(idx == 0),
            )
        )

    frames = []
    for i, t_val in enumerate(time_values):
        frame_traces = []
        for idx, field in enumerate(fields):
            # Only include value data - x, y, z are inherited from base traces
            # This dramatically reduces file size by avoiding coordinate duplication
            frame_traces.append(
                go.Volume(
                    value=frames_data[i][field],
                    opacity=0.1,
                    surface_count=15,
                    colorscale=colorscales[field],
                    cmin=global_min[field],
                    cmax=global_max[field],
                    colorbar={"title": field_labels[field]},
                )
            )
        frames.append(go.Frame(data=frame_traces, name=f"{t_val:.4f}"))

    slider_steps = [
        {
            "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": frame.name,
            "method": "animate",
        }
        for frame in frames
    ]

    field_buttons = []
    for idx, field in enumerate(fields):
        visibility = [False] * len(fields)
        visibility[idx] = True
        field_buttons.append(
            {
                "label": field_labels[field],
                "method": "update",
                "args": [
                    {"visible": visibility, "showscale": visibility},
                    {"title": f"3D Interactive {field_labels[field]}"},
                ],
            }
        )

    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        title="3D Interactive density",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
        },
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "t="},
                "pad": {"t": 50},
                "steps": slider_steps,
            }
        ],
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 150, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
                "type": "buttons",
                "direction": "left",
                "showactive": False,
                "x": 0.0,
                "y": 1.15,
            },
            {
                "buttons": field_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.5,
                "y": 1.15,
            },
        ],
        margin={"l": 0, "r": 0, "b": 0, "t": 80},
    )

    if save_path is None:
        from config import SNAPSHOT_DIR

        save_path = os.path.join(SNAPSHOT_DIR, "interactive_3d_plot.html")

    save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Use CDN for Plotly.js to reduce file size (~3MB savings)
    pio.write_html(fig, file=save_path, full_html=True, include_plotlyjs="cdn")
    return save_path
