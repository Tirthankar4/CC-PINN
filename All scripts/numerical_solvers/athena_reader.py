"""
athena_reader.py
================
Reads 2D Athena++ .athdf (HDF5) output files and builds an fd_cache dict
compatible with GRINN's create_5x3_comparison_table / compute_fd_data_cache.

The cache format expected by the comparison functions is:
    {
      t: {
          'x':   np.ndarray (Nx,),   # cell-centred x-coordinates
          'y':   np.ndarray (Ny,),   # cell-centred y-coordinates
          'rho': np.ndarray (Nx,Ny), # density field
          'vx':  np.ndarray (Nx,Ny), # x-velocity field
          'vy':  np.ndarray (Nx,Ny), # y-velocity field
          'phi': np.ndarray (Nx,Ny), # gravitational potential (zeros if unavailable)
      }
    }

Dependencies:
    h5py  (pip install h5py  or  conda install h5py)

Usage example (local comparison script):
    from numerical_solvers.athena_reader import build_athena_cache
    from visualization.plot_comparisons import create_5x3_comparison_table

    cache = build_athena_cache(
        data_dir="/path/to/athena/outputs",
        output_id="disk.out1",
        time_points=[0.0, 1.0, 2.0, 3.0, 4.0],
    )
    # load trained model + config, build initial_params, then:
    create_5x3_comparison_table(net, initial_params, which="density", fd_cache=cache)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency – give a helpful message if missing
# ---------------------------------------------------------------------------
try:
    import h5py
except ImportError:  # pragma: no cover
    raise ImportError(
        "h5py is required by athena_reader.  Install it with:\n"
        "    pip install h5py\n"
        "or:\n"
        "    conda install h5py"
    )


# ---------------------------------------------------------------------------
# Low-level: discover snapshot files
# ---------------------------------------------------------------------------

def discover_snapshots(
    data_dir: str,
    output_id: str = "disk.out1",
) -> Dict[float, Path]:
    """
    Scan *data_dir* for Athena++ HDF5 snapshot files matching
    ``{output_id}.NNNNN.athdf`` and return a dict mapping
    simulation time -> file path.

    Parameters
    ----------
    data_dir  : directory containing the .athdf files
    output_id : Athena++ output block name (e.g. ``"disk.out1"``)

    Returns
    -------
    dict  {time_float: pathlib.Path}

    Notes
    -----
    The simulation time is read from the HDF5 root attribute ``"Time"``.
    Files whose ``"Time"`` attribute is missing are skipped with a warning.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Athena data directory not found: {data_dir}")

    pattern = f"{output_id}.*.athdf"
    files = sorted(data_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {data_dir}"
        )

    time_map: Dict[float, Path] = {}
    for fp in files:
        try:
            with h5py.File(fp, "r") as f:
                t = float(f.attrs["Time"])
            time_map[t] = fp
        except Exception as exc:
            warnings.warn(f"Could not read Time from {fp}: {exc}")

    if not time_map:
        raise RuntimeError(
            f"No valid snapshots found in {data_dir} matching '{pattern}'"
        )

    return time_map


# ---------------------------------------------------------------------------
# Low-level: read a single 2-D snapshot
# ---------------------------------------------------------------------------

def load_athdf_2d(filepath: str | Path) -> Tuple[float, np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a single 2-D Athena++ .athdf snapshot.

    Supports simulations with one or many meshblocks (uniform grid,
    no adaptive mesh refinement).

    Parameters
    ----------
    filepath : path to the .athdf file

    Returns
    -------
    time : float       - simulation time
    x    : (Nx,)       - cell-centred x-coordinates
    y    : (Ny,)       - cell-centred y-coordinates
    rho  : (Nx, Ny)    - mass density  (prim[0])
    vx   : (Nx, Ny)    - x-velocity    (prim[1])
    vy   : (Nx, Ny)    - y-velocity    (prim[2])

    Notes
    -----
    * The routine handles multi-meshblock outputs by stitching the blocks
      together using their ``LogicalLocations`` indices.
    * For 2-D simulations the third spatial axis has size 1; that axis is
      squeezed out automatically.
    * ``prim`` is expected to follow the isothermal or adiabatic hydro
      convention: [density, v1, v2, ...].  Only the first three variables
      (indices 0-2) are used.
    """
    filepath = Path(filepath)

    with h5py.File(filepath, "r") as f:
        time    = float(f.attrs["Time"])
        n_vars  = int(f.attrs["NumVariables"][0])

        # prim shape is either:
        #   (nmb, nvar, nz, ny, nx)  -- classic multi-meshblock layout
        #   (nvar, nmb, nz, ny, nx)  -- layout used by some Athena++ builds
        # Disambiguate using the NumVariables attribute.
        prim_raw = f["prim"][:]
        if prim_raw.shape[0] == n_vars and prim_raw.shape[1] != n_vars:
            # axis-0 is variables: transpose to (nmb, nvar, nz, ny, nx)
            prim_raw = np.moveaxis(prim_raw, 0, 1)

        x1v_raw = f["x1v"][:]   # (n_meshblocks, nx1) or (nx1,)
        x2v_raw = f["x2v"][:]   # (n_meshblocks, nx2) or (nx2,)
        logloc  = f["LogicalLocations"][:]  # (n_meshblocks, 3)

    nmb  = prim_raw.shape[0]
    nvar = prim_raw.shape[1]
    nx3  = prim_raw.shape[2]   # should be 1 for 2-D
    nx2  = prim_raw.shape[3]   # cells per block in y
    nx1  = prim_raw.shape[4]   # cells per block in x

    if nvar < 3:
        raise ValueError(
            f"Expected at least 3 primitive variables (rho, vx, vy) "
            f"but found {nvar} in {filepath}"
        )

    # Ensure coordinate arrays are 2-D: (nmb, nx)
    if x1v_raw.ndim == 1:
        # Single meshblock: shape (nx1,)
        x1v_raw = x1v_raw[np.newaxis, :]
    if x2v_raw.ndim == 1:
        x2v_raw = x2v_raw[np.newaxis, :]

    # Determine the block-grid topology from logical locations
    # logloc[:, 0] -> block index along x1, logloc[:, 1] -> along x2
    i_locs = logloc[:, 0].astype(int)   # x-direction block indices
    j_locs = logloc[:, 1].astype(int)   # y-direction block indices

    ni = int(i_locs.max()) + 1   # number of blocks along x1
    nj = int(j_locs.max()) + 1   # number of blocks along x2

    global_nx = ni * nx1
    global_ny = nj * nx2

    # Allocate global arrays
    rho_global = np.zeros((global_nx, global_ny), dtype=np.float64)
    vx_global  = np.zeros((global_nx, global_ny), dtype=np.float64)
    vy_global  = np.zeros((global_nx, global_ny), dtype=np.float64)
    x_global   = np.zeros(global_nx, dtype=np.float64)
    y_global   = np.zeros(global_ny, dtype=np.float64)

    for mb in range(nmb):
        ix = i_locs[mb] * nx1   # starting x-index in global array
        jy = j_locs[mb] * nx2   # starting y-index in global array

        # prim_raw is now (nmb, nvar, nz, ny, nx) after axis normalisation above.
        # Squeeze nz (always 1 for 2-D); result: (nvar, ny, nx)
        block_data = prim_raw[mb, :, 0, :, :]  # (nvar, nx2, nx1)

        # Transpose to (nx1, nx2) for (x, y) convention
        rho_global[ix : ix + nx1, jy : jy + nx2] = block_data[0].T
        vx_global[ ix : ix + nx1, jy : jy + nx2] = block_data[1].T
        vy_global[ ix : ix + nx1, jy : jy + nx2] = block_data[2].T

        x_global[ix : ix + nx1] = x1v_raw[mb, :]
        y_global[jy : jy + nx2] = x2v_raw[mb, :]

    return time, x_global, y_global, rho_global, vx_global, vy_global


# ---------------------------------------------------------------------------
# High-level: build the fd_cache dict from a directory of snapshots
# ---------------------------------------------------------------------------

def build_athena_cache(
    data_dir: str,
    output_id: str = "disk.out1",
    time_points: Optional[List[float]] = None,
    time_tol: float = 0.05,
    snap_query_times: bool = True,
) -> Dict[float, dict]:
    """
    Build an ``fd_cache`` dict from Athena++ HDF5 snapshots.

    The returned dict has exactly the format expected by
    ``create_5x3_comparison_table`` and ``compute_fd_data_cache``::

        {
          t: {
              'x': np.ndarray, 'y': np.ndarray,
              'rho': np.ndarray, 'vx': np.ndarray,
              'vy': np.ndarray, 'phi': np.ndarray  (zeros)
          }
        }

    Parameters
    ----------
    data_dir         : directory containing the .athdf files
    output_id        : Athena++ output block name (default: ``"disk.out1"``)
    time_points      : list of simulation times to include.
                       If ``None``, *all* available snapshots are loaded.
    time_tol         : maximum allowed |dt| when matching a requested time to
                       the nearest available snapshot.  Raises a warning (not
                       an error) if the nearest snapshot is further away.
    snap_query_times : if ``True`` (default), each *requested* time is
                       replaced by the nearest available snapshot time in the
                       returned cache keys.  This avoids floating-point
                       mismatch when the comparison table looks up times.

    Returns
    -------
    dict  {snapped_time: cache_entry}

    Time-matching strategy
    ----------------------
    Athena++ writes snapshots at fixed intervals (e.g. dt_output = 0.1).
    GRINN's comparison table uses 5 equally-spaced times over [0, tmax].
    For the default tmax=4.0 those times are 0.0, 1.0, 2.0, 3.0, 4.0 --
    all multiples of 0.1, so they align exactly.  For other tmax values
    a *nearest-snapshot* approach is used: the closest available snapshot
    is selected for each requested time.  Set ``time_tol`` to control the
    warning threshold.
    """
    # 1. Discover all snapshots
    time_map = discover_snapshots(data_dir, output_id)
    available_times = np.array(sorted(time_map.keys()))

    print(
        f"[athena_reader] Found {len(available_times)} snapshots in {data_dir}  "
        f"(t in [{available_times[0]:.3f}, {available_times[-1]:.3f}])"
    )

    # 2. Determine which times to load
    if time_points is None:
        query_times = available_times.tolist()
    else:
        query_times = list(time_points)

    # 3. Build cache
    fd_cache: Dict[float, dict] = {}

    for t_req in query_times:
        # Find nearest available snapshot
        diffs       = np.abs(available_times - t_req)
        nearest_idx = int(np.argmin(diffs))
        nearest_t   = float(available_times[nearest_idx])
        delta       = float(diffs[nearest_idx])

        if delta > time_tol:
            warnings.warn(
                f"[athena_reader] Requested t={t_req:.4f} but nearest snapshot "
                f"is t={nearest_t:.4f} (dt={delta:.4f} > tol={time_tol}).  "
                f"Using nearest snapshot anyway."
            )

        filepath = time_map[nearest_t]

        print(
            f"[athena_reader] Loading t={t_req:.4f} -> snapshot t={nearest_t:.4f} "
            f"({filepath.name})"
        )

        _, x, y, rho, vx, vy = load_athdf_2d(filepath)

        # Cache key: use snapped time (nearest_t) if requested, else t_req
        cache_key = nearest_t if snap_query_times else t_req

        fd_cache[cache_key] = {
            "x":   x,
            "y":   y,
            "rho": rho,
            "vx":  vx,
            "vy":  vy,
            "phi": np.zeros_like(rho),   # Athena++ gravitational potential
                                          # not available in prim output
        }

    print(f"[athena_reader] Cache built with {len(fd_cache)} entries.")
    return fd_cache
