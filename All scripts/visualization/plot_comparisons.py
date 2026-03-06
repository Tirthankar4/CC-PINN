from .plot_utils import *
from .plot_fields import *

def create_1d_comparison_plots(net, initial_params, time_array_1d=None):
    """
    Create 1D cross section comparison plots between PINN, Linear Theory, and Finite Difference
    
    Args:
        net: Trained neural network
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_array_1d: List of time points for 1D plots (default: [0.5, 1.0, 1.5])
    """
    if time_array_1d is None:
        time_array_1d = [0.5, 1.0, 1.5]
    
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    # rho_o imported from config.py
    num_of_waves = (xmax - xmin) / lam
    
    print("Creating 1D cross section comparison plots...")
    
    # Create comparison plots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    # Collect all velocity data to determine appropriate y-axis limits
    all_velocity_data = []
    
    for i, time in enumerate(time_array_1d):
        print(f"Creating 1D comparison plots at t = {time}")
        
        # Get LAX solution (Finite Difference)
        x, rho, v, phi, n, rho_LT, rho_LT_max, rho_max_FD, v_LT = _timed_call(
            "LAX comparison slice (cpu)",
            lax_solution,
            time, FD_N_2D, _get_fd_nu(), lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=True, animation=True
        )
        
        # Get PINN solution
        X, rho_pred0, v_pred_x0, v_pred_y0, phi_pred0, rho_max_PN = plot_function(
            net, [time], initial_params, velocity=True, isplot=False, animation=True
        )
        
        # Interpolate LAX solutions to match PINN grid
        from scipy.interpolate import interp1d
        X_flat = X.flatten()
        
        # Interpolate Linear Theory and Finite Difference to PINN grid
        rho_LT_interp = interp1d(x, rho_LT, kind='linear', bounds_error=False, fill_value='extrapolate')(X_flat)
        rho_FD_interp = interp1d(x, rho[n-1,:], kind='linear', bounds_error=False, fill_value='extrapolate')(X_flat)
        v_LT_interp = interp1d(x, v_LT, kind='linear', bounds_error=False, fill_value='extrapolate')(X_flat)
        v_FD_interp = interp1d(x, v[n-1,:], kind='linear', bounds_error=False, fill_value='extrapolate')(X_flat)
        phi_FD_interp = interp1d(x, phi[n-1,:], kind='linear', bounds_error=False, fill_value='extrapolate')(X_flat)
        
        # Collect velocity data for limit calculation
        all_velocity_data.append(v_pred_x0)
        all_velocity_data.append(v_FD_interp)
        if (np.isclose(KY, 0.0)) and (a < 0.1):
            all_velocity_data.append(v_LT_interp)
        
        # Density comparison plots
        axes[i*3].plot(X, rho_pred0, color='c', linewidth=3, label="PINN")
        # Only plot Linear Theory when KY == 0 and amplitude is small
        if (np.isclose(KY, 0.0)) and (a < 0.1):
            axes[i*3].plot(X, rho_LT_interp, linestyle='dashed', color='firebrick', linewidth=2, label="Linear Theory")
        axes[i*3].plot(X, rho_FD_interp, linestyle='solid', color='black', linewidth=1, label="Finite Difference")
        axes[i*3].set_xlim(xmin, xmax)
        axes[i*3].set_title(f"Density at t={time:.1f}")
        axes[i*3].set_ylabel(r"$\rho$")
        axes[i*3].grid(True)
        axes[i*3].legend()
        axes[i*3].set_ylim(0.5*rho_o, 1.5*rho_o)
        
        # Velocity comparison plots
        axes[i*3+1].plot(X, v_pred_x0, color='c', linewidth=3, label="PINN")
        # Only plot Linear Theory when KY == 0 and amplitude is small
        if (np.isclose(KY, 0.0)) and (a < 0.1):
            axes[i*3+1].plot(X, v_LT_interp, linestyle='dashed', color='firebrick', linewidth=2, label="Linear Theory")
        axes[i*3+1].plot(X, v_FD_interp, linestyle='solid', color='black', linewidth=1, label="Finite Difference")
        axes[i*3+1].set_xlim(xmin, xmax)
        axes[i*3+1].set_title(f"Velocity at t={time:.1f}")
        axes[i*3+1].set_ylabel("$v_x$")
        axes[i*3+1].grid(True)
        axes[i*3+1].legend()
        
        # Potential comparison plots
        axes[i*3+2].plot(X, phi_pred0, color='c', linewidth=3, label="PINN")
        axes[i*3+2].plot(X, phi_FD_interp, linestyle='solid', color='black', linewidth=1, label="Finite Difference")
        axes[i*3+2].set_xlim(xmin, xmax)
        axes[i*3+2].set_title(f"Potential at t={time:.1f}")
        axes[i*3+2].set_ylabel(r"$\phi$")
        axes[i*3+2].set_xlabel("x")
        axes[i*3+2].grid(True)
        axes[i*3+2].legend()

    # Set velocity y-axis limits based on amplitude (same logic as create_1d_cross_sections_sinusoidal)
    # Calculate limits from all collected velocity data and apply consistently to all velocity plots
    if a < 0.1:
        # Default limits for small amplitude cases
        v_limu_default = 0.055
        v_liml_default = -0.055
    else:
        # Default limits for large amplitude cases
        v_limu_default = 0.6
        v_liml_default = -0.6
    
    # Calculate actual data range across all velocity data with padding
    if all_velocity_data:
        v_min = np.min([np.min(v_data) for v_data in all_velocity_data])
        v_max = np.max([np.max(v_data) for v_data in all_velocity_data])
        v_range = v_max - v_min
        # Add 10% padding to ensure all data stays within limits
        padding = max(0.01, 0.1 * v_range)
        v_liml_actual = v_min - padding
        v_limu_actual = v_max + padding
        
        # Ensure limits are at least as wide as default, but expand if data exceeds them
        v_liml_actual = min(v_liml_actual, v_liml_default)
        v_limu_actual = max(v_limu_actual, v_limu_default)
    else:
        v_liml_actual = v_liml_default
        v_limu_actual = v_limu_default
    
    # Apply consistent limits to all velocity plots
    for i in range(len(time_array_1d)):
        axes[i*3+1].set_ylim(v_liml_actual, v_limu_actual)

    plt.tight_layout()
    
    # Save the figure to output folder
    output_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "1d_comparison_plots.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 1D comparison plots to {save_path}")
    
    plt.show()



def create_growth_comparison_plot(net, initial_params, time_array_growth=None):
    """
    Create growth comparison plot showing density maximum evolution over time
    
    Args:
        net: Trained neural network
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_array_growth: Array of time points for growth analysis (default: 10 points from 0.1 to tmax)
    """
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    # rho_o imported from config.py
    num_of_waves = (xmax - xmin) / lam
    
    if time_array_growth is None:
        time_array_growth = np.linspace(0.1, tmax, 5)  # Reduced from 10 to 5 points
    
    print("Creating growth comparison plot...")
    
    Growth_LT_list = []
    Growth_FD_list = []
    Growth_PN_list = []

    for i, time in enumerate(time_array_growth):
        print(f"Processing growth point {i+1}/{len(time_array_growth)} at t={time:.2f}")
        # Get LAX solution with configured grid resolution
        x, rho, v, phi, n, rho_LT, rho_LT_max, rho_max_FD, v_LT = _timed_call(
            "LAX comparison slice (cpu)",
            lax_solution,
            time, FD_N_2D, _get_fd_nu(), lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=True, animation=True
        )
        
        # Get PINN solution
        X, rho_pred0, v_pred_x0, v_pred_y0, phi_pred0, rho_max_PN = plot_function(
            net, [time], initial_params, velocity=True, isplot=False, animation=True
        )
        
        Growth_LT = rho_LT_max - rho_o
        Growth_FD = rho_max_FD - rho_o  
        Growth_PN = rho_max_PN - rho_o
        
        Growth_LT_list.append(Growth_LT)
        Growth_FD_list.append(Growth_FD)
        Growth_PN_list.append(Growth_PN)

    # Plot growth comparison
    plt.figure(figsize=(8, 6))
    # Only plot Linear Theory when KY == 0 and amplitude is small
    if (np.isclose(KY, 0.0)) and (a < 0.1):
        plt.plot(time_array_growth, np.log(Growth_LT_list), marker='o', color='b', linewidth=2, label="Linear Theory")
    plt.plot(time_array_growth, np.log(Growth_FD_list), '--', marker='*', color='k', linewidth=3, label="Finite Difference")
    plt.plot(time_array_growth, np.log(Growth_PN_list), marker='^', markersize=8, linewidth=2, color='r', label="PINN")
    plt.xlabel("t", fontsize=14)
    plt.ylabel(r"$\log (\rho_{\rm max} - \rho_{0})$", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title("Growth Comparison: PINN vs Linear Theory vs Finite Difference")
    plt.tight_layout()
    
    # Save the figure to output folder
    output_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "growth_comparison_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved growth comparison plot to {save_path}")
    
    plt.show()



def create_all_plots(net, initial_params, include_growth=False,
                     fd_use_velocity_ps=None, fd_ps_index=None, fd_vel_rms=None, fd_random_seed=None):
    """
    Create only 2D surface plot grids (density and velocity). Optionally create FD grids and return figures.
    Uses cached FD data to avoid redundant solver calls.

    Args:
        net: Trained neural network
        initial_params: (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        include_growth: Unused now; kept for API compatibility.
        fd_use_velocity_ps: Override for FD velocity power spectrum flag (defaults to config)
        fd_ps_index: Override for FD power spectrum index (defaults to POWER_EXPONENT)
        fd_vel_rms: Override for FD velocity RMS (defaults to a*cs)
        fd_random_seed: Override for FD random seed (defaults to 1234)
    Returns:
        dict with figures/axes: {"pinn_density": (fig, axes), "pinn_velocity": (fig, axes),
                                 "fd_density": (fig, axes), "fd_velocity": (fig, axes)}
    """
    print("="*60)
    print("CREATING GRID VISUALIZATION PLOTS")
    print("="*60)
    
    # Use config defaults if not overridden to ensure consistency with PINN training
    if fd_use_velocity_ps is None:
        fd_use_velocity_ps = (str(PERTURBATION_TYPE).lower() == "power_spectrum")
    if fd_ps_index is None:
        fd_ps_index = POWER_EXPONENT
    if fd_vel_rms is None:
        fd_vel_rms = a * cs
    if fd_random_seed is None:
        fd_random_seed = RANDOM_SEED

    # Default time points for 2D surface plots
    time_points_2d = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Get time points for 1D cross-sections (if used)
    time_points_1d = TIMES_1D if isinstance(TIMES_1D, (list, tuple)) and len(TIMES_1D) > 0 else [0.5, 1.0, 1.5]
    
    # Combine all time points and remove duplicates
    all_time_points = sorted(list(set(time_points_2d + time_points_1d)))
    
    # Compute FD data cache once for all plots (2D and 1D)
    print(f"Computing FD cache for {len(all_time_points)} unique time points...")
    fd_cache = compute_fd_data_cache(
        initial_params, all_time_points,
        use_velocity_ps=fd_use_velocity_ps, ps_index=fd_ps_index,
        vel_rms=fd_vel_rms, random_seed=fd_random_seed
    )

    # 1. PINN density grid
    fig_den, axes_den = create_2d_surface_plots(net, initial_params, which="density", fd_cache=fd_cache)

    # 2. PINN velocity grid
    fig_vel, axes_vel = create_2d_surface_plots(net, initial_params, which="velocity", fd_cache=fd_cache)

    # 3. FD density grid - uses cached data
    fig_fd_den, axes_fd_den = create_2d_surface_plots_FD(initial_params, which="density",
                                                         use_velocity_ps=fd_use_velocity_ps, ps_index=fd_ps_index,
                                                         vel_rms=fd_vel_rms, random_seed=fd_random_seed,
                                                         fd_cache=fd_cache)

    # 4. FD velocity grid - uses cached data
    fig_fd_vel, axes_fd_vel = create_2d_surface_plots_FD(initial_params, which="velocity",
                                                         use_velocity_ps=fd_use_velocity_ps, ps_index=fd_ps_index,
                                                         vel_rms=fd_vel_rms, random_seed=fd_random_seed,
                                                         fd_cache=fd_cache)

    print("="*60)
    print("ALL GRID PLOTS COMPLETED!")
    print("="*60)

    # Display all created figures so they appear when called from train.py
    plt.show()

    result = {
        "pinn_density": (fig_den, axes_den),
        "pinn_velocity": (fig_vel, axes_vel),
        "fd_density": (fig_fd_den, axes_fd_den),
        "fd_velocity": (fig_fd_vel, axes_fd_vel),
        "fd_cache": fd_cache,  # Return cache for reuse in other plotting functions
    }
    
    return result



def create_density_growth_plot(net, initial_params, tmax, dt=0.1):
    """
    Create a PINN vs LAX density growth comparison plot.

    Plots over time: (1) maximum density, (2) log(rho_max - rho_o + eps).

    Args:
        net: trained PINN model
        initial_params: (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax_train)
        tmax: maximum time to plot (independent of training tmax)
        dt: temporal spacing (default 0.1)
    """
    xmin, xmax, ymin, ymax, rho_1, _alpha, lam, _output_folder, _tmax_train = initial_params
    
    # Unwrap network from list if needed
    net = net[0] if isinstance(net, list) else net
    num_of_waves = (xmax - xmin) / lam

    # Time grid (inclusive of tmax)
    time_points = np.arange(0.0, float(tmax) + 1e-9, float(dt))

    # Pre-compute FD cache for all time points to avoid redundant solver calls
    # This dramatically speeds up the plot generation, especially for 3D
    print(f"Pre-computing FD data cache for {len(time_points)} time points (this may take a moment)...")
    fd_cache = compute_fd_data_cache(
        initial_params, time_points, N=None, nu=_get_fd_nu(),
        use_velocity_ps=None, ps_index=None, vel_rms=None, random_seed=None
    )
    print("FD cache computed. Generating density growth plot...")

    pinn_max_list = []
    fd_max_list = []

    # PINN grid sampling settings (use N_GRID for consistency with FD solver)
    # Exclude right boundary for periodic domains to avoid double-counting
    Q = N_GRID
    xs = np.linspace(xmin, xmax, Q, endpoint=False)
    ys = np.linspace(ymin, ymax, Q, endpoint=False)
    TAU, PHI = np.meshgrid(xs, ys)
    if DIMENSION >= 3:
        ZETA = np.full_like(TAU, SLICE_Z)
    Xgrid = np.vstack([TAU.flatten(), PHI.flatten()]).T

    for idx, t in enumerate(time_points):
        # PINN evaluation on QxQ grid
        t_vec = t * np.ones(Q**2).reshape(Q**2, 1)
        pt_x = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
        pt_y = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device) if DIMENSION >= 2 else None
        pt_z = Variable(torch.from_numpy(ZETA.reshape(-1, 1)).float(), requires_grad=True).to(device) if DIMENSION >= 3 else None
        pt_t = Variable(torch.from_numpy(t_vec).float(), requires_grad=True).to(device)
        out = net(_build_input_list(pt_x, pt_t, pt_y, pt_z))
        rho_tensor, *_ = _split_outputs(out)
        rho_pinn = rho_tensor.detach().cpu().numpy().reshape(Q, Q)
        pinn_max_list.append(np.max(rho_pinn))

        # LAX/FD evaluation using cached data
        if DIMENSION == 3:
            # Use cached FD data
            if t in fd_cache:
                cache_data = fd_cache[t]
                rho_fd = cache_data['rho']
                fd_max_list.append(np.max(rho_fd))
            else:
                # Fallback: compute on the fly if cache miss (shouldn't happen)
                use_velocity_ps = (str(PERTURBATION_TYPE).lower() == "power_spectrum")
                result = _call_unified_3d_solver(
                    time=t, lam=lam, num_of_waves=num_of_waves, rho_1=rho_1, nu=_get_fd_nu(),
                    use_velocity_ps=use_velocity_ps, ps_index=POWER_EXPONENT,
                    vel_rms=a*cs, random_seed=RANDOM_SEED
                )
                rho_fd = result.density
                z_grid = result.coordinates['z']
                z_idx = np.argmin(np.abs(z_grid - SLICE_Z))
                fd_max_list.append(np.max(rho_fd[:, :, z_idx]))
        else:
            # Use cached FD data for 2D
            if t in fd_cache:
                cache_data = fd_cache[t]
                rho_fd = cache_data['rho']
                fd_max_list.append(np.max(rho_fd))
            else:
                # Fallback: compute on the fly if cache miss (shouldn't happen)
                if torch.cuda.is_available():
                    _clear_cuda_cache()
                    use_velocity_ps = (str(PERTURBATION_TYPE).lower() == "power_spectrum")
                    if idx == 0:
                        print(f"Using GPU solver for density growth plot (CUDA available)")
                    x_fd, rho_fd, _vx_fd, _vy_fd, _phi_fd, _n, _rho_max = _timed_call(
                        "LAX 2D (torch)",
                        lax_solution_torch,
                        time_val=t, N=N_GRID, nu=_get_fd_nu(), lam=lam, num_of_waves=num_of_waves, rho_1=rho_1,
                        gravity=True, use_velocity_ps=use_velocity_ps, ps_index=POWER_EXPONENT,
                        vel_rms=a*cs, random_seed=RANDOM_SEED
                    )
                else:
                    if str(PERTURBATION_TYPE).lower() == "power_spectrum":
                        if _shared_vx_np is not None and _shared_vy_np is not None:
                            n_fd_use = int(_shared_vx_np.shape[0])
                            x_fd, rho_fd, _vx_fd, _vy_fd, _phi_fd, _n, _rho_max = _timed_call(
                                "LAX 2D (shared-field cpu)",
                                lax_solution,
                                t, n_fd_use, _get_fd_nu(), lam, num_of_waves, rho_1,
                                gravity=True, isplot=False, comparison=False, animation=True,
                                vx0_shared=_shared_vx_np, vy0_shared=_shared_vy_np
                            )
                        else:
                            x_fd, rho_fd, _vx_fd, _vy_fd, _phi_fd, _n, _rho_max = _timed_call(
                                "LAX 2D (power cpu)",
                                lax_solution,
                                t, N_GRID, _get_fd_nu(), lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True,
                                use_velocity_ps=True, ps_index=POWER_EXPONENT, vel_rms=a*cs, random_seed=RANDOM_SEED
                            )
                    else:
                        x_fd, rho_fd, _vx_fd, _vy_fd, _phi_fd, _n, _rho_max = _timed_call(
                            "LAX 2D (sinusoidal cpu)",
                        lax_solution,
                        t, FD_N_2D, _get_fd_nu(), lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True,
                            use_velocity_ps=False
                        )
                fd_max_list.append(np.max(rho_fd))

    # Build figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # (1) Max density vs time
    axes[0].plot(time_points, fd_max_list, label="LAX", color='k', linewidth=2)
    axes[0].plot(time_points, pinn_max_list, label="PINN", color='c', linewidth=2)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"$\rho_{\max}$")
    axes[0].set_title("Maximum Density vs Time")
    axes[0].grid(True)
    axes[0].legend()
    # Annotate parameters on the plot
    try:
        param_str = f"a={a}, power_index={POWER_EXPONENT}"
        axes[0].text(0.02, 0.95, param_str, transform=axes[0].transAxes,
                     fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.6))
    except Exception:
        pass

    # (2) log growth vs time
    eps = 1e-12
    axes[1].plot(time_points, np.log(np.maximum(np.array(fd_max_list) - rho_o, 0.0) + eps), label="LAX", color='k', linewidth=2)
    axes[1].plot(time_points, np.log(np.maximum(np.array(pinn_max_list) - rho_o, 0.0) + eps), label="PINN", color='c', linewidth=2)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$\log(\rho_{\max} - \rho_0)$")
    axes[1].set_title("Density Growth (log)")
    axes[1].grid(True)
    axes[1].legend()
    # Mirror annotation on second axis
    try:
        param_str = f"a={a}, power_index={POWER_EXPONENT}"
        axes[1].text(0.02, 0.95, param_str, transform=axes[1].transAxes,
                     fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.6))
    except Exception:
        pass

    plt.tight_layout()

    # Save figure
    output_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "density_growth_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved density growth comparison plot to {save_path}")

    plt.show()

    return fig, axes



def create_1d_cross_sections_sinusoidal(net, initial_params, time_points=None, y_fixed=0.6, N_fd=1000, nu_fd=None,
                                        fd_cache=None):
    """
    Create 1D cross-section plots at fixed y for sinusoidal perturbations, comparing
    PINN vs Linear Theory vs 1D LAX (sinusoidal).

    Args:
        net: trained network
        initial_params: (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_points: list of times to plot
        y_fixed: y value for the 1D slice through the 2D domain
        N_fd: grid size for 1D LAX solver (only used if fd_cache is None)
        nu_fd: Courant number for 1D LAX solver (only used if fd_cache is None)
        fd_cache: Optional dictionary mapping time -> FD data (if provided, solver is not called)
    """
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, _output_folder, _tmax = initial_params
    if nu_fd is None:
        nu_fd = _get_fd_nu()
    
    # Unwrap network from list if needed
    net = net[0] if isinstance(net, list) else net
    num_of_waves = (xmax - xmin) / lam

    if time_points is None:
        time_points = TIMES_1D if isinstance(TIMES_1D, (list, tuple)) and len(TIMES_1D) > 0 else [0.5, 1.0, 1.5]

    # Use baseline density 1.0 for Linear Theory reference
    rho_base = 1.0
    jeans = np.sqrt(4*np.pi**2*cs**2/(const*G*rho_base))
    k = np.sqrt(KX**2 + KY**2 + KZ**2)
    v1_lt = (rho_1 / rho_base) * (alpha / k) if k > 1e-12 else 0.0

    # Build x grid for PINN slice
    X = np.linspace(xmin, xmax, 1000).reshape(1000, 1)
    Y = y_fixed * np.ones_like(X)
    z_fixed = SLICE_Z if DIMENSION >= 3 else None
    if DIMENSION >= 3:
        Z = z_fixed * np.ones_like(X)
    else:
        Z = None

    # Create 2 rows x T columns panel layout matching target style
    T = len(time_points)
    fig = plt.figure(figsize=(6*T, 8), constrained_layout=False)
    grid = plt.GridSpec(4, T, figure=fig, hspace=0.12, wspace=0.18)

    for row_idx, t in enumerate(time_points):
        # PINN predictions at fixed y (and z for 3D)
        t_arr = t * np.ones_like(X)
        pt_x = Variable(torch.from_numpy(X).float(), requires_grad=True).to(device)
        pt_y = Variable(torch.from_numpy(Y).float(), requires_grad=True).to(device)
        # For 3D power spectrum, align PINN z-slice with cached FD z
        if Z is not None and fd_cache is not None and t in fd_cache and DIMENSION == 3:
            z_val_use = fd_cache[t].get('z_val', SLICE_Z)
            Z_use = z_val_use * np.ones_like(X)
            pt_z = Variable(torch.from_numpy(Z_use).float(), requires_grad=True).to(device)
        else:
            pt_z = Variable(torch.from_numpy(Z).float(), requires_grad=True).to(device) if Z is not None else None
        pt_t = Variable(torch.from_numpy(t_arr).float(), requires_grad=True).to(device)
        # Use _build_input_list helper to ensure correct coordinate ordering
        inputs = _build_input_list(pt_x, pt_t, pt_y, pt_z)
        out = net(inputs)
        rho_pinn = out[:, 0:1].data.cpu().numpy().reshape(-1)
        vx_pinn = out[:, 1:2].data.cpu().numpy().reshape(-1)
        # potential not used in cross-section plots

        show_lt = np.isclose(KY, 0.0) and np.isclose(KZ, 0.0) and (a < 0.1)
        if show_lt:
            phase = KX * X[:, 0] + KY * y_fixed + (KZ * z_fixed if z_fixed is not None else 0.0)
            if lam >= jeans:
                rho_lt = rho_base + rho_1*np.exp(alpha * t)*np.cos(phase)
                if k > 0:
                    vx_lt = -v1_lt*np.exp(alpha * t)*np.sin(phase) * (KX / k)
                else:
                    vx_lt = -v1_lt*np.exp(alpha * t)*np.sin(phase)
            else:
                omega = np.sqrt(cs**2 * (KX**2 + KY**2 + KZ**2) - const*G*rho_base)
                rho_lt = rho_base + rho_1*np.cos(omega * t - phase)
                if k > 0:
                    vx_lt = v1_lt*np.cos(omega * t - phase) * (KX / k)
                else:
                    vx_lt = v1_lt*np.cos(omega * t - phase)

        from scipy.interpolate import interp1d
        
        # Use cached data if available
        if fd_cache is not None and t in fd_cache:
            cache_data = fd_cache[t]
            if DIMENSION == 3:
                # For 3D, use the volume data from cache
                if 'rho_vol' in cache_data:
                    rho_fd_3d = cache_data['rho_vol']
                    vx_fd_3d = cache_data['vx_vol']
                    phi_fd_3d = cache_data['phi_vol']
                    y_fd = cache_data['y']
                    z_fd = cache_data['z']
                    z_idx = cache_data.get('z_idx', np.argmin(np.abs(z_fd - SLICE_Z)))
                else:
                    # Fallback to 2D slice if volume not cached
                    rho_fd_3d = cache_data['rho'][:, :, np.newaxis]
                    vx_fd_3d = cache_data['vx'][:, :, np.newaxis]
                    phi_fd_3d = cache_data['phi'][:, :, np.newaxis] if cache_data['phi'] is not None else None
                    y_fd = cache_data['y']
                    z_fd = np.array([SLICE_Z])
                    z_idx = 0
                y_idx = np.argmin(np.abs(y_fd - y_fixed))
                rho_fd = rho_fd_3d[:, y_idx, z_idx]
                v_fd = vx_fd_3d[:, y_idx, z_idx]
                phi_fd_slice = phi_fd_3d[:, y_idx, z_idx] if phi_fd_3d is not None else None
                x_fd_line = cache_data['x']
            else:
                x_fd_2d = cache_data['x']
                y_fd_2d = cache_data['y']
                rho_fd_2d = cache_data['rho']
                vx_fd_2d = cache_data['vx']
                _phi_fd_2d = cache_data['phi'] if cache_data['phi'] is not None else np.zeros_like(rho_fd_2d)
                y_idx = np.argmin(np.abs(y_fd_2d - y_fixed))
                rho_fd = rho_fd_2d[:, y_idx]
                v_fd = vx_fd_2d[:, y_idx]
                phi_fd_slice = _phi_fd_2d[:, y_idx]
                x_fd_line = x_fd_2d
        else:
            # Compute FD data if not cached
            if DIMENSION == 3:
                # Use unified solver with proper IC type support
                use_velocity_ps = (str(PERTURBATION_TYPE).lower() == "power_spectrum")
                result = _call_unified_3d_solver(
                    time=t, lam=lam, num_of_waves=num_of_waves, rho_1=rho_1, nu=nu_fd,
                    use_velocity_ps=use_velocity_ps, ps_index=POWER_EXPONENT,
                    vel_rms=a*cs, random_seed=RANDOM_SEED
                )
                # Extract results from unified solver
                x_fd = result.coordinates['x']
                y_fd = result.coordinates['y']
                z_fd = result.coordinates['z']
                rho_fd_3d = result.density
                vx_fd_3d, vy_fd_3d, vz_fd_3d = result.velocity_components
                phi_fd_3d = result.potential
                y_idx = np.argmin(np.abs(y_fd - y_fixed))
                z_idx = np.argmin(np.abs(z_fd - SLICE_Z))
                rho_fd = rho_fd_3d[:, y_idx, z_idx]
                v_fd = vx_fd_3d[:, y_idx, z_idx]
                phi_fd_slice = phi_fd_3d[:, y_idx, z_idx]
                x_fd_line = x_fd
            else:
                if torch.cuda.is_available():
                    _clear_cuda_cache()
                    if row_idx == 0:
                        print(f"Using GPU solver for 1D cross section plot (CUDA available)")
                    use_velocity_ps = (str(PERTURBATION_TYPE).lower() == "power_spectrum")
                    x_fd_2d, rho_fd_2d, vx_fd_2d, vy_fd_2d, phi_fd_2d_torch, _n, _rho_max = _timed_call(
                        "LAX 2D slice (torch)",
                        lax_solution_torch,
                        time_val=t, N=FD_N_2D, nu=nu_fd, lam=lam, num_of_waves=num_of_waves, rho_1=rho_1,
                        gravity=True, use_velocity_ps=use_velocity_ps, ps_index=POWER_EXPONENT,
                        vel_rms=a*cs, random_seed=RANDOM_SEED
                    )
                    # Handle case where torch version returns None for phi (compute dummy phi if needed)
                    if phi_fd_2d_torch is None:
                        _phi_fd_2d = np.zeros_like(rho_fd_2d)
                    else:
                        _phi_fd_2d = phi_fd_2d_torch
                else:
                    x_fd_2d, rho_fd_2d, vx_fd_2d, vy_fd_2d, _phi_fd_2d, _n, _rho_max = _timed_call(
                        "LAX 2D slice (cpu)",
                        lax_solution,
                        t, FD_N_2D, nu_fd, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True
                    )
                y_fd_2d = np.linspace(0, lam * num_of_waves, rho_fd_2d.shape[1], endpoint=False)
                y_idx = np.argmin(np.abs(y_fd_2d - y_fixed))
                rho_fd = rho_fd_2d[:, y_idx]
                v_fd = vx_fd_2d[:, y_idx]
                phi_fd_slice = _phi_fd_2d[:, y_idx]
                x_fd_line = x_fd_2d

        rho_fd_interp = interp1d(x_fd_line, rho_fd, kind='linear', bounds_error=False, fill_value='extrapolate')(X[:, 0])
        v_fd_interp = interp1d(x_fd_line, v_fd, kind='linear', bounds_error=False, fill_value='extrapolate')(X[:, 0])
        phi_fd_interp = interp1d(x_fd_line, phi_fd_slice, kind='linear', bounds_error=False, fill_value='extrapolate')(X[:, 0])

        # Column index
        c = row_idx
        # Top row: density
        ax_rho = fig.add_subplot(grid[0, c])
        ax_rho.plot(X[:, 0], rho_pinn, label="GRINN", color='c', linewidth=2)
        # Only plot Linear Theory when enabled in config, KY == 0, and amplitude is small
        if SHOW_LINEAR_THEORY and show_lt:
            ax_rho.plot(X[:, 0], rho_lt, label="LT", linestyle='--', color='firebrick', linewidth=1.5)
        ax_rho.plot(X[:, 0], rho_fd_interp, label="FD", color='k', linewidth=1)
        ax_rho.set_title(f"t={t:.1f}")
        ax_rho.set_ylabel(r"$\rho$")
        ax_rho.grid(True)
        # Dynamic y-axis limits based on actual data with padding (similar to t=3.0 plot)
        # Collect all density values that are plotted
        rho_all = [rho_pinn, rho_fd_interp]
        if SHOW_LINEAR_THEORY and show_lt:
            rho_all.append(rho_lt)
        rho_min = min(np.min(rho) for rho in rho_all)
        rho_max = max(np.max(rho) for rho in rho_all)
        # Add padding: ~10% of the data range on each side (similar to t=3.0 example)
        rho_range = rho_max - rho_min
        padding = max(0.1 * rho_range, 0.05)  # At least 0.05 units of padding
        liml = rho_min - padding
        limu = rho_max + padding
        # Commented out hardcoded limits:
        # if a < 0.1:
        #     limu = 1.2*rho_o
        #     liml = .8*rho_o
        # else:
        #     limu = 3.0*rho_o
        #     liml = -1.0*rho_o
        ax_rho.set_ylim(liml, limu)
        if c == 0:
            ax_rho.legend(loc='upper right', fontsize=8)

        # Second row: epsilon for density using symmetric percent with absolute numerator
        # ε = 200 * |G - R| / (G + R) with more robust denominator
        eps_rho = 200.0 * np.abs(rho_pinn - rho_fd_interp) / (rho_pinn + rho_fd_interp + 1e-6)
        ax_eps_rho = fig.add_subplot(grid[1, c])
        ax_eps_rho.plot(X[:, 0], eps_rho, color='k', linewidth=1, label='FD')
        # Only plot Linear Theory epsilon when enabled in config, KY == 0, and amplitude is small
        if SHOW_LINEAR_THEORY and show_lt:
            eps_rho_lt = 200.0 * np.abs(rho_pinn - rho_lt) / (rho_pinn + rho_lt + 1e-6)
            ax_eps_rho.plot(X[:, 0], eps_rho_lt, color='firebrick', linestyle='--', linewidth=1, label='LT')
        ax_eps_rho.set_ylabel(r"$\varepsilon$")
        ax_eps_rho.grid(True)
        if c == 0:
            ax_eps_rho.legend(loc='upper right', fontsize=8)

        # Third row: velocity
        ax_v = fig.add_subplot(grid[2, c])
        ax_v.plot(X[:, 0], vx_pinn, label="GRINN", color='c', linewidth=2)
        # Only plot Linear Theory when enabled in config, KY == 0, and amplitude is small
        if SHOW_LINEAR_THEORY and show_lt:
            ax_v.plot(X[:, 0], vx_lt, label="LT", linestyle='--', color='firebrick', linewidth=1.5)
        ax_v.plot(X[:, 0], v_fd_interp, label="FD", color='k', linewidth=1)
        ax_v.set_ylabel(r"$v$")
        ax_v.grid(True)
        # Dynamic y-axis limits based on actual data with padding (similar to t=3.0 plot)
        # Collect all velocity values that are plotted
        v_all = [vx_pinn, v_fd_interp]
        if SHOW_LINEAR_THEORY and show_lt:
            v_all.append(vx_lt)
        v_min = min(np.min(v) for v in v_all)
        v_max = max(np.max(v) for v in v_all)
        # Add padding: ~10% of the data range on each side (similar to t=3.0 example)
        v_range = v_max - v_min
        padding = max(0.1 * v_range, 0.005)  # At least 0.005 units of padding
        liml = v_min - padding
        limu = v_max + padding
        # Commented out hardcoded limits:
        # if a < 0.1:
        #     limu = 0.055
        #     liml = -0.055
        # else:
        #     limu = 0.6
        #     liml = -0.6
        ax_v.set_ylim(liml, limu)
        if c == 0:
            ax_v.legend(loc='upper right', fontsize=8)

        # Fourth row: epsilon for velocity using notebook-style +1 offset (symmetric percent with shift)
        # ε = 200 * |(v_pred+1) - (v_ref+1)| / ((v_pred+1) + (v_ref+1)) = 200 * |v_pred - v_ref| / (v_pred + v_ref + 2)
        v_ref = v_fd_interp
        v_pred = vx_pinn
        eps_v = 200.0 * np.abs(v_pred - v_ref) / (v_pred + v_ref + 2.0)
        ax_eps_v = fig.add_subplot(grid[3, c])
        ax_eps_v.plot(X[:, 0], eps_v, color='k', linewidth=1, label='FD')
        # Only plot Linear Theory epsilon when enabled in config, KY == 0, and amplitude is small
        if SHOW_LINEAR_THEORY and show_lt:
            eps_v_lt = 200.0 * np.abs(v_pred - vx_lt) / (v_pred + vx_lt + 2.0)
            ax_eps_v.plot(X[:, 0], eps_v_lt, color='firebrick', linestyle='--', linewidth=1, label='LT')
        ax_eps_v.set_xlabel("x")
        ax_eps_v.set_ylabel(r"$\varepsilon$")
        ax_eps_v.grid(True)
        if c == 0:
            ax_eps_v.legend(loc='upper right', fontsize=8)

        # No potential subplot per request

    # Reduce outer margins similar to notebook style
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.10, wspace=0.18, hspace=0.12)
    
    # Save the figure to output folder
    output_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "1d_cross_sections_sinusoidal.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 1D cross sections plot to {save_path}")
    
    plt.show()

    return fig



def create_5x3_comparison_table(net, initial_params, which="density", N=200, nu=None,
                                use_velocity_ps=None, ps_index=None, vel_rms=None, random_seed=None,
                                fd_cache=None):
    """
    Create 5x3 comparison table showing PINN, FD, and epsilon metric at 5 time snapshots
    
    Args:
        net: Trained neural network
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        which: "density" or "velocity"
        N: Grid resolution for LAX solver (only used if fd_cache is None)
        nu: Courant number for LAX solver (only used if fd_cache is None)
        use_velocity_ps: Whether to use velocity power spectrum for FD (defaults to config) - only used if fd_cache is None
        ps_index: Power spectrum index for FD (defaults to POWER_EXPONENT) - only used if fd_cache is None
        vel_rms: Velocity RMS for FD (defaults to a*cs) - only used if fd_cache is None
        random_seed: Random seed for FD (defaults to 1234) - only used if fd_cache is None
        fd_cache: Optional dictionary mapping time -> FD data (if provided, solver is not called)
    """
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    if nu is None:
        nu = _get_fd_nu()
    
    # Use config defaults if not specified to ensure consistency with PINN training
    if use_velocity_ps is None:
        use_velocity_ps = (str(PERTURBATION_TYPE).lower() == "power_spectrum")
    if ps_index is None:
        ps_index = POWER_EXPONENT
    if vel_rms is None:
        vel_rms = a * cs
    if random_seed is None:
        random_seed = RANDOM_SEED
    
    # Unwrap network from list if needed
    net = net[0] if isinstance(net, list) else net
    
    # Generate 5 time points uniformly distributed over [0, tmax]
    time_points = np.linspace(0.0, float(tmax), 5)
    
    print(f"Creating 5x3 comparison table for {which}...")
    
    # Create 5x3 subplot grid
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    
    # Store data for consistent color limits
    pinn_data = []
    fd_data = []
    pinn_velocity_data = []  # Store velocity components separately
    fd_velocity_data = []    # Store velocity components separately
    
    # First pass: collect data to determine consistent color limits
    for i, t in enumerate(time_points):
        # print(f"Collecting data for t = {t:.2f}")  # Commented out to reduce output noise
        
        # Get PINN data - use N_GRID for consistency with FD solver
        # Exclude right boundary for periodic domains to avoid double-counting
        Q = N_GRID
        xs = np.linspace(xmin, xmax, Q, endpoint=False)
        ys = np.linspace(ymin, ymax, Q, endpoint=False)
        tau, phi = np.meshgrid(xs, ys) 
        Xgrid = np.vstack([tau.flatten(), phi.flatten()]).T
        t_00 = t * np.ones(Q**2).reshape(Q**2, 1)
        
        pt_x_collocation = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
        pt_y_collocation = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device) if DIMENSION >= 2 else None
        if DIMENSION >= 3:
            z_val_use = SLICE_Z
            if fd_cache is not None and t in fd_cache:
                z_val_use = fd_cache[t].get('z_val', SLICE_Z)
            pt_z_collocation = Variable(torch.from_numpy(np.full((Q**2, 1), z_val_use)).float(), requires_grad=True).to(device)
        else:
            pt_z_collocation = None
        pt_t_collocation = Variable(torch.from_numpy(t_00).float(), requires_grad=True).to(device)
        
        # Ensure inputs are on the same device as the model
        net_device = next(net.parameters()).device
        pt_x_collocation = pt_x_collocation.to(net_device)
        pt_t_collocation = pt_t_collocation.to(net_device)
        if pt_y_collocation is not None:
            pt_y_collocation = pt_y_collocation.to(net_device)
        if pt_z_collocation is not None:
            pt_z_collocation = pt_z_collocation.to(net_device)
        output_00 = net(_build_input_list(pt_x_collocation, pt_t_collocation, pt_y_collocation, pt_z_collocation))
        
        rho_tensor, vx_tensor, vy_tensor, _, _ = _split_outputs(output_00)
        if which == "density":
            pinn_field = rho_tensor.detach().cpu().numpy().reshape(Q, Q)
        else:
            vx_grid = vx_tensor.detach().cpu().numpy().reshape(Q, Q)
            vy_grid = vy_tensor.detach().cpu().numpy().reshape(Q, Q) if vy_tensor is not None else np.zeros_like(vx_grid)
            pinn_field = np.sqrt(vx_grid**2 + vy_grid**2)
        pinn_vx = vx_tensor.detach().cpu().numpy().reshape(Q, Q)
        pinn_vy = vy_tensor.detach().cpu().numpy().reshape(Q, Q) if vy_tensor is not None else np.zeros_like(pinn_vx)
        
        # Debug output (commented out to reduce noise)
        # print(f"  PINN {which} range: [{np.min(pinn_field):.6f}, {np.max(pinn_field):.6f}], std: {np.std(pinn_field):.6f}")
        
        # Get FD data - use cached data if available, otherwise compute
        num_of_waves = (xmax - xmin) / lam
        
        if fd_cache is not None and t in fd_cache:
            # Use cached data
            cache_data = fd_cache[t]
            if DIMENSION == 3:
                # For 3D, use volume data from cache if available, otherwise use 2D slice
                if 'rho_vol' in cache_data:
                    rho_vol = cache_data['rho_vol']
                    vx_vol = cache_data['vx_vol']
                    vy_vol = cache_data['vy_vol']
                    phi_vol = cache_data.get('phi_vol')
                    x_fd = cache_data['x']
                    y_fd = cache_data['y']
                    z_fd = cache_data['z']
                    z_idx = cache_data.get('z_idx', np.argmin(np.abs(cache_data['z'] - SLICE_Z)))
                else:
                    # Fallback to 2D slice
                    rho_vol = cache_data['rho'][:, :, np.newaxis]
                    vx_vol = cache_data['vx'][:, :, np.newaxis]
                    vy_vol = cache_data['vy'][:, :, np.newaxis]
                    phi_vol = cache_data.get('phi')
                    if phi_vol is not None:
                        phi_vol = phi_vol[:, :, np.newaxis]
                    x_fd = cache_data['x']
                    y_fd = cache_data['y']
                    z_fd = np.array([SLICE_Z])
                    z_idx = 0
                rho_fd = rho_vol[:, :, z_idx]
                vx_fd = vx_vol[:, :, z_idx]
                vy_fd = vy_vol[:, :, z_idx]
                phi_fd = phi_vol[:, :, z_idx] if phi_vol is not None else np.zeros_like(rho_fd)
                X_fd, Y_fd = np.meshgrid(x_fd, y_fd, indexing='ij')
            else:
                x_fd = cache_data['x']
                y_fd = cache_data['y']
                rho_fd = cache_data['rho']
                vx_fd = cache_data['vx']
                vy_fd = cache_data['vy']
                phi_fd = cache_data.get('phi')
                if phi_fd is None:
                    phi_fd = np.zeros_like(rho_fd)
                X_fd, Y_fd = np.meshgrid(x_fd, y_fd, indexing='ij')
        else:
            # Compute FD data if not cached
            if DIMENSION == 3:
                # Use unified solver with proper IC type support
                result = _call_unified_3d_solver(
                    time=t, lam=lam, num_of_waves=num_of_waves, rho_1=rho_1, nu=nu,
                    use_velocity_ps=use_velocity_ps, ps_index=ps_index,
                    vel_rms=vel_rms, random_seed=random_seed
                )
                # Extract results from unified solver
                x_fd = result.coordinates['x']
                y_fd = result.coordinates['y']
                z_fd = result.coordinates['z']
                rho_vol = result.density
                vx_vol, vy_vol, vz_vol = result.velocity_components
                phi_vol = result.potential
                z_idx = np.argmin(np.abs(z_fd - SLICE_Z))
                rho_fd = rho_vol[:, :, z_idx]
                vx_fd = vx_vol[:, :, z_idx]
                vy_fd = vy_vol[:, :, z_idx]
                phi_fd = phi_vol[:, :, z_idx] if phi_vol is not None else np.zeros_like(rho_fd)
                X_fd, Y_fd = np.meshgrid(x_fd, y_fd, indexing='ij')
            else:
                if str(PERTURBATION_TYPE).lower() == "power_spectrum":
                    if _shared_vx_np is not None and _shared_vy_np is not None:
                        x_fd, rho_fd, vx_fd, vy_fd, phi_fd, _n, _rho_max = _timed_call(
                            "LAX 2D (shared-field cpu)",
                            lax_solution,
                            t, N, nu, lam, num_of_waves, rho_1,
                            gravity=True, isplot=False, comparison=False, animation=True,
                            vx0_shared=_shared_vx_np, vy0_shared=_shared_vy_np
                        )
                    else:
                        if torch.cuda.is_available():
                            _clear_cuda_cache()
                            x_fd, rho_fd, vx_fd, vy_fd, phi_fd_torch, _n, _rho_max = _timed_call(
                                "LAX 2D (torch)",
                                lax_solution_torch,
                                time_val=t, N=N, nu=nu, lam=lam, num_of_waves=num_of_waves, rho_1=rho_1,
                                gravity=True, use_velocity_ps=True, ps_index=POWER_EXPONENT, vel_rms=a*cs, random_seed=RANDOM_SEED
                            )
                            phi_fd = phi_fd_torch if phi_fd_torch is not None else np.zeros_like(rho_fd)
                        else:
                            x_fd, rho_fd, vx_fd, vy_fd, phi_fd, _n, _rho_max = _timed_call(
                                "LAX 2D (power cpu)",
                                lax_solution,
                                t, N, nu, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True,
                                use_velocity_ps=True, ps_index=POWER_EXPONENT, vel_rms=a*cs, random_seed=RANDOM_SEED
                            )
                else:
                    if torch.cuda.is_available():
                        _clear_cuda_cache()
                        x_fd, rho_fd, vx_fd, vy_fd, phi_fd_torch, _n, _rho_max = _timed_call(
                            "LAX 2D (torch)",
                            lax_solution_torch,
                            time_val=t, N=N, nu=nu, lam=lam, num_of_waves=num_of_waves, rho_1=rho_1,
                            gravity=True, use_velocity_ps=False, ps_index=ps_index, vel_rms=vel_rms, random_seed=random_seed
                        )
                        phi_fd = phi_fd_torch if phi_fd_torch is not None else np.zeros_like(rho_fd)
                    else:
                        x_fd, rho_fd, vx_fd, vy_fd, phi_fd, _n, _rho_max = _timed_call(
                            "LAX 2D (sinusoidal cpu)",
                            lax_solution,
                            t, N, nu, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True,
                            use_velocity_ps=False, ps_index=ps_index, vel_rms=vel_rms, random_seed=random_seed
                        )
                Lx = lam * num_of_waves
                y_fd = np.linspace(0.0, Lx, rho_fd.shape[1], endpoint=False)
                X_fd, Y_fd = np.meshgrid(x_fd, y_fd, indexing='ij')
        
        if which == "density":
            fd_field = rho_fd
        else:  # velocity magnitude
            fd_field = np.sqrt(vx_fd**2 + vy_fd**2)
        
        # Interpolate FD data to PINN grid for comparison using single robust method
        from scipy.interpolate import RegularGridInterpolator
        points_fd = np.column_stack([X_fd.ravel(), Y_fd.ravel()])
        points_pinn = np.column_stack([tau.ravel(), phi.ravel()])
        
        # Use RegularGridInterpolator for robust interpolation with proper boundary handling
        # This avoids the interpolation cascade issues and provides consistent results
        interpolator = RegularGridInterpolator(
            (x_fd, y_fd), fd_field, 
            method='linear', 
            bounds_error=False, 
            fill_value=None  # extrapolate
        )
        fd_field_interp = interpolator(points_pinn)
        
        fd_field_interp = fd_field_interp.reshape(Q, Q)
        
        # Interpolate FD velocity components for vector plots using single robust method
        vx_interpolator = RegularGridInterpolator(
            (x_fd, y_fd), vx_fd, 
            method='linear', 
            bounds_error=False, 
            fill_value=None  # extrapolate
        )
        fd_vx_interp = vx_interpolator(points_pinn)
        
        vy_interpolator = RegularGridInterpolator(
            (x_fd, y_fd), vy_fd, 
            method='linear', 
            bounds_error=False, 
            fill_value=None  # extrapolate
        )
        fd_vy_interp = vy_interpolator(points_pinn)
        
        fd_vx_interp = fd_vx_interp.reshape(Q, Q)
        fd_vy_interp = fd_vy_interp.reshape(Q, Q)
        
        pinn_data.append(pinn_field)
        fd_data.append(fd_field_interp)
        
        # Store velocity components for vector plots (both density and velocity plots)
        pinn_velocity_data.append((pinn_vx, pinn_vy))
        fd_velocity_data.append((fd_vx_interp, fd_vy_interp))
    
    # Use individual color limits for each plot (like animation) to show dynamic range
    # This allows collapse features to be visible, rather than using global limits
    
    # Second pass: create plots
    for i, t in enumerate(time_points):
        pinn_field = pinn_data[i]
        fd_field = fd_data[i]
        
        # Extract velocity components for vector plots
        pinn_vx, pinn_vy = pinn_velocity_data[i]
        fd_vx, fd_vy = fd_velocity_data[i]
        
        # Calculate epsilon metric: error normalized by peak FD magnitude
        # Avoids near-zero blowup while staying relative to the field scale
        field_scale = np.max(np.abs(fd_field))
        if field_scale < 1e-12:
            field_scale = 1.0
        epsilon_metric = np.abs(pinn_field - fd_field) / field_scale * 100.0
        
        # Column 1: PINN - use individual color limits like animation
        ax_pinn = axes[i, 0]
        if which == "density":
            pc_pinn = ax_pinn.pcolormesh(tau, phi, pinn_field, shading='auto', cmap='YlOrBr', 
                                       vmin=np.min(pinn_field), vmax=np.max(pinn_field))
        else:
            pc_pinn = ax_pinn.pcolormesh(tau, phi, pinn_field, shading='auto', cmap='viridis', 
                                       vmin=np.min(pinn_field), vmax=np.max(pinn_field))
        
        # Add velocity vectors for both density and velocity plots
        if pinn_vx is not None and pinn_vy is not None:
            # Subsample vectors for clarity (similar to analyze_lax.py)
            skip_x = max(1, Q // 20)
            skip_y = max(1, Q // 20)
            skip = (slice(None, None, skip_x), slice(None, None, skip_y))
            ax_pinn.quiver(tau[skip], phi[skip], pinn_vx[skip], pinn_vy[skip], 
                          color='k', headwidth=3.0, width=0.003, alpha=0.7)
        
        ax_pinn.set_title(f"PINN {which.title()}, t={t:.2f}")
        ax_pinn.set_xlim(xmin, xmax)
        ax_pinn.set_ylim(ymin, ymax)
        cbar_pinn = plt.colorbar(pc_pinn, ax=ax_pinn, shrink=0.6)
        cbar_pinn.ax.set_title(r"$\rho$" if which == "density" else r"$|v|$", fontsize=14)
        
        # Column 2: FD - use individual color limits
        ax_fd = axes[i, 1]
        if which == "density":
            pc_fd = ax_fd.pcolormesh(tau, phi, fd_field, shading='auto', cmap='YlOrBr', 
                                    vmin=np.min(fd_field), vmax=np.max(fd_field))
        else:
            pc_fd = ax_fd.pcolormesh(tau, phi, fd_field, shading='auto', cmap='viridis', 
                                    vmin=np.min(fd_field), vmax=np.max(fd_field))
        
        # Add velocity vectors for both density and velocity plots
        if fd_vx is not None and fd_vy is not None:
            # Subsample vectors for clarity (similar to analyze_lax.py)
            skip_x = max(1, Q // 20)
            skip_y = max(1, Q // 20)
            skip = (slice(None, None, skip_x), slice(None, None, skip_y))
            ax_fd.quiver(tau[skip], phi[skip], fd_vx[skip], fd_vy[skip], 
                        color='k', headwidth=3.0, width=0.003, alpha=0.7)
        
        ax_fd.set_title(f"FD {which.title()}, t={t:.2f}")
        ax_fd.set_xlim(xmin, xmax)
        ax_fd.set_ylim(ymin, ymax)
        cbar_fd = plt.colorbar(pc_fd, ax=ax_fd, shrink=0.6)
        cbar_fd.ax.set_title(r"$\rho$" if which == "density" else r"$|v|$", fontsize=14)
        
        # Column 3: Epsilon Metric
        ax_diff = axes[i, 2]
        pc_diff = ax_diff.pcolormesh(tau, phi, epsilon_metric, shading='auto', cmap='coolwarm')
        ax_diff.set_title(f"ε (% of peak), t={t:.2f}")
        ax_diff.set_xlim(xmin, xmax)
        ax_diff.set_ylim(ymin, ymax)
        cbar_diff = plt.colorbar(pc_diff, ax=ax_diff, shrink=0.6)
        cbar_diff.ax.set_title("ε (% of peak)", fontsize=14)
        
        # Add x-axis labels only on bottom row
        if i == 4:
            ax_pinn.set_xlabel("x")
            ax_fd.set_xlabel("x")
            ax_diff.set_xlabel("x")
        
        # Add y-axis labels only on leftmost column
        ax_pinn.set_ylabel("y")
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{which}_comparison_5x3.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 5x3 comparison table to {save_path}")
    
    plt.show()
    return fig, axes

