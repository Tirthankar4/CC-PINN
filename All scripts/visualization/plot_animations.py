from .plot_utils import *
from .plot_fields import *
from .plot_comparisons import *

def create_2d_animation(net, initial_params, time_points=None, which="density", fps=2, save_path=None, fixed_colorbar=True, verbose=False):
    """
    Create an animated 2D surface plot showing evolution over time

    Args:
        net: Trained neural network
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_points: Array of time points for animation (default: 50 points from 0 to 2.0)
        which: "density" or "velocity"
        fps: Frames per second for animation
        save_path: Optional path to save the animation (e.g., 'animation.mp4')
    """
    if time_points is None:
        # Use the provided training tmax from initial_params to bound animation time
        _xmin, _xmax, _ymin, _ymax, _rho_1, _alpha, _lam, _output_folder, tmax = initial_params
        time_points = np.linspace(0.0, float(tmax), 80)
    
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    
    # Unwrap network from list if needed
    net = net[0] if isinstance(net, list) else net
    
    if verbose:
        print(f"Creating 2D animation with {len(time_points)} frames...")
    
    # Create output directory for saving plots: always use config.SNAPSHOT_DIR/GRINN
    output_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    
    # Get data for first frame to set up colorbar limits
    # Use N_GRID for consistency with FD solver resolution
    # Exclude right boundary for periodic domains to avoid double-counting
    Q = N_GRID
    xs = np.linspace(xmin, xmax, Q, endpoint=False)
    ys = np.linspace(ymin, ymax, Q, endpoint=False)
    tau, phi = np.meshgrid(xs, ys)
    if DIMENSION >= 3:
        zeta = np.full_like(tau, SLICE_Z)
    Xgrid = np.vstack([tau.flatten(), phi.flatten()]).T
    t_00 = time_points[0] * np.ones(Q**2).reshape(Q**2, 1)
    
    # Convert to tensors for first frame
    pt_x_collocation = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
    pt_y_collocation = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device) if DIMENSION >= 2 else None
    pt_z_collocation = Variable(torch.from_numpy(zeta.reshape(-1, 1)).float(), requires_grad=True).to(device) if DIMENSION >= 3 else None
    pt_t_collocation = Variable(torch.from_numpy(t_00).float(), requires_grad=True).to(device)
    
    # Get first frame data to set colorbar limits
    net_device = next(net.parameters()).device
    pt_x_collocation = pt_x_collocation.to(net_device)
    pt_t_collocation = pt_t_collocation.to(net_device)
    if pt_y_collocation is not None:
        pt_y_collocation = pt_y_collocation.to(net_device)
    if pt_z_collocation is not None:
        pt_z_collocation = pt_z_collocation.to(net_device)
    output_00 = net(_build_input_list(pt_x_collocation, pt_t_collocation, pt_y_collocation, pt_z_collocation))
    rho_first_tensor, vx_first_tensor, vy_first_tensor, _, _ = _split_outputs(output_00)
    rho_first = rho_first_tensor.detach().cpu().numpy().reshape(Q, Q)
    U_first = vx_first_tensor.detach().cpu().numpy().reshape(Q, Q)
    if vy_first_tensor is not None:
        V_first = vy_first_tensor.detach().cpu().numpy().reshape(Q, Q)
    else:
        V_first = np.zeros_like(U_first)
    
    # (removed temporary quick-check print of mean(U), mean(V))
    
    # Optionally precompute fixed color limits using first and last frames
    fixed_vmin = None
    fixed_vmax = None
    if which == "density" and fixed_colorbar:
        # Use N_GRID for consistency with FD solver resolution
        # Exclude right boundary for periodic domains to avoid double-counting
        Q = N_GRID
        xs = np.linspace(xmin, xmax, Q, endpoint=False)
        ys = np.linspace(ymin, ymax, Q, endpoint=False)
        tau, phi = np.meshgrid(xs, ys)
        Xgrid = np.vstack([tau.flatten(), phi.flatten()]).T
        # First frame
        t_first = time_points[0] * np.ones(Q**2).reshape(Q**2, 1)
        pt_x = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
        pt_y = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device) if DIMENSION >= 2 else None
        pt_z = Variable(torch.from_numpy(zeta.reshape(-1, 1)).float(), requires_grad=True).to(device) if DIMENSION >= 3 else None
        pt_t = Variable(torch.from_numpy(t_first).float(), requires_grad=True).to(device)
        # Ensure inputs are on the same device as the model
        net_device = next(net.parameters()).device
        pt_x = pt_x.to(net_device)
        pt_t = pt_t.to(net_device)
        if pt_y is not None:
            pt_y = pt_y.to(net_device)
        if pt_z is not None:
            pt_z = pt_z.to(net_device)
        rho_first = net(_build_input_list(pt_x, pt_t, pt_y, pt_z))[:, 0].data.cpu().numpy().reshape(Q, Q)
        # Last frame
        t_last = time_points[-1] * np.ones(Q**2).reshape(Q**2, 1)
        pt_t_last = Variable(torch.from_numpy(t_last).float(), requires_grad=True).to(device)
        pt_t_last = pt_t_last.to(net_device)
        rho_last = net(_build_input_list(pt_x, pt_t_last, pt_y, pt_z))[:, 0].data.cpu().numpy().reshape(Q, Q)
        fixed_vmin = min(np.min(rho_first), np.min(rho_last))
        fixed_vmax = max(np.max(rho_first), np.max(rho_last))
        if fixed_vmin == fixed_vmax:
            eps = 1e-6 if fixed_vmin == 0 else 1e-6 * abs(fixed_vmin)
            fixed_vmin, fixed_vmax = fixed_vmin - eps, fixed_vmax + eps

    # Set up initial plot
    if which == "density":
        # Handle flat initial frame by expanding color limits slightly
        if fixed_colorbar and fixed_vmin is not None:
            vmin_use, vmax_use = fixed_vmin, fixed_vmax
        else:
            rmin, rmax = np.min(rho_first), np.max(rho_first)
            if not np.isfinite(rmin) or not np.isfinite(rmax):
                rmin, rmax = 0.0, 1.0
            if rmin == rmax:
                eps = 1e-6 if rmin == 0 else 1e-6 * abs(rmin)
                rmin, rmax = rmin - eps, rmax + eps
            vmin_use, vmax_use = rmin, rmax
        pc = ax.pcolormesh(tau, phi, rho_first, shading='auto', cmap='YlOrBr', vmin=vmin_use, vmax=vmax_use)
        pert_str = "Sinusoidal" if str(PERTURBATION_TYPE).lower() == "sinusoidal" else "Power Spectrum"
        ax.set_title(f"{pert_str} Density, t={time_points[0]:.2f}")
        cbar = plt.colorbar(pc, shrink=0.6, location='right')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.set_title(r"$\rho$", fontsize=14)
    else:  # velocity magnitude surface plot
        Vmag_first = np.sqrt(U_first**2 + V_first**2)
        pc = ax.pcolormesh(tau, phi, Vmag_first, shading='auto', cmap='viridis')
        pert_str = "Sinusoidal" if str(PERTURBATION_TYPE).lower() == "sinusoidal" else "Power Spectrum"
        ax.set_title(f"{pert_str} Velocity, t={time_points[0]:.2f}")
        cbar = plt.colorbar(pc, shrink=0.6, location='right')
        cbar.ax.set_title(r" $|v|$", fontsize=14)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Do not save the initial frame by default; animation frames and optional snapshots cover needs
    
    def animate(frame):
        t = time_points[frame]
        if verbose:
            print(f"Animating frame {frame+1}/{len(time_points)} at t={t:.2f}")
        
        # Get data for current time
        t_00 = t * np.ones(Q**2).reshape(Q**2, 1)
        
        # Convert to tensors
        pt_x_collocation = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
        pt_y_collocation = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device) if DIMENSION >= 2 else None
        pt_z_collocation = Variable(torch.from_numpy(zeta.reshape(-1, 1)).float(), requires_grad=True).to(device) if DIMENSION >= 3 else None
        pt_t_collocation = Variable(torch.from_numpy(t_00).float(), requires_grad=True).to(device)
        
        # Evaluate network
        net_device = next(net.parameters()).device
        pt_x_collocation = pt_x_collocation.to(net_device)
        pt_t_collocation = pt_t_collocation.to(net_device)
        if pt_y_collocation is not None:
            pt_y_collocation = pt_y_collocation.to(net_device)
        if pt_z_collocation is not None:
            pt_z_collocation = pt_z_collocation.to(net_device)
        output_00 = net(_build_input_list(pt_x_collocation, pt_t_collocation, pt_y_collocation, pt_z_collocation))
        
        rho_tensor, vx_tensor, vy_tensor, _, _ = _split_outputs(output_00)
        rho = rho_tensor.detach().cpu().numpy().reshape(Q, Q)
        U = vx_tensor.detach().cpu().numpy().reshape(Q, Q)
        if vy_tensor is not None:
            V = vy_tensor.detach().cpu().numpy().reshape(Q, Q)
        else:
            V = np.zeros_like(U)
        
        # Update plot data
        if which == "density":
            pc.set_array(rho.ravel())
            # Update color limits only if not fixed
            if not fixed_colorbar:
                rmin, rmax = np.min(rho), np.max(rho)
                if rmin == rmax:
                    eps = 1e-6 if rmin == 0 else 1e-6 * abs(rmin)
                    rmin, rmax = rmin - eps, rmax + eps
                pc.set_clim(vmin=rmin, vmax=rmax)
            pert_str = "Sinusoidal" if str(PERTURBATION_TYPE).lower() == "sinusoidal" else "Power Spectrum"
            ax.set_title(f"{pert_str} Density, t={t:.2f}")
        else:  # velocity magnitude surface plot
            Vmag = np.sqrt(U**2 + V**2)
            pc.set_array(Vmag.ravel())
            pert_str = "Sinusoidal" if str(PERTURBATION_TYPE).lower() == "sinusoidal" else "Power Spectrum"
            ax.set_title(f"{pert_str} Velocity, t={t:.2f}")
        
        # Save every 10th frame for static snapshots (disabled to avoid extra plots)
        # if frame % 10 == 0:
        #     save_path_frame = os.path.join(output_dir, f"{which}_t_{t:.2f}.png")
        #     plt.savefig(save_path_frame, dpi=300, bbox_inches='tight')
        #     if verbose:
        #         print(f"Saved frame to {save_path_frame}")
        
        return pc
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(time_points), 
                                 interval=1000/fps, blit=False, repeat=True)
    
    # Save animation with appropriate writer/extension
    try:
        if animation.writers.is_available('ffmpeg'):
            animation_path = os.path.join(output_dir, f"{which}_animation.mp4")
            if verbose:
                print(f"Saving animation to {animation_path} with ffmpeg...")
            anim.save(animation_path, writer='ffmpeg', fps=fps)
            saved_format = 'mp4'
        else:
            animation_path = os.path.join(output_dir, f"{which}_animation.gif")
            if verbose:
                print(f"ffmpeg not available. Saving animation to {animation_path} with Pillow...")
            anim.save(animation_path, writer='pillow', fps=fps)
            saved_format = 'gif'
        if verbose:
            print("Animation saved successfully!")
    except Exception as e:
        print(f"Animation save failed: {e}")
        raise
    
    # Optional static snapshots uniformly over [0, tmax]
    # Static snapshots disabled to prevent extra plots
    # if SAVE_STATIC_SNAPSHOTS:
    #     snapshot_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    #     os.makedirs(snapshot_dir, exist_ok=True)
    #     # Determine tmax from initial_params
    #     _xmin, _xmax, _ymin, _ymax, _rho_1, _alpha, _lam, _output_folder, tmax_val = initial_params
    #     times_static = np.linspace(0.0, float(tmax_val), 5)
    #     if verbose:
    #         print(f"Saving {len(times_static)} static snapshots to {snapshot_dir} over [0, {tmax_val}]...")
    #     for t in times_static:
    #         t_00 = t * np.ones(Q**2).reshape(Q**2, 1)
    #         pt_x_collocation = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
    #         pt_y_collocation = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device)
    #         pt_t_collocation = Variable(torch.from_numpy(t_00).float(), requires_grad=True).to(device)
    #         output_00 = net([pt_x_collocation, pt_y_collocation, pt_t_collocation])
    #         rho = output_00[:, 0].data.cpu().numpy().reshape(Q, Q)
    #         U = output_00[:, 1].data.cpu().numpy().reshape(Q, Q)
    #         V = output_00[:, 2].data.cpu().numpy().reshape(Q, Q)
    #
    #         fig_static, ax_static = plt.subplots(figsize=(8, 8))
    #         if which == "density":
    #             pc_static = ax_static.pcolormesh(tau, phi, rho, shading='auto', cmap='YlOrBr')
    #             cbar_static = plt.colorbar(pc_static, shrink=0.6, location='right')
    #             cbar_static.formatter.set_powerlimits((0, 0))
    #             cbar_static.ax.set_title(r"$\rho$", fontsize=14)
    #         else:
    #             Vmag = np.sqrt(U**2 + V**2)
    #             pc_static = ax_static.pcolormesh(tau, phi, Vmag, shading='auto', cmap='viridis')
    #             cbar_static = plt.colorbar(pc_static, shrink=0.6, location='right')
    #             cbar_static.ax.set_title(r" $|v|$", fontsize=14)
    #         ax_static.set_xlim(xmin, xmax)
    #         ax_static.set_ylim(ymin, ymax)
    #         ax_static.set_xlabel("x")
    #         ax_static.set_ylabel("y")
    #         plt.tight_layout()
    #         static_save_path = os.path.join(snapshot_dir, f"{which}_static_t_{t:.2f}.png")
    #         plt.savefig(static_save_path, dpi=300, bbox_inches='tight')
    #         plt.close(fig_static)
    #         if verbose:
    #             print(f"Saved static snapshot to {static_save_path}")
    
    # Generate 5x3 comparison tables for both density and velocity (only once per animation call)
    if verbose:
        print("Generating 5x3 comparison tables...")
    
    # Only generate comparison tables if this is the density animation call
    # This prevents duplicate generation when both density and velocity animations are created
    if which == "density":
        # Compute cache for 5x3 comparison table time points (5 points uniformly over [0, tmax])
        comparison_time_points = np.linspace(0.0, float(tmax), 5)
        print(f"Computing FD cache for 5x3 comparison table ({len(comparison_time_points)} time points)...")
        fd_cache_5x3 = compute_fd_data_cache(
            initial_params, comparison_time_points,
            N=N_GRID, nu=_get_fd_nu()
        )
        
        print("Generating density comparison table...")
        # Create comparison table for density - use cached data
        create_5x3_comparison_table(net, initial_params, which="density", N=N_GRID, nu=_get_fd_nu(), fd_cache=fd_cache_5x3)
        
        print("Generating velocity comparison table...")
        # Create comparison table for velocity - use cached data
        create_5x3_comparison_table(net, initial_params, which="velocity", N=N_GRID, nu=_get_fd_nu(), fd_cache=fd_cache_5x3)
    
    # Display animation inline if in a notebook
    try:
        from IPython.display import HTML, display
        import base64
        with open(animation_path, 'rb') as f:
            data = f.read()
        data_base64 = base64.b64encode(data).decode('utf-8')
        if saved_format == 'mp4':
            video_html = f'''
    <div style="text-align: center;">
        <h3>{which.title()} Evolution Animation</h3>
        <video width="600" height="600" controls autoplay loop>
            <source src="data:video/mp4;base64,{data_base64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    '''
            display(HTML(video_html))
        else:
            img_html = f'''
    <div style="text-align: center;">
        <h3>{which.title()} Evolution Animation</h3>
        <img src="data:image/gif;base64,{data_base64}" width="600" height="600" />
    </div>
    '''
            display(HTML(img_html))
    except Exception as _:
        pass
    
    # Avoid displaying the figure in non-notebook runs
    plt.close(fig)
    
    return anim



