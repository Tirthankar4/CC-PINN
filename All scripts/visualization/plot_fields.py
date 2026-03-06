from .plot_utils import *

def plot_function(net, time_array, initial_params, velocity=False, isplot=False, animation=False):
    """
    Plot function for 1D slices through 2D domain
    
    Args:
        net: Trained neural network
        time_array: Array of times to plot
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        velocity: Whether to plot velocity
        isplot: Whether to save plots
        animation: Whether this is for animation
    """
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    
    # Unwrap network from list if needed
    net = net[0] if isinstance(net, list) else net  
    # rho_o imported from config.py
    num_of_waves_x = (xmax-xmin)/lam
    num_of_waves_y = (ymax-ymin)/lam
    if animation:
        ## Converting the float (time-input) to an numpy array for animation
        ## Ignore this when the function is called in isolation
        time_array = np.array([time_array])
        # print("time",np.asarray(time_array))
    
    rho_max_Pinns = []    
    peak_lst=[]
    pert_xscale=[]
    for t in time_array:
        print("Plotting at t=", t)
        
        # Create 1D slice through the domain (like in notebook: Y = 0.6)
        X = np.linspace(xmin, xmax, 1000).reshape(1000, 1)
        Y = SLICE_Y * np.ones(1000).reshape(1000, 1)  # Fixed Y slice
        if DIMENSION >= 3:
            Z = SLICE_Z * np.ones(1000).reshape(1000, 1)
        t_ = t * np.ones(1000).reshape(1000, 1)
        
        pt_x_collocation = Variable(torch.from_numpy(X).float(), requires_grad=True).to(device)
        pt_y_collocation = Variable(torch.from_numpy(Y).float(), requires_grad=True).to(device) if DIMENSION >= 2 else None
        pt_z_collocation = Variable(torch.from_numpy(Z).float(), requires_grad=True).to(device) if DIMENSION >= 3 else None
        pt_t_collocation = Variable(torch.from_numpy(t_).float(), requires_grad=True).to(device)
        
        # Evaluate network
        net_device = next(net.parameters()).device
        pt_x_collocation = pt_x_collocation.to(net_device)
        pt_t_collocation = pt_t_collocation.to(net_device)
        if pt_y_collocation is not None:
            pt_y_collocation = pt_y_collocation.to(net_device)
        if pt_z_collocation is not None:
            pt_z_collocation = pt_z_collocation.to(net_device)
        inputs = _build_input_list(
            pt_x_collocation,
            pt_t_collocation,
            pt_y_collocation,
            pt_z_collocation
        )
        output_0 = net(inputs)
        
        rho_tensor, vx_tensor, vy_tensor, vz_tensor, phi_tensor = _split_outputs(output_0)
        rho_pred0 = rho_tensor.detach().cpu().numpy()
        v_pred_x0 = vx_tensor.detach().cpu().numpy()
        v_pred_y0 = vy_tensor.detach().cpu().numpy() if vy_tensor is not None else None
        phi_pred0 = phi_tensor.detach().cpu().numpy()
 
        rho_max_PN = np.max(rho_pred0)
        
        ## Theoretical Values
        #rho_theory = np.max(rho_o + rho_1*np.exp(alpha * t)*np.cos(2*np.pi*X[:, 0:1]/lam))
        #rho_theory0 = np.max(rho_o + rho_1*np.exp(alpha * 0)*np.cos(2*np.pi*X[:, 0:1]/lam)) ## at t =0 
        
        #diff=abs(rho_max_PN-rho_theory)/abs(rho_max_PN+rho_theory) * 2  ## since the den is rhomax+rhotheory

        
#         ### Difference between peaks for the PINNs solution
        
#         rho_pred0Flat=rho_pred0.reshape(-1)
#         peaks,_=scipy.signal.find_peaks(rho_pred0Flat)
#         peak_lst.append(peaks)
        
#         growth_pert=(rho_theory-rho_theory0)/rho_theory0*100 ## growth percentage
        
#         peak_diff=(rho_pred0Flat[peaks[1]]-rho_pred0Flat[peaks[0]])/(rho_pred0Flat[peaks[1]]+rho_pred0Flat[peaks[0]])

        #g_pred0=phi_x = dde.grad.jacobian(phi_pred0, X, i=0, j=0)
        if isplot:              
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Density plot
            plt.figure(1)
            plt.plot(X, rho_pred0, label="t={}".format(round(t,2)))
            plt.ylabel(r"$\rho$")
            plt.xlabel("x")
            plt.grid()
            plt.legend(numpoints=1, loc='upper right', fancybox=True, shadow=True)
            plt.title(r"PINNs Solution for $\lambda$ = {} $\lambda_J$".format(round(lam/(2*np.pi),2)))
            #plt.savefig(output_folder+'/PINNS_density'+str(lam)+'_'+str(int(num_of_waves_x))+'_'+str(tmax)+'.png', dpi=300)

            if velocity == True:
                # Velocity plots
                plt.figure(2)
                plt.plot(X, v_pred_x0, '--', label="t={}".format(round(t,2)))
                plt.ylabel("$v_x$")
                plt.xlabel("x")
                plt.title("PINNs Solution Velocity")
                plt.legend(numpoints=1, loc='upper right', fancybox=True, shadow=True)
                #plt.savefig(output_folder+'/PINNS_velocity'+str(lam)+'_'+str(int(num_of_waves_x))+'_'+str(tmax)+'.png', dpi=300)

            # Potential plot
            plt.figure(3)
            plt.plot(X, phi_pred0, '--', label="t={}".format(round(t,2)))
            plt.ylabel(r"$\phi$")
            plt.xlabel("x")
            plt.title("PINNs Solution Potential")
            plt.legend(numpoints=1, loc='upper right', fancybox=True, shadow=True)
            #plt.savefig(output_folder+'/phi'+str(lam)+'_'+str(int(num_of_waves_x))+'_'+str(tmax)+'.png', dpi=300)

        
        else:  
            if animation:
                return X, rho_pred0, v_pred_x0, v_pred_y0, phi_pred0, rho_max_PN
            else:
                return X, rho_pred0, rho_max_PN



def Two_D_surface_plots(net, time, initial_params, ax=None, which="density", fd_cache=None):
    """
    Create 2D surface plots with velocity vectors

    Args:
        net: Trained neural network
        time: Time to plot
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        ax: Optional axis to plot on
        which: "density" or "velocity"
        fd_cache: Optional FD cache for z-slice alignment in 3D power spectrum cases
    """
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    
    # Unwrap network from list if needed
    net = net[0] if isinstance(net, list) else net
    
    # Use N_GRID for consistency with FD solver resolution
    # Exclude right boundary for periodic domains to avoid double-counting
    Q = N_GRID
    xs = np.linspace(xmin, xmax, Q, endpoint=False)
    ys = np.linspace(ymin, ymax, Q, endpoint=False)
    tau, phi = np.meshgrid(xs, ys) 
    Xgrid = np.vstack([tau.flatten(), phi.flatten()]).T
    t_00 = time * np.ones(Q**2).reshape(Q**2, 1)
    
    # Convert to tensors
    pt_x_collocation = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
    pt_y_collocation = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device) if DIMENSION >= 2 else None
    if DIMENSION >= 3:
        z_val_use = SLICE_Z
        if fd_cache is not None and time in fd_cache:
            z_val_use = fd_cache[time].get('z_val', SLICE_Z)
        pt_z_collocation = Variable(torch.from_numpy(np.full((Q**2, 1), z_val_use)).float(), requires_grad=True).to(device)
    else:
        pt_z_collocation = None
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

    if ax is None:  # for single plot
        plt.figure(figsize=(5, 5))
        ax = plt.gca() 

    # Clean velocity fields and avoid zero-length arrows
    U_clean = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
    V_clean = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    Vmag = np.sqrt(U_clean**2 + V_clean**2)
    mask = Vmag > 1e-12

    if which == "density":
        pc = ax.pcolormesh(tau, phi, rho, shading='auto', cmap='YlOrBr', vmin=np.min(rho), vmax=np.max(rho))
        skip = (slice(None, None, 5), slice(None, None, 5))
        ax.quiver(
            tau[skip][mask[skip]], phi[skip][mask[skip]],
            U_clean[skip][mask[skip]], V_clean[skip][mask[skip]],
            color='k', headwidth=3.0, width=0.003,
            scale_units='xy', angles='xy', scale=1.0, minlength=0.0, pivot='mid'
        )
        ax.set_title("Density, t={}".format(round(time, 2)))
        cbar = plt.colorbar(pc, shrink=0.6, location='right')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.set_title(r"$\rho$", fontsize=14)
    else:  # velocity magnitude surface plot
        pc = ax.pcolormesh(tau, phi, Vmag, shading='auto', cmap='viridis', vmin=np.min(Vmag), vmax=np.max(Vmag))
        skip = (slice(None, None, 5), slice(None, None, 5))
        ax.quiver(
            tau[skip][mask[skip]], phi[skip][mask[skip]],
            U_clean[skip][mask[skip]], V_clean[skip][mask[skip]],
            color='k', headwidth=3.0, width=0.003,
            scale_units='xy', angles='xy', scale=1.0, minlength=0.0, pivot='mid'
        )
        ax.set_title("Velocity, t={}".format(round(time, 2)))
        cbar = plt.colorbar(pc, shrink=0.6, location='right')
        cbar.ax.set_title(r" $|v|$", fontsize=14)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    
    return pc



def Two_D_surface_plots_FD(time, initial_params, N=200, nu=None, ax=None, which="density",
                           use_velocity_ps=None, ps_index=None, vel_rms=None, random_seed=None,
                           fd_cache=None):
    """
    Create 2D surface plots with velocity vectors using the Finite Difference (LAX) solver

    Args:
        time: Time to plot
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        N: Grid resolution for LAX solver (Nx = Ny = N) - only used if fd_cache is None
        nu: Courant number for LAX solver - only used if fd_cache is None
        ax: Optional matplotlib axis to plot on
        which: "density" or "velocity"
        use_velocity_ps: Whether to use velocity power spectrum (defaults to config) - only used if fd_cache is None
        ps_index: Power spectrum index (defaults to POWER_EXPONENT) - only used if fd_cache is None
        vel_rms: Velocity RMS amplitude (defaults to a*cs) - only used if fd_cache is None
        random_seed: Random seed (defaults to 1234) - only used if fd_cache is None
        fd_cache: Optional dictionary mapping time -> FD data (if provided, solver is not called)

    Returns:
        The QuadMesh object from pcolormesh
    """
    xmin, xmax, ymin, ymax, rho_1, _alpha, lam, _output_folder, _tmax = initial_params
    if nu is None:
        nu = _get_fd_nu()

    # Use cached data if available
    if fd_cache is not None and time in fd_cache:
        cache_data = fd_cache[time]
        x_fd = cache_data['x']
        y_fd = cache_data['y']
        rho_fd = cache_data['rho']
        vx_fd = cache_data['vx']
        vy_fd = cache_data['vy']
        X, Y = np.meshgrid(x_fd, y_fd, indexing='ij')
        Nx = x_fd.shape[0]
        Ny = y_fd.shape[0] if len(y_fd.shape) > 0 else rho_fd.shape[1]
    else:
        # Use config defaults if not specified to ensure consistency with PINN training
        if use_velocity_ps is None:
            use_velocity_ps = (str(PERTURBATION_TYPE).lower() == "power_spectrum")
        if ps_index is None:
            ps_index = POWER_EXPONENT
        if vel_rms is None:
            vel_rms = a * cs
        if random_seed is None:
            random_seed = RANDOM_SEED

        # Domain properties for LAX (Lx = Ly and Nx = Ny in solver)
        num_of_waves = (xmax - xmin) / lam

        # Decide grid resolution policy: use N_GRID for power spectrum; FD_N_2D for sinusoidal when N is None
        if str(PERTURBATION_TYPE).lower() == "power_spectrum":
            N_use = N_GRID
        else:
            N_use = FD_N_2D if N is None else N

        # Run LAX solver (finite difference) with self-gravity enabled to obtain 2D fields
        # Prefer using the exact shared velocity fields (if available) to ensure identical realization
        # across PINN ICs and all FD visualizations.
        # Returns (gravity=True, comparison=False, animation=True):
        #   x (Nx,), rho (Nx,Ny), vx (Nx,Ny), vy (Nx,Ny), phi (Nx,Ny), n, rho_max
        if DIMENSION == 3:
            # Use unified solver with proper IC type support
            result = _call_unified_3d_solver(
                time=time, lam=lam, num_of_waves=num_of_waves, rho_1=rho_1, nu=nu,
                use_velocity_ps=use_velocity_ps, ps_index=ps_index,
                vel_rms=vel_rms, random_seed=random_seed
            )
            # Extract results from unified solver
            x_fd = result.coordinates['x']
            y_fd = result.coordinates['y']
            z_fd = result.coordinates['z']
            rho_vol = result.density
            vx_vol, vy_vol, vz_vol = result.velocity_components
            z_idx = np.argmin(np.abs(z_fd - SLICE_Z))
            rho_fd = rho_vol[:, :, z_idx]
            vx_fd = vx_vol[:, :, z_idx]
            vy_fd = vy_vol[:, :, z_idx]
            X, Y = np.meshgrid(x_fd, y_fd, indexing='ij')
        else:
            if (str(PERTURBATION_TYPE).lower() == "power_spectrum" \
                and _shared_vx_np is not None and _shared_vy_np is not None):
                # Use native resolution of shared fields to avoid resampling artifacts
                n_fd_use = int(_shared_vx_np.shape[0])
                x_fd, rho_fd, vx_fd, vy_fd, _phi_fd, _n, _rho_max = _timed_call(
                    "LAX 2D (shared-field cpu)",
                    lax_solution,
                    time, n_fd_use, nu, lam, num_of_waves, rho_1,
                    gravity=True, isplot=False, comparison=False, animation=True,
                    vx0_shared=_shared_vx_np, vy0_shared=_shared_vy_np
                )
            else:
                if torch.cuda.is_available():
                    _clear_cuda_cache()
                    x_fd, rho_fd, vx_fd, vy_fd, phi_fd_torch, _n, _rho_max = _timed_call(
                        "LAX 2D (torch)",
                        lax_solution_torch,
                        time_val=time, N=N_use, nu=nu, lam=lam, num_of_waves=num_of_waves, rho_1=rho_1,
                        gravity=True, use_velocity_ps=use_velocity_ps, ps_index=ps_index,
                        vel_rms=vel_rms, random_seed=random_seed
                    )
                    _phi_fd = phi_fd_torch if phi_fd_torch is not None else np.zeros_like(rho_fd)
                else:
                    x_fd, rho_fd, vx_fd, vy_fd, _phi_fd, _n, _rho_max = _timed_call(
                        "LAX 2D (cpu)",
                        lax_solution,
                        time, N_use, nu, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True,
                        use_velocity_ps=use_velocity_ps, ps_index=ps_index, vel_rms=vel_rms, random_seed=random_seed
                    )

            Lx = lam * num_of_waves
            Nx = x_fd.shape[0]
            Ny = rho_fd.shape[1]
            y_fd = np.linspace(0.0, Lx, Ny, endpoint=False)
            X, Y = np.meshgrid(x_fd, y_fd, indexing='ij')

    if ax is None:  # for single plot
        plt.figure(figsize=(5, 5))
        ax = plt.gca()

    # Clean FD velocity fields and avoid zero-length arrows
    vx_c = np.nan_to_num(vx_fd, nan=0.0, posinf=0.0, neginf=0.0)
    vy_c = np.nan_to_num(vy_fd, nan=0.0, posinf=0.0, neginf=0.0)
    Vmag_fd = np.sqrt(vx_c**2 + vy_c**2)
    mask = Vmag_fd > 1e-12

    if which == "density":
        pc = ax.pcolormesh(X, Y, rho_fd, shading='auto', cmap='YlOrBr', vmin=np.min(rho_fd), vmax=np.max(rho_fd))
        skip = (slice(None, None, max(1, Nx // 20)), slice(None, None, max(1, Ny // 20)))
        ax.quiver(
            X[skip][mask[skip]], Y[skip][mask[skip]],
            vx_c[skip][mask[skip]], vy_c[skip][mask[skip]],
            color='k', headwidth=3.0, width=0.003,
            scale_units='xy', angles='xy', scale=1.0, minlength=0.0, pivot='mid'
        )
        ax.set_title("FD Density, t={}".format(round(time, 2)))
        cbar = plt.colorbar(pc, shrink=0.6, location='right')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.set_title(r"$\rho$", fontsize=14)
    else:
        pc = ax.pcolormesh(X, Y, Vmag_fd, shading='auto', cmap='viridis', vmin=np.min(Vmag_fd), vmax=np.max(Vmag_fd))
        skip = (slice(None, None, max(1, Nx // 20)), slice(None, None, max(1, Ny // 20)))
        ax.quiver(
            X[skip][mask[skip]], Y[skip][mask[skip]],
            vx_c[skip][mask[skip]], vy_c[skip][mask[skip]],
            color='k', headwidth=3.0, width=0.003,
            scale_units='xy', angles='xy', scale=1.0, minlength=0.0, pivot='mid'
        )
        ax.set_title("FD Velocity, t={}".format(round(time, 2)))
        cbar = plt.colorbar(pc, shrink=0.6, location='right')
        cbar.ax.set_title(r" $|v|$", fontsize=14)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return pc



def create_2d_surface_plots(net, initial_params, time_points=None, which="density", fd_cache=None):
    """
    Create 2D surface plots at multiple time points

    Args:
        net: Trained neural network
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_points: List of time points to plot (default: [0.0, 0.5, 1.0, 1.5, 2.0])
        which: "density" or "velocity"
        fd_cache: Optional FD cache for z-slice alignment in 3D power spectrum cases
    """
    if time_points is None:
        time_points = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    print("Creating 2D surface plots...")
    
    # Create subplots for multiple time points
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, t in enumerate(time_points):
        if i < len(axes):
            print(f"Plotting at t = {t}")
            Two_D_surface_plots(net, t, initial_params, ax=axes[i], which=which, fd_cache=fd_cache)

    # Remove the last empty subplot if needed
    if len(time_points) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    
    # Save the figure to output folder
    output_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"2d_surface_plots_{which}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 2D surface plots ({which}) to {save_path}")
    
    return fig, axes



def create_2d_surface_plots_FD(initial_params, time_points=None, which="density", N=200, nu=None,
                               use_velocity_ps=None, ps_index=None, vel_rms=None, random_seed=None,
                               fd_cache=None):
    """
    Create 2D surface plots at multiple time points using the LAX FD solver

    Args:
        initial_params: (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_points: list of times
        which: "density" or "velocity"
        N: grid size (only used if fd_cache is None)
        nu: Courant number (only used if fd_cache is None)
        use_velocity_ps: Whether to use velocity power spectrum (defaults to config) - only used if fd_cache is None
        ps_index: Power spectrum index (defaults to POWER_EXPONENT) - only used if fd_cache is None
        vel_rms: Velocity RMS amplitude (defaults to a*cs) - only used if fd_cache is None
        random_seed: Random seed (defaults to 1234) - only used if fd_cache is None
        fd_cache: Optional dictionary mapping time -> FD data (if provided, solver is not called)
    """
    if time_points is None:
        time_points = [0.0, 0.5, 1.0, 1.5, 2.0]
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

    print("Creating 2D FD surface plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, t in enumerate(time_points):
        if i < len(axes):
            print(f"FD plotting at t = {t}")
            # Enforce grid policy: use N_GRID for power spectrum; else pass through N
            if str(PERTURBATION_TYPE).lower() == "power_spectrum":
                N_call = N_GRID
            else:
                N_call = N
            Two_D_surface_plots_FD(t, initial_params, N=N_call, nu=nu, ax=axes[i], which=which,
                                   use_velocity_ps=use_velocity_ps, ps_index=ps_index, vel_rms=vel_rms, random_seed=random_seed,
                                   fd_cache=fd_cache)

    if len(time_points) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    
    # Save the figure to output folder
    output_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"2d_surface_plots_FD_{which}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 2D FD surface plots ({which}) to {save_path}")
    
    return fig, axes



