import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainingDiagnostics:
    """
    Comprehensive diagnostics for PINN performance analysis.
    
    Focus: Why does PINN fail at long times?
    Generates 6 diagnostic plots:
      - Training diagnostics: Loss convergence and balance
      - PDE residual evolution: When/where/which equations fail
      - Conservation violations: Mass and momentum drift
      - Temporal error accumulation: How errors compound over time
      - Spectral bias analysis: Frequency damping and mode evolution
      - Solution stability: Detecting unphysical behavior
    
    All diagnostics run automatically when ENABLE_TRAINING_DIAGNOSTICS = True
    """

    def __init__(self, save_dir='./diagnostics/', dimension=2, perturbation_type='power_spectrum'):
        self.save_dir = save_dir
        self.dimension = dimension
        self.perturbation_type = str(perturbation_type).lower()
        
        os.makedirs(save_dir, exist_ok=True)

        # Storage for tracking metrics over iterations
        self.history = {
            'iteration': [],
            'total_loss': [],
            'pde_loss': [],
            'ic_loss': [],
            'mean_rho': [],
            'max_rho': [],
        }

    def _prepare_inputs(self, geomtime_col):
        """Normalize collocation inputs to the model's expected format."""
        if isinstance(geomtime_col, (list, tuple)):
            return geomtime_col
        # Single tensor [N, D] -> list of [N,1]
        return [geomtime_col[:, i:i+1] for i in range(geomtime_col.shape[1])]

    def log_iteration(self, iteration, model, loss_dict, geomtime_col):
        """
        Call this every N iterations during training.
        
        Args:
            iteration: Current iteration number
            model: PINN model
            loss_dict: Dictionary with 'total', 'PDE', 'IC' losses
            geomtime_col: Collocation points
        """
        self.history['iteration'].append(iteration)
        self.history['total_loss'].append(float(loss_dict.get('total', np.nan)))
        self.history['pde_loss'].append(float(loss_dict.get('PDE', loss_dict.get('pde', np.nan))))
        self.history['ic_loss'].append(float(loss_dict.get('IC', loss_dict.get('ic', np.nan))))

        # Compute diagnostics from current model state
        with torch.no_grad():
            inputs = self._prepare_inputs(geomtime_col)
            pred = model(inputs)
            # Use first channel as density/log-density proxy
            rho = pred[:, 0].detach().cpu().numpy()

            # Density statistics
            self.history['mean_rho'].append(np.nanmean(rho))
            self.history['max_rho'].append(np.nanmax(rho))

    def plot_diagnostics(self, iteration=None):
        """
        Generate streamlined training diagnostic plot.
        Combined loss components and balance in one figure.
        """
        if len(self.history['iteration']) == 0:
            print("No diagnostic data to plot.")
            return
        
        iters = self.history['iteration']
        suffix = f"_iter_{iteration}" if iteration is not None else "_final"

        # Combined training diagnostics plot
        fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 3, figure=fig)
        
        # Subplot 1: Loss evolution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.semilogy(iters, self.history['total_loss'], 'b-', linewidth=2, label='Total')
        ax1.semilogy(iters, self.history['pde_loss'], 'r-', linewidth=2, label='PDE')
        ax1.semilogy(iters, self.history['ic_loss'], 'g-', linewidth=2, label='IC')
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Loss (log scale)', fontsize=11)
        ax1.set_title('Loss Components', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Loss balance ratio
        ax2 = fig.add_subplot(gs[0, 1])
        ic_arr = np.array(self.history['ic_loss'])
        pde_arr = np.array(self.history['pde_loss'])
        ratio = pde_arr / (ic_arr + 1e-10)
        ax2.plot(iters, ratio, 'purple', linewidth=2)
        ax2.axhline(1.0, color='k', linestyle='--', linewidth=1.5, label='Balanced')
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('PDE Loss / IC Loss', fontsize=11)
        ax2.set_title('Loss Balance', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Density evolution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(iters, self.history['mean_rho'], 'b-', linewidth=2, label='Mean ρ')
        ax3.plot(iters, self.history['max_rho'], 'r-', linewidth=2, label='Max ρ')
        ax3.set_xlabel('Iteration', fontsize=11)
        ax3.set_ylabel('Density', fontsize=11)
        ax3.set_title('Density Statistics', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_diagnostics{suffix}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training diagnostic plot saved to {self.save_dir}")

    def _get_domain_bounds(self, model, dimension):
        """Extract actual domain bounds from model, with fallback to [0,1]."""
        xmin = model.xmin if hasattr(model, 'xmin') and model.xmin is not None else 0.0
        xmax = model.xmax if hasattr(model, 'xmax') and model.xmax is not None else 1.0
        ymin = model.ymin if hasattr(model, 'ymin') and model.ymin is not None else 0.0
        ymax = model.ymax if hasattr(model, 'ymax') and model.ymax is not None else 1.0
        zmin = model.zmin if hasattr(model, 'zmin') and model.zmin is not None else 0.0
        zmax = model.zmax if hasattr(model, 'zmax') and model.zmax is not None else 1.0
        
        if dimension == 1:
            return xmin, xmax, None, None, None, None
        elif dimension == 2:
            return xmin, xmax, ymin, ymax, None, None
        else:  # dimension == 3
            return xmin, xmax, ymin, ymax, zmin, zmax

    # ==================== POST-TRAINING DIAGNOSTICS FOR HIGH-TMAX FAILURES ====================
    
    def compute_residual_heatmap(self, model, dimension, tmax, n_spatial=None, n_temporal=60):
        """
        Compute PDE residuals on a regular grid over space-time using CHUNKING to avoid OOM.
        """
        from core.losses import pde_residue
        from config import PERTURBATION_TYPE, N_GRID, N_GRID_3D
        
        device = next(model.parameters()).device
        
        # Get actual domain bounds from the model
        xmin, xmax, ymin, ymax, zmin, zmax = self._get_domain_bounds(model, dimension)
        
        # Determine grid size
        if n_spatial is None:
            if str(PERTURBATION_TYPE).lower() == "power_spectrum":
                n_spatial = min(150, N_GRID // 3) if dimension == 2 else min(100, N_GRID_3D // 3)
            else:
                n_spatial = 80
        
        print(f"  Computing residuals on {n_spatial}x{n_spatial}x{n_temporal} grid (Chunked)...")
        
        if dimension == 2:
            x = torch.linspace(xmin, xmax, n_spatial, device=device)
            y = torch.linspace(ymin, ymax, n_spatial, device=device)
            t = torch.linspace(0, tmax, n_temporal, device=device)
            
            X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
            
            # Flatten all points: (N_total, 1)
            X_flat = X.reshape(-1, 1)
            Y_flat = Y.reshape(-1, 1)
            T_flat = T.reshape(-1, 1)
            
            total_points = X_flat.shape[0]
            
            # Initialize result arrays (on CPU to save GPU memory)
            rho_res_all = np.zeros(total_points, dtype=np.float32)
            vx_res_all = np.zeros(total_points, dtype=np.float32)
            vy_res_all = np.zeros(total_points, dtype=np.float32)
            phi_res_all = np.zeros(total_points, dtype=np.float32)
            
            # Process in chunks
            chunk_size = 40000
            model.eval()
            
            for i in range(0, total_points, chunk_size):
                end_idx = min(i + chunk_size, total_points)
                
                x_chunk = X_flat[i:end_idx].detach().clone().requires_grad_(True)
                y_chunk = Y_flat[i:end_idx].detach().clone().requires_grad_(True)
                t_chunk = T_flat[i:end_idx].detach().clone().requires_grad_(True)
                
                colloc_chunk = [x_chunk, y_chunk, t_chunk]
                
                rho_r, vx_r, vy_r, phi_r = pde_residue(colloc_chunk, model, dimension=2)
                
                rho_res_all[i:end_idx] = rho_r.detach().cpu().numpy().flatten()
                vx_res_all[i:end_idx] = vx_r.detach().cpu().numpy().flatten()
                vy_res_all[i:end_idx] = vy_r.detach().cpu().numpy().flatten()
                phi_res_all[i:end_idx] = phi_r.detach().cpu().numpy().flatten()
                
                del x_chunk, y_chunk, t_chunk, colloc_chunk, rho_r, vx_r, vy_r, phi_r
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            
            return {
                'x': x.cpu().numpy(),
                'y': y.cpu().numpy(),
                't': t.cpu().numpy(),
                'rho_residual': rho_res_all.reshape(n_spatial, n_spatial, n_temporal),
                'vx_residual': vx_res_all.reshape(n_spatial, n_spatial, n_temporal),
                'vy_residual': vy_res_all.reshape(n_spatial, n_spatial, n_temporal),
                'phi_residual': phi_res_all.reshape(n_spatial, n_spatial, n_temporal)
            }
        
        elif dimension == 3:
            # For 3D, use coarser grid to save memory
            n_spatial_3d = max(32, n_spatial // 2)
            x = torch.linspace(xmin, xmax, n_spatial_3d, device=device)
            y = torch.linspace(ymin, ymax, n_spatial_3d, device=device)
            t = torch.linspace(0, tmax, n_temporal, device=device)
            
            z_slice_val = 0.5
            Z_slice = torch.full((n_spatial_3d, n_spatial_3d, n_temporal), z_slice_val, device=device)
            X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
            
            X_flat = X.reshape(-1, 1)
            Y_flat = Y.reshape(-1, 1)
            Z_flat = Z_slice.reshape(-1, 1)
            T_flat = T.reshape(-1, 1)
            
            total_points = X_flat.shape[0]
            
            rho_res_all = np.zeros(total_points, dtype=np.float32)
            vx_res_all = np.zeros(total_points, dtype=np.float32)
            vy_res_all = np.zeros(total_points, dtype=np.float32)
            vz_res_all = np.zeros(total_points, dtype=np.float32)
            phi_res_all = np.zeros(total_points, dtype=np.float32)
            
            chunk_size = 20000
            model.eval()
            
            for i in range(0, total_points, chunk_size):
                end_idx = min(i + chunk_size, total_points)
                
                x_chunk = X_flat[i:end_idx].detach().clone().requires_grad_(True)
                y_chunk = Y_flat[i:end_idx].detach().clone().requires_grad_(True)
                z_chunk = Z_flat[i:end_idx].detach().clone().requires_grad_(True)
                t_chunk = T_flat[i:end_idx].detach().clone().requires_grad_(True)
                
                colloc_chunk = [x_chunk, y_chunk, z_chunk, t_chunk]
                
                rho_r, vx_r, vy_r, vz_r, phi_r = pde_residue(colloc_chunk, model, dimension=3)
                
                rho_res_all[i:end_idx] = rho_r.detach().cpu().numpy().flatten()
                vx_res_all[i:end_idx] = vx_r.detach().cpu().numpy().flatten()
                vy_res_all[i:end_idx] = vy_r.detach().cpu().numpy().flatten()
                vz_res_all[i:end_idx] = vz_r.detach().cpu().numpy().flatten()
                phi_res_all[i:end_idx] = phi_r.detach().cpu().numpy().flatten()
                
                del x_chunk, y_chunk, z_chunk, t_chunk, colloc_chunk, rho_r, vx_r, vy_r, vz_r, phi_r
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            
            return {
                'x': x.cpu().numpy(),
                'y': y.cpu().numpy(),
                't': t.cpu().numpy(),
                'rho_residual': rho_res_all.reshape(n_spatial_3d, n_spatial_3d, n_temporal),
                'vx_residual': vx_res_all.reshape(n_spatial_3d, n_spatial_3d, n_temporal),
                'vy_residual': vy_res_all.reshape(n_spatial_3d, n_spatial_3d, n_temporal),
                'vz_residual': vz_res_all.reshape(n_spatial_3d, n_spatial_3d, n_temporal),
                'phi_residual': phi_res_all.reshape(n_spatial_3d, n_spatial_3d, n_temporal)
            }
    
    def plot_residual_heatmaps(self, residual_data, slice_idx=None):
        """
        Plot 2D heatmaps of PDE residuals over (x, t) at fixed y.
        Shows WHEN and WHERE each equation fails.
        """
        if slice_idx is None:
            slice_idx = residual_data['rho_residual'].shape[1] // 2
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        residuals = [
            ('Continuity Equation', residual_data['rho_residual']),
            ('Momentum X Equation', residual_data['vx_residual']),
            ('Momentum Y Equation', residual_data['vy_residual']),
            ('Poisson Equation', residual_data['phi_residual'])
        ]
        
        for idx, (name, resid) in enumerate(residuals):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            
            # Slice at fixed y
            resid_slice = resid[:, slice_idx, :].T  # (t, x) for imshow
            
            # Use log scale for better visualization
            resid_log = np.log10(np.abs(resid_slice) + 1e-16)
            actual_min = np.min(resid_log)
            data_max = np.max(resid_log)
            data_min = max(data_max - 4, -12)
            
            im = ax.imshow(resid_log, aspect='auto', origin='lower', cmap='hot',
                          extent=[residual_data['x'][0], residual_data['x'][-1],
                                 residual_data['t'][0], residual_data['t'][-1]],
                          vmin=data_min, vmax=data_max)
            ax.set_xlabel('x', fontsize=11)
            ax.set_ylabel('Time', fontsize=11)
            ax.set_title(
                f'{name} Residual\nMin: 10^{actual_min:.1f}, Max: 10^{data_max:.1f}',
                fontsize=12,
                fontweight='bold'
            )
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('log₁₀|residual|', fontsize=10)
        
        plt.suptitle(f'PDE Residuals Over Space-Time (y={residual_data["y"][slice_idx]:.2f})', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(f'{self.save_dir}/residual_heatmaps.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def check_conservation_laws(self, model, dimension, tmax, n_times=30, n_grid=None):
        """
        Check if mass and momentum are conserved over time.
        Critical for physical consistency - should be flat lines!
        
        Note: Uses higher resolution for power spectrum to ensure accurate integration.
        """
        from config import PERTURBATION_TYPE, N_GRID, N_GRID_3D
        
        device = next(model.parameters()).device
        
        # Get actual domain bounds from the model
        xmin, xmax, ymin, ymax, zmin, zmax = self._get_domain_bounds(model, dimension)
        
        # Use higher resolution for power spectrum
        if n_grid is None:
            if str(PERTURBATION_TYPE).lower() == "power_spectrum":
                n_grid = min(150, N_GRID // 3) if dimension == 2 else min(100, N_GRID_3D // 3)
            else:
                n_grid = 100
        
        print(f"  Checking conservation laws at {n_times} time points on {n_grid}x{n_grid} grid...")
        print(f"  Domain: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}]" + (f", z=[{zmin}, {zmax}]" if dimension == 3 else ""))
        
        conservation_data = {
            'times': [],
            'total_mass': [],
            'total_momentum_x': [],
            'total_momentum_y': [],
            'total_momentum_z': []  # For 3D cases
        }
        
        time_points = np.linspace(0, tmax, n_times)
        
        for t_val in time_points:
            if dimension == 2:
                x = torch.linspace(xmin, xmax, n_grid, device=device)
                y = torch.linspace(ymin, ymax, n_grid, device=device)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                T = torch.full_like(X, t_val)
                
                colloc = [X.reshape(-1, 1), Y.reshape(-1, 1), T.reshape(-1, 1)]
                
                with torch.no_grad():
                    pred = model(colloc)
                    rho = pred[:, 0].reshape(n_grid, n_grid).cpu().numpy()
                    vx = pred[:, 1].reshape(n_grid, n_grid).cpu().numpy()
                    vy = pred[:, 2].reshape(n_grid, n_grid).cpu().numpy()
                
                # Integrate using trapezoidal rule with correct spacing
                dx = (xmax - xmin) / (n_grid - 1)
                dy = (ymax - ymin) / (n_grid - 1)
                total_mass = np.trapezoid(np.trapezoid(rho, dx=dy, axis=1), dx=dx, axis=0)
                total_px = np.trapezoid(np.trapezoid(rho * vx, dx=dy, axis=1), dx=dx, axis=0)
                total_py = np.trapezoid(np.trapezoid(rho * vy, dx=dy, axis=1), dx=dx, axis=0)
                
                conservation_data['times'].append(t_val)
                conservation_data['total_mass'].append(total_mass)
                conservation_data['total_momentum_x'].append(total_px)
                conservation_data['total_momentum_y'].append(total_py)
            
            elif dimension == 3:
                # For 3D, use coarser grid
                n_grid_3d = max(40, n_grid // 2)
                x = torch.linspace(xmin, xmax, n_grid_3d, device=device)
                y = torch.linspace(ymin, ymax, n_grid_3d, device=device)
                z = torch.linspace(zmin, zmax, n_grid_3d, device=device)
                X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
                T = torch.full_like(X, t_val)
                
                colloc = [X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1), T.reshape(-1, 1)]
                
                with torch.no_grad():
                    pred = model(colloc)
                    rho = pred[:, 0].reshape(n_grid_3d, n_grid_3d, n_grid_3d).cpu().numpy()
                    vx = pred[:, 1].reshape(n_grid_3d, n_grid_3d, n_grid_3d).cpu().numpy()
                    vy = pred[:, 2].reshape(n_grid_3d, n_grid_3d, n_grid_3d).cpu().numpy()
                    vz = pred[:, 3].reshape(n_grid_3d, n_grid_3d, n_grid_3d).cpu().numpy()
                
                dx = (xmax - xmin) / (n_grid_3d - 1)
                dy = (ymax - ymin) / (n_grid_3d - 1)
                dz = (zmax - zmin) / (n_grid_3d - 1)
                total_mass = np.trapezoid(np.trapezoid(np.trapezoid(rho, dx=dz, axis=2), dx=dy, axis=1), dx=dx, axis=0)
                total_px = np.trapezoid(np.trapezoid(np.trapezoid(rho * vx, dx=dz, axis=2), dx=dy, axis=1), dx=dx, axis=0)
                total_py = np.trapezoid(np.trapezoid(np.trapezoid(rho * vy, dx=dz, axis=2), dx=dy, axis=1), dx=dx, axis=0)
                total_pz = np.trapezoid(np.trapezoid(np.trapezoid(rho * vz, dx=dz, axis=2), dx=dy, axis=1), dx=dx, axis=0)
                
                conservation_data['times'].append(t_val)
                conservation_data['total_mass'].append(total_mass)
                conservation_data['total_momentum_x'].append(total_px)
                conservation_data['total_momentum_y'].append(total_py)
                conservation_data['total_momentum_z'].append(total_pz)
        
        return conservation_data
    
    def plot_conservation_laws(self, conservation_data):
        """Plot conserved quantities over time - should be flat lines!"""
        fig = plt.figure(figsize=(16, 5))
        gs = GridSpec(1, 3, figure=fig)
        from config import cs
        
        times = conservation_data['times']
        
        # Mass conservation
        ax1 = fig.add_subplot(gs[0, 0])
        mass = np.array(conservation_data['total_mass'])
        mass_initial = mass[0]
        mass_drift = ((mass - mass_initial) / mass_initial) * 100  # Percent drift
        momentum_scale = mass_initial * cs
        ax1.plot(times, mass_drift, 'b-', linewidth=2.5)
        ax1.axhline(0, color='r', linestyle='--', linewidth=1.5, label='Perfect conservation')
        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Mass drift (%)', fontsize=11)
        ax1.set_title('Mass Conservation', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Momentum X conservation
        ax2 = fig.add_subplot(gs[0, 1])
        px = np.array(conservation_data['total_momentum_x'])
        px_initial = px[0]
        px_drift = ((px - px_initial) / momentum_scale) * 100
        ylabel = 'Momentum X drift (% of M₀cₛ)'
        title_suffix = ''
        
        ax2.plot(times, px_drift, 'g-', linewidth=2.5)
        ax2.axhline(0, color='r', linestyle='--', linewidth=1.5, label='Perfect conservation')
        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_ylabel(ylabel, fontsize=11)
        ax2.set_title('Momentum X Conservation' + title_suffix, fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Momentum Y conservation
        ax3 = fig.add_subplot(gs[0, 2])
        py = np.array(conservation_data['total_momentum_y'])
        py_initial = py[0]
        py_drift = ((py - py_initial) / momentum_scale) * 100
        ylabel = 'Momentum Y drift (% of M₀cₛ)'
        title_suffix = ''
        
        ax3.plot(times, py_drift, 'orange', linewidth=2.5)
        ax3.axhline(0, color='r', linestyle='--', linewidth=1.5, label='Perfect conservation')
        ax3.set_xlabel('Time', fontsize=11)
        ax3.set_ylabel(ylabel, fontsize=11)
        ax3.set_title('Momentum Y Conservation' + title_suffix, fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/conservation_laws.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def compute_spectral_content(self, model, dimension, tmax, n_times=5, n_grid=128):
        """
        Compute FFT of density field at different times.
        Reveals spectral bias and high-frequency mode damping.
        """
        device = next(model.parameters()).device
        
        # Get actual domain bounds
        xmin, xmax, ymin, ymax, zmin, zmax = self._get_domain_bounds(model, dimension)
        
        print(f"  Computing spectral content at {n_times} time points...")
        
        time_points = np.linspace(0, tmax, n_times)
        spectra = []
        
        for t_val in time_points:
            if dimension == 2:
                x = torch.linspace(xmin, xmax, n_grid, device=device)
                y = torch.linspace(ymin, ymax, n_grid, device=device)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                T = torch.full_like(X, t_val)
                
                colloc = [X.reshape(-1, 1), Y.reshape(-1, 1), T.reshape(-1, 1)]
                
                with torch.no_grad():
                    pred = model(colloc)
                    rho = pred[:, 0].reshape(n_grid, n_grid).cpu().numpy()
                
                # 2D FFT
                fft = np.fft.fft2(rho)
                power = np.abs(np.fft.fftshift(fft))**2
                
                # Create wavenumber grid with correct spacing
                Lx = xmax - xmin
                Ly = ymax - ymin
                dx = Lx / n_grid
                dy = Ly / n_grid
                kx = 2 * np.pi * np.fft.fftfreq(n_grid, d=dx)
                ky = 2 * np.pi * np.fft.fftfreq(n_grid, d=dy)
                kx_shift = np.fft.fftshift(kx)
                ky_shift = np.fft.fftshift(ky)
                KX, KY = np.meshgrid(kx_shift, ky_shift, indexing='ij')
                K = np.sqrt(KX**2 + KY**2)
                
                spectra.append({'time': t_val, 'power': power, 'k': K, 'rho': rho})
            
            elif dimension == 3:
                # For 3D, take a 2D slice at z=0.5 (middle of domain)
                n_grid_3d = max(64, n_grid // 2)
                x = torch.linspace(xmin, xmax, n_grid_3d, device=device)
                y = torch.linspace(ymin, ymax, n_grid_3d, device=device)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                Z = torch.full_like(X, (zmin + zmax) / 2.0)
                T = torch.full_like(X, t_val)
                
                colloc = [X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1), T.reshape(-1, 1)]
                
                with torch.no_grad():
                    pred = model(colloc)
                    rho = pred[:, 0].reshape(n_grid_3d, n_grid_3d).cpu().numpy()
                
                fft = np.fft.fft2(rho)
                power = np.abs(np.fft.fftshift(fft))**2
                
                # Use physical wavenumbers with correct spacing (consistent with 2D case)
                Lx = xmax - xmin
                Ly = ymax - ymin
                dx = Lx / n_grid_3d
                dy = Ly / n_grid_3d
                kx = 2 * np.pi * np.fft.fftfreq(n_grid_3d, d=dx)
                ky = 2 * np.pi * np.fft.fftfreq(n_grid_3d, d=dy)
                kx_shift = np.fft.fftshift(kx)
                ky_shift = np.fft.fftshift(ky)
                KX, KY = np.meshgrid(kx_shift, ky_shift, indexing='ij')
                K = np.sqrt(KX**2 + KY**2)
                
                spectra.append({'time': t_val, 'power': power, 'k': K, 'rho': rho})
        
        return spectra
    
    def plot_spectral_evolution(self, spectra_data):
        """
        Plot power spectrum evolution over time.
        Shows if high-frequency modes are being damped (spectral bias).
        """
        fig = plt.figure(figsize=(16, 5))
        gs = GridSpec(1, 2, figure=fig)
        
        # Left: Radially averaged power spectra
        ax1 = fig.add_subplot(gs[0, 0])
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(spectra_data)))
        
        for idx, spec in enumerate(spectra_data):
            # Radial binning
            k_max = np.max(spec['k'])
            k_bins = np.linspace(0, k_max, 25)
            power_avg = []
            k_centers = []
            
            for i in range(len(k_bins)-1):
                mask = (spec['k'] >= k_bins[i]) & (spec['k'] < k_bins[i+1])
                if mask.any():
                    power_avg.append(np.mean(spec['power'][mask]))
                    k_centers.append((k_bins[i] + k_bins[i+1]) / 2)
            
            if len(k_centers) > 0:
                ax1.loglog(k_centers, power_avg, '-o', color=colors[idx], 
                          linewidth=2, markersize=4, label=f"t={spec['time']:.2f}")
        
        ax1.set_xlabel('Wavenumber k', fontsize=11)
        ax1.set_ylabel('Power', fontsize=11)
        ax1.set_title('Power Spectrum Evolution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3, which='both')
        
        # Right: 2D power spectrum at final time
        ax2 = fig.add_subplot(gs[0, 1])
        final_spec = spectra_data[-1]
        power_log = np.log10(final_spec['power'] + 1e-12)
        
        im = ax2.imshow(power_log, origin='lower', cmap='hot', aspect='auto',
                       extent=[-np.max(final_spec['k']), np.max(final_spec['k']),
                              -np.max(final_spec['k']), np.max(final_spec['k'])])
        ax2.set_xlabel('k_x', fontsize=11)
        ax2.set_ylabel('k_y', fontsize=11)
        ax2.set_title(f'2D Power Spectrum at t={final_spec["time"]:.2f}', 
                     fontsize=12, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('log₁₀(Power)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/spectral_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def compute_temporal_statistics(self, model, dimension, tmax, n_times=50, n_spatial=100):
        """
        Compute field statistics over time to track error accumulation.
        """
        device = next(model.parameters()).device
        
        # Get actual domain bounds
        xmin, xmax, ymin, ymax, zmin, zmax = self._get_domain_bounds(model, dimension)
        
        print(f"  Computing temporal statistics at {n_times} time points...")
        
        time_points = np.linspace(0, tmax, n_times)
        stats = {
            'times': [],
            'rho_mean': [],
            'rho_std': [],
            'rho_max': [],
            'grad_rho_mean': [],
            'grad_rho_max': []
        }
        
        for t_val in time_points:
            if dimension == 2:
                x = torch.linspace(xmin, xmax, n_spatial, device=device)
                y = torch.linspace(ymin, ymax, n_spatial, device=device)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                T = torch.full_like(X, t_val)
                
                X_flat = X.reshape(-1, 1)
                Y_flat = Y.reshape(-1, 1)
                T_flat = T.reshape(-1, 1)
                
                X_flat.requires_grad_(True)
                Y_flat.requires_grad_(True)
                
                colloc = [X_flat, Y_flat, T_flat]
                
                pred = model(colloc)
                rho = pred[:, 0:1]
                
                # Compute gradients (retain_graph=True for first call since we need multiple gradients)
                grad_rho_x = torch.autograd.grad(rho.sum(), X_flat, create_graph=False, retain_graph=True)[0]
                grad_rho_y = torch.autograd.grad(rho.sum(), Y_flat, create_graph=False)[0]
                grad_mag = torch.sqrt(grad_rho_x**2 + grad_rho_y**2)
                
                rho_np = rho.detach().cpu().numpy().flatten()
                grad_np = grad_mag.detach().cpu().numpy().flatten()
                
                stats['times'].append(t_val)
                stats['rho_mean'].append(np.mean(rho_np))
                stats['rho_std'].append(np.std(rho_np))
                stats['rho_max'].append(np.max(rho_np))
                stats['grad_rho_mean'].append(np.mean(grad_np))
                stats['grad_rho_max'].append(np.max(grad_np))
            
            elif dimension == 3:
                # For 3D, take a 2D slice at z=middle for computational efficiency
                n_spatial_3d = max(50, n_spatial // 2)
                x = torch.linspace(xmin, xmax, n_spatial_3d, device=device)
                y = torch.linspace(ymin, ymax, n_spatial_3d, device=device)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                Z = torch.full_like(X, (zmin + zmax) / 2.0)
                T = torch.full_like(X, t_val)
                
                X_flat = X.reshape(-1, 1)
                Y_flat = Y.reshape(-1, 1)
                Z_flat = Z.reshape(-1, 1)
                T_flat = T.reshape(-1, 1)
                
                X_flat.requires_grad_(True)
                Y_flat.requires_grad_(True)
                Z_flat.requires_grad_(True)
                
                colloc = [X_flat, Y_flat, Z_flat, T_flat]
                
                pred = model(colloc)
                rho = pred[:, 0:1]
                
                # Compute gradients
                grad_rho_x = torch.autograd.grad(rho.sum(), X_flat, create_graph=False, retain_graph=True)[0]
                grad_rho_y = torch.autograd.grad(rho.sum(), Y_flat, create_graph=False, retain_graph=True)[0]
                grad_rho_z = torch.autograd.grad(rho.sum(), Z_flat, create_graph=False)[0]
                grad_mag = torch.sqrt(grad_rho_x**2 + grad_rho_y**2 + grad_rho_z**2)
                
                rho_np = rho.detach().cpu().numpy().flatten()
                grad_np = grad_mag.detach().cpu().numpy().flatten()
                
                stats['times'].append(t_val)
                stats['rho_mean'].append(np.mean(rho_np))
                stats['rho_std'].append(np.std(rho_np))
                stats['rho_max'].append(np.max(rho_np))
                stats['grad_rho_mean'].append(np.mean(grad_np))
                stats['grad_rho_max'].append(np.max(grad_np))
        
        return stats
    
    def plot_temporal_statistics(self, stats_data):
        """
        Plot field statistics over time.
        Shows if solution becomes unphysical or gradients explode.
        """
        fig = plt.figure(figsize=(16, 5))
        gs = GridSpec(1, 3, figure=fig)
        
        times = stats_data['times']
        
        # Density statistics
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(times, stats_data['rho_mean'], 'b-', linewidth=2.5, label='Mean')
        ax1.fill_between(times,
                         np.array(stats_data['rho_mean']) - np.array(stats_data['rho_std']),
                         np.array(stats_data['rho_mean']) + np.array(stats_data['rho_std']),
                         alpha=0.3, color='b', label='±1 std')
        ax1.plot(times, stats_data['rho_max'], 'r-', linewidth=2, label='Max')
        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Density Evolution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Gradient statistics
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.semilogy(times, stats_data['grad_rho_mean'], 'b-', linewidth=2.5, label='Mean')
        ax2.semilogy(times, stats_data['grad_rho_max'], 'r-', linewidth=2, label='Max')
        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_ylabel('|∇ρ| (log scale)', fontsize=11)
        ax2.set_title('Gradient Magnitude Evolution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')
        
        # Growth rate analysis
        ax3 = fig.add_subplot(gs[0, 2])
        rho_max = np.array(stats_data['rho_max'])
        if len(rho_max) > 5:
            # Fit exponential growth
            log_rho = np.log(rho_max + 1e-10)
            valid_idx = np.isfinite(log_rho)
            if valid_idx.sum() > 5:
                times_valid = np.array(times)[valid_idx]
                log_rho_valid = log_rho[valid_idx]
                coeffs = np.polyfit(times_valid, log_rho_valid, 1)
                growth_rate = coeffs[0]
                fit = np.exp(np.poly1d(coeffs)(times))
                
                ax3.semilogy(times, rho_max, 'b-', linewidth=2.5, label='Max ρ')
                ax3.semilogy(times, fit, 'r--', linewidth=2, 
                           label=f'Exp fit (γ={growth_rate:.4f})')
                ax3.set_xlabel('Time', fontsize=11)
                ax3.set_ylabel('Max Density (log scale)', fontsize=11)
                ax3.set_title('Density Growth Rate', fontsize=12, fontweight='bold')
                ax3.legend(fontsize=10)
                ax3.grid(True, alpha=0.3, which='both')
        
        # Add parameterization info for proper interpretation
        from config import PERTURBATION_TYPE, USE_PARAMETERIZATION
        param_info = f"[{PERTURBATION_TYPE}, ρ-param: {USE_PARAMETERIZATION}]"
        fig.text(0.99, 0.01, param_info, ha='right', va='bottom', fontsize=8, 
                 fontstyle='italic', color='gray', transform=fig.transFigure)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/temporal_statistics.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def compute_temporal_error_accumulation(self, model, dimension, tmax, n_times=30):
        """
        Track how prediction errors grow and accumulate over time.
        Helps identify when the solution starts diverging from physical behavior.
        
        Now computes ACTUAL PDE residuals and gradient norms including phi.
        """
        from core.losses import pde_residue
        
        device = next(model.parameters()).device
        time_points = np.linspace(0, tmax, n_times)
        
        error_data = {
            'times': time_points,
            'pde_residuals': {'continuity': [], 'momentum_x': [], 'momentum_y': [], 'poisson': []},
            'field_ranges': {'rho': [], 'vx': [], 'vy': []},
            'gradient_norms': {'rho': [], 'vx': [], 'vy': [], 'phi': []},
            'max_residuals': []
        }
        
        if dimension == 3:
            error_data['pde_residuals']['momentum_z'] = []
            error_data['field_ranges']['vz'] = []
            error_data['gradient_norms']['vz'] = []
        
        n_spatial = 80 if dimension == 2 else 60
        
        # Get actual domain bounds
        xmin, xmax, ymin, ymax, zmin, zmax = self._get_domain_bounds(model, dimension)
        
        for t_val in time_points:
            if dimension == 2:
                x = torch.linspace(xmin, xmax, n_spatial, device=device)
                y = torch.linspace(ymin, ymax, n_spatial, device=device)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                T = torch.full_like(X, t_val)
                
                X_flat = X.reshape(-1, 1).requires_grad_(True)
                Y_flat = Y.reshape(-1, 1).requires_grad_(True)
                T_flat = T.reshape(-1, 1).requires_grad_(True)
                
                colloc = [X_flat, Y_flat, T_flat]
            else:  # dimension == 3
                x = torch.linspace(xmin, xmax, n_spatial, device=device)
                y = torch.linspace(ymin, ymax, n_spatial, device=device)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                Z = torch.full_like(X, (zmin + zmax) / 2.0)
                T = torch.full_like(X, t_val)
                
                X_flat = X.reshape(-1, 1).requires_grad_(True)
                Y_flat = Y.reshape(-1, 1).requires_grad_(True)
                Z_flat = Z.reshape(-1, 1).requires_grad_(True)
                T_flat = T.reshape(-1, 1).requires_grad_(True)
                
                colloc = [X_flat, Y_flat, Z_flat, T_flat]
            
            # Compute ACTUAL PDE residuals using the proper residue function
            if dimension == 2:
                rho_r, vx_r, vy_r, phi_r = pde_residue(colloc, model, dimension=2)
                
                error_data['pde_residuals']['continuity'].append(torch.abs(rho_r).mean().item())
                error_data['pde_residuals']['momentum_x'].append(torch.abs(vx_r).mean().item())
                error_data['pde_residuals']['momentum_y'].append(torch.abs(vy_r).mean().item())
                error_data['pde_residuals']['poisson'].append(torch.abs(phi_r).mean().item())
                
            else:  # 3D
                rho_r, vx_r, vy_r, vz_r, phi_r = pde_residue(colloc, model, dimension=3)
                
                error_data['pde_residuals']['continuity'].append(torch.abs(rho_r).mean().item())
                error_data['pde_residuals']['momentum_x'].append(torch.abs(vx_r).mean().item())
                error_data['pde_residuals']['momentum_y'].append(torch.abs(vy_r).mean().item())
                error_data['pde_residuals']['momentum_z'].append(torch.abs(vz_r).mean().item())
                error_data['pde_residuals']['poisson'].append(torch.abs(phi_r).mean().item())
            
            # Get predictions for field statistics
            pred = model(colloc)
            rho = pred[:, 0:1]
            vx = pred[:, 1:2]
            vy = pred[:, 2:3]
            phi = pred[:, -1:]  # Last column is phi
            
            # Track field statistics
            error_data['field_ranges']['rho'].append(torch.std(rho).item())
            error_data['field_ranges']['vx'].append(torch.std(vx).item())
            error_data['field_ranges']['vy'].append(torch.std(vy).item())
            if dimension == 3:
                vz = pred[:, 3:4]
                error_data['field_ranges']['vz'].append(torch.std(vz).item())
            
            # Track gradient norms for ALL fields including phi (full gradient magnitude)
            with torch.enable_grad():
                # Compute full gradient magnitudes: |∇f| = sqrt((∂f/∂x)² + (∂f/∂y)² + ...)
                rho_grad_x = torch.autograd.grad(rho.sum(), X_flat, create_graph=False, retain_graph=True)[0]
                rho_grad_y = torch.autograd.grad(rho.sum(), Y_flat, create_graph=False, retain_graph=True)[0]
                vx_grad_x = torch.autograd.grad(vx.sum(), X_flat, create_graph=False, retain_graph=True)[0]
                vx_grad_y = torch.autograd.grad(vx.sum(), Y_flat, create_graph=False, retain_graph=True)[0]
                vy_grad_x = torch.autograd.grad(vy.sum(), X_flat, create_graph=False, retain_graph=True)[0]
                vy_grad_y = torch.autograd.grad(vy.sum(), Y_flat, create_graph=False, retain_graph=True)[0]
                phi_grad_x = torch.autograd.grad(phi.sum(), X_flat, create_graph=False, retain_graph=True)[0]
                phi_grad_y = torch.autograd.grad(phi.sum(), Y_flat, create_graph=False, retain_graph=True)[0]
                
                # Compute full gradient magnitudes
                rho_grad_mag = torch.sqrt(rho_grad_x**2 + rho_grad_y**2)
                vx_grad_mag = torch.sqrt(vx_grad_x**2 + vx_grad_y**2)
                vy_grad_mag = torch.sqrt(vy_grad_x**2 + vy_grad_y**2)
                phi_grad_mag = torch.sqrt(phi_grad_x**2 + phi_grad_y**2)
                
                if dimension == 3:
                    rho_grad_z = torch.autograd.grad(rho.sum(), Z_flat, create_graph=False, retain_graph=True)[0]
                    vx_grad_z = torch.autograd.grad(vx.sum(), Z_flat, create_graph=False, retain_graph=True)[0]
                    vy_grad_z = torch.autograd.grad(vy.sum(), Z_flat, create_graph=False, retain_graph=True)[0]
                    vz_grad_x = torch.autograd.grad(vz.sum(), X_flat, create_graph=False, retain_graph=True)[0]
                    vz_grad_y = torch.autograd.grad(vz.sum(), Y_flat, create_graph=False, retain_graph=True)[0]
                    vz_grad_z = torch.autograd.grad(vz.sum(), Z_flat, create_graph=False, retain_graph=True)[0]
                    phi_grad_z = torch.autograd.grad(phi.sum(), Z_flat, create_graph=False, retain_graph=True)[0]
                    
                    rho_grad_mag = torch.sqrt(rho_grad_x**2 + rho_grad_y**2 + rho_grad_z**2)
                    vx_grad_mag = torch.sqrt(vx_grad_x**2 + vx_grad_y**2 + vx_grad_z**2)
                    vy_grad_mag = torch.sqrt(vy_grad_x**2 + vy_grad_y**2 + vy_grad_z**2)
                    phi_grad_mag = torch.sqrt(phi_grad_x**2 + phi_grad_y**2 + phi_grad_z**2)
                    vz_grad_mag = torch.sqrt(vz_grad_x**2 + vz_grad_y**2 + vz_grad_z**2)
                    error_data['gradient_norms']['vz'].append(torch.norm(vz_grad_mag).item())
                
                error_data['gradient_norms']['rho'].append(torch.norm(rho_grad_mag).item())
                error_data['gradient_norms']['vx'].append(torch.norm(vx_grad_mag).item())
                error_data['gradient_norms']['vy'].append(torch.norm(vy_grad_mag).item())
                error_data['gradient_norms']['phi'].append(torch.norm(phi_grad_mag).item())
            
            # Max residual magnitude (now including Poisson!)
            max_res = max(
                error_data['pde_residuals']['continuity'][-1],
                error_data['pde_residuals']['momentum_x'][-1],
                error_data['pde_residuals']['momentum_y'][-1],
                error_data['pde_residuals']['poisson'][-1]
            )
            if dimension == 3:
                max_res = max(max_res, error_data['pde_residuals']['momentum_z'][-1])
            
            error_data['max_residuals'].append(max_res)
        
        return error_data
    
    def plot_temporal_error_accumulation(self, error_data):
        """
        Visualize how errors accumulate and grow over time.
        Shows which equations fail first and error growth rates.
        """
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        times = error_data['times']
        
        # Plot 1: PDE Residual Evolution (NOW WITH POISSON!)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.semilogy(times, error_data['pde_residuals']['continuity'], 'b-', linewidth=2, label='Continuity', marker='o', markersize=4)
        ax1.semilogy(times, error_data['pde_residuals']['momentum_x'], 'r-', linewidth=2, label='Momentum X', marker='s', markersize=4)
        ax1.semilogy(times, error_data['pde_residuals']['momentum_y'], 'g-', linewidth=2, label='Momentum Y', marker='^', markersize=4)
        # IMPORTANT: Now showing actual Poisson residuals!
        ax1.semilogy(times, error_data['pde_residuals']['poisson'], 'orange', linewidth=2.5, label='Poisson', marker='*', markersize=6)
        if 'momentum_z' in error_data['pde_residuals']:
            ax1.semilogy(times, error_data['pde_residuals']['momentum_z'], 'm-', linewidth=2, label='Momentum Z', marker='d', markersize=4)
        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Residual Magnitude (log)', fontsize=11)
        ax1.set_title('PDE Residual Evolution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3, which='both')
        
        # Plot 2: Field Range Evolution (variance growth)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(times, error_data['field_ranges']['rho'], 'b-', linewidth=2.5, label='ρ std', marker='o', markersize=4)
        ax2.plot(times, error_data['field_ranges']['vx'], 'r--', linewidth=2, label='vx std', marker='s', markersize=4)
        ax2.plot(times, error_data['field_ranges']['vy'], 'g--', linewidth=2, label='vy std', marker='^', markersize=4)
        if 'vz' in error_data['field_ranges']:
            ax2.plot(times, error_data['field_ranges']['vz'], 'm--', linewidth=2, label='vz std', marker='d', markersize=4)
        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_ylabel('Field Standard Deviation', fontsize=11)
        ax2.set_title('Field Variability Evolution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Maximum Residual Growth
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.semilogy(times, error_data['max_residuals'], 'k-', linewidth=2.5, marker='o', markersize=5)
        if len(times) > 5:
            # Fit exponential growth
            log_res = np.log(np.array(error_data['max_residuals']) + 1e-12)
            valid_idx = np.isfinite(log_res)
            if valid_idx.sum() > 3:
                coeffs = np.polyfit(np.array(times)[valid_idx], log_res[valid_idx], 1)
                growth_rate = coeffs[0]
                fit = np.exp(np.poly1d(coeffs)(times))
                ax3.semilogy(times, fit, 'r--', linewidth=2, label=f'Exp fit (γ={growth_rate:.4f})')
                ax3.legend(fontsize=10)
        ax3.set_xlabel('Time', fontsize=11)
        ax3.set_ylabel('Max Residual (log)', fontsize=11)
        ax3.set_title('Error Amplification Rate', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, which='both')
        
        # Plot 4: Gradient Norm Evolution (NOW WITH ALL FIELDS INCLUDING PHI!)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.semilogy(times, np.array(error_data['gradient_norms']['rho']) + 1e-12, 'b-', linewidth=2.5, label='∇ρ', marker='o', markersize=4)
        ax4.semilogy(times, np.array(error_data['gradient_norms']['phi']) + 1e-12, 'orange', linewidth=2.5, label='∇φ', marker='*', markersize=6)
        ax4.semilogy(times, np.array(error_data['gradient_norms']['vx']) + 1e-12, 'r--', linewidth=1.5, label='∇vx', marker='s', markersize=3, alpha=0.7)
        ax4.semilogy(times, np.array(error_data['gradient_norms']['vy']) + 1e-12, 'g--', linewidth=1.5, label='∇vy', marker='^', markersize=3, alpha=0.7)
        if 'vz' in error_data['gradient_norms']:
            ax4.semilogy(times, np.array(error_data['gradient_norms']['vz']) + 1e-12, 'm--', linewidth=1.5, label='∇vz', marker='d', markersize=3, alpha=0.7)
        ax4.set_xlabel('Time', fontsize=11)
        ax4.set_ylabel('Gradient Norm (log)', fontsize=11)
        ax4.set_title('Gradient Magnitude Evolution', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9, loc='best')
        ax4.grid(True, alpha=0.3, which='both')
        
        # Plot 5: Cumulative Error Metric
        ax5 = fig.add_subplot(gs[1, 1])
        cumulative_error = np.cumsum(error_data['max_residuals'])
        ax5.plot(times, cumulative_error, 'purple', linewidth=2.5, marker='o', markersize=5)
        ax5.set_xlabel('Time', fontsize=11)
        ax5.set_ylabel('Cumulative Error', fontsize=11)
        ax5.set_title('Accumulated Error Budget', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Error Growth Rate
        ax6 = fig.add_subplot(gs[1, 2])
        if len(times) > 1:
            error_rate = np.diff(error_data['max_residuals']) / np.diff(times)
            ax6.plot(times[:-1], error_rate, 'orange', linewidth=2.5, marker='o', markersize=5)
            ax6.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax6.set_xlabel('Time', fontsize=11)
            ax6.set_ylabel('Error Growth Rate', fontsize=11)
            ax6.set_title('Instantaneous Error Acceleration', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Temporal Error Accumulation Analysis', fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(f'{self.save_dir}/temporal_error_accumulation.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_spectral_bias_analysis(self, spectra_data):
        """
        Enhanced spectral analysis focused on detecting spectral bias.
        Shows frequency damping and mode evolution over time.
        """
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: Radially averaged power spectra at different times
        ax1 = fig.add_subplot(gs[0, :2])
        colors = plt.cm.viridis(np.linspace(0, 1, len(spectra_data)))
        
        for idx, spec in enumerate(spectra_data):
            # Radial binning
            k_max = np.max(spec['k'])
            k_bins = np.linspace(0, k_max, 25)
            power_avg = []
            k_centers = []
            
            for i in range(len(k_bins)-1):
                mask = (spec['k'] >= k_bins[i]) & (spec['k'] < k_bins[i+1])
                if mask.sum() > 0:
                    power_avg.append(np.mean(spec['power'][mask]))
                    k_centers.append((k_bins[i] + k_bins[i+1]) / 2)
            
            if len(k_centers) > 0:
                ax1.loglog(k_centers, power_avg, color=colors[idx], linewidth=2,
                          label=f't={spec["time"]:.2f}', marker='o', markersize=3)
        
        ax1.set_xlabel('Wavenumber k', fontsize=11)
        ax1.set_ylabel('Power P(k)', fontsize=11)
        ax1.set_title('Power Spectrum Evolution (Spectral Bias Check)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Plot 2: Spectral Damping Rate
        ax2 = fig.add_subplot(gs[0, 2])
        if len(spectra_data) >= 2:
            # Compare first and last spectra
            spec_0 = spectra_data[0]
            spec_f = spectra_data[-1]
            
            # Compute power ratio (damping)
            k_max = min(np.max(spec_0['k']), np.max(spec_f['k']))
            k_bins = np.linspace(0, k_max, 20)
            damping_ratio = []
            k_centers = []
            
            for i in range(len(k_bins)-1):
                mask_0 = (spec_0['k'] >= k_bins[i]) & (spec_0['k'] < k_bins[i+1])
                mask_f = (spec_f['k'] >= k_bins[i]) & (spec_f['k'] < k_bins[i+1])
                if mask_0.sum() > 0 and mask_f.sum() > 0:
                    p0 = np.mean(spec_0['power'][mask_0])
                    pf = np.mean(spec_f['power'][mask_f])
                    if p0 > 0:
                        damping_ratio.append(pf / p0)
                        k_centers.append((k_bins[i] + k_bins[i+1]) / 2)
            
            if len(k_centers) > 0:
                ax2.semilogx(k_centers, damping_ratio, 'b-', linewidth=2.5, marker='o', markersize=5)
                ax2.axhline(1.0, color='r', linestyle='--', linewidth=2, label='No damping')
                ax2.set_xlabel('Wavenumber k', fontsize=11)
                ax2.set_ylabel('Power Ratio (final/initial)', fontsize=11)
                ax2.set_title('Spectral Damping by Frequency', fontsize=12, fontweight='bold')
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
        
        # Plot 3: High-k Mode Tracking
        ax3 = fig.add_subplot(gs[1, 0])
        times = [spec['time'] for spec in spectra_data]
        
        # Track power in different k ranges
        low_k_power = []
        mid_k_power = []
        high_k_power = []
        
        for spec in spectra_data:
            k_flat = spec['k'].flatten()
            p_flat = spec['power'].flatten()
            
            k_max = np.max(k_flat)
            low_k_power.append(np.mean(p_flat[k_flat < k_max/3]))
            mid_k_power.append(np.mean(p_flat[(k_flat >= k_max/3) & (k_flat < 2*k_max/3)]))
            high_k_power.append(np.mean(p_flat[k_flat >= 2*k_max/3]))
        
        ax3.semilogy(times, low_k_power, 'b-', linewidth=2.5, label='Low-k (large scales)', marker='o', markersize=5)
        ax3.semilogy(times, mid_k_power, 'g-', linewidth=2.5, label='Mid-k', marker='s', markersize=5)
        ax3.semilogy(times, high_k_power, 'r-', linewidth=2.5, label='High-k (small scales)', marker='^', markersize=5)
        ax3.set_xlabel('Time', fontsize=11)
        ax3.set_ylabel('Average Power (log)', fontsize=11)
        ax3.set_title('Scale-Dependent Power Evolution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, which='both')
        
        # Plot 4: Spectral Index Evolution
        ax4 = fig.add_subplot(gs[1, 1])
        spectral_indices = []
        for spec in spectra_data:
            k_flat = spec['k'].flatten()
            p_flat = spec['power'].flatten()
            
            # Fit power law in log space
            valid = (k_flat > 0) & (p_flat > 0)
            if valid.sum() > 10:
                log_k = np.log(k_flat[valid])
                log_p = np.log(p_flat[valid])
                coeffs = np.polyfit(log_k, log_p, 1)
                spectral_indices.append(coeffs[0])  # Slope = spectral index
            else:
                spectral_indices.append(np.nan)
        
        ax4.plot(times, spectral_indices, 'purple', linewidth=2.5, marker='o', markersize=6)
        ax4.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Time', fontsize=11)
        ax4.set_ylabel('Spectral Index (slope)', fontsize=11)
        ax4.set_title('Power Law Evolution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Density field snapshots
        ax5 = fig.add_subplot(gs[1, 2])
        # Show first, middle, last density fields
        n_show = min(3, len(spectra_data))
        indices = [0, len(spectra_data)//2, -1] if n_show == 3 else [0, -1]
        
        for i, idx in enumerate(indices[:n_show]):
            rho = spectra_data[idx]['rho']
            # Compute structure function or variance
            rho_var = np.var(rho)
            ax5.bar(i, rho_var, color=colors[idx], edgecolor='black', linewidth=1.5)
            ax5.text(i, rho_var, f't={spectra_data[idx]["time"]:.2f}', 
                    ha='center', va='bottom', fontsize=9)
        
        ax5.set_xlabel('Time Snapshot', fontsize=11)
        ax5.set_ylabel('Density Variance', fontsize=11)
        ax5.set_title('Structure Growth', fontsize=12, fontweight='bold')
        ax5.set_xticks(range(n_show))
        ax5.set_xticklabels(['Early', 'Mid', 'Late'][:n_show])
        ax5.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Spectral Bias and Mode Damping Analysis', fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(f'{self.save_dir}/spectral_bias_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def compute_solution_stability(self, model, dimension, tmax, stats_data, n_check=20):
        """
        Check for signs of solution instability or unphysical behavior.
        """
        device = next(model.parameters()).device
        time_points = np.linspace(0, tmax, n_check)
        
        stability_data = {
            'times': time_points,
            'density_bounds': {'min': [], 'max': [], 'mean': []},
            'velocity_norms': [],
            'negative_density_fraction': [],
            'extreme_values': [],
            'solution_smoothness': []
        }
        
        n_spatial = 80 if dimension == 2 else 60
        
        # Get actual domain bounds
        xmin, xmax, ymin, ymax, zmin, zmax = self._get_domain_bounds(model, dimension)
        
        for t_val in time_points:
            if dimension == 2:
                x = torch.linspace(xmin, xmax, n_spatial, device=device)
                y = torch.linspace(ymin, ymax, n_spatial, device=device)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                T = torch.full_like(X, t_val)
                colloc = [X.reshape(-1, 1), Y.reshape(-1, 1), T.reshape(-1, 1)]
            else:
                x = torch.linspace(xmin, xmax, n_spatial, device=device)
                y = torch.linspace(ymin, ymax, n_spatial, device=device)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                Z = torch.full_like(X, (zmin + zmax) / 2.0)
                T = torch.full_like(X, t_val)
                colloc = [X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1), T.reshape(-1, 1)]
            
            with torch.no_grad():
                pred = model(colloc)
                rho = pred[:, 0].cpu().numpy()
                vx = pred[:, 1].cpu().numpy()
                vy = pred[:, 2].cpu().numpy()
                
                # Density bounds
                stability_data['density_bounds']['min'].append(np.min(rho))
                stability_data['density_bounds']['max'].append(np.max(rho))
                stability_data['density_bounds']['mean'].append(np.mean(rho))
                
                # Velocity norms
                v_norm = np.sqrt(vx**2 + vy**2)
                if dimension == 3:
                    vz = pred[:, 3].cpu().numpy()
                    v_norm = np.sqrt(vx**2 + vy**2 + vz**2)
                stability_data['velocity_norms'].append(np.max(v_norm))
                
                # Negative density fraction
                neg_frac = np.sum(rho < 0) / len(rho)
                stability_data['negative_density_fraction'].append(neg_frac)
                
                # Extreme value detection (more than 10x mean)
                extreme_frac = np.sum(np.abs(rho - np.mean(rho)) > 10 * np.std(rho)) / len(rho)
                stability_data['extreme_values'].append(extreme_frac)
                
                # Solution smoothness (Laplacian magnitude as proxy)
                if dimension == 2:
                    rho_2d = rho.reshape(n_spatial, n_spatial)
                    laplacian = np.abs(np.gradient(np.gradient(rho_2d, axis=0), axis=0) + 
                                     np.gradient(np.gradient(rho_2d, axis=1), axis=1))
                    stability_data['solution_smoothness'].append(np.mean(laplacian))
                else:
                    stability_data['solution_smoothness'].append(0.0)  # Placeholder for 3D
        
        return stability_data
    
    def plot_solution_stability(self, stability_data):
        """
        Visualize solution stability metrics to detect unphysical behavior.
        """
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        times = stability_data['times']
        
        # Plot 1: Density Bounds
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(times, stability_data['density_bounds']['min'], 'b-', linewidth=2.5, label='Min', marker='v', markersize=5)
        ax1.plot(times, stability_data['density_bounds']['mean'], 'g-', linewidth=2.5, label='Mean', marker='o', markersize=5)
        ax1.plot(times, stability_data['density_bounds']['max'], 'r-', linewidth=2.5, label='Max', marker='^', markersize=5)
        ax1.axhline(0, color='k', linestyle='--', alpha=0.5, label='Zero')
        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Density Bounds Check', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Velocity Norms
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.semilogy(times, np.array(stability_data['velocity_norms']) + 1e-12, 'purple', linewidth=2.5, marker='o', markersize=5)
        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_ylabel('Max Velocity Norm (log)', fontsize=11)
        ax2.set_title('Velocity Magnitude Evolution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        
        # Plot 3: Negative Density Fraction
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(times, np.array(stability_data['negative_density_fraction']) * 100, 'red', linewidth=2.5, marker='o', markersize=5)
        ax3.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Time', fontsize=11)
        ax3.set_ylabel('Negative Density (%)', fontsize=11)
        ax3.set_title('Unphysical Density Detection', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Extreme Values
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(times, np.array(stability_data['extreme_values']) * 100, 'orange', linewidth=2.5, marker='o', markersize=5)
        ax4.set_xlabel('Time', fontsize=11)
        ax4.set_ylabel('Extreme Value Fraction (%)', fontsize=11)
        ax4.set_title('Outlier Detection (>10σ)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Solution Smoothness
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.semilogy(times, np.array(stability_data['solution_smoothness']) + 1e-12, 'green', linewidth=2.5, marker='o', markersize=5)
        ax5.set_xlabel('Time', fontsize=11)
        ax5.set_ylabel('Smoothness Metric (log)', fontsize=11)
        ax5.set_title('Solution Regularity', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, which='both')
        
        # Plot 6: Stability Summary
        ax6 = fig.add_subplot(gs[1, 2])
        # Compute overall stability score (0-1, higher is worse)
        neg_dens_score = np.array(stability_data['negative_density_fraction'])
        extreme_score = np.array(stability_data['extreme_values'])
        overall_score = 0.5 * neg_dens_score + 0.5 * extreme_score
        
        ax6.plot(times, overall_score * 100, 'darkred', linewidth=3, marker='o', markersize=6)
        ax6.fill_between(times, 0, overall_score * 100, alpha=0.3, color='red')
        ax6.axhline(5, color='orange', linestyle='--', linewidth=2, label='Warning (5%)')
        ax6.axhline(1, color='yellow', linestyle='--', linewidth=2, label='Acceptable (1%)')
        ax6.set_xlabel('Time', fontsize=11)
        ax6.set_ylabel('Instability Score (%)', fontsize=11)
        ax6.set_title('Overall Stability Assessment', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # Add parameterization info for proper interpretation of density metrics
        from config import PERTURBATION_TYPE, USE_PARAMETERIZATION
        param_info = f"[{PERTURBATION_TYPE}, ρ-param: {USE_PARAMETERIZATION}]"
        fig.text(0.99, 0.01, param_info, ha='right', va='bottom', fontsize=8,
                 fontstyle='italic', color='gray', transform=fig.transFigure)
        
        plt.suptitle('Solution Stability and Physical Validity Analysis', fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(f'{self.save_dir}/solution_stability.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_comprehensive_diagnostics(self, model, dimension, tmax):
        """
        Run all post-training diagnostics for long-term evolution analysis.
        
        Generates 6 critical diagnostic plots:
        1. Training diagnostics (loss, balance, density) - already generated
        2. PDE residual evolution - WHEN/WHERE/WHICH equations fail
        3. Conservation law violations - Mass/momentum drift over time
        4. Temporal error accumulation - How errors compound over time
        5. Spectral bias analysis - Frequency damping and mode evolution
        6. Solution stability - Detecting unphysical behavior
        
        Args:
            model: Trained PINN model
            dimension: Spatial dimension (2 or 3)
            tmax: Maximum time to analyze
        """
        print("\n" + "="*70)
        print("  Running Long-Term Evolution Diagnostics")
        print("="*70)
        
        # Plot 2: PDE Residual Evolution
        print("\n[1/5] Computing PDE residual evolution...")
        residuals = self.compute_residual_heatmap(model, dimension, tmax)
        self.plot_residual_heatmaps(residuals)
        print("      [OK] PDE residual evolution saved")
        
        # Plot 3: Conservation Laws
        print("\n[2/5] Analyzing conservation law violations...")
        conservation = self.check_conservation_laws(model, dimension, tmax)
        self.plot_conservation_laws(conservation)
        print("      [OK] Conservation violations plot saved")
        
        # Plot 4: Temporal Error Accumulation
        print("\n[3/5] Computing temporal error accumulation...")
        error_data = self.compute_temporal_error_accumulation(model, dimension, tmax)
        self.plot_temporal_error_accumulation(error_data)
        print("      [OK] Error accumulation plot saved")
        
        # Plot 5: Spectral Bias Analysis
        print("\n[4/5] Analyzing spectral bias and mode damping...")
        spectra = self.compute_spectral_content(model, dimension, tmax)
        self.plot_spectral_bias_analysis(spectra)
        print("      [OK] Spectral bias analysis saved")
        
        # Plot 6: Solution Stability
        print("\n[5/5] Checking solution stability...")
        stats = self.compute_temporal_statistics(model, dimension, tmax)
        stability_data = self.compute_solution_stability(model, dimension, tmax, stats)
        self.plot_solution_stability(stability_data)
        print("      [OK] Solution stability diagnostics saved")
        
        print("\n" + "="*70)
        print(f"  All diagnostics saved to: {self.save_dir}")
        print("="*70)
        print("\nLong-Term Evolution Diagnostic Summary:")
        print("  1. training_diagnostics.png - Training convergence and loss balance")
        print("  2. residual_heatmaps.png - Spatiotemporal PDE violation patterns")
        print("  3. conservation_laws.png - Conservation law drift over time")
        print("  4. temporal_error_accumulation.png - Error growth and compounding")
        print("  5. spectral_bias_analysis.png - Frequency damping and spectral issues")
        print("="*70 + "\n")


# ==================== STANDALONE EXECUTION MODE ====================

def load_model_for_diagnostics(model_path, device='cuda', dimension=None):
    """
    Load a saved PINN model for standalone diagnostics.
    
    Args:
        model_path: Path to saved model (.pth file)
        device: Device to load model on ('cuda' or 'cpu')
        dimension: Spatial dimension (2 or 3). If None, uses DIMENSION from config.
    
    Returns:
        Loaded PINN model ready for diagnostics
    """
    from config import (DIMENSION, harmonics, wave, num_of_waves, 
                        xmin, ymin, zmin)
    from core.model_architecture import PINN
    
    if dimension is None:
        dimension = DIMENSION
    
    # Calculate domain bounds (same as train.py)
    lam = wave
    xmax = xmin + lam * num_of_waves
    ymax = ymin + lam * num_of_waves
    zmax = zmin + lam * num_of_waves
    
    # Create PINN with correct architecture
    net = PINN(n_harmonics=harmonics)
    
    # Set domain bounds for periodic features
    if dimension == 1:
        spatial_rmin = [xmin]
        spatial_rmax = [xmax]
    elif dimension == 2:
        spatial_rmin = [xmin, ymin]
        spatial_rmax = [xmax, ymax]
    else:  # dimension == 3
        spatial_rmin = [xmin, ymin, zmin]
        spatial_rmax = [xmax, ymax, zmax]
    
    net.set_domain(rmin=spatial_rmin, rmax=spatial_rmax, dimension=dimension)
    
    # Load saved weights
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net = net.to(device)
    net.eval()
    
    print(f"Loaded model from: {model_path}")
    print(f"  Dimension: {dimension}D")
    print(f"  Domain: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}]" + 
          (f", z=[{zmin}, {zmax}]" if dimension == 3 else ""))
    print(f"  Device: {device}")
    
    return net


def parse_args():
    """Parse command-line arguments for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive PINN diagnostics on a saved model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training_diagnostics.py --model ./GRINN/model.pth
  python training_diagnostics.py --model ./model.pth --output ./my_diagnostics/ --tmax 5.0
  python training_diagnostics.py --model ./model.pth --device cpu
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to saved model (.pth file)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./diagnostics/",
        help="Output directory for diagnostic plots (default: ./diagnostics/)"
    )
    
    parser.add_argument(
        "--tmax", "-t",
        type=float,
        default=None,
        help="Maximum time for analysis (default: from config.py)"
    )
    
    parser.add_argument(
        "--dimension", "-d",
        type=int,
        choices=[2, 3],
        default=None,
        help="Spatial dimension override (default: from config.py)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Force device (default: auto-detect)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for standalone diagnostics."""
    args = parse_args()
    
    # Determine device
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get tmax from config if not specified
    if args.tmax is None:
        from config import tmax as config_tmax
        tmax = config_tmax
    else:
        tmax = args.tmax
    
    # Get dimension from config if not specified
    if args.dimension is None:
        from config import DIMENSION
        dimension = DIMENSION
    else:
        dimension = args.dimension
    
    # Get perturbation type for diagnostics
    from config import PERTURBATION_TYPE
    
    print("\n" + "="*70)
    print("  PINN Diagnostic Tool - Standalone Mode")
    print("="*70)
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print(f"  tmax: {tmax}")
    print(f"  Dimension: {dimension}D")
    print(f"  Perturbation: {PERTURBATION_TYPE}")
    print("="*70 + "\n")
    
    # Load model
    net = load_model_for_diagnostics(
        model_path=args.model,
        device=device,
        dimension=dimension
    )
    
    # Create diagnostics instance
    diagnostics = TrainingDiagnostics(
        save_dir=args.output,
        dimension=dimension,
        perturbation_type=PERTURBATION_TYPE
    )
    
    # Run comprehensive diagnostics
    diagnostics.run_comprehensive_diagnostics(
        model=net,
        dimension=dimension,
        tmax=tmax
    )
    
    print("\nDiagnostics completed successfully!")


if __name__ == "__main__":
    main()
