import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class TrainingDiagnostics:
    """
    Comprehensive diagnostics for PINN performance analysis.
    
    TWO DIAGNOSTIC MODES:
    
    1. LONG-TERM EVOLUTION MODE (2D, 3D sinusoidal, 3D power spectrum at t>0)
       Focus: Why does PINN fail at long times?
       Generates 6 diagnostic plots:
         - Training diagnostics: Loss convergence and balance
         - PDE residual evolution: When/where/which equations fail
         - Conservation violations: Mass and momentum drift
         - Temporal error accumulation: How errors compound over time
         - Spectral bias analysis: Frequency damping and mode evolution
         - Solution stability: Detecting unphysical behavior
    
    2. IC FITTING MODE (3D power spectrum only)
       Focus: Why can't PINN fit complex initial conditions?
       Generates 5 diagnostic plots:
         - Training diagnostics: Loss convergence with IC component breakdown
         - IC spatial comparison: Predicted vs true IC fields
         - IC power spectrum: Scale-by-scale fit quality
         - IC component convergence: Which component fails to converge
         - IC metrics summary: Correlation and RMS error statistics
    
    All diagnostics run automatically when ENABLE_TRAINING_DIAGNOSTICS = True
    """

    def __init__(self, save_dir='./diagnostics/', dimension=2, perturbation_type='power_spectrum'):
        self.save_dir = save_dir
        self.dimension = dimension
        self.perturbation_type = str(perturbation_type).lower()
        self.is_3d_power_spectrum = (dimension == 3 and self.perturbation_type == 'power_spectrum')
        
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
        
        # Additional tracking for 3D power spectrum IC diagnostics
        if self.is_3d_power_spectrum:
            self.history.update({
                'ic_rho': [],
                'ic_vx': [],
                'ic_vy': [],
                'ic_vz': [],
                'ic_phi': [],
            })

    def _prepare_inputs(self, geomtime_col):
        """Normalize collocation inputs to the model's expected format."""
        if isinstance(geomtime_col, (list, tuple)):
            return geomtime_col
        # Single tensor [N, D] -> list of [N,1]
        return [geomtime_col[:, i:i+1] for i in range(geomtime_col.shape[1])]

    def log_iteration(self, iteration, model, loss_dict, geomtime_col, ic_component_losses=None):
        """
        Call this every N iterations during training.
        
        Args:
            iteration: Current iteration number
            model: PINN model
            loss_dict: Dictionary with 'total', 'PDE', 'IC' losses
            geomtime_col: Collocation points
            ic_component_losses: (Optional) Dict with component IC losses for 3D power spectrum
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
        
        # Track component IC losses for 3D power spectrum
        if self.is_3d_power_spectrum and ic_component_losses is not None:
            self.history['ic_rho'].append(float(ic_component_losses.get('rho', np.nan)))
            self.history['ic_vx'].append(float(ic_component_losses.get('vx', np.nan)))
            self.history['ic_vy'].append(float(ic_component_losses.get('vy', np.nan)))
            self.history['ic_vz'].append(float(ic_component_losses.get('vz', np.nan)))
            self.history['ic_phi'].append(float(ic_component_losses.get('phi', np.nan)))

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
            'total_momentum_y': []
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
                total_mass = np.trapz(np.trapz(rho, dx=dy, axis=1), dx=dx, axis=0)
                total_px = np.trapz(np.trapz(rho * vx, dx=dy, axis=1), dx=dx, axis=0)
                total_py = np.trapz(np.trapz(rho * vy, dx=dy, axis=1), dx=dx, axis=0)
                
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
                
                dx = (xmax - xmin) / (n_grid_3d - 1)
                dy = (ymax - ymin) / (n_grid_3d - 1)
                dz = (zmax - zmin) / (n_grid_3d - 1)
                total_mass = np.trapz(np.trapz(np.trapz(rho, dx=dz, axis=2), dx=dy, axis=1), dx=dx, axis=0)
                total_px = np.trapz(np.trapz(np.trapz(rho * vx, dx=dz, axis=2), dx=dy, axis=1), dx=dx, axis=0)
                total_py = np.trapz(np.trapz(np.trapz(rho * vy, dx=dz, axis=2), dx=dy, axis=1), dx=dx, axis=0)
                
                conservation_data['times'].append(t_val)
                conservation_data['total_mass'].append(total_mass)
                conservation_data['total_momentum_x'].append(total_px)
                conservation_data['total_momentum_y'].append(total_py)
        
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
                
                kx = np.fft.fftfreq(n_grid_3d, d=1.0/n_grid_3d)
                ky = np.fft.fftfreq(n_grid_3d, d=1.0/n_grid_3d)
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
            
            # Track gradient norms for ALL fields including phi
            with torch.enable_grad():
                rho_grad_x = torch.autograd.grad(rho.sum(), X_flat, create_graph=False, retain_graph=True)[0]
                vx_grad_x = torch.autograd.grad(vx.sum(), X_flat, create_graph=False, retain_graph=True)[0]
                vy_grad_x = torch.autograd.grad(vy.sum(), X_flat, create_graph=False, retain_graph=True)[0]
                phi_grad_x = torch.autograd.grad(phi.sum(), X_flat, create_graph=False, retain_graph=True)[0]
                
                error_data['gradient_norms']['rho'].append(torch.norm(rho_grad_x).item())
                error_data['gradient_norms']['vx'].append(torch.norm(vx_grad_x).item())
                error_data['gradient_norms']['vy'].append(torch.norm(vy_grad_x).item())
                error_data['gradient_norms']['phi'].append(torch.norm(phi_grad_x).item())
                
                if dimension == 3:
                    vz_grad_x = torch.autograd.grad(vz.sum(), X_flat, create_graph=False, retain_graph=True)[0]
                    error_data['gradient_norms']['vz'].append(torch.norm(vz_grad_x).item())
            
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
        print("  6. solution_stability.png - Physical validity and stability metrics")
        print("="*70 + "\n")
    
    
    def run_case_specific_diagnostics(self, model, tmax=None, true_ic_data=None):
        """
        Smart dispatcher: runs appropriate diagnostics based on case type.
        
        - 2D cases (any perturbation): High-tmax temporal evolution diagnostics
        - 3D power spectrum: IC fitting diagnostics  
        - 3D sinusoidal: High-tmax temporal evolution diagnostics (fallback to 2D approach)
        
        Args:
            model: Trained PINN model
            tmax: Maximum time (required for 2D or 3D non-power-spectrum cases)
            true_ic_data: Dict with IC data (required for 3D power spectrum)
                         {'colloc_IC', 'rho', 'vx', 'vy', 'vz'}
        """
        if self.is_3d_power_spectrum:
            # 3D power spectrum: Focus on IC fitting
            if true_ic_data is None:
                raise ValueError("true_ic_data required for 3D power spectrum diagnostics")
            self.run_3d_power_spectrum_diagnostics(model, true_ic_data)
        else:
            # All other cases: Focus on temporal evolution
            if tmax is None:
                raise ValueError("tmax required for temporal evolution diagnostics")
            self.run_comprehensive_diagnostics(model, self.dimension, tmax)
    # ==================== 3D POWER SPECTRUM IC DIAGNOSTICS ====================
    
    def compute_ic_spatial_comparison(self, model, true_ic_data, n_grid=80):
        """
        Compare predicted vs true ICs spatially for both 2D and 3D cases.
        Critical for diagnosing WHERE the network fails to fit the IC structure.
        
        Args:
            model: Trained PINN model
            true_ic_data: Dict with IC collocation points and true values
                         2D: {'colloc_IC', 'rho', 'vx', 'vy'}
                         3D: {'colloc_IC', 'rho', 'vx', 'vy', 'vz'}
            n_grid: Grid resolution per dimension (for visualization grid)
        
        Returns:
            Dict with predicted and true fields, plus errors
        """
        device = next(model.parameters()).device
        
        print(f"  Computing IC spatial comparison on training IC points...")
        
        # Use the actual IC collocation points from training
        colloc_IC = true_ic_data['colloc_IC']
        
        # Get predictions at IC points
        with torch.no_grad():
            pred = model(colloc_IC)
            rho_pred = pred[:, 0].cpu().numpy()
            vx_pred = pred[:, 1].cpu().numpy()
            vy_pred = pred[:, 2].cpu().numpy()
            if self.dimension == 3:
                vz_pred = pred[:, 3].cpu().numpy()
                phi_pred = pred[:, 4].cpu().numpy()
            else:
                phi_pred = pred[:, 3].cpu().numpy()
        
        # Get true ICs (detach in case they have gradients)
        rho_true = true_ic_data['rho'].detach().cpu().numpy()
        vx_true = true_ic_data['vx'].detach().cpu().numpy()
        vy_true = true_ic_data['vy'].detach().cpu().numpy()
        if self.dimension == 3 and true_ic_data['vz'] is not None:
            vz_true = true_ic_data['vz'].detach().cpu().numpy()
        
        # For visualization, create a regular grid
        # Get actual domain bounds
        xmin, xmax, ymin, ymax, zmin, zmax = self._get_domain_bounds(model, self.dimension)
        
        x = torch.linspace(xmin, xmax, n_grid, device=device)
        y = torch.linspace(ymin, ymax, n_grid, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        T = torch.zeros_like(X)
        
        if self.dimension == 3:
            # For 3D, take z=middle slice
            Z = torch.full_like(X, (zmin + zmax) / 2.0)
            colloc_viz = [X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1), T.reshape(-1, 1)]
        else:
            # For 2D
            colloc_viz = [X.reshape(-1, 1), Y.reshape(-1, 1), T.reshape(-1, 1)]
        
        # Get predictions on visualization grid
        with torch.no_grad():
            pred_viz = model(colloc_viz)
            rho_pred_viz = pred_viz[:, 0].reshape(n_grid, n_grid).cpu().numpy()
            vx_pred_viz = pred_viz[:, 1].reshape(n_grid, n_grid).cpu().numpy()
            vy_pred_viz = pred_viz[:, 2].reshape(n_grid, n_grid).cpu().numpy()
            if self.dimension == 3:
                vz_pred_viz = pred_viz[:, 3].reshape(n_grid, n_grid).cpu().numpy()
                phi_pred_viz = pred_viz[:, 4].reshape(n_grid, n_grid).cpu().numpy()
            else:
                phi_pred_viz = pred_viz[:, 3].reshape(n_grid, n_grid).cpu().numpy()
        
        # For "true" visualization, we need to interpolate from IC points to regular grid
        # Extract spatial coordinates from IC collocation points (detach first)
        x_ic = colloc_IC[0].detach().cpu().numpy().flatten()
        y_ic = colloc_IC[1].detach().cpu().numpy().flatten()
        
        if self.dimension == 3:
            z_ic = colloc_IC[2].detach().cpu().numpy().flatten()
            # Find points near z=middle for visualization
            zmid = (zmin + zmax) / 2.0
            z_tolerance = 0.1 * (zmax - zmin)  # 10% of domain
            mask_z = np.abs(z_ic - zmid) < z_tolerance
        else:
            mask_z = np.ones(len(x_ic), dtype=bool)
        
        if mask_z.sum() > 100:  # Need enough points for interpolation
            from scipy.interpolate import griddata
            points_ic = np.column_stack([x_ic[mask_z], y_ic[mask_z]])
            X_viz = X.cpu().numpy()
            Y_viz = Y.cpu().numpy()
            points_viz = np.column_stack([X_viz.flatten(), Y_viz.flatten()])
            
            # Interpolate with nearest neighbor fallback for NaN values
            rho_true_viz = griddata(points_ic, rho_true[mask_z], points_viz, method='linear', fill_value=np.nan).reshape(n_grid, n_grid)
            vx_true_viz = griddata(points_ic, vx_true[mask_z], points_viz, method='linear', fill_value=np.nan).reshape(n_grid, n_grid)
            vy_true_viz = griddata(points_ic, vy_true[mask_z], points_viz, method='linear', fill_value=np.nan).reshape(n_grid, n_grid)
            
            # For uniform density (all values the same), fill NaNs with the constant value
            # For varying fields, use nearest neighbor interpolation for NaN regions
            if np.isnan(rho_true_viz).any():
                if np.std(rho_true[mask_z]) < 1e-10:
                    # Uniform field - fill with mean
                    rho_true_viz = np.nan_to_num(rho_true_viz, nan=np.mean(rho_true[mask_z]))
                else:
                    # Varying field - use nearest neighbor for gaps
                    rho_nn = griddata(points_ic, rho_true[mask_z], points_viz, method='nearest').reshape(n_grid, n_grid)
                    rho_true_viz = np.where(np.isnan(rho_true_viz), rho_nn, rho_true_viz)
            
            if np.isnan(vx_true_viz).any():
                vx_nn = griddata(points_ic, vx_true[mask_z], points_viz, method='nearest').reshape(n_grid, n_grid)
                vx_true_viz = np.where(np.isnan(vx_true_viz), vx_nn, vx_true_viz)
            
            if np.isnan(vy_true_viz).any():
                vy_nn = griddata(points_ic, vy_true[mask_z], points_viz, method='nearest').reshape(n_grid, n_grid)
                vy_true_viz = np.where(np.isnan(vy_true_viz), vy_nn, vy_true_viz)
            
            if self.dimension == 3:
                vz_true_viz = griddata(points_ic, vz_true[mask_z], points_viz, method='linear', fill_value=np.nan).reshape(n_grid, n_grid)
                if np.isnan(vz_true_viz).any():
                    vz_nn = griddata(points_ic, vz_true[mask_z], points_viz, method='nearest').reshape(n_grid, n_grid)
                    vz_true_viz = np.where(np.isnan(vz_true_viz), vz_nn, vz_true_viz)
        else:
            # Not enough points for interpolation, use predicted as reference
            print("  [WARN] Not enough IC points for ground truth visualization")
            rho_true_viz = rho_pred_viz
            vx_true_viz = vx_pred_viz
            vy_true_viz = vy_pred_viz
            if self.dimension == 3:
                vz_true_viz = vz_pred_viz
        
        result = {
            'x': x.cpu().numpy(),
            'y': y.cpu().numpy(),
            'rho_pred': rho_pred_viz,
            'rho_true': rho_true_viz,
            'vx_pred': vx_pred_viz,
            'vx_true': vx_true_viz,
            'vy_pred': vy_pred_viz,
            'vy_true': vy_true_viz,
            'phi_pred': phi_pred_viz,
            # Also store scatter point data for metrics
            'rho_pred_scatter': rho_pred,
            'rho_true_scatter': rho_true,
            'vx_pred_scatter': vx_pred,
            'vx_true_scatter': vx_true,
            'vy_pred_scatter': vy_pred,
            'vy_true_scatter': vy_true,
        }
        
        if self.dimension == 3:
            result.update({
                'vz_pred': vz_pred_viz,
                'vz_true': vz_true_viz,
                'vz_pred_scatter': vz_pred,
                'vz_true_scatter': vz_true,
            })
        
        return result
    
    def plot_ic_power_spectrum_comparison(self, ps_data):
        """
        Plot power spectrum comparison for all velocity components.
        Shows which scales (k-modes) are missing or damped.
        """
        fig = plt.figure(figsize=(16, 5))
        gs = GridSpec(1, 3, figure=fig)
        
        components = [('vx', 'X-Velocity'), ('vy', 'Y-Velocity'), ('vz', 'Z-Velocity')]
        
        for idx, (comp, label) in enumerate(components):
            ax = fig.add_subplot(gs[0, idx])
            
            # Radial binning
            K = ps_data['K']
            power_pred = ps_data[f'power_{comp}_pred']
            power_true = ps_data[f'power_{comp}_true']
            
            k_max = np.max(K)
            k_bins = np.linspace(0, k_max, 30)
            power_pred_avg = []
            power_true_avg = []
            k_centers = []
            
            for i in range(len(k_bins)-1):
                mask = (K >= k_bins[i]) & (K < k_bins[i+1])
                if mask.any():
                    power_pred_avg.append(np.mean(power_pred[mask]))
                    power_true_avg.append(np.mean(power_true[mask]))
                    k_centers.append((k_bins[i] + k_bins[i+1]) / 2)
            
            if len(k_centers) > 0:
                ax.loglog(k_centers, power_true_avg, 'r-', linewidth=2.5, label='True IC', marker='o', markersize=4)
                ax.loglog(k_centers, power_pred_avg, 'b--', linewidth=2.5, label='Predicted', marker='s', markersize=4)
                
                ax.set_xlabel('Wavenumber k', fontsize=11)
                ax.set_ylabel('Power', fontsize=11)
                ax.set_title(f'{label} Power Spectrum', fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, which='both')
        
        plt.suptitle('IC Power Spectrum: Predicted vs True', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/ic_power_spectrum.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_ic_component_convergence(self):
        """
        Plot IC component loss convergence over training.
        Shows WHICH component (rho, vx, vy, vz, phi) is hardest to fit.
        """
        if not self.is_3d_power_spectrum or len(self.history['iteration']) == 0:
            print("IC component tracking not available.")
            return
        
        iters = self.history['iteration']
        
        # Replace zeros and very small values with a minimum threshold for log plotting
        def safe_log_data(data):
            """Replace zeros/negative values with small epsilon for log plotting."""
            data_array = np.array(data)
            data_array = np.where(data_array <= 0, 1e-12, data_array)
            return data_array
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: All IC components on same plot
        ax1 = fig.add_subplot(gs[0, :])
        
        # Check if rho is zero/constant (uniform field case)
        rho_is_uniform = np.max(self.history['ic_rho']) < 1e-10
        
        if not rho_is_uniform:
            ax1.semilogy(iters, safe_log_data(self.history['ic_rho']), linewidth=2, label='ρ', 
                        marker='o', markersize=3, markevery=max(1, len(iters)//20))
        else:
            # Plot a dashed line at the bottom to indicate uniform/zero loss
            ax1.axhline(1e-12, color='C0', linestyle='--', linewidth=2, label='ρ (uniform IC, no loss)', alpha=0.5)
        
        ax1.semilogy(iters, safe_log_data(self.history['ic_vx']), linewidth=2, label='vx', 
                    marker='s', markersize=3, markevery=max(1, len(iters)//20))
        ax1.semilogy(iters, safe_log_data(self.history['ic_vy']), linewidth=2, label='vy', 
                    marker='^', markersize=3, markevery=max(1, len(iters)//20))
        ax1.semilogy(iters, safe_log_data(self.history['ic_vz']), linewidth=2, label='vz', 
                    marker='d', markersize=3, markevery=max(1, len(iters)//20))
        ax1.semilogy(iters, safe_log_data(self.history['ic_phi']), linewidth=2, label='φ', 
                    marker='v', markersize=3, markevery=max(1, len(iters)//20))
        
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Loss (log scale)', fontsize=11)
        ax1.set_title('IC Component Losses', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10, ncol=5)
        ax1.grid(True, alpha=0.3)
        
        # Individual component plots
        components = [
            ('ic_rho', 'ρ (Density)', gs[1, 0], rho_is_uniform),
            ('ic_vx', 'vx (X-Velocity)', gs[1, 1], False),
            ('ic_vy', 'vy (Y-Velocity)', gs[1, 2], False),
        ]
        
        for hist_key, title, grid_pos, is_uniform in components:
            ax = fig.add_subplot(grid_pos)
            if is_uniform:
                # Show a flat line at bottom with annotation
                ax.axhline(1e-12, color='steelblue', linestyle='--', linewidth=2.5, alpha=0.5)
                ax.text(0.5, 0.5, 'Uniform IC\n(no spatial variation)', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                ax.set_ylim([1e-13, 1e-10])
            else:
                ax.semilogy(iters, safe_log_data(self.history[hist_key]), linewidth=2.5, color='steelblue')
            
            ax.set_xlabel('Iteration', fontsize=10)
            ax.set_ylabel('Loss (log scale)', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('IC Component Convergence Analysis', fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(f'{self.save_dir}/ic_component_convergence.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def compute_ic_correlation_metrics(self, model, true_ic_data):
        """
        Compute spatial correlation between predicted and true ICs.
        High correlation = good spatial structure capture.
        """
        print(f"  Computing IC correlation metrics on training IC points...")
        
        # Use IC collocation points
        colloc_IC = true_ic_data['colloc_IC']
        
        # Get predictions
        with torch.no_grad():
            pred = model(colloc_IC)
            rho_pred = pred[:, 0].cpu().numpy().flatten()
            vx_pred = pred[:, 1].cpu().numpy().flatten()
            vy_pred = pred[:, 2].cpu().numpy().flatten()
            if self.dimension == 3:
                vz_pred = pred[:, 3].cpu().numpy().flatten()
        
        # Get true ICs (detach in case they have gradients)
        rho_true = true_ic_data['rho'].detach().cpu().numpy().flatten()
        vx_true = true_ic_data['vx'].detach().cpu().numpy().flatten()
        vy_true = true_ic_data['vy'].detach().cpu().numpy().flatten()
        if self.dimension == 3 and true_ic_data['vz'] is not None:
            vz_true = true_ic_data['vz'].detach().cpu().numpy().flatten()
        
        # Compute correlations - handle uniform fields (zero variance)
        def safe_corrcoef(pred, true):
            """Compute correlation, handling constant fields gracefully."""
            # Check if either field is constant
            if np.std(pred) < 1e-10 or np.std(true) < 1e-10:
                # For uniform fields, correlation is undefined
                # If both are uniform and equal, perfect match (1.0)
                # If different constants, poor match (use normalized error)
                if np.std(pred) < 1e-10 and np.std(true) < 1e-10:
                    # Both uniform - check if they match
                    mean_diff = abs(np.mean(pred) - np.mean(true))
                    return 1.0 if mean_diff < 1e-6 else 0.0
                else:
                    # One uniform, one not - cannot capture structure
                    return 0.0
            else:
                return np.corrcoef(pred, true)[0, 1]
        
        correlations = {
            'rho': safe_corrcoef(rho_pred, rho_true),
            'vx': safe_corrcoef(vx_pred, vx_true),
            'vy': safe_corrcoef(vy_pred, vy_true),
        }
        
        # Compute RMS errors
        rms_errors = {
            'rho': np.sqrt(np.mean((rho_pred - rho_true)**2)),
            'vx': np.sqrt(np.mean((vx_pred - vx_true)**2)),
            'vy': np.sqrt(np.mean((vy_pred - vy_true)**2)),
        }
        
        if self.dimension == 3:
            correlations['vz'] = safe_corrcoef(vz_pred, vz_true)
            rms_errors['vz'] = np.sqrt(np.mean((vz_pred - vz_true)**2))
        
        return correlations, rms_errors
    
    def plot_ic_metrics_summary(self, correlations, rms_errors):
        """
        Summary plot: correlation and RMS error for each IC component.
        Quick visual diagnostic of IC fitting quality.
        """
        fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 2, figure=fig)
        
        components = ['rho', 'vx', 'vy', 'vz']
        labels = ['ρ', 'vx', 'vy', 'vz']
        
        # Correlation plot
        ax1 = fig.add_subplot(gs[0, 0])
        corr_vals = [correlations[c] for c in components]
        
        # Handle NaN/inf correlations from uniform fields
        corr_vals_safe = []
        for val in corr_vals:
            if np.isnan(val) or np.isinf(val):
                corr_vals_safe.append(0.0)  # Show as 0 for visualization
            else:
                corr_vals_safe.append(val)
        
        colors = ['green' if c > 0.9 else 'orange' if c > 0.7 else 'red' for c in corr_vals_safe]
        bars1 = ax1.bar(labels, corr_vals_safe, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.axhline(1.0, color='b', linestyle='--', linewidth=1.5, label='Perfect correlation')
        ax1.axhline(0.9, color='g', linestyle=':', linewidth=1.5, label='Good threshold')
        ax1.set_ylabel('Correlation Coefficient', fontsize=11)
        ax1.set_title('Spatial Correlation: Predicted vs True IC', fontsize=12, fontweight='bold')
        ax1.set_ylim([0, 1.05])
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val, orig_val in zip(bars1, corr_vals_safe, corr_vals):
            height = bar.get_height()
            if np.isnan(orig_val) or np.isinf(orig_val):
                label_text = 'N/A\n(uniform)'
                fontsize = 8
            else:
                label_text = f'{val:.3f}'
                fontsize = 10
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    label_text, ha='center', va='bottom', fontsize=fontsize, fontweight='bold')
        
        # RMS Error plot
        ax2 = fig.add_subplot(gs[0, 1])
        rms_vals = [rms_errors[c] for c in components]
        bars2 = ax2.bar(labels, rms_vals, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('RMS Error', fontsize=11)
        ax2.set_title('RMS Error: Predicted vs True IC', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars2, rms_vals):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle('IC Fitting Quality Metrics', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/ic_metrics_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_3d_power_spectrum_diagnostics(self, model, true_ic_data):
        """
        Run diagnostics for 3D power spectrum case.
        Focus on IC fitting quality, not temporal evolution.
        
        Generates 5 critical diagnostic plots:
        1. Training diagnostics (loss, balance, density) - already generated
        2. IC spatial comparison (predicted vs true fields)
        3. IC power spectrum comparison (scale-by-scale)
        4. IC component convergence (which component fails)
        5. IC metrics summary (correlation and RMS error)
        
        Args:
            model: Trained PINN model
            true_ic_data: Dict with IC data {'colloc_IC', 'rho', 'vx', 'vy', 'vz'}
        """
        print("\n" + "="*70)
        print("  Running 3D Power Spectrum IC Diagnostics")
        print("="*70)
        
        # Plot 2: IC Spatial Comparison
        print("\n[1/4] Computing IC spatial comparison...")
        ic_data = self.compute_ic_spatial_comparison(model, true_ic_data)
        self.plot_ic_spatial_comparison(ic_data)
        print("      [OK] IC spatial comparison saved")
        
        # Plot 3: IC Power Spectrum Comparison
        print("\n[2/4] Computing IC power spectrum comparison...")
        ps_data = self.compute_ic_power_spectrum_comparison(model, true_ic_data)
        self.plot_ic_power_spectrum_comparison(ps_data)
        print("      [OK] IC power spectrum comparison saved")
        
        # Plot 4: IC Component Convergence
        print("\n[3/4] Plotting IC component convergence...")
        self.plot_ic_component_convergence()
        print("      [OK] IC component convergence saved")
        
        # Plot 5: IC Metrics Summary
        print("\n[4/4] Computing IC metrics summary...")
        correlations, rms_errors = self.compute_ic_correlation_metrics(model, true_ic_data)
        self.plot_ic_metrics_summary(correlations, rms_errors)
        print("      [OK] IC metrics summary saved")
        
        print("\n" + "="*70)
        print(f"  All diagnostics saved to: {self.save_dir}")
        print("="*70)
        print("\nDiagnostic Summary:")
        print("  1. training_diagnostics.png - Training convergence analysis")
        print("  2. ic_spatial_comparison.png - Predicted vs true IC fields")
        print("  3. ic_power_spectrum.png - Scale-by-scale power spectrum fit")
        print("  4. ic_component_convergence.png - Which IC component fails")
        print("  5. ic_metrics_summary.png - Correlation and RMS error metrics")
        print("="*70 + "\n")
        
        # Print numerical summary
        print("\nIC Fitting Quality Summary:")
        print("-" * 50)
        print(f"{'Component':<15} {'Correlation':<15} {'RMS Error':<15}")
        print("-" * 50)
        for comp in ['rho', 'vx', 'vy', 'vz']:
            print(f"{comp:<15} {correlations[comp]:>14.4f} {rms_errors[comp]:>14.6f}")
        print("-" * 50)
        avg_corr = np.mean([correlations[c] for c in ['rho', 'vx', 'vy', 'vz']])
        print(f"{'Average':<15} {avg_corr:>14.4f}")
        print("-" * 50 + "\n")
    
    # ==================== UNIFIED DIAGNOSTICS (5 PLOTS FOR ALL CASES) ====================
    
    def run_unified_diagnostics(self, model, true_ic_data, final_iteration):
        """
        Unified diagnostics that generate exactly 5 plots for all cases (2D/3D, any perturbation).
        
        The 5 plots are:
        1. Training diagnostics (loss convergence and balance)
        2. IC spatial comparison (predicted vs true fields)
        3. IC power spectrum (for power_spectrum) or field spectrum (for sinusoidal)
        4. IC component convergence (which component fails)
        5. IC metrics summary (correlation and RMS error)
        
        Args:
            model: Trained PINN model
            true_ic_data: Dict with IC data {'colloc_IC', 'rho', 'vx', 'vy', 'vz'}
            final_iteration: Final iteration number
        """
        print("\n" + "="*70)
        print("  Running Unified Training Diagnostics (5 plots)")
        print("="*70)
        
        # Plot 1: Training diagnostics
        print("\n[1/5] Plotting training diagnostics...")
        self.plot_diagnostics(final_iteration)
        print("      [OK] Training diagnostics saved")
        
        # Plot 2: IC Spatial Comparison
        print("\n[2/5] Computing IC spatial comparison...")
        ic_data = self.compute_ic_spatial_comparison(model, true_ic_data)
        self.plot_ic_spatial_comparison(ic_data)
        print("      [OK] IC spatial comparison saved")
        
        # Plot 3: IC Power/Field Spectrum Comparison
        print("\n[3/5] Computing spectrum comparison...")
        if self.perturbation_type == 'power_spectrum':
            # For power spectrum: compare power spectra
            ps_data = self.compute_ic_power_spectrum_comparison(model, true_ic_data)
            self.plot_ic_power_spectrum_comparison(ps_data)
        else:
            # For sinusoidal: compute field spectra from predictions
            ps_data = self.compute_field_spectrum_comparison(model, true_ic_data)
            self.plot_field_spectrum_comparison(ps_data)
        print("      [OK] Spectrum comparison saved")
        
        # Plot 4: IC Component Convergence
        print("\n[4/5] Plotting IC component convergence...")
        self.plot_ic_component_convergence()
        print("      [OK] IC component convergence saved")
        
        # Plot 5: IC Metrics Summary
        print("\n[5/5] Computing IC metrics summary...")
        correlations, rms_errors = self.compute_ic_correlation_metrics(model, true_ic_data)
        self.plot_ic_metrics_summary(correlations, rms_errors)
        print("      [OK] IC metrics summary saved")
        
        print("\n" + "="*70)
        print(f"  All diagnostics saved to: {self.save_dir}")
        print("="*70)
        print("\nDiagnostic Summary:")
        print("  1. training_diagnostics.png - Training convergence analysis")
        print("  2. ic_spatial_comparison.png - Predicted vs true IC fields")
        print("  3. ic_power_spectrum.png - Spectrum comparison")
        print("  4. ic_component_convergence.png - IC component convergence")
        print("  5. ic_metrics_summary.png - Correlation and RMS error metrics")
        print("="*70 + "\n")
        
        # Print numerical summary
        print("\nIC Fitting Quality Summary:")
        print("-" * 50)
        print(f"{'Component':<15} {'Correlation':<15} {'RMS Error':<15}")
        print("-" * 50)
        comps = ['rho', 'vx', 'vy', 'vz'] if self.dimension == 3 else ['rho', 'vx', 'vy']
        for comp in comps:
            if comp in correlations:
                print(f"{comp:<15} {correlations[comp]:>14.4f} {rms_errors[comp]:>14.6f}")
        print("-" * 50)
        avg_corr = np.mean([correlations[c] for c in comps if c in correlations])
        print(f"{'Average':<15} {avg_corr:>14.4f}")
        print("-" * 50 + "\n")
    
    def compute_field_spectrum_comparison(self, model, true_ic_data, n_grid=128):
        """
        For sinusoidal cases: Compute power spectra of predicted vs true IC fields.
        Similar to power spectrum comparison but computed from actual fields.
        """
        device = next(model.parameters()).device
        
        print(f"  Computing field spectrum comparison on IC points...")
        
        # Use IC collocation points
        colloc_IC = true_ic_data['colloc_IC']
        
        # Get predictions
        with torch.no_grad():
            pred = model(colloc_IC)
            rho_pred = pred[:, 0].cpu().numpy()
            vx_pred = pred[:, 1].cpu().numpy()
            vy_pred = pred[:, 2].cpu().numpy()
            if self.dimension == 3:
                vz_pred = pred[:, 3].cpu().numpy()
        
        # Get true ICs
        rho_true = true_ic_data['rho'].detach().cpu().numpy()
        vx_true = true_ic_data['vx'].detach().cpu().numpy()
        vy_true = true_ic_data['vy'].detach().cpu().numpy()
        if self.dimension == 3:
            vz_true = true_ic_data['vz'].detach().cpu().numpy()
        
        # Need to interpolate to regular grid for FFT
        x_ic = colloc_IC[0].detach().cpu().numpy().flatten()
        y_ic = colloc_IC[1].detach().cpu().numpy().flatten()
        
        if self.dimension == 3:
            z_ic = colloc_IC[2].detach().cpu().numpy().flatten()
            # Use points near z=middle for 2D slice
            zmid = (zmin + zmax) / 2.0
            z_tolerance = 0.1 * (zmax - zmin)  # 10% of domain
            mask_z = np.abs(z_ic - zmid) < z_tolerance
            if mask_z.sum() < 100:
                mask_z = np.ones(len(z_ic), dtype=bool)
        else:
            mask_z = np.ones(len(x_ic), dtype=bool)
        
        # Create regular grid for FFT
        # Get actual domain bounds
        xmin, xmax, ymin, ymax, zmin, zmax = self._get_domain_bounds(model, self.dimension)
        
        x_grid = np.linspace(xmin, xmax, n_grid)
        y_grid = np.linspace(ymin, ymax, n_grid)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        # Interpolate from scattered IC points to regular grid
        from scipy.interpolate import griddata
        points_ic = np.column_stack([x_ic[mask_z], y_ic[mask_z]])
        points_grid = np.column_stack([X_grid.flatten(), Y_grid.flatten()])
        
        # Interpolate density
        rho_pred_grid = griddata(points_ic, rho_pred[mask_z], points_grid, method='linear', fill_value=0).reshape(n_grid, n_grid)
        rho_true_grid = griddata(points_ic, rho_true[mask_z], points_grid, method='linear', fill_value=0).reshape(n_grid, n_grid)
        
        # Interpolate velocities
        vx_pred_grid = griddata(points_ic, vx_pred[mask_z], points_grid, method='linear', fill_value=0).reshape(n_grid, n_grid)
        vy_pred_grid = griddata(points_ic, vy_pred[mask_z], points_grid, method='linear', fill_value=0).reshape(n_grid, n_grid)
        
        vx_true_grid = griddata(points_ic, vx_true[mask_z], points_grid, method='linear', fill_value=0).reshape(n_grid, n_grid)
        vy_true_grid = griddata(points_ic, vy_true[mask_z], points_grid, method='linear', fill_value=0).reshape(n_grid, n_grid)
        
        if self.dimension == 3:
            vz_pred_grid = griddata(points_ic, vz_pred[mask_z], points_grid, method='linear', fill_value=0).reshape(n_grid, n_grid)
            vz_true_grid = griddata(points_ic, vz_true[mask_z], points_grid, method='linear', fill_value=0).reshape(n_grid, n_grid)
        
        # Compute power spectra with correct domain spacing
        def compute_power_spectrum(field):
            fft = np.fft.fft2(field)
            power = np.abs(np.fft.fftshift(fft))**2
            
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
            
            return power, K
        
        power_rho_pred, K = compute_power_spectrum(rho_pred_grid)
        power_rho_true, _ = compute_power_spectrum(rho_true_grid)
        
        power_vx_pred, _ = compute_power_spectrum(vx_pred_grid)
        power_vx_true, _ = compute_power_spectrum(vx_true_grid)
        
        power_vy_pred, _ = compute_power_spectrum(vy_pred_grid)
        power_vy_true, _ = compute_power_spectrum(vy_true_grid)
        
        result = {
            'K': K,
            'power_rho_pred': power_rho_pred,
            'power_rho_true': power_rho_true,
            'power_vx_pred': power_vx_pred,
            'power_vx_true': power_vx_true,
            'power_vy_pred': power_vy_pred,
            'power_vy_true': power_vy_true,
        }
        
        if self.dimension == 3:
            power_vz_pred, _ = compute_power_spectrum(vz_pred_grid)
            power_vz_true, _ = compute_power_spectrum(vz_true_grid)
            result['power_vz_pred'] = power_vz_pred
            result['power_vz_true'] = power_vz_true
        
        return result
    
    def plot_field_spectrum_comparison(self, ps_data):
        """
        Plot field spectrum comparison for sinusoidal cases.
        Shows density and velocity spectra.
        """
        if self.dimension == 3:
            fig = plt.figure(figsize=(18, 5))
            gs = GridSpec(1, 4, figure=fig)
            components = [('rho', 'Density'), ('vx', 'X-Velocity'), ('vy', 'Y-Velocity'), ('vz', 'Z-Velocity')]
        else:
            fig = plt.figure(figsize=(14, 5))
            gs = GridSpec(1, 3, figure=fig)
            components = [('rho', 'Density'), ('vx', 'X-Velocity'), ('vy', 'Y-Velocity')]
        
        for idx, (comp, label) in enumerate(components):
            ax = fig.add_subplot(gs[0, idx])
            
            # Radial binning
            K = ps_data['K']
            power_pred = ps_data[f'power_{comp}_pred']
            power_true = ps_data[f'power_{comp}_true']
            
            k_max = np.max(K)
            k_bins = np.linspace(0, k_max, 30)
            power_pred_avg = []
            power_true_avg = []
            k_centers = []
            
            for i in range(len(k_bins)-1):
                mask = (K >= k_bins[i]) & (K < k_bins[i+1])
                if mask.any():
                    power_pred_avg.append(np.mean(power_pred[mask]))
                    power_true_avg.append(np.mean(power_true[mask]))
                    k_centers.append((k_bins[i] + k_bins[i+1]) / 2)
            
            if len(k_centers) > 0:
                ax.loglog(k_centers, power_true_avg, 'r-', linewidth=2.5, label='True IC', marker='o', markersize=4)
                ax.loglog(k_centers, power_pred_avg, 'b--', linewidth=2.5, label='Predicted', marker='s', markersize=4)
                
                ax.set_xlabel('Wavenumber k', fontsize=11)
                ax.set_ylabel('Power', fontsize=11)
                ax.set_title(f'{label} Spectrum', fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, which='both')
        
        plt.suptitle('IC Field Spectrum: Predicted vs True', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/ic_power_spectrum.png', dpi=150, bbox_inches='tight')
        plt.close()