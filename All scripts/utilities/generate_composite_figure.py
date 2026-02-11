"""
Generate a composite workflow figure for the PINN paper:
Left: Initial conditions (velocity field)
Center: PINN architecture schematic
Right: PINN predictions (density, velocity at final time)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy.interpolate import RegularGridInterpolator

# ============================================================================
# Configuration (hardcoded for 2D power spectrum case)
# ============================================================================
cs = 1.0
rho_o = 1.0
a = 0.1
wave = 7.0
num_of_waves = 2.0
tmax = 3.0
N_GRID = 400
POWER_EXPONENT = -4
RANDOM_SEED = 93
num_neurons = 64
harmonics = 3
num_layers = 5

# Domain
lam = wave
Lx = lam * num_of_waves
Ly = lam * num_of_waves
xmin, ymin = 0.0, 0.0
xmax, ymax = Lx, Ly

# ============================================================================
# Helper functions
# ============================================================================

def generate_velocity_field_power_spectrum(nx, ny, Lx, Ly, power_index=-4.0, amplitude=0.1, random_seed=42):
    """Generate 2D velocity field with power spectrum."""
    from scipy.fft import fft2, ifft2, fftfreq
    
    rng = np.random.default_rng(random_seed)
    
    dx = Lx / nx
    dy = Ly / ny
    
    kx = 2 * np.pi * fftfreq(nx, dx)
    ky = 2 * np.pi * fftfreq(ny, dy)
    kxg, kyg = np.meshgrid(kx, ky, indexing='ij')
    kk = np.sqrt(kxg**2 + kyg**2)
    
    def synthesize_component():
        field = rng.standard_normal((nx, ny))
        F = fft2(field)
        
        filt = np.zeros_like(kk)
        mask = kk > 0
        filt[mask] = kk[mask]**(power_index / 2.0)
        
        F_filtered = F * filt
        comp = np.real(ifft2(F_filtered))
        
        comp -= np.mean(comp)
        std = np.std(comp)
        if std > 0:
            comp = comp * (amplitude / std)
        return comp
    
    vx = synthesize_component()
    vy = synthesize_component()
    return vx, vy


class Sin(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)


class PINN(torch.nn.Module):
    """Simplified PINN for loading 2D model."""
    def __init__(self, num_neurons=64, num_layers=5, n_harmonics=3):
        super().__init__()
        self.num_neurons = num_neurons
        self.n_harmonics = n_harmonics
        self.num_layers = max(2, int(num_layers))
        
        self.xmin = self.xmax = self.ymin = self.ymax = None
        self.zmin = self.zmax = None
        
        def _make_branch(in_dim, out_dim):
            layers = []
            layers.append(torch.nn.Linear(in_dim, self.num_neurons))
            for _ in range(self.num_layers - 2):
                layers.append(Sin())
                layers.append(torch.nn.Linear(self.num_neurons, self.num_neurons))
            if self.num_layers > 2:
                layers.append(Sin())
            layers.append(torch.nn.Linear(self.num_neurons, out_dim))
            return torch.nn.Sequential(*layers)
        
        in_dim_1d = 2*self.n_harmonics + 1
        self.branch_1d = _make_branch(in_dim_1d, 3)
        
        in_dim_2d = 4*self.n_harmonics + 1
        self.branch_2d = _make_branch(in_dim_2d, 4)
        
        in_dim_3d = 6*self.n_harmonics + 1
        self.branch_3d = _make_branch(in_dim_3d, 5)
    
    def set_domain(self, rmin, rmax, dimension):
        if dimension >= 1:
            self.xmin, self.xmax = float(rmin[0]), float(rmax[0])
        if dimension >= 2:
            self.ymin, self.ymax = float(rmin[1]), float(rmax[1])
        if dimension >= 3:
            self.zmin, self.zmax = float(rmin[2]), float(rmax[2])
    
    def _periodic_features(self, u, umin, umax):
        L = umax - umin
        theta = 2*np.pi*(u - umin)/L
        features = []
        for k in range(1, self.n_harmonics+1):
            scale = 1.0 / np.sqrt(k)
            features.append(scale * torch.sin(k*theta))
            features.append(scale * torch.cos(k*theta))
        return torch.cat(features, dim=1) if features else u
    
    def forward(self, X):
        x, y, t = X[0], X[1], X[2]
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        y = y.unsqueeze(-1) if y.dim() == 1 else y
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        
        x_feat = self._periodic_features(x, self.xmin, self.xmax)
        y_feat = self._periodic_features(y, self.ymin, self.ymax)
        features = torch.cat([x_feat, y_feat, t], dim=1)
        
        outputs = self.branch_2d(features)
        
        # Apply density constraint (simplified)
        rho_hat = outputs[:, 0:1]
        other = outputs[:, 1:]
        STARTUP_DT = 0.01
        t_effective = torch.clamp(t - STARTUP_DT, min=0.0)
        rho = rho_o * torch.exp(torch.clamp(t_effective * rho_hat, min=-10, max=10))
        
        return torch.cat([rho, other], dim=1)


def draw_pinn_architecture(ax):
    """Draw a PINN architecture diagram with full connectivity."""
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    input_color = '#a8d5ba'
    hidden_color = '#e6e6fa'
    output_color = '#ffcccb'
    line_color = '#aaaaaa'
    arrow_color = '#555555'
    
    # Shift everything right by 0.8
    shift = 0.5
    
    # Input layer (x, y, t)
    input_labels = ['x', 'y', 't']
    input_y = [7, 5, 3]
    input_x = 1.0 + shift
    
    # Fourier features box position
    ff_left = 2.3 + shift
    ff_right = 3.7 + shift
    ff_center_x = (ff_left + ff_right) / 2
    ff_bottom = 2
    ff_top = 8
    ff_center_y = (ff_bottom + ff_top) / 2
    
    # Draw lines from inputs to Periodic Features box FIRST (so they're behind)
    for yy in input_y:
        ax.plot([input_x + 0.4, ff_left], [yy, ff_center_y], color=line_color, lw=1.2, zorder=1)
    
    # Draw input circles
    for i, (label, yy) in enumerate(zip(input_labels, input_y)):
        circle = Circle((input_x, yy), 0.4, facecolor=input_color, edgecolor='#2d5a3d', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(input_x, yy, label, ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)
    
    # Fourier features box - now showing the transformation
    ff_left = 2.3 + shift
    ff_right = 3.7 + shift
    ff_center_x = (ff_left + ff_right) / 2
    ff_bottom = 2
    ff_top = 8
    ff_center_y = (ff_bottom + ff_top) / 2
    
    # Draw outer box with lighter fill
    ff_box = FancyBboxPatch((ff_left, ff_bottom), ff_right - ff_left, ff_top - ff_bottom, 
                             boxstyle='round,pad=0.1', 
                             facecolor='#fffef0', edgecolor='#c9a227', linewidth=2, zorder=2)
    ax.add_patch(ff_box)
    
    # Title
    ax.text(ff_center_x, ff_top - 0.4, 'Periodic\nFeatures', ha='center', va='top', 
            fontsize=9, fontweight='bold', zorder=3)
    
    # Show representative feature examples stacked vertically
    features = [
        r'$\sin(\theta_x)$',
        r'$\cos(\theta_x)$',
        r'$\sin(\theta_y)$',
        r'$\cos(\theta_y)$',
        r'$\sin(2\theta_x)$',
        r'$\vdots$',
    ]
    
    # Position features vertically within the box
    y_start = ff_top - 1.35
    y_end = ff_bottom + 0.5
    n_features = len(features)
    spacing = (y_start - y_end) / (n_features - 1) if n_features > 1 else 0
    
    for i, feat in enumerate(features):
        y_pos = y_start - i * spacing
        ax.text(ff_center_x, y_pos, feat, ha='center', va='center', fontsize=9, zorder=3)
    
    # Hidden layers - shifted right
    hidden_x = [5.2 + shift, 6.4 + shift, 7.6 + shift]
    hidden_y_positions = [7.5, 6, 4.5, 3]
    
    # Draw connections: Periodic Features to first hidden layer
    for hy in hidden_y_positions:
        ax.plot([ff_right, hidden_x[0] - 0.3], [ff_center_y, hy], color=line_color, lw=0.8, zorder=1)
    
    # Draw connections between hidden layers
    for i in range(len(hidden_x) - 1):
        for hy1 in hidden_y_positions:
            for hy2 in hidden_y_positions:
                ax.plot([hidden_x[i] + 0.3, hidden_x[i+1] - 0.3], [hy1, hy2], 
                       color=line_color, lw=0.5, alpha=0.6, zorder=1)
    
    # Draw hidden layer circles
    for hx in hidden_x:
        for hy in hidden_y_positions:
            circle = Circle((hx, hy), 0.3, facecolor=hidden_color, edgecolor='#6a5acd', linewidth=1.5, zorder=2)
            ax.add_patch(circle)
    
    # Dots for more neurons
    ax.text(hidden_x[1], 1.8, '...', ha='center', va='center', fontsize=14, color='#666', zorder=3)
    
    # Output layer - shifted right
    output_labels = ['ρ', 'vₓ', 'vᵧ', 'φ']
    output_y = [7.5, 5.5, 3.5, 1.5]
    output_x = 9.8 + shift
    
    # Draw connections: last hidden to output
    for hy_hidden in hidden_y_positions:
        for hy_out in output_y:
            ax.plot([hidden_x[-1] + 0.3, output_x - 0.4], [hy_hidden, hy_out], 
                   color=line_color, lw=0.6, alpha=0.7, zorder=1)
    
    # Draw output circles
    for label, yy in zip(output_labels, output_y):
        circle = Circle((output_x, yy), 0.4, facecolor=output_color, edgecolor='#8b0000', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(output_x, yy, label, ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)
    
    # Labels
    ax.text(input_x, 9.0, 'Input', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(hidden_x[1], 9.0, 'Hidden Layers', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(output_x, 9.0, 'Output', ha='center', va='bottom', fontsize=10, fontweight='bold')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ========================================================================
    # Generate IC velocity field
    # ========================================================================
    print("Generating IC velocity field...")
    v_1 = a * cs
    vx_ic, vy_ic = generate_velocity_field_power_spectrum(
        N_GRID, N_GRID, Lx, Ly, 
        power_index=POWER_EXPONENT, 
        amplitude=v_1, 
        random_seed=RANDOM_SEED
    )
    
    x_grid = np.linspace(xmin, xmax, N_GRID, endpoint=False)
    y_grid = np.linspace(ymin, ymax, N_GRID, endpoint=False)
    
    # ========================================================================
    # Load model and generate predictions
    # ========================================================================
    print("Loading model...")
    model_path = r"C:\Users\tirth\OneDrive\Desktop\gravitational collapse results\2D power spectrum\model.pth"
    
    model = PINN(num_neurons=num_neurons, num_layers=num_layers, n_harmonics=harmonics)
    # The trained model is for DIMENSION=2 (2D spatial: x, y + time)
    model.set_domain([xmin, ymin], [xmax, ymax], dimension=2)  # spatial dimension = 2
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        model_loaded = True
    except Exception as e:
        print(f"Could not load model: {e}")
        model_loaded = False
    
    # Generate predictions at t=tmax
    if model_loaded:
        print(f"Generating predictions at t={tmax}...")
        n_pred = 200
        x_pred = np.linspace(xmin, xmax, n_pred)
        y_pred = np.linspace(ymin, ymax, n_pred)
        X_pred, Y_pred = np.meshgrid(x_pred, y_pred, indexing='ij')
        
        x_flat = torch.tensor(X_pred.flatten(), dtype=torch.float32, device=device)
        y_flat = torch.tensor(Y_pred.flatten(), dtype=torch.float32, device=device)
        t_flat = torch.full_like(x_flat, tmax)
        
        with torch.no_grad():
            outputs = model([x_flat, y_flat, t_flat])
            rho_pred = outputs[:, 0].cpu().numpy().reshape(n_pred, n_pred)
            vx_pred = outputs[:, 1].cpu().numpy().reshape(n_pred, n_pred)
            vy_pred = outputs[:, 2].cpu().numpy().reshape(n_pred, n_pred)
    
    # ========================================================================
    # Create composite figure
    # ========================================================================
    print("Creating composite figure...")
    
    fig = plt.figure(figsize=(16, 5.5))
    # Adjust ratios: account for colorbars in left and right columns
    gs = gridspec.GridSpec(2, 3, width_ratios=[0.90, 1.85, 0.75], 
                           height_ratios=[1, 1], hspace=0.30, wspace=0.12)
    
    # Left column: IC plots
    ax_ic_vx = fig.add_subplot(gs[0, 0])
    ax_ic_vy = fig.add_subplot(gs[1, 0])
    
    # Center: PINN architecture
    ax_pinn = fig.add_subplot(gs[:, 1])
    
    # Right column: Output plots
    ax_out_rho = fig.add_subplot(gs[0, 2])
    ax_out_v = fig.add_subplot(gs[1, 2])
    
    # ---- IC plots ----
    im_vx = ax_ic_vx.pcolormesh(x_grid, y_grid, vx_ic.T, shading='auto', cmap='viridis')
    ax_ic_vx.set_title(r'$v_x(\mathbf{x}, t=0)$', fontsize=11)
    ax_ic_vx.set_xlabel('x')
    ax_ic_vx.set_ylabel('y')
    ax_ic_vx.set_aspect('equal')
    plt.colorbar(im_vx, ax=ax_ic_vx, fraction=0.040, pad=0.02)
    
    im_vy = ax_ic_vy.pcolormesh(x_grid, y_grid, vy_ic.T, shading='auto', cmap='viridis')
    ax_ic_vy.set_title(r'$v_y(\mathbf{x}, t=0)$', fontsize=11)
    ax_ic_vy.set_xlabel('x')
    ax_ic_vy.set_ylabel('y')
    ax_ic_vy.set_aspect('equal')
    plt.colorbar(im_vy, ax=ax_ic_vy, fraction=0.040, pad=0.02)
    
    # ---- PINN architecture ----
    draw_pinn_architecture(ax_pinn)
    
    # ---- Output plots ----
    if model_loaded:
        im_rho = ax_out_rho.pcolormesh(x_pred, y_pred, rho_pred.T, shading='auto', cmap='YlOrBr')
        ax_out_rho.set_title(rf'$\rho(\mathbf{{x}}, t={tmax})$', fontsize=11)
        ax_out_rho.set_xlabel('x')
        ax_out_rho.set_ylabel('y')
        ax_out_rho.set_aspect('equal')
        plt.colorbar(im_rho, ax=ax_out_rho, fraction=0.040, pad=0.02)
        
        # Velocity magnitude
        v_mag = np.sqrt(vx_pred**2 + vy_pred**2)
        im_v = ax_out_v.pcolormesh(x_pred, y_pred, v_mag.T, shading='auto', cmap='viridis')
        ax_out_v.set_title(rf'$|\mathbf{{v}}|(\mathbf{{x}}, t={tmax})$', fontsize=11)
        ax_out_v.set_xlabel('x')
        ax_out_v.set_ylabel('y')
        ax_out_v.set_aspect('equal')
        plt.colorbar(im_v, ax=ax_out_v, fraction=0.040, pad=0.02)
    else:
        ax_out_rho.text(0.5, 0.5, 'Model not loaded', ha='center', va='center', transform=ax_out_rho.transAxes)
        ax_out_v.text(0.5, 0.5, 'Model not loaded', ha='center', va='center', transform=ax_out_v.transAxes)
    
    # Add section labels - aligned at same y position
    title_y = 0.96
    
    # Fine-tuned positions for better alignment
    left_center = 0.20    # Initial Conditions
    middle_center = 0.56  # PINN diagram (center)
    right_center = 0.88   # PINN Predictions
    
    fig.text(left_center, title_y, 'Initial Conditions', ha='center', fontsize=13, fontweight='bold')
    fig.text(middle_center, title_y, 'Physics-Informed Neural Network', ha='center', fontsize=13, fontweight='bold')
    fig.text(right_center, title_y, 'PINN Predictions', ha='center', fontsize=13, fontweight='bold')
    
    # Add vertical dotted lines as separators - positioned in the gaps
    # Line between IC and PINN (in the gap)
    line1_x = 0.315
    fig.add_artist(plt.Line2D([line1_x, line1_x], [0.01, 0.95], 
                              transform=fig.transFigure, color='gray', 
                              linestyle=':', linewidth=2.0, alpha=0.7))
    
    # Line between PINN and Predictions (in the gap)
    line2_x = 0.755
    fig.add_artist(plt.Line2D([line2_x, line2_x], [0.01, 0.95], 
                              transform=fig.transFigure, color='gray', 
                              linestyle=':', linewidth=2.0, alpha=0.7))
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.08)
    
    output_path = 'workflow_composite.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
