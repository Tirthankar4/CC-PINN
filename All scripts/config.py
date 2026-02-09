import numpy as np

# Random seed for reproducibility across all functions
RANDOM_SEED = 93

# Perturbation selection: "power_spectrum" or "sinusoidal"
PERTURBATION_TYPE = "power_spectrum"

xmin = 0.0
ymin = 0.0
zmin = 0.0
cs = 1.0
rho_o = 1.0
const = 1.0
G = 1.0

# Collocation point parameters
N_0 = 25000  # Number of initial condition points
N_r = 50000 # Number of residual/collocation points
DIMENSION = 2  # Spatial dimension

BATCH_SIZE = 75000  # Max collocation points per mini-batch (controls peak GPU memory)
NUM_BATCHES = 1  # Mini-batches per optimizer step; gradients accumulate, memory stays constant

a = 1.3

tmin = 0.
tmax = 3.0

num_neurons = 64
harmonics = 3
num_layers = 5
DEFAULT_ACTIVATION = 'sin'  # Default activation function for PINN ('sin', 'tanh', 'relu', etc.)

wave = 7.0
k = 2 * np.pi / wave
num_of_waves = 2.0

iteration_adam_2D = 1001
iteration_lbgfs_2D = 201

IC_WEIGHT = 1.0

# Extra collocation points at t=0 to strongly enforce Poisson equation
N_POISSON_IC = 0  # Number of extra spatial points at t=0 for Poisson
POISSON_IC_WEIGHT = 0.0  # Weight for Poisson residual at t=0
PHI_MEAN_CONSTRAINT_WEIGHT = 0.0  # Weight for mean(φ)=0 constraint at t=0 (fixes gauge freedom)

# Output/snapshot controls
SAVE_STATIC_SNAPSHOTS = False

# Directory to save snapshots; default keeps Kaggle working dir
SNAPSHOT_DIR = "/kaggle/working/"

# Training diagnostics
ENABLE_TRAINING_DIAGNOSTICS = True  # Enable automatic training diagnostics plots and logging

# Density growth comparison plot controls
PLOT_DENSITY_GROWTH = False
GROWTH_PLOT_TMAX = 4.0
GROWTH_PLOT_DT = 0.1

KX = k
KY = 0
KZ = 0
TIMES_1D = [2.5, 3.0, 3.5] # 1D cross-section times to plot (used for sinusoidal panel plots
FD_N_1D = 300  # Grid points for 1D LAX (when used)
FD_N_2D = 300  # Grid points per dimension for 2D LAX
FD_N_3D = 300  # Grid points per dimension for 3D LAX slices
FD_NU_SINUSOIDAL = 0.5  # Courant number for LAX solver
SLICE_Y = 0.6  # Default y slice for visualization/cross-sections
SLICE_Z = 0.6  # Default z slice for visualization/cross-sections
SHOW_LINEAR_THEORY = False

# 3D Interactive plot settings
ENABLE_INTERACTIVE_3D = True  # Toggle interactive 3D plot generation
INTERACTIVE_3D_RESOLUTION = 50  # Grid resolution per axis
INTERACTIVE_3D_TIME_STEPS = 10  # Number of time steps in slider

# Power spectrum parameters
N_GRID = 400  # Grid resolution for power spectrum generation
N_GRID_3D = 400 #In 3D
FD_NU_POWER = 0.25  # Courant number for LAX solver
POWER_EXPONENT = -4  # Power spectrum exponent
STARTUP_DT = 0.01 # Time offset after which PDE is enforced (ICs remain at t=0)
USE_PARAMETERIZATION = "exponential"  # Options: "exponential", "linear", "none"