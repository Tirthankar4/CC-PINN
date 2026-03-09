"""
Configuration system for GRINN (Gravity Informed Neural Network).

Uses @dataclass(frozen=True) for type-safe, immutable configuration.
Supports YAML serialization for reproducible experiment tracking.

Usage (new style — preferred):
    from config import CONFIG
    print(CONFIG.physics.cs)
    CONFIG.to_yaml("my_run_config.yaml")

Usage (legacy — still works):
    from config import cs, rho_o, DIMENSION, ...
"""

import math
from dataclasses import dataclass, field, asdict, fields
from typing import Optional, Tuple

import yaml

# ═══════════════════════════════════════════════════════════════
#  Sub-configuration dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PhysicsConfig:
    """Physical constants for Jeans instability simulation."""
    cs: float = 1.0            # Sound speed
    rho_o: float = 1.0         # Background density
    const: float = 1.0         # Poisson coupling constant (4πG)
    G: float = 1.0             # Gravitational constant
    a: float = 0.3             # Perturbation amplitude
    gravity: bool = True       # Enable/disable self-gravity coupling


@dataclass(frozen=True)
class DomainConfig:
    """Spatial/temporal domain and wave parameters."""
    dimension: int = 2         # Spatial dimension (1, 2, or 3)
    xmin: float = 0.0
    ymin: float = 0.0
    zmin: float = 0.0
    tmin: float = 0.0
    tmax: float = 4.5
    wave: float = 7.0          # Wavelength
    num_of_waves: float = 2.0  # Number of wavelengths in domain

    @property
    def k(self) -> float:
        """Wavenumber: 2π / wavelength."""
        return 2 * math.pi / self.wave


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters and collocation point settings."""
    N_0: int = 15000                       # Initial condition points
    N_r: int = 70000                       # Residual / collocation points
    batch_size: int = 85000                # Max points per mini-batch
    num_batches: int = 1                   # Mini-batches per optimizer step
    iteration_adam: int = 2001             # Adam iterations
    iteration_lbfgs: int = 201             # L-BFGS iterations
    ic_weight: float = 1.0                # IC loss weight
    # Causality
    startup_dt: float = 0.0              # PDE enforcement delay


@dataclass(frozen=True)
class ModelConfig:
    """Neural network architecture parameters."""
    num_neurons: int = 64
    num_layers: int = 5
    harmonics: int = 3                     # Fourier feature harmonics
    activation: str = "sin"                # Activation function
    use_parameterization: str = "exponential"  # Density parameterization


@dataclass(frozen=True)
class VisualizationConfig:
    """Visualization and output settings (perturbation-agnostic)."""
    save_static_snapshots: bool = True
    output_dir: str = "/kaggle/working/"     # Base directory for all run outputs
    enable_training_diagnostics: bool = True
    plot_density_growth: bool = False
    growth_plot_tmax: float = 4.0
    growth_plot_dt: float = 0.1
    # Spatial slice positions for cross-sections / 3D visualization
    slice_y: float = 0.6
    slice_z: float = 0.6
    # 3D interactive plot
    enable_interactive_3d: bool = True
    interactive_3d_resolution: int = 50
    interactive_3d_time_steps: int = 10


@dataclass(frozen=True)
class InitialConditionConfig:
    """Initial condition mode selection."""
    mode: str = "sinusoidal"  # "sinusoidal" or "power_spectrum"


@dataclass(frozen=True)
class SinusoidalConfig:
    """Sinusoidal perturbation parameters: wave vector, FD grids, cross-sections."""
    # Wave-vector components (KX defaults to domain.k when None)
    KX: Optional[float] = None
    KY: float = 0.0
    KZ: float = 0.0
    # 1D cross-section settings
    times_1d: Tuple[float, ...] = (1.0, 2.0, 3.0)
    fd_n_1d: int = 300
    fd_n_2d: int = 1000
    fd_n_3d: int = 300
    fd_nu: float = 0.1            # Courant number for LAX solver
    show_linear_theory: bool = False


@dataclass(frozen=True)
class PowerSpectrumConfig:
    """Power spectrum IC generation parameters."""
    n_grid: int = 400
    n_grid_3d: int = 400
    fd_nu_power: float = 0.25
    power_exponent: int = -4
    random_seed: int = 9        # Seed for IC field generation (big impact on field morphology)


@dataclass(frozen=True)
class AdaptiveCollocationConfig:
    """Adaptive collocation resampling parameters."""
    enabled: bool = True
    resample_every_n: int = 50       # Resample every N Adam iterations
    n_candidates: int = 5000         # Candidate points generated at each resample
    keep_fraction: float = 0.5       # Fraction of pool kept (highest-residual points)
    uniform_fraction: float = 0.5    # Fraction replaced with fresh uniform points


# ═══════════════════════════════════════════════════════════════
#  Top-level configuration
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SimulationConfig:
    """
    Top-level configuration for a GRINN simulation run.

    All sub-configs are frozen (immutable) after creation.
    Create custom configs by passing overrides to the constructor::

        cfg = SimulationConfig(
            physics=PhysicsConfig(cs=2.0),
            domain=DomainConfig(tmax=5.0, dimension=3),
        )

    Or load from a saved YAML::

        cfg = SimulationConfig.from_yaml("runs/run_42/config.yaml")
    """
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    initial_condition: InitialConditionConfig = field(default_factory=InitialConditionConfig)
    sinusoidal: SinusoidalConfig = field(default_factory=SinusoidalConfig)
    power_spectrum: PowerSpectrumConfig = field(default_factory=PowerSpectrumConfig)
    adaptive_collocation: AdaptiveCollocationConfig = field(default_factory=AdaptiveCollocationConfig)

    # ── Derived / resolved values ──────────────────────────────

    @property
    def perturbation_type(self) -> str:
        """Backward-compatible alias for selected IC mode."""
        return self.initial_condition.mode

    @property
    def random_seed(self) -> int:
        """Convenience accessor — delegates to power_spectrum.random_seed."""
        return self.power_spectrum.random_seed

    @property
    def KX(self) -> float:
        """Resolved KX: uses sinusoidal.KX if explicitly set, else domain.k."""
        if self.sinusoidal.KX is not None:
            return self.sinusoidal.KX
        return self.domain.k

    # ── Serialization ──────────────────────────────────────────

    def to_dict(self) -> dict:
        """Convert to nested dict for serialization."""
        d = asdict(self)
        # Tuples → lists for human-readable YAML
        if "sinusoidal" in d and "times_1d" in d["sinusoidal"]:
            d["sinusoidal"]["times_1d"] = list(d["sinusoidal"]["times_1d"])
        return d

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationConfig":
        """Construct a SimulationConfig from a (possibly partial) nested dict."""
        # Backward compat: migrate top-level random_seed → power_spectrum.random_seed
        if "random_seed" in d and "power_spectrum" not in d:
            d.setdefault("power_spectrum", {})["random_seed"] = d.pop("random_seed")
        elif "random_seed" in d:
            d.setdefault("power_spectrum", {}).setdefault("random_seed", d.pop("random_seed"))

        # Backward compat: migrate top-level perturbation_type → initial_condition.mode
        if "perturbation_type" in d and "initial_condition" not in d:
            d.setdefault("initial_condition", {})["mode"] = d.pop("perturbation_type")
        elif "perturbation_type" in d:
            d.setdefault("initial_condition", {}).setdefault("mode", d.pop("perturbation_type"))

        sub_configs = {}
        for f in fields(cls):
            if f.name not in d:
                continue  # use dataclass default
            if hasattr(f.type, "__dataclass_fields__"):
                sub_dict = dict(d[f.name])
                # Filter to known fields only (ignore stale / extra keys)
                known = {sf.name for sf in fields(f.type)}
                sub_dict = {k: v for k, v in sub_dict.items() if k in known}
                # Lists → tuples where needed
                if f.name == "sinusoidal" and "times_1d" in sub_dict:
                    sub_dict["times_1d"] = tuple(sub_dict["times_1d"])
                sub_configs[f.name] = f.type(**sub_dict)
            else:
                sub_configs[f.name] = d[f.name]
        return cls(**sub_configs)

    @classmethod
    def from_yaml(cls, path: str) -> "SimulationConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


# ═══════════════════════════════════════════════════════════════
#  Singleton instance (used by backward-compatible aliases below)
# ═══════════════════════════════════════════════════════════════

CONFIG = SimulationConfig()


# ═══════════════════════════════════════════════════════════════
#  Backward-compatible module-level aliases
#
#  Every existing  `from config import X`  continues to work.
#  These are plain values (not live references), matching the
#  original behaviour of module-level constants.
# ═══════════════════════════════════════════════════════════════

# --- Random seed & perturbation type ---
RANDOM_SEED = CONFIG.random_seed
PERTURBATION_TYPE = CONFIG.perturbation_type

# --- Physics ---
cs    = CONFIG.physics.cs
rho_o = CONFIG.physics.rho_o
const = CONFIG.physics.const
G     = CONFIG.physics.G
a     = CONFIG.physics.a
GRAVITY = CONFIG.physics.gravity

# --- Domain ---
DIMENSION    = CONFIG.domain.dimension
xmin         = CONFIG.domain.xmin
ymin         = CONFIG.domain.ymin
zmin         = CONFIG.domain.zmin
tmin         = CONFIG.domain.tmin
tmax         = CONFIG.domain.tmax
wave         = CONFIG.domain.wave
num_of_waves = CONFIG.domain.num_of_waves
k            = CONFIG.domain.k          # derived: 2π / wave

# --- Training ---
N_0                       = CONFIG.training.N_0
N_r                       = CONFIG.training.N_r
BATCH_SIZE                = CONFIG.training.batch_size
NUM_BATCHES               = CONFIG.training.num_batches
iteration_adam_2D         = CONFIG.training.iteration_adam
iteration_lbgfs_2D       = CONFIG.training.iteration_lbfgs
IC_WEIGHT                 = CONFIG.training.ic_weight
STARTUP_DT                = CONFIG.training.startup_dt

# --- Model ---
num_neurons          = CONFIG.model.num_neurons
num_layers           = CONFIG.model.num_layers
harmonics            = CONFIG.model.harmonics
DEFAULT_ACTIVATION   = CONFIG.model.activation
USE_PARAMETERIZATION = CONFIG.model.use_parameterization

# --- Visualization / output ---
SAVE_STATIC_SNAPSHOTS     = CONFIG.visualization.save_static_snapshots
SNAPSHOT_DIR              = CONFIG.visualization.output_dir  # legacy alias
ENABLE_TRAINING_DIAGNOSTICS = CONFIG.visualization.enable_training_diagnostics
PLOT_DENSITY_GROWTH       = CONFIG.visualization.plot_density_growth
GROWTH_PLOT_TMAX          = CONFIG.visualization.growth_plot_tmax
GROWTH_PLOT_DT            = CONFIG.visualization.growth_plot_dt
SLICE_Y                   = CONFIG.visualization.slice_y
SLICE_Z                   = CONFIG.visualization.slice_z
ENABLE_INTERACTIVE_3D     = CONFIG.visualization.enable_interactive_3d
INTERACTIVE_3D_RESOLUTION = CONFIG.visualization.interactive_3d_resolution
INTERACTIVE_3D_TIME_STEPS = CONFIG.visualization.interactive_3d_time_steps

# --- Sinusoidal perturbation ---
KX                        = CONFIG.KX     # resolved (falls back to domain.k)
KY                        = CONFIG.sinusoidal.KY
KZ                        = CONFIG.sinusoidal.KZ
TIMES_1D                  = list(CONFIG.sinusoidal.times_1d)
FD_N_1D                   = CONFIG.sinusoidal.fd_n_1d
FD_N_2D                   = CONFIG.sinusoidal.fd_n_2d
FD_N_3D                   = CONFIG.sinusoidal.fd_n_3d
FD_NU_SINUSOIDAL          = CONFIG.sinusoidal.fd_nu
SHOW_LINEAR_THEORY        = CONFIG.sinusoidal.show_linear_theory

# --- Power spectrum ---
N_GRID         = CONFIG.power_spectrum.n_grid
N_GRID_3D      = CONFIG.power_spectrum.n_grid_3d
FD_NU_POWER    = CONFIG.power_spectrum.fd_nu_power
POWER_EXPONENT = CONFIG.power_spectrum.power_exponent

# --- Adaptive collocation ---
ADAPTIVE_COLLOCATION = CONFIG.adaptive_collocation