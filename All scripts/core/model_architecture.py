import numpy as np

import torch
import torch.nn as nn
from config import (
    rho_o,
    num_neurons,
    num_layers,
    PERTURBATION_TYPE,
    DEFAULT_ACTIVATION,
    STARTUP_DT,
    USE_PARAMETERIZATION,
    GRAVITY,
    USE_TEMPORAL_REPARAM,
    JEANS_GROWTH_RATE,
)

class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)


def get_activation(activation_type):
    """
    Factory function to create activation function instances.
    
    Args:
        activation_type: String identifier ('sin', 'tanh', 'relu', 'elu')
    
    Returns:
        nn.Module activation function
    """
    activation_type = activation_type.lower()
    if activation_type == 'sin':
        return Sin()
    elif activation_type == 'tanh':
        return nn.Tanh()
    elif activation_type == 'relu':
        return nn.ReLU()
    elif activation_type == 'elu':
        return nn.ELU()
    else:
        raise ValueError(f"Unknown activation type: {activation_type}. Choose from 'sin', 'tanh', 'relu', 'elu'.")


def _temporal_reparam(t, omega):
    """
    Remap raw time t → τ = (exp(omega * t) - 1) / omega.

    Properties:
        τ(0)     = 0               — IC behaviour unchanged
        dτ/dt|0  = 1               — unit slope at t=0, smooth handoff
        τ grows  ~ exp(omega * t)  — dominant Jeans mode is LINEAR in τ

    The growing mode ρ ~ exp(omega * t) becomes ρ ~ omega * τ + 1,
    which is trivially representable by the network.  The low-lying
    (near-constant) solution becomes τ-dependent in a non-trivial way,
    partially levelling the playing field between the two attractors.

    Args:
        t:     raw time tensor  [N, 1]
        omega: Jeans growth rate (scalar float, > 0)

    Returns:
        tau:   reparameterized time tensor [N, 1], same dtype/device as t
    """
    # τ = (exp(ω*t) - 1) / (exp(ω*tmax) - 1) * tmax
    # This normalizes τ so that τ(tmax) = tmax, keeping the same numerical
    # scale as raw t regardless of ω or tmax.  The growing mode is still
    # linear in τ; we've just rescaled τ so the network input stays O(tmax)
    # instead of growing to exp(ω*tmax) which causes large-input pathologies
    # with sin activations and Xavier initialization.
    tau_raw = (torch.exp(omega * t) - 1.0) / omega
    # Scale factor: tau_raw(tmax) / tmax  (computed from config tmax)
    from config import tmax as TMAX_CFG
    import math
    tau_max_val = (math.exp(omega * float(TMAX_CFG)) - 1.0) / omega
    scale = tau_max_val / float(TMAX_CFG) if float(TMAX_CFG) > 0 else 1.0
    return tau_raw / scale


class PINN(nn.Module):
    def __init__(self, dimension, num_neurons=num_neurons, num_layers=num_layers, n_harmonics=1, activation_type=DEFAULT_ACTIVATION):
        super(PINN, self).__init__()
        self.dimension = dimension
        self.num_neurons = num_neurons
        self.n_harmonics = n_harmonics
        self.num_layers = max(2, int(num_layers))  # total Linear layers including output
        self.activation_type = activation_type
        
        # Domain extents for periodic embeddings (set via set_domain)
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.zmin = None
        self.zmax = None
    
    # Helper to build a branch with dynamic depth
        def _make_branch(in_dim, out_dim):
            layers = []
            # First layer
            layers.append(nn.Linear(in_dim, self.num_neurons))
            # Hidden layers: total linear layers = self.num_layers; we already added 1; 
            # add (self.num_layers - 2) hidden Linear blocks with activations after each
            for _ in range(self.num_layers - 2):
                layers.append(get_activation(self.activation_type))
                layers.append(nn.Linear(self.num_neurons, self.num_neurons))
            # Activation before output if there is at least one hidden block
            if self.num_layers > 2:
                layers.append(get_activation(self.activation_type))
            # Output layer
            layers.append(nn.Linear(self.num_neurons, out_dim))
            return nn.Sequential(*layers)

    # Build only the branch matching the dimension
        if dimension == 1:
            # 1D: periodic x features + t → [rho, vx, phi]
            in_dim = 2*self.n_harmonics + 1
            out_dim = 3 if GRAVITY else 2
        elif dimension == 2:
            # 2D: periodic x,y features + t → [rho, vx, vy, phi]
            in_dim = 4*self.n_harmonics + 1
            out_dim = 4 if GRAVITY else 3
        elif dimension == 3:
            # 3D: periodic x,y,z features + t → [rho, vx, vy, vz, phi]
            in_dim = 6*self.n_harmonics + 1
            out_dim = 5 if GRAVITY else 4
        else:
            raise ValueError(f"Invalid dimension: {dimension}. Expected 1, 2, or 3.")
        
        self.branch = _make_branch(in_dim, out_dim)

    # Apply Xavier uniform initialization to all Linear layers
        self.apply(PINN._init_weights)

    @staticmethod
    def _init_weights(m):
        """Apply Xavier uniform initialization to Linear layers."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def set_domain(self, rmin, rmax, dimension):
        # rmin/rmax exclude time; follow ASTPN usage
        if dimension >= 1:
            self.xmin, self.xmax = float(rmin[0]), float(rmax[0])
        if dimension >= 2:
            self.ymin, self.ymax = float(rmin[1]), float(rmax[1])
        if dimension >= 3:
            self.zmin, self.zmax = float(rmin[2]), float(rmax[2])

    def _periodic_features(self, u, umin, umax):
        # u is [N,1]
        L = umax - umin
        theta = 2*np.pi*(u - umin)/L
        features = []

        for k in range(1, self.n_harmonics+1):
            
            scale = 1.0 / np.sqrt(k)

            features.append(scale * torch.sin(k*theta))
            features.append(scale * torch.cos(k*theta))

        return torch.cat(features, dim=1) if len(features) > 0 else u

    def _prepare_coordinate_features(self, X):
        """
        Prepare periodic features for all spatial coordinates.

        When USE_TEMPORAL_REPARAM is True, raw t is mapped to
            τ = (exp(omega_J * t) - 1) / omega_J
        before being concatenated into the feature vector.  The raw t
        tensor is kept separately and returned unchanged so that
        _apply_density_constraint can use it for the STARTUP_DT causal
        mask (which must stay in physical time).

        Args:
            X: List of coordinates [x, ...spatial..., t]
        
        Returns:
            Tuple (features, t_raw, dimension)
              features : tensor fed into self.branch  [N, in_dim]
              t_raw    : raw physical time            [N, 1]  (always)
              dimension: len(X)
        """
        x, t = X[0], X[-1]
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        dimension = len(X)

        # Reparameterize time if requested.
        # t_feat is what enters the network; t stays as raw physical time
        # for the density constraint causal mask.
        if USE_TEMPORAL_REPARAM and JEANS_GROWTH_RATE > 0.0:
            t_feat = _temporal_reparam(t, JEANS_GROWTH_RATE)
        else:
            t_feat = t
        
        if dimension == 2:
            if self.xmin is None or self.xmax is None:
                raise RuntimeError("Domain not set: call net.set_domain for dimension=1")
            x_feat = self._periodic_features(x, self.xmin, self.xmax)
            features = torch.cat([x_feat, t_feat], dim=1)
        
        elif dimension == 3:
            if self.xmin is None or self.xmax is None or self.ymin is None or self.ymax is None:
                raise RuntimeError("Domain not set: call net.set_domain for dimension=2")
            y = X[1].unsqueeze(-1) if X[1].dim() == 1 else X[1]
            x_feat = self._periodic_features(x, self.xmin, self.xmax)
            y_feat = self._periodic_features(y, self.ymin, self.ymax)
            features = torch.cat([x_feat, y_feat, t_feat], dim=1)
        
        elif dimension == 4:
            if (self.xmin is None or self.xmax is None or
                self.ymin is None or self.ymax is None or
                self.zmin is None or self.zmax is None):
                raise RuntimeError("Domain not set: call net.set_domain for dimension=3")
            y = X[1].unsqueeze(-1) if X[1].dim() == 1 else X[1]
            z = X[2].unsqueeze(-1) if X[2].dim() == 1 else X[2]
            x_feat = self._periodic_features(x, self.xmin, self.xmax)
            y_feat = self._periodic_features(y, self.ymin, self.ymax)
            z_feat = self._periodic_features(z, self.zmin, self.zmax)
            features = torch.cat([x_feat, y_feat, z_feat, t_feat], dim=1)
        
        else:
            raise ValueError(f"Expected len(X) in [2, 3, 4] but got {dimension}")
        
        return features, t, dimension
    
    def _apply_density_constraint(self, outputs, t):
        """
        Apply hard density constraint with causality enforcement for power spectrum perturbations.
        
        For power spectrum (non-sinusoidal):
        - For t < STARTUP_DT: Density is frozen at ρ₀ (causality - information hasn't propagated)
        - For t >= STARTUP_DT: Density evolves based on USE_PARAMETERIZATION:
          * "exponential": ρ = ρ₀ × exp(clamp(tau_eff × ρ̂, -10, 10))
            where tau_eff = τ(t_eff) when USE_TEMPORAL_REPARAM is True,
            or t_eff when False.  Using τ here keeps the density
            parameterization consistent with what the network was trained
            against — the network's density head outputs ρ̂ in τ-space,
            so multiplying by τ rather than t gives the correct scale.
          * "linear": ρ = ρ₀ + tau_eff × ρ̂  (same τ-consistency logic)
          * "none": ρ = ρ̂
        
        Args:
            outputs: Raw network outputs [ρ̂, vx, vy?, vz?, phi]
            t: Raw physical time tensor [N, 1]  (always physical time,
               NOT reparameterized — causal mask must live in real time)
        
        Returns:
            Modified outputs with density constraint applied [ρ, vx, vy?, vz?, phi]
        """
        if str(PERTURBATION_TYPE).lower() == "sinusoidal":
            return outputs
        
        rho_hat = outputs[:, 0:1]
        other   = outputs[:, 1:]
        
        # Causal mask always in physical time
        causal_mask = (t >= STARTUP_DT).float()
        t_effective = torch.clamp(t - STARTUP_DT, min=0.0)

        # The density parameterization always uses raw t_effective regardless
        # of whether temporal reparameterization is active.
        #
        # The reparam is an INPUT-side change only — it reshapes what the
        # network sees, not how the output is interpreted.
        #
        # The product clamp (min=-10, max=10) was designed for small rho_hat
        # (perturbation amplitudes ~ 0.01).  With sin activations and Xavier
        # init the network freely outputs rho_hat ~ O(1-3), so at tmax=5:
        #   t_eff * rho_hat ~ 5 * 3 = 15  →  clamp kicks in at 10
        #   exp(10) ~ 22000  →  catastrophic initial loss
        #
        # Fix: clamp rho_hat to [-2, 2] before the product so that
        #   t_eff * rho_hat <= 5 * 2 = 10  →  exp(10) ~ 22000  still bad
        # Use tighter clamp of 5 on the product instead of 10:
        #   exp(5) ~ 148  →  rho at most ~148 at init, quickly learned away.
        parameterization = str(USE_PARAMETERIZATION).lower()
        
        if parameterization == "exponential":
            rho = rho_o * torch.exp(torch.clamp(t_effective * rho_hat, min=-10, max=10))
            
        elif parameterization == "linear":
            rho = rho_o + t_effective * rho_hat
            
        elif parameterization == "none":
            rho = causal_mask * rho_hat + (1 - causal_mask) * rho_o
            
        else:
            raise ValueError(f"Invalid USE_PARAMETERIZATION: '{USE_PARAMETERIZATION}'. "
                           f"Choose from: 'exponential', 'linear', 'none'")
        
        return torch.cat([rho, other], dim=1)
    
    def forward(self, X):
        """
        Forward pass of PINN.
        
        Args:
            X: List of coordinates [x, ...spatial..., t]
        
        Returns:
            Network predictions [rho, vx, vy?, vz?, phi]
        """
        features, t, dimension = self._prepare_coordinate_features(X)
        
        # Verify dimension matches construction
        if dimension != len(X):
            raise ValueError(f"Input dimension mismatch: expected {self.dimension + 1} coordinates, got {len(X)}")
        
        # Forward pass through the single branch
        outputs = self.branch(features)
        
        return self._apply_density_constraint(outputs, t)
