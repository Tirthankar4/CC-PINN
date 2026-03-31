# GRINN — Simplicity Bias: Experiments and Results

**Project:** Physics-Informed Neural Networks (PINNs) for self-gravitating molecular cloud collapse  
**System:** 2D compressible Euler equations with self-gravity (Jeans instability)  
**IC type:** Power spectrum velocity perturbations, uniform background density  
**Codebase:** GRINN (Gravitational Instability Neural Network)

---

## Physical Setup

GRINN enforces three PDEs via loss minimization:

$$\partial_t \rho + \nabla \cdot (\rho \mathbf{v}) = 0$$
$$\rho (\partial_t \mathbf{v} + \mathbf{v} \cdot \nabla \mathbf{v}) + c_s^2 \nabla \rho + \rho \nabla \phi = 0$$
$$\nabla^2 \phi = 4\pi G (\rho - \rho_0)$$

Default parameters: $c_s = \rho_0 = G = 1$, domain $L = 14$ (`wave=7`, `num_of_waves=2`). The Jeans wavenumber is $k_J = 1$; all modes $k < 1$ are unstable. The dominant unstable mode sits at $k_\text{min} = 2\pi/L \approx 0.449$ with growth rate:

$$\omega_J = \sqrt{1 - k_\text{min}^2} \approx 0.893$$

Training uses ~1500 Adam iterations followed by ~200 L-BFGS iterations.

---

## The Problem: Simplicity Bias at Large $t_\text{max}$

When the temporal domain exceeds a threshold, GRINN consistently learns the **wrong solution**: gas rotates weakly with negligible density growth, rather than undergoing gravitational collapse. This behaviour is reproducible across hundreds of configurations spanning perturbation strengths, random seeds, and architecture variants.

### The Two Competing Solutions

The PINN reliably converges to one of exactly two solutions:

1. **Correct — exponentially growing:** Gas accelerates under gravity into overdense regions; density grows as $\rho \sim e^{\omega_J t}$. Velocity field shows converging infall.
2. **Wrong — low-lying rotating:** Density barely changes; velocity field shows a persistent vortex-like rotating pattern.

### Why the Wrong Solution Wins

The PDE residuals of the low-lying rotating solution are **smaller in integrated magnitude** across a large temporal domain than those of the exponentially growing solution. The optimizer finds the wrong solution not because it is a local minimum — it is the **global minimum** of the PDE loss as currently formulated at large $t_\text{max}$. This is a loss landscape geometry problem, not a capacity or initialization problem.

### Key Diagnostics

**Linear regime test:** Even at vanishingly small perturbation amplitudes — where the solution is guaranteed to be fully linear throughout the entire domain — the PINN still fails for large $t_\text{max}$. Nonlinearity is ruled out as a cause. The failure is a pure competition between two linear solutions.

**Adam/LBFGS split:** Adam makes negligible density growth progress in all runs. All meaningful density growth happens during LBFGS — when it succeeds (small $t_\text{max}$), density grows rapidly during LBFGS; when it fails (large $t_\text{max}$), LBFGS shows the same negligible progress as Adam. The network is not stalling — it is actively optimizing into the wrong attractor.

**Training time heuristic:** Runs converging to the correct solution take ~15 minutes on the Kaggle P100 GPU; runs falling into the wrong solution take ~6–7 minutes. This was consistent enough to use as an early-stopping signal.

---

## Experiments

### 1. Adaptive Collocation

**Hypothesis:** Concentrating collocation points in high-residual regions improves the gradient signal toward the growing solution.

**Method:** Retain top 80% highest-residual collocation points; replace bottom 20% with fresh uniform samples every 50 Adam iterations.

**Result:** Negligible impact. The failure threshold in $t_\text{max}$ is identical with and without adaptive collocation.

**Why it failed:** When converging to the wrong solution, the adaptive method concentrates points where the *wrong solution* has high residual — shaping the training distribution to make the wrong basin easier to satisfy, not harder. The method functions correctly but is directed by the wrong signal.

---

### 2. PDE Time Weighting

**Hypothesis:** Late-time residuals from the growing mode are too small relative to those from the low-lying mode. Exponentially upweighting late times restores the balance.

**Method:** Weight each squared residual by $e^{\alpha t}$ before averaging. Tested $\alpha = 0.3$ as the primary value; also tested other nonzero values in both directions.

**Result:** No improvement at any tested $\alpha$ value.

**Why it failed:** The failure is not a loss magnitude imbalance problem. The optimizer cannot find a descent direction toward the growing solution basin regardless of how the existing residuals are weighted — reweighting does not change basin geometry.

---

### 3. Continuation Test

**Purpose:** Distinguish between two hypotheses about the failure mechanism:
- **Hypothesis A (initialization):** The growing-solution basin is simply unreachable from random initialization at large $t_\text{max}$.
- **Hypothesis B (landscape):** The wrong basin is genuinely dominant — it wins even from a good initialization.

**Method:** Take a successfully trained small-$T$ model ($t_\text{max} = 3$, confirmed correct solution via Athena++ comparison). Load its weights as initialization, regenerate collocation points over $[0, 5]$, retrain with fresh optimizers, compare against Athena++ at $t = 1, 2, 3, 4, 5$.

**Results:**

At $t = 1, 2$ (within the original training domain), errors increased from ~1% to ~2–6% compared to the original model — the wrong attractor at large $t_\text{max}$ was strong enough to partially pull weights away from the already-correct early-time solution.

At $t \geq 3$, density amplitude saturated: GRINN reached ~1.6 at $t = 4$ while Athena++ reached ~4.0. Velocity arrows showed rotation rather than infall. The signed residual (GRINN − Athena++) panels were uniformly blue — systematic amplitude under-prediction everywhere collapse occurs.

The max density growth curve on a log scale was concave upward — a polynomial approximating exponential, not exponential itself. Effective growth rate ~0.15 vs. the correct $\omega_J \approx 0.89$, approximately 1/6 of the correct rate.

**Conclusion:** Hypothesis B confirmed. The wrong basin does not just block access — it actively attracts weights away from the correct solution even when training starts inside the right basin.

---

### 4. Temporal Reparameterization

**Hypothesis:** The network cannot efficiently represent $e^{\omega_J t}$ from raw $t$ input due to spectral bias. Remapping $t \to \tau$ such that the growing mode becomes linear in $\tau$ removes the representational bottleneck.

**Reparameterization:** Normalized so that $\tau(t_\text{max}) = t_\text{max}$, preserving the original numerical scale of the network input:

$$\tau_\text{norm}(t) = \frac{(e^{\omega_J t} - 1)/\omega_J}{(e^{\omega_J t_\text{max}} - 1)/(\omega_J \cdot t_\text{max})}$$

The growing mode $\rho \sim e^{\omega_J t}$ becomes linear in $\tau_\text{norm}$, trivially representable by the network. Since autograd computes $d(\text{output})/dt$ through the chain rule and picks up a Jacobian factor $d\tau/dt$, time derivatives in the PDE residuals are corrected by dividing by this Jacobian.

**Result:** The density growth curve on a log scale became a clean straight line — genuinely exponential temporal structure, the correct functional form. However, the amplitude was ~1.75 at $t = 5$ vs Athena++'s ~40, and the velocity field still showed rotation rather than infall. The network learned exponential growth in the **wrong basin**, not the correct collapsing solution.

**Conclusion:** Reparameterization successfully fixed the representational problem — the network can now encode exponential temporal structure. But it did not change which solution the optimizer converges to. The wrong basin remained dominant. This confirms that the problem is not representational: it is purely in the loss landscape geometry.

---

### 5. Causal Training

**Hypothesis:** Standard PINNs allow the optimizer to satisfy late-time residuals before early-time behavior is correct. The low-lying rotating solution exploits this by satisfying late-time residuals cheaply without correct early-time dynamics. Enforcing temporal ordering of learning should prevent this.

**Method (Wang et al. 2022):** Assign each collocation point a weight based on how well the PDE is already satisfied at all earlier times:

$$w(t_i) = \exp\!\left(-\epsilon \sum_{t_j < t_i} \mathcal{L}(t_j)\right)$$

Implemented via a bin-based approximation: partition $[0, t_\text{max}]$ into $M$ ordered time bins, compute mean squared residual per bin, take a running cumulative sum, assign per-point weights normalised to mean = 1. Tested $\epsilon = 2.0$ and $\epsilon = 5.0$ with $M = 20$ bins.

**Result:** No improvement at either $\epsilon$ value. The rotating gas solution is learned in both cases, identically to the baseline.

**Why it likely failed:** The low-lying rotating solution has small PDE residuals at *all* times — early and late — so it passes the causal test and is never suppressed. The causal mechanism suppresses late-time gradients until early-time residuals are small, but the wrong solution already has small early-time residuals. The method would help if the wrong solution had characteristically large early-time residuals, but it does not.

---

## Summary

| Experiment | What It Targets | Result |
|---|---|---|
| Adaptive collocation | Sampling efficiency | No impact |
| PDE time weighting ($e^{\alpha t}$) | Loss magnitude balance | No impact |
| Continuation from correct initialization | Basin accessibility | Wrong basin wins even from correct start |
| Temporal reparameterization | Input representability | Fixed representation; wrong basin still dominates |
| Causal training ($\epsilon = 2.0, 5.0$) | Temporal ordering of learning | No impact |

All five interventions — spanning sampling, loss weighting, initialization, input representation, and gradient ordering — failed to push the PINN toward the correct collapsing solution at large $t_\text{max}$.

---

## Loss Landscape Interpolation

To directly quantify the loss landscape geometry, a linear interpolation was performed between the weights of two saved models: one trained successfully at small $T$ (the correct collapsing solution) and one trained at large $T$ (the wrong rotating solution). This is a standard technique for probing the geometry of the loss landscape between two known points in weight space without requiring any additional training.

**Method:** Given weight vectors $\theta_\text{correct}$ and $\theta_\text{wrong}$, define the interpolated model as:

$$\theta(\alpha) = (1 - \alpha)\,\theta_\text{correct} + \alpha\,\theta_\text{wrong}, \quad \alpha \in [0, 1]$$

At each $\alpha$, the large-$T$ PDE loss is evaluated on a fixed set of 5000 collocation points drawn uniformly from $[0, t_\text{max}=5]$. The result is a 1D cross-section through the loss landscape connecting the two solutions.

**Models used:**
- $\alpha = 0$: model trained at $t_\text{max} = 3$, confirmed correct collapsing solution
- $\alpha = 1$: model trained at $t_\text{max} = 5$, confirmed wrong rotating solution

**Results:**

| | PDE loss (large-$T$ domain) |
|---|---|
| Correct weights ($\alpha=0$) | $2.87 \times 10^{-1}$ |
| Wrong weights ($\alpha=1$) | $3.10 \times 10^{-4}$ |
| Ratio (wrong / correct) | $\approx 0.001$ |

The wrong solution is approximately **1000× lower loss** than the correct solution when evaluated on the large-$T$ domain. The interpolation curve on a log scale shows a clear barrier — the loss rises from $\alpha=0$, peaks near $\alpha \approx 0.7$ at around 1.0, then drops sharply to the minimum at $\alpha=1$. The two solutions are separated by a ridge in weight space, not connected by a downhill slope.

**What this tells us:**

The correct solution does not correspond to a minimum of the large-$T$ loss landscape at all. The weights from the small-$T$ model sit at loss 0.287 — on a hillside, not in a basin — because they were never trained to satisfy the PDEs at $t > 3$. The wrong solution is the sole deep minimum visible from essentially any starting point in weight space.

The three-orders-of-magnitude depth difference quantifies precisely why every intervention failed. Causal training, time weighting, and reparameterization are all small perturbations to the loss landscape. To redirect the optimizer toward the correct solution, any auxiliary signal would need to create a competing basin comparable in depth to $3 \times 10^{-4}$, which for a mode projection loss would likely require weighting it far more heavily than the PDE loss itself — raising questions about whether PDE accuracy would be preserved.

---

## Physical Interpretation

The system has two distinct attractors in the PDE loss landscape:

- **Correct attractor:** $\rho \sim e^{\omega_J t}$, spatially structured, requires the network to encode precise temporal and spatial collapse geometry. PDE residuals grow in time as the solution becomes more nonlinear.
- **Wrong attractor:** Near-constant density, rotating velocity field, minimal temporal structure. PDE residuals remain small and approximately uniform across the entire domain at all times.

At large $t_\text{max}$, the wrong attractor is the **global minimum** of the standard PINN loss — not a local minimum that better optimization could escape. The correct solution has higher integrated PDE residual because exponential growth produces larger gradients that are harder to satisfy everywhere simultaneously. Every intervention tried addressed symptoms of this imbalance rather than the imbalance itself.

The fundamental issue is that the PDE loss function, as standardly formulated, does not distinguish between the two solutions based on physical plausibility — it simply rewards whichever solution produces smaller squared residuals globally, and for large $t_\text{max}$ that is the wrong solution.