# PINN Diagnostics Guide for High-Tmax Failures

This guide explains the 5 optimized diagnostic plots and what they reveal about PINN performance degradation at high tmax.

## Overview

The enhanced `TrainingDiagnostics` class generates **5 critical plots** to diagnose why your PINN fails at high tmax:

1. **Training Diagnostics** - Training convergence analysis
2. **PDE Residual Heatmaps** - Spatiotemporal physics violations
3. **Conservation Laws** - Physical consistency check
4. **Spectral Evolution** - Frequency content analysis
5. **Temporal Statistics** - Field evolution tracking

---

## Plot 1: Training Diagnostics (`training_diagnostics.png`)

**What it shows:**
- Loss components (Total, PDE, IC) over training iterations
- Loss balance ratio (PDE/IC) - should be ~1 for balanced training
- Density statistics evolution during training

**What to look for:**
- ✅ **Good**: Losses decrease smoothly, balance ratio ~1, density grows steadily
- ❌ **Bad**: Loss plateaus early, imbalanced ratio, erratic density
- 🔍 **Diagnosis**: If PDE loss >> IC loss, network overfits to ICs and ignores physics

**Action items:**
- Imbalanced losses → Adjust `IC_WEIGHT` in config
- Early plateau → Increase model capacity or change activation
- Erratic density → Check learning rate or gradient clipping

---

## Plot 2: PDE Residual Heatmaps (`residual_heatmaps.png`)

**What it shows:**
- 2D heatmaps of residuals for each PDE equation over (x, time)
- Four subplots: Continuity, Momentum X, Momentum Y, Poisson
- Color indicates log₁₀ of residual magnitude (darker = larger error)

**What to look for:**
- ✅ **Good**: Uniformly dark (low residuals) across all space-time
- ❌ **Bad**: Bright regions appearing at late times or specific locations
- 🔍 **Diagnosis**: 
  - Bright at late times → Temporal error accumulation
  - Bright at boundaries → Boundary condition issues
  - Poisson bright → Gravitational potential calculation failing
  - Momentum bright → Advection/pressure terms failing

**Action items:**
- Late-time failures → Try causal training, time-marching, or domain decomposition
- Boundary issues → Strengthen boundary loss terms
- Specific equation fails → Add adaptive weighting for that equation
- Uniform degradation → Spectral bias or vanishing gradients (check other plots)

---

## Plot 3: Conservation Laws (`conservation_laws.png`)

**What it shows:**
- Mass, momentum X, and momentum Y drift over time (% change from initial)
- Should be flat horizontal lines at 0% for perfect conservation

**What to look for:**
- ✅ **Good**: All curves stay within ±1% of zero
- ❌ **Bad**: Significant drift (>5%) or systematic trends
- 🔍 **Diagnosis**:
  - Mass drift → Continuity equation not satisfied
  - Momentum drift → Momentum equations or pressure gradient issues
  - Exponential drift → Systematic bias in PDE enforcement

**Action items:**
- Mass drift → Increase weight on continuity residual
- Momentum drift → Check momentum equation implementation
- Large drift → Add explicit conservation constraints to loss
- Systematic trends → Network architecture may have inherent bias

---

## Plot 4: Spectral Evolution (`spectral_evolution.png`)

**What it shows:**
- Left: Radially averaged power spectrum at different times
- Right: 2D power spectrum at final time
- Shows energy distribution across spatial frequencies

**What to look for:**
- ✅ **Good**: Spectrum maintains shape over time, high-k modes present
- ❌ **Bad**: High-frequency modes (large k) disappear over time
- 🔍 **Diagnosis**:
  - Spectral bias → Network preferentially learns low frequencies
  - Damping at high-k → Numerical diffusion or insufficient resolution
  - Spectrum shifts → Physical vs numerical effects

**Action items:**
- Spectral bias → Use Fourier features, different activation (sin works better)
- High-k damping → Increase collocation points, use adaptive sampling
- Power law changes → Compare with numerical solver to verify if physical
- Missing modes → Increase network capacity or use multi-scale architecture

---

## Plot 5: Temporal Statistics (`temporal_statistics.png`)

**What it shows:**
- Left: Density evolution (mean ± std, max)
- Middle: Gradient magnitude evolution (mean, max)
- Right: Exponential growth rate fit

**What to look for:**
- ✅ **Good**: Smooth evolution, gradients stable, growth matches theory
- ❌ **Bad**: Sudden jumps, exploding gradients, wrong growth rate
- 🔍 **Diagnosis**:
  - Exploding gradients → Network becoming unstable
  - Plateauing density → Growth stalled (unphysical)
  - Wrong growth rate → Physics not captured correctly

**Action items:**
- Exploding gradients → Reduce learning rate, add gradient clipping
- Plateauing → Check if network saturating (activation functions)
- Wrong growth → Compare with linear theory, check PDE implementation
- Oscillations → Reduce time step in collocation sampling

---

## Usage Workflow

### During Training:
```python
diagnostics = TrainingDiagnostics(save_dir='./diagnostics/')

for iteration in range(max_iter):
    loss, loss_dict, geomtime_col = train_step(...)
    
    if iteration % 100 == 0:
        diagnostics.log_iteration(iteration, model, loss_dict, geomtime_col)

diagnostics.plot_diagnostics()  # Generates Plot 1
```

### After Training:
```python
# Generates Plots 2-5 for high-tmax analysis
diagnostics.run_comprehensive_diagnostics(
    model=trained_model,
    dimension=2,
    tmax=config.tmax
)
```

---

## Interpretation Decision Tree

```
Start: PINN fails at high tmax
│
├─ Check Plot 2 (Residuals)
│  ├─ Specific equation fails → Target that equation with adaptive weights
│  ├─ Late-time failure → Temporal accumulation (try causal training)
│  └─ Boundary regions fail → Strengthen boundary conditions
│
├─ Check Plot 3 (Conservation)
│  ├─ Mass drifts → Continuity equation issue
│  ├─ Momentum drifts → Momentum/pressure gradient issue
│  └─ All conserved → Physics OK, check numerical issues
│
├─ Check Plot 4 (Spectrum)
│  ├─ High-k damped → Spectral bias (try Fourier features)
│  ├─ Spectrum OK → Not a frequency resolution issue
│  └─ Power law wrong → Compare with numerical solver
│
├─ Check Plot 5 (Statistics)
│  ├─ Gradients explode → Instability (reduce LR, clip gradients)
│  ├─ Growth wrong → Physics implementation error
│  └─ Smooth but wrong → Systematic bias in network
│
└─ Check Plot 1 (Training)
   ├─ Losses imbalanced → Adjust loss weights
   ├─ Early plateau → Increase capacity or change architecture
   └─ Converged well → Problem is in inference, not training
```

---

## Common Failure Patterns

### Pattern 1: Spectral Bias
- **Symptoms**: Plot 4 shows high-k damping, Plot 2 shows uniform late-time failure
- **Solution**: Fourier features, sin activation, multi-scale architecture

### Pattern 2: Temporal Error Accumulation
- **Symptoms**: Plot 2 shows increasing residuals with time, Plot 5 shows wrong growth
- **Solution**: Causal training, time-marching, domain decomposition in time

### Pattern 3: Conservation Violation
- **Symptoms**: Plot 3 shows drift, Plot 2 shows continuity/momentum residuals
- **Solution**: Add conservation constraints, increase PDE loss weight

### Pattern 4: Gradient Issues
- **Symptoms**: Plot 5 shows exploding gradients, Plot 1 shows training instability
- **Solution**: Gradient clipping, lower learning rate, different optimizer

### Pattern 5: Loss Imbalance
- **Symptoms**: Plot 1 shows PDE >> IC or IC >> PDE, Plot 2 shows high residuals
- **Solution**: Adaptive loss weighting, adjust `IC_WEIGHT`

---

## References for Solutions

Based on diagnostic results, consider these techniques:

1. **Spectral Bias** → Fourier Features (Tancik et al. 2020)
2. **Temporal Failures** → Causal Training (Wang et al. 2022)
3. **Conservation** → Physics-Informed Losses (Raissi et al. 2019)
4. **Gradients** → Residual Connections (Wang et al. 2021)
5. **Loss Balance** → Adaptive Weights (Wang et al. 2021)

---

## Quick Reference

| Symptom | Check Plot | Likely Cause | Solution |
|---------|-----------|--------------|----------|
| Late-time failure | 2, 5 | Temporal accumulation | Causal training |
| High-freq damping | 4 | Spectral bias | Fourier features |
| Mass drift | 3 | Continuity violation | Increase continuity weight |
| Exploding gradients | 5 | Instability | Gradient clipping |
| Loss imbalance | 1 | Weight mismatch | Adjust IC_WEIGHT |
| Boundary errors | 2 | BC enforcement | Strengthen BC terms |
| Wrong growth rate | 5 | Physics error | Check PDE implementation |

---

**Note**: Always compare with your LAX numerical solver results to verify if issues are physical or numerical!
