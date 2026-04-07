"""
tmax calcs.py
=============
Physics-based tmax selector for GRINN training.

The goal is to choose a training horizon that:
  1. Covers the linear-growth phase so the PINN learns the correct growth rate.
  2. Extends slightly *past* the onset of gravitational collapse so the
     accelerating growth acts as an anchor — the network cannot satisfy the
     loss with a wrong (too-slow) growth rate once the field starts curving.
  3. Stops before the deep nonlinear phase, which exceeds the representational
     capacity of a PINN regardless of training strategy.

No numerical solver output is needed.  All inputs come from config.py.

Physical model
--------------
For a 2-D turbulent IC with a power-spectrum velocity field P_v(k) ~ k^n
and Mach coefficient a  (rms velocity = a * cs):

  1. The density seed at t = 0 comes from the compressive part of the
     velocity field acting on the background density (∇·v term):

         δρ_rms(t=0) = f_c · a · sqrt(∫_unstable k² P_v dk / ∫_all P_v dk)

     where the numerator integral runs only over Jeans-unstable modes
     (k < k_J = sqrt(const·G·ρ₀) / cs) because stable modes do not
     contribute to gravitational runaway.

  2. The peak of the density seed that controls local collapse is not the
     absolute pixel maximum but the peak of the *gravitationally coherent*
     (Jeans-filtered) field.  The domain contains N_Jeans = num_of_waves²
     independent Jeans-scale cells, so the expected peak is:

         δρ_peak(t=0) = C_peak · δρ_rms(t=0)
         C_peak = sqrt(2 · ln(N_Jeans))     [from extreme-value statistics]

  3. Under linear Jeans growth the fastest-growing mode (k → k_min) grows as
     exp(σ · t) where:

         σ = sqrt(const·G·ρ₀ − cs²·k_min²)

     (full dispersion relation, not just the k→0 limit used by the old script).

  4. The onset of collapse (t_onset) is when the local peak first reaches
     δρ/ρ₀ = 1:

         t_onset = ln(1 / δρ_peak) / σ       [if δρ_peak < 1]
                 = 0                           [already nonlinear at t=0]

  5. tmax is set to include a window of width α · τ_J past the onset:

         tmax = t_onset + alpha_window · τ_J

     alpha_window = 0.1 by default (about 10% of one Jeans time past onset).
     This is enough to anchor the growth rate but short enough to stay out of
     the deep nonlinear regime.

Why this is better than the previous approach
---------------------------------------------
• Old script:  tmax = safety_factor × t_nl,  where t_nl was computed with the
  seed estimated *at t_ff* (wrong reference time) and σ was the k→0 limit
  (ignores pressure stabilisation at k_min).  The result systematically
  overestimates t_nl, then retreats *away* from the onset — keeping tmax
  entirely in the linear regime where the growth rate is underconstrained.

• New script:  seed is computed at t=0 (correct IC), σ uses the full
  dispersion relation, and tmax is placed *just past* t_onset.  The collapse
  signal anchors σ_learned, so the PINN cannot converge to a slow-growth
  spurious solution.

  For supersonic cases (a ≳ 1) the local collapse happens early because
  δρ_peak is already a significant fraction of 1 at t=0 — this falls out
  naturally from the formula without a separate ad-hoc safety-factor branch.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Spectral moment helper
# ---------------------------------------------------------------------------

def _spectral_moment(n, k1, k2):
    """
    Compute ∫_{k1}^{k2} k^n dk analytically.
    Returns 0 if k2 ≤ k1.
    """
    if k2 <= k1:
        return 0.0
    if abs(n + 1) < 1e-12:          # logarithmic case
        return np.log(k2 / k1)
    return (k2 ** (n + 1) - k1 ** (n + 1)) / (n + 1)


# ---------------------------------------------------------------------------
# Core timescale computation
# ---------------------------------------------------------------------------

def compute_timescales(
    a,
    G=1.0,
    rho_0=1.0,
    cs=1.0,
    lam=7.0,
    num_of_waves=2,
    const=1.0,
    n_grid=400,
    power_exponent=-4,
    f_c=0.5,
):
    """
    Compute the physical timescales governing the PINN training horizon.

    Parameters
    ----------
    a              : Mach-like coefficient  (rms velocity = a · cs)
    G              : gravitational constant                 [config: G]
    rho_0          : background density                     [config: rho_o]
    cs             : sound speed                            [config: cs]
    lam            : wavelength                             [config: wave]
    num_of_waves   : number of wavelengths in domain        [config: num_of_waves]
    const          : Poisson coupling constant (= 4πG norm) [config: const]
    n_grid         : grid resolution (for Nyquist cutoff)   [config: N_GRID]
    power_exponent : spectral index of the IC velocity field [config: POWER_EXPONENT]
    f_c            : compressive fraction of turbulence     [default 0.5]

    Returns
    -------
    dict with keys:
        t_ff         : global free-fall time  sqrt(3π / 32Gρ₀)
        sigma        : Jeans growth rate at k_min  sqrt(const·G·ρ₀ − cs²·k_min²)
        tau_J        : Jeans growth time  1/σ
        k_min        : minimum (and most unstable) wavenumber  2π / L
        k_J          : Jeans wavenumber  sqrt(const·G·ρ₀) / cs
        delta_rms    : rms density seed at t=0 (from compressive velocity)
        delta_peak   : expected peak density seed (Jeans-coherent regions)
        t_onset      : time when the peak reaches δρ/ρ₀ = 1  (onset of collapse)
        t_onset_tff  : t_onset / t_ff
        regime       : 'subsonic' | 'transonic' | 'supersonic'
    """
    L     = lam * num_of_waves
    k_min = 2.0 * np.pi / L
    k_max = np.pi * n_grid / L          # Nyquist wavenumber
    k_J   = np.sqrt(const * G * rho_0) / cs   # Jeans wavenumber

    # ── 1. Growth rate at k_min (full dispersion relation) ──────────────────
    sigma_sq = const * G * rho_0 - cs ** 2 * k_min ** 2
    if sigma_sq <= 0.0:
        raise ValueError(
            f"k_min={k_min:.4f} is Jeans-stable (σ²={sigma_sq:.4g}). "
            "Increase the domain size or reduce cs."
        )
    sigma = np.sqrt(sigma_sq)
    tau_J = 1.0 / sigma

    # ── 2. Free-fall time ────────────────────────────────────────────────────
    t_ff = np.sqrt(3.0 * np.pi / (32.0 * G * rho_0))

    # ── 3. rms density seed from compressive velocity at t = 0 ──────────────
    #   Only Jeans-unstable modes (k < k_J) can drive collapse.
    n   = power_exponent
    k_unstable_max = min(k_J, k_max)

    # ∫ k² P_v dk  over unstable modes  (numerator ~ variance of ∇·v)
    m_unstable = _spectral_moment(n + 3, k_min, k_unstable_max)
    # ∫ P_v dk  over all modes           (denominator ~ total kinetic energy)
    m_total    = _spectral_moment(n + 1, k_min, k_max)

    if m_total <= 0.0 or m_unstable <= 0.0:
        delta_rms = 0.0
    else:
        delta_rms = f_c * a * np.sqrt(m_unstable / m_total)

    # ── 4. Peak seed over Jeans-coherent regions ────────────────────────────
    #   The domain contains N_Jeans = num_of_waves² independent Jeans cells.
    #   Expected maximum from extreme-value statistics for Gaussian field:
    #       C_peak = sqrt(2 · ln(N_Jeans))
    N_Jeans  = max(int(num_of_waves) ** 2, 2)
    C_peak   = np.sqrt(2.0 * np.log(N_Jeans))
    delta_peak = C_peak * delta_rms

    # ── 5. Onset of collapse ─────────────────────────────────────────────────
    if delta_peak <= 0.0:
        t_onset = np.inf
    elif delta_peak >= 1.0:
        t_onset = 0.0            # already nonlinear at t = 0
    else:
        t_onset = np.log(1.0 / delta_peak) / sigma

    # ── 6. Regime classification ─────────────────────────────────────────────
    if a < 1.0:
        regime = "subsonic"
    else:
        regime = "supersonic"

    return {
        "t_ff"        : t_ff,
        "sigma"       : sigma,
        "tau_J"       : tau_J,
        "k_min"       : k_min,
        "k_J"         : k_J,
        "delta_rms"   : delta_rms,
        "delta_peak"  : delta_peak,
        "C_peak"      : C_peak,
        "t_onset"     : t_onset,
        "t_onset_tff" : t_onset / t_ff if t_ff > 0 else np.inf,
        "regime"      : regime,
    }


# ---------------------------------------------------------------------------
# tmax suggestion
# ---------------------------------------------------------------------------

def suggest_tmax(a, alpha_window=0.1, **kwargs):
    """
    Suggest a tmax for PINN training.

    tmax is placed alpha_window · τ_J past the onset of gravitational collapse.
    This ensures the training window captures:
      • the full linear-growth phase  (anchors the growth rate)
      • a short initial segment of the collapse  (rules out slow-growth solutions)
    but stops before the deep nonlinear phase where PINN accuracy degrades.

    Parameters
    ----------
    a             : Mach coefficient  (same as compute_timescales)
    alpha_window  : fraction of τ_J to include past t_onset  [default 0.1]
    **kwargs      : forwarded to compute_timescales

    Returns
    -------
    tmax  : float
    ts    : dict  (full timescale dictionary from compute_timescales)
    """
    ts   = compute_timescales(a, **kwargs)
    tmax = ts["t_onset"] + alpha_window * ts["tau_J"]
    return tmax, ts


# ---------------------------------------------------------------------------
# Demo / diagnostic printout
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Pull defaults from config so the demo stays consistent with training runs.
    try:
        from config import G, rho_o as rho_0, cs, wave, num_of_waves, const, N_GRID, POWER_EXPONENT
        _kwargs = dict(G=G, rho_0=rho_0, cs=cs, lam=wave,
                       num_of_waves=num_of_waves, const=const,
                       n_grid=N_GRID, power_exponent=POWER_EXPONENT)
    except ImportError:
        _kwargs = {}
        print("(config.py not found — using default parameter values)\n")

    print(f"{'a':>5}  {'regime':>11}  {'tau_J':>6}  {'t_ff':>6}  "
          f"{'d_rms':>7}  {'d_peak':>7}  {'t_onset':>8}  "
          f"{'t_onset/t_ff':>12}  {'tmax':>6}")
    print("-" * 80)

    test_values = [0.2, 0.6, 1.1, 1.4]
    for a in test_values:
        try:
            tmax, ts = suggest_tmax(a, **_kwargs)
            print(
                f"{a:>5.1f}  {ts['regime']:>11}  {ts['tau_J']:>6.3f}  {ts['t_ff']:>6.3f}  "
                f"{ts['delta_rms']:>7.4f}  {ts['delta_peak']:>7.4f}  {ts['t_onset']:>8.3f}  "
                f"{ts['t_onset_tff']:>12.2f}  {tmax:>6.3f}"
            )
        except ValueError as e:
            print(f"{a:>5.1f}  ERROR: {e}")

    print()
    print("Notes")
    print("-----")
    print("* tmax = t_onset + 0.1 * tau_J  (just past collapse onset)")
    print("* t_onset: when the local density peak first reaches d_rho/rho_0 = 1")
    print("* d_peak uses extreme-value statistics over Jeans-coherent regions")
    print("* For supersonic (a > 1): d_peak is large at t=0 -> t_onset is short")
    print("  -> tmax comes out short naturally, no ad-hoc safety-factor branching")
    #print("* Use tmax from this script as the starting point; if training with")
    #print("  causal / adaptive collocation, you may extend by ~0.1-0.2 * tau_J")
    #print("  since those methods handle the early nonlinear phase better.")
