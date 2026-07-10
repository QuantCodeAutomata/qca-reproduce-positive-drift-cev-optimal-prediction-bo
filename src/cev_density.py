"""
CEV transition density for the optimal prediction problem.

The CEV process: dX_t = mu*X_t dt + sigma*X_t^{1+beta} dW_t

Zero-drift density (Schroder, 1989):
  p^(0)(x,z;t) = z^{-2β-3/2} * x^{1/2} / (σ²|β|t)
               * I_{ν}(x^{-β}z^{-β}/(σ²β²t))
               * exp(-(x^{-2β}+z^{-2β})/(2σ²β²t))
where ν = 1/(2β).

General drift density via Girsanov / scaling:
  p^(μ)(x,z;t) = exp(-μt) * p^(0)(x, exp(-μt)*z; T_t)
where T_t = (exp(2μβt) - 1) / (2μβ).

For numerical stability, we use:
  ive(ν, arg) = I_ν(arg) * exp(-arg)
to avoid overflow, so:
  I_ν(arg) * exp(-quad) = ive(ν, arg) * exp(arg - quad)
where arg - quad = -(x^{-β} - exp(μβt)z^{-β})^2 / (2σ²β²T_t) ≤ 0.
"""

import numpy as np
from scipy.special import ive  # exp-scaled Bessel I_ν(x)*exp(-x)
from scipy.integrate import quad


def T_scaling(mu: float, beta: float, t: float) -> float:
    """
    Compute the time-scaling factor T_t = (exp(2μβt)-1)/(2μβ).

    For |μβt| < 1e-8, use first-order Taylor: T_t ≈ t.
    """
    r = 2.0 * mu * beta * t
    if abs(r) < 1e-8:
        return t  # L'Hôpital: lim_{r->0} (e^r-1)/r * t = t
    return (np.exp(r) - 1.0) / (2.0 * mu * beta)


def _log_p0_scalar(x: float, z: float, t: float, sigma: float, beta: float) -> float:
    """
    Log of zero-drift CEV transition density (scalar version).

    Returns -inf if density is effectively 0 (e.g. z <= 0).
    """
    if z <= 0.0 or x <= 0.0 or t <= 0.0:
        return -np.inf

    nu = 1.0 / (2.0 * beta)
    c = sigma**2 * beta**2 * t  # σ²β²t

    x_mb = x ** (-beta)   # x^{-β}
    z_mb = z ** (-beta)   # z^{-β}

    arg = x_mb * z_mb / c                          # Bessel argument
    # quad - arg = (x^{-β} - z^{-β})^2 / (2σ²β²t) >= 0
    diff2 = (x_mb - z_mb) ** 2 / (2.0 * c)

    log_prefactor = (-2.0 * beta - 1.5) * np.log(z) + 0.5 * np.log(x)
    log_prefactor -= np.log(sigma**2 * abs(beta) * t)

    # log(ive(ν, arg)) is safe even for large arg
    log_ive = np.log(ive(nu, arg)) if arg > 0.0 else -np.inf
    if not np.isfinite(log_ive):
        return -np.inf

    # Full: log_prefactor + log(I_ν(arg)) - quad
    #      = log_prefactor + log(ive(ν,arg)) + arg - quad
    #      = log_prefactor + log(ive(ν,arg)) - diff2
    return log_prefactor + log_ive - diff2


def log_cev_density(
    x: float,
    z: float,
    t: float,
    mu: float,
    sigma: float,
    beta: float,
) -> float:
    """
    Log of the CEV transition density p^(μ)(x,z;t) (scalar).

    p^(μ)(x,z;t) = exp(-μt) * p^(0)(x, exp(-μt)*z; T_t)

    Parameters
    ----------
    x     : starting state > 0
    z     : ending state > 0
    t     : elapsed time > 0
    mu    : drift parameter
    sigma : volatility parameter
    beta  : elasticity parameter > 0

    Returns
    -------
    log density value (float, may be -inf)
    """
    if x <= 0.0 or z <= 0.0 or t <= 0.0:
        return -np.inf

    Tt = T_scaling(mu, beta, t)
    if Tt <= 0.0:
        return -np.inf

    w = np.exp(-mu * t) * z  # scaled z for p^(0)
    return -mu * t + _log_p0_scalar(x, w, Tt, sigma, beta)


def cev_density(
    x: float,
    z: np.ndarray | float,
    t: float,
    mu: float,
    sigma: float,
    beta: float,
) -> np.ndarray | float:
    """
    CEV transition density p^(μ)(x,z;t).

    Vectorized over z; x and t are scalars.

    Parameters
    ----------
    x     : starting state > 0 (scalar)
    z     : ending state(s) > 0 (array or scalar)
    t     : elapsed time > 0 (scalar)
    mu    : drift parameter
    sigma : volatility parameter
    beta  : elasticity parameter > 0

    Returns
    -------
    density values (same shape as z)
    """
    z = np.asarray(z, dtype=float)
    scalar_input = z.ndim == 0
    z = np.atleast_1d(z)

    if x <= 0.0 or t <= 0.0:
        result = np.zeros_like(z)
        return result.item() if scalar_input else result

    Tt = T_scaling(mu, beta, t)
    if Tt <= 0.0:
        result = np.zeros_like(z)
        return result.item() if scalar_input else result

    nu = 1.0 / (2.0 * beta)
    c = sigma**2 * beta**2 * Tt       # σ²β²T_t
    emu_t = np.exp(-mu * t)            # exp(-μt)

    w = emu_t * z                      # w = exp(-μt)*z (same shape as z)
    valid = (w > 0.0)

    result = np.zeros_like(z)
    if not np.any(valid):
        return result.item() if scalar_input else result

    wv = w[valid]
    x_mb = x ** (-beta)                # x^{-β}
    w_mb = wv ** (-beta)               # w^{-β}

    arg = x_mb * w_mb / c              # Bessel argument
    diff2 = (x_mb - w_mb) ** 2 / (2.0 * c)  # (x^{-β}-w^{-β})^2/(2σ²β²T_t)

    log_pre = (-2.0 * beta - 1.5) * np.log(wv) + 0.5 * np.log(x)
    log_pre -= np.log(sigma**2 * abs(beta) * Tt)

    # log(I_ν(arg)*exp(-quad)) = log(ive(ν,arg)) - diff2
    log_ive = np.log(np.where(arg > 0, ive(nu, arg), 0.0))
    log_val = -mu * t + log_pre + log_ive - diff2

    result[valid] = np.exp(log_val)
    return result.item() if scalar_input else result


def check_density_normalization(
    x: float,
    t: float,
    mu: float,
    sigma: float,
    beta: float,
    z_max: float = 50.0,
    tol: float = 0.05,
) -> dict:
    """
    Check that ∫_0^∞ p^(μ)(x,z;t) dz ≈ 1.

    Uses log-substitution z = exp(u) to map (0,∞) → (-∞,∞),
    then integrates over a wide but finite interval.

    Parameters
    ----------
    x, t     : starting state and time
    mu, sigma, beta : model parameters
    z_max    : upper truncation in z-space
    tol      : absolute tolerance for norm ≈ 1 check

    Returns
    -------
    dict with keys 'norm', 'error', 'pass'
    """
    def integrand(u: float) -> float:
        z = np.exp(u)
        return cev_density(x, z, t, mu, sigma, beta) * z

    u_lo = np.log(1e-6)
    u_hi = np.log(z_max)
    val, _ = quad(integrand, u_lo, u_hi, limit=200, epsabs=1e-8, epsrel=1e-8)
    return {"norm": val, "error": abs(val - 1.0), "pass": abs(val - 1.0) < tol}
