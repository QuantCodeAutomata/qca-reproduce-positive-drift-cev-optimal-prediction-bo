"""
Auxiliary integral kernels for the CEV optimal prediction free-boundary equations.

J(t, x) = E_x[G(X_{T-t})] = ∫_0^∞ G(z) * p^(μ)(x, z; T-t) dz

M(s, x, y, z) = ∫_y^z H(u) * p^(μ)(x, u; s) du

L(s, x, y) = ∫_0^y H(u) * p^(μ)(x, u; s) du
           = lim_{z->∞} M(s, x, 0, z) restricted to [0, y]

Numerical strategy:
- For J: use substitution z = exp(u) to map (0,∞) → ℝ, then finite-range quadrature.
- For M and L: direct quadrature with Gaussian nodes over the finite interval.
- All evaluations use log-stable cev_density.
"""

import numpy as np
from scipy.integrate import quad

from src.aspiration_laws import ExponentialAspirationLaw
from src.loss_functions import G_analytical, H
from src.cev_density import cev_density


def compute_J(
    t: float,
    x: float,
    T: float,
    mu: float,
    sigma: float,
    beta: float,
    law: ExponentialAspirationLaw,
    u_lo: float = -8.0,
    u_hi: float = 5.0,
    limit: int = 200,
    epsabs: float = 1e-7,
    epsrel: float = 1e-7,
) -> float:
    """
    Compute J(t, x) = ∫_0^∞ G(z) * p^(μ)(x, z; T-t) dz.

    Uses substitution z = exp(u) so the integral becomes
      ∫_{-∞}^{∞} G(exp(u)) * p^(μ)(x, exp(u); T-t) * exp(u) du.

    Parameters
    ----------
    t        : current time in [0, T)
    x        : current state > 0
    T        : horizon
    mu, sigma, beta : CEV parameters
    law      : aspiration law instance
    u_lo, u_hi : integration bounds in u = log(z) space
    limit, epsabs, epsrel : quad controls

    Returns
    -------
    J value (float)
    """
    elapsed = T - t
    if elapsed <= 0.0:
        return G_analytical(x, law)

    def integrand(u: float) -> float:
        z = np.exp(u)
        dens = cev_density(x, z, elapsed, mu, sigma, beta)
        return G_analytical(z, law) * dens * z  # Jacobian: dz = exp(u) du = z du

    val, _ = quad(integrand, u_lo, u_hi, limit=limit, epsabs=epsabs, epsrel=epsrel)
    return val


def compute_M(
    s: float,
    x: float,
    y: float,
    z: float,
    mu: float,
    sigma: float,
    beta: float,
    law: ExponentialAspirationLaw,
    limit: int = 100,
    epsabs: float = 1e-7,
    epsrel: float = 1e-7,
) -> float:
    """
    Compute M(s, x, y, z) = ∫_y^z H(u) * p^(μ)(x, u; s) du.

    Uses direct quadrature over [y, z].

    Parameters
    ----------
    s     : time increment > 0
    x     : starting state > 0
    y, z  : integration bounds, 0 < y <= z
    mu, sigma, beta : CEV parameters
    law   : aspiration law instance
    """
    if s <= 0.0 or y >= z:
        return 0.0

    def integrand(u: float) -> float:
        return H(u, mu, sigma, beta, law) * cev_density(x, u, s, mu, sigma, beta)

    val, _ = quad(integrand, y, z, limit=limit, epsabs=epsabs, epsrel=epsrel)
    return val


def compute_L(
    s: float,
    x: float,
    y: float,
    mu: float,
    sigma: float,
    beta: float,
    law: ExponentialAspirationLaw,
    u_lo: float = 1e-8,
    limit: int = 100,
    epsabs: float = 1e-7,
    epsrel: float = 1e-7,
) -> float:
    """
    Compute L(s, x, y) = ∫_0^y H(u) * p^(μ)(x, u; s) du.

    Uses direct quadrature over [u_lo, y].

    Parameters
    ----------
    s     : time increment > 0
    x     : starting state > 0
    y     : upper integration bound > 0
    mu, sigma, beta : CEV parameters
    law   : aspiration law instance
    u_lo  : small positive lower bound to avoid singularity at 0
    """
    if s <= 0.0 or y <= 0.0:
        return 0.0

    def integrand(u: float) -> float:
        return H(u, mu, sigma, beta, law) * cev_density(x, u, s, mu, sigma, beta)

    lower = max(u_lo, 1e-10)
    if y <= lower:
        return 0.0
    val, _ = quad(integrand, lower, y, limit=limit, epsabs=epsabs, epsrel=epsrel)
    return val
