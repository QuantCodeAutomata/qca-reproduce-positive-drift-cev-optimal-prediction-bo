"""
Reduced loss function G and generator term H for the CEV optimal prediction problem.

Given aspiration CDF F:
  G(x) = integral_0^x (F(y) - 1/2) dy
       = x/2 - E[l] + integral_x^inf (1 - F(y)) dy   (tail form)

For exponential law F(x) = 1 - exp(-lambda*x):
  G(x) = x/2 - 1/lambda + exp(-lambda*x)/lambda

Generator term (used in the free-boundary equations):
  H(x) = mu * x * (F(x) - 1/2) + (sigma^2/2) * x^{2*beta+2} * f(x)

The sign-change anchor alpha (positive drift, H changes sign in (0,m)) or
gamma (negative drift, H changes sign in (m,inf)) determines terminal boundaries.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from src.aspiration_laws import ExponentialAspirationLaw


def G_analytical(x: np.ndarray | float, law: ExponentialAspirationLaw) -> np.ndarray | float:
    """
    Reduced loss function G for exponential aspiration law (closed form).

    G(x) = x/2 - 1/lambda + exp(-lambda*x)/lambda
         = x/2 - E[l] + (1/lambda)*exp(-lambda*x)

    Parameters
    ----------
    x    : state value(s) >= 0
    law  : ExponentialAspirationLaw instance

    Returns
    -------
    G(x)
    """
    x = np.asarray(x, dtype=float)
    lam = law.lam
    return x / 2.0 - 1.0 / lam + np.exp(-lam * x) / lam


def G_integral(x: float, law: ExponentialAspirationLaw) -> float:
    """
    Reduced loss function G computed by direct integration:
    G(x) = integral_0^x (F(y) - 1/2) dy.

    This serves as a numerical check against G_analytical.
    """
    if x <= 0.0:
        return 0.0
    val, _ = quad(lambda y: law.F(y) - 0.5, 0.0, x)
    return val


def G_tail(x: float, law: ExponentialAspirationLaw) -> float:
    """
    Reduced loss function G via tail form:
    G(x) = x/2 - E[l] + integral_x^inf (1-F(y)) dy.

    For exponential: integral_x^inf exp(-lam*y) dy = exp(-lam*x)/lam.
    """
    return x / 2.0 - law.E_l + np.exp(-law.lam * x) / law.lam


def H(
    x: np.ndarray | float,
    mu: float,
    sigma: float,
    beta: float,
    law: ExponentialAspirationLaw,
) -> np.ndarray | float:
    """
    Generator term for the CEV optimal prediction problem:
    H(x) = mu * x * (F(x) - 1/2) + (sigma^2/2) * x^{2*beta+2} * f(x)

    For exponential law:
      F(x) - 1/2 = 1/2 - exp(-lam*x)
      f(x) = lam * exp(-lam*x)
    => H(x) = mu*x*(1/2 - exp(-lam*x)) + (sigma^2 * lam / 2)*x^{2*beta+2}*exp(-lam*x)

    Parameters
    ----------
    x    : state value(s) > 0
    mu   : drift parameter
    sigma: volatility parameter
    beta : elasticity parameter
    law  : ExponentialAspirationLaw instance
    """
    x = np.asarray(x, dtype=float)
    lam = law.lam
    exp_term = np.exp(-lam * x)
    drift_part = mu * x * (0.5 - exp_term)
    diff_part = (sigma**2 * lam / 2.0) * (x ** (2 * beta + 2)) * exp_term
    return drift_part + diff_part


def find_alpha(mu: float, sigma: float, beta: float, law: ExponentialAspirationLaw) -> float:
    """
    Find the sign-change anchor alpha in (0, m) for positive-drift regime.
    H(alpha) = 0, H < 0 on (0, alpha), H > 0 on (alpha, m).

    Only valid when mu > 0.

    Parameters
    ----------
    mu, sigma, beta : CEV parameters (mu > 0)
    law             : ExponentialAspirationLaw instance

    Returns
    -------
    alpha : float  - unique zero of H in (0, m)
    """
    if mu <= 0:
        raise ValueError("find_alpha is for positive drift (mu > 0)")
    m = law.median
    h = lambda x: H(x, mu, sigma, beta, law)
    # H(0+) = 0+ but approaching -inf? Let's check H at a small x.
    # H(eps) = mu*eps*(0.5 - 1 + O(eps)) + ... ≈ mu*eps*(-0.5) < 0 for mu > 0
    # H(m-eps) should be > 0 near m from left for positive drift? Let's just bracket.
    x_lo = 1e-6
    x_hi = m - 1e-8
    if h(x_lo) * h(x_hi) > 0:
        # Try adjusting bracket
        x_lo = 1e-8
        x_hi = m * 0.9999
    return brentq(h, x_lo, x_hi, xtol=1e-10, rtol=1e-10)


def find_gamma(mu: float, sigma: float, beta: float, law: ExponentialAspirationLaw) -> float:
    """
    Find the sign-change anchor gamma in (m, inf) for negative-drift regime.
    H(gamma) = 0, H > 0 on (m, gamma), H < 0 on (gamma, inf).

    Only valid when mu < 0.

    Parameters
    ----------
    mu, sigma, beta : CEV parameters (mu < 0)
    law             : ExponentialAspirationLaw instance

    Returns
    -------
    gamma : float  - unique zero of H in (m, inf)
    """
    if mu >= 0:
        raise ValueError("find_gamma is for negative drift (mu < 0)")
    m = law.median
    h = lambda x: H(x, mu, sigma, beta, law)
    # Search for sign change: H(m+eps) > 0 and H(x_hi) < 0
    x_lo = m + 1e-6
    # Search for x_hi where H < 0
    x_hi = m + 1.0
    while h(x_hi) > 0:
        x_hi *= 2.0
        if x_hi > 1e6:
            raise RuntimeError("Could not find upper bracket for gamma")
    return brentq(h, x_lo, x_hi, xtol=1e-10, rtol=1e-10)
