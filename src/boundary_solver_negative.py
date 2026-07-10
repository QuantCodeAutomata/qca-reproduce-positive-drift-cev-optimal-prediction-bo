"""
Backward induction solver for the negative-drift CEV optimal prediction problem.

Solves the scalar free-boundary integral equation for the one-sided stopping region
D = {(t,x): x <= b(t)}.

Free-boundary condition at x = b(t):
  V(t, b(t)) = G(b(t))

Using the integral representation:
  V(t, x) = J(t, x) - ∫_{t}^{T} L(s-t, x, b(s)) ds

The free-boundary equation becomes:
  J(t_i, b_i) - G(b_i) = ∫_{t_i}^{T} L(s-t_i, b_i, b(s)) ds

Backward induction: solve from t_{N-1} = T-dt backward to t_0 = 0.
Terminal condition: b(T-) = gamma.
"""

import numpy as np
from scipy.optimize import brentq
from typing import Tuple

from src.aspiration_laws import ExponentialAspirationLaw
from src.loss_functions import G_analytical, H, find_gamma
from src.kernels import compute_J, compute_L


def _time_integral_approx_L(
    t_i: float,
    x: float,
    future_times: np.ndarray,
    b_future: np.ndarray,
    mu: float,
    sigma: float,
    beta: float,
    law: ExponentialAspirationLaw,
    T: float = 1.0,
    gamma: float = None,
    epsabs: float = 1e-6,
    epsrel: float = 1e-6,
) -> float:
    """
    Approximate ∫_{t_i}^{T} L(s-t_i, x, b(s)) ds by trapezoidal rule
    over the future boundary nodes. Includes a terminal correction for the
    [t_{N-1}, T] segment using b(T-) = gamma as the terminal boundary value.

    Parameters
    ----------
    epsabs, epsrel : quadrature tolerances forwarded to compute_L (loosen for speed)
    """
    n = len(future_times)
    if n == 0:
        return 0.0

    l_vals = np.zeros(n)
    for j in range(n):
        s = future_times[j] - t_i
        if s > 1e-12:
            l_vals[j] = compute_L(
                s, x, b_future[j], mu, sigma, beta, law,
                limit=80, epsabs=epsabs, epsrel=epsrel,
            )

    all_times = np.concatenate([[t_i], future_times])
    all_l = np.concatenate([[0.0], l_vals])

    # Terminal correction: include [t_{N-1}, T] segment using b(T-) = gamma
    t_last = future_times[-1]
    dt_terminal = T - t_last
    if dt_terminal > 1e-14 and gamma is not None:
        s_terminal = T - t_i
        l_terminal = compute_L(
            s_terminal, x, gamma, mu, sigma, beta, law,
            limit=80, epsabs=max(epsabs, 1e-5), epsrel=max(epsrel, 1e-5),
        )
        all_times = np.append(all_times, T)
        all_l = np.append(all_l, l_terminal)

    return np.trapz(all_l, all_times)


def solve_negative_boundary(
    mu: float,
    sigma: float,
    beta: float,
    T: float,
    law: ExponentialAspirationLaw,
    N: int = 60,
    verbose: bool = True,
    epsabs: float = 1e-6,
    epsrel: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the negative-drift free-boundary integral equation by backward induction.

    Parameters
    ----------
    mu, sigma, beta : CEV parameters (mu < 0 for negative drift)
    T               : horizon
    law             : aspiration law instance
    N               : number of time steps
    verbose         : print progress
    epsabs, epsrel  : quadrature tolerances for L evaluations (loosen for speed)

    Returns
    -------
    times : np.ndarray of shape (N,) - time grid t_0,...,t_{N-1}
    b     : np.ndarray of shape (N,) - stopping boundary
    """
    gamma = find_gamma(mu, sigma, beta, law)
    m = law.median
    if verbose:
        print(f"  gamma = {gamma:.6f},  m = {m:.6f}")

    dt = T / N
    times = np.array([i * dt for i in range(N)])

    b = np.full(N, np.nan)

    # Terminal initialization: b(T-) = gamma
    b[N - 1] = gamma

    # ------------------------------------------------------------------ #
    # Backward induction: from i = N-2 down to 0                          #
    # ------------------------------------------------------------------ #
    for i in range(N - 2, -1, -1):
        t_i = times[i]
        future_idx = np.arange(i + 1, N)
        future_times = times[future_idx]
        b_fut = b[future_idx]

        J_i = lambda x: compute_J(t_i, x, T, mu, sigma, beta, law)
        G_i = lambda x: G_analytical(x, law)

        def rhs(bv: float) -> float:
            return _time_integral_approx_L(
                t_i, bv, future_times, b_fut, mu, sigma, beta, law,
                T=T, gamma=gamma, epsabs=epsabs, epsrel=epsrel,
            )

        def residual(bv: float) -> float:
            return J_i(bv) - G_i(bv) - rhs(bv)

        # b should lie in [m, gamma], increasing toward gamma as t -> T
        b_lo_bracket = m + 1e-6
        b_hi_bracket = gamma - 1e-8

        try:
            r_lo = residual(b_lo_bracket)
            r_hi = residual(b_hi_bracket)
            if r_lo * r_hi <= 0:
                b[i] = brentq(residual, b_lo_bracket, b_hi_bracket, xtol=1e-7, rtol=1e-7)
            else:
                # Fallback: use previous step value
                b[i] = b[i + 1] if i + 1 < N else gamma
        except Exception:
            b[i] = b[i + 1] if i + 1 < N else gamma

        if verbose and i % 10 == 0:
            print(f"  t[{i:3d}] = {t_i:.4f}: b = {b[i]:.6f}")

    return times, b


def compute_value_function_neg(
    t: float,
    x: float,
    times: np.ndarray,
    b: np.ndarray,
    T: float,
    mu: float,
    sigma: float,
    beta: float,
    law: ExponentialAspirationLaw,
) -> float:
    """
    Recover the reduced value function V(t, x) = J(t, x) - ∫_t^T L(s-t, x, b(s)) ds.
    """
    future_idx = times > t
    future_times = times[future_idx]
    b_fut = b[future_idx]

    from src.loss_functions import find_gamma
    _gamma = find_gamma(mu, sigma, beta, law)
    J_val = compute_J(t, x, T, mu, sigma, beta, law)
    integral = _time_integral_approx_L(
        t, x, future_times, b_fut, mu, sigma, beta, law,
        T=T, gamma=_gamma,
    )
    return J_val - integral
