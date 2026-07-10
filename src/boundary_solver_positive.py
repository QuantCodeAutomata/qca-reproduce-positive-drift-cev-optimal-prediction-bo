"""
Backward induction solver for the positive-drift CEV optimal prediction problem.

Solves the coupled free-boundary integral equations for the two-sided stopping band
D = {(t,x): a(t) <= x <= c(t)}.

Free-boundary conditions at x = a(t) and x = c(t):
  V(t, a(t)) = G(a(t))
  V(t, c(t)) = G(c(t))

Using the integral representation:
  V(t, x) = J(t, x) - ∫_{t}^{T} M(s-t, x, a(s), c(s)) ds

The free-boundary equations become:
  J(t_i, a_i) - G(a_i) = ∫_{t_i}^{T} M(s-t_i, a_i, a(s), c(s)) ds
  J(t_i, c_i) - G(c_i) = ∫_{t_i}^{T} M(s-t_i, c_i, a(s), c(s)) ds

Backward induction: solve from t_{N-1} = T-dt backward to t_0 = 0.
Terminal conditions: a(T-) = alpha, c(T-) = ∞ (handled as C_init).
"""

import numpy as np
from scipy.optimize import brentq
from typing import Tuple

from src.aspiration_laws import ExponentialAspirationLaw
from src.loss_functions import G_analytical, H, find_alpha
from src.kernels import compute_J, compute_M


def _time_integral_approx(
    t_i: float,
    x: float,
    future_times: np.ndarray,
    a_future: np.ndarray,
    c_future: np.ndarray,
    mu: float,
    sigma: float,
    beta: float,
    law: ExponentialAspirationLaw,
    T: float = 1.0,
    alpha: float = None,
    c_terminal: float = 50.0,
    epsabs: float = 1e-6,
    epsrel: float = 1e-6,
) -> float:
    """
    Approximate ∫_{t_i}^{T} M(s-t_i, x, a(s), c(s)) ds by trapezoidal rule
    over the future boundary nodes {future_times[j], a_future[j], c_future[j]}.

    The last future node is treated as the terminal segment. A terminal correction
    adds the [t_{N-1}, T] segment using boundary values (alpha, c_terminal).

    Parameters
    ----------
    epsabs, epsrel : quadrature tolerances forwarded to compute_M (loosen for speed)
    """
    n = len(future_times)
    if n == 0:
        return 0.0

    # Compute M at each future time node
    m_vals = np.zeros(n)
    for j in range(n):
        s = future_times[j] - t_i
        if s > 1e-12:
            m_vals[j] = compute_M(
                s, x, a_future[j], c_future[j], mu, sigma, beta, law,
                limit=80, epsabs=epsabs, epsrel=epsrel,
            )

    # Trapezoidal rule: weights based on future_times spacing
    # Prepend t_i with M=0 (boundary equation has M=0 at s=0)
    all_times = np.concatenate([[t_i], future_times])
    all_m = np.concatenate([[0.0], m_vals])

    # Terminal correction: include [t_{N-1}, T] segment
    # Use boundary values (alpha, c_terminal) as approximation for the terminal limits
    t_last = future_times[-1]
    dt_terminal = T - t_last
    if dt_terminal > 1e-14 and alpha is not None:
        s_terminal = T - t_i
        m_terminal = compute_M(
            s_terminal, x, alpha, c_terminal, mu, sigma, beta, law,
            limit=80, epsabs=max(epsabs, 1e-5), epsrel=max(epsrel, 1e-5),
        )
        all_times = np.append(all_times, T)
        all_m = np.append(all_m, m_terminal)

    integral = np.trapz(all_m, all_times)
    return integral


def solve_positive_boundary(
    mu: float,
    sigma: float,
    beta: float,
    T: float,
    law: ExponentialAspirationLaw,
    N: int = 60,
    C_init: float = 30.0,
    a_lo: float = 1e-4,
    c_bracket_max: float = 200.0,
    verbose: bool = True,
    epsabs: float = 1e-6,
    epsrel: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the positive-drift free-boundary integral equations by backward induction.

    Parameters
    ----------
    mu, sigma, beta : CEV parameters (mu > 0 for positive drift)
    T               : horizon
    law             : aspiration law instance
    N               : number of time steps (excluding terminal T)
    C_init          : initial guess for upper boundary near T
    a_lo            : lower search bracket for a (default small positive)
    c_bracket_max   : maximum bracket for upper boundary search
    verbose         : print progress
    epsabs, epsrel  : quadrature tolerances for M evaluations (loosen for speed)

    Returns
    -------
    times : np.ndarray of shape (N,) - time grid t_0,...,t_{N-1}
    a     : np.ndarray of shape (N,) - lower boundary
    c     : np.ndarray of shape (N,) - upper boundary
    """
    alpha = find_alpha(mu, sigma, beta, law)
    m = law.median
    if verbose:
        print(f"  alpha = {alpha:.6f},  m = {m:.6f}")

    dt = T / N
    times = np.array([i * dt for i in range(N)])  # t_0, ..., t_{N-1}

    a = np.full(N, np.nan)
    c = np.full(N, np.nan)

    # ------------------------------------------------------------------ #
    # Terminal initialization: last node t_{N-1} = T - dt                 #
    # For small dt, the integral term is negligible, so:                   #
    #   J(T-dt, a) ≈ G(a) and J(T-dt, c) ≈ G(c)                          #
    # We initialize a = alpha, c = C_init and refine with the residual.   #
    # ------------------------------------------------------------------ #
    i_last = N - 1
    t_last = times[i_last]

    a[i_last] = alpha
    c[i_last] = C_init

    # Refine last step slightly
    J_a = compute_J(t_last, alpha, T, mu, sigma, beta, law)
    G_a = G_analytical(alpha, law)
    if verbose:
        print(f"  Last step init: t={t_last:.4f}, a={alpha:.5f}, "
              f"J(a)-G(a) = {J_a - G_a:.2e}")

    # ------------------------------------------------------------------ #
    # Backward induction: from i = N-2 down to 0                          #
    # ------------------------------------------------------------------ #
    for i in range(N - 2, -1, -1):
        t_i = times[i]
        future_idx = np.arange(i + 1, N)
        future_times = times[future_idx]
        a_fut = a[future_idx]
        c_fut = c[future_idx]

        J_i = lambda x: compute_J(t_i, x, T, mu, sigma, beta, law)
        G_i = lambda x: G_analytical(x, law)

        # RHS: time integral approximation (depends on x)
        def rhs(x: float) -> float:
            return _time_integral_approx(
                t_i, x, future_times, a_fut, c_fut, mu, sigma, beta, law,
                T=T, alpha=alpha, c_terminal=C_init,
                epsabs=epsabs, epsrel=epsrel,
            )

        # Residuals for lower and upper boundaries
        def res_a(av: float) -> float:
            return J_i(av) - G_i(av) - rhs(av)

        def res_c(cv: float) -> float:
            return J_i(cv) - G_i(cv) - rhs(cv)

        # --- Solve for a_i in [alpha, m] ---
        # a should be between alpha and m, decreasing as t approaches T
        a_lo_bracket = alpha * 0.99
        a_hi_bracket = m

        try:
            r_lo = res_a(a_lo_bracket)
            r_hi = res_a(a_hi_bracket)
            if r_lo * r_hi <= 0:
                a[i] = brentq(res_a, a_lo_bracket, a_hi_bracket, xtol=1e-6, rtol=1e-6)
            else:
                # Fallback: use previous step value or alpha
                a[i] = a[i + 1] if i + 1 < N else alpha
        except Exception:
            a[i] = a[i + 1] if i + 1 < N else alpha

        # --- Solve for c_i in [m, c_bracket_max] ---
        # c should be above m, increasing as t approaches T
        c_lo_bracket = m
        c_hi_bracket = min(c[i + 1] * 3.0 if i + 1 < N else C_init * 3.0, c_bracket_max)

        # Check that bracket is valid (sign change required)
        try:
            r_lo_c = res_c(c_lo_bracket + 1e-4)
            r_hi_c = res_c(c_hi_bracket)
            if r_lo_c * r_hi_c <= 0:
                c[i] = brentq(
                    res_c, c_lo_bracket + 1e-4, c_hi_bracket, xtol=1e-5, rtol=1e-5
                )
            else:
                # Extend bracket upward if needed
                found = False
                bracket = c_hi_bracket
                while bracket < c_bracket_max * 2:
                    bracket *= 2.0
                    r_br = res_c(bracket)
                    if r_lo_c * r_br <= 0:
                        c[i] = brentq(
                            res_c, c_lo_bracket + 1e-4, bracket, xtol=1e-5, rtol=1e-5
                        )
                        found = True
                        break
                if not found:
                    c[i] = c[i + 1] if i + 1 < N else C_init
        except Exception:
            c[i] = c[i + 1] if i + 1 < N else C_init

        if verbose and i % 10 == 0:
            print(f"  t[{i:3d}] = {t_i:.4f}: a = {a[i]:.6f}, c = {c[i]:.6f}")

    return times, a, c


def compute_value_function(
    t: float,
    x: float,
    times: np.ndarray,
    a: np.ndarray,
    c: np.ndarray,
    T: float,
    mu: float,
    sigma: float,
    beta: float,
    law: ExponentialAspirationLaw,
) -> float:
    """
    Recover the reduced value function V(t, x) = J(t, x) - ∫_t^T M(s-t, x, a(s), c(s)) ds.

    Parameters
    ----------
    t       : current time in [0, T)
    x       : current state > 0
    times   : boundary time grid
    a, c    : boundary arrays
    T, mu, sigma, beta, law : model parameters
    """
    # Find future nodes
    future_idx = times > t
    future_times = times[future_idx]
    a_fut = a[future_idx]
    c_fut = c[future_idx]

    _alpha = find_alpha(mu, sigma, beta, law)
    c_last = float(c[-1]) if len(c) > 0 else 50.0
    J_val = compute_J(t, x, T, mu, sigma, beta, law)
    integral = _time_integral_approx(
        t, x, future_times, a_fut, c_fut, mu, sigma, beta, law,
        T=T, alpha=_alpha, c_terminal=c_last,
    )
    return J_val - integral


# Alias for backwards-compatibility with test imports
compute_value_function_pos = compute_value_function
