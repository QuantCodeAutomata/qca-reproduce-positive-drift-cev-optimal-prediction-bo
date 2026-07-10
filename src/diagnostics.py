"""
Diagnostics and validation utilities for the CEV optimal prediction experiments.

Provides:
- Loss-function reduction identity checks
- CEV density normalization checks
- Boundary residual computation
- Value-function consistency (V=G on stopping set, V<G on continuation)
- Monotonicity verification for computed boundaries
"""

import numpy as np
from scipy.integrate import quad

from src.aspiration_laws import ExponentialAspirationLaw
from src.loss_functions import G_analytical, G_integral, G_tail, H
from src.cev_density import cev_density, check_density_normalization
from src.kernels import compute_J, compute_M, compute_L


# ------------------------------------------------------------------ #
# 1. Loss function identity checks                                     #
# ------------------------------------------------------------------ #

def check_G_representations(
    x_grid: np.ndarray,
    law: ExponentialAspirationLaw,
    tol: float = 1e-5,
) -> dict:
    """
    Verify that three representations of G agree numerically:
    (a) Closed form: G(x) = x/2 - 1/λ + exp(-λx)/λ
    (b) Integral:    G(x) = ∫_0^x (F(y)-1/2) dy
    (c) Tail form:   G(x) = x/2 - E[l] + ∫_x^∞ (1-F(y)) dy

    Returns a dict with max absolute differences and pass/fail flags.
    """
    g_analytical = np.array([G_analytical(x, law) for x in x_grid])
    g_integral   = np.array([G_integral(x, law)   for x in x_grid])
    g_tail       = np.array([G_tail(x, law)        for x in x_grid])

    diff_int  = np.max(np.abs(g_analytical - g_integral))
    diff_tail = np.max(np.abs(g_analytical - g_tail))

    return {
        "max_diff_integral": diff_int,
        "max_diff_tail":     diff_tail,
        "integral_ok":       diff_int  < tol,
        "tail_ok":           diff_tail < tol,
    }


def check_reduction_identity(
    x_grid: np.ndarray,
    law: ExponentialAspirationLaw,
    tol: float = 1e-5,
) -> dict:
    """
    Verify E|x-l| = 2*G(x) + E[l] for each x in x_grid.

    E|x-l| is computed by numerical integration over l.
    """
    E_l = law.E_l
    errors = []
    for x in x_grid:
        # E|x-l| = ∫_0^∞ |x-y| f(y) dy = ∫_0^x (x-y)f(y)dy + ∫_x^∞ (y-x)f(y)dy
        def integrand(y: float) -> float:
            return abs(x - y) * law.f(y)
        E_abs, _ = quad(integrand, 0.0, 50.0, limit=200)
        lhs = E_abs
        rhs = 2.0 * G_analytical(x, law) + E_l
        errors.append(abs(lhs - rhs))

    return {
        "max_error": np.max(errors),
        "mean_error": np.mean(errors),
        "pass": np.max(errors) < tol,
    }


# ------------------------------------------------------------------ #
# 2. H sign-pattern checks                                             #
# ------------------------------------------------------------------ #

def check_H_sign_pattern_positive(
    alpha: float,
    m: float,
    mu: float,
    sigma: float,
    beta: float,
    law: ExponentialAspirationLaw,
    n_pts: int = 50,
) -> dict:
    """
    Verify H < 0 on (0, alpha) and H > 0 on (alpha, m) for positive drift.
    """
    x_below = np.linspace(1e-4, alpha * 0.999, n_pts)
    x_above = np.linspace(alpha * 1.001, m * 0.999, n_pts)
    h_below = H(x_below, mu, sigma, beta, law)
    h_above = H(x_above, mu, sigma, beta, law)
    return {
        "all_H_below_alpha_negative": bool(np.all(h_below < 0)),
        "all_H_above_alpha_positive": bool(np.all(h_above > 0)),
        "max_H_below": float(np.max(h_below)),
        "min_H_above": float(np.min(h_above)),
    }


def check_H_sign_pattern_negative(
    m: float,
    gamma: float,
    mu: float,
    sigma: float,
    beta: float,
    law: ExponentialAspirationLaw,
    n_pts: int = 50,
    x_hi: float = 5.0,
) -> dict:
    """
    Verify H > 0 on (m, gamma) and H < 0 on (gamma, inf) for negative drift.
    """
    x_between = np.linspace(m * 1.001, gamma * 0.999, n_pts)
    x_above   = np.linspace(gamma * 1.001, x_hi, n_pts)
    h_between = H(x_between, mu, sigma, beta, law)
    h_above   = H(x_above, mu, sigma, beta, law)
    return {
        "all_H_between_positive": bool(np.all(h_between > 0)),
        "all_H_above_gamma_negative": bool(np.all(h_above < 0)),
        "min_H_between": float(np.min(h_between)),
        "max_H_above":   float(np.max(h_above)),
    }


# ------------------------------------------------------------------ #
# 3. Density normalization                                             #
# ------------------------------------------------------------------ #

def check_density_norm_batch(
    x_vals: list,
    t_vals: list,
    mu: float,
    sigma: float,
    beta: float,
    z_max: float = 50.0,
) -> dict:
    """
    Check density normalization ∫ p^(μ)(x,z;t) dz ≈ 1 for multiple (x,t) pairs.
    """
    results = []
    for x in x_vals:
        for t in t_vals:
            chk = check_density_normalization(x, t, mu, sigma, beta, z_max=z_max)
            results.append({"x": x, "t": t, "norm": chk["norm"], "error": chk["error"]})
    max_err = max(r["error"] for r in results)
    return {"results": results, "max_error": max_err, "pass": max_err < 0.05}


# ------------------------------------------------------------------ #
# 4. Boundary residual diagnostics                                     #
# ------------------------------------------------------------------ #

def compute_positive_boundary_residuals(
    times: np.ndarray,
    a: np.ndarray,
    c: np.ndarray,
    T: float,
    mu: float,
    sigma: float,
    beta: float,
    law: ExponentialAspirationLaw,
) -> dict:
    """
    Compute residuals of the positive-drift free-boundary equations at each time node.

    Residual at (t_i, a_i): |J(t_i,a_i) - G(a_i) - integral_approx|
    Residual at (t_i, c_i): |J(t_i,c_i) - G(c_i) - integral_approx|
    """
    from src.boundary_solver_positive import _time_integral_approx

    N = len(times)
    res_a = np.full(N, np.nan)
    res_c = np.full(N, np.nan)

    for i in range(N - 1):
        t_i = times[i]
        future_idx = np.arange(i + 1, N)
        future_times = times[future_idx]
        a_fut = a[future_idx]
        c_fut = c[future_idx]

        from src.loss_functions import find_alpha
        _alpha = find_alpha(mu, sigma, beta, law)
        _c_last = float(c[-1])

        J_ai = compute_J(t_i, a[i], T, mu, sigma, beta, law)
        G_ai = G_analytical(a[i], law)
        int_a = _time_integral_approx(
            t_i, a[i], future_times, a_fut, c_fut, mu, sigma, beta, law,
            T=T, alpha=_alpha, c_terminal=_c_last,
        )
        res_a[i] = abs(J_ai - G_ai - int_a)

        J_ci = compute_J(t_i, c[i], T, mu, sigma, beta, law)
        G_ci = G_analytical(c[i], law)
        int_c = _time_integral_approx(
            t_i, c[i], future_times, a_fut, c_fut, mu, sigma, beta, law,
            T=T, alpha=_alpha, c_terminal=_c_last,
        )
        res_c[i] = abs(J_ci - G_ci - int_c)

    return {
        "res_a": res_a,
        "res_c": res_c,
        "max_res_a": float(np.nanmax(res_a)),
        "max_res_c": float(np.nanmax(res_c)),
        "mean_res_a": float(np.nanmean(res_a)),
        "mean_res_c": float(np.nanmean(res_c)),
    }


def compute_negative_boundary_residuals(
    times: np.ndarray,
    b: np.ndarray,
    T: float,
    mu: float,
    sigma: float,
    beta: float,
    law: ExponentialAspirationLaw,
) -> dict:
    """
    Compute residuals of the negative-drift free-boundary equation at each time node.
    """
    from src.boundary_solver_negative import _time_integral_approx_L

    N = len(times)
    res_b = np.full(N, np.nan)

    for i in range(N - 1):
        t_i = times[i]
        future_idx = np.arange(i + 1, N)
        future_times = times[future_idx]
        b_fut = b[future_idx]

        J_bi = compute_J(t_i, b[i], T, mu, sigma, beta, law)
        G_bi = G_analytical(b[i], law)
        int_b = _time_integral_approx_L(t_i, b[i], future_times, b_fut, mu, sigma, beta, law)
        res_b[i] = abs(J_bi - G_bi - int_b)

    return {
        "res_b": res_b,
        "max_res": float(np.nanmax(res_b)),
        "mean_res": float(np.nanmean(res_b)),
    }


# ------------------------------------------------------------------ #
# 5. Monotonicity checks                                               #
# ------------------------------------------------------------------ #

def check_boundary_monotonicity_positive(
    times: np.ndarray,
    a: np.ndarray,
    c: np.ndarray,
) -> dict:
    """
    For positive drift: a(t) should decrease and c(t) should increase as t -> T.
    (i.e., as index increases, a decreases and c increases.)
    """
    valid = ~(np.isnan(a) | np.isnan(c))
    a_v = a[valid]
    c_v = c[valid]
    t_v = times[valid]
    # a should be non-decreasing going forward (backward in time = forward in index)
    # Convention: times[0]=0, times[N-1]=T-dt, a(0) >= a(T-)=alpha
    a_mono = bool(np.all(np.diff(a_v) <= 1e-5))  # a non-increasing in t
    c_mono = bool(np.all(np.diff(c_v) >= -1e-5))  # c non-decreasing in t
    return {
        "a_non_increasing": a_mono,
        "c_non_decreasing": c_mono,
        "a_min": float(np.min(a_v)),
        "a_max": float(np.max(a_v)),
        "c_min": float(np.min(c_v)),
        "c_max": float(np.max(c_v)),
    }


def check_boundary_monotonicity_negative(
    times: np.ndarray,
    b: np.ndarray,
) -> dict:
    """
    For negative drift: b(t) should be non-decreasing as t increases toward T.
    """
    valid = ~np.isnan(b)
    b_v = b[valid]
    b_mono = bool(np.all(np.diff(b_v) >= -1e-6))
    return {
        "b_non_decreasing": b_mono,
        "b_min": float(np.min(b_v)),
        "b_max": float(np.max(b_v)),
    }
