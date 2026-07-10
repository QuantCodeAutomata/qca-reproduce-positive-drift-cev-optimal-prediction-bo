"""
Tests for src/boundary_solver_positive.py and src/boundary_solver_negative.py.
"""
import numpy as np
import pytest

from src.aspiration_laws import ExponentialAspirationLaw
from src.loss_functions import find_alpha, find_gamma, G_analytical
from src.boundary_solver_positive import solve_positive_boundary, compute_value_function_pos
from src.boundary_solver_negative import solve_negative_boundary, compute_value_function_neg


LAW   = ExponentialAspirationLaw(lam=2.0)
MU    =  0.15
MU_N  = -0.15
SIGMA =  0.30
BETA  =  0.70
T     =  1.0
N     =  30  # small grid for fast tests


class TestPositiveBoundary:

    @pytest.fixture(scope="class")
    def solved(self):
        """Solve positive boundary once for reuse."""
        times, a, c = solve_positive_boundary(
            MU, SIGMA, BETA, T, LAW, N=N, verbose=False
        )
        return times, a, c

    def test_shapes(self, solved):
        """times, a, c have consistent shapes."""
        times, a, c = solved
        assert len(times) == len(a) == len(c)

    def test_times_in_range(self, solved):
        """All time points are in [0, T)."""
        times, a, c = solved
        assert times[0] >= 0
        assert times[-1] < T + 1e-8

    def test_a_in_valid_range(self, solved):
        """Lower boundary a(t) in [alpha, m] for all t."""
        times, a, c = solved
        alpha = find_alpha(MU, SIGMA, BETA, LAW)
        m = LAW.median
        assert np.all(a >= alpha - 1e-3), f"a below alpha: min(a)={a.min():.5f}"
        assert np.all(a <= m + 1e-3),    f"a above m: max(a)={a.max():.5f}"

    def test_c_above_m(self, solved):
        """Upper boundary c(t) >= m for all t."""
        times, a, c = solved
        m = LAW.median
        assert np.all(c >= m - 1e-3), f"c below m: min(c)={c.min():.5f}"

    def test_a_less_than_c(self, solved):
        """a(t) < c(t) always (band is non-empty)."""
        times, a, c = solved
        assert np.all(a < c), "Stopping band collapsed: a >= c somewhere"

    def test_terminal_a(self, solved):
        """a(T-) ≈ alpha."""
        times, a, c = solved
        alpha = find_alpha(MU, SIGMA, BETA, LAW)
        assert a[-1] == pytest.approx(alpha, abs=0.02)

    def test_terminal_c_large(self, solved):
        """c(T-) is large (asymptotic divergence)."""
        times, a, c = solved
        m = LAW.median
        assert c[-1] > 5 * m, f"c(T-) too small: {c[-1]:.3f}"

    def test_a0_known_value(self, solved):
        """a(0) ≈ 0.34234 (paper target)."""
        times, a, c = solved
        assert a[0] == pytest.approx(0.34234, abs=0.02)

    def test_c0_known_value(self, solved):
        """c(0) ≈ 2.67449 (paper target)."""
        times, a, c = solved
        assert c[0] == pytest.approx(2.67449, abs=0.3)

    def test_value_matching_at_a(self, solved):
        """V(t, a(t)) ≈ G(a(t)) — value matching at lower boundary."""
        times, a, c = solved
        idx = 0  # t=0
        t0  = times[idx]
        V   = compute_value_function_pos(t0, a[idx], times, a, c, T, MU, SIGMA, BETA, LAW)
        G   = G_analytical(a[idx], LAW)
        assert abs(V - G) < 0.05, f"Value mismatch at lower boundary: V={V:.5f}, G={G:.5f}"

    def test_value_matching_at_c(self, solved):
        """V(t, c(t)) ≈ G(c(t)) — value matching at upper boundary."""
        times, a, c = solved
        idx = 0
        t0  = times[idx]
        V   = compute_value_function_pos(t0, c[idx], times, a, c, T, MU, SIGMA, BETA, LAW)
        G   = G_analytical(c[idx], LAW)
        assert abs(V - G) < 0.05, f"Value mismatch at upper boundary: V={V:.5f}, G={G:.5f}"


class TestNegativeBoundary:

    @pytest.fixture(scope="class")
    def solved(self):
        """Solve negative boundary once for reuse."""
        times, b = solve_negative_boundary(
            MU_N, SIGMA, BETA, T, LAW, N=N, verbose=False
        )
        return times, b

    def test_shapes(self, solved):
        """times and b have consistent shapes."""
        times, b = solved
        assert len(times) == len(b)

    def test_times_in_range(self, solved):
        """All time points are in [0, T)."""
        times, b = solved
        assert times[0] >= 0
        assert times[-1] < T + 1e-8

    def test_b_in_valid_range(self, solved):
        """Boundary b(t) in [m, gamma] for all t."""
        times, b = solved
        gamma = find_gamma(MU_N, SIGMA, BETA, LAW)
        m = LAW.median
        assert np.all(b >= m - 1e-3),    f"b below m: min(b)={b.min():.5f}"
        assert np.all(b <= gamma + 1e-3), f"b above gamma: max(b)={b.max():.5f}"

    def test_b_non_decreasing(self, solved):
        """b(t) is non-decreasing (increasing toward T)."""
        times, b = solved
        diffs = np.diff(b)
        assert np.all(diffs >= -1e-4), f"b not monotone: min diff={diffs.min():.4e}"

    def test_terminal_b(self, solved):
        """b(T-) ≈ gamma."""
        times, b = solved
        gamma = find_gamma(MU_N, SIGMA, BETA, LAW)
        assert b[-1] == pytest.approx(gamma, abs=0.02)

    def test_b0_known_value(self, solved):
        """b(0) ≈ 0.35320 (paper target)."""
        times, b = solved
        assert b[0] == pytest.approx(0.35320, abs=0.015)

    def test_value_matching_at_b(self, solved):
        """V(t, b(t)) ≈ G(b(t)) — value matching at boundary."""
        times, b = solved
        idx = 0
        t0  = times[idx]
        V   = compute_value_function_neg(t0, b[idx], times, b, T, MU_N, SIGMA, BETA, LAW)
        G   = G_analytical(b[idx], LAW)
        assert abs(V - G) < 0.05, f"Value mismatch: V={V:.5f}, G={G:.5f}"

    def test_stopping_region_one_sided(self, solved):
        """V < G for x above b(0) (continuation region)."""
        times, b = solved
        t0  = times[0]
        x_c = b[0] * 2.0  # well inside continuation
        V   = compute_value_function_neg(t0, x_c, times, b, T, MU_N, SIGMA, BETA, LAW)
        G   = G_analytical(x_c, LAW)
        # V should be strictly less than G in continuation region
        assert V <= G + 0.02, f"V={V:.5f} not <= G={G:.5f} in continuation"
