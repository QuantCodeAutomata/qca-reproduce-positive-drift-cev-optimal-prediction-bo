"""
Tests for src/loss_functions.py
"""
import numpy as np
import pytest
from scipy.integrate import quad

from src.aspiration_laws import ExponentialAspirationLaw
from src.loss_functions import G_analytical, G_integral, G_tail, H, find_alpha, find_gamma


LAW = ExponentialAspirationLaw(lam=2.0)
MU_POS = 0.15
MU_NEG = -0.15
SIGMA  = 0.30
BETA   = 0.70


class TestGAnalytical:

    def test_G_at_zero(self):
        """G(0) = 0."""
        assert G_analytical(0.0, LAW) == pytest.approx(0.0, abs=1e-12)

    def test_G_minimum_at_median(self):
        """G has a minimum at m = log(2)/2."""
        m = LAW.median
        eps = 1e-4
        assert G_analytical(m, LAW) <= G_analytical(m - eps, LAW)
        assert G_analytical(m, LAW) <= G_analytical(m + eps, LAW)

    def test_G_nonnegative_outside_min(self):
        """G(x) >= G(m) for all x (convex-shaped with minimum at m)."""
        x = np.linspace(0.01, 8.0, 200)
        g_vals = G_analytical(x, LAW)
        g_min = np.min(g_vals)
        assert g_min >= G_analytical(LAW.median, LAW) - 1e-6

    def test_G_integral_vs_analytical(self):
        """G_integral agrees with G_analytical."""
        x_grid = np.linspace(0.1, 4.0, 30)
        for x in x_grid:
            g_a = G_analytical(x, LAW)
            g_i = G_integral(x, LAW)
            assert g_a == pytest.approx(g_i, abs=1e-5), f"Failed at x={x}"

    def test_G_tail_vs_analytical(self):
        """G_tail agrees with G_analytical."""
        x_grid = np.linspace(0.1, 4.0, 30)
        for x in x_grid:
            g_a = G_analytical(x, LAW)
            g_t = G_tail(x, LAW)
            assert g_a == pytest.approx(g_t, abs=1e-6), f"Failed at x={x}"

    def test_reduction_identity(self):
        """E|x-l| = 2*G(x) + E[l]."""
        E_l = LAW.E_l
        x_test = [0.5, 1.0, 2.0, 3.0]
        for x in x_test:
            E_abs, _ = quad(lambda y: abs(x - y) * LAW.f(y), 0, 50.0)
            rhs = 2.0 * G_analytical(x, LAW) + E_l
            assert E_abs == pytest.approx(rhs, abs=1e-4), f"Failed at x={x}"

    def test_G_vectorized(self):
        """G_analytical handles array input."""
        x = np.array([0.5, 1.0, 2.0])
        g = G_analytical(x, LAW)
        expected = np.array([G_analytical(xi, LAW) for xi in x])
        np.testing.assert_allclose(g, expected)

    def test_G_closed_form_values(self):
        """Known numerical check: G(0.5) = 0.5/2 - 0.5 + 0.5*exp(-1)."""
        x = 0.5
        expected = 0.5 / 2 - 0.5 + 0.5 * np.exp(-2 * 0.5)
        assert G_analytical(x, LAW) == pytest.approx(expected, rel=1e-10)


class TestH:

    def test_H_zero_at_alpha(self):
        """H(alpha) ≈ 0 for positive drift."""
        alpha = find_alpha(MU_POS, SIGMA, BETA, LAW)
        assert abs(H(alpha, MU_POS, SIGMA, BETA, LAW)) < 1e-8

    def test_H_zero_at_gamma(self):
        """H(gamma) ≈ 0 for negative drift."""
        gamma = find_gamma(MU_NEG, SIGMA, BETA, LAW)
        assert abs(H(gamma, MU_NEG, SIGMA, BETA, LAW)) < 1e-8

    def test_H_sign_positive_drift(self):
        """Positive drift: H < 0 below alpha, H > 0 above alpha (up to m)."""
        alpha = find_alpha(MU_POS, SIGMA, BETA, LAW)
        m = LAW.median
        assert H(alpha / 2, MU_POS, SIGMA, BETA, LAW) < 0
        assert H((alpha + m) / 2, MU_POS, SIGMA, BETA, LAW) > 0

    def test_H_sign_negative_drift(self):
        """Negative drift: H > 0 in (m, gamma), H < 0 above gamma."""
        gamma = find_gamma(MU_NEG, SIGMA, BETA, LAW)
        m = LAW.median
        assert H((m + gamma) / 2, MU_NEG, SIGMA, BETA, LAW) > 0
        assert H(gamma * 2.0, MU_NEG, SIGMA, BETA, LAW) < 0

    def test_H_vectorized(self):
        """H handles array input."""
        x = np.array([0.1, 0.5, 1.0])
        h_arr = H(x, MU_POS, SIGMA, BETA, LAW)
        expected = np.array([H(xi, MU_POS, SIGMA, BETA, LAW) for xi in x])
        np.testing.assert_allclose(h_arr, expected)


class TestAlphaGamma:

    def test_alpha_known_value(self):
        """alpha ≈ 0.32581 for paper parameters."""
        alpha = find_alpha(MU_POS, SIGMA, BETA, LAW)
        assert alpha == pytest.approx(0.32581, abs=1e-3)

    def test_gamma_known_value(self):
        """gamma ≈ 0.37414 for paper parameters."""
        gamma = find_gamma(MU_NEG, SIGMA, BETA, LAW)
        assert gamma == pytest.approx(0.37414, abs=1e-3)

    def test_alpha_below_m(self):
        """alpha < m for positive drift."""
        alpha = find_alpha(MU_POS, SIGMA, BETA, LAW)
        assert alpha < LAW.median

    def test_gamma_above_m(self):
        """gamma > m for negative drift."""
        gamma = find_gamma(MU_NEG, SIGMA, BETA, LAW)
        assert gamma > LAW.median

    def test_find_alpha_raises_for_negative_mu(self):
        """find_alpha raises ValueError for mu <= 0."""
        with pytest.raises(ValueError):
            find_alpha(-0.1, SIGMA, BETA, LAW)

    def test_find_gamma_raises_for_positive_mu(self):
        """find_gamma raises ValueError for mu >= 0."""
        with pytest.raises(ValueError):
            find_gamma(0.1, SIGMA, BETA, LAW)
