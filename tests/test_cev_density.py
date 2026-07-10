"""
Tests for src/cev_density.py
"""
import numpy as np
import pytest
from scipy.integrate import quad

from src.cev_density import cev_density, T_scaling, check_density_normalization


MU_POS = 0.15
MU_NEG = -0.15
SIGMA  = 0.30
BETA   = 0.70


class TestTScaling:

    def test_T_scaling_positive_mu(self):
        """T_t is positive for mu > 0, t > 0."""
        t = 0.5
        Tt = T_scaling(MU_POS, BETA, t)
        assert Tt > 0

    def test_T_scaling_negative_mu(self):
        """T_t is positive for mu < 0, t > 0, |2 mu beta t| < 1."""
        t = 0.3
        Tt = T_scaling(MU_NEG, BETA, t)
        assert Tt > 0

    def test_T_scaling_small_mu(self):
        """For mu -> 0, T_t -> t (limit of (exp(2mu beta t)-1)/(2mu beta) as mu->0)."""
        t = 0.5
        Tt_small = T_scaling(1e-8, BETA, t)
        assert Tt_small == pytest.approx(t, rel=1e-5)

    def test_T_scaling_monotone(self):
        """T_t should be monotone increasing in t for mu > 0."""
        t_arr = np.linspace(0.01, 0.9, 10)
        Tt_arr = np.array([T_scaling(MU_POS, BETA, ti) for ti in t_arr])
        assert np.all(np.diff(Tt_arr) > 0)


class TestCEVDensity:

    def test_density_nonnegative(self):
        """Density is non-negative."""
        x, t = 1.0, 0.5
        z = np.linspace(0.01, 5.0, 50)
        dens = cev_density(x, z, t, MU_POS, SIGMA, BETA)
        assert np.all(dens >= 0)

    def test_density_normalization_positive_mu(self):
        """∫_0^∞ p^mu(x,z;t) dz ≈ 1 for representative (x,t)."""
        for x, t in [(0.5, 0.3), (1.0, 0.5), (2.0, 0.8)]:
            result = check_density_normalization(x, t, MU_POS, SIGMA, BETA)
            assert result["norm"] == pytest.approx(1.0, abs=0.05), \
                f"Norm={result['norm']:.4f} for x={x}, t={t}"

    def test_density_normalization_negative_mu(self):
        """∫_0^∞ p^mu(x,z;t) dz ≈ 1 for mu < 0."""
        for x, t in [(0.5, 0.3), (1.0, 0.5)]:
            result = check_density_normalization(x, t, MU_NEG, SIGMA, BETA)
            assert result["norm"] == pytest.approx(1.0, abs=0.05), \
                f"Norm={result['norm']:.4f} for x={x}, t={t}"

    def test_density_small_t_peaked(self):
        """At small t, density is more concentrated near x (lognormal-like)."""
        x = 1.0
        z_grid = np.linspace(0.01, 5.0, 300)
        dens_small = cev_density(x, z_grid, 0.05, MU_POS, SIGMA, BETA)
        dens_large = cev_density(x, z_grid, 0.8, MU_POS, SIGMA, BETA)
        # Peak of small-t density should be sharper (higher max)
        assert np.max(dens_small) > np.max(dens_large)

    def test_density_positive_for_interior(self):
        """Density is strictly positive in the interior of state space."""
        x, t = 1.0, 0.5
        z = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
        dens = cev_density(x, z, t, MU_POS, SIGMA, BETA)
        assert np.all(dens > 0)

    def test_density_shape_scalar(self):
        """Density works for scalar z input."""
        x, t, z = 1.0, 0.5, 1.0
        dens = cev_density(x, z, t, MU_POS, SIGMA, BETA)
        assert np.isscalar(dens) or dens.ndim == 0
        assert dens > 0

    def test_density_different_betas(self):
        """Density behaves correctly for different beta values."""
        x, t = 1.0, 0.5
        z = np.linspace(0.01, 4.0, 50)
        for beta in [0.3, 0.7, 1.0, 1.5]:
            dens = cev_density(x, z, t, MU_POS, SIGMA, beta)
            assert np.all(dens >= 0), f"Negative density for beta={beta}"

    def test_density_normalization_various_betas(self):
        """Density integrates to ~1 for various beta values."""
        x, t = 1.0, 0.5
        for beta in [0.3, 0.7, 1.0]:
            result = check_density_normalization(x, t, MU_POS, SIGMA, beta, tol=0.08)
            assert result["pass"], f"Norm={result['norm']:.4f} for beta={beta}"
