"""
Tests for src/aspiration_laws.py
"""
import numpy as np
import pytest
from src.aspiration_laws import ExponentialAspirationLaw


class TestExponentialAspirationLaw:

    def setup_method(self):
        self.law = ExponentialAspirationLaw(lam=2.0)

    def test_F_at_zero(self):
        """F(0) = 0."""
        assert self.law.F(0.0) == pytest.approx(0.0)

    def test_F_increasing(self):
        """CDF is strictly increasing."""
        x = np.linspace(0.01, 5.0, 100)
        F_vals = self.law.F(x)
        assert np.all(np.diff(F_vals) > 0)

    def test_F_range(self):
        """F(x) in [0,1] for x >= 0."""
        x = np.linspace(0, 10, 200)
        F_vals = self.law.F(x)
        assert np.all(F_vals >= 0)
        assert np.all(F_vals <= 1)

    def test_F_half_at_median(self):
        """F(m) = 0.5 at the median."""
        m = self.law.median
        assert self.law.F(m) == pytest.approx(0.5, abs=1e-12)

    def test_f_nonnegative(self):
        """PDF is non-negative."""
        x = np.linspace(0, 5, 100)
        assert np.all(self.law.f(x) >= 0)

    def test_f_integrates_to_1(self):
        """∫_0^∞ f(x) dx = 1 numerically."""
        from scipy.integrate import quad
        val, _ = quad(self.law.f, 0, 50)
        assert val == pytest.approx(1.0, abs=1e-4)

    def test_E_l_known_value(self):
        """E[l] = 1/lambda = 0.5."""
        assert self.law.E_l == pytest.approx(0.5, rel=1e-10)

    def test_median_known_value(self):
        """Median m = log(2)/2 ≈ 0.34657."""
        expected = np.log(2) / 2
        assert self.law.median == pytest.approx(expected, rel=1e-10)

    def test_invalid_lambda(self):
        """Constructor raises ValueError for lambda <= 0."""
        with pytest.raises(ValueError):
            ExponentialAspirationLaw(lam=-1.0)
        with pytest.raises(ValueError):
            ExponentialAspirationLaw(lam=0.0)

    def test_F_f_consistency(self):
        """F'(x) = f(x) via finite differences."""
        from scipy.misc import derivative
        x_test = np.array([0.5, 1.0, 2.0])
        for x in x_test:
            F_prime = derivative(self.law.F, x, dx=1e-6)
            assert F_prime == pytest.approx(self.law.f(x), rel=1e-4)

    def test_vectorized_F(self):
        """F is vectorized."""
        x = np.array([0.5, 1.0, 2.0])
        F_vals = self.law.F(x)
        expected = np.array([self.law.F(xi) for xi in x])
        np.testing.assert_allclose(F_vals, expected)

    def test_different_lambda(self):
        """Test with lambda=1 (standard exponential)."""
        law1 = ExponentialAspirationLaw(lam=1.0)
        assert law1.E_l == pytest.approx(1.0)
        assert law1.median == pytest.approx(np.log(2.0))
        assert law1.F(1.0) == pytest.approx(1 - np.exp(-1.0))
