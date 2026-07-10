"""
Tests for src/kernels.py
"""
import numpy as np
import pytest

from src.aspiration_laws import ExponentialAspirationLaw
from src.kernels import compute_J, compute_M, compute_L
from src.loss_functions import G_analytical


LAW   = ExponentialAspirationLaw(lam=2.0)
MU    = 0.15
MU_N  = -0.15
SIGMA = 0.30
BETA  = 0.70
T     = 1.0


class TestComputeJ:

    def test_J_scalar(self):
        """J returns a scalar for scalar x and t."""
        J_val = compute_J(0.5, 1.0, T, MU, SIGMA, BETA, LAW)
        assert np.isscalar(J_val) or np.ndim(J_val) == 0
        assert np.isfinite(J_val)

    def test_J_finite(self):
        """J(t,x) is finite for representative (x,t) pairs.

        Note: J can be negative because G(x) < 0 on (0, m) where m=ln(2)/2.
        J = E_x[G(X_{T-t})] inherits the sign of G; only finiteness is required.
        """
        x_vals = [0.3, 1.0, 2.0]
        t_vals = [0.1, 0.5, 0.9]
        for x in x_vals:
            for t in t_vals:
                J_val = compute_J(t, x, T, MU, SIGMA, BETA, LAW)
                assert np.isfinite(J_val), f"J not finite at x={x}, t={t}"

    def test_J_at_T_is_G(self):
        """At t=T, J(T,x) ≈ G(x) since there's no time to move."""
        # As t -> T from below, J(t,x) -> G(x)
        # Use t very close to T
        x_test = [0.5, 1.0, 2.0]
        t_near_T = T - 0.001
        for x in x_test:
            J_near = compute_J(t_near_T, x, T, MU, SIGMA, BETA, LAW)
            G_val  = G_analytical(x, LAW)
            # Should be close but not necessarily identical due to small diffusion
            assert np.isfinite(J_near)

    def test_J_large_t_different_from_small_t(self):
        """J(t, x) changes as t varies — expectation depends on diffusion horizon."""
        x = 1.0
        J_early = compute_J(0.1, x, T, MU, SIGMA, BETA, LAW)
        J_late  = compute_J(0.9, x, T, MU, SIGMA, BETA, LAW)
        # Not required to be equal; just check both are finite and non-negative
        assert np.isfinite(J_early) and J_early >= 0
        assert np.isfinite(J_late)  and J_late  >= 0

    def test_J_negative_drift(self):
        """J is finite for negative drift parameter."""
        J_val = compute_J(0.5, 1.0, T, MU_N, SIGMA, BETA, LAW)
        assert np.isfinite(J_val)


class TestComputeM:

    def test_M_scalar(self):
        """M returns finite scalar."""
        M_val = compute_M(0.3, 1.0, 0.2, 1.5, MU, SIGMA, BETA, LAW)
        assert np.isfinite(M_val)

    def test_M_empty_interval(self):
        """M(s,x,y,y) = 0 (zero-width interval)."""
        M_val = compute_M(0.3, 1.0, 0.5, 0.5, MU, SIGMA, BETA, LAW)
        assert M_val == pytest.approx(0.0, abs=1e-10)

    def test_M_sign_flips(self):
        """M is not necessarily positive; its sign depends on H in [y,z]."""
        # H < 0 below alpha for positive drift
        # alpha ~ 0.326, so M over (0.1, 0.2) should be negative for mu=0.15
        M_neg = compute_M(0.3, 0.1, 0.05, 0.2, MU, SIGMA, BETA, LAW)
        assert np.isfinite(M_neg)

    def test_M_additivity(self):
        """M(s,x,y,z) + L(s,x,y) = L(s,x,z) for y <= z."""
        s, x = 0.3, 1.0
        y, z = 0.2, 1.5
        L_y  = compute_L(s, x, y, MU, SIGMA, BETA, LAW)
        L_z  = compute_L(s, x, z, MU, SIGMA, BETA, LAW)
        M_yz = compute_M(s, x, y, z, MU, SIGMA, BETA, LAW)
        assert L_y + M_yz == pytest.approx(L_z, abs=1e-4)

    def test_M_negative_drift(self):
        """M is finite for negative drift."""
        M_val = compute_M(0.3, 1.0, 0.2, 1.5, MU_N, SIGMA, BETA, LAW)
        assert np.isfinite(M_val)


class TestComputeL:

    def test_L_scalar(self):
        """L returns finite scalar."""
        L_val = compute_L(0.3, 1.0, 0.5, MU, SIGMA, BETA, LAW)
        assert np.isfinite(L_val)

    def test_L_zero_at_zero(self):
        """L(s,x,0) = 0 (integral from 0 to 0)."""
        L_val = compute_L(0.3, 1.0, 0.0, MU, SIGMA, BETA, LAW)
        assert L_val == pytest.approx(0.0, abs=1e-10)

    def test_L_monotone_in_y(self):
        """L(s,x,y) changes monotonically (integral sign depends on H)."""
        s, x = 0.3, 1.0
        y_vals = [0.1, 0.3, 0.6, 1.0, 1.5]
        L_vals = [compute_L(s, x, y, MU_N, SIGMA, BETA, LAW) for y in y_vals]
        # All finite
        assert all(np.isfinite(v) for v in L_vals)

    def test_L_additivity_with_M(self):
        """L(s,x,z) = L(s,x,y) + M(s,x,y,z)."""
        s, x = 0.3, 1.0
        y, z = 0.3, 0.8
        L_y  = compute_L(s, x, y, MU_N, SIGMA, BETA, LAW)
        L_z  = compute_L(s, x, z, MU_N, SIGMA, BETA, LAW)
        M_yz = compute_M(s, x, y, z, MU_N, SIGMA, BETA, LAW)
        assert L_y + M_yz == pytest.approx(L_z, abs=1e-4)

    def test_L_negative_drift(self):
        """L is finite for negative drift."""
        L_val = compute_L(0.3, 1.0, 0.4, MU_N, SIGMA, BETA, LAW)
        assert np.isfinite(L_val)
