"""
Aspiration law interfaces for the CEV optimal prediction problem.

The aspiration law is the distribution of the target random variable l.
For the exponential law: l ~ Exp(lambda), F(x) = 1 - exp(-lambda*x).
"""

import numpy as np
from scipy.optimize import brentq


class ExponentialAspirationLaw:
    """
    Exponential aspiration law: l ~ Exp(lambda).

    CDF: F(x) = 1 - exp(-lambda * x),  x >= 0
    PDF: f(x) = lambda * exp(-lambda * x)
    Mean: E[l] = 1/lambda
    Median: m = M = log(2) / lambda
    """

    def __init__(self, lam: float = 2.0) -> None:
        """
        Parameters
        ----------
        lam : float
            Rate parameter lambda > 0 of the exponential distribution.
        """
        if lam <= 0:
            raise ValueError(f"lambda must be positive, got {lam}")
        self.lam = lam

    def F(self, x: np.ndarray | float) -> np.ndarray | float:
        """CDF F(x) = 1 - exp(-lambda*x)."""
        x = np.asarray(x, dtype=float)
        return 1.0 - np.exp(-self.lam * x)

    def f(self, x: np.ndarray | float) -> np.ndarray | float:
        """PDF f(x) = lambda * exp(-lambda*x)."""
        x = np.asarray(x, dtype=float)
        return self.lam * np.exp(-self.lam * x)

    @property
    def E_l(self) -> float:
        """Mean of the aspiration: E[l] = 1/lambda."""
        return 1.0 / self.lam

    @property
    def median(self) -> float:
        """Median of the aspiration: m = M = log(2)/lambda."""
        return np.log(2.0) / self.lam
