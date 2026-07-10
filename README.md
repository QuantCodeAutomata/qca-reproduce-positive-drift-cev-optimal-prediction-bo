# CEV Optimal Prediction: Positive and Negative Drift

Numerical reproduction of the finite-horizon optimal stopping problem for the
Constant Elasticity of Variance (CEV) process with exponential aspiration law,
as described in the paper on CEV optimal prediction.

## Problem Setup

The CEV process satisfies:
```
dX_t = μ X_t dt + σ X_t^{1+β} dW_t
```

The prediction error is minimised:
```
V_*(x) = inf_{0 ≤ τ ≤ T} E_x|X_τ - ℓ|
```
where ℓ ~ Exp(λ) is the aspiration random variable.

## Experiments

| ID   | Description                                       | Key Result                           |
|------|---------------------------------------------------|--------------------------------------|
| exp1 | Positive-drift two-sided stopping band            | α≈0.32581, a(0)≈0.34234, c(0)≈2.675 |
| exp2 | Negative-drift one-sided stopping boundary        | γ≈0.37414, b(0)≈0.35320              |
| exp3 | Comparative statics in β (positive drift)         | Wider band as β decreases            |
| exp4 | Comparative statics in β (negative drift)         | Higher boundary as β decreases       |
| exp5 | Core numerical validation (density, kernels, G)   | Confirms primitive accuracy          |

## Repository Structure

```
├── data/             # No empirical data (model-based experiments)
├── src/
│   ├── aspiration_laws.py         # Exponential law F, f, E[l], median
│   ├── loss_functions.py          # G, H, find_alpha, find_gamma
│   ├── cev_density.py             # CEV transition density p^(μ)
│   ├── kernels.py                 # Integral kernels J, L, M
│   ├── boundary_solver_positive.py # Backward induction for a(t), c(t)
│   ├── boundary_solver_negative.py # Backward induction for b(t)
│   └── diagnostics.py             # Normalization, residual, monotonicity checks
├── exp/
│   ├── exp1_positive_drift.py
│   ├── exp2_negative_drift.py
│   ├── exp3_comparative_statics_positive.py
│   ├── exp4_comparative_statics_negative.py
│   └── exp5_validation.py
├── tests/                         # pytest test suite
├── results/                       # Figures, JSON outputs, RESULTS.md
└── README.md
```

## Parameters (paper defaults)

| Parameter | Value  |
|-----------|--------|
| μ (pos)   | 0.15   |
| μ (neg)   | -0.15  |
| σ         | 0.30   |
| β         | 0.70   |
| λ         | 2.0    |
| T         | 1.0    |
| E[ℓ]      | 0.5    |
| m = M     | ln2/2 ≈ 0.34657 |

## Usage

### Run individual experiments

```bash
python exp/exp1_positive_drift.py
python exp/exp2_negative_drift.py
python exp/exp3_comparative_statics_positive.py
python exp/exp4_comparative_statics_negative.py
python exp/exp5_validation.py
```

### Run all tests

```bash
pytest tests/ -v
```

## Dependencies

- numpy, scipy, matplotlib
- pytest

## Mathematical Background

The reduced loss function is:
```
G(x) = ∫_0^x (F(y) - 1/2) dy = x/2 - 1/λ + exp(-λx)/λ
```

The generator term:
```
H(x) = μx(F(x) - 1/2) + (σ²/2)x^{2β+2}F'(x)
```

The CEV transition density uses the closed-form Bessel representation and is
evaluated in log-stabilized form via `scipy.special.ive`.
