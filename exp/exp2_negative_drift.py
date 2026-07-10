"""
Experiment 2: Negative-Drift CEV Optimal Prediction One-Sided Boundary.

Reproduces Example 5.2 from the paper:
  mu=-0.15, sigma=0.30, beta=0.70, lambda=2, T=1
  Expected: gamma ≈ 0.37414, b(0) ≈ 0.35320

Solves the scalar free-boundary integral equation by backward induction
and validates the one-sided stopping geometry.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json


class _NumpyEncoder(json.JSONEncoder):
    """Encode numpy scalars/arrays for JSON."""
    def default(self, obj):
        import numpy as _np
        if isinstance(obj, _np.integer): return int(obj)
        if isinstance(obj, _np.floating): return float(obj)
        if isinstance(obj, _np.bool_): return bool(obj)
        if isinstance(obj, _np.ndarray): return obj.tolist()
        return super().default(obj)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.aspiration_laws import ExponentialAspirationLaw
from src.loss_functions import G_analytical, H, find_gamma
from src.cev_density import check_density_normalization
from src.kernels import compute_J, compute_L
from src.boundary_solver_negative import solve_negative_boundary, compute_value_function_neg
from src.diagnostics import (
    check_G_representations,
    check_reduction_identity,
    check_H_sign_pattern_negative,
    check_density_norm_batch,
    check_boundary_monotonicity_negative,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_exp2() -> dict:
    """Run the full Experiment 2 pipeline and return summary metrics."""
    print("=" * 60)
    print("Experiment 2: Negative-Drift CEV Optimal Prediction")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Parameters                                                           #
    # ------------------------------------------------------------------ #
    mu     = -0.15
    sigma  =  0.30
    beta   =  0.70
    lam    =  2.0
    T      =  1.0
    N      =  60

    law = ExponentialAspirationLaw(lam=lam)
    m   = law.median
    E_l = law.E_l
    print(f"\nParameters: mu={mu}, sigma={sigma}, beta={beta}, lambda={lam}, T={T}")
    print(f"E[l] = {E_l:.5f}, median m = {m:.6f}")

    # ------------------------------------------------------------------ #
    # Step 1: Validate G representations                                   #
    # ------------------------------------------------------------------ #
    x_test = np.linspace(0.01, 5.0, 100)
    g_check = check_G_representations(x_test, law)
    print(f"\nG representation check:")
    print(f"  Max diff (integral):  {g_check['max_diff_integral']:.2e}  -> {'OK' if g_check['integral_ok'] else 'FAIL'}")

    # ------------------------------------------------------------------ #
    # Step 2: Validate reduction identity                                  #
    # ------------------------------------------------------------------ #
    x_test_small = np.linspace(0.1, 4.0, 20)
    red_check = check_reduction_identity(x_test_small, law)
    print(f"Reduction identity check: max error = {red_check['max_error']:.2e}  -> {'OK' if red_check['pass'] else 'FAIL'}")

    # ------------------------------------------------------------------ #
    # Step 3: Compute H and gamma                                          #
    # ------------------------------------------------------------------ #
    gamma = find_gamma(mu, sigma, beta, law)
    print(f"\nGenerator H and anchor:")
    print(f"  gamma = {gamma:.6f}  (expected ≈ 0.37414)")
    print(f"  H(gamma) = {H(gamma, mu, sigma, beta, law):.2e}")

    h_sign = check_H_sign_pattern_negative(m, gamma, mu, sigma, beta, law)
    print(f"  H > 0 on (m,gamma): {h_sign['all_H_between_positive']}")
    print(f"  H < 0 on (gamma,∞): {h_sign['all_H_above_gamma_negative']}")

    # ------------------------------------------------------------------ #
    # Step 4: Density normalization                                        #
    # ------------------------------------------------------------------ #
    # Use |mu| for normalization check (density same formula, drift sign enters through T_t)
    dens_check = check_density_norm_batch([0.5, 1.0, 2.0], [0.1, 0.5, 0.9], mu, sigma, beta)
    print(f"\nDensity normalization check: max error = {dens_check['max_error']:.4f}  -> {'OK' if dens_check['pass'] else 'FAIL'}")

    # ------------------------------------------------------------------ #
    # Step 5: Solve boundary                                               #
    # ------------------------------------------------------------------ #
    print(f"\nSolving negative-drift boundary (N={N} steps)...")
    times, b = solve_negative_boundary(mu, sigma, beta, T, law, N=N, verbose=True)

    b0 = b[0]
    print(f"\nKey results:")
    print(f"  b(0) = {b0:.6f}  (expected ≈ 0.35320)")
    print(f"  b(T-) = {b[-1]:.6f}  (expected ≈ gamma = {gamma:.5f})")

    # ------------------------------------------------------------------ #
    # Step 6: Monotonicity check                                           #
    # ------------------------------------------------------------------ #
    mono = check_boundary_monotonicity_negative(times, b)
    print(f"\nMonotonicity: b non-decreasing = {mono['b_non_decreasing']}")
    print(f"  b range: [{mono['b_min']:.5f}, {mono['b_max']:.5f}]")

    # ------------------------------------------------------------------ #
    # Step 7: Value function checks                                        #
    # ------------------------------------------------------------------ #
    t_check = times[0]
    V_at_b0 = compute_value_function_neg(t_check, b0, times, b, T, mu, sigma, beta, law)
    G_at_b0 = G_analytical(b0, law)
    print(f"\nValue matching at t=0:")
    print(f"  V(0,b0)={V_at_b0:.5f}, G(b0)={G_at_b0:.5f}, diff={abs(V_at_b0-G_at_b0):.2e}")

    # V < G for x > b(0) (continuation region)
    x_cont = 1.0
    V_cont = compute_value_function_neg(t_check, x_cont, times, b, T, mu, sigma, beta, law)
    G_cont = G_analytical(x_cont, law)
    print(f"  x=1.0 (above b0, continuation): V={V_cont:.5f}, G={G_cont:.5f}, V<G: {V_cont < G_cont}")

    # ------------------------------------------------------------------ #
    # Step 8: Plots                                                        #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, b, "g-o", markersize=3, label="$b(t)$ stopping boundary")
    ax.axhline(gamma, color="g", linestyle="--", alpha=0.7, label=f"$\\gamma={gamma:.4f}$")
    ax.axhline(m, color="gray", linestyle=":", alpha=0.7, label=f"$m={m:.4f}$")
    ax.fill_between(times, 0, b, alpha=0.2, color="green", label="Stopping region $x \\leq b(t)$")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Boundary value")
    ax.set_title("One-Sided Stopping Boundary $b(t)$ — Negative Drift CEV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, gamma * 1.5])
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp2_boundary.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Step 9: Summary                                                      #
    # ------------------------------------------------------------------ #
    results = {
        "experiment": "exp2_negative_drift",
        "parameters": {"mu": mu, "sigma": sigma, "beta": beta, "lambda": lam, "T": T, "N": N},
        "expected": {"gamma": 0.37414, "b0": 0.35320, "m": 0.34657},
        "computed": {
            "gamma": float(gamma),
            "m": float(m),
            "b0": float(b0),
            "b_terminal": float(b[-1]),
        },
        "checks": {
            "G_integral_ok": g_check["integral_ok"],
            "reduction_identity_ok": red_check["pass"],
            "density_norm_ok": dens_check["pass"],
            "H_sign_positive_between": h_sign["all_H_between_positive"],
            "H_sign_negative_above": h_sign["all_H_above_gamma_negative"],
            "b_non_decreasing": mono["b_non_decreasing"],
            "value_match_b0": abs(V_at_b0 - G_at_b0) < 0.05,
        },
    }

    with open(os.path.join(RESULTS_DIR, "exp2_results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=_NumpyEncoder)

    print("\n  Results saved to results/exp2_results.json")
    print("  Plot saved to results/exp2_boundary.png")
    return results


if __name__ == "__main__":
    run_exp2()
