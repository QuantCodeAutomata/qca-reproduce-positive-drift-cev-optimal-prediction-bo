"""
Experiment 1: Positive-Drift CEV Optimal Prediction Boundary Band.

Reproduces Example 5.1 from the paper:
  mu=0.15, sigma=0.30, beta=0.70, lambda=2, T=1
  Expected: alpha ≈ 0.32581, a(0) ≈ 0.34234, c(0) ≈ 2.67449

Solves the two-sided free-boundary integral equations by backward induction
and validates the stopping-band geometry.
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
from src.loss_functions import G_analytical, H, find_alpha
from src.cev_density import cev_density, check_density_normalization
from src.kernels import compute_J, compute_M
from src.boundary_solver_positive import solve_positive_boundary, compute_value_function
from src.diagnostics import (
    check_G_representations,
    check_reduction_identity,
    check_H_sign_pattern_positive,
    check_density_norm_batch,
    check_boundary_monotonicity_positive,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_exp1() -> dict:
    """Run the full Experiment 1 pipeline and return summary metrics."""
    print("=" * 60)
    print("Experiment 1: Positive-Drift CEV Optimal Prediction")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Parameters                                                           #
    # ------------------------------------------------------------------ #
    mu     = 0.15
    sigma  = 0.30
    beta   = 0.70
    lam    = 2.0
    T      = 1.0
    N      = 60   # time steps

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
    print(f"  Max diff (tail form): {g_check['max_diff_tail']:.2e}  -> {'OK' if g_check['tail_ok'] else 'FAIL'}")

    # ------------------------------------------------------------------ #
    # Step 2: Validate reduction identity E|x-l| = 2G(x) + E[l]          #
    # ------------------------------------------------------------------ #
    x_test_small = np.linspace(0.1, 4.0, 20)
    red_check = check_reduction_identity(x_test_small, law)
    print(f"\nReduction identity check:")
    print(f"  Max error: {red_check['max_error']:.2e}  -> {'OK' if red_check['pass'] else 'FAIL'}")

    # ------------------------------------------------------------------ #
    # Step 3: Compute H and alpha                                          #
    # ------------------------------------------------------------------ #
    alpha = find_alpha(mu, sigma, beta, law)
    print(f"\nGenerator H and anchor:")
    print(f"  alpha = {alpha:.6f}  (expected ≈ 0.32581)")
    print(f"  H(alpha) = {H(alpha, mu, sigma, beta, law):.2e}")

    h_sign = check_H_sign_pattern_positive(alpha, m, mu, sigma, beta, law)
    print(f"  H < 0 on (0,alpha): {h_sign['all_H_below_alpha_negative']}")
    print(f"  H > 0 on (alpha,m): {h_sign['all_H_above_alpha_positive']}")

    # ------------------------------------------------------------------ #
    # Step 4: Density normalization check                                  #
    # ------------------------------------------------------------------ #
    x_vals = [0.5, 1.0, 2.0]
    t_vals = [0.1, 0.5, 0.9]
    dens_check = check_density_norm_batch(x_vals, t_vals, mu, sigma, beta, z_max=50.0)
    print(f"\nDensity normalization check:")
    print(f"  Max error: {dens_check['max_error']:.4f}  -> {'OK' if dens_check['pass'] else 'FAIL'}")
    for r in dens_check["results"]:
        print(f"  x={r['x']}, t={r['t']}: norm={r['norm']:.5f}")

    # ------------------------------------------------------------------ #
    # Step 5: Solve boundaries                                             #
    # ------------------------------------------------------------------ #
    print(f"\nSolving positive-drift boundaries (N={N} steps)...")
    times, a, c = solve_positive_boundary(
        mu, sigma, beta, T, law, N=N, C_init=25.0,
        c_bracket_max=300.0, verbose=True,
    )

    a0 = a[0]
    c0 = c[0]
    print(f"\nKey results:")
    print(f"  a(0) = {a0:.6f}  (expected ≈ 0.34234)")
    print(f"  c(0) = {c0:.6f}  (expected ≈ 2.67449)")
    print(f"  a(T-) = {a[-1]:.6f}  (expected ≈ alpha = {alpha:.5f})")
    print(f"  c(T-) = {c[-1]:.4f}  (should be large)")

    # ------------------------------------------------------------------ #
    # Step 6: Monotonicity check                                           #
    # ------------------------------------------------------------------ #
    mono = check_boundary_monotonicity_positive(times, a, c)
    print(f"\nMonotonicity:")
    print(f"  a non-increasing in t: {mono['a_non_increasing']}")
    print(f"  c non-decreasing in t: {mono['c_non_decreasing']}")
    print(f"  a range: [{mono['a_min']:.5f}, {mono['a_max']:.5f}]")
    print(f"  c range: [{mono['c_min']:.4f}, {mono['c_max']:.4f}]")

    # ------------------------------------------------------------------ #
    # Step 7: Value function on stopping boundary (V = G check)           #
    # ------------------------------------------------------------------ #
    t_check = times[0]
    V_at_a0 = compute_value_function(t_check, a0, times, a, c, T, mu, sigma, beta, law)
    V_at_c0 = compute_value_function(t_check, c0, times, a, c, T, mu, sigma, beta, law)
    G_at_a0 = G_analytical(a0, law)
    G_at_c0 = G_analytical(c0, law)
    print(f"\nValue matching at t=0:")
    print(f"  V(0,a0)={V_at_a0:.5f}, G(a0)={G_at_a0:.5f}, diff={abs(V_at_a0-G_at_a0):.2e}")
    print(f"  V(0,c0)={V_at_c0:.5f}, G(c0)={G_at_c0:.5f}, diff={abs(V_at_c0-G_at_c0):.2e}")

    # Verify V < G for a continuation point
    x_cont = 1.0  # inside continuation region (between 0 and a(0) or above c(0)? No, cont is outside band)
    # Actually continuation is OUTSIDE the band: x < a(0) or x > c(0)
    x_cont_below = 0.2
    x_cont_above = 4.0
    V_cont_below = compute_value_function(t_check, x_cont_below, times, a, c, T, mu, sigma, beta, law)
    V_cont_above = compute_value_function(t_check, x_cont_above, times, a, c, T, mu, sigma, beta, law)
    G_cont_below = G_analytical(x_cont_below, law)
    G_cont_above = G_analytical(x_cont_above, law)
    print(f"\nContinuation inequalities (V < G outside band):")
    print(f"  x=0.2 (below a0): V={V_cont_below:.5f}, G={G_cont_below:.5f}, V<G: {V_cont_below < G_cont_below}")
    print(f"  x=4.0 (above c0): V={V_cont_above:.5f}, G={G_cont_above:.5f}, V<G: {V_cont_above < G_cont_above}")

    # ------------------------------------------------------------------ #
    # Step 8: Plots                                                        #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(times, a, "b-o", markersize=3, label="$a(t)$ lower boundary")
    ax.axhline(alpha, color="b", linestyle="--", alpha=0.7, label=f"$\\alpha={alpha:.4f}$")
    ax.axhline(m, color="gray", linestyle=":", alpha=0.7, label=f"$m={m:.4f}$")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Boundary value")
    ax.set_title("Lower Stopping Boundary $a(t)$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times, c, "r-o", markersize=3, label="$c(t)$ upper boundary")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Boundary value")
    ax.set_title("Upper Stopping Boundary $c(t)$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp1_boundaries.png"), dpi=120)
    plt.close(fig)

    # Stopping/continuation region heatmap
    x_grid = np.linspace(0.05, 5.0, 80)
    t_grid = times
    stop_map = np.zeros((len(x_grid), len(t_grid)))
    for ti, t_val in enumerate(t_grid):
        for xi, x_val in enumerate(x_grid):
            if a[ti] <= x_val <= c[ti]:
                stop_map[xi, ti] = 1.0

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    im = ax2.imshow(
        stop_map, aspect="auto", origin="lower",
        extent=[times[0], times[-1], x_grid[0], x_grid[-1]],
        cmap="RdYlGn", vmin=0, vmax=1,
    )
    ax2.plot(times, a, "b-", linewidth=2, label="$a(t)$")
    ax2.plot(times, np.minimum(c, 5.0), "r-", linewidth=2, label="$c(t)$ (clipped at 5)")
    ax2.set_xlabel("Time $t$")
    ax2.set_ylabel("State $x$")
    ax2.set_title("Stopping Region (green) and Continuation Region (red)\nPositive Drift CEV")
    ax2.legend()
    plt.colorbar(im, ax=ax2, label="Stop=1, Continue=0")
    fig2.savefig(os.path.join(RESULTS_DIR, "exp1_stopping_region.png"), dpi=120)
    plt.close(fig2)

    # ------------------------------------------------------------------ #
    # Step 9: Summary                                                      #
    # ------------------------------------------------------------------ #
    results = {
        "experiment": "exp1_positive_drift",
        "parameters": {"mu": mu, "sigma": sigma, "beta": beta, "lambda": lam, "T": T, "N": N},
        "expected": {"alpha": 0.32581, "a0": 0.34234, "c0": 2.67449, "m": 0.34657},
        "computed": {
            "alpha": float(alpha),
            "m": float(m),
            "a0": float(a0),
            "c0": float(c0),
            "a_terminal": float(a[-1]),
            "c_terminal": float(c[-1]),
        },
        "checks": {
            "G_integral_ok": g_check["integral_ok"],
            "G_tail_ok": g_check["tail_ok"],
            "reduction_identity_ok": red_check["pass"],
            "density_norm_ok": dens_check["pass"],
            "H_sign_positive": h_sign["all_H_above_alpha_positive"],
            "H_sign_negative": h_sign["all_H_below_alpha_negative"],
            "a_non_increasing": mono["a_non_increasing"],
            "c_non_decreasing": mono["c_non_decreasing"],
        },
        "value_matching": {
            "V_at_a0": float(V_at_a0),
            "G_at_a0": float(G_at_a0),
            "V_at_c0": float(V_at_c0),
            "G_at_c0": float(G_at_c0),
        },
    }

    # Save JSON
    with open(os.path.join(RESULTS_DIR, "exp1_results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=_NumpyEncoder)

    print("\n  Results saved to results/exp1_results.json")
    print("  Plots saved to results/exp1_boundaries.png, exp1_stopping_region.png")
    return results


if __name__ == "__main__":
    run_exp1()
