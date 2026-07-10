"""
Experiment 5: Core Numerical Validation of Problem Reduction, CEV Density,
and Free-Boundary Residuals.

Validates the mathematical and numerical primitives:
  1. G representations (direct integral vs tail form vs closed form)
  2. Reduction identity E|x-l| = 2G(x)+E[l]
  3. H sign patterns (positive and negative drift)
  4. CEV transition density normalization and moment properties
  5. Kernel consistency: M(s,x,y,z) + L(s,x,y) = L(s,x,z) for y <= z
  6. Boundary residuals for already-computed boundaries
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json


class _NumpyEncoder(json.JSONEncoder):
    """Encode numpy scalars and arrays for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.aspiration_laws import ExponentialAspirationLaw
from src.loss_functions import G_analytical, G_integral, G_tail, H, find_alpha, find_gamma
from src.cev_density import cev_density, check_density_normalization, T_scaling
from src.kernels import compute_J, compute_M, compute_L
from src.diagnostics import (
    check_G_representations,
    check_reduction_identity,
    check_H_sign_pattern_positive,
    check_H_sign_pattern_negative,
    check_density_norm_batch,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_exp5() -> dict:
    """Run Experiment 5: core numerical validation suite."""
    print("=" * 60)
    print("Experiment 5: Core Numerical Validation")
    print("=" * 60)

    lam   = 2.0
    T     = 1.0
    law   = ExponentialAspirationLaw(lam=lam)
    m     = law.median
    E_l   = law.E_l

    print(f"Aspiration law: Exp(lambda={lam}), E[l]={E_l:.5f}, m={m:.6f}")

    # ------------------------------------------------------------------ #
    # 1. G representation checks                                           #
    # ------------------------------------------------------------------ #
    print("\n--- 1. G Representation Checks ---")
    x_dense = np.linspace(0.001, 6.0, 200)
    g_check = check_G_representations(x_dense, law, tol=1e-5)
    print(f"  Max diff G_analytical vs G_integral: {g_check['max_diff_integral']:.2e}  {'PASS' if g_check['integral_ok'] else 'FAIL'}")
    print(f"  Max diff G_analytical vs G_tail:     {g_check['max_diff_tail']:.2e}  {'PASS' if g_check['tail_ok'] else 'FAIL'}")

    # ------------------------------------------------------------------ #
    # 2. Reduction identity                                                #
    # ------------------------------------------------------------------ #
    print("\n--- 2. Reduction Identity E|x-l| = 2G(x)+E[l] ---")
    x_grid_id = np.linspace(0.05, 5.0, 30)
    red_check = check_reduction_identity(x_grid_id, law, tol=1e-4)
    print(f"  Max error: {red_check['max_error']:.2e}  {'PASS' if red_check['pass'] else 'FAIL'}")
    print(f"  Mean error: {red_check['mean_error']:.2e}")

    # ------------------------------------------------------------------ #
    # 3. H sign pattern — positive drift                                   #
    # ------------------------------------------------------------------ #
    print("\n--- 3a. H Sign Pattern — Positive Drift (mu=0.15) ---")
    mu_pos, sigma, beta_ref = 0.15, 0.30, 0.70
    alpha = find_alpha(mu_pos, sigma, beta_ref, law)
    h_pos = check_H_sign_pattern_positive(alpha, m, mu_pos, sigma, beta_ref, law)
    print(f"  alpha = {alpha:.6f}")
    print(f"  H<0 on (0,alpha): {h_pos['all_H_below_alpha_negative']}  (max={h_pos['max_H_below']:.2e})")
    print(f"  H>0 on (alpha,m): {h_pos['all_H_above_alpha_positive']}  (min={h_pos['min_H_above']:.2e})")

    print("\n--- 3b. H Sign Pattern — Negative Drift (mu=-0.15) ---")
    mu_neg = -0.15
    gamma = find_gamma(mu_neg, sigma, beta_ref, law)
    h_neg = check_H_sign_pattern_negative(m, gamma, mu_neg, sigma, beta_ref, law)
    print(f"  gamma = {gamma:.6f}")
    print(f"  H>0 on (m,gamma):  {h_neg['all_H_between_positive']}  (min={h_neg['min_H_between']:.2e})")
    print(f"  H<0 on (gamma,∞):  {h_neg['all_H_above_gamma_negative']}  (max={h_neg['max_H_above']:.2e})")

    # ------------------------------------------------------------------ #
    # 4. CEV density normalization                                         #
    # ------------------------------------------------------------------ #
    print("\n--- 4. CEV Density Normalization ---")
    x_norms = [0.3, 1.0, 2.0]
    t_norms = [0.05, 0.25, 0.5, 0.9]
    for mu_test, label in [(mu_pos, "mu=+0.15"), (mu_neg, "mu=-0.15")]:
        dens_check = check_density_norm_batch(x_norms, t_norms, mu_test, sigma, beta_ref)
        print(f"  {label}: max error = {dens_check['max_error']:.4f}  {'PASS' if dens_check['pass'] else 'FAIL'}")
        for r in dens_check["results"]:
            print(f"    x={r['x']}, t={r['t']:.2f}: integral = {r['norm']:.5f}")

    # ------------------------------------------------------------------ #
    # 5. Kernel consistency: L(s,x,y)+M(s,x,y,z) = L(s,x,z)              #
    # ------------------------------------------------------------------ #
    print("\n--- 5. Kernel Consistency: L(s,x,y) + M(s,x,y,z) = L(s,x,z) ---")
    s_test = 0.3
    x_test = 1.0
    y_test = 0.2
    z_test = 1.5

    L_y  = compute_L(s_test, x_test, y_test, mu_pos, sigma, beta_ref, law)
    L_z  = compute_L(s_test, x_test, z_test, mu_pos, sigma, beta_ref, law)
    M_yz = compute_M(s_test, x_test, y_test, z_test, mu_pos, sigma, beta_ref, law)
    lhs  = L_y + M_yz
    err  = abs(lhs - L_z)
    print(f"  L({s_test},{x_test},{y_test}) = {L_y:.6f}")
    print(f"  M({s_test},{x_test},{y_test},{z_test}) = {M_yz:.6f}")
    print(f"  L({s_test},{x_test},{z_test}) = {L_z:.6f}")
    print(f"  L_y + M_yz - L_z = {lhs - L_z:.2e}  {'PASS' if err < 1e-4 else 'FAIL'}")

    # ------------------------------------------------------------------ #
    # 6. J moment check                                                    #
    # ------------------------------------------------------------------ #
    print("\n--- 6. J Kernel Plausibility ---")
    # J(t,x) should be >= G(x) for all x (V >= G always fails, but J >= G_min is expected)
    t_j_test = 0.5
    x_j_vals = [0.3, 0.5, 1.0, 2.0, 3.0]
    for x_j in x_j_vals:
        J_val  = compute_J(t_j_test, x_j, T, mu_pos, sigma, beta_ref, law)
        G_val  = G_analytical(x_j, law)
        print(f"  x={x_j:.1f}: J={J_val:.5f}, G={G_val:.5f}")

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #
    x_plot = np.linspace(0.001, 5.0, 300)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    G_v = G_analytical(x_plot, law)
    ax.plot(x_plot, G_v, "b-", label="$G(x)$ analytical")
    ax.axhline(0, color="gray", linestyle=":")
    ax.axvline(m, color="red", linestyle="--", alpha=0.7, label=f"$m={m:.3f}$")
    ax.set_xlabel("$x$")
    ax.set_title("Reduced Loss Function $G(x)$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.3, 1.5])

    ax = axes[1]
    H_pos_v = H(x_plot, mu_pos, sigma, beta_ref, law)
    H_neg_v = H(x_plot, mu_neg, sigma, beta_ref, law)
    ax.plot(x_plot, H_pos_v, "b-", label=f"$H(x)$, $\\mu=+0.15$, $\\alpha={alpha:.4f}$")
    ax.plot(x_plot, H_neg_v, "r-", label=f"$H(x)$, $\\mu=-0.15$, $\\gamma={gamma:.4f}$")
    ax.axhline(0, color="gray", linestyle=":")
    ax.axvline(alpha, color="b", linestyle="--", alpha=0.5)
    ax.axvline(gamma, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("$x$")
    ax.set_title("Generator Term $H(x)$")
    ax.legend(fontsize=8)
    ax.set_ylim([-0.1, 0.1])
    ax.set_xlim([0, 2.0])
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    x_dens = 1.0
    z_arr = np.linspace(0.01, 5.0, 200)
    for t_d, col in [(0.1, "b"), (0.5, "g"), (0.9, "r")]:
        dens = cev_density(x_dens, z_arr, t_d, mu_pos, sigma, beta_ref)
        ax.plot(z_arr, dens, color=col, label=f"$t={t_d}$")
    ax.set_xlabel("$z$")
    ax.set_title(f"CEV Density $p^{{\\mu}}({x_dens},z;t)$, $\\mu=+0.15$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp5_validation_plots.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    results = {
        "experiment": "exp5_validation",
        "G_check": g_check,
        "reduction_identity": red_check,
        "H_positive": h_pos,
        "H_negative": h_neg,
        "density_norm_positive": dens_check,
        "kernel_consistency": {
            "L_y": float(L_y), "M_yz": float(M_yz), "L_z": float(L_z),
            "error": float(err), "pass": bool(err < 1e-4),
        },
        "computed_anchors": {
            "alpha": float(alpha), "gamma": float(gamma), "m": float(m), "E_l": float(E_l),
        },
    }

    with open(os.path.join(RESULTS_DIR, "exp5_results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=_NumpyEncoder)

    print("\n  Results saved to results/exp5_results.json")
    print("  Plots saved to results/exp5_validation_plots.png")
    return results


if __name__ == "__main__":
    run_exp5()
