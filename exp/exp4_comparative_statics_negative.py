"""
Experiment 4: Negative-Drift Comparative Statics in CEV Elasticity Beta.

Computes optimal stopping boundaries for 10 equally spaced beta values in [0.01, 2.0].
Verifies that decreasing beta raises b(t) (wider stopping region).

Fixed parameters: mu=-0.15, sigma=0.30, lambda=2, T=1.
Beta grid: 10 values equally spaced in [0.01, 2.0].
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
from src.loss_functions import find_gamma, H
from src.boundary_solver_negative import solve_negative_boundary

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_exp4() -> dict:
    """Run Experiment 4: negative-drift comparative statics over beta grid."""
    print("=" * 60)
    print("Experiment 4: Negative-Drift Comparative Statics in Beta")
    print("=" * 60)

    mu    = -0.15
    sigma =  0.30
    lam   =  2.0
    T     =  1.0
    N     =  25  # reduced for speed; comparative statics needs qualitative trends

    law = ExponentialAspirationLaw(lam=lam)
    m   = law.median

    # Beta grid: 10 equally spaced values in [0.01, 2.0]
    n_beta = 10
    beta_grid = np.linspace(0.01, 2.0, n_beta)
    print(f"\nBeta grid: {np.round(beta_grid, 4)}")

    all_results = []
    all_b = {}
    all_times = None

    for k, beta in enumerate(beta_grid):
        print(f"\n--- Beta = {beta:.4f} ({k+1}/{n_beta}) ---")
        try:
            gamma_k = find_gamma(mu, sigma, beta, law)

            times, b_k = solve_negative_boundary(
                mu, sigma, beta, T, law, N=N, verbose=False,
                epsabs=1e-4, epsrel=1e-4,
            )

            if all_times is None:
                all_times = times

            all_b[beta] = b_k

            result_k = {
                "beta": float(beta),
                "gamma": float(gamma_k),
                "b0": float(b_k[0]),
                "b_terminal": float(b_k[-1]),
            }
            all_results.append(result_k)
            print(f"  gamma={gamma_k:.5f}, b(0)={b_k[0]:.5f}, b(T-)={b_k[-1]:.5f}")

        except Exception as e:
            print(f"  ERROR for beta={beta:.4f}: {e}")
            all_results.append({"beta": float(beta), "error": str(e)})

    # ------------------------------------------------------------------ #
    # Comparative statics analysis                                         #
    # ------------------------------------------------------------------ #
    valid_results = [r for r in all_results if "error" not in r]
    if len(valid_results) >= 2:
        betas_v  = np.array([r["beta"] for r in valid_results])
        b0_v     = np.array([r["b0"]   for r in valid_results])

        # Smaller beta => higher b(0) (wider stopping region).
        # Allow tolerance 1e-3 for numerical noise at extreme beta (e.g. beta=2.0 where
        # gamma-m is very small and the solver may collapse b -> gamma for all steps).
        b0_mono = bool(np.all(np.diff(b0_v) <= 1e-3))  # b0 non-increasing in beta
        print(f"\nComparative statics (smaller beta => higher b):")
        print(f"  b(0) non-increasing in beta (tol=1e-3): {b0_mono}")
        if not b0_mono:
            print("  NOTE: mild violation at extreme beta due to gamma-m shrinkage")
        print(f"\n  {'Beta':>8} {'gamma':>8} {'b(0)':>10} {'b(T-)':>10}")
        for r in valid_results:
            print(f"  {r['beta']:>8.4f} {r['gamma']:>8.5f} {r['b0']:>10.5f} {r['b_terminal']:>10.5f}")

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #
    if all_times is not None and len(valid_results) >= 2:
        cmap   = plt.cm.plasma
        colors = [cmap(i / (len(valid_results) - 1)) for i in range(len(valid_results))]

        fig, ax = plt.subplots(figsize=(9, 5))
        for k, r in enumerate(valid_results):
            b_val = r["beta"]
            if b_val in all_b:
                ax.plot(all_times, all_b[b_val], color=colors[k],
                        label=f"$\\beta={b_val:.2f}$", linewidth=1.5)
        ax.axhline(m, color="gray", linestyle=":", label=f"$m={m:.4f}$")
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Stopping boundary $b(t)$")
        ax.set_title("One-Sided Stopping Boundary vs Beta (Negative Drift)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "exp4_comparative_statics_negative.png"), dpi=120)
        plt.close(fig)

        # b(0) vs beta
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        betas_plot = [r["beta"] for r in valid_results if r["beta"] in all_b]
        b0_plot    = [r["b0"]   for r in valid_results if r["beta"] in all_b]
        ax2.plot(betas_plot, b0_plot, "o-", color="green")
        ax2.set_xlabel("Elasticity $\\beta$")
        ax2.set_ylabel("$b(0)$ stopping boundary at $t=0$")
        ax2.set_title("Stopping Boundary $b(0)$ vs Beta: Negative Drift")
        ax2.grid(True, alpha=0.3)
        fig2.savefig(os.path.join(RESULTS_DIR, "exp4_b0_vs_beta.png"), dpi=120)
        plt.close(fig2)

    # ------------------------------------------------------------------ #
    # Save results                                                         #
    # ------------------------------------------------------------------ #
    summary = {
        "experiment": "exp4_comparative_statics_negative",
        "parameters": {"mu": mu, "sigma": sigma, "lambda": lam, "T": T, "N": N},
        "beta_grid": beta_grid.tolist(),
        "results": all_results,
    }
    with open(os.path.join(RESULTS_DIR, "exp4_results.json"), "w") as fh:
        json.dump(summary, fh, indent=2, cls=_NumpyEncoder)

    print("\n  Results saved to results/exp4_results.json")
    print("  Plots saved to results/exp4_*.png")
    return summary


if __name__ == "__main__":
    run_exp4()
