"""
Experiment 3: Positive-Drift Comparative Statics in CEV Elasticity Beta.

Computes optimal stopping boundaries for 8 equally spaced beta values in [0.01, 0.30].
Verifies that decreasing beta:
  - Lowers the lower boundary a(t)
  - Raises the upper boundary c(t)
  - Widens the stopping band

Fixed parameters: mu=0.15, sigma=0.30, lambda=2, T=1.
Beta grid: 8 values equally spaced in [0.01, 0.30].
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
from src.loss_functions import find_alpha, G_analytical, H
from src.boundary_solver_positive import solve_positive_boundary

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_exp3() -> dict:
    """Run Experiment 3: positive-drift comparative statics over beta grid."""
    print("=" * 60)
    print("Experiment 3: Positive-Drift Comparative Statics in Beta")
    print("=" * 60)

    # Fixed parameters
    mu    = 0.15
    sigma = 0.30
    lam   = 2.0
    T     = 1.0
    # Use N=25 for speed; comparative statics needs qualitative trends, not exact values
    N     = 25

    law = ExponentialAspirationLaw(lam=lam)
    m   = law.median

    # Beta grid: 8 equally spaced values in [0.01, 0.30]
    n_beta = 8
    beta_grid = np.linspace(0.01, 0.30, n_beta)
    print(f"\nBeta grid: {np.round(beta_grid, 4)}")

    all_results = []
    all_a = {}
    all_c = {}
    all_times = None

    for k, beta in enumerate(beta_grid):
        print(f"\n--- Beta = {beta:.4f} ({k+1}/{n_beta}) ---")
        try:
            alpha_k = find_alpha(mu, sigma, beta, law)
            # Cap C_init and bracket to prevent unbounded search for small beta
            # Small beta leads to very large c(t); cap at 80 for tractability
            C_init_k = min(50.0 + 10.0 / (beta + 0.05), 80.0)
            c_max_k  = min(C_init_k * 2.0, 100.0)

            times, a_k, c_k = solve_positive_boundary(
                mu, sigma, beta, T, law,
                N=N, C_init=C_init_k, c_bracket_max=c_max_k,
                verbose=False, epsabs=1e-4, epsrel=1e-4,
            )

            if all_times is None:
                all_times = times

            all_a[beta] = a_k
            all_c[beta] = c_k

            result_k = {
                "beta": float(beta),
                "alpha": float(alpha_k),
                "a0": float(a_k[0]),
                "c0": float(c_k[0]),
                "a_terminal": float(a_k[-1]),
                "c_terminal": float(c_k[-1]),
                "max_c": float(np.max(c_k)),
            }
            all_results.append(result_k)
            print(f"  alpha={alpha_k:.5f}, a(0)={a_k[0]:.5f}, c(0)={c_k[0]:.4f}, max_c={np.max(c_k):.3f}")

        except Exception as e:
            print(f"  ERROR for beta={beta:.4f}: {e}")
            all_results.append({"beta": float(beta), "error": str(e)})

    # ------------------------------------------------------------------ #
    # Comparative statics analysis                                         #
    # ------------------------------------------------------------------ #
    valid_results = [r for r in all_results if "error" not in r]
    if len(valid_results) >= 2:
        betas_v  = np.array([r["beta"] for r in valid_results])
        a0_v     = np.array([r["a0"]   for r in valid_results])
        c0_v     = np.array([r["c0"]   for r in valid_results])
        alphas_v = np.array([r["alpha"] for r in valid_results])

        # Check monotonicity: larger beta -> larger a(0), smaller c(0)
        a0_mono  = bool(np.all(np.diff(a0_v) >= -1e-4))   # a0 non-decreasing in beta
        c0_mono  = bool(np.all(np.diff(c0_v) <= 1e-1))    # c0 non-increasing in beta
        print(f"\nComparative statics (smaller beta => lower a, higher c):")
        print(f"  a(0) non-decreasing in beta: {a0_mono}")
        print(f"  c(0) non-increasing in beta: {c0_mono}")
        print(f"\n  {'Beta':>8} {'alpha':>8} {'a(0)':>10} {'c(0)':>10}")
        for r in valid_results:
            print(f"  {r['beta']:>8.4f} {r['alpha']:>8.5f} {r['a0']:>10.5f} {r['c0']:>10.4f}")

        # Fixed-time cross section
        fixed_times = [0.0, 0.25, 0.5, 0.75]
        if all_times is not None:
            for t_fix in fixed_times:
                idx = np.argmin(np.abs(all_times - t_fix))
                a_cross = np.array([
                    all_a[r["beta"]][idx] for r in valid_results if r["beta"] in all_a
                ])
                c_cross = np.array([
                    all_c[r["beta"]][idx] for r in valid_results if r["beta"] in all_c
                ])
                beta_cross = np.array([
                    r["beta"] for r in valid_results if r["beta"] in all_a
                ])
                print(f"\n  Fixed t ≈ {all_times[idx]:.3f}: a-mono={bool(np.all(np.diff(a_cross)>=0))}, "
                      f"c-mono={bool(np.all(np.diff(c_cross)<=0))}")

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #
    if all_times is not None and len(valid_results) >= 2:
        cmap = plt.cm.viridis
        colors = [cmap(i / (len(valid_results) - 1)) for i in range(len(valid_results))]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        for k, r in enumerate(valid_results):
            b_val = r["beta"]
            if b_val in all_a:
                ax.plot(all_times, all_a[b_val], color=colors[k],
                        label=f"$\\beta={b_val:.3f}$", linewidth=1.5)
        ax.axhline(m, color="gray", linestyle=":", label=f"$m={m:.4f}$")
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Lower boundary $a(t)$")
        ax.set_title("Lower Stopping Boundary vs Beta (Positive Drift)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for k, r in enumerate(valid_results):
            b_val = r["beta"]
            if b_val in all_c:
                ax.semilogy(all_times, all_c[b_val], color=colors[k],
                            label=f"$\\beta={b_val:.3f}$", linewidth=1.5)
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Upper boundary $c(t)$ (log scale)")
        ax.set_title("Upper Stopping Boundary vs Beta (Positive Drift, log scale)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "exp3_comparative_statics_positive.png"), dpi=120)
        plt.close(fig)

        # c(0) vs beta plot
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        betas_plot = [r["beta"] for r in valid_results if r["beta"] in all_c]
        c0_plot    = [r["c0"] for r in valid_results if r["beta"] in all_c]
        ax2.semilogy(betas_plot, c0_plot, "o-", color="red")
        ax2.set_xlabel("Elasticity $\\beta$")
        ax2.set_ylabel("$c(0)$ upper boundary at $t=0$ (log scale)")
        ax2.set_title("Upper Boundary $c(0)$ vs Beta: GBM limit as $\\beta \\to 0$")
        ax2.grid(True, alpha=0.3)
        fig2.savefig(os.path.join(RESULTS_DIR, "exp3_c0_vs_beta.png"), dpi=120)
        plt.close(fig2)

    # ------------------------------------------------------------------ #
    # Save results                                                         #
    # ------------------------------------------------------------------ #
    summary = {
        "experiment": "exp3_comparative_statics_positive",
        "parameters": {"mu": mu, "sigma": sigma, "lambda": lam, "T": T, "N": N},
        "beta_grid": beta_grid.tolist(),
        "results": all_results,
    }
    with open(os.path.join(RESULTS_DIR, "exp3_results.json"), "w") as fh:
        json.dump(summary, fh, indent=2, cls=_NumpyEncoder)

    print("\n  Results saved to results/exp3_results.json")
    print("  Plots saved to results/exp3_*.png")
    return summary


if __name__ == "__main__":
    run_exp3()
