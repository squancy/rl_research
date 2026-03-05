import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import gaussian_kde

# Consistent color palette across all plots
COLORS = {
    "expert": "#2d3436",  # dark charcoal
    "policy": "#0984e3",  # blue
    "fill": "#74b9ff",  # light blue
    "bar_expert": "#2d3436",
    "bar_policy": "#0984e3",
    "error": "#d63031",  # red
}


class Plots:
    """
    Utility class for creating paper-quality visualizations.
    Plots are stored internally and rendered together via show().
    """

    def __init__(self):
        self._plot_fns = []

        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "lines.linewidth": 2,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.grid": True,
                "grid.linestyle": ":",
                "grid.alpha": 0.5,
                "grid.linewidth": 0.7,
            }
        )

    def loss_training_step(self, losses: list[float]):
        def _plot(ax):
            steps = np.arange(len(losses))
            ax.plot(steps, losses, color=COLORS["error"], alpha=0.9, lw=1.8)
            ax.fill_between(steps, losses, alpha=0.12, color=COLORS["error"])
            ax.set_yscale("log")
            ax.set_ylim(1e-10, None)
            ax.set_xlabel("Training step")
            ax.set_ylabel("Loss")
            ax.set_title("BC Training Loss")

        self._plot_fns.append(_plot)

    def action_time(
        self,
        trajectory_pairs: list[tuple[list[tuple], list[tuple]]],
        is_bc: bool = True,
    ):
        """Stacked sub-axes — one per trajectory — with expert & policy overlaid."""
        label_policy = "Behavior Cloning" if is_bc else "DAgger"

        def _to_floats(actions):
            return [
                a.item() if isinstance(a, torch.Tensor) else float(a) for a in actions
            ]

        parsed: list[tuple[list[float], list[float]]] = []
        for expert_traj, policy_traj in trajectory_pairs:
            expert_actions = _to_floats([step[1] for step in expert_traj])
            policy_actions = _to_floats([step[1] for step in policy_traj])
            parsed.append((expert_actions, policy_actions))

        n_trajs = len(parsed)

        def _plot_stacked(fig, sub_gs):
            """Draw n stacked sub-axes inside the given SubGridSpec."""
            inner = sub_gs.subgridspec(n_trajs, 1, hspace=0.3)
            axes = [fig.add_subplot(inner[j, 0]) for j in range(n_trajs)]

            for i, (expert_a, policy_a) in enumerate(parsed):
                ax = axes[i]
                steps = np.arange(len(expert_a))
                ax.plot(
                    steps,
                    expert_a,
                    color=COLORS["expert"],
                    alpha=0.9,
                    label="Expert" if i == 0 else None,
                )
                ax.plot(
                    steps,
                    policy_a,
                    color=COLORS["policy"],
                    alpha=0.8,
                    linestyle="--",
                    label=label_policy if i == 0 else None,
                )
                ax.set_ylabel(r"$\pi$", fontsize=8)
                ax.tick_params(labelsize=7)
                if i < n_trajs - 1:
                    ax.tick_params(labelbottom=False)

            axes[-1].set_xlabel("Time step")
            axes[0].set_title("Action vs Time")
            axes[0].legend(
                frameon=True,
                fancybox=True,
                framealpha=0.7,
                edgecolor="none",
                fontsize=7,
                loc="upper right",
            )

        # Mark as a stacked (multi-row) plot so show() knows to give it a full column
        _plot_stacked._is_stacked = True  # type: ignore[attr-defined]
        _plot_stacked._n_trajs = n_trajs  # type: ignore[attr-defined]
        self._plot_fns.append(_plot_stacked)

    def terminal_wealth_distr(
        self,
        X_expert: NDArray[np.float32],
        X_policy: NDArray[np.float32],
        is_bc: bool = True,
    ):
        X_expert = np.asarray(X_expert).reshape(-1)
        X_policy = np.asarray(X_policy).reshape(-1)

        X_expert = X_expert[np.isfinite(X_expert)]
        X_policy = X_policy[np.isfinite(X_policy)]

        def _plot(ax):
            if len(X_expert) == 0 or len(X_policy) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No finite terminal wealth values to plot",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                return
            label_policy = "Behavior Cloning" if is_bc else "DAgger"
            xmin = min(X_expert.min(), X_policy.min())
            xmax = max(X_expert.max(), X_policy.max())
            pad = 0.05 * (xmax - xmin)
            xs = np.linspace(xmin - pad, xmax + pad, 400)

            if len(X_expert) > 1:
                kde_expert = gaussian_kde(X_expert)
                ys = kde_expert(xs)
                ax.plot(xs, ys, color=COLORS["expert"], alpha=0.9, label="Expert")
                ax.fill_between(xs, ys, alpha=0.10, color=COLORS["expert"])
            else:
                ax.axvline(X_expert[0], color=COLORS["expert"], label="Expert")

            if len(X_policy) > 1:
                kde_policy = gaussian_kde(X_policy)
                ys = kde_policy(xs)
                ax.plot(
                    xs,
                    ys,
                    color=COLORS["policy"],
                    alpha=0.85,
                    linestyle="--",
                    label=label_policy,
                )
                ax.fill_between(xs, ys, alpha=0.10, color=COLORS["fill"])
            else:
                ax.axvline(
                    X_policy[0],
                    linestyle="--",
                    color=COLORS["policy"],
                    label=label_policy,
                )

            ax.set_xlabel("Terminal wealth $X_T$")
            ax.set_ylabel("Density")
            ax.set_title("Terminal Wealth Distribution")
            ax.legend(
                frameon=True,
                fancybox=True,
                framealpha=0.7,
                edgecolor="none",
                fontsize=8,
            )

        self._plot_fns.append(_plot)

    def expected_utility(self, U_expert: float, U_policy: float):
        def _plot(ax):
            values = [U_expert, U_policy]
            labels = ["Expert", "BC"]
            colors = [COLORS["bar_expert"], COLORS["bar_policy"]]

            bars = ax.bar(
                labels,
                values,
                width=0.55,
                color=colors,
                edgecolor="white",
                linewidth=1.2,
                alpha=0.85,
            )
            ax.set_ylabel("Expected utility")
            ax.set_title("Expected Utility")
            ax.grid(axis="y")

            for bar, v in zip(bars, values):
                offset = abs(v) * 0.02 if v >= 0 else -abs(v) * 0.04
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + offset,
                    f"{v:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        self._plot_fns.append(_plot)

    def rollout_drift(self, errors: NDArray[np.float32]):
        def _plot(ax):
            errs = np.asarray(errors).reshape(-1)
            steps = np.arange(len(errs))
            ax.plot(steps, errs, color=COLORS["error"], alpha=0.85, lw=1.8)
            ax.fill_between(steps, errs, alpha=0.12, color=COLORS["error"])
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

            ax.set_xlabel("Time step")
            ax.set_ylabel("Policy difference")
            ax.set_title("Action Error Over Time")

        self._plot_fns.append(_plot)

    def show(
        self,
        savepath: str | None = None,
        dpi: int = 300,
    ):
        plot_fns = self._plot_fns[:5]
        n = len(plot_fns)
        if n == 0:
            return

        # Layout: 3 top, 2 bottom (centred) — same as before.
        # Stacked plots (action_time) get a sub-gridspec *inside* their cell.
        nrows = 2 if n > 3 else 1
        base_height = 3.2
        fig = plt.figure(figsize=(12, base_height * nrows))

        gs = fig.add_gridspec(
            nrows=nrows,
            ncols=6,
            height_ratios=[1] * nrows,
            hspace=0.55,
            wspace=1.4,
        )

        # Map each slot to its GridSpec cell
        cells = []
        for j in range(min(n, 3)):
            cells.append(gs[0, 2 * j : 2 * j + 2])
        if n >= 4:
            cells.append(gs[1, 1:3])
        if n >= 5:
            cells.append(gs[1, 3:5])

        for cell, fn in zip(cells, plot_fns):
            if getattr(fn, "_is_stacked", False):
                # Stacked plot fills the cell with internal sub-axes
                fn(fig, cell)
            else:
                ax = fig.add_subplot(cell)
                fn(ax)

        fig.tight_layout()

        if savepath is not None:
            fig.savefig(
                savepath,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.02,
            )

        plt.show()
        plt.close(fig)
