import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import gaussian_kde


class Plots:
    """
    Utility class for creating paper-quality visualizations.
    Plots are stored internally and rendered together via show().
    """

    def __init__(self):
        self._plot_fns = []

        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "lines.linewidth": 2,
            }
        )

    def loss_training_step(self, losses: list[float]):
        def _plot(ax):
            ax.plot(losses, color="black")
            ax.set_yscale("log")
            ax.set_ylim(1e-10, None)
            ax.set_xlabel("Training step")
            ax.set_ylabel("MSE loss")
            ax.set_title("BC Training Loss")
            ax.grid(True, which="both", linestyle=":", linewidth=0.7)

        self._plot_fns.append(_plot)

    def action_time(self, expert_traj: list[tuple], policy_traj: list[tuple]):
        expert_actions = [step[1] for step in expert_traj]
        policy_actions = [step[1] for step in policy_traj]

        def _plot(ax):
            ax.plot(expert_actions, color="black", label="Expert")
            ax.plot(
                policy_actions,
                linestyle="--",
                color="tab:blue",
                label="Behavior cloning",
            )
            ax.set_xlabel("Time step")
            ax.set_ylabel("Portfolio weight")
            ax.set_title("Action vs Time")
            ax.legend(frameon=False)
            ax.grid(True, linestyle=":", linewidth=0.7)

        self._plot_fns.append(_plot)

    def terminal_wealth_distr(
        self,
        X_expert: NDArray[np.float32],
        X_policy: NDArray[np.float32],
    ):
        X_expert = np.asarray(X_expert).reshape(-1)
        X_policy = np.asarray(X_policy).reshape(-1)

        def _plot(ax):
            xmin = min(X_expert.min(), X_policy.min())
            xmax = max(X_expert.max(), X_policy.max())
            xs = np.linspace(xmin, xmax, 300)

            if len(X_expert) > 1:
                kde_expert = gaussian_kde(X_expert)
                ax.plot(xs, kde_expert(xs), color="black", label="Expert")
            else:
                ax.axvline(X_expert[0], color="black", label="Expert")

            if len(X_policy) > 1:
                kde_policy = gaussian_kde(X_policy)
                ax.plot(
                    xs,
                    kde_policy(xs),
                    linestyle="--",
                    color="tab:blue",
                    label="Behavior cloning",
                )
            else:
                ax.axvline(
                    X_policy[0],
                    linestyle="--",
                    color="tab:blue",
                    label="Behavior cloning",
                )

            ax.set_xlabel("Terminal wealth $X_T$")
            ax.set_ylabel("Density")
            ax.set_title("Terminal Wealth Distribution")
            ax.legend(frameon=False)
            ax.grid(True, linestyle=":", linewidth=0.7)

        self._plot_fns.append(_plot)

    def expected_utility(self, U_expert: float, U_policy: float):
        def _plot(ax):
            values = [U_expert, U_policy]
            labels = ["Expert", "BC"]

            ax.bar(labels, values, width=0.6)
            ax.set_ylabel("Expected utility")
            ax.set_title("Expected Utility Comparison")
            ax.grid(axis="y", linestyle=":", linewidth=0.7)

            for i, v in enumerate(values):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        self._plot_fns.append(_plot)

    def rollout_drift(self, errors: NDArray[np.float32]):
        def _plot(ax):
            ax.plot(errors, color="black")
            ax.axhline(0, color="gray", linestyle=":", linewidth=1)

            ax.set_xlabel("Time step")
            ax.set_ylabel(r"$|\pi_{\mathrm{BC}} - \pi_{\mathrm{expert}}|$")
            ax.set_title("BC Action Error Over Time")
            ax.grid(True, linestyle=":", linewidth=0.7)

        self._plot_fns.append(_plot)

    def show(
        self,
        savepath: str | None = None,
        dpi: int = 300,
    ):
        n = len(self._plot_fns)
        if n == 0:
            return

        # Cap at 5 plots (3 top, 2 bottom)
        plot_fns = self._plot_fns[:5]

        nrows = 2 if n > 3 else 1
        base_height = 2.6
        fig_width = 10

        fig = plt.figure(figsize=(fig_width, base_height * nrows))

        gs = fig.add_gridspec(
            nrows=nrows,
            ncols=6,
            height_ratios=[1] * nrows,
            hspace=0.50,
            wspace=1.5,
        )

        axes = []

        # --- First row: 3 plots ---
        if n >= 1:
            axes.append(fig.add_subplot(gs[0, 0:2]))
        if n >= 2:
            axes.append(fig.add_subplot(gs[0, 2:4]))
        if n >= 3:
            axes.append(fig.add_subplot(gs[0, 4:6]))

        # --- Second row: centered 2 plots ---
        if n >= 4:
            axes.append(fig.add_subplot(gs[1, 1:3]))
        if n >= 5:
            axes.append(fig.add_subplot(gs[1, 3:5]))

        for ax, plot_fn in zip(axes, plot_fns):
            plot_fn(ax)

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
