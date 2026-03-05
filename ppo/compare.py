from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.merton import create_merton_model
from plotting.utils import plot_kde
from policies.analytic import TimeDependentMertonPolicy
from policies.learnable import NNPolicy
from ppo.agent import PPO
from utils.consts import PPOConfig


def compare_ppo_vs_il_ppo(
    il_policy: NNPolicy | None = None,
    il_mean: torch.Tensor | None = None,
    il_std: torch.Tensor | None = None,
    config: PPOConfig | None = None,
    savepath: str | None = None,
    dpi: int = 300,
) -> tuple[PPO, PPO | None]:
    """
    Run PPO from scratch and (optionally) IL-pretrained PPO side by side.

    Args:
        il_policy (NNPolicy | None = None): Pre-trained NNPolicy from BC or DAgger (optional).
        il_mean (torch.Tensor | None = None): State mean from IL training dataset.
        il_std (torch.Tensor | None = None): State std from IL training dataset.
        config (PPOConfig | None = None): PPO configuration.
        savepath (str | None = None): Path to save comparison plot.
        dpi (int = 300): DPI resolution of the plot.

    Returns:
        tuple[PPO, PPO]: PPO scratch and PPO pretrained: the two trained PPO instances.
    """
    config = config or PPOConfig()

    print("=" * 60)
    print("Training PPO from scratch")
    print("=" * 60)
    ppo_scratch = PPO(config)
    ppo_scratch.train()

    ppo_pretrained = None
    if il_policy is not None:
        print("\n" + "=" * 60)
        print("Training PPO with IL pre-training")
        print("=" * 60)
        ppo_pretrained = PPO(config)
        ppo_pretrained.load_pretrained_actor(il_policy, il_mean, il_std)
        ppo_pretrained.train()

    fin_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    expert_util = fin_model.evaluate(
        policy=fin_model.expert_policy,
        m=config.eval_episodes,
        expert=True,
        state_type=config.state_type,
    )
    X_expert = fin_model.terminal_wealths(
        policy=fin_model.expert_policy,
        m=config.eval_episodes,
        expert=True,
        state_type=config.state_type,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1) Learning curves — y-axis clipped to readable range
    ax = axes[0]
    ax.plot(
        ppo_scratch.eval_steps,
        ppo_scratch.eval_utilities,
        color="#d63031",
        lw=2,
        label="PPO (random init)",
    )
    if ppo_pretrained is not None:
        ax.plot(
            ppo_pretrained.eval_steps,
            ppo_pretrained.eval_utilities,
            color="#0984e3",
            lw=2,
            label="PPO (IL pre-trained)",
        )
    ax.axhline(
        expert_util,
        color="#2d3436",
        linestyle="--",
        lw=1.5,
        label=f"Expert ({expert_util:.4f})",
    )
    # Symlog y-axis: linear near zero (where expert lives) and
    # log-compressed for extreme negatives from random-init PPO.
    linthresh = max(1.0, abs(expert_util) * 20)
    ax.set_yscale("symlog", linthresh=linthresh)
    # Set y-limits: top slightly above expert, bottom covers all data
    all_utils = ppo_scratch.eval_utilities[:]
    if ppo_pretrained is not None:
        all_utils += ppo_pretrained.eval_utilities
    finite_utils = [u for u in all_utils if np.isfinite(u)]
    if finite_utils:
        y_bottom = min(finite_utils) * 1.5
    else:
        y_bottom = expert_util * 100
    y_top = abs(expert_util) * 2 if expert_util < 0 else expert_util * 1.5
    ax.set_ylim(bottom=y_bottom, top=y_top)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(r"Expected utility $\mathbb{E}[U(X_T)]$")
    ax.set_title("Sample Efficiency: PPO vs IL+PPO")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    # 2) Median terminal wealth over training
    ax = axes[1]
    ax.plot(
        ppo_scratch.eval_steps,
        ppo_scratch.eval_mean_wealth,
        color="#d63031",
        lw=2,
        label="PPO (random init)",
    )
    if ppo_pretrained is not None:
        ax.plot(
            ppo_pretrained.eval_steps,
            ppo_pretrained.eval_mean_wealth,
            color="#0984e3",
            lw=2,
            label="PPO (IL pre-trained)",
        )
    ax.axhline(
        np.median(X_expert),
        color="#2d3436",
        linestyle="--",
        lw=1.5,
        label=f"Expert ({np.median(X_expert):.2f})",
    )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(r"Median $X_T$")
    ax.set_title("Median Terminal Wealth")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    # 3) Terminal wealth distribution (KDE)
    ax = axes[2]
    eval_scratch = ppo_scratch.get_eval_policy()
    X_scratch = fin_model.terminal_wealths(
        policy=eval_scratch, m=config.eval_episodes, state_type=config.state_type
    )
    X_expert_f = X_expert[np.isfinite(X_expert)]
    X_scratch_f = X_scratch[np.isfinite(X_scratch)]
    all_wealth = [X_expert_f, X_scratch_f]
    if ppo_pretrained is not None:
        eval_pre = ppo_pretrained.get_eval_policy()
        X_pre = fin_model.terminal_wealths(
            policy=eval_pre, m=config.eval_episodes, state_type=config.state_type
        )
        X_pre_f = X_pre[np.isfinite(X_pre)]
        all_wealth.append(X_pre_f)
    combined = np.concatenate(all_wealth)
    lo, hi = np.percentile(combined, [1, 95])
    pad = 0.05 * (hi - lo)
    xlim = (max(0, lo - pad), hi + pad)
    plot_kde(ax, X_expert_f, color="#2d3436", label="Expert", xlim=xlim)
    plot_kde(
        ax,
        X_scratch_f,
        color="#d63031",
        label="PPO (random)",
        linestyle="--",
        xlim=xlim,
    )
    if ppo_pretrained is not None:
        plot_kde(
            ax,
            X_pre_f,
            color="#0984e3",
            label="PPO (IL pre-trained)",
            linestyle="-.",
            xlim=xlim,
        )
    ax.set_xlim(xlim)
    ax.set_xlabel("Terminal wealth $X_T$")
    ax.set_ylabel("Density")
    ax.set_title("Terminal Wealth Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return ppo_scratch, ppo_pretrained
