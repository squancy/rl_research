from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import gaussian_kde
from torch.distributions import Normal

from consts import MertonConsts, PPOConfig
from merton_imitation import create_merton_model
from policy import NNPolicy, NormalizedPolicy, Policy, TimeDependentMertonPolicy
from ppo_env import MertonEnv, VectorizedMertonEnv
from seed import seed_everything

g = seed_everything(seed=42)


def _plot_kde(
    ax,
    data: np.ndarray,
    color: str,
    label: str,
    linestyle: str = "-",
    alpha: float = 0.85,
    fill_alpha: float = 0.10,
    xlim: tuple[float, float] | None = None,
) -> None:
    """
    Plot a KDE curve with optional fill on the given axes.

    Args:
        data (np.ndarray): Data to plot.
        color (str): Color of the KDE plot.
        label (str): Label of the plot.
        linestyle (str = "-"): Line style of the plot.
        alpha (float = 0.85): Transparency of the plot.
        fill_alpha (float = 0.10): Transparency of the fill.
        xlim (tuple[float, float]): Lower and upper bound for values
            on the the x-axis.
    """
    data = np.asarray(data).ravel()
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return
    if xlim is not None:
        data = data[(data >= xlim[0]) & (data <= xlim[1])]
        if len(data) < 2:
            return
        xs = np.linspace(xlim[0], xlim[1], 400)
    else:
        pad = 0.05 * (data.max() - data.min())
        xs = np.linspace(data.min() - pad, data.max() + pad, 400)
    kde = gaussian_kde(data)
    ys = kde(xs)
    ax.plot(xs, ys, color=color, alpha=alpha, lw=2, linestyle=linestyle, label=label)
    ax.fill_between(xs, ys, alpha=fill_alpha, color=color)


class PPOActorCritic(nn.Module):
    """
    Actor-Critic network for PPO with a Gaussian policy head.

    The actor outputs a mean action; a learnable log_std parameter
    controls the exploration noise. The critic outputs a scalar
    state-value estimate.

    Attributes:
        actor (nn.Sequential): Maps state to action mean.
        critic (nn.Sequential): Maps state to V(s).
        log_std (nn.Parameter): Log standard deviation of the Gaussian policy.
        action_scale (float): Soft action bound via tanh squashing.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        action_scale: float = 20.0,
        init_log_std: float = -0.5,
    ) -> None:
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.log_std = nn.Parameter(torch.tensor(init_log_std))
        self.action_scale = action_scale

    def forward_actor(self, state: torch.Tensor) -> Normal:
        """
        Given a state, returns a normal distribution with a mean
        (using the actor) and variance (using the learnable parameter).

        Args:
            state (torch.Tensor): State.

        Returns:
            Normal: normal distribution with given mean and variance.
        """
        raw_mean = self.actor(state).squeeze(-1)
        mean = self.action_scale * torch.tanh(raw_mean)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def forward_critic(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns the value function estimate V(s).

        Args:
            state (torch.Tensor): State.

        Returns:
            torch.Tensor: Value function estimate.
        """
        return self.critic(state).squeeze(-1)

    def act(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples an action.

        Args:
            state (torch.Tensor): State.
            deterministic (bool = False): True, if the policy is deterministic.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of
                sampled action, log probability of taking the sampled action and
                estimated value function using the critic.
        """
        dist = self.forward_actor(state)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.forward_critic(state)
        return action, log_prob, value

    def evaluate(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluates stored actions under the current policy.

        Args:
            states (torch.Tensor): List of states.
            actions (torch.Tensor): List of actions.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Log probabilities
                of taking the given actions, estimated values using the critic
                and the entropy of the distribution.
        """
        dist = self.forward_actor(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.forward_critic(states)
        return log_probs, values, entropy

    def load_actor_from_nn_policy(self, nn_policy: NNPolicy) -> None:
        """
        Copy weights from a pre-trained NNPolicy into the actor network.
        This enables warm-starting PPO from BC / DAgger.
        The NNPolicy.net architecture must match self.actor.

        Args:
            nn_policy (NNPolicy): Pretrained policy.
        """
        src_params = list(nn_policy.net.parameters())
        tgt_params = list(self.actor.parameters())
        assert len(src_params) == len(tgt_params), (
            f"Architecture mismatch: NNPolicy has {len(src_params)} param groups, "
            f"actor has {len(tgt_params)}."
        )
        for s, t in zip(src_params, tgt_params):
            assert s.shape == t.shape, f"Shape mismatch: {s.shape} vs {t.shape}"
            t.data.copy_(s.data)


class RolloutBuffer:
    """
    Stores transitions collected during rollout and computes returns / advantages.

    Attributes:
        states (list[torch.Tensor]): List of states collected during rollouts.
        actions (list[torch.Tensor]): List of actions collected during rollouts.
        log_probs (list[torch.Tensor]): List of log_probs collected during rollouts.
        rewards (list[torch.Tensor]): List of rewards collected during rollouts.
        values (list[torch.Tensor]): List of values collected during rollouts.
        dones (list[torch.Tensor]): List of dones collected during rollouts.
    """

    def __init__(self) -> None:
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []

    def store(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """
        Stores a set of new state, action, log probability, reward, value and done.

        Args:
            state (torch.Tensor): State to store.
            action (torch.Tensor): Action to store.
            log_prob (torch.Tensor): Log probability to store.
            reward (torch.Tensor): Reward to store.
            value (torch.Tensor): Value to store.
            done (torch.Tensor): Done to store.
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(
        self,
        last_value: torch.Tensor,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            last_value (torch.Tensor): V(s_{T+1}) for bootstrapping.
            gamma (float = 0.999): Discount factor.
            gae_lambda (float = 0.95): GAE lambda.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: tuple of advantages and returns
            both of shape (T * n_envs,).
        """
        T = len(self.rewards)
        values = torch.stack(self.values)  # (T, n_envs)
        rewards = torch.stack(self.rewards)  # (T, n_envs)
        dones = torch.stack(self.dones).float()  # (T, n_envs)

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(last_value)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        # Flatten (T, n_envs) -> (T * n_envs,)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        return advantages, returns

    def flatten(self) -> tuple[torch.Tensor, ...]:
        """
        Flatten stored states, actions and log probabilities to (T * n_envs, ...).

        Returns:
            tuple[torch.Tensor, ...]: tuple of states, actions and log probabilities.
        """
        states = torch.stack(self.states).reshape(-1, self.states[0].shape[-1])
        actions = torch.stack(self.actions).reshape(-1)
        log_probs = torch.stack(self.log_probs).reshape(-1)
        return states, actions, log_probs

    def clear(self) -> None:
        """
        Resets stored states, actions, log probabilities, rewards, values and dones
        to empty.
        """
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


class PPO:
    """
    PPO trainer for the time-dependent Merton portfolio allocation problem.

    Attributes:
        config (PPOConfig): Config variables.
        env (VectorizedMertonEnv): Environment for multiple trajectories.
        ac (PPOActorCritic): Actor-Critic network for PPO.
        optimizer (optim.Adam): Adam optimizer.
        state_mean (torch.Tensor | None = None): Mean of states.
        state_std (torch.Tensor | None = None): Standard deviation of states.
        total_steps: (int = 0): Total number of steps taken so far.
        eval_steps: (list[int] = []): Steps at which evaluation happened.
        eval_utilities (list[float] = []): Mean utilities at evaluation time.
        eval_mean_wealth (list[float] = []): Mean wealths at evaluation time.
        policy_losses (list[float] = []): List of PPO policy losses.
        value_losses (list[float] = []): List of value losses.
        entropy_values (list[float] = []): List of entropies.
        fin_model (MertonModel): Time-dependent merton model.
    """

    def __init__(self, config: PPOConfig | None = None) -> None:
        self.config = config or PPOConfig()
        c = self.config

        # Environment
        self.env = VectorizedMertonEnv(
            n_envs=c.n_envs,
            params=MertonConsts(),
            state_type=c.state_type,
            reward_type=c.reward_type,
            base_seed=c.env_base_seed,
        )

        # Actor-Critic
        self.ac = PPOActorCritic(
            state_dim=self.env.state_dim,
            hidden_dim=c.hidden_dim,
            action_scale=c.action_scale,
            init_log_std=c.init_log_std,
        )
        self.optimizer = optim.Adam(self.ac.parameters(), lr=c.lr, eps=1e-5)

        # Normalization stats (set when loading pre-trained actor)
        self.state_mean: torch.Tensor | None = None
        self.state_std: torch.Tensor | None = None

        # Running state normalization (if no pre-trained stats)
        self._state_sum = torch.zeros(self.env.state_dim)
        self._state_sq_sum = torch.zeros(self.env.state_dim)
        self._state_count = 0

        # Logging
        self.total_steps = 0
        self.eval_steps: list[int] = []
        self.eval_utilities: list[float] = []
        self.eval_mean_wealth: list[float] = []
        self.policy_losses: list[float] = []
        self.value_losses: list[float] = []
        self.entropy_values: list[float] = []

        # Financial model for evaluation
        self.fin_model = create_merton_model(policy_class=TimeDependentMertonPolicy)

    def load_pretrained_actor(
        self,
        nn_policy: NNPolicy,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        """
        Warm-start the PPO actor from a BC/DAgger pre-trained NNPolicy.

        Also copies the action_scale (pi_scale) from the source policy
        and sets the normalization statistics so that PPO uses the same
        input normalization as the IL policy did.

        Args:
            nn_policy (NNPolicy): Pre-trained NNPolicy (from BC or DAgger).
            mean (torch.Tensor): Dataset mean used during IL training.
            std (torch.Tensor): Dataset std used during IL training.
        """
        # Copy actor weights
        self.ac.load_actor_from_nn_policy(nn_policy)
        # Match action scale
        self.ac.action_scale = nn_policy.pi_scale
        # Store normalization stats
        self.state_mean = mean.clone()
        self.state_std = std.clone()
        print(
            f"[PPO] Loaded pre-trained actor. "
            f"action_scale={self.ac.action_scale}, "
            f"state_mean={self.state_mean.tolist()}, "
            f"state_std={self.state_std.tolist()}"
        )

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize states using either pre-trained or running stats.

        Args:
            state (torch.Tensor): States to normalize.

        Returns:
            torch.Tensor: Normalized states.
        """
        if self.state_mean is not None and self.state_std is not None:
            return (state - self.state_mean) / self.state_std
        # Running normalization
        if self._state_count > 1:
            mean = self._state_sum / self._state_count
            var = self._state_sq_sum / self._state_count - mean**2
            std = torch.sqrt(var.clamp(min=1e-8))
            return (state - mean) / std
        return state

    def _update_running_stats(self, states: torch.Tensor) -> None:
        """
        Update running mean/variance from a batch of states.

        Args:
            states (torch.Tensor): Batch of states.
        """
        if self.state_mean is not None:
            return  # Using pre-trained stats, skip
        # states: (n_envs, state_dim)
        self._state_sum += states.sum(dim=0)
        self._state_sq_sum += (states**2).sum(dim=0)
        self._state_count += states.shape[0]

    def train(self) -> None:
        """Run the full PPO training loop."""
        c = self.config
        states = self.env.reset()  # (n_envs, state_dim)
        self._update_running_stats(states)

        n_updates = c.total_timesteps // (c.rollout_steps * c.n_envs)

        for update in range(n_updates):
            buffer = RolloutBuffer()

            # Collect rollout
            for step in range(c.rollout_steps):
                norm_states = self._normalize_state(states)
                with torch.no_grad():
                    actions, log_probs, values = self.ac.act(norm_states)

                next_states, rewards, dones = self.env.step(actions)
                self._update_running_stats(next_states)

                buffer.store(
                    state=norm_states,
                    action=actions,
                    log_prob=log_probs,
                    reward=rewards,
                    value=values,
                    done=dones,
                )

                states = next_states

            # Bootstrap value for last state
            with torch.no_grad():
                norm_last = self._normalize_state(states)
                last_value = self.ac.forward_critic(norm_last)

            # GAE
            advantages, returns = buffer.compute_gae(
                last_value, gamma=c.gamma, gae_lambda=c.gae_lambda
            )

            # Flatten buffer
            flat_states, flat_actions, flat_log_probs = buffer.flatten()

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO update
            total_samples = flat_states.shape[0]
            update_policy_loss = 0.0
            update_value_loss = 0.0
            update_entropy = 0.0
            n_mini = 0

            for _epoch in range(c.n_epochs):
                indices = torch.randperm(total_samples)
                for start in range(0, total_samples, c.batch_size):
                    end = min(start + c.batch_size, total_samples)
                    idx = indices[start:end]

                    mb_states = flat_states[idx]
                    mb_actions = flat_actions[idx]
                    mb_old_log_probs = flat_log_probs[idx]
                    mb_advantages = advantages[idx]
                    mb_returns = returns[idx]

                    new_log_probs, new_values, entropy = self.ac.evaluate(
                        mb_states, mb_actions
                    )

                    # Clipped surrogate objective
                    ratio = (new_log_probs - mb_old_log_probs).exp()
                    surr1 = ratio * mb_advantages
                    surr2 = (
                        torch.clamp(ratio, 1.0 - c.clip_eps, 1.0 + c.clip_eps)
                        * mb_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss (clipped)
                    value_loss = nn.functional.mse_loss(new_values, mb_returns)

                    # Entropy bonus
                    entropy_loss = -entropy.mean()

                    loss = (
                        policy_loss + c.vf_coef * value_loss + c.ent_coef * entropy_loss
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), c.max_grad_norm)
                    self.optimizer.step()

                    update_policy_loss += policy_loss.item()
                    update_value_loss += value_loss.item()
                    update_entropy += entropy.mean().item()
                    n_mini += 1

            self.policy_losses.append(update_policy_loss / max(n_mini, 1))
            self.value_losses.append(update_value_loss / max(n_mini, 1))
            self.entropy_values.append(update_entropy / max(n_mini, 1))

            self.total_steps += c.rollout_steps * c.n_envs

            # Periodic evaluation
            if (
                len(self.eval_steps) == 0
                or self.total_steps - self.eval_steps[-1] >= c.eval_interval
            ):
                util, mean_w = self._evaluate()
                self.eval_steps.append(self.total_steps)
                self.eval_utilities.append(util)
                self.eval_mean_wealth.append(mean_w)
                print(
                    f"[PPO] Steps: {self.total_steps:>8d} | "
                    f"Policy loss: {self.policy_losses[-1]:.4f} | "
                    f"Value loss: {self.value_losses[-1]:.4f} | "
                    f"Entropy: {self.entropy_values[-1]:.4f} | "
                    f"E[U]: {util:.4f} | "
                    f"med(X_T): {mean_w:.4f}"
                )

            buffer.clear()

    @torch.no_grad()
    def _evaluate(self) -> tuple[float, float]:
        """
        Evaluate the current policy over fresh episodes.

        Returns:
            tuple[float, float]: tuple of mean_utility and mean_terminal_wealth.
        """
        c = self.config
        eval_env = MertonEnv(
            params=MertonConsts(),
            state_type=c.state_type,
            reward_type=c.reward_type,
        )

        utilities = []
        terminal_wealths = []
        gamma = MertonConsts().gamma

        for ep in range(c.eval_episodes):
            state = eval_env.reset(seed=c.eval_seed + ep)
            done = False
            while not done:
                norm_state = self._normalize_state(state)
                dist = self.ac.forward_actor(norm_state.unsqueeze(0))
                action = dist.mean.squeeze(0)  # Deterministic at eval
                state, reward, done = eval_env.step(action.item())
            X_T = eval_env.X
            U = (X_T ** (1 - gamma)) / (1 - gamma)
            utilities.append(U)
            terminal_wealths.append(X_T)

        return float(np.mean(utilities)), float(np.median(terminal_wealths))

    def get_eval_policy(self) -> NormalizedPolicy:
        """
        Return a NormalizedPolicy wrapping the PPO actor for use with
        the existing FinancialModel evaluation infrastructure.

        Returns:
            NormalizedPolicy: PPO actor policy normalized using
                the given mean and std or rolling statistics.
        """

        # Create an NNPolicy-like wrapper around the actor
        class _ActorAsPolicy(Policy):
            def __init__(self, ac: PPOActorCritic):
                super().__init__()
                self.ac = ac

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                dist = self.ac.forward_actor(x.unsqueeze(0) if x.dim() == 1 else x)
                return dist.mean.squeeze(0)

        actor_policy = _ActorAsPolicy(self.ac)
        if self.state_mean is not None and self.state_std is not None:
            return NormalizedPolicy(
                policy=actor_policy,
                mean=self.state_mean,
                std=self.state_std,
            )
        elif self._state_count > 1:
            mean = self._state_sum / self._state_count
            var = self._state_sq_sum / self._state_count - mean**2
            std = torch.sqrt(var.clamp(min=1e-8))
            return NormalizedPolicy(policy=actor_policy, mean=mean, std=std)
        else:
            return NormalizedPolicy(
                policy=actor_policy,
                mean=torch.zeros(self.env.state_dim),
                std=torch.ones(self.env.state_dim),
            )

    def plot_results(
        self,
        include_expert: bool = True,
        savepath: str | None = None,
        dpi: int = 300,
    ) -> None:
        """
        Plot PPO training curves and comparison to expert.

        Shows:
          1. Expected utility over training (y-axis clipped to readable range)
          2. Mean terminal wealth over training
          3. Terminal wealth distribution (KDE, PPO vs Expert)

        Args:
            include_expert (bool = True): True, if the expert policy should be included
                in the plots.
            savepath (str | None = None): Path to save the plot.
            dpi (int = 300): DPI resolution of the plot.
        """

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        expert_util = None
        if include_expert:
            expert_util = self.fin_model.evaluate(
                policy=self.fin_model.expert_policy,
                m=self.config.eval_episodes,
                expert=True,
                state_type=self.config.state_type,
            )

        # 1) Expected utility over training
        ax = axes[0]
        ax.plot(
            self.eval_steps,
            self.eval_utilities,
            color="#0984e3",
            lw=2,
            label="PPO",
        )
        if expert_util is not None:
            ax.axhline(
                expert_util,
                color="#2d3436",
                linestyle="--",
                lw=1.5,
                label=f"Expert ({expert_util:.4f})",
            )
            # Symlog y-axis: linear near zero, log-compressed for extremes
            linthresh = max(1.0, abs(expert_util) * 20)
            ax.set_yscale("symlog", linthresh=linthresh)
            finite_utils = [u for u in self.eval_utilities if np.isfinite(u)]
            if finite_utils:
                y_bottom = min(finite_utils) * 1.5
            else:
                y_bottom = expert_util * 100
            y_top = abs(expert_util) * 2 if expert_util < 0 else expert_util * 1.5
            ax.set_ylim(bottom=y_bottom, top=y_top)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Expected utility")
        ax.set_title("PPO Learning Curve")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

        # 2) Median terminal wealth over training
        ax = axes[1]
        ax.plot(
            self.eval_steps,
            self.eval_mean_wealth,
            color="#0984e3",
            lw=2,
            label="PPO",
        )
        if include_expert:
            X_expert = self.fin_model.terminal_wealths(
                policy=self.fin_model.expert_policy,
                m=self.config.eval_episodes,
                expert=True,
                state_type=self.config.state_type,
            )
            ax.axhline(
                np.median(X_expert),
                color="#2d3436",
                linestyle="--",
                lw=1.5,
                label=f"Expert ({np.median(X_expert):.2f})",
            )
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Median")
        ax.set_title("Median Terminal Wealth")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

        # 3) Terminal wealth distribution (KDE)
        ax = axes[2]
        eval_policy = self.get_eval_policy()
        X_ppo = self.fin_model.terminal_wealths(
            policy=eval_policy,
            m=self.config.eval_episodes,
            state_type=self.config.state_type,
        )
        X_ppo = X_ppo[np.isfinite(X_ppo)]
        all_wealth = np.concatenate(
            [X_expert[np.isfinite(X_expert)], X_ppo] if include_expert else [X_ppo]
        )
        lo, hi = np.percentile(all_wealth, [1, 95])
        pad = 0.05 * (hi - lo)
        xlim = (max(0, lo - pad), hi + pad)
        if include_expert:
            _plot_kde(
                ax,
                X_expert[np.isfinite(X_expert)],
                color="#2d3436",
                label="Expert",
                xlim=xlim,
            )
        _plot_kde(ax, X_ppo, color="#0984e3", label="PPO", linestyle="--", xlim=xlim)
        ax.set_xlim(xlim)
        ax.set_xlabel("Terminal wealth")
        ax.set_ylabel("Density")
        ax.set_title("Terminal Wealth Distribution")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        plt.show()
        plt.close(fig)


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
    _plot_kde(ax, X_expert_f, color="#2d3436", label="Expert", xlim=xlim)
    _plot_kde(
        ax,
        X_scratch_f,
        color="#d63031",
        label="PPO (random)",
        linestyle="--",
        xlim=xlim,
    )
    if ppo_pretrained is not None:
        _plot_kde(
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


if __name__ == "__main__":
    from il import BC

    config = PPOConfig(
        total_timesteps=500_000,
        n_envs=16,
        rollout_steps=1250,
        eval_interval=20_000,
        state_type="default",
    )

    # 1) Train BC to get a pre-trained policy
    print("=" * 60)
    print("Phase 1: Pre-training with Behavior Cloning")
    print("=" * 60)
    fin_model = create_merton_model(policy_class=TimeDependentMertonPolicy)
    trajectories = fin_model.generate_trajectories(m=100, state_type="default")
    il_policy = NNPolicy(in_dim=3, pi_scale=20.0)
    bc = BC(
        D=trajectories,
        policy=il_policy,
        lr=1e-3,
        epochs=10,
        batch_size=16,
        traj_dataset=False,
        optimizer="adam",
    )
    bc.train()

    il_mean = bc.dataset.mean
    il_std = bc.dataset.std

    # 2) Compare PPO (random) vs PPO (IL pre-trained)
    print("\n" + "=" * 60)
    print("Phase 2: PPO Training")
    print("=" * 60)
    ppo_scratch, ppo_pretrained = compare_ppo_vs_il_ppo(
        il_policy=il_policy,
        il_mean=il_mean,
        il_std=il_std,
        config=config,
        savepath="plots/ppo_vs_il_ppo.png",
    )
