import math

import numpy as np
import torch

from consts import MertonConsts


class MertonEnv:
    """
    Gym-like wrapper around the time-dependent Merton model.

    Each episode is one trajectory of N = T / delta_t steps.
    The agent chooses a portfolio weight pi at each step;
    the environment returns the next state and a scalar reward.

    Attributes:
        params (MertonConsts): Model parameters.
        N (int): Number of steps per episode.
        state_dim (int): Dimensionality of the state vector.
        reward_type (str): 'log_wealth' for per-step log-returns,
                           'terminal'  for terminal CRRA utility only.
        rng (Generator): Seeded random number generator.
        mu (float = 0.0): Mean of the trajectory.
        sigma (float = 0.0): Volatility of the trajectory.
        X (float = 0.0): Current wealth.
        t (int = 0): Current timestep.
        R_prev (float = 0.0): Previous return.
        returns (list[float] = []): List of returns throughout the trajectory.
        done (bool = True): True, if the trajectory is ended.
    """

    def __init__(
        self,
        params: MertonConsts | None = None,
        state_type: str = "default",
        reward_type: str = "log_wealth",
        seed: int = 0,
    ) -> None:
        self.params = params or MertonConsts()
        self.N = int(self.params.T / self.params.delta_t)
        self.state_type = state_type
        self.reward_type = reward_type
        self.rng = np.random.default_rng(seed=seed)

        # State dimensionality depends on state_type
        if state_type == "pomdp":
            self.state_dim = 1
        elif state_type in ("default", "full"):
            self.state_dim = 3
        else:
            self.state_dim = 3

        # Episode-level variables (set in reset)
        self.mu: float = 0.0
        self.sigma: float = 0.0
        self.X: float = 0.0
        self.t: int = 0
        self.R_prev: float = 0.0
        self.returns: list[float] = []
        self.done: bool = True

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """
        Resets the environment for a new episode.

        Args:
            seed (int | None = None): Optional seed for this episode's RNG.

        Returns:
            torch.Tensor: Initial state tensor.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

        # Sample per-trajectory mu, sigma (time-dependent Merton)
        self.mu = self.params.mu + self.params.sigma * self.rng.standard_normal()
        self.sigma = self.rng.lognormal(
            mean=math.log(self.params.sigma), sigma=self.params.sigma
        )

        self.X = self.params.init_wealth
        self.t = 0
        self.R_prev = 0.0
        self.returns = [0.0]
        self.done = False

        return self._get_state()

    def step(self, pi: float) -> tuple[torch.Tensor, float, bool]:
        """
        Advance one time step.

        Args:
            pi (float): Portfolio allocation to the risky asset.

        Returns:
            tuple[torch.Tensor, float, bool]: Tuple of (next_state, reward, done).
        """
        assert not self.done, "Episode finished — call reset()."

        epsilon = self.rng.standard_normal()
        R = (
            self.mu * self.params.delta_t
            + self.sigma * self.params.delta_t**0.5 * epsilon
        )

        X_old = self.X
        X_new = X_old * (
            1
            + self.params.r * self.params.delta_t
            + pi * (R - self.params.r * self.params.delta_t)
        )
        X_new = max(X_new, 1e-8)  # Prevent negative/zero wealth

        # Reward
        G = X_new / X_old  # Gross portfolio return
        if self.reward_type == "crra":
            # Per-step CRRA reward: r_t = (G^(1-γ) - 1) / (1-γ)
            # Myopic optimal action matches full-horizon Merton π*
            gamma = self.params.gamma
            reward = (G ** (1 - gamma) - 1) / (1 - gamma)
        elif self.reward_type == "log_wealth":
            reward = math.log(G)
        elif self.reward_type == "terminal":
            reward = 0.0
        else:
            reward = math.log(G)

        self.X = X_new
        self.R_prev = R
        self.returns.append(R)
        self.t += 1
        self.done = self.t >= self.N

        # Terminal CRRA bonus
        if self.done and self.reward_type == "terminal":
            gamma = self.params.gamma
            reward = (X_new ** (1 - gamma)) / (1 - gamma)

        next_state = self._get_state() if not self.done else self._terminal_state()

        return next_state, reward, self.done

    def _get_state(self) -> torch.Tensor:
        """
        Construct the state tensor for the current time step.

        Returns:
            torch.Tensor: Current state.
        """
        if self.state_type == "pomdp":
            return torch.tensor([self.R_prev], dtype=torch.float32)
        elif self.state_type == "full":
            return torch.tensor(
                [self.t / self.N, self.mu, self.sigma], dtype=torch.float32
            )
        else:  # default
            return torch.tensor(
                [self.t / self.N, self.R_prev, math.log(max(self.X, 1e-8))],
                dtype=torch.float32,
            )

    def _terminal_state(self) -> torch.Tensor:
        """
        Return a dummy terminal state (won't be used for actions).

        Returns:
            torch.Tensor: Dummy state.
        """
        return self._get_state()

    def expert_action(self) -> float:
        """
        Return the Merton optimal allocation for the current trajectory.

        Returns:
            float: Optimal allocation for the current trajectory.
        """
        return (self.mu - self.params.r) / (self.params.gamma * self.sigma**2)


class VectorizedMertonEnv:
    """
    Runs multiple MertonEnv instances in parallel (synchronous).
    All environments step together — when one finishes, it auto-resets.

    Attributes:
        n_envs (int): Number of parallel environments.
        envs (list[MertonEnv]): The individual environments.
        state_dim (int): Dimensions of a single state.
    """

    def __init__(
        self,
        n_envs: int = 16,
        params: MertonConsts | None = None,
        state_type: str = "default",
        reward_type: str = "log_wealth",
        base_seed: int = 0,
    ) -> None:
        self.n_envs = n_envs
        self.envs = [
            MertonEnv(
                params=params,
                state_type=state_type,
                reward_type=reward_type,
                seed=base_seed + i,
            )
            for i in range(n_envs)
        ]
        self.state_dim = self.envs[0].state_dim
        self._episode_seed_counter = base_seed + n_envs

    def reset(self) -> torch.Tensor:
        """
        Reset all environments.

        Returns:
            torch.Tensor: Stacked states of dimension (n_envs, state_dim).
        """
        states = []
        for env in self.envs:
            states.append(env.reset(seed=self._episode_seed_counter))
            self._episode_seed_counter += 1
        return torch.stack(states)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Step all environments.

        Args:
            actions (torch.Tensor): (n_envs,) tensor of portfolio weights.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                (next_states, rewards, dones) — each of shape (n_envs,) or (n_envs, dim).
        """
        next_states = []
        rewards = []
        dones = []

        for i, env in enumerate(self.envs):
            pi = actions[i].item()
            ns, r, d = env.step(pi)
            if d:
                # Auto-reset
                ns = env.reset(seed=self._episode_seed_counter)
                self._episode_seed_counter += 1
            next_states.append(ns)
            rewards.append(r)
            dones.append(d)

        return (
            torch.stack(next_states),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool),
        )

    def expert_actions(self) -> torch.Tensor:
        """
        Return expert actions for all envs.

        Returns:
            torch.Tensor: List of expert actions.
        """
        return torch.tensor(
            [env.expert_action() for env in self.envs], dtype=torch.float32
        )
