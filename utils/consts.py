from dataclasses import dataclass


@dataclass
class SystemConsts:
    """
    A data class for storing system constants.

    Attributes:
        train_base_seed (int): Base seed for training.
        test_base_seed_expert (int): Base seed for testing the expert policy.
        test_base_seed_policy (int): Base seed for testing the learned policy.
        terminal_wealth_base_seed_expert (int): Base seed for calculating terminal wealths using the expert policy.
        terminal_wealth_base_seed_policy (int): Base seed for calculating terminal wealths using the learned policy.
        dagger_base_seed (int): Base seed for DAgger.
    """

    train_base_seed: int = 1_000
    test_base_seed_expert: int = 2_000
    test_base_seed_policy: int = 3_000
    terminal_wealth_base_seed_expert: int = 4_000
    terminal_wealth_base_seed_policy: int = 5_000
    dagger_base_seed: int = 6_000


@dataclass
class GeneralConsts:
    """
    A data class for storing general constants.

    Attributes:
        init_wealth (float): Initial wealth.
        delta_t (float): Time step.
        T (int): Total horizon in years.
    """

    init_wealth: float = 1.0
    delta_t: float = 1 / 250  # daily
    T: int = 5


@dataclass
class MertonConsts(GeneralConsts):
    """
    A data class for storing the parameters of the Merton model.

    Attributes:
        mu (float): Expected risky return (annualized).
        r (float): Risk-free rate (annualized).
        sigma (float): Risky asset volatility (annualized).
        gamma (float): Risk aversion.
        A (float): Hyperparameter for determining time dependent risky return.
        distr_var (float): Variance of the distribution that generates per-trajectory
            mean and variance for the risky asset.
    """

    mu: float = 0.08
    r: float = 0.02
    sigma: float = 0.2
    gamma: float = 5
    A: float = 0.08
    distr_var: float = 0.1


@dataclass
class JumpDiffusionConsts(GeneralConsts):
    """
    A data class for storing parameters of the Jump Diffusion model.

    Attributes:
        mu (float): Expected risky return (annualized).
        r (float): Risk-free rate (annualized).
        sigma (float): Risky asset volatility (annualized).
        gamma (float): Risk aversion.
        lam (float): Intensity of the Poisson process.
        mu_J (float): Mean of jump.
        sigma_J (float): Variance of jump.
        distr_var (float): Variance of the distribution for per-trajectory sampling.
    """

    mu: float = 0.08
    r: float = 0.02
    sigma: float = 0.2
    gamma: float = 5
    lam: float = 0.5
    mu_J: float = -0.2
    sigma_J: float = 0.1
    distr_var: float = 0.1


@dataclass
class PPOConfig:
    """
    All PPO hyperparameters.

    Attributes:
        n_envs (int): Number of environments (trajectories).
        state_type (str): Type of the state used in trajectories.
        reward_type (str): Type of reward used in trajectories.
        hidden_dim (int): Hidden layer dimension size in the actor-critic network.
        action_scale (float): Number by which to scale actions.
        init_log_std (float): Initial log standard deviation.
        total_timesteps (int): Total number of timestaps to take.
        rollout_steps (int): Number of steps to take per episode.
        n_epochs (int): Number of epochs for PPO.
        batch_size (int): Batch size for PPO.
        gamma (float): Discount factor.
        gae_lambda (float): Lambda for generalized advantage estimation.
        clip_eps (float): Epsilon to use for clipping the PPO objective.
        vf_coef (float): Coefficient to use for the value function estimate
            in the loss function.
        ent_coef (float): Coefficient to use for the entropy estimate
            in the loss function.
        max_grad_norm (float): Maximum gradient norm used for gradient clipping.
        lr (float): Learning rate.
        eval_interval (int): Evaluate every N timesteps.
        eval_episodes (int): Number of episodes to evaluate on.
        eval_seed (int): Evaluation seed to reset the env.
        eval_base_seed (int): Evaluation starting seed.
    """

    # Environment
    n_envs: int = 16
    state_type: str = "default"
    reward_type: str = "crra"

    # Network
    hidden_dim: int = 64
    action_scale: float = 20.0
    init_log_std: float = -0.5

    # PPO
    total_timesteps: int = 500_000
    rollout_steps: int = 1250  # One full episode
    n_epochs: int = 10  # PPO update epochs per rollout
    batch_size: int = 256
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4

    # Evaluation
    eval_interval: int = 10_000  # Evaluate every N timesteps
    eval_episodes: int = 100
    eval_seed: int = SystemConsts.test_base_seed_policy

    # Seeding
    env_base_seed: int = 7000
