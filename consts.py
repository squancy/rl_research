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
    """

    mu: float = 0.08
    r: float = 0.02
    sigma: float = 0.2
    gamma: float = 5
    lam: float = 0.5
    mu_J: float = -0.2
    sigma_J: float = 0.1
