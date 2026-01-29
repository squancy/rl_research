from dataclasses import dataclass

@dataclass
class GeneralConsts:
    """A data class for storing general constants
    
    Attributes:
        init_wealth (float): Initial wealth.
    """
    init_wealth: float = 1.0

@dataclass
class MertonConsts:
    """A data class for storing the parameters of the Merton model

    Attributes:
        mu (float): Expected risky return (annualized).
        r (float): Risk-free rate (annualized).
        sigma (float): Risky asset volatility (annualized).
        gamma (float): Risk aversion. 
        delta_t (float): Time step.
        T (int): Total horizon in years.
        A (float): Hyperparameter for determining time dependent risky return.
    """
    mu: float = 0.08
    r: float = 0.02
    sigma: float = 0.2
    gamma: float = 5
    delta_t: float = 1 / 250 # daily
    T: int = 1
    A: float = 0.08