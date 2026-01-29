import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

class Plots:
    """
    Utility class for creating visualizations
    """
    def __init__(self):
        pass

    def loss_training_step(self, losses: list[float]):
        """
        Plots training loss against each time step.

        Args:
            losses (list[float]): List of training losses.
        """
        plt.figure()
        plt.plot(losses)
        plt.xlabel("Training step")
        plt.ylabel("MSE loss")
        plt.title("BC Training Loss")
        plt.grid(True)
        plt.show()

    def action_time(self, expert_traj: list[tuple], policy_traj: list[tuple]):
        """
        Plots actions of the expert policy and the learned policy using a single rollout
        and same randomness.

        Args:
            expert_traj (list[tuple]): Expert trajectory.
            policy_traj (list[tuple]): Policy trajectory.
        """
        expert_actions = [step[1] for step in expert_traj]
        policy_actions = [step[1] for step in policy_traj]
        plt.figure()
        plt.plot(expert_actions, label="Expert", linewidth=2)
        plt.plot(policy_actions, label="BC", linestyle="--")
        plt.xlabel("Time step")
        plt.ylabel("Portfolio weight")
        plt.title("Action vs Time")
        plt.legend()
        plt.grid(True)
        plt.ticklabel_format(axis="y", style="plain", useOffset=False)
        plt.show()

    def terminal_wealth_distr(self, X_expert: NDArray[np.float32], X_policy: NDArray[np.float32]):
        """
        Plots the terminal wealth distribution of the expert and learned policy.

        Args:
            X_expert (NDArray[np.float32]): List of terminal wealth values simulated using the expert policy.
            X_policy (NDArray[np.float32]): List of terminal wealth values simulated using the learnt policy.
        """
        plt.figure()
        plt.hist(X_expert, bins=50, alpha=0.6, label="Expert", density=True)
        plt.hist(X_policy, bins=50, alpha=0.6, label="BC", density=True)
        plt.xlabel("Terminal wealth $X_T$")
        plt.ylabel("Density")
        plt.title("Terminal Wealth Distribution")
        plt.legend()
        plt.grid(True)
        plt.show()

    def expected_utility(self, U_expert: np.float32, U_policy: np.float32):
        """
        Plots the expected utility for the expert and learnt policy.

        Args:
            U_expert (np.float32): Utility of the expert policy.
            U_policy (np.float32): Utility of the learnt policy.
        """
        plt.figure()
        plt.bar(["Expert", "BC"], [U_expert, U_policy])
        plt.ylabel("Expected Utility")
        plt.title("Expected Utility Comparison")
        plt.grid(axis="y")
        plt.show()

    def rollout_drift(self, errors: NDArray[np.float32]):
        """
        Plots the rollout drift between the expert and learnt policy.

        Args:
            errors (NDArray[np.float32]): List of errors at each time step.
        """
        plt.figure()
        plt.plot(errors)
        plt.xlabel("Time step")
        plt.ylabel("|π_BC - π_expert|")
        plt.title("BC Action Error Over Time")
        plt.grid(True)
        plt.show()
