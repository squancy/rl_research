from typing import Any

import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Wraps a PyTorch dataset around the per-trajectory dataset.
    This means that a single state and action are a set of states and actions
    visited by the policy along a trajectory, respectively.

    Attributes:
        actions (list[torch.Tensor]): List of actions.
        mean (torch.Tensor): Mean of states.
        std (torch.Tensor): Standard deviation of states.
        states (list[torch.Tensor]): Normalized states.
    """

    def __init__(self, trajectories: list[list[list[Any, Any]]]) -> None:
        self._raw_states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []

        for states, actions in trajectories:
            states_tensor = torch.stack(states)
            actions_tensor = torch.stack(actions)

            if states_tensor.ndim == 3 and states_tensor.shape[1] == 1:
                states_tensor = states_tensor.squeeze(1)

            if actions_tensor.ndim == 3 and actions_tensor.shape[1] == 1:
                actions_tensor = actions_tensor.squeeze(1)

            if actions_tensor.ndim == 1:
                actions_tensor = actions_tensor.unsqueeze(-1)

            self._raw_states.append(states_tensor.float())
            self.actions.append(actions_tensor.float())

        all_states = torch.cat(self._raw_states, dim=0)
        self.mean = all_states.mean(dim=0)
        self.std = all_states.std(dim=0) + 1e-8

        self.states = [(s - self.mean) / self.std for s in self._raw_states]

    def add(self, states: list[torch.Tensor], actions: list[torch.Tensor]) -> None:
        """
        Appends a single trajectory and recomputes normalization statistics.

        Args:
            states: List of state tensors for the new trajectory.
            actions: List of action tensors for the new trajectory.
        """
        states_tensor = torch.stack(states).float()
        actions_tensor = torch.stack(actions).float()

        if states_tensor.ndim == 3 and states_tensor.shape[1] == 1:
            states_tensor = states_tensor.squeeze(1)
        if actions_tensor.ndim == 3 and actions_tensor.shape[1] == 1:
            actions_tensor = actions_tensor.squeeze(1)
        if actions_tensor.ndim == 1:
            actions_tensor = actions_tensor.unsqueeze(-1)

        self._raw_states.append(states_tensor)
        self.actions.append(actions_tensor)

        all_states = torch.cat(self._raw_states, dim=0)
        self.mean = all_states.mean(dim=0)
        self.std = all_states.std(dim=0) + 1e-8
        self.states = [(s - self.mean) / self.std for s in self._raw_states]

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx]


class ExpertDataset(Dataset):
    """
    Wraps a PyTorch dataset around the expert dataset.

    Attributes:
        actions (torch.Tensor): Tensor of expert actions.
        mean (torch.Tensor): Mean of the input data.
        std (torch.Tensor): Variance of the input data.
        states (torch.Tensor): Normalized states.
        raw_states (torch.Tensor): Unnormalized states.
    """

    def __init__(
        self,
        data: list[tuple],
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        self.raw_states = torch.stack([x[0] for x in data]).float()
        self.actions = torch.stack(
            [torch.as_tensor(x[1], dtype=torch.float32).view(1) for x in data]
        ).float()

        if mean is None or std is None:
            self.mean = self.raw_states.mean(dim=0)
            self.std = self.raw_states.std(dim=0) + 1e-8
        else:
            self.mean = mean
            self.std = std
        self.states = ((self.raw_states - self.mean) / self.std).float()

    def add(self, states: list[torch.Tensor], actions: list[torch.Tensor]) -> None:
        """
        Adds data to the dataset. Uses the mean and variance
        of the instance.

        Args:
            states (list[torch.Tensor]): List of (state, action) paris to add.
        """
        s = torch.stack(states).float()
        a = torch.stack(
            [torch.as_tensor(action, dtype=torch.float32).view(1) for action in actions]
        ).float()
        self.raw_states = torch.cat([self.raw_states, s])
        self.actions = torch.cat([self.actions, a])
        self.mean = self.raw_states.mean(dim=0)
        self.std = self.raw_states.std(dim=0) + 1e-8
        self.states = ((self.raw_states - self.mean) / self.std).float()

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx]
