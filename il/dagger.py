from collections.abc import Sized
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from il.bc import BC
from models.jump_diffusion import create_jump_diffusion_model
from models.merton import create_merton_model
from policies.analytic import (
    JumpDiffusionPolicy,
    MertonPolicy,
    TimeDependentJumpDiffusionPolicy,
    TimeDependentMertonPolicy,
)
from policies.base import Policy
from policies.wrappers import MixturePolicy
from utils.consts import JumpDiffusionConsts, MertonConsts, SystemConsts
from utils.seed import g


class DAgger:
    """
    DAgger implementation.

    Attributes:
        financial_model (FinancialModel): Financial model with the expert policy.
        expert_dataset (list[tuple[Tensor, Tensor]]): Expert trajectories.
        epochs (int = 10): Number of epochs during training.
        batch_size (int = 32): Batch size.
        policy (Policy): Our policy.
        expert_policy (Policy): Expert policy.
        K (int = 5): Number of learner rollouts in each epoch.
        base_lr (float): Initial learning rate for DAgger. It will decay as the dataset grows.
        state_type (str = "default"): Determines what the states should be.
        traj_dataset (bool = False): Whether to use the per-trajectory dataset or not.
        bc_optimizer (str = "sgd"): Optimizer to use for BC.
        bc_epochs (int = 10): Number of epochs to use for BC.
        bc_batch_size (int = 32): Batch size to use for BC.
    """

    def __init__(
        self,
        policy: Policy,
        epochs: int = 10,
        batch_size: int = 32,
        expert_policy: str = "merton",
        K: int = 5,
        m: int = 100,
        base_lr: float = 3e-4,
        state_type: str = "default",
        traj_dataset: bool = False,
        bc_optimizer: str = "sgd",
        bc_epochs: int = 10,
        bc_batch_size: int = 32,
    ) -> None:
        if expert_policy == "merton":
            self.financial_model = create_merton_model(policy_class=MertonPolicy)
            self.expert_policy = MertonPolicy(params=MertonConsts())
        elif expert_policy == "time_dep_merton":
            self.financial_model = create_merton_model(
                policy_class=TimeDependentMertonPolicy
            )
            self.expert_policy = TimeDependentMertonPolicy(params=MertonConsts())
        elif expert_policy == "jump_diffusion":
            self.financial_model = create_jump_diffusion_model(
                policy_class=JumpDiffusionPolicy
            )
            self.expert_policy = JumpDiffusionPolicy(params=JumpDiffusionConsts())
        elif expert_policy == "time_dep_jump_diffusion":
            self.financial_model = create_jump_diffusion_model(
                policy_class=TimeDependentJumpDiffusionPolicy
            )
            self.expert_policy = TimeDependentJumpDiffusionPolicy(
                params=JumpDiffusionConsts()
            )
        self.state_type = state_type
        self.traj_dataset = traj_dataset
        if self.traj_dataset:
            self.expert_dataset = self.financial_model.generate_trajectories(
                m=m, state_type=self.state_type
            )
        else:
            self.expert_dataset = self.financial_model.generate_data(
                m=m, state_type=self.state_type
            )
        self.epochs = epochs
        self.batch_size = batch_size
        self.policy = policy
        self.K = K
        self.expert_policy.eval()
        for p in self.expert_policy.parameters():
            p.requires_grad_(False)
        self.system_consts = SystemConsts()
        self.base_lr = base_lr
        self.bc_optimizer = bc_optimizer
        self.traj_dataset = traj_dataset
        self.bc_epochs = bc_epochs
        self.bc_batch_size = bc_batch_size

    def train(self):
        """
        Trains DAgger on the expert dataset.
        """
        self.bc = BC(
            D=self.expert_dataset,
            policy=self.policy,
            lr=self.base_lr,
            optimizer=self.bc_optimizer,
            epochs=self.bc_epochs,
            batch_size=self.bc_batch_size,
            traj_dataset=self.traj_dataset,
        )
        N_0 = len(cast(Sized, self.bc.dataloader.dataset))
        for t in range(self.epochs):
            print(f"DAgger Epoch {t + 1}\n-------------------------")
            N_t = len(cast(Sized, self.bc.dataloader.dataset))
            E_t = int(min(20, np.ceil(10 * N_t / N_0)))
            self.bc.epochs = E_t
            lr_t = self.base_lr * np.sqrt(N_0 / N_t)
            for param_group in self.bc.optimizer.param_groups:
                param_group["lr"] = lr_t
            batch_size_t = max(
                16, int(self.bc.dataloader.batch_size * np.sqrt(N_0 / N_t))
            )
            self.bc.dataloader = DataLoader(
                self.bc.dataset,
                batch_size=batch_size_t,
                shuffle=True,
                generator=g,
                num_workers=0,
            )
            self.bc.train()
            rollout_seed = self.system_consts.dagger_base_seed + t * self.K
            rollout_rng = np.random.default_rng(seed=rollout_seed)
            mixture_policy = MixturePolicy(
                policy1=self.expert_policy,
                policy2=self.bc.policy,
                threshold=max(0.0, 1.0 - t / self.epochs),
                rng=rollout_rng,
            )
            for k in range(self.K):
                with torch.no_grad():
                    traj = self.financial_model.simulate_trajectory(
                        policy=mixture_policy,
                        seed=self.system_consts.dagger_base_seed + t * self.K + k,
                        state_type=self.state_type,
                    )
                states = []
                expert_actions = []
                for state, _, _, pi_star in traj:
                    states.append(state.detach().clone())
                    expert_actions.append(pi_star.detach().clone())
                self.bc.dataset.add(states=states, actions=expert_actions)
            print(f"Dataset size after epoch {t + 1}: {len(self.bc.dataset)}")
