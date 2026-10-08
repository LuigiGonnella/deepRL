"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        """
        state: (B, sd)
        out: (B, action_dim * chunk_dim)
        """
        super().__init__(state_dim, action_dim, chunk_size)

        dims = [state_dim] + list(hidden_dims)

        layers = []

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            layers += [
                nn.Linear(in_dim, out_dim),
                nn.ReLU()
            ]

        layers.append(
            nn.Linear(hidden_dims[-1], action_dim * chunk_size)
        )

        self.net = nn.Sequential(
            *layers, #unpack objects
        )


    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0]

        #### forward pass --> pred chunk
        pred = self.net(state).reshape((B, self.chunk_size, self.action_dim)) 

        ### loss
        loss = torch.mean((pred - action_chunk) ** 2) #MSE

        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        return self.net(state).reshape((B, self.chunk_size, self.action_dim))
        


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)


        dims = [state_dim + action_dim * chunk_size + 1] + list(hidden_dims) #time concatenated with state and actions

        layers = []
        
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]

        layers.append(nn.Linear(hidden_dims[-1], action_dim * chunk_size)) #vector field as output

        self.net = nn.Sequential(*layers) #unpack



    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor, #z
    ) -> torch.Tensor:

        B = state.shape[0]

        action_chunk = action_chunk.reshape((B, self.chunk_size * self.action_dim)) #(B, tot_act_dim), x1

        actions_noise = torch.randn_like(action_chunk) #x0

        tau = torch.rand((B,1), device=action_chunk.device, dtype=action_chunk.dtype) #tau

        actions_tau = tau * action_chunk + (1 - tau) * actions_noise

        sample = torch.cat([state, actions_tau, tau], dim = -1) #(B, tot_action_dim + state_dim + 1)

        target_v = action_chunk - actions_noise #(B, tot_action_dim)
        pred_v = self.net(sample) #(B, tot_action_dim)

        loss = torch.mean((pred_v - target_v)**2)

        return loss



    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:

        B = state.shape[0]
        tot_action_dim = self.chunk_size * self.action_dim

        actions_noise = torch.normal(torch.zeros((B, tot_action_dim)), torch.ones(B, tot_action_dim)).to(state.device)

        action_tau = actions_noise



        for i in range(num_steps):
            tau = torch.full((B, 1), i / num_steps, device=state.device, dtype = state.dtype) #(B, 1) with the same i / num_steps value

            sample = torch.cat([state, action_tau, tau], dim = -1)

            action_tau = action_tau + (1 / num_steps) * self.net(sample)

        return action_tau.reshape((B, self.chunk_size, self.action_dim))




PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
