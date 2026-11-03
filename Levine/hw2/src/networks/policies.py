import itertools
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        obs = np.asarray(obs)

        if obs.ndim == 1:
            obs = obs[None, :]  # (ob_dim,) -> (1, ob_dim)

        distribution = self(ptu.from_numpy(obs))
        action = distribution.sample()

        return ptu.to_numpy(action)

    def forward(self, obs: torch.FloatTensor) -> torch.FloatTensor:
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """

        # Categorical gestisce internamente la normalizzazione dei logits. Per le azioni continue, Independent(..., 1) considera l’intero vettore come un’unica azione, sommando le log-probabilità delle coordinate. Ottieni quindi log_prob(actions) di shape (B,) in entrambi i casi.

        if self.discrete:
            logits = self.logits_net(obs)  # (B, ac_dim)
            return D.Categorical(logits=logits)

        
        mean = self.mean_net(obs)
        std = self.logstd.exp()

        return D.Independent(D.Normal(mean, std), 1)

        

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        Performs one iteration of gradient descent on the provided batch of data. You don't need to implement this
        method in the base class, but you do need to implement it in the subclass.
        """
    


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs_t = ptu.from_numpy(obs)
        advantages_t = ptu.from_numpy(advantages).reshape(-1)

        if self.discrete:
            actions_t = ptu.from_numpy(actions).long().reshape(-1)
        else:
            actions_t = ptu.from_numpy(actions).reshape(obs_t.shape[0], -1)

        distribution = self(obs_t)
        log_probs = distribution.log_prob(actions_t)  # (B,)

        self.optimizer.zero_grad()
        # compute the policy gradient actor loss
        loss = -(log_probs * advantages_t).mean()

        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": loss.item(),
        }
