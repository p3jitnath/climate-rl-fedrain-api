"""Independent exploration used by the original DDPG implementation."""

import numpy as np
import torch


class IndependentExploration:
    """Sample independent actions and independent Gaussian actor noise."""

    def sample_initial_actions(self, action_space, batch_size):
        """Sample one action-space vector for every environment.

        Parameters
        ----------
        action_space : gymnasium.spaces.Space
            Environment action space.
        batch_size : int
            Number of action vectors.

        Returns
        -------
        numpy.ndarray
            Independently sampled action vectors.
        """
        return np.asarray([action_space.sample() for _ in range(batch_size)])

    def add_actor_noise(self, actions, action_scale, noise_scale):
        """Add independent Gaussian exploration noise to actor actions.

        Parameters
        ----------
        actions : torch.Tensor
            Deterministic actor actions.
        action_scale : torch.Tensor
            Per-dimension scale of the action range.
        noise_scale : float
            Exploration standard-deviation multiplier.

        Returns
        -------
        torch.Tensor
            Noisy actions before clipping.
        """
        return actions + torch.randn_like(actions) * action_scale * noise_scale
