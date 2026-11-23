"""Vertically correlated exploration for column action vectors."""

import numpy as np
import torch
import torch.nn.functional as functional


class VerticallyCorrelatedExploration:
    """Smooth random exploration over neighbouring action dimensions."""

    def __init__(self, smoothing_passes=2):
        """Initialise the number of three-point smoothing passes.

        Parameters
        ----------
        smoothing_passes : int, optional
            Number of neighbouring-level smoothing passes.

        Raises
        ------
        ValueError
            If fewer than one pass is requested.
        """
        if smoothing_passes < 1:
            raise ValueError("smoothing_passes must be at least one")
        self.smoothing_passes = int(smoothing_passes)

    def _smooth_numpy(self, values):
        """Smooth NumPy action vectors along their final dimension.

        Parameters
        ----------
        values : numpy.ndarray
            Batched action vectors.

        Returns
        -------
        numpy.ndarray
            Smoothed vectors with unchanged shape.
        """
        smoothed = values
        for _ in range(self.smoothing_passes):
            padded = np.pad(smoothed, ((0, 0), (1, 1)), mode="edge")
            smoothed = (
                0.25 * padded[:, :-2] + 0.5 * padded[:, 1:-1] + 0.25 * padded[:, 2:]
            )
        return smoothed

    def _smooth_torch(self, values):
        """Smooth tensor action vectors along their final dimension.

        Parameters
        ----------
        values : torch.Tensor
            Batched action vectors.

        Returns
        -------
        torch.Tensor
            Smoothed vectors with unchanged shape.
        """
        kernel = values.new_tensor((0.25, 0.5, 0.25)).reshape(1, 1, 3)
        smoothed = values.unsqueeze(1)
        for _ in range(self.smoothing_passes):
            padded = functional.pad(smoothed, (1, 1), mode="replicate")
            smoothed = functional.conv1d(padded, kernel)
        return smoothed.squeeze(1)

    def sample_initial_actions(self, action_space, batch_size):
        """Sample and vertically smooth initial action-space vectors.

        Parameters
        ----------
        action_space : gymnasium.spaces.Space
            Environment action space.
        batch_size : int
            Number of action vectors.

        Returns
        -------
        numpy.ndarray
            Smoothed samples in the action-space dtype.
        """
        actions = np.asarray([action_space.sample() for _ in range(batch_size)])
        return self._smooth_numpy(actions).astype(action_space.dtype, copy=False)

    def add_actor_noise(self, actions, action_scale, noise_scale):
        """Add vertically correlated Gaussian noise to actor actions.

        Parameters
        ----------
        actions : torch.Tensor
            Deterministic actor actions.
        action_scale : torch.Tensor
            Per-dimension action scale.
        noise_scale : float
            Exploration standard-deviation multiplier.

        Returns
        -------
        torch.Tensor
            Noisy actions before clipping.
        """
        noise = self._smooth_torch(torch.randn_like(actions))
        return actions + noise * action_scale * noise_scale
