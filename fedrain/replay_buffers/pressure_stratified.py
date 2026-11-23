"""Pressure-stratified replay for shared levelwise agents."""

from types import SimpleNamespace

import numpy as np
import torch

from fedrain.replay_buffers.level import LevelReplayBuffer


class PressureStratifiedLevelReplayBuffer:
    """Store and sample equal numbers of transitions from pressure bands."""

    def __init__(
        self,
        buffer_size,
        observation_channels,
        device,
        pressure_feature_index,
        pressure_bin_edges,
        max_transitions_per_band,
    ):
        """Initialise one uniform sub-buffer for each pressure band.

        Parameters
        ----------
        buffer_size : int
            Total transition capacity across pressure bands.
        observation_channels : int
            Features stored for each level.
        device : str or torch.device
            Device used for sampled tensors.
        pressure_feature_index : int
            Observation column containing normalised pressure.
        pressure_bin_edges : array_like
            Ordered boundaries defining pressure bands.
        max_transitions_per_band : int
            Maximum number of randomly selected transitions inserted into
            each pressure band per environment exchange.
        """
        self.buffer_size = int(buffer_size)
        self.device = device
        self.pressure_feature_index = int(pressure_feature_index)
        self.pressure_bin_edges = np.asarray(pressure_bin_edges, dtype=np.float32)
        self.max_transitions_per_band = int(max_transitions_per_band)
        if self.max_transitions_per_band <= 0:
            raise ValueError("max_transitions_per_band must be positive")
        band_count = self.pressure_bin_edges.size + 1
        band_capacity = self.buffer_size // band_count
        self.buffers = [
            LevelReplayBuffer(
                band_capacity,
                observation_channels,
                device,
            )
            for _ in range(band_count)
        ]

    def add(self, observations, next_observations, actions, rewards, dones):
        """Add transitions to sub-buffers selected by current pressure.

        Parameters
        ----------
        observations : numpy.ndarray
            Current level features including pressure.
        next_observations : numpy.ndarray
            Next level features.
        actions : array_like
            Scalar actions.
        rewards : array_like
            Scalar rewards.
        dones : array_like
            Episode termination indicators.
        """
        pressure_values = observations[:, self.pressure_feature_index]
        band_indices = np.digitize(pressure_values, self.pressure_bin_edges)
        for band_index, replay_buffer in enumerate(self.buffers):
            selected_indices = np.flatnonzero(band_indices == band_index)
            if selected_indices.size:
                if selected_indices.size > self.max_transitions_per_band:
                    selected_indices = np.random.choice(
                        selected_indices,
                        size=self.max_transitions_per_band,
                        replace=False,
                    )
                replay_buffer.add(
                    observations[selected_indices],
                    next_observations[selected_indices],
                    actions[selected_indices],
                    rewards[selected_indices],
                    dones[selected_indices],
                )

    def sample(self, batch_size):
        """Sample approximately equal counts from populated pressure bands.

        Parameters
        ----------
        batch_size : int
            Total transitions to sample.

        Returns
        -------
        types.SimpleNamespace
            Randomly permuted observation, action, reward, and done tensors.
        """
        populated = [
            replay_buffer
            for replay_buffer in self.buffers
            if replay_buffer.full or replay_buffer.pos > 0
        ]
        base_count, remainder = divmod(batch_size, len(populated))
        samples = []
        for index, replay_buffer in enumerate(populated):
            replay_buffer.device = self.device
            sample_count = base_count + (index < remainder)
            if sample_count:
                samples.append(replay_buffer.sample(sample_count))
        combined = SimpleNamespace(
            observations=torch.cat([sample.observations for sample in samples]),
            next_observations=torch.cat(
                [sample.next_observations for sample in samples]
            ),
            actions=torch.cat([sample.actions for sample in samples]),
            rewards=torch.cat([sample.rewards for sample in samples]),
            dones=torch.cat([sample.dones for sample in samples]),
        )
        permutation = torch.randperm(batch_size, device=self.device)
        return SimpleNamespace(
            **{
                field: getattr(combined, field)[permutation]
                for field in combined.__dict__
            }
        )
