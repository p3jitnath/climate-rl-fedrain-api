"""Replay buffer for levelwise reinforcement-learning transitions."""

from types import SimpleNamespace

import numpy as np
import torch


class LevelReplayBuffer:
    """Fixed-size replay buffer supporting bulk levelwise transitions."""

    def __init__(
        self,
        buffer_size,
        observation_channels,
        device,
    ):
        """Allocate compact arrays for levelwise transitions.

        Parameters
        ----------
        buffer_size : int
            Maximum number of transitions.
        observation_channels : int
            Features stored for each level.
        device : str or torch.device
            Device used for sampled tensors.
        """
        self.buffer_size = int(buffer_size)
        self.device = device
        self.observations = np.empty(
            (self.buffer_size, observation_channels), dtype=np.float32
        )
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.rewards = np.empty(self.buffer_size, dtype=np.float32)
        self.dones = np.empty(self.buffer_size, dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, observations, next_observations, actions, rewards, dones):
        """Add a batch of transitions, retaining the newest on overflow.

        Parameters
        ----------
        observations : numpy.ndarray
            Current level features.
        next_observations : numpy.ndarray
            Next level features.
        actions : array_like
            Scalar actions.
        rewards : array_like
            Scalar rewards.
        dones : array_like
            Episode termination indicators.
        """
        transition_count = observations.shape[0]
        if transition_count >= self.buffer_size:
            observations = observations[-self.buffer_size :]
            next_observations = next_observations[-self.buffer_size :]
            actions = actions[-self.buffer_size :]
            rewards = rewards[-self.buffer_size :]
            dones = dones[-self.buffer_size :]
            transition_count = self.buffer_size
        indices = (np.arange(transition_count) + self.pos) % self.buffer_size
        self.observations[indices] = observations
        self.next_observations[indices] = next_observations
        self.actions[indices] = np.asarray(actions).reshape(transition_count, 1)
        self.rewards[indices] = rewards
        self.dones[indices] = dones
        self.full = self.full or self.pos + transition_count >= self.buffer_size
        self.pos = (self.pos + transition_count) % self.buffer_size

    def sample(self, batch_size):
        """Sample a levelwise mini-batch as tensors on the agent device.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample with replacement.

        Returns
        -------
        types.SimpleNamespace
            Observation, action, reward, and termination tensors.
        """
        upper_bound = self.buffer_size if self.full else self.pos
        indices = np.random.randint(0, upper_bound, size=batch_size)
        return SimpleNamespace(
            observations=torch.as_tensor(
                self.observations[indices], device=self.device
            ),
            next_observations=torch.as_tensor(
                self.next_observations[indices], device=self.device
            ),
            actions=torch.as_tensor(self.actions[indices], device=self.device),
            rewards=torch.as_tensor(self.rewards[indices], device=self.device),
            dones=torch.as_tensor(self.dones[indices], device=self.device),
        )
