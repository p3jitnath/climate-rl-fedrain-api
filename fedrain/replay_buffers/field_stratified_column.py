"""Replay buffer balanced across complete theta, U and V columns."""

from types import SimpleNamespace

import numpy as np
import torch


class _ColumnReplayBuffer:
    """Store a fixed number of complete column transitions."""

    def __init__(self, capacity, observation_size, action_size):
        self.capacity = int(capacity)
        self.observations = np.empty(
            (self.capacity, observation_size), dtype=np.float32
        )
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty((self.capacity, action_size), dtype=np.float32)
        self.rewards = np.empty_like(self.actions)
        self.teacher_actions = np.empty_like(self.actions)
        self.dones = np.empty(self.capacity, dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(
        self, observations, next_observations, actions, rewards, dones, teacher_actions
    ):
        """Add complete-column transitions to the circular buffer."""
        transition_count = observations.shape[0]
        if transition_count >= self.capacity:
            selection = slice(-self.capacity, None)
            observations = observations[selection]
            next_observations = next_observations[selection]
            actions = actions[selection]
            rewards = rewards[selection]
            dones = dones[selection]
            teacher_actions = teacher_actions[selection]
            transition_count = self.capacity
        indices = (np.arange(transition_count) + self.pos) % self.capacity
        self.observations[indices] = observations
        self.next_observations[indices] = next_observations
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.dones[indices] = dones
        self.teacher_actions[indices] = teacher_actions
        self.full = self.full or self.pos + transition_count >= self.capacity
        self.pos = (self.pos + transition_count) % self.capacity

    def sample(self, sample_count):
        """Return randomly selected NumPy column transitions."""
        upper_bound = self.capacity if self.full else self.pos
        indices = np.random.randint(0, upper_bound, size=sample_count)
        return (
            self.observations[indices],
            self.next_observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.dones[indices],
            self.teacher_actions[indices],
        )

    def __len__(self):
        """Return the number of stored transitions."""
        return self.capacity if self.full else self.pos


class FieldStratifiedColumnReplayBuffer:
    """Store and sample equal numbers of complete columns from each field."""

    def __init__(
        self,
        buffer_size,
        observation_size,
        action_size,
        active_action_count,
        device,
        field_feature_index,
        field_values,
        max_columns_per_field,
    ):
        """Initialise one equally sized column buffer per physical field.

        Parameters
        ----------
        buffer_size : int
            Nominal capacity measured in level transitions.
        observation_size : int
            Flattened size of a complete column observation.
        action_size : int
            Number of vertical actions and rewards in each column.
        active_action_count : int
            Active levels represented by one complete-column transition.
        device : str or torch.device
            Device used for sampled tensors.
        field_feature_index : int
            Observation-channel index containing the field identity.
        field_values : array_like
            Identity values for the physical fields.
        max_columns_per_field : int
            Maximum columns retained from each field and exchange.
        """
        self.buffer_size = int(buffer_size)
        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        self.active_action_count = int(active_action_count)
        self.device = device
        self.field_feature_index = int(field_feature_index)
        self.field_values = np.asarray(field_values, dtype=np.float32)
        self.max_columns_per_field = int(max_columns_per_field)
        column_capacity = self.buffer_size // self.active_action_count
        field_capacity = column_capacity // self.field_values.size
        if field_capacity < 1:
            raise ValueError("Replay capacity is too small for all fields")
        if self.max_columns_per_field < 1:
            raise ValueError("max_columns_per_field must be positive")
        self.buffers = [
            _ColumnReplayBuffer(
                field_capacity,
                self.observation_size,
                self.action_size,
            )
            for _ in self.field_values
        ]

    def add(
        self,
        observations,
        next_observations,
        actions,
        rewards,
        dones,
        teacher_actions=None,
    ):
        """Add aligned column transitions and optional supervision targets.

        Parameters
        ----------
        observations, next_observations : numpy.ndarray
            Flattened pre-action and subsequent column states.
        actions, rewards : numpy.ndarray
            Applied actions and rewards, shaped as columns by levels.
        dones : numpy.ndarray
            Physical termination flags for each column.
        teacher_actions : numpy.ndarray, optional
            Training targets in action units, with the same shape as actions.
            Omitted targets are stored as zeros for unsupervised callers.
        """
        if teacher_actions is None:
            teacher_actions = np.zeros_like(actions)
        column_features = observations.reshape(
            observations.shape[0],
            self.observation_size // self.action_size,
            self.action_size,
        )
        observed_fields = column_features[:, self.field_feature_index, 0]
        for buffer, field_value in zip(self.buffers, self.field_values):
            selected_indices = np.flatnonzero(np.isclose(observed_fields, field_value))
            if selected_indices.size > self.max_columns_per_field:
                selected_indices = np.random.choice(
                    selected_indices,
                    size=self.max_columns_per_field,
                    replace=False,
                )
            if selected_indices.size:
                buffer.add(
                    observations[selected_indices],
                    next_observations[selected_indices],
                    actions[selected_indices],
                    rewards[selected_indices],
                    dones[selected_indices],
                    teacher_actions[selected_indices],
                )

    def sample(self, batch_size):
        """Sample an approximately equal mini-batch from every field."""
        populated_buffers = [buffer for buffer in self.buffers if len(buffer)]
        if not populated_buffers:
            raise RuntimeError("Cannot sample an empty replay buffer")
        base_count, remainder = divmod(batch_size, len(populated_buffers))
        samples = []
        for index, buffer in enumerate(populated_buffers):
            samples.append(buffer.sample(base_count + (index < remainder)))
        arrays = [np.concatenate(values, axis=0) for values in zip(*samples)]
        order = np.random.permutation(batch_size)
        return SimpleNamespace(
            observations=torch.as_tensor(arrays[0][order], device=self.device),
            next_observations=torch.as_tensor(arrays[1][order], device=self.device),
            actions=torch.as_tensor(arrays[2][order], device=self.device),
            rewards=torch.as_tensor(arrays[3][order], device=self.device),
            dones=torch.as_tensor(arrays[4][order], device=self.device),
            teacher_actions=torch.as_tensor(arrays[5][order], device=self.device),
        )
