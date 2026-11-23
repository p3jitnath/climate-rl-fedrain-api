"""Replay buffer balanced jointly across fields and pressure bands."""

import numpy as np

from fedrain.replay_buffers.level import LevelReplayBuffer
from fedrain.replay_buffers.pressure_stratified import (
    PressureStratifiedLevelReplayBuffer,
)


class FieldPressureStratifiedLevelReplayBuffer(PressureStratifiedLevelReplayBuffer):
    """Store and sample equal numbers from field-pressure strata."""

    def __init__(
        self,
        buffer_size,
        observation_channels,
        device,
        pressure_feature_index,
        pressure_bin_edges,
        field_feature_index,
        field_values,
        max_transitions_per_stratum,
    ):
        """Initialise one replay sub-buffer for each field-pressure pair."""
        self.buffer_size = int(buffer_size)
        self.device = device
        self.pressure_feature_index = int(pressure_feature_index)
        self.pressure_bin_edges = np.asarray(pressure_bin_edges, dtype=np.float32)
        self.field_feature_index = int(field_feature_index)
        self.field_values = np.asarray(field_values, dtype=np.float32)
        self.max_transitions_per_band = int(max_transitions_per_stratum)
        if self.max_transitions_per_band <= 0:
            raise ValueError("max_transitions_per_stratum must be positive")
        self.pressure_band_count = self.pressure_bin_edges.size + 1
        stratum_count = self.field_values.size * self.pressure_band_count
        stratum_capacity = self.buffer_size // stratum_count
        self.buffers = [
            LevelReplayBuffer(
                stratum_capacity,
                observation_channels,
                device,
            )
            for _ in range(stratum_count)
        ]

    def add(self, observations, next_observations, actions, rewards, dones):
        """Add a capped random sample to every field-pressure stratum."""
        pressure_bands = np.digitize(
            observations[:, self.pressure_feature_index],
            self.pressure_bin_edges,
        )
        observed_fields = observations[:, self.field_feature_index]
        for field_index, field_value in enumerate(self.field_values):
            field_mask = np.isclose(observed_fields, field_value)
            for pressure_band in range(self.pressure_band_count):
                selected_indices = np.flatnonzero(
                    field_mask & (pressure_bands == pressure_band)
                )
                if selected_indices.size > self.max_transitions_per_band:
                    selected_indices = np.random.choice(
                        selected_indices,
                        size=self.max_transitions_per_band,
                        replace=False,
                    )
                if selected_indices.size:
                    buffer_index = (
                        field_index * self.pressure_band_count + pressure_band
                    )
                    self.buffers[buffer_index].add(
                        observations[selected_indices],
                        next_observations[selected_indices],
                        actions[selected_indices],
                        rewards[selected_indices],
                        dones[selected_indices],
                    )
