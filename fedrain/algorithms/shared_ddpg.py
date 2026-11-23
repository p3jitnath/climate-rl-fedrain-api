"""DDPG implementation with actor weights shared across action dimensions."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fedrain.algorithms.ddpg import DDPG
from fedrain.replay_buffers import (
    FieldPressureStratifiedLevelReplayBuffer,
    LevelReplayBuffer,
    PressureStratifiedLevelReplayBuffer,
)


class SharedDDPGActor(nn.Module):
    """Apply one actor network independently across action dimensions.

    Observation channels are stored contiguously, with one value per action
    dimension in each channel. The shared network maps the channel values for
    one dimension to one continuous action.

    Parameters
    ----------
    envs : gym.Env or vectorized env
        Environment providing the observation and action spaces.
    layer_size : int
        Width of the shared hidden layers.

    """

    column_batch_size = 256

    def __init__(self, envs, layer_size):
        """Initialise the shared actor and action scaling buffers.

        Parameters
        ----------
        envs : gym.Env or vectorized env
            Environment providing observation and action spaces.
        layer_size : int
            Width of both hidden layers.

        Raises
        ------
        ValueError
            If observations cannot be divided evenly among actions.
        """
        super().__init__()
        observation_size = int(np.array(envs.single_observation_space.shape).prod())
        self.action_size = int(np.prod(envs.single_action_space.shape))
        if observation_size % self.action_size != 0:
            raise ValueError(
                f"Observation size {observation_size} is not divisible by "
                f"action size {self.action_size}"
            )
        self.observation_channels = observation_size // self.action_size
        self.fc1 = nn.Linear(self.observation_channels, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc_mu = nn.Linear(layer_size, 1)
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, observations):
        """Compute all actions with weights shared across dimensions.

        Parameters
        ----------
        observations : torch.Tensor
            Batched flattened column observations.

        Returns
        -------
        torch.Tensor
            Scaled actions for every column and action dimension.
        """
        action_batches = []
        for observation_batch in observations.split(
            self.column_batch_size,
            dim=0,
        ):
            level_features = observation_batch.reshape(
                observation_batch.shape[0],
                self.observation_channels,
                self.action_size,
            ).transpose(1, 2)
            level_features = F.relu(self.fc1(level_features))
            level_features = F.relu(self.fc2(level_features))
            action_batches.append(torch.tanh(self.fc_mu(level_features)).squeeze(-1))
        actions = torch.cat(action_batches, dim=0)
        return actions * self.action_scale + self.action_bias

    def forward_levels(self, level_features):
        """Compute scalar actions for a batch of individual levels.

        Parameters
        ----------
        level_features : torch.Tensor
            Batched features for individual levels.

        Returns
        -------
        torch.Tensor
            One scaled action per level.
        """
        level_features = F.relu(self.fc1(level_features))
        level_features = F.relu(self.fc2(level_features))
        actions = torch.tanh(self.fc_mu(level_features))
        return actions * self.action_scale[0] + self.action_bias[0]


class SharedDDPGCritic(nn.Module):
    """Estimate Q values for individual level state-action pairs."""

    def __init__(self, observation_channels, layer_size):
        """Initialise the levelwise critic network.

        Parameters
        ----------
        observation_channels : int
            Features describing each level.
        layer_size : int
            Width of both hidden layers.
        """
        super().__init__()
        self.fc1 = nn.Linear(observation_channels + 1, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, 1)

    def forward(self, observations, actions):
        """Return Q values for scalar level actions.

        Parameters
        ----------
        observations : torch.Tensor
            Batched level features.
        actions : torch.Tensor
            Batched scalar actions.

        Returns
        -------
        torch.Tensor
            Estimated Q value for every feature-action pair.
        """
        features = torch.cat((observations, actions), dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return self.fc3(features)


class SharedDDPG(DDPG):
    """DDPG using actor weights shared across action dimensions."""

    algorithm_name = "SharedDDPG"
    actor_class = SharedDDPGActor

    def __init__(
        self,
        *args,
        active_action_indices=None,
        level_replay_strategy="uniform",
        pressure_feature_index=None,
        pressure_bin_edges=None,
        max_transitions_per_pressure_band=None,
        field_feature_index=None,
        field_values=None,
        **kwargs,
    ):
        """Initialise shared DDPG with optional levelwise training.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to :class:`DDPG`.
        active_action_indices : array_like, optional
            Action dimensions included in levelwise training.
        level_replay_strategy : str, optional
            Level-transition sampling strategy: uniform, pressure-stratified,
            or field-pressure-stratified.
        pressure_feature_index : int, optional
            Feature containing pressure for stratified replay.
        pressure_bin_edges : array_like, optional
            Pressure-band boundaries for stratified replay.
        max_transitions_per_pressure_band : int, optional
            Maximum transitions retained from each pressure band and
            environment exchange.
        field_feature_index : int, optional
            Feature identifying the physical field in each transition.
        field_values : array_like, optional
            Field-identity values defining the replay strata.
        **kwargs : dict
            Keyword arguments forwarded to :class:`DDPG`.

        Raises
        ------
        ValueError
            If levelwise action bounds or replay configuration are invalid.
        """
        super().__init__(*args, **kwargs)
        self.level_replay_strategy = level_replay_strategy
        self.algorithm["level_replay_strategy"] = level_replay_strategy
        self.active_action_indices = active_action_indices
        self.levelwise_training = active_action_indices is not None
        if not self.levelwise_training:
            return

        self.active_action_indices = np.asarray(active_action_indices, dtype=np.int64)
        if not np.allclose(
            self.actor.action_scale,
            self.actor.action_scale[0],
        ) or not np.allclose(
            self.actor.action_bias,
            self.actor.action_bias[0],
        ):
            raise ValueError("Levelwise training requires uniform action bounds")
        observation_channels = self.actor.observation_channels
        self.qf1 = SharedDDPGCritic(
            observation_channels,
            self.critic_layer_size,
        ).to(self.device)
        self.qf1_target = SharedDDPGCritic(
            observation_channels,
            self.critic_layer_size,
        ).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.q_optimizer = optim.Adam(self.qf1.parameters(), lr=self.learning_rate)
        if level_replay_strategy == "uniform":
            self.rb = LevelReplayBuffer(
                self.buffer_size,
                observation_channels,
                self.device,
            )
        elif level_replay_strategy == "pressure_stratified":
            if (
                pressure_feature_index is None
                or pressure_bin_edges is None
                or max_transitions_per_pressure_band is None
            ):
                raise ValueError(
                    "Pressure-stratified replay requires a pressure feature "
                    "index, bin edges, and a per-band insertion limit"
                )
            self.rb = PressureStratifiedLevelReplayBuffer(
                self.buffer_size,
                observation_channels,
                self.device,
                pressure_feature_index,
                pressure_bin_edges,
                max_transitions_per_pressure_band,
            )
        elif level_replay_strategy == "field_pressure_stratified":
            if (
                pressure_feature_index is None
                or pressure_bin_edges is None
                or max_transitions_per_pressure_band is None
                or field_feature_index is None
                or field_values is None
            ):
                raise ValueError(
                    "Field-pressure-stratified replay requires pressure and "
                    "field features, pressure bins, field values, and a "
                    "per-stratum insertion limit"
                )
            self.rb = FieldPressureStratifiedLevelReplayBuffer(
                self.buffer_size,
                observation_channels,
                self.device,
                pressure_feature_index,
                pressure_bin_edges,
                field_feature_index,
                field_values,
                max_transitions_per_pressure_band,
            )
        else:
            raise ValueError(f"Unknown level replay strategy: {level_replay_strategy}")

    def _level_features(self, observations):
        """Extract active-level features from column observations.

        Parameters
        ----------
        observations : numpy.ndarray
            Batched flattened column observations.

        Returns
        -------
        numpy.ndarray
            Feature rows for active levels only.
        """
        column_count = observations.shape[0]
        level_features = observations.reshape(
            column_count,
            self.actor.observation_channels,
            self.actor.action_size,
        ).transpose(0, 2, 1)
        return level_features[:, self.active_action_indices, :].reshape(
            -1, self.actor.observation_channels
        )

    def _centre_actions(self, actions):
        """Extract the action associated with each active-level reward.

        Parameters
        ----------
        actions : numpy.ndarray
            Batched column actions.

        Returns
        -------
        numpy.ndarray
            Scalar action rows for active levels.
        """
        return actions[:, self.active_action_indices].reshape(-1, 1)

    def update(
        self,
        actions,
        next_obs,
        rewards,
        terminations,
        truncations,
        infos,
    ):
        """Update from independent active-level rewards when available.

        Parameters
        ----------
        actions : numpy.ndarray
            Actions applied to the environments.
        next_obs : numpy.ndarray
            Observations after the environment step.
        rewards : numpy.ndarray
            Aggregate environment rewards.
        terminations : numpy.ndarray
            Environment termination flags.
        truncations : numpy.ndarray
            Environment truncation flags.
        infos : dict
            Vector-environment metadata containing level rewards.

        Raises
        ------
        RuntimeError
            If levelwise training is active but level rewards are absent.
        """
        if not self.levelwise_training:
            return super().update(
                actions,
                next_obs,
                rewards,
                terminations,
                truncations,
                infos,
            )

        if "level_rewards" in infos:
            level_rewards = np.stack(infos["level_rewards"]).astype(np.float32)
        elif "final_info" in infos:
            level_rewards = np.stack(
                [info["level_rewards"] for info in infos["final_info"]]
            ).astype(np.float32)
        else:
            raise RuntimeError("Missing level_rewards in vector environment info")

        if self.check_episode_termination(infos):
            returns = np.array(
                [float(info["episode"]["r"][0]) for info in infos["final_info"]]
            )
            self.recorder.record_episodic_return(np.mean(returns), self.global_step)
            if self.record_step_data:
                self.recorder.sr.save(self.global_step, self.actor, returns)

        active_rewards = level_rewards[:, self.active_action_indices].reshape(-1)
        centre_actions = self._centre_actions(actions)
        active_dones = np.repeat(
            np.logical_or(terminations, truncations),
            self.active_action_indices.size,
        ).astype(np.float32)
        self.rb.add(
            self._level_features(self.obs),
            self._level_features(next_obs),
            centre_actions,
            active_rewards,
            active_dones,
        )
        if self.record_step_data:
            self.recorder.sr.add(
                self.global_step,
                self.obs,
                next_obs,
                actions,
                rewards,
            )
        self.obs = next_obs

        if self.train and self.global_step > self.learning_starts:
            for _ in range(self.gradient_steps):
                replay = self.rb.sample(self.batch_size)
                with torch.no_grad():
                    next_actions = self.target_actor.forward_levels(
                        replay.next_observations
                    )
                    next_q_values = self.qf1_target(
                        replay.next_observations,
                        next_actions,
                    ).view(-1)
                    target_q_values = (
                        replay.rewards
                        + (1.0 - replay.dones) * self.gamma * next_q_values
                    )

                q_values = self.qf1(
                    replay.observations,
                    replay.actions,
                ).view(-1)
                critic_loss = F.mse_loss(q_values, target_q_values)
                self.q_optimizer.zero_grad()
                critic_loss.backward()
                self.q_optimizer.step()
                self.gradient_step += 1

                if self.gradient_step % self.policy_frequency == 0:
                    actor_actions = self.actor.forward_levels(replay.observations)
                    actor_loss = -self.qf1(
                        replay.observations,
                        actor_actions,
                    ).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    for parameter, target_parameter in zip(
                        self.actor.parameters(), self.target_actor.parameters()
                    ):
                        target_parameter.data.copy_(
                            self.tau * parameter.data
                            + (1.0 - self.tau) * target_parameter.data
                        )
                    for parameter, target_parameter in zip(
                        self.qf1.parameters(), self.qf1_target.parameters()
                    ):
                        target_parameter.data.copy_(
                            self.tau * parameter.data
                            + (1.0 - self.tau) * target_parameter.data
                        )

        if self.train and self.fedRLConfig is not None and "final_info" in infos:
            if (
                self.global_step
                % (self.fedRLConfig["flwr_episodes"] * self.fedRLConfig["num_steps"])
                == 0
            ):
                self.fedRL.save_weights(self.global_step)
                if self.fedRLConfig.get("load_after_fit", True):
                    self.fedRL.load_weights(self.global_step)

        self.global_step += 1
