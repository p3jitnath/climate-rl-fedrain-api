"""Column-aware DDPG with compact shared vertical convolutions."""

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fedrain.algorithms.ddpg import DDPG
from fedrain.replay_buffers import FieldStratifiedColumnReplayBuffer


def _column_context(features, observations, pressure_feature_index):
    """Pool encoded levels uniformly or by trapezoidal pressure thickness.

    Pressure must use a linear normalisation; its constant scale cancels
    in the weighted mean. Model columns must span a nonzero pressure range.
    """
    if pressure_feature_index is None:
        return features.mean(dim=2, keepdim=True).expand_as(features)
    pressure = observations[:, int(pressure_feature_index.item()), :]
    pressure_thickness = (pressure[:, 1:] - pressure[:, :-1]).abs()
    interval_features = 0.5 * (features[:, :, 1:] + features[:, :, :-1])
    column_total = (interval_features * pressure_thickness[:, None, :]).sum(
        dim=2, keepdim=True
    )
    context = column_total / pressure_thickness.sum(dim=1)[:, None, None]
    return context.expand_as(features)


class ColumnDDPGActor(nn.Module):
    """Predict level actions using local and full-column context."""

    column_batch_size = 256

    def __init__(
        self,
        envs,
        layer_size,
        field_feature_index=None,
        field_values=(),
        context_pressure_feature_index=None,
        constant_observation=None,
    ):
        """Initialise the shared vertical encoder and action head.

        Parameters
        ----------
        envs : gym.Env or vectorized env
            Environment providing flattened column observations and actions.
        layer_size : int
            Width of the encoder and pointwise action-head layers.
        field_feature_index : int or None, optional
            Field-identity channel used to select a separate final output.
            None retains the shared output and legacy checkpoint shape.
        field_values : sequence, optional
            Field identities in output-channel order.
        context_pressure_feature_index : int or None, optional
            Linearly normalised pressure channel for mass-weighted context.
            None retains the uniform model-level mean.
        constant_observation : array_like or None, optional
            Fixed channel-by-level template replacing all atmospheric inputs.
            Field identity still selects the corresponding output head.

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
        self.encoder_channels = layer_size
        self.register_buffer(
            "constant_observation",
            (
                torch.as_tensor(constant_observation, dtype=torch.float32).unsqueeze(0)
                if constant_observation is not None
                else None
            ),
        )
        self.register_buffer(
            "context_pressure_index",
            (
                torch.tensor(context_pressure_feature_index, dtype=torch.int64)
                if context_pressure_feature_index is not None
                else None
            ),
        )
        self.encoder1 = nn.Conv1d(
            self.observation_channels,
            self.encoder_channels,
            kernel_size=3,
            padding=1,
        )
        self.encoder2 = nn.Conv1d(
            self.encoder_channels,
            self.encoder_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
        )
        self.fc1 = nn.Conv1d(
            2 * self.encoder_channels,
            layer_size,
            kernel_size=1,
        )
        self.fc2 = nn.Conv1d(layer_size, layer_size, kernel_size=1)
        self.field_feature_index = field_feature_index
        self.register_buffer(
            "field_values",
            (
                torch.as_tensor(field_values, dtype=torch.float32)
                if field_feature_index is not None
                else None
            ),
        )
        output_channels = len(field_values) if field_feature_index is not None else 1
        self.fc_mu = nn.Conv1d(layer_size, output_channels, kernel_size=1)
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

    def _column_features(self, observations):
        """Combine local convolutional features with global column context."""
        columns = observations.reshape(
            observations.shape[0],
            self.observation_channels,
            self.action_size,
        )
        if self.constant_observation is not None:
            columns = self.constant_observation.expand(columns.shape[0], -1, -1)
        features = F.relu(self.encoder1(columns))
        features = F.relu(self.encoder2(features))
        context = _column_context(features, columns, self.context_pressure_index)
        return torch.cat((features, context), dim=1)

    def forward(self, observations):
        """Predict all actions for a batch of complete columns.

        Parameters
        ----------
        observations : torch.Tensor
            Flattened observations with one row per complete column.

        Returns
        -------
        torch.Tensor
            Scaled action at every vertical level.
        """
        action_batches = []
        for observation_batch in observations.split(
            self.column_batch_size,
            dim=0,
        ):
            features = self._column_features(observation_batch)
            features = F.relu(self.fc1(features))
            features = F.relu(self.fc2(features))
            logits = self.fc_mu(features)
            if self.field_feature_index is not None:
                fields = observation_batch[
                    :, self.field_feature_index * self.action_size
                ]
                head_indices = (
                    (fields[:, None] - self.field_values[None, :]).abs().argmin(dim=1)
                )
                logits = logits.gather(
                    1, head_indices[:, None, None].expand(-1, 1, self.action_size)
                )
            action_batches.append(torch.tanh(logits).squeeze(1))
        actions = torch.cat(action_batches, dim=0)
        return actions * self.action_scale + self.action_bias


class ColumnDDPGCritic(nn.Module):
    """Estimate levelwise values using each action and its full column."""

    def __init__(
        self,
        observation_channels,
        layer_size,
        field_feature_index=None,
        field_values=(),
        context_pressure_feature_index=None,
        constant_observation=None,
    ):
        """Initialise the convolutional value network.

        Parameters
        ----------
        observation_channels : int
            Number of physical and coordinate channels in each column.
        layer_size : int
            Width of the encoder and pointwise value-head layers.
        field_feature_index : int or None, optional
            Field-identity channel used to select a separate final value.
        field_values : sequence, optional
            Field identities in output-channel order.
        context_pressure_feature_index : int or None, optional
            Linearly normalised pressure channel for mass-weighted context.
            None retains the uniform model-level mean.
        constant_observation : array_like or None, optional
            Fixed channel-by-level template replacing atmospheric inputs.
            Actions and categorical field-head selection remain available.
        """
        super().__init__()
        self.encoder_channels = layer_size
        self.register_buffer(
            "constant_observation",
            (
                torch.as_tensor(constant_observation, dtype=torch.float32).unsqueeze(0)
                if constant_observation is not None
                else None
            ),
        )
        self.register_buffer(
            "context_pressure_index",
            (
                torch.tensor(context_pressure_feature_index, dtype=torch.int64)
                if context_pressure_feature_index is not None
                else None
            ),
        )
        self.encoder1 = nn.Conv1d(
            observation_channels + 1,
            self.encoder_channels,
            kernel_size=3,
            padding=1,
        )
        self.encoder2 = nn.Conv1d(
            self.encoder_channels,
            self.encoder_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
        )
        self.fc1 = nn.Conv1d(
            2 * self.encoder_channels,
            layer_size,
            kernel_size=1,
        )
        self.fc2 = nn.Conv1d(layer_size, layer_size, kernel_size=1)
        self.field_feature_index = field_feature_index
        self.register_buffer(
            "field_values",
            (
                torch.as_tensor(field_values, dtype=torch.float32)
                if field_feature_index is not None
                else None
            ),
        )
        output_channels = len(field_values) if field_feature_index is not None else 1
        self.fc3 = nn.Conv1d(layer_size, output_channels, kernel_size=1)

    def forward(self, observations, actions):
        """Return one Q value for every level in each column.

        Parameters
        ----------
        observations : torch.Tensor
            Column features with shape ``(batch, channels, levels)``.
        actions : torch.Tensor
            Column actions with shape ``(batch, levels)``.

        Returns
        -------
        torch.Tensor
            Levelwise Q values with shape ``(batch, levels)``.
        """
        columns = observations
        if self.constant_observation is not None:
            columns = self.constant_observation.expand(columns.shape[0], -1, -1)
        features = torch.cat((columns, actions.unsqueeze(1)), dim=1)
        features = F.relu(self.encoder1(features))
        features = F.relu(self.encoder2(features))
        context = _column_context(features, columns, self.context_pressure_index)
        features = torch.cat((features, context), dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        values = self.fc3(features)
        if self.field_feature_index is not None:
            fields = observations[:, self.field_feature_index, 0]
            head_indices = (
                (fields[:, None] - self.field_values[None, :]).abs().argmin(dim=1)
            )
            values = values.gather(
                1, head_indices[:, None, None].expand(-1, 1, actions.shape[1])
            )
        return values.squeeze(1)


class _TwinColumnCritic(nn.Module):
    """Keep independent value estimates in one checkpointed module."""

    def __init__(self, observation_channels, layer_size, **network_arguments):
        super().__init__()
        self.critics = nn.ModuleList(
            [
                ColumnDDPGCritic(observation_channels, layer_size, **network_arguments)
                for _ in range(2)
            ]
        )

    def forward(self, observations, actions):
        return torch.stack([critic(observations, actions) for critic in self.critics])


class ColumnDDPG(DDPG):
    """Train a convolutional policy from field-balanced column replay."""

    algorithm_name = "ColumnDDPG"
    actor_class = ColumnDDPGActor

    def __init__(
        self,
        *args,
        active_action_indices,
        field_feature_index,
        field_values,
        max_columns_per_field,
        pressure_feature_index,
        pressure_bin_edges,
        supervision_weight=0.0,
        supervised_fields=None,
        training_fields=None,
        field_specific_heads=False,
        mass_weighted_context=False,
        twin_critics=False,
        actor_learning_rate=None,
        constant_observation=None,
        **kwargs,
    ):
        """Initialise column-aware networks and replay.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to :class:`DDPG`.
        active_action_indices : array_like
            Vertical action dimensions included in the losses.
        field_feature_index : int
            Observation channel containing the physical-field identity.
        field_values : array_like
            Identity values for theta, U and V columns.
        max_columns_per_field : int
            Maximum columns retained per field and exchange.
        pressure_feature_index : int
            Channel containing normalised pressure.
        pressure_bin_edges : array_like
            Normalised pressure boundaries for equal-band loss weighting.
        supervision_weight : float, optional
            Weight of the training-only squared teacher-action error.
            Zero preserves the unsupervised actor objective.
        supervised_fields : array_like or None, optional
            Field identities receiving teacher supervision. None includes all
            fields. Exclusion does not renormalise the remaining loss weights.
        training_fields : array_like or None, optional
            Field identities retained in replay for critic and actor updates.
            None includes all fields. Per-field replay capacities are unchanged.
        field_specific_heads : bool, optional
            Select separate final actor and critic outputs for each field,
            retaining shared column encoders. Requires matching checkpoints.
        mass_weighted_context : bool, optional
            Pool full-column context using pressure thickness. Requires a
            linearly normalised pressure feature and matching checkpoints.
        twin_critics : bool, optional
            Use the lower of two target values and clipped target-action
            noise, following TD3. Requires matching training checkpoints.
        actor_learning_rate : float or None, optional
            Actor rate for this forecast, reapplied after optimiser restore.
            None preserves the inherited optimiser learning rate.
        constant_observation : array_like or None, optional
            Fixed channel-by-level inputs for both actor and critics. Replay,
            pressure-band loss weights and teaching targets retain real data.
        **kwargs : dict
            Keyword arguments forwarded to :class:`DDPG`.
        """
        self.active_action_indices = np.asarray(active_action_indices, dtype=np.int64)
        self.field_feature_index = field_feature_index
        self.field_values = tuple(field_values)
        self.training_fields = tuple(
            self.field_values if training_fields is None else training_fields
        )
        if not self.training_fields or not set(self.training_fields).issubset(
            self.field_values
        ):
            raise ValueError(
                "Training fields must be a nonempty subset of field values"
            )
        self.max_columns_per_field = max_columns_per_field
        network_arguments = {}
        if field_specific_heads:
            network_arguments = {
                "field_feature_index": self.field_feature_index,
                "field_values": self.field_values,
            }
        if mass_weighted_context:
            network_arguments["context_pressure_feature_index"] = pressure_feature_index
        if constant_observation is not None:
            network_arguments["constant_observation"] = constant_observation
        if network_arguments:
            self.actor_class = partial(ColumnDDPGActor, **network_arguments)
        super().__init__(*args, **kwargs)
        self.algorithm["field_specific_heads"] = bool(field_specific_heads)
        self.algorithm["mass_weighted_context"] = bool(mass_weighted_context)
        self.algorithm["state_independent"] = constant_observation is not None
        self.algorithm["training_fields"] = list(self.training_fields)
        self.twin_critics = bool(twin_critics)
        self.actor_learning_rate = actor_learning_rate
        self.algorithm["twin_critics"] = self.twin_critics
        self.algorithm["actor_learning_rate"] = actor_learning_rate
        self.logger.info("Column field-specific outputs: %s", field_specific_heads)
        self.logger.info("Column pressure-weighted context: %s", mass_weighted_context)
        self.logger.info(
            "Column state-independent inputs: %s", constant_observation is not None
        )
        self.logger.info("Column replay training fields: %s", self.training_fields)
        if self.inference_only:
            self.recorder.record_algorithm()
            return
        self.supervision_weight = float(supervision_weight)
        self.pressure_feature_index = pressure_feature_index
        self.pressure_bin_edges = torch.as_tensor(
            pressure_bin_edges, dtype=torch.float32, device=self.device
        )
        self.supervised_fields = tuple(
            self.field_values if supervised_fields is None else supervised_fields
        )
        self.action_mask = torch.zeros(self.actor.action_size, device=self.device)
        self.action_mask[self.active_action_indices] = 1.0
        observation_channels = self.actor.observation_channels
        critic_class = _TwinColumnCritic if self.twin_critics else ColumnDDPGCritic
        self.qf1 = critic_class(
            observation_channels,
            self.critic_layer_size,
            **network_arguments,
        ).to(self.device)
        self.qf1_target = critic_class(
            observation_channels,
            self.critic_layer_size,
            **network_arguments,
        ).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.q_optimizer = optim.Adam(self.qf1.parameters(), lr=self.learning_rate)
        self.algorithm.update(
            replay_strategy="field_stratified_column",
            max_columns_per_field=int(max_columns_per_field),
            loss_weighting="equal_field_pressure_bands",
            pressure_bin_edges=list(pressure_bin_edges),
            mask_inactive_critic_actions=True,
            supervision_weight=self.supervision_weight,
            supervised_fields=list(self.supervised_fields),
            target_policy_noise=0.1 if self.twin_critics else 0.0,
            target_noise_clip=0.2 if self.twin_critics else 0.0,
        )
        self.training_configuration_logged = False
        self.recorder.record_algorithm()

    def create_replay_buffer(self):
        """Create the field-balanced column replay buffer directly.

        Returns
        -------
        FieldStratifiedColumnReplayBuffer
            Curated complete-column storage for each nudged field.
        """
        return FieldStratifiedColumnReplayBuffer(
            self.buffer_size,
            int(np.prod(self.envs.single_observation_space.shape)),
            self.actor.action_size,
            self.active_action_indices.size,
            self.device,
            self.field_feature_index,
            self.field_values,
            self.max_columns_per_field,
        )

    def _column_features(self, observations):
        """Reshape flattened observations for convolutional critics."""
        return observations.reshape(
            observations.shape[0],
            self.actor.observation_channels,
            self.actor.action_size,
        )

    def loss_weights(self, observations):
        """Give equal weight to fields and occupied pressure bands.

        Parameters
        ----------
        observations : torch.Tensor
            Replay states shaped ``(columns, channels, levels)``.

        Returns
        -------
        torch.Tensor
            Level weights summing to one, with inactive levels excluded.
        """
        pressure_bands = torch.bucketize(
            observations[:, self.pressure_feature_index, :].contiguous(),
            self.pressure_bin_edges,
            right=True,
        )
        fields = observations[:, self.field_feature_index, 0]
        weights = torch.zeros_like(pressure_bands, dtype=observations.dtype)
        for field_value in self.field_values:
            field_mask = torch.isclose(fields, fields.new_tensor(field_value))
            field_weights = torch.zeros_like(weights)
            for band in range(self.pressure_bin_edges.numel() + 1):
                selected = (
                    field_mask[:, None]
                    & (pressure_bands == band)
                    & self.action_mask.bool()[None, :]
                )
                field_weights += selected / selected.sum().clamp_min(1)
            weights += field_weights / field_weights.sum().clamp_min(1)
        return weights / weights.sum().clamp_min(1)

    def update(
        self,
        actions,
        next_obs,
        rewards,
        terminations,
        truncations,
        infos,
    ):
        """Store complete columns and update levelwise actor-critic losses.

        Notes
        -----
        Truncations preserve bootstrapping from the final observation;
        only physical terminations mask the future-value target.
        """
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

        real_next_obs = next_obs.copy()
        for index, truncated in enumerate(truncations):
            if truncated:
                real_next_obs[index] = infos["final_observation"][index]
        dones = np.asarray(terminations, dtype=np.float32)
        teacher_actions = (
            infos["teacher_actions"] if self.train and self.supervision_weight else None
        )
        retained = slice(None)
        if self.training_fields != self.field_values:
            fields = self._column_features(self.obs)[:, self.field_feature_index, 0]
            retained = np.isin(fields, self.training_fields)
        self.rb.add(
            self.obs[retained],
            real_next_obs[retained],
            actions[retained],
            level_rewards[retained],
            dones[retained],
            teacher_actions=(
                teacher_actions[retained] if teacher_actions is not None else None
            ),
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
            if self.actor_learning_rate is not None:
                for parameter_group in self.actor_optimizer.param_groups:
                    parameter_group["lr"] = self.actor_learning_rate
            if not self.training_configuration_logged:
                self.logger.info(
                    "COLUMN_TRAINING actor_lr=%g critic_lr=%g gamma=%g "
                    "twin_critics=%s supervision_weight=%g gradient_steps=%d",
                    self.actor_optimizer.param_groups[0]["lr"],
                    self.q_optimizer.param_groups[0]["lr"],
                    self.gamma,
                    self.twin_critics,
                    self.supervision_weight,
                    self.gradient_steps,
                )
                self.training_configuration_logged = True
            for _ in range(self.gradient_steps):
                replay = self.rb.sample(self.batch_size)
                observations = self._column_features(replay.observations)
                loss_weights = self.loss_weights(observations)
                next_observations = self._column_features(replay.next_observations)
                with torch.no_grad():
                    next_actions = self.target_actor(replay.next_observations)
                    if self.twin_critics:
                        target_noise = (0.1 * torch.randn_like(next_actions)).clamp(
                            -0.2, 0.2
                        )
                        next_actions = (next_actions + target_noise).clamp(
                            self.actor.action_bias - self.actor.action_scale,
                            self.actor.action_bias + self.actor.action_scale,
                        )
                    next_q_values = self.qf1_target(
                        next_observations,
                        next_actions * self.action_mask,
                    )
                    if self.twin_critics:
                        next_q_values = next_q_values.min(dim=0).values
                    target_q_values = (
                        replay.rewards
                        + (1.0 - replay.dones[:, None]) * self.gamma * next_q_values
                    )

                q_values = self.qf1(observations, replay.actions * self.action_mask)
                critic_loss = (
                    loss_weights * (q_values - target_q_values).square()
                ).sum()
                self.q_optimizer.zero_grad()
                critic_loss.backward()
                self.q_optimizer.step()
                self.gradient_step += 1

                if self.gradient_step % self.policy_frequency == 0:
                    self.qf1.requires_grad_(False)
                    actor_actions = self.actor(replay.observations)
                    actor_q_values = self.qf1(
                        observations,
                        actor_actions * self.action_mask,
                    )
                    if self.twin_critics:
                        actor_q_values = actor_q_values[0]
                    actor_loss = -(loss_weights * actor_q_values).sum()
                    if self.supervision_weight:
                        fields = observations[:, self.field_feature_index, 0]
                        supervised = torch.zeros_like(fields, dtype=torch.bool)
                        for field in self.supervised_fields:
                            supervised |= torch.isclose(
                                fields, fields.new_tensor(field)
                            )
                        teacher_loss = (
                            loss_weights
                            * supervised[:, None]
                            * (actor_actions - replay.teacher_actions).square()
                        ).sum()
                        actor_loss += self.supervision_weight * teacher_loss
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    self.qf1.requires_grad_(True)

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

        self.global_step += 1
