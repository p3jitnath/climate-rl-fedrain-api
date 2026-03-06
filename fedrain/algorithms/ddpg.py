"""DDPG algorithm implementation used by FedRAIN.

This module implements a compact DDPG agent (actor, critic, replay
buffer and training loop) intended for experiments and integration tests.
"""

import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer

from fedrain.algorithms.base import BaseAlgorithm
from fedrain.analysis.recorder import Recorder
from fedrain.fedrl import FedRL
from fedrain.utils import setup_logger


class DDPGActor(nn.Module):
    """Actor network for DDPG.

    The actor maps observations to continuous actions in the environment's
    action space. The final layer uses a tanh activation then scales and
    biases the output to match the environment action range.

    Parameters
    ----------
    envs : gym.Env or vectorized env
        Environment or vectorized environments providing ``single_observation_space``
        and ``single_action_space`` used to determine input/output sizes and
        action scaling.
    layer_size : int
        Width of the hidden layers.

    """

    def __init__(self, envs, layer_size):
        """Initialize the actor network layers and scaling buffers.

        Parameters
        ----------
        envs : gym.Env or vectorized env
            Environment used to infer input/output shapes.
        layer_size : int
            Width of the hidden layers.

        """
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(envs.single_observation_space.shape).prod(), layer_size
        )
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc_mu = nn.Linear(layer_size, np.prod(envs.single_action_space.shape))
        # Scale and bias to map tanh outputs into environment action range
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (envs.action_space.high - envs.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (envs.action_space.high + envs.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        """Compute action(s) from observation(s).

        Parameters
        ----------
        x : torch.Tensor
            Batch of observations with shape (batch_size, observation_dim).

        Returns
        -------
        torch.Tensor
            Batch of actions scaled to the environment action bounds.

        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class DDPGCritic(nn.Module):
    """Critic (Q-function) network for DDPG.

    The critic takes observations and actions and predicts a scalar Q-value.

    Parameters
    ----------
    envs : gym.Env or vectorized env
        Environment object used to infer input dimensions.
    layer_size : int
        Width of the hidden layers.

    """

    def __init__(self, envs, layer_size):
        """Initialize the critic network layers.

        Parameters
        ----------
        envs : gym.Env or vectorized env
            Environment used to infer input/output shapes.
        layer_size : int
            Width of the hidden layers.

        """
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(envs.single_observation_space.shape).prod()
            + np.prod(envs.single_action_space.shape),
            layer_size,
        )
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, 1)

    def forward(self, x, a):
        """Compute Q-value for given observations and actions.

        Parameters
        ----------
        x : torch.Tensor
            Batch of observations with shape (batch_size, observation_dim).
        a : torch.Tensor
            Batch of actions with shape (batch_size, action_dim).

        Returns
        -------
        torch.Tensor
            Predicted Q-values with shape (batch_size, 1).

        """
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DDPG(BaseAlgorithm):
    """Deep Deterministic Policy Gradient (DDPG) algorithm wrapper.

    This class provides a minimal DDPG implementation used in FedRAIN for
    continuous-action environments. It composes an actor and critic network,
    a replay buffer, and exposes ``predict`` and ``update`` methods that
    are used by the training loop.

    Parameters
    ----------
    envs : gym.Env or vectorized env
        Environment (or vectorized environments) providing observation and
        action spaces used to build networks and the replay buffer.
    seed : int
        Random seed used for environment reset and reproducibility.
    n_envs : int, optional
        Number of parallel environments (if using vectorized env) (default 1).
    learning_rate : float, optional
        Learning rate for both actor and critic optimizers (default 3e-4).
    buffer_size : int, optional
        Maximum size of the replay buffer (default 1e6).
    gamma : float, optional
        Discount factor for returns (default 0.99).
    tau : float, optional
        Soft-update coefficient for target networks (default 0.005).
    batch_size : int, optional
        Mini-batch size sampled from the replay buffer (default 256).
    exploration_noise : float, optional
        Scale of Gaussian exploration noise applied to actor outputs
        (default 0.1).
    learning_starts : int, optional
        Number of environment steps to collect before learning starts
        (default 1000).
    policy_frequency : int, optional
        How often (in gradient steps) the actor/target networks are updated
        relative to the critic (default 2).
    noise_clip : float, optional
        Clipping value for exploration noise (default 0.5).
    actor_layer_size : int, optional
        Hidden layer width for the actor network (default 256).
    critic_layer_size : int, optional
        Hidden layer width for the critic network (default 256).
    device : str, optional
        Torch device name (e.g. 'cpu' or 'cuda') to place tensors/models.
    level : int, optional
        Logging level forwarded to the internal logger (default logging.DEBUG).
    fedRLConfig : dict or None, optional
        If provided, configuration dict used to enable federated weight
        exchange via FedRL.
    record_dir : str or None, optional
        Directory where episodic-return record files are written.  Defaults
        to the ``$TMPDIR`` environment variable (or ``/tmp`` if unset).
    """

    def __init__(
        self,
        envs,
        seed,
        n_envs=1,
        learning_rate=3e-4,
        buffer_size=int(1e6),
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        exploration_noise=0.1,
        learning_starts=1000,
        policy_frequency=2,
        noise_clip=0.5,
        actor_layer_size=256,
        critic_layer_size=256,
        device="cpu",
        level=logging.DEBUG,
        fedRLConfig=None,
        record_dir=None,
    ):
        """Initialize the DDPG agent and its components.

        Notes
        -----
        The constructor creates actor/critic networks, their target copies,
        optimizers, and a replay buffer. If ``fedRLConfig`` is provided, a
        ``FedRL`` helper is instantiated and initial weights are exchanged.

        """
        super().__init__()

        self.name = "DDPG"

        self.envs = envs
        self.seed = seed
        self.n_envs = n_envs
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.learning_starts = learning_starts
        self.policy_frequency = policy_frequency
        self.noise_clip = noise_clip
        self.actor_layer_size = actor_layer_size
        self.critic_layer_size = critic_layer_size
        self.device = device
        self.logger = setup_logger(self.name, level)
        self.fedRLConfig = fedRLConfig

        self.algorithm = {
            "name": self.name.lower(),
            "learning_rate": self.learning_rate,
            "buffer_size": int(self.buffer_size),
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "exploration_noise": self.exploration_noise,
            "learning_starts": self.learning_starts,
            "policy_frequency": self.policy_frequency,
            "noise_clip": self.noise_clip,
            "actor_layer_size": self.actor_layer_size,
            "critic_layer_size": self.critic_layer_size,
            "device": self.device,
            "seed": int(self.seed),
        }

        if self.fedRLConfig is not None:
            self.algorithm["fedRLConfig"] = self.fedRLConfig

        self.recorder = Recorder(
            algorithm=self.algorithm, record_dir=record_dir, level=level
        )

        self.actor = DDPGActor(self.envs, self.actor_layer_size).to(self.device)
        self.qf1 = DDPGCritic(self.envs, self.critic_layer_size).to(self.device)
        self.qf1_target = DDPGCritic(self.envs, self.critic_layer_size).to(self.device)
        self.target_actor = DDPGActor(self.envs, self.actor_layer_size).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())

        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()), lr=self.learning_rate
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=self.learning_rate
        )

        self.rb = ReplayBuffer(
            self.buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            self.device,
            n_envs=self.n_envs,
            handle_timeout_termination=False,
        )

        if self.fedRLConfig is not None:
            self.fedRL = FedRL(self.actor, self.fedRLConfig["cid"], self.logger)
            self.fedRL.save_weights(0)
            self.fedRL.load_weights(0)

        self.obs, _ = envs.reset(seed=self.seed)
        self.global_step = 1

    def save(self, folder_path, timestep_offset=-1, replay_buffer=True):
        """Save model weights, optimizers and optionally replay buffer to *folder_path*.

        Parameters
        ----------
        folder_path : str
            Directory where checkpoint files will be written. The directory
            will be created if it does not exist.
        timestep_offset : int, optional
            Offset for the timestep when saving the weights (default is -1).
            This can be used to adjust the global timestep if saving a checkpoint during training.
        replay_buffer : bool, optional
            If True, save the replay buffer as well (default is True).

        Notes
        -----
        Files written:
        - actor.pt, qf1.pt, target_actor.pt, qf1_target.pt
        - actor_opt.pt, q_opt.pt
        - replay_buffer.pt (torch-saved object or dict fallback)
        - metadata.json

        """
        os.makedirs(folder_path, exist_ok=True)

        torch.save(self.actor.state_dict(), os.path.join(folder_path, "actor.pt"))
        torch.save(self.qf1.state_dict(), os.path.join(folder_path, "qf1.pt"))
        torch.save(
            self.target_actor.state_dict(), os.path.join(folder_path, "target_actor.pt")
        )
        torch.save(
            self.qf1_target.state_dict(), os.path.join(folder_path, "qf1_target.pt")
        )

        torch.save(
            self.actor_optimizer.state_dict(), os.path.join(folder_path, "actor_opt.pt")
        )
        torch.save(self.q_optimizer.state_dict(), os.path.join(folder_path, "q_opt.pt"))

        if replay_buffer:
            rb_file = os.path.join(folder_path, "replay_buffer.pt")
            try:
                torch.save(self.rb, rb_file)
            except Exception:
                try:
                    torch.save(self.rb.__dict__, rb_file)
                except Exception as e:
                    self.logger.warning(f"Failed to save replay buffer: {e}")

        meta = {
            "global_step": int(self.global_step) + timestep_offset,
            "seed": int(self.seed),
            "device": str(self.device),
        }
        with open(
            os.path.join(folder_path, "metadata.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump(meta, fh, indent=4)

    def load(self, folder_path, timestep_offset=1, replay_buffer=True):
        """Load model weights, optimizers and optionally replay buffer from *folder_path*.

        Parameters
        ----------
        folder_path : str
            Directory containing checkpoint files previously written by
            :meth:`save`.
        timestep_offset : int, optional
            Offset for the timestep when loading the weights (default is 1).
            This can be used to adjust the global timestep if resuming from a checkpoint.
        replay_buffer : bool, optional
            If True, load the replay buffer as well (default is True).

        """
        # models
        actor_file = os.path.join(folder_path, "actor.pt")
        qf1_file = os.path.join(folder_path, "qf1.pt")
        target_actor_file = os.path.join(folder_path, "target_actor.pt")
        qf1_target_file = os.path.join(folder_path, "qf1_target.pt")

        if os.path.exists(actor_file):
            self.actor.load_state_dict(torch.load(actor_file, map_location=self.device))
        if os.path.exists(qf1_file):
            self.qf1.load_state_dict(torch.load(qf1_file, map_location=self.device))
        if os.path.exists(target_actor_file):
            self.target_actor.load_state_dict(
                torch.load(target_actor_file, map_location=self.device)
            )
        if os.path.exists(qf1_target_file):
            self.qf1_target.load_state_dict(
                torch.load(qf1_target_file, map_location=self.device)
            )

        # optimizers
        actor_opt_file = os.path.join(folder_path, "actor_opt.pt")
        q_opt_file = os.path.join(folder_path, "q_opt.pt")
        if os.path.exists(actor_opt_file):
            try:
                self.actor_optimizer.load_state_dict(
                    torch.load(actor_opt_file, map_location=self.device)
                )
            except Exception:
                self.logger.warning("Failed to load actor optimizer state")
        if os.path.exists(q_opt_file):
            try:
                self.q_optimizer.load_state_dict(
                    torch.load(q_opt_file, map_location=self.device)
                )
            except Exception:
                self.logger.warning("Failed to load critic optimizer state")

        # replay buffer
        rb_file = os.path.join(folder_path, "replay_buffer.pt")
        if os.path.exists(rb_file) and replay_buffer:
            try:
                loaded = torch.load(rb_file, map_location=self.device)
                if isinstance(loaded, ReplayBuffer):
                    self.rb = loaded
                elif isinstance(loaded, dict):
                    try:
                        self.rb.__dict__.update(loaded)
                    except Exception:
                        self.logger.warning(
                            "Failed to apply replay buffer dict to current buffer"
                        )
                else:
                    # unknown object type; attempt to replace if possible
                    try:
                        self.rb = loaded
                    except Exception:
                        self.logger.warning("Loaded replay buffer could not be applied")
            except Exception as e:
                self.logger.warning(f"Failed to load replay buffer: {e}")

        # metadata
        meta_file = os.path.join(folder_path, "metadata.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r") as fh:
                    meta = json.load(fh)
                self.global_step = (
                    int(meta.get("global_step", self.global_step)) + timestep_offset
                )
                self.seed = int(meta.get("seed", self.seed))
            except Exception:
                self.logger.warning("Failed to read metadata.json")

    def predict(self, obs):
        """Compute actions for given observations.

        When the agent has not yet collected enough experience (global
        step < learning_starts) actions are sampled from the environment's
        action space. Otherwise the actor network is used with added
        exploration noise.
        """
        if self.global_step < self.learning_starts:
            actions = np.array(
                [
                    self.envs.single_action_space.sample()
                    for _ in range(self.envs.num_envs)
                ]
            )
        else:
            with torch.no_grad():
                actions = self.actor(torch.Tensor(obs).to(self.device))
                actions += torch.normal(
                    0, self.actor.action_scale * self.exploration_noise
                )
                actions = (
                    actions.cpu()
                    .numpy()
                    .clip(
                        self.envs.single_action_space.low,
                        self.envs.single_action_space.high,
                    )
                )
        return actions

    def update(self, actions, next_obs, rewards, terminations, truncations, infos):
        """Perform a training update using data from the replay buffer.

        Parameters
        ----------
        actions : array-like
            Actions taken by the agent.
        next_obs : array-like
            Next observations observed after taking ``actions``.
        rewards : array-like
            Rewards received.
        terminations : array-like
            Termination flags for episodes.
        truncations : array-like
            Truncation flags for episodes.
        infos : dict
            Additional environment-provided info structures.

        """
        if self.check_episode_termination(infos):
            returns_sum = 0
            for idx, info in enumerate(infos["final_info"]):
                if self.fedRLConfig:
                    self.logger.debug(
                        f"idx={idx}, seed={self.seed}, cid={self.fedRLConfig['cid']}, global_step={self.global_step}, episodic_return={info['episode']['r']}"
                    )
                else:
                    self.logger.debug(
                        f"idx={idx}, seed={self.seed}, global_step={self.global_step}, episodic_return={info['episode']['r']}"
                    )
                returns_sum += float(info["episode"]["r"])
            avg_return = returns_sum / len(infos["final_info"])
            self.recorder.record_episodic_return(avg_return, self.global_step)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        self.rb.add(self.obs, real_next_obs, actions, rewards, terminations, infos)

        self.obs = next_obs

        if self.global_step > self.learning_starts:

            data = self.rb.sample(self.batch_size)
            with torch.no_grad():
                next_state_actions = self.target_actor(data.next_observations)
                qf1_next_target = self.qf1_target(
                    data.next_observations, next_state_actions
                )
                target_q_values = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * self.gamma * (qf1_next_target).view(-1)

            qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, target_q_values)

            self.q_optimizer.zero_grad()
            qf1_loss.backward()
            self.q_optimizer.step()

            if self.global_step % self.policy_frequency == 0:
                actor_loss = -self.qf1(
                    data.observations, self.actor(data.observations)
                ).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            if self.global_step % self.policy_frequency == 0:
                for param, target_param in zip(
                    self.actor.parameters(), self.target_actor.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.qf1.parameters(), self.qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

        if self.fedRLConfig is not None and "final_info" in infos:
            if (
                self.global_step
                % (self.fedRLConfig["flwr_episodes"] * self.fedRLConfig["num_steps"])
                == 0
            ):
                # self.logger.debug(f"{self.seed} - Saving local weights")
                self.fedRL.save_weights(self.global_step)

                # self.logger.debug(f"{self.seed} - Loading global weights")
                self.fedRL.load_weights(self.global_step)

        self.global_step += 1
