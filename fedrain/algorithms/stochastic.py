"""Stochastic baseline algorithm used by FedRAIN.

This module implements a policy-free agent that independently samples each
action from the environment action space. It follows the rollout interface
used by the FedRAIN DDPG implementation without performing learning updates.
"""

import logging

import numpy as np

from fedrain.algorithms.base import BaseAlgorithm
from fedrain.analysis.recorder import Recorder
from fedrain.utils import setup_logger


class Stochastic(BaseAlgorithm):
    """Random-action algorithm wrapper.

    The stochastic agent provides the same ``predict`` and ``update`` rollout
    interface as the trainable FedRAIN algorithms. Observations do not affect
    action selection, and no policy, replay buffer, or optimizer is created.

    Parameters
    ----------
    envs : gym.Env or vectorized env
        Environment or vectorized environments providing
        ``single_action_space`` and ``num_envs``.
    seed : int
        Random seed used when resetting the environments.
    n_envs : int, optional
        Number of parallel environments (default 1).
    level : int, optional
        Logging level forwarded to the internal logger
        (default logging.DEBUG).
    record_dir : str or None, optional
        Directory where episodic-return record files are written. Defaults
        to the ``$TMPDIR`` environment variable or ``/tmp/climaterl/records``.
    retain_latest_only : bool, optional
        If True, retain only the latest step-data file (default False).
    train : bool, optional
        Algorithm mode recorded in the run metadata (default True). Action
        selection remains stochastic in both training and inference modes.

    """

    def __init__(
        self,
        envs,
        seed,
        n_envs=1,
        level=logging.DEBUG,
        record_dir=None,
        retain_latest_only=False,
        train=True,
    ):
        """Initialize the random-action agent and reset its environments.

        Parameters
        ----------
        envs : gym.Env or vectorized env
            Environment providing the vector action space.
        seed : int
            Environment reset seed.
        n_envs : int, optional
            Number of parallel environments.
        level : int, optional
            Logger level.
        record_dir : path-like, optional
            Output directory for rollout records.
        retain_latest_only : bool, optional
            Whether to retain only the newest step-data file.
        train : bool, optional
            Mode stored in algorithm metadata.

        Notes
        -----
        The constructor creates the episode recorder, resets the vectorized
        environments with ``seed``, and initializes the global step counter.
        It does not create any trainable model state.

        """
        super().__init__()

        self.name = "Stochastic"
        self.envs = envs
        self.seed = seed
        self.n_envs = n_envs
        self.train = train
        self.logger = setup_logger(self.name, level)
        self.algorithm = {
            "name": self.name.lower(),
            "seed": int(self.seed),
            "train": self.train,
        }
        self.recorder = Recorder(
            algorithm=self.algorithm,
            record_dir=record_dir,
            level=level,
            retain_latest_only=retain_latest_only,
        )
        self.obs, _ = self.envs.reset(seed=self.seed)
        self.global_step = 1

    def predict(self, obs):
        """Sample one action for each vectorized environment.

        Parameters
        ----------
        obs : array-like
            Current observations. They are accepted for compatibility with
            other FedRAIN algorithms but do not influence action selection.

        Returns
        -------
        numpy.ndarray
            Random actions with a leading dimension equal to
            ``envs.num_envs``.

        """
        del obs
        return np.array(
            [self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)]
        )

    def update(
        self,
        actions,
        next_obs,
        rewards,
        terminations,
        truncations,
        infos,
    ):
        """Record completed episodes and advance the rollout state.

        Parameters
        ----------
        actions : array-like
            Actions taken by the agent. No learning update is performed.
        next_obs : array-like
            Observations returned by the environments after the step.
        rewards : array-like
            Rewards returned by the environments. Episodic returns are read
            from ``infos`` when an episode terminates.
        terminations : array-like
            Termination flags returned by the environments.
        truncations : array-like
            Truncation flags returned by the environments.
        infos : dict
            Additional environment information, including ``final_info`` for
            completed episodes.

        Notes
        -----
        This method does not update a policy. It records completed episodic
        returns and step data, stores ``next_obs`` as the current observation,
        and increments ``global_step``.

        """
        del terminations, truncations
        if self.check_episode_termination(infos):
            returns = []
            for idx, info in enumerate(infos["final_info"]):
                self.logger.debug(
                    "idx=%s, seed=%s, global_step=%s, episodic_return=%s",
                    idx,
                    self.seed,
                    self.global_step,
                    info["episode"]["r"],
                )
                returns.append(float(info["episode"]["r"][0]))
            returns = np.array(returns)
            average_return = np.mean(returns)
            self.recorder.record_episodic_return(
                average_return,
                self.global_step,
            )
            self.recorder.sr.save(self.global_step, None, returns)

        self.recorder.sr.add(
            self.global_step,
            self.obs,
            next_obs,
            actions,
            rewards,
        )

        self.obs = next_obs
        self.global_step += 1
