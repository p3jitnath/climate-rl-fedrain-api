"""SmartRedis-based helper for exchanging actor weights.

This module implements :class:`FedRL`, a tiny abstraction used to publish
and retrieve flattened PyTorch actor parameters via a SmartRedis store.
"""

import os

import numpy as np
import torch
from smartredis import Client


class FedRL:
    """Helper for exchanging actor weights via a Redis-compatible store.

    The ``FedRL`` class provides a small abstraction around a remote tensor
    store (SmartRedis) used in this project to publish and retrieve flattened
    actor network weights. It assumes the environment variable ``SSDB`` points
    to the server address and that the actor provided exposes a standard
    PyTorch ``state_dict``/``parameters`` interface.

    Parameters
    ----------
    actor : torch.nn.Module
        The local actor network whose weights will be saved and updated.
    cid : int
        Client identifier used to form unique Redis tensor keys.
    logger : logging.Logger
        Logger used for debug/info messages.
    weights_folder : str or None, optional
        Optional local folder where received weights will be saved for
        inspection (default: None).

    Raises
    ------
    EnvironmentError
        If the ``SSDB`` environment variable is not set.

    """

    def __init__(self, actor, cid, logger, weights_folder=None):
        """Initialise the FedRL helper.

        Parameters
        ----------
        actor : torch.nn.Module
            Local actor network.
        cid : int
            Client identifier used to form unique Redis tensor keys.
        logger : logging.Logger
            Logger for debug/info messages.
        weights_folder : str or None, optional
            Optional folder to save received weights for inspection.

        """
        self.actor = actor
        self.logger = logger
        self.cid = cid
        self.weights_folder = None

        self.REDIS_ADDRESS = os.getenv("SSDB")
        if self.REDIS_ADDRESS is None:
            raise EnvironmentError("SSDB environment variable is not set.")
        self.redis = Client(address=self.REDIS_ADDRESS, cluster=False)
        self.logger.debug(f"FedRL - Connected to Redis server: {self.REDIS_ADDRESS}")

    def load_weights(self, step_count):
        """Load flattened actor weights from the remote store and apply to actor.

        This method blocks until a tensor with the key
        ``actor_network_weights_g2c_s{cid}`` is present in the store. The
        flattened array is reshaped into each parameter's shape and copied
        into the local actor parameters.

        Parameters
        ----------
        step_count : int
            Current training step count, used when optionally saving received
            weights to disk.

        Notes
        -----
        The method logs the L2-norm of the parameter differences after the
        update and deletes the remote tensor when finished.

        """
        if self.actor:
            while not self.redis.tensor_exists(
                f"actor_network_weights_g2c_s{self.cid}"
            ):
                pass

            actor_weights = self.redis.get_tensor(
                f"actor_network_weights_g2c_s{self.cid}"
            )

            old_actor_params = [
                param.clone().detach() for param in self.actor.parameters()
            ]

            offset = 0
            for param in self.actor.parameters():
                size = np.prod(param.shape)
                param.data.copy_(
                    torch.tensor(
                        actor_weights[offset : offset + size].reshape(param.shape)
                    )
                )
                offset += size

            if self.weights_folder:
                torch.save(
                    self.actor.state_dict(),
                    f"{self.weights_folder}/actor/actor-fedRL-{step_count}.pth",
                )

            actor_diff_norm = sum(
                torch.norm(old - new)
                for old, new in zip(old_actor_params, self.actor.parameters())
            )
            self.logger.debug(f"Actor norm: {actor_diff_norm}")
            self.redis.delete_tensor(f"actor_network_weights_g2c_s{self.cid}")

    def save_weights(self, step_count):
        """Serialize local actor parameters and publish them to the store.

        Parameters
        ----------
        step_count : int
            Current training step count. Provided for symmetry with
            :meth:`load_weights` (not used when publishing), and to maintain a
            consistent interface for callers.

        """
        if self.actor:
            actor_weights = np.concatenate(
                [
                    param.data.cpu().numpy().flatten()
                    for param in self.actor.parameters()
                ]
            )
            self.redis.put_tensor(
                f"actor_network_weights_c2g_s{self.cid}", actor_weights
            )
