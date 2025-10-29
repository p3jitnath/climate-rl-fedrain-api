"""Flower client adapter and helpers for FedRL experiments.

This module provides a Flower ``NumPyClient`` bridge that uses SmartRedis
to transfer flattened actor weights between the server and a local RL
process. It also includes a small factory helper used by the server.
"""

import logging
import multiprocessing as mp
import os

import flwr as fl
import numpy as np
import smartredis

from fedrain.utils import setup_logger


class FlowerClient(fl.client.NumPyClient):
    """Flower client adapter that bridges FL clients to local actor processes.

    This client implementation conforms to Flower's ``NumPyClient`` API and
    uses SmartRedis to transfer flattened actor weights between the server
    and a background RL process. The client optionally spawns a background
    process when a shared signal key is not present in Redis.

    Parameters
    ----------
    seed : int
        Random seed used by the spawned environment/process.
    cid : int
        Client identifier used to form unique Redis tensor keys.
    fn : callable
        Function used to create/start the client-side training process.
    actor : torch.nn.Module
        Local actor instance used to read/write parameter arrays.
    num_steps : int
        Number of local steps to run during ``fit`` (returned to server).

    """

    def __init__(self, seed, cid, fn, actor, num_steps):
        """Create a Flower client adapter and (optionally) start a worker.

        Parameters
        ----------
        seed : int
            Random seed used by the spawned environment/process.
        cid : int
            Client identifier used to form unique Redis tensor keys.
        fn : callable
            Function used to create/start the client-side training process.
        actor : torch.nn.Module
            Local actor instance used to read/write parameter arrays.
        num_steps : int
            Number of local steps to run during ``fit`` (reported to server).

        """
        super().__init__()

        self.cid = cid
        self.seed = seed
        self.actor = actor
        self.num_steps = num_steps

        self.logger = setup_logger(f"FLWR CLIENT {cid}", logging.DEBUG)

        self.REDIS_ADDRESS = os.getenv("SSDB")
        if self.REDIS_ADDRESS is None:
            raise EnvironmentError("SSDB environment variable is not set.")
        self.redis = smartredis.Client(address=self.REDIS_ADDRESS, cluster=False)
        self.logger.debug(f"Connected to Redis server: {self.REDIS_ADDRESS}")

        is_alive = self.redis.tensor_exists(f"SIGALIVE_S{self.cid}")
        self.logger.debug(f"is_alive: {is_alive}")
        if not is_alive:
            try:
                proc = mp.Process(target=fn, args=(seed, cid))
                proc.daemon = False
                proc.start()
                self.logger.debug(f"Started background process pid={proc.pid}")
                self.child_process = proc
            except Exception:
                self.logger.exception("Failed to start background process")
                raise

    def set_parameters(self, parameters):
        """Publish flattened parameter arrays to the remote store.

        Parameters
        ----------
        parameters : Sequence[numpy.ndarray]
            List of layer parameter arrays as provided by Flower's server.

        """
        if self.actor:
            actor_weights = np.concatenate([param.flatten() for param in parameters])

            self.redis.put_tensor(
                f"actor_network_weights_g2c_s{self.cid}", actor_weights
            )

    def get_parameters(self, config):
        """Retrieve flattened parameters published by the local actor process.

        This method blocks until the corresponding tensor is present in the
        remote store, then reconstructs the per-layer arrays matching the
        shapes of ``self.actor.parameters()``.

        Parameters
        ----------
        config : dict
            Configuration dict supplied by Flower (unused here but kept to
            satisfy the NumPyClient API).

        Returns
        -------
        list
            A list of numpy arrays matching the shapes of the actor's
            parameters (ready to be consumed by Flower's aggregation).

        """
        if self.actor:
            while not self.redis.tensor_exists(
                f"actor_network_weights_c2g_s{self.cid}"
            ):
                continue

            actor_weights = self.redis.get_tensor(
                f"actor_network_weights_c2g_s{self.cid}"
            )
            parameters = []
            offset = 0
            for param in self.actor.parameters():
                size = np.prod(param.shape)
                layer_weights = actor_weights[offset : offset + size].reshape(
                    param.shape
                )
                parameters.append(layer_weights)
                offset += size

            self.redis.delete_tensor(f"actor_network_weights_c2g_s{self.cid}")

        return parameters

    def fit(self, parameters, config):
        """Handle a ``fit`` round from the Flower server.

        The client sets incoming parameters on the local process, waits for the
        process to produce updated parameters, and returns them along with the
        number of local steps performed.

        Parameters
        ----------
        parameters : Sequence[numpy.ndarray]
            Parameters received from the server.
        config : dict
            Configuration metadata from the Flower server.

        Returns
        -------
        tuple
            Tuple of ``(updated_parameters, num_steps, metrics)`` matching the
            Flower client API.

        """
        # self.logger.debug(f"{config['server_round']} - Setting parameters")
        self.set_parameters(parameters)

        # self.logger.debug(f"{config['server_round']} - Loading parameters")
        updated_parameters = self.get_parameters(config)

        return (updated_parameters, self.num_steps, {})


def generate_client_fn(seed, fn, actor, num_steps):
    """Return a Flower-compatible client factory function.

    The returned function expects a Flower client context and produces a
    connected ``FlowerClient`` instance configured for the partition-id in
    the context.

    Parameters
    ----------
    seed : int
        Seed to pass to each client process.
    fn : callable
        Function used to start the client-side process.
    actor : torch.nn.Module
        Actor instance to share with clients.
    num_steps : int
        Number of local steps to report for each ``fit`` call.

    Returns
    -------
    callable
        A function accepting a Flower ``context`` and returning a
        ``FlowerClient`` bound to that context.

    """

    def client_fn(context):
        return FlowerClient(
            seed, int(context.node_config["partition-id"]), fn, actor, num_steps
        ).to_client()

    return client_fn
