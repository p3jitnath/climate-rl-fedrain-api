import logging
import multiprocessing as mp
import os

import flwr as fl
import numpy as np
import smartredis

from fedrain.utils import setup_logger


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, seed, cid, fn, actor, num_steps):
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
        if self.actor:
            actor_weights = np.concatenate([param.flatten() for param in parameters])

            self.redis.put_tensor(
                f"actor_network_weights_g2c_s{self.cid}", actor_weights
            )

    def get_parameters(self, config):
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
        # self.logger.debug(f"{config['server_round']} - Setting parameters")
        self.set_parameters(parameters)

        # self.logger.debug(f"{config['server_round']} - Loading parameters")
        updated_parameters = self.get_parameters(config)

        return (updated_parameters, self.num_steps, {})


def generate_client_fn(seed, fn, actor, num_steps):
    def client_fn(context):
        return FlowerClient(
            seed, int(context.node_config["partition-id"]), fn, actor, num_steps
        ).to_client()

    return client_fn
