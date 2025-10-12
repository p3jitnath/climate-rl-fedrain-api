import os

import numpy as np
import torch
from smartredis import Client


class FedRL:
    def __init__(self, actor, cid, logger, weights_folder=None):
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
        if self.actor:
            while not self.redis.tensor_exists(
                f"actor_network_weights_g2c_s{self.cid}"
            ):
                pass

            actor_weights = self.redis.get_tensor(
                f"actor_network_weights_g2c_s{self.cid}"
            )
            # self.logger.debug(f'Actor {self.cid} L {actor_weights[0:5]}')

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
