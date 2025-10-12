import functools

import flwr as fl
import gymnasium as gym
from flwr.common import FitIns

from fedrain.fedrl.client import generate_client_fn
from fedrain.utils import instantiate_redis_server, make_env


class FedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        min_fit_clients,
        min_available_clients,
        fraction_evaluate=0.0,
        **kwargs,
    ):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            fraction_evaluate=fraction_evaluate,
            **kwargs,
        )

    def configure_fit(self, server_round, parameters, client_manager):

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        fit_configurations = []
        for client in clients:
            fit_configurations.append(
                (client, FitIns(parameters, {"server_round": server_round}))
            )
        return fit_configurations


class FLWRServer:
    def __init__(self, num_clients, num_rounds, strategy=FedAvg):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.strategy = strategy(
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
            fraction_evaluate=0.0,
        )
        instantiate_redis_server()

    def generate_actor(self, env_class, actor_class, layer_size):
        envs = gym.vector.SyncVectorEnv(
            [make_env(functools.partial(env_class, cid=-1))]
        )
        self.actor = actor_class(envs, layer_size)

    def set_client(self, *args, **kwargs):
        if not hasattr(self, "actor") or self.actor is None:
            raise RuntimeError(
                "Actor not set. Call generate_actor(...) before set_client(...)."
            )
        kwargs["actor"] = self.actor
        self.client_fn = generate_client_fn(*args, **kwargs)

    def serve(self, cpus_per_client=3, gpus_per_client=0):
        total_cpus = self.num_clients * cpus_per_client + 1
        total_gpus = gpus_per_client
        ray_init_args = {
            "num_cpus": total_cpus,
            "num_gpus": total_gpus,
        }

        fl.simulation.start_simulation(
            client_fn=self.client_fn,
            num_clients=self.num_clients,
            strategy=self.strategy,
            ray_init_args=ray_init_args,
            client_resources={"num_cpus": cpus_per_client, "num_gpus": gpus_per_client},
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
        )
