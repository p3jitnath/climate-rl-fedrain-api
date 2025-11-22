"""Federated server and Flower strategy helpers for simulations.

This module contains lightweight server helpers used to orchestrate
Flower-based federated simulations in local experiments.
"""

import functools
import multiprocessing
import subprocess

import flwr as fl
import gymnasium as gym
import ray
from flwr.common import FitIns

from fedrain.fedrl.client import generate_client_fn
from fedrain.utils import RedisServer, make_env, set_deathsig


class FedAvg(fl.server.strategy.FedAvg):
    """Thin FedAvg strategy that exposes a configurable ``configure_fit``.

    This subclass exists to allow passing per-round configuration to clients
    during the simulation. It forwards all constructor arguments to the base
    Flower ``FedAvg`` implementation.
    """

    def __init__(
        self,
        min_fit_clients,
        min_available_clients,
        fraction_evaluate=0.0,
        **kwargs,
    ):
        """Construct a thin FedAvg strategy forwarding configuration.

        Parameters
        ----------
        min_fit_clients, min_available_clients : int
            Minimum number of clients required for fitting and availability.
        fraction_evaluate : float, optional
            Fraction of clients used for evaluation.
        **kwargs
            Forwarded to the underlying Flower strategy.

        """
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            fraction_evaluate=fraction_evaluate,
            **kwargs,
        )

    def configure_fit(self, server_round, parameters, client_manager):
        """Create fit configurations for selected clients.

        Parameters
        ----------
        server_round : int
            Current federated training round.
        parameters : list
            Model parameters broadcast to clients.
        client_manager : Flower client manager
            Manager used to sample available clients.

        Returns
        -------
        list
            A list of tuples ``(client, FitIns)`` describing which clients should
            run local training along with the associated instructions.

        """
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


class Server:
    """Lightweight process manager for federated simulation helpers.

    ``Server`` provides simple helpers to start background processes used in
    simulations and to perform a clean shutdown of those processes (and Ray).
    It is intended to be subclassed by concrete server implementations.
    """

    def __init__(self, with_redis=False):
        """Initialize the base Server process manager.

        The Server tracks started background processes and provides helpers
        to start and stop them.
        """
        self.process_fns, self._process_objs = [], []
        if with_redis:
            self.with_redis = True
            self.redis = RedisServer()
            self.redis.start()

    def start_process(self, cmd):
        """Start a background executable process running ``process_fn``."""

        proc = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=set_deathsig,
        )
        self._process_objs.append((proc, {"exec": True}))
        return proc

    def start_process_fn(self, process_fn):
        """Start a background process executing ``process_fn``.

        Parameters
        ----------
        process_fn : callable
            Function

        Returns
        -------
        multiprocessing.Process
            The started process object.

        """
        proc = multiprocessing.Process(target=process_fn)
        proc.daemon = True
        proc.start()
        self._process_objs.append((proc, {"exec": False}))
        return proc

    def run(self, fn, *args, **kwargs):
        """Run a function in a blocking manner."""
        return fn(*args, **kwargs)

    def stop(self):
        """Shutdown Ray and terminate any started background processes."""
        for proc, info in self._process_objs:
            proc.terminate()
            if not info["exec"]:
                proc.join()
            else:
                proc.wait()
        if self.with_redis:
            self.redis.stop()

    def serve(self, *args, **kwargs):
        """Serve method to be implemented by subclasses.

        Subclasses should implement this method to start the federated
        learning simulation or server loop.
        """
        raise NotImplementedError("Serve method not implemented.")


class FLWRServer(Server):
    """Federated learning server using Flower's simulation helpers.

    This class wraps the configuration and lifetime management for a Flower
    simulation run using Ray. It prepares an actor template used by clients,
    constructs a per-run client factory, and starts the simulation.
    """

    def __init__(self, num_clients, num_rounds, strategy=FedAvg):
        """Create an FLWRServer configured for a local simulation.

        Parameters
        ----------
        num_clients : int
            Number of simulated clients.
        num_rounds : int
            Number of federated training rounds.
        strategy : Strategy class, optional
            Flower strategy to use (default: FedAvg wrapper).

        """
        super().__init__(with_redis=True)
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.strategy = strategy(
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
            fraction_evaluate=0.0,
        )

    def generate_actor(self, env_class, actor_class, layer_size):
        """Instantiate a template actor used by client processes.

        Parameters
        ----------
        env_class : type
            Environment class used to construct a vectorized env for shape
            inference.
        actor_class : type
            Actor network class to instantiate.
        layer_size : int
            Hidden layer size passed to the actor constructor.

        """
        envs = gym.vector.SyncVectorEnv(
            [make_env(functools.partial(env_class, cid=-1))]
        )
        self.actor = actor_class(envs, layer_size)

    def set_client(self, *args, **kwargs):
        """Create a Flower client factory bound to the generated actor.

        The method requires that :meth:`generate_actor` has been called first.
        The generated client factory is stored on ``self.client_fn`` for use
        by :meth:`serve`.
        """
        if not hasattr(self, "actor") or self.actor is None:
            raise RuntimeError(
                "Actor not set. Call generate_actor(...) before set_client(...)."
            )
        kwargs["actor"] = self.actor
        self.client_fn = generate_client_fn(*args, **kwargs)

    def serve(self, cpus_per_client=3, gpus_per_client=0):
        """Start the Flower simulation using Ray.

        Parameters
        ----------
        cpus_per_client : int, optional
            Number of CPU cores to reserve for each simulated client.
        gpus_per_client : int, optional
            Number of GPUs to reserve for each simulated client.

        """
        total_cpus = self.num_clients * cpus_per_client + 1
        total_gpus = gpus_per_client
        ray_init_args = {
            "num_cpus": total_cpus,
            "num_gpus": total_gpus,
            "include_dashboard": False,
        }

        fl.simulation.start_simulation(
            client_fn=self.client_fn,
            num_clients=self.num_clients,
            strategy=self.strategy,
            ray_init_args=ray_init_args,
            client_resources={"num_cpus": cpus_per_client, "num_gpus": gpus_per_client},
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
        )

    def stop(self):
        """Stop the server and clean up resources (Ray, Redis, processes)."""
        ray.shutdown()
        super().stop()
