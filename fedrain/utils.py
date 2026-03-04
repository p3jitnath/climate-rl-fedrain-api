"""Utility helpers used across the FedRAIN project.

This module provides small, test-friendly utilities used by the FedRAIN
codebase such as logging setup, environment factory creation, seeding of
random number generators, and a lightweight Redis server manager used for
local simulations.

Functions
---------
setup_logger
    Create or retrieve a configured :class:`logging.Logger` instance.
make_env
    Return a thunk that creates a wrapped Gym environment instance.
set_seed
    Seed Python, NumPy and PyTorch RNGs for reproducibility.

Classes
-------
RedisServer
    Manage a local redis-server process used for development/testing.
"""

import logging
import os
import random
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time

import gymnasium as gym
import numpy as np
import psutil
import torch


def setup_logger(name, level=logging.INFO, only_rank0=True):
    """Create or return a configured logger.

    This helper creates a :class:`logging.Logger` with a stdout stream
    handler and a compact formatter when no handlers are attached yet. The
    function is idempotent: calling it multiple times for the same ``name``
    will return the same logger instance.

    Parameters
    ----------
    name : str
        Name of the logger.
    level : int, optional
        Logging level to apply to the logger (default: ``logging.INFO``).
    only_rank0 : bool, optional
        If True, the logger will output to stdout on rank 0 and be silent on
        other ranks. If False, the logger will output to stdout on all ranks (default: True).

    Returns
    -------
    logging.Logger
        Configured logger instance.

    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():

        # try to detect MPI rank; if unavailable assume rank 0.
        try:
            from mpi4py import MPI

            rank = MPI.COMM_WORLD.Get_rank()
        except Exception:
            rank = 0

        formatter = logging.Formatter(
            "[%(rank)03d, %(levelname)-5s] %(name)s: %(message)s"
        )

        if only_rank0 and rank != 0:
            logger.addHandler(logging.NullHandler())
        else:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger = logging.LoggerAdapter(logger, {"rank": rank})

    logger.setLevel(level)
    return logger


def make_env(env_class, seed=1, NUM_STEPS=200):
    """Create a thunk that builds a wrapped Gym environment.

    The returned function, when called, constructs an instance of
    ``env_class``, wraps it with a time limit and an episode statistics
    recorder, seeds its action space, and returns the environment. This
    pattern is convenient for vectorized environment constructors that
    expect a zero-argument callable.

    Parameters
    ----------
    env_class : type
        Callable/class that returns a Gym environment when invoked.
    seed : int, optional
        Seed used to seed the environment's action space (default: 1).
    NUM_STEPS : int, optional
        Maximum number of steps per episode enforced by ``TimeLimit``
        (default: 200).

    Returns
    -------
    callable
        A zero-argument function that constructs and returns the wrapped
        environment.

    """

    def thunk():
        env = env_class()
        env = gym.wrappers.TimeLimit(env, max_episode_steps=NUM_STEPS)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def set_seed(seed):
    """Seed Python, NumPy and PyTorch random number generators.

    This helper seeds the standard ``random`` module, NumPy and PyTorch to
    help make experiments reproducible. It also sets PyTorch's CuDNN
    deterministic flag for additional determinism on CUDA if available.

    Parameters
    ----------
    seed : int
        Seed value to apply.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_deathsig():
    """Set the parent-death signal in the child process.

    This ensures the spawned redis-server receives SIGTERM when the
    parent process dies, helping to avoid orphaned processes.
    """
    import ctypes

    libc = ctypes.CDLL("libc.so.6")
    PR_SET_PDEATHSIG = 1
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


def get_ip_from_interface(interface="lo"):
    """
    Get the IPv4 address for a given network interface.

    Parameters
    ----------
    interface : str, optional
        Name of the network interface to query (default 'lo').

    Returns
    -------
    str
        IPv4 address assigned to the specified interface.

    Raises
    ------
    ValueError
        If the interface is not present in psutil.net_if_addrs() or if it has no IPv4 address.

    Notes
    -----
    This function inspects psutil.net_if_addrs() and returns the first address whose
    family is socket.AF_INET.
    """
    available = psutil.net_if_addrs()
    if interface not in available:
        raise ValueError(
            f"{interface} is not a valid network interface. "
            f"Valid network interfaces are: {list(available.keys())}"
        )

    for info in available[interface]:
        if info.family == socket.AF_INET:
            return info.address
    raise ValueError(f"interface {interface} doesn't have an IPv4 address")


def get_urandom_redis_port():
    """
    Return a pseudorandom TCP port number intended for Redis.

    This function:
    - Reads 4 bytes from /dev/urandom and interprets them as an unsigned integer.
    - Chooses a base port uniformly at random from 12581 to 24580 (inclusive).
    - Adds (hrand % 1000) to the base port to produce the final port.

    Returns:
        int: A candidate port number in the inclusive range 12581..25579.

    Notes:
        - The function does not check whether the returned port is available; callers
          must verify and handle port collisions.
        - The final distribution is not purely uniform due to the additive modulo step.
    """
    with open("/dev/urandom", "rb") as f:
        hrand_bytes = f.read(4)
    hrand = struct.unpack("I", hrand_bytes)[0]
    redis_port = random.randint(12581, 24580)
    redis_port += hrand % 1000
    return redis_port


def get_ssdb_redis_port():
    """
    Return the Redis port extracted from the SSDB environment variable.

    Reads the SSDB environment variable, which is expected to be either a port
    ("6379") or a host:port string ("hostname:6379"), and returns the port as an int.

    Returns:
        int: Redis port number.

    Raises:
        EnvironmentError: If the SSDB environment variable is not set.
    """
    redis_port = os.getenv("SSDB")
    if not redis_port:
        raise EnvironmentError("The environment variable $SSDB is not set.")
    return int(redis_port.split(":")[-1])


class RedisServer:
    """Manage a local ``redis-server`` process for testing and simulations.

    ``RedisServer`` provides a minimal API to start and stop a local
    ``redis-server`` subprocess (optionally with RedisAI loaded). It sets the
    ``SSDB`` environment variable to the started server's address so other
    components can connect.

    Parameters
    ----------
    redis_address : str, optional
        Address to bind the redis-server process (default: "localhost")

    """

    def __init__(self, redis_port=6379):
        """Initialize the RedisServer manager.

        Parameters
        ----------
        redis_port : int, optional
            Port to bind the redis-server process (default: 6379).

        """
        self.redis_port = redis_port
        self.redis_proc = None

    def _instantiate_redis_server(self, interface, rediscli_config):
        redis_bin = shutil.which("redis-server")
        if redis_bin is None:
            raise RuntimeError(
                "redis-server binary not found on PATH. Please install Redis."
            )

        rdb_path = os.path.abspath("./dump.rdb")
        if os.path.exists(rdb_path):
            try:
                os.remove(rdb_path)
            except OSError:
                pass

        cmd = [
            redis_bin,
            "--port",
            str(self.redis_port),
            "--loadmodule",
            os.path.expanduser("~/redisai/redisai.so"),
        ]

        # Append additional CLI flags from rediscli_config as: --flag value
        for k, v in rediscli_config.items():
            flag = f"--{k}"
            cmd.extend([flag, str(v)])

        redis_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=set_deathsig,
        )

        for _ in range(50):
            try:
                with socket.create_connection(
                    ("localhost", self.redis_port), timeout=0.5
                ):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            redis_proc.terminate()
            raise RuntimeError(
                f"Failed to start redis-server on localhost:{self.redis_port}"
            )

        self.redis_address = get_ip_from_interface(interface) + f":{self.redis_port}"
        os.environ["SSDB"] = self.redis_address

        return redis_proc

    def _shutdown_redis_server(self, redis_proc):
        """Terminate the given redis subprocess if it exists.

        Parameters
        ----------
        redis_proc : subprocess.Popen or None
            Process object returned by :meth:`_instantiate_redis_server`.

        """
        if redis_proc is not None:
            redis_proc.terminate()
            redis_proc.wait()

    def start(self, interface="lo", rediscli_config={}):
        """Start a local redis-server and record its process handle."""
        self.redis_proc = self._instantiate_redis_server(interface, rediscli_config)

    def stop(self):
        """Stop the managed redis-server and clear the internal handle."""
        self._shutdown_redis_server(self.redis_proc)
        self.redis_proc = None
