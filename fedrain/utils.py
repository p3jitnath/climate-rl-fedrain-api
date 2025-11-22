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
import subprocess
import sys
import time

import gymnasium as gym
import numpy as np
import torch


def setup_logger(name, level=logging.INFO):
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

    Returns
    -------
    logging.Logger
        Configured logger instance.

    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
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


class RedisServer:
    """Manage a local ``redis-server`` process for testing and simulations.

    ``RedisServer`` provides a minimal API to start and stop a local
    ``redis-server`` subprocess (optionally with RedisAI loaded). It sets the
    ``SSDB`` environment variable to the started server's address so other
    components can connect.

    Parameters
    ----------
    redis_port : int, optional
        TCP port to bind the redis-server process (default: 6379).

    """

    def __init__(self, redis_port=6379):
        """Initialize the RedisServer manager.

        Parameters
        ----------
        redis_port : int, optional
            TCP port to bind the redis-server process (default: 6379).

        """
        self.redis_port = redis_port
        self.redis_proc = None

    def _instantiate_redis_server(self, redis_port=6379):
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
            str(redis_port),
            "--loadmodule",
            os.path.expanduser("~/redisai/redisai.so"),
        ]

        redis_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=set_deathsig,
        )

        for _ in range(50):
            try:
                with socket.create_connection(("127.0.0.1", redis_port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            redis_proc.terminate()
            raise RuntimeError("Failed to start redis-server on localhost")

        redis_address = f"127.0.0.1:{redis_port}"
        os.environ["SSDB"] = redis_address

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

    def start(self):
        """Start a local redis-server and record its process handle."""
        self.redis_proc = self._instantiate_redis_server(self.redis_port)

    def stop(self):
        """Stop the managed redis-server and clear the internal handle."""
        self._shutdown_redis_server(self.redis_proc)
        self.redis_proc = None
