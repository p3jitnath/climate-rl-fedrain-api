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
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def make_env(env_class, seed=1, NUM_STEPS=200):
    def thunk():
        env = env_class()
        env = gym.wrappers.TimeLimit(env, max_episode_steps=NUM_STEPS)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def instantiate_redis_server(redis_port=6379):
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

    def set_deathsig():
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)

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
