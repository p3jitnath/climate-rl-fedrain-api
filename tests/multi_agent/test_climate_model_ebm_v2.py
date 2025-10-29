"""Integration tests for the EBM v2 multi-client experiment.

These tests reproduce a short federated simulation using the EBM v2
environment and verify that the episodic returns recorded during the
simulation match precomputed expected values stored in TFRecord test data.

The module provides a small helper ``run_ebm`` used to run the environment
loop in a spawned process and a single test that starts a Flower-based
simulation with multiple clients.
"""

import functools
import glob
import json
import logging
import os
import tempfile

import gymnasium as gym
import numpy as np

from examples.climate_models.ebm_v2 import EnergyBalanceModelEnv
from fedrain.algorithms.ddpg import DDPGActor
from fedrain.api import FedRAIN
from fedrain.fedrl.server import FLWRServer
from fedrain.utils import make_env, set_seed
from tests.utils import retrieve_tfrecord_data

EBM_LATITUDES = 96
NUM_CLIENTS = 2
EBM_SUBLATITUDES = EBM_LATITUDES // NUM_CLIENTS

NUM_STEPS = 200
FLWR_EPISODES = 5
FLWR_ROUNDS = 5
TOTAL_TIMESTEPS = FLWR_EPISODES * FLWR_ROUNDS * NUM_STEPS

EPISODES = 10
EXP_ID = "ebm-v2-optim-L-20k-a2-fed05"
SEED = 1

CONFIG = {
    "learning_rate": 0.0009465344554592341,
    "tau": 0.021580739259456708,
    "batch_size": 128,
    "exploration_noise": 0.21660831219393845,
    "policy_frequency": 2,
    "noise_clip": 0.4,
    "actor_critic_layer_size": 256,
}

ACTOR_LAYER_SIZE = CONFIG["actor_critic_layer_size"]

test_data = [
    retrieve_tfrecord_data("ddpg", x, EPISODES)
    for x in sorted(glob.glob(f"tests/data/runs/{EXP_ID}_*/*_ddpg_*/*tfevents*"))
]


def run_ebm(seed, cid):
    """Run a short EBM training loop in a separate process.

    This function is intended to be started as a background process by the
    Flower client helper. It performs environment interaction for a fixed
    number of timesteps and writes a small JSON file with the episodic
    return for later verification.

    Parameters
    ----------
    seed : int
        Random seed to use for env and algorithm seeding.
    cid : int
        Client identifier used when writing the result file.

    """
    result_path = os.path.join(
        tempfile.gettempdir(), f"ebm_test_result_cid{cid}_{os.getpid()}.json"
    )
    set_seed(seed)
    envs = gym.vector.SyncVectorEnv(
        [make_env(functools.partial(EnergyBalanceModelEnv, cid=cid), seed, NUM_STEPS)]
    )

    params = CONFIG.copy()
    ac_size = params.pop("actor_critic_layer_size", None)
    params["actor_layer_size"] = params["critic_layer_size"] = ac_size
    params["fedRLConfig"] = {
        "cid": cid,
        "num_steps": NUM_STEPS,
        "flwr_episodes": FLWR_EPISODES,
    }

    api = FedRAIN()
    agent = api.set_algorithm(
        "DDPG", envs=envs, seed=SEED, **params, level=logging.DEBUG
    )

    obs, _ = envs.reset()
    episode_return = None
    for t in range(1, TOTAL_TIMESTEPS + 1):
        actions = agent.predict(obs)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        agent.update(actions, next_obs, rewards, terminations, truncations, infos)
        obs = next_obs

        if t == EPISODES * NUM_STEPS:
            if "final_info" in infos:
                for info in infos["final_info"]:
                    episode_return = info["episode"]["r"][0]
            expected = test_data[cid][cid][EPISODES - 1]["episodic_return"]
            with open(result_path, "w") as fn:
                json.dump(
                    {
                        "episode_return": str(episode_return),
                        "expected": str(expected),
                    },
                    fn,
                )


def test_ebm_v2_episodic_return_matches_expected():
    """Run a short federated EBM v2 simulation and compare episodic return.

    This test starts a small Flower-based federated simulation and asserts
    that the recorded episodic return matches the precomputed expected
    value from the TFRecord test data.
    """
    server = FLWRServer(NUM_CLIENTS, 3)
    server.generate_actor(EnergyBalanceModelEnv, DDPGActor, ACTOR_LAYER_SIZE)
    server.set_client(seed=SEED, fn=run_ebm, num_steps=NUM_STEPS)
    server.serve()
    server.stop()

    for cid in range(NUM_CLIENTS):
        pattern = os.path.join(
            tempfile.gettempdir(), f"ebm_test_result_cid{cid}_*.json"
        )
        matches = sorted(glob.glob(pattern))
        assert matches, f"No result file found for cid {cid} (pattern {pattern})"

        path = matches[-1]
        with open(path, "r") as fn:
            res = json.load(fn)
            os.remove(path)

        episode_return, expected = np.float32(res.get("episode_return")), np.float32(
            res.get("expected")
        )
        assert np.isclose(
            episode_return, expected, atol=1e-8, rtol=1e-6
        ), f"episodic_return {episode_return} != expected {expected}"
