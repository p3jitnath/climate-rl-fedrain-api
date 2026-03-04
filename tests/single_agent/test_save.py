"""Unit test for DDPG checkpoint save/load.

This test follows the project's single-agent test style: it seeds
randomness for determinism, constructs a tiny vectorized environment
thunk, instantiates a DDPG agent via the public API, performs a small
interaction/update step, saves a checkpoint to `ckpt_path` (created via
the `tmp_path` fixture) and verifies that weights and metadata are
restored correctly into a fresh agent.
"""

import json
import os

import gymnasium as gym
import pytest
import torch

from examples.climate_models.scbc_v0 import SimpleClimateBiasCorrectionEnv
from fedrain.api import FedRAIN
from fedrain.utils import make_env, set_seed


# create checkpoint path fixture before the test function per project pattern
@pytest.fixture
def ckpt_path(tmp_path):
    return tmp_path / "ckpt"


SEED = 1
NUM_STEPS = 200

CONFIG = {
    "learning_rate": 0.0046327801811340335,
    "tau": 0.07340809018042468,
    "batch_size": 128,
    "exploration_noise": 0.10076614958209602,
    "policy_frequency": 10,
    "noise_clip": 0.1,
    "actor_critic_layer_size": 128,
}


def test_ddpg_save_and_load(ckpt_path):
    """Save a DDPG checkpoint and load it into a new agent.

    Asserts that actor parameters match after loading and that
    `metadata.json` contains expected values.
    """
    set_seed(SEED)
    envs = gym.vector.SyncVectorEnv(
        [make_env(SimpleClimateBiasCorrectionEnv, SEED, NUM_STEPS)]
    )

    params = CONFIG.copy()
    ac_size = params.pop("actor_critic_layer_size", None)
    params["actor_layer_size"] = params["critic_layer_size"] = ac_size

    api = FedRAIN()
    agent = api.set_algorithm("DDPG", envs=envs, seed=SEED, **params)

    obs, _ = envs.reset()
    actions = agent.predict(obs)
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)
    agent.update(actions, next_obs, rewards, terminations, truncations, infos)

    ckpt_path_str = api.save_weights(agent, folder=str(ckpt_path))
    assert os.path.isdir(ckpt_path_str)
    assert os.path.exists(os.path.join(ckpt_path_str, "actor.pt"))
    assert os.path.exists(os.path.join(ckpt_path_str, "replay_buffer.pt"))
    assert os.path.exists(os.path.join(ckpt_path_str, "metadata.json"))

    # create a fresh agent and load
    agent2 = api.set_algorithm("DDPG", envs=envs, seed=SEED, **params)
    api.load_weights(agent2, ckpt_path_str)

    # actor parameters should match after load
    for p1, p2 in zip(agent.actor.parameters(), agent2.actor.parameters()):
        assert torch.allclose(p1.detach(), p2.detach())

    # metadata should have been restored
    with open(os.path.join(ckpt_path_str, "metadata.json"), "r") as fh:
        meta = json.load(fh)
    assert int(meta.get("seed", -1)) == SEED
    assert int(meta.get("global_step", -1)) == agent.global_step
