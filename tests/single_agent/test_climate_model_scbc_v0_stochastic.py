"""Single-agent stochastic integration test for the SCBC v0 climate model.

This test runs a small, deterministic agent-environment loop using the
``SimpleClimateBiasCorrectionEnv`` and compares the first ten episodic
returns against precomputed values from a seeded stochastic rollout.
"""

import logging
from unittest.mock import Mock

import gymnasium as gym
import numpy as np

from examples.climate_models.scbc_v0 import SimpleClimateBiasCorrectionEnv
from fedrain.api import FedRAIN
from fedrain.utils import make_env, set_seed

EPISODES = 10
NUM_STEPS = 200
TOTAL_TIMESTEPS = NUM_STEPS * EPISODES

SEED = 1

EXPECTED_EPISODIC_RETURNS = np.array(
    [
        -2.6306646,
        -2.5909991,
        -2.9344237,
        -3.1993594,
        -2.9337006,
        -3.1504486,
        -2.8909876,
        -2.7250060,
        -2.4942086,
        -2.7677166,
    ],
    dtype=np.float32,
)


def test_scbc_stochastic_episodic_returns_match_expected(monkeypatch):
    """Run the SCBC v0 loop and assert all episodic returns equal expected.

    The test seeds RNGs to ensure determinism, constructs a vectorized
    environment thunk, instantiates a stochastic agent via the public API,
    and runs the action-update loop for a fixed number of timesteps. All ten
    recorded episodic returns are compared bit for bit against precomputed
    values from the seeded rollout.
    """
    monkeypatch.setattr("fedrain.algorithms.stochastic.Recorder", Mock)
    set_seed(SEED)
    envs = gym.vector.SyncVectorEnv(
        [make_env(SimpleClimateBiasCorrectionEnv, SEED, NUM_STEPS)]
    )

    api = FedRAIN()
    agent = api.set_algorithm(
        "STOCHASTIC",
        envs=envs,
        seed=SEED,
        level=logging.INFO,
    )

    obs, _ = envs.reset()
    episodic_returns = []
    for t in range(1, TOTAL_TIMESTEPS + 1):
        actions = agent.predict(obs)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        agent.update(actions, next_obs, rewards, terminations, truncations, infos)

        if "final_info" in infos:
            for info in infos["final_info"]:
                episode_return = info["episode"]["r"][0]
                if episode_return is not None:
                    episodic_returns.append(episode_return)
                break

        obs = next_obs

    envs.close()
    assert episodic_returns, "No episodic return was recorded during the run"
    np.testing.assert_array_equal(
        np.asarray(episodic_returns),
        EXPECTED_EPISODIC_RETURNS,
    )
