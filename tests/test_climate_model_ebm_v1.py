import glob
import logging

import gymnasium as gym
import numpy as np

from examples.climate_models.ebm_v1 import EnergyBalanceModelEnv
from fedrain.api import FedRAIN
from fedrain.utils import make_env, set_seed
from tests.utils import retrieve_tfrecord_data

EPISODES = 10
NUM_STEPS = 200
TOTAL_TIMESTEPS = NUM_STEPS * EPISODES

EXP_ID = "ebm-v1-optim-L-20k"
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

test_data = retrieve_tfrecord_data(
    "ddpg", glob.glob(f"tests/runs/{EXP_ID}_*/*_ddpg_*/*tfevents*")[0]
)


def test_ebm_v1_episodic_return_matches_expected():

    set_seed(SEED)
    envs = gym.vector.SyncVectorEnv([make_env(EnergyBalanceModelEnv, SEED, NUM_STEPS)])

    params = CONFIG.copy()
    ac_size = params.pop("actor_critic_layer_size", None)
    params["actor_layer_size"] = params["critic_layer_size"] = ac_size

    api = FedRAIN()
    agent = api.set_algorithm(
        "DDPG", envs=envs, seed=SEED, **params, level=logging.DEBUG
    )

    obs, _ = envs.reset()
    episodic_returns = []
    for t in range(1, EPISODES * NUM_STEPS + 1):
        actions = agent.predict(obs)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        agent.update(actions, next_obs, rewards, terminations, truncations, infos)

        if "final_info" in infos:
            for info in infos["final_info"]:
                episode_return = info["episode"]["r"]
                if episode_return is not None:
                    episodic_returns.append(episode_return[0])
                break

        obs = next_obs

    assert episodic_returns, "No episodic return was recorded during the run"
    last_return = episodic_returns[-1]

    expected = test_data[1][EPISODES - 1]["episodic_return"]
    assert np.isclose(
        last_return, expected, atol=1e-8, rtol=1e-6
    ), f"episodic_return {last_return} != expected {expected}"
