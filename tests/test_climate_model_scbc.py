import random
from glob import glob

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from fedrain.api import FedRAIN
from tests.utils import retrieve_tfrecord_data

EPISODES = 10
MAX_EPISODE_STEPS = 200
TOTAL_TIMESTEPS = 2000

EXP_ID = "scbc-v0-optim-L-60k"
SEED = 1

CONFIG = {
    "learning_rate": 0.0046327801811340335,
    "tau": 0.07340809018042468,
    "batch_size": 128,
    "exploration_noise": 0.10076614958209602,
    "policy_frequency": 10,
    "noise_clip": 0.1,
    "actor_critic_layer_size": 128,
}

test_data = retrieve_tfrecord_data(
    "ddpg", glob(f"tests/runs/{EXP_ID}*/*/*tfevents*")[0]
)


class SimpleClimateBiasCorrectionEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None):
        self.min_temperature = 0.0
        self.max_temperature = 1.0
        self.min_heating_rate = -1.0
        self.max_heating_rate = 1.0
        self.dt = 1.0
        self.count = 0.0
        self.screen = None
        self.clock = None
        self.action_space = spaces.Box(
            low=self.min_heating_rate,
            high=self.max_heating_rate,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([self.min_temperature], dtype=np.float32),
            high=np.array([self.max_temperature], dtype=np.float32),
            dtype=np.float32,
        )
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def step(self, u):
        current_temperature = self.state[0]
        u = np.clip(u, -self.max_heating_rate, self.max_heating_rate)[0]
        observed_temperature = (321.75 - 273.15) / 100
        physics_temperature = (380 - 273.15) / 100
        division_constant = physics_temperature - observed_temperature
        new_temperature = current_temperature + u
        relaxation = (
            (physics_temperature - current_temperature) * 0.2 / division_constant
        )
        new_temperature += relaxation
        bias_correction = (
            (observed_temperature - new_temperature) * 0.1 / division_constant
        )
        new_temperature += bias_correction
        costs = bias_correction**2
        new_temperature = np.clip(
            new_temperature, self.min_temperature, self.max_temperature
        )
        self.state = np.array([new_temperature])
        return self._get_obs(), -costs, False, False, self._get_info()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([(300 - 273.15) / 100])
        return self._get_obs(), self._get_info()

    def _get_info(self):
        return {"_": None}

    def _get_obs(self):
        temperature = self.state[0]
        return np.array([temperature], dtype=np.float32)


def make_env(env_class, seed, max_episode_steps):
    def thunk():
        env = env_class()
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def test_scbc_episodic_return_matches_expected():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    envs = gym.vector.SyncVectorEnv(
        [make_env(SimpleClimateBiasCorrectionEnv, SEED, MAX_EPISODE_STEPS)]
    )
    api = FedRAIN()

    params = CONFIG.copy()
    ac_size = params.pop("actor_critic_layer_size", None)
    params["actor_layer_size"] = params["critic_layer_size"] = ac_size

    agent = api.set_algorithm(
        "DDPG",
        envs=envs,
        seed=SEED,
        **params,
    )

    obs, _ = envs.reset()
    episodic_returns = []
    for t in range(1, EPISODES * MAX_EPISODE_STEPS + 1):
        actions = agent.predict(obs)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        agent.update(actions, next_obs, rewards, terminations, truncations, infos)

        if "final_info" in infos:
            for info in infos["final_info"]:
                episode_return = info["episode"]["r"]
                if episode_return is not None:
                    episodic_returns.append(episode_return)
                break

        obs = next_obs

    assert episodic_returns, "No episodic return was recorded during the run"
    last_return = episodic_returns[-1][0]

    expected = test_data[1][EPISODES - 1]["episodic_return"]
    assert np.isclose(
        last_return, expected, atol=1e-8, rtol=1e-6
    ), f"episodic_return {last_return} != expected {expected}"
