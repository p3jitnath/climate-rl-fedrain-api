import random

import gymnasium as gym
import numpy as np
import pygame
import torch
from gymnasium import spaces

from fedrain.api import FedRAINAPI

MAX_EPISODE_STEPS = 200
TOTAL_TIMESTEPS = 2000
ACTOR_LAYER_SIZE, CRITIC_LAYER_SIZE = 64, 64


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
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), self._get_info()

    def _get_info(self):
        return {"_": None}

    def _get_obs(self):
        temperature = self.state[0]
        return np.array([temperature], dtype=np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        screen_width = 600
        screen_height = 400
        thermometer_height = 300
        thermometer_width = 50
        mercury_width = 30
        base_height = 10
        temp_range = self.max_temperature - self.min_temperature
        mercury_height = (
            (self.state[0] - self.min_temperature) / temp_range
        ) * thermometer_height
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                self.screen = pygame.Surface((screen_width, screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 16)
        self.screen.fill((255, 255, 255))
        thermometer_rect = pygame.Rect(
            (screen_width / 2) - (thermometer_width / 2),
            (screen_height / 2) - (thermometer_height / 2),
            thermometer_width,
            thermometer_height,
        )
        pygame.draw.rect(self.screen, (200, 200, 200), thermometer_rect)
        mercury_rect = pygame.Rect(
            (screen_width / 2) - (mercury_width / 2),
            (screen_height / 2) + (thermometer_height / 2) - mercury_height,
            mercury_width,
            mercury_height,
        )
        pygame.draw.rect(self.screen, (255, 0, 0), mercury_rect)
        base_rect = pygame.Rect(
            (screen_width / 2) - (thermometer_width / 2),
            (screen_height / 2) + (thermometer_height / 2),
            thermometer_width,
            base_height,
        )
        pygame.draw.rect(self.screen, (150, 150, 150), base_rect)
        observed_ratio = (321.75 - 273.15) / (380 - 273.15)
        observed_mark_y = (screen_height / 2) + (thermometer_height / 2)
        observed_mark_y -= thermometer_height * observed_ratio
        observed_mark_start = (screen_width / 2) - (thermometer_width / 2)
        observed_mark_end = (screen_width / 2) + (thermometer_width / 2)
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (observed_mark_start, observed_mark_y),
            (observed_mark_end, observed_mark_y),
            5,
        )
        min_temp_k = 273.15
        max_temp_k = 380
        temp_range_k = max_temp_k - min_temp_k
        marking_spacing_k = 20
        for temp_k in range(int(min_temp_k), int(max_temp_k) + 1, marking_spacing_k):
            normalized_temp = (temp_k - min_temp_k) / temp_range_k
            mark_y = (
                (screen_height / 2)
                + (thermometer_height / 2)
                - (normalized_temp * thermometer_height)
            )
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                ((screen_width / 2) - (thermometer_width / 2) - 10, mark_y),
                ((screen_width / 2) - (thermometer_width / 2), mark_y),
                2,
            )
            temp_text = self.font.render(f"{temp_k} K", True, (0, 0, 0))
            self.screen.blit(
                temp_text,
                (
                    (screen_width / 2) - (thermometer_width / 2) - 60,
                    mark_y - 10,
                ),
            )
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen).transpose([1, 0, 2])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None


def make_env(env_class, seed, max_episode_steps):
    def thunk():
        env = env_class()
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def run_scbc(seed):
    envs = gym.vector.SyncVectorEnv(
        [make_env(SimpleClimateBiasCorrectionEnv, seed, MAX_EPISODE_STEPS)]
    )
    api = FedRAINAPI()
    agent = api.set_algorithm(
        "DDPG",
        envs=envs,
        seed=seed,
        actor_layer_size=ACTOR_LAYER_SIZE,
        critic_layer_size=CRITIC_LAYER_SIZE,
    )

    obs, _ = envs.reset()
    for t in range(1, TOTAL_TIMESTEPS + 1):
        actions = agent.predict(obs)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        agent.update(actions, next_obs, rewards, terminations, truncations, infos)
        obs = next_obs


if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    run_scbc(seed=1)
