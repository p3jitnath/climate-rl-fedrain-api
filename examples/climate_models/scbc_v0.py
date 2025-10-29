"""Simple Climate Bias Correction example environment.

This module provides a toy Gym environment ``SimpleClimateBiasCorrectionEnv``
that simulates a single scalar temperature variable with a simple
physics-based relaxation and a bias-correction term. The environment is
intentionally minimal and suitable for demos, examples and federated RL
workflows used by the project.

The file also exposes ``run_scbc`` which creates a vectorized environment
and trains a DDPG agent via the project's ``FedRAIN`` API.
"""

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from fedrain.api import FedRAIN
from fedrain.utils import make_env, set_seed

NUM_STEPS = 200
TOTAL_TIMESTEPS = 2000
ACTOR_LAYER_SIZE, CRITIC_LAYER_SIZE = 64, 64


class SimpleClimateBiasCorrectionEnv(gym.Env):
    """A minimal environment simulating bias-corrected scalar temperature.

    This toy environment exposes a single temperature observation and a
    continuous heating-rate action. It is intended for demonstrations and
    small integration tests.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None):
        """Create the simple bias-correction environment.

        Parameters
        ----------
        render_mode : str or None
            If ``'human'`` the environment will create a visible Pygame
            window when rendering. If ``'rgb_array'`` rendering returns an
            RGB numpy array. If ``None`` rendering is disabled.

        """
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
        """Apply a heating-rate action and advance the environment one step.

        The dynamics are a simple Euler-like update: the agent's action
        ``u`` (heating rate) is applied, then the state relaxes partially
        toward a physics temperature and a small bias-correction term is
        applied to nudge the state toward an observed temperature. The
        reward returned is the negative squared bias correction (so the
        agent is incentivized to reduce bias).

        Parameters
        ----------
        u : array-like or float
            Action specifying the heating rate. The environment expects a
            single-element array-like (shape (1,)) and clips it to the
            allowed action bounds.

        Returns
        -------
        obs, reward, terminated, truncated, info
            Observation (1-D numpy array), scalar reward, termination and
            truncation flags (always False in this simple env), and an
            info dict (currently empty placeholder).

        """
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
        """Reset the environment to an initial temperature.

        Parameters
        ----------
        seed : int or None
            Seed forwarded to :meth:`gym.Env.reset` for deterministic
            behaviours in vectorized setups.
        options : dict or None
            Reset options (present for API compatibility, currently
            unused).

        Returns
        -------
        observation, info
            The initial observation (1-D ``float32`` array) and an info
            dictionary following the Gymnasium API.

        """
        super().reset(seed=seed)
        self.state = np.array([(300 - 273.15) / 100])
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), self._get_info()

    def _get_info(self):
        """Return the info dictionary for the current timestep.

        The environment does not currently populate diagnostic information
        so this returns a placeholder dictionary to satisfy the Gym API.
        """
        return {"_": None}

    def _get_obs(self):
        """Return the current observation (temperature) as a numpy array.

        Returns
        -------
        numpy.ndarray
            1-D float32 array containing the current temperature normalized
            to the environment's range.

        """
        temperature = self.state[0]
        return np.array([temperature], dtype=np.float32)

    def render(self):
        """Render the environment.

        For ``render_mode=='rgb_array'`` this returns an RGB image array.
        For ``render_mode=='human'`` rendering is handled in
        ``_render_frame`` (the window is updated there). If rendering is
        disabled this method is a no-op.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """Create or update the Pygame drawing for the thermometer.

        Returns
        -------
        numpy.ndarray or None
            When in ``'rgb_array'`` mode returns an (H, W, 3) uint8 RGB image
            representing the current frame. In ``'human'`` mode the frame
            is drawn to a live window and ``None`` is returned.

        """
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
        """Close any open rendering windows and release resources.

        Safe to call multiple times; after closing the Pygame display the
        environment will no longer render until a new display/surface is
        created on the next call to ``_render_frame``.
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None


def run_scbc(seed):
    """Run a simple training loop using the project's FedRAIN API.

    This helper creates a vectorized environment, sets up a DDPG agent via
    :class:`fedrain.api.FedRAIN`, and runs a fixed number of training
    timesteps. It is intended for quick demos and integration tests.

    Parameters
    ----------
    seed : int
        Random seed used to initialize environments and the agent.

    """
    set_seed(seed)
    envs = gym.vector.SyncVectorEnv(
        [make_env(SimpleClimateBiasCorrectionEnv, seed, NUM_STEPS)]
    )
    api = FedRAIN()
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
    run_scbc(seed)
