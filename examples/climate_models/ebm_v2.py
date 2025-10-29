"""Multi-client EBM example (version 2).

This module demonstrates a federated setup where different partitions of
the latitude band are controlled by separate clients. It provides an
``EnergyBalanceModelEnv`` class that accepts a ``cid`` argument to select
the sub-latitude region the client controls. The module also contains a
``run_ebm`` helper that is suitable for running in a background process
paired with the federated Flower server used in tests and examples.

Input observations are the entire global temperature field.
Rewards are computed based on the mean squared error between the
model's temperature field and observed climatological values ONLY over
the specified sub-latitude region.

"""

import functools
import logging
import os

import climlab
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from gymnasium import spaces
from matplotlib.gridspec import GridSpec
from smartredis import Client

from fedrain.algorithms.ddpg import DDPGActor
from fedrain.api import FedRAIN
from fedrain.fedrl.server import FLWRServer
from fedrain.utils import make_env, set_seed, setup_logger

EBM_LATITUDES = 96
NUM_CLIENTS = 2

EBM_SUBLATITUDES = EBM_LATITUDES // NUM_CLIENTS

NUM_STEPS = 200
FLWR_EPISODES = 5
FLWR_ROUNDS = 5
TOTAL_TIMESTEPS = FLWR_EPISODES * FLWR_ROUNDS * NUM_STEPS
ACTOR_LAYER_SIZE, CRITIC_LAYER_SIZE = 64, 64


class EBMUtils:
    """Utility helpers and cached climatologies used by the EBM examples.

    This container exposes file paths, a dataset download helper and
    precomputed climatological arrays used by the environment instances.
    """

    BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"
    DATASETS_DIR = f"{BASE_DIR}/datasets"

    fp_Ts = f"{DATASETS_DIR}/skt.sfc.mon.1981-2010.ltm.nc"
    ncep_url = (
        "http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/"
    )

    def download_and_save_dataset(url, filepath, dataset_name):
        """Download (or load) and return a named dataset.

        Parameters
        ----------
        url : str
            Remote URL of the dataset.
        filepath : str
            Local path to cache the dataset.
        dataset_name : str
            Human-readable dataset name used for logging.

        Returns
        -------
        xarray.Dataset
            The loaded dataset object.

        """
        logger = setup_logger("DATASET", logging.DEBUG)
        if not os.path.exists(filepath):
            logger.debug(f"Downloading {dataset_name} data ...")
            dataset = xr.open_dataset(url, decode_times=False)
            dataset.to_netcdf(filepath, format="NETCDF3_64BIT")
            logger.debug(f"{dataset_name} data saved to {filepath}")
        else:
            logger.debug(f"Loading {dataset_name} data ...")
            dataset = xr.open_dataset(
                filepath,
                decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
            )
        return dataset

    ncep_Ts = download_and_save_dataset(
        ncep_url + "surface_gauss/skt.sfc.mon.1981-2010.ltm.nc",
        fp_Ts,
        "NCEP surface temperature",
    ).sortby("lat")

    lat_ncep = ncep_Ts.lat
    lon_ncep = ncep_Ts.lon
    Ts_ncep_annual = ncep_Ts.skt.mean(dim=("lon", "time"))

    a0_ref = 0.354
    a2_ref = 0.25
    D_ref = 0.6
    A_ref = 2.1
    B_ref = 2


class EnergyBalanceModelEnv(gym.Env):
    """Gym environment exposing a sub-latitude partition of the EBM.

    Each environment instance controls a contiguous sub-latitude band
    selected by ``cid`` and communicates with a central EBM server via
    SmartRedis tensors. Observations are the temperature field (global in
    v2) and rewards are computed as the negative MSE against a
    climatological reference over the local band.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, cid=None, render_mode=None, level=logging.DEBUG):
        """Create a federated EBM environment for the given client id.

        Parameters
        ----------
        cid : int or None
            Client identifier selecting the sub-latitude region this
            environment controls.
        render_mode : str or None
            If provided, controls rendering mode (``'human'`` or
            ``'rgb_array'``).
        level : int
            Logging level (passed to the internal logger).

        """
        self.utils = EBMUtils()
        self.cid = cid

        self.logger = setup_logger(f"EBM {self.cid}", level)

        self.logger.debug(f"Environment ID: {self.cid}")
        self.logger.debug(f"Number of clients: {NUM_CLIENTS}")

        self.min_D = 0.55
        self.max_D = 0.65

        self.min_A = 1.4
        self.max_A = 4.2

        self.min_B = 1.95
        self.max_B = 2.05

        self.min_a0 = 0.3
        self.max_a0 = 0.4

        self.min_a2 = 0.2
        self.max_a2 = 0.3

        self.min_temperature = -90
        self.max_temperature = 90

        self.action_space = spaces.Box(
            low=np.array(
                [
                    self.min_D,
                    *[self.min_A for x in range(EBM_LATITUDES)],
                    *[self.min_B for x in range(EBM_LATITUDES)],
                    self.min_a0,
                    self.min_a2,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.max_D,
                    *[self.max_A for x in range(EBM_LATITUDES)],
                    *[self.max_B for x in range(EBM_LATITUDES)],
                    self.max_a0,
                    self.max_a2,
                ],
                dtype=np.float32,
            ),
            shape=(2 * EBM_LATITUDES + 3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=self.min_temperature,
            high=self.max_temperature,
            shape=(EBM_LATITUDES,),
            dtype=np.float32,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode

        self.REDIS_ADDRESS = os.getenv("SSDB")
        if self.REDIS_ADDRESS is None:
            raise EnvironmentError("SSDB environment variable is not set.")
        self.redis = Client(address=self.REDIS_ADDRESS, cluster=False)
        self.logger.debug(f"Connected to Redis server: {self.REDIS_ADDRESS}")

        self.redis.put_tensor(f"SIGALIVE_S{self.cid}", np.array([1], dtype=np.int32))

    def _get_obs(self):
        """Return the current observation for the environment.

        Returns
        -------
        numpy.ndarray
            1-D float32 array containing the current temperature field for
            all latitudes (shape ``(EBM_LATITUDES,)``).

        """
        return self._get_state()

    def _get_temp(self, model="RL"):
        """Return the temperature field from the chosen model process.

        Parameters
        ----------
        model : {'RL', 'climlab'}, optional
            Select which internal process to query: ``'RL'`` returns the
            reinforcement-learning controlled EBM, ``'climlab'`` returns
            the baseline ClimLab process. Default is ``'RL'``.

        Returns
        -------
        numpy.ndarray
            1-D float32 array of temperatures for all latitudes.

        """
        if model == "RL":
            ebm = self.ebm
        elif model == "climlab":
            ebm = self.climlab_ebm
        temp = np.array(ebm.Ts, dtype=np.float32).reshape(-1)
        return temp

    def _get_info(self):
        """Return an info dictionary for the current timestep.

        The environment currently returns a placeholder dict to satisfy the
        Gym API.
        """
        return {"_": None}

    def _get_params(self):
        """Return the flattened model parameter vector used by the agent.

        Returns
        -------
        numpy.ndarray
            1-D float32 array containing [D, A..., B..., a0, a2]. Values are
            extracted from the underlying climlab subprocess structures.

        """
        D = self.ebm.subprocess["diffusion"].D
        A, B = self.ebm.subprocess["LW"].A / 1e2, self.ebm.subprocess["LW"].B
        a0, a2 = (
            self.ebm.subprocess["albedo"].a0,
            self.ebm.subprocess["albedo"].a2,
        )
        params = np.array(
            [D, *(A.reshape(-1)), *(B.reshape(-1)), a0, a2], dtype=np.float32
        )
        return params

    def _get_state(self):
        """Return the environment state vector used as an observation.

        Returns
        -------
        numpy.ndarray
            1-D float32 array representing the flattened temperature field.

        """
        state = self._get_temp()
        return state

    def step(self, action):
        """Apply an action, step forward the EBM and return result.

        Parameters
        ----------
        action : array-like
            Action vector matching :pyattr:`action_space` (D, A..., B..., a0, a2).

        Returns
        -------
        obs, reward, terminated, truncated, info
            Observation, scalar reward (negative MSE), and Gym termination
            flags plus an info dictionary.

        """
        split_idx = EBM_LATITUDES
        D = action[0]
        A = np.array(action[1 : split_idx + 1]).reshape(-1, 1)
        B = np.array(action[split_idx + 1 : -2]).reshape(-1, 1)
        a0, a2 = action[-2], action[-1]

        D = np.clip(D, self.min_D, self.max_D)
        A = np.clip(A, self.min_A, self.max_A)
        B = np.clip(B, self.min_B, self.max_B)
        a0 = np.clip(a0, self.min_a0, self.max_a0)
        a2 = np.clip(a2, self.min_a2, self.max_a2)

        self.ebm.subprocess["diffusion"].D = D
        self.ebm.subprocess["LW"].A = A * 1e2
        self.ebm.subprocess["LW"].B = B
        self.ebm.subprocess["albedo"].a0 = a0
        self.ebm.subprocess["albedo"].a2 = a2

        self.ebm.step_forward()
        self.climlab_ebm.step_forward()

        costs = np.mean(
            (np.array(self.ebm.Ts).reshape(-1) - self.Ts_ncep_annual.values)[
                self.ebm_min_idx : self.ebm_max_idx
            ]
            ** 2
        )

        self.state = self._get_state()
        return self._get_obs(), -costs, False, False, self._get_info()

    def get_target_state(self):
        """Return the observational target state for the local band.

        Returns
        -------
        numpy.ndarray
            Target climatological temperatures for the environment's
            latitude band.

        """
        return np.array(self.Ts_ncep_annual.values[self.ebm_min_idx : self.ebm_max_idx])

    def reset(self, seed=None, options=None):
        """Reset the environment and initialize internal EBM state.

        Parameters
        ----------
        seed : int or None
            Optional random seed forwarded to ``super().reset`` for
            reproducibility. If ``None``, no explicit reseeding is
            performed here (Gym's default behavior applies).
        options : dict or None
            Gymnasium reset options (currently unused by this environment,
            present for API compatibility).

        Returns
        -------
        observation : numpy.ndarray
            The initial observation for the environment (temperature field
            over all latitudes) as a 1-D ``float32`` array with length
            ``EBM_LATITUDES``.
        info : dict
            An info dictionary following the Gym API (currently contains
            internal metadata via ``self._get_info()``).

        """
        super().reset(seed=seed)
        self.ebm = climlab.EBM_annual(
            a0=self.utils.a0_ref,
            a2=self.utils.a2_ref,
            D=self.utils.D_ref,
            A=np.array([self.utils.A_ref * 1e2 for x in range(EBM_LATITUDES)]).reshape(
                -1, 1
            ),
            B=np.array([self.utils.B_ref for x in range(EBM_LATITUDES)]).reshape(-1, 1),
            num_lat=EBM_LATITUDES,
            name="EBM Model w/ RL",
        )
        self.ebm.Ts[:] = 50.0
        self.Ts_ncep_annual = self.utils.Ts_ncep_annual.interp(
            lat=self.ebm.lat, kwargs={"fill_value": "extrapolate"}
        )

        self.climlab_ebm = climlab.process_like(self.ebm)
        self.climlab_ebm.name = "EBM Model"

        self.ebm_min_idx, self.ebm_max_idx = (
            self.cid * EBM_SUBLATITUDES,
            (self.cid + 1) * EBM_SUBLATITUDES,
        )
        self.phi = self.ebm.lat[self.ebm_min_idx : self.ebm_max_idx]

        self.state = self._get_state()
        return self._get_obs(), self._get_info()

    def _render_frame(self, save_fig=None, idx=None):
        """Create a matplotlib figure visualising parameters, state and error.

        Parameters
        ----------
        save_fig : str or None
            Optional path to save the generated figure.
        idx : int or None
            Optional frame index used when saving or annotating figures.

        Returns
        -------
        matplotlib.figure.Figure
            The constructed figure object.

        """
        fig = plt.figure(figsize=(28, 8))
        gs = GridSpec(1, 3, figure=fig)

        params = self._get_params()

        ax1 = fig.add_subplot(gs[0, 0])

        ax1_labels = ["D", "A", "B", "a0", "a2"]
        ax1_colors = [
            "tab:blue",
            "tab:blue",
            "tab:blue",
            "tab:blue",
            "tab:blue",
        ]
        ax1_bars = ax1.bar(
            ax1_labels,
            [
                params[0],
                np.mean(params[: EBM_LATITUDES + 1]),
                np.mean(params[EBM_LATITUDES + 1 : -2]),
                *params[-2:],
            ],
            color=ax1_colors,
            width=0.75,
        )
        ax1.set_ylim(0, 10)
        ax1.set_ylabel("Value", fontsize=14)

        for bar in ax1_bars:
            height = bar.get_height()
            ax1.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(
            self.ebm.lat[self.ebm_min_idx : self.ebm_max_idx],
            self.ebm.Ts[self.ebm_min_idx : self.ebm_max_idx],
            label="EBM Model w/ RL",
        )
        ax2.plot(
            self.climlab_ebm.lat[self.ebm_min_idx : self.ebm_max_idx],
            self.climlab_ebm.Ts[self.ebm_min_idx : self.ebm_max_idx],
            label="EBM Model",
        )
        ax2.plot(
            self.climlab_ebm.lat,
            self.Ts_ncep_annual,
            label="Observations",
            c="k",
        )
        ax2.set_ylabel("Temperature (°C)")
        ax2.set_xlabel("Latitude")
        ax2.set_xlim(-90, 90)
        ax2.set_xticks(np.arange(-90, 91, 30))
        ax2.legend()
        ax2.grid()

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(
            x=self.ebm.lat[self.ebm_min_idx : self.ebm_max_idx],
            height=np.abs(self.ebm.Ts.reshape(-1) - self.Ts_ncep_annual.values)[
                self.ebm_min_idx : self.ebm_max_idx
            ],
            label="EBM Model w/ RL",
        )
        ax3.bar(
            x=self.climlab_ebm.lat[self.ebm_min_idx : self.ebm_max_idx],
            height=np.abs(self.climlab_ebm.Ts.reshape(-1) - self.Ts_ncep_annual.values)[
                self.ebm_min_idx : self.ebm_max_idx
            ],
            label="EBM Model",
            zorder=-1,
        )
        ax3.set_ylabel("Error  (°C)")
        ax3.set_xlabel("Latitude")
        ax3.set_xlim(-90, 90)
        ax3.set_xticks(np.arange(-90, 91, 30))
        ax3.legend()
        ax3.grid()

        return fig

    def render(self, **kwargs):
        """Render the environment according to the configured mode.

        Supports ``'human'`` (display the figure) and ``'rgb_array'`` (return
        an RGB numpy array) modes as declared in :pyattr:`metadata`.
        """
        if self.render_mode == "human":
            self._render_frame(**kwargs)
            plt.show()
        elif self.render_mode == "rgb_array":
            fig = self._render_frame(**kwargs)
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape((height, width, 3))
            plt.close(fig)
            return image


def run_ebm(seed, cid):
    """Run the EBM client loop used by federated simulations.

    This function is designed to be executed in a background process by the
    federated client helper. It connects to the shared Redis server and
    runs the local agent for a fixed number of timesteps.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    cid : int
        Client identifier selecting which sub-latitude interval this client
        controls.

    """
    set_seed(seed)
    envs = gym.vector.SyncVectorEnv(
        [make_env(functools.partial(EnergyBalanceModelEnv, cid=cid), seed, NUM_STEPS)]
    )
    api = FedRAIN()
    agent = api.set_algorithm(
        "DDPG",
        envs=envs,
        seed=seed,
        actor_layer_size=ACTOR_LAYER_SIZE,
        critic_layer_size=CRITIC_LAYER_SIZE,
        fedRLConfig={
            "cid": cid,
            "num_steps": NUM_STEPS,
            "flwr_episodes": FLWR_EPISODES,
        },
    )

    obs, _ = envs.reset()
    for t in range(1, TOTAL_TIMESTEPS + 1):
        actions = agent.predict(obs)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        agent.update(actions, next_obs, rewards, terminations, truncations, infos)
        obs = next_obs


if __name__ == "__main__":
    seed = 1

    server = FLWRServer(NUM_CLIENTS, FLWR_ROUNDS)
    server.generate_actor(EnergyBalanceModelEnv, DDPGActor, ACTOR_LAYER_SIZE)
    server.set_client(seed=seed, fn=run_ebm, num_steps=NUM_STEPS)
    server.serve()
    server.stop()
