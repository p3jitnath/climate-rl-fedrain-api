"""Multi-client EBM example (version 3).

This module demonstrates a federated setup (similar to version 2) where different partitions of
the latitude band are controlled by separate clients. It provides an
``EnergyBalanceModelEnv`` class that accepts a ``cid`` argument to select
the sub-latitude region the client controls. The module also contains a
``run_ebm_subprocess`` helper that is suitable for running in a background process
paired with the federated Flower server used in tests and examples.

Input observations are now ONLY the temperature field for the sub-latitude region.
Rewards are computed based on the mean squared error between the
model's temperature field and observed climatological values ONLY over
the specified sub-latitude region.
"""

import functools
import logging
import os
import time

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

WAIT_TIME = 1e-4


class EBM:
    """Parent process for multi-client EBM federated simulations.

    ``EBM`` acts as the server-side driver that listens for start/compute
    signals from client workers via a shared Redis instance. It exposes a
    ``run`` method that enters a loop responding to client requests and
    orchestrating updates to the central ClimLab EBM model.
    """

    def __init__(self):
        """Initialize the EBM server helper.

        This constructor currently performs no actions; the heavy lifting is
        implemented in :meth:`run` which starts the server loop.
        """
        pass

    def run(self, level=logging.DEBUG):
        """Run the central EBM process loop.

        This method blocks and performs the following high-level steps in a
        tight loop:

        - Wait for clients to signal they are ready (``SIGSTART``).
        - Publish initial model state to requesting clients.
        - Wait for all clients to publish parameter proposals
          (``py2f_redis_s{cid}``).
        - Aggregate parameters, update the central ClimLab model and step
          it forward.
        - Publish the updated model state back to clients for the next step.

        Parameters
        ----------
        level : int, optional
            Logging level used when creating module loggers (default:
            ``logging.DEBUG``).

        """

        class EBMUtils:
            BASE_DIR = "."
            DATASETS_DIR = f"{BASE_DIR}/datasets"

            fp_Ts = f"{DATASETS_DIR}/skt.sfc.mon.1981-2010.ltm.nc"
            ncep_url = "http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/"

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

            logger = setup_logger("EBM MAIN", level)

            REDIS_ADDRESS = os.getenv("SSDB")
            if REDIS_ADDRESS is None:
                raise EnvironmentError("SSDB environment variable is not set.")
            redis = Client(address=REDIS_ADDRESS, cluster=False)
            logger.debug(f"Connected to Redis server: {REDIS_ADDRESS}")

        class ClimLabEBM:
            """Thin wrapper that builds the ClimLab EBM and related state.

            This helper constructs a :class:`climlab.EBM_annual` instance and
            prepares associated arrays used by the federated server for
            publishing model state to clients.
            """

            def __init__(self, utils):
                self.utils = utils
                self.ebm = climlab.EBM_annual(
                    a0=self.utils.a0_ref,
                    a2=self.utils.a2_ref,
                    D=self.utils.D_ref,
                    A=np.array(
                        [self.utils.A_ref * 1e2 for x in range(EBM_LATITUDES)]
                    ).reshape(-1, 1),
                    B=np.array(
                        [self.utils.B_ref for x in range(EBM_LATITUDES)]
                    ).reshape(-1, 1),
                    num_lat=EBM_LATITUDES,
                    name="EBM Model w/ RL",
                )
                self.ebm.Ts[:] = 50.0
                self.Ts_ncep_annual = np.array(
                    self.utils.Ts_ncep_annual.interp(
                        lat=self.ebm.lat, kwargs={"fill_value": "extrapolate"}
                    )
                ).reshape(-1, 1)

                self.climlab_ebm = climlab.process_like(self.ebm)
                self.climlab_ebm.name = "EBM Model"

        class Exists:
            """Small container tracking which client signals/files exist.

            The structure is used to track which clients have requested a
            start, requested a compute step, or have submitted parameter
            proposals.
            """

            def __init__(self):
                self.sigcompute = np.zeros(NUM_CLIENTS)
                self.sigstart = np.zeros(NUM_CLIENTS)
                self.params = np.zeros(NUM_CLIENTS)

        utils, exists = EBMUtils(), Exists()
        ctr = 0

        while True:

            for idx, cid in enumerate(range(NUM_CLIENTS)):
                if utils.redis.tensor_exists(f"SIGSTART_S{cid}"):
                    exists.sigstart[idx] = 1
                if utils.redis.tensor_exists(f"SIGCOMPUTE_S{cid}"):
                    exists.sigcompute[idx] = 1

            if np.sum(exists.sigstart) > 0:
                cebm = ClimLabEBM(utils)
                cebm.EBM_SUBLATITUDES = EBM_LATITUDES // NUM_CLIENTS

                for idx, cid in enumerate(range(NUM_CLIENTS)):
                    if exists.sigstart[idx] != 0:
                        utils.redis.delete_tensor(f"SIGSTART_S{cid}")
                        time.sleep(WAIT_TIME)

                        utils.redis.put_tensor(
                            f"f2py_redis_s{cid}",
                            np.array(
                                [
                                    cebm.ebm.Ts,
                                    cebm.climlab_ebm.Ts,
                                    cebm.Ts_ncep_annual,
                                    np.array(cebm.ebm.lat).reshape(-1, 1),
                                ],
                                dtype=np.float32,
                            ),
                        )
                        exists.sigstart[idx] = 0

            if np.sum(exists.sigcompute) == NUM_CLIENTS:
                params = [None for x in range(NUM_CLIENTS)]

                while sum(exists.params) != NUM_CLIENTS:
                    for idx, cid in enumerate(range(NUM_CLIENTS)):
                        if params[idx] is None:
                            if utils.redis.tensor_exists(f"py2f_redis_s{cid}"):
                                params[idx] = utils.redis.get_tensor(
                                    f"py2f_redis_s{cid}"
                                )
                                exists.params[idx] = 1
                                time.sleep(WAIT_TIME)
                                utils.redis.delete_tensor(f"py2f_redis_s{cid}")
                            else:
                                continue

                params = np.array(params)
                D = np.mean(params[:, 0])
                A = np.array(params[:, 1 : cebm.EBM_SUBLATITUDES + 1]).reshape(-1, 1)
                B = np.array(params[:, cebm.EBM_SUBLATITUDES + 1 : -2]).reshape(-1, 1)
                a0 = np.mean(params[:, -2])
                a2 = np.mean(params[:, -1])

                cebm.ebm.subprocess["diffusion"].D = D
                cebm.ebm.subprocess["LW"].A = A
                cebm.ebm.subprocess["LW"].B = B
                cebm.ebm.subprocess["albedo"].a0 = a0
                cebm.ebm.subprocess["albedo"].a2 = a2

                cebm.ebm.step_forward()
                cebm.climlab_ebm.step_forward()

                for idx, cid in enumerate(range(NUM_CLIENTS)):
                    utils.redis.delete_tensor(f"SIGCOMPUTE_S{cid}")
                    time.sleep(WAIT_TIME)
                    utils.redis.put_tensor(
                        f"f2py_redis_s{cid}",
                        np.array([cebm.ebm.Ts, cebm.climlab_ebm.Ts], dtype=np.float32),
                    )

                exists.params[:] = 0
                exists.sigcompute[:] = 0

            ctr += 1
            time.sleep(WAIT_TIME)


class EnergyBalanceModelEnv(gym.Env):
    """Gym environment exposing a sub-latitude partition of the EBM.

    This environment proxies actions to the central ``EBM`` process via a
    Redis-backed protocol. Each environment instance controls a contiguous
    sub-latitude band (selected by ``cid``) and exposes an observation that
    is the local temperature field for that band. Actions propose local
    parameter updates which the central server aggregates and applies to
    the ClimLab model.

    Notes
    -----
    Communication with the server uses SmartRedis tensors with keys like
    ``py2f_redis_s{cid}``, ``f2py_redis_s{cid}`` and control signals
    ``SIGSTART_S{cid}``, ``SIGCOMPUTE_S{cid}``.

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
        self.cid = cid

        self.logger = setup_logger(f"EBM SUB {self.cid}", level)
        self.logger.debug(f"Environment ID: {self.cid}")
        self.logger.debug(f"Number of clients: {NUM_CLIENTS}")

        self.D = 0.6
        self.min_D = 0.55
        self.max_D = 0.65

        self.A = np.array([2.1 for x in range(EBM_SUBLATITUDES)])
        self.min_A = 1.4
        self.max_A = 4.2

        self.B = np.array([2 for x in range(EBM_SUBLATITUDES)])
        self.min_B = 1.95
        self.max_B = 2.05

        self.a0 = 0.354
        self.min_a0 = 0.3
        self.max_a0 = 0.4

        self.a2 = 0.25
        self.min_a2 = 0.2
        self.max_a2 = 0.3

        self.min_temperature = -90
        self.max_temperature = 90

        self.action_space = spaces.Box(
            low=np.array(
                [
                    self.min_D,
                    *[self.min_A for _ in range(EBM_SUBLATITUDES)],
                    *[self.min_B for _ in range(EBM_SUBLATITUDES)],
                    self.min_a0,
                    self.min_a2,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.max_D,
                    *[self.max_A for _ in range(EBM_SUBLATITUDES)],
                    *[self.max_B for _ in range(EBM_SUBLATITUDES)],
                    self.max_a0,
                    self.max_a2,
                ],
                dtype=np.float32,
            ),
            shape=(2 * EBM_SUBLATITUDES + 3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=self.min_temperature,
            high=self.max_temperature,
            shape=(EBM_SUBLATITUDES,),
            dtype=np.float32,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode
        self.wait_time = 1e-4

        self.REDIS_ADDRESS = os.getenv("SSDB")
        if self.REDIS_ADDRESS is None:
            raise EnvironmentError("SSDB environment variable is not set.")
        self.redis = Client(address=self.REDIS_ADDRESS, cluster=False)
        self.logger.debug(f"Connected to Redis server: {self.REDIS_ADDRESS}")

        self.redis.put_tensor(f"SIGALIVE_S{self.cid}", np.array([1], dtype=np.int32))

    def _get_params(self):
        """Return the flattened parameter vector representing local proposals.

        Returns
        -------
        numpy.ndarray
            1-D float32 array containing [D, A..., B..., a0, a2] proposed by
            this client instance.

        """
        return np.array(
            [self.D, *(self.A), *(self.B), self.a0, self.a2], dtype=np.float32
        )

    def _get_obs(self):
        """Return the current observation for the local latitude band.

        Returns
        -------
        numpy.ndarray
            1-D float32 array containing the temperature field for this
            client's latitude partition (length ``EBM_SUBLATITUDES``).

        """
        return np.array(self.state, dtype=np.float32)

    def _get_info(self):
        """Return an info dictionary for the current timestep.

        The environment uses a placeholder info dict to satisfy the Gym API.
        """
        return {"_": None}

    def step(self, action):
        """Apply an action, synchronise with the central EBM and return result.

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
        self.D = action[0]
        self.A, self.B = (
            action[1 : EBM_SUBLATITUDES + 1],
            action[EBM_SUBLATITUDES + 1 : -2],
        )
        self.a0, self.a2 = action[-2], action[-1]

        self.D = np.clip(self.D, self.min_D, self.max_D)
        self.A = np.clip(self.A, self.min_A, self.max_A)
        self.B = np.clip(self.B, self.min_B, self.max_B)
        self.a0 = np.clip(self.a0, self.min_a0, self.max_a0)
        self.a2 = np.clip(self.a2, self.min_a2, self.max_a2)

        self.redis.put_tensor(
            f"py2f_redis_s{self.cid}",
            np.array(
                [self.D, *(self.A * 1e2), *(self.B), self.a0, self.a2],
                dtype=np.float32,
            ),
        )
        self.redis.put_tensor(f"SIGCOMPUTE_S{self.cid}", np.array([1], dtype=np.int32))

        self.ebm_Ts = None
        while self.ebm_Ts is None:
            if self.redis.tensor_exists(f"f2py_redis_s{self.cid}"):
                self.ebm_Ts, self.climlab_ebm_Ts = self.redis.get_tensor(
                    f"f2py_redis_s{self.cid}"
                )
                time.sleep(self.wait_time)
                self.redis.delete_tensor(f"f2py_redis_s{self.cid}")
            else:
                continue

        self.state = self.ebm_Ts[self.ebm_min_idx : self.ebm_max_idx].reshape(-1)

        costs = np.mean(
            (self.ebm_Ts - self.Ts_ncep_annual)[self.ebm_min_idx : self.ebm_max_idx]
            ** 2
        )

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
        """Reset the environment and request initial state from the server.

        The reset sends a ``SIGSTART`` message and blocks until the central
        process publishes the initial model arrays for this client. It
        returns the initial observation and info dict as required by Gym.

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

        self.redis.put_tensor(f"SIGSTART_S{self.cid}", np.array([1], dtype=np.int32))

        self.ebm_Ts = None
        while self.ebm_Ts is None:
            if self.redis.tensor_exists(f"f2py_redis_s{self.cid}"):
                (
                    self.ebm_Ts,
                    self.climlab_ebm_Ts,
                    self.Ts_ncep_annual,
                    self.ebm_lat,
                ) = self.redis.get_tensor(f"f2py_redis_s{self.cid}")
                time.sleep(self.wait_time)
                self.redis.delete_tensor(f"f2py_redis_s{self.cid}")
            else:
                continue

        self.ebm_min_idx, self.ebm_max_idx = (
            self.cid * EBM_SUBLATITUDES,
            (self.cid + 1) * EBM_SUBLATITUDES,
        )
        self.phi = self.ebm_lat[self.ebm_min_idx : self.ebm_max_idx]
        self.state = self.ebm_Ts[self.ebm_min_idx : self.ebm_max_idx].reshape(-1)

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
                np.mean(params[1 : EBM_SUBLATITUDES + 1]),
                np.mean(params[EBM_SUBLATITUDES + 1 : -2]),
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
            self.ebm_lat[self.ebm_min_idx : self.ebm_max_idx],
            self.ebm_Ts[self.ebm_min_idx : self.ebm_max_idx],
            label="EBM Model w/ RL",
        )
        ax2.plot(
            self.ebm_lat[self.ebm_min_idx : self.ebm_max_idx],
            self.climlab_ebm_Ts[self.ebm_min_idx : self.ebm_max_idx],
            label="EBM Model",
        )
        ax2.plot(
            self.ebm_lat,
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
            x=self.ebm_lat[self.ebm_min_idx : self.ebm_max_idx].reshape(-1),
            height=np.abs(
                self.ebm_Ts[self.ebm_min_idx : self.ebm_max_idx]
                - self.Ts_ncep_annual[self.ebm_min_idx : self.ebm_max_idx]
            ).reshape(-1),
            label="EBM Model w/ RL",
        )
        ax3.bar(
            x=self.ebm_lat[self.ebm_min_idx : self.ebm_max_idx].reshape(-1),
            height=np.abs(
                self.climlab_ebm_Ts[self.ebm_min_idx : self.ebm_max_idx]
                - self.Ts_ncep_annual[self.ebm_min_idx : self.ebm_max_idx]
            ).reshape(-1),
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


def run_ebm_subprocess(seed, cid):
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

    parent_ebm = EBM()
    server = FLWRServer(NUM_CLIENTS, FLWR_ROUNDS)
    server.generate_actor(EnergyBalanceModelEnv, DDPGActor, ACTOR_LAYER_SIZE)
    server.set_client(seed=seed, fn=run_ebm_subprocess, num_steps=NUM_STEPS)
    server.start_process_fn(parent_ebm.run)
    server.serve()
    server.stop()
