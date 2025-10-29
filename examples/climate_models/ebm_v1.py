"""Example: Energy Balance Model (EBM) environment (version 1).

This module provides a small OpenAI Gym-compatible environment that wraps
an annual energy-balance model (using ``climlab``) and exposes a
reinforcement-learning-friendly API used by the example scripts and tests.

Contents
--------
- ``EBMUtils``: dataset helpers and climatology loading.
- ``EnergyBalanceModelEnv``: the Gym environment exposing state, actions and
    a step/update loop compatible with DDPG agents.
- ``run_ebm``: convenience runner used for quick experiments.

The environment is intentionally lightweight and deterministic (when
seeded) so it is suitable for tests and small demos.
"""

import logging
import os

import climlab
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from gymnasium import spaces
from matplotlib.gridspec import GridSpec

from fedrain.api import FedRAIN
from fedrain.utils import make_env, set_seed, setup_logger

EBM_LATITUDES = 96

NUM_STEPS = 200
TOTAL_TIMESTEPS = 2000
ACTOR_LAYER_SIZE, CRITIC_LAYER_SIZE = 64, 64


class EBMUtils:
    """Utility helpers for dataset loading and climatologies.

    This small container provides file paths, dataset download helpers and
    cached climatological arrays used by the example EBM environments.
    """

    BASE_DIR = "."
    DATASETS_DIR = f"{BASE_DIR}/datasets"

    fp_Ts = f"{DATASETS_DIR}/skt.sfc.mon.1981-2010.ltm.nc"
    fp_ulwrf = f"{DATASETS_DIR}/ulwrf.ntat.mon.1981-2010.ltm.nc"
    fp_dswrf = f"{DATASETS_DIR}/dswrf.ntat.mon.1981-2010.ltm.nc"
    fp_uswrf = f"{DATASETS_DIR}/uswrf.ntat.mon.1981-2010.ltm.nc"

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
    ncep_ulwrf = download_and_save_dataset(
        ncep_url + "other_gauss/ulwrf.ntat.mon.1981-2010.ltm.nc",
        fp_ulwrf,
        "NCEP upwelling longwave radiation",
    ).sortby("lat")
    ncep_dswrf = download_and_save_dataset(
        ncep_url + "other_gauss/dswrf.ntat.mon.1981-2010.ltm.nc",
        fp_dswrf,
        "NCEP downwelling shortwave radiation",
    ).sortby("lat")
    ncep_uswrf = download_and_save_dataset(
        ncep_url + "other_gauss/uswrf.ntat.mon.1981-2010.ltm.nc",
        fp_uswrf,
        "NCEP upwelling shortwave radiation",
    ).sortby("lat")

    lat_ncep = ncep_Ts.lat
    lon_ncep = ncep_Ts.lon
    Ts_ncep_annual = ncep_Ts.skt.mean(dim=("lon", "time"))

    OLR_ncep_annual = ncep_ulwrf.ulwrf.mean(dim=("lon", "time"))
    ASR_ncep_annual = (ncep_dswrf.dswrf - ncep_uswrf.uswrf).mean(dim=("lon", "time"))

    a0_ref = 0.354
    a2_ref = 0.25
    D_ref = 0.6
    A_ref = 2.1
    B_ref = 2


class EnergyBalanceModelEnv(gym.Env):
    """Gym environment wrapping a energy-balance climate model.

    The agent controls a small set of radiative and diffusion parameters
    (D, A, B, a0, a2) that affect the model's temperature field. The
    observation is the vector of latitudinal surface temperatures and the
    reward is the negative mean-squared error relative to a reference
    climatology.

    Notes
    -----
    The environment uses ``climlab`` to create a physical EBM and exposes a
    straightforward step API compatible with vectorized Gym environments used
    in tests and examples.

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None):
        """Create the EBM environment and initialise action/observation spaces.

        Parameters
        ----------
        render_mode : str or None
            If provided, controls rendering mode (``'human'`` or
            ``'rgb_array'``). If ``None``, rendering is disabled.

        """
        self.utils = EBMUtils()

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
                    *[self.min_A for _ in range(EBM_LATITUDES)],
                    *[self.min_B for _ in range(EBM_LATITUDES)],
                    self.min_a0,
                    self.min_a2,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.max_D,
                    *[self.max_A for _ in range(EBM_LATITUDES)],
                    *[self.max_B for _ in range(EBM_LATITUDES)],
                    self.max_a0,
                    self.max_a2,
                ],
                dtype=np.float32,
            ),
            shape=(2 * EBM_LATITUDES + 3,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array(
                [self.min_temperature for _ in range(EBM_LATITUDES)], dtype=np.float32
            ).reshape(
                -1,
            ),
            high=np.array(
                [self.max_temperature for _ in range(EBM_LATITUDES)], dtype=np.float32
            ).reshape(
                -1,
            ),
            shape=(EBM_LATITUDES,),
            dtype=np.float32,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

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
        """Return the temperature field from the selected model process.

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
        ebm = self.ebm if model == "RL" else self.climlab_ebm
        temp = np.array(ebm.Ts, dtype=np.float32)
        return temp

    def _get_info(self):
        """Return an info dictionary for the current timestep.

        The environment currently does not populate diagnostics; this
        placeholder satisfies the Gym API.
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
        a0, a2 = self.ebm.subprocess["albedo"].a0, self.ebm.subprocess["albedo"].a2
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
        state = self._get_temp().reshape(
            -1,
        )
        return state

    def step(self, action):
        """Apply an action and advance the model one timestep.

        Parameters
        ----------
        action : array-like
            Array containing parameter values for D, A (per-lat), B (per-lat),
            a0 and a2. Values will be clipped to the environment parameter
            bounds.

        Returns
        -------
        obs, reward, done, trunc, info
            Standard Gym step tuple. ``done`` is always False in this episodic
            setup and ``trunc`` is False. ``info`` contains diagnostics.

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
        self.Ts_ncep_annual = self.utils.Ts_ncep_annual.interp(
            lat=self.ebm.lat, kwargs={"fill_value": "extrapolate"}
        )

        costs = np.mean(
            (np.array(self.ebm.Ts).reshape(-1) - self.Ts_ncep_annual.values) ** 2
        )

        self.state = self._get_state()

        return self._get_obs(), -costs, False, False, self._get_info()

    def get_target_state(self):
        """Return the target climatological temperature state.

        Returns
        -------
        numpy.ndarray
            Array of target temperatures used to compute the reward.

        """
        return np.array(self.Ts_ncep_annual.values)

    def reset(self, seed=None, options=None):
        """Reset the environment and initialize internal EBM state.

        This method resets the Gym environment's random seed (via
        :meth:`gym.Env.reset`), constructs the climlab EBM instance and a
        matching ClimLab process, interpolates the reference climatology to
        the model latitudes, and sets the initial observation stored in
        ``self.state``.

        Parameters
        ----------
        seed : int or None
            Optional random seed forwarded to ``super().reset`` for
            reproducibility. If ``None``, Gym's default behavior applies.
        options : dict or None
            Gymnasium reset options (currently unused by this environment).

        Returns
        -------
        observation, info
            The initial observation (1-D ``float32`` array) and an info
            dictionary following the Gym API.

        """
        super().reset(seed=seed)
        self.ebm = climlab.EBM_annual(
            a0=self.utils.a0_ref,
            a2=self.utils.a2_ref,
            D=self.utils.D_ref,
            A=np.array([self.utils.A_ref * 1e2 for _ in range(EBM_LATITUDES)]).reshape(
                -1, 1
            ),
            B=np.array([self.utils.B_ref for _ in range(EBM_LATITUDES)]).reshape(-1, 1),
            num_lat=EBM_LATITUDES,
            name="EBM Model w/ RL",
        )
        self.ebm.Ts[:] = 50.0
        self.Ts_ncep_annual = self.utils.Ts_ncep_annual.interp(
            lat=self.ebm.lat, kwargs={"fill_value": "extrapolate"}
        )

        self.climlab_ebm = climlab.process_like(self.ebm)
        self.climlab_ebm.name = "EBM Model"

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
        ax1_colors = ["tab:blue"] * len(ax1_labels)
        ax1_bars = ax1.bar(
            ax1_labels,
            [
                params[0],
                np.mean(params[1 : EBM_LATITUDES + 1]),
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
        ax2.plot(self.ebm.lat, self.ebm.Ts, label="EBM Model w/ RL")
        ax2.plot(self.climlab_ebm.lat, self.climlab_ebm.Ts, label="EBM Model")
        ax2.plot(self.climlab_ebm.lat, self.Ts_ncep_annual, label="Observations", c="k")
        ax2.set_ylabel("Temperature (°C)")
        ax2.set_xlabel("Latitude")
        ax2.set_xlim(-90, 90)
        ax2.set_xticks(np.arange(-90, 91, 30))
        ax2.legend()
        ax2.grid()

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(
            x=self.ebm.lat,
            height=np.abs(self.ebm.Ts.reshape(-1) - self.Ts_ncep_annual.values),
            label="EBM Model w/ RL",
        )
        ax3.bar(
            x=self.climlab_ebm.lat,
            height=np.abs(self.climlab_ebm.Ts.reshape(-1) - self.Ts_ncep_annual.values),
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


def run_ebm(seed):
    """Run a short training loop using the environment and a DDPG agent.

    This convenience runner is intended for manual experimentation and
    demonstration. It seeds RNGs, constructs a vectorized environment and
    runs the DDPG agent for ``TOTAL_TIMESTEPS`` steps.

    Parameters
    ----------
    seed : int
        Random seed used for reproducibility.

    """
    set_seed(seed)
    envs = gym.vector.SyncVectorEnv([make_env(EnergyBalanceModelEnv, seed, NUM_STEPS)])
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
    run_ebm(seed)
