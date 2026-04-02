"""Example: Radiative Convective Equilibrium (RCE)-17 environment (version 0).

This module mirrors the structure used by the EBM example and provides a
Gym-compatible RCE environment built on ``climlab``. The environment exposes
an RL-friendly action/observation API, helper plotting routines, and a small
training runner for manual experiments.

Contents
--------
- ``RCEUtils``: dataset helpers and pressure-level climatology loading.
- ``RadiativeConvectiveModelEnv``: Gym environment exposing RCE state,
  controls, and step/update dynamics.
- ``run_rce``: convenience runner for quick DDPG experiments.
"""

import logging
import os

import climlab
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tephi
import xarray as xr
from gymnasium import spaces
from matplotlib.gridspec import GridSpec
from metpy.plots import SkewT

from fedrain.api import FedRAIN
from fedrain.utils import make_env, set_seed, setup_logger

NUM_STEPS = 500
TOTAL_TIMESTEPS = 2000
ACTOR_LAYER_SIZE, CRITIC_LAYER_SIZE = 64, 64


class RCEUtils:
    """Utility helpers for dataset loading and observation climatology.

    The class lazily downloads and caches NCEP pressure-level temperature data
    and computes a global-weighted mean temperature profile used as the target
    in reward calculation.
    """

    BASE_DIR = "."
    DATASETS_DIR = f"{BASE_DIR}/datasets"

    fp_air = f"{DATASETS_DIR}/air.mon.ltm.1981-2010.nc"
    ncep_url = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Monthlies/"

    def download_and_save_dataset(url, filepath, dataset_name):
        """Download (or load) and return a named dataset.

        Parameters
        ----------
        url : str
            Remote URL of the dataset.
        filepath : str
            Local cache path for the dataset.
        dataset_name : str
            Human-readable dataset name for logging.

        Returns
        -------
        xarray.Dataset
            Loaded dataset.
        """
        logger = setup_logger("DATASET", logging.DEBUG)
        if not os.path.exists(filepath):
            logger.debug(f"Downloading {dataset_name} data ...")
            dataset = xr.open_dataset(
                url,
                decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
            )
            dataset.to_netcdf(filepath)
            logger.debug(f"{dataset_name} data saved to {filepath}")
        else:
            logger.debug(f"Loading {dataset_name} data ...")
            dataset = xr.open_dataset(
                filepath,
                decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
            )
        return dataset

    ncep_air = download_and_save_dataset(
        ncep_url + "pressure/air.mon.1981-2010.ltm.nc#mode=bytes",
        fp_air,
        "NCEP pressure-level temperature",
    )

    coslat = np.cos(np.deg2rad(ncep_air.lat))
    weight = coslat / coslat.mean(dim="lat")
    Tobs = (ncep_air.air * weight).mean(dim=("lat", "lon", "time"))


class RadiativeConvectiveModelEnv(gym.Env):
    """Gym environment wrapping a radiative-convective column model.

    The agent controls net atmospheric emissivity and a profile of convective
    adjustment lapse rates. The state is the temperature profile in degrees C,
    and reward is negative mean-squared error versus a target climatology.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None, locale="uk"):
        """Create the RCE environment and initialise spaces.

        Parameters
        ----------
        render_mode : str or None
            Rendering mode. Supported values are ``'human'`` and
            ``'rgb_array'``.
        locale : {'uk', 'us'}
            Plotting convention for thermodynamic diagrams.
            ``'uk'`` draws a tephigram; ``'us'`` draws a skew-T.
        """
        self.min_emissivity = 0.0
        self.max_emissivity = 1.0

        self.min_adj_lapse_rate = 5.5
        self.max_adj_lapse_rate = 9.8

        self.min_temperature = -90
        self.max_temperature = 90

        self.utils = RCEUtils()

        self.action_space = spaces.Box(
            low=np.array(
                [
                    self.min_emissivity,
                    *[
                        self.min_adj_lapse_rate
                        for _ in range(len(self.utils.Tobs.level))
                    ],
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.max_emissivity,
                    *[
                        self.max_adj_lapse_rate
                        for _ in range(len(self.utils.Tobs.level))
                    ],
                ],
                dtype=np.float32,
            ),
            shape=(1 + len(self.utils.Tobs.level),),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=self.min_temperature,
            high=self.max_temperature,
            shape=(len(self.utils.Tobs.level),),
            dtype=np.float32,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert locale in ["uk", "us"]
        self.render_mode = render_mode
        self.locale = locale

        self.reset()

    def _get_obs(self):
        """Return the current observation vector."""
        return self._get_temp().values

    def _get_temp(self, model="RL"):
        """Return the temperature profile from the selected process.

        Parameters
        ----------
        model : {'RL', 'climlab'}
            Source process. ``'RL'`` refers to the controlled model while
            ``'climlab'`` refers to the baseline clone.

        Returns
        -------
        xarray.DataArray
            Temperature profile in degrees C aligned to pressure levels.
        """
        rcm = self.rcm if model == "RL" else self.climlab_rcm
        temp = np.concatenate([rcm.Tatm, rcm.Ts], dtype=np.float32)
        temp -= climlab.constants.tempCtoK
        temp = xr.DataArray(temp, coords={"level": self.utils.Tobs.level.values[::-1]})
        return temp

    def _get_info(self):
        """Return a Gym info dictionary for the current timestep."""
        return {"_": None}

    def _get_params(self):
        """Return the flattened model parameter vector.

        Returns
        -------
        numpy.ndarray
            1-D float32 array containing [emissivity, lapse_rate...].
        """
        emissivity = self.rcm.subprocess["Radiation (net)"].emissivity
        adj_lapse_rate = self.rcm.subprocess["Convection"].adj_lapse_rate
        params = np.array([emissivity, *adj_lapse_rate], dtype=np.float32)
        return params

    def _get_state(self):
        """Return the state vector used as observation."""
        state = self._get_temp().values
        return state

    def step(self, action):
        """Apply an action and advance both controlled and baseline models.

        Parameters
        ----------
        action : array-like
            [emissivity, lapse_rate_0, ..., lapse_rate_N].

        Returns
        -------
        obs, reward, done, trunc, info
            Standard Gym step tuple. ``done`` and ``trunc`` are False.
        """
        emissivity, adj_lapse_rate = action[0], action[1:]

        emissivity = np.clip(emissivity, self.min_emissivity, self.max_emissivity)
        adj_lapse_rate = np.clip(
            adj_lapse_rate, self.min_adj_lapse_rate, self.max_adj_lapse_rate
        )

        self.rcm.subprocess["Radiation (net)"].emissivity = emissivity
        self.rcm.subprocess["Convection"].adj_lapse_rate = adj_lapse_rate
        self.rcm.step_forward()
        self.climlab_rcm.step_forward()

        Tprofile = self._get_temp().values
        costs = np.mean((Tprofile - self.utils.Tobs.values[::-1]) ** 2)

        self.state = self._get_state()
        return self._get_obs(), -costs, False, False, self._get_info()

    def reset(self, seed=None, options=None):
        """Reset the environment and initialize internal RCE processes.

        Parameters
        ----------
        seed : int or None
            Optional random seed forwarded to ``super().reset``.
        options : dict or None
            Gymnasium reset options (unused).

        Returns
        -------
        observation, info
            Initial observation and info dictionary.
        """
        super().reset(seed=seed)

        rce_state = climlab.column_state(lev=self.utils.Tobs.level[1:], water_depth=2.5)
        h2o = climlab.radiation.ManabeWaterVapor(
            state=rce_state,
            lev=self.utils.Tobs.level[1:],
        )
        rad = climlab.radiation.RRTMG(
            name="Radiation (net)",
            state=rce_state,
            specific_humidity=h2o.q,
            S0=1365.0,
            timestep=climlab.constants.seconds_per_day,
            albedo=0.25,
        )
        conv = climlab.convection.ConvectiveAdjustment(
            name="Convection",
            state=rce_state,
            adj_lapse_rate="MALR",
            timestep=rad.timestep,
        )

        self.rcm = climlab.couple([rad, conv], name="RCE Model w/ RL")

        # Start from an isothermal initial state for stable, repeatable resets.
        self.rcm.state.Tatm[:] = self.rcm.state.Ts

        self.climlab_rcm = climlab.process_like(self.rcm)
        self.climlab_rcm.name = "RCE Model"

        self.state = self._get_state()
        return self._get_obs(), self._get_info()

    def _render_frame(self):
        """Create a figure visualising parameters, thermodynamic profile and errors.

        Returns
        -------
        matplotlib.figure.Figure
            Constructed figure object.
        """
        fig = plt.figure(figsize=(29, 9))
        gs = GridSpec(1, 3, figure=fig)

        params = self._get_params()

        Tprofile_RL = self._get_temp()
        T_diff_RL = self.utils.Tobs.sel(level=[100, 200, 1000]) - Tprofile_RL.sel(
            level=[100, 200, 1000]
        )
        T_diff_RL = np.abs(T_diff_RL)

        Tprofile_climlab = self._get_temp(model="climlab")
        T_diff_climlab = self.utils.Tobs.sel(
            level=[100, 200, 1000]
        ) - Tprofile_climlab.sel(level=[100, 200, 1000])
        T_diff_climlab = np.abs(T_diff_climlab)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1_labels = ["Emissivity", "Mean Adj Lapse Rate"]
        ax1_colors = ["tab:blue", "tab:blue"]
        ax1_bars = ax1.bar(
            ax1_labels,
            [params[0], np.mean(params[1:])],
            color=ax1_colors,
            width=0.75,
        )
        ax1.set_ylim(0, 10)
        ax1.set_ylabel("Value", fontsize=14)
        ax1.set_title("Parameters")

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

        if self.locale == "us":
            skew = SkewT(fig, subplot=gs[0, 1], rotation=30)
            self._make_skewT(skew, "SkewT-logP")
            self._add_profile(skew, self.rcm)
            self._add_profile(skew, self.climlab_rcm)
        elif self.locale == "uk":
            ax2 = fig.add_subplot(gs[0, 1])
            tephi.ISOBAR_FIXED = [50, 1000, 100]
            tpg = tephi.Tephigram(
                figure=ax2.get_figure(),
                anchor=[(1000, -15), (70, -25)],
            )
            tpg.plot(
                self._get_tephigram_data(self.rcm),
                label=self.rcm.name,
                color="tab:blue",
            )
            tpg.plot(
                self._get_tephigram_data(self.climlab_rcm),
                label=self.climlab_rcm.name,
                color="tab:orange",
            )
            tpg.plot(
                zip(self.utils.Tobs.level, self.utils.Tobs),
                color="black",
                linestyle="-",
                linewidth=2,
                label="Observations",
            )
            tpg.axes.legend()
            ax2.set_title("Tephigram")
            ax2.set_axis_off()

        ax3 = fig.add_subplot(gs[0, 2])
        ax3_bar_width = 0.4
        ax3_labels = ["T_diff @ 100 hPa", "T_diff @ 200 hPa", "T_diff @ 1000 hPa"]
        ax3_ind = np.arange(1, 1 + len(ax3_labels))
        ax3_bars_RL = ax3.bar(
            ax3_ind - (ax3_bar_width / 2),
            T_diff_RL,
            color=["tab:blue", "tab:blue", "tab:blue"],
            width=ax3_bar_width,
        )
        ax3_bars_climlab = ax3.bar(
            ax3_ind + (ax3_bar_width / 2),
            T_diff_climlab,
            color=["tab:orange", "tab:orange", "tab:orange"],
            width=ax3_bar_width,
        )
        ax3.set_ylim(0, 20)
        ax3.set_ylabel("Difference [$\\degree$C]", fontsize=14)
        ax3.set_title("Temperature Differences")
        ax3.set_xticks(ax3_ind)
        ax3.set_xticklabels(ax3_labels)
        ax3.legend(
            (ax3_bars_RL[0], ax3_bars_climlab[0]), ("RCE Model w/ RL", "RCE Model")
        )

        ax3_bars = ax3_bars_RL + ax3_bars_climlab
        for bar in ax3_bars:
            height = bar.get_height()
            ax3.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        return fig

    def _get_tephigram_data(self, rcm):
        """Build pressure-temperature pairs used by tephi plotting.

        Parameters
        ----------
        rcm : climlab process
            Model process to convert to plotting data.

        Returns
        -------
        numpy.ndarray
            Two-column array of [pressure_hPa, temperature_C].
        """
        temp = np.concatenate([rcm.Tatm, rcm.Ts], dtype=np.float32)
        temp -= climlab.constants.tempCtoK
        levels = np.concatenate([rcm.lev, [1000]], dtype=np.float32)

        df = pd.DataFrame(zip(levels, temp))
        df.columns = ["pressure", "temperature"]
        return df.values

    def _make_skewT(self, skew, title=None):
        """Render observed profile and background adiabats on a skew-T."""
        skew.plot(
            self.utils.Tobs.level,
            self.utils.Tobs,
            color="black",
            linestyle="-",
            linewidth=2,
            label="Observations",
        )
        skew.ax.set_ylim(1050, 10)
        skew.ax.set_xlim(-90, 45)

        skew.plot_dry_adiabats(linewidth=0.5)
        skew.plot_moist_adiabats(linewidth=0.5)

        skew.ax.legend()
        skew.ax.set_xlabel("Temperature [$\\degree$C]", fontsize=14)
        skew.ax.set_ylabel("Pressure [hPa]", fontsize=14)
        if title:
            skew.ax.set_title(title)

    def _add_profile(self, skew, model):
        """Add one model profile to the skew-T diagram."""
        line = skew.plot(
            model.lev,
            model.Tatm - climlab.constants.tempCtoK,
            label=model.name,
            linewidth=2,
        )[0]
        skew.plot(
            1000,
            model.Ts - climlab.constants.tempCtoK,
            "o",
            markersize=8,
            color=line.get_color(),
        )
        skew.ax.legend()

    def render(self):
        """Render the environment according to the configured mode."""
        if self.render_mode == "human":
            self._render_frame()
            plt.show()
        elif self.render_mode == "rgb_array":
            fig = self._render_frame()
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape((height, width, 3))
            plt.close(fig)
            return image


def run_rce(seed):
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
    envs = gym.vector.SyncVectorEnv(
        [make_env(RadiativeConvectiveModelEnv, seed, NUM_STEPS)]
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
    run_rce(seed)
