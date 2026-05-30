"""Recorder utility for appending episodic returns to CSV files.

This small helper encapsulates the file handling previously implemented in
``BaseAlgorithm``. It provides a simple API for creating a timestamped CSV
and appending rows, and for resuming (warm-starting) from an existing CSV.
"""

import csv
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch

from fedrain.utils import setup_logger


class StepRecorder:
    """Recorder for step data (observations, actions, rewards) during training.

    This class accumulates step data in memory and saves it to a file when
    requested. It can be used to record detailed step information for debugging
    or analysis purposes.

    Parameters
    ----------
    record_dir : str
        Directory where step data files will be saved. Each file will be named with the global step number (e.g., "step_1000.pth").
    """

    def __init__(self, record_dir):
        self.record_dir = record_dir
        self.reset()

    def _clear(self):
        """Clear accumulated step data."""
        self.global_steps = []
        self.obs = []
        self.next_obs = []
        self.actions = []
        self.rewards = []

    def reset(self):
        """Reset step data by clearing all accumulated values."""
        self._clear()

    def add(self, global_step, obs, next_obs, actions, rewards):
        """Add step data to the recorder.

        Parameters
        ----------
        global_step : int
            Global step number.
        obs : array-like
            Observation data.
        next_obs : array-like
            Next observation data.
        actions : array-like
            Action data.
        rewards : array-like
            Reward data.
        """
        self.global_steps.append(global_step)
        self.obs.append(obs)
        self.next_obs.append(next_obs)
        self.actions.append(actions)
        self.rewards.append(rewards)

    def save(self, global_step, actor, episodic_return):
        """Save accumulated step data to a file.

        Parameters
        ----------
        global_step : int
            Global step number.
        actor : torch.nn.Module
            Actor network model.
        episodic_return : float
            Episodic return value.
        """
        torch.save(
            {
                "global_steps": np.array(self.global_steps).squeeze(),
                "obs": np.array(self.obs).squeeze(),
                "next_obs": np.array(self.next_obs).squeeze(),
                "actions": np.array(self.actions).squeeze(),
                "rewards": np.array(self.rewards).squeeze(),
                "actor": actor.state_dict(),
                "episodic_return": episodic_return,
            },
            f"{self.record_dir}/step_{global_step}.pth",
        )
        self.reset()


class Recorder:
    """Append episodic-return rows to a CSV file and record step data.

    Parameters
    ----------
    algorithm : dict
        A dictionary containing algorithm configuration details (for example,
        name, hyperparameters) to be recorded in a JSON file alongside the
        episodic returns CSV.
    record_dir : str or None
        Directory where new timestamped CSV files will be created. If ``None``
        the environment ``TMPDIR`` is used, falling back to ``/tmp``.
    level : int, optional
        Logging level for the recorder's logger (default is logging.DEBUG).
    """

    def __init__(self, algorithm, record_dir=None, level=logging.DEBUG):
        self._file_path = None
        self.algorithm = algorithm
        self.record_dir = os.path.join(
            record_dir or os.environ.get("TMPDIR", "/tmp/climaterl/records")
        )
        os.makedirs(self.record_dir, exist_ok=True)

        self.logger = setup_logger("recorder", level=level)
        self.logger.info(
            f"Initialised recorder for algorithm '{self.algorithm['name']}' in directory: {self.record_dir}"
        )

        self.timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )  # timestamp for file naming
        self.sr = StepRecorder(self.record_dir)
        self.sr.reset()
        self.record_algorithm()

    def record_algorithm(self):
        """Record the algorithm configuration to a JSON file in the record directory."""
        config_path = os.path.join(
            self.record_dir, f"algorithm_config_{self.timestamp}.json"
        )
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.algorithm, f, indent=4)
        self.logger.info("Recorded algorithm configuration to %s" % config_path)

    def record_episodic_return(self, episodic_return, global_step, idx=None):
        """Append a row (algorithm, global_step, episodic_return) to the CSV.

        The first call creates a new timestamped CSV in ``record_dir`` (or
        the recorder's configured directory / TMPDIR) and writes a header.
        Subsequent calls append rows.
        """
        if idx is not None:
            csv_path = os.path.join(
                self.record_dir, f"returns_{self.timestamp}_{idx}.csv"
            )
        else:
            csv_path = os.path.join(self.record_dir, f"returns_{self.timestamp}.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["algorithm", "global_step", "episodic_return"])

        with open(csv_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([self.algorithm["name"], global_step, episodic_return])

        log_msg = f"""Recorded episodic return: {episodic_return} at global step: {global_step} to {csv_path}"""
        if idx is not None:
            log_msg = f"idx={idx:03d} | {log_msg}"
        self.logger.debug(log_msg)
