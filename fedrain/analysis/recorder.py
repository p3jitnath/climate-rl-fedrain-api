"""Recorder utility for appending episodic returns to CSV files.

This small helper encapsulates the file handling previously implemented in
``BaseAlgorithm``. It provides a simple API for creating a timestamped CSV
and appending rows, and for resuming (warm-starting) from an existing CSV.
"""

import csv
import json
import logging
import os

from fedrain.utils import setup_logger


class Recorder:
    """Append episodic-return rows to a CSV file.

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
        self.record_dir = os.path.join(record_dir or os.environ.get("TMPDIR", "/tmp"))
        self.record_dir = os.path.join(self.record_dir, "records")
        os.makedirs(self.record_dir, exist_ok=True)

        self.logger = setup_logger("recorder", level=level)
        self.logger.info(
            f"Initialised recorder for algorithm '{self.algorithm['name']}' in directory: {self.record_dir}"
        )

        self.record_algorithm()

    def record_algorithm(self):
        """Record the algorithm configuration to a JSON file in the record directory."""
        config_path = os.path.join(self.record_dir, "algorithm_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.algorithm, f, indent=4)
        self.logger.info(f"Recorded algorithm configuration to {config_path}")

    def record_episodic_return(self, episodic_return, global_step, idx=None):
        """Append a row (algorithm, global_step, episodic_return) to the CSV.

        The first call creates a new timestamped CSV in ``record_dir`` (or
        the recorder's configured directory / TMPDIR) and writes a header.
        Subsequent calls append rows.
        """
        if idx is not None:
            csv_path = os.path.join(self.record_dir, f"returns_{idx}.csv")
        else:
            csv_path = os.path.join(self.record_dir, "returns.csv")
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
