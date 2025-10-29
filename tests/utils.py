"""Test utilities for reading precomputed TFRecord test artifacts.

This module contains helpers used by integration tests to extract episodic
return values from TensorFlow event (TFRecord) files produced during
reference training runs.
"""

import tensorflow as tf  # For GLIBC issues: export LD_LIBRARY_PATH="${CONDA_PREFIX:-$(conda info --base 2>/dev/null)/envs/${CONDA_DEFAULT_ENV:-base}}/lib${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
from tensorflow.core.util import event_pb2


def retrieve_tfrecord_data(algo, tfrecord_path, max_episodes=None):
    """Parse a TFRecord file and extract episodic return values.

    Parameters
    ----------
    algo : str
        Algorithm name (used for future extension; currently unused).
    tfrecord_path : str
        Path to a TensorFlow ``events``/TFRecord file containing summary
        events produced during training.
    max_episodes : int or None, optional
        If provided, parsing will stop after extracting this many episodic
        return entries.

    Returns
    -------
    dict
        Mapping from seed -> list of dictionaries with keys ``episode`` and
        ``episodic_return`` extracted from the event file.
    """
    data = {}
    seed = int(tfrecord_path.split("/")[-2].split("__")[-2])
    data[seed] = []
    episodic_idx = 0
    flag = False
    serialized_events = tf.data.TFRecordDataset(tfrecord_path)
    for serialized_example in serialized_events:
        e = event_pb2.Event.FromString(serialized_example.numpy())
        for v in e.summary.value:
            if v.HasField("simple_value") and v.tag == "charts/episodic_return":
                episodic_idx += 1
                data[seed].append(
                    {"episode": episodic_idx, "episodic_return": v.simple_value}
                )
                if max_episodes and episodic_idx == max_episodes:
                    flag = True
                    break
        if max_episodes and flag:
            break
    return data
