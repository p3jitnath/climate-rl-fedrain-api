import tensorflow as tf  # For GLIBC issues: export LD_LIBRARY_PATH="${CONDA_PREFIX:-$(conda info --base 2>/dev/null)/envs/${CONDA_DEFAULT_ENV:-base}}/lib${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
from tensorflow.core.util import event_pb2


def retrieve_tfrecord_data(algo, tfrecord_path):
    data = {}
    seed = int(tfrecord_path.split("/")[-2].split("__")[-2])
    data[seed] = []
    episodic_idx = 0
    serialized_events = tf.data.TFRecordDataset(tfrecord_path)
    for serialized_example in serialized_events:
        e = event_pb2.Event.FromString(serialized_example.numpy())
        for v in e.summary.value:
            if v.HasField("simple_value") and v.tag == "charts/episodic_return":
                episodic_idx += 1
                data[seed].append(
                    {"episode": episodic_idx, "episodic_return": v.simple_value}
                )
    return data
