"""High-level FedRAIN API.

This module exposes a simple, user-facing API for constructing and
configuring FedRAIN algorithm instances. It currently provides the
``FedRAIN`` facade which can be used to instantiate supported RL
algorithms (for example, DDPG) with the appropriate constructor
arguments.

Example:
-------
>>> from fedrain.api import FedRAIN
>>> api = FedRAIN()
>>> agent = api.set_algorithm('DDPG', envs, seed=0)

Notes:
-----
The available algorithm names are case-insensitive. New algorithms can be
added by extending :meth:`FedRAIN.set_algorithm`.

"""

import os
import tempfile

from fedrain.algorithms.ddpg import DDPG


class FedRAIN:
    """Facade for constructing FedRAIN RL algorithm instances.

    The ``FedRAIN`` class is a thin convenience wrapper around concrete
    algorithm implementations. It centralises algorithm selection and keeps
    user-facing code independent of concrete classes.

    Methods
    -------
    set_algorithm(name, *args, **kwargs)
        Create and return an instance of the algorithm identified by ``name``.

    """

    def __init__(self):
        """Create a new FedRAIN API facade.

        The base implementation has no state. Subclasses or future
        enhancements may add configuration options here.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        pass

    def set_algorithm(self, name, *args, **kwargs):
        """Instantiate a supported RL algorithm by name.

        Parameters
        ----------
        name : str
            Case-insensitive algorithm identifier (e.g., ``'DDPG'``).
        *args
            Positional arguments forwarded to the chosen algorithm's
            constructor (for example, the vectorized envs instance).
        **kwargs
            Keyword arguments forwarded to the chosen algorithm's
            constructor (for example, learning rate, seed, device).

        Returns
        -------
        object
            An instance of the requested algorithm (typically a subclass of
            ``BaseAlgorithm`` from ``fedrain.algorithms``).

        Raises
        ------
        ValueError
            If the provided ``name`` does not match a supported algorithm.

        """
        if name.upper() == "DDPG":
            agent = DDPG(*args, **kwargs)
            return agent
        else:
            raise ValueError(
                f"Algorithm '{name}' is not supported by FedRAIN. Please choose a valid RL algorithm (e.g., 'DDPG')."
            )

    def save_weights(self, agent, folder=None, replay_buffer=True):
        """Save an agent's weights and optionally its replay buffer to *folder*.

        If *folder* is None a temporary directory under `$TMPDIR` (or system
        default tempdir) is created and returned.

        Returns
        -------
        str
            Path to the folder where checkpoint files were written.
        """
        if folder is None:
            base = os.getenv("TMPDIR") or None
            folder = tempfile.mkdtemp(prefix="fedrain_ckpt_", dir=base)
        os.makedirs(folder, exist_ok=True)
        if hasattr(agent, "save"):
            agent.save(folder, replay_buffer=replay_buffer)
        else:
            raise AttributeError(
                "Provided agent does not implement 'save(folder, replay_buffer)'"
            )
        return folder

    def load_weights(self, agent, folder, replay_buffer=True):
        """Load an agent's weights and optionally its replay buffer from *folder*.

        Parameters
        ----------
        agent : object
            Agent instance created by the API (must implement ``load(folder, replay_buffer)``).
        folder : str
            Path to a checkpoint directory previously produced by
            :meth:`save_weights`.
        replay_buffer : bool, optional
            If True, load the replay buffer as well (default is True).
        """
        if not os.path.isdir(folder):
            raise ValueError(f"Checkpoint folder does not exist: {folder}")
        if hasattr(agent, "load"):
            agent.load(folder, replay_buffer=replay_buffer)
        else:
            raise AttributeError(
                "Provided agent does not implement 'load(folder, replay_buffer)'"
            )
