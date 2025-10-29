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
