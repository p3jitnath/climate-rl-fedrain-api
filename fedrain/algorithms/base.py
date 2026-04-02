"""Base classes for RL algorithms used by FedRAIN.

This module defines the minimal abstract interface expected from concrete
algorithm implementations used in the project.
"""


class BaseAlgorithm:
    """Base class for reinforcement-learning algorithms used in FedRAIN.

    This abstract base class defines the minimal public interface expected
    from any RL algorithm implementation used by the FedRAIN project. Concrete
    subclasses should implement the :meth:`predict` and :meth:`update`
    methods.

    Notes
    -----
    The base class does not prescribe specific argument types for
    ``predict`` or ``update``; those are defined by each concrete
    algorithm implementation (for example, they may accept NumPy arrays,
    PyTorch tensors, or dictionaries of tensors).

    """

    def __init__(self):
        """Initialize the base algorithm.

        Subclasses that override ``__init__`` should call ``super().__init__()``
        to ensure proper initialization of the base class.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Base class has no state by default; subclasses may initialize
        # algorithm-specific attributes here.
        pass

    def predict(self, *args, **kwargs):
        """Compute action(s) or prediction(s) given input data.

        This is an abstract method and must be implemented by subclasses.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to the concrete implementation
            (semantics depend on the subclass, e.g. observations, states).
        **kwargs : dict
            Keyword arguments forwarded to the concrete implementation.

        Returns
        -------
        object
            Model predictions. The concrete return type depends on the
            algorithm (for example, ``numpy.ndarray``, ``torch.Tensor``,
            or a dictionary of arrays/tensors).

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.

        """
        raise NotImplementedError("predict() must be implemented by subclass.")

    def update(self, *args, **kwargs):
        """Update the algorithm's parameters using provided training data.

        This is an abstract method and must be implemented by subclasses.
        Implementations typically perform a single training/update step or
        a batch of updates and may optionally return training statistics
        (for example, loss values or metrics).

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to the concrete implementation
            (e.g. data batches, gradients).
        **kwargs : dict
            Keyword arguments forwarded to the concrete implementation.

        Returns
        -------
        None or object
            Implementations may return training information (such as a loss
            scalar or dict of metrics). The base implementation does not
            return anything.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.

        """
        raise NotImplementedError("update() must be implemented by subclass.")

    def inference(self):
        """Perform inference using the algorithm.

        This method can be called to switch the algorithm into inference mode.
        which may involve setting internal flags or adjusting behavior to disable training-specific features (such as exploration noise or gradient updates). The exact behavior of this method depends on the concrete implementation in subclasses.
        """
        self.train = False
        self.algorithm["train"] = False
        self.logger.info("Switched to inference mode")

    @staticmethod
    def check_episode_termination(infos):
        """Check for episode termination signals from the environment.

        This method inspects the provided ``infos`` dictionary for termination
        signals (e.g., ``final_info``) and can be used to handle any cleanup
        or checkpoint saving before exiting.

        Parameters
        ----------
        infos : dict
            Dictionary containing environment information, typically returned
            by the environment's ``step`` method.

        Returns
        -------
        bool
            True if the episode has terminated, False otherwise.

        """
        return "final_info" in infos
