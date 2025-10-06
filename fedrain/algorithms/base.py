class BaseAlgorithm:
    """
    Base class for RL algorithms in FedRAIN.
    Provides common interface and shared functionality.
    """

    def __init__(self):
        pass

    def predict(self, *args, **kwargs):
        raise NotImplementedError("predict() must be implemented by subclass.")

    def update(self, *args, **kwargs):
        raise NotImplementedError("update() must be implemented by subclass.")
