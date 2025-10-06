from fedrain.algorithms.ddpg import DDPG


class FedRAIN:
    """
    Main API for climate RL users to interact with FedRAIN algorithms.
    """

    def __init__(self):
        pass

    def set_algorithm(self, name, *args, **kwargs):
        if name.upper() == "DDPG":
            agent = DDPG(*args, **kwargs)
            return agent
        else:
            raise ValueError(
                f"Algorithm '{name}' is not supported by FedRAIN. Please choose a valid RL algorithm (e.g., 'DDPG')."
            )
