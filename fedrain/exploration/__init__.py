"""Exploration strategies for continuous-action algorithms."""

from fedrain.exploration.independent import IndependentExploration
from fedrain.exploration.vertical import VerticallyCorrelatedExploration


def make_exploration_strategy(name, smoothing_passes=2):
    """Construct a named exploration strategy.

    Parameters
    ----------
    name : {"independent", "vertical"}
        Strategy identifier.
    smoothing_passes : int, optional
        Smoothing passes used by vertical exploration.

    Returns
    -------
    IndependentExploration or VerticallyCorrelatedExploration
        Configured exploration strategy.

    Raises
    ------
    ValueError
        If ``name`` is unknown.
    """
    strategies = {
        "independent": lambda: IndependentExploration(),
        "vertical": lambda: VerticallyCorrelatedExploration(
            smoothing_passes=smoothing_passes
        ),
    }
    try:
        return strategies[name]()
    except KeyError as error:
        raise ValueError(f"Unknown exploration strategy: {name}") from error


__all__ = [
    "IndependentExploration",
    "VerticallyCorrelatedExploration",
    "make_exploration_strategy",
]
