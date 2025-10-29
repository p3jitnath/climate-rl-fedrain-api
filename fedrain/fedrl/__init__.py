"""FedRL helpers: small utilities for federated weight exchange.

This subpackage provides the :class:`FedRL` helper used to publish and
retrieve flattened actor weights via a Redis-like tensor store.
"""

from .fedrl import FedRL

__all__ = ["FedRL"]
