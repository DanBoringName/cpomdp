from typing import Protocol, runtime_checkable

import numpy as np

from cpomdp.types import Belief


@runtime_checkable
class InferenceBackend(Protocol):
    """TODO: one-paragraph contract.
    Say: a backend is built from a model (front-loading happens there),
    and infer_states must stay cheap. Mention prior -> posterior.
    """

    def infer_states(
        self,
        observation: np.ndarray,
        prior: Belief,
        action: np.ndarray | None = None,
    ) -> Belief:
        """TODO: one recursive filter step — given the current belief (prior)
        and a new observation (and the action just taken), return the updated
        belief (posterior)."""
        ...       