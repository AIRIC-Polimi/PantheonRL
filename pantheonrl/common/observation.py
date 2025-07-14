from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Observation:
    """
    Representation of a single observation provided by an environment.

    obs: the (partial) observation that an agent receives
    state: the full state information
    action_mask: a mask specifying what actions are legal
        - if it is None, all actions are permitted
    """

    obs: np.ndarray
    state: np.ndarray
    action_mask: Optional[np.ndarray] = None

    def __init__(self, obs: np.ndarray, state: Optional[np.ndarray] = None, action_mask: Optional[np.ndarray] = None):
        self.obs = obs
        self.state = state if state is not None else obs
        self.action_mask = action_mask


def extract_obs(observation: Observation) -> np.ndarray:
    return observation.obs


def extract_partial_obs(observation: Observation) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    return (observation.obs, observation.action_mask)
