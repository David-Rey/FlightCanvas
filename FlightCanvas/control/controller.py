
from abc import ABC, abstractmethod

import numpy as np

class BaseController(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute_control_input(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        given state apply control input
        """
        pass



