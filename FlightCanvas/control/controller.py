
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

class BaseController(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute_control_input(self, t: Union[int, float], state: Optional[np.ndarray]) -> np.ndarray:
        """
        given state apply control input
        """
        pass



