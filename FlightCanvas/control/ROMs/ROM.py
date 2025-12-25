
from abc import ABC, abstractmethod
import numpy as np

class ROM(ABC):
    @abstractmethod
    def x_bar(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def rom(self, state: np.ndarray) -> np.ndarray:
        pass
