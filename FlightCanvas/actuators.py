import numpy as np
from typing import List
from abc import ABC, abstractmethod
import casadi as ca


class ActuatorModel(ABC):
    """
    Abstract base class for an actuator model.
    """

    def __init__(self, state_size: int):
        self.state_size = state_size

    @abstractmethod
    def get_derivative(self, state: np.ndarray, command: float) -> np.ndarray:
        """
        Calculates the derivative of the state vector for NumPy-based simulation.
        :param state: The current state vector of the actuator
        :param command: The control input command
        :return: The derivative of the state vector
        """
        pass

    @abstractmethod
    def get_casadi_expression(self) -> ca.Function:
        """
        Creates a CasADi symbolic function for the actuator dynamics.
        :return: A CasADi function representing the actuator's dynamics
        """
        pass


class FirstOrderDeflection(ActuatorModel):
    """
    First-Order Deflection Actuator Model
    """
    def __init__(self, time_constant: float, initial_deflection: float = 0.0):
        if time_constant <= 0:
            raise ValueError("Time constant must be positive.")
        super().__init__(state_size=1)
        self.tau = time_constant

    def get_derivative(self, state: np.ndarray, command: float) -> np.ndarray:
        deflection = state[0]
        deflection_dot = (command - deflection) / self.tau
        return np.array([deflection_dot])

    def get_casadi_expression(self) -> ca.Function:
        x = ca.MX.sym('x', self.state_size)
        u = ca.MX.sym('u')
        x_dot = (u - x[0]) / self.tau
        return ca.Function('first_order_deflection_actuator', [x, u], [x_dot], ['x', 'u'], ['x_dot'])