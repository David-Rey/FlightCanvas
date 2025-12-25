import numpy as np
import control as ct
from control import TransferFunction
from typing import Optional, Union
from scipy.signal import cont2discrete


class ActuatorDynamics:
    def __init__(self, num: Union[np.ndarray, list], den: Union[np.ndarray, list], num_actuators: int, init_state=0):
        self.tfs = []
        self.num_actuators = num_actuators
        for i in range(num_actuators):
            self.tfs.append(FCTransferFunction(num, den, init_state=init_state))

    def c2d(self, dt: float):
        for i in range(self.num_actuators):
            self.tfs[i] = self.tfs[i].c2d(dt)

    def update_deflections(self, deflections: Union[np.ndarray, list]) -> np.ndarray:
        true_deflections = np.zeros(self.num_actuators)
        for i in range(self.num_actuators):
            true_deflections[i] = self.tfs[i].update(deflections[i])
        return true_deflections


class FCTransferFunction(TransferFunction):
    """
    Extension of python-starship_control's TransferFunction with:
    - Continuous-to-discrete conversion
    - Explicit time-domain update via a difference equation
    """

    def __init__(self, num: Union[np.ndarray, list], den: Union[np.ndarray, list], init_state=0):
        super().__init__(num, den)

        #self.u_hist = np.repeat(init_state, len(self.num[0][0]))
        #self.y_hist = np.repeat(init_state, len(self.den[0][0]) - 1)

        self.u_hist = np.zeros(len(self.num[0][0]))
        self.y_hist = np.zeros(len(self.den[0][0]) - 1)

    def c2d(self, Ts: float, method='zoh') -> "FCTransferFunction":
        """
        Discretize the continuous-time transfer function
        """
        sys = (self.num[0][0], self.den[0][0])
        numd, dend, _ = cont2discrete(sys, Ts, method, None)
        numd = numd.flatten()
        dend = dend.flatten()
        fctf = FCTransferFunction(numd, dend)
        fctf.dt = Ts
        return fctf

    def update(self, u_k: float) -> float:
        """
        Advance the discrete-time system by one step using:
        y[k] = b0 u[k] + b1 u[k-1] + ... - (a1 y[k-1] + a2 y[k-2] + ...)
        """

        if self.dt == 0:
            raise RuntimeError("System must be discretized before calling update().")

        # Update input history
        if self.u_hist.size:
            self.u_hist[1:] = self.u_hist[:-1]
        self.u_hist[0] = u_k

        # Compute output
        feedforward = np.dot(self.num[0][0], self.u_hist)
        feedback = np.dot(self.den[0][0][1:], self.y_hist)
        y_k = feedforward - feedback

        # Update output history
        if self.y_hist.size:
            self.y_hist[1:] = self.y_hist[:-1]
        self.y_hist[0] = y_k

        return y_k


