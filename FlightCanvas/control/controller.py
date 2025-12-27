
import numpy as np
from control import TransferFunction
from typing import Optional, Union, Tuple
from scipy.signal import cont2discrete


class Controller(TransferFunction):
    """
    Extension of python-starship_control's TransferFunction with:
    - Continuous-to-discrete conversion
    - Explicit time-domain update via a difference equation
    """

    def __init__(self, num: Union[np.ndarray, list], den: Union[np.ndarray, list], limits: Optional[Tuple[float, float]] = None):
        super().__init__(num, den)

        # Store limits (min, max)
        self.limits = limits

        self.u_hist = np.zeros(len(self.num[0][0]))
        self.y_hist = np.zeros(len(self.den[0][0]) - 1)

    def c2d(self, Ts: float, method='zoh'):
        """
        Discretize the continuous-time transfer function
        """
        sys = (self.num[0][0], self.den[0][0])
        numd, dend, _ = cont2discrete(sys, Ts, method, None)
        self.num[0][0] = numd.flatten()
        self.den[0][0] = dend.flatten()
        self.dt = Ts

        #numd = numd.flatten()
        #dend = dend.flatten()
        #fctf = Controller(numd, dend)
        #fctf.dt = Ts
        #return fctf

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
        y_k_raw = feedforward - feedback

        if self.limits is not None:
            min_val, max_val = self.limits
            y_k = np.clip(y_k_raw, min_val, max_val)
        else:
            y_k = y_k_raw

        # Update output history
        if self.y_hist.size:
            self.y_hist[1:] = self.y_hist[:-1]
        self.y_hist[0] = y_k

        return y_k