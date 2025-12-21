import numpy as np
import control as ct
from control import TransferFunction
from typing import Optional, Union
from scipy.signal import cont2discrete

'''
class TransferFunctionSystem:
    def __init__(self, num, den, Ts, method='zoh'):
        """
        Initialize the system by converting S-domain to Z-domain
        and setting up the difference equation parameters.
        """
        self.Ts = Ts

        # Perform Discretization
        num_c = np.array(num).flatten()
        den_c = np.array(den).flatten()

        # Store continuous system for reference/comparison
        self.sys_s = ct.tf(num_c, den_c)

        # Assign the DISCRETE coefficients
        self.sys_z = self.sys_s.sample(Ts, method=method)
        self.num = self.sys_z.num[0][0]
        self.den = self.sys_z.den[0][0]

        # Normalize (den[0] should already be 1.0 from scipy, but keep for robustness)
        if self.den[0] != 1.0:
            self.num = self.num / self.den[0]
            self.den = self.den / self.den[0]

        # u_hist stores [u[k], u[k-1], u[k-2]...]. Length is len(num)
        self.u_hist = np.zeros(len(self.num))

        # y_hist stores [y[k-1], y[k-2]...]. Length is len(den) - 1
        self.y_hist = np.zeros(len(self.den) - 1)


    def update(self, u_k):
        """
        Step the system forward by one sample using the Difference Equation.
        y[k] = b0*u[k] + b1*u[k-1]... - (a1*y[k-1] + a2*y[k-2]...)
        """

        # 1. Update Input History (Linear Shift)
        # Shift everything right, discard the oldest, insert new input u_k
        if len(self.u_hist) > 0:
            self.u_hist[1:] = self.u_hist[:-1]
            self.u_hist[0] = u_k  # Insert new input at the front

        # 2. Compute Output y[k]
        feedforward = np.dot(self.num, self.u_hist)

        # Feedback: Sum of a * y
        # den[1:] is [a1, a2...]
        # y_hist is [y[k-1], y[k-2]...]
        feedback = np.dot(self.den[1:], self.y_hist)

        y_k = feedforward - feedback

        # 3. Update Output History
        # Shift everything right, discard the oldest, store new output y_k
        if len(self.y_hist) > 0:
            self.y_hist[1:] = self.y_hist[:-1]
            self.y_hist[0] = y_k  # Insert current output as next step's y[k-1]

        return y_k

    def reset(self):
        """Reset internal states to zero."""
        self.u_hist.fill(0)
        self.y_hist.fill(0)
'''


class FCTransferFunction(TransferFunction):
    """
    Extension of python-starship_control's TransferFunction with:
    - Continuous-to-discrete conversion
    - Explicit time-domain update via a difference equation
    """
    def __init__(self, num: Union[np.ndarray, list], den: Union[np.ndarray, list]):
        super().__init__(num, den)

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Continuous double integrator: 1 / s^2
    Ts = 0.5
    sys_c = FCTransferFunction(num=[1], den=[1, 0, 0])
    sys_d = sys_c.c2d(Ts=Ts, method="zoh")

    sim_time = 10.0
    steps = int(sim_time / Ts)

    time = np.arange(steps) * Ts
    output = np.zeros(steps)

    u = 1.0 # step input
    for k in range(steps):
        output[k] = sys_d.update(u)

    theoretical = 0.5 * time**2

    plt.figure(figsize=(10, 6))
    plt.step(time, output, where="post", label="Discrete Simulation")
    plt.plot(time, theoretical, "r--", label="0.5 t^2 (Continuous)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.title(f"Discrete Simulation of 1/s^2 (Ts = {Ts}s)")
    plt.grid(True)
    plt.legend()
    plt.show()
