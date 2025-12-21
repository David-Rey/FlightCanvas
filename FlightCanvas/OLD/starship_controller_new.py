
from typing import Optional, Union, Tuple
import control as ct
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('default')


class StarshipController:
    def __init__(self, pitch_sys):
        self.A = pitch_sys.A
        self.B = pitch_sys.B
        self.C = pitch_sys.C
        self.D = pitch_sys.D
        self.pitch_sys_ss = pitch_sys
        #self.pitch_feedback = ct.feedback(self.pitch_sys, -1)

    def compute_lqr(self, q_weights=None, r_weight=1.0) -> np.ndarray:
        """
        Computes LQR gain matrix K to stabilize the system.
        :param q_weights: List of 4 weights for the state error [u, w, q, theta]
        :param r_weight: Scalar weight for control effort (flap deflection)
        """
        # Default weights: prioritizing stabilizing Pitch (index 3) and Pitch Rate (index 2)
        if q_weights is None:
            # [u, w, q, theta]
            q_weights = [0.1, 0.1, 10.0, 10.0]

        Q = np.diag(q_weights)
        R = np.array([[r_weight]])

        # Compute the LQR gain matrix K
        # K is the matrix such that u = -Kx stabilizes (A - BK)
        K, S, E = ct.lqr(self.A, self.B, Q, R)

        # Create the closed-loop state-space system
        sys_cl = ct.ss(self.A - self.B @ K, self.B, self.C, self.D)

        print("LQR Gain Matrix K:", K)
        print("Closed-Loop Poles with LQR:", ct.poles(sys_cl))

        return K

    """
    def pzplot_closedloop(self):
        #response = ct.pole_zero_map(self.pitch_feedback)
        #ct.pole_zero_plot(response)
        ct.root_locus_map(self.pitch_feedback).plot()
        plt.show()

    def pzplot_openloop(self):
        response = ct.pole_zero_map(self.pitch_sys)
        ct.pole_zero_plot(response)
        plt.show()

    def formulate_PID(self):
        Kp = 1.0
        Ki = 0.5
        Kd = 0.5

        # Create the transfer function components
        s = ct.TransferFunction.s
        C = Kp + Ki / s + Kd * s

        T = ct.feedback(C * self.pitch_sys, 1)
        #ct.root_locus_map(T).plot()
        poles = ct.poles(T)

        print("System Poles:")
        print(poles)
    """


