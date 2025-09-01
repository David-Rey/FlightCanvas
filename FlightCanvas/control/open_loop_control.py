import numpy as np
import casadi as ca
from typing import List, Dict, Any, Union, Optional
import matplotlib.pyplot as plt
from FlightCanvas.control.controller import BaseController


class OpenLoopControl(BaseController):
    """
    Manages and generates open-loop control signals over time.
    """

    def __init__(self, num_inputs: int = 4):
        """
        Initializes the OpenLoopControl system.
        :param num_inputs: The total number of control inputs (e.g., u1, u2, ...).
                           This determines the size of the output vector.
        """
        super().__init__()
        self._signals: List[Dict[str, Any]] = []
        self.num_inputs = num_inputs

    def add_step(self, u_indices: Union[int, List[int]], start_time: float, value: float):
        """
        Adds a step input to one or more control channels.

        :param u_indices: The control channel index or list of indices (0 for u1, 1 for u2, etc.).
        :param start_time: The time (in seconds) when the step activates.
        :param value: The constant value of the signal after activation.
        """
        if isinstance(u_indices, int):
            u_indices = [u_indices]

        for u_index in u_indices:
            if not 0 <= u_index < self.num_inputs:
                raise ValueError(f"u_index {u_index} is out of bounds for {self.num_inputs} inputs.")

            signal = {
                "type": "step",
                "u_index": u_index,
                "start_time": start_time,
                "value": value
            }
            self._signals.append(signal)

    def add_ramp(self, u_indices: Union[int, List[int]], start_time: float, end_time: float, start_value: float,
                 end_value: float):
        """
        Adds a ramp input to one or more control channels.
        :param u_indices: The control channel index or list of indices (0 for u1, 1 for u2, etc.).
        :param start_time: The time (in seconds) when the ramp begins.
        :param end_time: The time (in seconds) when the ramp ends.
        :param start_value: The value of the signal at the start of the ramp.
        :param end_value: The value of the signal at the end of the ramp.
        """
        if isinstance(u_indices, int):
            u_indices = [u_indices]

        for u_index in u_indices:
            if not 0 <= u_index < self.num_inputs:
                raise ValueError(f"u_index {u_index} is out of bounds for {self.num_inputs} inputs.")
            if start_time >= end_time:
                raise ValueError("end_time must be greater than start_time for a ramp.")

            signal = {
                "type": "ramp",
                "u_index": u_index,
                "start_time": start_time,
                "end_time": end_time,
                "start_value": start_value,
                "end_value": end_value
            }
            self._signals.append(signal)

    def _get_u_casadi(self, t: ca.MX) -> ca.MX:
        """
        Calculates the symbolic control vector using CasADi
        """
        u = ca.MX.zeros(self.num_inputs)
        for signal in self._signals:
            u_index = signal["u_index"]

            if signal["type"] == "step":
                val = ca.if_else(t >= signal["start_time"], signal["value"], 0)
                u[u_index] += val

            elif signal["type"] == "ramp":
                start_time = signal["start_time"]
                end_time = signal["end_time"]
                start_value = signal["start_value"]
                end_value = signal["end_value"]
                duration = end_time - start_time
                progress = (t - start_time) / duration
                interp_val = start_value + progress * (end_value - start_value)
                val = ca.if_else(t < start_time, 0, ca.if_else(t >= end_time, end_value, interp_val))
                u[u_index] += val
        return u

    def _get_u_numpy(self, t: float) -> np.ndarray:
        """
        Calculates the numerical control vector using NumPy
        """
        u = np.zeros(self.num_inputs)
        for signal in self._signals:
            u_index = signal["u_index"]

            if signal["type"] == "step":
                if t >= signal["start_time"]:
                    u[u_index] += signal["value"]

            elif signal["type"] == "ramp":
                start_time = signal["start_time"]
                end_time = signal["end_time"]

                if t < start_time:
                    continue
                elif t >= end_time:
                    u[u_index] += signal["end_value"]
                else:
                    duration = end_time - start_time
                    progress = (t - start_time) / duration
                    value_range = signal["end_value"] - signal["start_value"]
                    current_value = signal["start_value"] + progress * value_range
                    u[u_index] += current_value
        return u

    def get_u(self, t: Union[float, ca.MX]) -> Union[np.ndarray, ca.MX]:
        """
        Calculates the control vector 'u' at a specific time 't'.
        :param t: The current time.
        :return: The control vector.
        """
        if isinstance(t, (ca.SX, ca.MX)):
            return self._get_u_casadi(t)
        else:
            return self._get_u_numpy(t)

    def compute_control_input(self, t: float, state: Optional[np.ndarray]) -> np.ndarray:
        return self._get_u_numpy(t)


if __name__ == '__main__':
    # Initialize a controller for 4 inputs (u1, u2, u3, u4)
    controls = OpenLoopControl(num_inputs=4)
    print("-" * 20)

    # Define the control sequence
    controls.add_ramp(u_indices=[0], start_time=3.0, end_time=5.0, start_value=0, end_value=10)  # u1
    controls.add_ramp(u_indices=[1], start_time=3.0, end_time=5.0, start_value=0, end_value=-5)  # u2
    controls.add_step(u_indices=[2, 3], start_time=7.0, value=2.5)  # u3
    print("-" * 30)

    # --- 1. Create a CasADi function from the symbolic expression ---
    t_sym = ca.MX.sym('t')
    u_symbolic = controls.get_u(t_sym)
    u_func = ca.Function('u_func', [t_sym], [u_symbolic])

    # --- 2. Generate data for plotting ---
    t_plot = np.linspace(0, 10, 500)  # Time vector for plotting

    # Evaluate numerically using the NumPy method
    u_numpy_results = np.array([controls.get_u(t) for t in t_plot]).T

    # Evaluate the CasADi function point-by-point to avoid the runtime error
    u_casadi_results = np.array([u_func(t).full().flatten() for t in t_plot]).T

    # --- 3. Plot the results for comparison ---
    fig, axs = plt.subplots(controls.num_inputs, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Open Loop Control Signals vs. Time', fontsize=16)

    for i in range(controls.num_inputs):
        axs[i].plot(t_plot, u_numpy_results[i], 'b-', label='NumPy (Direct)', linewidth=4)
        axs[i].plot(t_plot, u_casadi_results[i], 'r--', label='CasADi (Symbolic)', linewidth=2)
        axs[i].set_ylabel(f'u{i + 1}')
        axs[i].grid(True)
        axs[i].legend()

    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

