import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
import casadi as ca
import matplotlib.pyplot as plt


class ActuatorModel(ABC):
    """
    Abstract base class for an actuator model.
    """

    def __init__(self, state_size: int):
        """
        Initialize the actuator model base class
        :param state_size: size of actuator state space (either 0, 1 or 2)
        """
        self.state_size = state_size
        self.deflection_state = ca.MX.sym('delta_state', self.state_size)
        self.name = None

    def set_name(self, name: str):
        """
        Sets the name of the actuator model
        :param name: The name of the actuator model
        """
        self.name = name

    @abstractmethod
    def get_system_matrix(self) -> Optional[np.ndarray]:
        """
        Gets the system matrix (A matrix) for the actuator dynamics
        :return: The system matrix with size, state_size x state_size
        """
        pass

    @abstractmethod
    def get_control_matrix(self) -> Optional[np.ndarray]:
        """
        Gets the control matrix (B matrix) for the actuator dynamics
        :return: The control matrix with size, state_size x 1
        """
        pass

    def get_casadi_dynamics(self, state: ca.MX, command: ca.MX) -> ca.MX:
        """
        Constructs the symbolic state-space dynamics (Ax + Bu) for CasADi.
        :param state: The symbolic state vector (ca.MX).
        :param command: The symbolic control input (ca.MX).
        :return: The CasADi expression for the state derivative.
        """
        A = ca.MX(self.get_system_matrix())
        B = ca.MX(self.get_control_matrix())
        return A @ state + B @ command


class ZeroOrderDeflection(ActuatorModel):
    """
    Zero order deflection actuator model
    """
    def __init__(self):
        super().__init__(state_size=0)

    def get_system_matrix(self) -> None:
        """
        No system matrix for zero order deflection actuator
        """
        return None

    def get_control_matrix(self) -> None:
        """
        No control matrix for zero order deflection actuator
        """
        return None


class FirstOrderDeflection(ActuatorModel):
    """
    First-Order Deflection Actuator Model
    """
    def __init__(self, time_constant: float):
        if time_constant <= 0:
            raise ValueError("Time constant must be positive.")
        super().__init__(state_size=1)
        self.tau = time_constant

    def get_system_matrix(self) -> np.ndarray:
        """
        Gets the system matrix (A matrix) for the 2nd order actuator dynamics.
        """
        return np.array([-1 / self.tau])

    def get_control_matrix(self) -> np.ndarray:
        return np.array([1 / self.tau])


class DirectDerivative(ActuatorModel):
    """
    Direct derivative actuator model
    """
    def __init__(self):
        super().__init__(state_size=1)

    def get_system_matrix(self) -> np.ndarray:
        """
        Gets the system matrix (A matrix) for the Direct Derivative actuator dynamics
        """
        return np.array([0])

    def get_control_matrix(self) -> np.ndarray:
        """
        Gets the control matrix (B matrix) for the Direct Derivative actuator dynamics
        """
        return np.array([1])

class SecondOrderDeflection(ActuatorModel):
    """
    Second-Order Deflection Actuator Model
    """
    def __init__(self, natural_frequency: float, damping_ratio: float):
        if natural_frequency <= 0:
            raise ValueError("Natural frequency must be positive.")
        if damping_ratio <= 0:
            raise ValueError("Damping ratio must be positive.")
        super().__init__(state_size=2)
        self.wn = natural_frequency
        self.zeta = damping_ratio

    def get_system_matrix(self) -> Optional[np.ndarray]:
        """
        Gets the system matrix (A matrix) for the 2nd order actuator dynamics
        """
        return np.array([
            [0, 1],
            [-self.wn**2, -2 * self.zeta * self.wn]
        ])

    def get_control_matrix(self) -> Optional[np.ndarray]:
        """
        Gets the control matrix (B matrix) for the 2nd order actuator dynamics.
        """
        return np.array([0, self.wn**2])



if __name__ == "__main__":
    # --- 1. Simulation Setup ---
    # Define simulation parameters
    sim_time_total = 10.0  # Total simulation time in seconds
    dt = 0.05  # Time step in seconds

    # --- 2. Actuator Instantiation ---
    # Create an instance of the First-Order actuator model
    actuator_fo = FirstOrderDeflection(time_constant=0.2)
    actuator_fo.set_name("first-order-deflection")

    # Create an instance of the Second-Order actuator model
    # A damping ratio of ~0.7 is often a good balance (fast with minimal overshoot)
    actuator_so = SecondOrderDeflection(natural_frequency=10.0, damping_ratio=0.7)
    actuator_so.set_name("second-order-deflection")

    # --- 3. CasADi Integrator Setup ---
    # -- First-Order Integrator --
    ca_x_fo = ca.MX.sym('x_fo', actuator_fo.state_size)
    ca_u_fo = ca.MX.sym('u_fo')
    dynamics_fo = actuator_fo.get_casadi_dynamics(ca_x_fo, ca_u_fo)
    ode_fo = {'x': ca_x_fo, 'p': ca_u_fo, 'ode': dynamics_fo}
    integrator_options = {'tf': dt}
    integrator_fo = ca.integrator('F_fo', 'cvodes', ode_fo, integrator_options)

    # -- Second-Order Integrator --
    ca_x_so = ca.MX.sym('x_so', actuator_so.state_size)
    ca_u_so = ca.MX.sym('u_so')
    dynamics_so = actuator_so.get_casadi_dynamics(ca_x_so, ca_u_so)
    ode_so = {'x': ca_x_so, 'p': ca_u_so, 'ode': dynamics_so}
    integrator_so = ca.integrator('F_so', 'cvodes', ode_so, integrator_options)

    # --- 4. Simulation Loop ---
    # Set initial conditions
    current_state_fo = np.array([0.0])  # Initial deflection is 0
    current_state_so = np.array([0.0, 0.0])  # Initial deflection and rate are 0

    # Create history lists to store results for plotting
    time_steps = np.arange(0, sim_time_total, dt)
    state_history_fo = [current_state_fo]
    state_history_so = [current_state_so]
    command_history = []

    # Run the simulation
    for t in time_steps:
        # Define a command signal (a step input followed by a sine wave)
        if t < 1.0:
            command = 0.0
        elif t < 2.0:
            command = 15.0  # Step command
        else:
            # Sine wave: Amplitude=10, Frequency=0.25Hz, Offset=10
            command = 10.0 + 10.0 * np.sin(2 * np.pi * 0.25 * (t - 2.0))

        command_history.append(command)

        # Integrate First-Order model
        result_fo = integrator_fo(x0=current_state_fo, p=command)
        current_state_fo = result_fo['xf'].full().flatten()
        state_history_fo.append(current_state_fo)

        # Integrate Second-Order model
        result_so = integrator_so(x0=current_state_so, p=command)
        current_state_so = result_so['xf'].full().flatten()
        state_history_so.append(current_state_so)

    # Convert histories to a plottable format
    state_history_fo_np = np.array(state_history_fo[:-1])
    state_history_so_np = np.array(state_history_so[:-1])

    # --- 5. Plotting Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot the command signal
    ax.plot(time_steps, command_history, label='Command Signal', color='gray', linestyle=':', dashes=(4, 4),
            linewidth=2)

    # Plot the first-order actuator's deflection
    ax.plot(time_steps, state_history_fo_np[:, 0], label=f'1st-Order ($\\tau$={actuator_fo.tau})', color='royalblue',
            linewidth=2.5)

    # Plot the second-order actuator's deflection (first element of its state)
    ax.plot(time_steps, state_history_so_np[:, 0],
            label=f'2nd-Order ($\\omega_n$={actuator_so.wn}, $\\zeta$={actuator_so.zeta})', color='coral',
            linewidth=2.5)

    # Formatting the plot
    ax.set_title('Comparison of First-Order and Second-Order Actuator Response', fontsize=16, weight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Deflection (degrees)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set plot limits for better visualization
    ax.set_ylim(-5, 25)
    ax.set_xlim(0, sim_time_total)

    plt.tight_layout()
    plt.show()