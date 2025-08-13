import numpy as np
from typing import List
from abc import ABC, abstractmethod
import casadi as ca
import matplotlib.pyplot as plt


class ActuatorModel(ABC):
    """
    Abstract base class for an actuator model.
    """

    def __init__(self, state_size: int) -> None:
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
    def get_derivative(self, state: np.ndarray, command: float) -> np.ndarray:
        """
        Calculates the derivative of the state vector for NumPy-based simulation.
        :param state: The current state vector of the actuator
        :param command: The control input command
        :return: The derivative of the state vector
        """
        pass

    @abstractmethod
    def get_casadi_expression(self, u: ca.MX) -> ca.Function:
        """
        Creates a CasADi symbolic function for the actuator dynamics.
        :return: A CasADi function representing the actuator's dynamics  # delta_state: ca.MX,
        """
        pass


class FirstOrderDeflection(ActuatorModel):
    """
    First-Order Deflection Actuator Model
    """
    def __init__(self, time_constant: float) -> None:
        if time_constant <= 0:
            raise ValueError("Time constant must be positive.")
        super().__init__(state_size=1)
        self.tau = time_constant


    def get_derivative(self, state: np.ndarray, command: float) -> np.ndarray:
        deflection = state[0]
        deflection_dot = (command - deflection) / self.tau
        return np.array([deflection_dot])

    def get_casadi_expression(self, u: ca.MX) -> ca.Function:
        """
        Creates a CasADi symbolic function for the actuator dynamics
        :param u: The control input command
        :return: The derivative of the state vector of the actuator
        """
        x_dot = (u - self.deflection_state[0]) / self.tau
        return ca.Function('first_order_deflection_actuator', [self.deflection_state, u], [x_dot])


if __name__ == "__main__":
    # --- 1. Simulation Setup ---
    # Define simulation parameters
    sim_time_total = 10.0  # Total simulation time in seconds
    dt = 0.05  # Time step in seconds
    time_constant = 0.08  # Actuator time constant (tau)

    # Create an instance of the actuator model
    actuator = FirstOrderDeflection(time_constant=time_constant)
    actuator.set_name("first-order-deflection-1")

    # Define the ODE problem structure for CasADi
    ca_x = ca.MX.sym('x', actuator.state_size)
    ca_u = ca.MX.sym('u')

    # --- 2. CasADi Integrator Setup ---
    # Get the symbolic expression for the actuator dynamics
    actuator_dynamics = actuator.get_casadi_expression(ca_x, ca_u)

    ode = {'x': ca_x, 'p': ca_u, 'ode': actuator_dynamics(ca_x, ca_u)}

    # Create a CasADi integrator
    # We use the 'cvodes' plugin, which is a robust variable-step solver.
    # Note: Passing 'tf' in the options may raise a deprecation warning in some
    # CasADi versions, but is required for compatibility to avoid a runtime error.
    integrator_options = {'tf': dt, 'reltol': 1e-6, 'abstol': 1e-6}
    integrator = ca.integrator('F', 'cvodes', ode, integrator_options)

    # --- 3. Simulation Loop ---
    # Set initial conditions
    current_state = np.array([0.0])  # Initial deflection is 0

    # Create a history to store results for plotting
    time_steps = np.arange(0, sim_time_total, dt)
    state_history = [current_state]
    command_history = []

    # Run the simulation
    for t in time_steps:
        # Define a more complex command signal (a sine wave starting at t=1s)
        if t < 1.0:
            command = 0.0
        else:
            # Sine wave: Amplitude=10, Frequency=0.25Hz, Offset=10
            # This makes the command oscillate between 0 and 20
            command = 10.0 + 10.0 * np.sin(2 * np.pi * 0.25 * (t - 1.0))

        command_history.append(command)

        # Use the integrator to find the state at the next time step
        result = integrator(x0=current_state, p=command)
        current_state = result['xf'].full().flatten()

        # Store the new state
        state_history.append(current_state)

    # Convert history to a plottable format (remove the last state to match time vector length)
    state_history_np = np.array(state_history[:-1])

    # --- 4. Plotting Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the actuator's deflection
    ax.plot(time_steps, state_history_np[:, 0], label='Actuator Deflection (State)', color='royalblue', linewidth=2.5)

    # Plot the command signal
    ax.plot(time_steps, command_history, label='Command Signal (Sine Wave)', color='coral', linestyle='--',
            dashes=(5, 5))

    # Formatting the plot
    ax.set_title('First-Order Actuator Response to Sine Wave Command', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Deflection (degrees)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set plot limits for better visualization
    ax.set_ylim(-2, 22)
    ax.set_xlim(0, sim_time_total)

    plt.tight_layout()
    plt.show()