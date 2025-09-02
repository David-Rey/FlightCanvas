# aero_project/FlightCanvas/aero_vehicle.py

import pathlib
from typing import List, Tuple, Union

import aerosandbox.numpy as np

from FlightCanvas.actuators.actuator_dynamics import ActuatorDynamics
from FlightCanvas.vehicle.vehicle_dynamics import VehicleDynamics

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver
except ImportError:
    print("Warning: 'acados_template' not installed. Related functionality will be unavailable.")
    AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver = None, None, None, None, None

# Local application imports
from FlightCanvas.components.aero_component import AeroComponent
from FlightCanvas.control.open_loop_control import OpenLoopControl


class AeroVehicle:
    """
    Represents a complete aerodynamic vehicle composed of various FlightCanvas.
    Manages collective mesh generation and visualization.
    """

    def __init__(
            self,
            name: str,
            xyz_ref: Union[np.ndarray, List[float]],
            components: List[AeroComponent]
    ):
        """
        Initializes the AeroVehicle instance
        :param name: The name of the vehicle (e.g., "MyDrone")
        :param xyz_ref: The reference point [x, y, z] for the vehicle, typically the CG
        :param components: A list of AeroComponent instances that comprise the vehicle
        """
        self.name = name
        self.xyz_ref = np.array(xyz_ref)
        self.mass = 10
        self.moi = self.mass * np.eye(3)
        self.components = components

        self.actuator_dynamics = None

        self.vehicle_dynamics = None

        self.controller = None

        self.vehicle_path = f'vehicle_saves/{self.name}'

        for i in range(len(self.components)):
            self.components[i].update_id(i)

        [comp.set_parent(self) for comp in self.components]

        path_object = pathlib.Path(self.vehicle_path)
        path_object.mkdir(parents=True, exist_ok=True)

        # Crates a buildup manager for each component
        self.init_buildup_manager()

        # Update transformation matrices for all components
        self.update_transform()

        self.simX = None
        self.simU = None
        self.simT = None

    def update_transform(self):
        """
        Update transformation matrices for all components
        """
        [comp.update_transform() for comp in self.components]

    def set_mass(self, mass: float):
        """
        Sets the mass of the vehicle
        """
        self.mass = mass

    def set_moi_factor(self, moi_factor: float):
        """
        Sets the mass moment of inertia
        """
        self.moi = moi_factor * self.mass * np.eye(3)

    def set_moi_diag(self, moi_diag: Union[np.ndarray, List[float]]):
        """
        Sets the mass moment of inertia
        """
        self.moi = np.diag(np.array(moi_diag)) * self.mass

    def set_control_mapping(self, control_mapping: dict):
        """
        Sets control mapping and creates actuator dynamics
        :param control_mapping: A mapping of actuator names to component
        """
        self.actuator_dynamics = ActuatorDynamics(self.components, control_mapping)

    def init_vehicle_dynamics(self):
        """
        Creates dynamics in the form x_dot = f(t, x, u)
        """
        self.vehicle_dynamics = VehicleDynamics(self.mass, self.moi, self.components, self.actuator_dynamics)

    def run_sim(
        self,
        pos_0: np.ndarray,
        vel_0: np.ndarray,
        quat_0: np.ndarray,
        omega_0: np.ndarray,
        delta_0: np.ndarray,
        tf: float,
        dt: float = 0.02,
        gravity: bool = True,
        casadi: bool = True,
        print_debug: bool = False,
        open_loop_control: OpenLoopControl = None
    ):
        """
        Runs simulation of 6 Degree of Freedom model with no control
        :param pos_0: The initial position [x, y, z] (m)
        :param vel_0: The initial velocity [x, y, z] (m/s)
        :param quat_0: The initial quaternion [q0, q1, q2, q3]
        :param omega_0: The initial omega [x, y, z] (rad/s)
        :param delta_0: The initial flap deflection [fl, fr, al, af] (rad)
        :param tf: The time of simulation (s)
        :param dt: The fixed time step for the integrator
        :param gravity: Boolean for active gravity
        :param casadi: If True, uses the CasADi integrator; otherwise, uses SciPy
        :param print_debug: Boolean for printing debugging information
        :param open_loop_control: Open loop control object that commands the aero vehicle
        """
        self.simT, self.simX, self.simU = self.vehicle_dynamics.run_sim(pos_0, vel_0, quat_0, omega_0, delta_0, tf, dt, gravity, casadi, print_debug, open_loop_control)

    def run_mpc(
        self,
        pos_0: np.ndarray,
        vel_0: np.ndarray,
        quat_0: np.ndarray,
        omega_0: np.ndarray,
        delta_0: np.ndarray
    ):
        """
        Runs MPC with feedback control from acados
        :param pos_0: The initial position [x, y, z] (m)
        :param vel_0: The initial velocity [x, y, z] (m/s)
        :param quat_0: The initial quaternion [q0, q1, q2, q3]
        :param omega_0: The initial omega [x, y, z] (rad/s)
        :param delta_0: The initial flap deflection [fl, fr, al, af] (rad)
        """

        Nsim = self.controller.Nsim
        nx = self.controller.nx
        nu = self.controller.nu

        self.simX = np.zeros((nx, Nsim + 1))
        self.simU = np.zeros((nu, Nsim))
        self.simT = np.zeros(Nsim)

        state = np.concatenate((pos_0, vel_0, quat_0, omega_0, delta_0))
        N = self.controller.Nsim
        for i in range(N):
            t, state, u = self.controller.compute_control_input(i, state)
            self.simX[:, i] = state
            self.simU[:, i] = u
            self.simT[i] = t

    def init_buildup_manager(self):
        """
        Crates a buildup manager for each component
        """
        for component in self.components:
            component.init_buildup_manager(self.vehicle_path, component)

    def compute_buildup(self):
        """
        Computes the aerodynamic buildup data for all 'prime' FlightCanvas
        """
        print("Computing buildup data...")
        for component in self.components:
            component.compute_buildup()

    def save_buildup(self):
        """
        Saves the aerodynamic buildup data for all 'prime' FlightCanvas
        """
        print("Saving buildup data...")
        for component in self.components:
            component.save_buildup()

    def save_buildup_fig(self):
        """
        Saves the aerodynamic buildup figures for all 'prime' FlightCanvas
        """
        print("Saving buildup figures...")
        for component in self.components:
            component.save_buildup_figs()

    def load_buildup(self):
        """
        Loads the aerodynamic buildup data for all 'prime' FlightCanvas
        """
        print('Loading buildup data...')
        for component in self.components:
            component.load_buildup()

    def generate_mesh(self):
        """
        Generate the mesh for all FlightCanvas and applies their local translation
        """
        for component in self.components:
            component.generate_mesh()

    def get_control_history(self):
        """
        Return the complete control history for post-processing
        """
        return self.simT, self.simX, self.simU

    def test_new_buildup(self):

        comp = self.components[1]
        F_b, M_b = comp.buildup_manager.get_forces_and_moments(0.05, 0.05, 50, 0, 0, 0)
        print(f"F_b {F_b}")
        print(f"M_b {M_b}")


