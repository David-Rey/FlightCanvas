from FlightCanvas.components.aero_component import AeroComponent

from typing import List, Union, Tuple
import numpy as np
import casadi as ca
from FlightCanvas import utils

from casadi import Function


class VehicleDynamics:
    """
    Vehicle dynamics class
    TODO
    """

    full_dynamics: ca.Function

    def __init__(
            self,
            mass: float,
            moi: np.ndarray,
            components: List[AeroComponent],
            control_mapping):

        self.mass = mass
        self.moi = moi
        self.components = components
        self.control_mapping = control_mapping
        self.num_control_inputs = 0
        self.num_actuator_inputs_comp = len(components)
        if self.control_mapping is not None:
            self.num_control_inputs = len(control_mapping)
            self.allocation_matrix = self.create_allocation_matrix()

        self.create_casadi_model()

    def compute_forces_and_moments(
            self,
            state: Union[np.ndarray, ca.MX],
            true_deflections: Union[np.ndarray, ca.MX],
    ) -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
        """
        Computes the aerodynamic forces and moments on the vehicle. This function
        is type-aware and will use either NumPy or CasADi based on the input type.
        :param state: The current state of the vehicle (position, velocity, quaternion, angular_velocity)
        :param true_deflections: The command deflection angle
        :return: The computed forces and moments
        """
        is_casadi = isinstance(state, (ca.SX, ca.MX))

        if is_casadi:
            F_b = ca.MX.zeros(3, 1)
            M_b = ca.MX.zeros(3, 1)
        else:
            F_b = np.zeros(3)
            M_b = np.zeros(3)

        # For each component, look up the forces and moments based on its local flow conditions
        for i in range(len(self.components)):
            component = self.components[i]
            true_deflection = true_deflections[i]

            F_b_comp, M_b_comp = component.get_forces_and_moments(state, true_deflection)
            F_b += F_b_comp
            M_b += M_b_comp

        return F_b, M_b

    def _calculate_rigid_body_derivatives(
            self,
            state: Union[np.ndarray, ca.MX],
            deflections_true: Union[np.ndarray, ca.MX],
            g: Union[np.ndarray, ca.MX]) \
            -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
        """
        Calculates the rigid body derivatives given a state and deflections
        param state: The state to calculate the derivatives for
        param deflections_true: The true deflection of the flaps
        param g: The gravitational force acting on the body
        :return: The rigid body derivatives
        """

        # Extract quaterion and angular rate from state
        quat = state[6:10]
        omega_B = state[10:13]

        # check if state is a casadi object
        is_casadi = isinstance(state, (ca.MX, ca.SX))

        # if casadi use casadi functions, else use numpy
        if is_casadi:
            inv_func = ca.inv
            cross_func = ca.cross
            array_func = ca.MX
            dir_cosine_func = utils.dir_cosine_ca
            omega_matrix_func = utils.omega_ca
        else:
            inv_func = np.linalg.inv
            cross_func = np.cross
            array_func = np.array
            dir_cosine_func = utils.dir_cosine_np
            omega_matrix_func = utils.omega

        # Calculate external forces and moments as function
        F_B, M_B = self.compute_forces_and_moments(state, deflections_true)

        # compute direction cosine matrix
        C_B_I = dir_cosine_func(quat)   # INERTIAL frame to BODY frame.
        #C_I_B = C_B_I.T                 # BODY frame to INERTIAL frame.

        # Rotate Gravity into Body Frame
        g_body = C_B_I @ g

        # Body velocity
        v_body = state[3:6]

        # The Coriolis Term
        coriolis_accel = cross_func(omega_B, v_body)

        # force in inertial frame
        #F_I = (C_I_B @ F_B) + self.mass * g

        #F_B =

        # calculate inertial acceleration
        v_dot = (F_B / self.mass) + g_body - coriolis_accel

        # get moment of inertia
        J_B = array_func(self.moi)

        # angular rate calculation
        quat_dot = 0.5 * (omega_matrix_func(omega_B) @ quat)  # + quat_dot_correction

        # angular acceleration based on conservation of momentum
        omega_dot = inv_func(J_B) @ (M_B - cross_func(omega_B, J_B @ omega_B))

        return v_dot, omega_dot, quat_dot

    def create_casadi_model(self):
        """
        TODO
        """

        # Define Symbolic State and Dynamics
        pos_I = ca.MX.sym('pos_I', 3)
        vel_I = ca.MX.sym('vel_I', 3)
        quat = ca.MX.sym('quat', 4)
        omega_B = ca.MX.sym('omega_B', 3)

        # Define Symbolic Controls
        control_deflections = ca.MX.sym('control_deflections', self.num_actuator_inputs_comp)
        g = ca.MX.sym('g', 3)

        # concat state into a single variable
        state = ca.vertcat(pos_I, vel_I, quat, omega_B)

        # calculate x_dot
        v_dot, omega_dot, quat_dot = self._calculate_rigid_body_derivatives(state, control_deflections, g)

        state_dot = ca.vertcat(vel_I, v_dot, quat_dot, omega_dot)

        # create casadi function of dynamics
        self.full_dynamics = Function('dynamics', [state, control_deflections, g], [state_dot])

    def dynamics(self, state: np.ndarray, control_inputs: np.ndarray, gravity=True):
        if self.full_dynamics is None:
            self.create_casadi_model()
        if gravity:
            return self.full_dynamics(state, control_inputs, np.array([0, 0, -9.81]))
        else:
            return self.full_dynamics(state, control_inputs, np.array([0, 0, 0]))

    def create_allocation_matrix(self) -> np.ndarray:
        """
        Creates the starship_control allocation matrix for the vehicle.
        """
        # Get a sorted list of high-level command names for consistent column ordering.
        command_names = self.control_mapping.keys()
        command_to_col = {name: i for i, name in enumerate(command_names)}

        comp_lookup = {comp.name: comp for comp in self.components}
        comp_lookup_by_index = {comp.name: i for i, comp in enumerate(self.components)}

        # Initialize the matrix. Rows correspond to the individual actuator inputs,
        # and columns correspond to the high-level starship_control commands.
        allocation_matrix = np.zeros(
            (self.num_actuator_inputs_comp, self.num_control_inputs)
        )

        # Populate the matrix using the defined mapping.
        for command_name, component_map in self.control_mapping.items():
            # Get the column index for the current high-level command.
            col_idx = command_to_col[command_name]

            for actuator_name, gain in component_map.items():
                # Find the actuator's data (including its control_index) using its name.
                if actuator_name in comp_lookup:
                    row_idx = comp_lookup_by_index[actuator_name]
                    allocation_matrix[row_idx, col_idx] += gain
                else:
                    # Optional but recommended: A warning for names that don't match.
                    print(f"Warning: Actuator '{actuator_name}' in starship_control mapping not found in components.")

        return allocation_matrix

