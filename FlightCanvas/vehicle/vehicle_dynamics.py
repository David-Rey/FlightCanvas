from FlightCanvas.components.aero_component import AeroComponent
from FlightCanvas.actuators.actuator_dynamics import ActuatorDynamics
from FlightCanvas.control.open_loop_control import OpenLoopControl

from typing import List, Union, Tuple, Optional
import numpy as np
import casadi as ca
from scipy.integrate import solve_ivp
from FlightCanvas import utils

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver
except ImportError:
    print("Warning: 'acados_template' not installed. Related functionality will be unavailable.")
    AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver = None, None, None, None, None


class VehicleDynamics:
    """
    Vehicle dynamics class
    TODO
    """
    def __init__(
        self,
        mass: float,
        moi: np.ndarray,
        components: List[AeroComponent],
        actuator_dynamics: ActuatorDynamics):

        self.mass = mass
        self.moi = moi
        self.components = components
        self.actuator_dynamics = actuator_dynamics

        self.acados_model = None

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs simulation of 6 Degree of Freedom model with no control
        :param pos_0: The initial position [x, y, z] (m)
        :param vel_0: The initial velocity [x, y, z] (m/s)
        :param quat_0: The initial quaternion [q0, q1, q2, q3]
        :param omega_0: The initial omega [x, y, z] (rad/s)
        :param tf: The time of simulation (s)
        :param dt: The fixed time step for the integrator
        :param gravity: Boolean for active gravity
        :param casadi: If True, uses the CasADi integrator; otherwise, uses SciPy
        :param print_debug: Boolean for printing debugging information
        :param open_loop_control: Open loop control object that commands the aero vehicle
        :return: The time and state for every simulation step
        """
        if casadi:
            # Call the CasADi-specific simulation function
            return self.run_sim_acados(pos_0, vel_0, quat_0, omega_0, delta_0, tf, dt, gravity, open_loop_control)
        else:
            # Call the SciPy/NumPy-specific simulation function
            return self.run_sim_scipy(pos_0, vel_0, quat_0, omega_0, delta_0, tf, dt, gravity, print_debug, open_loop_control)

    def run_sim_scipy(
        self,
        pos_0: np.ndarray,
        vel_0: np.ndarray,
        quat_0: np.ndarray,
        omega_0: np.ndarray,
        delta_0: np.ndarray,
        tf: float,
        dt: float,
        gravity: bool,
        print_debug: bool,
        open_loop_control: Optional[OpenLoopControl]
    ) -> Tuple[np.ndarray, np.ndarray,  np.ndarray]:
        """
        Runs a 6DoF simulation using SciPy's adaptive-step ODE solver.
        """
        g = np.array([0, 0, -9.81]) if gravity else np.array([0, 0, 0])

        num_control_inputs = 0
        if self.actuator_dynamics is not None:
            num_control_inputs = self.actuator_dynamics.num_control_inputs

        def dynamics_6dof(t: float, state: np.ndarray) -> np.ndarray:
            vel_I = state[3:6]

            if print_debug:
                print(f"Time: {t:.2f} s")

            # Get number of states and control inputs
            if self.actuator_dynamics is not None:
                deflections_state = state[13:]

                deflections_true = np.array([
                    deflections_state[i] if i is not None else 0
                    for i in self.actuator_dynamics.deflection_indices
                ])
                if open_loop_control is not None:
                    control_inputs = open_loop_control.compute_control_input(t, state)
                    deflections_state_dot = self.actuator_dynamics.get_dynamics(deflections_state, control_inputs)
                else:
                    deflections_state_dot = np.zeros(len(deflections_state))
            else:
                deflections_state_dot = np.empty(0)
                deflections_true = np.zeros(len(self.components))

            v_dot, omega_dot, quat_dot = self._calculate_rigid_body_derivatives(state, deflections_true, g)

            return np.concatenate((vel_I, v_dot, quat_dot, omega_dot, deflections_state_dot))

        if delta_0 is None:
            delta_0 = np.empty(0)
        state_0 = np.concatenate((pos_0, vel_0, quat_0, omega_0, delta_0))
        t_span = (0, tf)

        num_points = int(tf / dt) + 1

        # Create the array of time points for the solver to output
        t_eval = np.linspace(t_span[0], t_span[1], num_points)

        import time
        start_time = time.perf_counter()
        solution = solve_ivp(dynamics_6dof, t_span, state_0, t_eval=t_eval, rtol=1e-5, atol=1e-5)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print("Simulation finished.")
        print(f"Total time for {num_points} nodes: {elapsed_time:.4f} seconds.")

        # Get control
        u_values = np.zeros((num_control_inputs, num_points))
        if open_loop_control is not None:
            u_values = np.array([open_loop_control.compute_control_input(t, None) for t in solution['t']]).T

        return solution['t'], solution['y'], u_values

    def _calculate_rigid_body_derivatives(
        self,
        state: Union[np.ndarray, ca.MX],
        deflections_true: Union[np.ndarray, ca.MX],
        g: Union[np.ndarray, ca.MX])\
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
        C_I_B = dir_cosine_func(quat).T

        # force in inertial frame
        F_I = (C_I_B @ F_B) + self.mass * g

        # calculate inertial acceleration
        v_dot = F_I / self.mass

        # get moment of inertia
        J_B = array_func(self.moi)

        # angular rate calculation
        quat_dot = 0.5 * (omega_matrix_func(omega_B) @ quat) #+ quat_dot_correction

        # angular acceleration based on conservation of momentum
        omega_dot = inv_func(J_B) @ (M_B - cross_func(omega_B, J_B @ omega_B))

        return v_dot, omega_dot, quat_dot

    def run_sim_acados(
        self,
        pos_0: np.ndarray,
        vel_0: np.ndarray,
        quat_0: np.ndarray,
        omega_0: np.ndarray,
        delta_0: np.ndarray,
        tf: float,
        dt: float,
        gravity: bool,
        open_loop_control: Optional[OpenLoopControl]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs a 6DoF simulation using a fixed-step CasADi RK4 integrator.
        """

        if self.acados_model is None:
            self.create_acados_model(gravity)

        N_sim = int(tf / dt) + 1

        sim = AcadosSim()

        sim.model = self.acados_model
        sim.solver_options.T = dt

        sim.solver_options.integrator_type = 'IRK'
        sim.solver_options.num_stages = 3
        sim.solver_options.num_steps = 3
        sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"

        nx = sim.model.x.rows()
        if not sim.model.u:
            nu = 0
        else:
            nu = sim.model.u.rows()

        if delta_0 is None:
            delta_0 = np.empty(0)
        x0 = np.concatenate((pos_0, vel_0, quat_0, omega_0, delta_0))

        acados_integrator = AcadosSimSolver(sim)

        sim_x = np.zeros((N_sim + 1, nx))
        sim_x[0, :] = x0
        sim_u = np.zeros((N_sim + 1, nu))
        sim_u[0, :] = np.zeros(nu)
        if open_loop_control is not None:
            sim_u[0, :] = open_loop_control.compute_control_input(0, None)

        sim_t = np.zeros(N_sim + 1)

        import time
        start_time = time.perf_counter()
        for i in range(N_sim):
            u_current = np.zeros(nu)
            if open_loop_control is not None:
                u_current = open_loop_control.compute_control_input(sim_t[i], None)
            sim_x[i + 1, :] = acados_integrator.simulate(x=sim_x[i, :], u=u_current)
            sim_u[i + 1, :] = u_current
            sim_t[i + 1] = (i + 1) * dt
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print("Simulation finished.")
        print(f"Total time for {N_sim} nodes: {elapsed_time:.4f} seconds.")

        return sim_t, sim_x.T, sim_u.T

    def create_acados_model(
        self,
        gravity: bool,
        control: bool = True
    ):
        """
        TODO
        """

        model = AcadosModel()
        model.name = "test"

        g = ca.MX([0, 0, -9.81]) if gravity else ca.MX([0, 0, 0])

        # Define Symbolic State and Dynamics
        pos_I = ca.MX.sym('pos_I', 3)
        vel_I = ca.MX.sym('vel_I', 3)
        quat = ca.MX.sym('quat', 4)
        omega_B = ca.MX.sym('omega_B', 3)

        # Get number of states and control inputs
        if self.actuator_dynamics is not None:
            num_deflection_states = self.actuator_dynamics.deflection_state_size
            num_control_inputs = self.actuator_dynamics.num_control_inputs

            deflections_state = ca.MX.sym('deflections_state', num_deflection_states)

            deflections_true_list = [
                deflections_state[i] if i is not None else ca.MX(0)
                for i in self.actuator_dynamics.deflection_indices
            ]
            deflections_true = ca.vertcat(*deflections_true_list)

            control_inputs = ca.MX.sym('control_inputs', num_control_inputs)
            state = ca.vertcat(pos_I, vel_I, quat, omega_B, deflections_state)
            deflections_state_dot = self.actuator_dynamics.get_dynamics(deflections_state, control_inputs)
            if not control:
                deflections_state_dot = ca.MX.zeros(num_deflection_states)

            model.x = state
            model.u = control_inputs
        else:
            state = ca.vertcat(pos_I, vel_I, quat, omega_B)
            model.x = state
            deflections_state_dot = np.empty(0)
            deflections_true = np.zeros(len(self.components))

        nx = state.shape[0]

        v_dot, omega_dot, quat_dot = self._calculate_rigid_body_derivatives(state, deflections_true, g)

        f_expl_expr = ca.vertcat(vel_I, v_dot, quat_dot, omega_dot, deflections_state_dot)

        xdot = ca.MX.sym('xdot', nx, 1)
        model.xdot = xdot
        f_impl_expr = f_expl_expr - xdot

        model.f_expl_expr = f_expl_expr
        model.f_impl_expr = f_impl_expr

        self.acados_model = model


