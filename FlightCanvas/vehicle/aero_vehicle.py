# aero_project/FlightCanvas/aero_vehicle.py

import pathlib
from typing import List, Tuple, Union, Optional
import scipy.linalg

import aerosandbox.numpy as np
import casadi as ca
import pyvista as pv
from scipy.integrate import solve_ivp

from FlightCanvas.actuators.actuator_dynamics import ActuatorDynamics
from FlightCanvas.vehicle.vehicle_dynamics import VehicleDynamics

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver
except ImportError:
    print("Warning: 'acados_template' not installed. Related functionality will be unavailable.")
    AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver = None, None, None, None, None

# Local application imports
from FlightCanvas import utils
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

        # Initialize a PyVista plotter for this vehicle
        self.pl = pv.Plotter()

        # List of Debug Actors
        self.cg_sphere = None

        # Update transformation matrices for all components
        self.update_transform()


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
        return self.vehicle_dynamics.run_sim(pos_0, vel_0, quat_0, omega_0, delta_0, tf, dt, gravity, casadi, print_debug, open_loop_control)

        #if casadi:
        #    # Call the CasADi-specific simulation function
        #    return self._run_sim_casadi(pos_0, vel_0, quat_0, omega_0, delta_0, tf, dt, gravity, open_loop_control)
        #else:
        #    # Call the SciPy/NumPy-specific simulation function
        #    return self._run_sim_scipy(pos_0, vel_0, quat_0, omega_0, delta_0, tf, dt, gravity, print_debug, open_loop_control)

    def _run_sim_scipy(
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
                    control_inputs = open_loop_control.compute_control_input(t)
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
            u_values = np.array([open_loop_control.compute_control_input(t) for t in solution['t']]).T

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

    def _create_acados_model(
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

        # Constraints
        #quat_con = ca.norm_2(quat) - 1  # quaterion
        #con_h_expr = ca.vertcat(
        #    quat_con,
        #)
        #model.con_h_expr = con_h_expr
        #model.con_h_expr_0 = con_h_expr

        v_dot, omega_dot, quat_dot = self._calculate_rigid_body_derivatives(state, deflections_true, g)

        f_expl_expr = ca.vertcat(vel_I, v_dot, quat_dot, omega_dot, deflections_state_dot)

        xdot = ca.MX.sym('xdot', nx, 1)
        model.xdot = xdot
        f_impl_expr = f_expl_expr - xdot

        model.f_expl_expr = f_expl_expr
        model.f_impl_expr = f_impl_expr
        self.acados_model = model


    def _run_sim_casadi(
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
            self._create_acados_model(gravity)

        N_sim = int(tf / dt) + 1

        sim = AcadosSim()

        sim.model = self.acados_model
        sim.solver_options.T = dt

        sim.solver_options.integrator_type = 'IRK'
        sim.solver_options.num_stages = 3
        sim.solver_options.num_steps = 3
        sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"

        nx = sim.model.x.rows()
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
            sim_u[0, :] = open_loop_control.compute_control_input(0)

        sim_t = np.zeros((N_sim + 1))

        import time
        start_time = time.perf_counter()
        for i in range(N_sim):
            u_current = np.zeros(nu)
            if open_loop_control is not None:
                u_current = open_loop_control.compute_control_input(sim_t[i])
            sim_x[i + 1, :] = acados_integrator.simulate(x=sim_x[i, :], u=u_current)
            sim_u[i + 1, :] = u_current
            sim_t[i + 1] = (i + 1) * dt
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print("Simulation finished.")
        print(f"Total time for {N_sim} nodes: {elapsed_time:.4f} seconds.")

        return sim_t, sim_x.T, sim_u.T

    def run_ocp(self):
        """
        Runs a 6DoF optimal control
        """
        if self.vehicle_dynamics.acados_model is None:
            self.vehicle_dynamics.create_acados_model(True)

        N = 50
        tf = 10
        dt = tf / N
        q_ref = utils.euler_to_quat((0, 0, 25))

        pos_0 = np.array([40, 0, 1000])  # Initial position
        vel_0 = np.array([0, 0, -30])  # Initial velocity
        quat_0 = utils.euler_to_quat((0, 0, 0))
        omega_0 = np.array([0, 0, 0])  # Initial angular velocity
        delta_0 = np.deg2rad(np.array([30, 30, 20, 20]))
        x0 = np.concatenate((pos_0, vel_0, quat_0, omega_0, delta_0))

        ocp = AcadosOcp()
        ocp.model = self.vehicle_dynamics.acados_model

        nx = ocp.model.x.rows()
        nu = ocp.model.u.rows()
        ny = nx + nu
        ocp.solver_options.N_horizon = N

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        ocp.dims.nx = nx
        ocp.dims.nu = nu

        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(nx)

        ocp.cost.yref[7:10] = q_ref[1:]
        ocp.cost.yref_e[7:10] = q_ref[1:]
        ocp.cost.yref[13:17] = np.deg2rad(np.array([30, 30, 20, 20]))
        ocp.cost.yref_e[13:17] = np.deg2rad(np.array([30, 30, 20, 20]))

        Q_omega = 5e1
        Q_pos = 2e-1
        Q_quat = 2e2
        R_controls = 1e1  # Very low weight on control effort
        Q_delta = 1e2

        Q_diag = np.zeros(nx)
        Q_diag[0:2] = Q_pos
        Q_diag[7:10] = Q_quat
        Q_diag[10:13] = Q_omega  # Weight on angular velocity
        Q_diag[13:] = Q_delta
        Q_mat = np.diag(Q_diag)

        # Create diagonal R matrix for control costs
        R_mat = np.diag(np.full(nu, R_controls))
        #R_mat = np.diag(nu, R_controls))

        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.W_e = Q_mat
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:, :nu] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        max_rate_rad_s = np.deg2rad(15.0)
        min_rate_rad_s = -max_rate_rad_s

        ocp.constraints.idxbu = np.arange(nu)  # Constrain all control inputs
        ocp.constraints.lbu = np.full(nu, min_rate_rad_s)
        ocp.constraints.ubu = np.full(nu, max_rate_rad_s)

        ocp.constraints.x0 = x0
        ocp.constraints.idxbx = np.array([13, 14, 15, 16])
        ocp.constraints.lbx = np.deg2rad(np.array([5, 5, 5, 5]))
        ocp.constraints.ubx = np.deg2rad(np.array([80, 80, 80, 80]))

        #lh = np.array([0])
        #uh = np.array([0])

        #ocp.constraints.lh = lh
        #ocp.constraints.uh = uh
        #ocp.constraints.lh_0 = lh
        #ocp.constraints.uh_0 = uh

        #soft_constraint_indices = [0]  # Removed index 2
        #ocp.constraints.idxsh = np.array(soft_constraint_indices)
        #ocp.constraints.idxsh_0 = np.array(soft_constraint_indices)

        # --- Slack Variables for Soft Constraints ---
        #num_soft_constraints = len(soft_constraint_indices)
        #penalty_weight = 1e2
        #ocp.cost.zl = penalty_weight * np.ones((num_soft_constraints,))
        #ocp.cost.zu = penalty_weight * np.ones((num_soft_constraints,))
        #ocp.cost.Zl = np.zeros_like(ocp.cost.zl)
        #ocp.cost.Zu = np.zeros_like(ocp.cost.zu)

        # 5. Set Solver Options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.regularize_method = 'GERSHGORIN_LEVENBERG_MARQUARDT'
        ocp.solver_options.levenberg_marquardt = 5e-2
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.tf = tf
        #ocp.solver_options.fixed_hess = 1

        # 6. Create and Solve
        print("Creating OCP solver...")
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

        _, init_x, init_u = self._run_sim_scipy(pos_0, vel_0, quat_0, omega_0, delta_0, tf, dt, True, False, None)
        for i in range(N):
            acados_ocp_solver.set(i, "x", init_x[:, i])
            acados_ocp_solver.set(i, "u", init_u[:, i])
        acados_ocp_solver.set(N, "x", init_x[:, N])

        simX = np.zeros((N, nx))
        simU = np.zeros((N, nu))

        xcurrent = x0.copy()
        #acados_ocp_solver.reset()

        print("Solving OCP...")

        for i in range(N):
            acados_ocp_solver.set(0, "lbx", xcurrent)
            acados_ocp_solver.set(0, "ubx", xcurrent)

            for _ in range(2):
                status = acados_ocp_solver.solve()
            if status != 0:
                acados_ocp_solver.print_statistics()

            u0 = acados_ocp_solver.get(0, "u")
            xcurrent = acados_ocp_solver.get(1, "x")

            simX[i, :] = xcurrent
            simU[i, :] = u0

        cost_value = acados_ocp_solver.get_cost()
        print(f"\nFinal Cost Function Value: {cost_value}")

        time_vec = np.linspace(0, tf, N)
        return time_vec, simX.T, simU.T


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

    def init_actors(self, **kwargs):
        """
        Get PyVista actors for all FlightCanvas
        :param: Additional keyword arguments to pass to init_actor
        """
        # Generate the geometric mesh for all FlightCanvas upon initialization
        self.generate_mesh()

        [comp.init_actor(self.pl, **kwargs) for comp in self.components]

    def update_actors(self, state: np.ndarray, true_deflection: np.ndarray):
        """
        Updates PyVista actors for all FlightCanvas
        :param state: The current state of the vehicle (position, velocity, quaternion, angular_velocity)
        :param true_deflection: The true deflection angle of the aero component
        """
        for i in range(len(self.components)):
            comp = self.components[i]
            comp.update_actor(state, float(true_deflection[i]))

    def init_debug(self, size=1, label=True):
        """
        Draws debug visuals for the vehicle and its FlightCanvas
        :param size: The size of the elements in the debug window
        :param label: If true, labels of the elements in the debug window
        """
        default_sphere_radius = 0.02
        sphere_radius = default_sphere_radius * 2 ** (size - 1)

        # Draw the vehicle's own reference point (e.g., Center of Gravity)
        sphere = pv.Sphere(radius=sphere_radius * 1.5, center=self.xyz_ref)
        self.cg_sphere = self.pl.add_mesh(sphere, color='yellow', show_edges=False, label='Vehicle CG')

        # Instruct each component to draw its own debug visuals
        for component in self.components:
            component.init_debug(self.pl, self.xyz_ref, size=size, label=label)

    def update_debug(self, state: np.ndarray, true_deflection: np.ndarray):
        """
        Updates debug visuals for the vehicle and its FlightCanvas given the current state
        :param state: The current state of the vehicle
        :param true_deflection: The true deflection angle of the aero component
        """
        for i in range(len(self.components)):
            comp = self.components[i]
            comp.update_debug(state, float(true_deflection[i]))

    def animate(self, t_arr: np.ndarray, x_arr: np.ndarray, u_arr: np.ndarray, debug=False, cam_distance=5):
        """
        Animates the aerodynamic visuals for all FlightCanvas
        :param t_arr: The time array
        :param x_arr: The state array
        :param u_arr: The control array
        :param debug: If true, draws debug visuals
        :param cam_distance: The distance from the camera to center of mass
        """
        grid = pv.Plane(
            center=(0, 0, 0),  # Center of the plane
            direction=(0, 0, 1),  # Normal vector, perpendicular to the XY plane
            i_size=2000,  # Width of the grid in X direction
            j_size=2000,  # Height of the grid in Y direction
            i_resolution=20,  # Number of subdivisions along X
            j_resolution=20  # Number of subdivisions along Y
        )
        # add grid to animation
        self.pl.add_mesh(grid, color="white", show_edges=True, edge_color="black")

        # set frames per second
        fps = 30
        dt = 1 / fps

        # calculate number of frames
        num_frames = int(np.floor(t_arr[-1] * fps))

        # output location
        video_filename = '../../animation.mp4'

        # start movie
        self.pl.open_movie(video_filename, framerate=fps, quality=9)

        for i in range(num_frames):
            sim_time = dt * i
            state, control = utils.interp_state(t_arr, x_arr, u_arr, sim_time)
            state[0] = -state[0]

            #
            deflection_state = state[13:]
            true_deflection = self.actuator_dynamics.get_component_deflection(deflection_state, control)

            # Update actors with interpolated state
            self.update_actors(state, true_deflection)
            if debug:
                self.update_debug(state, true_deflection)

            # extract rotation matrix from quaterion
            quat = state[6:10]
            C_B_I = utils.dir_cosine_np(quat)

            # get center of mass position
            pos = state[:3] + (C_B_I @ self.xyz_ref)

            # center camera focal point onto center of mass
            self.pl.camera.focal_point = pos

            # set camera location
            cam_offset = cam_distance * np.array([-1, 1, 1])
            self.pl.camera.position = pos + cam_offset

            # render and write the frame to the .mp4
            self.pl.render()
            self.pl.write_frame()

        self.pl.close()

    def show(self, **kwargs):
        """
        Displays the PyVista plotter window
        :param kwargs: Additional keyword arguments for pl.show()
        """
        self.pl.add_axes_at_origin(labels_off=True)
        self.pl.show(**kwargs)
