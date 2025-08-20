# aero_project/FlightCanvas/aero_vehicle.py

import pathlib
from typing import List, Tuple, Union

import aerosandbox.numpy as np
import casadi as ca
import pyvista as pv
from scipy.integrate import solve_ivp

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver
except ImportError:
    print("Warning: 'acados_template' not installed. Related functionality will be unavailable.")
    AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver = None, None, None, None, None

# Local application imports
from FlightCanvas import utils
from FlightCanvas.components.aero_component import AeroComponent
from FlightCanvas.open_loop_control import OpenLoopControl


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
        self.control_mapping = None
        self.allocation_matrix = None
        self.ca_u = None

        self.acados_model = None
        self.acados_path = '/home/david/Desktop/main/acados'  # NOTE: This will change from system to system

        self.vehicle_path = f'vehicle_saves/{self.name}'

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
        Sets the control mapping for the vehicle
        :param control_mapping: A dictionary that maps each component to its control
        """
        self.control_mapping = control_mapping

        # Get sorted lists of commands (for columns) and component names (for rows)
        command_names = control_mapping.keys()
        actuator_names = [comp.name for comp in self.components]

        # Create helper dictionaries to map names to matrix indices
        actuator_to_row = {name: i for i, name in enumerate(actuator_names)}
        command_to_col = {name: i for i, name in enumerate(command_names)}

        # Initialize a zero matrix with the correct dimensions
        num_actuators = len(actuator_names)
        num_commands = len(command_names)
        self.allocation_matrix = np.zeros((num_actuators, num_commands))

        # Populate the matrix with gains from the input dictionary
        for command, component_map in control_mapping.items():
            col_idx = command_to_col[command]
            for actuator_name, gain in component_map.items():
                if actuator_name in actuator_to_row:
                    row_idx = actuator_to_row[actuator_name]
                    self.allocation_matrix[row_idx, col_idx] = gain
                else:
                    # Warn if a name in the mapping doesn't match a component
                    print(
                        f"Warning: Actuator '{actuator_name}' in control mapping "
                        f"does not correspond to any component name."
                    )
        # Set up casadi control variables
        self.ca_u = ca.MX.sym('control', num_commands)

    def compute_forces_and_moments(
        self,
        state: Union[np.ndarray, ca.MX],
        true_deflections: Union[np.ndarray, ca.MX]) \
    -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
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

    def set_control(self, control: np.ndarray):
        """
        Sets the deflection angle for one or more control surfaces
        :param control: A list of corresponding control deflection angles in radians
        """

        # Get deflections based on allocation matrix
        deflections = self.allocation_matrix @ control

        # Get the sorted list of component names to serve as keys.
        actuator_names = [comp.name for comp in self.components]

        # Create a dictionary mapping each component name to its calculated deflection.
        deflection_map = {
            name: deflection for name, deflection in zip(actuator_names, deflections)
        }

        # Iterate through all components and apply their calculated deflection.
        for component in self.components:
            # Check if this component is a controllable surface.
            if component.name in deflection_map:
                # Get the specific deflection for this component from the map.
                rotation_command = deflection_map[component.name]

                # Update the component's transformation matrix with the new rotation.
                component.update_transform(rotation=rotation_command)

    def run_sim(
        self,
        pos_0: np.ndarray,
        vel_0: np.ndarray,
        quat_0: np.ndarray,
        omega_0: np.ndarray,
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
            return self._run_sim_casadi(pos_0, vel_0, quat_0, omega_0, tf, dt, gravity, open_loop_control)
        else:
            # Call the SciPy/NumPy-specific simulation function
            return self._run_sim_scipy(pos_0, vel_0, quat_0, omega_0, tf, dt, gravity, print_debug, open_loop_control)

    def _run_sim_scipy(
        self,
        pos_0: np.ndarray,
        vel_0: np.ndarray,
        quat_0: np.ndarray,
        omega_0: np.ndarray,
        tf: float,
        dt: float,
        gravity: bool,
        print_debug: bool,
        open_loop_control: OpenLoopControl
    ) -> Tuple[np.ndarray, np.ndarray,  np.ndarray]:
        """
        Runs a 6DoF simulation using SciPy's adaptive-step ODE solver.
        """
        g = np.array([0, 0, -9.81]) if gravity else np.array([0, 0, 0])

        def dynamics_6dof(t: float, state: np.ndarray) -> np.ndarray:
            vel_I = state[3:6]
            quat = state[6:10]
            omega_B = state[10:13]

            if print_debug:
                print(f"Time: {t:.2f} s")

            cmd_deflections = np.zeros(len(self.components))
            if open_loop_control is not None:
                control = open_loop_control.get_u(t)
                cmd_deflections = self.allocation_matrix @ control

            F_B, M_B = self.compute_forces_and_moments(state, cmd_deflections)
            C_I_B = utils.dir_cosine_np(quat).T
            F_I = (C_I_B @ F_B) + self.mass * g
            v_dot = F_I / self.mass

            J_B = np.array(self.moi)
            omega_dot = np.linalg.inv(J_B) @ (M_B - np.cross(omega_B, J_B @ omega_B))
            quat_dot = 0.5 * utils.omega(omega_B) @ quat
            return np.concatenate((vel_I, v_dot, quat_dot, omega_dot))

        state_0 = np.concatenate((pos_0, vel_0, quat_0, omega_0))
        t_span = (0, tf)

        num_points = int(tf / dt) + 1

        # Create the array of time points for the solver to output
        t_eval = np.linspace(t_span[0], t_span[1], num_points)

        solution = solve_ivp(dynamics_6dof, t_span, state_0, t_eval=t_eval, rtol=1e-5, atol=1e-5)

        # Get control
        u_values = np.empty(0)
        if open_loop_control is not None:
            u_values = np.array([open_loop_control.get_u(t) for t in solution['t']]).T

        return solution['t'], solution['y'], u_values

    def _get_deflection_states(self) -> Tuple[ca.MX, ca.MX, ca.MX]:
        """
        TODO
        """
        nd = self.allocation_matrix.shape[0]
        nu = self.allocation_matrix.shape[1]

        deflections = ca.MX.zeros(nd)
        deflections_dot = ca.MX.zeros(nd)

        u = ca.MX.sym('u', nu)
        cmd_deflections = self.allocation_matrix @ u

        for i in range(len(self.components)):
            component = self.components[i]
            if component.actuator_model is not None:
                deflection_state_size = component.actuator_model.state_size

                # Sanitize the name for CasADi (replace spaces with underscores)
                safe_name = component.name.replace(' ', '_')
                true_deflection = ca.MX.sym(safe_name, deflection_state_size)
                cmd_deflection = cmd_deflections[i]

                # Append the new symbolic Functino to the list
                deflection_derivative = component.actuator_model.get_casadi_expression(cmd_deflection, true_deflection)

                deflections[i] = true_deflection
                deflections_dot[i] = deflection_derivative

                # Append the new symbolic variable to the list

                #deflection_list.append(true_deflection)
                #deflection_dot_list.append(deflection_derivative)


        # After the loop, vertically stack all collected variables
        #if deflection_list:  # Check if the list is not empty
        #    deflections = ca.vertcat(*deflection_list)
        #    deflections_dot = ca.vertcat(*deflection_dot_list)
        #else:
        #    deflections = ca.MX([])
        #    deflections_dot = ca.MX([])

        return deflections, deflections_dot, u


    def _create_acados_model(
        self,
        gravity: bool,
        open_loop_control: OpenLoopControl
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
        true_deflections, deflections_dot, u = self._get_deflection_states()

        state = ca.vertcat(pos_I, vel_I, quat, omega_B, true_deflections)
        model.x = state
        model.u = u
        nx = state.shape[0]

        #true_deflections = np.zeros(len(self.components))
        #if deflections.numel() > 0:
        #    u = ca.MX.sym('u', self.allocation_matrix.shape[1])
        #    true_deflections = self.allocation_matrix @ u
        #    model.u = u

        # if open_loop_control is not None:
        #    u = ca.MX.sym('u', open_loop_control.num_inputs)
        #    cmd_deflections = self.allocation_matrix @ u
        #    model.u = u

        F_B, M_B = self.compute_forces_and_moments(state, true_deflections)
        C_I_B = utils.dir_cosine_ca(quat).T
        F_I = (C_I_B @ F_B) + self.mass * g
        v_dot = F_I / self.mass

        J_B = ca.MX(self.moi)
        omega_dot = ca.inv(J_B) @ (M_B - ca.cross(omega_B, J_B @ omega_B))
        quat_dot = 0.5 * (utils.omega_ca(omega_B) @ quat)
        f_expl_expr = ca.vertcat(vel_I, v_dot, quat_dot, omega_dot, deflections_dot)

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
        tf: float,
        dt: float,
        gravity: bool,
        open_loop_control: OpenLoopControl
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs a 6DoF simulation using a fixed-step CasADi RK4 integrator.
        """

        if self.acados_model is None:
            self._create_acados_model(gravity, open_loop_control)

        N_sim = int(tf / dt) + 1

        sim = AcadosSim(acados_path=self.acados_path)

        sim.model = self.acados_model
        sim.solver_options.T = dt

        sim.solver_options.integrator_type = 'IRK'
        sim.solver_options.num_stages = 3
        sim.solver_options.num_steps = 3
        sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"

        nx = sim.model.x.rows()
        nu = sim.model.u.rows()

        acados_integrator = AcadosSimSolver(sim)

        x0 = np.concatenate((pos_0, vel_0, quat_0, omega_0))

        sim_x = np.zeros((N_sim + 1, nx))
        sim_x[0, :] = x0
        sim_u = np.zeros((N_sim + 1, nu))
        sim_u[0, :] = open_loop_control.get_u(0)
        sim_t = np.zeros((N_sim + 1))

        import time
        start_time = time.perf_counter()
        for i in range(N_sim):
            u_current = open_loop_control.get_u(sim_t[i])
            sim_x[i + 1, :] = acados_integrator.simulate(x=sim_x[i, :], u=u_current)
            sim_u[i + 1, :] = u_current
            sim_t[i + 1] = (i + 1) * dt
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print("Simulation finished.")
        print(f"Total time for {N_sim} nodes: {elapsed_time:.4f} seconds.")

        #u_values = np.empty(0)
        #if open_loop_control is not None:
        #    u_values = np.array([open_loop_control.get_u(t) for t in sim_t]).T

        return sim_t, sim_x.T, sim_u.T

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
        video_filename = '../animation.mp4'

        # start movie
        self.pl.open_movie(video_filename, framerate=fps, quality=9)

        for i in range(num_frames):
            sim_time = dt * i
            state, control = utils.interp_state(t_arr, x_arr, u_arr, sim_time)
            state[0] = -state[0]

            cmd_deflection = np.zeros(len(self.components))
            if control is not None:
                cmd_deflection = self.allocation_matrix @ control

            # Update actors with interpolated state
            self.update_actors(state, cmd_deflection)
            if debug:
                self.update_debug(state, cmd_deflection)

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
