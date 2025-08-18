# aero_project/FlightCanvas/aero_vehicle.py

from typing import List, Union, Tuple, Optional, Any
import pyvista as pv
import aerosandbox.numpy as np
from FlightCanvas import utils
import pathlib
from scipy.integrate import solve_ivp
import casadi as ca

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
        cmd_deflections: Union[np.ndarray, ca.MX]) \
    -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
        """
        Computes the aerodynamic forces and moments on the vehicle. This function
        is type-aware and will use either NumPy or CasADi based on the input type.
        :param state: The current state of the vehicle (position, velocity, quaternion, angular_velocity)
        :param cmd_deflections: The command deflection angle
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
            cmd_deflection = cmd_deflections[i]
            true_deflection = cmd_deflection  # TODO
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

    #def test_new_buildup(self):
    #    """
    #    Compares the numerical output of the NumPy and CasADi lookup functions
    #    for a given operating point to ensure they are consistent.
    #    """
    #    comp = self.components[3]
    #    alpha_deg = 10.0
    #    beta_deg = 1.0
    #    speed = 100.0
#
    #    alpha_rad = np.deg2rad(alpha_deg)
    #    beta_rad = np.deg2rad(beta_deg)
#
    #    print(f"\n--- Comparing Buildup Manager for '{comp.name}' ---")
    #    print(f"Test Point: alpha={alpha_deg:.2f} deg, beta={beta_deg:.2f} deg, speed={speed:.2f} m/s")
#
    #    # --- 1. NumPy Calculation ---
    #    # The type-aware function will automatically call the NumPy version
    #    F_b_numpy, M_b_numpy = comp.buildup_manager.get_forces_and_moments(alpha_rad, beta_rad, speed)
    #    print("\n--- NumPy Result ---")
    #    print(f"Forces: {F_b_numpy}")
    #    print(f"Moments: {M_b_numpy}")
#
    #    # --- 2. CasADi Calculation ---
    #    # Create symbolic variables as placeholders
    #    alpha_sym = ca.MX.sym('alpha')
    #    beta_sym = ca.MX.sym('beta')
    #    speed_sym = ca.MX.sym('speed')
#
    #    # Get the symbolic output expressions by calling the type-aware function with symbols
    #    F_b_sym, M_b_sym = comp.buildup_manager.get_forces_and_moments(alpha_sym, beta_sym, speed_sym)
#
    #    # Create a callable CasADi Function from the symbolic graph
    #    evaluate_casadi_lookup = ca.Function(
    #        'evaluate_casadi',
    #        [alpha_sym, beta_sym, speed_sym],
    #        [F_b_sym, M_b_sym]
    #    )
#
    #    # Call the CasADi function with the numerical inputs
    #    F_b_casadi_dm, M_b_casadi_dm = evaluate_casadi_lookup(alpha_rad, beta_rad, speed)
#
    #    # Convert CasADi's matrix type to NumPy arrays for comparison
    #    F_b_casadi = F_b_casadi_dm.full().flatten()
    #    M_b_casadi = M_b_casadi_dm.full().flatten()
#
    #    print("\n--- CasADi Result ---")
    #    print(f"Forces: {F_b_casadi}")
    #    print(f"Moments: {M_b_casadi}")

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
            return self._run_sim_casadi(pos_0, vel_0, quat_0, omega_0, tf, dt, gravity)
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

    def _run_sim_casadi(
        self,
        pos_0: np.ndarray,
        vel_0: np.ndarray,
        quat_0: np.ndarray,
        omega_0: np.ndarray,
        tf: float,
        dt: float,
        gravity: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs a 6DoF simulation using a fixed-step CasADi RK4 integrator.
        """
        g = ca.MX([0, 0, -9.81]) if gravity else ca.MX([0, 0, 0])

        # Define Symbolic State and Dynamics
        pos_I = ca.MX.sym('pos_I', 3)
        vel_I = ca.MX.sym('vel_I', 3)
        quat = ca.MX.sym('quat', 4)
        omega_B = ca.MX.sym('omega_B', 3)
        state = ca.vertcat(pos_I, vel_I, quat, omega_B)
        cmd_deflections = np.zeros(len(self.components))

        F_B, M_B = self.compute_forces_and_moments(state, cmd_deflections)
        C_I_B = utils.dir_cosine_ca(quat).T
        F_I = (C_I_B @ F_B) + self.mass * g
        v_dot = F_I / self.mass

        J_B = ca.MX(self.moi)
        omega_dot = ca.inv(J_B) @ (M_B - ca.cross(omega_B, J_B @ omega_B))
        quat_dot = 0.5 * utils.omega_ca(omega_B) @ quat
        f_expl_expr = ca.vertcat(vel_I, v_dot, quat_dot, omega_dot)

        # Create the Integrator
        ode = {'x': state, 'ode': f_expl_expr}
        integ_options = {'t0': 0, 'tf': dt, 'simplify': True, 'number_of_finite_elements': 4}
        integrator = ca.integrator('integrator', 'rk', ode, integ_options)

        # Run the Simulation Loop
        x0 = np.concatenate([pos_0, vel_0, quat_0, omega_0])
        num_steps = int(tf / dt)
        x_history = [x0]
        current_x = x0

        for _ in range(num_steps):
            res = integrator(x0=current_x)
            current_x = res['xf'].full().flatten()
            current_x[6:10] /= np.linalg.norm(current_x[6:10])  # Normalize quaternion
            x_history.append(current_x)

        t_eval = np.linspace(0, tf, num_steps + 1)
        return t_eval, np.array(x_history).T, np.empty(0)

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
