# aero_project/FlightCanvas/aero_vehicle.py

from typing import List, Union, Tuple
import pyvista as pv
import aerosandbox.numpy as np
from FlightCanvas import utils
import pathlib
from scipy.integrate import solve_ivp

from FlightCanvas.components.aero_component import AeroComponent


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

    def compute_forces_and_moments_lookup(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the aerodynamic forces and moments on the vehicle by looking up
        pre-computed data for each component
        :param state: The current state of the vehicle (position, velocity, quaternion, angular_velocity)
        """

        F_b = np.zeros(3)
        M_b = np.zeros(3)

        # For each component, look up the forces based on its local flow conditions
        for component in self.components:
            F_b_comp, M_b_comp = component.get_forces_and_moment_lookup(state)
            F_b += F_b_comp
            M_b += M_b_comp

        return F_b, M_b

    def set_control(self, surface_name: List[str], control: List[float]):
        """
        Sets the deflection angle for one or more control surfaces
        :param surface_name: A list of names of the FlightCanvas to actuate
        :param control: A list of corresponding control deflection angles in radians
        """
        for i, surface_name in enumerate(surface_name):
            # Find the component with the matching name
            for component in self.components:
                if component.name == surface_name:
                    # Update the component's transformation matrix with the new rotation
                    component.update_transform(rotation=control[i])
                    break  # Move to the next surface name once found

    def run_sim(self, pos_0, vel_0, quat_0, omega_0, tf, N=500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs simulation of 6 Degree of Freedom model with no control
        :param: pos_0: The initial position [x, y, z] (m)
        :param: vel_0: The initial velocity [x, y, z] (m/s)
        :param: quat_0: The initial quaternion [q0, q1, q2, q3]
        :param: omega_0: The initial omega [x, y, z] (rad/s)
        :param: tf: The time of simulation (s)
        :param: N: The number of simulation steps
        :return: The time and state for every simulation step
        """
        # Initial state
        state_0 = np.concatenate((pos_0, vel_0, quat_0, omega_0))

        # Time span for the simulation
        t_span = (0, tf)  # Simulate for 10 seconds
        t_eval = np.linspace(t_span[0], t_span[1], N)  # Time points for output
        solution = solve_ivp(self.dynamics_6DOF, t_span, state_0, t_eval=t_eval, rtol=1e-5, atol=1e-5)
        return t_eval, solution['y']

    def dynamics_6DOF(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Propagates 6 Degree of Freedom dynamics and returns the derivative of the state vector (x_dot)
        :param t: The current time
        :param state: The current state of the vehicle (position, velocity, quaternion, angular_velocity)
        :return: The derivative of the state vector (x_dot)
        """
        pos_I = state[:3]  # Position in the inertial frame
        vel_I = state[3:6]  # Velocity in the inertial frame
        quat = state[6:10]  # Orientation as a quaternion
        omega_B = state[10:13]  # Angular velocity in the body frame

        print(f"  Time {t}\n")

        # Gravity in the inertial frame
        g = np.array([0, 0, -9.81])
        #g = np.array([0, 0, 0])

        # Forces and moments (in the body frame)
        F_B, M_B = self.compute_forces_and_moments_lookup(state)

        C_B_I = utils.dir_cosine_np(quat)  # From body to internal
        C_I_B = C_B_I.transpose()

        F_I = (C_I_B @ F_B) + self.mass * g

        # Equations of motion for a 6DoF object
        v_dot = F_I / self.mass
        J_B = np.array(self.moi)
        omega_dot = np.linalg.inv(J_B) @ (M_B - np.cross(omega_B, J_B @ omega_B))
        quat_dot = 0.5 * utils.omega(omega_B) @ quat
        return np.concatenate((vel_I, v_dot, quat_dot, omega_dot))

    def print_states(self, t_arr: np.ndarray, x_arr: np.ndarray):
        """
        :param t_arr: The time array
        :param x_arr: The state array
        """
        for i in range(len(t_arr)):
            t = t_arr[i]
            state = x_arr[:, i]
            pos_I = state[:3]  # Position in the inertial frame
            vel_I = state[3:6]  # Velocity in the inertial frame
            quat = state[6:10]  # Orientation as a quaternion
            omega_B = state[10:13]  # Angular velocity in the body frame

            print(f"  Time {t}")
            print(f"  Position (Inertial): {pos_I}")
            print(f"  Velocity (Inertial): {vel_I}")
            print(f"  Quaternion: {quat}")
            print(f"  Angular Velocity (Body): {omega_B}")
            print("\n")

    def init_buildup_manager(self):
        """
        Crates a buildup manager for each component
        """
        for component in self.components:
            component.init_buildup_manager(self.vehicle_path)

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

    def update_actors(self, state: np.ndarray):
        """
        Updates PyVista actors for all FlightCanvas
        """
        [comp.update_actor(state) for comp in self.components]

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

    def update_debug(self, state: np.ndarray):
        """
        Updates debug visuals for the vehicle and its FlightCanvas given the current state
        :param state: The current state of the vehicle
        """
        [comp.update_debug(state) for comp in self.components]

    def animate(self, t_arr: np.ndarray, x_arr: np.ndarray, debug=False):
        """
        Animates the aerodynamic visuals for all FlightCanvas
        :param t_arr: The time array
        :param x_arr: The state array
        :param debug: If true, draws debug visuals
        """
        grid = pv.Plane(
            center=(0, 0, 0),  # Center of the plane
            direction=(0, 0, 1),  # Normal vector, perpendicular to the XY plane
            i_size=2000,  # Width of the grid in X direction
            j_size=2000,  # Height of the grid in Y direction
            i_resolution=20,  # Number of subdivisions along X
            j_resolution=20  # Number of subdivisions along Y
        )
        self.pl.add_mesh(grid, color="white", show_edges=True, edge_color="black")

        fps = 30
        dt = 1 / fps
        num_frames = int(np.floor(t_arr[-1] * fps))

        video_filename = '../animation.mp4'

        self.pl.open_movie(video_filename, framerate=fps, quality=9)

        for i in range(num_frames):
            sim_time = dt * i
            state = utils.interp_state(t_arr, x_arr, sim_time)
            state[0] = -state[0]

            # Update actors with interpolated state
            self.update_actors(state)
            if debug:
                self.update_debug(state)

            quat = state[6:10]
            C_B_I = utils.dir_cosine_np(quat)
            pos = state[:3] + (C_B_I @ self.xyz_ref)
            self.pl.camera.focal_point = pos
            cam_offset = 60 * np.array([-1, 1, 1])
            self.pl.camera.position = pos + cam_offset

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


# print(f"  Time {t}")
# print(f"  Position (Inertial): {pos_I}")
# print(f"  Velocity (Inertial): {vel_I}")
# print(f"  Quaternion: {quat}")
# print(f"  Angular Velocity (Body): {omega_B}")
# print("\n")

'''
def draw_buildup(self, name: str, ID: str, **kwargs):
    """
    Draws a contour plot of a specified aerodynamic coefficient from the
    buildup data
    :param name: The name of the component
    :param ID: The identifier for the data to plot (e.g., "CL", "CD", "F_b")
    :param kwargs: Additional keyword arguments for the plot
    """
    for component in self.FlightCanvas:
        if component.name == name:
            component.draw_buildup(name, ID, **kwargs)

    def dynamics(self, t, state):
        pos_I = state[:3]  # Position in the inertial frame
        vel_I = state[3:6]  # Velocity in the inertial frame
        quat = state[6:10]  # Orientation as a quaternion
        omega_B = state[10:13]  # Angular velocity in the body frame

        # Gravity in the inertial frame
        g = np.array([0, 0, -9.81])

        # Forces and moments (in the body frame)
        F_B, M_B = self.compute_forces_and_moments_lookup(quat, vel_I, omega_B)
        F_B = F_B.flatten()
        M_B = M_B.flatten()

        C_B_I = dir_cosine_np(quat)
        C_I_B = C_B_I.transpose()

        F_I = C_I_B @ F_B + self.mass_const * g

        v_dot = F_I / self.mass_const
        J_B = np.array(self.moi_const)
        omega_dot = np.linalg.inv(J_B) @ (M_B - np.cross(omega_B, J_B @ omega_B))
        quat_dot = 0.5 * omega(omega_B) @ quat
        return np.concatenate((vel_I, v_dot, quat_dot, omega_dot))

    def test_remove_me(self):
        # Initial state
        vel = np.array([10, 0, 0])  # Initial velocity
        quat = utils.euler_to_quat((0, 0, 0))
        omega = np.array([0, 0, 0])  # Initial angular velocity

        F_B, M_B = self.compute_forces_and_moments_lookup(quat, vel, omega, self.xyz_ref)
        print(f"F_B: {F_B}")
        print(f"M_B: {M_B}")

    def draw(self, debug=False):
        """
        Prepares the vehicle for rendering in the PyVista plotter
        :param debug: If True, enables debug visuals like axes and reference points
        """
        #pos_0 = np.array([0, 0, 0])  # Initial position
        #vel_0 = np.array([0, 0, 0])  # Initial velocity
        #quat_0 = utils.euler_to_quat((0, 0, 0))
        #omega_0 = np.array([0, 0, 0])  # Initial angular velocity
        #state = np.concatenate((pos_0, vel_0, quat_0, omega_0))

        # Ensure all actors are updated with their latest transforms
        #self.update_actors()

        if debug:
            self.pl.add_axes_at_origin(labels_off=True)
            self.init_debug()
     
'''

# TODO: Review this coordinate system correction.
# The negative sign suggests a mismatch between the simulation's inertial
# frame (e.g., North-East-Down) and the aerodynamic standard frame
# (forward-right-down). This needs verification.
# vel[0] = -vel[0]


# print(f"  Time {t}")
# print(f"  Position (Inertial): {pos_I}")
# print(f"  Velocity (Inertial): {vel_I}")
# print(f"  Quaternion: {quat}")
# print(f"  Angular Velocity (Body): {omega_B}")
# print("\n")
