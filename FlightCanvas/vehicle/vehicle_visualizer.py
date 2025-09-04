
from FlightCanvas.vehicle.aero_vehicle import AeroVehicle
import pyvista as pv
from pyvista import Text, TextProperty
from typing import Optional
import numpy as np
from FlightCanvas import utils


class VehicleVisualizer:
    def __init__(self, vehicle: AeroVehicle):
        self.vehicle = vehicle

        self.cg_sphere = None

        self.pl = pv.Plotter()

        # Text elements
        self.stage_text = None
        self.info_text = None
        self.state_vec_text = None

        # Text display settings
        self.text_prop = TextProperty(font_size=20, color='black', justification_horizontal='left', justification_vertical='top')
        self.stage_text_prop = TextProperty(font_size=20, color='black', justification_horizontal='right', justification_vertical='top')

    def init_text(self):
        """
        Initialize text elements for the visualization
        """
        # Init Text elements at different screen positions
        self.info_text = Text("", [20, 720], prop=self.text_prop)
        self.state_vec_text = Text("", [600, 720], prop=self.stage_text_prop)

        # Add to plot
        self.pl.add_actor(self.info_text)
        self.pl.add_actor(self.state_vec_text)

    def init_actors(self, **kwargs):
        """
        Get PyVista actors for all FlightCanvas
        :param: Additional keyword arguments to pass to init_actor
        """
        # Generate the geometric mesh for all FlightCanvas upon initialization
        self.vehicle.generate_mesh()

        [comp.init_actor(self.pl, **kwargs) for comp in self.vehicle.components]

    def update_actors(self, state: np.ndarray, true_deflection: Optional[np.ndarray]):
        """
        Updates PyVista actors for all FlightCanvas
        :param state: The current state of the vehicle (position, velocity, quaternion, angular_velocity)
        :param true_deflection: The true deflection angle of the aero component
        """
        for i in range(len(self.vehicle.components)):
            comp = self.vehicle.components[i]
            if true_deflection is not None:
                comp.update_actor(state, float(true_deflection[i]))
            else:
                comp.update_actor(state, 0)

    def init_debug(self, size=1, label=True):
        """
        Draws debug visuals for the vehicle and its FlightCanvas
        :param size: The size of the elements in the debug window
        :param label: If true, labels of the elements in the debug window
        """
        default_sphere_radius = 0.02
        sphere_radius = default_sphere_radius * 2 ** (size - 1)

        # Draw the vehicle's own reference point (e.g., Center of Gravity)
        sphere = pv.Sphere(radius=sphere_radius * 1.5, center=self.vehicle.xyz_ref)
        self.cg_sphere = self.pl.add_mesh(sphere, color='yellow', show_edges=False, label='Vehicle CG')

        # Instruct each component to draw its own debug visuals
        for component in self.vehicle.components:
            component.init_debug(self.pl, self.vehicle.xyz_ref, size=size, label=label)

    def update_debug(self, state: np.ndarray, true_deflection: np.ndarray):
        """
        Updates debug visuals for the vehicle and its FlightCanvas given the current state
        :param state: The current state of the vehicle
        :param true_deflection: The true deflection angle of the aero component
        """
        for i in range(len(self.vehicle.components)):
            comp = self.vehicle.components[i]
            comp.update_debug(state, float(true_deflection[i]))

    def draw_text(self, sim_time, state, control, true_deflection):
        """
        Draw text information on screen
        :param sim_time: Current simulation time
        :param state: Current vehicle state
        :param control: Current control inputs
        :param true_deflection: Current true deflection angles
        """
        # Basic flight information
        velocity = np.linalg.norm(state[3:6])
        altitude = state[2]  # Assuming negative Z is up

        info_str = (
            f"Time: {sim_time:.2f} s\n"
            f"Altitude: {altitude:.2f} m\n"
            f"Velocity: {velocity:.2f} m/s\n"
            f"Mach: {velocity / 343:.3f}\n"
            f"Dynamic Pressure: {0.5 * 1.225 * velocity ** 2:.2f} Pa"
        )

        self.info_text.input = info_str
        self.info_text.prop = self.text_prop

        # Detailed state vector information
        angular_rates_deg = np.degrees(state[10:13])  # Convert rad/s to deg/s
        deflection_angles_deg = np.degrees(state[13:17])  # Convert deflection angles to degrees

        state_vec_str = (
            f"Position: [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}] m\n"
            f"Velocity: [{state[3]:.2f}, {state[4]:.2f}, {state[5]:.2f}] m/s\n"
            f"Quaternion: [{state[6]:.3f}, {state[7]:.3f}, {state[8]:.3f}, {state[9]:.3f}]\n"
            f"Angular Rate: [{angular_rates_deg[0]:.2f}, {angular_rates_deg[1]:.2f}, {angular_rates_deg[2]:.2f}] deg/s\n"
            f"Flap Angles: [{deflection_angles_deg[0]:.2f}, {deflection_angles_deg[1]:.2f}, {deflection_angles_deg[2]:.2f}, {deflection_angles_deg[3]:.2f}] deg"
        )

        self.state_vec_text.input = state_vec_str
        self.state_vec_text.prop = self.text_prop

    def add_grid(self):
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

    def animate(self, debug=False, show_text=True, cam_distance=5, fps=60):
        """
        Animates the aerodynamic visuals for all FlightCanvas
        :param debug: If true, draws debug visuals
        :param show_text: If true, draws text information
        :param cam_distance: The distance from the camera to center of mass
        :param fps: The frames per second of animation
        """
        t_arr, x_arr, u_arr = self.vehicle.get_control_history()

        if show_text:
            self.init_text()

        # set frames per second
        dt = 1 / fps

        # calculate number of frames
        num_frames = int(np.floor(t_arr[-1] * fps))

        # output location
        # TODO: fix this
        video_filename = '../../animation.mp4'

        # start movie
        self.pl.open_movie(video_filename, framerate=fps, quality=9)

        for i in range(num_frames):
            sim_time = dt * i
            state, control = utils.interp_state(t_arr, x_arr, u_arr, sim_time)
            state[0] = -state[0]

            true_deflection = None
            if self.vehicle.actuator_dynamics is not None:
                deflection_state = state[13:]
                true_deflection = self.vehicle.actuator_dynamics.get_component_deflection(deflection_state, control)

            # Update actors with interpolated state
            self.update_actors(state, true_deflection)
            if debug:
                self.update_debug(state, true_deflection)

            if show_text:
                self.draw_text(sim_time, state, control, true_deflection)

            # get center of mass position
            pos = state[:3] # + (C_B_I @ self.vehicle.xyz_ref)

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