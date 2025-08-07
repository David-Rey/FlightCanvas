# aero_project/FlightCanvas/aero_component.py

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple

import numpy as np
from FlightCanvas import utils
import pyvista as pv

from FlightCanvas.buildup_manager import BuildupManager


def get_rel_alpha_beta(v_rel: Union[np.ndarray, List[float]]):
    """
    Calculates the relative angle of attack (alpha) and sideslip angle (beta)
    from a local-frame relative velocity vector.
    :param v_rel: The relative velocity vector [u, v, w] in the component's local frame
    :return: A tuple containing the angle of attack (alpha) and sideslip angle (beta)
    """
    v_a = np.linalg.norm(v_rel)  # Total airspeed
    alpha = np.arctan2(v_rel[2], v_rel[0])  # Angle of attack
    beta = np.arcsin(v_rel[1] / v_a)  # Sideslip angle
    return alpha, beta


class AeroComponent(ABC):
    """
    An abstract base class for any meshable, translatable part of a vehicle.
    Represents a single, physical component.
    """

    def __init__(self, name: str,
                 ref_direction: Union[np.ndarray, List[float]],
                 control_pivot=None,
                 is_prime=True,
                 symmetric_comp: Optional['AeroComponent'] = None, ):
        """
        :param name: The name of the component
        :param ref_direction: The primary axis of the component, used for rotation (e.g., hinge axis for a control surface)
        :param control_pivot: The axis at which the component will rotate given a control input
        :param is_prime: Flag to indicate if this is a primary component. If False, control surface rotations are inverted
        :param symmetric_comp: The AeroComponent object that is symmetric to the current AeroComponent object
        """

        self.name = name
        self.ref_direction = np.array(ref_direction)
        self.control_pivot = control_pivot
        self.is_prime = is_prime
        self.symmetric_comp = symmetric_comp
        self.symmetry_type = ''  # either 'xz-plane' or 'x-radial'
        self.radial_angle = None  # degrees of rotation around x-axis
        self.parent = None

        # Mesh and Position Attributes
        self.mesh: Optional[pv.PolyData] = None
        self.xyz_ref = np.array([0., 0., 0.])
        self.static_transform_matrix = np.eye(4)
        self.dynamic_transform_matrix = np.eye(4)

        # AeroSandbox Attributes
        self.asb_object = None  # To hold the asb.Wing or asb.Fuselage
        self.asb_airplane = None  # To hold the asb.Airplane

        # Buildup Manager Object
        self.buildup_manager = None

        # PyVista Attributes
        self.pv_actor = None  # pyvista actor

        # Debug actors
        self.arrow_actor = None
        self.ref_actor = None
        self.label_actor = None
        self.ref_direction_actor = None
        self.control_pivot_actor = None
        self.force_actors = []

    def set_parent(self, parent: 'AeroComponent'):
        """
        Sets self.parent to the AeroVehicle for higher level information such as center of mass
        :param: parent: The AeroVehicle that the component belongs to
        """
        self.parent = parent

    def get_forces_and_moment_lookup(self, state) -> Tuple[np.ndarray, np.ndarray]:
        """
        Looks up aerodynamic forces by interpolating pre-computed buildup data
        :param state: The current state of the vehicle (position, velocity, quaternion, angular_velocity)
        :return: Forces and moments from lookup table
        """
        # Extracts velocity, orientation and angular rate from state vector
        vel = state[3:6]
        quat = state[6:10]
        angular_rate = state[10:]

        # Calculate Direction Cosine Matrix from the quaternion
        C_B_I = utils.dir_cosine_np(quat)  # Body to Inertial

        # Transform inertial velocity into the vehicle's body frame
        v_B = C_B_I @ vel

        T = self.static_transform_matrix
        R = T[:3, :3]  # Extract the 3x3 rotation matrix from the transform

        # Transform velocity from body frame to the component's local frame
        v_comp = R @ v_B - np.cross(angular_rate, self.xyz_ref)

        # Get angle of attack (alpha) and sideslip (beta)
        alpha, beta = get_rel_alpha_beta(v_comp)

        # if component is main then use buildup manager, else use symmetric component buildup manager
        if self.is_prime:
            F_b, M_b = self.buildup_manager.get_forces_and_moments(alpha, beta, v_comp)
        else:
            # if the component is reflected around xz plane then use get_forces_and_moment_xz_plane function
            if self.symmetry_type == 'xz-plane':
                F_b, M_b = self.get_forces_and_moment_xz_plane(alpha, beta, v_comp)
            # if the component is rotated around x-axis then use get_forces_and_moment_x_axial function
            elif self.symmetry_type == 'x-radial':
                F_b, M_b = self.get_forces_and_moment_x_axial(v_comp)
            else:
                raise ValueError("self.symmetry_type needed to be either 'xz-plane' or 'x-radial'")

        # compute distance from component to vehicle center of mass
        lever_arm = self.parent.xyz_ref - self.xyz_ref

        # compute moment arm
        M_b_cross = np.cross(lever_arm, F_b)

        # sum aero-moments and moments due to forces
        M_b_total = M_b + M_b_cross
        return F_b, M_b_total

    def get_forces_and_moment_xz_plane(self, alpha, beta, v_comp) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the forces and moments from buildup data for FlightCanvas that are reflected in the xz plane.
        :param alpha: Angle of attack of component
        :param beta: Side slip angle of component
        :param v_comp: Velocity of component
        :return: Forces and moments from buildup data for FlightCanvas that are reflected in the xz plane
        """
        F_b, M_b = self.symmetric_comp.buildup_manager.get_forces_and_moments(alpha, -beta, v_comp)
        F_b[1] = -F_b[1]
        M_b[0] = -M_b[0]
        M_b[2] = -M_b[2]
        return F_b, M_b

    def get_forces_and_moment_x_axial(self, v_comp) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the forces and moments from buildup data for FlightCanvas that are rotated around the x-axis.
        :param v_comp: Velocity of component
        :return: Forces and moments from buildup data for FlightCanvas that are reflected in the xz plane
        """

        T = self.static_transform_matrix
        R_Comp_Body = T[:3, :3]  # Extract the 3x3 rotation matrix from the transform
        R_Body_Comp = R_Comp_Body.T

        alpha, beta = get_rel_alpha_beta(v_comp)

        F_b_local, M_b_local = self.symmetric_comp.buildup_manager.get_forces_and_moments(alpha, beta, v_comp)
        F_b = R_Body_Comp @ F_b_local
        M_b = R_Body_Comp @ M_b_local
        return F_b, M_b

    def init_buildup_manager(self, vehicle_path):
        """
        Adds buildup manager that holds aerodynamic forces and moments to object
        :param vehicle_path: The path to the buildup manager file
        """
        if self.is_prime:
            self.buildup_manager = BuildupManager(self.name, vehicle_path)

    def compute_buildup(self):
        """
        Computes the aerodynamic buildup data for the component over a range
        of alpha and beta angles at a specified velocity
        """
        if self.is_prime:
            self.buildup_manager.compute_buildup(self.asb_airplane)

    def save_buildup(self):
        """
        Saves aerodynamic coefficient from the buildup data
        """
        if self.is_prime:
            if self.buildup_manager.asb_data is None:
                self.compute_buildup()
            self.buildup_manager.save_buildup()

    def save_buildup_figs(self):
        """
        Saves a contour plot of a specified aerodynamic coefficient from the buildup data
        """
        if self.is_prime:
            if self.buildup_manager.asb_data is None:
                self.compute_buildup()
            self.buildup_manager.save_buildup_figs()

    def load_buildup(self):
        """
        Loads buildup that holds aerodynamic forces and moments to object
        """
        if self.is_prime:
            self.buildup_manager.load_buildup()

    @abstractmethod
    def generate_mesh(self):
        """
        Each component must know how to generate its own mesh
        """
        pass

    def init_actor(self, pl: pv.Plotter, **kwargs):
        """
        Adds the component's mesh to a PyVista plotter to create a renderable actor
        :param pl: The `pyvista.Plotter` instance to add the mesh to
        :param kwargs: Additional keyword arguments to pass to `pl.add_mesh()`
        """
        if self.mesh is None:
            self.generate_mesh()

        self.pv_actor = pl.add_mesh(self.mesh, **kwargs)

        # The user_matrix allows for efficient transformation of the actor
        self.pv_actor.user_matrix = self.static_transform_matrix

    def get_transform(self, rotation=0) -> np.ndarray:
        """
        Calculates the 4x4 homogeneous transformation matrix for the component
        :param rotation: The rotation angle [radians] around the `ref_direction`
        :return: The 4x4 transformation matrix
        """
        x_vec = np.array([1, 0, 0])
        ref = self.xyz_ref
        flip_matrix = np.eye(4)
        axial_rotation = np.eye(4)

        # If this is not a 'prime' component invert the rotation for symmetric control deflection
        if not self.is_prime:
            # if the component is reflected around xz plane then flip around y-axis
            if self.symmetry_type == 'xz-plane':
                rotation = -rotation
                flip_matrix[1, 1] = -1
            # check for x-radial symmetry
            elif self.symmetry_type == 'x-radial':
                axial_rotation[:3, :3] = utils.rotate_z(self.radial_angle)
            else:
                raise ValueError("self.symmetry_type needed to be either 'xz-plane' or 'x-radial'")

        # rotate from the standard body X-axis to the component's defined axis
        transform_from_axis_vec = utils.rotation_matrix_from_vectors(x_vec, self.ref_direction)

        # rotate around that new axis (for control deflection)
        transform_from_control = np.eye(4)
        if self.control_pivot is not None:
            transform_from_control = utils.rotation_matrix_from_axis_angle(self.control_pivot, rotation)

        # translate the rotated component to its reference position in the body frame
        transform_from_ref = utils.translation_matrix(ref)

        # The final matrix is the product of these transformations
        static_transform_matrix = transform_from_ref @ transform_from_control @ transform_from_axis_vec @ flip_matrix @ axial_rotation
        return static_transform_matrix

    def update_transform(self, **kwargs):
        """
        Computes and updates the component's stored transformation matrix
        :param kwargs: Additional keyword arguments to pass to get_transform
        """
        self.static_transform_matrix = self.get_transform(**kwargs)

    def update_dynamic_transform(self, state: np.ndarray):
        """
        Updates the component's dynamic transformation matrix used for animation
        :param state: The current state of the vehicle (position, velocity, quaternion, angular_velocity)
        """

        pos_I = state[:3]  # Position in the inertial frame
        quat = state[6:10]  # Orientation as a quaternion

        # Construct a transformation matrix
        R = utils.dir_cosine_np(quat)
        static_to_dynamic_transform = np.eye(4)
        static_to_dynamic_transform[:3, 3] = pos_I
        static_to_dynamic_transform[:3, :3] = R

        self.dynamic_transform_matrix = static_to_dynamic_transform @ self.static_transform_matrix

    def update_actor(self, state: np.ndarray):
        """
        Updates the PyVista actor's transformation matrix in the 3D scene
        :param state: The current state of the vehicle (position, velocity, quaternion, angular_velocity)
        """
        self.update_dynamic_transform(state)
        self.pv_actor.user_matrix = self.dynamic_transform_matrix

    def init_debug(self, pl: pv.Plotter, com: np.ndarray, size: float = 1.0, label=True):
        """
        Draws debug visuals for this component in a PyVista plotter
        :param pl: The `pyvista.Plotter` to draw on
        :param com: The vehicle's center of mass coordinates [x, y, z]
        :param size: The size of the elements in the debug window
        :param label: If true, draw a label on this component
        """

        default_sphere_radius = 0.02
        sphere_radius = default_sphere_radius * 2 ** (size - 1)

        # Draw a sphere at the component's reference point
        sphere = pv.Sphere(radius=sphere_radius, center=self.xyz_ref)
        self.ref_actor = pl.add_mesh(sphere, color='red', show_edges=False, label=f"{self.asb_object.name} Ref")

        # Draw an arrow from the component's ref point to the vehicle's CoM
        self.arrow_actor = utils.plot_line_from_points(pl, self.xyz_ref, com, color='grey')

        # Add a text label at the reference point
        if label:
            self.label_actor = pl.add_point_labels(np.array([self.xyz_ref]), [f'{self.name}'])

        self.ref_direction_actor = self.draw_ref_direction(pl, size=size)
        self.control_pivot_actor = self.draw_control_pivot(pl, size=size)

    def update_debug(self, state: np.ndarray):
        """
        Updates debug visuals for this component in a PyVista plotter
        :param state: The current state of the vehicle (position, velocity, quaternion, angular_velocity)
        """
        pos_inertial = state[:3]
        quat = state[6:10]
        R = utils.dir_cosine_np(utils.normalize_quaternion(quat))  # body to inertial rotation

        # Clear any existing thrust visuals
        for actor in self.force_actors:
            self.parent.pl.remove_actor(actor)
        self.force_actors.clear()

        start = pos_inertial + R @ self.xyz_ref

        F_b, _ = self.get_forces_and_moment_lookup(state)

        k = .15
        direction = (R @ F_b) * k
        end = start + direction

        line = pv.Line(start, end)
        actor = self.parent.pl.add_mesh(line, color='red', line_width=3)
        self.force_actors.append(actor)

    def translate(self, xyz: Union[np.ndarray, List[float]]) -> "AeroComponent":
        """
        Sets the component's reference position relative to the vehicle's origin
        :param xyz: The new reference position [x, y, z] in the vehicle's body frame
        :return: The instance of the component (`self`)
        """
        self.xyz_ref = np.array(xyz)
        return self

    def set_translate(self, xyz: Union[np.ndarray, List[float]]):
        """
        Sets the component's reference position. This is an alias for `translate`
        :param xyz: The new reference position [x, y, z] in the vehicle's body frame
        """
        self.translate(xyz)

    def draw_ref_direction(self, pl: pv.Plotter, size: float = 1.0) -> pv.Actor:
        """
        Draws the component's `ref_direction` in a PyVista plot
        :param pl: The PyVista plotter to draw on
        :param size: The size of the elements in the debug window
        """
        default_length = 1
        length = default_length * 2 ** (size - 1)
        return utils.draw_line_from_point_and_vector(pl, self.xyz_ref, self.ref_direction, color='green', line_width=4,
                                                     length=length)

    def draw_control_pivot(self, pl: pv.Plotter, size: float = 1.0) -> Union[pv.Actor, None]:
        """
        Draws the component's `control_pivot` in a PyVista plot
        :param pl: The PyVista plotter to draw on
        :param size: The size of the elements in the debug window
        """
        default_length = 0.6
        length = default_length * 2 ** (size - 1)
        if self.control_pivot is not None:
            return utils.draw_line_from_point_and_vector(pl, self.xyz_ref, self.control_pivot, color='blue',
                                                         line_width=6, length=length)
        return None

        # T = self.static_transform_matrix
        # R_Comp_Body = T[:3, :3]  # Extract the 3x3 rotation matrix from the transform
        # R_Body_Comp = R_Comp_Body.T

        # F_b_local, M_b_local = self.symmetric_comp.buildup_manager.get_forces_and_moments(alpha, beta, v_comp)
        # F_b = R_Body_Comp @ F_b_local
        # M_b = R_Body_Comp @ M_b_local
        # return F_b, M_b

    '''
    def get_forces_and_moment_lookup(self, v_B: np.ndarray) -> np.ndarray:
        """
        Looks up aerodynamic forces by interpolating pre-computed buildup data
        :param v_B: The velocity vector of the vehicle in the body frame
        :return: Forces and moments from lookup table
        """
        alpha, beta = self.get_alpha_beta(v_B)

        # Get the alpha and beta axes from the pre-computed grid (in radians)
        alpha_lin_rad = np.deg2rad(self.alpha_grid[:, 0])
        beta_lin_rad = np.deg2rad(self.beta_grid[0, :])

        # Reshape force data to be compatible with interpolation
        data = self.asb_data["F_b"][0].reshape(self.alpha_grid.shape)
        data_3d = data[:, :, np.newaxis]

        # Perform linear interpolation to find the force at the current alpha/beta
        F_b = utils.linear_interpolation(alpha_lin_rad, beta_lin_rad, alpha, beta, data_3d)
        return F_b
    '''


'''
    def compute_buildup(self):
        """
        Computes the aerodynamic buildup data for the component over a range
        of alpha and beta angles at a specified velocity
        """
        self.buildup_manager.compute_buildup()
        # Create a meshgrid of alpha and beta values to analyze
        self.beta_grid, self.alpha_grid = np.meshgrid(
            np.linspace(-90, 90, 100),
            np.linspace(-180, 180, 150)
        )
        # Define the operating points for the analysis
        op_point = asb.OperatingPoint(
            velocity=10,
            alpha=self.alpha_grid.flatten(),
            beta=self.beta_grid.flatten()
        )
        # Run the AeroBuildup analysis
        self.asb_data = asb.AeroBuildup(
            airplane=self.asb_airplane,
            op_point=op_point
        ).run()

    def draw_buildup(self, name: str, ID: str, index=None):
        """
        Draws a contour plot of a specified aerodynamic coefficient from the
        buildup data
        :param name: The name of the component
        :param ID: The identifier for the data to plot (e.g., "CL", "CD", "F_b")
        :param index: The index if the data is a vector (e.g., 0 for Fx in F_b)
        """
        if index is None:
            data = self.asb_data[ID]
            title = f"`{name}` {ID}"
        else:
            data = self.asb_data[ID][index]
            title = f"`{name}` {ID} [{index}]"

        # Create the contour plot
        p.contour(
            self.beta_grid, self.alpha_grid, data.reshape(self.alpha_grid.shape),
            colorbar_label=f"${ID}$ [-]",
            linelabels_format=lambda x: f"{x:.2f}",
            linelabels_fontsize=7,
            cmap="RdBu",
            alpha=0.6
        )
        p.set_ticks(15, 5, 15, 5)
        p.show_plot(
            title,
            r"Sideslip angle $\beta$ [deg]",
            r"Angle of Attack $\alpha$ [deg]",
            set_ticks=False
        )
        plt.show()
'''

# def get_alpha_beta(self, v_comp: np.ndarray) -> Tuple[float, float]:
"""
Calculates the component's local angle of attack and sideslip.
:param v_comp: The velocity vector of the component in the body frame
:return: A tuple containing the component's local angle of attack (alpha) and
    sideslip angle (beta) in radians
"""
# T = self.get_transform()
# R = T[:3, :3]  # Extract the 3x3 rotation matrix from the transform


# alpha, beta = get_rel_alpha_beta(v_comp)
# return alpha, beta

# self.alpha_grid = None  # Hold grid of an angle of attack for buildup
# self.beta_grid = None  # Hold grid of sideslip angles for buildup
# self.asb_data = None  # To hold the aero build data

# else:
#    F_b, M_b = self.symmetric_comp.buildup_manager.get_forces_and_moments(alpha, -beta, v_comp)
#    F_b[1] = -F_b[1]
#    M_b[0] = -M_b[0]
#    M_b[2] = -M_b[2]
