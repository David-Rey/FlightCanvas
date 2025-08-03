# aero_project/components/aero_component.py

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple

import numpy as np
#from ASB6DOF import utils
import utils
import pyvista as pv

from components.buildup_manager import BuildupManager


def get_rel_alpha_beta(v_rel:  Union[np.ndarray, List[float]]):
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
                 axis_vector: Union[np.ndarray, List[float]],
                 is_prime=True,
                 parent: Optional['AeroComponent'] = None,):
        """
        :param name: The name of the component
        :param axis_vector: The primary axis of the component, used for rotation (e.g., hinge axis for a control surface)
        :param is_prime: Flag to indicate if this is a primary component. If False, control surface rotations are inverted
        :param parent: The AeroComponent object that is symmetric to the current AeroComponent object
        """

        self.name = name
        self.axis_vector = np.array(axis_vector)
        self.is_prime = is_prime
        self.parent = parent

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
        self.axis_ref = None

        # Set transform_matrix
        #self.transform_matrix = self.get_transform()

    def get_alpha_beta(self, v_B: np.ndarray, angular_rate: np.ndarray) -> Tuple[float, float]:
        """
        Calculates the component's local angle of attack and sideslip.
        :param v_B: The velocity vector of the vehicle in the body frame
        :param angular_rate: Angular rotation of the vehicle in the body frame
        :return: A tuple containing the component's local angle of attack (alpha) and
            sideslip angle (beta) in radians
        """
        T = self.get_transform()
        R = T[:3, :3]  # Extract the 3x3 rotation matrix from the transform

        # Transform velocity from body frame to the component's local frame
        v_comp = R @ v_B + np.cross(angular_rate, self.xyz_ref)
        alpha, beta = get_rel_alpha_beta(v_comp)
        return alpha, beta

    def get_forces_and_moment_lookup(self, v_B: np.ndarray, angular_rate: np.ndarray, com: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Looks up aerodynamic forces by interpolating pre-computed buildup data
        :param v_B: The velocity vector of the vehicle in the body frame
        :param angular_rate: Angular rotation of the vehicle in the body frame
        :return: Forces and moments from lookup table
        """
        alpha, beta = self.get_alpha_beta(v_B, angular_rate)
        if self.is_prime:
            F_b, M_b = self.buildup_manager.get_forces_and_moments(alpha, beta, v_B)
        else:
            F_b, M_b = self.parent.buildup_manager.get_forces_and_moments(alpha, -beta, v_B)
            F_b[1] = -F_b[1]
            M_b[0] = -M_b[0]
            M_b[2] = -M_b[2]

        lever_arm = self.xyz_ref - com
        M_b_cross = np.cross(-lever_arm, F_b)
        M_b_total = M_b + M_b_cross
        #M_b_total = M_b_cross
        return F_b, M_b_total

    def init_buildup_manager(self, vehicle_path):
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

        self.update_transform()

        # The user_matrix allows for efficient transformation of the actor
        self.pv_actor.user_matrix = self.static_transform_matrix

    def get_transform(self, rotation=0) -> np.ndarray:
        """
        Calculates the 4x4 homogeneous transformation matrix for the component
        :param rotation: The rotation angle [radians] around the `axis_vector`
        :return: The 4x4 transformation matrix
        """
        x_vec = np.array([1, 0, 0])
        ref = self.xyz_ref
        flip_matrix = np.eye(4)

        # If this is not a 'prime' component invert the rotation for symmetric control deflection
        if not self.is_prime:
            rotation = -rotation
            flip_matrix[1, 1] = -1

        # rotate from the standard body X-axis to the component's defined axis
        transform_from_axis_vec = utils.rotation_matrix_from_vectors(x_vec, self.axis_vector)

        # rotate around that new axis (for control deflection)
        transform_from_control = utils.rotation_matrix_from_axis_angle(self.axis_vector, rotation)

        # translate the rotated component to its reference position in the body frame
        transform_from_ref = utils.translation_matrix(ref)

        # The final matrix is the product of these transformations
        static_transform_matrix = transform_from_ref @ transform_from_control @ transform_from_axis_vec @ flip_matrix
        return static_transform_matrix

    def update_transform(self, **kwargs):
        """
        Computes and updates the component's stored transformation matrix
        :param kwargs: Additional keyword arguments to pass to get_transform
        """
        self.static_transform_matrix = self.get_transform(**kwargs)

    def update_dynamic_transform(self, state: np.ndarray):
        """
        TODO
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
        """
        self.update_dynamic_transform(state)
        self.pv_actor.user_matrix = self.dynamic_transform_matrix


    def init_debug(self, pl: pv.Plotter, com: np.ndarray, sphere_radius=0.02):
        """
        Draws debug visuals for this component in a PyVista plotter
        :param pl: The `pyvista.Plotter` to draw on
        :param com: The vehicle's center of mass coordinates [x, y, z]
        :param sphere_radius: The radius of the sphere marking the reference point
        """

        # Draw a sphere at the component's reference point
        sphere = pv.Sphere(radius=sphere_radius, center=self.xyz_ref)
        #self.ref_actor = pl.add_mesh(sphere, color='red', show_edges=False, label=f"{self.asb_object.name} Ref")

        # Draw an arrow from the component's ref point to the vehicle's CoM
        self.arrow_actor = utils.plot_arrow_from_points(pl, self.xyz_ref, com, color='grey')

        # Add a text label at the reference point
        self.label_actor = pl.add_point_labels(np.array([self.xyz_ref]), [f'{self.name}'])

        # Specific debug visuals for subclasses like AeroWing
        from .aero_wing import AeroWing
        if isinstance(self, AeroWing):
            self.axis_ref = self.draw_axis_vector(pl)

    def update_debug(self, state: np.ndarray):
        """
        TODO
        """
        #self.update_dynamic_transform(state)
        #self.ref_actor.user_matrix = self.dynamic_transform_matrix
        #if self.arrow_actor is not None:
        #    self.arrow_actor.user_matrix = self.dynamic_transform_matrix
        #self.label_actor.user_matrix = self.dynamic_transform_matrix

        #from .aero_wing import AeroWing
        #if isinstance(self, AeroWing):
        #    self.axis_ref.user_matrix = self.dynamic_transform_matrix

        pass

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

# self.alpha_grid = None  # Hold grid of an angle of attack for buildup
# self.beta_grid = None  # Hold grid of sideslip angles for buildup
# self.asb_data = None  # To hold the aero build data