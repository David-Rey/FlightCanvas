# aero_project/FlightCanvas/aero_component.py

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple

import casadi as ca
import numpy as np
from FlightCanvas import utils
import pyvista as pv

from FlightCanvas.buildup_manager import BuildupManager
from FlightCanvas.actuators import ActuatorModel


def get_rel_alpha_beta(v_rel: Union[np.ndarray, ca.MX]) -> Tuple[Union[float, ca.MX], Union[float, ca.MX]]:
    """
    Calculates the relative angle of attack (alpha) and sideslip angle (beta)
    from a local-frame relative velocity vector. This function is type-aware.
    :param v_rel: The relative velocity vector [u, v, w] in the component's local frame
    :return: A tuple containing the angle of attack (alpha) and sideslip angle (beta)
    """
    is_casadi = isinstance(v_rel, (ca.SX, ca.MX))
    epsilon = 1e-10

    if is_casadi:
        v_a = ca.norm_2(v_rel)
        alpha = ca.atan2(v_rel[2], v_rel[0])
        beta = ca.asin(v_rel[1] / (v_a + epsilon))
    else:
        v_a = np.linalg.norm(v_rel)
        alpha = np.arctan2(v_rel[2], v_rel[0])
        beta = np.arcsin(v_rel[1] / (v_a + epsilon))

    return alpha, beta


class AeroComponent(ABC):
    """
    An abstract base class for any meshable, translatable part of a vehicle.
    Represents a single, physical component.
    """

    def __init__(
        self,
        name: str,
        ref_direction: Union[np.ndarray, List[float]],
        control_pivot=None,
        is_prime=True,
        symmetric_comp: Optional['AeroComponent'] = None,
        actuator_model: Optional[ActuatorModel] = None, ):
        """
        :param name: The name of the component
        :param ref_direction: The primary axis of the component, used for rotation (e.g., hinge axis for a control surface)
        :param control_pivot: The axis at which the component will rotate given a control input
        :param is_prime: Flag to indicate if this is a primary component. If False, control surface rotations are inverted
        :param symmetric_comp: The AeroComponent object that is symmetric to the current AeroComponent object
        :param actuator_model: The actuator model object to use
        """

        self.name = name
        self.ref_direction = np.array(ref_direction)
        self.control_pivot = control_pivot
        self.is_prime = is_prime
        self.symmetric_comp = symmetric_comp
        self.symmetry_type = ''  # either 'xz-plane' or 'x-radial'
        self.radial_angle = None  # degrees of rotation around x-axis
        self.parent = None
        self.actuator_model = actuator_model

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

    def set_actuator(self, actuator: ActuatorModel):
        """
        TODO
        """
        self.actuator_model = actuator


    def get_forces_and_moments(
        self,
        state: Union[np.ndarray, ca.MX]) \
    -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
        """
        Calculates the aerodynamic forces and moments on the component.
        This function is type-aware and will use either NumPy or CasADi based on the input type.
        :param state: The state vector (position, velocity, quaternion, angular velocity) in either CasADi or NumPy.
        """
        is_casadi = isinstance(state, (ca.SX, ca.MX))

        vel = state[3:6]
        quat = state[6:10]
        angular_rate = state[10:]

        # Select the correct library functions and types based on the input
        if is_casadi:
            lib = ca
            dir_cosine_func = utils.dir_cosine_ca
            to_type = ca.MX
            norm = ca.norm_2
        else:
            lib = np
            dir_cosine_func = utils.dir_cosine_np
            to_type = lambda x: x
            norm = np.linalg.norm

        # Compute direction cosine function
        C_B_I = dir_cosine_func(quat)

        # get velocity of the body
        v_B = C_B_I @ vel

        # get rotation matrix from transform matrix
        T = to_type(self.static_transform_matrix)
        R = T[:3, :3]

        # get reference point to either numpy or casadi
        xyz_ref_typed = to_type(self.xyz_ref)

        # get local velocity of the component
        v_comp = R @ v_B - lib.cross(angular_rate, xyz_ref_typed)

        # get airspeed of component
        speed = norm(v_comp)

        # compute angle of attack and sideslip
        alpha, beta = get_rel_alpha_beta(v_comp)

        # if component is main then use buildup manager, else use symmetric component buildup manager
        if self.is_prime:
            F_b, M_b = self.buildup_manager.get_forces_and_moments(alpha, beta, speed)
        else:
            # if the component is reflected around xz plane then use get_forces_and_moment_xz_plane function
            if self.symmetry_type == 'xz-plane':
                F_b, M_b = self.get_forces_and_moment_xz_plane(alpha, beta, speed)
            # if the component is rotated around x-axis then use get_forces_and_moment_x_axial function
            elif self.symmetry_type == 'x-radial':
                F_b, M_b = self.get_forces_and_moment_x_axial(v_comp)
            else:
                raise ValueError("self.symmetry_type needed to be either 'xz-plane' or 'x-radial'")

        # compute distance from component to vehicle center of mass
        lever_arm = to_type(self.parent.xyz_ref - self.xyz_ref)

        # compute moment arm
        M_b_cross = lib.cross(lever_arm, F_b)

        # Sum aero-moments and moments due to forces
        M_b_total = M_b + M_b_cross
        return F_b, M_b_total

    def get_forces_and_moment_xz_plane(
        self,
        alpha: Union[float, ca.MX],
        beta: Union[float, ca.MX],
        speed: Union[float, ca.MX]
    ) -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
        """
        Returns the forces and moments reflected in the xz plane.
        Switches between NumPy and CasADi based on input type.
        :param alpha: Angle of attack (float for NumPy, ca.MX for CasADi)
        :param beta: Side slip angle (float for NumPy, ca.MX for CasADi)
        :param speed: Velocity (float for NumPy, ca.MX for CasADi)
        :return: Forces and moments (np.ndarray or ca.MX)
        """
        # Check if inputs are CasADi symbolic variables
        F_b, M_b = self.symmetric_comp.buildup_manager.get_forces_and_moments(alpha, -beta, speed)
        if isinstance(alpha, ca.MX):
            #F_b, M_b = self.symmetric_comp.buildup_manager.get_forces_and_moments_casadi(alpha, -beta, speed)
            F_b = ca.vertcat(F_b[0], -F_b[1], F_b[2])
            M_b = ca.vertcat(-M_b[0], M_b[1], -M_b[2])
        else:
            F_b[1] *= -1
            M_b[0] *= -1
            M_b[2] *= -1

        return F_b, M_b

    def get_forces_and_moment_x_axial(
        self, v_comp: Union[np.ndarray, ca.MX]
    ) -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
        """
        Returns the forces and moments rotated from the component's local frame
        to the body frame. Switches between NumPy and CasADi based on input type.

        :param v_comp: Velocity of the component (np.ndarray or ca.MX)
        :return: Forces and moments (np.ndarray or ca.MX)
        """
        # Set up library-specific functions and variables
        if isinstance(v_comp, ca.MX):
            T = ca.MX(self.static_transform_matrix)
            norm_func = ca.norm_2
        else:
            T = self.static_transform_matrix
            norm_func = np.linalg.norm

        # Perform the calculation using the selected functions
        R_Comp_Body = T[:3, :3]
        R_Body_Comp = R_Comp_Body.T

        alpha, beta = get_rel_alpha_beta(v_comp)
        speed = norm_func(v_comp)

        F_b_local, M_b_local = self.symmetric_comp.buildup_manager.get_forces_and_moments(alpha, beta, speed)

        # Matrix multiplication works for both NumPy and CasADi
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

    def get_transform(self, rotation: Union[float, ca.MX] = 0) -> Union[np.ndarray, ca.MX]:
        """
        Calculates the 4x4 homogeneous transformation matrix for the component
        :param rotation: The rotation angle [radians] around the `ref_direction`
        :return: The 4x4 transformation matrix
        """
        is_casadi = isinstance(rotation, (ca.SX, ca.MX))

        if is_casadi:
            # Use CasADi functions and types
            eye_func = ca.MX.eye
            to_type = ca.MX
        else:
            # Use NumPy functions and types
            eye_func = np.eye
            to_type = lambda x: x

        # Calculate Static Transformation Matrices
        x_vec = np.array([1, 0, 0])
        flip_matrix = np.eye(4)
        axial_rotation_matrix = np.eye(4)

        # If this is not a 'prime' component, apply symmetry rules
        if not self.is_prime:
            if self.symmetry_type == 'xz-plane':
                rotation = -rotation  # Invert rotation for symmetric deflection
                flip_matrix[1, 1] = -1
            elif self.symmetry_type == 'x-radial':
                axial_rotation_matrix[:3, :3] = utils.rotate_z(self.radial_angle)
            else:
                raise ValueError("self.symmetry_type must be either 'xz-plane' or 'x-radial'")

        # Static rotations based on component geometry (can be calculated with NumPy)
        transform_from_axis_vec = utils.rotation_matrix_from_vectors(x_vec, self.ref_direction)
        transform_from_ref = utils.translation_matrix(self.xyz_ref)

        # Calculate the Dynamic Control Deflection Matrix
        # Use the dispatched functions to create a matrix of the correct type
        transform_from_control = eye_func(4)
        if self.control_pivot is not None:
            transform_from_control = utils.rotation_matrix_from_axis_angle(self.control_pivot, rotation)

        # Combine Transformations
        final_transform = (
                to_type(transform_from_ref) @
                transform_from_control @
                to_type(transform_from_axis_vec) @
                to_type(flip_matrix) @
                to_type(axial_rotation_matrix)
        )

        return final_transform


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

        F_b, _ = self.get_forces_and_moments(state)

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
