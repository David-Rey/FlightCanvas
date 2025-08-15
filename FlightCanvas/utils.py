import aerosandbox.geometry.mesh_utilities as mesh_utils
import pyvista as pv
import numpy as np
from typing import List, Union
import casadi as ca


def get_mesh(abs_mesh, translation_vector=None, **kwargs):
    """
    Converts AeroSandbox mesh object to pyvista mesh object
    """
    # add body
    points, faces = abs_mesh.mesh_body(**kwargs)

    # Apply translation if a vector is provided
    if translation_vector is not None:
        if not isinstance(translation_vector, (list, tuple, np.ndarray)) or len(translation_vector) != 3:
            raise ValueError("`translation_vector` must be a 3-element list, tuple, or numpy array.")
        points = points + np.array(translation_vector)  # Add the translation vector to each point

    mesh = pv.PolyData(
        *mesh_utils.convert_mesh_to_polydata_format(points, faces)
    )
    return mesh


def plot_line_from_points(
        plotter: pv.Plotter,
        start_point: np.ndarray,
        end_point: np.ndarray,
        width: float = 8,
        color: str = 'grey',
        label: str = None,
        **kwargs  # Any additional kwargs for plotter.add_mesh (e.g., opacity)
) -> pv.Actor | None:  # Can return None if line length is zero
    """
    Plots a PyVista line from a starting point to an ending point.

    Args:
        plotter: The PyVista Plotter instance to add the line to.
        start_point: A 3-element NumPy array or list representing the line's start.
        end_point: A 3-element NumPy array or list representing the line's end.
        width: The width of the line. Defaults to 5.
        color: The color of the line (e.g., 'red', 'blue', '#FF00FF'). Defaults to 'grey'.
        label: Optional label for the line in the legend.
        **kwargs: Additional keyword arguments passed directly to `plotter.add_mesh()`.

    Returns:
        The PyVista Actor object for the added line, or None if the line length is zero.
    """
    start_point = np.asarray(start_point, dtype=float)
    end_point = np.asarray(end_point, dtype=float)

    # Handle zero-length lines to prevent issues
    if np.allclose(start_point, end_point):
        print(f"Warning: Line start and end points are identical ({start_point}). Not plotting line.")
        return None

    # Create the line mesh directly from the two points
    line_mesh = pv.Line(start_point, end_point)

    # Add the line mesh to the plotter, controlling its thickness with `line_width`
    actor = plotter.add_mesh(
        line_mesh,
        color=color,
        label=label,
        line_width=width,
        **kwargs
    )
    return actor


def plot_arrow_from_points(
        plotter: pv.Plotter,
        start_point: np.ndarray,
        end_point: np.ndarray,
        size: float = 1,
        color: str = 'grey',
        label: str = None,
        **kwargs  # Any additional kwargs for plotter.add_mesh (e.g., opacity, show_edges)
) -> pv.Actor | None:  # Can return None if arrow length is zero
    """
    Plots a PyVista arrow from a starting point to an ending point with a constant absolute width.

    Args:
        plotter: The PyVista Plotter instance to add the arrow to.
        start_point: A 3-element NumPy array or list representing the arrow's tail.
        end_point: A 3-element NumPy array or list representing the arrow's head.
        size: The size of the arrow.
        color: The color of the arrow (e.g., 'red', 'blue', '#FF00FF'). Defaults to 'red'.
        label: Optional label for the arrow in the legend.
        **kwargs: Additional keyword arguments passed directly to `plotter.add_mesh()`.

    Returns:
        The PyVista Actor object for the added arrow, or None if the arrow length is zero.
    """

    default_shaft_radius = 0.01  # Absolute radius of the arrow shaft
    default_tip_radius = 0.03  # Absolute radius of the arrow tip
    tip_length_ratio = 0.15  # Tip length as fraction of total arrow length

    shaft_radius = default_shaft_radius * 2 ** (size - 1)
    tip_radius = default_tip_radius * 2 ** (size - 1)

    start_point = np.asarray(start_point, dtype=float)
    end_point = np.asarray(end_point, dtype=float)

    # Calculate the direction vector
    direction_vector = end_point - start_point

    # Calculate the actual length of the arrow
    arrow_length = np.linalg.norm(direction_vector)

    # Handle zero-length arrows to prevent division by zero
    if arrow_length < 1e-9:  # Use a small epsilon for floating point comparison
        print(f"Warning: Arrow start and end points are too close ({start_point} to {end_point}). Not plotting arrow.")
        # Optionally, you could plot a small sphere at start_point here if desired
        # sphere = pv.Sphere(radius=fixed_tip_radius, center=start_point)
        # return plotter.add_mesh(sphere, color=color, label=label, **kwargs)
        return None

    # Calculate the fractional radii required by pv.Arrow
    # These are relative to the 'scale' argument (which is the total arrow length)
    fractional_shaft_radius = shaft_radius / arrow_length
    fractional_tip_radius = tip_radius / arrow_length

    # Ensure tip length is not larger than arrow length (fractional)
    tip_length = tip_length_ratio * arrow_length

    # Create the arrow mesh
    arrow_mesh = pv.Arrow(
        start=start_point,
        direction=direction_vector,
        scale=arrow_length,  # Set scale to the actual length
        tip_length=tip_length,  # This is now absolute length
        tip_radius=fractional_tip_radius,
        shaft_radius=fractional_shaft_radius
    )

    # Add the arrow mesh to the plotter
    actor = plotter.add_mesh(
        arrow_mesh,
        color=color,
        label=label,
        **kwargs
    )
    return actor


def draw_line_from_point_and_vector(
        pl: pv.Plotter,
        point: Union[np.ndarray, List[float]],
        vector: Union[np.ndarray, List[float]],
        length: float = 1.0,
        **kwargs
) -> pv.Actor:
    """
    Draws a line through a given point, oriented along a specified vector.

    The line extends equally in both directions from the point, with a total length.

    Args:
        pl (pv.Plotter): The PyVista plotter object to add the line to.
        point (Union[np.ndarray, List[float]]): The [x, y, z] coordinates of a point
                                                 that the line should pass through.
        vector (Union[np.ndarray, List[float]]): The [dx, dy, dz] FlightCanvas of the
                                                  direction vector for the line.
        length (float, optional): The total length of the line. Defaults to 1.0.
        **kwargs: Additional keyword arguments to pass to `pl.add_mesh()` for styling
                  the line (e.g., `color='blue'`, `line_width=3`).

    Returns:
        pv.Actor: The PyVista actor representing the drawn line.
    """
    point = np.array(point)
    vector = np.array(vector)

    # Normalize the vector to get a unit direction
    if np.linalg.norm(vector) == 0:
        raise ValueError("The direction vector cannot be a zero vector.")
    unit_vector = vector / np.linalg.norm(vector)

    # Calculate half the length
    half_length = length / 2.0

    # Calculate the start and end points of the line
    start_point = point - unit_vector * half_length
    end_point = point + unit_vector * half_length

    # Create and add the line mesh to the plotter
    line_mesh = pv.Line(start_point, end_point)
    actor = pl.add_mesh(line_mesh, **kwargs)

    return actor


def flip_y(vector: Union[np.ndarray, List[float]]) -> np.ndarray:
    if vector is None:
        return None
    new_vector = np.array([vector[0], -vector[1], vector[2]])
    return new_vector


def rotation_matrix_from_vectors(
        vec_from: Union[np.ndarray, List[float]],
        vec_to: Union[np.ndarray, List[float]]
) -> np.ndarray:
    """
    Computes the 3x3 rotation matrix that rotates one 3D vector to align with another.

    This function finds the smallest rotation to transform `vec_from` into `vec_to`.

    Args:
        vec_from (Union[np.ndarray, List[float]]): The 3D vector to rotate (source).
        vec_to (Union[np.ndarray, List[float]]): The 3D vector to rotate to (destination).

    Returns:
        np.ndarray: A 3x3 rotation matrix.

    Raises:
        ValueError: If either input vector is a zero vector.
    """
    vec_from = np.array(vec_from, dtype=float)
    vec_to = np.array(vec_to, dtype=float)

    # Check for zero vectors
    norm_from = np.linalg.norm(vec_from)
    norm_to = np.linalg.norm(vec_to)

    if norm_from == 0:
        raise ValueError("The 'vec_from' vector cannot be a zero vector.")
    if norm_to == 0:
        raise ValueError("The 'vec_to' vector cannot be a zero vector.")

    # Normalize the vectors
    vec_from_norm = vec_from / norm_from
    vec_to_norm = vec_to / norm_to

    # Calculate the dot product and clamp it to handle potential floating point inaccuracies
    dot_product = np.clip(np.dot(vec_from_norm, vec_to_norm), -1.0, 1.0)

    # Calculate the angle of rotation
    angle = np.arccos(dot_product)

    # Handle special cases: parallel or anti-parallel vectors
    if np.isclose(angle, 0.0):
        # Vectors are already parallel, no rotation needed
        return np.identity(4)
    elif np.isclose(angle, np.pi):
        # Vectors are anti-parallel (180 degree rotation)
        # We need an arbitrary axis perpendicular to vec_from_norm
        # Try cross product with [1,0,0]. If parallel, try [0,1,0]. One must work.
        axis = np.cross(vec_from_norm, np.array([1, 0, 0]))
        if np.linalg.norm(axis) < 1e-6:  # Check if cross product is effectively zero
            axis = np.cross(vec_from_norm, np.array([0, 1, 0]))

        # Normalize the chosen axis
        axis_norm = axis / np.linalg.norm(axis)

        # Rotation matrix for 180 degrees around 'axis_norm'
        # R = 2 * (axis_norm @ axis_norm.T) - I
        return 2 * np.outer(axis_norm, axis_norm) - np.identity(4)
    else:
        # General case: calculate the rotation axis from the cross product
        axis = np.cross(vec_from_norm, vec_to_norm)
        axis_norm = axis / np.linalg.norm(axis)

        # Construct the skew-symmetric matrix (K) from the unit rotation axis
        k_x, k_y, k_z = axis_norm
        K = np.array([
            [0, -k_z, k_y],
            [k_z, 0, -k_x],
            [-k_y, k_x, 0]
        ])

        # Apply Rodrigues' rotation formula
        # R = I + sin(theta) * K + (1 - cos(theta)) * K^2
        # A more numerically stable form: R = cos(theta) * I + (1 - cos(theta)) * outer(k, k) + sin(theta) * K
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        identity_matrix = np.identity(3)
        outer_product_axis = np.outer(axis_norm, axis_norm)  # k * k.T

        R = cos_theta * identity_matrix + (1 - cos_theta) * outer_product_axis + sin_theta * K

        T = np.eye(4)

        # Insert the 3x3 rotation matrix into the top-left block
        T[:3, :3] = R

        return T


def rotation_matrix_from_axis_angle_casadi(
        axis: Union[np.ndarray, List[float]],
        angle_rad: ca.MX
) -> ca.MX:
    """
    Computes a 4x4 transformation matrix that rotates around a given 3D axis
    by a specified symbolic angle using CasADi.

    This function implements Rodrigues' Rotation Formula symbolically.

    Args:
        axis (Union[np.ndarray, List[float]]): The 3D vector representing the
                                                 axis of rotation. This is a
                                                 fixed, numerical value. It
                                                 will be normalized internally.
        angle_rad (ca.MX): The symbolic rotation angle in radians (a CasADi variable).

    Returns:
        ca.MX: A 4x4 symbolic transformation matrix.

    Raises:
        ValueError: If the input axis vector is a zero vector.
    """
    # --- 1. Handle the numerical axis vector ---
    axis_np = np.array(axis, dtype=float)

    # Normalize the axis vector
    norm_axis = np.linalg.norm(axis_np)
    if norm_axis < 1e-9:  # Use a small tolerance for floating point
        raise ValueError("The rotation 'axis' vector cannot be a zero vector.")
    unit_axis = axis_np / norm_axis

    # Extract components of the unit axis vector
    kx, ky, kz = unit_axis

    # --- 2. Handle the symbolic angle ---
    # Pre-calculate sine and cosine of the symbolic angle
    cos_theta = ca.cos(angle_rad)
    sin_theta = ca.sin(angle_rad)

    # Calculate (1 - cos_theta) for efficiency
    one_minus_cos_theta = 1 - cos_theta

    # --- 3. Construct constant matrices in CasADi format ---
    # Construct the skew-symmetric cross-product matrix (K)
    K = ca.MX(np.array([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ]))

    # The identity matrix
    identity_matrix = ca.MX.eye(3)

    # K_squared = K @ K
    # This can be pre-calculated as it does not depend on the angle.
    K_squared = K @ K

    # --- 4. Apply Rodrigues' rotation formula symbolically ---
    # R = I + sin(theta) * K + (1 - cos(theta)) * K^2
    R = identity_matrix + sin_theta * K + one_minus_cos_theta * K_squared

    # --- 5. Embed the 3x3 rotation into a 4x4 transformation matrix ---
    # Start with a 4x4 identity matrix
    T = ca.MX.eye(4)

    # Insert the 3x3 symbolic rotation matrix into the top-left block
    T[:3, :3] = R

    return T


def rotation_matrix_from_axis_angle(
        axis: Union[np.ndarray, List[float]],
        angle_rad: float
) -> np.ndarray:
    """
    Computes a 3x3 rotation matrix that rotates around a given 3D axis by a specified angle.

    Uses Rodrigues' Rotation Formula.

    Args:
        axis (Union[np.ndarray, List[float]]): The 3D vector representing the axis of rotation.
                                                 It will be normalized internally.
        angle_rad (float): The rotation angle in radians.

    Returns:
        np.ndarray: A 3x3 rotation matrix.

    Raises:
        ValueError: If the input axis vector is a zero vector.
    """
    axis = np.array(axis, dtype=float)

    # Normalize the axis vector
    norm_axis = np.linalg.norm(axis)
    if norm_axis == 0:
        raise ValueError("The rotation 'axis' vector cannot be a zero vector.")
    unit_axis = axis / norm_axis

    # Extract FlightCanvas of the unit axis vector
    kx, ky, kz = unit_axis

    # Pre-calculate sine and cosine of the angle
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    # Calculate (1 - cos_theta) for efficiency
    one_minus_cos_theta = 1 - cos_theta

    # Construct the skew-symmetric cross-product matrix (K)
    K = np.array([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ])

    # Apply Rodrigues' rotation formula:
    # R = I + sin(theta) * K + (1 - cos(theta)) * K^2
    identity_matrix = np.identity(3)

    # K_squared = K @ K (equivalent to np.outer(unit_axis, unit_axis) - identity_matrix)
    K_squared = np.array([
        [kx * kx - 1, kx * ky, kx * kz],
        [ky * kx, ky * ky - 1, ky * kz],
        [kz * kx, kz * ky, kz * kz - 1]
    ])  # Using pre-calculated FlightCanvas for K^2

    # Rodrigues' formula
    R = identity_matrix + sin_theta * K + one_minus_cos_theta * K_squared

    T = np.eye(4)

    # Insert the 3x3 rotation matrix into the top-left block
    T[:3, :3] = R

    return T


def translation_matrix(
        translation_vector: Union[np.ndarray, List[float]]
) -> np.ndarray:
    """
    Creates a 4x4 homogeneous transformation matrix representing a pure translation.

    Args:
        translation_vector (Union[np.ndarray, List[float]]): A 3D vector [tx, ty, tz]
                                                              representing the translation.

    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix.

    Raises:
        ValueError: If translation_vector is not a 3-element array.
    """
    translation_vector = np.array(translation_vector, dtype=float)
    if translation_vector.shape != (3,):
        raise ValueError("`translation_vector` must be a 3-element list, tuple, or NumPy array.")

    # Create a 4x4 identity matrix
    transform_matrix = np.eye(4)

    # Insert the 3x1 translation vector into the last column (index 3)
    # of the first three rows (indices 0, 1, 2)
    transform_matrix[:3, 3] = translation_vector

    return transform_matrix


def dir_cosine_np(q) -> np.ndarray:
    """
    Returns a 3x3 numpy array that rotates from BODY frame to INERTIAL frame.
    """
    return np.array([
        [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])],
        [2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])],
        [2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]
    ])


def dir_cosine_ca(q) -> ca.Function:
    """
    Returns a 3x3 casadi matrix that rotates from BODY frame to INERTIAL frame.
    """
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    # DCM for q = [w, x, y, z]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)),
        ca.horzcat(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)),
        ca.horzcat(2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2))
    )


def euler_to_quat(a):
    """
    Convert Euler angles to a quaternion.
    """
    a = np.deg2rad(a)

    cy = np.cos(a[1] * 0.5)  # Cosine of half pitch
    sy = np.sin(a[1] * 0.5)  # Sine of half pitch
    cr = np.cos(a[0] * 0.5)  # Cosine of half roll
    sr = np.sin(a[0] * 0.5)  # Sine of half roll
    cp = np.cos(a[2] * 0.5)  # Cosine of half yaw
    sp = np.sin(a[2] * 0.5)  # Sine of half yaw

    q = np.zeros(4)

    q[0] = cy * cr * cp + sy * sr * sp  # w component
    q[1] = cy * sr * cp - sy * cr * sp  # x component
    q[2] = sy * cr * cp - cy * sr * sp  # y component
    q[3] = cy * cr * sp + sy * sr * cp  # z component

    return q


#@jit(nopython=True)
def linear_interpolation(alpha_grid, beta_grid, alpha, beta, data):
    """
    Perform bilinear interpolation for a 2D grid with multiple output dimensions.
    """
    # Find indices for x and y
    i = np.searchsorted(alpha_grid, alpha) - 1
    j = np.searchsorted(beta_grid, beta) - 1
    i = max(0, min(i, len(alpha_grid) - 2))
    j = max(0, min(j, len(beta_grid) - 2))

    # Get grid corners for each output dimension
    x0, x1 = alpha_grid[i], alpha_grid[i + 1]
    y0, y1 = beta_grid[j], beta_grid[j + 1]

    # Initialize the interpolated values
    output_dim = data.shape[2]
    interpolated_values = np.zeros(output_dim)

    for k in range(output_dim):
        Q11, Q21 = data[i, j, k], data[i + 1, j, k]
        Q12, Q22 = data[i, j + 1, k], data[i + 1, j + 1, k]

        # Bilinear interpolation formula for each output dimension
        interpolated_values[k] = (
                                         Q11 * (x1 - alpha) * (y1 - beta) +
                                         Q21 * (alpha - x0) * (y1 - beta) +
                                         Q12 * (x1 - alpha) * (beta - y0) +
                                         Q22 * (alpha - x0) * (beta - y0)
                                 ) / ((x1 - x0) * (y1 - y0))

    return interpolated_values


def omega(w):
    return np.array([
        [0, -w[0], -w[1], -w[2]],
        [w[0], 0, w[2], -w[1]],
        [w[1], -w[2], 0, w[0]],
        [w[2], w[1], -w[0], 0],
    ])


def omega_ca(w: ca.MX) -> ca.MX:
    # Extract the components of the angular velocity vector
    wx, wy, wz = w[0], w[1], w[2]

    # Construct the Omega matrix using CasADi's horzcat and vertcat
    # for creating matrices from symbolic components.
    Omega = ca.vertcat(
        ca.horzcat(0, -wx, -wy, -wz),
        ca.horzcat(wx, 0, wz, -wy),
        ca.horzcat(wy, -wz, 0, wx),
        ca.horzcat(wz, wy, -wx, 0)
    )

    return Omega


def normalize_quaternion(q):
    """
    Normalizes quaternion so that it is unit quaternion
    """
    norm = np.sqrt(np.sum(q ** 2))
    return q / norm


def rotate_z(rotate_angle_deg):
    """
    Generate a Z-axis rotation matrix.
    :param rotate_angle_deg: Rotation angle in degrees.
    """
    rotate_angle = np.deg2rad(rotate_angle_deg)
    cos_phi, sin_phi = np.cos(rotate_angle), np.sin(rotate_angle)

    R_Comp_Body = np.array([
        [1, 0, 0],
        [0, cos_phi, sin_phi],
        [0, -sin_phi, cos_phi],
    ])

    return R_Comp_Body


def interp_state(t_arr, x_arr, sim_time):
    index = np.searchsorted(t_arr, sim_time) - 1
    index = np.clip(index, 0, x_arr.shape[1] - 2)

    t0 = t_arr[index]
    t1 = t_arr[index + 1]
    alpha = (sim_time - t0) / (t1 - t0)

    # Interpolate state
    state0 = x_arr[:, index]
    state1 = x_arr[:, index + 1]
    state = state0 + alpha * (state1 - state0)

    return state


def print_states(t_arr: np.ndarray, x_arr: np.ndarray):
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


'''
def rotate_mesh_by_matrix(
    mesh: pv.PolyData,
    rotation_matrix: np.ndarray
) -> pv.PolyData:
    """
    Rotates a PyVista mesh by a given 3x3 rotation matrix.

    This function creates a new mesh with the transformed points.

    Args:
        mesh (pv.PolyData): The PyVista PolyData mesh to rotate.
        rotation_matrix (np.ndarray): A 3x3 NumPy array representing the rotation matrix.

    Returns:
        pv.PolyData: A new PyVista PolyData mesh with the rotated points.

    Raises:
        ValueError: If the rotation_matrix is not a 3x3 array.
    """
    # Validate the input rotation matrix shape
    if rotation_matrix.shape != (3, 3):
        raise ValueError("The 'rotation_matrix' must be a 3x3 NumPy array.")

    # Create a 4x4 homogeneous transformation matrix from the 3x3 rotation matrix.
    # The top-left 3x3 block is the rotation, and the last column (translation part) is zero.
    # The bottom-right element is 1.
    transform_matrix = np.eye(4) # Start with a 4x4 identity matrix
    transform_matrix[:3, :3] = rotation_matrix # Insert the 3x3 rotation matrix

    # Use PyVista's built-in transform method to apply the transformation.
    # By setting inplace=False, a new mesh object is returned, preserving the original.
    rotated_mesh = mesh.transform(transform_matrix, inplace=False)

    return rotated_mesh
'''
