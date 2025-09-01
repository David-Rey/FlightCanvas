
from FlightCanvas.vehicle.aero_vehicle import AeroVehicle
from FlightCanvas.vehicle.vehicle_visualizer import VehicleVisualizer
import pyvista as pv
import numpy as np
from FlightCanvas import utils

class StarshipVisualizer(VehicleVisualizer):
    def __init__(self, vehicle: AeroVehicle):
        super().__init__(vehicle)


    def generate_square_traj(self):
        t_arr, x_arr, u_arr = self.vehicle.controller.get_control_history()

        tf = t_arr[-1]
        hoops_per_sec = 4

        t_hoops = np.arange(0, tf, hoops_per_sec)

        for t in t_hoops:
            state, control = utils.interp_state(t_arr, x_arr, u_arr, t)
            state[0] = -state[0]

            quat = state[6:10]
            C_B_I = utils.dir_cosine_np(quat)
            pos = state[:3] + (C_B_I @ self.vehicle.xyz_ref)

            normal = C_B_I @ np.array([0, 0 ,1])

            square = self.create_square(pos, normal, 200, 200)
            self.pl.add_mesh(square, color='red', line_width=3)

    def generate_z_line(self):
        n = 1500
        points = np.array([
            [0.0, 0.0, 0.0],  # Start point
            [0.0, 0.0, n]  # End point
        ])

        # Define the line connectivity (2 points, indices 0 and 1)
        lines = np.array([2, 0, 1])

        # Create the PyVista line mesh
        line_mesh = pv.PolyData(points, lines=lines)
        self.pl.add_mesh(line_mesh, color='#808080', line_width=4)


    @staticmethod
    def create_square(position: np.ndarray, normal: np.ndarray, width: float, height: float) -> pv.PolyData:
        """
        Create a square (plane) in 3D space at a given position with specified normal vector.
        :param position: 3D coordinates [x, y, z] for the center of the square
        :param normal: Normal vector [nx, ny, nz] defining the orientation of the square
        :param width: Width of the square
        :param height: Height of the square
        :return: yVista mesh representing the square
        """
        # Convert inputs to numpy arrays
        position = np.array(position, dtype=float)
        normal = np.array(normal, dtype=float)

        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)

        # Create two orthogonal vectors in the plane of the square
        # Find a vector that's not parallel to the normal
        if abs(normal[0]) < 0.9:
            temp = np.array([1.0, 0.0, 0.0])
        else:
            temp = np.array([0.0, 1.0, 0.0])

        # Create first tangent vector (cross product with temp vector)
        tangent1 = np.cross(normal, temp)
        tangent1 = tangent1 / np.linalg.norm(tangent1)

        # Create second tangent vector (cross product of normal and first tangent)
        tangent2 = np.cross(normal, tangent1)
        tangent2 = tangent2 / np.linalg.norm(tangent2)

        # Define the four corners of the square relative to center
        half_width = width / 2.0
        half_height = height / 2.0

        corners = np.array([
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height]
        ])

        # Transform corners to 3D space using the tangent vectors
        vertices = []
        for corner in corners:
            vertex = position + corner[0] * tangent1 + corner[1] * tangent2
            vertices.append(vertex)

        vertices = np.array(vertices)

        # Create edges (lines forming the square outline)
        edges = np.hstack([
            [2, 0, 1],  # Edge from vertex 0 to 1
            [2, 1, 2],  # Edge from vertex 1 to 2
            [2, 2, 3],  # Edge from vertex 2 to 3
            [2, 3, 0]  # Edge from vertex 3 to 0
        ])

        # Create the mesh with lines
        square_mesh = pv.PolyData(vertices, lines=edges)

        return square_mesh