
from FlightCanvas.vehicle.aero_vehicle import AeroVehicle
from FlightCanvas.analysis.vehicle_visualizer import VehicleVisualizer
import pyvista as pv
import numpy as np
from FlightCanvas import utils
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class StarshipVisualizer(VehicleVisualizer):
    def __init__(self, vehicle: AeroVehicle):
        super().__init__(vehicle)


    def generate_square_traj(self):
        t_arr, x_arr, u_arr = self.vehicle.get_control_history()

        tf = t_arr[-1]
        hoops_per_sec = 3

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

    def animate_mpc_horizon(self, save_path=None, interval=100):
        """
        Create an animated plot showing MPC horizon evolution for position and flap deflection

        Args:
            save_path: Optional path to save animation as gif/mp4
            interval: Animation interval in milliseconds
        """
        controller = self.vehicle.controller
        # Extract data
        simX = controller.simX
        simXhorizon = controller.simXhorizon
        simU = controller.simU
        simUhorizon = controller.simUhorizon  # Control horizon
        simT = controller.simT
        dt = controller.dt
        N_horizon = controller.N_horizon
        Nsim = controller.Nsim

        # Create time vectors
        sim_time = np.arange(Nsim + 1) * dt  # Actual simulation time
        horizon_time_relative = np.arange(N_horizon + 1) * dt  # Relative horizon time
        horizon_time_relative_u = np.arange(N_horizon) * dt  # Control horizon time

        # Set up the figure with 2x2 layout
        fig, ((ax_xz, ax_flaps), (ax_yz, ax_controls)) = plt.subplots(2, 2, figsize=(16, 10))

        # Initialize empty line objects
        lines = {}

        # XZ trajectory plot (left top)
        lines['actual_xz'], = ax_xz.plot([], [], 'b-', linewidth=2, label='Actual')
        lines['horizon_xz'], = ax_xz.plot([], [], 'r--', linewidth=2, alpha=0.7, label='Predicted')
        lines['current_xz'], = ax_xz.plot([], [], 'go', markersize=8, label='Current')

        # YZ trajectory plot (left bottom)
        lines['actual_yz'], = ax_yz.plot([], [], 'b-', linewidth=2, label='Actual')
        lines['horizon_yz'], = ax_yz.plot([], [], 'r--', linewidth=2, alpha=0.7, label='Predicted')
        lines['current_yz'], = ax_yz.plot([], [], 'go', markersize=8, label='Current')

        # Flap deflections (4 flaps) - right top
        flap_colors = ['red', 'blue', 'green', 'orange']
        flap_labels = ['Front Flap', 'Front Flap Star', 'Aft Flap', 'Aft Flap Star']

        for i in range(4):
            lines[f'actual_flap_{i}'], = ax_flaps.plot([], [], color=flap_colors[i],
                                                       linewidth=2, label=f'{flap_labels[i]} Actual')
            lines[f'horizon_flap_{i}'], = ax_flaps.plot([], [], color=flap_colors[i],
                                                        linestyle='--', linewidth=2, alpha=0.7,
                                                        label=f'{flap_labels[i]} Predicted')
            lines[f'current_flap_{i}'], = ax_flaps.plot([], [], 'o', color=flap_colors[i], markersize=6)

        # Control inputs (4 controls) - right bottom
        control_colors = ['red', 'blue', 'green', 'orange']
        control_labels = ['Pitch Control', 'Roll Control', 'Yaw Control', 'Drag Control']

        for i in range(4):
            lines[f'actual_control_{i}'], = ax_controls.plot([], [], color=control_colors[i],
                                                             linewidth=2, label=f'{control_labels[i]} Actual')
            lines[f'horizon_control_{i}'], = ax_controls.plot([], [], color=control_colors[i],
                                                              linestyle='--', linewidth=2, alpha=0.7,
                                                              label=f'{control_labels[i]} Predicted')
            lines[f'current_control_{i}'], = ax_controls.plot([], [], 'o', color=control_colors[i], markersize=6)

        # Set up axes labels and titles
        ax_xz.set_xlabel('X Position (m)')
        ax_xz.set_ylabel('Z Position (m)')
        ax_xz.set_title('XZ Trajectory')
        ax_xz.grid(True)
        ax_xz.legend()
        ax_xz.invert_yaxis()  # Invert Z axis for typical aerospace convention

        ax_yz.set_xlabel('Y Position (m)')
        ax_yz.set_ylabel('Z Position (m)')
        ax_yz.set_title('YZ Trajectory')
        ax_yz.grid(True)
        ax_yz.legend()
        ax_yz.invert_yaxis()  # Invert Z axis for typical aerospace convention

        ax_flaps.set_xlabel('Time (s)')
        ax_flaps.set_ylabel('Flap Deflection (deg)')
        ax_flaps.set_title('Flap Deflections vs Time')
        ax_flaps.grid(True)
        ax_flaps.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        ax_controls.set_xlabel('Time (s)')
        ax_controls.set_ylabel('Control Rate (deg/s)')
        ax_controls.set_title('Control Rates vs Time')
        ax_controls.grid(True)
        ax_controls.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Set axis limits
        x_min, x_max = np.min(simX[0, :]), np.max(simX[0, :])
        y_min, y_max = np.min(simX[1, :]), np.max(simX[1, :])
        z_min, z_max = np.min(simX[2, :]), np.max(simX[2, :])

        margin = 0.1
        ax_xz.set_xlim(x_min - margin * abs(x_max - x_min), x_max + margin * abs(x_max - x_min))
        ax_xz.set_ylim(z_min - margin * abs(z_max - z_min), z_max + margin * abs(z_max - z_min))

        ax_yz.set_xlim(y_min - margin * abs(y_max - y_min), y_max + margin * abs(y_max - y_min))
        ax_yz.set_ylim(z_min - margin * abs(z_max - z_min), z_max + margin * abs(z_max - z_min))

        t_max = sim_time[-1]
        ax_flaps.set_xlim(0, t_max)
        ax_controls.set_xlim(0, t_max)

        # Flap limits in degrees
        flap_min = np.min(np.rad2deg(simX[13:17, :]))
        flap_max = np.max(np.rad2deg(simX[13:17, :]))
        ax_flaps.set_ylim(flap_min - 5, flap_max + 5)

        # Control limits in deg/s
        control_min = np.min(np.rad2deg(simU))
        control_max = np.max(np.rad2deg(simU))
        ax_controls.set_ylim(control_min - 2, control_max + 2)

        # Animation function
        def animate(frame):
            k = frame
            if k >= Nsim:
                k = Nsim - 1

            current_time = k * dt

            # Ensure we don't go out of bounds
            k_states = min(k + 1, simX.shape[1] - 1)  # For states (Nsim + 1)
            k_controls = min(k, simU.shape[1] - 1)  # For controls (Nsim)

            # Update actual trajectories (up to current time)
            actual_time_states = sim_time[:k_states + 1]
            actual_time_controls = sim_time[:k_controls + 1] if k_controls >= 0 else np.array([0])

            # XZ trajectory
            lines['actual_xz'].set_data(simX[0, :k_states + 1], simX[2, :k_states + 1])
            lines['current_xz'].set_data([simX[0, k]], [simX[2, k]])

            # YZ trajectory
            lines['actual_yz'].set_data(simX[1, :k_states + 1], simX[2, :k_states + 1])
            lines['current_yz'].set_data([simX[1, k]], [simX[2, k]])

            # Flap deflections (actual)
            for i in range(4):
                lines[f'actual_flap_{i}'].set_data(actual_time_states, np.rad2deg(simX[13 + i, :k_states + 1]))
                lines[f'current_flap_{i}'].set_data([current_time], [np.rad2deg(simX[13 + i, k])])

            # Control inputs (actual)
            for i in range(4):
                if k_controls >= 0 and len(actual_time_controls) > 0:
                    control_data = np.rad2deg(simU[i, :k_controls + 1])
                    if len(control_data) == len(actual_time_controls):
                        lines[f'actual_control_{i}'].set_data(actual_time_controls, control_data)
                    if k_controls < simU.shape[1]:
                        lines[f'current_control_{i}'].set_data([current_time], [np.rad2deg(simU[i, k_controls])])

            # Update predicted horizon
            if k < Nsim and k < simXhorizon.shape[1]:
                horizon_time = current_time + horizon_time_relative
                horizon_time_u = current_time + horizon_time_relative_u

                # XZ predicted trajectory
                horizon_x = simXhorizon[0, k, :]
                horizon_z = simXhorizon[2, k, :]
                lines['horizon_xz'].set_data(horizon_x, horizon_z)

                # YZ predicted trajectory
                horizon_y = simXhorizon[1, k, :]
                lines['horizon_yz'].set_data(horizon_y, horizon_z)

                # Flap deflections prediction
                for i in range(4):
                    horizon_flap = np.rad2deg(simXhorizon[13 + i, k, :])
                    lines[f'horizon_flap_{i}'].set_data(horizon_time, horizon_flap)

                # Control inputs prediction (only if simUhorizon exists)
                if hasattr(controller, 'simUhorizon') and k < simUhorizon.shape[1]:
                    for i in range(4):
                        try:
                            horizon_control = np.rad2deg(simUhorizon[i, k, :])
                            # Debug: print shapes to understand the issue
                            #print(
                            #    f"Debug - Control {i}: horizon_control shape: {horizon_control.shape}, horizon_time_u shape: {horizon_time_u.shape}")

                            if len(horizon_control) == len(horizon_time_u):
                                lines[f'horizon_control_{i}'].set_data(horizon_time_u, horizon_control)
                            else:
                                # Handle shape mismatch - truncate or pad as needed
                                min_len = min(len(horizon_control), len(horizon_time_u))
                                lines[f'horizon_control_{i}'].set_data(horizon_time_u[:min_len],
                                                                       horizon_control[:min_len])
                        except Exception as e:
                            print(f"Error plotting starship_control horizon {i}: {e}")
                            # Clear the line if there's an error
                            lines[f'horizon_control_{i}'].set_data([], [])

            return list(lines.values())

        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=Nsim, interval=interval,
                                       blit=False, repeat=True)

        plt.tight_layout()

        # Save animation if path provided
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=10)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=10)
            print(f"Animation saved to: {save_path}")

        plt.show()

        return anim