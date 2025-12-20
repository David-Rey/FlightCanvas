
from FlightCanvas.analysis.log import Log
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p


class Analysis:
    def __init__(self, log: Log):
        self.log = log
        self.num_samples = log.current_idx

    def generate_velocity_plot(self):
        time = self.log.time[0:self.log.current_idx + 1]

        vx = self.log.get_state("vx")
        vy = self.log.get_state("vy")
        vz = self.log.get_state("vz")

        plt.figure()
        plt.plot(time, vx, label='vx (m/s)')
        plt.plot(time, vy, label='vy (m/s)')
        plt.plot(time, vz, label='vz (m/s)')

        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (deg)')
        plt.title('Velocity Components (Inertial)')
        plt.legend()
        plt.grid(True)
        plt.gcf().set_size_inches(6, 4)
        p.show_plot(rotate_axis_labels=False)

    def generate_euler_angle_plot(self):
        time = self.log.time[0:self.log.current_idx+1]

        q0 = self.log.get_state("q0")
        q1 = self.log.get_state("q1")
        q2 = self.log.get_state("q2")
        q3 = self.log.get_state("q3")

        quat_matrix = np.vstack((-q1, -q2, -q3, q0)).T

        # Perform vectorized conversion
        rot = R.from_quat(quat_matrix)
        euler = rot.as_euler('zyx', degrees=True)  # Returns (N, 3) matrix

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(time, euler[:, 0], label='Yaw (Z)')
        plt.plot(time, euler[:, 1], label='Pitch (Y)')
        plt.plot(time, euler[:, 2], label='Roll (X)')

        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title('Euler Angles')
        plt.legend()
        plt.grid(True)
        plt.gcf().set_size_inches(6, 4)
        p.show_plot(rotate_axis_labels=False)

