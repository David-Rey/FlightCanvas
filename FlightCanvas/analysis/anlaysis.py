
from FlightCanvas.analysis.log import Log
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p


class Analysis:
    def __init__(self, log: Log):
        self.log = log
        self.num_samples = log.current_idx

    def generate_velocity_plot(self, include_vz=True):
        time = self.log.time[0:self.log.current_idx + 1]

        vx = self.log.get_state("vx")
        vy = self.log.get_state("vy")
        vz = self.log.get_state("vz")

        plt.figure()
        plt.plot(time, vx, label='vx (m/s)')
        plt.plot(time, vy, label='vy (m/s)')
        if include_vz:
            plt.plot(time, vz, label='vz (m/s)')

        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (deg)')
        plt.title('Velocity Components (Body)')
        plt.legend()
        plt.grid(True)
        plt.gcf().set_size_inches(6, 4)
        p.show_plot(rotate_axis_labels=False)

    def generate_position_plot(self):
        time = self.log.time[0:self.log.current_idx + 1]

        px = self.log.get_state("x")
        py = self.log.get_state("y")
        pz = self.log.get_state("z")

        plt.figure()
        plt.plot(time, px, label='vx (m)')
        plt.plot(time, py, label='vy (m)')
        plt.plot(time, pz, label='vz (m)')

        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Position Components (Inertial)')
        plt.legend()
        plt.grid(True)
        plt.gcf().set_size_inches(6, 4)
        p.show_plot(rotate_axis_labels=False)

    def generate_quat_norm_plot(self):
        time = self.log.time[0:self.log.current_idx + 1]

        q0 = self.log.get_state("q0")
        q1 = self.log.get_state("q1")
        q2 = self.log.get_state("q2")
        q3 = self.log.get_state("q3")

        norm = np.sqrt(q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        plt.figure()
        plt.plot(time, norm, label='q_norm')

        plt.xlabel('Time (s)')
        plt.ylabel('q_norm')
        plt.title('Quaterion Norm')
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

        pitch = np.rad2deg(np.arcsin(2 * (q3 * q1 - q2 * q0)))

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(time, euler[:, 0], label='Yaw (Z)')

        plt.plot(time, euler[:, 1], label='Pitch (Y)')
        #plt.plot(time, pitch, label='Pitch (Y)')
        plt.plot(time, euler[:, 2], label='Roll (X)')

        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title('Euler Angles')
        plt.legend()
        plt.grid(True)
        plt.gcf().set_size_inches(6, 4)
        p.show_plot(rotate_axis_labels=False)

    def generate_angular_velocity_plot(self):
        time = self.log.time[0:self.log.current_idx+1]

        wx = np.rad2deg(self.log.get_state("wx"))
        wy = np.rad2deg(self.log.get_state("wy"))
        wz = np.rad2deg(self.log.get_state("wz"))

        plt.figure()
        plt.plot(time, wx, label='wx (deg/s)')
        plt.plot(time, wy, label='wy (deg/s)')
        plt.plot(time, wz, label='wz (deg/s)')

        plt.xlabel('Time (s)')
        plt.ylabel('Angular Rate (deg)')
        plt.title('Angular Rate')
        plt.legend()
        plt.grid(True)
        plt.gcf().set_size_inches(6, 4)
        p.show_plot(rotate_axis_labels=False)

    def generate_control_plot(self):
        time = self.log.time[0:self.log.current_idx + 1]

        pitch = np.rad2deg(self.log.get_control_input("pitch"))
        yaw = np.rad2deg(self.log.get_control_input("yaw"))
        roll = np.rad2deg(self.log.get_control_input("roll"))

        plt.figure()
        plt.plot(time, pitch, label='pitch (deg)')
        plt.plot(time, yaw, label='yaw (deg)')
        plt.plot(time, roll, label='roll (deg)')

        plt.xlabel('Time (s)')
        plt.ylabel('Control (deg)')
        plt.title('Control')
        plt.legend()
        plt.grid(True)
        plt.gcf().set_size_inches(6, 4)
        p.show_plot(rotate_axis_labels=False)

    def generate_angle_of_attack_plot(self):
        time = self.log.time[0:self.log.current_idx + 1]

        u = self.log.get_state("vx")
        w = self.log.get_state("vz")
        alpha = np.rad2deg(np.arctan2(w, u))

        plt.figure()
        plt.plot(time, alpha, label='alpha (deg)')
        plt.xlabel('Time (s)')
        plt.ylabel('Alpha (deg)')
        plt.title('Angle of Attack')
        plt.legend()
        plt.grid(True)
        plt.gcf().set_size_inches(6, 4)
        p.show_plot(rotate_axis_labels=False)

    def generate_true_deflections_plot(self):
        time = self.log.time[0:self.log.current_idx + 1]

        fl_deflections = np.rad2deg(self.log.get_deflection('f_left'))
        fr_deflections = np.rad2deg(self.log.get_deflection('f_right'))
        bl_deflections = np.rad2deg(self.log.get_deflection('b_left'))
        br_deflections = np.rad2deg(self.log.get_deflection('b_right'))

        plt.figure()
        plt.plot(time, fl_deflections, label='Front Left Flap (deg)')
        plt.plot(time, fr_deflections, label='Front Right Flap (deg)')
        plt.plot(time, bl_deflections, label='Back Left Flap (deg)')
        plt.plot(time, br_deflections, label='Back Right Flap (deg)')
        plt.xlabel('Time (s)')
        plt.ylabel('Deflections (deg)')
        plt.title('Angle of Attack')
        plt.legend()
        plt.grid(True)
        plt.gcf().set_size_inches(6, 4)
        p.show_plot(rotate_axis_labels=False)

    #def generate_cmd_deflections_plot(self, allocation_matrix):
        #time = self.log.time[0:self.log.current_idx + 1]

        #control = self.log.control_inputs
        #deflections = allocation_matrix @ control

