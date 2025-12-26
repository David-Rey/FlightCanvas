from FlightCanvas.vehicle.vehicle_dynamics import VehicleDynamics
import numpy as np
from FlightCanvas import utils
import numdifftools as nd
from scipy.optimize import minimize
import control as ct
from matplotlib import pyplot as plt


class Trimpoint:
    x_star: np.ndarray      # The trimmed 13 dimensional state vector (13)
    z_star: np.ndarray      # The trimmed design variable (3)
    u_star: np.ndarray      # The trimmed deflections (5)
    c_star: np.ndarray      # The Trimmed control (4)

    x_star_pitch: np.ndarray
    u_star_pitch: np.ndarray

    A_full: np.ndarray
    B_full: np.ndarray

    A_pitch: np.ndarray
    B_pitch: np.ndarray
    K_pitch: np.ndarray

    A_yaw: np.ndarray
    B_yaw: np.ndarray
    K_yaw: np.ndarray

    A_roll: np.ndarray
    B_roll: np.ndarray
    K_roll: np.ndarray

    x_dot_full: np.ndarray
    x_dot_pitch: np.ndarray

    nominal_pos = np.array([0, 0, 0])
    nominal_vel = np.array([0, 0, 0])
    nominal_quat = utils.euler_to_quat((0, 0, 0))
    nominal_omega = np.array([0, 0, 0])

    # Pitch params
    default_pitch = -0.6
    default_drag = np.deg2rad(40)


    def __init__(self, vehicle_dynamics: VehicleDynamics):
        self.vehicle_dynamics = vehicle_dynamics

        self.pitch_ROM = self.pitch_vel_ROM

    def x_bar_pitch(self, z: np.ndarray) -> np.ndarray:
        """
        Takes in design parameters z (w [speed in down direction], u [speed in the forward direction], delta [control pitch])
        and returns full state
        """
        state = np.concatenate((self.nominal_pos, self.nominal_vel, self.nominal_quat, self.nominal_omega))
        state[5] = z[0]
        state[3] = z[1]
        quat_pitch = utils.euler_to_quat((0, self.default_pitch, 0))
        state[6:10] = quat_pitch
        return state


    def nominal_control(self):
        return np.array([self.z_star[2], 0, 0, self.default_drag])

    def u_bar_pitch(self, z: np.ndarray) -> np.ndarray:
        """
        Takes in design parameters z (w [speed in down direction], u [speed in the forward direction], delta [control pitch])
        and returns control deflections
        """
        control_inputs = np.array([z[2], 0, 0, self.default_drag])
        component_deflections = self.vehicle_dynamics.allocation_matrix @ control_inputs
        return component_deflections

    def control2deflections(self, pitch, yaw, roll):
        control_inputs = np.array([pitch, yaw, roll, self.default_drag])
        component_deflections = self.vehicle_dynamics.allocation_matrix @ control_inputs
        return component_deflections

    def trim_objective(self, z: np.ndarray) -> float:
        """
        Takes in initial guess z and returns a cost of the objective function
        """
        x = self.x_bar_pitch(z)
        u = self.u_bar_pitch(z)

        jac_fun = nd.Jacobian(self.pitch_ROM)
        Jg = jac_fun(x)
        x_dot = self.vehicle_dynamics.dynamics(x, u)

        x_rom_dot = Jg @ x_dot
        cost = np.linalg.norm(x_rom_dot)
        return cost

    def get_trimpoint(self, z_guess: np.ndarray):
        self.perform_trim_optimization(z_guess)
        self.find_pitch_ROM()
        self.find_yaw_ROM()

    def perform_trim_optimization(self, z_guess: np.ndarray):

        opt_settings = {
            'maxiter': 1000,
            'ftol': 1e-9,
            'disp': True,
            'eps': 1.49e-08
        }

        res_pitch = minimize(self.trim_objective, z_guess, method='SLSQP', options=opt_settings)
        self.z_star = res_pitch.x
        self.x_star = self.x_bar_pitch(self.z_star)
        self.u_star = self.u_bar_pitch(self.z_star)

        jac_fun = nd.Jacobian(self.pitch_ROM)
        Jg = jac_fun(self.x_star)
        self.x_dot_full = self.vehicle_dynamics.dynamics(self.x_star, self.u_star)

        self.x_dot_pitch = Jg @ self.x_dot_full
        self.x_star_pitch = Jg @ self.x_star

        print(f"Z: {res_pitch.x}")
        print(f"Cost: {res_pitch.fun}")
        print(f"ROM dot: {self.x_dot_pitch}")

    def find_pitch_ROM(self):
        pitch_jac = nd.Jacobian(self.pitch_ROM)
        Jg_pitch = pitch_jac(self.x_star)

        fx = lambda x: self.vehicle_dynamics.dynamics(x, self.u_star)
        fu = lambda pitch: self.vehicle_dynamics.dynamics(self.x_star, self.control2deflections(pitch[0], 0, 0))

        A_jac_fun = nd.Jacobian(fx)
        B_jac_fun = nd.Jacobian(fu)

        A = A_jac_fun(self.x_star).squeeze()
        B = B_jac_fun(self.z_star[2]).reshape(-1, 1)

        T_pitch = np.linalg.pinv(Jg_pitch)

        self.A_pitch = Jg_pitch @ A @ T_pitch
        self.B_pitch = Jg_pitch @ B

    def find_yaw_ROM(self):
        yaw_jac = nd.Jacobian(self.yaw_ROM)
        Jg_yaw = yaw_jac(self.x_star)
        nominal_control = self.nominal_control()
        nominal_deflections = self.control2deflections(nominal_control[0], nominal_control[1], nominal_control[2])
        nominal_deflections[3] = 0

        fx = lambda x: self.vehicle_dynamics.dynamics(x, self.u_star)

        fu = lambda yaw: self.vehicle_dynamics.dynamics(self.x_star, nominal_deflections + self.control2deflections(0, yaw[0], 0))

        A_jac_fun = nd.Jacobian(fx)
        B_jac_fun = nd.Jacobian(fu)

        A = A_jac_fun(self.x_star).squeeze()
        B = B_jac_fun(0).reshape(-1, 1)

        T_yaw = np.linalg.pinv(Jg_yaw)

        self.A_yaw = Jg_yaw @ A @ T_yaw
        self.B_yaw = Jg_yaw @ B
        print(1)

    def init_LQR(self):
        self.init_LQR_pitch()
        self.init_LQR_yaw()

    def init_LQR_pitch(self):
        q_weights = [0.001, 0, 0, 0]  # [u w pitch_rate pitch]
        r_weight = 1

        Q = np.diag(q_weights)
        R = np.array([[r_weight]])

        # Compute the LQR gain matrix K
        self.K_pitch, S, E = ct.lqr(self.A_pitch, self.B_pitch, Q, R)

    def init_LQR_yaw(self):
        q_weights = [1, 0.01]  # [yaw_rate yaw]
        r_weight = 1

        Q = np.diag(q_weights)
        R = np.array([[r_weight]])

        # Compute the LQR gain matrix K
        self.K_yaw, S, E = ct.lqr(self.A_yaw, self.B_yaw, Q, R)

    def get_control(self, state: np.ndarray) -> np.ndarray:
        pitch_rom = self.pitch_ROM(state)
        yaw_rom = self.yaw_ROM(state)
        roll_rom = self.roll_ROM(state)

        # Pitch
        pitch_ref = np.array([-5, 0, 0, 0])  # [vx, vz, pitch_rate, pitch]
        pitch_error = pitch_ref - pitch_rom
        rel_pitch_control = self.K_pitch @ pitch_error
        pitch_control = np.array([rel_pitch_control[0], 0, 0, 0])

        # Yaw
        yaw_ref = np.deg2rad(np.array([0, 1]))
        yaw_error = yaw_ref - yaw_rom
        rel_yaw_control = self.K_yaw @ yaw_error
        yaw_control = np.array([0, rel_yaw_control[0], 0, 0])

        nominal_control = self.nominal_control()
        control = nominal_control + pitch_control + yaw_control
        return control

    def plot_pz_with_feedback(self):
        # 1. Define the Open-Loop System (G)
        C = np.array([[0, 0, 1]])  # Note: ensure C is 2D for ss
        D = np.array([[0]])
        sys_open = ct.ss(self.A_pitch[:3, :3], self.B_pitch[:3], C, D)

        # 2. Define the Feedback Gain (K)
        # You can start with K=1.0 and increase it if the system remains unstable.
        K = 20

        # 3. Create the Closed-Loop System (T)
        # T = G / (1 + G*K)
        sys_closed = ct.feedback(sys_open, K)

        # 4. Extract Poles and Zeros for Closed-Loop
        p_cl = ct.poles(sys_closed)
        z_cl = ct.zeros(sys_closed)

        print(f"Closed-Loop Poles: {p_cl}")
        print(f"Closed-Loop Zeros: {z_cl}")

        # 5. Plotting
        plt.figure(figsize=(10, 5))

        # Open Loop Map
        plt.subplot(1, 2, 1)
        ct.pzmap(sys_open, title="Open-Loop Pole/Zero")
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.grid(True)

        # Closed Loop Map
        plt.subplot(1, 2, 2)
        ct.pzmap(sys_closed, title=f"Closed-Loop Pole/Zero (K={K})")
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def roll_ROM(full_state: np.ndarray) -> np.ndarray:
        v = full_state[4]
        w = full_state[5]

        q0 = full_state[6]
        q1 = full_state[7]
        q2 = full_state[8]
        q3 = full_state[9]

        roll_rate = full_state[10]
        roll = np.arctan2(2 * (q3 * q0 + q1 * q2), 1 - 2 * (q0 ** 2 + q1 ** 2))

        rom = np.array([v, w, roll_rate, roll])
        return rom

    @staticmethod
    def yaw_ROM(full_state: np.ndarray) -> np.ndarray:
        q0 = full_state[6]
        q1 = full_state[7]
        q2 = full_state[8]
        q3 = full_state[9]

        yaw_rate = full_state[12]
        yaw = np.arctan2(2 * (q3 * q2 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2))

        rom = np.array([yaw_rate, yaw])
        return rom


    @staticmethod
    def pitch_vel_ROM(full_state: np.ndarray) -> np.ndarray:
        u = full_state[3]
        w = full_state[5]

        q0 = full_state[6]
        q1 = full_state[7]
        q2 = full_state[8]
        q3 = full_state[9]

        pitch = np.arcsin(2 * (q3 * q1 - q2 * q0))
        pitch_rate = full_state[11]

        rom = np.array([u, w, pitch_rate, pitch])
        return rom


    @staticmethod
    def pitch_aoa_ROM(full_state: np.ndarray) -> np.ndarray:
        u = full_state[3]
        v = full_state[4]
        w = full_state[5]

        q0 = full_state[6]
        q1 = full_state[7]
        q2 = full_state[8]
        q3 = full_state[9]

        pitch = np.arcsin(2 * (q3 * q1 - q2 * q0))
        alpha = np.arctan2(w, u)
        q = full_state[11]
        speed = np.linalg.norm([u, v, w])

        rom = np.array([alpha, q, pitch, speed])
        return rom

    @staticmethod
    def get_yaw_error(state, target_yaw):
        target_quat = utils.euler_to_quat([0, 0, target_yaw])
        # Calculate Error Quaternion
        q_curr = state[6:10]
        q_inv = utils.quat_inverse(q_curr)
        q_err = utils.quat_multiply(target_quat, q_inv)

        # 2. Extract the Y-axis (Pitch) component
        # For [w, x, y, z] convention, y is index 2
        w_err = q_err[0]
        z_err = q_err[3]

        # Multiplied by 2 to get approximate angle in radians for small errors
        yaw_error_rad = 2 * np.sign(w_err) * z_err

        return yaw_error_rad