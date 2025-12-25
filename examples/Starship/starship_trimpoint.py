from FlightCanvas.vehicle.vehicle_dynamics import VehicleDynamics
import numpy as np
from FlightCanvas import utils
import numdifftools as nd
from scipy.optimize import minimize
import control as ct
from matplotlib import pyplot as plt


class Trimpoint:
    x_star: np.ndarray
    z_star: np.ndarray
    u_star: np.ndarray

    x_star_pitch: np.ndarray
    u_star_pitch: np.ndarray

    A_full: np.ndarray
    B_full: np.ndarray

    A_pitch: np.ndarray
    B_pitch: np.ndarray
    K_pitch: np.ndarray

    x_dot_full: np.ndarray
    x_dot_pitch: np.ndarray

    nominal_pos = np.array([0, 0, 0])
    nominal_vel = np.array([0, 0, 0])
    nominal_quat = utils.euler_to_quat((0, 0, 0))
    nominal_omega = np.array([0, 0, 0])

    # Pitch params
    default_pitch = -0.6
    default_drag = 40

    def __init__(self, vehicle_dynamics: VehicleDynamics):
        self.vehicle_dynamics = vehicle_dynamics

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

    def u_bar_pitch(self, z: np.ndarray) -> np.ndarray:
        """
        Takes in design parameters z (w [speed in down direction], u [speed in the forward direction], delta [control pitch])
        and returns control deflections
        """
        control_inputs = np.deg2rad(np.array([z[2], 0, 0, self.default_drag]))
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
        opt_settings = {
            'maxiter': 1000,  # Default is often too low for complex aero models
            'ftol': 1e-9,  # Precision on the objective function
            'disp': True,  # Prints convergence messages so you can see WHY it stopped
            'eps': 1.49e-08  # Step size used for numerical approximation of the Jacobian
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

        fx = lambda x: self.vehicle_dynamics.dynamics(x, self.u_star)
        fu = lambda pitch: self.vehicle_dynamics.dynamics(self.x_star, self.u_bar_pitch(np.array([0, 0, pitch[0]])))

        A_jac_fun = nd.Jacobian(fx)
        B_jac_fun = nd.Jacobian(fu)

        A = A_jac_fun(self.x_star).squeeze()
        B = B_jac_fun(np.deg2rad(self.z_star[2])).reshape(-1, 1)

        T = np.linalg.pinv(Jg)

        self.A_pitch = Jg @ A @ T
        self.B_pitch = Jg @ B

        print(f"Z: {res_pitch.x}")
        print(f"Cost: {res_pitch.fun}")
        print(f"ROM dot: {self.x_dot_pitch}")

    def init_LQR(self, q_weights=None, r_weight=1.0):

        if q_weights is None:
            # [u, w, q, theta]
            q_weights = [1, 0.1, 0.1, 0.1]

        Q = np.diag(q_weights)
        R = np.array([[r_weight]])

        # Compute the LQR gain matrix K
        # K is the matrix such that u = -Kx stabilizes (A - BK)
        self.K_pitch, S, E = ct.lqr(self.A_pitch, self.B_pitch, Q, R)

    def get_control(self, state: np.ndarray) -> np.ndarray:
        rom = self.pitch_ROM(state)

        ref = np.deg2rad(np.array([-95, 0, 0, 0]))

        state_error = ref - rom + self.x_star_pitch
        rel_pitch_control = np.rad2deg(self.K_pitch @ state_error)

        #nominal_control = np.deg2rad(np.array([self.z_star[1], 0, 0, self.drag_control]))
        pitch_control = np.deg2rad(np.array([self.z_star[2] + rel_pitch_control[0], 0, 0, self.default_drag]))
        #pitch_control = np.array([rel_pitch_control, 0, 0, 0])
        #pitch_control = self.u_star + np.array([rel_pitch_control[0], 0, 0, 0])

        return pitch_control

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
    def pitch_ROM_2(full_state: np.ndarray) -> np.ndarray:
        vx = full_state[3]
        vz = full_state[5]

        q0 = full_state[6]
        q1 = full_state[7]
        q2 = full_state[8]
        q3 = full_state[9]

        pitch = np.arcsin(2 * (q3 * q1 - q2 * q0))
        pitch_rate = full_state[11]

        rom = np.array([vx, vz, pitch_rate, pitch])
        return rom


    @staticmethod
    def pitch_ROM(full_state: np.ndarray) -> np.ndarray:
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

        x4 = np.array([alpha, q, pitch, speed])
        return x4
