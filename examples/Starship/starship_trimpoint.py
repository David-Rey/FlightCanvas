from FlightCanvas.vehicle.vehicle_dynamics import VehicleDynamics
import numpy as np
from FlightCanvas import utils
import numdifftools as nd
from scipy.optimize import minimize
import control as ct
from matplotlib import pyplot as plt
from FlightCanvas.control.controller import Controller


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

    A_lateral: np.ndarray
    B_lateral: np.ndarray
    K_lateral: np.ndarray

    x_dot_full: np.ndarray
    x_dot_pitch: np.ndarray

    nominal_pos = np.array([0, 0, 0])
    nominal_vel = np.array([0, 0, 0])
    nominal_quat = utils.euler_to_quat((0, 0, 0))
    nominal_omega = np.array([0, 0, 0])

    # Pitch params
    default_pitch = 0.6
    default_drag = np.deg2rad(35)

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
        self.find_lateral_ROM()

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

    def find_lateral_ROM(self):
        lateral_jac = nd.Jacobian(self.lateral_ROM_quaternion, step=1e-7, method='central')
        Jg = lateral_jac(self.x_star)
        nominal_control = self.nominal_control()
        nominal_deflections = self.control2deflections(nominal_control[0], nominal_control[1], nominal_control[2])
        nominal_deflections[3] = 0

        fx = lambda x: self.vehicle_dynamics.dynamics(x, self.u_star)
        fu = lambda control: self.vehicle_dynamics.dynamics(self.x_star, nominal_deflections + self.control2deflections(0, control[0], control[1]))

        A_jac_fun = nd.Jacobian(fx, step=1e-4, method='central')
        B_jac_fun = nd.Jacobian(fu, step=1e-4, method='central')

        A = A_jac_fun(self.x_star).squeeze()
        B = B_jac_fun([0, 0]).squeeze()

        T_yaw = np.linalg.pinv(Jg)

        self.A_lateral = Jg @ A @ T_yaw
        self.B_lateral = Jg @ B

    def init_LQR(self):
        self.init_LQR_pitch()
        self.init_LQR_lateral()

    def init_LQR_pitch(self):
        q_weights = [0.001, 0, 0, 0]  # [u w pitch_rate pitch]
        r_weight = 1

        Q = np.diag(q_weights)
        R = np.array([[r_weight]])

        # Compute the LQR gain matrix K
        self.K_pitch, S, E = ct.lqr(self.A_pitch, self.B_pitch, Q, R)

    def init_LQR_lateral(self):
        q_weights = [0, 0.01, 0.5, 0.01, 1]  #
        r_weight = [1, 1]

        Q = np.diag(q_weights)
        R = np.diag(r_weight)

        # Compute the LQR gain matrix K
        self.K_lateral, S, E = ct.lqr(self.A_lateral, self.B_lateral, Q, R)

    def get_control(self, state: np.ndarray) -> np.ndarray:
        pitch_rom = self.pitch_ROM(state)
        lateral_rom = self.lateral_ROM_quaternion(state)

        # Pitch
        pitch_ref = np.array([16, 0, 0, 0])  # [vx, vz, pitch_rate, pitch]
        #pitch_ref = -np.deg2rad(3)
        pitch_error = pitch_ref - pitch_rom
        #rel_pitch_control = self.pitch_controller.update(pitch_error)
        rel_pitch_control = self.K_pitch @ pitch_error
        pitch_control = np.array([rel_pitch_control[0], 0, 0, 0])

        # Roll-Yaw
        lateral_rom[0] = 0
        lateral_ref = np.array([0, 0, 0, 0, np.deg2rad(0)])
        lateral_error = lateral_ref - lateral_rom
        #lateral_error[3] = self.wrap_to_pi(lateral_error[3])
        #lateral_error[4] = self.wrap_to_pi(lateral_error[4])
        rel_lateral_control = self.K_lateral @ lateral_error
        rel_lateral_control[0] = np.clip(rel_lateral_control[0], np.deg2rad(-2), np.deg2rad(2))
        rel_lateral_control[1] = np.clip(rel_lateral_control[1], np.deg2rad(-2), np.deg2rad(2))
        lateral_control = np.array([0, rel_lateral_control[0], rel_lateral_control[1], 0])

        nominal_control = self.nominal_control()
        control = nominal_control + pitch_control + lateral_control
        return control

    def pitch_control_analysis(self):
        # 1. Define the Open-Loop System (G)
        C = np.array([[1, 0, 0, 0]])  # Note: ensure C is 2D for ss
        D = np.array([[0]])
        sys_open = ct.ss(self.A_pitch, self.B_pitch, C, D)

        Ku, Tu = self.find_ultimate_params(sys_open)

        #Kp = 0.6 * Ku
        #Ti = 0.5 * Tu
        #Td = 0.125 * Tu

        # Convert to standard form gains
        #Ki = Kp / Ti
        #Kd = Kp * Td


        #print(f"Ultimate Gain (Ku): {Ku:.2f}")
        #print(f"Ultimate Period (Tu): {Tu:.2f}")
        #print(f"Calculated PID: Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}")

        Kp = 0.05
        Ki = 0.0
        Kd = 0.0

        controller = ct.TransferFunction([Kd, Kp, Ki], [1, 0])
        sys_closed = ct.feedback(controller * sys_open, 1)

        tau = 1
        #self.pitch_controller = Controller([Kp*tau + Kd,   Kp + Ki*tau,   Ki], [tau, 1, 0])

        # 4. Extract Poles and Zeros for Closed-Loop
        p_cl = ct.poles(sys_closed)
        z_cl = ct.zeros(sys_closed)

        print(f"Closed-Loop Poles: {p_cl}")
        print(f"Closed-Loop Zeros: {z_cl}")

        # 5. Plotting
        plt.figure(figsize=(10, 5))

        # Open Loop Map
        x_bounds = [-1.5, 1.5]
        y_bounds = [-1.5, 1.5]
        plt.subplot(1, 2, 1)
        ct.pzmap(sys_open, title="Open-Loop Pole/Zero")
        plt.xlim(x_bounds)
        plt.ylim(y_bounds)
        plt.grid(True)

        # Closed Loop Map
        plt.subplot(1, 2, 2)
        ct.pzmap(sys_closed, title=f"Closed-Loop Pole/Zero")
        plt.xlim(x_bounds)
        plt.ylim(y_bounds)
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
    def lateral_ROM(full_state: np.ndarray) -> np.ndarray:
        v = full_state[4]

        q0 = full_state[6]
        q1 = full_state[7]
        q2 = full_state[8]
        q3 = full_state[9]

        p = full_state[10]
        r = full_state[12]
        yaw = np.arctan2(2 * (q3 * q2 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2))
        roll = np.arctan2(2 * (q3 * q0 + q1 * q2), 1 - 2 * (q0 ** 2 + q1 ** 2))

        rom = np.array([v, p, r, roll, yaw])
        return rom

    def lateral_ROM_quaternion(self, full_state: np.ndarray) -> np.ndarray:
        """
        Returns [v, p, r, qe_x, qe_z]
        where qe represents the deviation from the trim orientation.
        """
        v = full_state[4]  # Lateral velocity
        p = full_state[10]  # Roll rate
        r = full_state[12]  # Yaw rate

        # Current orientation quaternion
        q_curr = full_state[6:10]

        # Orientation at trim (pre-calculated during perform_trim_optimization)
        q_star = self.x_star[6:10]

        # Calculate Error Quaternion: q_err = inv(q_star) * q_curr
        # This represents the "deviation" from trim
        q_star_inv = utils.quat_inverse(q_star)
        q_err = utils.quat_multiply(q_star_inv, q_curr)

        # We use the vector components (x, z) as proxies for Roll and Yaw error.
        # This is linear and has no wrapping issues at the trim point.
        qe_roll = q_err[1]
        qe_yaw = q_err[3]

        return np.array([v, p, r, qe_roll, qe_yaw])

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

    @staticmethod
    def find_ultimate_params(sys):
        # Use margin() to find where phase is -180 degrees
        # gm is the gain margin, which is the gain needed to reach instability
        gm, pm, wg, wp = ct.margin(sys)

        ku = gm  # Ultimate Gain
        wu = wg  # Ultimate Frequency (rad/s)
        tu = (2 * np.pi) / wu  # Ultimate Period
        return ku, tu

    @staticmethod
    def wrap_to_pi(angle):
        """
        Wraps an angle (radians) to the range [-pi, pi].
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    '''
    def yaw_control_analysis(self):
        C = np.array([[0, 0, 1]])  # Note: ensure C is 2D for ss
        D = np.array([[0]])
        sys_open = ct.ss(self.A_yaw, self.B_yaw, C, D)

        Ku, Tu = self.find_ultimate_params(sys_open)
        Ku = np.minimum(Ku, 10)

        Kp = 0.6 * Ku
        Ti = 0.5 * Tu
        Td = 0.125 * Tu

        # Convert to standard form gains
        Ki = Kp / Ti
        Kd = Kp * Td

        print(f"Ultimate Gain (Ku): {Ku:.2f}")
        print(f"Ultimate Period (Tu): {Tu:.2f}")
        print(f"Calculated PID: Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}")

        #Kp = 2
        #Kd = 0.5
        #Ki = 0.5

        tau = 0.01
        controller = ct.TransferFunction([Kp*tau + Kd,   Kp + Ki*tau,   Ki], [tau, 1, 0])
        sys_closed = ct.feedback(controller * sys_open, 1)

        p_cl = ct.poles(sys_closed)
        z_cl = ct.zeros(sys_closed)

        print(f"Closed-Loop Poles: {p_cl}")
        print(f"Closed-Loop Zeros: {z_cl}")

        # 5. Plotting
        plt.figure(figsize=(10, 5))

        # Open Loop Map
        x_bounds = [-1.5, 1.5]
        y_bounds = [-0.5, 0.5]
        plt.subplot(1, 2, 1)
        ct.pzmap(sys_open, title="Open-Loop Pole/Zero")
        plt.xlim(x_bounds)
        plt.ylim(y_bounds)
        plt.grid(True)

        # Closed Loop Map
        plt.subplot(1, 2, 2)
        ct.pzmap(sys_closed, title=f"Closed-Loop Pole/Zero")
        plt.xlim(x_bounds)
        plt.ylim(y_bounds)
        plt.grid(True)

        #plt.tight_layout()
        plt.show()
    '''

