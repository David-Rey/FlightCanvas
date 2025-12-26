from FlightCanvas.vehicle.vehicle_dynamics import VehicleDynamics
import numpy as np
import numdifftools as nd
from FlightCanvas import utils
from scipy.optimize import minimize
import control as ct


class Trimpoints:
    def __init__(self, vehicle_dynamics: VehicleDynamics):
        self.vehicle_dynamics = vehicle_dynamics
        self.nominal_pos = np.array([0, 0, 0])
        self.nominal_vel = np.array([0, 0, 0])
        self.nominal_quat = utils.euler_to_quat((0, 0, 0))
        self.nominal_omega = np.array([0, 0, 0])

        self.K = None
        self.x_star = None
        self.x_rom_star = None
        self.u_star = None
        self.z_star = None
        self.drag_control = 41

    def x_bar(self, z):
        state = np.concatenate((self.nominal_pos, self.nominal_vel, self.nominal_quat, self.nominal_omega))
        state[5] = z[0]
        state[3] = z[2]
        #pitch = z[1]
        pitch = 0
        quat_pitch = utils.euler_to_quat((0, pitch, 0))
        state[6:10] = quat_pitch
        return state

    def u_bar(self, z):
        #component_deflections = np.array([0, z[1], z[1], z[2], z[2]])
        #return component_deflections
        control_inputs = np.deg2rad(np.array([z[1], 0, 0, self.drag_control]))
        if self.vehicle_dynamics.allocation_matrix is not None:
            component_deflections = self.vehicle_dynamics.allocation_matrix @ control_inputs
            return component_deflections
        return np.zeros(self.vehicle_dynamics.num_actuator_inputs_comp)

    def trim_objective(self, z: np.ndarray) -> float:
        x = self.x_bar(z)
        u = self.u_bar(z)

        jac_fun = nd.Jacobian(self.pitch_ROM)
        Jg = jac_fun(x)
        x_dot = self.vehicle_dynamics.dynamics(x, u)

        x_rom_dot = Jg @ x_dot
        cost = np.linalg.norm(x_rom_dot)
        return cost

    def find_trimpoint(self):
        z_0 = np.array([-57, 20, 0])

        res = minimize(self.trim_objective, z_0, method='SLSQP')
        self.z_star = res.x

        print(res.x)
        print(res.fun)

        self.x_star = self.x_bar(self.z_star)
        self.x_rom_star = self.pitch_ROM(self.x_star)
        self.u_star = self.u_bar(self.z_star)

        return self.x_star, self.u_star, self.z_star

    def calculate_ROM(self):
        rom_jac_fun = nd.Jacobian(self.pitch_ROM)

        def control(pitch):
            control_inputs = np.array([pitch[0], 0, 0, np.deg2rad(self.drag_control)])
            deflections = self.vehicle_dynamics.allocation_matrix @ control_inputs
            return deflections

        fx = lambda x: self.vehicle_dynamics.dynamics(x, self.u_star)
        fu = lambda pitch: self.vehicle_dynamics.dynamics(self.x_star, control(pitch))

        A_jac_fun = nd.Jacobian(fx)
        B_jac_fun = nd.Jacobian(fu)
        Jg_jac_fun = nd.Jacobian(self.pitch_ROM)

        A = A_jac_fun(self.x_star).squeeze()
        B = B_jac_fun(np.deg2rad(self.z_star[1])).reshape(-1, 1)
        Jg = rom_jac_fun(self.x_star)

        T = np.linalg.pinv(Jg)

        A_rom = Jg @ A @ T
        B_rom = Jg @ B

        return A_rom, B_rom

    def get_LQR_control(self, q_weights=None, r_weight=1.0):
        A, B = self.calculate_ROM()

        if q_weights is None:
            # [u, w, q, theta]
            q_weights = [0.001, 0.001, 1.0, 1.0]

        Q = np.diag(q_weights)
        R = np.array([[r_weight]])

        # Compute the LQR gain matrix K
        # K is the matrix such that u = -Kx stabilizes (A - BK)
        self.K, S, E = ct.lqr(A, B, Q, R)

        #C = np.array([[1, 0, 0, 0]])
        #D = 0

        #sys_ss = ct.ss(A, B, C, D)
        #ss_tf = ct.ss2tf(sys_ss)
        #ss_tf_feedback = ct.feedback(ss_tf, 1)

        #from matplotlib import pyplot as plt

        #ct.root_locus_map(ss_tf_feedback).plot()

        #plt.show()


        # Create the closed-loop state-space system
        #sys_cl = ct.ss(A - B @ self.K, B, C, D)

    def get_control(self, state):
        rom = self.pitch_ROM(state)

        ref = np.array([-10, 0, 0, 0])

        state_error = ref - rom + self.x_rom_star
        rel_pitch_control = self.K @ state_error
        nominal_control = np.deg2rad(np.array([self.z_star[1], 0, 0, self.drag_control]))
        pitch_control = nominal_control + np.array([rel_pitch_control[0], 0, 0, 0])

        return pitch_control


    @staticmethod
    def pitch_ROM(full_state: np.ndarray) -> np.ndarray:
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
