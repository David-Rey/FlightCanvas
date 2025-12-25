from FlightCanvas.vehicle.vehicle_dynamics import VehicleDynamics
import numpy as np


class StarshipROM:

    @staticmethod
    def pitch_ROM(full_state: np.ndarray) -> np.ndarray:
        u = full_state[3]
        w = full_state[5]

        q0 = full_state[6]
        q1 = full_state[7]
        q2 = full_state[8]
        q3 = full_state[9]

        pitch = np.arcsin(2 * (q3 * q1 - q2 * q0))
        alpha = np.arctan2(w, u)
        q = full_state[11]

        x3 = np.array([alpha, q, pitch])
        return x3

    @staticmethod
    def pitch_ROM_old(full_state: np.ndarray) -> np.ndarray:
        """
        Calculates reduced order model
        :param full_state: The current state of the vehicle (position, velocity, quaternion, angular_velocity)
        :return: The reduced order model (u, w, q, theta)
        """
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
    def lateral_ROM(full_state: np.ndarray) -> np.ndarray:
        v = full_state[4]
        p = full_state[10]
        r = full_state[12]

        q0 = full_state[6]
        q1 = full_state[7]
        q2 = full_state[8]
        q3 = full_state[9]

        yaw = np.arctan2(2 * (q3 * q2 + q0 * q1), 1 - 2 * (q1**2 + q2**2))
        roll = np.arctan2(2 * (q3 * q0 + q1 * q2), 1 - 2 * (q0 ** 2 + q1 ** 2))

        rom = np.array([v, p, r, roll, yaw])
        return rom






