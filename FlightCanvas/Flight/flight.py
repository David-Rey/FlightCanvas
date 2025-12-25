from FlightCanvas.analysis.log import Log
from FlightCanvas.vehicle.aero_vehicle import AeroVehicle
import numpy as np
import FlightCanvas.utils as utils


class Flight:
    def __init__(self, aero_vehicle: AeroVehicle, final_time, dt=0.01, gravity=True):
        self.aero_vehicle = aero_vehicle
        self.final_time = final_time
        self.dt = dt
        self.gravity = gravity
        self.steps = int(final_time / dt)

    def run_sim(self, init_state: np.array, trim, log: Log):
        """
        TODO
        """

        self.aero_vehicle.actuator_dynamics.c2d(self.dt)

        time = 0.0
        log.initialize_timestep(time)
        state = init_state
        control = trim.get_control(state)

        states_dot = self.aero_vehicle.dynamics(state, control)

        log.add(time, "states", state)
        log.add(time, "state_dots", states_dot)
        log.add(time, "control", control)
        log.add(time, "deflections", self.aero_vehicle.get_true_deflections())

        for i in range(1, self.steps):
            time = i * self.dt
            log.initialize_timestep(time)

            control = trim.get_control(state)
            aero_vehicle_dyn = lambda state: self.aero_vehicle.dynamics(state, control)

            state = utils.rk4(aero_vehicle_dyn, state, self.dt)
            states_dot = self.aero_vehicle.dynamics(state, control)

            log.add(time, "states", state)
            log.add(time, "state_dots", states_dot)
            log.add(time, "control", control)
            log.add(time, "deflections", self.aero_vehicle.get_true_deflections())

        log.trim()






