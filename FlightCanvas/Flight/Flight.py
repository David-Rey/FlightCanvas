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

    def run_sim(self, init_state: np.array, log: Log):
        """
        TODO
        """

        time = 0.0
        log.initialize_timestep(time)
        num_control_inputs = self.aero_vehicle.vehicle_dynamics.num_control_inputs
        control = np.zeros(num_control_inputs)
        state = init_state

        dynamics_6dof = lambda state, control: self.aero_vehicle.vehicle_dynamics.dynamics(state, control).full().flatten()

        states_dot = dynamics_6dof(state, control)

        log.add(time, "states", state)
        log.add(time, "state_dots", states_dot)

        for i in range(1, self.steps):
            time = i * self.dt
            log.initialize_timestep(time)

            # TODO: Replace with controller class
            control = np.zeros(num_control_inputs)

            aero_vehicle_dyn = lambda state: dynamics_6dof(state, control)

            state = utils.rk4(aero_vehicle_dyn, state, self.dt)
            states_dot = dynamics_6dof(state, control)

            log.add(time, "states", state)
            log.add(time, "state_dots", states_dot)

        log.trim()






