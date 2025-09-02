from abc import abstractmethod

from FlightCanvas.control.controller import BaseController
from FlightCanvas.vehicle.aero_vehicle import AeroVehicle
from typing import Tuple

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver
except ImportError:
    print("Warning: 'acados_template' not installed. Related functionality will be unavailable.")
    AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver = None, None, None, None, None

import numpy as np

class OptimalController(BaseController):
    def __init__(self, vehicle: AeroVehicle, Nsim: int, N_horizon: int, tf: float):
        super().__init__()

        self.Nsim = Nsim

        self.N_horizon = N_horizon

        self.tf = tf

        self.dt = self.tf / self.N_horizon

        self._vehicle = vehicle
        if self._vehicle.vehicle_dynamics.acados_model is None:
            self._vehicle.vehicle_dynamics.create_acados_model(True)

        self.acados_model = self._vehicle.vehicle_dynamics.acados_model

        self.nx = self.acados_model.x.rows()
        self.nu = self.acados_model.u.rows()
        self.ny = self.nx + self.nu

        self.ocp = AcadosOcp()
        self.ocp.model = self.acados_model

        # Set horizon and time
        self.ocp.solver_options.N_horizon = self.N_horizon
        self.ocp.solver_options.tf = self.tf

        # Initialize storage arrays
        self.simX = np.zeros((self.nx, self.Nsim + 1))
        self.simU = np.zeros((self.nu, self.Nsim))
        self.simT = np.zeros(self.Nsim)

    @abstractmethod
    def compute_control_input(self, k: int, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def init_first_step(self, x0: np.ndarray):
        pass