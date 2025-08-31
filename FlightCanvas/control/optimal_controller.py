from abc import abstractmethod

from FlightCanvas.control.controller import BaseController
from FlightCanvas.vehicle.aero_vehicle import AeroVehicle

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver
except ImportError:
    print("Warning: 'acados_template' not installed. Related functionality will be unavailable.")
    AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver = None, None, None, None, None

import numpy as np

class OptimalController(BaseController):
    def __init__(self, vehicle: AeroVehicle):
        super().__init__()

        self._vehicle = vehicle
        if self._vehicle.vehicle_dynamics.acados_model is None:
            self._vehicle.vehicle_dynamics.create_acados_model(True)

        self.nx = self._vehicle.vehicle_dynamics.acados_model.x.row()
        self.nu = self._vehicle.vehicle_dynamics.acados_model.u.row()

        self.ocp = AcadosOcp()

    def set_ocp_options(self):
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'IRK'
        self.ocp.solver_options.regularize_method = 'GERSHGORIN_LEVENBERG_MARQUARDT'
        self.ocp.solver_options.levenberg_marquardt = 5e-2
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'

    @abstractmethod
    def compute_control_input(self, t: float, state: np.ndarray) -> np.ndarray:
        pass