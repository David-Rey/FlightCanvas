
from FlightCanvas.control.optimal_controller import OptimalController
import scipy.linalg
from FlightCanvas.vehicle.aero_vehicle import AeroVehicle

import numpy as np
from FlightCanvas import utils


try:
    from acados_template import AcadosModel, AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver
except ImportError:
    print("Warning: 'acados_template' not installed. Related functionality will be unavailable.")
    AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver = None, None, None, None, None



class StarshipController(OptimalController):
    """
    Model Predictive Controller (MPC) for the Starship vehicle using Acados
    """
    def __init__(self, vehicle: AeroVehicle, Nsim: int, N_horizon: int, tf: float):
        """
        Initialize the Starship MPC controller
        """
        super().__init__(vehicle, Nsim, N_horizon, tf)

        self.vehicle = vehicle

        # Set problem dimensions
        self.ocp.dims.nx = self.nx
        self.ocp.dims.nu = self.nu

        # Configure the optimal control problem
        self.set_ocp_options()
        self.set_cost_function()
        self.set_constraints()

        # Create solver instances
        self.ocp_solver = AcadosOcpSolver(self.ocp, verbose=True)
        self.integrator = AcadosSimSolver(self.ocp)

        # Initialize storage arrays
        self.simX = np.zeros((self.Nsim + 1, self.nx))
        self.simU = np.zeros((self.Nsim, self.nu))
        self.simT = np.zeros(self.Nsim)

        # Performance tracking
        self.t_preparation = np.zeros(self.Nsim)
        self.t_feedback = np.zeros(self.Nsim)


    def set_ocp_options(self):
        """
        Configure solver options for the optimal control problem
        """
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'IRK'
        self.ocp.solver_options.regularize_method = 'GERSHGORIN_LEVENBERG_MARQUARDT'
        self.ocp.solver_options.levenberg_marquardt = 5e-2
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.tf = self.tf

    def set_cost_function(self):
        """
        Define the cost function for the optimal control problem
        """
        Q_omega = 2e1
        Q_pos = 1e-1
        Q_quat = 1e2
        R_controls = 2e2
        Q_delta = 1e0

        Q_diag = np.zeros(self.nx)
        Q_diag[0:2] = Q_pos
        Q_diag[7:10] = Q_quat
        Q_diag[10:13] = Q_omega  # Weight on angular velocity
        Q_diag[13:] = Q_delta
        Q_mat = np.diag(Q_diag)

        q_ref = utils.euler_to_quat((0, 0, 25))

        # Create diagonal R matrix for control costs
        R_mat = np.diag(np.full(self.nu, R_controls))

        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'

        self.ocp.cost.yref = np.zeros(self.ny)
        self.ocp.cost.yref_e = np.zeros(self.nx)

        self.ocp.cost.yref[7:10] = q_ref[1:]
        self.ocp.cost.yref_e[7:10] = q_ref[1:]
        self.ocp.cost.yref[13:17] = np.deg2rad(np.array([30, 30, 20, 20]))
        self.ocp.cost.yref_e[13:17] = np.deg2rad(np.array([30, 30, 20, 20]))

        self.ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        self.ocp.cost.W_e = Q_mat
        self.ocp.cost.Vx = np.zeros((self.ny, self.nx))
        self.ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)
        self.ocp.cost.Vu = np.zeros((self.ny, self.nu))
        self.ocp.cost.Vu[self.nx:, :self.nu] = np.eye(self.nu)
        self.ocp.cost.Vx_e = np.eye(self.nx)

    def set_constraints(self):
        """
        Define constraints for the optimal control problem.

        Sets up three types of constraints:
        1. Control input constraints (actuator rate limits)
        2. State constraints (flap angle limits)
        3. Initial state constraints (for MPC feedback)
        """
        max_rate_rad_s = np.deg2rad(15.0)
        min_rate_rad_s = -max_rate_rad_s

        self.ocp.constraints.idxbu = np.arange(self.nu)  # Constrain all control inputs
        self.ocp.constraints.lbu = np.full(self.nu, min_rate_rad_s)
        self.ocp.constraints.ubu = np.full(self.nu, max_rate_rad_s)

        self.ocp.constraints.idxbx = np.array([13, 14, 15, 16])
        self.ocp.constraints.lbx = np.deg2rad(np.array([5, 5, 5, 5]))
        self.ocp.constraints.ubx = np.deg2rad(np.array([80, 80, 80, 80]))

        # Initial state constraints
        self.ocp.constraints.idxbx_0 = np.arange(self.nx)  # Constrain all states at initial time
        self.ocp.constraints.lbx_0 = np.zeros(self.nx)  # Will be updated with current state
        self.ocp.constraints.ubx_0 = np.zeros(self.nx)  # Will be updated with current state

    def init_first_step(self, x0: np.ndarray):
        """
        Initialize the MPC solver with a warm-start trajectory
        """
        pos_0 = x0[0:3]
        vel_0 = x0[3:6]
        quat_0 = x0[6:10]
        omega_0 = x0[10:13]
        delta_0 = x0[13:17]
        gravity = True
        print_debug = False
        open_loop_control = None
        _, init_x, init_u = self.vehicle.vehicle_dynamics.run_sim_scipy(pos_0, vel_0, quat_0, omega_0, delta_0, self.tf,
                                                                        self.dt, gravity,
                                                                        print_debug, open_loop_control)
        for i in range(self.N_horizon):
            self.ocp_solver.set(i, "x", init_x[:, i])
            self.ocp_solver.set(i, "u", init_u[:, i])
        self.ocp_solver.set(self.N_horizon, "x", init_x[:, self.N_horizon])
        self.simX[0, :] = x0  # Use the actual initial state passed in


    def compute_control_input(self, k: int, state: np.ndarray) -> np.ndarray:
        """
        Compute optimal control input using MPC at time step k
        """
        # set initial state
        if k == 0:
            self.init_first_step(state)

        self.ocp_solver.set(0, "lbx", state)
        self.ocp_solver.set(0, "ubx", state)

        self.ocp_solver.options_set('rti_phase', 1)
        status = self.ocp_solver.solve()
        self.t_preparation[k] = self.ocp_solver.get_stats('time_tot')

        if status not in [0, 2, 5]:
            raise Exception(f'acados returned status {status}. Exiting.')

        # feedback phase
        self.ocp_solver.options_set('rti_phase', 2)
        self.ocp_solver.solve()
        self.t_feedback[k] = self.ocp_solver.get_stats('time_tot')

        self.simU[k, :] = self.ocp_solver.get(0, "u")
        self.simX[k + 1, :] = self.integrator.simulate(x=self.simX[k, :], u=self.simU[k, :])
        self.simT[k] = k * self.dt

        return self.simX[k + 1, :]

    def get_control_history(self):
        """
        Return the complete control history for post-processing
        """
        return self.simT, self.simX.T, self.simU.T