
import aerosandbox as asb
import aerosandbox.numpy as np
from scipy.interpolate import splprep, splev
import scipy.linalg

from FlightCanvas.vehicle.aero_vehicle import AeroVehicle
from FlightCanvas.components.aero_fuselage import AeroFuselage
from FlightCanvas.components.aero_wing import create_planar_wing_pair, AeroWing
from FlightCanvas.actuators.actuators import DirectDerivative
from FlightCanvas.control.optimal_controller import OptimalController
from FlightCanvas.vehicle.vehicle_visualizer import VehicleVisualizer

from typing import Dict, List

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
        Q_omega = 5e1
        Q_pos = 2e-1
        Q_quat = 1e2
        R_controls = 3e1
        Q_delta = 1e1

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


class Starship:
    """
    A class that defines and simulates a Starship-like aero vehicle.

    This class encapsulates the geometry, aerodynamics, mass properties,
    and control systems for the vehicle, and provides methods to run
    simulations and visualizations.
    """

    def __init__(self, cg_x=19.0, height=50.0, diameter=9.0):
        """
        Initializes and builds the Starship vehicle model
        :param cg_x: The initial cg in x direction in m
        :param height: The height in m
        :param diameter: The diameter in m
        """
        # Store geometric parameters
        self.cg_x = cg_x
        self.height = height
        self.diameter = diameter

        # Create all geometric components
        body = self._create_body()
        front_flaps = self._create_front_flaps()
        back_flaps = self._create_back_flaps()

        all_components = [body, *front_flaps, *back_flaps]

        # Define the control mapping
        control_mapping = self._get_control_mapping()

        # Assemble the AeroVehicle
        self.vehicle = AeroVehicle(
            name="Starship",
            xyz_ref=[self.cg_x, 0, 0],
            components=all_components,
        )
        self.vehicle.set_control_mapping(control_mapping)

        # Set mass and inertia properties
        self._set_mass_properties()

        # Load pre-computed aerodynamic data
        print("Loading aerodynamic buildup data...")
        try:
            self.vehicle.load_buildup()
        except FileNotFoundError:
            print("Build up data not found. Creating new aerodynamic buildup data...")
            self.vehicle.compute_buildup()
            self.vehicle.save_buildup()

        # Define vehicle dynamics
        self.vehicle.init_vehicle_dynamics()

        # Define OptimalController
        Nsim = 120
        N_horizon = 50
        tf = 8  # from horizon (dt = tf/N_horizon)
        # true tf = dt * Nsim
        self.vehicle.controller = StarshipController(self.vehicle, Nsim, N_horizon, tf)

        # Create Visualizer
        self.visualizer = VehicleVisualizer(self.vehicle)

    def save_buildup(self):
        """
        Saves the aerodynamic buildup data
        """
        self.vehicle.save_buildup()

    def _set_mass_properties(self):
        """
        Sets the mass and moment of inertia for the vehicle
        """
        self.vehicle.set_mass(95000)
        # MOI for a cylinder
        radius = self.diameter / 2
        I_s = (1 / 2) * radius ** 2  # Inertia about the spin axis (x)
        I_a = ((1 / 4) * radius ** 2) + ((1 / 12) * self.height ** 2)  # Inertia about transverse axes (y, z)
        self.vehicle.set_moi_diag([I_s, I_a, I_a])

    def _create_body(self) -> AeroFuselage:
        """
        Creates the body of starship
        :return: AeroFuselage object of starship body
        """
        n_points = 100
        nosecone_coords = self._get_nosecone_cords(self.diameter, n_points=n_points)
        end_cord = np.array([[self.height, nosecone_coords[-1, 1]], [self.height, 0]])
        nosecone_coords = np.vstack((nosecone_coords, end_cord))

        fuselage_xsecs = [
            asb.FuselageXSec(
                xyz_c=[x - self.cg_x, 0, 0],
                radius=z,
            ) for x, z in nosecone_coords
        ]

        return AeroFuselage(
            name="Fuselage",
            xsecs=fuselage_xsecs,
        ).translate([self.cg_x, 0, 0])

    def _create_front_flaps(self) -> List[AeroWing]:
        """
        Creates the pair of front flap components
        """
        flap_airfoil = asb.Airfoil(coordinates=self._flat_plate_airfoil(thickness=0.02))
        front_flap_xsecs = [
            asb.WingXSec(xyz_le=[0, 0, 0], chord=8, airfoil=flap_airfoil),
            asb.WingXSec(xyz_le=[6, 4.8, 0], chord=2.5, airfoil=flap_airfoil)
        ]
        return create_planar_wing_pair(
            name="Front Flap",
            xsecs=front_flap_xsecs,
            translation=[5, 2.9, 0],
            ref_direction=[1, 0.18, 0],
            control_pivot=[1, 0.18, 0],
            actuator_model=DirectDerivative()
        )

    def _create_back_flaps(self) -> List[AeroWing]:
        """
        Creates the pair of aft flap components
        """
        flap_airfoil = asb.Airfoil(coordinates=self._flat_plate_airfoil(thickness=0.02))
        back_flap_xsecs = [
            asb.WingXSec(xyz_le=[0, 0, 0], chord=15, airfoil=flap_airfoil),
            asb.WingXSec(xyz_le=[8, 5.8, 0], chord=6, airfoil=flap_airfoil)
        ]
        return create_planar_wing_pair(
            name="Aft Flap",
            xsecs=back_flap_xsecs,
            translation=[35, 4.5, 0],
            ref_direction=[1, 0, 0],
            control_pivot=[1, 0, 0],
            actuator_model=DirectDerivative()
        )

    @staticmethod
    def _get_control_mapping() -> Dict[str, Dict[str, float]]:
        """
        Define how abstract control commands map to individual flap deflections
        :return: Control mapping from abstract commands to flap deflections
        """

        return {
            "pitch control": {
                "Front Flap": 1.0,
                "Front Flap Star": 1.0,
                "Aft Flap": -1.0,
                "Aft Flap Star": -1.0
            },
            "roll control": {
                "Front Flap": 1.0,
                "Front Flap Star": -1.0,
                "Aft Flap": 1.0,
                "Aft Flap Star": -1.0
            },
            "yaw control": {
                "Front Flap": -1.0,
                "Front Flap Star": 1.0,
                "Aft Flap": 1.0,
                "Aft Flap Star": -1.0
            },
            "drag control": {
                "Front Flap": 1.0,
                "Front Flap Star": 1.0,
                "Aft Flap": 1.0,
                "Aft Flap Star": 1.0
            }
        }

    # --- Helper methods for geometry creation ---
    @staticmethod
    def _smooth_path(points: np.ndarray, smoothing_factor: float = 0.0, n_points: int = 500) -> np.ndarray:
        """
        Create smooth spline interpolation through given points
        """
        tck, u = splprep(points.T, s=smoothing_factor)
        u_fine = np.linspace(0, 1, n_points)
        return np.array(splev(u_fine, tck)).T

    def _get_nosecone_cords(self, diameter, smoothed=True, n_points=500) -> np.ndarray:
        """
        Generate nosecone profile coordinates based on Starship-like proportions
        :param diameter: The diameter of the starship
        :param smoothed: If true, smooth profile coordinates
        :param n_points: Number of points in final profile
        :return: The nosecone profile coordinates
        """
        points = np.array([
            [0.010000, 0.000000], [0.057585, 0.238814], [0.286398, 0.495763],
            [2.231314, 1.601695], [3.222839, 2.097458], [6.502500, 3.394068],
            [10.697415, 4.309322], [14.320297, 4.500000]
        ])
        scaled_points = np.copy(points)
        scaled_points[:, 1] *= diameter / 4.5 / 2
        return self._smooth_path(scaled_points, smoothing_factor=0.002, n_points=n_points) if smoothed else scaled_points

    @staticmethod
    def _flat_plate_airfoil(thickness=0.01, n_points=100) -> np.ndarray:
        """
        Generate flat plate airfoil coordinates for control surfaces
        """
        x = np.linspace(1, 0, n_points)
        y_upper = thickness / 2 * np.ones_like(x)
        y_lower = -thickness / 2 * np.ones_like(x)
        x_coords = np.concatenate([x, x[::-1]])
        y_coords = np.concatenate([y_upper, y_lower[::-1]])
        return np.vstack([x_coords, y_coords]).T

    def run_ocp(self):
        #t_arr, x_arr, u_arr = self.vehicle.run_ocp()
        pos_0 = np.array([1000, 0, 1000])  # Initial position
        vel_0 = np.array([0, 0, -30])  # Initial velocity
        quat_0 = utils.euler_to_quat((0, 0, 0))
        omega_0 = np.array([0, 0, 0])  # Initial angular velocity
        delta_0 = np.deg2rad(np.array([30, 30, 20, 20]))

        self.vehicle.run_mpc(pos_0, vel_0, quat_0, omega_0, delta_0)
        t_arr, x_arr, u_arr = self.vehicle.controller.get_control_history()
        self.vehicle.init_actors(color='lightblue', show_edges=False, opacity=1)
        self.vehicle.animate(t_arr, x_arr, u_arr, cam_distance=60, debug=False)

    def run_sim(self):
        pos_0 = np.array([0, 0, 1000])  # Initial position
        vel_0 = np.array([0, 0, -1])  # Initial velocity
        quat_0 = utils.euler_to_quat((0, 0, -30))
        omega_0 = np.array([0, 0, 0])  # Initial angular velocity
        delta_0 = np.deg2rad(np.array([20, 20, 20, 20]))
        tf = 30

        t_arr, x_arr, u_arr = self.vehicle.run_sim(pos_0, vel_0, quat_0, omega_0, delta_0, tf,
                            casadi=False, open_loop_control=None, gravity=True)
        self.vehicle.init_actors(color='lightblue', show_edges=False, opacity=1)
        self.vehicle.animate(t_arr, x_arr, u_arr, cam_distance=60, debug=False)

if __name__ == '__main__':
    # Create an instance of the entire Starship model
    starship = Starship()
    #starship.compute_buildup()
    starship.run_ocp()
    #starship.run_sim()




