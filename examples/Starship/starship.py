import aerosandbox as asb
import aerosandbox.numpy as np
from scipy.interpolate import splprep, splev

from FlightCanvas.vehicle.aero_vehicle import AeroVehicle
from FlightCanvas.components.aero_fuselage import AeroFuselage
from FlightCanvas.components.aero_wing import create_planar_wing_pair, AeroWing
from FlightCanvas.OLD.actuators.actuators import DirectDerivative
from FlightCanvas.Flight.flight import Flight
from FlightCanvas.analysis.log import Log
from FlightCanvas.analysis.vehicle_visualizer import VehicleVisualizer
from FlightCanvas.analysis.anlaysis import Analysis
from examples.Starship.starship_trimpoint import Trimpoint
from FlightCanvas.vehicle.actuator_dynamics import Actuator

from typing import Dict, List


class Starship:
    """
    A class that defines and simulates a Starship-like aero vehicle.

    This class encapsulates the geometry, aerodynamics, mass properties,
    and starship_control systems for the vehicle, and provides methods to run
    simulations and visualizations.
    """

    def __init__(self, cg_x=25, height=50.0, diameter=9.0):
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

        # Define the starship_control mapping
        control_mapping = self._get_control_mapping()

        # Assemble the AeroVehicle
        self.vehicle = AeroVehicle(
            name="Starship",
            xyz_ref=[self.cg_x, 0, 0],
            components=all_components,
        )

        # Set mass and inertia properties
        self._set_mass_properties()

        # Load pre-computed aerodynamic data
        print("Loading aerodynamic buildup data...")
        try:
            self.vehicle.load_buildup()
        except:
            print("Build up data not found. Creating new aerodynamic buildup data...")
            self.vehicle.compute_buildup()
            self.vehicle.save_buildup()

        self.update_moment()

        # Init vehicle dynamics
        self.vehicle.init_vehicle_dynamics(control_mapping)

    def save_buildup(self):
        """
        Saves the aerodynamic buildup data
        """
        self.vehicle.save_buildup()

    def save_buildup_figs(self):
        """
        Saves the aerodynamic buildup figures
        """
        self.vehicle.save_buildup_fig()

    def _set_mass_properties(self):
        """
        Sets the mass and moment of inertia for the vehicle
        """
        self.vehicle.set_mass(135000)
        # MOI for a cylinder
        radius = self.diameter / 2
        I_s = (1 / 2) * radius ** 2  # Inertia about the spin axis (x)
        I_a = ((1 / 4) * radius ** 2) + ((1 / 12) * self.height ** 2)  # Inertia about transverse axes (y, z)
        self.vehicle.set_moi_diag([I_s, I_a, I_a])

    def update_moment(self):
        """
        Updates the body moments to get the cg to be reasonable
        """
        # Retrieve the tuple from the dictionary
        F_b_tuple = self.vehicle.components[0].buildup_manager.asb_data_static["M_b"]

        # Convert the tuple to a list to make it mutable
        F_b_list = list(F_b_tuple)

        # Access the moment value and update it within the new list
        F_b_list[1] = F_b_list[1] * 0.50
        #F_b_list[0] = F_b_list[0] * 0
        #F_b_list[2] = F_b_list[2] * 0


        for i in [1, 3]:
        #for i in []:
            F_b_temp = list(self.vehicle.components[i].buildup_manager.asb_data_static["F_b"])
            M_b_temp = list(self.vehicle.components[i].buildup_manager.asb_data_static["M_b"])
            F_b_temp[0] = F_b_temp[0] * 0
            F_b_temp[1] = F_b_temp[1] * 0
            M_b_temp[0] = M_b_temp[0] * 0
            M_b_temp[2] = M_b_temp[2] * 0
            self.vehicle.components[i].buildup_manager.asb_data_static["F_b"] = F_b_temp
            self.vehicle.components[i].buildup_manager.asb_data_static["M_b"] = M_b_temp



        #F_b_temp = list(self.vehicle.components[2].buildup_manager.asb_data_static["F_b"])
        #F_b_temp[0] = F_b_temp[0] * 0
        #F_b_temp[1] = F_b_temp[1] * 0
        #self.vehicle.components[1].buildup_manager.asb_data_static["F_b"] = F_b_temp

        # Reassign the updated list back to the dictionary
        self.vehicle.components[0].buildup_manager.asb_data_static["M_b"] = F_b_list

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
            actuator_model=Actuator([1], [0.1, 1])
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
            actuator_model=Actuator([1], [0.1, 1])
        )

    @staticmethod
    def _get_control_mapping() -> Dict[str, Dict[str, float]]:
        """
        Define how abstract starship_control commands map to individual flap deflections
        :return: Control mapping from abstract commands to flap deflections
        """
        return {
            "pitch starship_control": {
                "Front Flap": 1.0,
                "Front Flap Star": 1.0,
                "Aft Flap": -1.0,
                "Aft Flap Star": -1.0
            },
            "roll starship_control": {
                "Front Flap": 1.0,
                "Front Flap Star": -1.0,
                "Aft Flap": 1.0,
                "Aft Flap Star": -1.0
            },
            "yaw starship_control": {
                "Front Flap": -1.0,
                "Front Flap Star": 1.0,
                "Aft Flap": 1.0,
                "Aft Flap Star": -1.0
            },
            "drag starship_control": {
                "Front Flap": 1.0,
                "Front Flap Star": 1.0,
                "Aft Flap": 1.0,
                "Aft Flap Star": 1.0
            }
        }

    # Helper methods for geometry creation
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
        return self._smooth_path(scaled_points, smoothing_factor=0.002,
                                 n_points=n_points) if smoothed else scaled_points

    @staticmethod
    def _flat_plate_airfoil(thickness=0.01, n_points=100) -> np.ndarray:
        """
        Generate flat plate airfoil coordinates for starship_control surfaces
        """
        x = np.linspace(1, 0, n_points)
        y_upper = thickness / 2 * np.ones_like(x)
        y_lower = -thickness / 2 * np.ones_like(x)
        x_coords = np.concatenate([x, x[::-1]])
        y_coords = np.concatenate([y_upper, y_lower[::-1]])
        return np.vstack([x_coords, y_coords]).T

    def run_sim(self):
        """
        Runs static simulation
        """
        state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'q0', 'q1', 'q2', 'q3', 'wx', 'wy', 'wz']
        control_names = ['pitch', 'yaw', 'roll', 'drag']
        deflection_names = ['b', 'f_left', 'f_right', 'b_left', 'b_right']
        maxSteps = 2000
        log = Log(state_names, control_names, deflection_names, maxSteps)

        dt = 0.01
        tf = 10
        flight = Flight(self.vehicle, tf, dt=dt)

        trim = Trimpoint(self.vehicle.vehicle_dynamics)
        z_guess = np.array([-60, 0, np.deg2rad(12)])
        trim.get_trimpoint(z_guess)
        trim.init_LQR()
        #trim.yaw_control_analysis()

        #trim.plot_pz_with_feedback()

        inital_state = trim.x_star
        #inital_state = trim.u_star
        #starship_control = StarshipController(sys)
        #starship_control.compute_lqr()
        #inital_state, inital_control, _ = trim.get_trimpoint()
        #trim.get_LQR_control()

        inital_state[2] = 800
        inital_state[11] = 0.001

        #trim.draw_wrench_space()
        flight.run_sim(inital_state, trim, log)

        analysis = Analysis(log)
        analysis.generate_control_plot()
        analysis.generate_velocity_plot(include_vz=False)
        #analysis.generate_position_plot()
        analysis.generate_euler_angle_plot()
        analysis.generate_angular_velocity_plot()
        #analysis.generate_quat_norm_plot()
        #analysis.generate_angle_of_attack_plot()
        #analysis.generate_true_deflections_plot()

        vv = VehicleVisualizer(self.vehicle, log)
        vv.init_actors()
        vv.add_grid()
        #vv.generate_square_traj()
        vv.animate(cam_distance=80, zoom=1.5)



if __name__ == '__main__':
    # Create an instance of the entire Starship model
    starship = Starship()
    #starship.save_buildup_figs()

    starship.run_sim()
