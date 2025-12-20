# aero_project/fin_tabs.py

import aerosandbox as asb
import numpy as np

from typing import List

from FlightCanvas.vehicle.aero_vehicle import AeroVehicle
from FlightCanvas.components.aero_fuselage import AeroFuselage
from FlightCanvas.components.aero_wing import AeroWing, create_axial_wing_pair

from FlightCanvas.analysis.vehicle_visualizer import VehicleVisualizer

from FlightCanvas import utils


class FinTabs:
    def __init__(self):
        # Nose cone parameters
        self.nose_cone_length = 0.61  # [m]
        self.nose_cone_rho = 1  # []

        # Main body parameters
        self.total_length = 2.64  # This includes nose code [m]
        self.rocket_diameter = 0.157  # [m]

        # MOI and CG
        self.longitudinal_moi = 9.15  # [kg * m^2]
        self.rotational_moi = 0.112  # [kg * m^2]
        #self.cg_x = 0  # [m]
        self.cg_x = 1.84  # [m]

        # Fins
        self.num_fins = 4  # []
        self.root_chord = 0.203  # [m]
        self.tip_chord = 0.17  # [m]
        self.fin_span = 0.144  # [m]
        self.sweep_angle = 15  # [deg]

        #body = self._create_body()
        fins = self._create_fins()

        #all_components = [body, *fins]
        all_components = [*fins]

        # Assemble the AeroVehicle
        self.vehicle = AeroVehicle(
            name="FinTabs",
            xyz_ref=[0, 0, 0],
            components=all_components,
        )

    def _create_body(self) -> AeroFuselage:
        """Creates the fuselage geometry for a rocket with a tangent ogive nose cone."""

        # Define key geometric parameters for clarity
        L = self.nose_cone_length  # Length of the nose cone
        R = self.rocket_diameter / 2  # Radius of the rocket body

        # Calculate the ogive radius (rho) based on L and R
        rho = (R ** 2 + L ** 2) / (2 * R)

        # Generate coordinates for the nose cone curve
        nose_x = np.linspace(0, L, 100)  # Use 100 points for a smooth curve

        # This is the standard equation for a tangent ogive's profile
        nose_y = np.sqrt(rho ** 2 - (L - nose_x) ** 2) + R - rho

        nose_coords = np.vstack([nose_x, nose_y]).T

        # Define the cylindrical body section
        body_coords = np.array([
            [self.total_length, R],
            [self.total_length, 0]
        ])

        # Combine all coordinates into a single profile
        full_body_coords = np.vstack((nose_coords, body_coords))

        # Create AeroSandbox FuselageXSec objects
        body_xsecs = [
            asb.FuselageXSec(
                xyz_c=[x - self.cg_x, 0, 0],
                radius=y,  # The y-coordinate from our profile is the radius
            ) for x, y in full_body_coords
        ]

        return AeroFuselage(
            name="RocketBody",
            xsecs=body_xsecs
        )

    def _create_fins(self) -> List[AeroWing]:
        """
        Creates the pair of front flap components
        """

        ref_delta = 0
        delta = self.total_length - self.cg_x - self.root_chord
        sweep_length = self.fin_span * np.sin(np.deg2rad(self.sweep_angle))

        flap_airfoil = asb.Airfoil(coordinates=self._flat_plate_airfoil(thickness=0.01))
        front_flap_xsecs = [
            asb.WingXSec(xyz_le=[-ref_delta, 0, 0], chord=self.root_chord, airfoil=flap_airfoil),
            asb.WingXSec(xyz_le=[-ref_delta + sweep_length, self.fin_span, 0], chord=self.tip_chord,
                         airfoil=flap_airfoil)
        ]
        return create_axial_wing_pair(
            name="Fins",
            xsecs=front_flap_xsecs,
            translation=[delta + ref_delta, self.rocket_diameter / 2, 0],
            ref_direction=[1, 0, 0],
            num_wings=4
        )

    @staticmethod
    def _flat_plate_airfoil(thickness=0.01, n_points=100) -> np.ndarray:
        """
        Generate flat plate airfoil coordinates
        :param thickness: Percent thickness of flat plate airfoil
        :param n_points: Number of flat plate airfoil points
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
        pos_0 = np.array([0, 0, 1000])  # Initial position
        vel_0 = np.array([50, 0, 0])  # Initial velocity
        quat_0 = utils.euler_to_quat((0, 0.1, 0))
        omega_0 = np.array([0, 0, 0])  # Initial angular velocity
        delta_0 = np.array([])
        tf = 20

        self.vehicle.run_sim(pos_0, vel_0, quat_0, omega_0, delta_0, tf,
                             casadi=False, open_loop_control=None, gravity=False)

    def test_dynamics(self):
        pos_0 = np.array([0, 0, 1000])  # Initial position
        vel_0 = np.array([50, 0, 0])  # Initial velocity
        quat_0 = utils.euler_to_quat((0, 0, 0))
        omega_0 = np.array([0, 0, 0])  # Initial angular velocity
        delta_0 = np.array([])
        state_0 = np.concatenate((pos_0, vel_0, quat_0, omega_0, delta_0))
        F_b, M_b = self.vehicle.vehicle_dynamics.compute_forces_and_moments(state_0, true_deflections=np.array([0, 0, 0, 0]))

        print(F_b)
        print(M_b)



if __name__ == '__main__':
    rocket = FinTabs()
    #rocket.vehicle.load_buildup()
    rocket.vehicle.init_vehicle_dynamics()
    rocket.vehicle.compute_buildup()
    #rocket.vehicle.save_buildup()
    #rocket.vehicle.save_buildup_fig()
    rocket.test_dynamics()

    vis = VehicleVisualizer(rocket.vehicle)
    vis.init_actors(color='lightblue', show_edges=False, opacity=1)
    #vis.add_grid()
    #vis.animate()
    vis.init_debug(size=1)
    vis.show()

    #tail_airfoil = asb.Airfoil("naca0010")
    '''
    # Create the horizontal tail using the factory function
    h_tail_xsecs = [
        asb.WingXSec(xyz_le=[-0.05, 0, 0], chord=0.1, twist=0, airfoil=tail_airfoil),
        asb.WingXSec(xyz_le=[-0.03, 0.17, 0], chord=0.08, twist=0, airfoil=tail_airfoil)
    ]

    fins = create_axial_wing_pair(
        name="Fin",
        xsecs=h_tail_xsecs,
        translation=[0.3, 0.1, 0],  # Apply translation to the whole pair
        ref_direction=[1, 0, 0],
        control_pivot=[0, 1, 0],
        num_wings=4
    )

    all_components = [
        *fins,
    ]

    aero_vehicle = AeroVehicle(
        name="David's Rocket",
        xyz_ref=[0, 0, 0],  # Vehicle's Center of Gravity
        components=all_components
    )

    aero_vehicle.compute_buildup()
    #aero_vehicle.save_buildup()
    #aero_vehicle.save_buildup_fig()
    #aero_vehicle.load_buildup()
    aero_vehicle.init_vehicle_dynamics()

    pos_0 = np.array([0, 0, 950])  # Initial position
    vel_0 = np.array([100, 0, 0])  # Initial velocity
    quat_0 = utils.euler_to_quat((0, 0, 0))
    omega_0 = np.array([0, 0, 2])  # Initial angular velocity
    delta_0 = []
    tf = 10
    dt = 0.02

    aero_vehicle.run_sim(pos_0, vel_0, quat_0, omega_0, delta_0, tf, dt, gravity=False, casadi=False)

    vis = VehicleVisualizer(aero_vehicle)
    vis.init_actors(color='lightblue', show_edges=False, opacity=1)
    vis.add_grid()
    vis.animate(show_text=False, cam_distance=5, fps=30)
    '''
