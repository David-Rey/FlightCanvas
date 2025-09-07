# aero_project/fin_tabs.py

import aerosandbox as asb
import numpy as np

from typing import List

from FlightCanvas.vehicle.aero_vehicle import AeroVehicle
from FlightCanvas.components.aero_fuselage import AeroFuselage
from FlightCanvas.components.aero_wing import create_axial_wing_pair
from FlightCanvas.components.aero_wing import AeroWing

from FlightCanvas.vehicle.vehicle_visualizer import VehicleVisualizer


#from FlightCanvas import utils


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
        self.cg_x = 1.84  # [m]

        # Fins
        self.num_fins = 4  # []
        self.root_chord = 0.203  # [m]
        self.tip_chord = 0.17  # [m]
        self.fin_height = 0.144  # [m]
        self.sweep_length = 0.033  # [m]

        body = self._create_body()

        all_components = [body]

        # Assemble the AeroVehicle
        self.vehicle = AeroVehicle(
            name="Starship",
            xyz_ref=[self.cg_x, 0, 0],
            components=all_components,
        )

    def _create_body(self) -> AeroFuselage:
        nose_x = np.linspace(0, self.nose_cone_length)
        rho = self.nose_cone_rho
        L = self.nose_cone_length
        C = self.nose_cone_rho / self.rocket_diameter
        R = self.rocket_diameter * (C ** 2 + 0.25)
        #R = self.rocket_diameter / 2
        nose_y = np.sqrt(rho**2 - (L - nose_x) ** 2) + R - rho
        nose_corrds = np.vstack([nose_x, nose_y]).T

        # Add body
        end_cord = np.array([[self.total_length, self.total_length], [R, 0]])
        body_corrds = np.vstack((nose_corrds, end_cord))

        body_xsecs = [
            asb.FuselageXSec(
                xyz_c=[x - self.cg_x, 0, 0],
                radius=z,
            ) for x, z in body_corrds
        ]

        return AeroFuselage(
            name="Body",
            xsecs=body_xsecs,
        ).translate([self.cg_x, 0, 0])

    def _create_fins(self) -> List[AeroWing]:
        pass

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


if __name__ == '__main__':
    rocket = FinTabs()
    vis = VehicleVisualizer(rocket.vehicle)

    vis.init_actors(color='lightblue', show_edges=False, opacity=0.8)
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
