# aero_project/rocket.py

import aerosandbox as asb
import aerosandbox.numpy as np

from FlightCanvas.components.aero_wing import create_axial_wing_pair
from FlightCanvas.vehicle.aero_vehicle import AeroVehicle
from FlightCanvas.vehicle.vehicle_visualizer import VehicleVisualizer
from FlightCanvas import utils

if __name__ == '__main__':
    tail_airfoil = asb.Airfoil("naca0010")

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
