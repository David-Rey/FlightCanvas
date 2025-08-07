# aero_project/rocket_main.py

import aerosandbox as asb
import aerosandbox.numpy as np

from components.aero_wing import create_axial_wing_pair
from components.aero_vehicle import AeroVehicle
from components import utils

tail_airfoil = asb.Airfoil("naca0010")

# Create the horizontal tail using the factory function
h_tail_xsecs = [
    asb.WingXSec(xyz_le=[-0.05, 0, 0], chord=0.1, twist=0, airfoil=tail_airfoil),
    asb.WingXSec(xyz_le=[-0.03, 0.17, 0], chord=0.08, twist=0, airfoil=tail_airfoil)
]

h_tail_wings = create_axial_wing_pair(
    name="Fin",
    xsecs=h_tail_xsecs,
    translation=[0.3, 0.1, 0],  # Apply translation to the whole pair
    ref_direction=[1, 0, 0],
    control_pivot=[0, 1, 0],
    num_wings=4
)

all_components = [
    *h_tail_wings,
]

aero_vehicle = AeroVehicle(
    name="David's Rocket",
    xyz_ref=[0, 0, 0],  # Vehicle's Center of Gravity
    components=all_components
)

# DEBUG
animate = 0

aero_vehicle.compute_buildup()
aero_vehicle.save_buildup()
#aero_vehicle.save_buildup_fig()
aero_vehicle.load_buildup()

if animate:
    pos_0 = np.array([0, 0, 950])  # Initial position
    vel_0 = np.array([100, 0, 0])  # Initial velocity
    quat_0 = utils.euler_to_quat((0, 0, 0))
    omega_0 = np.array([0, 0, 2])  # Initial angular velocity
    tf = 10

    t_arr, x_arr = aero_vehicle.run_sim(pos_0, vel_0, quat_0, omega_0, tf)
    aero_vehicle.init_actors(color='lightblue', show_edges=False, opacity=1)
    aero_vehicle.animate(t_arr, x_arr, debug=False)
else:
    aero_vehicle.init_actors(color='lightblue', show_edges=False, opacity=0.8)
    aero_vehicle.init_debug()
    aero_vehicle.show()

#aero_vehicle.set_control(["Fin"], [np.deg2rad(30)])
#aero_vehicle.init_actors(color='lightblue', show_edges=False, opacity=0.8)
#aero_vehicle.init_debug(label=True)
#aero_vehicle.show()
