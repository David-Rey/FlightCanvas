# aero_project/rocket_main.py

import aerosandbox as asb
import aerosandbox.numpy as np

from components.aero_fuselage import AeroFuselage
from components.aero_wing import create_axial_wing_pair, AeroWing
from aero_vehicle import AeroVehicle

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


aero_vehicle.set_control(["Fin"], [np.deg2rad(10)])
aero_vehicle.init_actors(color='lightblue', show_edges=False, opacity=0.8)
aero_vehicle.init_debug(label=True)
aero_vehicle.show()
