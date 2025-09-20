import aerosandbox as asb
import aerosandbox.numpy as np

from FlightCanvas.components.aero_fuselage import AeroFuselage
from FlightCanvas.components.aero_wing import create_planar_wing_pair, AeroWing
from FlightCanvas.vehicle.aero_vehicle import AeroVehicle

wing_airfoil = asb.Airfoil("sd7037")
tail_airfoil = asb.Airfoil("naca0010")

# Create aerosandbox airplane
airplane = asb.Airplane(
    name="Peter's Glider",
    xyz_ref=[0, 0, 0],  # CG location
    wings=[
        asb.Wing(
            name="Main Wing",
            symmetric=True,  # Should this wing be mirrored across the XZ plane?
            xsecs=[
                asb.WingXSec(xyz_le=[0, 0, 0], chord=0.18, twist=2, airfoil=wing_airfoil),
                asb.WingXSec(xyz_le=[0.01, 0.5, 0], chord=0.16, twist=0, airfoil=wing_airfoil),
                asb.WingXSec(xyz_le=[0.08, 1, 0.1], chord=0.08, twist=-2, airfoil=wing_airfoil),
            ]
        ),
        asb.Wing(
            name="Horizontal Stabilizer",
            symmetric=True,
            xsecs=[
                asb.WingXSec(xyz_le=[0, 0, 0], chord=0.1, twist=-10, airfoil=tail_airfoil),
                asb.WingXSec(xyz_le=[0.02, 0.17, 0], chord=0.08, twist=-10, airfoil=tail_airfoil)
            ]
        ).translate([0.6, 0, 0.06]),
        asb.Wing(
            name="Vertical Stabilizer",
            symmetric=False,
            xsecs=[
                asb.WingXSec(xyz_le=[0, 0, 0], chord=0.1, twist=0, airfoil=tail_airfoil),
                asb.WingXSec(xyz_le=[0.04, 0, 0.15], chord=0.06, twist=0, airfoil=tail_airfoil)
            ]
        ).translate([0.6, 0, 0.07])
    ],
    fuselages=[
        asb.Fuselage(
            name="Fuselage",
            xsecs=[
                asb.FuselageXSec(
                    xyz_c=[0.8 * xi - 0.1, 0, 0.1 * xi - 0.03],
                    radius=0.6 * asb.Airfoil("dae51").local_thickness(x_over_c=xi)
                )
                for xi in np.cosspace(0, 1, 30)
            ]
        )
    ]
)

# Crate Flight Canvas Airplane

# Create the main wing using the factory function
main_wing_xsecs = [
    asb.WingXSec(xyz_le=[0, 0, 0], chord=0.18, twist=2, airfoil=wing_airfoil),
    asb.WingXSec(xyz_le=[0.01, 0.5, 0], chord=0.16, twist=0, airfoil=wing_airfoil),
    asb.WingXSec(xyz_le=[0.08, 1, 0.1], chord=0.08, twist=-2, airfoil=wing_airfoil),
]
main_wings = create_planar_wing_pair(
    name="Main Wing",
    xsecs=main_wing_xsecs,
    translation=[0, 0, 0],
    ref_direction=[1, 0, 0]
)

# Create the horizontal tail using the factory function
h_tail_xsecs = [
    asb.WingXSec(xyz_le=[0, 0, 0], chord=0.1, twist=-10, airfoil=tail_airfoil),
    asb.WingXSec(xyz_le=[0.02, 0.17, 0], chord=0.08, twist=-10, airfoil=tail_airfoil)
]
h_tail_wings = create_planar_wing_pair(
    name="Horizontal Stabilizer",
    xsecs=h_tail_xsecs,
    translation=[0.6, 0, 0.06],  # Apply translation to the whole pair
    ref_direction=[1, 0, 0]
)

# Create the fuselage
fuselage_xsecs = [
    asb.FuselageXSec(
        xyz_c=[0.8 * xi - 0.1, 0, 0.1 * xi - 0.03],
        radius=0.6 * asb.Airfoil("dae51").local_thickness(x_over_c=xi)
    )
    for xi in np.cosspace(0, 1, 30)
]
fuselage = AeroFuselage(
    name="Fuselage",
    xsecs=fuselage_xsecs
)

# Crate the Vertical Stabilizer
vertical_xsecs = [
    asb.WingXSec(xyz_le=[0, 0, 0], chord=0.1, twist=0, airfoil=tail_airfoil),
    asb.WingXSec(xyz_le=[0.04, 0, 0.15], chord=0.06, twist=0, airfoil=tail_airfoil)
]

v_tail_wing = AeroWing(
    name="Vertical Stabilizer",
    xsecs=vertical_xsecs,
    ref_direction=[1, 0, 0],
).translate([0.6, 0, 0.07])

# Assemble the Vehicle
all_components = [
    fuselage,
    *main_wings,
    *h_tail_wings,
    v_tail_wing,
]

aero_vehicle = AeroVehicle(
    name="David's Glider",
    xyz_ref=[0, 0, 0],  # Vehicle's Center of Gravity
    components=all_components
)

aero_vehicle.compute_buildup()



