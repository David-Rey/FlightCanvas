# aero_project/main.py

import aerosandbox as asb
import aerosandbox.numpy as np

from components.aero_fuselage import AeroFuselage
from components.aero_wing import create_symmetric_wing_pair, AeroWing
from aero_vehicle import AeroVehicle
import utils
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # Define Airfoils
    wing_airfoil = asb.Airfoil("sd7037")
    tail_airfoil = asb.Airfoil("naca0010")

    # Create the main wing using the factory function
    main_wing_xsecs = [
        asb.WingXSec(xyz_le=[0, 0, 0], chord=0.18, twist=2, airfoil=wing_airfoil),
        asb.WingXSec(xyz_le=[0.01, 0.5, 0], chord=0.16, twist=0, airfoil=wing_airfoil),
        asb.WingXSec(xyz_le=[0.08, 1, 0.1], chord=0.08, twist=-2, airfoil=wing_airfoil),
    ]
    main_wings = create_symmetric_wing_pair(
        name="Main Wing",
        xsecs=main_wing_xsecs,
        translation=[0.08, 0.05, 0],
        axis_vector=[1, 0, 0]
    )

    # Create the horizontal tail using the factory function
    h_tail_xsecs = [
        asb.WingXSec(xyz_le=[0, 0, 0], chord=0.1, twist=-0, airfoil=tail_airfoil),
        asb.WingXSec(xyz_le=[0.02, 0.17, 0], chord=0.08, twist=-0, airfoil=tail_airfoil)
    ]
    h_tail_wings = create_symmetric_wing_pair(
        name="Horizontal Stabilizer",
        xsecs=h_tail_xsecs,
        translation=[0.6, 0, 0.06],  # Apply translation to the whole pair
        axis_vector=[1, 0, 0]
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
        #translation=[0.6, 0, 0.07],  # FIX
        axis_vector=[1, 0, 0],
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
        xyz_ref=[0.05, 0, 0],  # Vehicle's Center of Gravity
        components=all_components
    )

    #aero_vehicle.compute_buildup()
    #aero_vehicle.save_buildup()
    #aero_vehicle.save_buildup_fig()
    aero_vehicle.load_buildup()
    #aero_vehicle.compute_forces_and_moments_lookup(quat, vel, omega)
    #aero_vehicle.set_control(["Horizontal Stabilizer",
    #                          "Horizontal Stabilizer Star"],
    #                         np.deg2rad([0, 30]))
    #aero_vehicle.test_remove_me()

    t_arr, x_arr = aero_vehicle.run_sim(20)

    '''
    plt.figure()
    plt.plot(t_arr, x_arr[2])

    plt.figure()
    plt.plot(t_arr, x_arr[-2])
    plt.show()
    '''

    # Visualize the Vehicle
    aero_vehicle.init_actors(color='lightblue', show_edges=False, opacity=1)
    #aero_vehicle.init_debug()

    aero_vehicle.animate(t_arr, x_arr)
    #aero_vehicle.set_control(["Main Wing", "Main Wing Star"], np.deg2rad([30, 0]))
    #aero_vehicle.draw(debug=True)
    #aero_vehicle.update_actors(state)
    aero_vehicle.show()

    #aero_vehicle.draw_buildup("Horizontal Stabilizer", "M_b", index=2)
