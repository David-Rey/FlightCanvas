# aero_project/FlightCanvas/examples/glider.py

import aerosandbox as asb
import aerosandbox.numpy as np

from FlightCanvas.analysis.anlaysis import Analysis
from FlightCanvas.components.aero_fuselage import AeroFuselage
from FlightCanvas.components.aero_wing import create_planar_wing_pair, AeroWing
from FlightCanvas.vehicle.aero_vehicle import AeroVehicle
from FlightCanvas.analysis.log import Log
from FlightCanvas.Flight.flight import Flight
from FlightCanvas.analysis.vehicle_visualizer import VehicleVisualizer

from FlightCanvas import utils

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
    main_wings = create_planar_wing_pair(
        name="Main Wing",
        xsecs=main_wing_xsecs,
        translation=[0.08, 0.05, 0],
        ref_direction=[1, 0, 0]
    )

    # Create the horizontal tail using the factory function
    h_tail_xsecs = [
        asb.WingXSec(xyz_le=[0, 0, 0], chord=0.1, twist=0, airfoil=tail_airfoil),
        asb.WingXSec(xyz_le=[0.02, 0.17, 0], chord=0.08, twist=0, airfoil=tail_airfoil)
    ]
    h_tail_wings = create_planar_wing_pair(
        name="Horizontal Stabilizer",
        xsecs=h_tail_xsecs,
        translation=[0.6, 0, 0.06],  # Apply translation to the whole pair
        ref_direction=[1, 0, -0.02]
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
        xyz_ref=[0.15, 0, 0],  # Vehicle's Center of Gravity
        components=all_components
    )
    aero_vehicle.compute_buildup()
    #aero_vehicle.save_buildup()
    #aero_vehicle.load_buildup()
    aero_vehicle.init_vehicle_dynamics(None)

    state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'q0', 'q1', 'q2', 'q3', 'wx', 'wy', 'wz']
    maxSteps = 1000
    log = Log(state_names, [], maxSteps)

    dt = 0.01
    tf = 22
    flight = Flight(aero_vehicle, tf, dt=dt)

    pos_0 = np.array([0, 0, 950])  # Initial position
    vel_0 = np.array([90, 0, 0.01])  # Initial velocity
    quat_0 = utils.euler_to_quat((0, 0, 0))
    omega_0 = np.array([0, 0, .1])  # Initial angular velocity
    inital_state = np.concatenate((pos_0, vel_0, quat_0, omega_0))

    flight.run_sim(inital_state, log)

    analysis = Analysis(log)
    analysis.generate_euler_angle_plot()

    #vv = VehicleVisualizer(aero_vehicle, log)
    #vv.init_actors()
    #vv.add_grid()
    #vv.animate()


    #aero_vehicle.compute_buildup()
    #aero_vehicle.save_buildup()
    #aero_vehicle.save_buildup_fig()


    #aero_vehicle.test_new_buildup()
    '''
    if animate:
        pos_0 = np.array([0, 0, 950])  # Initial position
        vel_0 = np.array([90, 0, 0.01])  # Initial velocity
        quat_0 = utils.euler_to_quat((0, 0, 0))
        omega_0 = np.array([0, 0, 0])  # Initial angular velocity
        tf = 40

        #t_arr, x_arr = aero_vehicle.run_sim(pos_0, vel_0, quat_0, omega_0, tf, print_debug=False)
        t_arr, x_arr, u_arr = aero_vehicle.run_sim(pos_0, vel_0, quat_0, omega_0, tf,
                                                   casadi=True)
        aero_vehicle.init_actors(color='lightblue', show_edges=False, opacity=1)
        aero_vehicle.animate(t_arr, x_arr, u_arr, cam_distance=4)
    else:
        aero_vehicle.init_actors(color='lightblue', show_edges=False, opacity=0.8)
        aero_vehicle.init_debug()
        aero_vehicle.show()
    '''
