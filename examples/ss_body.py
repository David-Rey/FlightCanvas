import aerosandbox as asb
import aerosandbox.numpy as np

from scipy.interpolate import splprep, splev

from FlightCanvas.actuators import ActuatorModel, FirstOrderDeflection
from FlightCanvas.components.aero_fuselage import AeroFuselage
from FlightCanvas.aero_vehicle import AeroVehicle
from FlightCanvas.components.aero_wing import create_planar_wing_pair, AeroWing

from FlightCanvas import utils
from FlightCanvas.open_loop_control import OpenLoopControl

def _smooth_path(points, smoothing_factor=0, n_points=500):
    tck, u = splprep(points.T, s=smoothing_factor)
    u_fine = np.linspace(0, 1, n_points)  # Increase the number of points for smoothness
    smoothed_points = np.array(splev(u_fine, tck)).T
    return smoothed_points


def _get_nosecone_cords(diameter, smoothed=True, n_points=500):
    points = np.array([
        [0.010000, 0.000000],
        [0.057585, 0.238814],
        [0.286398, 0.495763],
        [2.231314, 1.601695],
        [3.222839, 2.097458],
        [6.502500, 3.394068],
        [10.697415, 4.309322],
        [14.320297, 4.500000]
    ])

    scaled_points = np.zeros_like(points)
    scaled_points[:, 0] = points[:, 0]
    scaled_points[:, 1] = points[:, 1] * diameter / 4.5 / 2

    if smoothed:
        return _smooth_path(scaled_points, smoothing_factor=0.002, n_points=n_points)
    else:
        return scaled_points

def model_body(height: float, diameter: float) -> AeroFuselage:
    """
    Returns an AeroSandbox Wing object (the flap) and its PyVista mesh.
    """
    n_points = 100
    nosecone_coords = _get_nosecone_cords(diameter, n_points=n_points)
    end_cord = np.array([[height, nosecone_coords[-1, 1]], [height, 0]])
    nosecone_coords = np.vstack((nosecone_coords, end_cord))

    k = 20  # remove me (for testing)

    fuselage_xsecs = [asb.FuselageXSec(
        xyz_c=[x - k, 0, 0],  # Place the sections based on the nosecone coordinates
        radius=z,  # Set a proportional radius for the fuselage

    )
        for x, z in nosecone_coords
    ]

    fuselage = AeroFuselage(
        name="Fuselage",
        xsecs=fuselage_xsecs,
    ).translate([k, 0, 0])

    return fuselage


if __name__ == '__main__':
    height = 50
    diameter = 9

    body = model_body(height, diameter)

    all_components = [
        body,
    ]

    # Values are another dict mapping component names to their gain.
    aero_vehicle = AeroVehicle(
        name="SS_body",
        xyz_ref=[20, 0, 0],  # Vehicle's Center of Gravity
        components=all_components,
    )

    aero_vehicle.set_mass(95000)
    # MOI calculation
    I_s = (1/2) * 4.5 ** 2
    I_a = ((1/4) * 4.5 ** 2) + ((1/12) * 50 ** 2)

    aero_vehicle.set_moi_diag([I_s, I_a, I_a])

    # DEBUG
    animate = 1

    #aero_vehicle.compute_buildup()
    #aero_vehicle.save_buildup()
    #aero_vehicle.save_buildup_fig()
    aero_vehicle.load_buildup()

    # Add Open Look Control

    if animate:
        pos_0 = np.array([0, 0, 1000])  # Initial position
        vel_0 = np.array([0, 0, -1])  # Initial velocity
        quat_0 = utils.euler_to_quat((0, 0, 0))
        omega_0 = np.array([0, 0, 1])  # Initial angular velocity
        tf = 40

        t_arr, x_arr, u_arr = aero_vehicle.run_sim(pos_0, vel_0, quat_0, omega_0, tf,
                            casadi=False, open_loop_control=None, gravity=False)
        aero_vehicle.init_actors(color='lightblue', show_edges=False, opacity=1)
        aero_vehicle.animate(t_arr, x_arr, u_arr, cam_distance=60, debug=False)
    else:
        aero_vehicle.init_actors(color='lightblue', show_edges=False, opacity=0.8)
        aero_vehicle.init_debug(size=5.5)
        aero_vehicle.show()
