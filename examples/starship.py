import aerosandbox as asb
import aerosandbox.numpy as np

from scipy.interpolate import splprep, splev

from FlightCanvas.actuators import ActuatorModel, FirstOrderDeflection
from FlightCanvas.components.aero_fuselage import AeroFuselage
from FlightCanvas.aero_vehicle import AeroVehicle
from FlightCanvas.components.aero_wing import create_planar_wing_pair, AeroWing

from FlightCanvas import utils


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


def _flat_plate_airfoil(thickness=0.01, n_points=100):
    """
    Creates a flat plate airfoil with a specified thickness.

    Args:
        thickness: The maximum thickness of the flat plate, as a fraction of chord length.
        n_points: Number of points used to define the airfoil.

    Returns:
        A numpy array of airfoil coordinates.
    """
    x = np.linspace(1, 0, n_points)
    y_upper = thickness / 2 * np.ones_like(x)  # Upper surface
    y_lower = -thickness / 2 * np.ones_like(x)  # Lower surface

    # Combine upper and lower surfaces to form the airfoil
    x_coords = np.concatenate([x, x[::-1]])
    y_coords = np.concatenate([y_upper, y_lower[::-1]])

    return np.vstack([x_coords, y_coords]).T


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

    flap_airfoil = asb.Airfoil(coordinates=_flat_plate_airfoil(thickness=0.02))

    front_flap_length = 8
    front_flap_width = 4.8
    front_flap_p1 = 6
    front_flap_p2 = 8.5
    front_flap_xsecs = [
        asb.WingXSec(
            xyz_le=[0, 0, 0],
            chord=front_flap_length,
            twist=0,
            airfoil=flap_airfoil,
        ),
        asb.WingXSec(
            xyz_le=[front_flap_p1,
                    front_flap_width,
                    0],
            chord=front_flap_p2 - front_flap_p1,
            twist=0,
            airfoil=flap_airfoil
        )
    ]

    front_flaps = create_planar_wing_pair(
        name="Front Flap",
        xsecs=front_flap_xsecs,
        translation=[5, 2.9, 0],  # Apply translation to the whole pair
        ref_direction=[1, 0.18, 0],
        control_pivot=[1, 0.18, 0],
        actuator_model=FirstOrderDeflection(time_constant=0.1)
    )

    back_flap_length = 15
    back_flap_width = 5.8
    back_flap_p1 = 8
    back_flap_p2 = 14
    back_flap_xsecs = [
        asb.WingXSec(
            xyz_le=[0, 0, 0],
            chord=back_flap_length,
            twist=0,
            airfoil=flap_airfoil,
        ),
        asb.WingXSec(
            xyz_le=[back_flap_p1,
                    back_flap_width,
                    0],
            chord=back_flap_p2 - back_flap_p1,
            twist=0,
            airfoil=flap_airfoil
        )
    ]

    back_flaps = create_planar_wing_pair(
        name="Back Flap",
        xsecs=back_flap_xsecs,
        translation=[35, 4.5, 0],  # Apply translation to the whole pair
        ref_direction=[1, 0, 0],
        control_pivot=[1, 0, 0]
    )

    all_components = [
        body,
        *front_flaps,
        *back_flaps,
    ]

    # Values are another dict mapping component names to their gain.
    control_mapping = {
        "control 1": {
            "Front Flap": 1.0
        },
        "control 2": {
            "Front Flap Star": 1.0
        },
        "control 3": {
            "Back Flap": 1.0
        },
        "control 4": {
            "Back Flap Star": 1.0
        }
    }

    aero_vehicle = AeroVehicle(
        name="Starship",
        xyz_ref=[20, 0, 0],  # Vehicle's Center of Gravity
        components=all_components,
    )
    aero_vehicle.set_mass(95000)
    aero_vehicle.set_moi_factor(70)
    aero_vehicle.set_control_mapping(control_mapping)

    # DEBUG
    animate = 1

    aero_vehicle.compute_buildup()
    #aero_vehicle.save_buildup()
    #aero_vehicle.save_buildup_fig()
    #aero_vehicle.load_buildup()

    front_flap_del = np.deg2rad(40)
    back_flap_del = np.deg2rad(10)
    control_arr = np.array([front_flap_del, front_flap_del, back_flap_del, back_flap_del])
    aero_vehicle.set_control(control_arr)

    if animate:
        pos_0 = np.array([0, 0, 950])  # Initial position
        vel_0 = np.array([0, 0, -10])  # Initial velocity
        quat_0 = utils.euler_to_quat((0, 0, 0))
        omega_0 = np.array([0, 0, 0])  # Initial angular velocity
        tf = 20

        t_arr, x_arr = aero_vehicle.run_sim(pos_0, vel_0, quat_0, omega_0, tf, N=200)
        aero_vehicle.init_actors(color='lightblue', show_edges=False, opacity=1)
        aero_vehicle.animate(t_arr, x_arr, cam_distance=60)
    else:
        aero_vehicle.init_actors(color='lightblue', show_edges=False, opacity=0.8)
        aero_vehicle.init_debug(size=5.5)
        aero_vehicle.show()

"""
    front_flap_del = 60
    back_flap_del = 0
    
  Position (Inertial): [1.76196968e+02 1.03554441e-04 2.87400366e+01]
  Velocity (Inertial): [ 1.73064557e+01  1.99982349e-04 -5.22171862e+01]
  Quaternion: [ 9.99971588e-01 -5.11811703e-05  7.52580123e-03 -1.00681805e-04]
  Angular Velocity (Body): [-0.00019237 -0.00608645 -0.00038755]
"""
