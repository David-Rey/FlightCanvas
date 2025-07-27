import aerosandbox as asb
import aerosandbox.numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional
import pyvista as pv
import utils


class AeroWingSet:
    def __init__(
            self,
            name: Optional[str] = None,
            xsecs: Optional[List["asb.WingXSec"]] = None,
            color: Optional[Union[str, Tuple[float]]] = None,
            analysis_specific_options: Optional[Dict[type, Dict[str, Any]]] = None,
            controllable: Optional[bool] = False,
            pivot: Optional[np.ndarray] = np.array([-1, 0, 0]),
    ):
        self.name = name

        main_wing = AeroWing(
            name=name,
            xsecs=xsecs,
            color=color,
            analysis_specific_options=analysis_specific_options,
            controllable=controllable,
            pivot=pivot,
        )

        mirrored_xsecs = []
        if xsecs is not None:  # Ensure xsecs is not None before iterating
            for xsec in xsecs:
                new_xyz_le = xsec.xyz_le.copy()  # Make a copy to avoid modifying the original array
                new_xyz_le[1] *= -1  # Flip the y-coordinate (index 1)

                mirrored_xsecs.append(
                    asb.WingXSec(  # Create a new WingXSec instance
                        xyz_le=new_xyz_le,
                        chord=xsec.chord,
                        twist=-xsec.twist,
                        airfoil=xsec.airfoil,
                    )
                )

        secondary_wing = AeroWing(
            name=f"{name}_mirrored" if name else "Mirrored Wing",
            xsecs=mirrored_xsecs,
            color=color,
            analysis_specific_options=analysis_specific_options,
            controllable=controllable,
            pivot=pivot,
        )

        self.wings = [main_wing, secondary_wing]
        self.wing_actors: List[pv.Actor] = []

    def generate_mesh(self):
        [wing.generate_mesh() for wing in self.wings]

    def draw_debug(self, pl: pv.Plotter, com: np.ndarray, sphere_radius=0.02):
        [wing.draw_debug(pl, com ,sphere_radius=sphere_radius) for wing in self.wings]

    def get_actors(self, pl: pv.Plotter, **kwargs):
        self.wing_actors = [wing.get_actor(pl, **kwargs) for wing in self.wings]

    def translate(self, xyz: Union[np.ndarray, List[float]]) -> "AeroWingSet":
        self.wings = [wing.translate(xyz) for wing in self.wings]
        return self

    def set_translate(self, xyz: Union[np.ndarray, List[float]]):
        [wing.set_translate(xyz) for wing in self.wings]


class AeroWing:
    """
    A wrapper for an AeroSandbox Wing, providing mesh generation and visualization capabilities.

    This class facilitates the creation of a 3D mesh representation of an AeroSandbox
    Wing object and its integration into a PyVista plotting environment.
    """

    def __init__(
            self,
            name: Optional[str] = None,
            xsecs: Optional[List["asb.WingXSec"]] = None,
            color: Optional[Union[str, Tuple[float]]] = None,
            analysis_specific_options: Optional[Dict[type, Dict[str, Any]]] = None,
            controllable: Optional[bool] = False,
            pivot: Optional[np.ndarray] = np.array([-1, 0, 0]),
    ):
        """
        Initializes an AeroWing instance.

        Args:
            name (Optional[str]): The name of the wing. Defaults to None.
            xsecs (Optional[List["asb.WingXSec"]]): A list of AeroSandbox WingXSec objects
                that define the wing's geometry. Defaults to an empty list if None.
            color (Optional[Union[str, Tuple[float]]]): The color of the wing for visualization.
                Defaults to None.
            analysis_specific_options (Optional[Dict[type, Dict[str, Any]]]):
                Analysis-specific options for the AeroSandbox Wing. Defaults to an empty dict if None.
            symmetric (bool): Indicates if the wing is symmetric across the XZ-plane.
                This is passed to the underlying asb.Wing. Defaults to False.
        """
        self.wing = asb.Wing(
            name=name,
            xsecs=xsecs if xsecs is not None else [],
            color=color,
            analysis_specific_options=analysis_specific_options if analysis_specific_options is not None else {}
        )
        self.mesh: Optional[pv.PolyData] = None
        self.xyz_ref = np.array([0, 0, 0])
        self.controllable = controllable
        self.pivot = pivot

    def generate_mesh(self):
        self.mesh = utils.get_mesh(self.wing, translation_vector=self.xyz_ref)

    def draw_debug(self, pl: pv.Plotter, com: np.ndarray, sphere_radius=0.02):
        sphere = pv.Sphere(radius=sphere_radius, center=self.xyz_ref)
        pl.add_mesh(sphere, color='red', show_edges=False, label='Translation Point')
        utils.plot_arrow_from_points(pl, self.xyz_ref, com)

    def get_actor(self, pl: pv.Plotter, **kwargs):
        actor = pl.add_mesh(self.mesh, **kwargs)
        return actor

    def translate(self, xyz: Union[np.ndarray, List[float]]) -> "AeroWing":
        self.xyz_ref = np.array(xyz)
        return self

    def set_translate(self, xyz: Union[np.ndarray, List[float]]):
        self.xyz_ref = np.array(xyz)


class AeroFuselage:
    """
    A wrapper for an AeroSandbox Fuselage, providing mesh generation and visualization.

    This class allows for the creation of a 3D mesh representation of an AeroSandbox
    Fuselage object and its integration into a PyVista plotting environment.
    """

    def __init__(
            self,
            name: Optional[str] = "Untitled",
            xsecs: Optional[List["asb.FuselageXSec"]] = None,
            color: Optional[Union[str, Tuple[float]]] = None,
            analysis_specific_options: Optional[Dict[type, Dict[str, Any]]] = None,
    ):
        """
        Initializes an AeroFuselage instance.

        Args:
            name (Optional[str]): The name of the fuselage. Defaults to "Untitled".
            xsecs (Optional[List["asb.FuselageXSec"]]): A list of AeroSandbox FuselageXSec
                objects that define the fuselage's geometry. Defaults to an empty list if None.
            color (Optional[Union[str, Tuple[float]]]): The color of the fuselage for
                visualization. Defaults to None.
            analysis_specific_options (Optional[Dict[type, Dict[str, Any]]]):
                Analysis-specific options for the AeroSandbox Fuselage. Defaults to an empty dict if None.
        """
        self.fuselage = asb.Fuselage(
            name=name,
            xsecs=xsecs if xsecs is not None else [],
            symmetric=False,
            color=color,
            analysis_specific_options=analysis_specific_options if analysis_specific_options is not None else {}
        )
        self.mesh: Optional[pv.PolyData] = None
        self.xyz_ref = np.array([0, 0, 0])

    def generate_mesh(self):
        self.mesh = utils.get_mesh(self.fuselage, tangential_resolution=100)

    def draw_debug(self, pl: pv.Plotter, com: np.ndarray, sphere_radius=0.02):
        sphere = pv.Sphere(radius=sphere_radius, center=self.xyz_ref)
        pl.add_mesh(sphere, color='red', show_edges=False, label='Translation Point')
        utils.plot_arrow_from_points(pl, self.xyz_ref, com)

    def get_actor(self, pl, **kwargs):
        actor = pl.add_mesh(self.mesh, **kwargs)
        return actor

    def translate(self, xyz: Union[np.ndarray, List[float]]) -> "AeroFuselage":
        self.xyz_ref = np.array(xyz)
        return self

    def set_translate(self, xyz: Union[np.ndarray, List[float]]):
        self.xyz_ref = np.array(xyz)


class AeroVehicle:
    """
    Represents a complete aerodynamic vehicle composed of wings and fuselages.

    This class manages a collection of AeroWing and AeroFuselage objects,
    facilitating collective mesh generation and visualization of the entire vehicle.
    """

    def __init__(
            self,
            name: Optional[str] = None,
            xyz_ref: Optional[Union[np.ndarray, List[float]]] = None,
            wing_sets: Optional[List[AeroWingSet]] = None,
            wings: Optional[List[AeroWing]] = None,
            fuselages: Optional[List[AeroFuselage]] = None,
    ):
        """
        Initializes an AeroVehicle instance.

        Args:
            name (Optional[str]): The name of the vehicle. Defaults to None.
            xyz_ref (Optional[Union[np.ndarray, List[float]]]): A 3-element list or array
                representing the reference coordinates (e.g., CG) of the vehicle.
                Defaults to [0, 0, 0] if None.
            wings (Optional[List[AeroWing]]): A list of AeroWing objects that are part of
                this vehicle. Defaults to an empty list if None.
            wing_sets (Optional[List[AeroWingSet]]): A list of AeroWingSets objects
                that are symmetric wings and part of the vehicle. Defaults to an empty
                 list if None.
            fuselages (Optional[List[AeroFuselage]]): A list of AeroFuselage objects
                that are part of this vehicle. Defaults to an empty list if None.
        """
        self.name = name if name is not None else "Untitled Vehicle"
        self.xyz_ref = np.array(xyz_ref) if xyz_ref is not None else np.array([0., 0., 0.])
        self.wings = wings if wings is not None else []
        self.wing_sets = wing_sets if wing_sets is not None else []
        self.fuselages = fuselages if fuselages is not None else []

        self.wing_actors: List[pv.Actor] = []
        self.fuselage_actors: List[pv.Actor] = []

    def generate_mesh(self):
        [wing.generate_mesh() for wing in self.wings]
        [wing_set.generate_mesh() for wing_set in self.wing_sets]
        [fuselage.generate_mesh() for fuselage in self.fuselages]

    def get_actors(self, pl: pv.Plotter, **kwargs):
        self.wing_actors = [wing.get_actor(pl, **kwargs) for wing in self.wings]
        self.fuselages_actors = [fuselage.get_actor(pl, **kwargs) for fuselage in self.fuselages]
        [wing_sets.get_actors(pl) for wing_sets in self.wing_sets]

    def draw_debug(self, pl: pv.Plotter, sphere_radius=0.02):
        [wing.draw_debug(pl, self.xyz_ref, sphere_radius=sphere_radius) for wing in self.wings]
        [fuselage.draw_debug(pl, self.xyz_ref, sphere_radius=sphere_radius) for fuselage in self.fuselages]
        [wing_set.draw_debug(pl, self.xyz_ref, sphere_radius=sphere_radius) for wing_set in self.wing_sets]

        sphere = pv.Sphere(radius=sphere_radius, center=self.xyz_ref)
        pl.add_mesh(sphere, color='yellow', show_edges=False, label='Translation Point')


if __name__ == '__main__':
    wing_airfoil = asb.Airfoil("sd7037")
    tail_airfoil = asb.Airfoil("naca0010")

    airplane = asb.Airplane(
        name="Peter's Glider ASB",
        xyz_ref=[0.1, 0, 0],  # CG location
        wings=[
            asb.Wing(
                name="Main Wing",
                symmetric=True,  # Should this wing be mirrored across the XZ plane?
                xsecs=[  # The wing's cross ("X") sections
                    asb.WingXSec(  # Root
                        xyz_le=[0, 0, 0],
                        # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                        chord=0.18,
                        twist=2,  # degrees
                        airfoil=wing_airfoil,  # Airfoils are blended between a given XSec and the next one.
                    ),
                    asb.WingXSec(  # Mid
                        xyz_le=[0.01, 0.5, 0],
                        chord=0.16,
                        twist=0,
                        airfoil=wing_airfoil,
                    ),
                    asb.WingXSec(  # Tip
                        xyz_le=[0.08, 1, 0.1],
                        chord=0.08,
                        twist=-2,
                        airfoil=wing_airfoil,
                    ),
                ]
            ),
            asb.Wing(
                name="Horizontal Stabilizer",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(  # root
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                        twist=-10,
                        airfoil=tail_airfoil,
                    ),
                    asb.WingXSec(  # tip
                        xyz_le=[0.02, 0.17, 0],
                        chord=0.08,
                        twist=-10,
                        airfoil=tail_airfoil
                    )
                ]
            ).translate([0.6, 0, 0.06]),
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

    aero_vehicle = AeroVehicle(
        name="Peter's Glider ASB",
        xyz_ref=[0.1, 0, 0],  # CG location
        wing_sets=[
            AeroWingSet(
                name="Main Wing",
                xsecs=[  # The wing's cross ("X") sections
                    asb.WingXSec(  # Root
                        xyz_le=[0, 0, 0],
                        # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                        chord=0.18,
                        twist=2,  # degrees
                        airfoil=wing_airfoil,  # Airfoils are blended between a given XSec and the next one.
                    ),
                    asb.WingXSec(  # Mid
                        xyz_le=[0.01, 0.5, 0],
                        chord=0.16,
                        twist=0,
                        airfoil=wing_airfoil,
                    ),
                    asb.WingXSec(  # Tip
                        xyz_le=[0.08, 1, 0.1],
                        chord=0.08,
                        twist=-2,
                        airfoil=wing_airfoil,
                    ),
                ]
            ),
            #],
            #wings=[
            AeroWingSet(
                name="Horizontal Stabilizer",
                xsecs=[
                    asb.WingXSec(  # root
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                        twist=-10,
                        airfoil=tail_airfoil,
                    ),
                    asb.WingXSec(  # tip
                        xyz_le=[0.02, 0.17, 0],
                        chord=0.08,
                        twist=-10,
                        airfoil=tail_airfoil
                    )
                ]
            ).translate([0.6, 0, 0.06]),
        ],
        fuselages=[
            AeroFuselage(
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
    pl = pv.Plotter()
    aero_vehicle.generate_mesh()
    aero_vehicle.get_actors(pl, opacity=0.8)
    aero_vehicle.draw_debug(pl)
    pl.add_axes()
    pl.show_grid()

    pl.show()
    #aero_vehicle.draw()

    #airplane.draw()

    #aero_vehicle.draw(show=False)
    #show_kwargs = {
    #    "show_edges": True,
    #    "show_grid": True,
    #}
    #plt.plot(**show_kwargs)
