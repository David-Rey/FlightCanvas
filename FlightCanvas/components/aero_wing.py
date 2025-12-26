# aero_project/FlightCanvas/aero_wing.py

from typing import List, Union, Optional
import aerosandbox as asb
import aerosandbox.numpy as np

from .aero_component import AeroComponent
from FlightCanvas import utils
from FlightCanvas.vehicle.actuator_dynamics import Actuator


class AeroWing(AeroComponent):
    """
    A wrapper for an AeroSandbox Wing, providing mesh generation and
    visualization capabilities through the AeroComponent interface.
    """

    def __init__(
        self,
        name: str,
        xsecs: List["asb.WingXSec"],
        ref_direction: Union[np.ndarray, List[float]],
        control_pivot=None,
        is_prime: bool = True,
        symmetric_comp: Optional['AeroComponent'] = None,
        actuator_model: Optional[Actuator] = None,
        symmetry_type=None,
        **kwargs
    ):
        """
        Initializes the AeroWing component
        :param name: The name of the wing or wing section
        :param xsecs: A list of `aerosandbox.WingXSec` objects that define the wing's cross-sections
        :param ref_direction: The primary axis for rotation, typically representing the hinge line of a starship_control surface
        :param control_pivot: The axis at which the component will rotate given a starship_control input
        :param is_prime: Inherited from AeroComponent. Used to identify the primary wing in a symmetric pair
        :param symmetric_comp: The symmetric component of the wing
        :param actuator_model: The actuator model of the wing deflection around the starship_control pivot
        :param kwargs:  Additional keyword arguments to be passed to the `aerosandbox.Wing` constructor
        """
        super().__init__(name, ref_direction, control_pivot=control_pivot, is_prime=is_prime,
                         symmetric_comp=symmetric_comp, actuator_model=actuator_model)

        # Set translation
        self.set_translate(ref_direction)

        # Set symmetry type
        self.symmetry_type = symmetry_type

        # Ensure wing is not symmetric for visualization
        kwargs['symmetric'] = False

        # Create the underlying AeroSandbox Wing object
        self.wing = asb.Wing(name=name, xsecs=xsecs, **kwargs)

        # Link this specific instance to the generic asb_object attribute
        self.asb_object = self.wing

        # Create a temporary single-component airplane for isolated analysis
        self.asb_airplane = asb.Airplane(name=name, wings=[self.wing])

    def generate_mesh(self):
        """
        Generates a 3D PyVista mesh from the AeroSandbox Wing definition
        """
        # The translation will be applied to the mesh after generation.
        self.mesh = utils.get_mesh(self.wing)


def create_planar_wing_pair(
    name: str,
    xsecs: List["asb.WingXSec"],
    translation: Union[np.ndarray, List[float]] = (0, 0, 0),
    ref_direction: Union[np.ndarray, List[float]] = (1, 0, 0),
    control_pivot=None,
    actuator_model: Optional[Actuator] = None,
    **kwargs
) -> List[AeroWing]:
    """
    Creates a symmetric pair of AeroWing objects from a half-span definition
    :param name: The name of the wing or wing section
    :param xsecs: A list of `aerosandbox.WingXSec` objects that define the wing's cross-sections
    :param translation: The new reference position [x, y, z]
    :param ref_direction: The primary axis for rotation, typically representing the hinge line of a starship_control surface
    :param control_pivot: The axis at which the component will rotate given a starship_control input
    :param actuator_model: The actuator model of the wing deflection around the starship_control pivot
    :param kwargs:  Additional keyword arguments to be passed to the `aerosandbox.Wing` constructor
    """
    # Create the right-hand wing wrapper from the provided cross-sections
    right_aero_wing = AeroWing(
        name=f"{name}",
        xsecs=xsecs,
        ref_direction=ref_direction,
        control_pivot=control_pivot,
        actuator_model=actuator_model,
        **kwargs
    )

    # Manually create the mirrored cross-sections for the left-hand wing
    mirrored_xsecs = []
    if xsecs is not None:
        for xsec in xsecs:
            new_xyz_le = xsec.xyz_le.copy()

            mirrored_xsecs.append(
                asb.WingXSec(
                    xyz_le=new_xyz_le,
                    chord=xsec.chord,
                    twist=xsec.twist,
                    airfoil=xsec.airfoil
                )
            )

    # Create the left-hand wing wrapper with the new mirrored cross-sections
    left_aero_wing = AeroWing(
        name=f"{name} Star",
        xsecs=mirrored_xsecs,
        ref_direction=utils.flip_y(ref_direction),
        control_pivot=utils.flip_y(control_pivot),
        actuator_model=actuator_model,
        is_prime=False,
        symmetric_comp=right_aero_wing,
        symmetry_type='xz-plane',
        **kwargs
    )

    # Apply the overall translation to both wings and return them as a list
    right_aero_wing.translate(translation)

    # Flip y-axis of translation vector
    left_translation = utils.flip_y(translation)
    left_aero_wing.translate(left_translation)

    return [right_aero_wing, left_aero_wing]


def create_axial_wing_pair(
    name: str,
    xsecs: List["asb.WingXSec"],
    translation: Union[np.ndarray, List[float]] = (0, 0, 0),
    ref_direction: Union[np.ndarray, List[float]] = (1, 0, 0),
    control_pivot: Union[np.ndarray, List[float]] = None,
    actuator_model: Optional[Actuator] = None,
    num_wings: int = 2,
    **kwargs
) -> List[AeroWing]:
    """
    Creates an axial symmetric pair of AeroWing objects
    :param name: The name of the wing or wing section
    :param xsecs: A list of `aerosandbox.WingXSec` objects that define the wing's cross-sections
    :param translation: The new reference position [x, y, z]
    :param ref_direction: The primary axis for rotation, typically representing the hinge line of a starship_control surface
    :param control_pivot: The axis at which the component will rotate given a starship_control input
    :param actuator_model: The actuator model of the wing deflection around the starship_control pivot
    :param num_wings: The number of wings to create
    :param kwargs:  Additional keyword arguments to be passed to the `aerosandbox.Wing` constructor
    """
    if control_pivot is None:
        control_pivot = ref_direction

    # Create the right-hand wing wrapper from the provided cross-sections
    main_aero_wing = AeroWing(
        name=f"{name}",
        xsecs=xsecs,
        ref_direction=ref_direction,
        control_pivot=control_pivot,
        actuator_model=actuator_model,
        **kwargs
    ).translate(translation)

    wing_array = [main_aero_wing]

    angle_array = np.linspace(0, 360, num_wings + 1)

    for i in range(num_wings - 1):
        R_Comp_Body = utils.rotate_z(angle_array[i + 1])

        new_wing = AeroWing(
            name=f"{name} {i + 1}",
            xsecs=xsecs,
            ref_direction=R_Comp_Body @ ref_direction,
            control_pivot=R_Comp_Body @ control_pivot,
            actuator_model=actuator_model,
            is_prime=False,
            symmetric_comp=main_aero_wing,
            symmetry_type='x-radial',
            **kwargs
        ).translate(R_Comp_Body @ translation)
        new_wing.radial_angle = angle_array[i + 1]
        wing_array.append(new_wing)

    return wing_array
