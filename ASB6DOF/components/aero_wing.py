# aero_project/components/aero_wing.py

from typing import List, Union, Optional
import aerosandbox as asb
import aerosandbox.numpy as np
import pyvista as pv

from .aero_component import AeroComponent
import utils


class AeroWing(AeroComponent):
    """
    A wrapper for an AeroSandbox Wing, providing mesh generation and
    visualization capabilities through the AeroComponent interface.
    """

    def __init__(
            self,
            name: str,
            xsecs: List["asb.WingXSec"],
            axis_vector: Union[np.ndarray, List[float]],
            is_prime: bool = True,
            symmetric_comp: Optional['AeroComponent'] = None,
            **kwargs
    ):
        """
        Initializes the AeroWing component
        :param name: The name of the wing or wing section
        :param xsecs: A list of `aerosandbox.WingXSec` objects that define the wing's cross-sections
        :param axis_vector: The primary axis for rotation, typically representing the hinge line of a control surface
        :param is_prime: Inherited from AeroComponent. Used to identify the primary wing in a symmetric pair
        :param kwargs:  Additional keyword arguments to be passed to the `aerosandbox.Wing` constructor
        """
        super().__init__(name, axis_vector, is_prime=is_prime, symmetric_comp=symmetric_comp)

        self.set_translate(axis_vector)

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

    def draw_axis_vector(self, pl: pv.Plotter) -> pv.Actor:
        """
        Draws the component's `axis_vector` in a PyVista plot
        :param pl: The PyVista plotter to draw on
        """
        return utils.draw_line_from_point_and_vector(pl, self.xyz_ref, self.axis_vector, color='green', line_width=4)


def create_symmetric_wing_pair(
        name: str,
        xsecs: List["asb.WingXSec"],
        translation: Union[np.ndarray, List[float]] = (0, 0, 0),
        axis_vector: Union[np.ndarray, List[float]] = (1, 0, 0),
        **kwargs
) -> List[AeroWing]:
    """
    Creates a symmetric pair of AeroWing objects from a half-span definition
    :param name: The name of the wing or wing section
    :param xsecs: A list of `aerosandbox.WingXSec` objects that define the wing's cross-sections
    :param translation: The new reference position [x, y, z]
    :param axis_vector: The primary axis for rotation, typically representing the hinge line of a control surface
    :param kwargs:  Additional keyword arguments to be passed to the `aerosandbox.Wing` constructor
    """
    # Create the right-hand wing wrapper from the provided cross-sections
    right_aero_wing = AeroWing(
        name=f"{name}",
        xsecs=xsecs,
        axis_vector=axis_vector,
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
        axis_vector=utils.flip_y(axis_vector),
        is_prime=False,
        symmetric_comp=right_aero_wing,
        **kwargs
    )

    # Apply the overall translation to both wings and return them as a list
    right_aero_wing.translate(translation)

    # Flip y-axis of translation vector
    left_translation = utils.flip_y(translation)
    left_aero_wing.translate(left_translation)

    return [right_aero_wing, left_aero_wing]
