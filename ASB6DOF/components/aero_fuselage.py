# aero_project/components/aero_fuselage.py

from typing import List, Union
import aerosandbox as asb
import numpy as np

from .aero_component import AeroComponent
import utils


class AeroFuselage(AeroComponent):
    """
    A wrapper for an AeroSandbox Fuselage, providing mesh generation and
    visualization capabilities through the AeroComponent interface.
    """

    def __init__(
            self,
            name: str,
            xsecs: List["asb.FuselageXSec"],
            axis_vector: Union[np.ndarray, List[float]] = (1, 0, 0),
            is_prime: bool = True,
            **kwargs
    ):
        """
        Initializes the AeroFuselage component
        :param name: The name of the fuselage
        :param xsecs: A list of `aerosandbox.FuselageXSec` objects that define the fuselage's cross-sections from nose to tail
        :param axis_vector: The primary axis of the fuselage, typically aligned with the body x-axis. Defaults to (1, 0, 0)
        :param is_prime: Inherited from AeroComponent. Since fuselages are rarely mirrored, this usually remains True
        :param kwargs: Additional keyword arguments to be passed to the `aerosandbox.Fuselage` constructor
        """
        super().__init__(name, axis_vector, is_prime)

        # Ensure fuselage is not symmetric for visualization
        kwargs['symmetric'] = False

        # Create the underlying AeroSandbox Fuselage object
        self.fuselage = asb.Fuselage(name=name, xsecs=xsecs, **kwargs)

        # Link this specific instance to the generic asb_object attribute
        self.asb_object = self.fuselage

        # # For aerodynamic analysis, AeroBuildup requires an Airplane object
        self.asb_airplane = asb.Airplane(name=name, fuselages=[self.fuselage])

    def generate_mesh(self):
        """
        Generates a 3D PyVista mesh from the AeroSandbox Fuselage definition
        """
        self.mesh = utils.get_mesh(self.fuselage, tangential_resolution=100)
