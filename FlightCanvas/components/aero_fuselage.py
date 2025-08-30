# aero_project/FlightCanvas/aero_fuselage.py

import aerosandbox as asb
import numpy as np

from .aero_component import AeroComponent
from FlightCanvas import utils
from FlightCanvas.actuators.actuators import ActuatorModel
from typing import Optional, Union, List


class AeroFuselage(AeroComponent):
    """
    A wrapper for an AeroSandbox Fuselage, providing mesh generation and
    visualization capabilities through the AeroComponent interface.
    """

    def __init__(
        self,
        name: str,
        xsecs: List["asb.FuselageXSec"],
        ref_direction: Union[np.ndarray, List[float]] = (1, 0, 0),
        is_prime: bool = True,
        symmetric_comp: Optional['AeroComponent'] = None,
        actuator_model: Optional[ActuatorModel] = None,
        **kwargs,
    ):
        """
        Initializes the AeroFuselage component
        :param name: The name of the fuselage
        :param xsecs: A list of `aerosandbox.FuselageXSec` objects that define the fuselage's cross-sections from nose to tail
        :param ref_direction: The primary axis of the fuselage, typically aligned with the body x-axis. Defaults to (1, 0, 0)
        :param is_prime: Inherited from AeroComponent. Since fuselages are rarely mirrored, this usually remains True
        :param symmetric_comp: The AeroComponent object that is symmetric to the current AeroComponent object
        :param actuator_model: The actuator model object to use
        :param kwargs: Additional keyword arguments to be passed to the `aerosandbox.Fuselage` constructor
        """
        super().__init__(name, ref_direction, is_prime=is_prime, symmetric_comp=symmetric_comp, actuator_model=actuator_model)

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
