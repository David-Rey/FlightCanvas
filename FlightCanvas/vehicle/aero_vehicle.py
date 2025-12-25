# aero_project/FlightCanvas/aero_vehicle.py

import pathlib
from typing import List, Union

import aerosandbox.numpy as np

from FlightCanvas.vehicle.vehicle_dynamics import VehicleDynamics

# Local application imports
from FlightCanvas.components.aero_component import AeroComponent


class AeroVehicle:
    """
    Represents a complete aerodynamic vehicle composed of various FlightCanvas.
    Manages collective mesh generation and visualization.
    """

    def __init__(
            self,
            name: str,
            xyz_ref: Union[np.ndarray, List[float]],
            components: List[AeroComponent]
    ):
        """
        Initializes the AeroVehicle instance
        :param name: The name of the vehicle (e.g., "MyDrone")
        :param xyz_ref: The reference point [x, y, z] for the vehicle, typically the CG
        :param components: A list of AeroComponent instances that comprise the vehicle
        """
        self.name = name
        self.xyz_ref = np.array(xyz_ref)
        self.mass = 10
        self.moi = self.mass * np.eye(3)
        self.components = components
        self.num_components = len(components)

        self.vehicle_dynamics = None
        self.actuator_dynamics = None
        self.vehicle_path = f'vehicle_saves/{self.name}'

        for i in range(len(self.components)):
            self.components[i].update_id(i)

        [comp.set_parent(self) for comp in self.components]

        path_object = pathlib.Path(self.vehicle_path)
        path_object.mkdir(parents=True, exist_ok=True)

        # Crates a buildup manager for each component
        self.init_buildup_manager()

        # Update transformation matrices for all components
        self.update_transform()

    def update_transform(self):
        """
        Update transformation matrices for all components
        """
        [comp.update_transform() for comp in self.components]

    def set_mass(self, mass: float):
        """
        Sets the mass of the vehicle
        """
        self.mass = mass

    def set_moi_factor(self, moi_factor: float):
        """
        Sets the mass moment of inertia
        """
        self.moi = moi_factor * self.mass * np.eye(3)

    def set_moi_diag(self, moi_diag: Union[np.ndarray, List[float]]):
        """
        Sets the mass moment of inertia
        """
        self.moi = np.diag(np.array(moi_diag)) * self.mass

    def init_vehicle_dynamics(self, control_mapping: Union[None, dict]):
        """
        Creates dynamics in the form x_dot = f(x, u)
        """
        self.vehicle_dynamics = VehicleDynamics(self.mass, self.moi, self.components, control_mapping)

    def init_buildup_manager(self):
        """
        Crates a buildup manager for each component
        """
        for component in self.components:
            component.init_buildup_manager(self.vehicle_path, component)

    def compute_buildup(self):
        """
        Computes the aerodynamic buildup data for all 'prime' FlightCanvas
        """
        print("Computing buildup data...")
        for component in self.components:
            component.compute_buildup()

    def save_buildup(self):
        """
        Saves the aerodynamic buildup data for all 'prime' FlightCanvas
        """
        print("Saving buildup data...")
        for component in self.components:
            component.save_buildup()

    def save_buildup_fig(self):
        """
        Saves the aerodynamic buildup figures for all 'prime' FlightCanvas
        """
        print("Saving buildup figures...")
        for component in self.components:
            component.save_buildup_figs()

    def load_buildup(self):
        """
        Loads the aerodynamic buildup data for all 'prime' FlightCanvas
        """
        print('Loading buildup data...')
        for component in self.components:
            component.load_buildup()

    def generate_mesh(self):
        """
        Generate the mesh for all FlightCanvas and applies their local translation
        """
        for component in self.components:
            component.generate_mesh()

    def dynamics(self, state: np.ndarray, control: np.ndarray):
        """
        Wrapper for 6-Degree of freedom dynamics in vehicle dynamics class
        """
        if self.vehicle_dynamics.allocation_matrix is None:
            raise ValueError("Vehicle dynamics is not allocated")

        cmd_deflections = self.vehicle_dynamics.allocation_matrix @ control

        true_deflections = self.actuator_dynamics.update_deflections(cmd_deflections)

        return self.vehicle_dynamics.dynamics(state, true_deflections).full().flatten()

    def get_true_deflections(self):
        true_deflections = np.zeros(self.num_components)
        for i in range(self.num_components):
            true_deflections[i] = self.actuator_dynamics.tfs[i].y_hist[0]
        return true_deflections
