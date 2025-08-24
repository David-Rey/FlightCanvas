from dataclasses import dataclass

from FlightCanvas.components.aero_component import AeroComponent
from typing import List, Dict, Optional
import numpy as np
import casadi as ca

@dataclass
class ActuatorData:
    component_id: int
    component_name: str
    control_index: Optional[int]
    deflection_index: Optional[int]
    order: Optional[int]
    A: Optional[np.ndarray]
    B: Optional[np.ndarray]

class ActuatorDynamics:
    def __init__(self, components: List[AeroComponent], actuator_mapping: Dict[str, Dict[str, float]]):
        self.components = components
        self.num_components = len(components)
        self.num_control_inputs = len(actuator_mapping)
        self.actuator_mapping = actuator_mapping

        self.actuators: List[ActuatorData] = []
        self.num_actuator_inputs_comp = 0
        self.deflection_state_size = 0
        self.create_actuator_dataset()
        self.deflection_indices = [actuator.deflection_index for actuator in self.actuators]

        self.allocation_matrix = self.create_allocation_matrix()

        self.full_system_matrix = np.zeros((self.deflection_state_size, self.deflection_state_size))
        self.full_control_matrix = np.zeros((self.deflection_state_size, self.num_actuator_inputs_comp))

        self.populate_system()


    def get_dynamics(self, state: ca.MX, control: ca.MX) -> ca.MX:
        """
        Gets the actuator symbolic linear dynamics based
        :param state: casadi expression representing the deflection states
        :param control: casadi expression representing the control inputs
        :return: casadi expression representing rate of change of dynamics
        """
        A = ca.MX(self.full_system_matrix)
        B = ca.MX(self.full_control_matrix)
        H = ca.MX(self.allocation_matrix)
        return A @ state + B @ H @ control

    def get_component_deflection(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        TODO
        """
        true_deflections = np.zeros(self.num_components)

        for i in range(self.num_components):
            actuator = self.actuators[i]
            true_deflection = 0
            if actuator.deflection_index is not None:
                true_deflection = state[actuator.deflection_index]
            elif actuator.control_index is not None:
                true_deflection = control[actuator.control_index]
            true_deflections[i] = true_deflection

        return true_deflections



    def create_allocation_matrix(self) -> np.ndarray:
        """
        Creates the control allocation matrix for the vehicle.
        """
        # Get a sorted list of high-level command names for consistent column ordering.
        command_names = sorted(self.actuator_mapping.keys())
        command_to_col = {name: i for i, name in enumerate(command_names)}

        # Create a helper dictionary to quickly find an actuator's data by its name.
        actuator_name_to_data = {act.component_name: act for act in self.actuators}

        # Initialize the matrix. Rows correspond to the individual actuator inputs,
        # and columns correspond to the high-level control commands.
        allocation_matrix = np.zeros(
            (self.num_actuator_inputs_comp, self.num_control_inputs)
        )

        # Populate the matrix using the defined mapping.
        for command_name, component_map in self.actuator_mapping.items():
            # Get the column index for the current high-level command.
            col_idx = command_to_col[command_name]

            for actuator_name, gain in component_map.items():
                # Find the actuator's data (including its control_index) using its name.
                if actuator_name in actuator_name_to_data:
                    actuator_data = actuator_name_to_data[actuator_name]
                    row_idx = actuator_data.control_index

                    # Ensure the component is actually a controllable actuator.
                    if row_idx is not None:
                        # Place the gain at the correct row (actuator input) and column (command).
                        allocation_matrix[row_idx, col_idx] += gain
                else:
                    # Optional but recommended: A warning for names that don't match.
                    print(f"Warning: Actuator '{actuator_name}' in control mapping not found in components.")

        return allocation_matrix

    def create_actuator_dataset(self):
        """
        Creates the actuator dynamics dataset.
        """
        control_index_counter = 0
        deflection_index_counter = 0
        for i in range(self.num_components):
            component = self.components[i]
            component_id = component.id
            component_name = component.name

            control_index = None
            deflection_index = None
            actuator_order = None
            system_matrix = None
            control_matrix = None

            actuator_model = component.actuator_model
            if actuator_model is not None:
                actuator_order = component.actuator_model.state_size
                system_matrix = component.actuator_model.get_system_matrix()
                control_matrix = component.actuator_model.get_control_matrix()

                control_index = control_index_counter
                deflection_index = deflection_index_counter
                if actuator_order == 0:
                    deflection_index = None

                control_index_counter += 1
                deflection_index_counter += actuator_order

            actuator_data = ActuatorData(component_id, component_name, control_index, deflection_index, actuator_order, system_matrix, control_matrix)
            self.actuators.append(actuator_data)

        self.num_actuator_inputs_comp = control_index_counter
        self.deflection_state_size = deflection_index_counter


    def populate_system(self):
        """
        TODO
        """
        for actuator in self.actuators:
            order = actuator.order
            deflection_index = actuator.deflection_index
            if deflection_index is not None:
                control_index = actuator.control_index
                index_start = deflection_index
                index_end = deflection_index + order
                self.full_system_matrix[index_start:index_end, index_start:index_end] = actuator.A
                self.full_control_matrix[index_start:index_end, control_index] = actuator.B



if __name__ == "__main__":
    import aerosandbox as asb
    import aerosandbox.numpy as np

    from scipy.interpolate import splprep, splev

    from FlightCanvas.actuators import FirstOrderDeflection, SecondOrderDeflection
    from FlightCanvas.components.aero_fuselage import AeroFuselage
    from FlightCanvas.components.aero_wing import create_planar_wing_pair



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

        k = 19.5  # remove me (for testing)

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
        actuator_model=FirstOrderDeflection(time_constant=0.3)
        #actuator_model=FirstOrderDeflection(natural_frequency=10, damping_ratio=0.7)
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
        control_pivot=[1, 0, 0],
        actuator_model=SecondOrderDeflection(natural_frequency=10, damping_ratio=0.7)
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
            "Back Flap": 1.1,
            "Back Flap Star": 1.0
        }
    }

    control_mapping_2 = {
        "control 1": {
            "Front Flap": 1.0,
            "Front Flap Star": 1.0
        },
        "control 2": {
            "Back Flap": 1.0,
            "Back Flap Star": 1.0
        }
    }

    dyn = ActuatorDynamics(all_components, control_mapping_2)
    print(dyn)

    """
    def create_allocation_matrix_old(self) -> np.ndarray:

        # Get sorted lists of commands (for columns) and component names (for rows)
        command_names = self.actuator_mapping.keys()
        actuator_names = [comp.name for comp in self.components]

        # Create helper dictionaries to map names to matrix indices
        actuator_to_row = {name: i for i, name in enumerate(actuator_names)}
        command_to_col = {name: i for i, name in enumerate(command_names)}

        allocation_matrix = np.zeros((self.deflection_state_size, self.num_control_inputs))
        #input_index = 0
        # Populate the matrix with gains from the input dictionary
        for command, component_map in self.actuator_mapping.items():
            #col_idx = command_to_col[command]
            for actuator_name, gain in component_map.items():
                if actuator_name in actuator_to_row:
                    row_idx = actuator_to_row[actuator_name]
                    #deflection_index = self.actuators[row_idx].deflection_index
                    control_index = self.actuators[row_idx].control_index
                    #allocation_matrix[deflection_index, control_index] = gain
                    allocation_matrix[control_index, input_index] = gain
                    #input_index += 1
        return allocation_matrix



    def create_allocation_matrix_old(self) -> np.ndarray:
        # Get sorted lists of commands (for columns) and component names (for rows)
        command_names = self.actuator_mapping.keys()
        actuator_names = [comp.name for comp in self.components]

        # Create helper dictionaries to map names to matrix indices
        actuator_to_row = {name: i for i, name in enumerate(actuator_names)}
        command_to_col = {name: i for i, name in enumerate(command_names)}

        # Initialize a zero matrix with the correct dimensions
        num_actuators = len(actuator_names)
        num_commands = len(command_names)
        allocation_matrix = np.zeros((num_actuators, num_commands))

        # Populate the matrix with gains from the input dictionary
        for command, component_map in self.actuator_mapping.items():
            col_idx = command_to_col[command]
            for actuator_name, gain in component_map.items():
                if actuator_name in actuator_to_row:
                    row_idx = actuator_to_row[actuator_name]
                    allocation_matrix[row_idx, col_idx] = gain

        return allocation_matrix
    """


