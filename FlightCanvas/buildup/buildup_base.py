from abc import ABC, abstractmethod
import numpy as np
import os
import pathlib
import pickle

import aerosandbox as asb
from aerosandbox import OperatingPoint, Atmosphere
import aerosandbox.tools.pretty_plots as p

from typing import Union, Tuple, List, Dict, Any, Optional
import casadi as ca
from matplotlib import pyplot as plt
import scipy.ndimage


class BuildupBase(ABC):
    """
    Manages the aerodynamic buildup process for a vehicle component, handling both static and damping aero-data
    """

    def __init__(
            self,
            name: str,
            vehicle_path: str,
            aero_component: ['AeroComponent'],
            coefficients_config: Optional[Dict[str, Any]] = None,
            alpha_grid_size: int = 150,
            beta_grid_size: int = 100,
            operating_velocity: float = 50,
            compute_damping: bool = False,
            smooth_data: bool = True,
    ):
        """
        Initialize the BuildupManager object
        :param name: Name of the component
        :param vehicle_path: Path to the vehicle file buildup
        :param aero_component: AeroComponent object
        :param coefficients_config: Dictionary defining coefficients to extract and their properties
        :param alpha_grid_size: Grid size for angle of attack
        :param beta_grid_size: Grid size for angle of sideslip
        :param operating_velocity: Operating mach number
        :param compute_damping: Whether to compute damping
        :param smooth_data: Whether to smooth data aerodynamic data to avoid discontinuity
        """

        self.name = name
        self.vehicle_path = vehicle_path
        self.aero_component = aero_component
        self.compute_damping = compute_damping
        self.smooth_data = smooth_data

        self.alpha_grid_size = alpha_grid_size
        self.beta_grid_size = beta_grid_size
        self.operating_velocity = operating_velocity

        # Set up default coefficients configuration if none provided
        if coefficients_config is None:
            self.coefficients_config = self._get_default_coefficients_config()
        else:
            self.coefficients_config = coefficients_config

        self.asb_atm = Atmosphere()
        self.sos = self.asb_atm.speed_of_sound()

        # Create a meshgrid of alpha and beta values to analyze
        # Hold grid of an angle of attack and sideslip for buildup
        self.beta_grid, self.alpha_grid = np.meshgrid(
            np.linspace(-90, 90, self.beta_grid_size),
            np.linspace(-180, 180, self.alpha_grid_size)
        )
        # Define the operating points for the analysis
        self.op_point = asb.OperatingPoint(
            velocity=self.operating_velocity,
            alpha=self.alpha_grid.flatten(),
            beta=self.beta_grid.flatten()
        )

        self.asb_data_static = None  # To hold the aero build data
        self.asb_data_damping = None
        self.static_interpolants = None  # To hold the CasADi interpolant object
        self.damping_interpolant = None
        self.stacked_coeffs_data_static = None  # To hold the pre-processed NumPy data
        self.stacked_coeffs_data_damping = None

    @abstractmethod
    def _get_default_coefficients_config(self) -> Dict[str, Any]:
        """
        Returns the default coefficients configuration.
        Must be implemented by subclasses to define their specific coefficient structure.
        """
        pass

    def _extract_coefficient_data(self, data_dict: Dict, coeff_name: str, coeff_config: Dict) -> np.ndarray:
        """
        Extract coefficient data from asb_data, handling both simple keys and indexed tuples
        :param data_dict: The data dictionary (e.g., asb_data_static)
        :param coeff_name: Name of the coefficient
        :param coeff_config: Configuration for this coefficient
        :return: Extracted data as numpy array
        """
        key = coeff_config['key']
        index = coeff_config.get('index', None)

        if key not in data_dict:
            raise KeyError(f"Key '{key}' not found in data dictionary")

        data = data_dict[key]

        # Handle indexed access (e.g., for tuples like Fb[0])
        if index is not None:
            if isinstance(data, (list, tuple)):
                if index >= len(data):
                    raise IndexError(f"Index {index} out of range for {key} with length {len(data)}")
                data = data[index]
            else:
                raise TypeError(f"Cannot index into {key} - it's not a list or tuple")

        return np.array(data)

    @abstractmethod
    def get_forces_and_moments(
            self,
            alpha: Union[float, ca.MX],
            beta: Union[float, ca.MX],
            speed: Union[float, ca.MX],
            p: Union[float, ca.MX],
            q: Union[float, ca.MX],
            r: Union[float, ca.MX]
    ) -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
        """
        Calculates aerodynamic forces and moments using either NumPy or CasADi.
        :param alpha: Angle of attack (rad)
        :param beta: Sideslip angle (rad)
        :param speed: Airspeed (m/s)
        :param p: Body-axis roll rate (rad/s)
        :param q: Body-axis pitch rate (rad/s)
        :param r: Body-axis yaw rate (rad/s)
        :return: A tuple containing the force and moment vectors in the body frame.
        """
        pass

    def _pre_process_static_data(self):
        """
        Reshapes and stacks the raw coefficient data into a single 3D NumPy array
        for efficient interpolation. This should only be called once.
        """
        if self.asb_data_static is None:
            raise RuntimeError("Aero data not available. Run compute_buildup() or load_buildup() first.")

        static_coeffs = self.coefficients_config['static_coeffs']
        coeff_data_list = []

        for coeff_name, coeff_config in static_coeffs.items():
            # Extract the coefficient data
            data = self._extract_coefficient_data(self.asb_data_static, coeff_name, coeff_config)

            # Reshape into 2D grid
            if len(data.shape) == 1:
                data_2d = np.column_stack(data).reshape(self.alpha_grid.shape)
            else:
                data_2d = data.reshape(self.alpha_grid.shape)

            coeff_data_list.append(data_2d)

        # Stack all coefficient data into a single 3D array
        self.stacked_coeffs_data_static = np.stack(coeff_data_list, axis=-1)

    def _pre_process_damping_data(self):
        """
        Pre-processes damping data for NumPy interpolation
        """
        if self.asb_data_damping is None:
            raise RuntimeError("Damping data not available.")

        damping_coeffs = self.coefficients_config['damping_coeffs']
        damping_data_list = []

        for coeff_name, coeff_config in damping_coeffs.items():
            data = self._extract_coefficient_data(self.asb_data_damping, coeff_name, coeff_config)
            damping_data_list.append(data)

        self.stacked_coeffs_data_damping = np.column_stack(damping_data_list)

    def _create_static_interpolants(self):
        """
        Private helper method to create and cache the CasADi interpolant objects
        """
        if self.stacked_coeffs_data_static is None:
            self._pre_process_static_data()

        # Get grid points in radians
        alpha_lin_rad = np.deg2rad(self.alpha_grid[:, 0])
        beta_lin_rad = np.deg2rad(self.beta_grid[0, :])
        grid_axes = [alpha_lin_rad, beta_lin_rad]

        coeffs_data_grid_transposed = np.transpose(self.stacked_coeffs_data_static, axes=(1, 0, 2))

        # Flatten the pre-stacked data for the interpolant.
        coeffs_data_flat = coeffs_data_grid_transposed.ravel(order='C')

        # Create a single interpolant for all coefficients
        sanitized_name = self.name.replace(" ", "_")
        self.static_interpolants = ca.interpolant(
            f'{sanitized_name}_CoeffsLookup',
            'bspline',  # bspline (or) linear
            grid_axes,
            coeffs_data_flat
        )

    def _create_damping_interpolant(self):
        """
        Private helper method to create and cache the CasADi interpolant for damping
        """
        if self.stacked_coeffs_data_damping is None:
            self._pre_process_damping_data()

        alpha_sweep_rad = np.deg2rad(self.alpha_sweep_1D)
        sanitized_name = self.name.replace(" ", "_")

        # Convert NumPy arrays to the list format that CasADi expects
        grid_list = [alpha_sweep_rad.tolist()]
        data_list = self.stacked_coeffs_data_damping.T.ravel().tolist()

        self.damping_interpolant = ca.interpolant(
            f'{sanitized_name}_DampingLookup',
            'linear',
            grid_list,
            data_list
        )

    def compute_buildup(self):
        """
        Computes the aerodynamic buildup data for the component over a range
        of alpha and beta angles at a specified velocity
        """
        # Run the AeroBuildup analysis
        self.asb_data_static = asb.AeroBuildup(
            airplane=self.aero_component.asb_airplane,
            op_point=self.op_point
        ).run()

        if self.compute_damping:
            self.asb_data_damping = asb.AeroBuildup(
                airplane=self.aero_component.asb_airplane,
                op_point=self.op_point_1D_alpha_sweep
            ).run_with_stability_derivatives(p=True, q=True, r=True)

        if self.smooth_data:
            self._apply_smoothing()

    def _apply_smoothing(self):
        """
        Apply Gaussian smoothing to specified coefficients
        """
        sigma_to_tune = 4
        static_coeffs = self.coefficients_config['static_coeffs']

        for coeff_name, coeff_config in static_coeffs.items():
            if coeff_config.get('smooth', False):
                key = coeff_config['key']
                index = coeff_config.get('index', None)

                if key in self.asb_data_static:
                    if index is not None:
                        # Handle tuple/list case
                        data_list = list(self.asb_data_static[key])
                        data_list[index] = scipy.ndimage.gaussian_filter(
                            data_list[index], sigma=sigma_to_tune
                        )
                        self.asb_data_static[key] = tuple(data_list) if isinstance(
                            self.asb_data_static[key], tuple
                        ) else data_list
                    else:
                        # Handle simple case
                        self.asb_data_static[key] = scipy.ndimage.gaussian_filter(
                            self.asb_data_static[key], sigma=sigma_to_tune
                        )

    def save_buildup(self):
        """
        Saves buildup data to buildup folder
        """
        # Put variables in a dictionary for easy loading
        variables_to_save = {
            'name': self.name,
            'coefficients_config': self.coefficients_config,
            'alpha_grid_size': self.alpha_grid_size,
            'beta_grid_size': self.beta_grid_size,
            'operating_velocity': self.operating_velocity,
            'asb_data_static': self.asb_data_static,
            'asb_data_damping': self.asb_data_damping,
        }

        folder_path = os.path.join(self.vehicle_path, 'buildup_data')
        path_object = pathlib.Path(folder_path)
        path_object.mkdir(parents=True, exist_ok=True)
        file_name = f"{self.name}.pkl"
        full_path = os.path.join(folder_path, file_name)

        # Open the file in binary write mode ('wb')
        with open(full_path, 'wb') as file:
            pickle.dump(variables_to_save, file)

    def load_buildup(self):
        """
        Loads buildup data from buildup folder
        """
        folder_path = os.path.join(self.vehicle_path, 'buildup_data')
        file_name = f"{self.name}.pkl"
        full_path = os.path.join(folder_path, file_name)

        # Check if the file exists before trying to open it
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Could not find the buildup file at: {full_path}")

        # Open the file in binary read mode ('rb') and load the data
        with open(full_path, 'rb') as file:
            loaded_data = pickle.load(file)

        # set variables form the loaded file to current object
        self.name = loaded_data['name']
        self.coefficients_config = loaded_data.get('coefficients_config', self._get_default_coefficients_config())
        self.alpha_grid_size = loaded_data['alpha_grid_size']
        self.beta_grid_size = loaded_data['beta_grid_size']
        self.operating_velocity = loaded_data['operating_velocity']
        self.asb_data_static = loaded_data['asb_data_static']
        self.asb_data_damping = loaded_data['asb_data_damping']

        self.beta_grid, self.alpha_grid = np.meshgrid(
            np.linspace(-90, 90, self.beta_grid_size),
            np.linspace(-180, 180, self.alpha_grid_size)
        )

    def save_buildup_figs(self):
        """
        Draws a contour plot of a specified aerodynamic coefficient from the
        buildup data
        """
        static_coeffs = self.coefficients_config['static_coeffs']

        folder_path = os.path.join(self.vehicle_path, 'buildup_figs', self.name)
        path_object = pathlib.Path(folder_path)
        path_object.mkdir(parents=True, exist_ok=True)

        # Generate plots for all configured static coefficients
        for coeff_name, coeff_config in static_coeffs.items():
            display_name = coeff_config.get('display_name', coeff_name)
            self.draw_buildup_figs(coeff_name, display_name, folder_path)

    def draw_buildup_figs(self, coeff_name: str, display_name: str, folder_path: str):
        """
        Saves buildup figure which a function of angle of attack and sideslip for aerodynamic coefficient
        :param coeff_name: Internal name of the aerodynamic coefficient
        :param display_name: Display name for the coefficient (for filename and labels)
        :param folder_path: Path to the buildup figure
        """
        coeff_config = self.coefficients_config['static_coeffs'][coeff_name]
        data = self._extract_coefficient_data(self.asb_data_static, coeff_name, coeff_config)

        title = f"`{self.name}` {display_name}"
        file_name = f"{self.name}_{display_name}.png"
        full_path = os.path.join(folder_path, file_name)

        # Create the contour plot
        plt.figure()
        p.contour(
            self.beta_grid, self.alpha_grid, data.reshape(self.alpha_grid.shape),
            colorbar_label=f"${coeff_name}$ [-]",
            linelabels_format=lambda x: f"{x:.2f}",
            linelabels_fontsize=7,
            cmap="RdBu",
            alpha=0.6
        )
        p.set_ticks(15, 5, 15, 5)
        plt.clim(*np.array([-1, 1]) * np.max(np.abs(data)))
        p.show_plot(
            title,
            r"Sideslip angle $\beta$ [deg]",
            r"Angle of Attack $\alpha$ [deg]",
            set_ticks=False,
            savefig=full_path,
            show=False
        )

        plt.close()

    def get_coefficient_names(self) -> List[str]:
        """
        Returns list of all configured coefficient names
        """
        static_names = list(self.coefficients_config['static_coeffs'].keys())
        damping_names = list(self.coefficients_config['damping_coeffs'].keys()) if self.compute_damping else []
        return static_names + damping_names

    def add_coefficient(self, coeff_name: str, key: str, index: Optional[int] = None,
                        smooth: bool = False, display_name: Optional[str] = None,
                        is_damping: bool = False):
        """
        Add a new coefficient to the configuration
        :param coeff_name: Internal name for the coefficient
        :param key: Key in the asb_data dictionary
        :param index: Index for tuple/list data (e.g., 0 for Fb[0])
        :param smooth: Whether to apply smoothing to this coefficient
        :param display_name: Display name for plots and files
        :param is_damping: Whether this is a damping coefficient
        """
        config = {
            'key': key,
            'index': index,
            'smooth': smooth,
            'display_name': display_name or coeff_name
        }

        if is_damping:
            self.coefficients_config['damping_coeffs'][coeff_name] = config
        else:
            self.coefficients_config['static_coeffs'][coeff_name] = config

    @staticmethod
    def _convert_axes_np(alpha: float, beta: float, F_w: np.ndarray) -> np.ndarray:
        """
        Convert axis from wind to body
        :param alpha: Angle of attack in radians
        :param beta: Angle of sideslip in radians
        :param F_w: Vector in the wind frame
        """
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)

        # Rotation matrix from wind to body axes
        R_b_w = np.array([
            [ca * cb, -ca * sb, -sa],
            [sb, cb, 0],
            [sa * cb, -sa * sb, ca]
        ])

        F_b = R_b_w @ F_w
        return F_b

    @staticmethod
    def _convert_axes_ca(alpha: ca.MX, beta: ca.MX, F_w: ca.MX) -> ca.MX:
        """
        Convert axis from wind to body
        :param alpha: Angle of attack in radians
        :param beta: Angle of sideslip in radians
        :param F_w: Vector in the wind frame
        """
        ca_ = ca.cos(alpha)
        sa = ca.sin(alpha)
        cb = ca.cos(beta)
        sb = ca.sin(beta)

        # Rotation matrix from wind to body axes
        R_b_w = ca.vertcat(
            ca.horzcat(ca_ * cb, -ca_ * sb, -sa),
            ca.horzcat(sb, cb, 0),
            ca.horzcat(sa * cb, -sa * sb, ca_)
        )

        F_b = R_b_w @ F_w
        return F_b