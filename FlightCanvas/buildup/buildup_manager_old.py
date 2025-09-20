import aerosandbox as asb
import numpy as np
from matplotlib import pyplot as plt
import aerosandbox.tools.pretty_plots as p
import os
import pathlib
import pickle
from FlightCanvas import utils
from typing import Tuple
import casadi as ca
from typing import Union


class BuildupManagerOld:
    def __init__(self, name: str,
                 vehicle_path: str,
                 aero_component: ['AeroComponent'],
                 alpha_grid_size: int = 150,
                 beta_grid_size: int = 100,
                 operating_velocity: float = 50.0,
                 include_Fx: bool = True,
                 include_Fy: bool = True,
                 include_Fz: bool = True,
                 include_Mx: bool = True,
                 include_My: bool = True,
                 include_Mz: bool = True):

        self.name = name
        self.vehicle_path = vehicle_path
        self.aero_component = aero_component

        self.alpha_grid_size = alpha_grid_size
        self.beta_grid_size = beta_grid_size
        self.operating_velocity = operating_velocity

        self.include_Fx = include_Fx
        self.include_Fy = include_Fy
        self.include_Fz = include_Fz
        self.include_Mx = include_Mx
        self.include_My = include_My
        self.include_Mz = include_Mz
        self.include_arr = np.array([include_Fx, include_Fy, include_Fz, include_Mx, include_My, include_Mz])

        self.alpha_grid = None  # Hold grid of an angle of attack for buildup
        self.beta_grid = None  # Hold grid of sideslip angles for buildup
        self.asb_data_static = None  # To hold the aero build data
        self.aero_interpolants = None  # To hold the CasADi interpolant object

    def get_forces_and_moments(self, alpha: Union[float, ca.MX], beta: Union[float, ca.MX], speed: Union[float, ca.MX],
                               p, q, r) -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
        """
        Computes forces and moments. This function is type-aware and will use
        either NumPy or CasADi based on the input type.
        """
        is_casadi = isinstance(speed, (ca.SX, ca.MX))

        if is_casadi:
            return self._get_forces_and_moments_ca(alpha, beta, speed)
        else:
            return self._get_forces_and_moments_np(alpha, beta, speed)

    def _get_forces_and_moments_np(self, alpha: float, beta: float, speed: float) -> tuple[np.ndarray, np.ndarray]:
        """
        TODO
        """
        scale_factor = (speed / self.operating_velocity) ** 2

        # Get the alpha and beta axes from the pre-computed grid (in radians)
        alpha_lin_rad = np.deg2rad(self.alpha_grid[:, 0])
        beta_lin_rad = np.deg2rad(self.beta_grid[0, :])

        # Reshape force data to be compatible with interpolation
        F_b_data = np.column_stack(self.asb_data_static["F_b"]).reshape(self.alpha_grid.shape + (3,))
        M_b_data = np.column_stack(self.asb_data_static["M_b"]).reshape(self.alpha_grid.shape + (3,))

        # Perform linear interpolation to find the force at the current alpha/beta
        F_b = utils.linear_interpolation(alpha_lin_rad, beta_lin_rad, alpha, beta, F_b_data)
        M_b = utils.linear_interpolation(alpha_lin_rad, beta_lin_rad, alpha, beta, M_b_data)

        # Set elements to zero if they are not included
        F_b = np.where(self.include_arr[:3], F_b, 0) * scale_factor
        M_b = np.where(self.include_arr[3:], M_b, 0) * scale_factor
        return F_b, M_b

    def _create_aero_interpolants(self):
        """
        Private helper method to create and cache the CasADi interpolant objects.
        This is a one-time setup operation.
        """
        # 1. Get grid points in radians
        alpha_lin_rad = np.deg2rad(self.alpha_grid[:, 0])
        beta_lin_rad = np.deg2rad(self.beta_grid[0, :])
        grid_axes = [alpha_lin_rad, beta_lin_rad]

        # 2. Reshape data for forces and moments separately
        F_b_data_grid = np.column_stack(self.asb_data_static["F_b"]).reshape(self.alpha_grid.shape + (3,))
        M_b_data_grid = np.column_stack(self.asb_data_static["M_b"]).reshape(self.alpha_grid.shape + (3,))

        F_b_data_grid_transposed = np.transpose(F_b_data_grid, axes=(1, 0, 2))
        M_b_data_grid_transposed = np.transpose(M_b_data_grid, axes=(1, 0, 2))

        # Flatten the data for each interpolant in column-major ('F') order.
        F_b_data_flat = F_b_data_grid_transposed.ravel(order='C')
        M_b_data_flat = M_b_data_grid_transposed.ravel(order='C')

        # 3. Create two separate interpolants, mirroring the NumPy function's logic
        sanitized_name = self.name.replace(" ", "_")
        F_b_interpolant = ca.interpolant(
            f'{sanitized_name}_ForcesLookup',
            'linear',
            grid_axes,
            F_b_data_flat
        )
        M_b_interpolant = ca.interpolant(
            f'{sanitized_name}_MomentsLookup',
            'linear',
            grid_axes,
            M_b_data_flat
        )

        # Store both interpolants
        self.aero_interpolants = (F_b_interpolant, M_b_interpolant)

    def _get_forces_and_moments_ca(self, alpha: ca.MX, beta: ca.MX, speed: ca.MX) -> tuple[ca.MX, ca.MX]:
        """
        Computes aerodynamic forces and moments using pre-computed CasADi interpolants.
        """
        if self.asb_data_static is None:
            raise RuntimeError("Aero data not available. Run compute_buildup() or load_buildup() first.")

        # If the interpolants haven't been created yet, create them now.
        if self.aero_interpolants is None:
            self._create_aero_interpolants()

        # Unpack the force and moment interpolant functions
        F_b_interp_func, M_b_interp_func = self.aero_interpolants

        # --- Perform Symbolic Lookup ---
        scale_factor = (speed / self.operating_velocity) ** 2
        interp_input = ca.vcat([alpha, beta])

        # Perform the lookups separately
        F_b_interp_flat = F_b_interp_func(interp_input)
        M_b_interp_flat = M_b_interp_func(interp_input)

        # The output of a vector interpolant is a flat list, so reshape to a 3x1 column vector
        F_b_interp = ca.reshape(F_b_interp_flat, 3, 1)
        M_b_interp = ca.reshape(M_b_interp_flat, 3, 1)

        # Apply the include_arr mask and scale factor.
        F_b = F_b_interp * self.include_arr[:3].reshape(-1, 1) * scale_factor
        M_b = M_b_interp * self.include_arr[3:].reshape(-1, 1) * scale_factor

        return F_b, M_b

    def compute_buildup(self):
        """
        Computes the aerodynamic buildup data for the component over a range
        of alpha and beta angles at a specified velocity
        """
        # Create a meshgrid of alpha and beta values to analyze
        self.beta_grid, self.alpha_grid = np.meshgrid(
            np.linspace(-90, 90, self.beta_grid_size),
            np.linspace(-180, 180, self.alpha_grid_size)
        )
        # Define the operating points for the analysis
        op_point = asb.OperatingPoint(
            velocity=self.operating_velocity,
            alpha=self.alpha_grid.flatten(),
            beta=self.beta_grid.flatten()
        )
        # Run the AeroBuildup analysis
        self.asb_data_static = asb.AeroBuildup(
            airplane=self.aero_component.asb_airplane,
            op_point=op_point
        ).run()

    def save_buildup(self):

        # Put variables in a dictionary for easy loading
        variables_to_save = {
            'name': self.name,
            'alpha_grid_size': self.alpha_grid_size,
            'beta_grid_size': self.beta_grid_size,
            'operating_velocity': self.operating_velocity,
            'asb_data_static': self.asb_data_static,
        }

        folder_path = os.path.join(self.vehicle_path, 'aero_data')
        path_object = pathlib.Path(folder_path)
        path_object.mkdir(parents=True, exist_ok=True)
        file_name = f"{self.name}.pkl"
        full_path = os.path.join(folder_path, file_name)

        # Open the file in binary write mode ('wb')
        with open(full_path, 'wb') as file:
            pickle.dump(variables_to_save, file)

    def load_buildup(self):
        folder_path = os.path.join(self.vehicle_path, 'aero_data')
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
        self.alpha_grid_size = loaded_data['alpha_grid_size']
        self.beta_grid_size = loaded_data['beta_grid_size']
        self.operating_velocity = loaded_data['operating_velocity']
        self.asb_data_static = loaded_data['asb_data_static']

        self.beta_grid, self.alpha_grid = np.meshgrid(
            np.linspace(-90, 90, self.beta_grid_size),
            np.linspace(-180, 180, self.alpha_grid_size)
        )

    def save_buildup_figs(self):
        """
        Draws a contour plot of a specified aerodynamic coefficient from the
        buildup data
        """
        list_names = ["F_b", "M_b"]
        index_data = [0, 1, 2]

        folder_path = os.path.join(self.vehicle_path, 'buildup', self.name)
        path_object = pathlib.Path(folder_path)
        path_object.mkdir(parents=True, exist_ok=True)

        # ID is the identifier for the data to plot (e.g., "CL", "CD", "F_b")
        # The index if the data is a vector (e.g., 0 for Fx in F_b)
        for ID in list_names:
            # Inner loop for indices
            for index in index_data:
                # Data from the buildup
                self.draw_buildup_figs(ID, index, folder_path)

    def draw_buildup_figs(self, ID: str, index: int, folder_path: str):
        data = self.asb_data_static[ID][index]
        title = f"`{self.name}` {ID} [{index}]"
        file_name = f"{self.name}_{ID}_{index}.png"
        full_path = os.path.join(folder_path, file_name)

        # Create the contour plot
        plt.figure()
        p.contour(
            self.beta_grid, self.alpha_grid, data.reshape(self.alpha_grid.shape),
            colorbar_label=f"${ID}$ [-]",
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
