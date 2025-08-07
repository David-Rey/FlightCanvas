# aero_project/FlightCanvas/buildup_manager.py

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


class BuildupManager:
    def __init__(self, name: str,
                 vehicle_path: str,
                 alpha_grid_size: int = 150,
                 beta_grid_size: int = 100,
                 operating_velocity: float = 10.0,
                 include_Fx: bool = True,
                 include_Fy: bool = True,
                 include_Fz: bool = True,
                 include_Mx: bool = True,
                 include_My: bool = True,
                 include_Mz: bool = True):

        self.name = name
        self.vehicle_path = vehicle_path

        self.alpha_grid_size = alpha_grid_size
        self.beta_grid_size = beta_grid_size
        self.operating_velocity = operating_velocity

        self.include_Fx = include_Fx
        self.include_Fy = include_Fy
        self.include_Fz = include_Fz
        self.include_Mx = include_Mx
        self.include_My = include_My
        self.include_Mz = include_Mz
        self.include_arr = [include_Fx, include_Fy, include_Fz, include_Mx, include_My, include_Mz]

        self.alpha_grid = None  # Hold grid of an angle of attack for buildup
        self.beta_grid = None  # Hold grid of sideslip angles for buildup
        self.asb_data = None  # To hold the aero build data

    def get_forces_and_moments(self, alpha: float, beta: float, velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        TODO
        """
        scale_factor = (np.linalg.norm(velocity) / self.operating_velocity) ** 2

        # Get the alpha and beta axes from the pre-computed grid (in radians)
        alpha_lin_rad = np.deg2rad(self.alpha_grid[:, 0])
        beta_lin_rad = np.deg2rad(self.beta_grid[0, :])

        # Reshape force data to be compatible with interpolation
        F_b_data = np.column_stack(self.asb_data["F_b"]).reshape(self.alpha_grid.shape + (3,))
        M_b_data = np.column_stack(self.asb_data["M_b"]).reshape(self.alpha_grid.shape + (3,))

        # Perform linear interpolation to find the force at the current alpha/beta
        F_b = utils.linear_interpolation(alpha_lin_rad, beta_lin_rad, alpha, beta, F_b_data)
        M_b = utils.linear_interpolation(alpha_lin_rad, beta_lin_rad, alpha, beta, M_b_data)

        # Set elements to zero if they are not included
        F_b = np.where(self.include_arr[:3], F_b, 0) * scale_factor
        M_b = np.where(self.include_arr[3:], M_b, 0) * scale_factor
        return F_b, M_b

    def compute_buildup(self, asb_airplane: asb.Airplane):
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
        self.asb_data = asb.AeroBuildup(
            airplane=asb_airplane,
            op_point=op_point
        ).run()

    def save_buildup(self):

        # Put variables in a dictionary for easy loading
        variables_to_save = {
            'name': self.name,
            'alpha_grid_size': self.alpha_grid_size,
            'beta_grid_size': self.beta_grid_size,
            'operating_velocity': self.operating_velocity,
            'asb_data': self.asb_data,
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
        self.asb_data = loaded_data['asb_data']

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
        data = self.asb_data[ID][index]
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

    def compute_symbolic_lookup(self) -> Tuple[ca.Function, ca.Function]:
        """
        Creates CasADi functions for looking up aerodynamic forces and moments
        """
        if self.asb_data is None:
            raise RuntimeError("Aero data not available. Run compute_buildup() or load_buildup() first.")

        alpha_sym = ca.MX.sym('alpha', 1)  # Angle of attack [rad]
        beta_sym = ca.MX.sym('beta', 1)  # Sideslip angle [rad]
        velocity_sym = ca.MX.sym('velocity', 3)  # Velocity vector [m/s]

        # Get the grid axes in radians, which is required for the interpolant
        alpha_lin_rad = np.deg2rad(self.alpha_grid[:, 0])
        beta_lin_rad = np.deg2rad(self.beta_grid[0, :])
        grid_axes = [alpha_lin_rad, beta_lin_rad]

        # Reshape the force and moment data to match the grid dimensions
        # The required shape is (n_alpha, n_beta, n_dims), where n_dims is 3 for a 3D vector.
        F_b_data_grid = np.column_stack(self.asb_data["F_b"]).reshape(self.alpha_grid.shape + (3,))
        M_b_data_grid = np.column_stack(self.asb_data["M_b"]).reshape(self.alpha_grid.shape + (3,))

        # The interpolant requires the data to be flattened. CasADi handles the C-style unraveling internally.
        F_b_interpolant = ca.interpolant('F_b_interp', 'bspline', grid_axes, F_b_data_grid.flatten())
        M_b_interpolant = ca.interpolant('M_b_interp', 'bspline', grid_axes, M_b_data_grid.flatten())

        # The input to the interpolant function must be a stacked vector of the grid variables
        interp_input = ca.vertcat(alpha_sym, beta_sym)

        # Perform the symbolic lookup
        F_b_base = F_b_interpolant(interp_input)
        M_b_base = M_b_interpolant(interp_input)

        # The output of a vector interpolant is flattened, so we must reshape it
        F_b_base = ca.reshape(F_b_base, 3, 1)
        M_b_base = ca.reshape(M_b_base, 3, 1)

        # Forces and moments scale with velocity squared
        scale_factor = (ca.norm_2(velocity_sym) / self.operating_velocity) ** 2

        # Use the boolean flags to selectively include/exclude components
        include_F = ca.DM(self.include_arr[:3]).T  # Transpose to enable element-wise multiplication
        include_M = ca.DM(self.include_arr[3:]).T

        F_b_sym = F_b_base * include_F * scale_factor
        M_b_sym = M_b_base * include_M * scale_factor

        inputs = [alpha_sym, beta_sym, velocity_sym]
        input_names = ['alpha', 'beta', 'velocity']

        # Create symbolic force function
        F_b_func = ca.Function(
            f'{self.name}_forces',
            inputs,
            [F_b_sym],
            input_names,
            ['F_b']
        )

        # Create symbolic moment function
        M_b_func = ca.Function(
            f'{self.name}_moments',
            inputs,
            [M_b_sym],
            input_names,
            ['M_b']
        )

        return F_b_func, M_b_func

    def compute_symbolic(self):
        pass

    def save_symbolic(self):
        pass

    def load_symbolic(self):
        pass

    def save_symbolic_figs(self, show=False):
        pass

    def draw_symbolic_figs(self, ID: str, index=None):
        pass

    def draw_error(self):
        pass
