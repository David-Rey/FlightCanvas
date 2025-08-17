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
from typing import Union


class BuildupManager:
    def __init__(self, name: str,
        vehicle_path: str,
        aero_component: ['AeroComponent'],
        alpha_grid_size: int = 250,
        beta_grid_size: int = 200,
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

        self.include_Fx = include_Fx
        self.include_Fy = include_Fy
        self.include_Fz = include_Fz
        self.include_Mx = include_Mx
        self.include_My = include_My
        self.include_Mz = include_Mz
        self.include_arr = np.array([include_Fx, include_Fy, include_Fz, include_Mx, include_My, include_Mz])

        self.asb_data = None  # To hold the aero build data
        self.aero_interpolants = None  # To hold the CasADi interpolant object
        self.stacked_coeffs_data = None  # To hold the pre-processed NumPy data

    def get_forces_and_moments(self, alpha: Union[float, ca.MX], beta: Union[float, ca.MX], speed: Union[float, ca.MX]) -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
        """
        Computes forces and moments. This function is type-aware and will use
        either NumPy or CasADi based on the input type.
        """
        is_casadi = isinstance(speed, (ca.SX, ca.MX))

        if is_casadi:
            return self._get_forces_and_moments_ca(alpha, beta, speed)
        else:
            return self._get_forces_and_moments_np(alpha, beta, speed)




    def _get_forces_and_moments_np(self, alpha: float, beta: float, speed: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        TODO
        """
        if self.stacked_coeffs_data is None:
            self._pre_process_data()

        # Get the alpha and beta axes from the pre-computed grid (in radians)
        alpha_lin_rad = np.deg2rad(self.alpha_grid[:, 0])
        beta_lin_rad = np.deg2rad(self.beta_grid[0, :])

        # Perform a single, fast interpolation on the pre-stacked data
        interpolated_coeffs = utils.linear_interpolation(
            alpha_lin_rad, beta_lin_rad, alpha, beta, self.stacked_coeffs_data
        )

        # Unpack the results.
        CL, CY, CD, Cl, Cm, Cn = interpolated_coeffs

        density = 1.225
        dynamic_pressure = 0.5 * density * speed ** 2

        s = self.aero_component.asb_airplane.s_ref
        c = self.aero_component.asb_airplane.c_ref
        b = self.aero_component.asb_airplane.b_ref
        qS = dynamic_pressure * s

        L = -CL * qS  # Lift
        Y = CY * qS   # Side Force
        D = -CD * qS  # Drag
        F_w = np.array([D, Y, L])

        l_b = Cl * qS * b  # rolling moment
        m_b = Cm * qS * c  # pitching moment
        n_b = Cn * qS * b  # yawing moment
        M_b = np.array([l_b, m_b, n_b])
        F_b = self._convert_axes_np(alpha, beta, F_w)

        return F_b, M_b

    def _pre_process_data(self):
        """
        Reshapes and stacks the raw coefficient data into a single 3D NumPy array
        for efficient interpolation. This should only be called once.
        """
        if self.asb_data is None:
            raise RuntimeError("Aero data not available. Run compute_buildup() or load_buildup() first.")

        # Reshape each coefficient's data into a 2D grid
        CL_data = np.column_stack(self.asb_data["CL"]).reshape(self.alpha_grid.shape)
        CY_data = np.column_stack(self.asb_data["CY"]).reshape(self.alpha_grid.shape)
        CD_data = np.column_stack(self.asb_data["CD"]).reshape(self.alpha_grid.shape)
        Cl_data = np.column_stack(self.asb_data["Cl"]).reshape(self.alpha_grid.shape)
        Cm_data = np.column_stack(self.asb_data["Cm"]).reshape(self.alpha_grid.shape)
        Cn_data = np.column_stack(self.asb_data["Cn"]).reshape(self.alpha_grid.shape)

        # Stack the 2D grids into a single 3D data array and store it
        self.stacked_coeffs_data = np.stack(
            [CL_data, CY_data, CD_data, Cl_data, Cm_data, Cn_data],
            axis=-1
        )

    @staticmethod
    def _convert_axes_np(alpha: float, beta: float, F_w: np.ndarray) -> np.ndarray:
        """
        Convert axis from wind to body
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
    def _convert_axes_ca(alpha, beta, F_w):
        """
        Convert axis from wind to body
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


    def _create_aero_interpolants(self):
        """
        Private helper method to create and cache the CasADi interpolant objects.
        This is a one-time setup operation.
        """
        if self.stacked_coeffs_data is None:
            self._pre_process_data()

        # Get grid points in radians
        alpha_lin_rad = np.deg2rad(self.alpha_grid[:, 0])
        beta_lin_rad = np.deg2rad(self.beta_grid[0, :])
        grid_axes = [alpha_lin_rad, beta_lin_rad]

        coeffs_data_grid_transposed = np.transpose(self.stacked_coeffs_data, axes=(1, 0, 2))

        # Flatten the pre-stacked data for the interpolant.
        coeffs_data_flat = coeffs_data_grid_transposed.ravel(order='C')

        # Create a single interpolant for all 6 coefficients
        sanitized_name = self.name.replace(" ", "_")
        self.aero_interpolant = ca.interpolant(
            f'{sanitized_name}_CoeffsLookup',
            'linear',
            grid_axes,
            coeffs_data_flat
        )

    def _get_forces_and_moments_ca(self, alpha: ca.MX, beta: ca.MX, speed: ca.MX) -> Tuple[ca.MX, ca.MX]:
        """
        Computes aerodynamic forces and moments using pre-computed CasADi interpolants.
        """
        if self.asb_data is None:
            raise RuntimeError("Aero data not available. Run compute_buildup() or load_buildup() first.")

        if self.aero_interpolants is None:
            self._create_aero_interpolants()

        # Perform Symbolic Lookup
        interp_input = ca.vcat([alpha, beta])
        interpolated_coeffs_flat = self.aero_interpolant(interp_input)
        interpolated_coeffs = ca.reshape(interpolated_coeffs_flat, 6, 1)

        # Unpack the results
        CL, CY, CD, Cl, Cm, Cn = [interpolated_coeffs[i] for i in range(6)]

        # Perform Physics Calculation Symbolically
        density = 1.225
        dynamic_pressure = 0.5 * density * speed ** 2

        s = self.aero_component.asb_airplane.s_ref
        c = self.aero_component.asb_airplane.c_ref
        b = self.aero_component.asb_airplane.b_ref
        qS = dynamic_pressure * s

        L = CL * qS  # Lift
        Y = CY * qS  # Side Force
        D = CD * qS  # Drag
        F_w = ca.vertcat(-D, Y, -L)

        l_b = Cl * qS * b  # rolling moment
        m_b = Cm * qS * c  # pitching moment
        n_b = Cn * qS * b  # yawing moment
        M_b = ca.vertcat(l_b, m_b, n_b)
        F_b = self._convert_axes_ca(alpha, beta, F_w)

        return F_b, M_b

    def compute_buildup(self):
        """
        Computes the aerodynamic buildup data for the component over a range
        of alpha and beta angles at a specified velocity
        """

        # Run the AeroBuildup analysis
        self.asb_data = asb.AeroBuildup(
            airplane=self.aero_component.asb_airplane,
            op_point=self.op_point
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
        list_names = ["CL", "CY", "CD", "Cl", "Cm", "Cn"]

        folder_path = os.path.join(self.vehicle_path, 'buildup', self.name)
        path_object = pathlib.Path(folder_path)
        path_object.mkdir(parents=True, exist_ok=True)

        # ID is the identifier for the data to plot (e.g., "CL", "CD", "F_b")
        for ID in list_names:
             # Data from the buildup
            self.draw_buildup_figs(ID, folder_path)

    def draw_buildup_figs(self, ID: str, folder_path: str):
        data = self.asb_data[ID]
        title = f"`{self.name}` {ID}"
        file_name = f"{self.name}_{ID}.png"
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

    #def test_lookup_consistency(self, alpha_deg: float, beta_deg: float, velocity_vec: np.ndarray):
    #    """
    #    Compares the numerical output of the NumPy and CasADi lookup functions
    #    for a given operating point to ensure they are consistent.
    #    """
    #    print("\n--- Running Lookup Table Consistency Check ---")
    #    if self.asb_data is None:
    #        raise RuntimeError("Aero data not available. Run compute_buildup() or load_buildup() first.")
#
    #    # --- Numerical Inputs ---
    #    alpha_rad = np.deg2rad(alpha_deg)
    #    beta_rad = np.deg2rad(beta_deg)
#
    #    print(f"Test Point: alpha={alpha_deg:.2f} deg, beta={beta_deg:.2f} deg, velocity={velocity_vec}")
#
    #    # --- 1. NumPy-based Calculation ---
    #    F_b_numpy, M_b_numpy = self.get_forces_and_moments(alpha_rad, beta_rad, velocity_vec)
#
    #    # --- 2. CasADi-based Calculation ---
    #    # Create symbolic variables as placeholders for the inputs
    #    alpha_sym = ca.MX.sym('alpha')
    #    beta_sym = ca.MX.sym('beta')
    #    velocity_sym = ca.MX.sym('velocity')
#
    #    # Get the symbolic output expressions from the CasADi function
    #    F_b_sym, M_b_sym = self.get_forces_and_moments_casadi(alpha_sym, beta_sym, velocity_sym)
#
    #    # Create a callable CasADi Function from the symbolic graph
    #    evaluate_casadi_lookup = ca.Function(
    #        'evaluate_casadi',
    #        [alpha_sym, beta_sym, velocity_sym],
    #        [F_b_sym, M_b_sym]
    #    )
#
    #    # Call the CasADi function with the numerical inputs
    #    F_b_casadi_dm, M_b_casadi_dm = evaluate_casadi_lookup(alpha_rad, beta_rad, velocity_vec)
#
    #    # Convert CasADi's DM matrix type to NumPy arrays for comparison
    #    F_b_casadi = F_b_casadi_dm.full().flatten()
    #    M_b_casadi = M_b_casadi_dm.full().flatten()
#
    #    # --- 3. Comparison ---
    #    print("\n--- NumPy Result ---")
    #    print(f"Forces: {F_b_numpy}")
    #    print(f"Moments: {M_b_numpy}")
#
    #    print("\n--- CasADi Result ---")
    #    print(f"Forces: {F_b_casadi}")
    #    print(f"Moments: {M_b_casadi}")
#
    #    force_diff = np.linalg.norm(F_b_numpy - F_b_casadi)
    #    moment_diff = np.linalg.norm(M_b_numpy - M_b_casadi)
#
    #    print("\n--- Difference (Norm) ---")
    #    print(f"Force Difference:  {force_diff}")
    #    print(f"Moment Difference: {moment_diff}")

    #def _get_forces_and_moments_np(self, alpha: float, beta: float, speed: float) -> Tuple[np.ndarray, np.ndarray]:
    #    """
    #    TODO
    #    """
    #    scale_factor = (speed / self.operating_velocity) ** 2
#
    #    # Get the alpha and beta axes from the pre-computed grid (in radians)
    #    alpha_lin_rad = np.deg2rad(self.alpha_grid[:, 0])
    #    beta_lin_rad = np.deg2rad(self.beta_grid[0, :])
#
    #    # Reshape force data to be compatible with interpolation
    #    F_b_data = np.column_stack(self.asb_data["F_b"]).reshape(self.alpha_grid.shape + (3,))
    #    M_b_data = np.column_stack(self.asb_data["M_b"]).reshape(self.alpha_grid.shape + (3,))
#
    #    # Perform linear interpolation to find the force at the current alpha/beta
    #    F_b = utils.linear_interpolation(alpha_lin_rad, beta_lin_rad, alpha, beta, F_b_data)
    #    M_b = utils.linear_interpolation(alpha_lin_rad, beta_lin_rad, alpha, beta, M_b_data)
#
    #    # Set elements to zero if they are not included
    #    F_b = np.where(self.include_arr[:3], F_b, 0) * scale_factor
    #    M_b = np.where(self.include_arr[3:], M_b, 0) * scale_factor
    #    return F_b, M_b