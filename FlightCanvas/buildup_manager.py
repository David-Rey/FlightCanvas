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
        alpha_grid_size: int = 150,
        beta_grid_size: int = 100,
        operating_velocity: float = 50.0,
        compute_damping: bool = True):

        self.name = name
        self.vehicle_path = vehicle_path
        self.aero_component = aero_component
        self.compute_damping = compute_damping

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

        self.alpha_sweep_1D = np.linspace(-180, 180, self.alpha_grid_size)
        self.op_point_1D_alpha_sweep = asb.OperatingPoint(
            velocity=self.operating_velocity,
            alpha=self.alpha_sweep_1D
        )

        self.asb_data_static = None  # To hold the aero build data
        self.asb_data_damping = None
        self.static_interpolants = None  # To hold the CasADi interpolant object
        self.damping_interpolant = None
        self.stacked_coeffs_data_static = None  # To hold the pre-processed NumPy data
        self.stacked_coeffs_data_damping = None

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
        # 1. --- Type Detection and Library-Specific Setup ---
        is_casadi = isinstance(speed, (ca.SX, ca.MX))

        if is_casadi:
            # Use CasADi-specific functions and methods
            def vertcat_func(*args):
                return ca.vertcat(*args)
            convert_axes = self._convert_axes_ca

            # Perform symbolic lookup for static coefficients
            if self.static_interpolants is None:
                self._create_static_interpolants()

            interp_input = ca.vcat([alpha, beta])
            coeffs_flat = self.static_interpolants(interp_input)
            coeffs = ca.reshape(coeffs_flat, 6, 1)
            CL, CY, CD, Cl, Cm, Cn = [coeffs[i] for i in range(6)]

        else:
            def vertcat_func(*args):
                return np.array(args)
            convert_axes = self._convert_axes_np

            # Perform numerical interpolation for static coefficients
            if self.stacked_coeffs_data_static is None:
                self._pre_process_static_data()

            alpha_lin_rad = np.deg2rad(self.alpha_grid[:, 0])
            beta_lin_rad = np.deg2rad(self.beta_grid[0, :])

            coeffs = utils.linear_interpolation(
                alpha_lin_rad, beta_lin_rad, alpha, beta, self.stacked_coeffs_data_static
            )
            CL, CY, CD, Cl, Cm, Cn = coeffs

        density = 1.225
        dynamic_pressure = 0.5 * density * speed ** 2

        # Get airplane dimensions
        s = self.aero_component.asb_airplane.s_ref
        c = self.aero_component.asb_airplane.c_ref
        b = self.aero_component.asb_airplane.b_ref
        qS = dynamic_pressure * s

        L = -CL * qS  # Lift
        Y = CY * qS  # Side Force
        D = -CD * qS  # Drag

        # Force in wind frame
        F_w = vertcat_func(D, Y, L)

        # Force in body frame
        F_b = convert_axes(alpha, beta, F_w)

        l_b = Cl * qS * b  # rolling moment
        m_b = Cm * qS * c  # pitching moment
        n_b = Cn * qS * b  # yawing moment
        M_b_static = vertcat_func(l_b, m_b, n_b)

        M_b_damping = ca.MX.zeros(3) if is_casadi else np.zeros(3)
        # Damping Moment Calculation
        if self.asb_data_damping is not None and self.compute_damping:
            # Get damping coefficients using the appropriate method
            if is_casadi:
                if self.damping_interpolant is None:
                    self._create_damping_interpolant()
                damping_coeffs_flat = self.damping_interpolant(alpha)
                damping_coeffs = ca.reshape(damping_coeffs_flat, 3, 1)
                Clp, Cmq, Cnr = [damping_coeffs[i] for i in range(3)]
            else:
                if self.stacked_coeffs_data_damping is None:
                    self._pre_process_damping_data()
                alpha_sweep_rad = np.deg2rad(self.alpha_sweep_1D)
                damping_coeffs = utils.linear_interpolation_1d(
                    alpha_sweep_rad, alpha, self.stacked_coeffs_data_damping
                )
                Clp, Cmq, Cnr = damping_coeffs

            # Non-dimensionalization factor for rates
            # Use a small epsilon to avoid division by zero at speed=0
            V_inv = 1 / (speed + 1e-9)  # Epsilon for stability at speed=0
            p_hat = p * b * 0.5 * V_inv
            q_hat = q * c * 0.5 * V_inv
            r_hat = r * b * 0.5 * V_inv

            # Calculate damping moments
            M_b_damping_values = vertcat_func(
                Clp * qS * b * p_hat,
                Cmq * qS * c * q_hat,
                Cnr * qS * b * r_hat
            )
            M_b_damping = M_b_damping_values

        M_b_total = M_b_static + M_b_damping
        return F_b, M_b_total

    def get_forces_and_moments_old(
        self,
        alpha: Union[float, ca.MX],
        beta: Union[float, ca.MX],
        speed: Union[float, ca.MX],
        p: Union[float, ca.MX],
        q: Union[float, ca.MX],
        r: Union[float, ca.MX]
    ) -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
        """
        Computes forces and moments. This function is type-aware and will use
        either NumPy or CasADi based on the input type.
        """
        is_casadi = isinstance(speed, (ca.SX, ca.MX))

        if is_casadi:
            return self._get_forces_and_moments_ca(alpha, beta, speed, p, q, r)
        else:
            return self._get_forces_and_moments_np(alpha, beta, speed, p, q, r)

    def _get_forces_and_moments_np(self, alpha: float, beta: float, speed: float, p: float, q: float, r: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        TODO
        """
        if self.stacked_coeffs_data_static is None:
            self._pre_process_static_data()

        # Get the alpha and beta axes from the pre-computed grid (in radians)
        alpha_lin_rad = np.deg2rad(self.alpha_grid[:, 0])
        beta_lin_rad = np.deg2rad(self.beta_grid[0, :])

        # Perform a single, fast interpolation on the pre-stacked data
        interpolated_coeffs = utils.linear_interpolation(
            alpha_lin_rad, beta_lin_rad, alpha, beta, self.stacked_coeffs_data_static
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
        F_b = self._convert_axes_np(alpha, beta, F_w)

        l_b = Cl * qS * b  # rolling moment
        m_b = Cm * qS * c  # pitching moment
        n_b = Cn * qS * b  # yawing moment
        M_b_static = np.array([l_b, m_b, n_b])

        M_b_damping = np.zeros(3)
        # Damping Moment Calculation ---
        if self.asb_data_damping is not None and self.compute_damping:
            if self.stacked_coeffs_data_damping is None:
                self._pre_process_damping_data()

            # 1D lookup for damping derivatives
            alpha_sweep_rad = np.deg2rad(self.alpha_sweep_1D)
            damping_coeffs = utils.linear_interpolation_1d(
                alpha_sweep_rad, alpha, self.stacked_coeffs_data_damping
            )
            Clp, Cmq, Cnr = damping_coeffs

            # Non-dimensionalization factor for rates
            # Use a small epsilon to avoid division by zero at speed=0
            V_inv = 1 / (speed + 1e-9)
            p_hat = p * b * 0.5 * V_inv
            q_hat = q * c * 0.5 * V_inv
            r_hat = r * b * 0.5 * V_inv

            # Add damping moments
            M_b_damping[0] = Clp * qS * b * p_hat
            M_b_damping[1] = Cmq * qS * c * q_hat
            M_b_damping[2] = Cnr * qS * b * r_hat

        M_b_total = M_b_static + M_b_damping
        return F_b, M_b_total

    def _pre_process_static_data(self):
        """
        Reshapes and stacks the raw coefficient data into a single 3D NumPy array
        for efficient interpolation. This should only be called once.
        """
        if self.asb_data_static is None:
            raise RuntimeError("Aero data not available. Run compute_buildup() or load_buildup() first.")

        # Reshape each coefficient's data into a 2D grid
        CL_data = np.column_stack(self.asb_data_static["CL"]).reshape(self.alpha_grid.shape)
        CY_data = np.column_stack(self.asb_data_static["CY"]).reshape(self.alpha_grid.shape)
        CD_data = np.column_stack(self.asb_data_static["CD"]).reshape(self.alpha_grid.shape)
        Cl_data = np.column_stack(self.asb_data_static["Cl"]).reshape(self.alpha_grid.shape)
        Cm_data = np.column_stack(self.asb_data_static["Cm"]).reshape(self.alpha_grid.shape)
        Cn_data = np.column_stack(self.asb_data_static["Cn"]).reshape(self.alpha_grid.shape)

        # Stack the 2D grids into a single 3D data array and store it
        self.stacked_coeffs_data_static = np.stack(
            [CL_data, CY_data, CD_data, Cl_data, Cm_data, Cn_data],
            axis=-1
        )

    def _pre_process_damping_data(self):
        """Pre-processes damping data for NumPy interpolation."""
        self.stacked_coeffs_data_damping = np.column_stack([
            self.asb_data_damping["Clp"],
            self.asb_data_damping["Cmq"],
            self.asb_data_damping["Cnr"]
        ])

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

        # Create a single interpolant for all 6 coefficients
        sanitized_name = self.name.replace(" ", "_")
        self.static_interpolants = ca.interpolant(
            f'{sanitized_name}_CoeffsLookup',
            'linear',
            grid_axes,
            coeffs_data_flat
        )

    def _get_forces_and_moments_ca(self, alpha: ca.MX, beta: ca.MX, speed: ca.MX, p: ca.MX, q: ca.MX, r: ca.MX) -> Tuple[ca.MX, ca.MX]:
        """
        Computes aerodynamic forces and moments using pre-computed CasADi interpolants.
        """
        if self.asb_data_static is None:
            raise RuntimeError("Aero data not available. Run compute_buildup() or load_buildup() first.")

        if self.static_interpolants is None:
            self._create_static_interpolants()

        # Perform Symbolic Lookup
        interp_input = ca.vcat([alpha, beta])
        interpolated_coeffs_flat = self.static_interpolants(interp_input)
        interpolated_coeffs = ca.reshape(interpolated_coeffs_flat, 6, 1)

        # Unpack the results
        CL, CY, CD, Cl, Cm, Cn = [interpolated_coeffs[i] for i in range(6)]

        # Perform Physics Calculation Symbolically
        density = 1.225  # hard code density for now
        dynamic_pressure = 0.5 * density * speed ** 2

        # Get airplane dimensions
        s = self.aero_component.asb_airplane.s_ref
        c = self.aero_component.asb_airplane.c_ref
        b = self.aero_component.asb_airplane.b_ref
        qS = dynamic_pressure * s

        L = CL * qS  # Lift
        Y = CY * qS  # Side Force
        D = CD * qS  # Drag

        # Force in wind frame
        F_w = ca.vertcat(-D, Y, -L)

        # Force in body frame
        F_b = self._convert_axes_ca(alpha, beta, F_w)

        l_b = Cl * qS * b  # rolling moment
        m_b = Cm * qS * c  # pitching moment
        n_b = Cn * qS * b  # yawing moment
        M_b_static = ca.vertcat(l_b, m_b, n_b)

        M_b_damping = ca.MX.zeros(3)
        if self.asb_data_damping is not None and self.compute_damping:
            if self.damping_interpolant is None:
                self._create_damping_interpolant()

            damping_coeffs_flat = self.damping_interpolant(alpha)
            damping_coeffs = ca.reshape(damping_coeffs_flat, 3, 1)
            Clp, Cmq, Cnr = [damping_coeffs[i] for i in range(3)]

            V_inv = 1 / (speed + 1e-9)
            p_hat = p * b * 0.5 * V_inv
            q_hat = q * c * 0.5 * V_inv
            r_hat = r * b * 0.5 * V_inv

            M_b_damping[0] = Clp * qS * b * p_hat
            M_b_damping[1] = Cmq * qS * c * q_hat
            M_b_damping[2] = Cnr * qS * b * r_hat

        M_b_total = M_b_static + M_b_damping
        return F_b, M_b_total

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

    def save_buildup(self):
        """
        Saves buildup data to buildup folder
        """
        # Put variables in a dictionary for easy loading
        variables_to_save = {
            'name': self.name,
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
        list_names = ["CL", "CY", "CD", "Cl", "Cm", "Cn"]

        filename_map = {
            "CL": "CL",  # Lift coefficient
            "CY": "CY",
            "CD": "CD",
            "Cl": "Cl_moment",  # Rolling moment coefficient
            "Cm": "Cm",
            "Cn": "Cn"
        }

        folder_path = os.path.join(self.vehicle_path, 'buildup_figs', self.name)
        path_object = pathlib.Path(folder_path)
        path_object.mkdir(parents=True, exist_ok=True)

        # ID is the identifier for the data to plot (e.g., "CL", "CD", "F_b")
        for ID in list_names:
            file_id = filename_map.get(ID, ID)
            # Data from the buildup
            self.draw_buildup_figs(ID, file_id, folder_path)

    def draw_buildup_figs(self, ID: str, file_id: str, folder_path: str):
        """
        Saves buildup figure which a function of angle of attack and sideslip for aerodynamic coefficient
        :param ID: Name of the aerodynamic coefficient
        :param file_id: Name of the buildup figure
        :param folder_path: Path to the buildup figure
        """
        data = self.asb_data_static[ID]
        title = f"`{self.name}` {file_id}"
        file_name = f"{self.name}_{file_id}.png"
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