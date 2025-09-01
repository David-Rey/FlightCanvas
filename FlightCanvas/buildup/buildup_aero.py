from FlightCanvas.buildup.buildup_base import BuildupBase
from typing import Union, Tuple, Dict, Any, Optional
import casadi as ca
import numpy as np
from FlightCanvas import utils


class BuildupAero(BuildupBase):
    def __init__(self,
                 name: str,
                 vehicle_path: str,
                 aero_component: ['AeroComponent'],
                 coefficients_config: Optional[Dict[str, Any]] = None,
                 alpha_grid_size: int = 150,
                 beta_grid_size: int = 100,
                 operating_mach: float = 0.3,
                 compute_damping: bool = False,
                 smooth_data: bool = True
                 ):
        super().__init__(
            name,
            vehicle_path,
            aero_component,
            coefficients_config,
            alpha_grid_size,
            beta_grid_size,
            operating_mach,
            compute_damping,
            smooth_data)

    def _get_default_coefficients_config(self) -> Dict[str, Any]:
        """
        Returns the default aerodynamic coefficients configuration for BuildupAero
        """
        return {
            'static_coeffs': {
                'CL': {'key': 'CL', 'index': None, 'smooth': True, 'display_name': 'CL'},
                'CY': {'key': 'CY', 'index': None, 'smooth': True, 'display_name': 'CY'},
                'CD': {'key': 'CD', 'index': None, 'smooth': True, 'display_name': 'CD'},
                'Cl': {'key': 'Cl', 'index': None, 'smooth': True, 'display_name': 'Cl_moment'},
                'Cm': {'key': 'Cm', 'index': None, 'smooth': True, 'display_name': 'Cm'},
                'Cn': {'key': 'Cn', 'index': None, 'smooth': True, 'display_name': 'Cn'}
            },
            'damping_coeffs': {
                'Clp': {'key': 'Clp', 'index': None, 'display_name': 'Clp'},
                'Cmq': {'key': 'Cmq', 'index': None, 'display_name': 'Cmq'},
                'Cnr': {'key': 'Cnr', 'index': None, 'display_name': 'Cnr'}
            }
        }

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

            # Get number of static coefficients dynamically
            num_static_coeffs = len(self.coefficients_config['static_coeffs'])
            coeffs = ca.reshape(coeffs_flat, num_static_coeffs, 1)

            # Extract coefficients in the order they appear in the config
            static_coeffs_list = list(self.coefficients_config['static_coeffs'].keys())
            coeff_values = {name: coeffs[i] for i, name in enumerate(static_coeffs_list)}

            # Get the specific coefficients we need (with fallback to zero if not present)
            CL = coeff_values.get('CL', 0)
            CY = coeff_values.get('CY', 0)
            CD = coeff_values.get('CD', 0)
            Cl = coeff_values.get('Cl', 0)
            Cm = coeff_values.get('Cm', 0)
            Cn = coeff_values.get('Cn', 0)

        else:
            # Use NumPy-specific functions and methods
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

            # Extract coefficients in the order they appear in the config
            static_coeffs_list = list(self.coefficients_config['static_coeffs'].keys())
            coeff_values = {name: coeffs[i] if i < len(coeffs) else 0
                            for i, name in enumerate(static_coeffs_list)}

            # Get the specific coefficients we need (with fallback to zero if not present)
            CL = coeff_values.get('CL', 0)
            CY = coeff_values.get('CY', 0)
            CD = coeff_values.get('CD', 0)
            Cl = coeff_values.get('Cl', 0)
            Cm = coeff_values.get('Cm', 0)
            Cn = coeff_values.get('Cn', 0)

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

                # Get number of damping coefficients dynamically
                num_damping_coeffs = len(self.coefficients_config['damping_coeffs'])
                damping_coeffs = ca.reshape(damping_coeffs_flat, num_damping_coeffs, 1)

                # Extract damping coefficients in the order they appear in the config
                damping_coeffs_list = list(self.coefficients_config['damping_coeffs'].keys())
                damping_coeff_values = {name: damping_coeffs[i]
                                        for i, name in enumerate(damping_coeffs_list)}

                # Get the specific damping coefficients we need
                Clp = damping_coeff_values.get('Clp', 0)
                Cmq = damping_coeff_values.get('Cmq', 0)
                Cnr = damping_coeff_values.get('Cnr', 0)

            else:
                if self.stacked_coeffs_data_damping is None:
                    self._pre_process_damping_data()
                alpha_sweep_rad = np.deg2rad(self.alpha_sweep_1D)
                damping_coeffs = utils.linear_interpolation_1d(
                    alpha_sweep_rad, alpha, self.stacked_coeffs_data_damping
                )

                # Extract damping coefficients in the order they appear in the config
                damping_coeffs_list = list(self.coefficients_config['damping_coeffs'].keys())
                damping_coeff_values = {name: damping_coeffs[i] if i < len(damping_coeffs) else 0
                                        for i, name in enumerate(damping_coeffs_list)}

                # Get the specific damping coefficients we need
                Clp = damping_coeff_values.get('Clp', 0)
                Cmq = damping_coeff_values.get('Cmq', 0)
                Cnr = damping_coeff_values.get('Cnr', 0)

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
