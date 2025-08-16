def test_casadi(self, pos_0, vel_0, quat_0, omega_0):
        def get_rel_alpha_beta(v_rel: Union[np.ndarray, List[float]]):
            epsilon = 1e-10
            v_a = np.linalg.norm(v_rel)  # Total airspeed
            alpha = np.arctan2(v_rel[2], v_rel[0])  # Angle of attack
            beta = np.arcsin(v_rel[1] / (v_a + epsilon))  # Sideslip angle
            return alpha, beta

        def get_rel_alpha_beta_casadi(v_rel: ca.MX) -> tuple[ca.MX, ca.MX]:

            epsilon = 1e-10
            v_a = ca.norm_2(v_rel)
            alpha = ca.atan2(v_rel[2], v_rel[0])
            beta = ca.asin(v_rel[1] / (v_a + epsilon))

            return alpha, beta

        state_num = np.concatenate((pos_0, vel_0, quat_0, omega_0))

        # --- Alpha/Beta Comparison ---
        print("\n--- Alpha/Beta Comparison ---")

        # 1. NumPy Calculation
        print("\n--- NumPy Alpha/Beta ---")
        vel_num = state_num[3:6]
        quat_num = state_num[6:10]
        angular_rate_num = state_num[10:]
        v_B_num = utils.dir_cosine_np(quat_num).T @ vel_num

        for component in self.components:
            R_num = component.static_transform_matrix[:3, :3]
            v_comp_num = R_num @ v_B_num - np.cross(angular_rate_num, component.xyz_ref)
            alpha_num, beta_num = get_rel_alpha_beta(v_comp_num)
            print(f"Component '{component.name}': alpha={np.rad2deg(alpha_num):.4f}, beta={np.rad2deg(beta_num):.4f}")

        # 2. CasADi Calculation
        print("\n--- CasADi Alpha/Beta ---")
        state_sym = ca.MX.sym('state', state_num.shape[0])
        vel_sym = state_sym[3:6]
        quat_sym = state_sym[6:10]
        angular_rate_sym = state_sym[10:]
        v_B_sym = utils.dir_cosine_ca(quat_sym).T @ vel_sym

        alpha_beta_sym_list = []
        for component in self.components:
            R_sym = ca.MX(component.static_transform_matrix)[:3, :3]
            xyz_ref_sym = ca.MX(component.xyz_ref)
            v_comp_sym = R_sym @ v_B_sym - ca.cross(angular_rate_sym, xyz_ref_sym)
            alpha_sym, beta_sym = get_rel_alpha_beta_casadi(v_comp_sym)
            alpha_beta_sym_list.append(ca.vertcat(alpha_sym, beta_sym))

        calculate_ab_casadi = ca.Function(
            'calculate_alpha_beta',
            [state_sym],
            alpha_beta_sym_list
        )

        casadi_results = calculate_ab_casadi(state_num)

        # Handle the case where the function returns a single DM vs. a list of DMs
        if not isinstance(casadi_results, list):
            casadi_results = [casadi_results]

        for i, component in enumerate(self.components):
            alpha_casadi = casadi_results[i][0].full().item()
            beta_casadi = casadi_results[i][1].full().item()
            print(
                f"Component '{component.name}': alpha={np.rad2deg(alpha_casadi):.4f}, beta={np.rad2deg(beta_casadi):.4f}")

        # --- Original Force/Moment Comparison ---
        print("\n--- Force/Moment Comparison ---")
        F_B_sym, M_B_sym = self.compute_forces_and_moments_casadi(state_sym)

        calculate_aero_casadi = ca.Function(
            'calculate_aero',
            [state_sym],
            [F_B_sym, M_B_sym]
        )

        F_B_casadi_result, M_B_casadi_result = calculate_aero_casadi(state_num)

        F_B_casadi_np = F_B_casadi_result.full().flatten()
        M_B_casadi_np = M_B_casadi_result.full().flatten()

        F_B_numpy, M_B_numpy = self.compute_forces_and_moments_lookup(state_num)

        print("--- CasADi Result ---")
        print("Forces:", F_B_casadi_np)
        print("Moments:", M_B_casadi_np)

        print("\n--- NumPy Result ---")
        print("Forces:", F_B_numpy)
        print("Moments:", M_B_numpy)

        #self.components[0].buildup_manager.test_lookup_consistency(10, 1, 100)