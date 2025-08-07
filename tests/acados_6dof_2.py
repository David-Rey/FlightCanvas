import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import casadi as ca
import scipy as sp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def dir_cosine_casadi(q):
    """Converts a quaternion to a 3x3 direction cosine matrix."""
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    # DCM for q = [w, x, y, z]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)),
        ca.horzcat(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)),
        ca.horzcat(2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2))
    )


def omega_casadi(w):
    """Returns the omega matrix for quaternion kinematics."""
    wx, wy, wz = w[0], w[1], w[2]
    return ca.vertcat(
        ca.horzcat(0, -wx, -wy, -wz),
        ca.horzcat(wx, 0, wz, -wy),
        ca.horzcat(wy, -wz, 0, wx),
        ca.horzcat(wz, wy, -wx, 0)
    )


def create_rocket_model() -> AcadosModel:
    """
    Creates a 6DOF rocket model with time augmentation.
    """
    model = AcadosModel()
    model.name = 'rocket'

    ### Rocket Parameters
    g = 9.81  # m/s^2
    Isp = 300  # s
    Lr = 4.5  # m
    Lh = 50  # m
    Lcm = 20  # m
    delta_max_rad = np.deg2rad(15.0)

    # Moment of Inertia
    m = ca.MX.sym('m', 1)

    Ixx = 1 / 4 * m[0] * Lr ** 2 + 1 / 12 * m[0] * Lh ** 2
    Iyy = 1 / 4 * m[0] * Lr ** 2 + 1 / 12 * m[0] * Lh ** 2
    Izz = 1 / 2 * m[0] * Lr ** 2

    epsilon_m = 1e-5
    Jb = ca.diag(ca.vertcat(Ixx, Iyy, Izz))
    Jb_inv = ca.inv(Jb + ca.DM.eye(3) * epsilon_m)

    # Gravity VEctor
    g_vec = ca.MX([0, 0, -g])

    # Distance from thrust to CM
    rT = ca.MX([0, 0, -Lcm])

    # States
    r = ca.MX.sym('r', 3)
    v = ca.MX.sym('v', 3)
    q = ca.MX.sym('q', 4)
    w = ca.MX.sym('w', 3)
    t = ca.MX.sym('t', 1)

    # Control
    Tb = ca.MX.sym('Tb', 3)

    # Derivatives
    m_dot = -1 / (9.81 * Isp) * ca.norm_2(Tb)
    r_dot = v
    v_dot = (1 / m[0] * dir_cosine_casadi(q) @ Tb) + g_vec
    q_dot = 1 / 2 * omega_casadi(w) @ q
    w_dot = Jb_inv @ (ca.cross(rT, Tb) - ca.cross(w, Jb @ w))
    t_dot = 0

    # State vector
    x = ca.vertcat(m, r, v, q, w, t)
    f_physical = ca.vertcat(m_dot, r_dot, v_dot, q_dot, w_dot) # <<< MODIFY: Add t_dot
    u = ca.vertcat(Tb)

    f_scaled = t * f_physical

    f_expl_expr = ca.vertcat(f_scaled, t_dot)

    x_dot_sym = ca.MX.sym('x_dot_sym', x.shape) # Shape is now (15, 1)

    # Set model attributes
    model.x = x
    model.u = u
    model.xdot = x_dot_sym
    model.f_expl_expr = f_expl_expr
    model.f_impl_expr = f_expl_expr - x_dot_sym

    # Constraints
    quat_con = ca.norm_2(q) - 1  # quaterion
    thrust_con = ca.norm_2(Tb)  # thrust
    tilt_con = 1 - 2 *(q[2] ** 2 + q[3] ** 2)  # max tilt
    angular_rate_con = ca.dot(w, w)  # max angular rate

    e1 = ca.MX([1, 0, 0])
    e2 = ca.MX([0, 1, 0])
    e3 = ca.MX([0, 0, 1])
    H12 = ca.horzcat(e1, e2)
    H12T = ca.transpose(H12)

    gimbal_con = ca.dot(e3, Tb) - ca.norm_2(Tb) * np.cos(delta_max_rad)
    flight_angle = ca.dot(e3, r) / (ca.norm_2(H12T @ r) + epsilon_m)

    con_h_expr = ca.vertcat(
        quat_con,
        thrust_con,
        gimbal_con,
        tilt_con,
        angular_rate_con,
        flight_angle
    )

    model.con_h_expr = con_h_expr
    model.con_h_expr_0 = con_h_expr

    model.cost_expr_ext_e = -m

    return model


def main():
    """
    Sets up and solves the free-time optimal control problem for the rocket landing.
    """
    ocp = AcadosOcp()
    ocp.model = create_rocket_model()
    ocp.code_export_directory = 'tests/export'

    # --- Horizon and Solver Settings ---
    N = 100
    # The final time is now a variable to be optimized, so we don't set ocp.solver_options.tf
    ocp.dims.N = N
    ocp.solver_options.tf = 1

    # --- Get dimensions ---
    nx = ocp.model.x.size1()  # Will be 15
    nu = ocp.model.u.size1()
    ny = nx + nu

    # --- Initial State ---
    m0 = 400.0
    r0 = np.array([0.0, 30.0, 100.0])
    v0 = np.array([0.0, -15.0, -20.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    w0 = np.array([0.0, 0.0, 0.0])

    x0_physical = np.concatenate([np.array([m0]), r0, v0, q0, w0])

    # --- Cost Function (LINEAR_LS to minimize fuel) ---
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    Q = np.diag([0.0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0])
    R = np.diag([1, 1, 1]) * 0
    ocp.cost.W = sp.linalg.block_diag(Q, R)

    # Increase weight on final mass to give a stronger incentive to save fuel
    W_e = np.diag([100.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0])
    ocp.cost.W_e = W_e

    yref_e = np.zeros(nx)
    yref_e[0] = m0
    yref = np.zeros(ny)
    yref[7:11] = np.array([1.0, 0, 0, 0])
    ocp.cost.yref = yref
    ocp.cost.yref_e = yref_e

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:, :nu] = np.eye(nu)
    ocp.cost.Vx_e = np.eye(nx)

    # --- Constraints ---
    # 1. Initial State Constraint (on physical states only)
    ocp.constraints.idxbx_0 = np.arange(nx - 1)
    ocp.constraints.lbx_0 = x0_physical
    ocp.constraints.ubx_0 = x0_physical

    # 2. Path Constraints (on mass and final time Tf)
    dry_mass = 300.0
    Tf_min = 3.0
    Tf_max = 40.0
    ocp.constraints.idxbx = np.array([0, 14])
    ocp.constraints.lbx = np.array([dry_mass, Tf_min])
    ocp.constraints.ubx = np.array([m0, Tf_max])

    # 3. Path Constraints (on controls and other states)
    T_min = 2000
    T_max = 9000
    eta_max = 20
    omega_max = 2
    glide_slope = 89
    inf = 1e9
    lh = np.array([0, T_min, 0, np.cos(np.deg2rad(eta_max)), 0, np.tan(np.deg2rad(glide_slope))])
    uh = np.array([0, T_max, inf, inf, omega_max ** 2, inf])

    ocp.constraints.lh = lh
    ocp.constraints.uh =uh
    ocp.constraints.lh_0 = lh
    ocp.constraints.uh_0 = uh

    # 4. Terminal State Constraints (on physical states only)
    terminal_phys_indices = np.arange(1, 14)
    terminal_r = [0, 0, 0]
    terminal_v = [0, 0, 0]
    terminal_q = [1, 0, 0, 0]
    terminal_w = [0, 0, 0]
    x_target_e = np.concatenate([terminal_r, terminal_v, terminal_q, terminal_w])

    ocp.constraints.idxbx_e = terminal_phys_indices
    ocp.constraints.lbx_e = x_target_e
    ocp.constraints.ubx_e = x_target_e
    ocp.dims.nbx_e = len(terminal_phys_indices)

    #ocp.constraints.idxbu = np.array([2])  # Indices of controls to constrain (all of them)
    #ocp.constraints.lbu = np.array([0])  # Lower bound
    #ocp.constraints.ubu = np.array([T_max])  # Upper bound

    soft_constraint_indices = [0, 1, 2, 3, 4, 5] # Removed index 2
    ocp.constraints.idxsh = np.array(soft_constraint_indices)
    ocp.constraints.idxsh_0 = np.array(soft_constraint_indices)

    # --- Slack Variables for Soft Constraints ---
    num_soft_constraints = len(soft_constraint_indices)
    penalty_weight = 1e2
    ocp.cost.zl = penalty_weight * np.ones((num_soft_constraints,))
    ocp.cost.zu = penalty_weight * np.ones((num_soft_constraints,))
    ocp.cost.Zl = np.zeros_like(ocp.cost.zl)
    ocp.cost.Zu = np.zeros_like(ocp.cost.zu)

    # --- Solver Configuration ---
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.regularize_method = 'GERSHGORIN_LEVENBERG_MARQUARDT'
    ocp.solver_options.levenberg_marquardt = 1e-2
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.tol = 1e-2
    ocp.solver_options.qp_solver_cond_N = N

    # --- Create and Run Solver ---
    #ocp_solver = AcadosOcpSolver(ocp, json_file='test/acados_ocp_rocket_free_time.json')
    ocp_solver = AcadosOcpSolver(ocp)

    # --- Initialize Trajectory ---
    # 1. Create guess for physical states
    init_x_physical = np.zeros((N + 1, nx - 1))
    x_target_physical = np.concatenate([np.array([dry_mass]), x_target_e])
    for i in range(N + 1):
        alpha = i / N
        init_x_physical[i, :] = (1 - alpha) * x0_physical + alpha * x_target_physical
        q_part = init_x_physical[i, 7:11]
        if np.linalg.norm(q_part) > 1e-6:
            init_x_physical[i, 7:11] = q_part / np.linalg.norm(q_part)

    # 2. Create guess for time state
    Tf_guess = 15.0
    init_t_state = np.linspace(0, Tf_guess, N + 1).reshape(N + 1, 1)

    # 3. Combine into the full initial trajectory guess
    init_x = np.hstack([init_x_physical, init_t_state])

    # 4. Create initial guess for controls
    init_u = np.zeros((N, nu))
    init_u[:, 2] = m0 * 9.81  # Hover thrust

    # 5. CORRECTED: Set the initial guess for each stage using a loop
    for i in range(N):
        ocp_solver.set(i, "x", init_x[i, :])
        ocp_solver.set(i, "u", init_u[i, :])
    ocp_solver.set(N, "x", init_x[N, :])  # Set state at final node

    status = ocp_solver.solve()

    # --- Get timing statistics ---
    total_time = ocp_solver.get_stats('time_tot')
    print(f"Solver finished in {total_time * 1000:.3f} ms.")

    ocp_solver.print_statistics()
    cost_value = ocp_solver.get_cost()
    print(f"\nFinal Cost Function Value: {cost_value}")

    # --- Plotting ---
    # Extract solution
    simX = np.zeros((N + 1, nx))
    simU = np.zeros((N, nu))
    for i in range(N):
        simX[i, :] = ocp_solver.get(i, "x")
        simU[i, :] = ocp_solver.get(i, "u")
    simX[N, :] = ocp_solver.get(N, "x")

    # Get the optimal final time from the last state
    Tf = simX[-1, -1]
    print(f"\nOptimal Landing Time: {Tf:.3f} s\n")

    # Time vector
    t = np.linspace(0, Tf, N + 1)

    # 3D Trajectory Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(simX[:, 1], simX[:, 2], simX[:, 3], 'b-')
    ax.plot(simX[0, 1], simX[0, 2], simX[0, 3], 'go', markersize=10, label='Start')
    ax.plot(simX[-1, 1], simX[-1, 2], simX[-1, 3], 'ro', markersize=10, label='End')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Rocket 3D Trajectory')
    ax.legend()
    ax.grid(True)

    # State Trajectories Plot
    fig2, axs = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
    state_labels = ['Mass (kg)', 'Position (m)', 'Velocity (m/s)', 'Quaternion', 'Angular Velocity (rad/s)']
    state_indices = [(0, 1), (1, 4), (4, 7), (7, 11), (11, 14)]

    for i, label in enumerate(state_labels):
        start, end = state_indices[i]
        axs[i].plot(t, simX[:, start:end])
        axs[i].set_ylabel(label)
        axs[i].grid(True)
        if i == 0:
            axs[i].legend(['m'])
        elif i == 1:
            axs[i].legend(['r_x', 'r_y', 'r_z'])
        elif i == 2:
            axs[i].legend(['v_x', 'v_y', 'v_z'])
        elif i == 3:
            axs[i].legend(['q0', 'q1', 'q2', 'q3'])
        elif i == 4:
            axs[i].legend(['w_x', 'w_y', 'w_z'])

    axs[-1].set_xlabel('Time [s]')
    fig2.suptitle('State Trajectories', fontsize=16)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Control Trajectories Plot
    fig3, ax_u = plt.subplots(figsize=(12, 6))
    ax_u.step(t[:-1], simU[:, 0], where='post', label='Tb_x')
    ax_u.step(t[:-1], simU[:, 1], where='post', label='Tb_y')
    ax_u.step(t[:-1], simU[:, 2], where='post', label='Tb_z')
    ax_u.set_xlabel('Time [s]')
    ax_u.set_ylabel('Thrust [N]')
    ax_u.set_title('Control Inputs (Thrust Vector)')
    ax_u.legend()
    ax_u.grid(True)

    fig.savefig('tests/figs/rocket_trajectory_3d.png')
    fig2.savefig('tests/figs/rocket_states.png')
    fig3.savefig('tests/figs/rocket_controls.png')

    plt.show()
    #plt.close('all')  # Close all figures to free up memory


if __name__ == '__main__':
    main()


    """
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    # State and control penalty matrices for stage cost
    Q = np.diag([
        0,  # mass
        5.0, 5.0, 10.0,  # r_x, r_y, r_z
        2.0, 2.0, 5.0,  # v_x, v_y, v_z
        0.1, 0.1, 0.1, 0.1,  # q (penalize deviation from reference)
        0.1, 0.1, 0.1,  # w_x, w_y, w_z
        0.0  # t_state (no penalty on running time)
    ])
    R = np.diag([0.001, 0.001, 0.001])  # Penalty on thrust control

    # Terminal state penalty matrix
    W_time = 1.0  # Weight for minimizing the final time
    Q_e = np.diag([
        0,  # mass_e
        50, 50, 200,  # r_xe, r_ye, r_ze
        50, 50, 400,  # v_xe, v_ye, v_ze
        10, 10, 10, 10,  # q_e
        10, 10, 10,  # w_e
        W_time  # Penalty on final time t(Tf)
    ])

    # Set weighting matrices
    ocp.cost.W = sp.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    # Set up mapping matrices
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:, :] = np.eye(nu)

    ocp.cost.Vx_e = np.eye(ny_e)

    # --- Reference Trajectory (Target State) ---
    # Target state for terminal cost
    yref_e = np.array([
        0,  # Final mass doesn't matter
        0, 0, 0,  # Land at origin
        0, 0, 0,  # Zero velocity
        1, 0, 0, 0,  # Upright attitude
        0, 0, 0,  # Zero angular rate
        0.0  # Reference for final time is 0, so cost is W_time * (t(Tf) - 0)^2
    ])

    # Target for stage cost (state part is yref_e, control part is zero)
    yref = np.concatenate([yref_e, np.zeros(nu)])

    ocp.cost.yref = yref
    ocp.cost.yref_e = yref_e
    """

    """


        x0_physical = np.concatenate([np.array([m0]), r0, v0, q0, w0])
        ocp.constraints.idxbx_0 = np.arange(nx - 1)  # Indices 0 to 13
        ocp.constraints.lbx_0 = x0_physical
        ocp.constraints.ubx_0 = x0_physical

        # Target values for the terminal state
        terminal_r = [0.0, 0.0, 0.0]
        terminal_v = [0.0, 0.0, 0.0]
        terminal_q = [1.0, 0.0, 0.0, 0.0]
        terminal_w = [0.0, 0.0, 0.0]
        x_target_e = np.concatenate([terminal_r, terminal_v, terminal_q, terminal_w])

        # Set lower and upper bounds to the same value for equality constraints
        terminal_phys_indices = np.arange(1, 14)
        ocp.constraints.idxbx_e = terminal_phys_indices
        ocp.constraints.lbx_e = x_target_e
        ocp.constraints.ubx_e = x_target_e
        ocp.dims.nbx_e = len(terminal_phys_indices)

        # --- Cost Function ---
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        # Running Cost Weights
        # Small penalty on states for regularization
        Q = np.diag([
            0.0,                # mass
            0.0, 0.0, 0.0,      # position
            0.0, 0.0, 0.0,      # velocity
            0.1, 0.1, 0.1, 0.1, # quaternion
            0.1, 0.1, 0.1,      # angular velocity
            0.0                 # time
        ])
        # Main penalty on control usage to minimize fuel
        R = np.diag([0.01, 0.01, 0.01])
        ocp.cost.W = sp.linalg.block_diag(Q, R)

        # Terminal Cost Weights
        # Large penalty on final mass to maximize it
        W_e = np.diag([
            10.0,             # mass (large weight to maximize)
            0, 0, 0,            # position (hard constrained)
            0, 0, 0,            # velocity (hard constrained)
            0, 0, 0, 0,         # quaternion (hard constrained)
            0, 0, 0,            # angular velocity (hard constrained)
            0                   # time (no penalty)
        ])
        ocp.cost.W_e = W_e

        # --- Reference Trajectories ---
        # Terminal reference
        yref_e = np.zeros(nx)
        yref_e[0] = m0 # Reference for final mass is initial mass

        # Running reference
        yref = np.zeros(ny)
        yref[7:11] = np.array([1.0, 0, 0, 0]) # Reference for quaternion is upright

        ocp.cost.yref = yref
        ocp.cost.yref_e = yref_e

        # --- Cost-to-Variable Mapping ---
        ocp.cost.Vx = np.zeros((ny, nx)); ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu)); ocp.cost.Vu[nx:, :nu] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        # --- Constraints ---

        dry_mass = 300

        # Bound the final time to provide a search window for the solver
        Tf_min = 6.0
        Tf_max = 20.0

        # The time state is the last state, at index 14
        ocp.constraints.idxbx = np.array([0, 14])
        ocp.constraints.lbx = np.array([dry_mass, Tf_min])
        ocp.constraints.ubx = np.array([m0, Tf_max])

        # Terminal constraints on time
        ocp.constraints.idxbx_e = np.array([14])
        ocp.constraints.lbx_e = np.array([Tf_min])
        ocp.constraints.ubx_e = np.array([Tf_max])

        T_min = 2000
        T_max = 7000
        eta_max = 20  # max tilt angle (deg)
        omega_max = 2  # max angular rate (rad/s)
        inf = 1e9

        ocp.constraints.lh = np.array([
            0,
            T_min,
            0,
            np.cos(np.deg2rad(eta_max)),
            0
        ])
        ocp.constraints.uh = np.array([
            0,
            T_max,
            inf,
            inf,
            omega_max ** 2
        ])

        # --- Slack Variables for Soft Constraints ---
        ocp.constraints.idxsh = np.arange(len(ocp.constraints.lh))
        penalty_weight = 1e3
        ocp.cost.zl = penalty_weight * np.ones((len(ocp.constraints.lh),))
        ocp.cost.zu = penalty_weight * np.ones((len(ocp.constraints.lh),))
        ocp.cost.Zl = np.zeros_like(ocp.cost.zl)
        ocp.cost.Zu = np.zeros_like(ocp.cost.zu)
        """


    #delta_max_val = 15.0  # degrees
    #tan_delta_sq = np.tan(np.deg2rad(delta_max_val)) ** 2
    #gimbal_con = Tb[2] ** 2 * tan_delta_sq - (Tb[0] ** 2 + Tb[1] ** 2)