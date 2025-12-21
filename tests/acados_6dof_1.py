import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import casadi as ca
import scipy as sp
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
    Creates a 6DOF rocket model.
    """
    model = AcadosModel()
    model.name = 'rocket'

    ### Rocket Parameters
    g = 9.81  # m/s^2
    Isp = 300  # s
    Lr = 4.5  # m
    Lh = 50  # m
    Lcm = 20  # m

    # Moment of Inertia
    m = ca.MX.sym('m', 1)

    Ixx = 1 / 4 * m[0] * Lr ** 2 + 1 / 12 * m[0] * Lh ** 2
    Iyy = 1 / 4 * m[0] * Lr ** 2 + 1 / 12 * m[0] * Lh ** 2
    Izz = 1 / 2 * m[0] * Lr ** 2

    epsilon_m = 1e-7
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

    # Control
    Tb = ca.MX.sym('Tb', 3)

    # Derivatives
    m_dot = -1 / (9.81 * Isp) * ca.norm_2(Tb)
    r_dot = v
    v_dot = (1 / m[0] * dir_cosine_casadi(q) @ Tb) + g_vec
    q_dot = 1 / 2 * omega_casadi(w) @ q
    w_dot = Jb_inv @ (ca.cross(rT, Tb) - ca.cross(w, Jb @ w))

    x = ca.vertcat(m, r, v, q, w,)
    f_expl_expr = ca.vertcat(m_dot, r_dot, v_dot, q_dot, w_dot)
    u = ca.vertcat(Tb)

    x_dot_sym = ca.MX.sym('x_dot_sym', x.shape)

    # Set model attributes
    model.x = x
    model.u = u
    model.xdot = x_dot_sym
    model.f_expl_expr = f_expl_expr
    model.f_impl_expr = f_expl_expr - x_dot_sym

    # Constraints
    quat_con = ca.norm_2(q) - 1  # quaterion
    thrust_con = ca.norm_2(Tb)  # thrust
    #gimbal_con = ca.dot(e3, Tb) / (ca.norm_2(Tb) + epsilon)  # gimble angle
    tilt_con = 1 - 2 *(q[2] ** 2 + q[3] ** 2)  # max tilt
    angular_rate_con = ca.dot(w, w)  # max angular rate

    delta_max_val = 15.0  # degrees
    tan_delta_sq = np.tan(np.deg2rad(delta_max_val)) ** 2
    gimbal_con = Tb[2] ** 2 * tan_delta_sq - (Tb[0] ** 2 + Tb[1] ** 2)

    model.con_h_expr = ca.vertcat(
        quat_con,
        thrust_con,
        gimbal_con,
        tilt_con,
        angular_rate_con
    )

    return model


def main():

    ocp = AcadosOcp()
    ocp.model = create_rocket_model()

    # --- Horizon and Solver Settings ---
    N = 150
    Tf = 12.0 # Time horizon in seconds
    ocp.dims.N = N
    ocp.solver_options.tf = Tf

    # --- Get dimensions ---
    nx = ocp.model.x.size1()
    nu = ocp.model.u.size1()
    ny = nx + nu  # Cost dimension for stage cost

    # --- Initial State ---
    m0 = 500.0
    r0 = np.array([5.0, 100.0, 100.0])
    v0 = np.array([5.0, -10.0, -25.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    w0 = np.array([0.0, 0.0, 0.0])
    x0 = np.concatenate([np.array([m0]), r0, v0, q0, w0])

    # --- Cost Function ---
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    # State and starship_control penalty matrices
    Q = np.diag([
        0,              # mass
        5.0, 5.0, 10.0, # r_x, r_y, r_z
        2.0, 2.0, 5.0,  # v_x, v_y, v_z
        0.1, 0.1, 0.1, 0.1, # q (penalize deviation from reference)
        0.1, 0.1, 0.1   # w_x, w_y, w_z
    ])
    R = np.diag([0.001, 0.001, 0.001]) # Penalty on thrust starship_control

    # Terminal state penalty matrix
    Q_e = np.diag([
        0,                # mass_e
        50, 50, 200,   # r_xe, r_ye, r_ze
        50, 50, 400,   # v_xe, v_ye, v_ze
        10, 10, 10, 10, # q_e
        10, 10, 10     # w_e
    ])

    # Set weighting matrices
    ocp.cost.W = sp.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    # Set up mapping matrices
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:, :nu] = np.eye(nu)

    ocp.cost.Vx_e = np.eye(nx)

    # --- Reference Trajectory (Target State) ---
    # Target state for terminal cost
    yref_e = np.array([
        0,  # Final mass doesn't matter
        0, 0, 0,  # Land at origin
        0, 0, 0,  # Zero velocity
        1, 0, 0, 0,  # Upright attitude
        0, 0, 0  # Zero angular rate
    ])

    # Target for stage cost (state part is yref_e, starship_control part is zero)
    yref = np.concatenate([yref_e, np.zeros(nu)])

    ocp.cost.yref = yref
    ocp.cost.yref_e = yref_e

    # --- Constraints ---
    ocp.constraints.x0 = x0
    dry_mass = 300

    ocp.constraints.idxbx = np.array([0])  # Index of mass in state vector x is 0
    ocp.constraints.lbx = np.array([dry_mass]) # Lower bound on mass
    ocp.constraints.ubx = np.array([m0])      # Upper bound on mass

    T_min = 2000
    T_max = 7000
    delta_max = 15  # max gimbal angle (deg)
    eta_max = 20  # max tilt angle (deg)
    omega_max = 2  # max angular rate (rad/s)

    inf = 1e9  # inf

    '''
    ocp.constraints.lh = np.array([
        0,  # Lower bound for ||q||-1 is 0
        T_min,  # Lower bound for ||Tb|| is T_min
        np.cos(np.deg2rad(delta_max)),
        np.cos(np.deg2rad(eta_max)),
        0
    ])
    ocp.constraints.uh = np.array([
        0,  # Upper bound for ||q||-1 is 0
        T_max,  # Upper bound for ||Tb|| is T_max
        inf,
        inf,
        omega_max
    ])
    '''
    ocp.constraints.lh = np.array([
        0,      # Lower bound for ||q||-1 is 0
        T_min,  # Lower bound for ||Tb|| is T_min
        0,      # Lower bound for new gimbal_con is 0
        np.cos(np.deg2rad(eta_max)),
        0
    ])
    ocp.constraints.uh = np.array([
        0,      # Upper bound for ||q||-1 is 0
        T_max,  # Upper bound for ||Tb|| is T_max
        inf,    # Upper bound for new gimbal_con is inf
        inf,
        omega_max**2
    ])
    ocp.constraints.idxsh = np.array([0, 1, 2, 3, 4])

    penalty_weight = 1e4

    # Penalty on lower bound slack variables (zl)
    ocp.cost.zl = np.array([
        penalty_weight * 1e2,
        penalty_weight,  # Corresponds to idxsh[0] = 1 (thrust_con lower)
        penalty_weight,  # Corresponds to idxsh[1] = 2 (gimbal_con lower)
        penalty_weight,  # Corresponds to idxsh[2] = 3 (tilt_con lower)
        0  # Corresponds to idxsh[3] = 4 (angular_rate_con lower, not needed)
    ])

    # Penalty on upper bound slack variables (zu)
    ocp.cost.zu = np.array([
        penalty_weight * 1e2,
        penalty_weight,  # Corresponds to idxsh[0] = 1 (thrust_con upper)
        0,  # Corresponds to idxsh[1] = 2 (gimbal_con upper, not needed)
        0,  # Corresponds to idxsh[2] = 3 (tilt_con upper, not needed)
        penalty_weight  # Corresponds to idxsh[3] = 4 (angular_rate_con upper)
    ])

    # The Z penalties are for the rate of change of the slack variables.
    # It's usually enough to just penalize the magnitude (zl, zu).
    ocp.cost.Zl = np.zeros_like(ocp.cost.zl)
    ocp.cost.Zu = np.zeros_like(ocp.cost.zu)

    # --- Solver Configuration ---
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    #ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.regularize_method = 'GERSHGORIN_LEVENBERG_MARQUARDT'
    ocp.solver_options.levenberg_marquardt = 1e-1
    #ocp.solver_options.tol = 1e-3
    ocp.solver_options.nlp_solver_max_iter = 20

    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'

    # Disable warm-starting the QP solver as a diagnostic step.
    # 0: disabled, 1: enabled (default)
    #ocp.solver_options.qp_solver_warm_start = 0

    # --- Create and Run Solver ---
    json_file = 'acados_ocp_rocket.json'
    #ocp_solver = AcadosOcpSolver(ocp, json_file=json_file)
    ocp_solver = AcadosOcpSolver(ocp)

    init_x = np.zeros((N + 1, nx))
    for i in range(N + 1):
        init_x[i, :] = x0 + (yref_e - x0) * (i / N)

    init_u = np.zeros((N, nu))
    init_u[:, 2] = 400  # Small vertical thrust component

    for i in range(N):
        ocp_solver.set(i, "x", init_x[i, :])
        ocp_solver.set(i, "u", init_u[i, :])
    ocp_solver.set(N, "x", init_x[N, :])

    status = ocp_solver.solve()

    # --- Get timing statistics ---
    total_time = ocp_solver.get_stats('time_tot')
    print(f"Solver finished in {total_time * 1000:.3f} ms.")

    ocp_solver.print_statistics()

    if status != 0:
        print(f'Solver failed with status {status}!')
    else:
        print('Optimal solution found!')

    # --- Plotting ---
    # Extract solution
    simX = np.zeros((N + 1, nx))
    simU = np.zeros((N, nu))
    for i in range(N):
        simX[i, :] = ocp_solver.get(i, "x")
        simU[i, :] = ocp_solver.get(i, "u")
    simX[N, :] = ocp_solver.get(N, "x")

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

    fig.savefig('rocket_trajectory_3d.png')
    fig2.savefig('rocket_states.png')
    fig3.savefig('rocket_controls.png')

    plt.close('all') # Close all figures to free up memory


if __name__ == '__main__':
    main()
