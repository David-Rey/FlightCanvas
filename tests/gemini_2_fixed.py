import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat, sin, cos, tan


# Set the ACADOS_SOURCE_DIR environment variable
# Use the correct path to your acados installation
#os.environ['ACADOS_SOURCE_DIR'] = '/home/david/Desktop/main/acados'

def create_rocket_model_variable_mass() -> AcadosModel:
    """
    Creates a 3DOF rocket model with variable mass.
    """
    model = AcadosModel()
    model.name = 'rocket_3dof_var_mass'

    # -----------------
    # PARAMETERS (as constants)
    # -----------------
    Iyy = 0.5  # Moment of inertia for pitch [kg*m^2]
    Isp = 200.0  # Specific impulse of the engine [s]
    g0 = 9.80665  # Standard gravity for Isp calculation [m/s^2]
    g = 9.81  # Local gravity

    # -----------------
    # STATES (x) - NOW WITH MASS
    # -----------------
    x_pos = SX.sym('x_pos')
    z_pos = SX.sym('z_pos')
    vx = SX.sym('vx')
    vz = SX.sym('vz')
    theta = SX.sym('theta')
    omega = SX.sym('omega')
    m = SX.sym('m')

    x = vertcat(x_pos, z_pos, vx, vz, theta, omega, m)

    # -----------------
    # CONTROLS (u)
    # -----------------
    T = SX.sym('T')  # Thrust
    tau = SX.sym('tau')  # Torque
    u_vec = vertcat(T, tau)

    # -----------------
    # STATE DERIVATIVES (x_dot)
    # -----------------
    x_dot_sym = SX.sym('x_dot_sym', x.shape)

    # -----------------
    # DYNAMICS (f_expl_expr)
    # -----------------
    # Note: acceleration now depends on the state 'm'
    f_expl_expr = vertcat(
        vx,
        vz,
        (-T * sin(theta)) / m,
        (T * cos(theta)) / m - g,
        omega,
        tau / Iyy,
        -T / (Isp * g0),
    )
    f_expl_expr = f_expl_expr


    # Set model attributes
    model.x = x
    model.u = u_vec
    model.xdot = x_dot_sym
    model.f_expl_expr = f_expl_expr
    model.f_impl_expr = f_expl_expr - x_dot_sym

    return model


def main():
    """
    Main function to set up and solve the fuel-optimal rocket landing OCP.
    """
    # -----------------
    # OCP SETUP
    # -----------------
    ocp = AcadosOcp()
    ocp.model = create_rocket_model_variable_mass()

    # Horizon parameters
    N = 200
    Tf = 6
    ocp.dims.N = N
    ocp.solver_options.tf = Tf

    ocp.solver_options.N_horizon = N

    # Rocket mass properties
    mass_wet = 10.0  # Initial wet mass [kg]
    mass_dry = 6.0  # Final dry mass [kg]

    # -----------------
    # COST FUNCTION (EXTERNAL - for non-linear objective)
    # -----------------
    # Set the cost type for all stages
    #ocp.cost.cost_type_0 = 'LINEAR_LS'
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    # Get model dimensions
    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    # Define cost dimensions
    ny_0 = nx + nu
    ny = nx + nu
    ny_e = nx

    # --- Define Weighting Matrices ---
    # To have NO running cost, we set the running cost weights to zero.
    Q= np.zeros((nx, nx))
    #R= np.zeros((nu, nu))
    R = np.diag([0.05, 0.01])
    W = np.block([[Q, np.zeros((nx, nu))], [np.zeros((nu, nx)), R]])

    # The terminal cost weight matrix remains large to enforce the final state.
    W_e = np.diag([500, 500, 500, 500, 100, 100, 0])  # High weights on final position/velocity

    # Set the weighting matrices.
    #ocp.cost.W_0 = W  # Zero cost for initial stage
    ocp.cost.W = W  # Zero cost for running stages
    ocp.cost.W_e = W_e  # High cost for terminal stage

    # --- Define Mapping Matrices (Vx, Vu) ---
    # These are still required for a complete problem formulation.
    #Vx_0 = np.zeros((ny_0, nx))
    #Vx_0[:nx, :nx] = np.eye(nx)
    #Vu_0 = np.zeros((ny_0, nu))
    #Vu_0[nx:, :nu] = np.eye(nu)
    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx:, :nu] = np.eye(nu)
    Vx_e = np.eye(ny_e)

    # Set mapping matrices
    #ocp.cost.Vx_0 = Vx_0
    #ocp.cost.Vu_0 = Vu_0
    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = Vx_e

    # --- Define References (y_ref) ---
    # The reference is to land at the origin (x=0, z=0, etc.) and use zero starship_control input.
    #ocp.cost.yref_0 = np.zeros(ny_0)
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # -----------------
    # CONSTRAINTS
    # -----------------
    # Initial state: 50m high, 20m downrange, with initial mass
    #x0 = np.array([100.0, 50.0, -5.0, -10.0, 0.0, 0.0, mass_wet, 10])
    x0_physical = np.array([50.0, 70.0, 0.0, -10.0, 0.0, 0.0, mass_wet])
    ocp.constraints.x0 = x0_physical

    # Set initial bounds: physical states are fixed, T_f is in a range.
    #ocp.constraints.lbx_0 = np.concatenate((x0_physical, np.array([Tf_min])))
    #ocp.constraints.ubx_0 = np.concatenate((x0_physical, np.array([Tf_max])))
    #ocp.constraints.idxbx_0 = np.array(np.arange(nx))

    # Control limits
    T_max = 2.0 * mass_wet * 9.81  # Max thrust
    T_min = 0.5 * mass_wet * 9.81  # Min thrust
    tau_max = 1.0
    ocp.constraints.lbu = np.array([T_min, -tau_max])
    ocp.constraints.ubu = np.array([T_max, tau_max])

    # Get the number of controls 'nu' directly from the model
    nu = ocp.model.u.size()[0]
    ocp.constraints.idxbu = np.arange(nu)

    # State limits: z >= 0, |theta| <= 45 deg, m >= m_dry
    angle_limit = np.pi / 4.0
    ocp.constraints.lbx = np.array([0, -angle_limit, mass_dry])  # z_pos, theta, m
    ocp.constraints.ubx = np.array([1e5, angle_limit, mass_wet])
    ocp.constraints.idxbx = np.array([1, 4, 6])  # Indices of z_pos, theta, and mass

    terminal_state_indices = np.array([7])

    # The desired final value for all these states is 0.
    # We set both the lower and upper bounds to 0 to enforce this exactly.
    #ocp.constraints.lbx_e = np.array([2])
    #ocp.constraints.ubx_e = np.array([20])
    #ocp.constraints.idxbx_e = terminal_state_indices

    # -----------------
    # SOLVER OPTIONS
    # -----------------
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.regularize_method = 'GERSHGORIN_LEVENBERG_MARQUARDT'  # CHANGE 5: Add regularization
    ocp.solver_options.levenberg_marquardt = 1e-2
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'

    ocp.solver_options.nlp_solver_max_iter = 200 # Default is often 50

    ## --- FIX (Optional): Add regularization for stubborn problems ---
    # ocp.solver_options.regularize_method = 'CONVEXIFY'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    # ocp.solver_options.nlp_solver_step_length = 0.9  # Try values between 0.5 and 1.0

    # -----------------
    # CREATE AND SOLVE
    # -----------------
    solver_json = 'acados_ocp_' + ocp.model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)

    # Initialize solver with a guess
    #for i in range(N + 1):
    #    x_guess = np.array([0,0,0,0,0,0,0, 10])
    #    acados_ocp_solver.set(i, 'x', x_guess)

    print("Solving fuel-optimal OCP...")
    status = acados_ocp_solver.solve()
    acados_ocp_solver.print_statistics()

    if status != 0:
        print(f"Solver failed with status {status}!")
    else:
        print("Solver succeeded!")

    # Get optimal trajectory
    nx = ocp.model.x.size()[0]
    # nu is already defined above
    x_traj = np.array([acados_ocp_solver.get(i, "x") for i in range(N + 1)])
    u_traj = np.array([acados_ocp_solver.get(i, "u") for i in range(N)])

    final_mass = x_traj[-1, 6]
    fuel_used = mass_wet - final_mass
    #Tf = x_traj[-1, 7]

    #print(x_traj)

    print("\n--- RESULTS ---")
    print(f"Initial Mass: {mass_wet:.2f} kg")
    print(f"Final Mass:   {final_mass:.2f} kg")
    print(f"Fuel Used:    {fuel_used:.2f} kg")
    print(f"Final Time: {Tf:.2f} s")

    print("\n--- Predicted Final State ---")
    print(f"Position: (x={x_traj[-1, 0]:.2f}, z={x_traj[-1, 1]:.2f}) m")
    print(f"Velocity: (vx={x_traj[-1, 2]:.2f}, vz={x_traj[-1, 3]:.2f}) m/s")

    # Optional: Plotting
    try:
        import matplotlib.pyplot as plt
        t = np.linspace(0, Tf, N + 1)

        plt.figure(figsize=(12, 10))
        # Plot 1: Trajectory
        plt.subplot(3, 1, 1)
        plt.plot(x_traj[:, 0], x_traj[:, 1], '-o', markersize=2)
        plt.title('Optimal Rocket Trajectory (Fuel Minimized)')
        plt.xlabel('Horizontal Position x [m]')
        plt.ylabel('Altitude z [m]')
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=2)
        plt.axvline(0, color='black', linewidth=1)
        plt.axis('equal')

        # Plot 2: Controls
        plt.subplot(3, 1, 2)
        plt.step(t[:-1], u_traj[:, 0], where='post', label='Thrust [N]')
        plt.step(t[:-1], u_traj[:, 1], where='post', label='Torque [Nm]')
        plt.title('Control Inputs')
        plt.xlabel('Time [s]')
        plt.grid(True)
        plt.axhline(y=T_max, color='r', linestyle='--', label=f'Max Thrust ({T_max:.1f} N)')
        plt.axhline(y=T_min, color='g', linestyle='--', label='Min Thrust')
        plt.legend()

        # Plot 3: Mass
        plt.subplot(3, 1, 3)
        plt.plot(t, x_traj[:, 6])
        plt.title('Mass vs. Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Mass [kg]')
        plt.grid(True)
        plt.axhline(mass_dry, color='r', linestyle='--', label=f'Dry Mass ({mass_dry} kg)')
        plt.legend()

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Skipping plot.")


if __name__ == '__main__':
    main()