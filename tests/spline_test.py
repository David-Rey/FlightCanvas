#!/usr/bin/env python3
"""
Fixed Acados-based MPC for spline tracking with spline parameter as state.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import SX, vertcat, sumsqr, sqrt, fmax


class SplineProgressionMPC:
    def __init__(self, waypoints, tube_radius=1.5, T_horizon=3.0, N=30):
        """
        Initialize the spline tracking MPC with spline parameter as state.

        Args:
            waypoints: List of (x, y) waypoints defining the path
            tube_radius: Radius of constraint tube around spline
            T_horizon: Prediction horizon in seconds
            N: Number of discretization steps
        """
        self.waypoints = np.array(waypoints)
        self.tube_radius = tube_radius
        self.T_horizon = T_horizon
        self.N = N
        self.dt = T_horizon / N

        # Create spline from waypoints
        self._create_spline()

        # Setup and solve the OCP
        try:
            self.ocp = self._setup_ocp()
            self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp_spline.json')
            print("Acados solver initialized successfully")
        except Exception as e:
            print(f"Error initializing Acados solver: {e}")
            raise

    def _create_spline(self):
        """Create cubic spline from waypoints."""
        # Parameterize by cumulative distance along path
        distances = np.zeros(len(self.waypoints))
        for i in range(1, len(self.waypoints)):
            distances[i] = distances[i - 1] + np.linalg.norm(
                self.waypoints[i] - self.waypoints[i - 1]
            )

        # Create splines for x and y coordinates
        self.spline_x = CubicSpline(distances, self.waypoints[:, 0])
        self.spline_y = CubicSpline(distances, self.waypoints[:, 1])
        self.max_distance = distances[-1]
        print(f"Spline created with total length: {self.max_distance:.2f} m")

    def _get_spline_reference(self, s):
        """Get spline position and derivatives at parameter s."""
        # Clamp s to valid range
        s = np.clip(s, 0, self.max_distance)

        x_ref = self.spline_x(s)
        y_ref = self.spline_y(s)
        dx_ref = self.spline_x(s, 1)  # First derivative
        dy_ref = self.spline_y(s, 1)  # First derivative

        return x_ref, y_ref, dx_ref, dy_ref

    def _setup_ocp(self):
        """Setup the optimal control problem with spline parameter as state."""
        ocp = AcadosOcp()

        # Model dimensions
        nx = 6  # [px, py, vx, vy, s, vs] - added spline parameter and velocity
        nu = 3  # [ax, ay, as] - added spline acceleration
        np_param = 4  # [spline_x_ref, spline_y_ref, spline_dx_ref, spline_dy_ref]

        # Symbolic variables
        x = SX.sym('x', nx)
        u = SX.sym('u', nu)
        p = SX.sym('p', np_param)  # Parameters for spline reference at current s

        # Extract states
        px, py, vx, vy, s, vs = x[0], x[1], x[2], x[3], x[4], x[5]
        ax, ay, as_ = u[0], u[1], u[2]  # as is reserved keyword

        # Extract spline reference parameters
        spline_x_ref = p[0]
        spline_y_ref = p[1]
        spline_dx_ref = p[2]
        spline_dy_ref = p[3]

        # System dynamics
        f_expl = vertcat(
            vx,  # dpx/dt = vx
            vy,  # dpy/dt = vy
            ax,  # dvx/dt = ax
            ay,  # dvy/dt = ay
            vs,  # ds/dt = vs (velocity along spline)
            as_  # dvs/dt = as (acceleration along spline)
        )

        # FIXED: More balanced cost function weights
        W_tracking = 50.0  # Position tracking weight
        W_vs = 10.0  # Spline velocity tracking weight
        W_control = 1.0  # Control effort weight
        W_progress = 200.0  # Spline progress weight (reduced)

        # Target spline velocity
        vs_target = 2.0

        # Stage cost - SIMPLIFIED: Remove hard tube constraint, use only tracking
        # Position tracking cost
        pos_error = vertcat(px - spline_x_ref, py - spline_y_ref)
        cost_tracking = W_tracking * sumsqr(pos_error)

        # Spline velocity tracking
        #cost_vs = W_vs * (vs - vs_target) ** 2

        #squared_distance = (px - spline_x_ref) ** 2 + (py - spline_y_ref) ** 2

        # 2. Define the squared radius of the "zero-cost" tube
        #deadband_radius_sq = self.tube_radius ** 2

        # 3. Calculate the violation (how much we are outside the tube)
        #    This is 0 if inside the tube, positive otherwise.
        #violation = fmax(0, squared_distance - deadband_radius_sq)

        # 4. The new tracking cost penalizes only this violation.
        #    NOTE: A much higher weight is often needed for this to be effective.
        #W_violation = 0.1  # High penalty for leaving the tube
        #cost_tracking = W_violation * violation

        # Spline velocity tracking
        cost_vs = W_vs * (vs - vs_target) ** 2

        # Control effort
        cost_control = W_control * sumsqr(u)

        # Progress reward (encourage forward motion)
        cost_progress = -W_progress * vs  # Negative cost = reward

        #stage_cost = cost_tracking + cost_vs + cost_control + cost_progress

        # Terminal cost: position tracking + spline velocity tracking
        #terminal_cost = cost_tracking + W_vs * (vs - vs_target) ** 2

        stage_cost = cost_control
        terminal_cost = cost_tracking + cost_progress

        # Create model
        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.p = p
        model.name = "spline_progression_model"

        # Set model
        ocp.model = model

        # Dimensions
        ocp.dims.N = self.N

        # Cost function setup - using EXTERNAL cost
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.model.cost_expr_ext_cost = stage_cost
        ocp.model.cost_expr_ext_cost_e = terminal_cost

        # FIXED: Add reasonable bounds for all states and controls
        max_acceleration = 4.0  # Reduced from 6.0
        max_velocity = 8.0  # Reasonable velocity bound
        max_position = 50.0  # Reasonable position bound
        max_spline_velocity = 4.0
        max_spline_acceleration = 2.0  # Reduced from 3.0

        # Control bounds
        ocp.constraints.lbu = np.array([-max_acceleration, -max_acceleration, -max_spline_acceleration])
        ocp.constraints.ubu = np.array([max_acceleration, max_acceleration, max_spline_acceleration])
        ocp.constraints.idxbu = np.array([0, 1, 2])

        # FIXED: Add bounds for all states
        ocp.constraints.lbx = np.array([
            -max_position,  # px lower bound
            -max_position,  # py lower bound
            -max_velocity,  # vx lower bound
            -max_velocity,  # vy lower bound
            0.0,  # s lower bound
            -max_spline_velocity  # vs lower bound
        ])
        ocp.constraints.ubx = np.array([
            max_position,  # px upper bound
            max_position,  # py upper bound
            max_velocity,  # vx upper bound
            max_velocity,  # vy upper bound
            self.max_distance,  # s upper bound
            max_spline_velocity  # vs upper bound
        ])
        ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5])  # Bound all states

        # FIXED: Use soft constraint for tube instead of hard constraint
        # This is now handled in the cost function only

        # Initial state constraint (will be updated)
        ocp.constraints.x0 = np.zeros(nx)

        # Initialize parameter values
        ocp.parameter_values = np.zeros(np_param)

        # FIXED: More conservative solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.tf = self.T_horizon
        ocp.solver_options.print_level = 1  # Enable some output for debugging
        ocp.solver_options.nlp_solver_max_iter = 100  # Reduced from 150
        ocp.solver_options.qp_solver_iter_max = 50  # Reduced from 100
        ocp.solver_options.nlp_solver_tol_stat = 1e-4  # Less strict
        ocp.solver_options.nlp_solver_tol_eq = 1e-4  # Less strict
        ocp.solver_options.nlp_solver_tol_ineq = 1e-4  # Less strict

        return ocp

    def solve(self, current_state_2d, current_spline_param, current_spline_velocity=0.0):
        """
        Solve the MPC problem.

        Args:
            current_state_2d: Current 2D state [px, py, vx, vy]
            current_spline_param: Current parameter along spline
            current_spline_velocity: Current velocity along spline

        Returns:
            Optimal control input [ax, ay, as]
        """
        # Construct full state vector
        current_state = np.concatenate([
            current_state_2d,
            [current_spline_param, current_spline_velocity]
        ])

        # Set initial state
        self.solver.set(0, "lbx", current_state)
        self.solver.set(0, "ubx", current_state)

        # FIXED: Predict spline parameter evolution and set consistent references
        for k in range(self.N + 1):  # Include terminal stage
            # Predict where s will be at stage k
            # Simple prediction: assume constant velocity
            if k == 0:
                s_k = current_spline_param
            else:
                # Rough prediction based on target velocity
                dt_pred = k * self.dt
                s_k = current_spline_param + 2.0 * dt_pred  # Assume target vs=2.0
                s_k = np.clip(s_k, 0, self.max_distance)

            # Get spline reference at predicted s
            x_ref, y_ref, dx_ref, dy_ref = self._get_spline_reference(s_k)

            params = np.array([x_ref, y_ref, dx_ref, dy_ref])
            self.solver.set(k, "p", params)

        # FIXED: Initialize with reasonable guess
        # Initialize states and controls with smooth progression
        for k in range(self.N):
            # Initialize states
            if k == 0:
                self.solver.set(k, "x", current_state)
            else:
                # Simple forward prediction
                dt_pred = k * self.dt
                s_pred = current_spline_param + 2.0 * dt_pred
                s_pred = np.clip(s_pred, 0, self.max_distance)
                x_pred, y_pred, _, _ = self._get_spline_reference(s_pred)

                state_pred = np.array([
                    x_pred,  # px
                    y_pred,  # py
                    2.0,  # vx (reasonable guess)
                    0.0,  # vy (reasonable guess)
                    s_pred,  # s
                    2.0  # vs (target velocity)
                ])
                self.solver.set(k, "x", state_pred)

            # Initialize controls to zero
            self.solver.set(k, "u", np.zeros(3))

        # Set terminal state
        dt_pred = self.N * self.dt
        s_terminal = current_spline_param + 2.0 * dt_pred
        s_terminal = np.clip(s_terminal, 0, self.max_distance)
        x_terminal, y_terminal, _, _ = self._get_spline_reference(s_terminal)

        terminal_state = np.array([
            x_terminal, y_terminal, 2.0, 0.0, s_terminal, 2.0
        ])
        self.solver.set(self.N, "x", terminal_state)

        # Solve
        status = self.solver.solve()

        if status != 0:
            print(f"Solver failed with status {status}")
            # Return safe fallback control
            return np.array([0.0, 0.0, 1.0])  # Small forward progress

        # Get optimal control
        u_opt = self.solver.get(0, "u")
        return u_opt

    def get_prediction(self):
        """Get the predicted trajectory."""
        trajectory = []
        for k in range(self.N + 1):
            x_pred = self.solver.get(k, "x")
            trajectory.append(x_pred)
        return np.array(trajectory)

    def get_control_sequence(self):
        """Get the predicted control sequence."""
        controls = []
        for k in range(self.N):
            u_pred = self.solver.get(k, "u")
            controls.append(u_pred)
        return np.array(controls)

    def plot_spline_with_tube(self, num_points=200):
        """Plot the reference spline with constraint tube."""
        s_vals = np.linspace(0, self.max_distance, num_points)
        spline_points = []

        for s in s_vals:
            x_ref, y_ref, dx_ref, dy_ref = self._get_spline_reference(s)
            spline_points.append([x_ref, y_ref, dx_ref, dy_ref])

        spline_points = np.array(spline_points)

        plt.figure(figsize=(12, 8))

        # Plot spline centerline
        plt.plot(spline_points[:, 0], spline_points[:, 1], 'b-',
                 linewidth=3, label='Reference Spline')

        # Plot constraint tube
        for i in range(0, len(spline_points), max(1, len(spline_points) // 50)):
            x_center, y_center = spline_points[i, 0], spline_points[i, 1]
            dx, dy = spline_points[i, 2], spline_points[i, 3]

            # Normal vector to spline (perpendicular)
            if dx ** 2 + dy ** 2 > 1e-6:
                norm = np.sqrt(dx ** 2 + dy ** 2)
                nx, ny = -dy / norm, dx / norm  # 90 degree rotation

                # Tube boundaries
                x_inner = x_center - self.tube_radius * nx
                y_inner = y_center - self.tube_radius * ny
                x_outer = x_center + self.tube_radius * nx
                y_outer = y_center + self.tube_radius * ny

                plt.plot([x_inner, x_outer], [y_inner, y_outer], 'r-', alpha=0.3, linewidth=0.5)

        # Add circle patches for clearer tube visualization
        theta = np.linspace(0, 2 * np.pi, 100)
        for i in range(0, len(spline_points), max(1, len(spline_points) // 20)):
            x_center, y_center = spline_points[i, 0], spline_points[i, 1]
            x_circle = x_center + self.tube_radius * np.cos(theta)
            y_circle = y_center + self.tube_radius * np.sin(theta)
            plt.plot(x_circle, y_circle, 'r--', alpha=0.2, linewidth=0.5)

        plt.plot(self.waypoints[:, 0], self.waypoints[:, 1], 'go',
                 markersize=8, label='Waypoints')
        plt.grid(True)
        plt.axis('equal')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title(f'Reference Spline with Constraint Tube (r={self.tube_radius}m)')
        plt.legend()
        plt.show()


def simulate_spline_progression():
    """Simulate the spline progression tracking controller."""
    # Define waypoints for a more complex curved path
    waypoints = [
        [0, 0],
        [3, 2],
        [6, 4],
        [10, 5],
        [14, 4],
        [17, 1],
        [20, -1],
        [22, -3],
        [24, -2],
        [26, 0]
    ]

    # Create MPC controller with more conservative settings
    tube_radius = 2.0
    mpc = SplineProgressionMPC(waypoints, tube_radius=tube_radius, T_horizon=2.0, N=15)  # Shorter horizon

    # Plot the reference spline with tube
    mpc.plot_spline_with_tube()

    # Simulation parameters
    dt_sim = 0.05  # Simulation time step
    T_sim = 15.0  # Total simulation time

    # FIXED: Start closer to the spline
    pos_init = np.array([0.2, -1.1])  # Smaller offset
    vel_init = np.array([1.0, 0.1])  # Smaller initial velocity
    state_2d = np.concatenate([pos_init, vel_init])
    spline_param = 0.0
    spline_velocity = 1.0  # Start with some forward velocity

    # Storage for results
    times = []
    states_2d = []
    spline_params = []
    spline_velocities = []
    controls = []
    spline_refs = []
    predictions = []

    time = 0.0
    step = 0

    print("Starting spline progression simulation...")
    print(f"Initial position: ({pos_init[0]:.2f}, {pos_init[1]:.2f})")
    print(f"Tube radius: {tube_radius:.2f}m")

    while time < T_sim and spline_param < mpc.max_distance:
        try:
            # Solve MPC
            u_opt = mpc.solve(state_2d, spline_param, spline_velocity)

            # Get prediction for visualization
            if step % 20 == 0:  # Every 20 steps
                pred = mpc.get_prediction()
                predictions.append(pred.copy())

            # Store results
            times.append(time)
            states_2d.append(state_2d.copy())
            spline_params.append(spline_param)
            spline_velocities.append(spline_velocity)
            controls.append(u_opt.copy())

            # Get current spline reference
            x_ref, y_ref, dx_ref, dy_ref = mpc._get_spline_reference(spline_param)
            spline_refs.append([x_ref, y_ref])

            # Simulate system (Euler integration)
            # 2D dynamics
            state_2d_dot = np.array([state_2d[2], state_2d[3], u_opt[0], u_opt[1]])
            state_2d = state_2d + state_2d_dot * dt_sim

            # Spline parameter dynamics
            spline_velocity = spline_velocity + u_opt[2] * dt_sim
            spline_param = spline_param + spline_velocity * dt_sim

            # Clamp spline parameter to valid range
            spline_param = np.clip(spline_param, 0, mpc.max_distance)

            time += dt_sim
            step += 1

            if step % 100 == 0:  # Print every 5 seconds
                pos_error = np.linalg.norm(state_2d[:2] - np.array([x_ref, y_ref]))
                speed = np.linalg.norm(state_2d[2:])
                print(f"Time: {time:.1f}s, Pos: ({state_2d[0]:.2f}, {state_2d[1]:.2f}), "
                      f"Spline: {spline_param:.2f}/{mpc.max_distance:.2f}, "
                      f"Error: {pos_error:.3f}m, Speed: {speed:.2f}m/s, vs: {spline_velocity:.2f}m/s")

        except Exception as e:
            print(f"Error at time {time:.2f}: {e}")
            break

    print(f"\nSimulation completed!")
    print(
        f"Final spline progress: {spline_params[-1]:.2f}/{mpc.max_distance:.2f} ({100 * spline_params[-1] / mpc.max_distance:.1f}%)")

    # Convert to arrays and create basic plots
    times = np.array(times)
    states_2d = np.array(states_2d)
    spline_params = np.array(spline_params)
    spline_velocities = np.array(spline_velocities)
    controls = np.array(controls)
    spline_refs = np.array(spline_refs)

    # Basic trajectory plot
    plt.figure(figsize=(12, 8))
    plt.plot(spline_refs[:, 0], spline_refs[:, 1], 'b--', linewidth=2, label='Reference', alpha=0.7)
    plt.plot(states_2d[:, 0], states_2d[:, 1], 'r-', linewidth=2, label='Actual Trajectory')
    plt.plot(mpc.waypoints[:, 0], mpc.waypoints[:, 1], 'go', markersize=6, label='Waypoints')
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Trajectory Tracking Results')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    simulate_spline_progression()