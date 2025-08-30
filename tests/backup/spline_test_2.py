import casadi as ca
import numpy as np


def make_symbolic_spline_distance(n_spline_waypoints, degree=3, n_samples=100):
    """
    Returns a true CasADi function f(xq, yq, spline_waypoints) -> distance.
    This function is fully symbolic and can be used for optimization.
    """

    # --- Step 1: Define Symbolic Variables ---
    xq = ca.MX.sym("xq")
    yq = ca.MX.sym("yq")
    query_point = ca.vertcat(xq, yq)
    spline_w = ca.MX.sym("spline_w", 2, n_spline_waypoints)

    # --- Step 2: Create a Symbolic B-spline Evaluator ---
    # Generate the knot vector for a standard clamped spline
    n_knots = n_spline_waypoints + degree + 1
    knots = np.concatenate([
        np.zeros(degree),
        np.linspace(0, 1, n_spline_waypoints - degree + 1),
        np.ones(degree)
    ])

    # Define a symbolic parameter 'u' that goes from 0 to 1
    u = ca.MX.sym("u")

    # **THE FIX:** Use the correct bspline function signature
    # The bspline function expects: bspline(u, control_points, knots, derivatives, degree, options)
    # For a 2D spline, we need to handle x and y coordinates separately or use the matrix form

    # Create the spline point using the correct bspline signature
    # We need to transpose spline_w to have control points as columns
    spline_point = ca.bspline(u, spline_w.T, ca.DM(knots), [[0]], degree, {})

    # Create a reusable CasADi Function to evaluate the spline
    spline_evaluator = ca.Function(
        "spline_evaluator",
        [u, spline_w],
        [spline_point],
        ["u", "waypoints"],
        ["pos"]
    )

    # --- Step 3: Sample the Spline Symbolically ---
    u_samples = np.linspace(0, 1, n_samples)

    # Evaluate spline at all sample points
    sample_points = []
    for u_val in u_samples:
        point = spline_evaluator(u_val, spline_w)
        sample_points.append(point)

    # Stack the points horizontally to create a polyline
    polyline = ca.horzcat(*sample_points)

    # --- Step 4: Calculate Distance to the Symbolic Polyline ---
    dists = []
    for i in range(n_samples - 1):
        p1 = polyline[:, i]
        p2 = polyline[:, i + 1]
        v = p2 - p1
        w = query_point - p1
        t = ca.dot(w, v) / (ca.dot(v, v) + 1e-9)
        t_clamped = ca.fmax(0, ca.fmin(1, t))
        closest_point = p1 + t_clamped * v
        dist = ca.norm_2(query_point - closest_point)
        dists.append(dist)

    min_dist = dists[0]
    for d in dists[1:]:
        min_dist = ca.fmin(min_dist, d)

    # --- Step 5: Build and Return the Final CasADi Function ---
    f = ca.Function(
        "symbolic_spline_distance",
        [xq, yq, spline_w],
        [min_dist],
        ["xq", "yq", "spline_waypoints"],
        ["distance"]
    )
    return f


# --- Example Usage ---
waypoints = np.array([[0, 1, 3, 4],
                      [0, 2, 3, 0]])

# Build the fully symbolic distance function
dist_fun_symbolic = make_symbolic_spline_distance(n_spline_waypoints=4, n_samples=100)
print("Symbolic function output:", dist_fun_symbolic(2.5, 1.0, waypoints))