import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


# Use the smooth spline generator from the previous code
def make_symbolic_smooth_spline_simple(n_waypoints, n_output_points=100, smoothness=0.3):
    """
    Simplified smooth spline using weighted control points.
    This approach ensures ALL segments are curved, including endpoints.
    """

    waypoints = ca.MX.sym("waypoints", 2, n_waypoints)

    # Generate evaluation points
    t_values = np.linspace(0, 1, n_output_points)

    def smooth_interpolation_1d(control_points, t_eval):
        """
        Smooth interpolation that creates curves everywhere
        """
        n = len(control_points)
        output_points = []

        for t_val in t_eval:
            result = control_points[0]

            # Find which interval and interpolate with cubic polynomials
            for i in range(n - 1):
                t_start = i / (n - 1)
                t_end = (i + 1) / (n - 1)

                # Check if t_val is in this interval
                in_interval = ca.logic_and(t_val >= t_start, t_val < t_end)
                if i == n - 2:  # Last interval includes endpoint
                    in_interval = ca.logic_or(in_interval, t_val >= t_end)

                # Local parameter
                s = (t_val - t_start) / (t_end - t_start)
                s = ca.fmax(0, ca.fmin(1, s))

                # Get control points for smooth curve
                p0 = control_points[i]
                p3 = control_points[i + 1]

                # Create intermediate control points for smoothness
                if i == 0:
                    # First segment: extrapolate backward
                    if n > 2:
                        direction = control_points[1] - control_points[0]
                        p_virtual = control_points[0] - direction * smoothness
                    else:
                        p_virtual = control_points[0]
                    p1 = control_points[0] + smoothness * (control_points[1] - p_virtual)
                else:
                    # Use previous point
                    p1 = control_points[i] + smoothness * (control_points[i + 1] - control_points[i - 1])

                if i == n - 2:
                    # Last segment: extrapolate forward
                    if n > 2:
                        direction = control_points[n - 1] - control_points[n - 2]
                        p_virtual = control_points[n - 1] + direction * smoothness
                    else:
                        p_virtual = control_points[n - 1]
                    p2 = control_points[i + 1] + smoothness * (p_virtual - control_points[i])
                else:
                    # Use next point
                    p2 = control_points[i + 1] + smoothness * (control_points[i] - control_points[i + 2])

                # Cubic Bezier interpolation
                t = s
                t2 = t * t
                t3 = t2 * t
                mt = 1 - t
                mt2 = mt * mt
                mt3 = mt2 * mt

                segment_value = mt3 * p0 + 3 * mt2 * t * p1 + 3 * mt * t2 * p2 + t3 * p3

                result = ca.if_else(in_interval, segment_value, result)

            output_points.append(result)

        return output_points

    # Apply to both coordinates
    x_smooth = smooth_interpolation_1d([waypoints[0, i] for i in range(n_waypoints)], t_values)
    y_smooth = smooth_interpolation_1d([waypoints[1, i] for i in range(n_waypoints)], t_values)

    # Combine results
    spline_points = ca.horzcat(*[ca.vertcat(x_smooth[i], y_smooth[i]) for i in range(n_output_points)])

    spline_func = ca.Function(
        "smooth_spline_generator",
        [waypoints],
        [spline_points],
        ["waypoints"],
        ["spline_points"]
    )

    return spline_func


# Distance calculation function (your existing working version)
def make_symbolic_linear_spline_distance(smooth_points):
    """
    Calculate distance from query point to a set of connected line segments
    smooth_points: 2 x n_points matrix of points on the spline
    """

    xq = ca.MX.sym("xq")
    yq = ca.MX.sym("yq")
    query_point = ca.vertcat(xq, yq)

    # Calculate distance to each line segment
    dists = []
    n_points = smooth_points.shape[1]

    for i in range(n_points - 1):
        p1 = smooth_points[:, i]
        p2 = smooth_points[:, i + 1]
        v = p2 - p1
        w = query_point - p1

        v_dot_v = ca.dot(v, v)
        t = ca.if_else(v_dot_v > 1e-12, ca.dot(w, v) / v_dot_v, 0)
        t_clamped = ca.fmax(0, ca.fmin(1, t))

        closest_point = p1 + t_clamped * v
        dist = ca.norm_2(query_point - closest_point)
        dists.append(dist)

    min_dist = dists[0]
    for d in dists[1:]:
        min_dist = ca.fmin(min_dist, d)

    distance_func = ca.Function(
        "spline_distance",
        [xq, yq],
        [min_dist],
        ["xq", "yq"],
        ["distance"]
    )

    return distance_func


# Create complete pipeline: waypoints → smooth spline → distance function
def create_complete_spline_distance_function(n_waypoints, n_spline_points=100, smoothness=0.3):
    """
    Complete pipeline that creates a distance function for a smooth spline
    """

    # Step 1: Create spline generator
    spline_generator = make_symbolic_smooth_spline_simple(n_waypoints, n_spline_points, smoothness)

    # Step 2: Create the complete function
    xq = ca.MX.sym("xq")
    yq = ca.MX.sym("yq")
    waypoints = ca.MX.sym("waypoints", 2, n_waypoints)
    query_point = ca.vertcat(xq, yq)

    # Generate smooth spline points
    smooth_points = spline_generator(waypoints)

    # Calculate distances to all segments
    dists = []
    for i in range(n_spline_points - 1):
        p1 = smooth_points[:, i]
        p2 = smooth_points[:, i + 1]
        v = p2 - p1
        w = query_point - p1

        v_dot_v = ca.dot(v, v)
        t = ca.if_else(v_dot_v > 1e-12, ca.dot(w, v) / v_dot_v, 0)
        t_clamped = ca.fmax(0, ca.fmin(1, t))

        closest_point = p1 + t_clamped * v
        dist = ca.norm_2(query_point - closest_point)
        dists.append(dist)

    min_dist = dists[0]
    for d in dists[1:]:
        min_dist = ca.fmin(min_dist, d)

    complete_func = ca.Function(
        "complete_spline_distance",
        [xq, yq, waypoints],
        [min_dist],
        ["xq", "yq", "waypoints"],
        ["distance"]
    )

    return complete_func


# Now create the contour plot with smooth splines
if __name__ == "__main__":
    waypoints = np.array([[0, 1, 3, 4],
                          [0, 2, 3, 0]])

    smoothness = 0.1

    print("Creating complete smooth spline distance function...")
    dist_function = create_complete_spline_distance_function(n_waypoints=4, n_spline_points=100, smoothness=smoothness)

    # Create grid for contour plot
    print("Setting up grid for contour plot...")
    x_min, x_max = -1, 5
    y_min, y_max = -1, 4
    resolution = 60

    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Evaluate distance function over the grid
    print("Evaluating distance function...")
    Z = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            try:
                distance = dist_function(X[i, j], Y[i, j], waypoints)
                Z[i, j] = float(distance)
            except Exception as e:
                Z[i, j] = np.nan

    # Create the contour plot
    print("Creating contour plot...")
    plt.figure(figsize=(14, 10))

    # Main contour plot
    plt.subplot(2, 2, (1, 2))
    contour = plt.contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.8)
    contour_lines = plt.contour(X, Y, Z, levels=15, colors='white', alpha=0.4, linewidths=0.8)

    # Add colorbar
    cbar = plt.colorbar(contour, shrink=0.8)
    cbar.set_label('Distance to Smooth Spline', rotation=270, labelpad=20)

    # Plot the original waypoints
    #plt.plot(waypoints[0, :], waypoints[1, :], 'ro-', linewidth=3, markersize=10,
    #         label='Original Waypoints', markerfacecolor='red', markeredgecolor='white', markeredgewidth=2)

    # Plot the actual smooth spline for reference
    spline_gen = make_symbolic_smooth_spline_simple(4, 100, smoothness)
    spline_result = spline_gen(waypoints)
    spline_points = np.array(spline_result).T
    plt.plot(spline_points[:, 0], spline_points[:, 1], 'w--', linewidth=3, alpha=0.9,
             label='Smooth Spline Path')

    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('2D Contour Plot: Distance to Smooth Spline\n(All segments curved)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Add a zoomed-in view around the spline
    plt.subplot(2, 2, 3)
    x_zoom = np.linspace(-0.5, 4.5, 40)
    y_zoom = np.linspace(-0.5, 3.5, 40)
    X_zoom, Y_zoom = np.meshgrid(x_zoom, y_zoom)
    Z_zoom = np.zeros_like(X_zoom)

    for i in range(40):
        for j in range(40):
            distance = dist_function(X_zoom[i, j], Y_zoom[i, j], waypoints)
            Z_zoom[i, j] = float(distance)

    plt.contourf(X_zoom, Y_zoom, Z_zoom, levels=20, cmap='plasma', alpha=0.8)
    plt.contour(X_zoom, Y_zoom, Z_zoom, levels=10, colors='white', alpha=0.5, linewidths=0.5)
    plt.plot(waypoints[0, :], waypoints[1, :], 'wo-', linewidth=2, markersize=8)
    plt.plot(spline_points[:, 0], spline_points[:, 1], 'w--', linewidth=2, alpha=0.8)
    plt.title('Zoomed View (Different Colormap)')
    plt.grid(True, alpha=0.3)

    # Show distance values along a line
    plt.subplot(2, 2, 4)
    y_slice = 1.5
    x_slice = np.linspace(-1, 5, 100)
    distances = []
    for x_val in x_slice:
        dist = dist_function(x_val, y_slice, waypoints)
        distances.append(float(dist))

    plt.plot(x_slice, distances, 'b-', linewidth=2, label=f'Distance along y={y_slice}')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Distance')
    plt.title('Distance Profile (Cross-section)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Print statistics
    print(f"\nContour Plot Statistics:")
    print(f"Distance range: {np.nanmin(Z):.3f} to {np.nanmax(Z):.3f}")
    print(f"Grid resolution: {resolution}x{resolution}")
    print(f"Spline uses {len(spline_points)} smooth points")
    print(f"Min distance in slice: {min(distances):.3f}")

    plt.show()

    # Verify the spline is actually smooth
    print(f"\nSpline verification:")
    print(f"Waypoints shape: {waypoints.shape if hasattr(waypoints, 'shape') else 'symbolic'}")
    print(f"Generated spline points: {spline_points.shape}")
    print(f"Spline path length: {np.sum(np.linalg.norm(np.diff(spline_points, axis=0), axis=1)):.2f}")