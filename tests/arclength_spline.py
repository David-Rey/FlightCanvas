import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


def make_symbolic_smooth_spline_simple(n_waypoints, n_output_points=100, smoothness=0.3):
    """
    Smooth spline generator (same as before)
    """
    waypoints = ca.MX.sym("waypoints", 2, n_waypoints)
    t_values = np.linspace(0, 1, n_output_points)

    def smooth_interpolation_1d(control_points, t_eval):
        n = len(control_points)
        output_points = []

        for t_val in t_eval:
            result = control_points[0]

            for i in range(n - 1):
                t_start = i / (n - 1)
                t_end = (i + 1) / (n - 1)

                in_interval = ca.logic_and(t_val >= t_start, t_val < t_end)
                if i == n - 2:
                    in_interval = ca.logic_or(in_interval, t_val >= t_end)

                s = (t_val - t_start) / (t_end - t_start)
                s = ca.fmax(0, ca.fmin(1, s))

                p0 = control_points[i]
                p3 = control_points[i + 1]

                if i == 0:
                    if n > 2:
                        direction = control_points[1] - control_points[0]
                        p_virtual = control_points[0] - direction * smoothness
                    else:
                        p_virtual = control_points[0]
                    p1 = control_points[0] + smoothness * (control_points[1] - p_virtual)
                else:
                    p1 = control_points[i] + smoothness * (control_points[i + 1] - control_points[i - 1])

                if i == n - 2:
                    if n > 2:
                        direction = control_points[n - 1] - control_points[n - 2]
                        p_virtual = control_points[n - 1] + direction * smoothness
                    else:
                        p_virtual = control_points[n - 1]
                    p2 = control_points[i + 1] + smoothness * (p_virtual - control_points[i])
                else:
                    p2 = control_points[i + 1] + smoothness * (control_points[i] - control_points[i + 2])

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

    x_smooth = smooth_interpolation_1d([waypoints[0, i] for i in range(n_waypoints)], t_values)
    y_smooth = smooth_interpolation_1d([waypoints[1, i] for i in range(n_waypoints)], t_values)

    spline_points = ca.horzcat(*[ca.vertcat(x_smooth[i], y_smooth[i]) for i in range(n_output_points)])

    spline_func = ca.Function(
        "smooth_spline_generator",
        [waypoints],
        [spline_points],
        ["waypoints"],
        ["spline_points"]
    )

    return spline_func


def create_combined_spline_functions(n_waypoints, n_spline_points=100, smoothness=0.3):
    """
    Creates both distance and arc length functions in one go for efficiency

    Returns:
        tuple: (distance_func, arc_length_func, total_length_func)
    """

    spline_generator = make_symbolic_smooth_spline_simple(n_waypoints, n_spline_points, smoothness)

    xq = ca.MX.sym("xq")
    yq = ca.MX.sym("yq")
    waypoints = ca.MX.sym("waypoints", 2, n_waypoints)
    query_point = ca.vertcat(xq, yq)

    smooth_points = spline_generator(waypoints)

    # Calculate cumulative arc lengths
    arc_lengths = [0]
    for i in range(n_spline_points - 1):
        p1 = smooth_points[:, i]
        p2 = smooth_points[:, i + 1]
        segment_length = ca.norm_2(p2 - p1)
        arc_lengths.append(arc_lengths[-1] + segment_length)

    # Find closest point
    closest_arc_length = arc_lengths[0]
    min_distance = ca.inf

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

        arc_length_at_closest = arc_lengths[i] + t_clamped * ca.norm_2(v)

        is_closer = dist < min_distance
        closest_arc_length = ca.if_else(is_closer, arc_length_at_closest, closest_arc_length)
        min_distance = ca.if_else(is_closer, dist, min_distance)

    # Create functions
    distance_func = ca.Function(
        "spline_distance", [xq, yq, waypoints], [min_distance],
        ["xq", "yq", "waypoints"], ["distance"]
    )

    arc_length_func = ca.Function(
        "spline_arc_length", [xq, yq, waypoints], [closest_arc_length],
        ["xq", "yq", "waypoints"], ["arc_length"]
    )

    total_length_func = ca.Function(
        "spline_total_length", [waypoints], [arc_lengths[-1]],
        ["waypoints"], ["total_length"]
    )

    return distance_func, arc_length_func, total_length_func


# Example usage and visualization
if __name__ == "__main__":
    waypoints = np.array([[0, 1, 2, 2],
                          [0, 2, -2, -6]])

    smoothness = 0.1

    print("Creating arc length function...")
    distance_func, arc_length_func, total_length_func = create_combined_spline_functions(
        n_waypoints=4, n_spline_points=50, smoothness=smoothness
    )

    # Test the functions
    total_length = float(total_length_func(waypoints))
    print(f"Total spline length: {total_length:.3f}")

    # Test some points
    test_points = [
        (0.5, 1.0, "Near start"),
        (2.0, 2.5, "Middle"),
        (3.5, 1.5, "Near end"),
        (2.0, 0.5, "Below path")
    ]

    print("\nTest points:")
    for x, y, desc in test_points:
        distance = float(distance_func(x, y, waypoints))
        arc_length = float(arc_length_func(x, y, waypoints))
        progress = arc_length / total_length * 100
        print(f"{desc} ({x}, {y}): distance={distance:.3f}, arc_length={arc_length:.3f} ({progress:.1f}% along path)")

    # Create visualization
    print("\nCreating arc length visualization...")

    # Create grid
    x_min, x_max = -1, 4
    y_min, y_max = -7, 4
    resolution = 100

    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Evaluate arc length function over the grid
    Z_arc = np.zeros_like(X)
    Z_dist = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            arc_length = float(arc_length_func(X[i, j], Y[i, j], waypoints))
            distance = float(distance_func(X[i, j], Y[i, j], waypoints))
            Z_arc[i, j] = arc_length
            Z_dist[i, j] = distance

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Arc length contour plot
    contour1 = axes[0].contourf(X, Y, Z_arc, levels=20, cmap='coolwarm', alpha=0.8)
    axes[0].contour(X, Y, Z_arc, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour1, ax=axes[0], label='Arc Length Along Path')

    # Distance contour plot for comparison
    contour2 = axes[1].contourf(X, Y, Z_dist, levels=20, cmap='viridis', alpha=0.8)
    axes[1].contour(X, Y, Z_dist, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour2, ax=axes[1], label='Distance to Path')

    # Combined visualization
    # Show arc length with distance as contour lines
    contour3 = axes[2].contourf(X, Y, Z_arc, levels=20, cmap='coolwarm', alpha=0.6)
    axes[2].contour(X, Y, Z_dist, levels=[0.1, 0.2, 0.5, 1.0, 2.0], colors='black',
                    linewidths=1, linestyles='--', alpha=0.7)
    plt.colorbar(contour3, ax=axes[2], label='Arc Length')

    # Add spline path to all plots
    spline_gen = make_symbolic_smooth_spline_simple(4, 100, smoothness)
    spline_result = spline_gen(waypoints)
    spline_points = np.array(spline_result).T

    for ax in axes:
        ax.plot(waypoints[0, :], waypoints[1, :], 'ko-', linewidth=3, markersize=8,
                label='Waypoints', markerfacecolor='white', markeredgecolor='black', markeredgewidth=2)
        ax.plot(spline_points[:, 0], spline_points[:, 1], 'w-', linewidth=3, alpha=0.9,
                label='Smooth Spline')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')

    axes[0].set_title('Arc Length Along Path\n(Red = End, Blue = Start)')
    axes[1].set_title('Distance to Path\n(Dark = Close, Light = Far)')
    axes[2].set_title('Combined: Arc Length + Distance Contours\n(Dashed lines = constant distance)')

    plt.tight_layout()
    plt.show()

    # Create a detailed analysis plot
    print("\nCreating detailed analysis...")

    # Sample points along a line perpendicular to the path
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Line through the middle of the path
    x_line = np.linspace(-0.5, 4.5, 100)
    y_line = np.full_like(x_line, 1.5)

    arc_lengths_line = []
    distances_line = []

    for x_val, y_val in zip(x_line, y_line):
        arc_len = float(arc_length_func(x_val, y_val, waypoints))
        dist = float(distance_func(x_val, y_val, waypoints))
        arc_lengths_line.append(arc_len)
        distances_line.append(dist)

    # Plot arc length profile
    axes[0, 0].plot(x_line, arc_lengths_line, 'r-', linewidth=2, label='Arc Length')
    axes[0, 0].set_xlabel('X coordinate')
    axes[0, 0].set_ylabel('Arc Length Along Path')
    axes[0, 0].set_title('Arc Length vs X (at y=1.5)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot distance profile
    axes[0, 1].plot(x_line, distances_line, 'b-', linewidth=2, label='Distance')
    axes[0, 1].set_xlabel('X coordinate')
    axes[0, 1].set_ylabel('Distance to Path')
    axes[0, 1].set_title('Distance vs X (at y=1.5)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Scatter plot: distance vs arc length
    axes[1, 0].scatter(distances_line, arc_lengths_line, c=x_line, cmap='plasma', alpha=0.7)
    axes[1, 0].set_xlabel('Distance to Path')
    axes[1, 0].set_ylabel('Arc Length Along Path')
    axes[1, 0].set_title('Arc Length vs Distance\n(Color = X position)')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])

    # Show the path with arc length markers
    axes[1, 1].plot(spline_points[:, 0], spline_points[:, 1], 'b-', linewidth=3, label='Spline Path')
    axes[1, 1].plot(waypoints[0, :], waypoints[1, :], 'ro-', linewidth=2, markersize=8,
                    label='Waypoints')

    # Add arc length markers along the path
    marker_intervals = [0, 0.25, 0.5, 0.75, 1.0]
    for frac in marker_intervals:
        idx = int(frac * (len(spline_points) - 1))
        marker_point = spline_points[idx]
        axes[1, 1].plot(marker_point[0], marker_point[1], 'go', markersize=10,
                        markerfacecolor='yellow', markeredgecolor='green', markeredgewidth=2)
        axes[1, 1].text(marker_point[0], marker_point[1] + 0.15, f'{frac * 100:.0f}%',
                        ha='center', fontweight='bold', fontsize=9)

    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].set_title('Spline Path with Arc Length Markers')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')

    plt.tight_layout()
    plt.show()

    print(f"\nFunction Analysis:")
    print(f"Total path length: {total_length:.3f}")
    print(f"Arc length range: 0 to {total_length:.3f}")
    print(f"Distance range: {min(distances_line):.3f} to {max(distances_line):.3f}")

    #return distance_func, arc_length_func, total_length_func


#def create_spline_arc_length_function(n_waypoints, n_spline_points=100, smoothness=0.3):
#    """
#    Creates a function that returns the arc length along the spline path
#    where the closest point to the query occurs.
#
#    Returns:
#        CasADi Function: f(xq, yq, waypoints) -> arc_length_at_closest_point
#    """
#
#    # Step 1: Generate smooth spline points
#    spline_generator = make_symbolic_smooth_spline_simple(n_waypoints, n_spline_points, smoothness)
#
#    # Symbolic inputs
#    xq = ca.MX.sym("xq")
#    yq = ca.MX.sym("yq")
#    waypoints = ca.MX.sym("waypoints", 2, n_waypoints)
#    query_point = ca.vertcat(xq, yq)
#
#    # Generate smooth spline points
#    smooth_points = spline_generator(waypoints)
#
#    # Step 2: Calculate cumulative arc lengths along the spline
#    arc_lengths = [0]  # Start with 0 arc length
#    for i in range(n_spline_points - 1):
#        p1 = smooth_points[:, i]
#        p2 = smooth_points[:, i + 1]
#        segment_length = ca.norm_2(p2 - p1)
#        arc_lengths.append(arc_lengths[-1] + segment_length)
#
#    # Step 3: Find closest point and return its arc length
#    closest_arc_length = arc_lengths[0]
#    min_distance = ca.inf
#
#    for i in range(n_spline_points - 1):
#        p1 = smooth_points[:, i]
#        p2 = smooth_points[:, i + 1]
#        v = p2 - p1
#        w = query_point - p1
#
#        v_dot_v = ca.dot(v, v)
#        t = ca.if_else(v_dot_v > 1e-12, ca.dot(w, v) / v_dot_v, 0)
#        t_clamped = ca.fmax(0, ca.fmin(1, t))
#
#        closest_point = p1 + t_clamped * v
#        dist = ca.norm_2(query_point - closest_point)
#
#        # Arc length at this closest point
#        arc_length_at_closest = arc_lengths[i] + t_clamped * ca.norm_2(v)
#
#        # Update if this is the closest point so far
#        is_closer = dist < min_distance
#        closest_arc_length = ca.if_else(is_closer, arc_length_at_closest, closest_arc_length)
#        min_distance = ca.if_else(is_closer, dist, min_distance)
#
#    arc_length_func = ca.Function(
#        "spline_arc_length",
#        [xq, yq, waypoints],
#        [closest_arc_length],
#        ["xq", "yq", "waypoints"],
#        ["arc_length"]
#    )
#
#    return arc_length_func
