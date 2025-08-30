import casadi as ca
import numpy as np
import matplotlib.pyplot as plt



# Alternative simpler approach using linear interpolation with more segments
def make_symbolic_piecewise_linear_distance(n_spline_waypoints):
    """
    Simplest version: just compute distance to the piecewise linear path
    connecting the waypoints directly (no intermediate sampling).
    """

    xq = ca.MX.sym("xq")
    yq = ca.MX.sym("yq")
    query_point = ca.vertcat(xq, yq)
    spline_w = ca.MX.sym("spline_w", 2, n_spline_waypoints)

    # Calculate distance to each line segment between consecutive waypoints
    dists = []
    for i in range(n_spline_waypoints - 1):
        p1 = spline_w[:, i]
        p2 = spline_w[:, i + 1]
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

    f = ca.Function(
        "symbolic_piecewise_linear_distance",
        [xq, yq, spline_w],
        [min_dist],
        ["xq", "yq", "spline_waypoints"],
        ["distance"]
    )
    return f


# --- Example Usage ---
if __name__ == "__main__":
    # Create the distance function
    waypoints = np.array([[0, 1, 3, 4],
                          [0, 2, 3, 0]])

    print("Creating symbolic distance function...")
    dist_function = make_symbolic_piecewise_linear_distance(n_spline_waypoints=4)

    # Create a grid for contour plotting
    print("Setting up grid...")
    x_min, x_max = -1, 5
    y_min, y_max = -1, 4
    resolution = 50  # Grid resolution

    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Evaluate distance function over the grid
    print("Evaluating distance function over grid...")
    Z = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            try:
                # Evaluate the CasADi function at each grid point
                distance = dist_function(X[i, j], Y[i, j], waypoints)
                Z[i, j] = float(distance)
            except Exception as e:
                print(f"Error at ({X[i, j]}, {Y[i, j]}): {e}")
                Z[i, j] = np.nan

    print("Creating contour plot...")

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot contour
    contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
    contour_lines = plt.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)

    # Add colorbar
    cbar = plt.colorbar(contour)
    cbar.set_label('Distance to Spline', rotation=270, labelpad=20)

    # Plot the original waypoints
    plt.plot(waypoints[0, :], waypoints[1, :], 'ro-', linewidth=3, markersize=8,
             label='Original Waypoints', markerfacecolor='red', markeredgecolor='white', markeredgewidth=2)

    # Plot the interpolated spline path for reference
    # Create a dense version for visualization
    n_vis_samples = 200
    vis_points = []
    for i in range(len(waypoints[0]) - 1):
        n_seg = n_vis_samples // (len(waypoints[0]) - 1)
        if i == len(waypoints[0]) - 2:
            n_seg = n_vis_samples - len(vis_points)

        for j in range(n_seg):
            t = j / (n_seg - 1) if n_seg > 1 else 0
            x = (1 - t) * waypoints[0, i] + t * waypoints[0, i + 1]
            y = (1 - t) * waypoints[1, i] + t * waypoints[1, i + 1]
            vis_points.append([x, y])

    if vis_points:
        vis_points = np.array(vis_points)
        plt.plot(vis_points[:, 0], vis_points[:, 1], 'w--', linewidth=2, alpha=0.8,
                 label='Interpolated Spline Path')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Contour Plot of Distance to Linear Spline\n(Darker = Closer to Spline)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # Add some statistics
    print(f"Distance range: {np.nanmin(Z):.3f} to {np.nanmax(Z):.3f}")
    print(f"Grid resolution: {resolution}x{resolution}")

    plt.tight_layout()
    plt.show()



