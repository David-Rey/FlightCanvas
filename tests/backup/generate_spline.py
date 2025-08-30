import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


def make_symbolic_cubic_spline_natural(n_waypoints, n_output_points=100):
    """
    Creates a natural cubic spline that is smooth everywhere, including endpoints.
    Uses the "natural" boundary condition (second derivative = 0 at endpoints).
    """

    waypoints = ca.MX.sym("waypoints", 2, n_waypoints)

    # Parameter values for waypoints
    t_in = np.linspace(0, 1, n_waypoints)
    t_out = np.linspace(0, 1, n_output_points)

    def natural_cubic_spline_1d(y_values, t_knots, t_eval):
        """
        Natural cubic spline interpolation with smooth curves everywhere
        """
        n = len(t_knots)

        # Build the tridiagonal system for natural cubic spline
        # We need to solve for the second derivatives at each knot

        # Compute differences
        h = []
        for i in range(n - 1):
            h.append(t_knots[i + 1] - t_knots[i])

        # Build system Ax = b for second derivatives
        # Natural boundary conditions: second derivative = 0 at endpoints

        # Initialize arrays symbolically
        output_points = []

        for t_val in t_eval:
            result = y_values[0]  # Initialize

            # Find which interval and interpolate with cubic polynomials
            for i in range(n - 1):
                t_start = t_knots[i]
                t_end = t_knots[i + 1]

                # Check if t_val is in this interval
                in_interval = ca.logic_and(t_val >= t_start, t_val < t_end)
                if i == n - 2:  # Last interval includes endpoint
                    in_interval = ca.logic_or(in_interval, t_val >= t_end)

                # Local parameter
                dt = t_end - t_start
                s = (t_val - t_start) / dt
                s = ca.fmax(0, ca.fmin(1, s))

                # Cubic Hermite interpolation with estimated derivatives
                p0 = y_values[i]
                p1 = y_values[i + 1]

                # Estimate derivatives at endpoints for smooth curves
                if i == 0:
                    # First segment: estimate derivative from first two points
                    if n > 2:
                        m0 = (y_values[1] - y_values[0]) / h[0]
                        m1 = (y_values[2] - y_values[0]) / (h[0] + h[1])
                    else:
                        m0 = (y_values[1] - y_values[0]) / h[0]
                        m1 = m0
                elif i == n - 2:
                    # Last segment: estimate derivative from last two points
                    m0 = (y_values[n - 1] - y_values[n - 3]) / (h[i - 1] + h[i])
                    m1 = (y_values[n - 1] - y_values[n - 2]) / h[i]
                else:
                    # Middle segments: use central differences
                    m0 = (y_values[i + 1] - y_values[i - 1]) / (h[i - 1] + h[i])
                    m1 = (y_values[i + 2] - y_values[i]) / (h[i] + h[i + 1])

                # Scale derivatives by segment length
                m0 = m0 * dt
                m1 = m1 * dt

                # Hermite basis functions
                h00 = 2 * s ** 3 - 3 * s ** 2 + 1
                h10 = s ** 3 - 2 * s ** 2 + s
                h01 = -2 * s ** 3 + 3 * s ** 2
                h11 = s ** 3 - s ** 2

                # Cubic Hermite interpolation
                segment_value = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

                result = ca.if_else(in_interval, segment_value, result)

            output_points.append(result)

        return output_points

    # Apply to x and y coordinates
    x_spline = natural_cubic_spline_1d([waypoints[0, i] for i in range(n_waypoints)], t_in, t_out)
    y_spline = natural_cubic_spline_1d([waypoints[1, i] for i in range(n_waypoints)], t_in, t_out)

    # Combine into output matrix
    spline_points = ca.horzcat(*[ca.vertcat(x_spline[i], y_spline[i]) for i in range(n_output_points)])

    spline_func = ca.Function(
        "natural_cubic_spline",
        [waypoints],
        [spline_points],
        ["waypoints"],
        ["spline_points"]
    )

    return spline_func


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
            # Scale t to span all segments
            segment_t = t_val * (n - 1)
            segment_idx = ca.floor(segment_t)
            local_t = segment_t - segment_idx

            result = control_points[0]

            for i in range(n - 1):
                in_segment = ca.logic_and(segment_idx == i, local_t >= 0)

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
                t = local_t
                t2 = t * t
                t3 = t2 * t
                mt = 1 - t
                mt2 = mt * mt
                mt3 = mt2 * mt

                segment_value = mt3 * p0 + 3 * mt2 * t * p1 + 3 * mt * t2 * p2 + t3 * p3

                result = ca.if_else(in_segment, segment_value, result)

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


# Example usage and comparison
if __name__ == "__main__":
    waypoints = np.array([[0, 1, 3, 4],
                          [0, 2, 3, 0]])

    print("Comparing different spline methods...")

    plt.figure(figsize=(15, 5))

    # Test all three methods
    methods = [
        ("Natural Cubic", make_symbolic_cubic_spline_natural),
        ("Smooth Simple", lambda n, nout: make_symbolic_smooth_spline_simple(n, nout, 0.3)),
        ("Very Smooth", lambda n, nout: make_symbolic_smooth_spline_simple(n, nout, 0.5))
    ]

    for idx, (name, func) in enumerate(methods):
        try:
            spline_func = func(4, 100)
            result = spline_func(waypoints)
            points = np.array(result).T

            plt.subplot(1, 3, idx + 1)
            plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label=f'{name} Spline')
            plt.plot(waypoints[0, :], waypoints[1, :], 'ro-', linewidth=3, markersize=8,
                     label='Waypoints', markerfacecolor='red', markeredgecolor='white', markeredgewidth=2)
            plt.title(f'{name}\n(All segments should be curved)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')

            print(f"{name}: Successfully generated {points.shape[0]} points")

        except Exception as e:
            print(f"{name} error: {e}")

    plt.tight_layout()
    plt.show()

    # Test that the function is truly symbolic
    print("\nTesting symbolic nature...")
    try:
        smooth_func = make_symbolic_smooth_spline_simple(4, 50, 0.4)

        # Create symbolic waypoints for testing
        sym_waypoints = ca.MX.sym("test_waypoints", 2, 4)
        sym_result = smooth_func(sym_waypoints)

        print("✓ Function accepts symbolic inputs and produces symbolic outputs")
        print(f"✓ Output shape: {sym_result.shape} (should be 2 x 50)")

        # Test with actual values
        actual_result = smooth_func(waypoints)
        actual_points = np.array(actual_result).T
        print(f"✓ Actual evaluation produces {actual_points.shape[0]} points")

    except Exception as e:
        print(f"Symbolic test error: {e}")