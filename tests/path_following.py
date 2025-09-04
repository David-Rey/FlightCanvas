import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from scipy.interpolate import CubicSpline

import numpy as np
import casadi as ca
from scipy.interpolate import splprep, splev

def make_spline_from_waypoints(waypoints, smoothing=0, k=3, num_points=200):
    waypoints = np.asarray(waypoints)
    assert waypoints.shape[1] == 2, "Waypoints must be shape (N,2)."

    # --- Fit parametric spline using scipy ---
    tck, u = splprep([waypoints[:, 0], waypoints[:, 1]], s=smoothing, k=k)

    # Sample the spline finely
    unew = np.linspace(0, 1, num_points)
    x_spline, y_spline = splev(unew, tck)

    # Compute arc-length parameterization
    dx = np.diff(x_spline)
    dy = np.diff(y_spline)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate(([0], np.cumsum(ds)))  # arc-length
    L = s[-1]

    # Normalize to [0, L]
    s_normalized = s

    # CasADi interpolants (1D lookup tables)
    px_fun = ca.interpolant("px", "bspline", [s_normalized.tolist()], x_spline)
    py_fun = ca.interpolant("py", "bspline", [s_normalized.tolist()], y_spline)

    return px_fun, py_fun, L

# Define waypoints
waypoints = np.array([
    [0, 0],
    [2, 1],
    [4, 0],
    [6, -1],
    [8, 0]
])

px_fun, py_fun, L = make_spline_from_waypoints(waypoints)

# Evaluate spline in CasADi
s = ca.MX.sym("s")
x_pos = px_fun(s)
y_pos = py_fun(s)

path_fun = ca.Function("path", [s], [x_pos, y_pos])

print("Total path length:", L)

# Test evaluation
print("Position at s=L/2:", path_fun(L/2))

# -----------------
# System dynamics
# -----------------
px = ca.MX.sym('px')
py = ca.MX.sym('py')
vx = ca.MX.sym('vx')
vy = ca.MX.sym('vy')
x = ca.vertcat(px, py, vx, vy)

ux = ca.MX.sym('ux')
uy = ca.MX.sym('uy')
u = ca.vertcat(ux, uy)

# Algebraic variable = spline parameter
z = ca.MX.sym('z')

# Continuous dynamics
xdot = ca.vertcat(vx, vy, ux, uy)

# -----------------
# Path tube constraint
# -----------------
px_path_z, py_path_z = path_fun(z)
dist_vec = ca.vertcat(px - px_path_z, py - py_path_z)
dist_sq = ca.dot(dist_vec, dist_vec)   # squared distance
tube_radius = 0.5   # allowable tube around path

# -----------------
# ACADOS model
# -----------------
model = AcadosModel()
model.name = "drone_path_following"
model.x = x
model.u = u
model.z = z
model.xdot = xdot
#model.con_h_expr = dist_sq  # tube constraint expression
model.f_impl_expr = ca.vertcat(
    xdot[0] - vx,
    xdot[1] - vy,
    xdot[2] - ux,
    xdot[3] - uy,
    0
)
#model.f_expl_expr = ca.vertcat(vx, vy, ux, uy)

# -----------------
# OCP
# -----------------
N = 40          # horizon steps
T = 4.0         # horizon length

ocp = AcadosOcp()
ocp.model = model
ocp.dims.N = N
ocp.solver_options.tf = T

# -----------------
# Cost
# -----------------
w_tube = 10.0
w_u = 1e-2
w_s = 5.0

px_path_z, py_path_z = path_fun(z)
dist_sq = (x[0]-px_path_z)**2 + (x[1]-py_path_z)**2

stage_cost = w_tube*dist_sq + w_u*(ux**2 + uy**2) - w_s*z
model.cost_expr_ext_cost = stage_cost

# Terminal cost: distance to end of path
px_end = px_fun(L)
py_end = py_fun(L)
dist_sq_end = (x[0]-px_end)**2 + (x[1]-py_end)**2
#model.cost_expr_ext_cost_e = dist_sq_end

# constraints
#ocp.constraints.lh = np.array([0.0])         # lower bound dist_sq >= 0
#ocp.constraints.uh = np.array([tube_radius**2]) # enforce ||p - path(z)||^2 <= r^2
#ocp.constraints.idxh = np.array([0])

# initial condition
x0 = np.array([0, 0, 0, 0])
ocp.constraints.x0 = x0

# solver
ocp_solver = AcadosOcpSolver(ocp, json_file='drone_path.json')

#ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
#ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'IRK'
#ocp.solver_options.nlp_solver_type = 'SQP'
#ocp.solver_options.regularize_method = 'GERSHGORIN_LEVENBERG_MARQUARDT'
#ocp.solver_options.levenberg_marquardt = 1e-2
#ocp.solver_options.nlp_solver_max_iter = 200
#ocp.solver_options.tol = 1e-2
ocp.solver_options.qp_solver_cond_N = N

# -----------------
# Run OCP
# -----------------
simX = [x0]
simU = []
for k in range(N):
    ocp_solver.set(0, "lbx", x0)
    ocp_solver.set(0, "ubx", x0)
    status = ocp_solver.solve()
    if status != 0:
        raise Exception(f"acados returned status {status} at step {k}")

    u_opt = ocp_solver.get(0, "u")
    simU.append(u_opt)

    # propagate with simple Euler (for demo)
    x0 = simX[-1] + T/N * np.array([x0[2], x0[3], u_opt[0], u_opt[1]])
    simX.append(x0)

simX = np.array(simX)
simU = np.array(simU)

print("Final position:", simX[-1, :2])
