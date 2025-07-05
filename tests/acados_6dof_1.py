import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import casadi as ca
import scipy as sp

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

    Jb = ca.diag(ca.vertcat(Ixx, Iyy, Izz))
    Jb_inv = ca.inv(Jb)

    # Gravity VEctor
    g_vec = ca.MX([0, 0, g])

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

    return model


def main():

    ocp = AcadosOcp()
    ocp.model = create_rocket_model()

    # Horizon parameters
    N = 100
    Tf = 10
    ocp.dims.N = N
    ocp.solver_options.tf = Tf
    ocp.solver_options.N_horizon = N

    # Cost type
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    # Get model dimensions
    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    # set cost
    Q_mat = 2 * np.diag([1e3, 1e3, 1e-4, 1e-3, 1e-3])  # [x,y,x_d,y_d,th,th_d]
    R_mat = 2 * 5 * np.diag([1e-1, 1e-2])

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ny = nx + nu
    ny_e = nx

    ocp.cost.W_e = Q_mat
    ocp.cost.W = sp.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    print(nx)



if __name__ == '__main__':
    main()
