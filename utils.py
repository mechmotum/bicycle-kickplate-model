from bicycleparameters.models import Meijaard2007Model
from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
from control import ctrb
from scipy.linalg import solve_continuous_are
from sympy import ImmutableMatrix, zeros
from sympy.core.cache import cacheit
import numpy as np
import sympy as sm
import sympy.physics.mechanics as mec


def calc_lqr_gains(par, speed):

    """
    x = |roll angle | = |phi     |
        |steer angle|   |delta   |
        |roll rate  |   |phidot  |
        |steer rate |   |deltadot|
        |yaw angle  |   |psi     |

    u = |roll torque | = |Tphi  |
        |steer torque|   |Tdelta|

    psid = (v*delta + c*deltad)/w*cos(lam)
    yp = v*psi
    yq = (v + w)*psi - c*cos(lam)*delta

    """
    # TODO : I don't seem to have a function that converts the benchmark
    # parameters set to my parameter set.

    #par = {k: str(v) for k, v in sym_par_map.items()}
    try:
        par['v']
    except KeyError:
        par['v'] = 0.0
    par_set = Meijaard2007ParameterSet(par, True)
    model = Meijaard2007Model(par_set)
    A_, B_ = model.form_state_space_matrices(v=speed)  # 4x4, 4x2
    A1 = np.hstack((A_, np.zeros((4, 1))))
    coef = np.cos(par['lam'])/par['w']
    A = np.vstack((A1, [0.0, speed*coef, 0.0, par['c']*coef, 0.0]))
    B = np.vstack((B_, [0.0, 0.0]))
    C = ctrb(A, B[:, 1:2])
    print("Controllability rank:", np.linalg.matrix_rank(C))
    Q = np.eye(5)
    Q = np.diag([1.0, 1.0, 1.0, 1.0, 200.0])
    R = np.eye(1)
    S = solve_continuous_are(A, B[:, 1:2], Q, R)  # steer torque control
    K = (np.linalg.inv(R) @ B[:, 1:2].T @ S).squeeze()

    return K


def print_syms(expr, note='', log=True):
    if log:
        dy_syms = list(sm.ordered(mec.find_dynamicsymbols(expr)))
        co_syms = list(sm.ordered(expr.free_symbols))
        try:
            co_syms.remove(mec.dynamicsymbols._t)
        except ValueError:
            pass
        print('{}\n    {}\n    {}'.format(note, dy_syms, co_syms))


@cacheit
def det_laplace(matrix):
    n = matrix.shape[0]
    if n == 1:
        return matrix[0]
    elif n == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    else:
        return sum((-1) ** i * matrix[0, i] *
                   det_laplace(matrix.minor_submatrix(0, i)) for i in range(n))


def cramer_solve(A, b, det_method=det_laplace):
    def entry(i, j):
        return b[i, sol] if j == col else A[i, j]

    A_imu = ImmutableMatrix(A)  # Convert to immutable for cache purposes
    det_A = det_method(A_imu)
    x = zeros(*b.shape)
    for sol in range(b.shape[1]):
        for col in range(b.shape[0]):
            x[col, sol] = det_method(ImmutableMatrix(*A.shape, entry)) / det_A
    return x


def linear_coefficients(exprs, vars):
    """Finds the linear coefficients of the variables in the expressions and
    returns them in a Jacobian form.

    Warning: expr must be linear in vars!

    Parameters
    ==========
    exprs : sequence of expressions, shape(n,)
    vars: sequence of variables, shape(m,)

    Returns
    =======
    coeffs : Matrix, shape(n, m)
        Jacobian exprs wrt vars.
    """
    coeffs = sm.zeros(len(exprs), len(vars))
    for i, expr in enumerate(exprs):
        for j, var in enumerate(vars):
            coeffs[i, j] = expr.coeff(var)
    return coeffs


def decompose_linear_parts(F, *x):
    """Returns the linear coefficient matrices associated with the provided
    vectors and the remainder vector. F must be able to be put into the
    following form:

    F = A1*x1 + A2*x2 + ... + An*xm + B = 0

    - F : n x 1 vector of expressions
    - Ai : n x pi matrix of expressions
    - xi : pi x 1 vector of variables
    - pi : length of vector xi
    - m : number of xi vectors
    - B : n x 1 vector of expressions

    Parameters
    ==========
    F : Matrix, shape(n, 1)
        Column matrix of expressions that linearly depend on entires of
        x1,...,xm.
    x : Sequence[Expr]
        Column matrices representing x1,...,xm.

    Returns
    =======
    Ai, ..., An : Matrix
    B : Matrix, shape(n, 1)

    Notes
    =====
    If xi = xj', then make sure xj'is passed in first to guarantee proper
    replacement.

    """
    F = sm.Matrix(F)
    matrices = []
    for xi in x:
        Ai = linear_coefficients(F, xi)
        matrices.append(Ai)
        repl = {xij: 0 for xij in xi}
        F = F.xreplace(repl)  # remove Ai*xi from F
    matrices.append(F)
    return tuple(matrices)


class ReferenceFrame(mec.ReferenceFrame):
    """Subclass that enforces the desired unit vector index style."""

    def __init__(self, *args, **kwargs):

        kwargs.pop('indices', None)
        kwargs.pop('latexs', None)

        lab = args[0].lower()
        tex = r'\hat{{{}}}_{}'

        super(ReferenceFrame, self).__init__(*args, indices=('1', '2', '3'),
                                             latexs=(tex.format(lab, '1'),
                                                     tex.format(lab, '2'),
                                                     tex.format(lab, '3')),
                                             **kwargs)
