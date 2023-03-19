import sympy as sm
from sympy import ImmutableMatrix, zeros
from sympy.core.cache import cacheit
import sympy.physics.mechanics as mec
import numpy as np


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
        Ai = F.jacobian(xi)
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


def euler_integrate(rhs_func, tspan, x0_vals, p_vals, delt=0.03):
    """Returns state trajectory and corresponding values of time resulting
    from integrating the ordinary differential equations with Euler's
    Method.

    Parameters
    ==========
    rhs_func : function
       Python function that evaluates the derivative of the state and takes
       this form ``dxdt = f(t, x, p)``.
    tspan : 2-tuple of floats
       The initial time and final time values: (t0, tf).
    x0_vals : array_like, shape(2*n,)
       Values of the state x at t0.
    p_vals : array_like, shape(o,)
       Values of constant parameters.
    delt : float
       Integration time step in seconds/step.

    Returns
    =======
    ts : ndarray(m, )
       Monotonically increasing values of time.
    xs : ndarray(m, 2*n)
       State values at each time in ts.

    """
    # generate monotonically increasing values of time.
    duration = tspan[1] - tspan[0]
    num_samples = round(duration/delt) + 1
    ts = np.arange(tspan[0], tspan[0] + delt*num_samples, delt)

    # create an empty array to hold the state values.
    x = np.empty((len(ts), len(x0_vals)))

    # set the initial conditions to the first element.
    x[0, :] = x0_vals

    # use a for loop to sequentially calculate each new x.
    for i, ti in enumerate(ts[:-1]):
        x[i + 1, :] = x[i, :] + delt*rhs_func(ti, x[i, :], p_vals)

    return ts, x
