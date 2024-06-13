from nonlin_sym import *

import sympy as sm
from sympy.solvers import linear_eq_to_matrix
from sympy.physics.mechanics.functions import center_of_mass

# A_all*x = B_all
# [Mp  0,  0, -Mf]*[up ] = [F     ]
# [0, Cf,  0,  Cz] [fyp]   [-Fy   ]
# [0,  0, Df,  Dz] [mzp]   [-Mz   ]
# [Ap, 0,  0,  Af] [fz ]   [-B_aux]
# TODO : Maybe add these rows for completeness
# [I,  0,  0,  0 ] [qp ] = [0     ]

# f(x', x, r) = f(xeq', xeq, req) + Jfx(xeq', xeq, req)*([x, r] - [xeq, req])
# f(x', x, r) = Mp*x' + Mx*x + Mr*r
# x' = A*x + B*r, where A = Mp

total_mass = mc + md + me + mf
mass_center = center_of_mass(dn, rear_frame, rear_wheel, front_frame,
                             front_wheel)

# These are approximations of the mass center that will be exact when the
# equilibrium points are subsituted in.
# total_mass*g*x = Ffz*w
# TODO : Check if these shoudl be positive or negative.
w = fn.pos_from(dn).dot(A.x)
Ffz_eq = total_mass*g*mass_center.dot(A.x)/w
# total_mass*g*(w-x) = Frz*w
Frz_eq = total_mass*g*(mass_center - w*A.x).dot(A.x)/w

q5eq, q7eq, u6eq, u8eq = sm.symbols('q5eq, q7eq, u6eq, u8eq')

Fryp, Ffyp, Mrzp, Mfzp = sm.symbols('Fryp, Ffyp, Mrzp, Mfzp')
u1p, u2p, u3p, u4p, u5p, u6p, u7p, u8p = sm.symbols('u1p, u2p, u3p, u4p, u5p, u6p, u7p, u8p')

xp = sm.Matrix([
    u1p,
    u2p,
    u3p,
    u4p,
    u5p,
    u6p,
    u7p,
    u8p,
    Fryp,
    Ffyp,
    Mrzp,
    Mfzp,
    Frz,
    Ffz,
])

x = sm.Matrix([
    u1,
    u2,
    u3,
    u4,
    u5,
    u6,
    u7,
    u8,
    Fry,
    Ffy,
    Mrz,
    Mfz,
])

r = sm.Matrix([
    T4,
    T6,
    T7,
    Fkp,
])

v0 = {
    # dx/dt
    u1p: 0,
    u2p: 0,
    u3p: 0,
    u4p: 0,
    u5p: 0,
    u6p: 0,
    u7p: 0,
    u8p: 0,
    Fryp: 0,
    Ffyp: 0,
    Mrzp: 0,
    Mfzp: 0,
    Frz: Frz_eq,
    Ffz: Ffz_eq,
    # x
    u1: 0,
    u2: 0,
    u3: 0,
    u4: 0,
    u5: 0,
    u6: u6eq,
    u7: 0,
    u8: u8eq,
    q1: 0,
    q2: 0,
    q3: 0,
    q4: 0,
    q5: q5eq,
    q6: 0,
    q7: q7eq,  # needed due to divide-by-zero issues
    q8: 0,
    # r
    T4: 0,
    T6: 0,
    T7: 0,
    Fkp: 0,
}

v = sm.Matrix(list(v0.keys()))
v_val = sm.Matrix(list(v0.values()))
v_zerod = {k: 0 for k, _ in v0.items()}

f = A_all*xp - b_all
f_lin = f.xreplace(v0) + f.jacobian(v).xreplace(v0)*(v - v_val)

Mp_lin = linear_eq_to_matrix(f_lin, xp)[0]
Mx_lin = linear_eq_to_matrix(f_lin, x)[0]
Mr_lin = linear_eq_to_matrix(f_lin, r)[0]
remainder = f_lin.xreplace(v_zerod)

eval_lin = sm.lambdify([qs, us, fs, rs, ps, eqs],
                       [Mp_lin, Mx_lin, Mr_lin],
                       cse=True)
