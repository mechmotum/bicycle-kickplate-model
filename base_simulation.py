"""Simulate the standard condition: initial conditions are zero except for an
initial forward speed and apply a pulse kick."""

import numpy as np
import matplotlib.pyplot as plt

from symbols import rr, rf, ps
from inputs import (calc_kick_motion_constant_acc, calc_linear_tire_force,
                    calc_nonlinear_tire_force)
from inputs import calc_full_state_feedback_steer_torque as calc_steer_torque
from parameters import balanceassistv1_gabriele as p_vals
from tire_data import SchwalbeT03_500kPa as tire
from simulate import setup_initial_conditions, rhs, simulate
from plot import (plot_all, plot_kick_motion, plot_wheel_paths,
                  plot_tire_curves)
from generated_functions import eval_angles

p_arr = np.array([p_vals[pi] for pi in ps])

# TODO : Move to parameters.py
# LQR gains for Whipple model, rider Gabriele (635 N), at 3 m/s
kq4 = -19.5679
ku4 = -6.7665
kq7 = 15.4934
ku7 = 1.5876


def calc_inputs(t, x, p):
    """Returns all specified forces and torques.

    Parameters
    ==========
    t : float
        Time in seconds.
    x : array_like, shape(24,)
        State values where x = [q1, q2, q3, q4, q5, q6, q7, q8, q11, q12,
                                u1, u2, u3, u4, u5, u6, u7, u8, u11, u12,
                                Fry, Ffy, Mrz, Mfz].
    p : array_like, shape(44,)
        Constant values.
        [c_af, c_ar, c_f, c_maf, c_mar, c_mpf, c_mpr, c_pf, c_pr, c_r, d1, d2,
        d3, g, ic11, ic22, ic31, ic33, id11, id22, ie11, ie22, ie31, ie33,
        if11, if22, k_f, k_r, l1, l2, l3, l4, mc, md, me, mf, r_tf, r_tr, rf,
        rr, s_yf, s_yr, s_zf, s_zr]

    Returns
    =======
    r : ndarray, shape(8,)
        r = [T4, T6, T7, fkp, y, yd, ydd, Fry, Ffy, Mrz, Mfz].

    """

    q = x[:10]
    u = x[10:20]

    q11, q12, u11, u12 = q[8], q[9], u[8], u[9]
    c_f, c_r, k_f, k_r = p[2], p[9], p[26], p[27]
    Frz = -k_r*q11 - c_r*u11  # negative when in compression
    Ffz = -k_f*q12 - c_f*u12  # negative when in compression

    # plate motion
    y, yd, ydd = calc_kick_motion_constant_acc(t)

    c_af, c_ar = p[0], p[1]
    c_maf, c_mar, c_mpf = p[3], p[4], p[5]
    c_mpr, c_pf, c_pr = p[6], p[7], p[8]
    alphar, alphaf, phir, phif = eval_angles(q, u, [y, yd], p)
    Fry, Mrz = calc_linear_tire_force(alphar, phir, Frz, c_ar, c_pr, c_mar,
                                      c_mpr)
    Ffy, Mfz = calc_linear_tire_force(alphaf, phif, Ffz, c_af, c_pf, c_maf,
                                      c_mpf)
    #print('Linear: ', Fry, Ffy)

    Fry, Mrz = calc_nonlinear_tire_force(alphar, phir, Frz, tire)
    Ffy, Mfz = calc_nonlinear_tire_force(alphaf, phif, Ffz, tire)
    #print('Non-linear: ', Fry, Ffy)

    # steer, rear wheel, roll torques set to zero
    T4, T6, T7 = 0.0, 0.0, calc_steer_torque(t, x, [kq4, ku4, kq7, ku7])

    # kick plate force
    fkp = 0.0

    # NOTE : Self-aligning moment has a destabilizing effect, you can disable
    # it by uncommenting the following line.
    #Mrz, Mfz = 0.0, 0.0

    r = [T4, T6, T7, fkp, y, yd, ydd, Fry, Ffy, Mrz, Mfz]

    return r


# initial coordinates
q_vals = np.array([
    0.0,  # q1
    0.0,  # q2
    0.0,  # q3
    0.0,  # q4
    np.nan,  # q5
    0.0,  # q6
    1e-14,  # q7, setting to zero gives singular matrix
    0.0,  # q8
    # TODO : these can be generated from equilibrium_eq (copied for now)
    0.0, #0.00664797028,  # q11
    0.0, #0.00220163072,  # q12
])

# initial speeds
initial_speed = 3.0  # m/s
u_vals = np.array([
    np.nan,  # u1
    np.nan,  # u2
    0.0,  # u3, rad/s
    0.0,  # u4, rad/s
    np.nan,  # u5, rad/s
    -initial_speed/p_vals[rr],  # u6
    0.0,  # u7
    -initial_speed/p_vals[rf],  # u8
    0.0,  # u11
    0.0,  # u12
])

# initial tire forces
# TODO : Need to figure out how we know the initial state of the tire forces.
f_vals = np.array([0.0, 0.0, 0.0, 0.0])

initial_conditions = setup_initial_conditions(q_vals, u_vals, f_vals, p_arr)
print('Initial conditions:')
print(initial_conditions)

print('Test rhs with initial conditions and correct constants:')
print(rhs(0.0, initial_conditions, calc_inputs, p_arr))

fps = 400  # frames per second
duration = 6.0  # seconds
res = simulate(duration, calc_inputs, initial_conditions, p_arr, fps=fps)

plot_all(*res)
plot_kick_motion(res[0], res[-1])
plot_wheel_paths(res[1], res[-3], res[-2], res[-1][:, 4])
plot_tire_curves(p_vals)

if __name__ == "__main__":
    plt.show()
