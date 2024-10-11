"""Generates the plot for our ICSC 2023 poster and abstract. Simulate the
standard condition: initial conditions are zero except for an initial forward
speed and apply a pulse kick."""

import numpy as np
import matplotlib.pyplot as plt

from simulate import *
from parameters import browser_jason_par as p_vals
from inputs import calc_full_state_feedback_steer_torque as calc_steer_torque

p_arr = np.array([p_vals[pi] for pi in ps])

# LQR gains for Whipple model Broswer-Jason at 6 m/s
kq4 = -2.2340917377023612
kq7 = 4.90641020775064
ku4 = -0.5939384880650549
ku7 = 0.4340987861323103


def calc_fkp(t):
    """Returns the lateral forced applied to the tire by the kick plate."""

    if t > 0.5 and t < 1.0:
        return 500.0  # Newtons
    else:
        return 0.0


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
    y, yd, ydd = 0.0, 0.0, 0.0

    c_af, c_ar = p[0], p[1]
    c_maf, c_mar, c_mpf = p[3], p[4], p[5]
    c_mpr, c_pf, c_pr = p[6], p[7], p[8]
    alphar, alphaf, phir, phif = eval_angles(q, u, [y, yd], p)
    Fry, Mrz = calc_linear_tire_force(alphar, phir, Frz, c_ar, c_pr, c_mar,
                                      c_mpr)
    Ffy, Mfz = calc_linear_tire_force(alphaf, phif, Ffz, c_af, c_pf, c_maf,
                                      c_mpf)

    # steer, rear wheel, roll torques set to zero
    T4, T6, T7 = 0.0, 0.0, calc_steer_torque(t, x, [kq4, ku4, kq7, ku7])

    # kick plate force
    fkp = calc_fkp(t)

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
    0.0,  # q11
    0.0,  # q12
])

# initial speeds
initial_speed = 6.0  # m/s
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

(times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj, q9_traj,
 q10_traj, r_traj) = simulate(2.0, calc_inputs, initial_conditions, p_arr,
                              fps=1000)

axes, torqax = plot_minimal(times, q_traj[:, 6], slip_traj[:, 0],
                            slip_traj[:, 1], r_traj[:, -1], r_traj[:, 2],
                            f_traj[:, 0], f_traj[:, 1])

factor = 1.3
p_vals[c_af] = p_vals[c_af]*factor
p_vals[c_ar] = p_vals[c_ar]*factor
p_vals[c_maf] = p_vals[c_maf]*factor
p_vals[c_mar] = p_vals[c_mar]*factor
p_vals[c_pf] = p_vals[c_pf]*factor
p_vals[c_pr] = p_vals[c_pr]*factor
p_arr = np.array([p_vals[pi] for pi in ps])
initial_conditions = setup_initial_conditions(q_vals, u_vals, f_vals, p_arr)

(times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj, q9_traj,
 q10_traj, r_traj) = simulate(2.0, calc_inputs, initial_conditions, p_arr,
                              fps=1000)

plot_minimal(times, q_traj[:, 6], slip_traj[:, 0], slip_traj[:, 1],
             r_traj[:, -1], r_traj[:, 2], f_traj[:, 0], f_traj[:, 1],
             axes=axes, torqax=torqax, linestyle=':')

plt.show()
