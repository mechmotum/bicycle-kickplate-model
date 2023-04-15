"""Simulate the standard condition: initial conditions are zero except for an
initial forward speed and apply a pulse kick."""

import numpy as np
import matplotlib.pyplot as plt

from simulate import (rr, rf, p_vals, p_arr, setup_initial_conditions,
                      simulate, plot_minimal, c_af, c_ar, c_maf, c_mar, c_pf,
                      c_pr, ps)


def calc_fkp(t):
    """Returns the lateral forced applied to the tire by the kick plate."""

    if t > 0.5 and t < 1.0:
        return 100.0
    else:
        return 0.0


def calc_steer_torque(t, x):

    q4 = x[3]
    q7 = x[6]
    u4 = x[11]
    u7 = x[14]

    # LQR gains for Whipple model at 6 m/s
    kq4 = -2.2340917377023612
    kq7 = 4.90641020775064
    ku4 = -0.5939384880650549
    ku7 = 0.4340987861323103

    # LQR gains for Whipple model at 3 m/s
    #kq4 = -23.183400610625647
    #kq7 = 17.086409893261113
    #ku4 = -7.999852938163112
    #ku7 = 1.8634394089384874

    return -(kq4*q4 + kq7*q7 + ku4*u4 + ku7*u7)


def calc_inputs(t, x, p):

    # steer, rear wheel, roll torques set to zero
    T4, T6, T7 = 0.0, 0.0, calc_steer_torque(t, x)

    # kick plate force
    fkp = calc_fkp(t)

    r = [T4, T6, T7, fkp]

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
])

# initial speeds
initial_speed = 6.0  # m/s
u_vals = np.array([
    np.nan,  # u1
    np.nan,  # u2
    0.0,  # u3, rad/s
    1e-10,  # u4, rad/s
    np.nan,  # u5, rad/s
    -initial_speed/p_vals[rr],  # u6
    0.0,  # u7
    -initial_speed/p_vals[rf],  # u8
])

# initial tire forces
# TODO : Need to figure out how we know the initial state of the tire forces.
f_vals = np.array([0.0, 0.0, 0.0, 0.0])

initial_conditions = setup_initial_conditions(q_vals, u_vals, f_vals, p_arr)

(times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj, q9_traj,
 q10_traj, r_traj) = simulate(2.0, calc_inputs, initial_conditions, p_arr,
                              fps=1000)

axes = plot_minimal(times, q_traj[:, 6], slip_traj[:, 0], slip_traj[:, 1],
                    r_traj[:, -1], f_traj[:, 0], f_traj[:, 1])

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
             r_traj[:, -1], f_traj[:, 0], f_traj[:, 1], axes=axes,
             linestyle=':')

plt.show()
