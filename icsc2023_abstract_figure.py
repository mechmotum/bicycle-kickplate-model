"""Simulate the standard condition: initial conditions are zero except for an
initial forward speed and apply a pulse kick."""

import numpy as np
import matplotlib.pyplot as plt

from simulate import (rr, rf, p_vals, p_arr, rhs, setup_initial_conditions,
                      simulate, plot_minimal, calc_fkp, c_af, c_ar, c_maf,
                      c_mar, c_pf, c_pr, ps)


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

times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj, q9_traj, q10_traj = simulate(
    2.0, rhs, initial_conditions, p_arr, fps=1000)

axes = plot_minimal(times, q_traj[:, 6], slip_traj[:, 0], slip_traj[:, 1],
                    calc_fkp(times), f_traj[:, 0], f_traj[:, 1])

factor = 1.3
p_vals[c_af] = p_vals[c_af]*factor
p_vals[c_ar] = p_vals[c_ar]*factor
p_vals[c_maf] = p_vals[c_maf]*factor
p_vals[c_mar] = p_vals[c_mar]*factor
p_vals[c_pf] = p_vals[c_pf]*factor
p_vals[c_pr] = p_vals[c_pr]*factor
p_arr = np.array([p_vals[pi] for pi in ps])
initial_conditions = setup_initial_conditions(q_vals, u_vals, f_vals, p_arr)

times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj, q9_traj, q10_traj = simulate(
    2.0, rhs, initial_conditions, p_arr, fps=1000)

plot_minimal(times, q_traj[:, 3], slip_traj[:, 0], slip_traj[:, 1],
             calc_fkp(times), f_traj[:, 0], f_traj[:, 1], axes=axes,
             linestyle=':')

plt.show()
