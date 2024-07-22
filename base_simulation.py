"""Simulate the standard condition: initial conditions are zero except for an
initial forward speed and apply a pulse kick."""

import numpy as np
import matplotlib.pyplot as plt

from simulate import (rr, rf, p_vals, p_arr, setup_initial_conditions, rhs,
                      simulate, plot_all, plot_wheel_paths, equilibrium_eq,
                      calc_linear_tire_force, calc_nonlinear_tire_force,
                      eval_angles)


def calc_fkp(t):
    """Returns the lateral forced applied to the tire by the kick plate. The
    force is modeled as a sinusoidal pulse."""

    start = 0.4  # seconds
    stop = 0.6  # seconds
    magnitude = 400  # Newtons

    period = stop - start
    frequency = 1.0/period
    omega = 2*np.pi*frequency  # rad/s

    if start < t < stop:
        return magnitude/2.0*(1.0 - np.cos(omega*(t - start)))
    else:
        return 0.0


def calc_steer_torque(t, x):
    """Simple LQR control based on linear Carvallo-Whipple model."""

    q = x[:10]
    u = x[10:20]

    q4 = q[3]
    q7 = q[6]
    u4 = u[3]
    u7 = u[6]

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
        r = [T4, T6, T7, fkp, Fry, Ffy, Mrz, Mfz].

    """

    q = x[:10]
    u = x[10:20]

    q11, q12, u11, u12 = q[8], q[9], u[8], u[9]
    c_f, c_r, k_f, k_r = p[2], p[9], p[26], p[27]
    Frz = k_r*q11 + c_r*u11  # positive when in compression
    Ffz = k_f*q12 + c_f*u12  # positive when in compression

    c_af, c_ar = p[0], p[1]
    c_maf, c_mar, c_mpf, c_mpr, c_pf, c_pr = p[3:9]
    alphar, alphaf, phir, phif = eval_angles(q, u, p)
    Fry, Mrz = calc_linear_tire_force(alphar, phir, Frz, c_ar, c_pr, c_mar,
                                      c_mpr)
    Ffy, Mfz = calc_linear_tire_force(alphaf, phif, Ffz, c_af, c_pf, c_maf,
                                      c_mpf)
    #print('Linear: ', Fry, Ffy)
    Fry, Mrz = calc_nonlinear_tire_force(alphar, phir, Frz)
    Ffy, Mfz = calc_nonlinear_tire_force(alphaf, phif, Ffz)
    #print('Non-linear: ', Fry, Ffy)

    # steer, rear wheel, roll torques set to zero
    T4, T6, T7 = 0.0, 0.0, calc_steer_torque(t, x)

    # kick plate force
    fkp = calc_fkp(t)

    r = [T4, T6, T7, fkp, Fry, Ffy, Mrz, Mfz]

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
    0.00664797028,  # q11
    0.00220163072,  # q12
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

print('Test rhs with initial conditions and correct constants:')
print(rhs(0.0, initial_conditions, calc_inputs, p_arr))

fps = 100  # frames per second
duration = 6.0  # seconds
res = simulate(duration, calc_inputs, initial_conditions, p_arr, fps=fps)

plot_all(*res)
plot_wheel_paths(res[1], res[-3], res[-2])

if __name__ == "__main__":
    plt.show()
