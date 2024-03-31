"""Simulate the standard condition: initial conditions are zero except for an
initial forward speed and apply a pulse kick."""

import numpy as np
import matplotlib.pyplot as plt

from simulate import (rr, rf, p_vals, p_arr, setup_initial_conditions, rhs,
                      simulate, plot_all, plot_wheel_paths, equilibrium_eq)


def calc_fkp(t):
    """Returns the lateral forced applied to the tire by the kick plate."""

    if t > 0.5 and t < 1.0:
        return 100.0
    else:
        return 0.0


def calc_steer_torque(t, x):

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
    # TODO : these can be generated from equilibrium_eq (copied for now)
    -0.00664797028,  # q11
    -0.00220163072,  # q12
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
    0.0,  # u11
    0.0,  # u12
])

# initial tire forces
# TODO : Need to figure out how we know the initial state of the tire forces.
f_vals = np.array([0.0, 0.0, 0.0, 0.0])

initial_conditions = setup_initial_conditions(q_vals, u_vals, f_vals, p_arr)

print('equilibrium')
print(equilibrium_eq(initial_conditions[0:10], p_arr))

print('Test rhs with initial conditions and correct constants:')
print(rhs(0.0, initial_conditions, calc_inputs, p_arr))

fps = 100  # frames per second
duration = 6.0  # seconds
duration = 0.5  # seconds
res = simulate(duration, calc_inputs, initial_conditions, p_arr, fps=fps)

plot_all(*res)
plot_wheel_paths(res[1], res[-3], res[-2])

plt.show()
