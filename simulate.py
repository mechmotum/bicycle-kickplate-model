from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm

from nonlin_sym import *

##########################
# Check evaluation of EoMs
##########################

print('Lambdifying full equations of motion.')
eval_dynamic = sm.lambdify([qs, us, fs, rs, ps], [A_all, b_all], cse=True)
print('Test eval_dynamics with all ones: ')
print(eval_dynamic(*[np.ones_like(a) for a in [qs, us, fs, rs, ps]]))

############################
# Create ODE right hand side
############################


def calc_y(t):

    L = 0.4  # height
    k = 100.0  # steepness
    t0 = 1.0  # shift to the right

    # logistic function
    y = L/(1+np.exp(-k*(t-t0)))
    yd = L*k*np.exp(-k*(t - t0))/(1 + np.exp(-k*(t - t0)))**2
    ydd = (-L*k**2*(1 - 2*np.exp(-k*(t - t0))/(1 + np.exp(-k*(t - t0))))*
           np.exp(-k*(t - t0))/(1 + np.exp(-k*(t - t0)))**2)

    return y, yd, ydd


def rhs(t, x, p):
    """
    Parameters
    ==========
    t : float
        Time value in seconds.
    x : array_like, shape(20,)
        State values where x = [q1, q2, q3, q4, q5, q6, q7, q8, u1, u2, u3, u4,
        u5, u6, u7, u8, Fry, Ffy, Mrz, Mfz].
    p : array_like, shape(28,)
        Constant values.

    Returns
    =======
    xdot : ndarray, shape(20,)
        Time derivative of the state.
    force : ndarray, shape(2,)
        Ground normal force magnitudes at the rear and front wheel contacts.
        force = [Frz, Ffz]

    """
    # TODO : add new argument to pass in a function to calculate r.

    q = x[:8]
    u = x[8:16]
    f = x[16:20]

    # steer, rear wheel, roll torques set to zero
    T4, T6, T7 = 0.0, 0.0, 0.0

    # kickplate motion set to zero
    y, yd, ydd = calc_y(t)

    r = [T4, T6, T7, y, yd, ydd]

    # This solves for the generalized accelerations and the normal forces at
    # the tire contact.
    A, b = eval_dynamic(q, u, f, r, p)
    # xplus = [us', Frz, Ffz]
    xplus = np.linalg.solve(A, b).squeeze()

    return np.hstack((u, xplus[:12])), xplus[-2:]


########################
# Setup Numerical Values
########################

p_vals = {
   c_af: 11.46,  # estimates from Andrew's dissertation (done by him)
   c_ar: 11.46,
   c_pf: 0.573,
   c_pr: 0.573,
   c_maf: 0.01,  # need real numbers for this
   c_mar: 0.01,  # need real numbers for this
   c_mpf: 0.01,  # need real numbers for this
   c_mpr: 0.01,  # need real numbers for this
   d1: 0.9534570696121849,
   d2: 0.2676445084476887,
   d3: 0.03207142672761929,
   g: 9.81,
   ic11: 7.178169776497895,
   ic22: 11.0,
   ic31: 3.8225535938357873,
   ic33: 4.821830223502103,
   id11: 0.0603,
   id22: 0.12,
   ie11: 0.05841337700152972,
   ie22: 0.06,
   ie31: 0.009119225261946298,
   ie33: 0.007586622998470264,
   if11: 0.1405,
   if22: 0.28,
   l1: 0.4707271515135145,
   l2: -0.47792881146460797,
   l3: -0.00597083392418685,
   l4: -0.3699518200282974,
   mc: 85.0,
   md: 2.0,
   me: 4.0,
   mf: 3.0,
   rf: 0.35,
   rr: 0.3,
   s_yf: 0.15,  # need real numbers for this
   s_yr: 0.15,  # need real numbers for this
   s_zf: 0.15,  # need real numbers for this
   s_zr: 0.15,  # need real numbers for this
}
p_arr = np.array(list(p_vals.values()))

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
eval_holonomic = sm.lambdify((q5, q4, q7, d1, d2, d3, rf, rr), holonomic)
initial_pitch_angle = float(fsolve(eval_holonomic, 0.0,
                                   args=(q_vals[3],  # q4
                                         q_vals[6],  # q7
                                         p_vals[d1],
                                         p_vals[d2],
                                         p_vals[d3],
                                         p_vals[rf],
                                         p_vals[rr])))
print('Initial pitch angle:', np.rad2deg(initial_pitch_angle))
q_vals[4] = initial_pitch_angle
print('Initial coordinates: ', q_vals)

# initial speeds
initial_speed = 4.6  # m/s
u_vals = np.array([
    np.nan,  # u1
    np.nan,  # u2
    0.0,  # u3, rad/s
    0.1,  # u4, rad/s
    np.nan,  # u5, rad/s
    -initial_speed/p_vals[rr],  # u6
    0.0,  # u7
    -initial_speed/p_vals[rf],  # u8
])

eval_dep_speeds = sm.lambdify([qs, u_ind, [yd], ps], [A_nh, -B_nh], cse=True)
A_nh_vals, B_nh_vals = eval_dep_speeds(q_vals, u_vals[[2, 3, 5, 6, 7]], [0.0],
                                       p_arr)
u_vals[[0, 1, 4]] = np.linalg.solve(A_nh_vals, B_nh_vals).squeeze()
print('Initial dependent speeds (u1, u2, u5): ',
      u_vals[0], u_vals[1], np.rad2deg(u_vals[4]))
print('Initial speeds: ', u_vals)

# initial tire forces
# TODO : Need to figure out how we know the initial state of the tire forces.
f_vals = np.array([0.0, 0.0, 0.0, 0.0])

initial_conditions = np.hstack((q_vals, u_vals, f_vals))

print('Test rhs with initial conditions and correct constants:')
print(rhs(0.0, initial_conditions, p_arr))

##########
# Simulate
##########

fps = 100  # frames per second
duration = 6.0  # seconds
t0 = 0.0
tf = t0 + duration
times = np.linspace(t0, tf, num=int(duration*fps))

res = solve_ivp(lambda t, x: rhs(t, x, p_arr)[0], (t0, tf),
                initial_conditions, t_eval=times, method='LSODA')
x_traj = res.y.T
times = res.t

holonomic_vs_time = eval_holonomic(x_traj[:, 4],  # q5
                                   x_traj[:, 3],  # q4
                                   x_traj[:, 6],  # q7
                                   p_vals[d1],
                                   p_vals[d2],
                                   p_vals[d3],
                                   p_vals[rf],
                                   p_vals[rr])

deg = [False, False, True, True, True, True, True, True]
fig, axes = plt.subplots(11, 2, sharex=True)
q_traj = x_traj[:, :8]
u_traj = x_traj[:, 8:16]
f_traj = x_traj[:, 16:]
fig.set_size_inches(8, 10)
for i, (ax, traj, s, degi) in enumerate(zip(axes[:, 0], q_traj.T, qs, deg)):
    unit = '[m]'
    if degi:
        traj = np.rad2deg(traj)
        unit = '[deg]'
    ax.plot(times, traj)
    ax.set_ylabel(str(s) + '\n' + unit)
for i, (ax, traj, s, degi) in enumerate(zip(axes[:, 1], u_traj.T, us, deg)):
    unit = '[m/s]'
    if degi:
        traj = np.rad2deg(traj)
        unit = '[deg/s]'
    ax.plot(times, traj)
    ax.set_ylabel(str(s) + '\n' + unit)

axes[8, 0].plot(times, f_traj[:, 0])
axes[8, 0].set_ylabel(str(fs[0]) + '\n[N]')
axes[9, 0].plot(times, f_traj[:, 1])
axes[9, 0].set_ylabel(str(fs[1]) + '\n[N]')
axes[8, 1].plot(times, f_traj[:, 2])
axes[8, 1].set_ylabel(str(fs[2]) + '\n[N-m]')
axes[9, 1].plot(times, f_traj[:, 3])
axes[9, 1].set_ylabel(str(fs[3]) + '\n[N-m]')

axes[-1, 0].plot(times, calc_y(times)[0])
axes[-1, 0].set_ylabel('y\n[m]')
axes[-1, 0].set_xlabel('Time [s]')
axes[-1, 1].plot(times, holonomic_vs_time)
axes[-1, 1].set_ylabel('constraint\n[m]')
axes[-1, 1].set_xlabel('Time [s]')
plt.tight_layout()
plt.show()