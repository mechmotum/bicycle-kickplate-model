from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, least_squares
import matplotlib.pyplot as plt
import numpy as np

from nonlin_sym import *
from tire_data import SchwalbeT03_500kPa as TIRE
from inputs import calc_linear_tire_force, calc_nonlinear_tire_force

##########################
# Check evaluation of EoMs
##########################

print('Test eval_dynamics with all ones: ')
print(eval_dynamic(*[np.ones_like(a) for a in [qs, us, fs, rs, ps]]))

############################
# Create ODE right hand side
############################


def rhs(t, x, r_func, p):
    """Evaluates the time derivative of the state.

    Parameters
    ==========
    t : float
        Time value in seconds.
    x : array_like, shape(24,)
        State values where x = [q1, q2, q3, q4, q5, q6, q7, q8, q11, q12,
                                u1, u2, u3, u4, u5, u6, u7, u8, u11, u12,
                                Fry, Ffy, Mrz, Mfz].
    r_func : function
        Function of the form ``r = f(t, x, p)``. Returns all specified inputs
        where: r = [T4, T6, T7, fkp, y, yd, ydd, Fry, Ffy, Mrz, Mfz].
    p : array_like, shape(44,)
        Constant values.
        [c_af, c_ar, c_f, c_maf, c_mar, c_mpf, c_mpr, c_pf, c_pr, c_r, d1, d2,
        d3, g, ic11, ic22, ic31, ic33, id11, id22, ie11, ie22, ie31, ie33,
        if11, if22, k_f, k_r, l1, l2, l3, l4, mc, md, me, mf, r_tf, r_tr, rf,
        rr, s_yf, s_yr, s_zf, s_zr]

    Returns
    =======
    xdot : ndarray, shape(24,)
        Time derivative of the state.

    """

    q = x[:10]
    u = x[10:20]
    f = x[20:]
    r = r_func(t, x, p)

    A, b = eval_dynamic(q, u, f, r, p)
    udot = np.linalg.solve(A, b).squeeze()

    return np.hstack((u, udot))


def fall_detector(t, x):
    """Returns the difference in the absoluate value of the roll angle and the
    maximum tolerable roll angle magnitude."""
    max_roll = np.deg2rad(45.0)
    return max_roll - np.abs(x[3])


fall_detector.terminal = True
fall_detector.direction = -1


def equilibrium_eq(q, p):
    """Returns the static equilibrium configuration of the model."""
    # TODO : When I change the tire vertical stiffness values I don't get a
    # change in equilibrium state. So this doesn't seem to work in a full proof
    # way.
    #u = np.ones(10)*1e-13  # divide by zeros if u is simple all zeros
    u = np.zeros(10)
    f = np.zeros(4)  # Fry, Ffy, Mrz, Mfz
    r = np.zeros(11)  # T4, T6, T7, Fkp, y, yd, ydd, Fry_, Ffy_, Mrz_, Mfz_

    def zeros(x):
        """
        x = [q5, q11, q12]
        """
        q_ = q.copy()
        q_[4] = x[0]
        q_[8] = x[1]
        q_[9] = x[2]
        return eval_equilibrium(q_, u, f, r, p).squeeze()

    sol = least_squares(zeros, q[[4, 8, 9]])
    q[4] = sol.x[0]
    q[8] = sol.x[1]
    q[9] = sol.x[2]
    return q


def setup_initial_conditions(q_vals, u_vals, f_vals, p_arr):
    """Calculates dependent coordinates and speeds given the independent
    coordinates and speeds. The dependent coordinates and speeds in q_vals and
    u_vals will be overwritten.

    Parameters
    ==========
    q_vals : array_like, shape(10,)
        [q1, q2, q3, q4, q5, q6, q7, q8, q11, q12]
    u_vals : array_like, shape(10,)
        [u1, u2, u3, u4, u5, u6, u7, u8, u11, u12]
    f_vals: array_like, shape(4,)
        [Fry, Ffy, Mrz, Mfz]
    p_arr: array_like, shape(42,)

    """

    # NOTE : The following equilibrium state calculated assuming that the tire
    # vertical stiffness is very high and that the bicycle is in the nominal
    # configuration. It does not solve the general equilibrium state. It sets
    # the pitch angle to be that of the bike with no tire deflection and it
    # sets the tire deflection based on the force balance of the bike with
    # infintely stiff tires. This gives reliable results for the initial
    # configuration being the nomrinal configuration.

    ehom_args = (
        q_vals[3],  # q4
        q_vals[6],  # q7
        q_vals[8],  # q11
        q_vals[9],  # q12
        p_arr[10],  # d1
        p_arr[11],  # d2
        p_arr[12],  # d3
        p_arr[36],  # r_tf
        p_arr[37],  # r_tr
        p_arr[38],  # rf
        p_arr[39],  # rr
    )
    initial_pitch_angle = float(fsolve(eval_holonomic, np.pi/10,
                                       args=ehom_args))
    print('Initial pitch angle:', np.rad2deg(initial_pitch_angle))
    q_vals[4] = initial_pitch_angle

    total_mass, wheelbase, com_d = eval_balance(q_vals, p_arr)
    print('Total mass:', total_mass)
    print('Wheelbase:', wheelbase)
    print('Longitudinal distance to total mass center:', com_d)
    q11 = (wheelbase - com_d)*total_mass*p_arr[13]/wheelbase/p_arr[27]
    q12 = com_d*total_mass*p_arr[13]/wheelbase/p_arr[26]
    print('q11, q12 :', q11, q12)
    q_vals[-2] = q11
    q_vals[-1] = q12

    # TODO : Find a reliable method of calculating the general equilibrium
    # state.
    #q_eq = equilibrium_eq(q_vals, p_arr)
    #print('Equilibrium coordinates: ', q_eq)
    #q_vals[4] = q_eq[4]
    #q_vals[-2:] = q_eq[-2:]
    #print('Initial pitch angle:', np.rad2deg(q_eq[4]))

    print('Independent generalized speeds:', u_vals[[2, 3, 5, 6, 7, 8, 9]])
    A_nh_vals, B_nh_vals = eval_dep_speeds(q_vals,
                                           u_vals[[2, 3, 5, 6, 7, 8, 9]],
                                           [0.0, 0.0],  # y, yd
                                           p_arr)
    res = np.linalg.solve(A_nh_vals, B_nh_vals.squeeze())
    print('res', res)
    u_vals[[0, 1, 4]] = np.linalg.solve(A_nh_vals, B_nh_vals).squeeze()
    print('zero', A_nh_vals @ u_vals[[0, 1, 4]] - B_nh_vals.squeeze())

    print('Initial dependent speeds (u1 [m/s], u2 [m/s], u5 [deg/s]): ',
          u_vals[0], u_vals[1], np.rad2deg(u_vals[4]))
    print('Initial speeds: ', u_vals)
    # TODO: When the speed is higher than about 4.6, the initial lateral speed
    # is non-zero. Need to investigate. For now, force to zero.
    # The singularity with q7=0.0 causes this and it is very sensitive to the
    # value of q7.
    u_vals[1] = 0.0

    return np.hstack((q_vals, u_vals, f_vals))


def simulate(dur, calc_inputs, x0, p, fps=60):
    """Simulate the model given the duration, constant parameters, initial
    conditions, and inputs and calculate any output variables."""

    t0 = 0.0
    tf = t0 + dur
    times = np.linspace(t0, tf, num=int(dur*fps) + 1)

    res = solve_ivp(lambda t, x: rhs(t, x, calc_inputs, p), (t0, tf),
                    x0, t_eval=times, events=fall_detector, method='LSODA',
                    rtol=1e-12)

    times = res.t
    x_traj = res.y.T
    q_traj = x_traj[:, :10]
    u_traj = x_traj[:, 10:20]
    f_traj = x_traj[:, 20:]

    con_traj = eval_holonomic(
        q_traj[:, 4],  # q5
        q_traj[:, 3],  # q4
        q_traj[:, 6],  # q7
        q_traj[:, 8],  # q11
        q_traj[:, 9],  # q12
        p[10],  # d1
        p[11],  # d2
        p[12],  # d3
        p[36],  # r_tf
        p[37],  # r_tr
        p[38],  # rf
        p[39],  # rr
    )

    fz_traj = np.zeros((len(times), 2))
    slip_traj = np.zeros((len(times), 4))
    q9_traj = np.zeros_like(times)
    q10_traj = np.zeros_like(times)
    r_traj = np.zeros((len(times), 7))
    for i, (ti, qi, ui, fi) in enumerate(zip(times, q_traj, u_traj, f_traj)):
        statei = np.hstack((qi, ui, fi))
        r_traj[i] = calc_inputs(ti, statei, p)[:7]
        fz_traj[i, :] = np.array([-p[27]*qi[8] - p[9]*ui[8],
                                  -p[26]*qi[9] - p[2]*ui[9]])
        slip_traj[i, :] = eval_angles(qi, ui, r_traj[i, 3:5], p)
        q9_traj[i], q10_traj[i] = eval_front_contact(qi, p)

    return (times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj,
            q9_traj, q10_traj, r_traj)


def plot_all(times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj,
             q9_traj, q10_traj, r_traj):

    deg = [False, False, True, True, True, True, True, True, False, False]
    fig, axes = plt.subplots(16, 2, sharex=True, layout='constrained')
    fig.set_size_inches(8, 10)
    # fills right 10 rows
    for i, (ax, traj, s, degi) in enumerate(zip(axes[:, 0], q_traj.T, qs, deg)):
        unit = '[m]'
        if degi:
            traj = np.rad2deg(traj)
            unit = '[deg]'
        ax.plot(times, traj)
        ax.set_ylabel(str(s) + '\n' + unit)
    # fills left 10 rows
    for i, (ax, traj, s, degi) in enumerate(zip(axes[:, 1], u_traj.T, us, deg)):
        unit = '[m/s]'
        if degi:
            traj = np.rad2deg(traj)
            unit = '[deg/s]'
        ax.plot(times, traj)
        ax.set_ylabel(str(s) + '\n' + unit)

    axes[10, 0].plot(times, f_traj[:, 0])
    axes[10, 0].set_ylabel(str(fs[0]) + '\n[N]')
    axes[10, 1].plot(times, f_traj[:, 1])
    axes[10, 1].set_ylabel(str(fs[1]) + '\n[N]')
    axes[11, 0].plot(times, f_traj[:, 2])
    axes[11, 0].set_ylabel(str(fs[2]) + '\n[N-m]')
    axes[11, 1].plot(times, f_traj[:, 3])
    axes[11, 1].set_ylabel(str(fs[3]) + '\n[N-m]')

    axes[12, 0].plot(times, fz_traj[:, 0])
    axes[12, 0].set_ylabel(str('Frz') + '\n[N]')
    axes[12, 1].plot(times, fz_traj[:, 1])
    axes[12, 1].set_ylabel(str('Ffz') + '\n[N]')

    axes[13, 0].plot(times, np.rad2deg(slip_traj[:, 0]))
    axes[13, 0].set_ylabel('alphar\n[deg]')
    axes[13, 1].plot(times, np.rad2deg(slip_traj[:, 1]))
    axes[13, 1].set_ylabel('alphaf\n[deg]')
    axes[14, 0].plot(times, np.rad2deg(slip_traj[:, 2]))
    axes[14, 0].set_ylabel('phir\n[deg]')
    axes[14, 1].plot(times, np.rad2deg(slip_traj[:, 3]))
    axes[14, 1].set_ylabel('phif\n[deg]')

    axes[-1, 0].plot(times, r_traj[:, -1])
    axes[-1, 0].set_ylabel('$\ddot{y}$\n[m/s/s]')
    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].plot(times, con_traj)
    axes[-1, 1].set_ylabel('constraint\n[m]')
    axes[-1, 1].set_xlabel('Time [s]')


def plot_kick_motion(times, r_traj):

    fig, axes = plt.subplots(3, 1, sharex=True, layout='constrained')
    axes[0].plot(times, r_traj[:, -1])
    axes[0].set_ylabel(r'$\ddot{y}$ [m/s/s]')
    axes[1].plot(times, r_traj[:, -2])
    axes[1].set_ylabel(r'$\dot{y}$ [m/s]')
    axes[2].plot(times, r_traj[:, -3])
    axes[2].set_ylabel(r'$y$ [m]')
    axes[2].set_xlabel('Time [s]')

    return axes


def plot_wheel_paths(q_traj, q9_traj, q10_traj, kick_displacement):
    fig, ax = plt.subplots(1, 1)
    ax.plot(q_traj[:, 0], q_traj[:, 1], label='Rear Wheel Contact')
    ax.plot(q9_traj, q10_traj, label='Front Wheel Contact')
    # NOTE : I plot the kickplate displacement vs rear wheel longitudinal
    # motion for comparison purposes.
    ax.plot(q_traj[:, 0], kick_displacement, label='Kick Plate Displacement')
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\hat{n}_1$')
    ax.set_ylabel(r'$\hat{n}_2$')
    ax.invert_yaxis()
    ax.grid()
    ax.legend()
    return ax


def plot_tire_curves(p_vals):

    camber_range = np.deg2rad(45.0)
    camber_angles = np.linspace(-camber_range, camber_range)

    slip_range = np.deg2rad(20.0)
    slip_angles = np.linspace(-slip_range, slip_range)

    normal_forces = [-200.0, -400.0, -600.0, -800.0]
    colors = ['C0', 'C1', 'C2', 'C3']

    fig, axes = plt.subplots(2, 2, layout='constrained')

    # Update "tire" to plot the current tire characteristics you are using for
    # simulations
    for Fz, color in zip(normal_forces, colors):
        Fys, Mzs = [], []
        Fys_lin, Mzs_lin = [], []
        for alpha in slip_angles:
            Fy, Mz = calc_nonlinear_tire_force(alpha, 0.0, Fz, TIRE)
            Fy_lin, Mz_lin = calc_linear_tire_force(alpha, 0.0, Fz,
                                                    p_vals[c_ar],
                                                    p_vals[c_pr],
                                                    p_vals[c_mar],
                                                    p_vals[c_mpr])
            Fys.append(Fy)
            Mzs.append(Mz)
            Fys_lin.append(Fy_lin)
            Mzs_lin.append(Mz_lin)
        axes[0, 0].plot(np.rad2deg(slip_angles), Fys,
                        color=color,
                        label='Fz = {} N'.format(Fz))
        axes[1, 0].plot(np.rad2deg(slip_angles), Mzs,
                        color=color,
                        label='Fz = {} N'.format(Fz))
        axes[0, 0].plot(np.rad2deg(slip_angles), Fys_lin,
                        color=color,
                        linestyle='--',
                        label='Fz = {} N'.format(Fz))
        axes[1, 0].plot(np.rad2deg(slip_angles), Mzs_lin,
                        color=color,
                        linestyle='--',
                        label='Fz = {} N'.format(Fz))
        Fys, Mzs = [], []
        Fys_lin, Mzs_lin = [], []
        for phi in camber_angles:
            Fy, Mz = calc_nonlinear_tire_force(0.0, phi, Fz, TIRE)
            Fy_lin, Mz_lin = calc_linear_tire_force(0.0, phi, Fz,
                                                    p_vals[c_ar],
                                                    p_vals[c_pr],
                                                    p_vals[c_mar],
                                                    p_vals[c_mpr])
            Fys.append(Fy)
            Mzs.append(Mz)
            Fys_lin.append(Fy_lin)
            Mzs_lin.append(Mz_lin)
        axes[0, 1].plot(np.rad2deg(camber_angles), Fys,
                        color=color,
                        label='Fz = {} N'.format(Fz))
        axes[1, 1].plot(np.rad2deg(camber_angles), Mzs,
                        color=color,
                        label='Fz = {} N'.format(Fz))
        axes[0, 1].plot(np.rad2deg(camber_angles), Fys_lin,
                        color=color,
                        linestyle='--',
                        label='Fz = {} N'.format(Fz))
        axes[1, 1].plot(np.rad2deg(camber_angles), Mzs_lin,
                        color=color,
                        linestyle='--',
                        label='Fz = {} N'.format(Fz))

    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Slip angle [deg]')
    axes[0, 0].set_ylabel('Lateral Force [N]')
    axes[0, 0].set_ylim(-1000, 1000)
    axes[0, 0].grid()

    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Slip angle [deg]')
    axes[1, 0].set_ylabel('Self-aligning Moment [N-m]')
    axes[1, 0].set_ylim(-25, 25)
    axes[1, 0].grid()

    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Camber angle [deg]')
    axes[0, 1].set_ylabel('Lateral Force [N]')
    axes[0, 1].grid()

    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Camber angle [deg]')
    axes[1, 1].set_ylabel('Self-aligning Moment [N-m]')
    axes[1, 1].grid()

    return axes


# simplified figure
# compare normal tire numbers and 10% change in slip coefficient
# plot slip angle, lateral force for front and rear steer angle and force input
def plot_minimal(t, q7, ar, af, fkp, T7, fyr, fyf, axes=None, torqax=None,
                 **kwargs):

    if axes is None:
        fig, axes = plt.subplots(2, 1, sharex=True)
    if torqax is None:
        torqax = axes[1].twinx()

    axes[0].plot(t, np.rad2deg(q7), color='C0', label=r'$\delta$',
                 **kwargs)
    axes[0].plot(t, np.rad2deg(ar), color='C1', label=r'$\alpha_r$',
                 **kwargs)
    axes[0].plot(t, np.rad2deg(af), color='C2', label=r'$\alpha_f$',
                 **kwargs)
    axes[1].plot(t, fkp, color='C0', label='$F_{kp}$', **kwargs)
    axes[1].plot(t, fyr, color='C1', label='$F_{yr}$', **kwargs)
    axes[1].plot(t, fyf, color='C2', label='$F_{yf}$', **kwargs)
    torqax.plot(t, T7, color='C3', label=r'$T_\delta$', **kwargs)

    axes[0].set_ylabel('Angle [deg]')
    axes[1].set_ylabel('Force [N]')
    torqax.set_ylabel('Torque [N-m]')
    axes[1].set_xlabel('Time [s]')
    axes[0].set_xlim((0.0, 2.0))
    axes[1].set_xlim((0.0, 2.0))

    axes[0].legend()
    axes[1].legend()
    torqax.legend()

    plt.tight_layout()
    return axes, torqax
