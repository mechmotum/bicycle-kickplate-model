from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

from nonlin_sym import *

##########################
# Check evaluation of EoMs
##########################

print('Test eval_dynamics with all ones: ')
print(eval_dynamic(*[np.ones_like(a) for a in [qs, us, fs, rs, ps]]))

############################
# Create ODE right hand side
############################


def rhs(t, x, r_func, p):
    """
    Parameters
    ==========
    t : float
        Time value in seconds.
    x : array_like, shape(20,)
        State values where x = [q1, q2, q3, q4, q5, q6, q7, q8, u1, u2, u3, u4,
        u5, u6, u7, u8, Fry, Ffy, Mrz, Mfz].
    p : array_like, shape(38,)
        Constant values.

    Returns
    =======
    xdot : ndarray, shape(20,)
        Time derivative of the state.
    force : ndarray, shape(2,)
        Ground normal force magnitudes at the rear and front wheel contacts.
        force = [Frz, Ffz]

    """

    q = x[:8]
    u = x[8:16]
    f = x[16:20]
    r = r_func(t, x, p)

    # This solves for the state derivatives and the normal forces at the tire
    # contact.
    A, b = eval_dynamic(q, u, f, r, p)
    # xplus = [us', Frz, Ffz]
    xplus = np.linalg.solve(A, b).squeeze()

    return np.hstack((u, xplus[:12])), xplus[-2:]


########################
# Setup Numerical Values
########################

# batavus browser with Jason sitting on it, tire parameters from
# Andrew/Gabriele
p_vals = {
    c_af: 11.46,  # estimates from Andrew's dissertation (done by him)
    c_ar: 11.46,
    c_maf: 0.33,  # 0.33 is rough calc from gabriele's data
    c_mar: 0.33,
    c_mpf: 0.0,  # need real numbers for this
    c_mpr: 0.0,  # need real numbers for this
    c_pf: 0.573,
    c_pr: 0.573,
    d1: 0.9631492634872098,
    d2: 0.4338396131640938,
    d3: 0.0705000000001252,
    g: 9.81,
    ic11: 11.519805885486146,
    ic22: 12.2177848012,
    ic31: 1.57915608541552,
    ic33: 2.959474124693854,
    id11: 0.0883819364527,
    id22: 0.152467620286,
    ie11: 0.2811355367159554,
    ie22: 0.246138810935,
    ie31: 0.0063377219110826045,
    ie33: 0.06782113764394461,
    if11: 0.0904106601579,
    if22: 0.149389340425,
    l1: 0.5384415640161426,
    l2: -0.531720230353059,
    l3: -0.07654646159268344,
    l4: -0.47166687226492093,
    mc: 81.86,
    md: 3.11,
    me: 3.22,
    mf: 2.02,
    rf: 0.34352982332,
    rr: 0.340958858855,
    s_yf: 0.175,  # Andrew's estimates from his dissertation data
    s_yr: 0.175,
    s_zf: 0.175,
    s_zr: 0.175,
}
p_arr = np.array([p_vals[pi] for pi in ps])


def setup_initial_conditions(q_vals, u_vals, f_vals, p_arr):
    """Calculates dependent coordinates and speeds given the independent
    coordinates and speeds. The dependent coordinates and speeds in q_vals and
    u_vals will be overwritten.

    Parameters
    ==========
    q_vals : array_like, shape(8,)
        [q1, q2, q3, q4, q5, q6, q7, q8]
    u_vals : array_like, shape(8,)
        [u1, u2, u3, u4, u5, u6, u7, u8]
    f_vals: array_like, shape(4,)
        [Fry, Ffy, Mrz, Mfz]
    p_arr: array_like, shape(38,)

    """

    ehom_args = (
        q_vals[3],  # q4
        q_vals[6],  # q7
        p_arr[8],  # d1
        p_arr[9],  # d2
        p_arr[10],  # d3
        p_arr[32],  # rf
        p_arr[33],  # rr
    )
    initial_pitch_angle = float(fsolve(eval_holonomic, np.pi/10,
                                       args=ehom_args))
    print('Initial pitch angle:', np.rad2deg(initial_pitch_angle))
    q_vals[4] = initial_pitch_angle
    print('Initial coordinates: ', q_vals)

    A_nh_vals, B_nh_vals = eval_dep_speeds(q_vals, u_vals[[2, 3, 5, 6, 7]],
                                           p_arr)
    u_vals[[0, 1, 4]] = np.linalg.solve(A_nh_vals, B_nh_vals).squeeze()
    print('Initial dependent speeds (u1, u2, u5): ', u_vals[0], u_vals[1],
          np.rad2deg(u_vals[4]))
    print('Initial speeds: ', u_vals)
    # TODO: When the speed is higher than about 4.6, the initial lateral speed
    # is non-zero. Need to investigate. For now, force to zero.
    u_vals[1] = 0.0

    return np.hstack((q_vals, u_vals, f_vals))


def simulate(dur, calc_inputs, x0, p, fps=60):
    """Simulate the model given the duration, constant parameters, initial
    conditions, and inputs and calcaluate any output variables."""

    t0 = 0.0
    tf = t0 + dur
    times = np.linspace(t0, tf, num=int(dur*fps) + 1)

    res = solve_ivp(lambda t, x: rhs(t, x, calc_inputs, p)[0], (t0, tf),
                    x0, t_eval=times, method='LSODA')

    times = res.t
    x_traj = res.y.T
    q_traj = x_traj[:, :8]
    u_traj = x_traj[:, 8:16]
    f_traj = x_traj[:, 16:]

    con_traj = eval_holonomic(
        q_traj[:, 4],  # q5
        q_traj[:, 3],  # q4
        q_traj[:, 6],  # q7
        p[8],  # d1
        p[9],  # d2
        p[10],  # d3
        p[32],  # rf
        p[33],  # rr
    )

    fz_traj = np.zeros((len(times), 2))
    slip_traj = np.zeros((len(times), 4))
    q9_traj = np.zeros_like(times)
    q10_traj = np.zeros_like(times)
    r_traj = np.zeros((len(times), 4))
    for i, (ti, qi, ui, fi) in enumerate(zip(times, q_traj, u_traj, f_traj)):
        statei = np.hstack((qi, ui, fi))
        _, fz_traj[i, :] = rhs(ti, statei, calc_inputs, p)
        slip_traj[i, :] = eval_angles(qi, ui, p)
        q9_traj[i], q10_traj[i] = eval_front_contact(qi, p)
        r_traj[i] = calc_inputs(ti, statei, p)

    return (times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj,
            q9_traj, q10_traj, r_traj)


def plot_all(times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj,
             q9_traj, q10_traj, r_traj):

    deg = [False, False, True, True, True, True, True, True]
    fig, axes = plt.subplots(14, 2, sharex=True)
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
    axes[8, 1].plot(times, f_traj[:, 1])
    axes[8, 1].set_ylabel(str(fs[1]) + '\n[N]')
    axes[9, 0].plot(times, f_traj[:, 2])
    axes[9, 0].set_ylabel(str(fs[2]) + '\n[N-m]')
    axes[9, 1].plot(times, f_traj[:, 3])
    axes[9, 1].set_ylabel(str(fs[3]) + '\n[N-m]')

    axes[10, 0].plot(times, fz_traj[:, 0])
    axes[10, 0].set_ylabel(str(Frz) + '\n[N]')
    axes[10, 1].plot(times, fz_traj[:, 1])
    axes[10, 1].set_ylabel(str(Ffz) + '\n[N]')

    axes[11, 0].plot(times, np.rad2deg(slip_traj[:, 0]))
    axes[11, 0].set_ylabel('alphar\n[deg]')
    axes[11, 1].plot(times, np.rad2deg(slip_traj[:, 1]))
    axes[11, 1].set_ylabel('alphaf\n[deg]')
    axes[12, 0].plot(times, np.rad2deg(slip_traj[:, 2]))
    axes[12, 0].set_ylabel('phir\n[deg]')
    axes[12, 1].plot(times, np.rad2deg(slip_traj[:, 3]))
    axes[12, 1].set_ylabel('phif\n[deg]')

    axes[-1, 0].plot(times, r_traj[:, -1])
    axes[-1, 0].set_ylabel('$F_{kp}$\n[N]')
    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].plot(times, con_traj)
    axes[-1, 1].set_ylabel('constraint\n[m]')
    axes[-1, 1].set_xlabel('Time [s]')
    plt.tight_layout()

    return axes


def plot_wheel_paths(q_traj, q9_traj, q10_traj):
    fig, ax = plt.subplots(1, 1)
    ax.plot(q_traj[:, 0], q_traj[:, 1])
    ax.plot(q9_traj, q10_traj)
    ax.set_aspect('equal')
    return ax


# simplified figure
# compare normal tire numbers and 10% change in slip coefficient
# plot slip angle, lateral force for front and rear steer angle and force input
def plot_minimal(t, q7, ar, af, fkp, fyr, fyf, axes=None, **kwargs):
    if axes is None:
        fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(t, np.rad2deg(q7), color='C0', label=r'$\delta$',
                 **kwargs)
    axes[0].plot(t, np.rad2deg(ar), color='C1', label=r'$\alpha_r$',
                 **kwargs)
    axes[0].plot(t, np.rad2deg(af), color='C2', label=r'$\alpha_f$',
                 **kwargs)
    axes[0].set_ylabel('Angle [deg]')
    axes[0].legend()
    axes[1].plot(t, fkp, color='C0', label='$F_{kp}$', **kwargs)
    axes[1].plot(t, fyr, color='C1', label='$F_{yr}$', **kwargs)
    axes[1].plot(t, fyf, color='C2', label='$F_{yf}$', **kwargs)
    axes[1].set_ylabel('Force [N]')
    axes[1].set_xlabel('Time [s]')
    axes[0].set_xlim((0.0, 2.0))
    axes[1].set_xlim((0.0, 2.0))
    axes[1].legend()
    plt.tight_layout()
    return axes
