import numpy as np
import matplotlib.pyplot as plt

from symbols import qs, us, fs, c_ar, c_mpr, c_pr, c_mar
from tire_data import SchwalbeT03_500kPa as TIRE
from inputs import calc_linear_tire_force, calc_nonlinear_tire_force


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

    axes[-1, 0].plot(times, r_traj[:, 6])
    axes[-1, 0].set_ylabel('$\ddot{y}$\n[m/s/s]')
    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].plot(times, con_traj)
    axes[-1, 1].set_ylabel('constraint\n[m]')
    axes[-1, 1].set_xlabel('Time [s]')


def plot_kick_motion(times, r_traj):

    fig, axes = plt.subplots(3, 1, sharex=True, layout='constrained')
    axes[0].plot(times, r_traj[:, 6])
    axes[0].set_ylabel(r'$\ddot{y}$ [m/s/s]')
    axes[1].plot(times, r_traj[:, 5])
    axes[1].set_ylabel(r'$\dot{y}$ [m/s]')
    axes[2].plot(times, r_traj[:, 4])
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
