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
        where: r = [T4, T6, T7, fkp, Fry, Ffy, Mrz, Mfz].
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
    max_roll = np.deg2rad(45.0)
    return max_roll - np.abs(x[3])


fall_detector.terminal = True
fall_detector.direction = -1


def equilibrium_eq(q, p):
    """Returns the static equilibrium configuration of the model."""
    # TODO : When I change the tire vertical stiffness values I don't get a
    # change in equilibrium state. So this doesn't seem to work in a full proof
    # way.
    # TODO : May be better to solve Fr + Frstar + holonomic constraint all
    # equal zero.

    u = np.ones(10)*1e-13  # divide by zeros if u is simple all zeros
    f = np.zeros(4)  # Fry, Ffy, Mrz, Mfz
    r = np.zeros(8)  # T4, T6, T7, Fkp, Fry_, Ffy_, Mrz_, Mfz_

    def zeros(x, p):
        _, b = eval_dynamic(x, u, f, r, p)
        return np.squeeze(b[0:10])

    return fsolve(zeros, q, args=p, xtol=1e-11)


def calc_linear_tire_force(alpha, phi, Fz, c_a, c_p, c_ma, c_mp):
    """Returns the lateral force and self-aligning moment at the contact patch
    acting on the tire.

    Parameters
    ==========
    alpha : float
        Lateral slip angle, positive is yaw to the right.
    phi : float
        Camber angle, positive is roll to the right.
    Fz : float
        Normal force, positive in compression.
    c_a, c_p, c_ma, c_mp : floats
        Laterial slip and camber coefficients for force and moment, all
        positive values.

    Returns
    =======
    Fy : float
        Lateral force, positive to the right.
    Mz : float
        Self-aligning moment, positive moment will turn wheel to the right.

    """
    Fy = (-c_a*alpha + c_p*phi)*Fz
    # TODO : Make the sign of the camber effect on self-aligning moment is
    # correct.
    # TODO : This -c_ma*alpha*Fz should be positive! Don't understand this as
    # it implies that a positive slip angle velocity vector pointing to right
    # causes a negative moment pushing the wheel away from the velocity vector.
    Mz = -(c_ma*alpha + c_mp*phi)*Fz
    return Fy, Mz


def calc_nonlinear_tire_force(alpha, phi, Fz):
    """Returns the lateral force and self-aligning moment at the contact patch
    acting on the tire.

    Parameters
    ==========
    alpha : float
        Lateral slip angle in radians, positive is yaw to the right.
    phi : float
        Camber angle in radians, positive is roll to the right.
    Fz : float
        Normal force in Newtons, negative in compression.

    Returns
    =======
    Fy : float
        Lateral force in Newtons, positive to the right.
    Mz : float
        Self-aligning moment in Newton-Meters, positive moment will turn wheel
        to the right.

    """

    Fz = Fz/1000.00  # MUST be in [kN]

    ## Old simple case
    opt_Pac_fy = [
        # 0.296295594442076,  #values from OptPar_deg
        # 3896.98838593407,
        # 3964.30596402491,
        # 115.333054984316,
        # 0.638805088213141,
        # 0.000552109911180092,
        # 4.88569512529486,
        # -0.0832906867626918,
        # 0.0354269547190490,
        # -0.116088892942258,
        # -0.00175245144219161,
        # 6.50409689362419,
        # -4.56517105416890,
        # -1.26062915456958,
        4802.49127129661,           #values from Opt Par_rad
        -411.483943584675,
        1039.78973919905,
        7770.72927522235,
        0.990788636848624,
        0.000470000000000000,
        195.543489325314,
        432.125989933535,
        0.0400000000000000,
        0.00462948605575456,
        -0.00204252002863379,
        23,
        -28.7133576301737,
        5.90903596659257,
        # 0.0421157577662902,	    # From OptParameterMF_T03R01_camber0_p300.mat
        # 6921.70703980922,
        # 36184.4484921848,
        # 117.679709488861,
        # 0.671843568786086,
        # 0.00244548261240910,
        # 3.28647094098780,
        # 0.567243118050733,
        # 0.0405091858662170,
        # 0.0665947740791520,
        # -0.0660233172268580,
        # 5.13196785763573,
        # -8.70159208401445,
        # -0.228798626988778,
    ]

    opt_Pac_Mz = [
        # 4.64291579665827,   #values from OptPar_deg
        # 10.0307510263452,
        # 2.93733400697400,
        # 0.00538014988781111,
        # -3.09108719250574,
        # -0.958523366802777,
        # -0.00614502879858124,
        # 1.85716282505503,
        # -1.53447284898031,
        # 0.318387396748285,
        # -9.15012640518871,
        # 0.0728219670843347,
        # 0.411869078464016,
        # -0.240103083582358,
        # -0.000111504690469878,
        # 0.329856499993466,
        # 0.491921917244135,
        # -0.0986453301580708
        0.187591378220407,  #values from Opt Par_rad
        137.765934193791,
        1.71813617983282,
        -35.2039167373913,
        -230.570331500149,
        -0.462270268994748,
        0.00276990000000000,
        -10.9248978046150,
        10.4554960206743,
        -0.921974478744944,
        0.0255010000000000,
        -0.0235700000000000,
        -0.00106432726384250,
        0.000198958173929200,
        0.0211329000000000,
        0.894690000000000,
        0.131350350303294,
        0.0878457704136251,
        # 1.37808763179523,       	# From OptParameterMF_T03R01_camber0_p300.mat
        # 4.24701824645087,
        # 5.86957508131411,
        # 0.000408850229834906,
        # -5.88153147501883,
        # 0.219451579832919,
        # 0.00418578530343426,
        # -44.7275441651533,
        # 25.9062723115695,
        # -2.36085049584848,
        # 4.62449995053518,
        # 0.0871018174594679,
        # -0.226373443969147,
        # 0.0909174155928722,
        # 0.232326352204219,
        # 0.215539422811174,
        # 0.0268209223191697,
        # 0.139175737361296
    ]

    C_mz = opt_Pac_Mz[0]  # Shape factor
    D_mz = (opt_Pac_Mz[1]*Fz**2 + opt_Pac_Mz[2]*Fz)  # Peak factor
    BCD_mz = ((opt_Pac_Mz[3]*Fz**2 + opt_Pac_Mz[4]*Fz) *
              (1 - opt_Pac_Mz[6]*np.abs(phi))*np.exp(-opt_Pac_Mz[5]*Fz))
    B_mz= BCD_mz/(C_mz*D_mz)  # Stiffness factor
    Sh_mz = (opt_Pac_Mz[11]*phi + opt_Pac_Mz[12]*Fz +
             opt_Pac_Mz[13])  # Horizontal shift
    Sv_mz = ((opt_Pac_Mz[14]*Fz**2 + opt_Pac_Mz[15]*Fz)*phi +
             opt_Pac_Mz[16]*Fz + opt_Pac_Mz[17])  # Vertical shift
    X1_mz = alpha + Sh_mz  # Composite
    E_mz = ((opt_Pac_Mz[7]*Fz**2 + opt_Pac_Mz[8]*Fz + opt_Pac_Mz[9])*
            (1 - opt_Pac_Mz[10]*np.abs(phi)))  # Curvature factor

    # TODO : Mz causes a slightly unstable oscillation.
    # Evaluation of Mz
    Mz = (D_mz*np.sin(C_mz*np.arctan(B_mz*X1_mz -
          E_mz*(B_mz*X1_mz - np.arctan(B_mz*X1_mz))))) + Sv_mz

    C_fy = opt_Pac_fy[0]  # Shape factor
    D_fy = (opt_Pac_fy[1]*Fz**2 + opt_Pac_fy[2]*Fz)  # Peak factor
    BCD_fy = (opt_Pac_fy[3]*np.sin(np.arctan(Fz/opt_Pac_fy[4])*2)*
              (1 - opt_Pac_fy[5]*np.abs(phi)))
    B_fy = BCD_fy/(C_fy*D_fy)  # Stiffness factor
    Sh_fy = (opt_Pac_fy[9]*Fz + opt_Pac_fy[10] +
             opt_Pac_fy[8]*phi)  # Horizontal shift
    Sv_fy = (opt_Pac_fy[11]*Fz*phi + opt_Pac_fy[12]*Fz +
             opt_Pac_fy[13])  # Vertical shift
    X1_fy = alpha + Sh_fy  # Composite
    E_fy = opt_Pac_fy[6]*Fz + opt_Pac_fy[7]

    # Evaluation of Fy
    Fy = (D_fy*np.sin(C_fy*np.arctan(B_fy*X1_fy -
          E_fy*(B_fy*X1_fy - np.arctan(B_fy*X1_fy))))) + Sv_fy

    return -Fy, Mz


########################
# Setup Numerical Values
########################

# batavus browser with Jason sitting on it, tire parameters from
# Andrew/Gabriele
p_vals = {
    c_af: 11.46,  # estimates from Andrew's dissertation (done by him)
    c_ar: 11.46,
    c_f: 4000.0,  # guess
    c_maf: 0.33,  # 0.33 is rough calc from gabriele's data
    c_mar: 0.33,
    c_mpf: 0.0,  # need real numbers for this
    c_mpr: 0.0,  # need real numbers for this
    c_pf: 0.573,
    c_pr: 0.573,
    c_r: 4000.0,  # guess
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
    k_f: 120000.0,  # ~ twice the stiffness of a 1.25" tire from Rothhamel 2024
    k_r: 120000.0,  # ~ twice the stiffness of a 1.25" tire from Rothhamel 2024
    l1: 0.5384415640161426,
    l2: -0.531720230353059,
    l3: -0.07654646159268344,
    l4: -0.47166687226492093,
    mc: 81.86,
    md: 3.11,
    me: 3.22,
    mf: 2.02,
    r_tf: 0.01,
    r_tr: 0.01,
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
    q_vals : array_like, shape(10,)
        [q1, q2, q3, q4, q5, q6, q7, q8, q11, q12]
    u_vals : array_like, shape(10,)
        [u1, u2, u3, u4, u5, u6, u7, u8, u11, u12]
    f_vals: array_like, shape(4,)
        [Fry, Ffy, Mrz, Mfz]
    p_arr: array_like, shape(42,)

    """

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

    q_eq = equilibrium_eq(q_vals, p_arr)
    print('Equilibrium coordinates: ', q_eq)

    q_vals[-2:] = q_eq[-2:]

    print('Independent generalized speeds:', u_vals[[2, 3, 5, 6, 7, 8, 9]])
    A_nh_vals, B_nh_vals = eval_dep_speeds(q_vals,
                                           u_vals[[2, 3, 5, 6, 7, 8, 9]],
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
    conditions, and inputs and calcaluate any output variables."""

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
    r_traj = np.zeros((len(times), 4))
    for i, (ti, qi, ui, fi) in enumerate(zip(times, q_traj, u_traj, f_traj)):
        statei = np.hstack((qi, ui, fi))
        fz_traj[i, :] = np.array([p[27]*qi[8] + p[9]*ui[8],
                                  p[26]*qi[9] + p[2]*ui[9]])
        slip_traj[i, :] = eval_angles(qi, ui, p)
        q9_traj[i], q10_traj[i] = eval_front_contact(qi, p)
        r_traj[i] = calc_inputs(ti, statei, p)[:4]

    return (times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj,
            q9_traj, q10_traj, r_traj)


def plot_all(times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj,
             q9_traj, q10_traj, r_traj):

    deg = [False, False, True, True, True, True, True, True, False, False]
    fig, axes = plt.subplots(16, 2, sharex=True)
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
