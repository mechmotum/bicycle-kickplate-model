import numpy as np


def calc_linear_tire_force(alpha, phi, Fz, c_a, c_p, c_ma, c_mp):
    """Returns the lateral force and self-aligning moment at the contact patch
    acting on the tire.

    ::

       Fy = (c_a*alpha + c_p*phi)*Fz
       Mz = -(c_ma*alpha - c_mp*phi)*Fz

    Parameters
    ==========
    alpha : float
        Lateral slip angle, positive is yaw to the right.
    phi : float
        Camber angle, positive is roll to the right.
    Fz : float
        Normal force, negative in compression.
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
    Fy = (c_a*alpha + c_p*phi)*Fz
    Mz = -(c_ma*alpha - c_mp*phi)*Fz
    return Fy, Mz


def calc_nonlinear_tire_force(alpha, phi, Fz, tire_data):
    """Returns the lateral force and self-aligning moment at the contact patch
    acting on the tire. This is an implementation of Pacejka 1989 tire model
    developed in Dell'Orto et al. 2024.

    Parameters
    ==========
    alpha : float
        Lateral slip angle in radians, positive is yaw to the right.
    phi : float
        Camber angle in radians, positive is roll to the right.
    Fz : float
        Normal force in Newtons, negative in compression.
    tire_data : TireCoefficients
        Pacejka 89 tire model constants based on data produced in Dell'Orto
        2024.

    Returns
    =======
    Fy : float
        Lateral force in Newtons, positive to the right.
    Mz : float
        Self-aligning moment in Newton-Meters, positive moment will turn wheel
        to the right.

    References
    ==========

    Bakker E, Pacejka HB, Lidner L. A new tire model with an application in
    vehicle dynamics studies. SAE Tech Pap. 1989;98:101–113. doi:10.4271/890087

    Dell’Orto, G., Mastinu, G., Happee, R., & Moore, J. K. (2024). Measurement
    of lateral characteristics and identification of Magic Formula parameters
    of city and cargo bicycle tyres. Vehicle System Dynamics (Under Review).
    https://doi.org/10.1080/00423114.2024.2338143

    """

    Fz = -Fz/1000.00  # MUST be in [kN]
    alpha = np.rad2deg(alpha)    # angles input in [deg]
    phi = np.rad2deg(phi)        # angles input in [deg]

    opt_Pac_fy = tire_data.Fy_coef
    opt_Pac_Mz = tire_data.Mz_coef

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

    # Used to adjust the Friction coefficient indoor test-rig VS kickplate
    # sandpaper
    # Obtained as: Friction coeff kickplate / Friction coeff test-rig
    # TODO : This should only be applied to the rear wheel.
    Friction_coeff = 1.31917  # 1.279368
    Fy = Fy * Friction_coeff
    Mz = Mz * Friction_coeff

    return -Fy, -Mz


def calc_full_state_feedback_steer_torque(t, x, k):
    """Simple LQR control based on linear Carvallo-Whipple model.

    Parameters
    ==========
    t : float
        Time in seconds.
    x : array_like, shape(24,)
        State values where x = [q1, q2, q3, q4, q5, q6, q7, q8, q11, q12,
                                u1, u2, u3, u4, u5, u6, u7, u8, u11, u12,
                                Fry, Ffy, Mrz, Mfz].
    k : array_like, shape(4,)
        Gains: [kq4, ku4, kq7, ku7]

    """

    q = x[:10]
    u = x[10:20]

    q4 = q[3]
    q7 = q[6]
    u4 = u[3]
    u7 = u[6]

    kq4, ku4, kq7, ku7 = k

    return -(kq4*q4 + kq7*q7 + ku4*u4 + ku7*u7)


def calc_kick_force_pulse(t):
    """Returns the lateral forced applied to the tire by the kick plate. The
    force is modeled as a sinusoidal pulse."""

    start = 0.4  # seconds
    stop = 0.6  # seconds
    magnitude = 700  # Newtons

    period = stop - start
    frequency = 1.0/period
    omega = 2*np.pi*frequency  # rad/s

    if start + period/2 < t < stop:
        return magnitude/2.0*(1.0 - np.cos(omega*(t - start)))
    else:
        return 0.0


def calc_kick_motion_constant_acc(t, duration=0.15):
    """Returns the kick plate displacement, velocity, and acceleration assuming
    a constant acceleration and instaneous deceleration with a plate
    displacement of 15 cm in the specified duration in seconds. Constant
    acceleration is assumed because the air cylinder force is approximately
    constant based on the pressure sensor measurement.

    Parameters
    ==========
    duration : float, optional
        Duration in seconds of the kick plate displacement.

    Returns
    =======
    y : float
        Displacement at time ``t``.
    yd : float
        Speed at time ``t``.
    ydd : float
        Acceleration at time ``t``.

    """

    kick_displacement = 0.15  # meters

    # y(t) = m*t**2
    # y'(t) = 2*m*t
    # y''(t) = 2*m
    # y(duration) = d = m*duration**2 -> d = m*duration**2 -> m = d/(duration**2)

    m = kick_displacement/(duration**2)
    if 0.0 <= t < duration:
        y, yd, ydd = m*t**2, 2.0*m*t, 2.0*m
    elif t >= duration:
        y, yd, ydd = kick_displacement, 0.0, 0.0
    else:
        y, yd, ydd = 0.0, 0.0, 0.0

    return y, yd, ydd


def calc_kick_motion_pulse_acc(t):
    """Returns the kick plate displacement, velocity, and acceleration assuming
    a sinusoidal pulse acceleration."""

    start = 0.4  # seconds
    stop = 0.6  # seconds
    magnitude = 20.0  # m/s/s

    period = stop - start
    frequency = 1.0/period
    omega = 2*np.pi*frequency  # rad/s

    # TODO : figure out how to calculate the integration constants (-0.2 and
    # -1.0)
    if start < t < stop:
        y = magnitude/2.0*(t**2/2.0 - (-np.cos(omega*(t - start))/omega)/omega) - 0.8
        yd = magnitude/2.0*(t - np.sin(omega*(t - start))/omega) - 4.0
        ydd = magnitude/2.0*(1.0 - np.cos(omega*(t - start)))
    elif t >= stop:
        y, yd, ydd = 1.0, 0.0, 0.0
    else:
        y, yd, ydd = 0.0, 0.0, 0.0

    return y, yd, ydd
