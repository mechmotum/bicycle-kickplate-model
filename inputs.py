import numpy as np


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
