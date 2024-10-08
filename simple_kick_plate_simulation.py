"""This is a simple single mass simulation to try to understand how the
kickplate may move when hit with the force from the pressurized cylinder. Most
of the constants are best guesses, but the basic acceleration profile can be
seen and compared with the accelerometer measurement taken on the kickplate.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

m = 25.0  # mass of plate + some portion of bike/rider [kg]
mu = 1.2  # coefficient of friction of plate with rails
g = 9.81  # acceleration due to gravity [m/s/s]
k = 10000000.0  # stopper Hunt/Crossley model stiffness
c = 1000.0  # stopper Hunt/Crossley model damping
x_wall = 0.15  # plate travel distance [m]


def rhs(t, s):

    x = s[0]
    v = s[1]

    # constant piston force
    if t < 0.05:
        Fp = 1500.0  # N
    else:
        Fp = 0.0

    # stopper collision force
    if x > 0.15:
        xterm = (x - x_wall)**(3.0/2.0)
        Fw = k*xterm + c*v*xterm
    else:
        Fw = 0.0

    # friction force
    if np.abs(v) < 0.01:
        Ff = 0.0
    else:
        Ff = np.sign(v)*mu*m*g

    dxdt = v
    dvdt = (-Ff - Fw + Fp)/m

    return dxdt, dvdt, Fp, Fw


t0, tf, s0 = 0.0, 0.3, np.array([0.0, 0.0])
t = np.linspace(t0, tf, num=1001)

sol = solve_ivp(lambda t, x: rhs(t, x)[0:2],
                (t0, tf), s0, t_eval=t) #, method='LSODA')

a, Fp, Fw = [], [], []
for ti, si in zip(sol.t, sol.y.T):
    _, ai, Fpi, Fwi = rhs(ti, si)
    a.append(ai)
    Fp.append(Fpi)
    Fw.append(Fwi)
a = np.array(a)
Fp = np.array(Fp)
Fw = np.array(Fw)

fig, axes = plt.subplots(5, 1, sharex=True)

axes[0].plot(sol.t, Fp)
axes[0].set_ylabel(r'$F_p$')
axes[1].plot(sol.t, Fw)
axes[1].set_ylabel(r'$F_w$')
axes[2].plot(sol.t, sol.y[0])
axes[2].set_ylabel(r'$x$')
axes[3].plot(sol.t, sol.y[1])
axes[3].set_ylabel(r'$v$')
axes[4].plot(sol.t, a)
axes[4].set_ylabel(r'$a$')

plt.show()
