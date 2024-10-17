import numpy as np
import matplotlib.pyplot as plt
from symmeplot.matplotlib import Scene3D

from symbols import rr, rf, y, qs, us, rs, ps, d1, d2, d3, Fry_, Ffy_
# TODO : loading these from model.py forces the EoMs to be rebuilt
from model import (N, A, C, E, o, p, do, fo, nd, fn, g1_hat, g2_hat,
                   rear_wheel, front_wheel)
from base_simulation import p_arr, res, FPS, INITIAL_SPEED, KICKDUR

(times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj, q9_traj,
 q10_traj, r_traj) = res

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

scene = Scene3D(N,
                o.locatenew('e', nd.pos_from(o).dot(N.x)*N.x),
                ax=ax, scale=1.0)

# Adds a kick plate.
scene.add_frame(N, p)  # kick plate
kick_length = INITIAL_SPEED*KICKDUR
scene.add_line([
    o,
    o.locatenew('pp', kick_length*N.x),
    p.locatenew('pp', kick_length*N.x),
    p,
    o
], color="k")  # kick plate
scene.add_point(nd, color='C0')
scene.add_point(fn, color='C2')

# Adds moving lines along the "lane".
for side in [-1.0, 1.0]:
    scene.add_line([o.locatenew(f'o{i}',
                                o.pos_from(o) + side*N.y + (0.5*i - 10.0)*N.x)
                    for i in range(40)], marker='.', color='grey')

# Adds the bicycle.
rear_wheel_plot = scene.add_body(rear_wheel,
                                 plot_frame_properties={"scale": 0.3})
rear_wheel_plot.attach_circle(
    rear_wheel.masscenter,
    rr,
    rear_wheel.frame.y,
    facecolor="C0", alpha=0.4, edgecolor="black")

front_wheel_plot = scene.add_body(front_wheel,
                                  plot_frame_properties={"scale": 0.3})
front_wheel_plot.attach_circle(
    front_wheel.masscenter,
    rf,
    front_wheel.frame.y,
    facecolor="C2", alpha=0.4, edgecolor="black")

scene.add_line([
    fo,
    fo.locatenew('d3_half', -d3*E.x),
    fo.locatenew('d2_half', -d3*E.x - d2*E.z),
    do.locatenew('d1_half', d1*C.x),
    do,

], color='k', linewidth=2)

# Adds velocity vectors at wheel contacts to show slip angles.
scene.add_vector(A.x, nd, color="black")
scene.add_vector(nd.vel(N)/INITIAL_SPEED, nd, color="C0")
scene.add_vector(Fry_*A['2']/300, nd, color="C4")

scene.add_vector(g1_hat, fn, color="black")
scene.add_vector(fn.vel(N)/INITIAL_SPEED, fn, color="C2")
scene.add_vector(Ffy_*g2_hat/300, fn, color="C4")

scene.lambdify_system(qs + us + list(rs) + list(ps))
scene.evaluate_system(*np.hstack((q_traj[0, :], u_traj[0, :], r_traj[0, :],
                                  p_arr)))

scene.plot()

ax.invert_zaxis()

ax.set_xlim((-0.5, 1.5))
ax.set_ylim((-1.0, 1.0))
ax.set_zlim((1.0, -1.0))

slow_factor = 3  # int
ani = scene.animate(lambda i: np.hstack((q_traj[i, :], u_traj[i, :],
                                         r_traj[i, :], p_arr)),
                    frames=len(times), interval=slow_factor/FPS*1000)
ani.save("animation.gif", fps=FPS//slow_factor)
