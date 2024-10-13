import numpy as np
import matplotlib.pyplot as plt
from symmeplot.matplotlib import Scene3D

from symbols import rr, rf, y, qs, us, ps, d1, d2, d3
# TODO : loading these from model.py forces the EoMs to be rebuilt
from model import (N, A, C, E, o, p, do, fo, nd, fn, g1_hat, rear_wheel,
                   front_wheel)
from base_simulation import p_arr, res, FPS, INITIAL_SPEED, KICKDUR

(times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj, q9_traj,
 q10_traj, r_traj) = res

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

scene = Scene3D(N,
                o.locatenew('e', nd.pos_from(o).dot(N.x)*N.x),
                ax=ax, scale=2.0)

scene.add_frame(N, p)  # kick plate
kick_length = INITIAL_SPEED*KICKDUR
scene.add_line([
    o,
    o.locatenew('pp', kick_length*N.x),
    p.locatenew('pp', kick_length*N.x),
    p,
    o
], color="k")  # kick plate
scene.add_point(nd, color='C1')
scene.add_point(fn, color='C1')

rear_wheel_plot = scene.add_body(rear_wheel)
rear_wheel_plot.attach_circle(
    rear_wheel.masscenter,
    rr,
    rear_wheel.frame.y,
    facecolor="none", edgecolor="C0")

front_wheel_plot = scene.add_body(front_wheel)
front_wheel_plot.attach_circle(
    front_wheel.masscenter,
    rf,
    front_wheel.frame.y,
    facecolor="none", edgecolor="C2")

scene.add_line([
    fo,
    fo.locatenew('d3_half', -d3*E.x),
    fo.locatenew('d2_half', -d3*E.x - d2*E.z),
    do.locatenew('d1_half', d1*C.x),
    do,

], color='k')

scene.add_vector(A.x, nd, color="black")
scene.add_vector(nd.vel(N).normalize(), nd, color="C0")

# TODO : Adding this makes the script significantly slower.
scene.add_vector(g1_hat, fn, color="black")
scene.add_vector(fn.vel(N).normalize(), fn, color="C2")

scene.lambdify_system(qs + us + [y] + list(ps))
scene.evaluate_system(*np.hstack((q_traj[0, :], u_traj[0, :], r_traj[0, 4:5],
                                  p_arr)))

scene.plot()

ax.invert_zaxis()

slow_factor = 3  # int
ani = scene.animate(lambda i: np.hstack((q_traj[i, :], u_traj[i, :],
                                         r_traj[i, 4:5], p_arr)),
                    frames=len(times), interval=slow_factor/FPS*1000)
ani.save("animation.gif", fps=FPS//slow_factor)
