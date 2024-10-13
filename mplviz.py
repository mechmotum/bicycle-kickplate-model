import numpy as np
import matplotlib.pyplot as plt

from symbols import rr, rf, y, qs, ps, d1, d2, d3
# TODO : loading these from model.py forces the EoMs to be rebuilt
from model import N, C, E, o, p, do, fo, nd, rear_wheel, front_wheel
from symmeplot.matplotlib import Scene3D
from base_simulation import p_arr, res, fps, initial_speed, KICKDUR

(times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj, q9_traj,
 q10_traj, r_traj) = res

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

scene = Scene3D(N,
                o.locatenew('e', nd.pos_from(o).dot(N.x)*N.x),
                ax=ax, scale=2.0)

scene.add_frame(N, p)  # kick plate
kick_length = initial_speed*KICKDUR
scene.add_line([o, o.locatenew('pp', kick_length*N.x)])  # reference
scene.add_line([p, p.locatenew('pp', kick_length*N.x)])  # kick plate
scene.add_point(nd, color='green')

rear_wheel_plot = scene.add_body(rear_wheel)
rear_wheel_plot.attach_circle(
    rear_wheel.masscenter,
    rr,
    rear_wheel.frame.y,
    facecolor="none", edgecolor="k")

front_wheel_plot = scene.add_body(front_wheel)
front_wheel_plot.attach_circle(
    front_wheel.masscenter,
    rf,
    front_wheel.frame.y,
    facecolor="none", edgecolor="k")

scene.add_line([
    fo,
    fo.locatenew('d3_half', -d3*E.x),
    fo.locatenew('d2_half', -d3*E.x - d2*E.z),
    do.locatenew('d1_half', d1*C.x),
    do,
])

scene.lambdify_system(qs + [y] + list(ps))
scene.evaluate_system(*np.hstack((q_traj[0, :], r_traj[0, 4:5], p_arr)))

scene.plot()

ax.invert_zaxis()

slow_factor = 3  # int
ani = scene.animate(lambda i: np.hstack((q_traj[i, :], r_traj[i, 4:5], p_arr)),
                    frames=len(times), interval=slow_factor/fps*1000)
ani.save("animation.gif", fps=fps//slow_factor)
