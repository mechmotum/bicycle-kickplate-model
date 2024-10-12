import matplotlib.pyplot as plt

from symbols import rr, rf, y, qs, ps
from model import N, o, rear_wheel, front_wheel
from symmeplot.matplotlib import Scene3D
from base_simulation import *

(times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj, q9_traj,
 q10_traj, r_traj) = res

scene = Scene3D(N, o) #, scale=0.5)
rear_wheel_plot = scene.add_body(rear_wheel)
rear_wheel_plot.attach_circle(
    rear_wheel.masscenter,
    rr,
    rear_wheel.frame.y,
    facecolor="red", edgecolor="k")

front_wheel_plot = scene.add_body(front_wheel)
front_wheel_plot.attach_circle(
    front_wheel.masscenter,
    rf,
    front_wheel.frame.y,
    facecolor="red", edgecolor="k")

#scene.add_body(rear_frame_body)
#scene.add_line(some_points_describing_the_rear_frame)
#
#scene.add_body(front_frame_body)
#scene.add_line(some_points_describing_the_front_frame)

# For a single state plot:
scene.lambdify_system(qs + [y] + list(ps))
scene.evaluate_system(*np.hstack((q_traj[0, :], r_traj[0, 4:5], p_arr)))

plt.show()

#ani = scene.animate(lambda q: (q,), frames=np.linspace(0, 2 * np.pi, 60))
#ani.save("animation.gif", fps=30)
