import numpy as np
from pydy.viz import (Sphere, Cylinder, VisualizationFrame, Scene,
                      PerspectiveCamera)

from simulate import *


def calc_fkp(t):
    """Returns the lateral forced applied to the tire by the kick plate."""

    if t > 0.5 and t < 1.0:
        return 500.0
    else:
        return 0.0


def calc_steer_torque(t, x):

    q4 = x[3]
    q7 = x[6]
    u4 = x[11]
    u7 = x[14]

    # LQR gains for Whipple model at 6 m/s
    kq4 = -2.2340917377023612
    kq7 = 4.90641020775064
    ku4 = -0.5939384880650549
    ku7 = 0.4340987861323103

    # LQR gains for Whipple model at 3 m/s
    #kq4 = -23.183400610625647
    #kq7 = 17.086409893261113
    #ku4 = -7.999852938163112
    #ku7 = 1.8634394089384874

    return -(kq4*q4 + kq7*q7 + ku4*u4 + ku7*u7)


def calc_inputs(t, x, p):

    # steer, rear wheel, roll torques set to zero
    T4, T6, T7 = 0.0, 0.0, calc_steer_torque(t, x)

    # kick plate force
    fkp = calc_fkp(t)

    r = [T4, T6, T7, fkp]

    return r


# initial coordinates
q_vals = np.array([
    0.0,  # q1
    0.0,  # q2
    0.0,  # q3
    0.0,  # q4
    np.nan,  # q5
    0.0,  # q6
    1e-14,  # q7, setting to zero gives singular matrix
    0.0,  # q8
])

# initial speeds
initial_speed = 6.0  # m/s
u_vals = np.array([
    np.nan,  # u1
    np.nan,  # u2
    0.0,  # u3, rad/s
    1e-10,  # u4, rad/s
    np.nan,  # u5, rad/s
    -initial_speed/p_vals[rr],  # u6
    0.0,  # u7
    -initial_speed/p_vals[rf],  # u8
])

# initial tire forces
# TODO : Need to figure out how we know the initial state of the tire forces.
f_vals = np.array([0.0, 0.0, 0.0, 0.0])

initial_conditions = setup_initial_conditions(q_vals, u_vals, f_vals, p_arr)

fps = 60  # frames per second
duration = 6.0  # seconds

times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj, q9_traj, q10_traj, r_traj = simulate(
    duration, calc_inputs, initial_conditions, p_arr, fps=fps)

rear_wheel_circle = Cylinder(radius=p_vals[rr], length=0.01,
                             color="green", name='rear wheel')
front_wheel_circle = Cylinder(radius=p_vals[rf], length=0.01,
                              color="green", name='front wheel')

rear_wheel_vframe = VisualizationFrame(B, do, rear_wheel_circle)
front_wheel_vframe = VisualizationFrame(E, fo, front_wheel_circle)

d1_cylinder = Cylinder(radius=0.02, length=p_vals[d1],
                       color='black', name='rear frame d1')
d2_cylinder = Cylinder(radius=0.02, length=p_vals[d2],
                       color='black', name='front frame d2')
d3_cylinder = Cylinder(radius=0.02, length=p_vals[d3],
                       color='black', name='front frame d3')

d1_frame = VisualizationFrame(C.orientnew('C_r', 'Axis', (sm.pi/2, C.z)),
                              do.locatenew('d1_half', d1/2*C.x), d1_cylinder)
d2_frame = VisualizationFrame(E.orientnew('E_r', 'Axis', (-sm.pi/2, E.x)),
                              fo.locatenew('d2_half', -d3*E.x - d2/2*E.z), d2_cylinder)
d3_frame = VisualizationFrame(E.orientnew('E_r', 'Axis', (sm.pi/2, E.z)),
                              fo.locatenew('d3_half', -d3/2*E.x), d3_cylinder)

co_sphere = Sphere(radius=0.05, color='blue', name='rear frame co')
eo_sphere = Sphere(radius=0.05, color='blue', name='rear frame eo')
co_frame = VisualizationFrame(C, co, co_sphere)
eo_frame = VisualizationFrame(E, eo, eo_sphere)

# TODO : Moving camera does not seem to work.
camera_loc = o.locatenew('p_camera', -10*N.z + q1*N.x)
camera = PerspectiveCamera('camera', N, camera_loc)

scene = Scene(N, o) #, cameras=[camera])
scene.visualization_frames = [front_wheel_vframe, rear_wheel_vframe, d1_frame,
                              d2_frame, d3_frame, co_frame, eo_frame]

scene.times = times
scene.constants = p_vals
scene.states_symbols = qs + us
scene.states_trajectories = np.hstack((q_traj, u_traj))
