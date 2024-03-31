import numpy as np
from pydy.viz import (Sphere, Cylinder, VisualizationFrame, Scene,
                      PerspectiveCamera)
from simulate import *
from base_simulation import *

(times, q_traj, u_traj, slip_traj, f_traj, fz_traj, con_traj, q9_traj,
 q10_traj, r_traj) = res

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
