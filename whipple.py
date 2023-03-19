#!/usr/bin/env python

"""This file derives the non-linear equations of motion of the Carvallo-Whipple
bicycle model ([Carvallo1899]_, [Whippl1899]_) following the description and
nomenclature in [Moore2012]_ and produces Octave functions that calculate the
lateral wheel-ground constraint force for each wheel given the essential
kinematics of the vehicle.

References
==========

.. [Whipple1899] Whipple, Francis J. W. "The Stability of the Motion of a
   Bicycle." Quarterly Journal of Pure and Applied Mathematics 30 (1899):
   312–48.
.. [Carvallo1899] Carvallo, E. Théorie Du Mouvement Du Monocycle et de La
   Bicyclette. Paris, France: Gauthier- Villars, 1899.
.. [Moore2012] Moore, Jason K. "Human Control of a Bicycle." Doctor of
   Philosophy, University of California, 2012.
   http://moorepants.github.io/dissertation.

"""

import os

import sympy as sm
import sympy.physics.mechanics as mec
from pydy.codegen.octave_code import OctaveMatrixGenerator
from scipy.optimize import fsolve
import numpy as np
from scipy.integrate import solve_ivp

from utils import ReferenceFrame, decompose_linear_parts, cramer_solve, euler_integrate

##################
# Reference Frames
##################

print('Defining reference frames.')

# Newtonian Frame
N = ReferenceFrame('N')
# Yaw Frame
A = ReferenceFrame('A')
# Roll Frame
B = ReferenceFrame('B')
# Rear Frame
C = ReferenceFrame('C')
# Rear Wheel Frame
D = ReferenceFrame('D')
# Front Frame
E = ReferenceFrame('E')
# Front Wheel Frame
F = ReferenceFrame('F')

####################################
# Generalized Coordinates and Speeds
####################################

# All the following are a function of time.
t = mec.dynamicsymbols._t

print('Defining time varying symbols.')

# q1: perpendicular distance from the n2> axis to the rear contact
#     point in the ground plane
# q2: perpendicular distance from the n1> axis to the rear contact
#     point in the ground plane
# q3: frame yaw angle
# q4: frame roll angle
# q5: frame pitch angle
# q6: rear wheel rotation angle
# q7: steering rotation angle
# q8: front wheel rotation angle
# q9: perpendicular distance from the n2> axis to the front contact
#     point in the ground plane
# q10: perpendicular distance from the n1> axis to the front contact
#     point in the ground plane
q1, q2, q3, q4 = mec.dynamicsymbols('q1, q2, q3, q4')
q5, q6, q7, q8 = mec.dynamicsymbols('q5, q6, q7, q8')
q10, q11 = mec.dynamicsymbols('q10, q11')

# u1: speed of the rear wheel contact point in the n1> direction
# u2: speed of the rear wheel contact point in the n2> direction
# u3: frame yaw angular rate
# u4: frame roll angular rate
# u5: frame pitch angular rate
# u6: rear wheel rotation angular rate
# u7: steering rotation angular rate
# u8: front wheel rotation angular rate
u1, u2, u3, u4 = mec.dynamicsymbols('u1, u2, u3, u4')
u5, u6, u7, u8 = mec.dynamicsymbols('u5, u6, u7, u8')

# u9: speed of the front wheel contact point in the n1> direction
# u10: speed of the front wheel contact point in the n2> direction
# u11: auxiliary speed to determine the rear tire vertical force
# u12: auxiliary speed to determine the front tire vertical force
u9, u10, u11, u12 = mec.dynamicsymbols('u9, u10, u11, u12')

###########
# Specified
###########

# kickplate lateral position
y, yd, ydd = mec.dynamicsymbols('y, y_d, y_dd')

# control torques
# T4 : roll torque
# T6 : rear wheel torque
# T7 : steer torque
T4, T6, T7 = mec.dynamicsymbols('T4, T6, T7')

# Fry : rear wheel-ground contact lateral force
# Frz : rear wheel-ground contact normal force
# Mrz : rear wheel-ground contact self-aligning moment
# Ffy : front wheel-ground contact lateral force
# Ffz : front wheel-ground contact normal force
# Mfz : front rear wheel-ground contact self-aligning moment
Fry, Frz, Mrz, Ffy, Ffz, Mfz = mec.dynamicsymbols(
    'Fry, Frz, Mrz, Ffy, Ffz, Mfz')

#################################
# Orientation of Reference Frames
#################################

print('Orienting frames.')

# The following defines a 3-1-2 Tait-Bryan rotation with yaw (q3), roll
# (q4), pitch (q5) angles to orient the rear frame relative to the ground
# (Newtonian frame). The front frame is then rotated through the steer
# angle (q7) about the rear frame's 3 axis. The wheels are not oriented, as
# q6 and q8 end up being ignorable coordinates.

# rear frame yaw
A.orient(N, 'Axis', (q3, N['3']))
# rear frame roll
B.orient(A, 'Axis', (q4, A['1']))
# rear frame pitch
C.orient(B, 'Axis', (q5, B['2']))
# front frame steer
E.orient(C, 'Axis', (q7, C['3']))

# create a front "yaw" frame that is equivalent to the A frame for the rear
# wheel.
# G['1'] lies in the ground plane and points in the direction of the wheel
# contact path E['2'] X A['3'] gives this unit vector.
# G['2'] lies in the ground plane and points perpendicular to the wheel
# contact path. A['3'] X G['1'] gives this unit vector.
g1_hat = E['2'].cross(A['3'])
g2_hat = A['3'].cross(g1_hat)

###########
# Constants
###########

print('Defining constants.')

# geometry
# rf: radius of front wheel
# rr: radius of rear wheel
# d1: the perpendicular distance from the steer axis to the center
#     of the rear wheel (rear offset)
# d2: the distance between wheels along the steer axis
# d3: the perpendicular distance from the steer axis to the center
#     of the front wheel (fork offset)
# l1: the distance in the c1> direction from the center of the rear
#     wheel to the frame center of mass
# l2: the distance in the c3> direction from the center of the rear
#     wheel to the frame center of mass
# l3: the distance in the e1> direction from the front wheel center to
#     the center of mass of the fork
# l4: the distance in the e3> direction from the front wheel center to
#     the center of mass of the fork
rf, rr = sm.symbols('rf, rr')
d1, d2, d3 = sm.symbols('d1, d2, d3')
l1, l2, l3, l4 = sm.symbols('l1, l2, l3, l4')

# acceleration due to gravity
g = sm.symbols('g')

# mass for each rigid body: C, D, E, F
mc, md, me, mf = sm.symbols('mc, md, me, mf')

# inertia components for each rigid body: C, D, E, F
ic11, ic22, ic33, ic31 = sm.symbols('ic11, ic22, ic33, ic31')
id11, id22 = sm.symbols('id11, id22')
ie11, ie22, ie33, ie31 = sm.symbols('ie11, ie22, ie33, ie31')
if11, if22 = sm.symbols('if11, if22')

##################
# Position Vectors
##################

print('Defining position vectors.')

# point fixed on the ground
o = mec.Point('o')

# rear wheel contact point, y is the kickplate lateral location
dn = mec.Point('dn')
dn.set_pos(o, q1*N['1'] + (y + q2)*N['2'])

# newtonian origin to rear wheel center
do = mec.Point('do')
do.set_pos(dn, -rr*B['3'])

# rear wheel center to bicycle frame center
co = mec.Point('co')
co.set_pos(do, l1*C['1'] + l2*C['3'])

# rear wheel center to steer axis point
ce = mec.Point('ce')
ce.set_pos(do, d1*C['1'])

# steer axis point to the front wheel center
fo = mec.Point('fo')
fo.set_pos(ce, d2*E['3'] + d3*E['1'])

# front wheel center to front frame center
eo = mec.Point('eo')
eo.set_pos(fo, l3*E['1'] + l4*E['3'])

# front wheel contact point
fn = mec.Point('fn')
fn.set_pos(fo, rf*E['2'].cross(A['3']).cross(E['2']).normalize())

######################
# Holonomic Constraint
######################

print('Defining holonomic constraints.')

# this constraint is enforced so that the front wheel contacts the ground
holonomic = fn.pos_from(dn).dot(A['3'])

####################################
# Kinematical Differential Equations
####################################

print('Defining kinematical differential equations.')

kinematical = [
    q1.diff(t) - u1,  # rear x contact speed
    q2.diff(t) - u2,  # rear y contact speed
    q3.diff(t) - u3,  # yaw
    q4.diff(t) - u4,  # roll
    q5.diff(t) - u5,  # pitch
    q6.diff(t) - u6,  # rear wheel rotation
    q7.diff(t) - u7,  # steer
    q8.diff(t) - u8,  # front wheel rotation
]

####################
# Angular Velocities
####################

print('Defining angular velocities.')

# Note that the wheel angular velocities are defined relative to the frame
# they are attached to.

A.set_ang_vel(N, u3*N['3'])  # yaw rate
B.set_ang_vel(A, u4*A['1'])  # roll rate
C.set_ang_vel(B, u5*B['2'])  # pitch rate
D.set_ang_vel(C, u6*C['2'])  # rear wheel rate
E.set_ang_vel(C, u7*C['3'])  # steer rate
F.set_ang_vel(E, u8*E['2'])  # front wheel rate

###################
# Linear Velocities
###################

print('Defining linear velocities.')

# rear wheel contact stays in ground plane and does not slip but the auxiliary
# speed, u11, is included which corresponds to the vertical force
o.set_vel(N, 0)
dn.set_vel(N, u1*N['1'] + (y.diff(t) + u2)*N['2'])
dn_ = mec.Point('dn')
dn_.set_pos(dn, 0)
dn_.set_vel(N, dn.vel(N) + u11*A['3'])

# mass centers
do.v2pt_theory(dn_, N, D)  # ensures u11 is present in velocities
co.v2pt_theory(do, N, C)
ce.v2pt_theory(do, N, C)
fo.v2pt_theory(ce, N, E)
eo.v2pt_theory(fo, N, E)

# front wheel contact velocity
fn.v2pt_theory(fo, N, F)

fn_ = mec.Point('fn')
fn_.set_pos(fn, 0)
fn_.set_vel(N, fn.vel(N) + u12*A['3'])  # includes u11 and u12

# Slip angle components
# project the velocity vectors at the contact point onto each wheel's yaw
# direction

yd_repl = {y.diff(t, 2): ydd, y.diff(t): yd}

N_v_dn1 = dn.vel(N).dot(A['1']).xreplace(yd_repl)
N_v_dn2 = dn.vel(N).dot(A['2']).xreplace(yd_repl)
N_v_fn1 = dn.vel(N).dot(g1_hat).xreplace(yd_repl)
N_v_fn2 = dn.vel(N).dot(g2_hat).xreplace(yd_repl)

####################
# Motion Constraints
####################

# impose rolling without longitudinal slip on the front and rear wheel
# contacts, but allow lateral slip
# add a velocity contraint for the holonomic constraint (front wheel contacts
# the ground)

print('Defining nonholonomic constraints.')

nonholonomic = [
    dn_.vel(N).dot(A['1']),  # no rear longitudinal slip
    fn_.vel(N).dot(g1_hat),  # no front longitudinal slip
    fn_.vel(N).dot(A['3']),  # time derivative of the holonomic constraint
]

#########
# Inertia
#########

print('Defining inertia.')

Ic = mec.inertia(C, ic11, ic22, ic33, 0, 0, ic31)
Id = mec.inertia(C, id11, id22, id11, 0, 0, 0)
Ie = mec.inertia(E, ie11, ie22, ie33, 0, 0, ie31)
If = mec.inertia(E, if11, if22, if11, 0, 0, 0)

##############
# Rigid Bodies
##############

print('Defining the rigid bodies.')

rear_frame = mec.RigidBody('Rear Frame', co, C, mc, (Ic, co))
rear_wheel = mec.RigidBody('Rear Wheel', do, D, md, (Id, do))
front_frame = mec.RigidBody('Front Frame', eo, E, me, (Ie, eo))
front_wheel = mec.RigidBody('Front Wheel', fo, F, mf, (If, fo))

bodies = [rear_frame, rear_wheel, front_frame, front_wheel]

###########################
# Generalized Active Forces
###########################

print('Defining the forces and torques.')

# gravity
Fco = (co, mc*g*A['3'])
Fdo = (do, md*g*A['3'])
Feo = (eo, me*g*A['3'])
Ffo = (fo, mf*g*A['3'])

# tire-ground lateral forces
Fydn = (dn, Fry*A['2'])
Fyfn = (fn, Ffy*g2_hat)

# tire-ground normal forces (non-contributing), need equal and opposite forces
Fzdn = (dn, Frz*A['3'])
Fzdn_ = (dn_, -Frz*A['3'])
Fzfn = (fn, -Ffz*A['3'])
Fzfn_ = (fn_, Ffz*A['3'])

# input torques
Tc = (C, T4*A['1'] - T6*B['2'] - T7*C['3'])
Td = (D, T6*C['2'] + Mrz*A['3'])
Te = (E, T7*C['3'])
Tf = (F, Mfz*A['3'])

loads = [
    Fco, Fdo, Feo, Ffo,
    Fydn, Fyfn, Fzdn, Fzfn,
    Fzdn_, Fzfn_,
    Tc, Td, Te, Tf
]

####################
# Prep symbolic data
####################

newto = N
# rear contact x, rear contact y, yaw, roll, rear wheel angle, steer, front
# wheel angle
q_ind = (q1, q2, q3, q4, q6, q7, q8)
q_dep = (q5,)  # pitch
qs = tuple(sm.ordered(q_ind + q_dep))

# longitudinal rear speed, roll rate, yaw rate, rear wheel rate, steer rate
u_ind = (u1, u3, u4, u6, u7)
u_dep = (u2, u5, u8)  # lateral rear speed, pitch rate, front wheel rate
u_aux = (u11, u12)
us = tuple(sm.ordered(u_ind + u_dep))

ps = (d1, d2, d3, g, ic11, ic22, ic31, ic33, id11, id22, ie11, ie22, ie31,
      ie33, if11, if22, l1, l2, l3, l4, mc, md, me, mf, rf, rr)
rs = (T4, T6, T7, Fry, Frz, Mrz, Ffy, Ffz, Mfz, y, yd, ydd)
holon = [holonomic]
nonho = tuple(nonholonomic)

###############
# Kane's Method
###############

print("Generating Kane's equations.")

kane = mec.KanesMethod(
    newto,
    q_ind,
    u_ind,
    kd_eqs=kinematical,
    q_dependent=q_dep,
    configuration_constraints=holon,
    u_dependent=u_dep,
    velocity_constraints=nonho,
    u_auxiliary=u_aux,
    constraint_solver=cramer_solve,
)

Fr, Frstar = kane.kanes_equations(bodies, loads=loads)

###########################
# Generate Octave Functions
###########################

u1p, u3p, u4p, u6p, u7p = mec.dynamicsymbols('u1p, u3p, u4p, u6p, u7p')
u2p, u5p, u8p = mec.dynamicsymbols('u2p, u5p, u8p')
u_dots = [mec.dynamicsymbols(ui.name + 'p') for ui in us]
ups = tuple(sm.ordered(u_dots))
u_dot_subs = {ui.diff(): upi for ui, upi in zip(us, u_dots)}

gen = OctaveMatrixGenerator([[q4, q5, q7],
                             [d1, d2, d3, rf, rr]],
                            [sm.Matrix([holonomic])])
gen.write('eval_holonomic', path=os.path.dirname(__file__))

# Create matrices for solving for the dependent speeds.
nonholonomic = sm.Matrix(nonholonomic).xreplace({u11: 0, u12: 0, y.diff(t): yd})

print('The nonholonomic constraints a function of these dynamic variables:')
print(list(sm.ordered(mec.find_dynamicsymbols(nonholonomic))))

A_nh, B_nh = decompose_linear_parts(nonholonomic, u_dep)
gen = OctaveMatrixGenerator([[q3, q4, q5, q7],
                             u_ind,
                             [yd],
                             [d1, d2, d3, rf, rr]],
                            [A_nh, -B_nh])
gen.write('eval_dep_speeds', path=os.path.dirname(__file__))

# Create function for solving for the derivatives of the dependent speeds.
nonholonomic_dot = sm.Matrix(nonholonomic).diff(t).xreplace(kane.kindiffdict())

nonholonomic_dot = nonholonomic_dot.xreplace(u_dot_subs).xreplace({yd.diff(t): ydd})

print('The derivative of the nonholonomic constraints a function of these '
      'dynamic variables:')
print(list(sm.ordered(mec.find_dynamicsymbols(nonholonomic_dot))))

A_pnh, B_pnh = decompose_linear_parts(nonholonomic_dot, [u2p, u5p, u8p])
gen = OctaveMatrixGenerator([[q3, q4, q5, q7],
                             [u1, u2, u3, u4, u5, u6, u7, u8],
                             [yd, ydd],
                             [u1p, u3p, u4p, u6p, u7p],
                             [d1, d2, d3, rf, rr]],
                            [A_pnh, -B_pnh])
gen.write('eval_dep_speeds_derivs', path=os.path.dirname(__file__))

# Create function for solving for the derivatives of the dependent speeds.

print('The mass matrix is a function of these dynamic variables:')
print(list(sm.ordered(mec.find_dynamicsymbols(kane.mass_matrix))))

print('The forcing function is a function of these dynamic variables:')
print(list(sm.ordered(mec.find_dynamicsymbols(kane.forcing))))

A_dyn = kane.mass_matrix
B_dyn = kane.forcing.xreplace({
    u11.diff(t): 0,
    u12.diff(t): 0,
    u11: 0,
    u12: 0,
    y.diff(t, 2): ydd,
    y.diff(t): yd,
    Ffz: 0
})

gen = OctaveMatrixGenerator([[q3, q4, q5, q7],
                             [u1, u2, u3, u4, u5, u6, u7, u8],
                             [T4, T6, T7, yd, ydd, Fry, Mrz, Ffy, Mfz],
                             ps],
                            [A_dyn, B_dyn])
gen.write('eval_dynamic_eqs', path=os.path.dirname(__file__))

# Create function for solving for the lateral forces.
"""
Should be linear in the forces? Or even always F1 + F2 + ... = 0, i.e.
coefficient is 1?

A(q, t)*[Ffz] - b(u', u, q, t) = 0
        [Frz]

"""
aux_eqs = kane.auxiliary_eqs.xreplace(u_dot_subs).xreplace({y.diff(t, 2): ydd,
                                                            y.diff(t): yd})

print('The auxiliary equations are a function of these dynamic variables:')
print(list(sm.ordered(mec.find_dynamicsymbols(aux_eqs))))

A_normal, b_normal = decompose_linear_parts(aux_eqs, [Frz, Ffz])

print('A is a function of these dynamic variables:')
print(list(sm.ordered(mec.find_dynamicsymbols(A_normal))))
print('b is a function of these dynamic variables:')
print(list(sm.ordered(mec.find_dynamicsymbols(b_normal))))

slip_components = sm.Matrix([N_v_dn1, N_v_dn2, N_v_fn1, N_v_fn2])

gen = OctaveMatrixGenerator([[q3, q4, q5, q7],
                             [u1, u2, u3, u4, u5, u6, u7, u8],
                             [T4, T6, T7, yd, ydd, Fry, Mrz, Ffy, Mfz],
                             [u1p, u2p, u3p, u4p, u5p, u6p, u7p, u8p],
                             ps],
                            [A_normal, -b_normal, slip_components])
gen.write('eval_tire_force_inputs', path=os.path.dirname(__file__))

# Test simulation

p_vals = {
   d1: 0.9534570696121849,
   d2: 0.2676445084476887,
   d3: 0.03207142672761929,
   g: 9.81,
   ic11: 7.178169776497895,
   ic22: 11.0,
   ic31: 3.8225535938357873,
   ic33: 4.821830223502103,
   id11: 0.0603,
   id22: 0.12,
   ie11: 0.05841337700152972,
   ie22: 0.06,
   ie31: 0.009119225261946298,
   ie33: 0.007586622998470264,
   if11: 0.1405,
   if22: 0.28,
   l1: 0.4707271515135145,
   l2: -0.47792881146460797,
   l3: -0.00597083392418685,
   l4: -0.3699518200282974,
   mc: 85.0,
   md: 2.0,
   me: 4.0,
   mf: 3.0,
   rf: 0.35,
   rr: 0.3,
}

# We need to solve the dynamic equations and the auxiliary equations
# simultaneously to avoid having to solve the dynamic equations first and then
# substitute in the deritavies of the speeds. So reconstruct the equations of
# motion.
fr_plus_fr_star = kane.mass_matrix*kane.u.diff(t) - kane.forcing.xreplace({
    u11.diff(t): 0,
    u12.diff(t): 0,
    u11: 0,
    u12: 0,
    y.diff(t, 2): ydd,
    y.diff(t): yd,
    Ffz: 0
})
aux_eqs = kane.auxiliary_eqs.xreplace({y.diff(t, 2): ydd, y.diff(t): yd})
all_dyn_eqs = fr_plus_fr_star.col_join(aux_eqs)

x_all = tuple(ui.diff(t) for ui in us) + (Frz, Ffz)
x_all_zerod = {xi: 0 for xi in x_all}

A_all = all_dyn_eqs.jacobian(x_all)
b_all = -all_dyn_eqs.xreplace(x_all_zerod)

print('A_all is a function of these dynamic variables:')
print(list(sm.ordered(mec.find_dynamicsymbols(A_all))))
print('b_all is a function of these dynamic variables:')
print(list(sm.ordered(mec.find_dynamicsymbols(b_all))))

eval_dynamic = sm.lambdify([qs, us, rs, ps], [A_all, b_all, slip_components],
                           cse=True)
print(eval_dynamic(*[np.ones_like(a) for a in [qs, us, rs, ps]]))

last_vals = np.zeros(6)


def rhs(t, x, p):
    q = x[:8]
    u = x[8:16]

    # grab the values calculated from the last time step
    # TODO : need way to capture and use output quantities (things that aren't
    # states)
    Frz, Ffz, rear_lon, rear_lat, front_lon, front_lat = last_vals

    # steer, rear wheel, roll torques set to zero
    T4, T6, T7 = 0.0, 0.0, 0.0

    # kickplate motion set to zero
    y, yd, ydd = 0.0, 0.0, 0.0

    # set self-aligning moments to zero
    Mrz, Mfz = 0.0, 0.0

    # calculate later tire forces
    # coefficient estimating form Fig 11 in Dressel & Rahman 2012
    normalized_cornering_coeff = (0.55 - 0.1)/np.deg2rad(3.0 - 0.5)  # about 10
    normalized_cornering_coeff = 0.0001
    # use 1e-12 to avoid divide by zero
    alphar = -np.sign(rear_lat)*np.arctan(rear_lat/(rear_lon + 1e-12))
    alphaf = -np.sign(front_lat)*np.arctan(front_lat/(front_lon + 1e-12))
    Fry = Frz*normalized_cornering_coeff*alphar
    Ffy = Ffz*normalized_cornering_coeff*alphaf

    r = [T4, T6, T7, Fry, Frz, Mrz, Ffy, Ffz, Mfz, y, yd, ydd]

    A, b, slip = eval_dynamic(q, u, r, p)

    slip = slip.squeeze()

    xplus = np.linalg.solve(A, b).squeeze()

    last_vals[:] = [xplus[8], xplus[9], slip[0], slip[1], slip[2], slip[3]]

    return np.hstack((u, xplus[:8]))


initial_speed = 4.6  # m/s
initial_roll_rate = 0.5  # rad/s

eval_holonomic = sm.lambdify((q5, q4, q7, d1, d2, d3, rf, rr), holonomic)
initial_pitch_angle = float(fsolve(eval_holonomic, 0.0,
                                   args=(0.0,  # q4
                                         1e-8,  # q7
                                         p_vals[d1],
                                         p_vals[d2],
                                         p_vals[d3],
                                         p_vals[rf],
                                         p_vals[rr])))

print('Initial pitch angle:', np.rad2deg(initial_pitch_angle))

initial_conditions = [
    0.0,  # q1
    0.0,  # q2
    1e-10,  # q3  (divide by zero)
    0.0,  # q4
    initial_pitch_angle,  # q5
    0.0,  # q6
    0.0,  # q7
    0.0,  # q8
    initial_speed,  # u1
    0.0,  # u2
    0.0,  # u3
    initial_roll_rate,  # u4
    0.0,  # u5
    -initial_speed/p_vals[rr],  # u6
    0.0,  # u7
    -initial_speed/p_vals[rf],  # u8
]

print('here')
print(last_vals)
print(rhs(0.0, initial_conditions, list(p_vals.values())))
print(last_vals)

fps = 30  # frames per second
duration = 4.0  # seconds
t0 = 0.0
tf = t0 + duration
times = np.linspace(t0, tf, num=int(duration*fps))

res = solve_ivp(lambda t, x: rhs(t, x, list(p_vals.values())), (t0, tf),
            initial_conditions, t_eval=times) #, method='LSODA')
x_traj = res.y.T
times = res.t

#times, x_traj = euler_integrate(rhs, (t0, tf), initial_conditions,
                                #list(p_vals.values()), delt=0.001)

#holonomic_vs_time  = eval_holonomic(x_trajectory[:, 3],  # q5
                                    #x_trajectory[:, 1],  # q4
                                    #x_trajectory[:, 2],  # q7
                                    #sys.constants[d1],
                                    #sys.constants[d2],
                                    #sys.constants[d3],
                                    #sys.constants[rf],
                                    #sys.constants[rr])

import matplotlib.pyplot as plt
fig, axes = plt.subplots(x_traj.shape[1], 1, sharex=True)
fig.set_size_inches(8, 10)
for ax, traj, s in zip(axes, x_traj.T, qs + us):
    ax.plot(times, traj)
    ax.set_ylabel(s)
axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.show()
