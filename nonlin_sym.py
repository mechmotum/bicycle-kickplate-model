#!/usr/bin/env python

"""This file derives the non-linear equations of motion of the Carvallo-Whipple
bicycle model ([Carvallo1899]_, [Whippl1899]_) following the description and
nomenclature in [Moore2012]_ but with the tire-ground lateral slip nonholonomic
constraint removed and replaced with a lateral tire force and self-aligning
moment that are functions of the slip and camber angle of the wheels as well as
a specified lateral displacement of the rear wheel contact to simulate
perturbations applied by a kick plate.

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
import inspect

import sympy as sm
import sympy.physics.mechanics as mec

from utils import (ReferenceFrame, decompose_linear_parts, print_syms)

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
# q11: rear tire vertical extension
# q12: front tire vertical extension
q1, q2, q3, q4 = mec.dynamicsymbols('q1, q2, q3, q4')
q5, q6, q7, q8 = mec.dynamicsymbols('q5, q6, q7, q8')
q11, q12 = mec.dynamicsymbols('q11, q12')

# q's that will have kinematical differential equations
qs = [q1, q2, q3, q4, q5, q6, q7, q8, q11, q12]

# u1: speed of the rear wheel contact point in the n1> direction
# u2: speed of the rear wheel contact point in the n2> direction
# u3: frame yaw angular rate
# u4: frame roll angular rate
# u5: frame pitch angular rate
# u6: rear wheel rotation angular rate
# u7: steering rotation angular rate
# u8: front wheel rotation angular rate
# u9: speed of the front wheel contact point in the n1> direction
# u10: speed of the front wheel contact point in the n2> direction
# u11: rear tire vertical extension speed
# u12: front tire vertical extension speed
u1, u2, u3, u4 = mec.dynamicsymbols('u1, u2, u3, u4')
u5, u6, u7, u8 = mec.dynamicsymbols('u5, u6, u7, u8')
u9, u10, u11, u12 = mec.dynamicsymbols('u9, u10, u11, u12')

# u's that will have dynamical differential equations
us = [u1, u2, u3, u4, u5, u6, u7, u8, u11, u12]

# variables for the derivatives of the u's
ups = tuple([mec.dynamicsymbols(ui.name + 'p') for ui in us])

###########
# Specified
###########

# control torques
# T4 : roll torque
# T6 : rear wheel torque
# T7 : steer torque
T4, T6, T7 = mec.dynamicsymbols('T4, T6, T7')

# Fry : rear wheel-ground contact lateral force
# Ffy : front wheel-ground contact lateral force
Fry, Ffy = mec.dynamicsymbols('Fry, Ffy')

# Mrz : rear wheel-ground contact self-aligning moment
# Mfz : front rear wheel-ground contact self-aligning moment
Mrz, Mfz = mec.dynamicsymbols('Mrz, Mfz')

# kickplate force
Fkp = mec.dynamicsymbols('Fkp')

##############
# Replacements
##############

qdot_repl = {qi.diff(t): ui for qi, ui in zip(qs, us)}
udot_repl = {ui.diff(t): upi for ui, upi in zip(us, ups)}

#################################
# Orientation of Reference Frames
#################################

print('Orienting frames.')

# The following defines a 3-1-2 Tait-Bryan rotation with yaw (q3), roll (q4),
# pitch (q5) angles to orient the rear frame relative to the ground (Newtonian
# frame). The front frame is then rotated through the steer angle (q7) about
# the rear frame's 3 axis. The wheels are not oriented, as q6 and q8 end up
# being ignorable coordinates.

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
# contact path (E['2'] X A['3'])/|E['2'] X A['3']| gives this unit vector.
# G['2'] lies in the ground plane and points perpendicular to the wheel
# contact path. A['3'] X G['1'] gives this unit vector.
g1_hat = E['2'].cross(A['3']).normalize()
g2_hat = A['3'].cross(g1_hat)
g3_hat = A['3']

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

# vertical tire stiffness and damping, tire cross section rolling radius
k_f, k_r, c_r, c_f, r_tf, r_tr = sm.symbols('k_f, k_r, c_r, c_f, r_tf, r_tr')
# vertical load normalized cornering coefficients for lateral force
c_ar, c_af, c_pr, c_pf = sm.symbols('c_ar, c_af, c_pr, c_pf')
# vertical load normalized coefficients for self aligning moment
c_mar, c_maf, c_mpr, c_mpf = sm.symbols('c_mar, c_maf, c_mpr, c_mpf')
# relaxation lengths for the lateral force and self-aligning moments
s_yr, s_yf, s_zr, s_zf = sm.symbols('s_yr, s_yf, s_zr, s_zf')

##################
# Position Vectors
##################

print('Defining position vectors.')

# point fixed on the ground
o = mec.Point('o')

# rear wheel contact point, moves in ground plane, y is the kickplate lateral
# location
nd = mec.Point('nd')
nd.set_pos(o, q1*N['1'] + q2*N['2'])

# rear rim point
dt = mec.Point('dt')
dt.set_pos(nd, -(2*r_tr + q11)*A['3'])

# rim point to rear wheel center
do = mec.Point('do')
do.set_pos(dt, -rr*B['3'])

# point fixed on wheel at contact point (used for longitudinal holonomic
# constraint), could also make nd=dn
dn = mec.Point('nd')
dn.set_pos(do, rr*B['3'])

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

# front rim point
ft = mec.Point('ft')
ft.set_pos(fo, rf*E['2'].cross(A['3']).cross(E['2']).normalize())

# front wheel contact point
fn = mec.Point('fn')
# TODO : I think the sign of q12 here should be negative.
fn.set_pos(ft, (2*r_tf + q12)*A['3'])

######################
# Holonomic Constraint
######################

print('Defining holonomic constraints.')

# this constraint is enforced so that the front wheel contacts the ground
holonomic = fn.pos_from(nd).dot(A['3'])

print_syms(holonomic, "Holonomic constraint is a function of: ")

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
    q11.diff(t) - u11,  # front tire squish
    q12.diff(t) - u12,  # rear tire squish
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

# rear wheel contact stays in ground plane
o.set_vel(N, 0)
nd.set_vel(N, u1*N['1'] + u2*N['2'])

dt.set_vel(N, nd.vel(N) - u11*A['3'])

# mass centers
do.v2pt_theory(dt, N, B)
co.v2pt_theory(do, N, C)
ce.v2pt_theory(do, N, C)
fo.v2pt_theory(ce, N, E)
eo.v2pt_theory(fo, N, E)

# rear and front wheel points fixed on wheels
dn.v2pt_theory(do, N, D)  # TODO : is dt, not dn
ft.v2pt_theory(fo, N, F)

fn.set_vel(F, u12*A['3'])
fn.set_vel(N, fn.v1pt_theory(fo, N, F).xreplace(qdot_repl))

# Slip angle components
# project the velocity vectors at the contact point onto each wheel's yaw
# direction
N_v_nd1 = nd.vel(N).dot(A['1'])
N_v_nd2 = nd.vel(N).dot(A['2'])
# NOTE : Be careful on calculating these velocities for the slip angles. ft is
# both the point fixed in the wheel (as shown in ft.v2pt_theory(fo, N, F) and
# the point moving in the ground plane (as below). Probably should distinguish
# them better.
N_v_ft1 = ft.pos_from(o).dt(N).dot(g1_hat).xreplace(qdot_repl)
N_v_ft2 = ft.pos_from(o).dt(N).dot(g2_hat).xreplace(qdot_repl)

print_syms(N_v_nd1, "N_v_nd1 is a function of: ")
print_syms(N_v_nd2, "N_v_nd2 is a function of: ")
print_syms(N_v_ft1, "N_v_ft1 is a function of: ")
print_syms(N_v_ft2, "N_v_ft2 is a function of: ")

####################
# Motion Constraints
####################

# impose rolling without longitudinal slip on the front and rear wheel
# contacts, but allow lateral slip
# add a velocity contraint for the holonomic constraint (front wheel contacts
# the ground)

print('Defining nonholonomic constraints.')

nonholonomic = [
    # no rear longitudinal slip
    sm.trigsimp(dn.vel(N).dot(A['1'])),
    # no front longitudinal slip
    ft.vel(N).dot(g1_hat),
    # front contact cannot move vertically wrt ground
    holonomic.diff(t).xreplace(qdot_repl),
]

print_syms(nonholonomic[0], "rear slip constraint is a function of: ")
print_syms(nonholonomic[1], "front slip constraint is a function of: ")
print_syms(nonholonomic[2],
           "wheel vertical contact vel constraint is a function of: ")

common = mec.find_dynamicsymbols(nonholonomic[0]).intersection(
    mec.find_dynamicsymbols(nonholonomic[0])).intersection(
        mec.find_dynamicsymbols(nonholonomic[0]))
print('Nonholonomic constraints share these time varying parameters: ', common)

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

# The forces acting on the tire from the ground are defined in this convention:
# TODO : Change sign of Fz, so it is consistent with the other measure numbers.
# TODO : Not sure I'm following this with the new tire compliance model.
# Fr = Fx*A['1'] + Fy*A['2'] - Fz*A['3']
# Mr = Mrz*A['3']
# Ff = Fx*G['1'] + Fy*G['2'] - Fz*G['3']
# Mf = Mfz*G['3']

# lateral forces acting on the tire at ground contact
Fydn = (nd, Fry*A['2'])
Fyfn = (fn, Ffy*g2_hat)

# kickplate force (acts on tire at ground contact)
Fykp = (nd, Fkp*N['2'])

# Frz : rear wheel-ground contact normal force
# Ffz : front wheel-ground contact normal force
# tire-ground normal forces, need equal and opposite forces, compression is
# positive
Frz = -k_r*q11-c_r*u11  # positive when in compression
Ffz = -k_f*q12-c_f*u12  # positive when in compression
Fzdn = (nd, Frz*A['3'])
Fzdt = (dt, -Frz*A['3'])
Ffzn = (fn, Ffz*A['3'])
Ffzt = (ft, -Ffz*A['3'])

# input torques
Tc = (C, T4*A['1'] - T6*B['2'] - T7*C['3'])
Td = (D, T6*C['2'] + Mrz*A['3'])
Te = (E, T7*C['3'])
Tf = (F, Mfz*A['3'])

loads = [
    Fco, Fdo, Feo, Ffo,
    Fydn, Fyfn, Fzdn, Ffzn,
    Fzdt, Ffzt,
    Fykp,
    Tc, Td, Te, Tf
]

####################
# Prep symbolic data
####################

newto = N
# rear contact x, rear contact y, yaw, roll, rear wheel angle, steer, front
# wheel angle
q_ind = (q1, q2, q3, q4, q6, q7, q8, q11, q12)
q_dep = (q5,)  # pitch

# yaw rate, roll rate, rear wheel rate, steer rate, front wheel rate
u_ind = (u3, u4, u6, u7, u8, u11, u12)
# longitudinal rear speed, lateral rear speed, pitch rate
u_dep = (u1, u2, u5)
fs = (Fry, Ffy, Mrz, Mfz)
# the constants rely on being sorted
ps = (
    c_af,
    c_ar,
    c_f,
    c_maf,
    c_mar,
    c_mpf,
    c_mpr,
    c_pf,
    c_pr,
    c_r,
    d1,
    d2,
    d3,
    g,
    ic11,
    ic22,
    ic31,
    ic33,
    id11,
    id22,
    ie11,
    ie22,
    ie31,
    ie33,
    if11,
    if22,
    k_f,
    k_r,
    l1,
    l2,
    l3,
    l4,
    mc,
    md,
    me,
    mf,
    r_tf,
    r_tr,
    rf,
    rr,
    s_yf,
    s_yr,
    s_zf,
    s_zr,
)
rs = (T4, T6, T7, Fkp)
holon = (holonomic,)
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
    constraint_solver='CRAMER',
)

kane.kanes_equations(bodies, loads=loads)

###################
# Assemble Full EoM
###################

# Tire forces
# Relaxation length differential equation looks like so:
# (s_yr/N_v_nd1)*Fyr' + Fyr = (-c_ar*alphar + c_pr*phir)*Frz
# (s_yf/N_v_fn1)*Fyf' + Fyf = (-c_af*alphaf + c_pf*phif)*Ffz
# slip angle
alphar = sm.atan(N_v_nd2/N_v_nd1)
print_syms(alphar, 'Rear slip angle is a function of: ')
alphaf = sm.atan(N_v_ft2/N_v_ft1)
print_syms(alphaf, 'Front slip angle is a function of: ')
# camber angle
phir = q4
phif = -sm.atan((mec.dot(fo.pos_from(ft), g2_hat) /
                 mec.dot(fo.pos_from(ft), A['3'])))
Cf = sm.Matrix([
    [(s_yr/sm.Abs(N_v_nd1)), 0],
    [0, (s_yf/sm.Abs(N_v_ft1))],
])
Df = sm.Matrix([
    [(s_zr/sm.Abs(N_v_nd1)), 0],
    [0, (s_zf/sm.Abs(N_v_ft1))],
])
Fry_, Ffy_, Mrz_, Mfz_ = mec.dynamicsymbols('Fry_, Ffy_, Mrz_, Mfz_')
rs = rs + (Fry_, Ffy_, Mrz_, Mfz_)
#Fry_ = (-c_ar*alphar + c_pr*phir)*Frz
#Ffy_ = (-c_af*alphaf + c_pf*phif)*Ffz
#Mrz_ = (-c_mar*alphar + c_mpr*phir)*Frz
#Mfz_ = (-c_maf*alphaf + c_mpf*phif)*Ffz
nFy = sm.Matrix([
    [-Fry + Fry_],
    [-Ffy + Ffy_],
])
# TODO : Make the sign of the camber effect on self-aligning moment is correct.
nMz = sm.Matrix([
    [-Mrz + Mrz_],
    [-Mfz + Mfz_],
])

# With the additional differential equations for relaxation length, the full
# dynamical differential equations take this form:
# [M,  0,  0]*[up ] = [F(Fy, Mz)     ]
# [0, Cf,  0] [fyp]   [-Fy + f(Fz(q))]
# [0,  0, Df] [mzp]   [-Mz + f(Fz(q))]
print('Assembling full equations of motion.')
# KanesMethod stores the qs and us unordered, so fix.
# kane.u[:] = [u3, u4, u6, u7, u8, u11, u12, u1, u2, u5]
new_order = [2, 3, 5, 6, 7, 8, 9, 0, 1, 4]
mass_matrix = sm.zeros(*kane.mass_matrix.shape)
forcing = sm.zeros(*kane.forcing.shape)
forcing_orig = kane.forcing
for i in range(mass_matrix.shape[0]):
    forcing[new_order[i], 0] = forcing_orig[i, 0]
    for j in range(mass_matrix.shape[1]):
        mass_matrix[new_order[i], new_order[j]] = kane.mass_matrix[i, j]

num_udots = mass_matrix.shape[0]
row1 = mass_matrix.row_join(sm.zeros(num_udots, 4))
row2 = sm.zeros(2, num_udots).row_join(Cf).row_join(sm.zeros(2, 2))
row3 = sm.zeros(2, num_udots).row_join(sm.zeros(2, 2)).row_join(Df)

A_all = row1.col_join(row2).col_join(row3)
b_all = forcing.col_join(nFy).col_join(nMz)

print_syms(A_all, 'A_all is a function of these dynamic variables: ')
print_syms(b_all, 'b_all is a function of these dynamic variables: ')

# Create matrices for solving for the dependent speeds.
nonholonomic = sm.Matrix(nonholonomic)
print_syms(nonholonomic,
           'The nonholonomic constraints are a function of these variables:')
A_nh, B_nh = decompose_linear_parts(nonholonomic, u_dep)

# Front wheel contact point position.
# q9: perpendicular distance from the n2> axis to the front contact
#     point in the ground plane
# q10: perpendicular distance from the n1> axis to the front contact
#     point in the ground plane
q9 = fn.pos_from(o).dot(N['1'])
q10 = fn.pos_from(o).dot(N['2'])

print('Lambdifying equations of motion.')
eval_holonomic = sm.lambdify((q5, q4, q7, q11, q12, d1, d2, d3, r_tf, r_tr, rf,
                              rr), holonomic, cse=True)
eval_dep_speeds = sm.lambdify([qs, u_ind, ps], [A_nh, -B_nh], cse=True)
eval_dynamic = sm.lambdify([qs, us, fs, rs, ps], [A_all, b_all], cse=True)
eval_angles = sm.lambdify((qs, us, ps), [alphar, alphaf, phir, phif], cse=True)
eval_front_contact = sm.lambdify((qs, ps), [q9, q10], cse=True)

with open('eval_dynamic.py', 'w') as file:
    file.write(inspect.getsource(eval_dynamic))
