import sympy as sm
import sympy.physics.mechanics as mec

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
# q11: rear tire vertical sinkage
# q12: front tire vertical sinkage
q1, q2, q3, q4 = mec.dynamicsymbols('q1, q2, q3, q4')
q5, q6, q7, q8 = mec.dynamicsymbols('q5, q6, q7, q8')
q11, q12 = mec.dynamicsymbols('q11, q12')

# q's that will have kinematical differential equations
qs = [  # index
    q1,  # 0
    q2,  # 1
    q3,  # 2
    q4,  # 3
    q5,  # 4
    q6,  # 5
    q7,  # 6
    q8,  # 7
    q11,  # 8
    q12,  # 9
]

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
us = [
    u1,  # 0, dep
    u2,  # 1, dep
    u3,  # 2, ind
    u4,  # 3, ind
    u5,  # 4, dep
    u6,  # 5, ind
    u7,  # 6, ind
    u8,  # 7, ind
    u11,  # 8, ind
    u12,  # 9, ind
]

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

fs = (Fry, Ffy, Mrz, Mfz)


# kickplate force
Fkp = mec.dynamicsymbols('Fkp')

# kickplate displacement, speed, and acceleration
y, yd, ydd = mec.dynamicsymbols('y, yd, ydd')

rs = (T4, T6, T7, Fkp, y, yd, ydd)

# states for relaxation length
Fry_, Ffy_, Mrz_, Mfz_ = mec.dynamicsymbols('Fry_, Ffy_, Mrz_, Mfz_')
rs = rs + (Fry_, Ffy_, Mrz_, Mfz_)

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

# the constants rely on being sorted
ps = (
    c_af,  # 0
    c_ar,  # 1
    c_f,  # 2
    c_maf,  # 3
    c_mar,  # 4
    c_mpf,  # 5
    c_mpr,  # 6
    c_pf,  # 7
    c_pr,  # 8
    c_r,  # 9
    d1,  # 10
    d2,  # 11
    d3,  # 12
    g,  # 13
    ic11,  # 14
    ic22,  # 15
    ic31,  # 16
    ic33,  # 17
    id11,  # 18
    id22,  # 19
    ie11,  # 20
    ie22,  # 21
    ie31,  # 22
    ie33,  # 23
    if11,  # 24
    if22,  # 25
    k_f,  # 26
    k_r,  # 27
    l1,  # 28
    l2,  # 29
    l3,  # 30
    l4,  # 31
    mc,  # 32
    md,  # 33
    me,  # 34
    mf,  # 35
    r_tf,  # 36
    r_tr,  # 37
    rf,  # 38
    rr,  # 39
    s_yf,  # 40
    s_yr,  # 41
    s_zf,  # 42
    s_zr,  # 43
)
