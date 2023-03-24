import os

from pydy.codegen.octave_code import OctaveMatrixGenerator
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

print_syms(nonholonomic,
           'The nonholonomic constraints a function of these dynamic variables:')

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

print_syms(nonholonomic_dot, 'The derivative of the nonholonomic constraints'
           'a function of these dynamic variables: ')

A_pnh, B_pnh = decompose_linear_parts(nonholonomic_dot, [u2p, u5p, u8p])
gen = OctaveMatrixGenerator([[q3, q4, q5, q7],
                             [u1, u2, u3, u4, u5, u6, u7, u8],
                             [yd, ydd],
                             [u1p, u3p, u4p, u6p, u7p],
                             [d1, d2, d3, rf, rr]],
                            [A_pnh, -B_pnh])
gen.write('eval_dep_speeds_derivs', path=os.path.dirname(__file__))


# Create function for solving for the derivatives of the dependent speeds.

print_syms(kane.mass_matrix,
           'The mass matrix is a function of these dynamic variables:')
print_syms(kane.forcing,
           'The forcing function is a function of these dynamic variables: ')

A_dyn = kane.mass_matrix
B_dyn = kane.forcing.xreplace({
    u11.diff(t): 0,
    u12.diff(t): 0,
    u11: 0,
    u12: 0,
    y.diff(t, 2): ydd,
    y.diff(t): yd,
    # TODO : Should I be setting this to zero here?
    Ffz: 0
})

gen = OctaveMatrixGenerator([qs, us, rs, ps],
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

print_syms(aux_eqs,
           'The auxiliary equations are a function of these dynamic variables: ')

A_normal, b_normal = decompose_linear_parts(aux_eqs, [Frz, Ffz])

print_syms(A_normal, 'A_aux is a function of these dynamic variables: ')
print_syms(b_normal, 'b_aux is a function of these dynamic variables: ')

slip_components = sm.Matrix([N_v_nd1, N_v_nd2, N_v_fn1, N_v_fn2])

gen = OctaveMatrixGenerator([qs, us, rs, ups, ps],
                            [A_normal, -b_normal, slip_components])
gen.write('eval_tire_force_inputs', path=os.path.dirname(__file__))

