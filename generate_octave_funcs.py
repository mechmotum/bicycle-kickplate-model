import os

from pydy.codegen.octave_code import OctaveMatrixGenerator

from nonlin_sym import *

# holonomic constraint
gen = OctaveMatrixGenerator([[q4, q5, q7],
                             [d1, d2, d3, rf, rr]],
                            [sm.Matrix([holonomic])])
gen.write('eval_holonomic', path=os.path.dirname(__file__))

# nonholonomic constraints: calculate dependent speeds
gen = OctaveMatrixGenerator([[q3, q4, q5, q7],
                             u_ind,
                             [d1, d2, d3, rf, rr]],
                            [A_nh, -B_nh])
gen.write('eval_dep_speeds', path=os.path.dirname(__file__))

# dynamic equations and normal forces
gen = OctaveMatrixGenerator([qs, us, fs, rs, ps],
                            [A_all, b_all])
gen.write('eval_dynamic_eqs', path=os.path.dirname(__file__))
