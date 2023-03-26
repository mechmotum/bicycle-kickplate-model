Dynamic model of a bicycle that includes linear lateral tire forces, linear
tire self-aligning moments, relaxation lengths for both, and an input capable
of moving the contact point under the rear wheel laterally to simulate a
kick plate.

This script requires the development version of SymPy and PyDy depends on
SymPy, so install both from development versions.

::

   mamba env create -n env.yml
   mamba activate bicycle-kickplate-model
   git clone https://github.com/sympy/sympy.git
   cd sympy
   python setup.py develop
   git clone https://github.com/pydy/pydy.git
   cd pydy
   python setup.py develop
   cd ..
   python simulate.py
