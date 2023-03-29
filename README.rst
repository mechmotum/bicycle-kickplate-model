Dynamic model of a bicycle that includes linear lateral tire forces, linear
tire self-aligning moments, relaxation lengths for both, and an input capable
of moving the contact point under the rear wheel laterally to simulate a
kick plate.

This script requires the development version of SymPy and PyDy depends on
SymPy, so install both from development versions.

::

   conda env create -f env.yml
   conda activate bicycle-kickplate-model
   python -m pip install git+https://github.com/sympy/sympy.git
   python -m pip install git+https://github.com/pydy/pydy.git

After that you can run the simulation, e.g.::

   python simulate.py
