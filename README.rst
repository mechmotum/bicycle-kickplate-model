Description
===========

Dynamic model of a bicycle that includes both a nonlinear and linear tire
model. The model includes lateral forces, self-aligning moments, relaxation
length, and an input capable of applying either a force or displacement to the
contact point under the rear wheel to simulate a lateral kick plate.

The model is based on the nonlinear Carvallo-Whipple model presented in Moore
2012. The lateral nonholonomic constraints at the wheel-ground contact
locations are removed and replaced with a lateral force and yaw moment at the
contact patch. A tire carcass radius creates a torodial tire and vertical tire
compression is added.

Usage
=====

Create the Conda environment and activate it::

   conda env create -f bicycle-kickplate-model-env.yml
   conda activate bicycle-kickplate-model

After that you can run the simulation, e.g.::

   python base_simulation.py

The symbolic nonlinear equations of motion and functions that evaluate the
equations are generated with::

   python nonlin_sym.py

``simulate.py`` houses generic functions.

License
=======

The source code and documentation in this repository are licensed under the
3-clause BSD open source software license.
