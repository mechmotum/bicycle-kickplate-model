Description
===========

https://github.com/user-attachments/assets/71bbadf5-80d3-4f66-bf41-2daf0c47d846

Dynamic model of a bicycle that includes both a nonlinear and linear tire
model. The model includes lateral forces, self-aligning moments, relaxation
length, and an input capable of applying either a force or displacement to the
contact point under the rear wheel to simulate a lateral kick plate.

The model is based on the nonlinear Carvallo-Whipple model presented in
[Moore2012]_. The lateral nonholonomic constraints at the wheel-ground contact
locations are removed and replaced with a lateral force and self-aligning
moment at the contact patch. A tire carcass radius creates a toroidal tire and
vertical tire compression is added.

We first presented preliminary results of this model at the ICSC 2023
conference [DellOrto2023]_.

.. [Moore2012] Moore, J. K. (2012). Human Control of a Bicycle [Doctor of
   Philosophy, University of California].
   http://moorepants.github.io/dissertation
.. [DellOrto2023] Dell’Orto, G., Alizadehsaravi, L., Happee, R., & Moore, J. K.
   (2023, November 16). Kick-plate test for assessing bicycle dynamics and tyre
   effect [Poster]. International Cycling Safety Conference, The Hague, The
   Netherlands.

Usage
=====

Create the Conda environment and activate it::

   conda env create -f bicycle-kickplate-model-env.yml
   conda activate bicycle-kickplate-model

After that you can run the sample simulation, e.g.::

   python base_simulation.py

or recreate the plot from our ICSC 2023 poster presentation::

   python icsc2023_abstract_figure.py

After ``model.py`` is run once, it generates a Python module
``generated_functions.py`` which caches the equations of motion. If you edit
``symbols.py`` or ``model.py`` you'll need to delete the
``generated_functions.py`` and rerun the model construction. Delete the
``generated_functions.py`` anytime you want to rebuild the model.

The various files in the repository are described below:

- ``symbols.py``: contains all symbol symbol definitions
- ``model.py``: formulates the symbolic equations of motion and generates
  numeric functions to evaluate them
- ``inputs.py``: contains different function for specified inputs (kick plate
  displacement, steer torque, etc.)
- ``parameters.py``: holds dictionaries mapping numerical values to all model
  constants
- ``simulate.py``: simulation functions and plotting functions
- ``viz.py``: sets up a pythreejs animation using PyDy
- ``visualize.ipynb``: displays a 3D animation of the ``base_simualtion.py``
- ``simple_kick_plate_simulation.py``: 1D simulation of a kick plate mass
  hitting the stoppers used to understand the possible acceleration profiles
- ``tire_data.py``: Pajecka Magic formula constants for the nonlinear tire
  model measured from Gabriele Dell'Orto's work
- ``utils.py``: helper functions

License
=======

The source code and documentation in this repository are licensed under the
3-clause BSD open source software license.
