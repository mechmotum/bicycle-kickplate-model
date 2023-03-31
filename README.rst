Description
===========

Dynamic model of a bicycle that includes linear lateral tire forces, linear
tire self-aligning moments, relaxation length, and an input capable of applying
a force to the contact point under the rear wheel to simulate a lateral kick
plate.

Usage
=====

Install the software into a conda environment and activate::

   conda env create -f env.yml
   conda activate bicycle-kickplate-model

After that you can run the simulation, e.g.::

   python simulate.py

License
=======

The source code and documentation in this repository are licensed under the
3-clause BSD open source software license.
