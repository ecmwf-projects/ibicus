.. _installing:

Installing
==========


Pip install
-----------

To install ibicus, just run the following command:

.. code-block:: bash

  pip install ibicus

The ibicus ``pip`` package has been tested successfully with the latest versions of
its dependencies (`setup.cfg <https://github.com/ecmwf-projects/ibicus/blob/main/setup.cfg>`_).

Conda install
-------------

No conda package has been created yet.
``pip install ibicus`` can be used in a conda environment.

.. note::

  Mixing ``pip`` and ``conda`` could create some dependencies issues,
  we recommend installing as many dependencies as possible with conda,
  then install ibicus with ``pip``, `as recommended by the anaconda team
  <https://www.anaconda.com/blog/using-pip-in-a-conda-environment>`_.


Troubleshooting
---------------

Python 3.8 or above is required
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ibicus requires Python 3.8 or above. Depending on your installation,
  you may need to substitute ``pip`` to ``pip3`` in the examples below.
