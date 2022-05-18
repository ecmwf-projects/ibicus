.. _installing:

Installing
==========


Pip install
-----------

To install PACKAGE_NAME, just run the following command:

.. code-block:: bash

  pip install PACKAGE_NAME

The PACKAGE_NAME ``pip`` package has been tested successfully with the latest versions of
its dependencies (`build logs <https://github.com/PROJECT/PACKAGE_NAME/PATH/TO/test-and-release.yml>`_).

Conda install
-------------

No conda package has been created yet.
``pip install PACKAGE_NAME`` can be used in a conda environment.

.. note::

  Mixing ``pip`` and ``conda`` could create some dependencies issues,
  we recommend installing as many dependencies as possible with conda,
  then install PACKAGE_NAME with ``pip``, `as recommended by the anaconda team
  <https://www.anaconda.com/blog/using-pip-in-a-conda-environment>`_.


Troubleshooting
---------------

Python 3.7 or above is required
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PACKAGE_NAME requires Python 3.7 or above. Depending on your installation,
  you may need to substitute ``pip`` to ``pip3`` in the examples below.


