.. ibicus documentation master file, created by
   sphinx-quickstart on Wed Mar 30 16:04:07 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ibicus's documentation!
========================================

**ibicus provides a flexible and user-friendly toolkit for the bias correction of climate models and associated evaluation.**

ibicus implements a variety of methods for bias correction (8 currently) published in peer-reviewed literature, including ISIMIP (Lange 2019) and CDFt (Michelangeli et al. 2009) and provides a unified interface for their usage. The package enables the user to modify and refine their behavior with settings and parameters, and provides an evaluation framework to assess marginal, temporal, spatial, and multivariate properties of the bias corrected climate model. 

On the following pages you can find information on the usage of ibicus. Have a look at the `Overview <getting_started/overview.html>`_ the `What is debiasing? <getting_started/whatisdebiasing.html>`_ and the `API reference <reference/api.html>`_ for both the `debias <reference/debias.html>`_ and `evaluate <reference/evaluate.html>`_ module.

Documentation
_____________

**Getting Started**

* :doc:`getting_started/overview`
* :doc:`getting_started/installing`
* :doc:`getting_started/whatisdebiasing`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started
   
   getting_started/overview
   getting_started/installing
   getting_started/whatisdebiasing

**Tutorials**

- `00 Download and Preprocess Data <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/00%20Download%20and%20Preprocess.ipynb>`_
- `01 Getting Started <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/01%20Getting%20Started.ipynb>`_
- `02 Adjusting Debiasers <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/02%20Adjusting%20Debiasers.ipynb>`_
- `03 Evaluation <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/03%20Evaluation.ipynb>`_

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tutorials
   
   00 Download and Preprocess Data <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/00%20Download%20and%20Preprocess.ipynb>
   01 Getting Started <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/01%20Getting%20Started.ipynb>
   02 Adjusting Debiasers <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/02%20Adjusting%20Debiasers.ipynb>
   03 Evaluation <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/03%20Evaluation.ipynb>

**Documentation / API reference**

* :doc:`reference/api`
   * :doc:`reference/debias`
   * :doc:`reference/evaluate`
   * :doc:`reference/utils`
   * :doc:`reference/variables`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Documentation

   ibicus.debias module <reference/debias>
   ibicus.evaluate module <reference/evaluate>
   ibicus.utils module <reference/utils>
   ibicus.variables module <reference/variables>

License
-------

ibicus is available under the open source `Apache-2.0 License`__.

__ https://github.com/ecmwf-projects/ibicus/blob/main/LICENSE


Acknowledgements
----------------

The development of this package was supported by the European Centre for Mid-term Weather Forecasts (ECMWF) as part of the `ECMWF Summer of Weather Code <https://esowc.ecmwf.int/>`_ 

.. image:: images/logos.png
   :width: 800
   :alt: ECMWF logos
   