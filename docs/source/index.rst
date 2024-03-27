.. ibicus documentation master file, created by
   sphinx-quickstart on Wed Mar 30 16:04:07 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ibicus's documentation!
========================================

**ibicus provides a flexible and user-friendly toolkit bias adjustment of climate models and associated evaluation.**

ibicus implements eight peer-reviewed bias adjustment methods in a common framework, including ISIMIP (Lange 2019) and CDFt (Michelangeli et al. 2009) and provides a unified interface for their application. The package enables the user to modify and refine the bias adjustment methods through settings and parameters. The evaluation framework introduced in ibicus allows the user to analyse changes to the marginal, spatiotemporal and inter-variable structure of user-defined climate indices and distributional properties, as well as any alteration of the climate change trend simulated in the model. ibicus operates on a numerical level and can therefore be integrated with any existing pre-processing pipeline and easily parallelized and integrated into high performance computing environments.

The ibicus documentation presented here provides a detailed overview of the different methods implemented, their default settings and possible modifications in parameters under `Documentation - ibicus.debias <reference/debias.html>`_, as well as a detailed description of the evaluation framework under `Documentation - ibicus.evaluate <reference/debias.html>`_ For a hands-on introduction to the package see our tutorial notebooks.

The documentation also provides a brief introduction to bias adjustment and possible issues with the approach under `Getting started <getting_started>`_. For a more detailed introduction to bias adjustment, as well as an overview of relevant literature on existing methods and issues, we refer to our paper published in Geoscientific Model Development:

How to cite: Spuler, F. R., Wessel, J. B., Comyn-Platt, E., Varndell, J., and Cagnazzo, C.: ibicus: a new open-source Python package and comprehensive interface for statistical bias adjustment and evaluation in climate modelling (v1.0.1), Geosci. Model Dev., 17, 1249–1269, https://doi.org/10.5194/gmd-17-1249-2024, 2024.

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
- `04 Parallelization and Advanced Topics <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/04%20Parallelization%20and%20Advanced%20Topics.ipynb>`_


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tutorials

   00 Download and Preprocess Data <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/00%20Download%20and%20Preprocess.ipynb>
   01 Getting Started <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/01%20Getting%20Started.ipynb>
   02 Adjusting Debiasers <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/02%20Adjusting%20Debiasers.ipynb>
   03 Evaluation <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/03%20Evaluation.ipynb>
   04 Parallelization and Advanced Topics <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/04%20Parallelization%20and%20Advanced%20Topics.ipynb>

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
