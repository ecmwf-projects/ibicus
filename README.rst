
******
ibicus
******

|pypi_release| |pypi_status| |PyPI license| |pypi_downloads| |docs| |PyPI pyversions| |made-with-python| |made-with-sphinx-doc| |Maintenance yes| |Ask Me Anything !| |GitHub contributors|


**Ibicus provides a flexible and user-friendly toolkit for the bias correction of climate models and associated evaluation.**

Ibicus implements a variety of methods for bias correction (8 currently) published in peer-reviewed literature, including ISIMIP (Lange 2019) and CDFt (Michelangeli et al. 2009) and provides a unified interface for their usage.
The package enables the user to modify and refine their behavior with settings and parameters, and provides an evaluation framework to assess marginal, temporal, spatial, and multivariate properties of the bias corrected climate model.

Given future climate model data to debias (``cm_future``), climate model data during a reference period (``cm_hist``) and observational or reanalysis data during the same reference period (``obs``) running a debiaser is as easy as:

>>> from ibicus import CDFt
>>> debiaser = CDFt.from_variable("pr")
>>> debiased_cm_future = debiaser.apply(obs, cm_hist, cm_future)

Evaluating dry spell length can be as easy as:

>>> from ibicus.evaluate.metrics import dry_days
>>> spell_length = dry_days.calculate_spell_length(minimum_length: 4, obs = obs, raw = cm_future, ISIMIP = debiased_cm_future)


For more information on the usage have a look at `our docs <https://ibicus.readthedocs.io/en/latest/>`_.



Install
-------

Ibicus releases are available via PyPI. Just write::

   pip install ibicus

For more information about installation and requirements see the `install documentation <https://ibicus.readthedocs.io/en/latest/getting_started/installing.html>`_ in the docs.


Contact
-------

If you have feedback on the package, suggestions for additions, questions you'd like to ask or would like to contribute, please contact us under `ibicus.py@gmail.com <mailto:ibicus.py@gmail.com>`_.
Similarly should you encounter bugs or issues using the package please `open an issue <https://github.com/ecmwf-projects/ibicus/issues>`_. or write to us using the email adress above.


.. |pypi_release| image:: https://img.shields.io/pypi/v/ibicus?color=green
    :target: https://pypi.org/project/ibicus

.. |pypi_status| image:: https://img.shields.io/pypi/status/ibicus
    :target: https://pypi.org/project/ibicus

.. |pypi_downloads| image:: https://img.shields.io/pypi/dm/ibicus
  :target: https://pypi.org/project/ibicus

.. |docs| image:: https://readthedocs.org/projects/ibicus/badge/?version=latest
  :target: https://ibicus.readthedocs.io/en/latest/?badge=latest

.. |Maintenance yes| image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://github.com/ecmwf-projects/ibicus/graphs/commit-activity

.. |Website ibicus| image:: https://img.shields.io/website-up-down-green-red/http/monip.org.svg
   :target: https://readthedocs.org/

.. |Ask Me Anything !| image:: https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg
   :target: mailto:ibicus.py@gmail.com

.. |made-with-python| image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/

.. |made-with-sphinx-doc| image:: https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg
   :target: https://www.sphinx-doc.org/

.. |PyPI download month| image:: https://img.shields.io/pypi/dm/ibicus
   :target: https://pypi.org/project/ibicus/

.. |PyPI version shields.io| image:: https://img.shields.io/pypi/v/ibicus
   :target: https://pypi.org/project/ibicus/

.. |PyPI license| image:: https://img.shields.io/pypi/l/ibicus
   :target: https://pypi.org/project/ibicus/

.. |PyPI pyversions| image:: https://img.shields.io/pypi/pyversions/ibicus
   :target: https://pypi.org/project/ibicus/

.. |PyPI status| image:: https://img.shields.io/pypi/status/ibicus
   :target: https://pypi.org/project/ibicus/

.. |GitHub contributors| image:: https://img.shields.io/github/contributors/ecmwf-projects/ibicus
   :target: https://github.com/ecmwf-projects/ibicus
