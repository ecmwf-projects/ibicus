
******
ibicus
******

**Ibicus provides a flexible and user-friendly toolkit for the bias correction of climate models and associated evaluation.**

Ibicus currently implements eight methods for bias correction published in peer-reviewed literature, including ISIMIP (Lange 2019) and CDFt (Michelangeli et al. 2009). 
The package enables the user to modify and refine their behavior with settings and parameters, and provides an evaluation framework to assess marginal, temporal, spatial, and multivariate properties of the bias corrected climate model.

|pypi_release| |pypi_status| |PyPI license| |pypi_downloads| |docs| 

|PyPI pyversions| |made-with-python| |made-with-sphinx-doc|

|Maintenance yes| |Ask Me Anything !| |GitHub contributors|















[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest)

[![PyPI download week](https://img.shields.io/pypi/dw/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![PyPi version](https://badgen.net/pypi/v/pip/)](https://pypi.com/project/pip)
[![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![PyPi license](https://badgen.net/pypi/license/pip/)](https://pypi.com/project/pip/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-sphinx-doc](https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg)](https://www.sphinx-doc.org/)

[![GitHub commits](https://badgen.net/github/commits/Naereen/Strapdown.js)](https://GitHub.com/Naereen/StrapDown.js/commit/)
[![GitHub contributors](https://img.shields.io/github/contributors/Naereen/badges.svg)](https://GitHub.com/Naereen/badges/graphs/contributors/)

Debiasers implemented:

- Linear Scaling
- Delta Change
- Quantile Mapping (with delta change)
- Quantile Delta Mapping following Cannon et al. 2015
- Equidistant CDF Matching following Li et al. 2010
- Scaled Distribution Mapping following Switanek 2017
- CDFt following Michelangeli et al. 2009
- ISIMIP following Lange 2019

A template repository for developing Python packages

**Quick start**

Follow these steps to create a new repository from this template.

#. Click the `Use this template <https://github.com/esowc/python-package-template/generate>`_
   button and create a new repository with your desired name, location and visibility.

#. Clone the repository::

     git clone git@github.com:esowc/<your-repository-name>.git
     cd <your-repository-name>

#. Remove sample code::

     rm ibicus/sample.py
     rm tests/test_sample.py

#. Replace ``ibicus`` with your chosen package name::

     NEW_ibicus=<your-package-name>
     mv ibicus $NEW_ibicus
     sed -i "" "s/ibicus/$NEW_ibicus/g" setup.py \
        docs/source/conf.py \
        docs/source/getting_started/installing.rst \
        docs/source/index.rst \
        $NEW_ibicus/__meta__.py

#. Modify the contents of ``__meta__.py`` to reflect your repository. Note that there
   is no need to update this same information in ``setup.py``, as it will be imported
   directly from ``__meta__.py``.

#. Modify the project url in ``setup.py`` to reflect your project's home in GitHub.

#. Modify ``README.rst`` to reflect your repository. A number of `shield <https://shields.io/>`_
   templates are included, and will need to be updated to match your repository if you want
   to use them.

**Usage tips**

* Create an executable called ``qa`` containing the following::

    black .
    isort .

  Add this to your path, and run it from the top-level of your repository before
  committing changes::

    qa .

.. |pypi_release| image:: https://img.shields.io/pypi/v/thermofeel?color=green
    :target: https://pypi.org/project/thermofeel

.. |pypi_status| image:: https://img.shields.io/pypi/status/thermofeel
    :target: https://pypi.org/project/thermofeel

.. |pypi_downloads| image:: https://img.shields.io/pypi/dm/thermofeel
  :target: https://pypi.org/project/thermofeel
  
.. |docs| image:: https://readthedocs.org/projects/thermofeel/badge/?version=latest
  :target: https://thermofeel.readthedocs.io/en/latest/?badge=latest

.. |Maintenance yes| image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity

.. |Website ibicus| image:: https://img.shields.io/website-up-down-green-red/http/monip.org.svg
   :target: https://readthedocs.org/

.. |Ask Me Anything !| image:: https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg
   :target: mailto:ibicus.py@gmail.com

.. |made-with-python| image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/

.. |made-with-sphinx-doc| image:: https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg
   :target: https://www.sphinx-doc.org/

.. |PyPI download month| image:: https://img.shields.io/pypi/dm/ansicolortags.svg
   :target: https://pypi.python.org/pypi/ansicolortags/

.. |PyPI version shields.io| image:: https://img.shields.io/pypi/v/ansicolortags.svg
   :target: https://pypi.python.org/pypi/ansicolortags/

.. |PyPI license| image:: https://img.shields.io/pypi/l/ansicolortags.svg
   :target: https://pypi.python.org/pypi/ansicolortags/

.. |PyPI pyversions| image:: https://img.shields.io/pypi/pyversions/ansicolortags.svg
   :target: https://pypi.python.org/pypi/ansicolortags/

.. |PyPI status| image:: https://img.shields.io/pypi/status/ansicolortags.svg
   :target: https://pypi.python.org/pypi/ansicolortags/

.. |Documentation Status| image:: https://readthedocs.org/projects/ansicolortags/badge/?version=latest
   :target: http://ansicolortags.readthedocs.io/?badge=latest

.. |GitHub contributors| image:: https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg
   :target: https://GitHub.com/Naereen/StrapDown.js/graphs/contributors/
