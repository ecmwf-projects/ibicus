.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. # define a double hard line break for HTML
.. |brr| raw:: html

   <br /> <br />


PACKAGE\_NAME.utils module
==========================

.. automodule:: PACKAGE_NAME.utils
   :members:
   :undoc-members:
   :show-inheritance:


PACKAGE\_NAME.utils Convert variables
-------------------------------------

.. autofunction:: get_tasrange

.. autofunction:: get_tasskew

.. autofunction:: get_tasmin

.. autofunction:: get_tasmax

.. autofunction:: get_tasmin_tasmax

.. autofunction:: get_tasrange_tasskew

.. autofunction:: get_prsnratio

.. autofunction:: get_pr

.. autofunction:: get_prsn



PACKAGE\_NAME.utils StatisticalModel abstract-class
----------------------------------------------------

.. autoclass:: StatisticalModel
	:members: fit, cdf, ppf


PACKAGE\_NAME.utils gen\_PrecipitationIgnoreZeroValuesModel-class
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. autoclass:: gen_PrecipitationIgnoreZeroValuesModel
	:members: fit, cdf, ppf
   

PACKAGE\_NAME.debias gen\_PrecipitationHurdleModel-class
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. autoclass:: gen_PrecipitationHurdleModel
	:members: fit, cdf, ppf


PACKAGE\_NAME.utils gen\_PrecipitationGammaLeftCensoredModel-class
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. autoclass:: gen_PrecipitationGammaLeftCensoredModel
	:members: fit, cdf, ppf
   

PACKAGE\_NAME.utils Mathematical helpers
-----------------------------------------

.. autofunction:: ecdf

.. autofunction:: iecdf

.. autofunction:: quantile_map_non_parametically
