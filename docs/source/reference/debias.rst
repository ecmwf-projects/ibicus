.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. # define a double hard line break for HTML
.. |brr| raw:: html

   <br /> <br />


PACKAGE\_NAME.debias module
===========================

.. automodule:: PACKAGE_NAME.debias
   :members:
   :undoc-members:
   :show-inheritance:
   

PACKAGE\_NAME.debias Debiaser abstract-class
--------------------------------------------

.. autoclass:: Debiaser
	:members: from_variable, apply, apply_location, _from_variable

   
PACKAGE\_NAME.debias LinearScaling class
----------------------------------------

.. autoclass:: LinearScaling
	:members: from_variable, apply, apply_location

PACKAGE\_NAME.debias DeltaChange class
--------------------------------------

.. autoclass:: DeltaChange
	:members: from_variable, apply, apply_location


PACKAGE\_NAME.debias QuantileMapping class
------------------------------------------

.. autoclass:: QuantileMapping
	:members: from_variable, for_precipitation, apply, apply_location


PACKAGE\_NAME.debias ScaledDistributionMapping class
----------------------------------------------------

.. autoclass:: ScaledDistributionMapping
	:members: from_variable, apply, apply_location
ScaledDistributionMapping


PACKAGE\_NAME.debias CDFt class
-------------------------------

.. autoclass:: CDFt
	:members: from_variable, apply, apply_location


PACKAGE\_NAME.debias ECDFM class
--------------------------------

.. autoclass:: ECDFM
	:members: from_variable, for_precipitation, apply, apply_location
	
	
PACKAGE\_NAME.debias QuantileDeltaMapping class
-----------------------------------------------

.. autoclass:: QuantileDeltaMapping
	:members: from_variable, for_precipitation, apply, apply_location

PACKAGE\_NAME.debias ISIMIP class
---------------------------------

.. autoclass:: ISIMIP
	:members: from_variable, apply, apply_location, step1, step2, step3, step4, step5, step6, step7, step8






