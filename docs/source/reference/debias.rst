.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. # define a double hard line break for HTML
.. |brr| raw:: html

   <br /> <br />


ibicus.debias module
===========================

.. automodule:: ibicus.debias
   :members:
   :undoc-members:
   :show-inheritance:

----

ibicus.debias Debiaser abstract-class
--------------------------------------------

.. autoclass:: Debiaser
	:members: from_variable, apply, apply_location

ibicus.debias RunningWindowDebiaser abstract-class
--------------------------------------------

.. autoclass:: RunningWindowDebiaser
	:members: from_variable, apply, apply_on_window

ibicus.debias LinearScaling class
----------------------------------------

.. autoclass:: LinearScaling
	:members: from_variable, apply

ibicus.debias DeltaChange class
--------------------------------------

.. autoclass:: DeltaChange
	:members: from_variable, apply


ibicus.debias QuantileMapping class
------------------------------------------

.. autoclass:: QuantileMapping
	:members: from_variable, for_precipitation, apply


ibicus.debias ScaledDistributionMapping class
----------------------------------------------------

.. autoclass:: ScaledDistributionMapping
	:members: from_variable, apply


ibicus.debias CDFt class
-------------------------------

.. autoclass:: CDFt
	:members: from_variable, apply


ibicus.debias ECDFM class
--------------------------------

.. autoclass:: ECDFM
	:members: from_variable, for_precipitation, apply


ibicus.debias QuantileDeltaMapping class
-----------------------------------------------

.. autoclass:: QuantileDeltaMapping
	:members: from_variable, for_precipitation, apply

ibicus.debias ISIMIP class
---------------------------------

.. autoclass:: ISIMIP
	:members: from_variable, apply, step1, step2, step3, step4, step5, step6, step7, step8
