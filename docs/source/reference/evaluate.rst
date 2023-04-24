.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. # define a double hard line break for HTML
.. |brr| raw:: html

   <br /> <br />


ibicus.evaluate module
=============================

.. automodule:: ibicus.evaluate
   :members:
   :undoc-members:
   :show-inheritance:


ibicus.evaluate.metrics
------------------------------

.. automodule:: ibicus.evaluate.metrics


ibicus.evaluate.metrics.ThresholdMetric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ThresholdMetric
	:members: from_quantile, calculate_instances_of_threshold_exceedance, filter_threshold_exceedances, calculate_exceedance_probability, calculate_number_annual_days_beyond_threshold, calculate_spell_length, calculate_spatial_extent, calculate_spatiotemporal_clusters, violinplots_clusters


ibicus.evaluate.metrics.AccumulativeThresholdMetric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AccumulativeThresholdMetric
	:members: from_quantile, calculate_percent_of_total_amount_beyond_threshold, calculate_annual_value_beyond_threshold, calculate_intensity_index


Concrete metrics
^^^^^^^^^^^^^^^^

.. autodata:: dry_days

.. autodata:: wet_days

.. autodata:: R10mm

.. autodata:: R20mm

.. autodata:: warm_days

.. autodata:: cold_days

.. autodata:: frost_days

.. autodata:: tropical_nights

.. autodata:: summer_days

.. autodata:: icing_days

----

ibicus.evaluate.marginal
-------------------------------

.. automodule:: ibicus.evaluate.marginal
   :members:
   :undoc-members:
   :show-inheritance:

----

ibicus.evaluate.multivariate
-----------------------------------

.. automodule:: ibicus.evaluate.multivariate
   :members:
   :undoc-members:
   :show-inheritance:

----

ibicus.evaluate.correlation
----------------------------------

.. automodule:: ibicus.evaluate.correlation
   :members:
   :undoc-members:
   :show-inheritance:

----

ibicus.evaluate.trend
----------------------------

.. automodule:: ibicus.evaluate.trend
   :members:
   :undoc-members:
   :show-inheritance:

----

ibicus.evaluate.assumptions
----------------------------------

.. automodule:: ibicus.evaluate.assumptions
   :members:
   :undoc-members:
   :show-inheritance:
