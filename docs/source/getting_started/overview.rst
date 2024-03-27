.. _overview:

.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. # define a double hard line break for HTML
.. |brr| raw:: html

   <br /> <br />

Overview
========


**ibicus provides a flexible and user-friendly toolkit for the bias adjustment of climate models and associated evaluation.**

ibicus currently implements eight state of the art methods for bias adjustment published in peer-reviewed literature, including ISIMIP (Lange 2019) and CDFt (Michelangeli et al. 2009). The package enables the user to modify and refine their settings and parameters, and provides an evaluation framework to assess marginal, temporal, spatial, and multivariate properties as well as a range of climate indices of the bias adjusted climate model.

You can find an introduction-video to ibicus `here <https://www.youtube.com/watch?v=n8QlGLU2gIo>`_.

What is bias adjustment?
------------------------

Even though climate models have gained impressive skill over the recent years, simulations of both global and regional climate models still exhibit biases. This means that there are systematic discrepancies between statistics of the model output and observational distribution. These discrepancies in the two distributions become especially relevant when using the output of climate models for local impact studies that often require focus on specific threshold metrics such as dry days.

Biases could be due to unresolved topography, the parametrization of processes such as convection occurring below grid-cell level, or the misplacement of large-scale atmospheric patterns. Bias adjustment cannot fundamentally address these issues. Rather, bias adjustment calibrates a statistical transfer function between simulated and observed distributional parameters in order to improve (“bias adjust”) the climate model. While bias adjustment can reduce the location-wise bias of the climate model in many cases, the approach is also prone to misuse and can introduce changes in the spatial, temporal and multivariate structure of the climate model as well as the climate change trend. Nevertheless, bias adjustment has de-facto become a standard pre-processing step in climate impact studies.

For a more detailed introduction, as well as an overview of relevant literature on existing methods and issues, we refer to our paper published in Geoscientific Model Development:

How to cite: Spuler, F. R., Wessel, J. B., Comyn-Platt, E., Varndell, J., and Cagnazzo, C.: ibicus: a new open-source Python package and comprehensive interface for statistical bias adjustment and evaluation in climate modelling (v1.0.1), Geosci. Model Dev., 17, 1249–1269, https://doi.org/10.5194/gmd-17-1249-2024, 2024.

What is ibicus?
---------------

*A user-friendly toolkit to bias adjust climate models…*

A variety of methods have been developed for the bias adjustment of climate models. Some are better suited for certain variables; others will introduce modifications to the climate change trend while others are explicitly trend-preserving. ibicus provides a unified interface for applying a variety of different methods (8 currently) for bias adjustment published in peer reviewed literature.

Given climate model data: during a reference period (``cm_hist``) and future / application period (``cm_future``) as well as observations or reanalysis data during the reference period (``obs``), ibicus provides a standardized user-interface for initializing and applying a bias adjustment method, for example ISIMIP:

>>> from ibicus import ISIMIP
>>> debiaser = ISIMIP.from_variable("tas")
>>> debiased_cm_future = debiaser.apply(obs, cm_hist, cm_future)

The methods currently implemented in ibicus include:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * -
     - References
     - Brief description
   * - :py:class:`ISIMIP`
     - * Lange 2019 |brr|
       * Lange 2021
     - ISIMIP is a semi-parametric quantile mapping method that attempts to be trend-preserving by generating ‘pseudo future observations’ and executing the quantile mapping between the future climate model and the pseudo future observations. ISIMIP includes special cases for each of the variables, and for a complete description of the methodology we refer to the ISIMIP documentation.
   * - :py:class:`LinearScaling`
     - * Maraun 2016
     - Linear scaling corrects a climate model by the difference in the mean of observations and the mean of the climate model on the reference period, either additively or multiplicatively.
   * - :py:class:`QuantileMapping`
     - * Cannon et al. 2015 |brr|
       * Maraun 2016
     - (Parametric) quantile mapping maps every quantile of the climate model distribution to the corresponding quantile in observations during the reference period. Optionally, additive or multiplicative detrending of the mean can be applied to make the method trend preserving. Most bias adjustment methods build on quantile mapping.
   * - :py:class:`ScaledDistributionMapping`
     - * Switanek et al. 2017
     - SDM is conceptually similar to QDM, and in the same ‘family’ as CDFt and ECDFM. It is a parametric quantile mapping approach that also attempts to be trend preserving in all quantiles. In addition to the quantile mapping the method also contains an event likelihood adjustment.
   * - :py:class:`CDFt`
     - * Michelangeli et al. 2009 |brr|
       * Vrac et al. 2012 |brr|
       * Famien et al. 2018 |brr|
       * Vrac et al. 2016
     - CDFt is a non-parametric quantile mapping method that attempts to be trend-preserving in all quantiles. CDFt applies a concatenation between a quantile mapping of future and historical climate model data and a quantile mapping of the future climate model with historical observations. It also includes a running window over the future period to account for changes in the simulated trend.
   * - :py:class:`ECDFM`
     - * Li et al. 2010
     - ECDFM is a parametric quantile mapping method that attempts to be trend-preserving in all quantiles. ECDFM applies quantilewise correction by adding the difference between a quantile mapping of observations and future values and a quantile mapping of historical climate model values to the future climate model ones.
   * - :py:class:`QuantileDeltaMapping`
     - * Cannon et al. 2015
     - QDM is a parametric quantile mapping method that also attempts to be trend-preserving. It extends ECDFM such that the two quantile mappings defined there can not only added but also divided by each other to create multiplicative corrections. Furthermore it includes both a running window over the year: to account for seasonality, as well as one over the future period to account for changes in trends.
   * - :py:class:`DeltaChange`
     - * Maraun 2016
     - Delta Change applies the trend from historical to future climate model to the observations. Although technically not a bias adjustment method, as no transformation is applied to the climate model, it is included here as it provides an adjusted future climatology.

Users can modify the settings of different debiasers to adapt them to their use-case, for example:

>>> pr_debiaser1 = QuantileMapping.for_precipitation(model_type = "hurdle")
>>> pr_debiaser2 = pr_debiaser2 = QuantileMapping.for_precipitation(model_type = "censored")

*… as well as a framework for evaluating the performance of different bias adjustment methods:*

Bias adjustment is prone to misuse and can generate seemingly meaningful results even if applied to variables that have no physical link whatsoever. Any bias adjustment approach should therefore include a thorough evaluation of the obtained results, not only of marginal aspects of the corrected statistics, but also comparing the multivariate, temporal and spatial structure of observations, the raw climate model and the bias corrected climate model. Furthermore users should ideally evaluate wether bias adjustment modifies derived quantities of interest such as climate indices.

ibicus includes a framework that enables the user to conduct this evaluation as part of the bias adjustment process. The evaluation framework consists of three parts:

- Evaluation of the method on a validation period: This component enables the user to compare the bias corrected model to the ‘raw’ model and observations / reanalysis data, all on a chosen validation period. Both statistical properties as well as threshold based climate indices (threshold metrics) such as dry days, hot days, etc. that are often used for calculating climate impacts can be compared. The following table summarises the types of analysis that can be conducted in this component:

+----------------+------------------------+-----------------------+
|                | Statistical properties | Threshold metrics     |
+================+========================+=======================+
| Marginal       | x                      |  x                    |
+----------------+------------------------+-----------------------+
| Temporal       |                        |  x (spell length)     |
+----------------+------------------------+-----------------------+
| Spatial        | x (RMSE)               | x (spatial extent)    |
+----------------+------------------------+-----------------------+
| Spatiotemporal |                        |  x (cluster size)     |
+----------------+------------------------+-----------------------+
| Multivariate   | x (correlation)        |  x (joint exceedance) |
+----------------+------------------------+-----------------------+

- Analysis of trend preservation: Bias adjustment can significantly modify the trend projected in the climate model simulation. This component helps the user assess whether a certain method preserves the climate model trend or not, in order to provide the basis for an informed choice on whether trend modification is desirable for the application at hand.

- Assumptions testing: this component helps the user check some assumptions underlying the use of different bias adjustment methods to choose the most appropriate method and refine its parameters.

What is ibicus not?
-------------------

After trying to convince you of the advantages of using ibicus, we also want to alert you to what ibicus currently does not do:

1. ibicus offers a multivariate evaluation of the bias adjusted climate model but does not currently support multivariate bias adjustment, meaning the correction of spatial or inter-variable structure. Whether or not to correct for example the inter-variable structure, which could be seen as an integral feature of the climate model, is a contentious and debated topic of research. If such correction is necessary, the excellent `MBC <https://cran.r-project.org/web/packages/MBC/index.html>`_ or `SBCK <https://github.com/yrobink/SBCK>`_ package are suitable solutions. For a more detailed discussion of the advantages and possible drawbacks of multivariate bias adjustment we refer to Spuler et al. (2023) cited above. |brr|

2. ibicus is not suitable for 'downscaling' the climate model which is a term for methods used to increase the spatial resolution of climate models. Although bias adjustment methods have been used for downscaling, in general they are not appropriate, since they do not reproduce the local scale variability that is crucial on those scales. Maraun 2016 argues that for downscaling, stochastic methods have great advantages. An example of a package addressing the problem of downscaling is: `Rglimclim <https://www.ucl.ac.uk/~ucakarc/work/glimclim.html>`_. |brr|

3. 'Garbage in, garbage out'. ibicus cannot guarantee that the climate model is suitable for the problem at hand. As mentioned above, although bias adjustment can help with misspecifications, it cannot solve fundamental problems within climate models. The evaluation framework can help you identify whether such fundamental issues exist in the chosen climate model. However, this cannot replace careful climate model selection before starting a climate impact study. |brr|

About the authors
-----------------

Fiona is a PhD student at the Department of Meteorology, working with Prof Marlene Kretschmer and Prof Ted Shepherd. Her work focuses on developing methods for combining dynamical and machine learning models to boost S2S forecasts of extreme weather events. Prior to starting her PhD, Fiona worked for two years at the not-for-profit organisation “2° Investing Initiative” on the alignment of the European financial sectors with climate mitigation goals. Last year, she was part of an international fellowship programme by the German foreign ministry working on finance for resilience with the Coalition for Climate Resilient Investing. Fiona holds a degree in Physics (MSc, University of Edinburgh; BSc, University of Goettingen) as well as Environmental Change and Management (MSc, University of Oxford).

Jakob Wessel is a PhD student at the University of Exeter where he is working on "Statistical post-processing of ensemble forecasts of compound weather risk" under supervision of Dr Frank Kwasniok and Dr Chris Ferro, in cooperation with the UK Met Office. Jakob is currently based at the Alan Turing Institute as part of their PhD enrichment scheme. Jakob holds an MSc in Data Science (Statistics) from University College London where he worked on an MSc dissertation about improving methods for climate model downscaling, under supervision of Prof Richard Chandler, winning the price for the best MSc dissertation. He worked as Research Analyst at the 2° Investing Initiative and gained experience as a project manager and data analyst at Serlo Education. He holds a BSc in Mathematics from Technical University Berlin and a BA in Philosophy and Political Science from Free University Berlin.


Get in touch
------------

If you have suggestions on additional methods we could add, questions you'd like to ask, issues that you are finding in the application of the methods that are already implemented, or bugs in the code, please contact us under ibicus.py@gmail.com or `raise an issue on github <https://github.com/ecmwf-projects/ibicus/issues>`_.


Cite the package
----------------

If you use ibicus for your research, please cite our publication in Geoscientific Model Development:

Spuler, F. R., Wessel, J. B., Comyn-Platt, E., Varndell, J., and Cagnazzo, C.: ibicus: a new open-source Python package and comprehensive interface for statistical bias adjustment and evaluation in climate modelling (v1.0.1), Geosci. Model Dev., 17, 1249–1269, https://doi.org/10.5194/gmd-17-1249-2024, 2024.

References
----------

- Maraun, D. Bias Correcting Climate Change Simulations - a Critical Review. Curr Clim Change Rep 2, 211–220 (2016). https://doi.org/10.1007/s40641-016-0050-x
- Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles and Extremes? In Journal of Climate (Vol. 28, Issue 17, pp. 6938–6959). American Meteorological Society. https://doi.org/10.1175/jcli-d-14-00754.1
- Switanek, M. B., Troch, P. A., Castro, C. L., Leuprecht, A., Chang, H.-I., Mukherjee, R., & Demaria, E. M. C. (2017). Scaled distribution mapping: a bias correction method that preserves raw climate model projected changes. In Hydrology and Earth System Sciences (Vol. 21, Issue 6, pp. 2649–2666). Copernicus GmbH. https://doi.org/10.5194/hess-21-2649-2017.
- Michelangeli, P.-A., Vrac, M., & Loukos, H. (2009). Probabilistic downscaling approaches: Application to wind cumulative distribution functions. In Geophysical Research Letters (Vol. 36, Issue 11). American Geophysical Union (AGU). https://doi.org/10.1029/2009gl038401
- Famien, A. M., Janicot, S., Ochou, A. D., Vrac, M., Defrance, D., Sultan, B., & Noël, T. (2018). A bias-corrected CMIP5 dataset for Africa using the CDF-t method – a contribution to agricultural impact studies. In Earth System Dynamics (Vol. 9, Issue 1, pp. 313–338). Copernicus GmbH. https://doi.org/10.5194/esd-9-313-2018
- Vrac, M., Drobinski, P., Merlo, A., Herrmann, M., Lavaysse, C., Li, L., & Somot, S. (2012). Dynamical and statistical downscaling of the French Mediterranean climate: uncertainty assessment. In Natural Hazards and Earth System Sciences (Vol. 12, Issue 9, pp. 2769–2784). Copernicus GmbH. https://doi.org/10.5194/nhess-12-2769-2012
- Vrac, M., Noël, T., & Vautard, R. (2016). Bias correction of precipitation through Singularity Stochastic Removal: Because occurrences matter. In Journal of Geophysical Research: Atmospheres (Vol. 121, Issue 10, pp. 5237–5258). American Geophysical Union (AGU). https://doi.org/10.1002/2015jd024511
- Li, H., Sheffield, J., and Wood, E. F. (2010), Bias correction of monthly precipitation and temperature fields from Intergovernmental Panel on Climate Change AR4 models using equidistant quantile matching, J. Geophys. Res., 115, D10101, doi:10.1029/2009JD012882.
- Lange, S. (2019). Trend-preserving bias adjustment and statistical downscaling with ISIMIP3BASD (v1.0). In Geoscientific Model Development (Vol. 12, Issue 7, pp. 3055–3070). Copernicus GmbH. https://doi.org/10.5194/gmd-12-3055-2019
- Lange, S. (2022). ISIMIP3BASD (3.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.6758997
