.. _whatisdebiasing:

What is bias adjustment?
========================

Why do we need bias adjustment?
-------------------------------

Even though climate models (CMs), both regional and global, have gained impressive skill in recent decades,
model errors and biases do still exist. Model biases are systematic differences between the simulated climate
statistic and the corresponding real-world statistic over a historic period where observations exist (Maraun 2016).

These biases could be due to inaccurately resolved topography of the model, due to misrepresentation of
convection processes, due to misplacement of large scale processes such as the midlatitude circulation
or many other factors. These model errors then manifest as discrepancies in the values and statistical properties of meteorological variables.
With biases in place it can be difficult to use this 'raw' climate model output to project real-world impact of climate change.

'Bias adjustment' methods, which have now become a standard pre-processing step for climate impact studies, calibrate an empirical transfer function
between simulated and observed distribution over a historical period that can be used to adjust the climate model output over a future period.

Overview of existing methods for bias adjustment
------------------------------------------------

This section provides a brief overview of different methods for bias adjustment that have been developed over the years. For a more detailed overview see the ibicus paper (Spuler et al. 2023), and for a more in-depth explanation of the specific bias adjustment methods implemented in this package and options for modifying them, please have a look at the `documentation of the debiaser class <../reference/debias.html>`_. For a more detailed general overview of different methods see Spuler et al 2023.

Two simple methods include the delta method (adding or multiplying a climate change signal to historical observational data) and
linear scaling (calculating the present-day bias in mean and variance, leaving the ratio constant and applying it to the future climate model output).

Quantile mapping is currently the most prominent method for bias adjustment in the literature (Maraun 2016, Holthuijzen et al. 2021) -- it adjusts not only the mean / variance
but provides a quantile-by-quantile mapping of the simulated to the observed distribution. Non-parametric methods can be more accurate in
terms of the fit, but can run into issues of overfitting, whilst parametric methods are more robust and might even allow
extrapolation outside of observed quantiles (Themeßl et al. 2012) but obviously need the fit to be good enough to produce meaningful results.
One issue with quantile mapping is that it is in general not trend preserving.

A range of more complex methods build on quantile mapping in different ways: these include Quantile Delta Mapping, CDFt, Equidistant CDF Matching or the ISIMIP3BASD approach. Those
are oftentimes trend preserving in the mean and other quantiles, based on different assumptions. For an overview of the inner workings
of these and other methods, please have a look at the :py:mod:`debias`-module.

The methods named so far all implement univariate bias adjustment, meaning they work location-wise and do not provide an adjustment of
either spatial or inter-variable structure. Multivariate and spatial quantile mapping methods exist (eg. Vrac et al. 2015) -- allowing for the
correction of dependences, next to marginal properties, for multiple meteorological variables, or singular ones at multiple locations.
Whether or not to correct for example the inter-variable structure, which could be seen as an integral feature of the climate model, and how such a correction could be evaluated
is a contentious and debated topic of research. If such correction is necessary, the excellent
MBC (https://cran.r-project.org/web/packages/MBC/index.html) or SBCK (https://github.com/yrobink/SBCK) package are suitable solutions.

ibicus currently implements eight methods for bias adjustment published in peer-reviewed literature. The package enables the user to
modify and refine their settings and parameters, and provides an evaluation framework to assess marginal, temporal, spatial, and
multivariate properties of the bias corrected climate model.


Limitations of bias adjustment
------------------------------

When applying a bias adjustment method, it is important to keep in mind that there are some issues that bias adjustment
will be able to correct, and some that it will not. A bias adjustment method might in general be able to iron out discrepancies
in local-scale marginal statistics, but not move and reassemble a large-scale pattern such as the North Atlantic storm track to
a more suitable location. Similarly, marginal bias adjustment cannot correct larger-scale temporal or spatial structures or
feedback such as a misrepresentation of regional responses to large-scale processes. Put differently, no bias adjustment methodology
cannot correct fundamental misrepresentations of the climate model. We refer to Maraun et al. 2017 for a good overview of issues with bias adjustment
and an appeal to place a solid understanding of the process meant to be bias corrected at the center of any bias adjustment exploration.

Further limitations include the assumption that the climate model bias of today can tell us anything about the bias the simulations
of the climate model in the future (i.e. that the bias is stationary), as well as biases and errors in the observational
and reanalysis data, that is often used to bias correct.

Applying ibicus in a case study over the Mediterranean region using seven CMIP6 global circulation models, Spuler et al (2023) find that the most appropriate bias adjustment method depends on the variable and impact studied. This finding highlight the importance of a use-case-specific choice of method and the need for a rigorous multivariate evaluation of results when applying statistical bias adjustment. Furthermore, the authors find and that even methods that aim to preserve the climate change trend can modify it.




The importance of evaluation as well as climate model and method selection
--------------------------------------------------------------------------

So what do these limitations of bias adjustment mean in concrete terms for applying it in practice?

- The first point is certainly to understand the source of present-day biases in the chosen climate model. CMIP6 literature and software such as ESMValTool can help you explore whether a particular climate model is suitable for your problem.
- Understanding whether the trend should be considered plausible or should rather be bias corrected. This is an area of emerging research, where for example research on emergent constraints can prove useful. If you consider a trend to be plausible, a trend-preserving bias adjustment method should be applied.
- The most directly applicable of these three points is to evaluate the performance of different bias adjustment methods for the use case at hand, select the best one, and transparently communicate shortcomings of the method. For this, the user can make us of the evaluation framework provided as part of ibicus to not only evaluate the marginal aspects (location-wise correction of each quantile), but also investigate the implications of bias adjustment for temporal, spatial, and spatio-temporal aspects of the data. In addition, users should investigate climate metrics (such as dry days, hot days, etc.) that are particularly relevant to their specific use-case. This is possible using the ``metrics`` module with a number of metrics are provided as default and the ability to define new ones.


Bias adjustment vs downscaling
------------------------------

Bias adjustment is also used often to downscale the climate model output, however studies
have shown that common methods produce artefacts and cannot explain sub-grid variability (Maraun 2013, Maraun 2016). Therefore we recommend the application
of bias adjustment and ibicus at constant resolution.

See also
--------

- Tutorial notebooks `01 Getting started <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/01%20Getting%20Started.ipynb>`_, `02 Adjusting debiasers <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/02%20Adjusting%20Debiasers.ipynb>`_ and `03 Evaluation <https://nbviewer.org/github/ecmwf-projects/ibicus/blob/main/notebooks/03%20Evaluation.ipynb>`_, as well as the documentation of the  `debias <../reference/debias.html>`_ and `evaluate <../reference/evaluate.hmtl>`_ module give information on how to use ibicus.
- This `guidance from the UK MET Office <https://www.metoffice.gov.uk/binaries/content/assets/metofficegovuk/pdf/research/ukcp/ukcp18-guidance---how-to-bias-correct.pdf>`_ provides an good introduction into bias adjustment: Fung, F (2018). How to Bias Correct, UKCP18 Guidance, Met Office.
- `Maraun 2016 <https://doi.org/10.1007/s40641-016-0050-x>`_ provides an excellent review of some of the issues with bias adjustment: Maraun, D. (2016). Bias Correcting Climate Change Simulations - a Critical Review. In Current Climate Change Reports (Vol. 2, Issue 4, pp. 211-220). Springer Science and Business Media LLC.

**References:**

- Holthuijzen, M. F., Beckage, B., Clemins, P. J., Higdon, D., & Winter, J. M. (2021). Constructing High-Resolution, Bias-Corrected Climate Products: A Comparison of Methods. In Journal of Applied Meteorology and Climatology (Vol. 60, Issue 4, pp. 455–475). American Meteorological Society. https://doi.org/10.1175/jamc-d-20-0252.1
- Maraun, D. (2013). Bias Correction, Quantile Mapping, and Downscaling: Revisiting the Inflation Issue. In Journal of Climate (Vol. 26, Issue 6, pp. 2137–2143). American Meteorological Society. https://doi.org/10.1175/jcli-d-12-00821.1
- Maraun, D. Bias Correcting Climate Change Simulations - a Critical Review. Curr Clim Change Rep 2, 211–220 (2016). https://doi.org/10.1007/s40641-016-0050-x
- Maraun, D., Shepherd, T. G., Widmann, M., Zappa, G., Walton, D., Gutiérrez, J. M., Hagemann, S., Richter, I., Soares, P. M. M., Hall, A., & Mearns, L. O. (2017). Towards process-informed bias correction of climate change simulations. In Nature Climate Change (Vol. 7, Issue 11, pp. 764–773). Springer Science and Business Media LLC. https://doi.org/10.1038/nclimate3418
- Spuler, F. R., Wessel, J. B., Comyn-Platt, E., Varndell, J., and Cagnazzo, C.: ibicus: a new open-source Python package and comprehensive interface for statistical bias adjustment and evaluation in climate modelling (v1.0.1), Geosci. Model Dev., 17, 1249–1269, https://doi.org/10.5194/gmd-17-1249-2024, 2024.
- Themeßl, M. J., Gobiet, A., & Heinrich, G. (2011). Empirical-statistical downscaling and error correction of regional climate models and its impact on the climate change signal. In Climatic Change (Vol. 112, Issue 2, pp. 449–468). Springer Science and Business Media LLC. https://doi.org/10.1007/s10584-011-0224-4
- Vrac, M., & Friederichs, P. (2014). Multivariate—Intervariable, Spatial, and Temporal—Bias Correction*. In Journal of Climate (Vol. 28, Issue 1, pp. 218–237). American Meteorological Society. https://doi.org/10.1175/jcli-d-14-00059.1
