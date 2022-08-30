.. _whatisdebiasing:

What is debiasing?
==================

Why do we need bias correction?
-------------------------------

Even though climate models (CMs), both regional and global, have gained impressive skill in recent decades, 
model errors and biases can persist. Model biases are systematic differences between the simulated climate 
statistic and the corresponding real-world statistic over a historic period where observations exist [1]. 

These biases could be due to an inaccurately resolved topography of the model, due to a misrepresentation of 
convection processes overall, due to a misplacement of large scale processes such as the midlatitude circulation 
or many other factors. These biases then manifest as discrepancies in the values and statistical properties of meteorological variables.
With biases in place, these ‘raw’ climate model output cannot not be used for studies to project real-world impact of climate change. 

This is where ‘bias correction’ methods come in, which have now become standard procedure for many climate impact studies. 
The general idea behind bias correction is to calibrate an empirical transfer function between simulated and observed distributional 
parameters that bias adjusts the climate model output. 

A broad overview of existing methods for bias correction
--------------------------------------------------------
				
A variety of different methods exist for bias correction that have been developed over the years. We will just give a very brief overview here - 
for a detailed explanation of the specific bias correction methods implemented in this package, please have a look at the documentation of the 
Debiaser class on this page. 

Two simple methods include the delta method (adding or multiplying a climate change signal to historical observational data) and 
linear scaling (calculating the present-day bias in mean and variance, leaving the ratio constant and applying it to the future climate model output).
		
Quantile mapping is currently the most prominent method for bias correction in the literature [2] - it adjusts not only the mean / variance 
but provides a quantile-by-quantile mapping of the simulated to the observed distribution. Non-parametric methods can be more accurate in 
terms of the fit, but can quickly run into issues of overfitting, whilst parametric methods are more robust and might even allow 
extrapolation outside of observed quantiles [3] but obviously need the fit to be good enough to produce meaningful results. 
One issue with quantile mapping is that it is in general not trend preserving.

More complex methods include for example Quantile Delta Mapping, CDFt, Equidistant CDF Matching or the ISIMIP3BASD approach. Those 
are oftentimes trend preserving in the mean and other quantiles. The methods are most of the time based on some form of quantile 
mapping as core, but with different additional adjustments. They also have different assumptions. For an overview of the inner workings 
of those and some other methods, please have a look at the debias-module (LINK).

The methods named so far all provide univariate bias adjustment: meaning they work location-wise and do not provide an adjustment of 
either spatial or inter-variable structure. Also, multivariate and spatial quantile mapping methods exist [4] – allowing for the 
correction of dependences, next to marginal properties, for multiple meteorological variables, or singular ones at multiple locations.
Whether or not to correct for example the inter-variable structure, which could be seen as an integral feature of the climate model, 
is a contentious and debated topic of research. If such correction is necessary, the excellent 
MBC (https://cran.r-project.org/web/packages/MBC/index.html) or SBCK (https://github.com/yrobink/SBCK) package are suitable solutions.

Ibicus currently implements eight methods for bias correction published in peer-reviewed literature. The package enables the user to 
modify and refine their settings and parameters, and provides an evaluation framework to assess marginal, temporal, spatial, and 
multivariate properties of the bias corrected climate model.


Limitations of bias correction
------------------------------

When applying a bias correction method, it is important to keep in mind that there are some issues that bias correction 
will be able to correct, and some that it will not. A bias correction method might in general be able to iron out discrepancies
in local-scale marginal statistics but not move and reassemble a large-scale pattern such as the North Atlantic storm track to
a more suitable location. Similarly, marginal bias correction cannot correct larger-scale temporal or spatial structures or 
feedback such as a misrepresentation of regional responses to large-scale processes. Put differently, no bias correction methodology
cannot correct fundamental misrepresentations of the climate model. We refer to [5] for a good overview of issues with bias correction
and an appeal to place a solid understanding of the process meant to be bias corrected at the center of any bias correction exploration.

Further limitations include the assumptions that exploring the climate model bias of today can tell us anything about the bias the simulations 
of the climate model into the future has under different scenarios (i.e. that the bias is stationary), as well as the issues with observational 
and reanalysis data, that is often used to bias correct.


The importance of evaluation as well as climate model and method selection
--------------------------------------------------------------------------

So what do these limitations of bias correction mean in concrete terms for applying it in practice? 

- The first point is certainly to understand the source of present-day biases in the chosen climate model. CMIP6 literature and software such as ESMValTool can help you explore whether a particular climate model is suitable for your problem.
- Understanding whether the trend should be considered plausible or should rather be bias corrected. This is an area of emerging research, where for example research on emergent constraints can prove useful. If you consider a trend to be plausible, a trend-preserving bias correction method should be applied.
- The most directly applicable of these three points is to evaluate the performance of different bias correction methods for the use case at hand, select the best one and transparently communicate shortcoming of the method. For this, the user can make us of the evaluation framework provided as part of ibicus to not only evaluate the marginal aspects of the bias correction (location-wise correction of each quantile), but also investigate the bias correction of temporal, spatial, and spatio-temporal aspects of the data - and the In addition, you should investigate metrics such as ‘frost days’ that are particularly relevant to your specific use-case. A number of metrics are provided as ‘default metrics’, however you can specify your own metrics using the ThresholdMetrics and AccumulativeThresholdMetrics classes (see documentation)


Bias correction vs downscaling
------------------------------

Oftentimes, bias correction is also used to downscale the climate model output, however since recent studies ([1], [6]) 
have shown that common methods produce artefacts and cannot explain sub-grid variability, we will leave this discussion aside
in this proposal and focus on bias correction in our project.	

See also
--------

- Documentation of the  debias and evaluate module providing info on how to use ibicus
- This [guidance from the UK MET Office](https://www.metoffice.gov.uk/binaries/content/assets/metofficegovuk/pdf/research/ukcp/ukcp18-guidance---how-to-bias-correct.pdf) provides an excellent intro into bias correction: Fung, F (2018). How to Bias Correct, UKCP18 Guidance, Met Office.
- This [publication by Maraun 2016](https://link.springer.com/article/10.1007/s40641-016-0050-x) provides an excellent review of some of the issues with bias correction: Maraun, D. (2016). Bias Correcting Climate Change Simulations - a Critical Review. In Current Climate Change Reports (Vol. 2, Issue 4, pp. 211–220). Springer Science and Business Media LLC. https://doi.org/10.1007/s40641-016-0050-x

**References:**

[1] D. Maraun, “Bias Correcting Climate Change Simulations - a Critical Review,” Current Climate Change Reports, vol. 2, no. 4, pp. 211–220, Dec. 2016, doi: 10.1007/s40641-016-0050-x.

[2] M. F. Holthuijzen, B. Beckage, P. J. Clemins, D. Higdon, and J. M. Winter, “Constructing High-Resolution, Bias-Corrected Climate Products: A Comparison of Methods,” Journal of Applied Meteorology and Climatology, vol. 60, no. 4, pp. 455–475, Apr. 2021, doi: 10.1175/JAMC-D-20-0252.1.

[3] M. J. Themeßl, A. Gobiet, and G. Heinrich, “Empirical-statistical downscaling and error correction of regional climate models and its impact on the climate change signal,” Climatic Change, vol. 112, no. 2, pp. 449–468, May 2012, doi: 10.1007/s10584-011-0224-4.

[4] M. Vrac and P. Friederichs, “Multivariate—Intervariable, Spatial, and Temporal—Bias Correction*,” Journal of Climate, vol. 28, no. 1, pp. 218–237, Jan. 2015, doi: 10.1175/JCLI-D-14-00059.1.

[5] D. Maraun et al., “Towards process-informed bias correction of climate change simulations,” in Nature Climate Change, Nov. 2017, vol. 7, no. 11, pp. 764–773. doi: 10.1038/nclimate3418.

[6] D. Maraun, “Bias Correction, Quantile Mapping, and Downscaling: Revisiting the Inflation Issue,” Journal of Climate, vol. 26, no. 6, pp. 2137–2143, Mar. 2013, doi: 10.1175/JCLI-D-12-00821.1

