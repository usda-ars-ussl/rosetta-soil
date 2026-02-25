This package provides an implementation of **Rosetta**, a group of
neural network model for predicting unasturated soil hydraulic parameters
from basic soil characterization data.

For many use cases it is not necessary to install this package
-- there is a web browser interface and an api to ``rosetta-soil`` that
is available at `<https://www.handbook60.org/rosetta>`_

 
Installation
============
::

    pip install rosetta-soil

Quickstart
==========
::

    >>> from rosetta import rosetta

    # required ordering for data records:
    # [sa (%), si (%), cl (%), bd (g/cm3), th33, th1500]
    # sa, si, and cl are required; others optional

    >>> data = [
            [30,30,40,1.5,0.3,0.1],
            [20,60,20],
            [55,25,20,1.1]
        ]

    >>> mean, stdev, codes = rosetta(3, data)

    >>> with np.printoptions(precision=4, suppress=True):
    ...     print(mean)

    [[ 0.0687  0.3839  0.0037  1.5123 10.4522  1.0203  0.2242]
     [ 0.0899  0.4301  0.0038  1.4993 15.8995  0.8909  0.1726]
     [ 0.0913  0.485   0.0097  1.4172 84.7834  2.9015 -0.3463]]

    >>> with np.printoptions(precision=4, suppress=True):
    ...  print(stdev)

    [[ 0.0136  0.015   0.0012  0.1176  4.816   0.4968  1.4017]
     [ 0.0067  0.0088  0.0007  0.0454  3.1789  0.449   1.6119]
     [ 0.0128  0.0131  0.0022  0.0577 27.0357  1.8203  1.2248]]

    >>> print(codes)

    [5 2 3]

Background
==========

The Rosetta pedotransfer functions predict seven parameters for the van
Genuchten model of unsaturated soil hydraulic properties

* theta_r      : residual volumetric water content
* theta_s      : saturated volumetric water content
* alpha        : retention shape parameter (1/cm)
* n            : retention shape parameter
* ksat         : saturated hydraulic conductivity (cm/d)
* k0           : unsaturated conductivity matching point (cm/d)
* L            : unsaturated conductivity exponent

Rosetta provides four models for predicting the seven parameters from soil
characterization data. The models differ in the required input data

+------------+------------------------+
| Model Code | Input Data             |
+============+========================+
|      2     | sa, si, cl (SSC)       |
+------------+------------------------+
|      3     | SSC, bulk density (BD) |
+------------+------------------------+
|      4     | SSC, BD, th33          |
+------------+------------------------+
|      5     | SSC, BD, th33, th1500  |
+------------+------------------------+

where

* sa, si, cl are percentages of sand, silt and clay
* BD is soil bulk density (g/cm3)
* th33 is the soil volumetric water content at 33 kPa
* th1500 is the soil volumetric water content at 1500 kPa

Three versions of Rosetta are available for predicting theta_r, theta_s
alpha, n, and ksat. The versions effectively represent three alternative
calibrations or trainings of the four Rosetta models. The references
that should be cited when using Rosetta versions 1, 2, and 3 are,
respectively:

[1] Schaap, M.G., Leij, F.J., and Van Genuchten, M.T. 2001. ROSETTA: a
computer program for estimating soil hydraulic parameters with
hierarchical pedotransfer functions. Journal of Hydrology 251(3-4): 163-176.
doi: `10.1016/S0022-1694(01)00466-8 <https://doi.org/10.1016/S0022-1694(01)00466-8)>`_

[2] Schaap, M.G., A. Nemes, and M.T. van Genuchten. 2004. Comparison of Models
for Indirect Estimation of Water Retention and Available Water in Surface Soils.
Vadose Zone Journal 3(4): 1455-1463.
doi: `10.2136/vzj2004.1455 <https://doi.org/10.2136/vzj2004.1455>`_

[3] Zhang, Y. and Schaap, M.G. 2017. Weighted recalibration of the Rosetta
pedotransfer model with improved estimates of hydraulic parameter
distributions and summary statistics (Rosetta3). Journal of Hydrology 547: 39-53.
doi: `10.1016/j.jhydrol.2017.01.004 <https://doi.org/10.1016/j.jhydrol.2017.01.004>`_

The parameters k0 and L are predicted with a neural network model from
[1] that uses as inputs values for the retention parameters theta_r,
theta_s, alpha, and n. In the current implementation, the same model is
to predict k0 and L regardless of the version of Rosetta used to estimate
the retention paramters.


Usage
=====

Soil data should be orgainzed as a collection of data records,

::

    >>> data = [
    ...     [30, 30, 40, 1.5, 0.3, 0.1],  
    ...     [20, 60, 20],
    ...     [55, 25, 20, 1.1]
    ... ]

where each element contains soil data in this order::

    [%sand, %silt, %clay, bulk density, th33, th1500]

Sand, silt, and clay are required in each element; the others are optional.

Two functions, ``rosetta`` and ``rosesoil``, are provided for predicting soil
hydraulic parameters. The two functions have the same arguments and return
the same results, but differ in the format of the returned results.
The function ``rosetta`` returns the predicted parameters in numpy arrays,
whereas ``rosesoil`` returns ``RosettaResults`` object.

``rosetta`` and ``rosesoil`` take as arguments the Rosetta version to be used (1,
2, or 3) and the array-like soil data. A third optional argument,
``estimate_type``, specifies the parameter estimate to be returned for
alpha, npar, Ksat, and K0:
::

    estimate_type: Literal['linear', 'log', 'geo']
       'linear' (default):
           Estimate is arithmetic mean.
       'log':
           Estimate is mean of log10 transformed parameter value.
       'geo':
           Estimate is geometric mean.

The function ``rosetta`` returns a 3-tuple of numpy arrays
::

    >>> from rosetta import rosetta

    >>> data = [
    ...     [30,30,40,1.5,0.3,0.1],
    ...     [20,60,20],
    ...     [55,25,20,1.1]
    ... ]

    >>> mean, stdev, codes = rosetta(3, data)

    >>> with np.printoptions(precision=4, suppress=True):
    ...     print(mean)

    [[ 0.0687  0.3839  0.0037  1.5123 10.4522  1.0203  0.2242]
     [ 0.0899  0.4301  0.0038  1.4993 15.8995  0.8909  0.1726]
     [ 0.0913  0.485   0.0097  1.4172 84.7834  2.9015 -0.3463]]

    >>> with np.printoptions(precision=4, suppress=True):
    ...  print(stdev)

    [[ 0.0136  0.015   0.0012  0.1176  4.816   0.4968  1.4017]
     [ 0.0067  0.0088  0.0007  0.0454  3.1789  0.449   1.6119]
     [ 0.0128  0.0131  0.0022  0.0577 27.0357  1.8203  1.2248]]

    >>> print(codes)

    [5 2 3]

In the above, ``mean`` is a 2D array with shape (nsamples, 7). The ith
row holds predicted soil hydraulic parameters for ith entry in ``data``.
The columns are:

+-------+---------------------------------------------------------------+
|Column | Parameter                                                     |
+=======+===============================================================+
|   0   | theta_r, residual water content                               |
+-------+---------------------------------------------------------------+
|   1   | theta_s, saturated water content                              |
+-------+---------------------------------------------------------------+
|   2   | alpha or log10(alpha), 'alpha' shape parameter, 1/cm          | 
+-------+---------------------------------------------------------------+
|   3   | npar or log10(npar), 'n' shape parameter                      |
+-------+---------------------------------------------------------------+
|   4   | Ksat or log10(Ksat), saturated conductivity, cm/day           |
+-------+---------------------------------------------------------------+
|   5   | K0 or log10(K0), conductivity matching point, cm/day          |
+-------+---------------------------------------------------------------+
|   6   | lpar, unsaturated conductivity exponent,  cm/day              |
+-------+---------------------------------------------------------------+

``stdev`` is 2D array holding the corresponding parameter standard
deviations. Note these are standard deviations for the bootstrap resamples,
not the standard error of the mean estimate.

``codes`` is a 1D array with the ith entry indicating the Rosetta model
and input data used to predict the ith row of ``mean`` and ``stdev``.

+------+--------------------------------------------------------+
| Code | Data used                                              |
+======+========================================================+
|    2 | sand, silt, clay (SSC)                                 |
+------+--------------------------------------------------------+
|    3 | SSC + bulk density (BD)                                |
+------+--------------------------------------------------------+
|    4 | SSC + BD + field capacity water content (TH33)         | 
+------+--------------------------------------------------------+
|    5 | SSC + BD + TH33 + wilting point water content (TH1500) |
+------+--------------------------------------------------------+
|   -1 | no result returned, inadequate or erroneous data       |
+------+--------------------------------------------------------+


The function ``rosesoil`` works the same way:
::

    >>> from rosetta import rosesoil


    >>> data = [
    ...     [30,30,40,1.5,0.3,0.1],
    ...     [20,60,20],
    ...     [55,25,20,1.1]
    ... ]

    >>> result = rosesoil(3, data)

    >>> print(result)

    RosettaResults([RosettaResult(sand=30.0000, silt=30.0000, clay=40.0000,
    rhob=1.5000, th33=0.3000, th1500=0.1000, version=3, estimate_type='linear',
    code=5, thr=0.0687, ths=0.3839, alpha=0.0037, npar=1.5123, ksat=10.4522,
    k0=1.0203, lpar=0.2242, thr_std=0.0136, ths_std=0.0150, alpha_std=0.0012,
    npar_std=0.1176, ksat_std=4.8160, k0_std=0.4968, lpar_std=1.4017),
    RosettaResult(sand=20.0000, silt=60.0000, clay=20.0000, rhob=None,
    th33=None, th1500=None, version=3, estimate_type='linear', code=2,
    thr=0.0899, ths=0.4301, alpha=0.0038, npar=1.4993, ksat=15.8995, k0=0.8909,
    lpar=0.1726, thr_std=0.0067, ths_std=0.0088, alpha_std=0.0007,
    npar_std=0.0454, ksat_std=3.1789, k0_std=0.4490, lpar_std=1.6119),
    RosettaResult(sand=55.0000, silt=25.0000, clay=20.0000, rhob=1.1000,
    th33=None, th1500=None, version=3, estimate_type='linear', code=3,
    thr=0.0913, ths=0.4850, alpha=0.0097, npar=1.4172, ksat=84.7834, k0=2.9015,
    lpar=-0.3463, thr_std=0.0128, ths_std=0.0131, alpha_std=0.0022,
    npar_std=0.0577, ksat_std=27.0357, k0_std=1.8203, lpar_std=1.2248)])


    To get the results as a list of dictionaries, use `.asdicts()`

    >>> print(result.asdicts())
    
    [{'sand': 30.0, 'silt': 30.0, 'clay': 40.0, 'rhob': 1.5, 'th33': 0.3,
    'th1500': 0.1, 'version': 3, 'estimate_type': 'linear', 'code': 5, 'thr':
    0.06872133198419333, 'ths': 0.3839050853475144, 'alpha':
    0.0036914540698334396, 'npar': 1.5122960137476906, 'ksat':
    10.45220672557941, 'k0': 1.020283781688175, 'lpar': 0.22415802649553881,
    'thr_std': 0.013635805076704855, 'ths_std': 0.014976665455951114,
    'alpha_std': 0.0012009965476429596, 'npar_std': 0.11756697611413282,
    'ksat_std': 4.816027639984864, 'k0_std': 0.4967836368490979, 'lpar_std':
    1.4016987918707104}, {'sand': 20.0, 'silt': 60.0, 'clay': 20.0, 'rhob':
    None, 'th33': None, 'th1500': None, 'version': 3, 'estimate_type':
    'linear', 'code': 2, 'thr': 0.08994502219206939, 'ths': 0.4301366480210401,
    'alpha': 0.0038032951212636345, 'npar': 1.4992978643076722, 'ksat':
    15.899549552218094, 'k0': 0.8908630387641998, 'lpar': 0.17264425930013833,
    'thr_std': 0.006710949117601514, 'ths_std': 0.008790220586404331,
    'alpha_std': 0.0006666891009750212, 'npar_std': 0.04539463666512193,
    'ksat_std': 3.178862086739814, 'k0_std': 0.4489771681065423, 'lpar_std':
    1.6118830509323745}, {'sand': 55.0, 'silt': 25.0, 'clay': 20.0, 'rhob':
    1.1, 'th33': None, 'th1500': None, 'version': 3, 'estimate_type': 'linear',
    'code': 3, 'thr': 0.09130753033144609, 'ths': 0.485031958049669, 'alpha':
    0.009748718589139088, 'npar': 1.4171964842691995, 'ksat': 84.7833726853498,
    'k0': 2.9015059553301388, 'lpar': -0.3463378623447919, 'thr_std':
    0.012777797583517441, 'ths_std': 0.013068706563916467, 'alpha_std':
    0.0022148332082555515, 'npar_std': 0.05767077407189314, 'ksat_std':
    27.035739850886362, 'k0_std': 1.8202787951521058, 'lpar_std':
    1.2248193041204392}]


Alternative usage
-----------------

Predictions can also be made using the Rosetta and UnsaturatedK classes.
These classes return the raw bootstrap ensembles of the predicted parameters,
which may be of interest to researchers who want to compute alternative
summary statistics or who want to use the resamples for other purposes.

::

    >>> from rosetta import Rosetta

The Rosetta class is instantiated for a particular Rosetta version and model,

::

    >>> rose33 = Rosetta(rosetta_version=3, model_code=3)

The ``.predict()`` method returns a 2-tuple of numpy arrays. The first 
has shape (nboot, nsample, nouput=4), where the 4 outputs are: [theta_r,
theta_s, log10(alpha), log10(npar)]. The second array has shape
(nboot, nsample, nouput=1), where the 1 ouput is [log10(Ksat)].

Note that whereas ``rosetta()`` and ``rosesoil()`` allow data elements
of differing lengths, ``.predict()`` requires that data be compatible
with a full 2D array that has the correct number of columns for the
instantiated model (ncols = data.shape[1] = model_code + 1).

::

    >>> from rosetta import Rosetta
    >>> rose33 = Rosetta(rosetta_version=3, model_code=3)
    >>> data =[
    ...     # sa, si, cl, bd
    ...     [30, 30, 40, 1.5],
    ...     [55, 25, 20, 1.1]
    ... ]
    >>> retc_boot, ksat_boot = rose33.predict(data)
    >>> print(retc_boot.shape)
    (1000, 2, 4)
    >>> print(ksat_boot.shape)
    (1000, 2, 1)

Statistics for the bootstrap resamples are computed along the zero axis,
e.g.

::

    >>> import numpy as np
    >>> print(np.mean(retc_boot, axis=0))

    [[ 0.11535773  0.41791199 -2.06713851  0.11201021]
    [ 0.09130753  0.48503196 -2.02238809  0.15107161]]

The UnsaturatedK class is used to predict K0 and L from the retention
parameters theta_r, theta_s, alpha, and npar. The input retention
parameters must be linear values, not log10 transformed. The
``.predict()`` method returns a 3D array with shape
(nboot, nsamples, nouput=2), where the two outputs are [log10(K0), Lpar].

::

    >>> from rosetta import UnsaturatedK
    >>> unsatk = UnsaturatedK()
    >>> retc_params = [
    ...     # theta_r, theta_s, alpha, npar
    ...     [0.12, 0.42, 0.008, 1.29],
    ...     [0.09, 0.49, 0.009, 1.41]
    ... ]
    >>> unsat_boot = unsatk.predict(retc_params)
    >>> print(unsat_boot.shape)
    (100, 2, 2)
    >>> print(np.mean(unsat_boot, axis=0))
    [[-0.04057941, -1.03302704],
    [ 0.35788353, -0.25048901]]


Acknowledgments
===============

The values for the neural network weights and biases were obtained from
`Marcel Schaap`_ and `Yonggen Zhang`_ at the University of Arizona.

.. _Marcel Schaap: https://envs.arizona.edu/person/marcel-g-schaap
.. _Yonggen Zhang: https://scholar.google.com/citations?user=u46LEeQAAAAJ&hl=en

