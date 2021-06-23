This package provides an implementation of **Rosetta**, a neural
network-based model for predicting unasturated soil hydraulic parameters
from basic soil characterization data.

How to avoid installing this package
====================================

For most Rosetta use cases, we recommend using the web browser interface
to ``rosetta-soil`` that is available at
`<https://www.handbook60.org/rosetta>`_

There is also an api available at ``handbook60.org``.  For example::

    import requests

    data = { 
        "soildata": [
            [30, 30, 40, 1.5, 0.3, 0.1],  
            [20, 60, 20],
            [55, 25, 20, 1.1],
        ]
    }

    def url(rosetta_version: int) -> str:
        return f"http://www.handbook60.org/api/v1/rosetta/{rosetta_version}"
    r = requests.post(url(3), json=data)

returns the following::

    print(r.json())

    {'model_codes': [5, 2, 3], 'rosetta_version': 3, 'stdev': 
    [[0.013628985468838103, 0.01496917525020338, 0.12948704319399928, 
    0.0347739236276485, 0.1747797749074611], [0.0067075928037543765, 
    0.00878582437678383, 0.07413139323912403, 0.013230683219936165, 
    0.08709445948355408], [0.01277140708670187, 0.013062170576228887, 
    0.10020312250396954, 0.01763982447621485, 0.14163566888667592]], 
    'van_genuchten_params': [[0.06872133198419336, 0.38390508534751433, 
    -2.452968871563431, 0.17827394547955497, 0.9827227259550619], 
    [0.08994502219206943, 0.4301366480210401, -2.4262357492034043, 
    0.17568732926631986, 1.192731130984082], [0.09130753033144606, 
    0.485031958049669, -2.0223880878467875, 0.151071612216524, 
    1.9060147751706147]]}

See below for information on the expected structure and content of the
submitted ``soildata`` and returned json payload.

[If your use case involves, e.g., thousands of repeated requests, then
please install and use ``rosetta-soil`` locally rather than use the api.]
 
Installation
============
::

    pip install rosetta-soil

Quickstart
==========
::

    >>> from rosetta import rosetta, SoilData

    >>> data = [
            [30,30,40,1.5,0.3,0.1],  
            [20,60,20],
            [55,25,20,1.1]
        ]

    >>> mean, stdev, codes = rosetta(3, SoilData.from_array(data))

    >>> print(mean)
    [[ 0.06872133  0.38390509 -2.45296887  0.17827395  0.98272273]
    [ 0.08994502  0.43013665 -2.42623575  0.17568733  1.19273113]
    [ 0.09130753  0.48503196 -2.02238809  0.15107161  1.90601478]]

    >>> print(stdev)
    [[ 0.01362899 0.01496918 0.12948704 0.03477392 0.17477977]
    [ 0.00670759 0.00878582 0.07413139 0.01323068 0.08709446]
    [ 0.01277141 0.01306217 0.10020312 0.01763982 0.14163567]]

    >>> print(codes)
    [5 2 3]


Background
==========

The Rosetta pedotransfer function predicts five parameters for the van
Genuchten model of unsaturated soil hydraulic properties

* theta_r      : residual volumetric water content
* theta_s      : saturated volumetric water content
* log10(alpha) : retention shape parameter [log10(1/cm)]
* log10(n)     : retention shape parameter
* log10(ksat)  : saturated hydraulic conductivity [log10(cm/d)]

Rosetta provides four models for predicting the five parameters from soil
characterization data. The models differ in the required input data

+------------+------------------------+
| Model Code | Input Data             |
+============+========================+
|      2     | sa, si, cl (SSC) |
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

Three versions of Rosetta are available. The versions effectively represent
three alternative calibrations of the four Rosetta models. 
The references that should be cited when using Rosetta versions 1, 2,
and 3 are, respectively:

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


Usage
=====
::

    from rosetta import rosetta, SoilData

The imported function ``rosetta`` predicts soil hydraulic parameters from
soil characterization data. It has two required arguments::

    rosetta_version : int, {1, 2, 3}
    soildata : SoilData

The second argument is a ``SoilData`` instance. Normally, the instance is
created from an array-like collection of soil characterization data
using the ``from_array`` method.
::

    data = [
        [30,30,40,1.5,0.3,0.1],  
        [20,60,20],
        [55,25,20,1.1]
    ]
    soildata = SoilData.from_array(data)

Each element of the array-like data contains soil data in this order::

    [%sand, %silt, %clay, buld density, th33, th1500]

Sand, silt, and clay are required; the others are optional. For each
entry, ``rosetta`` selects the best availabe Rosetta model based on
the given data.  Note that even if you are predicting for only a single
soil record, ``data`` still needs to 2D array-like::

    data = [[30,30,40]]
    soildata = SoilData.from_array(data)

The function ``rosetta`` returns a 3-tuple
::

   mean, stdev, codes = rosetta(3, soildata)

``mean`` is a 2D numpy array. The ith row holds predicted soil hydraulic
parameters for ith entry in ``soildata``. The array columns are

+-------+---------------------------------------------------------------+
|Column | Parameter                                                     |
+=======+===============================================================+
|   0   | theta_r, residual water content                               |
+-------+---------------------------------------------------------------+
|   1   | theta_s, saturated water content                              |
+-------+---------------------------------------------------------------+
|   2   | log10(alpha), 'alpha' shape parameter, log10(1/cm)            | 
+-------+---------------------------------------------------------------+
|   3   | log10(npar), 'n' shape parameter                              |
+-------+---------------------------------------------------------------+
|   4   | log10(Ksat), saturated hydraulic conductivity, log10(cm/day)  |
+-------+---------------------------------------------------------------+

``stdev`` is 2D numpy array holding the corresponding parameter standard
deviations.

``codes`` is a 1D numpy array with the ith entry indicating the
Rosetta model and input data used to predict the ith row of ``mean``
and ``stdev``.

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

Alternative usage
-----------------

Predictions can also be made using the Rostta class
::

    import numpy as np
    from rosetta import Rosetta

The class is instantiated for a particular Rosetta version and model.
Predictions are then made using a numpy array of soil data.
::

    rose33 = Rosetta(rosetta_version=3, model_code=3)
    data = np.array([[30,30,40,1.5],[55,25,20,1.1]], dtype=float)
    mean, stdev = rose33.predict(data)

The 2D numpy array ``data`` has to be ``data.shape[1] = model_code + 1``.
Compared with the function rosetta.rosetta, Rosetta.predict offers
fewer checks on arguments and data.


Notes
=====

This module wraps files taken from
`research code <http://www.u.arizona.edu/~ygzhang/download.html>`_
developed by Marcel Schaap and Yonggen Zhang at the University of
Arizona. 

The Rosetta class described above has another method,
Rosetta.ann_predict, which returns additional statistical quantities
computed by the Schaap and Zhang code and which may be of interest to
researchers. The usage is the same as Rosetta.predict,
::

    rose33 = Rosetta(rosetta_version=3, model_code=3)
    data = np.array([[30,30,40,1.5],[55,25,20,1.1]], dtype=float)
    results = rose33.ann_predict(data, sum_data=True)

However, in this case, the returned ``results`` is a dictionay of parameters
and statistical results. Note the arrays in ``results`` are the transpose 
of what is returned by other functions and methods in ``rosetta-soil``
See the file ``ANN_Module.py`` and the code base of 
`Schaap and Zhang <http://www.u.arizona.edu/~ygzhang/download.html>`_
for more information.
