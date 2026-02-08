from collections.abc import Sequence, Iterable
from dataclasses import dataclass
import importlib.resources
from typing import NamedTuple, Self, TypeAlias

import numpy as np

from . import ANN_Module
from . import DB_Module

Array1D: TypeAlias = np.ndarray
Array2D: TypeAlias = np.ndarray

ROSETTA_VERSIONS = {1, 2, 3}
ROSETTA_MODEL_CODES = {2, 3, 4, 5}
ROSE2_COEFFICIENTS = {
    2: [(0.969, 0.000), (1.103, -0.048), (1.050, 0.159), (0.655, 0.004)],
    3: [(0.995, 0.000), (1.000, 0.002), (1.028, 0.162), (0.660, 0.003)],
    4: [(1.156, 0.000), (0.974, 0.011), (1.039, 0.067), (0.796, 0.006)],
    5: [(1.220, 0.000), (1.045, -0.020), (0.956, -0.040), (0.877, -0.003)],
}


class RosettaError(Exception):
    def __init__(self, message):
        self.message = message


def to_float(any) -> float:
    try:
        return float(any)
    except (ValueError, TypeError):
        return np.nan


def to_floats(seq: Sequence) -> list[float]:
    floats = []
    for s in seq:
        try:
            floats.append(float(s))
        except (ValueError, TypeError):
            floats.append(np.nan)
    return floats


class SoilDatum(NamedTuple):
    sand: float = np.nan
    silt: float = np.nan
    clay: float = np.nan
    rhob: float | None = None
    th33: float | None = None
    th1500: float | None = None

    def is_valid_separates(self) -> bool:
        return (
            99.0 <= sum([self.sand, self.silt, self.clay]) <= 101.0
            and self.sand >= 0.0
            and self.silt >= 0.0
            and self.clay >= 0.0
        )

    def is_valid_bulkdensity(self) -> bool:
        return False if self.rhob is None else 0.5 <= self.rhob <= 2.0

    def is_valid_fieldcap(self) -> bool:
        return False if self.th33 is None else 0.0 < self.th33 < 1.0

    def is_valid_wiltpoint(self) -> bool:
        return False if self.th1500 is None else 0.0 < self.th1500 < 1.0

    def is_valid_fieldcap_and_wiltpoint(self) -> bool:
        if self.th33 is None or self.th1500 is None:
            return False
        return 0.0 < self.th1500 < self.th33 < 1.0

    def is_valid(self, index: int | None = None) -> bool:
        """Test validity of data in self[:index].

        Requirements for valid data:
            99 <= sand + silt + clay <= 101
            0.5 <= rho_b <= 2
            0 < th1500 < th33 < 1

        """
        if index is None:
            index = len(self)
        if not (3 <= index <= len(self)):
            return False
        tests = [
            self.is_valid_separates(),
            self.is_valid_bulkdensity(),
            self.is_valid_fieldcap(),
            self.is_valid_fieldcap_and_wiltpoint(),
        ]
        return all(tests[: index - 2])

    def best_index(self) -> int:
        """Max value of index for which self.is_valid(index) is True."""
        best = 0
        for index in range(3, len(self) + 1):
            if self.is_valid(index):
                best = index
            else:
                break
        return best


@dataclass
class SoilData:
    data: list[SoilDatum]

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, indx):
        return self.data[indx]

    def __len__(self):
        return len(self.data)

    @classmethod
    def from_array(cls, array: Iterable[Sequence[float]]) -> Self:
        return cls([SoilDatum(*to_floats(row)) for row in array])

    @classmethod
    def from_dicts(cls, dicts: Sequence[dict[str, float]]) -> Self:
        keys = ["sand", "silt", "clay", "rhob", "th33", "th1500"]
        dicts = [{k: to_float(d[k]) for k in keys if k in d} for d in dicts]
        return cls([SoilDatum(**row) for row in dicts])

    def to_array(self) -> Array2D:
        return np.array([list(datum) for datum in self], dtype=float)

    def to_dicts(self) -> list[dict]:
        return [datum._asdict() for datum in self.data]

    def is_valid(self, index: int | None = None) -> list[bool]:
        # ith element of returned list is validity of self[i][SoilDatum[:index]]
        return [datum.is_valid(index) for datum in self]


class Rosetta:
    def __init__(self, rosetta_version: int, model_code: int) -> None:
        if rosetta_version not in ROSETTA_VERSIONS:
            raise RosettaError(f"rosetta_version must be in {ROSETTA_VERSIONS}")
        if model_code not in ROSETTA_MODEL_CODES:
            raise RosettaError(f"model_code must be in {ROSETTA_MODEL_CODES}")
        self.rosetta_version = rosetta_version
        self.model_code = model_code
        code = model_code if rosetta_version == 3 else model_code + 100
        data_dir = importlib.resources.files("rosetta").joinpath("data")
        rosetta_db_path = data_dir / "Rosetta.sqlite"
        with DB_Module.DB(0, 0, 0, sqlite_path=rosetta_db_path) as db:
            self.ptf_model = ANN_Module.PTF_MODEL(code, db)

    def predict(self, X: Array2D) -> tuple[Array2D, Array2D]:
        predicted_params = self.ptf_model.predict(X.T, sum_data=True)
        mean = predicted_params["sum_res_mean"].T
        stdev = predicted_params["sum_res_std"].T

        if self.rosetta_version == 2:
            for jcol, (slope, offset) in enumerate(ROSE2_COEFFICIENTS[self.model_code]):
                mean[:, jcol] = slope * mean[:, jcol] + offset
                stdev[:, jcol] *= slope

        return mean, stdev

    def ann_predict(self, X: Array2D, sum_data: bool = True) -> dict:
        if self.rosetta_version == 2:
            raise RosettaError("ann_predict not enabled for Rosetta version 2")
        return self.ptf_model.predict(X.T, sum_data=sum_data)


def rosetta(
    rosetta_version: int, soildata: SoilData
) -> tuple[Array2D, Array2D, Array1D]:
    """Predict soil hydraulic parameters from soil characterization data.

    Parameters
    ----------
    rosetta_version : {1, 2, 3}

    soildata : :class:`~rosetta.rosetta.SoilData`
        List of soil characterization data, one entry per soil

    Returns
    -------
    (mean, stdev, codes)

    mean : 2D np.array, dtype=float

        ith row holds predicted soil hydraulic parameters for ith entry
        in `soildata`.

        array  |
        column | parameter
        -----------------
           0   | theta_r, residual water content
           1   | theta_s, saturated water content
           2   | log10(alpha), van Genuchten 'alpha' parameter (1/cm)
           3   | log10(npar), van Genuchten 'n' parameter
           4   | log10(Ksat), saturated hydraulic conductivity (cm/day)

    stdev: 2D np.array, dtype=float

        Standard deviations for parameters in `mean`.

    codes : 1D np.array, dtype=int

        ith entry is a code indicating the neural network model and
        input data used to predict the ith row of `mean` and `stdev`.

         code | data used
        ---------------
           2  | sand, silt, clay (SSC)
           3  | SSC + bulk density (BD)
           4  | SSC + BD + field capacity water content (TH33)
           5  | SSC + BD + TH33 + wilting point water content (TH1500))
          -1  | no result returned, insufficient or erroneous data

    Example
    -------
    >>> from rosetta import rosetta, SoilData

    # required ordering for data records:
    # [sa (%), si (%), cl (%), bd (g/cm3), th33, th1500]

    # sa, si, and cl are required; others optional

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

    """
    if rosetta_version not in ROSETTA_VERSIONS:
        raise RosettaError(f"rosetta_version must be in {ROSETTA_VERSIONS}")
    if not isinstance(soildata, SoilData):
        raise RosettaError("soildata must be a SoilData instance")

    # codes[i] is -1 if soildata[i] lacks minimun required data
    codes = np.array([data.best_index() - 1 for data in soildata], dtype=int)
    features = soildata.to_array()
    mean = np.full((features.shape[0], 5), np.nan, dtype=float)
    stdev = np.full((features.shape[0], 5), np.nan, dtype=float)

    for code in set(codes) - {-1}:
        rose = Rosetta(rosetta_version, code)
        rows = codes == code
        mean[rows], stdev[rows] = rose.predict(features[rows, : code + 1])

    return mean, stdev, codes


def rosesoil(rosetta_version: int, soildata: SoilData) -> list[dict]:
    """Predict soil hydraulic parameters from soil characterization data.

    Parameters
    ----------
    rosetta_version : {1, 2, 3}

    soildata : :class:`~rosetta.rosetta.SoilData`
        List of soil characterization data, one entry per soil

    Returns
    -------
    [ 
      {
        "sand": float,        # sand %
        "silt": float,        # silt %
        "clay": float,        # clay %
        "rhob": float,        # soil bulk density (g/cm3)
        "th33": float,        # third bar soil water content
        "th1500": float,      # 15 bar soil water content
        "code": int,          # rosetta model code
        "thr": float,         # residual water content
        "ths": float,         # saturatued water content
        "alp": float,         # log10 alpha parameter (1/cm)
        "npar": float,        # log10 n parameter
        "ksat": float,        # log10 saturated conductivity (cm/d)
        "thr-stdev": float,   # standard dev of residual water content
        "ths-stdev": float,   # standard dev of saturatued water content
        "alp-stdev": float,   # standard dev of log10 alpha parameter
        "npar-stdev": float,  # standard dev of log10 n parameter
        "ksat-stdev": float,  # standard dev of log10 saturated conductivity
        "version": int,       # Rosetta version
      }, ...
    ]

    rosetta model codes :

         code | data used
        ---------------
           2  | sand, silt, clay (SSC)
           3  | SSC + bulk density (BD)
           4  | SSC + BD + field capacity water content (TH33)
           5  | SSC + BD + TH33 + wilting point water content (TH1500))
          -1  | no result returned, insufficient or erroneous data

    Example
    -------
    >>> from rosetta import rosetta, SoilData

    # required ordering for data records:
    # [sa (%), si (%), cl (%), bd (g/cm3), th33, th1500]

    # sa, si, and cl are required; others optional

    >>> data = [
            [30,30,40,1.5,0.3,0.1],
            [20,60,20],
            [55,25,20,1.1]
        ]

    >>> result = rosesoil(3, SoilData.from_array(data))

    >>> print(result)
    [ {
        "sand": 30, "silt": 30, "clay": 40, "rhob": 1.5, "th33": 0.3,
        "th1500" 0.1, "code": 5, "thr": 0.06872133, "ths": 0.38390509,
        "alp": -2.45296887, "npar": 0.17827395, "ksat": 0.98272273,
        "thr-stdev": 0.01362899, "ths-stdev": 0.01496918,
        "alp-stdev": 0.12948704, "npar-stdev": 0.03477392,
        "ksat-stdev": 0.17477977, "version": 3
      },
      {
        "sand": 20, "silt": 60, "clay": 20, "rhob": nan, "th33": nan,
        "th1500" nan, "code": 2, "thr": 0.08994502, "ths": 0.43013665,
        "alp": -2.42623575, "npar": 0.17568733, "ksat": 1.19273113,
        "thr-stdev": 0.00670759, "ths-stdev": 0.00878582,
        "alp-stdev": 0.07413139, "npar-stdev": 0.01323068,
        "ksat-stdev": 0.08709446, "version": 3
      },
      {
        "sand": 55, "silt": 25, "clay": 20, "rhob": 1.1, "th33": nan,
        "th1500" nan, "code": 3, "thr": 0.09130753, "ths": 0.48503196,
        "alp": -2.02238809, "npar": 0.15107161, "ksat": 1.90601478,
        "thr-stdev": 0.01277141, "ths-stdev": 0.01306217,
        "alp-stdev": 0.10020312, "npar-stdev": 0.01763982,
        "ksat-stdev": 0.14163567, "version": 3
      },
    ]

    """
    mean, stdev, codes = rosetta(rosetta_version, soildata)

    vg_keys = ["thr", "ths", "alp", "npar", "ksat"]
    std_keys = [s + "-stdev" for s in vg_keys]
    vang_params = [dict(zip(vg_keys, vals.tolist())) for vals in mean]
    vang_stds = [dict(zip(std_keys, vals.tolist())) for vals in stdev]
    codes=[{"code": c, "version": rosetta_version} for c in codes.tolist()]
    data = soildata.to_dicts()

    results = [
        {**d, **c, **p, **s} for d, c, p, s in zip(data, codes, vang_params, vang_stds)
    ]

    return results

