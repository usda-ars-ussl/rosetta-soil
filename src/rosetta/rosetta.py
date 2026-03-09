from collections.abc import Sequence, Iterable
from dataclasses import dataclass
from numbers import Number
from typing import NamedTuple, Self, TypeAlias, Literal

import numpy as np
from numpy.typing import ArrayLike

from .nn_model import NNModel

Array1D: TypeAlias = np.ndarray
Array2D: TypeAlias = np.ndarray
Array3D: TypeAlias = np.ndarray

ROSETTA_VERSIONS = [1, 2, 3]
ROSETTA_MODEL_CODES = [2, 3, 4, 5]
ROSE2_COEFFICIENTS = {
    2: [(0.969, 0.000), (1.103, -0.048), (1.050, 0.159), (0.655, 0.004)],
    3: [(0.995, 0.000), (1.000, 0.002), (1.028, 0.162), (0.660, 0.003)],
    4: [(1.156, 0.000), (0.974, 0.011), (1.039, 0.067), (0.796, 0.006)],
    5: [(1.220, 0.000), (1.045, -0.020), (0.956, -0.040), (0.877, -0.003)],
}


class RosettaError(Exception):
    def __init__(self, message):
        self.message = message


def _validate_rosetta_input(rosetta_version: int, model_code: int | None = None):
    if rosetta_version not in ROSETTA_VERSIONS:
        raise RosettaError(f"{rosetta_version=} must be one of {ROSETTA_VERSIONS}")
    if model_code is not None and model_code not in ROSETTA_MODEL_CODES:
        raise RosettaError(f"{model_code=} must be one of {ROSETTA_MODEL_CODES}")

def _as_2d_array(input: ArrayLike, expected_ninput: int, src: str) -> Array2D:
    """
    Convert 1D or 2D array-like input to 2D np.ndarray and verify result
    has `expected_ninput` columns.
    """
    input = np.asarray(input, dtype=np.float64)
    init_shape = input.shape
    if input.ndim == 1:
        input = input[np.newaxis, :]
    if input.ndim != 2 or input.shape[1] != expected_ninput:
        raise RosettaError(
            f"Incorrectly shaped input. Input was shaped {init_shape}. "
            f"Input to {src} must be shaped (nsample, {expected_ninput}), "
            f"or alternatively ({expected_ninput},) if nsample = 1.)"
        )
    return input

def _to_float(any) -> float:
    try:
        return float(any)
    except (ValueError, TypeError):
        return np.nan


def _to_floats(input: Iterable) -> list[float]:
    floats = []
    for x in input:
        try:
            floats.append(float(x))
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

    def __repr__(self):
        return (
            f"SoilDatum(sand={float(self.sand)}, silt={float(self.silt)}, "
            f"clay={float(self.clay)}, "
            f"rhob={np.nan if not self.rhob else float(self.rhob)}, "
            f"th33={np.nan if not self.th33 else float(self.th33)}, "
            f"th1500={np.nan if not self.th1500 else float(self.th1500)})"
        )

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

        Note this tests the validity of values in the slice :index, not
        the values for self[index] and below. E.g., .is_valid(index=3)
        tests the values at indices 0, 1, and 2, not 0, 1, 2, and 3.

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
    def from_iter(cls, input: Iterable[Iterable[Number]] | Iterable[Number]) -> Self:
        try:
            first = next(iter(input))
        except StopIteration:
            # Empty input
            rows = []
        else:
            if isinstance(first, Iterable) and not isinstance(first, (str, bytes)):
                rows = input
            else:
                rows = [input]
        return cls([SoilDatum(*_to_floats(row)) for row in rows])

    @classmethod
    def from_dicts(cls, dicts: Sequence[dict[str, float]]) -> Self:
        keys = ["sand", "silt", "clay", "rhob", "th33", "th1500"]
        dicts = [{k: _to_float(d[k]) for k in keys if k in d} for d in dicts]
        return cls([SoilDatum(**row) for row in dicts])

    def to_array(self) -> Array2D:
        return np.array([list(datum) for datum in self], dtype=float)

    def to_dicts(self) -> list[dict]:
        return [datum._asdict() for datum in self.data]

    def is_valid(self, index: int | None = None) -> list[bool]:
        # ith element of returned list is validity of self[i][SoilDatum[:index]]
        return [datum.is_valid(index) for datum in self]


class UnsaturatedK:
    """
    Predict unsaturated hydraulic conductivity parameters K0 and L from
    van Genuchten retention parameters. Predictions are returned for the
    full ensemble of bootstrap resamples.
    """

    def __init__(self, validate_network: bool = False) -> None:
        self.model = NNModel.from_npz_resource(
            "rose1_unsat_k", validate_network=validate_network
        )
        self.model.validate_input = self._validate_input

    @staticmethod
    def _validate_input(p):
        thr, ths, alp, npar = p
        return (0 <= thr < ths < 1) and alp > 0 and npar > 1

    def predict(self, input: ArrayLike) -> Array3D:
        """
        Parameters
        ----------
        inputs: ArrayLike with shape of (4,) or (nsample, 4)
            The 4 inputs are the VG retention parameters (in this order):
            [theta_r, theta_s, alpha, npar]. These need to be linear
            (not log tranformed) parameter values.

        Returns
        -------
        np.ndarray: shape is (nboot, nsample, noutput=2), ouput is [log10(K0), Lpar]

        """
        name = UnsaturatedK.predict.__qualname__
        input = _as_2d_array(input, expected_ninput=4, src=name)
        return self.model.predict(input)


class Rosetta:
    """
    Predict van Genuchten parameters and saturated conductivity from
    basic soil characterization data. Predictions are returned for the
    full ensemble of bootstrap resamples.
    """

    def __init__(
        self, rosetta_version: int, model_code: int, validate_network: bool = False
    ) -> None:
        _validate_rosetta_input(rosetta_version, model_code)

        if rosetta_version == 3:
            nn_name = f"rose3_mod{model_code}"
        else:
            nn_name = f"rose1_mod{model_code}"
        model0 = NNModel.from_npz_resource(
            nn_name + "_0", validate_network=validate_network
        )

        # Rosetta 1 and 2 have separate models for VG params and Ksat.
        # Rosetta 3 has one model for both.
        if rosetta_version == 3:
            model1 = None
        else:
            model1 = NNModel.from_npz_resource(
                nn_name + "_1", validate_network=validate_network
            )

        self.rosetta_version = rosetta_version
        self.model_code = model_code
        self.model0 = model0
        self.model1 = model1

    @property
    def _ninput(self) -> int:
        return self.model_code + 1

    def predict(self, input: np.ndarray) -> tuple[Array2D, Array2D]:
        """
        Parameters
        ----------
        input: ArrayLike with shape of (ninput,) or (nsample, ninput)

        Returns
        -------
        (np.ndarray, np.ndarray)

        First array is bootstrap resamples of VG retention parameters.
        Shape is (nboot, nsample, nouput=4),
        where the 4 outputs are [theta_r, theta_s, log10(alpha), log10(npar)

        Second array is bootstrap resamples of saturated conductivity.
        Shape is (nboot, nsample, nouput=1),
        where the 1 ouput is [log10(Ksat)]

        Note nboot is not necessarily the same for the two arrays.
        """
        name = Rosetta.predict.__qualname__
        input = _as_2d_array(input, expected_ninput=self._ninput, src=name)

        out0_boot = self.model0.predict(input)
        if self.rosetta_version == 3:
            retc_boot = out0_boot[:, :, :4]
            ksat_boot = out0_boot[:, :, 4:5]
        else:
            retc_boot = out0_boot
            ksat_boot = self.model1.predict(input)

        if self.rosetta_version == 2:
            for idx, (slope, offset) in enumerate(ROSE2_COEFFICIENTS[self.model_code]):
                retc_boot[:, :, idx] = slope * retc_boot[:, :, idx] + offset

        return retc_boot, ksat_boot


def rosetta(
    rosetta_version: int,
    soildata: SoilData | Iterable[Iterable[Number]] | Iterable[Number],
    estimate_type: Literal["arith", "log", "geo"] = "arith",
) -> tuple[Array2D, Array2D, Array1D]:
    """Predict soil hydraulic parameters from soil characterization data.

    For each soil datum, the "best-available" prediction model is applied.

    Parameters
    ----------
    rosetta_version : {1, 2, 3}

    soildata : SoilData | Iterable[Iterable] | Iterable
        List of soil characterization data, one entry per soil

    estimate_type: Literal["arith", "log", "geo"]
        Specifies the parameter estimate to be returned from the bootsrap
        ensemble of parameter values for alpha, npar, Ksat, and K0.
        'arith' (default):
            Estimate is aritmetic mean.
        'log':
            Estimate is mean of log10 transformed parameter value.
        'geo':
            Estimate is geometric mean.
        Esimates of theta_r, theta_s, and Lpar are always arithmetic
        means regardless of `estimate_type`.

    Returns
    -------
    (mean, stdev, codes)

    mean: 2D np.array, dtype=float, shape is (nsample, noutput=7)

        ith row holds predicted soil hydraulic parameters for ith sample
        in `soildata`. The type of estimate (arithmetic mean, geometric
        mean, mean of log10 values) in cols 2 through 5 depends on the
        `estimate_type` as indicated above.

        column | parameter
        -----------------
           0   | theta_r, residual water content
           1   | theta_s, saturated water content
           2   | alpha (or log10), van Genuchten 'alpha' parameter (1/cm)
           3   | npar (or log10), van Genuchten 'n' parameter
           4   | Ksat (or log10), saturated hydraulic conductivity (cm/day)
           5   | K0 (or log10), unsaturated conductivity match point (cm/day)
           6   | Lpar, unsaturated conductivity exponent

    stdev: 2D np.array, dtype=float

        Standard deviations corresponding to the parameters in `mean`.
        Note these are the standard deviations of the bootstrap
        resamples, not the standard error of the mean estimates.

    codes : 1D np.array, dtype=int

        ith entry is a code indicating the "best-available" neural
        network model and input data that were used to predict the ith
        row of `mean` and `stdev`.

         code | data used
        ---------------
           2  | sand, silt, clay (SSC)
           3  | SSC + bulk density (BD)
           4  | SSC + BD + field capacity water content (TH33)
           5  | SSC + BD + TH33 + wilting point water content (TH1500))
          -1  | no result returned, insufficient or erroneous data

    Example
    -------
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

    """
    _validate_rosetta_input(rosetta_version)
    if not isinstance(soildata, SoilData):
        soildata = SoilData.from_iter(soildata)
    unsatk = UnsaturatedK()

    # The integer codes 2 thru 5 indicate the best available Rosetta
    # model code. The model code is equal to the value returned by
    # SoilData.best_index() minus one. best_index() returns 0 if no
    # minimum viable data so the code is -1 in that case.
    codes = np.array([data.best_index() - 1 for data in soildata], dtype=int)

    inputs = soildata.to_array()
    nsample = inputs.shape[0]
    mean = np.full((nsample, 7), np.nan, dtype=float)
    stdev = np.full((nsample, 7), np.nan, dtype=float)

    for code in ROSETTA_MODEL_CODES:
        if code not in codes:
            continue
        rose = Rosetta(rosetta_version, code)
        rowmask = codes == code
        retc_boot, ksat_boot = rose.predict(inputs[rowmask, : code + 1])

        retc_linboot = retc_boot.copy()
        ksat_linboot = ksat_boot.copy()
        retc_linboot[:, :, 2:] = np.power(10, retc_linboot[:, :, 2:])
        ksat_linboot[:, :, 0] = np.power(10, ksat_linboot[:, :, 0])

        retc_arith_mean = np.mean(retc_linboot, axis=0)
        ksat_arith_mean = np.mean(ksat_linboot, axis=0)
        retc_arith_std = np.std(retc_linboot, axis=0, ddof=1)
        ksat_arith_std = np.std(ksat_linboot, axis=0, ddof=1)

        retc_log_mean = np.mean(retc_boot, axis=0)
        ksat_log_mean = np.mean(ksat_boot, axis=0)
        retc_log_std = np.std(retc_boot, axis=0, ddof=1)
        ksat_log_std = np.std(ksat_boot, axis=0, ddof=1)


        retc_geo_mean = retc_log_mean.copy()
        ksat_geo_mean = ksat_log_mean.copy()
        retc_geo_mean[:, 2:] = np.power(10, retc_geo_mean[:, 2:])
        ksat_geo_mean[:, 0] = np.power(10, ksat_geo_mean[:, 0])
        retc_geo_std = retc_arith_std
        ksat_geo_std = ksat_arith_std

        if estimate_type == "log":
            mean[rowmask, :4] = retc_log_mean
            mean[rowmask, 4:5] = ksat_log_mean
            stdev[rowmask, :4] = retc_log_std
            stdev[rowmask, 4:5] = ksat_log_std

            unsatboot = unsatk.predict(retc_geo_mean)
            mean[rowmask, 5:] = np.mean(unsatboot, axis=0)
            stdev[rowmask, 5:] = np.std(unsatboot, axis=0, ddof=1)

        elif estimate_type == "arith":
            mean[rowmask, :4] = retc_arith_mean
            mean[rowmask, 4:5] = ksat_arith_mean
            stdev[rowmask, :4] = retc_arith_std
            stdev[rowmask, 4:5] = ksat_arith_std

            unsatboot = unsatk.predict(retc_arith_mean)
            unsatboot[:, :, 0] = np.power(10, unsatboot[:, :, 0])
            mean[rowmask, 5:] = np.mean(unsatboot, axis=0)
            stdev[rowmask, 5:] = np.std(unsatboot, axis=0, ddof=1)

        elif estimate_type == "geo":
            mean[rowmask, :4] = retc_geo_mean
            mean[rowmask, 4:5] = ksat_geo_mean
            stdev[rowmask, :4] = retc_geo_std
            stdev[rowmask, 4:5] = ksat_geo_std

            unsatboot = unsatk.predict(retc_geo_mean)
            k0_linboot = np.power(10, unsatboot[:, :, 0])
            k0_arith_std = np.std(k0_linboot, axis=0, ddof=1)
            k0_geo_mean = np.power(10, np.mean(unsatboot[:, :, 0], axis=0))
            lpar_arith_mean = np.mean(unsatboot[:, :, 1], axis=0)
            lpar_arith_std = np.std(unsatboot[:, :, 1], axis=0, ddof=1)

            mean[rowmask, 5] = k0_geo_mean
            mean[rowmask, 6] = lpar_arith_mean
            stdev[rowmask, 5] = k0_arith_std
            stdev[rowmask, 6] = lpar_arith_std

    return mean, stdev, codes


@dataclass
class RosettaResult:
    sand: float
    silt: float
    clay: float
    rhob: float | None
    th33: float | None
    th1500: float | None
    version: int
    estimate_type: Literal["arith", "log", "geo"]
    code: int
    thr: float
    ths: float
    alpha: float
    npar: float
    ksat: float
    k0: float
    lpar: float
    thr_std: float
    ths_std: float
    alpha_std: float
    npar_std: float
    ksat_std: float
    k0_std: float
    lpar_std: float

    def __repr__(self):
        fields = []
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is None:
                val_str = "None"
            elif isinstance(value, float):
                val_str = f"{float(value):.4f}"
            elif isinstance(value, int):
                val_str = str(int(value))
            else:
                try:
                    val_str = f"{float(value):.4f}"
                except Exception:
                    val_str = repr(value)
            fields.append(f"{field}={val_str}")
        return f"RosettaResult({', '.join(fields)})"

    def asdict(self) -> dict:
        return self.__dict__

@dataclass
class RosettaResults:
    results: list[RosettaResult]

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, indx):
        return self.results[indx]

    def __len__(self):
        return len(self.results)

    def asdicts(self) -> list[dict]:
        return [res.asdict() for res in self.results]

    def __repr__(self):
        results_str = ', '.join([repr(res) for res in self.results])
        return f"RosettaResults([{results_str}])"


def rosesoil(
    rosetta_version: int,
    soildata: SoilData | Iterable[Iterable[Number]] | Iterable[Number],
    estimate_type: Literal["arith", "log", "geo"] = "arith",
) -> list[dict] | RosettaResults:
    """Predict soil hydraulic parameters from soil characterization data.

    For each soil datum, the "best-available" prediction model is applied.

    This function does the same thing as `rosetta()` but returns the
    results as RosettaResults.

    Parameters
    ----------
    rosetta_version : {1, 2, 3}

    soildata : SoilData | Iterable[Iterable] | Iterable
        List of soil characterization data, one entry per soil

    estimate_type: Literal["arith", "log", "geo"]
        Specifies the parameter estimate to be returned from the bootsrap
        ensemble of parameter values for alpha, npar, Ksat, and K0.
        'arith' (default):
            Estimate is aritmetic mean.
        'log':
            Estimate is mean of log10 transformed parameter value.
        'geo':
            Estimate is geometric mean.
        Esimates of theta_r, theta_s, and Lpar are always arithmetic
        means regardless of `estimate_type`.

    Returns
    -------
    RosettaResuts

    Notes
    -----
    The code returned in each RosettaResult indicates the neural network
    model and input data used to predict soil hydraulic parameters for
    that result:

         code | data used
        ---------------
           2  | sand, silt, clay (SSC)
           3  | SSC + bulk density (BD)
           4  | SSC + BD + field capacity water content (TH33)
           5  | SSC + BD + TH33 + wilting point water content (TH1500))
          -1  | no result returned, insufficient or erroneous data

    Example
    -------
    >>> from rosetta import rosesoil

    # required ordering for data records:
    # [sa (%), si (%), cl (%), bd (g/cm3), th33, th1500]

    # sa, si, and cl are required; others optional

    >>> data = [
            [30,30,40,1.5,0.3,0.1],
            [20,60,20],
            [55,25,20,1.1]
        ]

    >>> result = rosesoil(3, data))

    >>> print(result)

    RosettaResults([RosettaResult(sand=30.0000, silt=30.0000, clay=40.0000,
    rhob=1.5000, th33=0.3000, th1500=0.1000, version=3, estimate_type='arith',
    code=5, thr=0.0687, ths=0.3839, alpha=0.0037, npar=1.5123, ksat=10.4522,
    k0=1.0203, lpar=0.2242, thr_std=0.0136, ths_std=0.0150, alpha_std=0.0012,
    npar_std=0.1176, ksat_std=4.8160, k0_std=0.4968, lpar_std=1.4017),
    RosettaResult(sand=20.0000, silt=60.0000, clay=20.0000, rhob=None,
    th33=None, th1500=None, version=3, estimate_type='arith', code=2,
    thr=0.0899, ths=0.4301, alpha=0.0038, npar=1.4993, ksat=15.8995, k0=0.8909,
    lpar=0.1726, thr_std=0.0067, ths_std=0.0088, alpha_std=0.0007,
    npar_std=0.0454, ksat_std=3.1789, k0_std=0.4490, lpar_std=1.6119),
    RosettaResult(sand=55.0000, silt=25.0000, clay=20.0000, rhob=1.1000,
    th33=None, th1500=None, version=3, estimate_type='arith', code=3,
    thr=0.0913, ths=0.4850, alpha=0.0097, npar=1.4172, ksat=84.7834, k0=2.9015,
    lpar=-0.3463, thr_std=0.0128, ths_std=0.0131, alpha_std=0.0022,
    npar_std=0.0577, ksat_std=27.0357, k0_std=1.8203, lpar_std=1.2248)])


    To get the results as a list of dictionaries, use `.asdicts()`

    >>> print(result.asdicts())

    [{'sand': 30.0, 'silt': 30.0, 'clay': 40.0, 'rhob': 1.5, 'th33': 0.3,
    'th1500': 0.1, 'version': 3, 'estimate_type': 'arith', 'code': 5, 'thr':
    0.06872133198419333, 'ths': 0.3839050853475144, 'alpha':
    0.0036914540698334396, 'npar': 1.5122960137476906, 'ksat':
    10.45220672557941, 'k0': 1.020283781688175, 'lpar': 0.22415802649553881,
    'thr_std': 0.013635805076704855, 'ths_std': 0.014976665455951114,
    'alpha_std': 0.0012009965476429596, 'npar_std': 0.11756697611413282,
    'ksat_std': 4.816027639984864, 'k0_std': 0.4967836368490979, 'lpar_std':
    1.4016987918707104}, {'sand': 20.0, 'silt': 60.0, 'clay': 20.0, 'rhob':
    None, 'th33': None, 'th1500': None, 'version': 3, 'estimate_type':
    'arith', 'code': 2, 'thr': 0.08994502219206939, 'ths': 0.4301366480210401,
    'alpha': 0.0038032951212636345, 'npar': 1.4992978643076722, 'ksat':
    15.899549552218094, 'k0': 0.8908630387641998, 'lpar': 0.17264425930013833,
    'thr_std': 0.006710949117601514, 'ths_std': 0.008790220586404331,
    'alpha_std': 0.0006666891009750212, 'npar_std': 0.04539463666512193,
    'ksat_std': 3.178862086739814, 'k0_std': 0.4489771681065423, 'lpar_std':
    1.6118830509323745}, {'sand': 55.0, 'silt': 25.0, 'clay': 20.0, 'rhob':
    1.1, 'th33': None, 'th1500': None, 'version': 3, 'estimate_type': 'arith',
    'code': 3, 'thr': 0.09130753033144609, 'ths': 0.485031958049669, 'alpha':
    0.009748718589139088, 'npar': 1.4171964842691995, 'ksat': 84.7833726853498,
    'k0': 2.9015059553301388, 'lpar': -0.3463378623447919, 'thr_std':
    0.012777797583517441, 'ths_std': 0.013068706563916467, 'alpha_std':
    0.0022148332082555515, 'npar_std': 0.05767077407189314, 'ksat_std':
    27.035739850886362, 'k0_std': 1.8202787951521058, 'lpar_std':
    1.2248193041204392}]
    
    """
    if not isinstance(soildata, SoilData):
        soildata = SoilData.from_iter(soildata)
    means, stds, codes = rosetta(rosetta_version, soildata, estimate_type)
    results = []
    for sdat, mean, std, code in zip(soildata, means, stds, codes):
        results.append(
            RosettaResult(
                sand=sdat.sand,
                silt=sdat.silt,
                clay=sdat.clay,
                rhob=sdat.rhob,
                th33=sdat.th33,
                th1500=sdat.th1500,
                code=int(code),
                version=rosetta_version,
                estimate_type=estimate_type,
                thr=float(mean[0]),
                ths=float(mean[1]),
                alpha=float(mean[2]),
                npar=float(mean[3]),
                ksat=float(mean[4]),
                k0=float(mean[5]),
                lpar=float(mean[6]),
                thr_std=float(std[0]),
                ths_std=float(std[1]),
                alpha_std=float(std[2]),
                npar_std=float(std[3]),
                ksat_std=float(std[4]),
                k0_std=float(std[5]),
                lpar_std=float(std[6]),
            )
        )
    return RosettaResults(results)

