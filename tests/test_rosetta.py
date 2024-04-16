import numpy as np
import numpy.testing as npt


def test_soildatum() -> None:
    from rosetta.rosetta import SoilDatum

    assert SoilDatum(10, 20, 70).is_valid_separates() is True
    assert SoilDatum(10, 20, 70).is_valid() is False
    assert SoilDatum(10, 20, 70).is_valid(3) is True
    assert SoilDatum(8.9, 20, 70).is_valid_separates() is False
    assert SoilDatum(8.9, 20, 70).is_valid(3) is False
    assert SoilDatum(10, 20, 71).is_valid_separates() is True
    assert SoilDatum(10, 20, 71.1).is_valid_separates() is False
    assert SoilDatum(10, 20, 70, 1.1).is_valid_bulkdensity() is True
    assert SoilDatum(10, 20, 70, 2.1).is_valid_bulkdensity() is False
    assert SoilDatum(10, 20, 70, 1.1).is_valid(4) is True
    assert SoilDatum(10, 20, 70, 2.1).is_valid(4) is False
    assert SoilDatum(10, 20, 70, 1.1, 0.3).is_valid_fieldcap() is True
    assert SoilDatum(10, 20, 70, 1.1, 30).is_valid_fieldcap() is False
    assert SoilDatum(10, 20, 70, 1.1, 0.3).is_valid(5) is True
    assert SoilDatum(10, 20, 70, 1.1, 30).is_valid(5) is False
    assert SoilDatum(10, 20, 70, 1.1, 0.3, 0.1).is_valid_wiltpoint() is True
    assert SoilDatum(10, 20, 70, 1.1, 0.3, 10).is_valid_wiltpoint() is False
    assert SoilDatum(10, 20, 70, 1.1, 0.3, 0.1).is_valid() is True
    assert SoilDatum(10, 20, 70, 1.1, 0.3, 10).is_valid() is False
    assert SoilDatum(10, 20, 70, 1.1, 0.1, 0.3).is_valid(6) is False
    assert SoilDatum(0, 20, 70, 1.1, 0.3, 0.1).is_valid(6) is False
    assert SoilDatum(10, 20, 70).is_valid(2) is False
    assert SoilDatum(10, 20, 70).is_valid(-99) is False
    assert SoilDatum(10, 20, 70).is_valid(99) is False
    assert SoilDatum(10, 20, 70).is_valid(3) is True
    assert SoilDatum(8.9, 20, 70).is_valid(3) is False
    assert SoilDatum(10, 20, 70, 1.1, 0.3, 10).is_valid(3) is True
    assert SoilDatum(10, 20, 70, 1.1, 0.3, 10).is_valid(4) is True
    assert SoilDatum(10, 20, 70, 1.1, 0.3, 10).is_valid(5) is True
    assert SoilDatum(10, 20, 70, 1.1, 0.3, 10).is_valid(6) is False
    assert SoilDatum(10, 20, 70, 1.1, 0.3, 0.1).is_valid(3) is True
    assert SoilDatum(10, 20, 70, 1.1, 0.3, 0.1).is_valid(4) is True
    assert SoilDatum(10, 20, 70, 1.1, 0.3, 0.1).is_valid(5) is True
    assert SoilDatum(10, 20, 70, 1.1, 0.3, 0.1).is_valid(6) is True
    assert SoilDatum(10, 20, 70).is_valid(4) is False
    assert SoilDatum(0, 20, 70).best_index() == 0
    assert SoilDatum(10, 20, 70).best_index() == 3
    assert SoilDatum(10, 20, 70, 1.1, 0.3, 10).best_index() == 5
    assert SoilDatum(10, 20, 70, 1.1, 0.3, 0.1).best_index() == 6
    assert SoilDatum(10, 20, 70, 2.1).best_index() == 3
    assert SoilDatum(10, 20, 70, 1.1).best_index() == 4


def test_soildata() -> None:
    from rosetta import SoilData
    from rosetta.rosetta import SoilDatum

    dat = [[10, 20, 70]]
    dat2 = [dat[0], [0, 10, 20]]
    dat3 = [
        [30, 40, 30, 1, 0.3, 0.1],
        [30, 40, 30, 1, 3, 0.1],
        [30, 40, 30, 1, 0.3, 0.1],
    ]
    sf = SoilData.from_array(dat)
    sf2 = SoilData.from_array(dat2)
    sf3 = SoilData.from_array(dat3)

    assert sf[0] == SoilDatum(*list(map(float, dat[0])))
    assert sf[0] == SoilDatum(*list(map(float, dat2[0])))
    assert sf2[1] == SoilDatum(*list(map(float, dat2[1])))
    assert len(sf) == 1 and len(sf2) == 2 and len(sf3) == 3
    assert sf2.is_valid(3) == [True, False]
    assert sf3.is_valid(3) == [True, True, True]
    assert sf3.is_valid() == [True, False, True]
    assert sf3.is_valid(5) == [True, False, True]

    npt.assert_array_equal(sf.to_array(), np.array([dat2[0] + 3 * [np.nan]]))
    #npt.assert_array_equal(
    #    sf2.to_array(4), np.array([dat2[0] + [np.nan], dat2[1] + [np.nan]])
    #)
    npt.assert_array_equal(sf3.to_array(), np.array(dat3))
    #npt.assert_array_equal(
    #    sf3.to_array(4), np.array([[30, 40, 30, 1], [30, 40, 30, 1], [30, 40, 30, 1]])
    #)


def test_rosetta() -> None:
    import itertools
    from typing import List

    from rosetta import rosetta, SoilData

    from data import (
        FEATURES,
        ROSE1_MOD2,
        ROSE1_MOD3,
        ROSE1_MOD4,
        ROSE1_MOD5,
        ROSE2_MOD2,
        ROSE2_MOD3,
        ROSE2_MOD4,
        ROSE2_MOD5,
        ROSE3_MOD2,
        ROSE3_MOD3,
        ROSE3_MOD4,
        ROSE3_MOD5,
    )

    def parse(raw: str) -> List[List[float]]:
        return [
            [float(s) for s in row]
            for row in [r.split() for r in raw.strip().split("\n")]
        ]

    FEATURES = parse(FEATURES)

    DESIRED = dict(
        r1m2=ROSE1_MOD2,
        r1m3=ROSE1_MOD3,
        r1m4=ROSE1_MOD4,
        r1m5=ROSE1_MOD5,
        r2m2=ROSE2_MOD2,
        r2m3=ROSE2_MOD3,
        r2m4=ROSE2_MOD4,
        r2m5=ROSE2_MOD5,
        r3m2=ROSE3_MOD2,
        r3m3=ROSE3_MOD3,
        r3m4=ROSE3_MOD4,
        r3m5=ROSE3_MOD5,
    )
    DESIRED = {k: np.array(parse(v), dtype=np.float64) for k, v in DESIRED.items()}
    DESIRED["r1m0"] = DESIRED["r1m5"]
    DESIRED["r2m0"] = DESIRED["r2m5"]
    DESIRED["r3m0"] = DESIRED["r3m5"]

    for ver, mod in itertools.product((1, 2, 3), (2, 3, 4, 5)):
        jcol = mod + 1
        arr, stdev, codes = rosetta(ver, SoilData.from_array([f[:jcol] for f in FEATURES]))
        npt.assert_equal(codes, np.array(len(FEATURES) * [mod], dtype=int))
        npt.assert_array_almost_equal(
            arr, DESIRED[f"r{ver}m{mod}"], decimal=5 if ver == 2 else 10
        )

    features = FEATURES[:]
    features[0][0] = np.nan
    features[1][3] = "NA"
    features[2][4] = -99.0
    features[3][5] = 100.0
    arr, stdev, codes = rosetta(1, SoilData.from_array(features))
    npt.assert_equal(codes[:5], np.array([-1, 2, 3, 4, 5], dtype=int))
    npt.assert_array_equal(arr[0], np.full(5, np.nan))
    npt.assert_allclose(arr[1], DESIRED["r1m2"][1])
    npt.assert_allclose(arr[2], DESIRED["r1m3"][2])
    npt.assert_allclose(arr[3], DESIRED["r1m4"][3])
    npt.assert_allclose(arr[4], DESIRED["r1m5"][4])
    npt.assert_allclose(arr[5], DESIRED["r1m5"][5])

    feat3 = [f[:3] for f in features]
    arr, stdev, codes = rosetta(1, SoilData.from_array(feat3))
    npt.assert_equal(codes[:5], np.array([-1, 2, 2, 2, 2], dtype=int))
    npt.assert_array_equal(arr[0], np.full(5, np.nan))
    npt.assert_allclose(arr[1], DESIRED["r1m2"][1])
    npt.assert_allclose(arr[2], DESIRED["r1m2"][2])
    npt.assert_allclose(arr[3], DESIRED["r1m2"][3])

    feat4 = [f[:4] for f in features]
    arr, stdev, codes = rosetta(1, SoilData.from_array(feat4))
    npt.assert_equal(codes[:5], np.array([-1, 2, 3, 3, 3], dtype=int))
    npt.assert_array_equal(arr[0], np.full(5, np.nan))
    npt.assert_allclose(arr[1], DESIRED["r1m2"][1])
    npt.assert_allclose(arr[2], DESIRED["r1m3"][2])
    npt.assert_allclose(arr[3], DESIRED["r1m3"][3])
    npt.assert_allclose(arr[4], DESIRED["r1m3"][4])


def test_error() -> None:
    import pytest
    from rosetta import rosetta, RosettaError, SoilData

    with pytest.raises(RosettaError) as e:
        _ = rosetta(-99, SoilData.from_array([[1, 2, 3]]))
    with pytest.raises(RosettaError) as e:
        _ = rosetta(4, SoilData.from_array([[1, 2, 3]]))
    with pytest.raises(RosettaError) as e:
        _ = rosetta(1, [[1, 2, 3]])


if __name__ == "__main__":
    test_soildatum()
    test_soildata()
    test_rosetta()
    test_error()
