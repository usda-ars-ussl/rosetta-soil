from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from rosetta import rosetta

def parse(raw: str) -> list[list[float]]:
    return [
        [float(s) for s in row]
        for row in [r.split() for r in raw.strip().split("\n")]
    ]

# Prepare features and expected results
inputs, means, stds = [], [], []
for mod_id in (2, 3, 4, 5):
    path = Path(__file__).parent / f"data/rose1_mod{mod_id}_inputs.txt"
    raw_input = path.read_text().strip()
    inputs.append(np.array(parse(raw_input), dtype=np.float64))

    path = Path(__file__).parent / f"data/rose1_mod{mod_id}_mean.txt"
    raw_mean = path.read_text().strip()
    means.append(np.array(parse(raw_mean), dtype=np.float64))

    path = Path(__file__).parent / f"data/rose1_mod{mod_id}_std.txt"
    raw_std = path.read_text().strip()
    stds.append(np.array(parse(raw_std), dtype=np.float64))


@pytest.mark.parametrize("input, expected_mean, expected_std", zip(inputs, means, stds))
def test_unsat_k_prediction(input, expected_mean, expected_std):
    # pyrosetta Rosetta 1 code used to generate reference data
    # returned geometric means for alp, npar, ksat, and k0,
    # and stdev's for log(alp), log(npar), log(ksat), and log(k0)
    # Need to split the comparision here because rosetta.rosetta
    # returns std for antilog params when estimated_type is "geo" 
    mean, _, _ = rosetta(1, input, estimate_type="geo")
    npt.assert_allclose(mean, expected_mean, rtol=1e-5, atol=1e-5)
    _, std, _ = rosetta(1, input, estimate_type="log")
    npt.assert_allclose(std, expected_std, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
