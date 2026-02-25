import numpy as np
import pytest

from rosetta.nn_model import NNModel, NNModelError
from rosetta.rosetta import _as_2d_array, RosettaError


def _make_simple_model():
    weights = [[np.array([[1.0, 1.0]], dtype=float)]]
    biases = [[np.array(0.0)]]
    layers = [{"activate": "linear"}]
    inputs = [{"scale": "asis", "params": []}, {"scale": "asis", "params": []}]
    outputs = [{"scale": "asis", "params": []}]
    return NNModel(
        name="simple",
        ninput=2,
        noutput=1,
        nboot=1,
        nlayer=1,
        weights=weights,
        biases=biases,
        layers=layers,
        inputs=inputs,
        outputs=outputs,
    )


def test_nnmodel_predict_shape_validation():
    model = _make_simple_model()
    good = model.predict([1.0, 2.0])
    assert good.shape == (1, 1, 1)
    with pytest.raises(NNModelError):
        model.predict([1.0])  # wrong input length


def test_bias_normalization_from_npz_data():
    npz_data = {
        "name": np.array("simple", dtype=object),
        "ninput": np.array(2),
        "noutput": np.array(1),
        "nboot": np.array(1),
        "nlayer": np.array(1),
        "weights": np.array([[np.array([[1.0, 1.0]])]], dtype=object),
        "biases": np.array([[np.array([0.0])]], dtype=object),  # shape (1,) bias
        "layers": np.array([{"activate": "linear"}], dtype=object),
        "inputs": np.array(
            [{"scale": "asis", "params": []}, {"scale": "asis", "params": []}],
            dtype=object,
        ),
        "outputs": np.array([{"scale": "asis", "params": []}], dtype=object),
    }
    model = NNModel.from_npz_data(npz_data)
    assert np.isscalar(model.biases[0][0])


def test_validate_network_rejects_bad_shapes():
    weights = [[np.zeros((1, 3))]]  # expects 2 inputs, mismatch
    biases = [[np.array(0.0)]]
    layers = [{"activate": "linear"}]
    inputs = [{"scale": "asis", "params": []}, {"scale": "asis", "params": []}]
    outputs = [{"scale": "asis", "params": []}]
    with pytest.raises(AssertionError):
        NNModel(
            name="bad",
            ninput=2,
            noutput=1,
            nboot=1,
            nlayer=1,
            weights=weights,
            biases=biases,
            layers=layers,
            inputs=inputs,
            outputs=outputs,
        )


def test_as_2d_array_raises_on_bad_shape():
    with pytest.raises(RosettaError):
        _ = _as_2d_array(np.array([1.0, 2.0, 3.0]), expected_ninput=2, src="unit")

