"""Bootstrap ensemble feed-forward neural network model"""

from dataclasses import dataclass
import importlib.resources
from typing import Callable, Self

import numpy as np
from numpy.typing import ArrayLike


class NNModelError(Exception):
    def __init__(self, message):
        self.message = message


@dataclass
class NNModel:
    """
    weights: list[list[np.array]]
        Outer list length is nboot, inner list length is nlayer.
        weights[i][j] is the weight matrix W (ndim=2) for the jth network
        layer of the ith bootstrap ensemble member.

    biases: list[list[np.array]]
        Outer list length is nboot, inner list length is nlayer.
        biases[i][j] are the biases b (ndim=1) for the jth network layer
        of the ith bootstrap ensemble member.

    """
    name: str
    ninput: int
    noutput: int
    nboot: int
    nlayer: int
    weights: list[list[np.ndarray]]
    biases: list[list[np.ndarray]]
    layers: list[dict]
    inputs: list[dict]
    outputs: list[dict]
    validate_input: Callable[[np.ndarray], bool] | None = None
    validate_network: bool = True

    def __post_init__(self):
        if self.validate_network:
            self._validate_network()
        self._input_scalers = [(self._scale_fn(c["scale"]), c["params"]) for c in self.inputs]
        self._output_scalers = [(self._scale_fn(c["scale"]), c["params"]) for c in self.outputs]
        self._layer_activations = [self._activate_fn(c["activate"]) for c in self.layers]

        self._W_stacked = [
            np.stack([self.weights[b][i].T for b in range(self.nboot)])
            for i in range(self.nlayer)
        ]
        self._b_stacked = [
            np.stack([np.atleast_2d(self.biases[b][i]) for b in range(self.nboot)])
            for i in range(self.nlayer)
        ]

    def _validate_network(self):
        assert len(self.inputs) == self.ninput, "inputs metadata length mismatch"
        assert len(self.outputs) == self.noutput, "outputs metadata length mismatch"
        assert len(self.layers) == self.nlayer, "layers metadata length mismatch"
        assert len(self.weights) == len(self.biases) == self.nboot, "bootstrap count mismatch"
        for boot_idx in range(self.nboot):
            ws = self.weights[boot_idx]
            bs = self.biases[boot_idx]
            assert len(ws) == len(bs) == self.nlayer, f"layer count mismatch in boot {boot_idx}"
            prev_units = self.ninput
            for layer_idx in range(self.nlayer):
                W = ws[layer_idx]
                b = bs[layer_idx]
                assert W.ndim == 2, f"W must be 2D (boot {boot_idx}, layer {layer_idx})"
                out_units, in_units = W.shape
                assert in_units == prev_units, (
                    f"W shape mismatch (boot {boot_idx}, layer {layer_idx}): "
                    f"expected {prev_units} inputs, got {in_units}"
                )
                if np.isscalar(b):
                    assert out_units == 1, (
                        f"scalar bias only allowed for single-output layer "
                        f"(boot {boot_idx}, layer {layer_idx})"
                    )
                else:
                    b_arr = np.asarray(b)
                    if b_arr.ndim == 0:
                        assert out_units == 1, (
                            f"scalar bias only allowed for single-output layer "
                            f"(boot {boot_idx}, layer {layer_idx})"
                        )
                    elif b_arr.ndim == 1:
                        assert b_arr.shape[0] == out_units, (
                            f"bias length mismatch (boot {boot_idx}, layer {layer_idx}): "
                            f"expected {out_units}, got {b_arr.shape[0]}"
                        )
                    elif b_arr.ndim == 2:
                        assert b_arr.shape == (out_units, 1), (
                            f"bias shape mismatch (boot {boot_idx}, layer {layer_idx}): "
                            f"expected ({out_units}, 1), got {b_arr.shape}"
                        )
                    else:
                        raise AssertionError(
                            f"bias must be 1D or column vector (boot {boot_idx}, layer {layer_idx}); "
                            f"got ndim={b_arr.ndim}"
                        )
                prev_units = out_units
            assert prev_units == self.noutput, (
                f"final layer outputs {prev_units} != noutput {self.noutput} (boot {boot_idx})"
            )

    @classmethod
    def from_npz_resource(
        cls,
        model_name: str,
        anchor="rosetta.data",
        validate_input: Callable | None = None,
        validate_network: bool = True,
    ) -> Self:
        resource = (importlib.resources.files(anchor).joinpath(f'{model_name}.npz'))
        with importlib.resources.as_file(resource) as npz_path:
            data = np.load(npz_path, allow_pickle=True)
        return cls.from_npz_data(data, validate_input, validate_network)

    @classmethod
    def from_npz_file(
        cls,
        filepath,
        validate_input: Callable | None = None,
        validate_network: bool = True,
    ) -> Self:
        data = np.load(filepath, allow_pickle=True)
        return cls.from_npz_data(data, validate_input, validate_network)

    @classmethod
    def from_npz_data(
        cls,
        npz_data: dict,
        validate_input: Callable | None = None,
        validate_network: bool = True,
    ) -> Self:
        weights = npz_data["weights"].tolist()
        # ensure weights are numpy arrays for matrix ops
        for boot_idx, boot_weights in enumerate(weights):
            for layer_idx, W in enumerate(boot_weights):
                weights[boot_idx][layer_idx] = np.asarray(W)
        biases = npz_data["biases"].tolist()
        # normalize length-1 bias arrays to scalars to avoid mixed shapes
        for boot_idx, boot_biases in enumerate(biases):
            for layer_idx, b in enumerate(boot_biases):
                arr = b if isinstance(b, np.ndarray) else np.asarray(b)
                if arr.shape == (1,):
                    biases[boot_idx][layer_idx] = arr.item()
                else:
                    biases[boot_idx][layer_idx] = arr
        return cls(
            name = str(npz_data['name'].item()),
            ninput = int(npz_data['ninput'].item()),
            noutput = int(npz_data['noutput'].item()),
            nboot = int(npz_data['nboot'].item()),
            nlayer = int(npz_data['nlayer'].item()),
            weights = weights,
            biases = biases,
            layers = npz_data['layers'].tolist(),
            inputs = npz_data['inputs'].tolist(),
            outputs = npz_data['outputs'].tolist(),
            validate_input = validate_input,
            validate_network = validate_network,
        )

    @staticmethod
    def _activate_fn(name: str) -> Callable[[np.ndarray], np.ndarray]:
        name = name.lower()
        if name == 'relu':
            return lambda x: np.maximum(0, x)
        elif name == 'sigmoid':
            return lambda x: 1.0 / (1.0 + np.exp(-x))
        elif name == 'tansig':
            return lambda x: -1.0 + 2.0 / (1.0 + np.exp(-2.0 * x))
        elif name == 'linear':
            return lambda x: x.copy()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    @staticmethod
    def _scale_fn(name: str) -> Callable[[float, np.ndarray], float]:
        name = name.lower()
        if name == 'asis':
            return lambda x, _: x
        elif name == 'division':
            return lambda x, p: x / p[0]
        elif name == 'logtr':
            return lambda x, p: np.log10(x) / p[0] + p[1]
        elif name == 'affine':
            return lambda x, p: p[0] * (x + p[1])
        elif name == 'minmax':
            return lambda x, p: (x - p[0]) * p[1] + p[2]
        elif name == 'linear':
            return lambda x, p: x * p[0] + p[1]
        else:
            raise ValueError(f"Unknown scaler method: {name}")

    def _scale(self, raw_vals: np.ndarray, scalers: list[tuple[Callable, list]]) -> np.ndarray:
        scaled = np.zeros_like(raw_vals)
        for i, (fn, params) in enumerate(scalers):
            scaled[..., i] = fn(raw_vals[..., i], params)
        return scaled

    def _forward_single(self, scaled_input: np.ndarray, boot_idx: int) -> np.ndarray:
        """Forward pass through the layers of a single network"""
        a = scaled_input
        for layer_idx in range(self.nlayer):
            activation = self._layer_activations[layer_idx]
            W = self.weights[boot_idx][layer_idx]
            b = self.biases[boot_idx][layer_idx]
            z = W @ a + b
            a = activation(z)
        return a

    def _forward_vectorized(self, scaled_inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the layers of multiple networks"""
        a = scaled_inputs[np.newaxis, :, :]
        for i in range(self.nlayer):
            z = np.matmul(a, self._W_stacked[i]) + self._b_stacked[i]
            a = self._layer_activations[i](z)
        return a

    def predict(self, inputs: ArrayLike) -> np.ndarray:
        """
        Parameters
        ----------
        inputs: shape (nsamples, ninput)

        Returns
        -------
        np.ndarray (nboot, nsamples, noutput)
        """
        inputs = np.asarray(inputs, dtype=np.float64)
        init_shape = inputs.shape
        if inputs.ndim == 1:
            inputs = inputs[np.newaxis, :]
        if inputs.ndim != 2 or inputs.shape[1] != self.ninput:
            raise NNModelError(
                f"Incorrectly shaped input. Input was shaped {init_shape}. "
                f"Input to {self.name} must be shaped (nsample, {self.ninput}), "
                f"or alternatively ({self.ninput},) if nsample = 1.)"
            )

        # return NaN for invalid samples
        invalid_mask = np.zeros(inputs.shape[0], dtype=bool)
        if self.validate_input:
            invalid_mask = np.array([not self.validate_input(inp) for inp in inputs])

        outputs = np.full((self.nboot, inputs.shape[0], self.noutput), np.nan)
        valid_mask = ~invalid_mask

        if np.any(valid_mask):
            valid_inputs = inputs[valid_mask]
            scaled_inputs = self._scale(valid_inputs, self._input_scalers)
            raw_outputs = self._forward_vectorized(scaled_inputs)
            outputs[:, valid_mask, :] = self._scale(raw_outputs, self._output_scalers)

        return outputs

