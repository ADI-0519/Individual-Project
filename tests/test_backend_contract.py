"""Pure stubs check extraction bookkeeping, not PyTorch/model correctness.

These checks intentionally need no model weights or torch installation. A real
model smoke run must still verify the library API, device placement and causal
prefix invariance on the chosen checkpoint.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from latent_probes.backends import HFBackend


class TensorStub:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, index):
        return TensorStub(self.values[index])

    def detach(self):
        return self

    def to(self, **kwargs):
        return self

    def numpy(self):
        return self.values


class HandleStub:
    def __init__(self, block, callback):
        self.block = block
        self.callback = callback

    def remove(self):
        self.block.callbacks.remove(self.callback)


class BlockStub:
    def __init__(self):
        self.callbacks = []

    def register_forward_hook(self, callback):
        self.callbacks.append(callback)
        return HandleStub(self, callback)


class DecoderStub:
    def __init__(self, *, fail=False, nonfinite=False, omit_last=False):
        self.layers = [BlockStub(), BlockStub()]
        self.fail = fail
        self.nonfinite = nonfinite
        self.omit_last = omit_last
        self.values = []

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        if self.fail:
            raise RuntimeError("injected forward failure")
        for layer, block in enumerate(self.layers):
            values = np.arange(5)[:, None] * 10 + np.arange(4)[None, :] + layer * 100
            values = values[np.newaxis].astype(float)
            if self.nonfinite and layer == 1:
                values[0, 4, 0] = np.nan
            self.values.append(values)
            if self.omit_last and layer == 1:
                continue
            output = TensorStub(values)
            # Cover both decoder output forms accepted by the hook adapter.
            output = (output,) if layer == 0 else output
            for callback in list(block.callbacks):
                callback(block, (), output)


def backend_stub(**kwargs):
    backend = object.__new__(HFBackend)
    backend.settings = SimpleNamespace(capture_layers=[0, 1])
    backend.torch = SimpleNamespace(
        inference_mode=nullcontext,
        float32=np.float32,
        ones_like=lambda values: np.ones_like(values),
    )
    backend.base = DecoderStub(**kwargs)
    return backend


class BackendContractTests(unittest.TestCase):
    def test_exact_token_positions_and_layer_keys_are_preserved(self):
        backend = backend_stub()
        features, positions = backend._capture(np.array([[1, 2, 3, 4, 5]]), 3, 4)
        self.assertEqual(positions, {"prompt_last": 2, "response_last": 4})
        self.assertEqual(
            set(features),
            {
                "layer_0_prompt_last",
                "layer_0_response_last",
                "layer_1_prompt_last",
                "layer_1_response_last",
            },
        )
        np.testing.assert_array_equal(features["layer_0_prompt_last"], [20, 21, 22, 23])
        np.testing.assert_array_equal(features["layer_1_response_last"], [140, 141, 142, 143])
        self.assertTrue(all(not layer.callbacks for layer in backend.base.layers))

    def test_replay_does_not_request_all_hidden_states_or_a_cache(self):
        backend = backend_stub()
        sequence = np.array([[1, 2, 3, 4, 5]])
        backend._capture(sequence, 3, 4)
        self.assertIs(backend.base.kwargs["input_ids"], sequence)
        self.assertFalse(backend.base.kwargs["use_cache"])
        self.assertFalse(backend.base.kwargs["output_hidden_states"])

    def test_features_are_copied_instead_of_aliasing_forward_storage(self):
        backend = backend_stub()
        features, _ = backend._capture(np.array([[1, 2, 3, 4, 5]]), 3, 4)
        backend.base.values[0][:] = 999
        np.testing.assert_array_equal(features["layer_0_prompt_last"], [20, 21, 22, 23])

    def test_forward_failure_cleans_up_every_hook(self):
        backend = backend_stub(fail=True)
        with self.assertRaisesRegex(RuntimeError, "injected forward failure"):
            backend._capture(np.array([[1, 2, 3, 4, 5]]), 3, 4)
        self.assertTrue(all(not layer.callbacks for layer in backend.base.layers))

    def test_nonfinite_feature_is_rejected_and_hooks_are_cleaned_up(self):
        backend = backend_stub(nonfinite=True)
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            backend._capture(np.array([[1, 2, 3, 4, 5]]), 3, 4)
        self.assertTrue(all(not layer.callbacks for layer in backend.base.layers))

    def test_missing_requested_layer_capture_is_rejected(self):
        backend = backend_stub(omit_last=True)
        with self.assertRaisesRegex(RuntimeError, "Not all requested"):
            backend._capture(np.array([[1, 2, 3, 4, 5]]), 3, 4)


if __name__ == "__main__":
    unittest.main()
