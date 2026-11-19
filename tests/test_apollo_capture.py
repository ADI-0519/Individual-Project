"""Offline token-capture contracts; these fixtures do not validate a real model.

The character tokenizer makes boundaries inspectable, while deterministic block
outputs expose token order, replay identity and hook lifecycle mistakes.
"""

import unittest
from contextlib import nullcontext
from itertools import count
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from latent_probes.backends import GenerationResult, HFBackend
from latent_probes.config import GenerationSettings


class TensorFixture:
    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, index):
        return TensorFixture(self.values[index])

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def numpy(self):
        return self.values

    def tolist(self):
        return self.values.tolist()


class HookFixture:
    def __init__(self, block, callback):
        self.block, self.callback = block, callback

    def remove(self):
        self.block.callbacks.remove(self.callback)


class BlockFixture:
    def __init__(self):
        self.callbacks = []

    def register_forward_hook(self, callback):
        self.callbacks.append(callback)
        return HookFixture(self, callback)


class DecoderFixture:
    def __init__(self, *, fail=False, nonfinite=False, omit_last=False):
        self.layers = [BlockFixture(), BlockFixture()]
        self.fail, self.nonfinite, self.omit_last = fail, nonfinite, omit_last
        self.forward_calls = []
        self.storage = []

    def __call__(self, **kwargs):
        self.forward_calls.append(kwargs)
        if self.fail:
            raise RuntimeError("injected token replay failure")
        length = kwargs["input_ids"].shape[1]
        for layer, block in enumerate(self.layers):
            values = (layer * 100 + np.arange(length)[:, None] * 10 + np.arange(3)[None, :])[
                None, :, :
            ].astype(float)
            if self.nonfinite and layer == 1:
                values[0, 1, 0] = np.nan
            self.storage.append(values)
            if self.omit_last and layer == 1:
                continue
            output = TensorFixture(values)
            if layer == 0:
                output = (output,)
            for callback in list(block.callbacks):
                callback(block, (), output)


class EncodingFixture(dict):
    def __init__(self, text, missing_offsets=()):
        super().__init__(input_ids=TensorFixture([[ord(c) for c in text]]))
        self.missing_offsets = set(missing_offsets)
        self.text = text

    def char_to_token(self, batch_index, character_index):
        assert batch_index == 0
        if character_index in self.missing_offsets:
            return None
        return character_index if 0 <= character_index < len(self.text) else None


class CharacterTokenizerFixture:
    """One fixture token per character; deliberately unlike real BPE tokenizers."""

    is_fast = True
    all_special_ids = [0, 99]

    def __init__(self):
        self.missing_offsets = set()
        self.break_template = False

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        if add_generation_prompt:
            return "<user>" + messages[-1]["content"] + "</user><assistant>"
        prefix = "<user>" + messages[-2]["content"] + "</user><assistant>"
        return (
            prefix + (" " if self.break_template else "") + messages[-1]["content"] + "</assistant>"
        )

    def __call__(self, text, *, add_special_tokens, return_tensors, return_offsets_mapping):
        assert add_special_tokens is False
        assert return_tensors == "pt" and return_offsets_mapping is True
        return EncodingFixture(text, self.missing_offsets)

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


class WordEncodingFixture(dict):
    def __init__(self, segments):
        super().__init__(input_ids=TensorFixture([[1000 + i for i in range(len(segments))]]))
        self.offsets = []
        cursor = 0
        for segment in segments:
            self.offsets.append((cursor, cursor + len(segment)))
            cursor += len(segment)

    def char_to_token(self, batch_index, character_index):
        assert batch_index == 0
        return next(
            (i for i, (start, end) in enumerate(self.offsets) if start <= character_index < end),
            None,
        )


class WordTokenizerFixture(CharacterTokenizerFixture):
    """Merged word/leading-space tokens test offset use, without assuming real BPE."""

    def __call__(self, text, **kwargs):
        header, separator, rest = text.partition("<assistant>")
        response, terminator, tail = rest.partition("</assistant>")
        assert separator and terminator and not tail
        words = response.split(" ")
        self.segments = [header + separator, words[0]]
        self.segments.extend(" " + word for word in words[1:])
        self.segments.append(terminator)
        return WordEncodingFixture(self.segments)

    def decode(self, ids, **kwargs):
        return "".join(self.segments[i - 1000] for i in ids)


def backend_fixture(**kwargs):
    backend = object.__new__(HFBackend)
    backend.settings = SimpleNamespace(capture_layers=[0, 1], max_input_tokens=1024)
    backend.torch = SimpleNamespace(
        inference_mode=nullcontext,
        float32=np.float32,
        ones_like=lambda tensor: TensorFixture(np.ones_like(tensor.values)),
        tensor=lambda values, **kw: TensorFixture(values),
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
    )
    backend.base = DecoderFixture(**kwargs)
    backend.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048))
    backend.input_device = "cpu"
    backend.tokenizer = CharacterTokenizerFixture()
    return backend


class TokenMatrixCaptureTests(unittest.TestCase):
    def test_selected_row_order_layers_and_independent_storage(self):
        backend = backend_fixture()
        sequence = TensorFixture([[1, 2, 3, 4, 5]])
        matrices = backend._capture_token_rows(sequence, [3, 1])
        np.testing.assert_array_equal(matrices["layer_0_tokens"], [[30, 31, 32], [10, 11, 12]])
        np.testing.assert_array_equal(
            matrices["layer_1_tokens"], [[130, 131, 132], [110, 111, 112]]
        )
        backend.base.storage[0][:] = -999
        self.assertEqual(matrices["layer_0_tokens"][0, 0], 30)
        call = backend.base.forward_calls[0]
        self.assertIs(call["input_ids"], sequence)
        self.assertFalse(call["use_cache"])
        self.assertFalse(call["output_hidden_states"])
        self.assertTrue(all(not block.callbacks for block in backend.base.layers))

    def test_invalid_positions_fail_before_forward_or_hook_registration(self):
        for positions in ([], [1, 1], [-1], [5], [True], [1.5]):
            with self.subTest(positions=positions):
                backend = backend_fixture()
                with self.assertRaises(ValueError):
                    backend._capture_token_rows(TensorFixture([[1, 2, 3, 4, 5]]), positions)
                self.assertFalse(backend.base.forward_calls)
                self.assertTrue(all(not block.callbacks for block in backend.base.layers))

    def test_failure_nonfinite_and_missing_capture_clean_up_hooks(self):
        for failure in ({"fail": True}, {"nonfinite": True}, {"omit_last": True}):
            with self.subTest(failure=failure):
                backend = backend_fixture(**failure)
                with self.assertRaises(RuntimeError):
                    backend._capture_token_rows(TensorFixture([[1, 2, 3]]), [1, 2])
                self.assertTrue(all(not block.callbacks for block in backend.base.layers))


class ForcedResponseMaskTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "user", "content": "FIXTURE PROMPT"}]
        self.response = "one two three four five six seven eight"

    def test_word_trim_masks_prefix_without_removing_suffix_from_context(self):
        backend = backend_fixture()
        matrices, record = backend.forced_token_activations(self.messages, self.response, 5)
        retained = "one two three"
        prefix = backend.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        expected_positions = list(range(len(prefix), len(prefix) + len(retained)))
        self.assertEqual(record["selected_token_positions"], expected_positions)
        self.assertEqual(record["selected_token_ids"], [ord(c) for c in retained])
        self.assertEqual(record["selected_decoded_text"], retained)
        self.assertEqual(record["excluded_suffix"], " four five six seven eight")
        self.assertEqual(record["character_span"], [len(prefix), len(prefix) + len(retained)])
        full_input = backend.base.forward_calls[0]["input_ids"].tolist()[0]
        self.assertEqual("".join(map(chr, full_input)), prefix + self.response + "</assistant>")
        self.assertEqual(matrices["layer_0_tokens"].shape, (len(retained), 3))

    def test_untrimmed_response_excludes_template_terminator(self):
        backend = backend_fixture()
        _, record = backend.forced_token_activations(self.messages, self.response)
        self.assertEqual(record["selected_decoded_text"], self.response)
        self.assertEqual(record["excluded_suffix"], "")

    def test_offsets_select_whole_merged_tokens_instead_of_counting_characters(self):
        backend = backend_fixture()
        backend.tokenizer = WordTokenizerFixture()
        matrices, record = backend.forced_token_activations(self.messages, self.response, 5)
        self.assertEqual(record["selected_token_positions"], [1, 2, 3])
        self.assertEqual(record["selected_token_ids"], [1001, 1002, 1003])
        self.assertEqual(record["selected_decoded_text"], "one two three")
        self.assertEqual(matrices["layer_0_tokens"].shape, (3, 3))
        self.assertEqual(len(record["input_token_ids"]), 10)

    def test_unmapped_character_boundaries_fail_instead_of_guessing(self):
        for boundary in ("start", "end"):
            with self.subTest(boundary=boundary):
                backend = backend_fixture()
                prefix = backend.tokenizer.apply_chat_template(
                    self.messages, tokenize=False, add_generation_prompt=True
                )
                point = len(prefix) + (0 if boundary == "start" else len("one two three"))
                backend.tokenizer.missing_offsets.add(point)
                with self.assertRaisesRegex(ValueError, "boundary"):
                    backend.forced_token_activations(self.messages, self.response, 5)
                self.assertFalse(backend.base.forward_calls)

    def test_nonpreserving_template_and_slow_tokenizer_are_rejected(self):
        backend = backend_fixture()
        backend.tokenizer.break_template = True
        with self.assertRaisesRegex(ValueError, "preserve"):
            backend.forced_token_activations(self.messages, self.response)
        backend = backend_fixture()
        backend.tokenizer.is_fast = False
        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            backend.forced_token_activations(self.messages, self.response)

    def test_context_limits_reject_complete_dialogue_without_truncation(self):
        for limit_type in ("configured", "model"):
            with self.subTest(limit_type=limit_type):
                backend = backend_fixture()
                if limit_type == "configured":
                    backend.settings.max_input_tokens = 32
                else:
                    backend.model.config.max_position_embeddings = 32
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    backend.forced_token_activations(self.messages, self.response)
                self.assertFalse(backend.base.forward_calls)

    def test_fixed_response_profiling_waits_for_device_work(self):
        backend = backend_fixture()
        events = []
        backend.torch.cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            reset_peak_memory_stats=lambda i: events.append(("reset", i)),
            synchronize=lambda i: events.append(("sync", i)),
            max_memory_allocated=lambda i: (i + 1) * 2**30,
        )

        def clock():
            events.append("clock")
            return len(events)

        with patch("latent_probes.backends.time.perf_counter", side_effect=clock):
            _, record = backend.forced_token_activations(self.messages, self.response, 5)
        for device in (0, 1):
            self.assertIn(("reset", device), events)
            self.assertIn(("sync", device), events)
            self.assertLess(events.index(("sync", device)), len(events) - 1)
        self.assertEqual(events[-1], "clock")
        self.assertEqual(record["peak_allocated_gib"], {"0": 1.0, "1": 2.0})


class GeneratedResponseReplayTests(unittest.TestCase):
    def test_replay_uses_exact_ids_and_scores_only_content_positions(self):
        backend = backend_fixture()
        result = GenerationResult(
            text="FIXTURE GENERATED TEXT",
            prompt_token_ids=[11, 12],
            generated_token_ids=[5, 99, 7, 0],
            elapsed_seconds=3.0,
        )
        calls = []

        def generate(messages, settings, seed, capture):
            calls.append((messages, settings, seed, capture))
            return result

        backend.generate = generate
        settings = GenerationSettings(do_sample=False)
        with patch("latent_probes.backends.time.perf_counter", side_effect=count(1)):
            matrices, record = backend.generated_token_activations([], settings, 17)
        self.assertEqual(calls, [([], settings, 17, False)])
        self.assertEqual(
            backend.base.forward_calls[0]["input_ids"].tolist(), [[11, 12, 5, 99, 7, 0]]
        )
        self.assertEqual(record["selected_token_ids"], [5, 7])
        self.assertEqual(record["selected_token_positions"], [2, 4])
        self.assertEqual(matrices["layer_0_tokens"].shape, (2, 3))
        self.assertGreater(record["extraction_seconds"], 0)

    def test_all_special_generation_has_no_score_and_no_replay(self):
        backend = backend_fixture()
        backend.generate = lambda *args, **kwargs: GenerationResult(
            text="", prompt_token_ids=[11], generated_token_ids=[99, 0]
        )
        with self.assertRaisesRegex(ValueError, "no scoreable"):
            backend.generated_token_activations([], GenerationSettings(do_sample=False), 17)
        self.assertFalse(backend.base.forward_calls)
