# Frozen specificity-pilot inputs

These are byte-identical curated copies of the prospectively prepared local
inputs. They are tracked so offline tests and the source distribution work in
a fresh checkout. No model answers or behavior labels appear in these panels.
The full originals and preparation evidence remain in ignored local storage.

- `mask_specificity_v1.json`: eight purposively selected original MASK cases,
  with original belief/pressure questions and logged explicit-incentive deletions.
  MASK/Ren et al., Center for AI Safety and Scale AI.
  [Dataset](https://huggingface.co/datasets/cais/MASK/tree/74a44111218d762985b7a82f39b0eb5bddadad2a),
  [paper](https://arxiv.org/abs/2503.03750),
  [CC-BY-4.0 data license](https://creativecommons.org/licenses/by/4.0/).
  The upstream implementation is separately MIT licensed.
- `mask_neutral_calibration_v1.json`: 32 selected original Stanford Alpaca
  instructions, without source answers, rendered in a helpful chat context.
  Stanford CRFM/Taori et al.
  [Pinned source](https://github.com/tatsu-lab/stanford_alpaca/tree/761dc5bfbdeeffa89b8bff5d038781a4055f796a),
  [CC-BY-NC-4.0 data license](https://github.com/tatsu-lab/stanford_alpaca/blob/761dc5bfbdeeffa89b8bff5d038781a4055f796a/DATA_LICENSE).
  This use is noncommercial research; upstream code is Apache-2.0 licensed.

See the [source audit](../../docs/MASK_SOURCE_AUDIT.md) for exact hashes,
selection, modifications, attribution and interpretation limits. A neutral-score
percentile is not a validated honest-response false-positive rate. Behavioral
annotations remain separate agent-assisted drafts requiring human review.
