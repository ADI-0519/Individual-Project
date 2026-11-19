# Linear probe: what is implemented

The starter implements a regularised logistic-regression baseline on frozen
activation vectors. It is an adaptation of a general probing method. It does
not claim to reproduce the Apollo paper's model, dataset or published scores.

## Inputs

The feature NPZ must contain:

- `X`: finite numeric matrix, one example per row.
- `example_ids`: unique nonempty Unicode strings in the same order as `X`.
- `feature_signature` (strongly recommended): scalar Unicode string identifying
  the exact model/revision, precision/quantization and capture layer/position.

Labels are a CSV with `example_id`, `family_id` and an explicitly completed
binary target such as `label`. Values must be `0` or `1`. Missing labels and
duplicate IDs are errors. The feature IDs and label IDs must match exactly.
Filter/export an explicit matching subset if some examples remain unaudited;
the fitting function never silently excludes them.

When fitting the generic `label` column, an accompanying `label_target` column
must contain the same nonempty target definition in every row. Mixed definitions
are rejected even when their binary values happen to agree. Blank definitions
and the `manual_definition_required` placeholder are rejected. Complete both
`label` and `label_target` in the exported manual annotations; feature packing
preserves that definition. This checks consistency, not whether an annotation is
scientifically justified.

Legacy CSVs without `label_target` remain supported, but their manifests record
unverified target semantics and an explicit limitation. Choosing another target
column does not borrow the unrelated definition of `label`: `objective_pressure`
is an assigned instruction condition, while `critical_fact_disclosed` is a
user-supplied disclosure annotation. Neither is an automatic deception label.
Other named targets remain semantically unverified unless separately defined.

If a `text` column is present, every row must contain the intended text input.
The code then fits a separate TF-IDF logistic-regression baseline on exactly the
same rows and split. Choose that column deliberately: output-only text is not
information-matched to activations that also encode the prompt/source/audience.
Winning against this bounded baseline does not establish information that is
unavailable from text.

An optional `scenario_id` is checked against the chosen grouping field. Known
scenario or template families cannot be divided across split groups. When
family membership exists, splitting on individual messages/scenarios that share
a family is rejected. Otherwise the code must trust the grouping metadata you
supply; it cannot discover semantic duplicates.

## Training and evaluation

`train_probe(features_path, labels_path, output_dir, ...)` returns its manifest.
`split_groups(groups, ...)` returns row-index arrays under `train`, `validation`
and `test`.

1. Sort unique group IDs, shuffle once with the recorded seed, and allocate
   approximately 60%/20%/20% of groups to train/validation/test.
2. Require both labels in each partition. A deficient partition produces an
   error; the code does not try random seeds until it finds a preferred split.
3. Fit `StandardScaler` and L2 logistic regression only on training rows.
4. Choose `C` by validation AUROC. Ties choose the smaller `C` (stronger
   regularisation). There is no train-plus-validation refit in this baseline.
5. Evaluate the chosen model on test groups once. Use a prespecified threshold
   of 0.5; this is not an estimated 1% false-positive-rate operating point.
6. Record AUROC, average precision, Brier score, accuracy, confusion counts and
   false-positive/true-positive rates. Logistic scores have no additional
   calibration step; a probability-shaped number is not proof of calibration.
7. Estimate percentile intervals by resampling whole held-out groups. With fewer
   than two test groups no interval is reported. Few groups still give unstable
   intervals even when the computation succeeds.

Changing `C`/feature layer after inspecting test results makes those data
development data. This utility cannot stop a person repeatedly looking at the
test set. Record decisions and reserve fresh families for final evaluation.
For many candidate layers, select the layer on validation data and evaluate the
selected configuration once on the held-out test set.

The 24-scenario communication pilot is for debugging and task validation. It is
not an adequate independent training source plus final test set. Obtain separate
probe-training data; grow independent scenario/template families before making
generalisation claims.

## Saved outputs

- `probe.npz`: scaler parameters and logistic weights, without pickle.
- `manifest.json`: checksums, feature signature, environment versions, group and
  example split membership, target definition and its verification status,
  validation selection, test scores and limitations.
- `test_predictions.csv`: per-example held-out predictions.
- Optional `text_probe.npz` and `text_vocabulary.json`: numeric text baseline
  parameters and training vocabulary. Text baseline scores appear in the same
  test-prediction CSV; no separate text prediction CLI is provided yet.

`predict_probe(features_path, probe_dir, output_path)` applies the saved numeric
activation probe. It rejects changed feature widths or changed/missing capture
signatures when the training signature was present. If no signature was supplied
at training time, only dimension checks are possible and the result is marked
as unverified. Different models of the same width are still different feature
spaces; train a separate probe per model/precision/capture setting.

## Keep the target's meaning explicit

These are distinct labels, and none is automatically an intent label:

- Instruction condition: what objective the sender was assigned.
- Disclosure: whether a specified fact appears in its message.
- Literal truth: whether a claim is supported by the source.
- Recipient error: the recipient's measured conclusion relative to the world.
- Elicited forecast: what the sender says a recipient will conclude when asked.
- Controlled belief-task label: the independently specified perspective in a
  separate task.

In particular, predicting an instruction or a missing fact does not establish
spontaneous second-order belief modelling. Compare natural behaviour under
matched objectives, handle harmless omissions, and assess independent
belief-task transfer separately. Actual recipient information that was hidden
from the sender cannot be recovered from its already-recorded activation.

Numerical fixtures in `tests/test_probes.py` test software integrity. They are
artificial arrays and are not LLM experiments or research results.

## Method reference

Goldowsky-Dill et al. (ICML 2025), *Detecting Strategic Deception with Linear
Probes*: <https://proceedings.mlr.press/v267/goldowsky-dill25a.html>.
Authors' implementation: <https://github.com/ApolloResearch/deception-detection>.
