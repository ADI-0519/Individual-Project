# Frozen belief readout: independent initial-belief controls

**21 November 2026. Real local GPU execution, completed once.** The queried
readout passes all primary floors at **48/64**, and responds in the correct
direction in **29/32 informative receipt pairs**. It fails full qualification:
the new no-change controls score **43/64**, and receipt changes flip its verdict
in **13/32 pairs whose attributed belief stays fixed**. No future-goal conditions
or new probe fits followed.

This is evidence of partial prompted belief information and a controlled
robustness failure. It is not a working spontaneous anticipation monitor, a
demonstration of deceptive intent, or an established activation advantage.

[Verified complete record](validation/2026-11-21-crossed-belief-validation.json)
· [Independent raw audit](validation/2026-11-21-crossed-belief-independent.json)
· [Frozen protocol](CROSSED_BELIEF_VALIDATION_PROTOCOL_V1.md)
· [Independent prospective review](CROSSED_BELIEF_DESIGN_REVIEW_2026-11-21.md)

![Fixed readout and output results](figures/2026-11-21/crossed_belief_validation.png)

[Vector PDF](figures/2026-11-21/crossed_belief_validation.pdf)
· [Editable vector SVG](figures/2026-11-21/crossed_belief_validation.svg)

## What this study adds

Earlier message-transfer cases made the initial reader belief the opposite of
the message. The new bank varies initial belief I, message M, trusted receipt B
and world truth W independently within every one of four fresh wording/entity
templates. Two physical-name mappings give **128 unique sender inputs**. Actual
reading A is hidden from the sender; two private A records per input create
256 fictional rows, not additional model observations.

The sender knows I and M, trusts B, knows the reader's update rule, and cannot
independently see delivery. The stipulated sender-attributed reader location is
`T = M if B else I`; actual fictional reader belief is `R = M if A else I`.
Class 1 means belief in the fixed place_b proposition, not belief accuracy.
These are narrated fictional states, not independently measured LM beliefs.

The primary 64 inputs have I != M, so the receipt should change T. The other
64 have I=M, so the receipt should not change T. A probe that preserves the old
assumption I=1-M could pass the primary subset and fail the controls. Independent
review caught this before execution, and the frozen protocol requires success
on both subsets. Every template crosses every source-factor combination.

The unchanged head is `nested_question__sender__nested__layer_16`, including its
source scaler, row normalization, coefficients, intercept and threshold. Capture
is the last complete chat-template token before an answer, block16/HF state17.
The model is pinned Llama 3.1 8B Instruct, NF4 weights/BF16 compute. This belief
readout is separate from the project's Apollo-style deception-probe adaptation.

Each input was captured with and without the original explicit nested-belief
question: **256 measurement forwards** and **128 queried greedy answers**.
Each forward saved the activation and full vocabulary logits together. No answer
was generated for the pre-question condition. Source-vector reproduction and a
separate layer-index check preceded the new inputs. The 460-515-token prefixes
passed preflight; no truncation, prompt repair, retries or refitting occurred.

## Main results and prospective decision

| Measure | Primary: I differs from M | Controls: I equals M |
| --- | ---: | ---: |
| Queried frozen probe correct | **48/64** | **43/64** |
| Queried frozen probe AUROC | .782227 | .759766 |
| Queried output YES-NO AUROC | .769531 | .825684 |
| Fixed output YES/NO decision correct | 33/64 | 33/64 |
| Strict greedy answer correct | 26/64 | 31/64 |
| Invalid, capped answers | 15/64 | 22/64 |
| Pre-question frozen probe correct | 32/64 | 32/64 |
| Pre-question frozen probe AUROC | .591797 | .666016 |

The queried probe passes all primary floors: 48/64 overall, 24/32 in each
physical mapping, template accuracies 13/16, 10/16, 13/16 and 12/16, and 29/32
positive gold-oriented receipt contrasts. It fails the additional control
floors: 43/64 overall, 21/32 in mapping 0 and 22/32 in mapping 1, below the
required 48/64 and 24/32. Every control template individually meets its 10/16
floor. **The full probe qualification is failed.**

The separately declared **primary output-ranking qualification passes**,
including both mappings and all templates. This does not qualify greedy
behavior, the whole conditional-update task, or any old gate. Behavior fails
with 57/128 strict answers correct and 37 capped invalids. Among 91 valid
answers, 90 are NO and one YES. Fixed YES/NO next-token decisions score only
66/128, so the weakness cannot be attributed solely to the answer-length cap.
The cap and parser were not changed after seeing these outputs.

Across the full bank, the queried probe scores **91/128 (71.1%)**. I-only and
M-only shortcuts each score **96/128 (75%)**, since I=M on half the inputs. They
each score 32/64 on the primary subset. A symbolic application of the stipulated
rule scores 128/128 by construction. These results do not establish a latent
advantage over capable text reasoning. The small primary AUROC difference
(.782 versus .770) does not establish statistical superiority or independent
information beyond output scores; output ranking is stronger on the controls.

Before the question, the same frozen head predicts class 1 on **all 128 inputs**.
Its fixed-threshold transfer therefore fails. Its non-flat AUROCs mean that this
is not evidence of zero label information. A different coding geometry or token
position could matter. Both prefixes also retain a candidate proposition and
assessment framing; even a positive result would not demonstrate natural,
spontaneous sender anticipation.

## Counterfactual evidence

| Queried-head comparison | Pairs | Observed result |
| --- | ---: | --- |
| Receipt changes, I != M: T changes | 32 | Correctly oriented margin in 29; both verdicts correct in 16 |
| Receipt changes, I=M: T fixed | 32 | Verdict changes in **13**; both correct in 15 |
| World changes: T fixed | 64 | Verdict changes in **15**; both correct in 38 |
| Physical names exchanged: T fixed | 64 | Verdict changes in **7**; both correct in 42 |

The mean absolute queried-margin movement is .144554 for informative receipt
pairs, .101106 for invariant receipt pairs, .109273 for world pairs and .042642
for name swaps. These are descriptive paired contrasts, without a population
significance claim. The irrelevant changes need not always make an answer worse:
a flip can correct an initially incorrect verdict as well as introduce an error.

One **posthoc illustrative pair** holds the world, message and initial belief
at the amber strongbox. Lennox thinks Daria believes the token is there whether
or not she reads a message repeating that location. The queried proposition
concerns the teal strongbox and is false in both versions. Changing only the
trusted reading receipt moves the probe's positive probability from .512997
(incorrect YES) to .498965 (correct NO), crossing its fixed threshold. The model
answers NO correctly in both cases. The exact pair is
`cb_t01_i0_m0_b0_w0_map0` versus `cb_t01_i0_m0_b1_w0_map0` in the committed bank.
This illustrates readout sensitivity with unchanged gold and correct verbal
answers; it does not prove the model's internal belief stayed unchanged.

The primary hidden-reader negative control is exactly 50% accuracy and AUROC
in each capture condition, as required when identical inputs receive opposing
private A labels. Over all authored rows the hard-accuracy ceiling is 75%,
because I=M determines R without knowing A. Do not pool A duplicates or physical
mapping pairs as independent stories. There are four templates and one update
rule family; human semantic validation is still pending.

## Verification and reproducibility

- Frozen execution commit: `f576a5a8e79c65464fd510f78d7e2b18c40dc213`.
- Raw bank SHA256: `3fc31d095b75ac948cd04bb5eb2bafdad62a2bf2a872243657e4367effa8cc6c`.
- Prepared protocol SHA256: `71f2ac9dc316bb1976551ba74fe0de8967b883378c424a03038dfa8814c48b36`.
- Native run: `/home/adirj/kcl-latent-probes-runs/crossed_belief_validation_v1_20261121`.
- Execution: 16:06:43-16:14:36 UTC, including checkpoint loading.
- Ignored Windows mirror: `runs/crossed_belief_validation_v1_20261121`.
  **517 files, 68,462,262 bytes**, verified identical to the native file set and bytes.
- The frozen CPU analyzer re-derived labels, paired metrics, strict answers,
  tokenizer prefixes and all scores. A separate agent auditor imported no
  project code and independently reconstructed **257 activation/full-vocabulary
  arrays** including the reference, all 256 prefixes and all 128 answers. Maximum
  margin discrepancy was 3.33e-16; output-scalar discrepancy 8.94e-12. It also
  checked 18 policy files and all inherited source, known-run and fit files.
- The independent auditor's original record correctly has mirror verification
  false: it audited the native data before copying finished. The main verified
  record separately establishes mirror identity; neither record is relabeled.
- A third agent independently reconstructed counts, AUROCs, counterfactual
  effects and qualification from raw JSON using PowerShell.
- A separate real-data [Windows/WSL CPU replay](validation/2026-11-21-crossed-belief-cpu-reproduction.json)
  reproduces every scientific record and metric. Both verify complete native/
  mirror byte identity; tokenizer reconstruction is explicitly WSL-only because
  the native lightweight environment does not include Transformers. No model
  inference is repeated by either analyzer.
- Before execution: **911 local tests and 48 subtests passed**; Ruff passed.
  All five [CI jobs for the frozen commit](https://github.com/ADI-0519/Individual-Project/actions/runs/44922585075)
  passed, including the measured Progress 2 rebuild. These engineering checks
  are separate from semantic validation and model measurements.

Rebuild the verified record under the original WSL project environment:

```bash
HF_HOME=.cache/huggingface-wsl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-wsl/bin/python scripts/analyse_crossed_belief_validation.py \
  --run runs/crossed_belief_validation_v1_20261121 \
  --native /home/adirj/kcl-latent-probes-runs/crossed_belief_validation_v1_20261121 \
  --verify-tokenizer --output .cache/crossed-belief-reanalysis.json

.venv-wsl/bin/python scripts/plot_crossed_belief_validation.py \
  --record docs/validation/2026-11-21-crossed-belief-validation.json \
  --output .cache/crossed-belief-figure
```

Do not execute the runner again over this attempt or alter its frozen files.
The independent audit script's exact executed bytes are preserved locally at
`runs/crossed_belief_independent_audit_v1.py`, SHA256
`02d4394ba7408ac0c8ce914e975a66a70a5d550d488255842dbfef3a35d137fc`.

## Research decision

The defensible finding is that this queried readout often responds correctly
to belief-relevant evidence **and also responds when that evidence should not
change the attributed belief**. This is directly relevant to Variant 2's
robustness and transferability brief. Generic belief decoding and prompt
brittleness already have substantial precedent; see the
[focused literature update](CROSSED_BELIEF_LITERATURE_ADDENDUM_2026-11-21.md).
No publication-level novelty or spontaneous anticipation is established.

Do not expand into the old goal/archive conditions or fit a replacement on
these validation outcomes. For the original anticipation project, the next
bounded stage should qualify the neutral behavior of one prospectively selected
reference checkpoint, without prompt search or retries. This bank is now
development/model-selection material. A replacement model-specific probe would
then need fresh fully crossed training material and an untouched update-rule
family before a new goal experiment. A full-BF16 reference on CREATE would help
test a precision difference that the local NF4 study cannot resolve; no evidence
currently attributes this failure to quantization or guarantees that a larger
model would solve it.

The [64-account blinded review packet](review/crossed_belief_validation_v1_human/review.md)
is ready, with blank annotations. Human review is an outstanding scientific
input, not a result of having generated a packet. Existing failed gates remain
failed. This study completes an instrument-validation milestone; it does not
complete the research project.

## Material for the next supervision meeting

**Project:** evaluating whether latent belief probes track a sender's attributed
reader belief reliably under changes to evidence and context. This is an
instrument-validation extension within Variant 2; it remains separate from the
Apollo-style deception-probe baseline.

**Achieved:** completed a prospectively frozen 128-input factorial validation,
256 prefix captures and 128 answers. The probe passes informative-update cases
but fails no-change controls. Saved all raw vectors/logits, independently audited
them, reproduced the analysis on Windows and WSL, and produced the figure above.

**Barrier:** the current readout is sensitive to changes that leave the target
belief fixed, while the model's explicit neutral answers are unreliable. Human
semantic annotations remain pending. This prevents a defensible claim about
unchanged current belief under future communication goals.

**Next steps:** human review, one fixed neutral reference/checkpoint comparison,
then fresh fully crossed training and untouched rule-family validation if
justified. CREATE would support a full-BF16 reference and a larger preselected
checkpoint; current data do not show which precision/capacity choice will help.
The overall plan is behind the intended goal-communication milestone, but the
instrument-validity requirement has now produced a concrete measured result.
