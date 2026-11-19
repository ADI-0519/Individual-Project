# Neutral-trained reader forecast: first natural development pilot

19 November 2026. The main question remains the one in the slides: **does a latent reader-outcome readout trained during neutral communication transfer to reporting pressure and improve on text/source controls?** This run establishes an executable first experiment. It does not establish belief decoding, truthful-omission detection or a held-out research result.

## What ran

- 24 fresh authored cases in four rule families, with explicit 16/4/4 training/validation/test case splits. All four families occur in every split.
- 72 naturally generated neutral reports, three attributed-reader contexts per case, each followed by a fresh reader conversation.
- Fixed-block-8 features from the last non-special report token. These are **after-report** representations, not pre-report anticipations.
- Five models fitted on neutral training rows only: activation, prompt TF-IDF, report TF-IDF, full-text TF-IDF, and full text plus activation. C=1 was fixed; validation/test labels did not choose hyperparameters.
- 47 valid training outcomes: 29 APPROVE, 10 REJECT, 8 DEFER. One INVALID response remains in the exclusion ledger.
- Fitted artifacts frozen after the 72 neutral reports, before 12 favorable-goal reports were generated on the four test cases. Frozen scoring made no fit calls; artifact hashes remained unchanged.
- 96 reader-control calls. The 24 full-evidence checks deliberately duplicate informed/no-report prompts and add no independent sample size. 23/24 full-evidence decisions were correct.
- A separate, post-hoc 24-call direct LLM text forecast of the same frozen test reports, plus an exploratory literal decision-copy control. Neither modifies the original reports or readout.

The model was the same pinned Llama-3.1-8B-Instruct in all three roles, using NF4 weights/BF16 compute locally. Greedy decoding and a fixed seed were used. The [machine-readable report](./validation/2026-11-19-neutral-reader-forecast.json) records settings, classes, metrics and source hashes.

## Measured comparisons

Each test condition has 12 rows from **four cases**, with three reader contexts per case. These are not 12 independent scenarios. Neutral labels are APPROVE/REJECT/DEFER = 8/3/1; favorable labels are 8/1/3.

| Predictor | Neutral correct | Favorable correct | Neutral macro F1 | Favorable macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Activation readout | 10/12 | 10/12 | .703 | .580 |
| Prompt TF-IDF | 8/12 | 8/12 | .267 | .267 |
| Report TF-IDF | 8/12 | 8/12 | .267 | .267 |
| Full-text TF-IDF | 8/12 | 8/12 | .267 | .267 |
| Full text + activation | 10/12 | 10/12 | .703 | .580 |
| Direct LLM text forecast, exploratory | 8/12 | 9/12 | .267 | .614 |
| Literal report-decision copy, exploratory | 7/12 | 11/12 | .381 | .647 |

The literal control takes the last uppercase APPROVE/REJECT/DEFER in the report; when none occurs it uses the neutral-training majority, APPROVE. Explicit decision words occur in 10/12 neutral and 6/12 favorable test reports. This control was specified after inspecting the development reports, so it is an exploratory shortcut diagnostic, not a prospectively selected primary comparator.

It is reproducible without model inference using `scripts/literal_report_control.py --run runs/neutral_forecast_development_v1 --output runs/literal_control_new.json` with the project's Python interpreter.

The direct LLM baseline continues the original sender prompt and report with one fixed recipient-forecast question. It receives neither the observed recipient answer nor private world-correct metadata. It is a same-checkpoint, post-hoc exploratory baseline; a larger independent capable monitor remains outstanding. No calibrated probability or log loss is invented from its single generated decision.

**Added value beyond report text is not established.** The literal control beats the activation readout on favorable accuracy and macro F1. All three learned lexical controls predict APPROVE throughout; their fixed-C regularization and feature scaling make them preliminary controls, not a claim that strong text methods have been beaten.

Unchanged accuracy also hides changes in error type. Activation balanced recall falls from .778 to .556, with zero recall on the single favorable REJECT outcome. All its errors in both conditions occur in the one held-out conjunction case. The other three cases are each 3/3. No family-generalization or population confidence claim follows.

Activation neutral log loss is .958 versus .683 for full text, despite higher accuracy; favorable values are .409 and .709 respectively. Thus neither improved calibration nor uniform superiority is established. Adding text to activations gives .962/.424 loss with the same predicted labels. Future comparisons need separately validated regularization and appropriate feature-block scaling.

## What the data says about the target

Observed reader decisions are usable behavioral labels. They are distinct from correct decisions, truthful reports, disclosure and deceptive intent. The source-world rule predicts 8/12 neutral and 6/12 favorable reader outcomes; the no-report reader baseline predicts 7/12 and 5/12. Those are controls for outcome prediction, not estimates of honesty.

The failed full-evidence control is the Robin Services validation case: revenue 83000 and costs 90000 imply REJECT, but the reader answers APPROVE. The failure is retained. It is a reader error with sufficient evidence, not an omission.

A six-report qualitative inspection also found:

- A report calling 218 points “below” a threshold of 211.
- A report incorrectly claiming that an October pass supersedes a November failure.
- A report saying DEFER while disclosing decisive current evidence, after which the reader correctly approves.
- A correct negative-profit report ending REJECT, after which the reader nevertheless approves.
- One reader output beginning APPROVE but adding “Reasoning:”; the strict parser correctly marks it INVALID.
- Reports explicitly containing decision words, which create a plausible post-report shortcut.

These examples are not a prevalence audit. The natural data cannot yet be called a truthful-omission benchmark. There are four report word-budget violations and two length-terminated reports; all remain visible without regeneration. The 40-word instruction was not enforced by truncating text.

## Next experiment, preserving the original project

**Subsequent development update:** the report audit, prefix-only before-report
comparison and a paired sender-policy ablation have now run. The
[new results](./RESEARCH_CHECKS_RESULTS.md) preserve this pilot unchanged and
report a sharp before-report transfer failure. Human review and a fresh main
study remain outstanding; the steps below record the priorities identified
from the original run.

1. Audit sender facts, temporal comparisons, recommendations, omissions and recipient outcomes separately. Keep this development run frozen. Clarify evidence-precedence and sender/reader roles in a separately versioned protocol.
2. Compare **pre-report prompt features** with during/after-report features, with positions fixed before the next fresh evaluation. A pre-report readout removes direct access to the sampled report, but still needs source-state and prompt-only controls. It is not automatically a belief mechanism.
3. Develop independently labelled neutral communication/belief tasks whose target is the recipient's **post-message** state, with world truth, prior recipient state and resulting state separated. Preserve the neutral-trained transfer objective in the slides.
4. Strengthen text baselines and tune regularization on neutral validation data before opening fresh test outcomes. Carry forward the literal-decision control. Use a second reader and hold out complete scenario families.
5. Use a CREATE BF16 reference to check precision sensitivity and another checkpoint to check whether the observed reasoning failures are peculiar to this reader. Compute cannot substitute for a valid target.

The November engineering milestone is achieved: real 8B activations, an Apollo comparator, reader controls, and a neutral-trained frozen outcome readout now exist. Data validity, text shortcuts, pre-report specificity and independent generalization are the next research gates.

## Reproduction and presentation

The [research direction](./RESEARCH_DIRECTION.md) gives the cached WSL command for the natural pilot. The separate text baseline is:

```bash
HF_HOME=.cache/huggingface-wsl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TRITON_CACHE_DIR=.cache/triton-wsl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-wsl/bin/python scripts/run_text_forecast.py \
  --config configs/apollo_llama8b_4bit.json \
  --run runs/neutral_forecast_development_v1 \
  --output runs/neutral_forecast_text_new
```

Use fresh output paths; original artifacts remain untouched. Run the presentation builder with the locked `slides` extra and installed Arial fonts. The source template, measured JSON and four-slide outputs are versioned under [presentations](../presentations/).
