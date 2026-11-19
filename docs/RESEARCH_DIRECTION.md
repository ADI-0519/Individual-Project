# Research decision after the baseline audit

19 November 2026. This is a decision record, not a preregistration or a claim of publication-level novelty.

**Keep the question presented in Progress 1 and the Progress 2 draft central: train a latent reader-outcome readout under neutral communication, freeze it, and evaluate transfer under reporting pressure against text and source controls.** Apollo is the deception comparator. The earlier recommendation to make frozen-Apollo context sensitivity primary was a proposed narrowing, not an agreed interpretation of the slides; it is superseded here after the user requested alignment.

The [first real pilot](./PILOT_RESULTS.md) is complete, including a failed restoration control and a separately recorded follow-up. It supports progressing through data-validity checks rather than immediately scaling or claiming a working omission detector.

## What the project should ask

> Can a latent readout trained to predict readers' decisions during neutral communication transfer to pressured communication, and improve on prompt, report-text and source-state controls?

The first operational target is an observed APPROVE/REJECT/DEFER decision under the reader evidence attributed to the sender. It is a behavioral proxy for the listener-belief motivation, not a direct belief or intent label. The first readout uses the final non-special natural-report token at fixed decoder block 8, so it is an **after-report outcome readout**, not a demonstrated pre-report anticipation mechanism. A neutral belief-task readout transferring into communication remains a stronger subsequent test.

Separate source truth, reporting incentive, sender-attributed recipient evidence, report text, actual reader evidence, and the reader's eventual decision. This permits harmless omissions, consequential omissions, unsuccessful attempts to mislead, and harmful mistakes without attributing intent from an error alone.

The initial study should keep sender-attributed and actual evidence aligned. Subsequently cross them independently. A sender activation saved before independently assigning actual reader evidence cannot identify that later assignment. Give an outcome monitor the actual reader context and give its text baseline the same context. A representation extracted during a later reread is a monitoring representation, not evidence of the sender's original beliefs.

## Why this is narrower than the original pitch

The [deep probe audit](./LITERATURE_PROBE_AUDIT.md) and [recipient-context design](./RECIPIENT_CONTEXT_DESIGN.md) record primary sources and reading scopes. Several ingredients are already occupied:

- Apollo already reports false alarms on honest responses under misleading incentives. [Goldowsky-Dill et al., ICML 2025](https://proceedings.mlr.press/v267/goldowsky-dill25a.html)
- Probe instructions and deception categories already receive extensive robustness analysis. [Natarajan et al., ICML 2026](https://proceedings.mlr.press/v306/natarajan26a.html)
- Fact coverage, reader decisions and pre-generation coverage probes are already studied together. [Kim et al., EMNLP 2026](https://aclanthology.org/2026.emnlp-main.656/)
- Instructed-to-spontaneous omission transfer already exists, with documented omission-label validity problems. [Luikham, 2026](https://arxiv.org/html/2609.00180)
- Incentives and recipient knowledge already appear in candidate-response preference experiments. [Shkolnikov, 2026](https://arxiv.org/html/2609.04166v1)
- Belief decoding, second-order tasks and prompt brittleness already have precedents. [Zhu et al., ICML 2024](https://proceedings.mlr.press/v235/zhu24o.html), [Bortoletto et al., 2025](https://aclanthology.org/2025.findings-emnlp.1226/)

The candidate contribution is a controlled neutral-to-pressure transfer evaluation with natural reports, measured recipient outcomes, matched-information text baselines, world-state controls and fresh held-out scenario families. The fixed-report/recipient-evidence/restoration assay is supporting validation. Existing work overlaps strongly with belief decoding, coverage probes and omission transfer; the scientific contribution must come from the precise transfer result and controls, not merely combining familiar components. No novelty guarantee is claimed.

## What the completed baseline actually establishes

The saved experiment contains 536 forced dialogues, 219 training and 49 validation fact pairs, and 3,200 scored token rows. The target is **instructed honesty condition**. Both conditions use the same true continuation for each fact; the dishonest instruction does not make that continuation an observed lie.

An independent auditor recomputes normalization, frozen margins, response aggregation and AUROC, checks artifact hashes and record alignment, examines shared causal histories, and runs train/validation-independent pairwise label flips. It also fits prompt-only and response-only text classifiers. Their AUROCs are respectively **1.0 and 0.5**. Prompt information alone therefore suffices to solve this validation target; this does not establish that the activation probe uses only the prompt.

There are 178 validation token rows whose entire causal histories occur in training, including 130 rows with exactly duplicated vectors. Ten validation dialogues have all scored histories covered. Removing any pair containing a shared scored history leaves only eight validation pairs, still with instruction AUROC 1.0. The overlap limits independence; it does not explain away all instruction discrimination. Preserve the original frozen fit and report these diagnostics rather than silently replace the split after seeing scores.

## Progress gates

1. **Engineering:** verify source recipe adaptations, exact layer convention, token boundaries, numerical replay, causal masking, train-only normalization and data lineage. Preserve the NF4/BF16 identity; do not treat it as a BF16 precision replication.
2. **Reader feasibility:** four authored development cases, three reader packets, four report versions. Preserve correct, wrong, deferred and invalid decisions. Measure whether restoration helps and whether the irrelevant addition produces a similar effect. No population confidence claim from four authored cases.
3. **Probe specificity:** run the separate small MASK-derived pressure pilot with blinded, whole-response labels and operational belief checks. Keep refusals and mixed assertions separate. Compare natural honest and belief-contradicting assertions within the same pressure prompt where both exist. This validates the adaptation; the already-known pressure confound is not a novelty claim.
4. **Neutral-to-pressure readout:** collect natural neutral reports and observed reader decisions; fit on neutral training cases only. Freeze fitted weights and preprocessing before pressured test reports are generated. Compare activation, prompt-only, report-only, full-context text and text-plus-activation models. Include a programmatic source-world decision control and the no-report reader baseline. Preserve all three outcomes and report invalids separately.
5. **Generalization:** author and independently audit fresh scenario families; keep development cases out of held-out evaluation. Hold out scenario families, then replay the same frozen reports with a second reader. Report failures, uncertainties and denominator changes.

The main comparison is the change in frozen outcome-prediction performance between neutral and pressured communication on the same held-out cases. Report class counts, multiclass loss, per-class recall and macro F1, together with comparable text controls. Repeated contexts within a case are not independent cases. Evaluate an eventual genuinely held-out study with case/family-level uncertainty.

The supporting reader contrast compares the reduction in wrong definite decisions after restoring a fact across stale and informed contexts. Correct-decision gains, deferrals, invalid outputs and no-report transitions remain separate. A wrong-to-defer change differs from wrong-to-correct. These fixed-text diagnostics do not replace natural communication as the main evidence.

## First natural development run

The new panel contains 24 fresh authored cases, four rule families and explicit 16/4/4 train/validation/test case splits. All families appear across splits, so this is a case-held-out development test, **not family OOD**. The plan has 72 neutral natural reports, then 12 favorable-goal reports on the four test cases. Each report receives a fresh reader replay under the sender-attributed packet. There are also 96 explicit reader checks, including duplicate informed/no-report prompts that add no independent sample size.

The configuration is fixed before generation: the pinned 8B checkpoint, NF4/BF16 runtime, block 8, greedy decoding, one seed, 40-word report instruction and 96-token cap. Word-budget violations and truncation are recorded without regeneration. C=1 is fixed for the initial readouts; validation/test labels do not select hyperparameters. The 72 neutral records are exported and the models saved before favorable reports are generated. Frozen evaluation accepts only declared test cases and verifies artifact hashes.

If training lacks a decision class, the primary three-class readout is marked untrainable. An explicitly labeled two-class feasibility diagnostic may still verify fitting and frozen transfer, with the unsupported class and clipped loss disclosed. Do not manufacture DEFER labels. If the source-world rule already predicts all reader decisions, the panel has not established a listener-specific target. Lexical TF-IDF controls are preliminary; the strong LLM text monitor remains outstanding.

## Scope to defer

Keep causal activation editing and mechanistic second-order-belief claims as extensions after the outcome readout is validated. Its eventual belief target must be message-conditioned recipient belief; a classic false-belief story about an agent's prior belief is not automatically second-order belief decoding. Decodability alone does not establish causal use. Do not demote the neutral-trained transfer question described in the slides into an optional side project.

Postpone correction-benefit ranking until reader effects show reproducible variation worth predicting. It adds treatment-effect estimation and policy evaluation beyond the core Variant 2 question. Avoid simultaneously adding multilingual shifts, cross-model probe transfer, new steering methods, and a large retrieval system.

## Compute and research quality

The RTX 5080 supports the present 8B NF4/BF16 pilot. A justified CREATE request is one 40 GB A100 for a controlled BF16 reference and a second-reader replay, with profiled time/memory and frozen jobs. Confirm CREATE's installed driver/runtime first. More compute or a larger model is not itself a scientific contribution.

The work becomes research grade through a defensible target, independent labels, fair information-matched baselines, strong controls, held-out evaluation and reproducible results. Neither engineering maturity nor an encouraging pilot guarantees a paper, marks, or admission.

## Reproduce the independent audit

From the repository in the working WSL environment:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-wsl/bin/python -m latent_probes.baseline_audit \
  --capture runs/apollo_llama8b_full_v1 \
  --probe runs/apollo_llama8b_full_probe_v1 \
  --output runs/apollo_full_independent_audit_new.json --permutations 20
```

The genuine GPU audit, reader panel and fixed-report probe diagnostic share one model load:

```bash
HF_HOME=.cache/huggingface-wsl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TRITON_CACHE_DIR=.cache/triton-wsl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-wsl/bin/python scripts/run_research_audit.py \
  --capture runs/apollo_llama8b_full_v1 \
  --probe runs/apollo_llama8b_full_probe_v1 \
  --panel data/reader_panel/four_cases_v1.json \
  --output runs/research_audit_new
```

Choose fresh output paths. Original captures, fitted weights, plans, raw decisions and failed runs remain frozen. These commands reuse the established environment; they do not reinstall PyTorch or require another Hugging Face login.

Run the natural reader-forecast milestone from the repository in WSL:

```bash
HF_HOME=.cache/huggingface-wsl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TRITON_CACHE_DIR=.cache/triton-wsl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-wsl/bin/python scripts/run_forecast_pilot.py \
  --config configs/apollo_llama8b_4bit.json \
  --panel data/reader_forecast/development_v1.json \
  --apollo-probe runs/apollo_llama8b_full_probe_v1 \
  --output runs/neutral_forecast_new
```
