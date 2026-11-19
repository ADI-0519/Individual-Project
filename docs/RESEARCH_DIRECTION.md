# Research decision after the baseline audit

19 November 2026. This is a decision record, not a preregistration or a claim of publication-level novelty.

**Keep Variant 2. Use Apollo as a frozen reference probe and make recipient-context robustness the central experiment.** The instruction-pair baseline is the correct first engineering milestone. Its perfect instruction-label AUROC is not the project contribution.

The [first real pilot](./PILOT_RESULTS.md) is complete, including a failed restoration control and a separately recorded follow-up. It supports progressing through data-validity checks rather than immediately scaling or claiming a working omission detector.

## What the project should ask

> When literal report content is matched, how do established deception monitors respond to changes in the knowledge attributed to the recipient, and how does that sensitivity relate to measured reader decisions?

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

The candidate contribution is the controlled combination: the same truthful report and fixed factual edits, independently varied starting evidence, actual reader replay, and frozen-monitor comparisons. Not finding that exact combination in the inspected papers is a bounded literature-search result, not proof of novelty.

## What the completed baseline actually establishes

The saved experiment contains 536 forced dialogues, 219 training and 49 validation fact pairs, and 3,200 scored token rows. The target is **instructed honesty condition**. Both conditions use the same true continuation for each fact; the dishonest instruction does not make that continuation an observed lie.

An independent auditor recomputes normalization, frozen margins, response aggregation and AUROC, checks artifact hashes and record alignment, examines shared causal histories, and runs train/validation-independent pairwise label flips. It also fits prompt-only and response-only text classifiers. Their AUROCs are respectively **1.0 and 0.5**. Prompt information alone therefore suffices to solve this validation target; this does not establish that the activation probe uses only the prompt.

There are 178 validation token rows whose entire causal histories occur in training, including 130 rows with exactly duplicated vectors. Ten validation dialogues have all scored histories covered. Removing any pair containing a shared scored history leaves only eight validation pairs, still with instruction AUROC 1.0. The overlap limits independence; it does not explain away all instruction discrimination. Preserve the original frozen fit and report these diagnostics rather than silently replace the split after seeing scores.

## Progress gates

1. **Engineering:** verify source recipe adaptations, exact layer convention, token boundaries, numerical replay, causal masking, train-only normalization and data lineage. Preserve the NF4/BF16 identity; do not treat it as a BF16 precision replication.
2. **Reader feasibility:** four authored development cases, three reader packets, four report versions. Preserve correct, wrong, deferred and invalid decisions. Measure whether restoration helps and whether the irrelevant addition produces a similar effect. No population confidence claim from four authored cases.
3. **Probe specificity:** run the separate small MASK-derived pressure pilot with blinded, whole-response labels and operational belief checks. Keep refusals and mixed assertions separate. Compare natural honest and belief-contradicting assertions within the same pressure prompt where both exist. This validates the adaptation; the already-known pressure confound is not a novelty claim.
4. **Audience-context evaluation:** freeze the layer, probe, pooling and any independently calibrated threshold before the main study. Compare paired context sensitivity against actual reader effects, a capable full-context text monitor, an instructions-only baseline, and simple evidence checks.
5. **Generalization:** author and independently audit fresh scenario families; keep development cases out of held-out evaluation. Hold out scenario families, then replay the same frozen reports with a second reader. Report failures, uncertainties and denominator changes.

The primary reader contrast is the reduction in wrong definite decisions after restoring the decisive fact in stale-evidence contexts, minus that reduction for informed readers. Also report correct-decision gains, deferrals, invalid outputs and no-report transitions. A wrong-to-defer change is useful but differs from a wrong-to-correct change. An outdated belief that persists after a report differs from an error induced by it.

## Scope to defer

Keep a neutral recipient-belief readout as a secondary transfer experiment. Its target must be message-conditioned recipient belief; a classic false-belief story about an agent's prior belief is not automatically second-order belief decoding. Decodability alone does not establish causal use.

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
