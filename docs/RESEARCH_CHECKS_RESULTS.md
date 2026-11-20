# Before-report and sender-policy development checks

19 November 2026. **These two diagnostics are complete; the main research study and dissertation remain unfinished.** They extend the [first natural pilot](./NEUTRAL_FORECAST_RESULTS.md) while preserving its reports, outcomes and fitted models. The question remains neutral-trained reader-outcome prediction under reporting pressure, compared with text and source controls.

The [machine-readable record](./validation/2026-11-19-prereport-and-policy-checks.json) contains exact metrics, model identity, hashes and limitations. The executed scripts are preserved at commit `084f5ac2d37c07e20fda29881b120b363380197f`. Subsequent guards strengthen reuse without rewriting this run.

## 1. Does the result survive removing access to the generated report?

Captured 84 new activation vectors using **only saved sender-prompt tokens**, at the last prompt token of decoder block 8. No response tokens enter the fitted features. Every text input's report field is the same `[REPORT NOT YET GENERATED]` placeholder. A new scaler and three-class logistic model are fitted on the original neutral training outcomes only, using fixed C=1. The after-report model is not reused on these different features.

This is an **exploratory replay of already observed outcomes**. Freezing the new weights before scoring prevents refitting on pressure rows, but does not turn known development outcomes into a fresh prospective test. There are 47 valid training rows and one recorded INVALID exclusion. Evaluation has four cases and three dependent contexts per case. Families occur in every partition.

| Predictor and information time | Neutral correct | Favourable-pressure correct | Neutral / pressure macro F1 | Neutral / pressure log loss |
| --- | ---: | ---: | ---: | ---: |
| New before-report activation | 8/12 | 1/12 | .425 / .051 | 1.255 / 9.917 |
| New before-report text + activation | 8/12 | 1/12 | .425 / .051 | 1.291 / 9.851 |
| New prompt-only text | 8/12 | 8/12 | .267 / .267 | .688 / .721 |
| New full pre-report text | 8/12 | 8/12 | .267 / .267 | .688 / .722 |
| New constant-report/prior control | 8/12 | 8/12 | .267 / .267 | .856 / .894 |
| Earlier after-report activation | 10/12 | 10/12 | .703 / .580 | .958 / .409 |

Both new activation methods predict **REJECT on all 12 pressure rows**. The observed labels are eight APPROVE, one REJECT and three DEFER. The before-report activation model therefore exhibits a substantial transfer failure in this development setting. Its neutral accuracy equals the majority-class control, although its predicted labels and macro F1 differ. The learned text controls predict APPROVE throughout and remain preliminary fixed-C baselines.

The earlier after-report result does not establish anticipation before composing the report. It still precedes the reader's answer and can legitimately forecast that later answer; the report-text shortcut prevents a stronger recipient-representation interpretation here. Comparing the two time points changes both token position and available information. It is an information/timing diagnostic, not a matched-information claim of representation superiority. A single block, position, checkpoint, quantization and regularization choice does not establish that all prospective latent forecasts fail. Neither time point establishes a second-order-belief mechanism or deceptive intent. The [20 November follow-up](./BELIEF_DIAGNOSTIC_RESULTS.md) diagnoses fitting sensitivity and directly tests a separate nested-perspective target.

### Numerical checks and retained failure

An independent agent checked all 84 capture bindings and all 120 prediction rows, independently recomputed metrics, and verified all 377 original files, 106 diagnostic outputs and ten new frozen artifacts against their recorded hashes.

- All eight prefix-only repeats and eight full-sequence repeats are exact.
- All 16 same-shape mutations of later tokens leave the earlier captured vector exactly unchanged.
- Prefix-only versus longer-sequence values pass the prespecified one-adjacent-BF16-value bound for only **2/8** examples. Six fail; the maximum absolute difference is **0.03125**.

The numerical status remains **inconclusive**. The tolerance was not widened. These differences do not establish future-token leakage; changing sequence shape can alter numerical execution. The exported training features come exclusively from prefix-only forwards, whose actual inputs contain no future tokens. Longer sequences are used only in the separately identified controls.

## 2. Does a sender instruction encourage decision recommendations?

On eight original **training** cases, two per family and all three reader contexts, removed exactly this sentence from the sender system message:

> If the available facts do not establish APPROVE or REJECT under the decision rule, choose DEFER.

Generated 24 new reports and 24 fresh reader answers. Source facts, attributed reader packets, reader policy, checkpoint, greedy decoding, seed and generation budgets match the original paired conditions. The selection rule was recorded before these new generations, after inspecting v1. Original and new outputs are retained, including length violations and truncations.

| Measurement across 24 paired reports / eight cases | Original | Sentence removed |
| --- | ---: | ---: |
| Reports containing an uppercase decision word | 22 | 11 |
| Reports containing uppercase DEFER | 12 | 0 |
| Reader correct / wrong / deferred / invalid | 8 / 10 / 6 / 0 | 15 / 8 / 1 / 0 |
| Reports exceeding the 40-word instruction | 1 | 3 |
| Reports terminated at the token cap | 1 | 2 |

This intervention changes reporting and reader behavior in this small deterministic panel. Its aggregate result does not establish general improvement or a belief mechanism. In particular, fewer deferrals must be distinguished from fewer wrong decisions.

A separate agent reviewed the new sender texts and sources without reader outcomes or probe scores. The [24-row draft audit](./SENDER_POLICY_AUDIT_DRAFT.md) labels 18 reports factually supported and six contradicted. The matched original draft labels are 15 supported, eight contradicted and one unclear. **Three previously supported reports become contradicted.** All labels require human review; apparent aggregate improvement does not justify declaring the prompt repaired.

The [original 84-report draft audit](./REPORT_AUDIT_DRAFT.md) labels 52 supported, 30 contradicted and two unclear. Coverage, factual claims, rule reasoning and recommendation words are separately annotated. These drafts are not validated truthful-omission or deception labels. Their raw quotes and byte hashes are preserved in the [annotation tables](../data/annotations/).

## What follows

1. Independently review case semantics and report annotations, including new contradictions, temporal precedence, arithmetic and ambiguous historical claims. Keep original artifacts and disagreements.
2. Develop a separately versioned protocol with clear sender/reader roles. Choose the primary representation time and target before opening fresh evaluation outcomes. The current after-report behavioral readout is an operational precursor to the stronger belief-transfer claim.
3. Validate competitive text/source baselines, regularization and feature-block scaling on neutral development/validation data. Retain the literal-decision shortcut diagnostic.
4. Use fresh audited cases and complete held-out families for the main transfer study, with case/family-aware uncertainty, calibrated probabilities where applicable, and a second reader replay of frozen reports.

The [main-study gates](./MAIN_STUDY_GATES.md) define the remaining evidence, sample/precision planning and deliverables. Progress 2 can report a functioning pipeline, a real neutral-to-pressure pilot, text-shortcut concerns, and this new before-report failure. The project is still in development; no main-study conclusion, publication outcome or final-project completion is claimed.

## Reproduce

Use the established WSL environment, cached checkpoint and a fresh output path:

```bash
HF_HOME=.cache/huggingface-wsl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_HUB_DISABLE_TELEMETRY=1 TRITON_CACHE_DIR=.cache/triton-wsl \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv-wsl/bin/python scripts/run_next_research_checks.py \
  --config configs/apollo_llama8b_4bit.json \
  --run runs/neutral_forecast_development_v1 \
  --output runs/research_checks_new
```

The combined runner shares one model load and closes it after both checks. Subsequent code rejects source-contained output before writing, and checks saved job bindings, reader prompts, source-plan hash, generation policy and backend/template identity before the paired intervention. Independent inspection confirmed these properties already hold for the completed v2 artifacts. No PyTorch reinstall, model download or new authentication is required.
